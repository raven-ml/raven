(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The operations rune2 differentiates, each as a function of a float64 value
   whose elements a case draws: every kind of scalar program and every
   operation, real and complex. *)

open Windtrap
open Compose
module D = Nx.Dtype
module L = Nx_array.Layout
module P = Nx_kernel.Prog
module T = Nx.Ptree
module C = Nx_support.Counting
module Count = (val Nx.devices ~kernels:(module C) [ Nx_support.memory 0 ])

let f64 = Nx.float64
let c128 = Nx.complex128

type f64_t = (float, D.float64_elt, Nx.host) Nx.t
type c128_t = (Complex.t, D.complex64_elt, Nx.host) Nx.t

let invalid ~sub f = raises_match (Exn.invalid_arg ~substring:sub) f
let v xs = Nx.create f64 [| Array.length xs |] xs

(* Operands *)

(* Values a case draws: uniform in a range per row of axis 0, or spaced apart,
   [lo + (j + 0.3) step] in a random order, so that no step of the differences
   crosses a kink. *)
type values = Uniform of (float * float) list | Spaced of float * float

type case = {
  name : string;
  empty : bool;  (** Whether its shapes may have no element. *)
  shape : int array Gen.t;
  values : values;
  f : f64_t -> f64_t;
  z : ((f64_t -> c128_t) * (c128_t -> c128_t)) option;
      (** A complex function, and the complex value it reads from [f]'s operand,
          where [f] is it on the parts. *)
}

let case ?(empty = true) ?(values = Uniform [ (-2., 2.) ])
    ?(shape = Gen.(map (fun n -> [| n |]) (int_range 0 5))) ?z name f =
  { name; empty; shape; values; f; z }

let rows k = Gen.(map (fun n -> [| k; n |]) (int_range 0 4))

let matrix =
  Gen.(map (fun (m, n) -> [| m; n |]) (pair (int_range 0 3) (int_range 0 4)))

(* Matrices whose rows and columns are never empty, for extremes. *)
let square = Gen.(map (fun n -> [| n + 1; n + 1 |]) (int_range 0 2))

let full =
  Gen.(
    map
      (fun (m, n) -> [| m + 1; n + 1 |])
      (pair (int_range 0 2) (int_range 0 3)))

let cube =
  Gen.(map (fun (m, n) -> [| 2; m; n |]) (pair (int_range 0 3) (int_range 0 4)))

let draw values s =
  let n = numel s in
  let open Gen in
  match values with
  | Spaced (lo, step) ->
      map
        (fun p ->
          Array.of_list
            (List.map (fun j -> lo +. ((Float.of_int j +. 0.3) *. step)) p))
        (permutation (List.init n Fun.id))
  | Uniform ranges ->
      let k = List.length ranges in
      let per = if Array.length s = 0 then n else n / Int.max 1 s.(0) in
      let row r = List.nth ranges (if k = 1 then 0 else r) in
      let rec go i =
        if i = n then constant []
        else
          let lo, hi = row (if per = 0 then 0 else i / per) in
          let* x = float_range lo hi in
          let+ rest = go (i + 1) in
          x :: rest
      in
      map Array.of_list (go 0)

let gen c =
  let open Gen in
  (let* s = c.shape in
   let* x = draw c.values s in
   let+ t = array ~size:(constant (numel s)) (float_range (-1.) 1.) in
   (s, x, t))
  |> with_pp (fun ppf (s, x, t) ->
      Format.fprintf ppf "%a x %a t %a" pp_shape s
        (Format.pp_print_list Format.pp_print_float)
        (Array.to_list x)
        (Format.pp_print_list Format.pp_print_float)
        (Array.to_list t))

let row k x = Nx.get [ k ] x
let two f x = f (row 0 x) (row 1 x)
let three f x = f (row 0 x) (row 1 x) (row 2 x)
let complex x = Nx.complex c128 ~re:(row 0 x) ~im:(row 1 x)
let parts z = Nx.stack [ Nx.real f64 z; Nx.imag f64 z ]
let on_complex f x = parts (f (complex x))
let contiguous s = L.contiguous (Iarray.of_array s)
let any = D.Any f64
let identity = P.of_node ~ins:[| any |] (Op1 (Copy, any, 0))

(* A loop of one load along axis 1 of [x], as nx's functions build none. *)
let reduce1 reduction x =
  Nx.Prim.eval ~by:"test"
    (Reduce
       {
         layout = contiguous (Nx.shape x);
         axes = [| 1 |];
         prog = identity;
         reductions = reduction;
         loads = [| Plain x |];
       })

let scan1 ?carry reduction x =
  Nx.Prim.eval ~by:"test"
    (Scan
       {
         layout = contiguous (Nx.shape x);
         axis = 1;
         prog = identity;
         reduction;
         loads = [| Plain x |];
         carry = Option.map (fun c -> Nx.Prim.Any c) carry;
       })

(* [x] of shape [m; n + 1]: a carry, column 0, and rows, the others. *)
let carried x =
  let n = Nx.dim 1 x in
  (Nx.slice [ A; R (0, 1) ] x, Nx.slice [ A; R (1, n) ] x)

let padded_shape =
  Gen.(
    map
      (fun (m, n) -> [| m + 1; n + 1 |])
      (pair (int_range 0 2) (int_range 0 3)))

(* A map of several nodes and two outputs, reading a padded load and the
   coordinates: [x0 sin (x1)], and [x0 + coord 0] over [x1] padded by one on
   each side of axis 1. *)
let multi x =
  let m = Nx.dim 0 x and n = Nx.dim 1 x in
  let pad : Nx_kernel.pad =
    { lo = [| 0; 1 |]; hi = [| 0; 1 |]; interior = [| 0; 0 |]; windows = [||] }
  in
  let x0 = Nx.pad [| (0, 0); (1, 1) |] 1.5 x in
  let prog =
    P.v ~ins:[| any; any |]
      [|
        In 0;
        In 1;
        Op1 (Unary Sin, any, 1);
        Op2 (Binary Mul, 0, 2);
        Coord 0;
        Op1 (Cast, any, 4);
        Op2 (Binary Add, 0, 5);
      |]
      ~outs:[| 3; 6 |]
  in
  let y, (z, ()) =
    Nx.Prim.eval ~by:"test"
      (Map
         {
           layout = contiguous [| m; n + 2 |];
           prog;
           outs = [ f64; f64 ];
           loads = [| Plain x0; Padded { x; fill = 0.25; pad } |];
         })
  in
  Nx.concatenate ~axis:0 [ y; z ]

(* [Copy] and a cast there and back, as kinds of a program. *)
let kinds x =
  let c64 = D.Any c128 in
  let prog =
    P.v ~ins:[| any |]
      [| In 0; Op1 (Copy, any, 0); Op1 (Cast, c64, 1); Op1 (Cast, any, 2) |]
      ~outs:[| 3 |]
  in
  fst
    (Nx.Prim.eval ~by:"test"
       (Map
          {
            layout = contiguous (Nx.shape x);
            prog;
            outs = [ f64 ];
            loads = [| Plain x |];
          }))

(* A program of twenty operands, [Σ_k x_k Π_(j<k) sin x_j], past the arrays one
   block of nx.cpu's holds: its derivative is one loop of forty loads. *)
let wide = 20

let wide_prog =
  let x k = P.E.input k Nx_array.Dtype.Float64 in
  let rec go k acc prod =
    if k = wide then acc
    else go (k + 1) P.E.(acc + (x k * prod)) P.E.(prod * sin (x k))
  in
  P.of_exprs
    ~ins:(Iarray.init wide (fun _ -> any))
    [| P.E.any (go 1 (x 0) (P.E.sin (x 0))) |]

let wide_loads x = Iarray.init wide (fun k -> Nx.Prim.Plain (row k x))
let rows_wide = Gen.(map (fun n -> [| wide; n |]) (int_range 0 4))

let wide_map x =
  fst
    (Nx.Prim.eval ~by:"test"
       (Map
          {
            layout = contiguous [| Nx.dim 1 x |];
            prog = wide_prog;
            outs = [ f64 ];
            loads = wide_loads x;
          }))

let wide_reduce reduction x =
  fst
    (Nx.Prim.eval ~by:"test"
       (Reduce
          {
            layout = contiguous [| Nx.dim 1 x |];
            axes = [| 0 |];
            prog = wide_prog;
            reductions = [ reduction ];
            loads = wide_loads x;
          }))

(* Two sums of a program's two outputs, the second's first. *)
let sums x =
  let prog =
    P.v ~ins:[| any |]
      [| In 0; Op1 (Unary Sin, any, 0); Op2 (Binary Mul, 0, 0) |]
      ~outs:[| 1; 2 |]
  in
  let a, (b, ()) =
    Nx.Prim.eval ~by:"test"
      (Reduce
         {
           layout = contiguous (Nx.shape x);
           axes = [| 1 |];
           prog;
           reductions = [ Monoid (Sum, 1, f64); Monoid (Sum, 0, f64) ];
           loads = [| Plain x |];
         })
  in
  Nx.stack [ a; b ]

let idx xs =
  Nx.create Nx.int64 [| Array.length xs |] (Array.map Int64.of_int xs)

(* Positions of a scatter of [n] updates into 3 targets, two on target 0. *)
let targets n = idx (Array.init n (fun k -> if k < 2 then 0 else k mod 3))

let scatter combine x =
  let n = Nx.dim 1 x in
  let into = Nx.slice [ I 0; R (0, Int.min n 3) ] x in
  let into =
    if Nx.dim 0 into < 3 then Nx.pad [| (0, 3 - Nx.dim 0 into) |] 0.5 into
    else into
  in
  Nx.scatter ~combine ~axis:0 (targets n) (row 1 x) into

let pattern = Nx.Pattern.v "i j, j k -> i k | j"

(* Contractions with block scales and segments, of twenty draws: [a], then
   [b]. *)
let contract_of x ~a ~b ~batch ~contracting ?a_scale ?b_scale ?segments () =
  let take n off shape = Nx.reshape shape (Nx.slice [ R (off, off + n) ] x) in
  let na = Array.fold_left ( * ) 1 a in
  let a = take na 0 a and b = take (Array.fold_left ( * ) 1 b) na b in
  Nx.Prim.eval ~by:"test"
    (Contract
       {
         batch;
         contracting;
         acc = any;
         out = f64;
         a;
         b;
         init = None;
         a_scale;
         b_scale;
         segments;
       })

let offsets = Nx.create Nx.int64 [| 3 |] [| 0L; 1L; 4L |]

let contract_cases =
  let twenty = Gen.constant [| 20 |] in
  let c name f =
    case ~empty:false ~shape:twenty name (fun x -> Nx.reshape [| -1 |] (f x))
  in
  [
    c "a contraction of scaled operands" (fun x ->
        contract_of x ~a:[| 2; 2 |] ~b:[| 2; 2 |] ~batch:[||]
          ~contracting:[| (1, 0) |]
          ~a_scale:(Value (Nx.create f64 [| 2; 1 |] [| 1.5; 0.5 |]))
          ~b_scale:(Exponent (Nx.create Nx.uint8 [| 1; 2 |] [| 128; 126 |]))
          ());
    c "a contraction segmented by rows" (fun x ->
        contract_of x ~a:[| 4; 2 |] ~b:[| 2; 2; 3 |] ~batch:[||]
          ~contracting:[| (1, 1) |]
          ~segments:(Rows { offsets; a_axis = 0; b_axis = 0 })
          ());
    c "a contraction segmented along its pair" (fun x ->
        contract_of x ~a:[| 4; 2 |] ~b:[| 4; 3 |] ~batch:[||]
          ~contracting:[| (0, 0) |]
          ~segments:(Contracting { offsets; pair = 0 })
          ());
    c "a batched contraction of transposed operands" (fun x ->
        contract_of x ~a:[| 2; 3; 2 |] ~b:[| 2; 2 |]
          ~batch:[| (0, 1) |]
          ~contracting:[| (2, 0) |]
          ());
  ]

let real_cases =
  let unary name ?values f = case ?values name f in
  let binary name ?(values = Uniform [ (-2., 2.); (-2., 2.) ]) f =
    case ~shape:(rows 2) ~values name (two f)
  in
  let spaced = Spaced (-3., 0.5) and positive = Uniform [ (0.2, 3.) ] in
  [
    unary "neg" Nx.neg;
    unary "recip" ~values:positive Nx.recip;
    unary "abs" ~values:spaced Nx.abs;
    unary "sign" ~values:spaced Nx.sign;
    unary "sqrt" ~values:positive Nx.sqrt;
    unary "exp" Nx.exp;
    unary "exp2" Nx.exp2;
    unary "log" ~values:positive Nx.log;
    unary "log2" ~values:positive Nx.log2;
    unary "log1p" ~values:(Uniform [ (-0.8, 3.) ]) Nx.log1p;
    unary "expm1" Nx.expm1;
    unary "sin" Nx.sin;
    unary "cos" Nx.cos;
    unary "tan" ~values:(Uniform [ (-1.2, 1.2) ]) Nx.tan;
    unary "asin" ~values:(Uniform [ (-0.9, 0.9) ]) Nx.asin;
    unary "acos" ~values:(Uniform [ (-0.9, 0.9) ]) Nx.acos;
    unary "atan" Nx.atan;
    unary "sinh" Nx.sinh;
    unary "cosh" Nx.cosh;
    unary "tanh" Nx.tanh;
    unary "asinh" Nx.asinh;
    unary "acosh" ~values:(Uniform [ (1.1, 4.) ]) Nx.acosh;
    unary "atanh" ~values:(Uniform [ (-0.9, 0.9) ]) Nx.atanh;
    unary "erf" Nx.erf;
    unary "floor" ~values:spaced Nx.floor;
    unary "ceil" ~values:spaced Nx.ceil;
    unary "round" ~values:spaced Nx.round;
    unary "trunc" ~values:spaced Nx.trunc;
    unary "sigmoid" Nx.sigmoid;
    unary "softplus" ~values:spaced Nx.softplus;
    unary "copy and casts, as kinds" kinds;
    binary "add" Nx.add;
    binary "sub" Nx.sub;
    binary "mul" Nx.mul;
    binary "div" ~values:(Uniform [ (-2., 2.); (0.3, 2.) ]) Nx.div;
    binary "mod"
      ~values:(Spaced (0.1, 0.37))
      (fun a b -> Nx.mod_ a (Nx.add_s b 2.));
    binary "pow" ~values:(Uniform [ (0.3, 3.); (-2., 2.) ]) Nx.pow;
    binary "atan2" ~values:(Uniform [ (-2., 2.); (0.3, 2.) ]) Nx.atan2;
    binary "maximum" ~values:spaced Nx.maximum;
    binary "minimum" ~values:spaced Nx.minimum;
    binary "hypot" ~values:(Uniform [ (0.3, 2.); (0.3, 2.) ]) Nx.hypot;
    binary "where" ~values:spaced (fun a b ->
        Nx.where (Nx.less a b) (Nx.mul a b) (Nx.sin b));
    case ~shape:(rows 3) "fma" (three Nx.fma);
    case ~empty:false ~shape:padded_shape
      "a map of several outputs, a padded load and coordinates" multi;
    case ~shape:rows_wide "a map of twenty operands" wide_map;
    case ~shape:rows_wide "a sum over twenty operands"
      (wide_reduce (Monoid (Sum, 0, f64)));
    case ~empty:false
      ~shape:Gen.(map (fun n -> [| wide; n + 1 |]) (int_range 0 3))
      ~values:(Spaced (-3., 0.1))
      "a max over twenty operands"
      (wide_reduce (Monoid (Max, 0, f64)));
    case ~shape:matrix "sum" (Nx.sum ~axes:[ 1 ]);
    case ~shape:matrix "sum and its tangent in one loop" (fun x -> Nx.sum x);
    case ~shape:matrix "two sums of two outputs, in another order" sums;
    case ~shape:matrix "prod" (Nx.prod ~axes:[ 1 ]);
    case ~shape:matrix
      ~values:(Uniform [ (-1., 1.) ])
      "prod over both axes"
      (fun x -> Nx.prod x);
    case ~empty:false ~shape:full ~values:spaced "max" (Nx.max ~axes:[ 1 ]);
    case ~empty:false ~shape:full ~values:spaced "min" (Nx.min ~axes:[ 0 ]);
    case ~shape:matrix "mean" (Nx.mean ~axes:[ 1 ]);
    case ~shape:matrix "var" (Nx.var ~axes:[ 1 ]);
    case ~shape:matrix "logsumexp" (Nx.logsumexp ~axes:[ 1 ]);
    case ~empty:false ~shape:full ~values:spaced "an arg reduction's extreme"
      (fun x ->
        let (y, _), () = reduce1 [ Arg (Max, 0, f64) ] x in
        y);
    case ~shape:matrix "moments" (fun x ->
        let (m, var), () = reduce1 [ Moments (0, f64) ] x in
        Nx.stack [ m; var ]);
    case ~shape:matrix "cumsum" (Nx.cumsum ~axis:1);
    case ~shape:matrix "cumprod" (Nx.cumprod ~axis:1);
    case ~shape:matrix ~values:spaced "cummax" (Nx.cummax ~axis:1);
    case ~shape:matrix ~values:spaced "cummin" (Nx.cummin ~axis:0);
    case ~shape:matrix "logcumsumexp" (Nx.logcumsumexp ~axis:1);
    case ~shape:matrix "a running moments" (fun x ->
        let m, var = scan1 (Moments (0, f64)) x in
        Nx.stack [ m; var ]);
    case ~shape:matrix ~values:spaced "a running arg's extreme" (fun x ->
        fst (scan1 (Arg (Min, 0, f64)) x));
    case ~empty:false ~shape:padded_shape "a running sum from a carry" (fun x ->
        let c, r = carried x in
        scan1 ~carry:c (Monoid (Sum, 0, f64)) r);
    case ~empty:false ~shape:padded_shape
      ~values:(Uniform [ (-1.5, 1.5) ])
      "a running product from a carry"
      (fun x ->
        let c, r = carried x in
        scan1 ~carry:c (Monoid (Prod, 0, f64)) r);
    case ~empty:false ~shape:padded_shape ~values:spaced
      "a running max from a carry" (fun x ->
        let c, r = carried x in
        scan1 ~carry:c (Monoid (Max, 0, f64)) r);
    case ~shape:matrix "take" (Nx.take ~axis:1 (idx [| 2; 0; 7; 1 |]));
    case ~shape:matrix "take_along_axis" (fun x ->
        let n = Nx.dim 1 x in
        Nx.take_along_axis ~axis:1
          (Nx.broadcast_to
             [| Nx.dim 0 x; n |]
             (idx (Array.init n (fun k -> k * 3 mod Int.max 1 n))))
          x);
    case ~shape:(rows 2) "scatter add" (scatter Add);
    case ~shape:(rows 2) "scatter set" (scatter Set);
    case ~shape:(rows 2) ~values:spaced "scatter max" (scatter Max);
    case ~shape:(rows 2) ~values:spaced "scatter min" (scatter Min);
    case ~shape:matrix ~values:spaced "sort" (fun x -> fst (Nx.sort ~axis:1 x));
    case ~empty:false
      ~shape:Gen.(map (fun n -> [| n + 2 |]) (int_range 0 4))
      ~values:spaced "top_k"
      (fun x -> fst (Nx.top_k ~k:2 x));
    case ~empty:false ~shape:padded_shape "pad" (Nx.pad [| (1, 0); (2, 1) |] 3.);
    case ~empty:false
      ~shape:Gen.(map (fun n -> [| 2; n + 3 |]) (int_range 0 3))
      "extract_patches"
      (fun x ->
        Nx.extract_patches ~kernel_size:[| 2 |] ~stride:[| 1 |]
          ~dilation:[| 1 |]
          ~padding:[| (1, 0) |]
          x);
    case ~empty:false
      ~shape:Gen.(map (fun l -> [| 2; 2; l + 1 |]) (int_range 0 3))
      "combine_patches"
      (fun x ->
        let l = Nx.dim 2 x in
        Nx.combine_patches
          ~output_size:[| l + 1 |]
          ~kernel_size:[| 2 |] ~stride:[| 1 |] ~dilation:[| 1 |]
          ~padding:[| (0, 0) |]
          x);
    case ~empty:false ~shape:full "set, an assembly over its operand" (fun x ->
        Nx.set [ R (0, 1) ] (Nx.slice [ R (0, 1) ] (Nx.sin (row 0 x))) (row 0 x));
    case ~empty:false ~shape:square "a strided slice, a diagonal" Nx.diagonal;
    case ~shape:matrix "a reversed strided slice" (fun x ->
        Nx.slice [ A; Rs (-1, -100, -2) ] x);
    case ~shape:(rows 2) "concatenate" (fun x ->
        Nx.concatenate ~axis:0 [ row 1 x; Nx.sin (row 0 x); row 1 x ]);
    case ~shape:cube "matmul" (fun x ->
        Nx.matmul (Nx.transpose (row 0 x)) (row 1 x));
    case ~shape:cube "contract from an init" (fun x ->
        let a = Nx.transpose (row 0 x) and b = row 1 x in
        let n = Nx.dim 1 b in
        Nx.contract
          ~init:(Nx.sin (Nx.slice [ A; R (0, n) ] (Nx.matmul a b)))
          f64 pattern a b);
    case ~shape:matrix "copy" (fun x -> Nx.copy (Nx.sin x));
    case ~shape:matrix "moves" (fun x ->
        Nx.reshape [| -1 |] (Nx.flip ~axes:[ 0 ] (Nx.transpose x)));
    case ~shape:matrix "broadcast" (fun x ->
        Nx.broadcast_to [| 2; Nx.dim 0 x; Nx.dim 1 x |] (Nx.exp x));
    case ~shape:matrix "place" (fun x ->
        Nx.place Nx.Host.on (Nx.sin (Nx.place Count.on x)));
    case ~shape:matrix "check" (fun x ->
        Nx.check T.tensor (Nx.less_s x 100.) x (fun _ _ -> Failure "check");
        Nx.cos x);
  ]

(* Factorisations

   Each case builds a well-conditioned matrix from its draws in [-1, 1]: one
   whose factors are continuous there, with spaced eigenvalues and singular
   values. A vector defined up to its sign is read squared. *)

let tall_or_wide =
  Gen.(
    map
      (fun (m, n) -> [| m + 1; n + 1 |])
      (pair (int_range 0 3) (int_range 0 3)))

let flat xs = Nx.concatenate ~axis:0 (List.map (Nx.reshape [| -1 |]) xs)
let adjoint x = Nx.conjugate (Nx.matrix_transpose x)
let half x = Nx.mul x (Nx.full_like x (D.of_float (Nx.dtype x) 0.5))

(* [x / 2] plus a spaced diagonal [3, 6, …] of its shape and a fixed matrix of
   no special form, so that no draw, zero included, makes a matrix at which a
   factorisation changes its branch, as Householder reflections do at a column
   already reduced. *)
let spaced x =
  let m = Nx.dim (-2) x and n = Nx.dim (-1) x in
  let k = Int.min m n in
  let d = Nx.mul_s (Nx.cast f64 (Nx.arange Nx.int32 1 (k + 1) 1)) 3. in
  let diag = Nx.mul (Nx.eye f64 k) (Nx.unsqueeze ~axes:[ 0 ] d) in
  let generic =
    Nx.create f64 [| m; n |]
      (Array.init (m * n) (fun k ->
           0.3
           *. Float.sin
                ((1.7 *. Float.of_int (k / n))
                +. (2.9 *. Float.of_int (k mod n))
                +. 0.4)))
  in
  let fixed = Nx.add generic (Nx.pad [| (0, m - k); (0, n - k) |] 0. diag) in
  Nx.add (half x) (Nx.cast (Nx.dtype x) fixed)

let symmetric x = half (Nx.add x (adjoint x))

let positive x =
  let n = Nx.dim (-1) x in
  let shift = Nx.cast (Nx.dtype x) (Nx.mul_s (Nx.eye f64 n) (Float.of_int n)) in
  Nx.add (Nx.matmul x (adjoint x)) shift

(* [x]'s rows moved one down, the last first: an LU of it pivots. *)
let rotated x =
  let m = Nx.dim (-2) x and r = Nx.ndim x in
  if m < 2 then x
  else
    let rows a b =
      Nx.slice
        (List.init r (fun i -> if i = r - 2 then Nx.R (a, b) else Nx.A))
        x
    in
    Nx.concatenate ~axis:(r - 2) [ rows (m - 1) m; rows 0 (m - 1) ]

(* [x]'s squared magnitudes, which a vector's sign or phase leaves alone. *)
let squared x = Nx.mul x (Nx.conjugate x)
let batched = Gen.(map (fun n -> [| 2; n + 1; n + 1 |]) (int_range 0 2))

let wider =
  Gen.(
    map
      (fun (m, d) -> [| m + 1; m + d + 2 |])
      (pair (int_range 0 1) (int_range 0 1)))

let factor_cases =
  let unit = Uniform [ (-1., 1.) ] in
  let m name shape f = case ~empty:false ~shape ~values:unit name f in
  let rhs x = Nx.reshape [| Nx.dim 0 x; 1 |] (Nx.sin (Nx.sum ~axes:[ 1 ] x)) in
  [
    m "cholesky" square (fun x -> Nx.cholesky (positive x));
    (* The upper triangle, which Cholesky does not read, moves too. *)
    m "cholesky of a lower triangle" square (fun x ->
        Nx.cholesky (Nx.add (positive x) (Nx.triu ~k:1 x)));
    m "batched cholesky" batched (fun x -> Nx.cholesky (positive x));
    m "cholesky of the upper triangle" square (fun x ->
        Nx.cholesky ~triangle:Upper (positive x));
    m "lu" tall_or_wide (fun x ->
        let _, l, u = Nx.lu (spaced x) in
        flat [ l; u ]);
    m "qr" tall_or_wide (fun x ->
        let q, r = Nx.qr (spaced x) in
        flat [ q; r ]);
    m "lu, pivoting" tall_or_wide (fun x ->
        let _, l, u = Nx.lu (rotated (spaced x)) in
        flat [ l; u ]);
    m "batched qr" batched (fun x ->
        let q, r = Nx.qr (spaced x) in
        flat [ q; r ]);
    m "qr complete of a wide matrix" wider (fun x ->
        let q, r = Nx.qr ~factors:Complete (spaced x) in
        flat [ q; r ]);
    m "svd's values" tall_or_wide (fun x ->
        let _, s, _ = Nx.svd ~vectors:false f64 (spaced x) in
        s);
    m "svd" tall_or_wide (fun x ->
        let u, s, vh = Nx.svd f64 (spaced x) in
        flat [ s; Nx.square u; Nx.square vh ]);
    m "eigh's values" square (fun x ->
        fst (Nx.eigh ~vectors:false f64 (spaced (symmetric x))));
    m "eigh" square (fun x ->
        let w, v = Nx.eigh f64 (spaced (symmetric x)) in
        flat [ w; Nx.square v ]);
    (* The triangle eigh does not read moves too. *)
    m "eigh of a lower triangle" square (fun x ->
        let w, v = Nx.eigh f64 (spaced x) in
        flat [ w; Nx.square v ]);
    m "eigh of the upper triangle" square (fun x ->
        let w, v = Nx.eigh ~triangle:Upper f64 (spaced (symmetric x)) in
        flat [ w; Nx.square v ]);
    m "eig's values" square (fun x ->
        let w, _ = Nx.eig ~vectors:false c128 (spaced (Nx.mul_s x 0.3)) in
        Nx.real f64 w);
    m "eig" square (fun x ->
        let w, v = Nx.eig c128 (spaced (Nx.mul_s x 0.3)) in
        flat [ Nx.real f64 w; Nx.real f64 (squared v) ]);
    m "solve_triangular" square (fun x ->
        let a = spaced (Nx.tril x) in
        flat
          [
            Nx.solve_triangular a (rhs x);
            Nx.solve_triangular ~transpose:Transpose a (rhs x);
          ]);
    m "solve_triangular, upper" square (fun x ->
        let a = spaced x in
        flat
          [
            Nx.solve_triangular ~triangle:Upper a (rhs x);
            Nx.solve_triangular ~triangle:Upper ~transpose:Transpose a (rhs x);
            Nx.solve_triangular ~triangle:Upper ~transpose:Adjoint a (rhs x);
          ]);
    m "solve_triangular, unit diagonal" square (fun x ->
        flat
          [
            Nx.solve_triangular ~triangle:Upper ~unit_diag:true x (rhs x);
            Nx.solve_triangular ~unit_diag:true ~transpose:Transpose x (rhs x);
          ]);
    m "solve" square (fun x -> Nx.solve (spaced x) (rhs x));
    m "inv" square (fun x -> Nx.inv (spaced x));
    m "det" square (fun x -> Nx.det (spaced x));
    m "solve, inv and det, pivoting" square (fun x ->
        let a = rotated (spaced x) in
        flat [ Nx.solve a (rhs x); Nx.inv a; Nx.det a ]);
    m "slogdet" square (fun x -> snd (Nx.slogdet f64 (spaced x)));
  ]

(* [spaced z] with a fixed imaginary part of no special form too: a complex
   reflection changes its branch at a real column. *)
let zspaced z =
  let m = Nx.dim (-2) z and n = Nx.dim (-1) z in
  let im =
    Nx.create f64 [| m; n |]
      (Array.init (m * n) (fun k ->
           0.2 *. Float.cos ((1.3 *. Float.of_int k) +. 0.7)))
  in
  Nx.add (spaced z) (Nx.complex c128 ~re:(Nx.zeros f64 [| m; n |]) ~im)

let complex_factor_cases =
  let unit = Uniform [ (-1., 1.); (-1., 1.) ] in
  let zsq = Gen.(map (fun n -> [| 2; n + 1; n + 1 |]) (int_range 0 2)) in
  let ztw =
    Gen.(
      map
        (fun (m, n) -> [| 2; m + 1; n + 1 |])
        (pair (int_range 0 2) (int_range 0 2)))
  in
  let m name shape f =
    case ~empty:false ~shape ~values:unit ~z:(complex, f) name (on_complex f)
  in
  let rhs z = Nx.reshape [| Nx.dim 0 z; 1 |] (Nx.sin (Nx.sum ~axes:[ 1 ] z)) in
  [
    m "complex cholesky" zsq (fun z -> Nx.cholesky (positive z));
    m "complex cholesky of the upper triangle" zsq (fun z ->
        Nx.cholesky ~triangle:Upper (Nx.add (positive z) (Nx.tril ~k:(-1) z)));
    m "complex eig" zsq (fun z ->
        let w, v = Nx.eig c128 (zspaced (half z)) in
        flat [ w; squared v ]);
    m "complex svd" ztw (fun z ->
        let u, s, vh = Nx.svd f64 (zspaced z) in
        let s = Nx.cast c128 s in
        let usvh = Nx.matmul (Nx.mul u (Nx.unsqueeze ~axes:[ 0 ] s)) vh in
        flat [ s; squared u; squared vh; usvh ]);
    m "complex eigh" zsq (fun z ->
        let w, v = Nx.eigh f64 (zspaced z) in
        flat [ Nx.cast c128 w; squared v ]);
    m "complex qr complete of a wide matrix"
      Gen.(map (fun n -> [| 2; 2; n + 3 |]) (int_range 0 1))
      (fun z ->
        let q, r = Nx.qr ~factors:Complete (zspaced z) in
        flat [ q; r ]);
    m "complex solve_triangular, unit diagonal and upper" zsq (fun z ->
        flat
          [
            Nx.solve_triangular ~triangle:Upper ~unit_diag:true z (rhs z);
            Nx.solve_triangular ~triangle:Upper ~transpose:Adjoint (zspaced z)
              (rhs z);
          ]);
    m "complex lu" ztw (fun z ->
        let _, l, u = Nx.lu (zspaced z) in
        flat [ l; u ]);
    m "complex qr" ztw (fun z ->
        let q, r = Nx.qr (zspaced z) in
        flat [ q; r ]);
    m "complex svd's values" ztw (fun z ->
        let _, s, _ = Nx.svd ~vectors:false f64 (zspaced z) in
        Nx.cast c128 s);
    m "complex eigh's values" zsq (fun z ->
        Nx.cast c128 (fst (Nx.eigh ~vectors:false f64 (zspaced (symmetric z)))));
    m "complex solve_triangular" zsq (fun z ->
        let a = spaced (Nx.tril z) in
        flat
          [
            Nx.solve_triangular a (rhs z);
            Nx.solve_triangular ~transpose:Transpose a (rhs z);
            Nx.solve_triangular ~transpose:Adjoint a (rhs z);
          ]);
    m "complex solve" zsq (fun z -> Nx.solve (zspaced z) (rhs z));
    m "complex inv" zsq (fun z -> Nx.inv (zspaced z));
    m "complex det" zsq (fun z -> Nx.det (zspaced z));
  ]

let complex_cases =
  let z ?(values = Uniform [ (0.2, 2.); (0.2, 1.) ]) name f =
    case ~shape:(rows 2) ~values ~z:(complex, f) name (on_complex f)
  in
  (* Two complex operands, rows 0 and 1 of one value. *)
  let pair x =
    Nx.stack
      [
        Nx.complex c128 ~re:(row 0 x) ~im:(row 1 x);
        Nx.complex c128 ~re:(row 2 x) ~im:(row 3 x);
      ]
  in
  let zz name f =
    let g z = f (row 0 z) (row 1 z) in
    case ~shape:(rows 4)
      ~values:(Uniform [ (0.3, 2.); (0.2, 1.); (0.3, 1.5); (-1., 1.) ])
      ~z:(pair, g) name
      (fun x -> parts (g (pair x)))
  in
  [
    z "complex neg" Nx.neg;
    z "complex recip" Nx.recip;
    z "complex sign" Nx.sign;
    z "complex sqrt" Nx.sqrt;
    z "complex exp" Nx.exp;
    z "complex exp2" Nx.exp2;
    z "complex log" Nx.log;
    z "complex log2" Nx.log2;
    z "complex sin" Nx.sin;
    z "complex cos" Nx.cos;
    z "complex tan" Nx.tan;
    z "complex asin" Nx.asin;
    z "complex acos" Nx.acos;
    z "complex atan" Nx.atan;
    z "complex sinh" Nx.sinh;
    z "complex cosh" Nx.cosh;
    z "complex tanh" Nx.tanh;
    z "complex asinh" Nx.asinh;
    z "complex acosh" Nx.acosh;
    z "complex atanh" Nx.atanh;
    z "conjugate" Nx.conjugate;
    zz "complex add" Nx.add;
    zz "complex sub" Nx.sub;
    zz "complex mul" Nx.mul;
    zz "complex div" Nx.div;
    zz "complex pow" Nx.pow;
    zz "complex matmul" Nx.matmul;
    z "complex prod" (fun z -> Nx.prod z);
    zz "complex where" (fun a b ->
        Nx.where (Nx.less_s (Nx.real f64 a) 1.17) a b);
    case ~shape:(rows 2)
      ~values:(Uniform [ (0.2, 2.); (0.2, 1.) ])
      "magnitude and angle"
      (fun x ->
        let z = complex x in
        Nx.stack [ Nx.magnitude f64 z; Nx.angle f64 z ]);
    case "a real cast to complex" (fun x ->
        parts (Nx.mul (Nx.cast c128 x) (Nx.cast c128 x)));
    case ~shape:(rows 2) "a complex cast to real" (fun x ->
        Nx.cast f64 (Nx.exp (complex x)));
    case ~shape:(rows 2)
      ~z:(complex, fun z -> Nx.fft z)
      "fft"
      (fun x -> parts (Nx.fft (complex x)));
    case "rfft" (fun x -> parts (Nx.rfft c128 x));
    case ~shape:matrix "rfft along a leading axis" (fun x ->
        parts (Nx.rfft c128 ~axes:[ 0 ] x));
    case ~shape:matrix "rfft over two axes" (fun x ->
        parts (Nx.rfft c128 ~axes:[ 0; 1 ] x));
    case ~shape:(rows 2)
      ~z:(complex, fun z -> Nx.ifft z)
      "ifft"
      (fun x -> parts (Nx.ifft (complex x)));
    case ~empty:false
      ~shape:Gen.(map (fun n -> [| 2; n + 1; 2 |]) (int_range 0 3))
      "irfft along a leading axis"
      (fun x -> Nx.irfft f64 ~axes:[ 0 ] (complex x));
    case ~empty:false
      ~shape:Gen.(map (fun n -> [| 2; n + 1; 3 |]) (int_range 0 2))
      "irfft over two axes"
      (fun x -> Nx.irfft f64 ~axes:[ 0; 1 ] (complex x));
    case ~empty:false
      ~shape:Gen.(map (fun n -> [| 2; n + 1 |]) (int_range 0 4))
      "irfft"
      (fun x -> Nx.irfft f64 (complex x));
    (let f z =
       Nx.concatenate ~axis:0 [ Nx.sum ~keepdims:true z; Nx.cumprod z ]
     in
     case ~shape:(rows 2) ~z:(complex, f) "complex sum and cumprod" (fun x ->
         parts (f (complex x))));
  ]
