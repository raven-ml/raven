(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.jvp: tangents against central differences at float64 for every kind of
   scalar program and every operation that has a derivative, complex functions
   as maps of their parts, the kinks' tangents, jvp of a compiled function
   against jvp, and a map's tangent computed with its primal in one kernel. *)

open Windtrap
open Compose
module Rune = Rune2
module D = Nx.Dtype
module L = Nx_array.Layout
module P = Nx_kernel.Prog
module T = Nx.Ptree
module C = Nx_support.Counting
module Count = (val Nx.devices ~kernels:(module C) [ Nx_support.memory 0 ])

let f64 = Nx.float64
let c128 = Nx.complex128
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
  f :
    (float, D.float64_elt, Nx.host) Nx.t -> (float, D.float64_elt, Nx.host) Nx.t;
}

let case ?(empty = true) ?(values = Uniform [ (-2., 2.) ])
    ?(shape = Gen.(map (fun n -> [| n |]) (int_range 0 5))) name f =
  { name; empty; shape; values; f }

let rows k = Gen.(map (fun n -> [| k; n |]) (int_range 0 4))

let matrix =
  Gen.(map (fun (m, n) -> [| m; n |]) (pair (int_range 0 3) (int_range 0 4)))

(* Matrices whose rows and columns are never empty, for extremes. *)
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

let h = 1e-6

let law c (s, xs, ts) =
  if c.empty then cover "a zero-size shape" (numel s = 0);
  cover "a shape of several elements" (numel s > 1);
  let x = Nx.create f64 s xs and t = Nx.create f64 s ts in
  let y, dy = Rune.jvp' c.f x t in
  equal ~msg:"primal" bits_t (bits (c.f x)) (bits y);
  let at e =
    Nx.to_array
      (c.f (Nx.create f64 s (Array.mapi (fun k x -> x +. (e *. ts.(k))) xs)))
  in
  let plus = at h and minus = at (-.h) in
  let dy = Nx.to_array dy in
  equal ~msg:"tangent count" int (Array.length plus) (Array.length dy);
  Array.iteri
    (fun k d ->
      let fd = (plus.(k) -. minus.(k)) /. (2. *. h) in
      (* Differences of non-finite values say nothing. *)
      if
        Float.is_finite plus.(k)
        && Float.is_finite minus.(k)
        && Float.is_finite fd
      then
        less
          ~msg:(Printf.sprintf "tangent %d: %h, differences %h" k d fd)
          float_exact
          ~than:(1e-5 *. (1. +. Float.abs fd))
          (Float.abs (d -. fd)))
    dy

(* Helpers over cases' operands *)

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

(* A program of twenty operands, [Σ_k x_k Π_(j<k) sin x_j], past the arrays
   one block of nx.cpu's holds: its derivative is one loop of forty loads. *)
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

let complex_cases =
  let z ?(values = Uniform [ (0.2, 2.); (0.2, 1.) ]) name f =
    case ~shape:(rows 2) ~values name (on_complex f)
  in
  let zz name f =
    case ~shape:(rows 4)
      ~values:(Uniform [ (0.3, 2.); (0.2, 1.); (0.3, 1.5); (-1., 1.) ])
      name
      (fun x ->
        let a = Nx.complex c128 ~re:(row 0 x) ~im:(row 1 x)
        and b = Nx.complex c128 ~re:(row 2 x) ~im:(row 3 x) in
        parts (f a b))
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
    case ~shape:(rows 2) "fft" (fun x -> parts (Nx.fft (complex x)));
    case "rfft" (fun x -> parts (Nx.rfft c128 x));
    case ~empty:false
      ~shape:Gen.(map (fun n -> [| 2; n + 1 |]) (int_range 0 4))
      "irfft"
      (fun x -> Nx.irfft f64 (complex x));
    case ~shape:(rows 2) "complex sum and cumprod" (fun x ->
        let z = complex x in
        parts (Nx.concatenate ~axis:0 [ Nx.sum ~keepdims:true z; Nx.cumprod z ]));
  ]

let tangent_props =
  List.map (fun c -> prop c.name (gen c) (law c)) (real_cases @ complex_cases)

(* Values the conventions state *)

let floats = array (float 1e-12)
let tangent f x t = Nx.to_array (snd (Rune.jvp' f x t))

(* JAX's: [abs] passes the tangent at either zero; tied operands of an extreme
   share it by halves, and a NaN takes none; tied elements along a reduction's
   axes share it equally; a running extreme takes the first element's. *)
let test_kinks () =
  equal ~msg:"abs" floats [| 1.; 1.; 1.; -1. |]
    (tangent Nx.abs (v [| 0.; -0.; 2.; -2. |]) (v [| 1.; 1.; 1.; 1. |]));
  let pair = T.(pair tensor tensor) in
  let extreme f x y tx ty =
    Nx.to_array
      (snd
         (Rune.jvp pair T.tensor (fun (a, b) -> f a b) (v x, v y) (v tx, v ty)))
  in
  equal ~msg:"maximum" floats [| 2.; 1.; 3.; 0. |]
    (extreme Nx.maximum
       [| 1.; 2.; 0.; Float.nan |]
       [| 1.; 0.; 2.; 1. |] [| 1.; 1.; 1.; 1. |] [| 3.; 3.; 3.; 3. |]);
  equal ~msg:"minimum" floats [| 2.; 3.; 1.; 0. |]
    (extreme Nx.minimum [| 1.; 2.; 0.; 1. |]
       [| 1.; 0.; 2.; Float.nan |]
       [| 1.; 1.; 1.; 1. |] [| 3.; 3.; 3.; 3. |]);
  equal ~msg:"max over an axis" floats [| 3. |]
    (tangent (Nx.max ~keepdims:true) (v [| 1.; 3.; 3. |]) (v [| 1.; 2.; 4. |]));
  equal ~msg:"min over an axis" floats [| 3. |]
    (tangent (Nx.min ~keepdims:true) (v [| 2.; 1.; 1. |]) (v [| 5.; 2.; 4. |]));
  equal ~msg:"cummax" floats [| 1.; 2.; 2.; 2. |]
    (tangent (Nx.cummax ~axis:0)
       (v [| 1.; 3.; 3.; 2. |])
       (v [| 1.; 2.; 4.; 8. |]));
  equal ~msg:"pow at a zero base" floats [| 0.; 0.; 0. |]
    (tangent
       (fun y -> Nx.pow (v [| 0.; 0.; 0. |]) y)
       (v [| -1.; 2.; 0.5 |])
       (v [| 1.; 1.; 1. |]));
  equal ~msg:"pow at a zero exponent" floats [| 0. |]
    (Nx.to_array
       (snd
          (Rune.jvp pair T.tensor
             (fun (a, b) -> Nx.pow a b)
             (v [| 0. |], v [| 0. |])
             (v [| 1. |], v [| 0. |]))));
  equal ~msg:"a real sign" floats [| 0.; 0. |]
    (tangent Nx.sign (v [| 0.; 2. |]) (v [| 1.; 1. |]))

(* Values the derivatives' definitions give at the edges: zeros, infinities,
   narrow dtypes whose rounding of a primal must not reach its tangent, a
   reduction into another dtype, ties stated by the kinks' rules. *)
let test_edges () =
  let tangent' f x t = Nx.to_array (Nx.cast f64 (snd (Rune.jvp' f x t))) in
  let close tol = array (float tol) in
  equal ~msg:"prod at a zero" floats [| 6. |]
    (tangent (Nx.prod ~keepdims:true) (v [| 0.; 2.; 3. |]) (v [| 1.; 0.; 0. |]));
  equal ~msg:"cumprod at a zero" floats [| 1.; 2.; 6. |]
    (tangent Nx.cumprod (v [| 0.; 2.; 3. |]) (v [| 1.; 0.; 0. |]));
  equal ~msg:"cumprod's first at an infinity" floats [| 1. |]
    [| (tangent Nx.cumprod (v [| Float.infinity; 1. |]) (v [| 1.; 0. |])).(0) |];
  equal ~msg:"logcumsumexp after -inf" floats [| 1.; 2.; 5. |]
    (tangent Nx.logcumsumexp
       (v [| Float.neg_infinity; Float.neg_infinity; 0. |])
       (v [| 1.; 3.; 5. |]));
  equal ~msg:"logcumsumexp from -inf" floats [| 0.; 1. |]
    (tangent Nx.logcumsumexp (v [| Float.neg_infinity; 0. |]) (v [| 0.; 1. |]));
  let narrow dt xs = Nx.cast dt (v xs) in
  let bf = Nx.bfloat16 and h = Nx.float16 in
  equal ~msg:"logsumexp shifted, bfloat16" (array float_exact) [| 1. |]
    (tangent'
       (Nx.logsumexp ~keepdims:false)
       (narrow bf [| 100.; 100. |])
       (narrow bf [| 1.; 1. |]));
  equal ~msg:"var shifted, float16" (array float_exact) [| 0. |]
    (tangent'
       (fun x -> Nx.reshape [| 1 |] (Nx.var x))
       (narrow h [| 2000.; 2001. |])
       (narrow h [| 1.; 1. |]));
  equal ~msg:"var along one, float16" (array float_exact) [| 0.5 |]
    (tangent'
       (fun x -> Nx.reshape [| 1 |] (Nx.var x))
       (narrow h [| 2000.; 2001. |])
       (narrow h [| 0.; 1. |]));
  equal ~msg:"prod in its carrier, float16" (close 1.) [| 300. |]
    (tangent'
       (fun x -> Nx.reshape [| 1 |] (Nx.prod x))
       (narrow h [| 300.; 300.; 1. /. 300.; 1. /. 300. |])
       (narrow h [| 0.; 0.; 1.; 0. |]));
  (* A float64 output reduced into float32, and into int32. *)
  let into dt x =
    fst
      (Nx.Prim.eval ~by:"test"
         (Reduce
            {
              layout = contiguous (Nx.shape x);
              axes = [| 0 |];
              prog = identity;
              reductions = [ Monoid (Max, 0, dt) ];
              loads = [| Plain x |];
            }))
  in
  equal ~msg:"max into float32" (array float_exact) [| 2. |]
    (tangent'
       (fun x -> Nx.reshape [| 1 |] (into Nx.float32 x))
       (v [| 0.1; 0.2 |])
       (v [| 1.; 2. |]));
  let prod_into dt x =
    fst
      (Nx.Prim.eval ~by:"test"
         (Reduce
            {
              layout = contiguous (Nx.shape x);
              axes = [| 0 |];
              prog = identity;
              reductions = [ Monoid (Prod, 0, dt) ];
              loads = [| Plain x |];
            }))
  in
  equal ~msg:"prod into float32, in the output's carrier" (float 1e24) 1e30
    (tangent'
       (fun x -> Nx.reshape [| 1 |] (prod_into Nx.float32 x))
       (v [| 1e30; 1e30; 1e-30; 1e-30 |])
       (v [| 0.; 0.; 1.; 0. |])).(0);
  equal ~msg:"max into int32" (array int32) [| 0l |]
    (Nx.to_array
       (snd
          (Rune.jvp'
             (fun x -> Nx.reshape [| 1 |] (into Nx.int32 x))
             (v [| 1.5; 2.5 |])
             (v [| 1.; 2. |]))));
  (* Ties: a scatter's target with the update that ties it, an arg's extreme, a
     running maximum's carry against the element tying it. *)
  let s = T.(pair tensor tensor) in
  let _, t =
    Rune.jvp s T.tensor
      (fun (into, u) -> Nx.scatter ~combine:Max ~axis:0 (idx [| 1; 0 |]) u into)
      (v [| 1.; 5. |], v [| 5.; 0.5 |])
      (v [| 1.; 2. |], v [| 4.; 8. |])
  in
  equal ~msg:"scatter max" floats [| 1.; 3. |] (Nx.to_array t);
  let arg x =
    Nx.reshape [| 1 |] (fst (fst (reduce1 [ Arg (Max, 0, f64) ] x)))
  in
  equal ~msg:"arg's extreme" floats [| 3. |]
    (tangent arg
       (Nx.reshape [| 1; 3 |] (v [| 1.; 3.; 3. |]))
       (Nx.reshape [| 1; 3 |] (v [| 1.; 2.; 4. |])));
  let running x =
    let c, r = carried x in
    Nx.reshape [| 2 |] (scan1 ~carry:c (Monoid (Max, 0, f64)) r)
  in
  equal ~msg:"running max from a tying carry" floats [| 9.; 9. |]
    (tangent running
       (Nx.reshape [| 1; 3 |] (v [| 3.; 3.; 1. |]))
       (Nx.reshape [| 1; 3 |] (v [| 9.; 5.; 7. |])));
  (* What has no derivative raises. *)
  invalid ~sub:"Rune.jvp': Cholesky has no derivative yet" (fun () ->
      Rune.jvp' Nx.cholesky (Nx.eye f64 2) (Nx.eye f64 2));
  invalid ~sub:"Rune.jvp': a contraction's scale has a tangent" (fun () ->
      Rune.jvp'
        (fun s ->
          Nx.Prim.eval ~by:"test"
            (Contract
               {
                 batch = [||];
                 contracting = [| (1, 0) |];
                 acc = any;
                 out = f64;
                 a = v [| 1.; 2. |] |> Nx.reshape [| 1; 2 |];
                 b = Nx.reshape [| 2; 1 |] (v [| 3.; 4. |]);
                 init = None;
                 a_scale = Some (Value s);
                 b_scale = None;
                 segments = None;
               }))
        (Nx.reshape [| 1; 2 |] (v [| 1.; 1. |]))
        (Nx.reshape [| 1; 2 |] (v [| 1.; 1. |])))

(* A tangent is the directional derivative: [f'(z) v] for a holomorphic [f],
   [conj v] for the conjugate, [Re (conj z v) / |z|] for the modulus. *)
let test_complex () =
  let z =
    Nx.create c128 [| 2 |] [| { re = 0.5; im = -1. }; { re = -2.; im = 0.25 } |]
  in
  let t =
    Nx.create c128 [| 2 |] [| { re = 1.; im = 2. }; { re = -0.5; im = 1. } |]
  in
  let close = array (pair (float 1e-12) (float 1e-12)) in
  let parts z =
    Array.map (fun (c : Complex.t) -> (c.re, c.im)) (Nx.to_array z)
  in
  equal ~msg:"exp" close
    (parts (Nx.mul (Nx.exp z) t))
    (parts (snd (Rune.jvp' Nx.exp z t)));
  equal ~msg:"mul by a constant" close
    (parts (Nx.mul z t))
    (parts (snd (Rune.jvp' (fun w -> Nx.mul w z) z t)));
  equal ~msg:"conjugate" close
    (parts (Nx.conjugate t))
    (parts (snd (Rune.jvp' Nx.conjugate z t)));
  let expected =
    Array.map2
      (fun (z : Complex.t) (t : Complex.t) ->
        ((z.re *. t.re) +. (z.im *. t.im)) /. Complex.norm z)
      (Nx.to_array z) (Nx.to_array t)
  in
  equal ~msg:"magnitude" floats expected
    (Nx.to_array (snd (Rune.jvp' (Nx.magnitude f64) z t)))

(* jvp of a compiled function *)

type jit_case =
  | Test_case :
      (float, 's) Nx.dtype
      * int array
      * operand
      * operand
      * operand
      * operand
      * expr
      -> jit_case

let jit_cases =
  let open Gen in
  (let* s = shape in
   let* e = expr ~rank:(Array.length s) in
   let* x = operand s
   and+ y = operand s
   and+ tx = operand s
   and+ ty = operand s in
   let+ (D.Any dt) = of_list [ D.Any Nx.float32; Any f64 ] in
   match D.kind dt with
   | Float -> Test_case (dt, s, x, y, tx, ty, e)
   | _ -> assert false)
  |> with_pp (fun ppf (Test_case (dt, s, x, y, _, _, e)) ->
      Format.fprintf ppf "%a %a, x %a, y %a: %a" Nx.pp_dtype dt pp_shape s
        pp_view x.view pp_view y.view pp e)

let law_jit (Test_case (dt, s, xo, yo, txo, tyo, e)) =
  let x = make dt s xo and y = make dt s yo in
  let tx = make dt s txo and ty = make dt s tyo in
  cover "a zero-size shape" (numel s = 0);
  cover "a one-element shape" (numel s = 1 && Array.length s > 0);
  cover "a 0-d shape" (Array.length s = 0);
  cover "a view operand" (xo.view <> Plain || yo.view <> Plain);
  cover "a composition of over 6 functions" (size e > 6);
  let f x y = eval x y e in
  let pair = T.(pair tensor tensor) in
  let both (p, t) = (bits p, bits t) in
  let eager =
    outcome (fun () ->
        both (Rune.jvp pair T.tensor (fun (x, y) -> f x y) (x, y) (tx, ty)))
  in
  let jitted =
    let g = Rune.jit T.(tensor @-> tensor @-> returns tensor) f in
    outcome (fun () ->
        both (Rune.jvp pair T.tensor (fun (x, y) -> g x y) (x, y) (tx, ty)))
  in
  let staged =
    let g =
      Rune.jit
        T.(
          tensor @-> tensor @-> tensor @-> tensor
          @-> returns (pair tensor tensor))
        (fun x y tx ty ->
          Rune.jvp pair T.tensor (fun (x, y) -> f x y) (x, y) (tx, ty))
    in
    outcome (fun () -> both (g x y tx ty))
  in
  cover "a refused operation" (match eager with Raised _ -> true | _ -> false);
  equal ~msg:"jvp of jit" (outcome_t (Testable.pair bits_t bits_t)) eager jitted;
  equal ~msg:"jit of jvp" (outcome_t (Testable.pair bits_t bits_t)) eager staged

(* Every dtype, through the functions every dtype has: complex rules staged and
   evaluated in a call, integers refused alike. *)
type any_case =
  | Any_case :
      ('v, 's) Nx.dtype
      * int array
      * operand
      * operand
      * operand
      * operand
      * expr
      -> any_case

let any_cases =
  let open Gen in
  (let* s = shape in
   let* e = expr ~rank:(Array.length s) in
   let* x = operand s
   and+ y = operand s
   and+ tx = operand s
   and+ ty = operand s in
   let+ (D.Any dt) = of_list D.all in
   Any_case (dt, s, x, y, tx, ty, generic e))
  |> with_pp (fun ppf (Any_case (dt, s, x, y, _, _, e)) ->
      Format.fprintf ppf "%a %a, x %a, y %a: %a" Nx.pp_dtype dt pp_shape s
        pp_view x.view pp_view y.view pp e)

let law_jit_any (Any_case (dt, s, xo, yo, txo, tyo, e)) =
  let x = make_any dt s xo and y = make_any dt s yo in
  let tx = make_any dt s txo and ty = make_any dt s tyo in
  cover "a complex dtype" (D.is Complex dt);
  cover "a narrow float dtype"
    (D.is Float dt && not (D.equal dt f64 || D.equal dt Nx.float32));
  cover "an integer dtype" (D.is Signed dt || D.is Unsigned dt);
  let f x y = eval_any x y e in
  let pair = T.(pair tensor tensor) in
  let both (p, t) = (codes p, codes t) in
  let eager =
    outcome (fun () ->
        both (Rune.jvp pair T.tensor (fun (x, y) -> f x y) (x, y) (tx, ty)))
  in
  let jitted =
    let g = Rune.jit T.(tensor @-> tensor @-> returns tensor) f in
    outcome (fun () ->
        both (Rune.jvp pair T.tensor (fun (x, y) -> g x y) (x, y) (tx, ty)))
  in
  let staged =
    let g =
      Rune.jit
        T.(
          tensor @-> tensor @-> tensor @-> tensor
          @-> returns (pair tensor tensor))
        (fun x y tx ty ->
          Rune.jvp pair T.tensor (fun (x, y) -> f x y) (x, y) (tx, ty))
    in
    outcome (fun () -> both (g x y tx ty))
  in
  let t = outcome_t (Testable.pair codes_t codes_t) in
  equal ~msg:"jvp of jit" t eager jitted;
  equal ~msg:"jit of jvp" t eager staged

(* One kernel per map *)

let test_one_kernel () =
  let x = Nx.place Count.on (v [| 0.5; -1.; 2. |]) in
  let t = Nx.place Count.on (v [| 1.; 1.; 1. |]) in
  let calls f =
    C.reset ();
    ignore (Rune.jvp' f x t);
    (C.calls (), C.positions ())
  in
  equal ~msg:"sin" (pair int int) (1, 3) (calls Nx.sin);
  equal ~msg:"a chain of three" (pair int int) (3, 9)
    (calls (fun x -> Nx.exp (Nx.mul (Nx.sin x) x)));
  equal ~msg:"a sum and its tangent" (pair int int) (1, 3)
    (calls (fun x -> Nx.sum x));
  (* A derivative wider than a block of nx.cpu's holds is one loop too. *)
  let w = Nx.place Count.on (Nx.full f64 [| wide; 3 |] 0.5) in
  let tw = Nx.place Count.on (Nx.full f64 [| wide; 3 |] 1.) in
  let wide_calls f =
    C.reset ();
    ignore (Rune.jvp' f w tw);
    (C.calls (), C.positions ())
  in
  equal ~msg:"a map of twenty operands" (pair int int) (1, 3)
    (wide_calls wide_map);
  equal ~msg:"a sum over twenty operands" (pair int int) (1, 3)
    (wide_calls (wide_reduce (Monoid (Sum, 0, f64))))

(* Structures and constructs *)

let test_structures () =
  let p = T.(pair tensor tensor) in
  invalid ~sub:"Rune.jvp: 1: shape [2] in the parameters, [3] in the tangents"
    (fun () ->
      Rune.jvp p T.tensor
        (fun (a, _) -> a)
        (v [| 1.; 2. |], v [| 1.; 2. |])
        (v [| 1.; 2. |], v [| 1.; 2.; 3. |]));
  invalid ~sub:"Rune.jvp': the parameters hold no real or complex tensor"
    (fun () ->
      Rune.jvp' Fun.id (Nx.zeros Nx.int32 [| 2 |]) (Nx.zeros Nx.int32 [| 2 |]));
  (* An integer result has a zero tangent, a result that depends on no parameter
     too, and an integer parameter's tangent is ignored. *)
  let i = Nx.create Nx.int32 [| 2 |] [| 3l; 4l |] in
  let s = T.(pair tensor tensor) in
  let (a, b), (ta, tb) =
    Rune.jvp
      T.(pair tensor tensor)
      s
      (fun (x, n) -> (Nx.cast Nx.int32 x, Nx.add n n))
      (v [| 1.5; 2. |], i)
      (v [| 1.; 1. |], i)
  in
  ignore (a, b);
  equal ~msg:"integer result" (array int32) [| 0l; 0l |] (Nx.to_array ta);
  equal ~msg:"integer parameter" (array int32) [| 0l; 0l |] (Nx.to_array tb);
  let _, t = Rune.jvp' (fun _ -> v [| 7. |]) (v [| 1. |]) (v [| 1. |]) in
  equal ~msg:"a constant" floats [| 0. |] (Nx.to_array t);
  (* A tangent is fresh memory: the identity's can be donated and its seed lives
     on. *)
  let x = v [| 1.; 2. |] and dx = v [| 3.; 4. |] in
  let _, t = Rune.jvp' Fun.id x dx in
  ignore (Nx.add (Nx.donate t) x);
  equal floats [| 3.; 4. |] (Nx.to_array dx)

(* A parameter of every set, a constant, with a placed tangent: the dual takes
   the constant's form, and a loop over it is traced again where its step places
   the carry. *)
let test_constant_primal () =
  let zero = Nx.zeros f64 [| 2 |] and t = v [| 1.; 2. |] in
  let y, dy = Rune.jvp' (fun x -> Nx.add (Nx.mul x x) x) zero t in
  equal ~msg:"primal" floats [| 0.; 0. |] (Nx.to_array y);
  equal ~msg:"tangent" floats [| 1.; 2. |] (Nx.to_array dy);
  let xs = Nx.create f64 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let f c =
    fst (Rune.scan' ~f:(fun c x -> (Nx.add c (Nx.mul c x), c)) ~init:c xs)
  in
  let _, dy = Rune.jvp' f zero t in
  (* c_3 = c_0 (1 + x_0) (1 + x_1) (1 + x_2), linear in c_0. *)
  equal ~msg:"through a scan" floats
    [| 1. *. 2. *. 4. *. 6.; 2. *. 3. *. 5. *. 7. |]
    (Nx.to_array dy)

let test_detach () =
  let x = v [| 2.; -1. |] and t = v [| 1.; 3. |] in
  equal ~msg:"detach" floats [| 2.; -3. |]
    (tangent (fun x -> Nx.mul x (Rune.detach x)) x t);
  (* Every differentiation around the call detaches it. *)
  let one = v [| 1. |] in
  let inner x = fst (Rune.jvp' (fun y -> Rune.detach (Nx.mul x y)) one one) in
  equal ~msg:"nested" floats [| 0. |] (tangent inner (v [| 2. |]) one)

(* An OCaml branch on a value reads its primal. *)
let test_read () =
  let f x = if Nx.item [ 0 ] x > 0. then Nx.sin x else Nx.cos x in
  equal floats [| Float.cos 1. |] (tangent f (v [| 1. |]) (v [| 1. |]))

(* Nested forward modes: each tangent belongs to its own. *)
let test_nesting () =
  let one = v [| 1. |] in
  let inner x = snd (Rune.jvp' Nx.sin x one) in
  equal ~msg:"second derivative" floats
    [| -.Float.sin 0.5 |]
    (tangent inner (v [| 0.5 |]) one);
  let f x = Nx.mul x (snd (Rune.jvp' (fun y -> Nx.add x y) one one)) in
  equal ~msg:"perturbations apart" floats [| 1. |] (tangent f one one)

let test_jacfwd () =
  let x = v [| 1.; 2.; -3. |] in
  let j = Rune.jacfwd' (fun x -> Nx.mul x x) x in
  equal ~msg:"shape" (array int) [| 3; 3 |] (Nx.shape j);
  equal floats [| 2.; 0.; 0.; 0.; 4.; 0.; 0.; 0.; -6. |] (Nx.to_array j);
  (* A linear map's Jacobian is its matrix, rows the result's elements. *)
  let a = Nx.create f64 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let j = Rune.jacfwd' (fun x -> Nx.matmul a x) x in
  equal ~msg:"a matrix's shape" (array int) [| 2; 3 |] (Nx.shape j);
  equal ~msg:"a matrix" floats (Nx.to_array a) (Nx.to_array j);
  (* A transpose's: [J i j k l] is 1 where [(k, l) = (j, i)], in C order. *)
  let m = Nx.create f64 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let j = Rune.jacfwd' Nx.transpose m in
  equal ~msg:"a transpose's shape" (array int) [| 2; 2; 2; 2 |] (Nx.shape j);
  let expected =
    Array.init 16 (fun n ->
        let i = n / 8
        and jj = n / 4 mod 2
        and k = n / 2 mod 2
        and l = n mod 2 in
        if k = jj && l = i then 1. else 0.)
  in
  equal ~msg:"a transpose" floats expected (Nx.to_array j);
  (* A complex argument's columns are the tangents along its real units. *)
  let z =
    Nx.create c128 [| 2 |] [| { re = 1.; im = 2. }; { re = -1.; im = 0.5 } |]
  in
  let j = Rune.jacfwd' (fun z -> Nx.mul z z) z in
  let two_z =
    Array.map (fun c -> Complex.mul { re = 2.; im = 0. } c) (Nx.to_array z)
  in
  equal ~msg:"complex"
    (array (pair (float 1e-12) (float 1e-12)))
    (Array.map
       (fun (c : Complex.t) -> (c.re, c.im))
       [| two_z.(0); Complex.zero; Complex.zero; two_z.(1) |])
    (Array.map (fun (c : Complex.t) -> (c.re, c.im)) (Nx.to_array j));
  let j = Rune.jacfwd' (fun x -> Nx.sum x) (Nx.zeros f64 [| 0 |]) in
  equal ~msg:"no element" (array int) [| 0 |] (Nx.shape j);
  invalid ~sub:"Rune.jacfwd': the parameters hold no real or complex tensor"
    (fun () -> Rune.jacfwd' Fun.id (Nx.zeros Nx.int32 [| 2 |]));
  invalid ~sub:"Rune.jacfwd': Cholesky has no derivative yet" (fun () ->
      Rune.jacfwd' Nx.cholesky (Nx.zeros f64 [| 0; 0 |]))

(* Column [k] of the Jacobian, its elements at index [k] of the argument's axes
   in C order, is the tangent along element [k], bit for bit. *)
let jac_cases =
  let open Gen in
  (let* s = shape in
   let* e = expr ~rank:(Array.length s) in
   let* k = int_range 0 (Int.max 0 (numel s - 1)) in
   let+ x = operand s in
   (s, x, k, e))
  |> with_pp (fun ppf (s, x, k, e) ->
      Format.fprintf ppf "%a, x %a, column %d: %a" pp_shape s pp_view x.view k
        pp e)

let law_jacobian (s, xo, k, e) =
  let x = make f64 s xo in
  let n = numel s in
  cover "a zero-size shape" (n = 0);
  cover "several elements" (n > 1);
  let f x = eval x x e in
  let jacobian = outcome (fun () -> Rune.jacfwd' f x) in
  cover "a refused function"
    (match jacobian with Raised _ -> true | _ -> false);
  match jacobian with
  | Raised m ->
      (* It raises as jvp' does along any tangent, under its own name. *)
      let renamed m =
        let jvp = "Rune.jvp'" and n = String.length m in
        let k = String.length jvp in
        if n >= k && String.sub m 0 k = jvp then
          "Rune.jacfwd'" ^ String.sub m k (n - k)
        else m
      in
      let along =
        match Rune.jvp' f x (Nx.zeros_like x) with
        | _ -> "no raise"
        | exception Invalid_argument m -> renamed m
      in
      equal ~msg:"raises as jvp'" string along m
  | Value j ->
      let sy = Array.sub (Nx.shape j) 0 (Nx.ndim j - Array.length s) in
      equal ~msg:"shape" (array int) (Array.append sy s) (Nx.shape j);
      if n > 0 then begin
        let column =
          Nx.reshape sy (Nx.slice [ A; I k ] (Nx.reshape [| numel sy; n |] j))
        in
        let unit =
          Nx.reshape s
            (Nx.create f64 [| n |]
               (Array.init n (fun i -> if i = k then 1. else 0.)))
        in
        equal ~msg:"column" bits_t
          (bits (snd (Rune.jvp' f x unit)))
          (bits column)
      end

(* Loops under forward mode: a scan is its fold's, an iterate its while loop's,
   the step traced then evaluated with duals, bit for bit, eagerly and under
   jit. Rows and carry both have tangents. *)
let fold step ~init xs =
  let n = (Nx.shape xs).(0) in
  let rec go i c ys =
    if i = n then (c, List.rev ys)
    else
      let c, y = step c (Nx.get [ i ] xs) in
      go (i + 1) c (y :: ys)
  in
  go 0 init []

type loop_case =
  | Loop_case :
      (float, 's) Nx.dtype
      * int array
      * int
      * operand
      * operand
      * operand
      * operand
      * expr
      * expr
      -> loop_case

let loop_cases =
  let open Gen in
  (let* s = shape in
   let* n = of_list [ 0; 1; 2; 5 ] in
   let rank = Array.length s in
   let* g = expr ~rank and+ h = expr ~rank in
   let rows = Array.append [| n |] s in
   let* c = operand s and+ xs = operand rows in
   let* tc = operand s and+ txs = operand rows in
   let+ (D.Any dt) = of_list [ D.Any Nx.float32; Any f64 ] in
   match D.kind dt with
   | Float -> Loop_case (dt, s, n, c, xs, tc, txs, g, h)
   | _ -> assert false)
  |> with_pp (fun ppf (Loop_case (dt, s, n, c, xs, _, _, g, h)) ->
      Format.fprintf ppf "%a %a, %d rows, carry %a, xs %a: carry %a, output %a"
        Nx.pp_dtype dt pp_shape s n pp_view c.view pp_view xs.view pp g pp h)

let law_scan (Loop_case (dt, s, n, co, xso, tco, txso, g, h)) =
  let rows = Array.append [| n |] s in
  let init = make dt s co and xs = make dt rows xso in
  let tinit = make dt s tco and txs = make dt rows txso in
  cover "no row" (n = 0);
  cover "several rows" (n > 1);
  cover "a zero-size row" (numel s = 0);
  cover "a view operand" (co.view <> Plain || xso.view <> Plain);
  let step c x =
    (Nx.broadcast_to s (eval c x g), Nx.broadcast_to s (eval c x h))
  in
  let p = T.(pair tensor tensor) in
  let both ((c, ys), (tc, tys)) = ((bits c, bits ys), (bits tc, bits tys)) in
  let folded (c, xs) =
    (* The step is traced at the call, so with no row its errors raise. *)
    if n = 0 then ignore (step c (Nx.zeros dt s));
    let c, ys = fold step ~init:c xs in
    (c, if n = 0 then Nx.zeros dt rows else Nx.stack ~axis:0 ys)
  in
  let scanned (c, xs) = Rune.scan' ~f:step ~init:c xs in
  let jvp f = Rune.jvp p p f (init, xs) (tinit, txs) in
  let expected = outcome (fun () -> both (jvp folded)) in
  cover "a refused step" (match expected with Raised _ -> true | _ -> false);
  let t = Testable.pair bits_t bits_t in
  equal ~msg:"eager"
    (outcome_t (Testable.pair t t))
    expected
    (outcome (fun () -> both (jvp scanned)));
  let staged =
    Rune.jit
      T.(tensor @-> tensor @-> tensor @-> tensor @-> returns (pair p p))
      (fun c xs tc txs -> Rune.jvp p p scanned (c, xs) (tc, txs))
  in
  equal ~msg:"under jit"
    (outcome_t (Testable.pair t t))
    expected
    (outcome (fun () -> both (staged init xs tinit txs)))

type iterate_case =
  | Iterate_case :
      (float, 's) Nx.dtype * int array * int * operand * operand * expr
      -> iterate_case

let iterate_cases =
  let open Gen in
  (let* s = shape in
   let* trips = int_range 0 3 in
   let* g = expr ~rank:(Array.length s) in
   let* c = operand s and+ tc = operand s in
   let+ (D.Any dt) = of_list [ D.Any Nx.float32; Any f64 ] in
   match D.kind dt with
   | Float -> Iterate_case (dt, s, trips, c, tc, g)
   | _ -> assert false)
  |> with_pp (fun ppf (Iterate_case (dt, s, trips, _, _, g)) ->
      Format.fprintf ppf "%a %a, %d trips: %a" Nx.pp_dtype dt pp_shape s trips
        pp g)

let law_iterate (Iterate_case (dt, s, trips, co, tco, g)) =
  cover "no trip" (trips = 0);
  cover "several trips" (trips > 1);
  let init = make dt s co and tinit = make dt s tco in
  let step (x, k) =
    (Nx.broadcast_to s (eval x x g), Nx.add k (Nx.scalar Nx.int32 1l))
  in
  let until (_, k) =
    Nx.greater_equal k (Nx.scalar Nx.int32 (Int32.of_int trips))
  in
  let k0 = Nx.scalar Nx.int32 0l in
  let p = T.(pair tensor tensor) in
  let both ((x, _), (t, _)) = (bits x, bits t) in
  let looped c =
    ignore (step c);
    let rec go j c = if j = trips then c else go (j + 1) (step c) in
    go 0 c
  in
  let iterated c = Rune.iterate p ~max:trips ~until ~f:step c in
  let jvp f = Rune.jvp p p f (init, k0) (tinit, k0) in
  let expected = outcome (fun () -> both (jvp looped)) in
  cover "a refused step" (match expected with Raised _ -> true | _ -> false);
  equal
    (outcome_t (Testable.pair bits_t bits_t))
    expected
    (outcome (fun () -> both (jvp iterated)))

(* An addition under forward mode adds its primal. *)
let test_total () =
  let total = Rune.Total.make () in
  let _, sum =
    Rune.Total.collect total ~zero:(v [| 0. |]) (fun () ->
        Rune.jvp'
          (fun x ->
            Rune.Total.add total x;
            Nx.sin x)
          (v [| 2. |]) (v [| 1. |]))
  in
  equal floats [| 2. |] (Nx.to_array sum)

let () =
  exit
    (run "rune2 jvp"
       [
         group "tangents" tangent_props;
         group "conventions"
           [
             test "kinks take JAX's tangents" test_kinks;
             test "complex tangents are directional derivatives" test_complex;
             test "edges take their definitions' values" test_edges;
           ];
         group "jit"
           [
             prop "jvp of jit gives jvp's bits" jit_cases law_jit;
             prop "jvp of jit gives jvp's bits at every dtype" any_cases
               law_jit_any;
           ];
         group "loops"
           [
             prop "jvp of scan is jvp of its fold" loop_cases law_scan;
             prop "jvp of iterate is jvp of its loop" iterate_cases law_iterate;
           ];
         group "kernels"
           [ test "a map's tangent is one kernel" test_one_kernel ];
         group "interface"
           [
             test "structures and zero tangents" test_structures;
             test "a constant parameter" test_constant_primal;
             test "detach" test_detach;
             test "a read reads the primal" test_read;
             test "nesting" test_nesting;
             test "jacfwd'" test_jacfwd;
             prop "jacfwd' applied to a tangent is jvp's" jac_cases law_jacobian;
             test "totals" test_total;
           ];
       ])
