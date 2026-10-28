(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.vmap: a map equals its lanes computed one by one and stacked, for every
   operation, its operands lanes and constants, of every dtype and view; named
   maps nest; a lane's reads raise, and so do lanes and lane_index outside their
   map; totals, forward mode, compiled functions, loops, placements and checks
   inside maps. *)

open Windtrap
open Compose
module Rune = Rune2
module D = Nx.Dtype
module L = Nx_array.Layout
module P = Nx_kernel.Prog
module T = Nx.Ptree
module C = Nx_support.Counting
module Count = (val Nx.devices ~kernels:(module C) [ Nx_support.memory 0 ])
module Two = (val Nx.devices [ Nx_support.memory 1; Nx_support.memory 2 ])

let f64 = Nx.float64
let c128 = Nx.complex128
let invalid ~sub f = raises_match (Exn.invalid_arg ~substring:sub) f
let v xs = Nx.create f64 [| Array.length xs |] xs
let floats = array float_exact

type f64_t = (float, D.float64_elt, Nx.host) Nx.t

(* The lanes one by one: [f] of each row of [xs], stacked along a new axis 0.
   With no row, [f] of a row of ones gives the shape. *)
let one_by_one f xs =
  let n = Nx.dim 0 xs in
  if n > 0 then Nx.stack ~axis:0 (List.init n (fun k -> f (Nx.get [ k ] xs)))
  else
    let row = Array.sub (Nx.shape xs) 1 (Nx.ndim xs - 1) in
    let y = f (Nx.ones (Nx.dtype xs) row) in
    Nx.zeros (Nx.dtype y) (Array.append [| 0 |] (Nx.shape y))

(* Operands *)

(* A constant of shape [s], its values cycled through the pool. *)
let konst s =
  let n = numel s and m = Array.length pool in
  Nx.create f64 s (Array.init n (fun k -> pool.(((k * 7) + 3) mod m)))

(* Values a case draws: the pool's and uniform ones, or uniform in a range. *)
let any_value =
  Gen.(
    frequency
      [
        (3, map (fun i -> pool.(i)) (int_range 0 (Array.length pool - 1)));
        (2, float_range (-4.) 4.);
      ])

type case = {
  name : string;
  shape : int array Gen.t;  (** A lane's. *)
  values : float Gen.t;
  f : f64_t -> f64_t;
}

let case ?(values = any_value)
    ?(shape = Gen.(map (fun n -> [| n |]) (int_range 0 4))) name f =
  { name; shape; values; f }

let lanes = Gen.int_range 0 3
let rows k = Gen.(map (fun n -> [| k; n |]) (int_range 0 4))

let matrix =
  Gen.(map (fun (m, n) -> [| m; n |]) (pair (int_range 0 3) (int_range 0 4)))

let full =
  Gen.(
    map
      (fun (m, n) -> [| m + 1; n + 1 |])
      (pair (int_range 0 2) (int_range 0 3)))

let square = Gen.(map (fun n -> [| n; n |]) (int_range 0 3))
let uniform = Gen.float_range (-2.) 2.

(* [n] lanes of a case's shape, as a view drawn, and its values. *)
let gen c =
  let open Gen in
  (let* n = lanes in
   let* s = c.shape in
   let s = Array.append [| n |] s in
   let* view = of_list [ Plain; Flipped; Strided; Broadcast ] in
   let size =
     match view with
     | Plain | Flipped -> numel s
     | Strided -> numel s * 2
     | Broadcast -> numel s / Int.max 1 s.(0)
   in
   let+ values = array ~size:(constant size) c.values in
   (s, { view; values }))
  |> with_pp (fun ppf (s, o) ->
      Format.fprintf ppf "%a %a %a" pp_shape s pp_view o.view
        (Format.pp_print_list Format.pp_print_float)
        (Array.to_list o.values))

(* Bit for bit: on nx.cpu a reduction adds each output's terms in lane order
   whatever its other extents, and a contraction does too while its rows or its
   columns are at most 4 (nx_cpu.mli, Order), as every case's are. A wider
   contraction is held to its bound below. *)
let law c (s, o) =
  cover "no lane" (s.(0) = 0);
  cover "several lanes" (s.(0) > 1);
  let xs = make f64 s o in
  equal codes_t (codes (one_by_one c.f xs)) (codes (Rune.vmap' c.f xs))

(* Operations built from their parts, as nx's functions build none *)

let row k x = Nx.get [ k ] x
let contiguous s = L.contiguous (Iarray.of_array s)
let any = D.Any f64
let identity = P.of_node ~ins:[| any |] (Op1 (Copy, any, 0))

let idx xs =
  Nx.create Nx.int64 [| Array.length xs |] (Array.map Int64.of_int xs)

(* A map of two outputs reading a padded load and the coordinates: [x0 sin
   (x1)], and [x0 + coord 0] over [x1] padded by one on each side of axis 1. *)
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
           at = None;
         })
  in
  Nx.concatenate ~axis:0 [ y; z ]

(* A loop of one load along axis 1 of [x], from a carry of extent 1 there. *)
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

let reduce1 reductions x =
  Nx.Prim.eval ~by:"test"
    (Reduce
       {
         layout = contiguous (Nx.shape x);
         axes = [| 1 |];
         prog = identity;
         reductions;
         loads = [| Plain x |];
       })

let column x = Nx.slice [ A; R (0, 1) ] x

(* Positions of a scatter of [n] updates into 3 targets, two on target 0. *)
let targets n = idx (Array.init n (fun k -> if k < 2 then 0 else k mod 3))

(* Positions in [0, 3) a lane's values give. *)
let positions x =
  Nx.cast Nx.int64
    (Nx.mod_
       (Nx.floor (Nx.abs (Nx.where (Nx.isfinite x) x (Nx.zeros_like x))))
       (Nx.scalar_like x 3.))

let scatter combine ~lane x =
  let n = Nx.dim 1 x in
  let into = konst [| 3 |] and updates = row 1 x in
  match lane with
  | `Updates -> Nx.scatter ~combine ~axis:0 (targets n) updates into
  | `Into ->
      let into = Nx.pad [| (0, 3) |] 0.5 (row 0 x) |> Nx.slice [ R (0, 3) ] in
      Nx.scatter ~combine ~axis:0 (targets n) (konst [| n |]) into
  | `Positions ->
      Nx.scatter ~combine ~axis:0 (positions (row 0 x)) (konst [| n |]) into

let pattern = Nx.Pattern.v "i j, j k -> i k | j"
let complex x = Nx.complex c128 ~re:(row 0 x) ~im:(row 1 x)
let parts z = Nx.stack [ Nx.real f64 z; Nx.imag f64 z ]

(* A symmetric positive definite matrix from a lane's square. *)
let spd x =
  let n = Nx.dim 0 x in
  Nx.add (Nx.matmul x (Nx.transpose x)) (Nx.mul_s (Nx.eye f64 n) 3.)

let cases =
  [
    case "a unary map" Nx.sin;
    case "a map of two lanes" (fun x -> Nx.mul x (Nx.flip x));
    case "a map of a lane and a constant" (fun x ->
        Nx.sub (konst (Nx.shape x)) x);
    case "a map of a lane and a formula" (fun x ->
        Nx.pow x (Nx.full f64 (Nx.shape x) 0.5));
    case "where of a constant condition" (fun x ->
        let k = konst (Nx.shape x) in
        Nx.where (Nx.less k (Nx.zeros_like k)) x k);
    case "fma of a lane and constants" (fun x ->
        Nx.fma x (konst (Nx.shape x)) (Nx.scalar_like x 0.5));
    case
      ~shape:
        Gen.(
          map
            (fun (m, n) -> [| m + 1; n |])
            (pair (int_range 0 2) (int_range 0 3)))
      "a map of two outputs, a padded load and coordinates" multi;
    case ~shape:matrix "sum over an axis" (Nx.sum ~axes:[ 1 ]);
    case ~shape:matrix "sum over every axis" (fun x -> Nx.sum x);
    case ~shape:matrix "prod" (Nx.prod ~axes:[ 0 ]);
    case ~shape:full "max" (Nx.max ~axes:[ 1 ]);
    case ~shape:matrix "logsumexp" (Nx.logsumexp ~axes:[ 1 ]);
    case ~shape:matrix "var" (Nx.var ~axes:[ 0 ]);
    case ~shape:matrix "moments" (fun x ->
        let (m, var), () = reduce1 [ Moments (0, f64) ] x in
        Nx.stack [ m; var ]);
    case ~shape:full "an arg reduction" (fun x ->
        let (y, p), () = reduce1 [ Arg (Min, 0, f64) ] x in
        Nx.stack [ y; Nx.cast f64 p ]);
    case ~shape:matrix "a sum of a lane by a constant" (fun x ->
        Nx.sum ~axes:[ 1 ] (Nx.mul x (konst (Nx.shape x))));
    case ~shape:matrix "cumsum" (Nx.cumsum ~axis:1);
    case ~shape:matrix "cumprod" (Nx.cumprod ~axis:0);
    case ~shape:matrix "cummax" (Nx.cummax ~axis:1);
    case ~shape:matrix "logcumsumexp" (Nx.logcumsumexp ~axis:1);
    case ~shape:matrix "a running sum from a constant carry" (fun x ->
        scan1 ~carry:(konst [| Nx.dim 0 x; 1 |]) (Monoid (Sum, 0, f64)) x);
    case ~shape:full "a running max from a lane's carry" (fun x ->
        scan1 ~carry:(column x) (Monoid (Max, 0, f64)) x);
    case ~shape:full "a running sum of a constant from a lane's carry" (fun x ->
        scan1 ~carry:(column x) (Monoid (Sum, 0, f64)) (konst (Nx.shape x)));
    case ~shape:matrix "take at constant positions"
      (Nx.take ~axis:1 (idx [| 2; 0; 7; 1 |]));
    case "take of a constant at a lane's positions" (fun x ->
        Nx.take ~axis:0 (positions x) (konst [| 3 |]));
    case ~shape:matrix "take_along_axis at a lane's positions" (fun x ->
        Nx.take_along_axis ~axis:1 (positions x)
          (Nx.pad [| (0, 0); (0, 3) |] 1. x));
    case ~shape:(rows 2) "scatter add of a lane into a constant"
      (scatter Add ~lane:`Updates);
    case ~shape:(rows 2) "scatter set of a constant into a lane"
      (scatter Set ~lane:`Into);
    case ~shape:(rows 2) "scatter max at a lane's positions"
      (scatter Max ~lane:`Positions);
    case ~shape:(rows 2) "scatter min of a lane" (scatter Min ~lane:`Updates);
    case ~shape:matrix "sort" (fun x -> fst (Nx.sort ~axis:1 x));
    case ~shape:matrix "argsort" (fun x -> Nx.cast f64 (Nx.argsort ~axis:0 x));
    case
      ~shape:Gen.(map (fun n -> [| n + 2 |]) (int_range 0 3))
      "top_k"
      (fun x ->
        let y, p = Nx.top_k ~k:2 x in
        Nx.stack [ y; Nx.cast f64 p ]);
    case ~shape:full "pad" (Nx.pad [| (1, 0); (2, 1) |] 3.);
    case
      ~shape:Gen.(map (fun n -> [| 2; n + 3 |]) (int_range 0 3))
      "extract_patches"
      (fun x ->
        Nx.extract_patches ~kernel_size:[| 2 |] ~stride:[| 1 |]
          ~dilation:[| 1 |]
          ~padding:[| (1, 0) |]
          x);
    case
      ~shape:Gen.(map (fun l -> [| 2; 2; l + 1 |]) (int_range 0 3))
      "combine_patches"
      (fun x ->
        let l = Nx.dim 2 x in
        Nx.combine_patches
          ~output_size:[| l + 1 |]
          ~kernel_size:[| 2 |] ~stride:[| 1 |] ~dilation:[| 1 |]
          ~padding:[| (0, 0) |]
          x);
    case "concatenate with constants" (fun x ->
        Nx.concatenate ~axis:0 [ konst [| 2 |]; x; Nx.sin x; konst [| 1 |] ]);
    case ~shape:matrix "matmul of a lane by a constant" (fun x ->
        Nx.matmul x (konst [| Nx.dim 1 x; 2 |]));
    case ~shape:matrix "matmul of a constant by a lane" (fun x ->
        Nx.matmul (konst [| 3; Nx.dim 0 x |]) x);
    case ~shape:matrix "matmul of two lanes" (fun x ->
        Nx.matmul (Nx.transpose x) x);
    case
      ~shape:
        Gen.(
          map
            (fun (m, n) -> [| 2; m; n |])
            (pair (int_range 0 3) (int_range 0 3)))
      "a batched matmul of a lane by a constant"
      (fun x -> Nx.matmul x (konst [| 2; Nx.dim 2 x; 3 |]));
    case
      ~shape:
        Gen.(
          map
            (fun (m, n) -> [| 2; m; n |])
            (pair (int_range 0 3) (int_range 0 3)))
      "a batched matmul of a constant by a lane"
      (fun x -> Nx.matmul (konst [| 2; 2; Nx.dim 1 x |]) x);
    case ~shape:matrix "a contraction whose init alone has lanes" (fun x ->
        let m = Nx.dim 0 x and n = Nx.dim 1 x in
        Nx.contract ~init:x f64 pattern (konst [| m; 2 |]) (konst [| 2; n |]));
    case ~shape:matrix "a contraction of a lane from a lane's init" (fun x ->
        let n = Nx.dim 1 x in
        Nx.contract ~init:(Nx.sin x) f64 pattern x (konst [| n; n |]));
    case ~shape:(rows 2) "fft" (fun x -> parts (Nx.fft (complex x)));
    case "rfft" (fun x -> parts (Nx.rfft c128 x));
    case
      ~shape:Gen.(map (fun n -> [| 2; n + 1 |]) (int_range 0 3))
      "irfft"
      (fun x -> Nx.irfft f64 (complex x));
    case ~shape:(rows 2) "complex exp" (fun x -> parts (Nx.exp (complex x)));
    case ~shape:(rows 2) "complex as float pairs" (fun x ->
        Nx.bitcast f64 (complex x));
    case ~values:uniform ~shape:square "cholesky" (fun x -> Nx.cholesky (spd x));
    case ~values:uniform ~shape:matrix "qr" (fun x ->
        let q, r = Nx.qr x in
        Nx.concatenate ~axis:0 [ Nx.reshape [| -1 |] q; Nx.reshape [| -1 |] r ]);
    case ~values:uniform ~shape:square "solve_triangular of a constant"
      (fun x ->
        let n = Nx.dim 0 x in
        Nx.solve_triangular (Nx.cholesky (spd x)) (konst [| n; 2 |]));
    case ~values:uniform ~shape:square "solve_triangular by a constant"
      (fun x ->
        let n = Nx.dim 0 x in
        Nx.solve_triangular (Nx.cholesky (spd (konst [| n; n |]))) x);
    case ~values:uniform ~shape:square "eigh" (fun x ->
        let w, vs = Nx.eigh f64 (spd x) in
        Nx.concatenate ~axis:0 [ Nx.reshape [| 1; -1 |] w; Nx.abs vs ]);
    case ~values:uniform ~shape:square "lu" (fun x ->
        let p, l, u = Nx.lu (spd x) in
        Nx.concatenate ~axis:0 [ Nx.reshape [| 1; -1 |] (Nx.cast f64 p); l; u ]);
    case ~shape:matrix "copy of a transposed view" (fun x ->
        Nx.copy (Nx.transpose x));
    case ~shape:matrix "reshape of a transposed view" (fun x ->
        Nx.reshape [| -1 |] (Nx.transpose x));
    case ~shape:matrix "broadcast to new axes" (fun x ->
        Nx.broadcast_to [| 2; Nx.dim 0 x; Nx.dim 1 x |] (Nx.exp x));
    case ~shape:matrix "slices with steps" (fun x ->
        Nx.slice [ Rs (0, Nx.dim 0 x, 2); Rs (Nx.dim 1 x - 1, -1, -1) ] x);
    case "bitcast to integers" (fun x ->
        Nx.cast f64 (Nx.rshift (Nx.bitcast Nx.int64 x) 40));
    case ~shape:matrix "place" (fun x ->
        Nx.place Nx.Host.on (Nx.sin (Nx.place Count.on x)));
    case ~shape:matrix "a passing check" (fun x ->
        let nan = Nx.isnan x in
        Nx.check T.tensor
          (Nx.logical_or nan (Nx.logical_not nan))
          x
          (fun _ _ -> Failure "check");
        Nx.cos x);
  ]

let operation_props = List.map (fun c -> prop c.name (gen c) (law c)) cases

(* Contractions with segments and scales, each operand a lane's or a constant *)

type role = Lane | Constant

let pp_role ppf r =
  Format.pp_print_string ppf
    (match r with Lane -> "lane" | Constant -> "constant")

type contraction = {
  m : int;
  k : int;
  n : int;
  g : int;
  segs : [ `None | `Rows | `Contracting ];
  a : role;
  b : role;
  init : role option;
  offsets : role;
  a_scale : role option;
  b_scale : role option;
}

let contraction =
  let open Gen in
  let role = of_list [ Lane; Constant ] in
  let* m = int_range 0 3 and+ k = int_range 0 3 and+ n = int_range 0 3 in
  let* g = int_range 0 2 and+ segs = of_list [ `None; `Rows; `Contracting ] in
  let* a = role and+ b = role and+ offsets = role in
  let+ init = option role
  and+ a_scale = option role
  and+ b_scale = option role in
  { m; k; n; g; segs; a; b; init; offsets; a_scale; b_scale }

let pp_contraction ppf c =
  let opt ppf = function
    | None -> Format.pp_print_string ppf "none"
    | Some r -> pp_role ppf r
  in
  Format.fprintf ppf
    "m %d k %d n %d g %d %s, a %a b %a init %a offsets %a scales %a %a" c.m c.k
    c.n c.g
    (match c.segs with
    | `None -> "unsegmented"
    | `Rows -> "rows"
    | `Contracting -> "contracting")
    pp_role c.a pp_role c.b opt c.init pp_role c.offsets opt c.a_scale opt
    c.b_scale

(* The contraction over a lane's values [x] of [width c] elements: each operand
   with lanes is cut from [x], and the others are constants. *)
let shapes c =
  let a = [| c.m; c.k |] in
  let b =
    match c.segs with `Rows -> [| c.g; c.k; c.n |] | _ -> [| c.k; c.n |]
  in
  let r =
    match c.segs with
    | `Contracting -> [| c.g; c.m; c.n |]
    | _ -> [| c.m; c.n |]
  in
  let sb = match c.segs with `Rows -> [| 1; 1; c.n |] | _ -> [| 1; c.n |] in
  (a, b, r, [| c.g + 1 |], [| c.m; 1 |], sb)

let width c =
  let a, b, r, o, sa, sb = shapes c in
  numel a + numel b + numel r + numel o + numel sa + numel sb

let contract c x =
  let sa_, sb_, sr, so, ssa, ssb = shapes c in
  let at = ref 0 in
  let cut s =
    let k = numel s in
    let y = Nx.reshape s (Nx.slice [ R (!at, !at + k) ] x) in
    at := !at + k;
    y
  in
  let operand role s = match role with Lane -> cut s | Constant -> konst s in
  let a = operand c.a sa_ and b = operand c.b sb_ in
  let init = Option.map (fun r -> operand r sr) c.init in
  let scale role s =
    Option.map
      (fun r -> Nx.Prim.Value (Nx.add_s (Nx.abs (operand r s)) 0.5))
      role
  in
  let a_scale = scale c.a_scale ssa and b_scale = scale c.b_scale ssb in
  let offsets () =
    let o = operand c.offsets so in
    Nx.cast Nx.int64
      (Nx.mod_
         (Nx.floor (Nx.abs (Nx.where (Nx.isfinite o) o (Nx.zeros_like o))))
         (Nx.scalar_like o 4.))
  in
  let segments : _ Nx_kernel.segments option =
    match c.segs with
    | `None -> None
    | `Rows -> Some (Rows { offsets = offsets (); a_axis = 0; b_axis = 0 })
    | `Contracting -> Some (Contracting { offsets = offsets (); pair = 0 })
  in
  let contracting : (int * int) iarray =
    match c.segs with `Rows -> [| (1, 1) |] | _ -> [| (1, 0) |]
  in
  Nx.Prim.eval ~by:"test"
    (Contract
       {
         batch = [||];
         contracting;
         acc = any;
         out = f64;
         a;
         b;
         init;
         a_scale;
         b_scale;
         segments;
       })

let contractions =
  let open Gen in
  (let* c = contraction in
   let* n = lanes in
   let+ xs = array ~size:(constant (n * width c)) uniform in
   (c, n, xs))
  |> with_pp (fun ppf (c, n, _) ->
      Format.fprintf ppf "%a, %d lanes" pp_contraction c n)

let law_contract (c, n, xs) =
  cover "segments" (c.segs <> `None);
  cover "a scale" (c.a_scale <> None || c.b_scale <> None);
  cover "both operands' lanes" (c.a = Lane && c.b = Lane);
  cover "one operand's lanes" (c.a <> c.b);
  cover "an init's lanes alone"
    (c.a = Constant && c.b = Constant && c.init = Some Lane);
  let xs = Nx.create f64 [| n; width c |] xs in
  equal codes_t
    (codes (one_by_one (contract c) xs))
    (codes (Rune.vmap' (contract c) xs))

(* A contraction whose lanes are its rows': nx.cpu adds one lane's four rows in
   lane order and the map's in chain order, so each lane is within the bound
   [Nx.contract] states of the lane alone, [γ(K + 1, 2u) Σ|a||b|]. *)
let wide =
  let open Gen in
  (let* n = int_range 2 4 and+ k = int_range 1 6 in
   let* a = array ~size:(constant (n * 4 * k)) uniform in
   let+ b = array ~size:(constant (k * 8)) uniform in
   (n, k, a, b))
  |> with_pp (fun ppf (n, k, _, _) ->
      Format.fprintf ppf "%d lanes, depth %d" n k)

let law_wide (n, k, a, b) =
  let a = Nx.create f64 [| n; 4; k |] a and b = Nx.create f64 [| k; 8 |] b in
  let f a = Nx.matmul a b in
  let mapped = Nx.to_array (Rune.vmap' f a) in
  let alone = Nx.to_array (one_by_one f a) in
  let size =
    Nx.to_array (one_by_one (fun a -> Nx.matmul (Nx.abs a) (Nx.abs b)) a)
  in
  let v = Float.of_int (k + 1) *. Float.epsilon in
  let gamma = v /. (1. -. v) in
  Array.iteri
    (fun e y ->
      less
        ~msg:(Printf.sprintf "element %d" e)
        float_exact
        ~than:((gamma *. size.(e)) +. Float.min_float)
        (Float.abs (y -. alone.(e))))
    mapped

(* Compositions *)

(* [f] of each lane's rows of [x] and [y], stacked. *)
let one_by_one2 f x y =
  let n = Nx.dim 0 x in
  if n > 0 then
    Nx.stack ~axis:0
      (List.init n (fun k -> f (Nx.get [ k ] x) (Nx.get [ k ] y)))
  else
    let row x =
      Nx.ones (Nx.dtype x) (Array.sub (Nx.shape x) 1 (Nx.ndim x - 1))
    in
    let z = f (row x) (row y) in
    Nx.zeros (Nx.dtype z) (Array.append [| 0 |] (Nx.shape z))

type compose_case =
  | Pair :
      ('v, 's) Nx.dtype * int array * operand * operand * bool * expr
      -> compose_case

(* Two arrays of [n] lanes, or one and a constant of a lane's shape, [y]. *)
let compositions ~dtypes ~expr:e =
  let open Gen in
  (let* n = lanes and+ s = shape in
   let* ex = e ~rank:(Array.length s) in
   let full = Array.append [| n |] s in
   let* both = bool in
   let* x = operand full and+ y = operand (if both then full else s) in
   let+ (D.Any dt) = of_list dtypes in
   Pair (dt, full, x, y, both, ex))
  |> with_pp (fun ppf (Pair (dt, s, x, y, both, e)) ->
      Format.fprintf ppf "%a %a, x %a, y %a%s: %a" Nx.pp_dtype dt pp_shape s
        pp_view x.view pp_view y.view
        (if both then "" else " (a constant)")
        pp e)

let law_compose ~kinds (Pair (dt, s, xo, yo, both, e)) =
  cover "no lane" (s.(0) = 0);
  cover "0-d lanes" (Array.length s = 1);
  cover "a constant operand" (not both);
  if kinds then begin
    cover "a complex dtype" (D.is Complex dt);
    cover "an integer dtype" (D.is Signed dt || D.is Unsigned dt)
  end;
  let x = make_any dt s xo in
  let lane = Array.sub s 1 (Array.length s - 1) in
  let y = make_any dt (if both then s else lane) yo in
  let f x y = eval_any x y e in
  let mapped, by_lane =
    if both then
      ( outcome (fun () ->
            codes (Rune.vmap T.(tensor @-> tensor @-> returns tensor) f x y)),
        outcome (fun () -> codes (one_by_one2 f x y)) )
    else
      ( outcome (fun () -> codes (Rune.vmap' (fun x -> f x y) x)),
        outcome (fun () -> codes (one_by_one (fun x -> f x y) x)) )
  in
  equal (outcome_t codes_t) by_lane mapped

let float_compositions =
  compositions ~dtypes:[ D.Any Nx.float32; Any f64 ] ~expr:(fun ~rank ->
      expr ~rank)

let every_dtype =
  compositions ~dtypes:D.all ~expr:(fun ~rank -> Gen.map generic (expr ~rank))

(* Named maps *)

let lanes_of_two = Nx.create f64 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]

(* An outer map [a] over the two rows, an inner [b] over each row's three
   elements: element [i; j] of the result is lane [i] of [a] and [j] of [b]. *)
let nested ?(outer = Rune.axis ()) ?(inner = Rune.axis ()) f =
  Rune.vmap' ~axis:outer (Rune.vmap' ~axis:inner (f outer inner)) lanes_of_two

let index a x = Nx.cast f64 (Rune.lane_index a x)

let test_named () =
  let y =
    nested (fun a b x ->
        let tens = Nx.mul_s (index a x) 10. in
        Nx.add x (Nx.add tens (Nx.mul_s (index b x) 100.)))
  in
  equal ~msg:"indices" floats
    [| 1.; 102.; 203.; 14.; 115.; 216. |]
    (Nx.to_array y);
  (* An inner map of the outer's name hides it. *)
  let a = Rune.axis () in
  let y = nested ~outer:a ~inner:a (fun a _ x -> Nx.add x (index a x)) in
  equal ~msg:"hidden" floats [| 1.; 3.; 5.; 4.; 6.; 8. |] (Nx.to_array y);
  (* [lanes] of the outer map gathers each inner lane across the outer's: at [i;
     j], column [j]. Of the inner, row [i]. *)
  let y = nested (fun a _ x -> Rune.lanes a x) in
  equal ~msg:"lanes of the outer's shape" (array int) [| 2; 3; 2 |] (Nx.shape y);
  equal ~msg:"lanes of the outer" floats
    [| 1.; 4.; 2.; 5.; 3.; 6.; 1.; 4.; 2.; 5.; 3.; 6. |]
    (Nx.to_array y);
  let y = nested (fun _ b x -> Rune.lanes b x) in
  equal ~msg:"lanes of the inner" floats
    [| 1.; 2.; 3.; 1.; 2.; 3.; 1.; 2.; 3.; 4.; 5.; 6.; 4.; 5.; 6.; 4.; 5.; 6. |]
    (Nx.to_array y);
  (* A value every lane shares, gathered: as many copies as lanes. *)
  let y =
    nested (fun a _ x -> Nx.add x (Nx.sum (Rune.lanes a (v [| 0.5 |]))))
  in
  equal ~msg:"lanes of a constant" floats
    [| 2.; 3.; 4.; 5.; 6.; 7. |]
    (Nx.to_array y);
  (* The sum over a map's lanes of a lane's own index. *)
  let y = nested (fun a _ x -> Nx.add x (Nx.sum (Rune.lanes a (index a x)))) in
  equal ~msg:"lanes of the index" floats
    [| 2.; 3.; 4.; 5.; 6.; 7. |]
    (Nx.to_array y)

(* A lane's elements, and lanes and lane_index outside their map *)

let test_reads () =
  let xs = v [| 1.; 2. |] in
  invalid ~sub:"Nx.item: the value is traced by Rune.vmap" (fun () ->
      Rune.vmap' (fun x -> if Nx.item [] x > 0. then x else Nx.neg x) xs);
  let a = Rune.axis () and b = Rune.axis () in
  let no = " no map named by the axis is around the call" in
  invalid ~sub:("Rune.lanes:" ^ no) (fun () ->
      Rune.vmap' ~axis:a (fun x -> Rune.lanes b x) xs);
  invalid ~sub:("Rune.lane_index:" ^ no) (fun () ->
      Rune.vmap' ~axis:a (fun x -> Nx.add x (index b x)) xs);
  (* An anonymous map is read by no name. *)
  invalid ~sub:("Rune.lane_index:" ^ no) (fun () ->
      Rune.vmap' (fun x -> Nx.add x (index a x)) xs);
  (* After the map returned. *)
  let escaped = ref (fun () -> Nx.zeros f64 [||]) in
  ignore
    (Rune.vmap' ~axis:a
       (fun x ->
         (escaped := fun () -> index a x);
         x)
       xs);
  invalid ~sub:("Rune.lane_index:" ^ no) (fun () -> !escaped ());
  (* Under a compiled function and in a loop's step inside the map, they read
     it. *)
  let y =
    Rune.vmap' ~axis:a
      (Rune.jit' (fun x -> Nx.add x (Nx.sum (Rune.lanes a (index a x)))))
      xs
  in
  equal ~msg:"under jit" floats [| 2.; 3. |] (Nx.to_array y);
  let y =
    Rune.vmap' ~axis:a
      (fun x ->
        fst
          (Rune.scan' ~init:x
             ~f:(fun c r -> (Nx.add c (Nx.mul r (index a r)), r))
             (v [| 1.; 10. |])))
      xs
  in
  equal ~msg:"in a step" floats [| 1.; 13. |] (Nx.to_array y)

(* Errors *)

let test_errors () =
  let s = T.(tensor @-> tensor @-> returns tensor) in
  invalid ~sub:"Rune.vmap: 1: 3 rows along axis 0, 0: 2" (fun () ->
      Rune.vmap s Nx.add (v [| 1.; 2. |]) (v [| 1.; 2.; 3. |]));
  invalid ~sub:"Rune.vmap': 0 is 0-d" (fun () ->
      Rune.vmap' Nx.sin (Nx.scalar f64 1.));
  invalid ~sub:"Rune.vmap: the arguments hold no array" (fun () ->
      Rune.vmap T.(unit @-> returns tensor) (fun () -> v [| 1. |]) ())

(* Totals *)

let test_totals () =
  let t = Rune.Total.make () in
  let xs = Nx.create f64 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 0.5; -0.25 |] in
  let collect ?(zero = Nx.zeros f64 [| 2 |]) f =
    snd (Rune.Total.collect t ~zero (fun () -> ignore (Rune.vmap' f xs)))
  in
  equal ~msg:"lanes' values" floats [| 4.5; 5.75 |]
    (Nx.to_array
       (collect (fun x ->
            Rune.Total.add t x;
            x)));
  equal ~msg:"a shared value, once per lane" floats [| 3.; 6. |]
    (Nx.to_array
       (collect (fun x ->
            Rune.Total.add t (v [| 1.; 2. |]);
            x)));
  let none = Nx.create f64 [| 0; 2 |] [||] in
  let _, total =
    Rune.Total.collect t
      ~zero:(v [| -0.; -0. |])
      (fun () ->
        Rune.vmap'
          (fun x ->
            Rune.Total.add t x;
            x)
          none)
  in
  equal ~msg:"no lane adds nothing" (array int64)
    [| Int64.bits_of_float (-0.); Int64.bits_of_float (-0.) |]
    (Array.map Int64.bits_of_float (Nx.to_array total));
  (* A collect inside the map receives its own lane's additions. *)
  let per_lane =
    Rune.vmap'
      (fun x ->
        snd
          (Rune.Total.collect t ~zero:(Nx.zeros f64 [| 2 |]) (fun () ->
               Rune.Total.add t x;
               Rune.Total.add t x)))
      xs
  in
  equal ~msg:"per lane" floats
    [| 2.; 4.; 6.; 8.; 1.; -0.5 |]
    (Nx.to_array per_lane)

(* Forward mode *)

let test_jvp () =
  let xs = Nx.create f64 [| 3; 2 |] [| 0.5; -1.; 2.; 0.; 1.5; -0.25 |] in
  let ts = Nx.create f64 [| 3; 2 |] [| 1.; 2.; -1.; 0.5; 3.; 1. |] in
  let f x = Nx.mul (Nx.sin x) (Nx.sum x) in
  let jvp2 x t = snd (Rune.jvp' f x t) in
  let expected = codes (one_by_one2 jvp2 xs ts) in
  equal ~msg:"vmap of jvp" codes_t expected
    (codes (Rune.vmap T.(tensor @-> tensor @-> returns tensor) jvp2 xs ts));
  equal ~msg:"jvp of vmap" codes_t expected
    (codes (snd (Rune.jvp' (Rune.vmap' f) xs ts)));
  (* lanes is linear: under forward mode its tangent is the tangents' lanes. *)
  let a = Rune.axis () in
  let g x = Nx.sum (Rune.lanes a (Nx.mul x x)) in
  let y =
    Rune.vmap
      T.(tensor @-> tensor @-> returns tensor)
      ~axis:a
      (fun x t -> snd (Rune.jvp' g x t))
      xs ts
  in
  (* The map's lanes move together: every lane's [lanes a (x x)] moves by each
     lane's [2 x t], and so does their sum. *)
  let moved =
    List.fold_left ( +. ) 0.
      (List.init 6 (fun e ->
           let k = e / 2 and i = e mod 2 in
           2. *. Nx.item [ k; i ] xs *. Nx.item [ k; i ] ts))
  in
  equal ~msg:"lanes under jvp" floats [| moved; moved; moved |] (Nx.to_array y)

(* A loop's output that reads no lane is shared: forward mode under a map of
   tangents computes its primal once, and reads of it read. *)
let test_shared_primal () =
  let x = Nx.place Count.on (v [| 1.; 2.; 3. |]) in
  let ts = Nx.place Count.on (Nx.ones f64 [| 4; 3 |]) in
  C.reset ();
  ignore (Rune.vmap' (fun t -> snd (Rune.jvp' Nx.sin x t)) ts);
  equal ~msg:"the primal at one lane's positions, the tangent at four"
    (pair int int) (2, 15)
    (C.calls (), C.positions ());
  let f x =
    let y = Nx.sin x in
    if Nx.item [ 0 ] y > 0. then y else Nx.neg y
  in
  equal ~msg:"a read under jacfwd'"
    (array (float 1e-15))
    [| Float.cos 1.; 0.; 0.; Float.cos 2. |]
    (Nx.to_array (Rune.jacfwd' f (v [| 1.; 2. |])))

(* A scan of two loads whose output reads the constant one alone: shared,
   computed once at one lane's positions. *)
let test_shared_scan () =
  let ins = Iarray.of_list [ any; any ] in
  let prog =
    P.v ~ins (Iarray.of_list [ P.In 0; In 1 ]) ~outs:(Iarray.of_list [ 1 ])
  in
  let k = Nx.place Count.on (v [| 1.; -2.; 0.5 |]) in
  let scan x =
    Nx.Prim.eval ~by:"test"
      (Scan
         {
           layout = contiguous [| 3 |];
           axis = 0;
           prog;
           reduction = Monoid (Sum, 0, f64);
           loads = [| Plain x; Plain k |];
           carry = None;
         })
  in
  let xs = Nx.place Count.on (Nx.ones f64 [| 4; 3 |]) in
  C.reset ();
  let y = Rune.vmap' scan xs in
  equal ~msg:"one lane's positions" (pair int int) (1, 3)
    (C.calls (), C.positions ());
  equal ~msg:"every lane's sums" floats
    (Array.concat (List.init 4 (fun _ -> [| 1.; -1.; -0.5 |])))
    (Nx.to_array (Nx.place Nx.Host.on y))

(* detach in a map, inside and around forward mode. *)
let test_detach () =
  let xs = Nx.create f64 [| 2; 2 |] [| 0.5; -1.; 2.; 3. |] in
  let ts = Nx.create f64 [| 2; 2 |] [| 1.; 2.; -1.; 0.5 |] in
  let f y = Nx.mul y (Rune.detach y) in
  let expected = Nx.to_array (Nx.mul xs ts) in
  equal ~msg:"jvp in the map" floats expected
    (Nx.to_array
       (Rune.vmap
          T.(tensor @-> tensor @-> returns tensor)
          (fun x t -> snd (Rune.jvp' f x t))
          xs ts));
  equal ~msg:"jvp of the map" floats expected
    (Nx.to_array (snd (Rune.jvp' (Rune.vmap' f) xs ts)))

(* jacfwd' is one map of jvp: its kernels do not grow with [x]'s elements. *)
let test_jacfwd_one_pass () =
  let calls n =
    let x = Nx.place Count.on (Nx.linspace f64 0.1 1. n) in
    C.reset ();
    ignore (Rune.jacfwd' (fun x -> Nx.mul (Nx.sin x) x) x);
    C.calls ()
  in
  equal ~msg:"3 against 7 elements" int (calls 3) (calls 7)

(* Reverse mode *)

let objective f x = Nx.sum (f x)

(* Per lane, the gradient of the sum of a case's function, mapped over the lanes
   and of the sum over the map's lanes. *)
let law_grad c (s, o) =
  cover "no lane" (s.(0) = 0);
  cover "several lanes" (s.(0) > 1);
  let xs = make f64 s o in
  let g = Rune.grad' (objective c.f) in
  let by_lane = outcome (fun () -> codes (one_by_one g xs)) in
  cover "a gradient" (match by_lane with Value _ -> true | Raised _ -> false);
  equal ~msg:"vmap of grad" (outcome_t codes_t) by_lane
    (outcome (fun () -> codes (Rune.vmap' g xs)));
  equal ~msg:"grad of vmap" (outcome_t codes_t) by_lane
    (outcome (fun () -> codes (Rune.grad' (objective (Rune.vmap' c.f)) xs)))

let grad_props = List.map (fun c -> prop c.name (gen c) (law_grad c)) cases

let law_grad_compose (Pair (dt, s, xo, yo, both, e)) =
  cover "a constant operand" (not both);
  cover "several lanes" (s.(0) > 1);
  let x = make_any dt s xo in
  let lane = Array.sub s 1 (Array.length s - 1) in
  let y = make_any dt (if both then s else lane) yo in
  let g x y = Rune.grad' (fun x -> Nx.sum (eval_any x y e)) x in
  let t = outcome_t codes_t in
  if both then begin
    let by_lane = outcome (fun () -> codes (one_by_one2 g x y)) in
    cover "a gradient of two lanes"
      (match by_lane with Value _ -> true | Raised _ -> false);
    equal ~msg:"vmap of grad" t by_lane
      (outcome (fun () ->
           codes (Rune.vmap T.(tensor @-> tensor @-> returns tensor) g x y)));
    equal ~msg:"grad of vmap" t by_lane
      (outcome (fun () ->
           codes
             (Rune.grad'
                (fun x ->
                  Nx.sum
                    (Rune.vmap
                       T.(tensor @-> tensor @-> returns tensor)
                       (fun x y -> eval_any x y e)
                       x y))
                x)))
  end
  else begin
    let by_lane = outcome (fun () -> codes (one_by_one (fun x -> g x y) x)) in
    cover "a gradient of a lane and a constant"
      (match by_lane with Value _ -> true | Raised _ -> false);
    equal ~msg:"vmap of grad" t by_lane
      (outcome (fun () -> codes (Rune.vmap' (fun x -> g x y) x)));
    equal ~msg:"grad of vmap" t by_lane
      (outcome (fun () ->
           codes
             (Rune.grad'
                (fun x -> Nx.sum (Rune.vmap' (fun x -> eval_any x y e) x))
                x)))
  end

(* [lanes a] under reverse mode: a lane's cotangent collects every lane's use of
   its value. Lane [l]'s objective weighs every lane's [x x] by [w] and by [l +
   1], so the gradient at lane [k] is [Σ_l (l + 1) 2 x_k w_k], or [12 x_k w_k]
   over three lanes; one whose weights are complex and linear in [z] has the
   weights' sum, [6 c_k]. *)
let test_grad_lanes () =
  let xs = Nx.create f64 [| 3; 2 |] [| 0.5; -1.; 2.; 0.; 1.5; -0.25 |] in
  let w = Nx.create f64 [| 3; 2 |] [| 1.; -2.; 0.5; 3.; -1.5; 4. |] in
  let a = Rune.axis () in
  let weight () = Nx.mul w (Nx.add_s (index a w) 1.) in
  let f x = Nx.sum (Nx.mul (Rune.lanes a (Nx.mul x x)) (weight ())) in
  let expected = Nx.to_array (Nx.mul_s (Nx.mul xs w) 12.) in
  equal ~msg:"in the map" floats expected
    (Nx.to_array (Rune.vmap' ~axis:a (Rune.grad' f) xs));
  equal ~msg:"compiled in the map" floats expected
    (Nx.to_array (Rune.vmap' ~axis:a (Rune.jit' (Rune.grad' f)) xs));
  let c =
    Nx.complex c128 ~re:w
      ~im:(Nx.create f64 [| 3; 2 |] [| 0.5; 1.; -1.; 2.; 0.; -3. |])
  in
  let zs = Nx.complex c128 ~re:xs ~im:(Nx.neg xs) in
  let g z =
    let k = Nx.cast c128 (Nx.add_s (index a z) 1.) in
    Nx.real f64 (Nx.sum (Nx.mul (Rune.lanes a z) (Nx.mul c k)))
  in
  let parts z =
    Array.map (fun (c : Complex.t) -> (c.re, c.im)) (Nx.to_array z)
  in
  equal ~msg:"complex"
    (array (pair float_exact float_exact))
    (parts (Nx.mul_s c { Complex.re = 6.; im = 0. }))
    (parts (Rune.vmap' ~axis:a (Rune.grad' g) zs));
  (* Outside the map, lanes are the map's batch, and the objective is the sum of
     the lanes' objectives. *)
  let g = Rune.grad' (fun xs -> Nx.sum (Rune.vmap' ~axis:a f xs)) xs in
  equal ~msg:"grad of the map" floats expected (Nx.to_array g)

(* jacrev' is one map of the pullback: its kernels do not grow with the result's
   elements. *)
let test_jacrev_one_pass () =
  let calls n =
    let x = Nx.place Count.on (Nx.linspace f64 0.1 1. n) in
    C.reset ();
    ignore (Rune.jacrev' (fun x -> Nx.mul (Nx.sin x) x) x);
    C.calls ()
  in
  equal ~msg:"3 against 7 elements" int (calls 3) (calls 7)

(* Compiled functions *)

(* A compiled call is eager's within rune2.mli's classes: bit for bit where
   each function is exact at the dtype, as test_jit's laws state it. *)
let law_jit (Pair (dt, s, xo, _, _, e)) =
  let e = exact_any dt e in
  let x = make_any dt s xo in
  let f x = eval_any x x e in
  let mapped = outcome (fun () -> codes (Rune.vmap' f x)) in
  equal ~msg:"vmap of jit" (outcome_t codes_t) mapped
    (outcome (fun () -> codes (Rune.vmap' (Rune.jit' f) x)));
  equal ~msg:"jit of vmap" (outcome_t codes_t) mapped
    (outcome (fun () -> codes (Rune.jit' (Rune.vmap' f) x)))

(* Loops *)

(* A step that reads a lane's value it captures, from a carry every lane shares:
   the carry gains lanes after the first step. *)
let test_scan () =
  let xs = Nx.create f64 [| 3; 2 |] [| 0.5; -1.; 2.; 0.; 1.5; -0.25 |] in
  let rows = Nx.create f64 [| 4; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |] in
  let f x =
    let c, ys =
      Rune.scan' ~init:(Nx.zeros f64 [| 2 |])
        ~f:(fun c r -> (Nx.add (Nx.mul c x) r, Nx.sin c))
        rows
    in
    Nx.concatenate ~axis:0 [ c; Nx.reshape [| -1 |] ys ]
  in
  equal codes_t (codes (one_by_one f xs)) (codes (Rune.vmap' f xs));
  (* An iterate whose until every lane shares steps its lanes together; one
     whose until reads a lane stops each lane on its own, and names the first
     lane still running at the bound. *)
  let c = T.(pair tensor tensor) in
  let g x =
    snd
      (Rune.iterate c ~max:5
         ~until:(fun (k, _) -> Nx.greater_equal_s k 3.)
         ~f:(fun (k, y) -> (Nx.add_s k 1., Nx.mul (Nx.sin y) x))
         (Nx.scalar f64 0., x))
  in
  equal ~msg:"iterate" codes_t
    (codes (one_by_one g xs))
    (codes (Rune.vmap' g xs));
  let h =
    Rune.iterate' ~max:5
      ~until:(fun y -> Nx.greater_s (Nx.sum y) 1.)
      ~f:(fun y -> Nx.add_s (Nx.mul_s y 2.) 0.5)
  in
  equal ~msg:"lanes stopping apart" codes_t
    (codes (one_by_one h xs))
    (codes (Rune.vmap' h xs));
  invalid
    ~sub:"Rune.iterate': until is still false after max = 5 steps, in lane 0"
    (fun () ->
      Rune.vmap'
        (Rune.iterate' ~max:5
           ~until:(fun y -> Nx.greater_s (Nx.sum y) 10.)
           ~f:(fun y -> Nx.mul_s y 2.))
        xs)

(* Placements *)

let test_place () =
  let host = Nx.create f64 [| 4; 3 |] (Array.init 12 Float.of_int) in
  let cut = Nx.place (Two.split ~axis:0) host in
  let placement y = Nx.placement y in
  (* A lane of a batch cut along axis 0 lies whole on each device. *)
  let seen = ref None in
  let y =
    Rune.vmap'
      (fun x ->
        seen := Some (Nx.Placement.equal (placement x) Two.on);
        Nx.place (placement x) (Nx.sin x))
      cut
  in
  equal ~msg:"a lane's placement" (option bool) (Some true) !seen;
  equal ~msg:"each lane computes on its device" bool true
    (Nx.Placement.equal (placement (Rune.vmap' Nx.exp cut)) (Two.split ~axis:0));
  equal ~msg:"placed at its own placement, the cut stays" bool true
    (Nx.Placement.equal (placement y) (Two.split ~axis:0));
  equal ~msg:"values" codes_t
    (codes (Nx.sin host))
    (codes (Nx.place Nx.Host.on y));
  (* Each lane placed elsewhere: stacked, cut along the lanes' axis 0. *)
  let y =
    Rune.vmap' (fun x -> Nx.place (Two.split ~axis:0) x) (Nx.place Two.on host)
  in
  equal ~msg:"each lane cut" bool true
    (Nx.Placement.equal (placement y) (Two.split ~axis:1));
  equal ~msg:"its values" codes_t (codes host) (codes (Nx.place Nx.Host.on y))

(* A set no kernel library computes on, as a GPU set is outside a compiler:
   a memory device and an io device, which does not run on the host. *)
module Bare =
  (val Nx.devices [ Nx_support.memory 3; Nx_array_support.io_device () ])

(* A creation in a lane reads no lane, so the map computes it now: on a set
   without kernels the host makes it, and every lane holds it. *)
let test_bare () =
  let xs = Nx.zeros_at Bare.on f64 [| 2; 3 |] in
  let check msg want f =
    let y = Rune.vmap' f xs in
    equal ~msg:(msg ^ ": placement") (Nx_support.placement ()) Bare.on
      (Nx.placement y);
    equal ~msg codes_t (codes want) (codes (Nx.place Nx.Host.on y))
  in
  check "eye_at" (Nx.stack [ Nx.eye f64 3; Nx.eye f64 3 ]) (fun x ->
      Nx.eye_at (Nx.placement x) f64 3);
  check "full_like" (Nx.full f64 [| 2; 3 |] 2.) (fun x -> Nx.full_like x 2.);
  (* An operation that needs none of its lane's elements still computes
     beside it, so it raises as it does outside the map. *)
  invalid ~sub:"has no kernels to compute" (fun () ->
      Rune.vmap'
        (fun x ->
          Nx.matrix_power (Nx.broadcast_to [| 3; 3 |] (Nx.get [ 0 ] x)) 0)
        xs)

(* A check raises its first failing lane's exception, from that lane's index and
   data. *)
exception Failed of int array * float

let test_check () =
  let xs = Nx.create f64 [| 3; 4 |] (Array.init 12 Float.of_int) in
  raises ~msg:"lane 1, index 2"
    (Failed ([| 2 |], 6.))
    (fun () ->
      Rune.vmap'
        (fun x ->
          Nx.check T.tensor (Nx.less_s x 6.) x (fun i d ->
              Failed (i, Nx.item [] d));
          x)
        xs)

(* A draw without a key is one every lane shares; one from each lane's key is
   its own. *)
let test_draws () =
  let xs = Nx.zeros f64 [| 3; 2 |] in
  let y = Rune.vmap' (fun x -> Nx.add x (Nx.Rng.uniform f64 [| 2 |])) xs in
  equal ~msg:"shared" floats
    (Nx.to_array (Nx.get [ 0 ] y))
    (Nx.to_array (Nx.get [ 2 ] y));
  let a = Rune.axis () and k = Nx.Rng.key 7 in
  let own () =
    Nx.Rng.fold_in_tensor k (Rune.lane_index a (Nx.Rng.to_tensor k))
  in
  let y =
    Rune.vmap' ~axis:a
      (fun x -> Nx.add x (Nx.Rng.uniform ~key:(own ()) f64 [| 2 |]))
      xs
  in
  for i = 0 to 2 do
    equal
      ~msg:(Printf.sprintf "lane %d's key" i)
      floats
      (Nx.to_array (Nx.Rng.uniform ~key:(Nx.Rng.fold_in k i) f64 [| 2 |]))
      (Nx.to_array (Nx.get [ i ] y))
  done

let () =
  exit
  @@ run "rune2 vmap"
       [
         group "operations" operation_props;
         group "contractions"
           [
             prop "segments and scales, any operand's lanes" contractions
               law_contract;
             prop "rows past nx.cpu's lane order, within the bound" wide
               law_wide;
           ];
         group "compositions"
           [
             prop "float compositions" float_compositions
               (law_compose ~kinds:false);
             prop "every dtype" every_dtype (law_compose ~kinds:true);
             prop "under jit" every_dtype law_jit;
           ];
         group "reverse mode"
           (grad_props
           @ [
               prop "compositions" float_compositions law_grad_compose;
               test "lanes" test_grad_lanes;
               test "jacrev' in one pass" test_jacrev_one_pass;
             ]);
         group "names"
           [
             test "nested maps" test_named;
             test "reads raise" test_reads;
             test "errors" test_errors;
           ];
         group "constructs"
           [
             test "totals" test_totals;
             test "forward mode" test_jvp;
             test "jacfwd' in one pass" test_jacfwd_one_pass;
             test "a shared primal" test_shared_primal;
             test "a shared scan" test_shared_scan;
             test "detach" test_detach;
             test "loops" test_scan;
             test "placements" test_place;
             test "creations on a set without kernels" test_bare;
             test "checks" test_check;
             test "draws" test_draws;
           ];
       ]
