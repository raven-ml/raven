(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.jvp: tangents against central differences at float64 for every kind of
   scalar program and every operation that has a derivative, complex functions
   as maps of their parts, the kinks' tangents, and a map's tangent computed
   with its primal in one kernel. *)

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

(* A program of nine operands, [Σ_k x_k Π_(j<k) sin x_j], whose derivative takes
   more operands than one loop does. *)
let wide = 9

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
    case ~shape:rows_wide "a map of nine operands" wide_map;
    case ~empty:false ~shape:padded_shape "pad" (Nx.pad [| (1, 0); (2, 1) |] 3.);
    case ~empty:false
      ~shape:Gen.(map (fun n -> [| 2; n + 3 |]) (int_range 0 3))
      "extract_patches"
      (fun x ->
        Nx.extract_patches ~kernel_size:[| 2 |] ~stride:[| 1 |]
          ~dilation:[| 1 |]
          ~padding:[| (1, 0) |]
          x);
    case ~shape:(rows 2) "concatenate" (fun x ->
        Nx.concatenate ~axis:0 [ row 1 x; Nx.sin (row 0 x); row 1 x ]);
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
  ]

let tangent_props =
  List.map (fun c -> prop c.name (gen c) (law c)) (real_cases @ complex_cases)

(* Values the conventions state *)

let floats = array (float 1e-12)
let tangent f x t = Nx.to_array (snd (Rune.jvp' f x t))

(* JAX's: [abs] passes the tangent at either zero; tied operands of an extreme
   share it by halves, and a NaN takes none. *)
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
    (calls (fun x -> Nx.exp (Nx.mul (Nx.sin x) x)))

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
           ];
         group "kernels"
           [ test "a map's tangent is one kernel" test_one_kernel ];
         group "interface"
           [
             test "structures and zero tangents" test_structures;
             test "detach" test_detach;
             test "a read reads the primal" test_read;
             test "nesting" test_nesting;
             test "totals" test_total;
           ];
       ])
