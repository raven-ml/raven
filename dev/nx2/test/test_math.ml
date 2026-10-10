(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The float functions nx composes from the kernels' kinds: each within its
   stated ulps of the C library's at float32 and float64, over values drawn
   from bytes, its special values as stated, and a narrower float computed at
   float32 and rounded once. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout

let invalid ~by f = raises_match (Exn.invalid_arg ~substring:(by ^ ": ")) f

(* A host value of [dt] and shape [[n]] over drawn bytes. *)
let drawn (type s) (dt : (float, s) D.t) n : (float, s, Nx.host) Nx.t Gen.t =
  let open Gen in
  let+ data = string_of ~size:(constant (D.bytes dt n)) char in
  Nx.Repr.of_array Nx.Host.v
    (A.v dt (L.contiguous [| n |]) (Rig.Buffer.of_string data))

(* The distance between [r] and the next float of [dt]'s format away from
   zero: the unit in the last place at [r]. *)
let ulp (type s) (dt : (float, s) D.t) r =
  let r = Float.abs r in
  match dt with
  | D.Float32 ->
      let b = Int32.bits_of_float r in
      Int32.float_of_bits (Int32.succ b) -. Int32.float_of_bits b
  | _ -> Float.succ r -. r

(* [v] is within [k] ulps of [r], or [r] is a NaN, an infinity or a zero, in
   [dt]'s format, and [v] is that value, sign included. *)
let close dt k r v =
  let r = if Float.is_finite (D.of_float dt r) then r else D.of_float dt r in
  if Float.is_nan r then Float.is_nan v
  else if r = 0. || Float.abs r = infinity then
    Int64.equal (Int64.bits_of_float r) (Int64.bits_of_float v)
  else Float.abs (v -. r) <= k *. ulp dt r

let float_pp ppf v = Format.fprintf ppf "%h" v

let within dt k xs refs got =
  Array.iteri
    (fun i r ->
      let v = got.(i) in
      if not (close dt k r v) then
        failf "at %a: %a, the C library's %a, beyond %g ulps" float_pp xs.(i)
          float_pp v float_pp r k)
    refs

let law_unary (type s) (dt : (float, s) D.t)
    (f : (float, s, Nx.host) Nx.t -> (float, s, Nx.host) Nx.t) libm x =
  let xs = Nx.to_array x in
  within dt 3. xs (Array.map libm xs) (Nx.to_array (f x))

let samples = 64

let accuracy name (f : 's. (float, 's, Nx.host) Nx.t -> (float, 's, Nx.host) Nx.t) libm =
  group name
    [
      prop "float64 within 3 ulps" (drawn D.Float64 samples) (fun x ->
          law_unary D.Float64 f libm x);
      prop "float32 within 3 ulps" (drawn D.Float32 samples) (fun x ->
          law_unary D.Float32 f libm x);
    ]

let narrow_dtypes =
  List.filter
    (fun (D.Any dt) -> D.is D.Float dt && D.bits dt < 32)
    D.all

(* Every code of [dt], a float narrower than float32. *)
let codes (type s) (dt : (float, s) D.t) : (float, s, Nx.host) Nx.t =
  let every =
    match D.bits dt with
    | 4 -> Nx.bitcast dt (Nx.arange Nx.uint4 0 16 1)
    | 8 -> Nx.bitcast dt (Nx.arange Nx.uint8 0 256 1)
    | _ -> Nx.bitcast dt (Nx.arange Nx.uint16 0 65536 1)
  in
  Nx.place Nx.Host.on every

let same a b =
  (Float.is_nan a && Float.is_nan b)
  || Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

(* At a float narrower than float32, [f x] is [f] at float32 rounded once. *)
let law_narrow (f : 's. (float, 's, Nx.host) Nx.t -> (float, 's, Nx.host) Nx.t)
    (D.Any dt) =
  match D.kind dt with
  | D.Float ->
      let x = codes dt in
      let expected = Nx.cast dt (f (Nx.cast Nx.float32 x)) in
      Array.iter2
        (fun e g -> if not (same e g) then failf "%h, not %h" g e)
        (Nx.to_array expected) (Nx.to_array (f x))
  | _ -> ()

let narrow name (f : 's. (float, 's, Nx.host) Nx.t -> (float, 's, Nx.host) Nx.t) =
  cases (name ^ " at a narrow float is float32's, rounded once")
    ~name:(fun (D.Any dt) -> D.name dt)
    narrow_dtypes
    (law_narrow (fun x -> f x))

(* Special values *)

let values f dt xs = Nx.to_array (f (Nx.create dt [| Array.length xs |] xs))

let specials () =
  let f64 = Nx.float64 in
  equal (array float_exact) [| -0.; 0.; infinity; neg_infinity; nan |]
    (values Nx.asinh f64 [| -0.; 0.; infinity; neg_infinity; nan |]);
  equal (array float_exact) [| 0.; nan; nan; infinity |]
    (values Nx.acosh f64 [| 1.; 0.999; neg_infinity; infinity |]);
  equal (array float_exact) [| infinity; neg_infinity; nan; nan; -0. |]
    (values Nx.atanh f64 [| 1.; -1.; 1.5; -2.; -0. |]);
  equal (array float_exact) [| infinity; neg_infinity; nan |]
    (values Nx.rsqrt f64 [| 0.; -0.; -1. |])

let hypot_specials () =
  let v a b = Nx.item [] (Nx.hypot (Nx.scalar Nx.float32 a) (Nx.scalar Nx.float32 b)) in
  equal float_exact infinity (v nan infinity);
  equal float_exact infinity (v neg_infinity nan);
  equal float_exact 0. (v 0. (-0.));
  equal bool true (Float.is_nan (v nan 1.));
  equal float_exact 5. (v 3. 4.);
  equal float_exact 0x5p100 (v 0x3p100 0x4p100)

let law_hypot (type s) (dt : (float, s) D.t) (x, y) =
  let xs = Nx.to_array x and ys = Nx.to_array y in
  within dt 3. xs (Array.map2 Float.hypot xs ys) (Nx.to_array (Nx.hypot x y))

let test_square () =
  equal (array int) [| -112; 0; 1 |]
    (Nx.to_array (Nx.square (Nx.create Nx.int8 [| 3 |] [| 12; -128; -1 |])));
  equal (array float_exact) [| 0.; 9.; infinity |]
    (Nx.to_array (Nx.square (Nx.create Nx.float32 [| 3 |] [| -0.; 3.; 1e20 |])));
  equal (array bool) [| true |]
    (Nx.to_array
       (Nx.equal
          (Nx.square (Nx.create Nx.complex64 [| 1 |] [| { Complex.re = 1.; im = 2. } |]))
          (Nx.create Nx.complex64 [| 1 |] [| { Complex.re = -3.; im = 4. } |])))

let math =
  [
    accuracy "asinh" (fun x -> Nx.asinh x) Float.asinh;
    accuracy "acosh" (fun x -> Nx.acosh x) Float.acosh;
    accuracy "atanh" (fun x -> Nx.atanh x) Float.atanh;
    group "hypot"
      [
        prop "float64 within 3 ulps"
          (Gen.pair (drawn D.Float64 samples) (drawn D.Float64 samples))
          (law_hypot D.Float64);
        prop "float32 within 3 ulps"
          (Gen.pair (drawn D.Float32 samples) (drawn D.Float32 samples))
          (law_hypot D.Float32);
        test "special values" hypot_specials;
      ];
    group "narrow floats"
      [
        narrow "asinh" (fun x -> Nx.asinh x);
        narrow "acosh" (fun x -> Nx.acosh x);
        narrow "atanh" (fun x -> Nx.atanh x);
        narrow "rsqrt" (fun x -> Nx.rsqrt x);
        narrow "hypot" (fun x -> Nx.hypot x (Nx.flip x));
      ];
    group "special values"
      [
        test "square wraps integers and squares complex numbers" test_square;
        test "zeros, ends of the domain, infinities and NaN" specials;
        cases "a function of floats refuses another dtype"
          ~name:(fun (n, _) -> n)
          [
            ("Nx.rsqrt", fun () -> ignore (Nx.rsqrt (Nx.zeros Nx.int32 [| 1 |])));
            ("Nx.hypot", fun () ->
                let z = Nx.zeros Nx.complex64 [| 1 |] in
                ignore (Nx.hypot z z));
            ("Nx.asinh", fun () -> ignore (Nx.asinh (Nx.zeros Nx.bool [| 1 |])));
            ("Nx.acosh", fun () -> ignore (Nx.acosh (Nx.zeros Nx.uint8 [| 1 |])));
            ("Nx.atanh", fun () -> ignore (Nx.atanh (Nx.zeros Nx.int64 [| 1 |])));
            ("Nx.square", fun () -> ignore (Nx.square (Nx.zeros Nx.bool [| 1 |])));
          ]
          (fun (by, f) -> invalid ~by f);
      ];
  ]

let () = exit (run "nx math" math)
