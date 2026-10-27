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

(* C99 Annex F's special values, F.10, at float64 and float32. *)
let specials () =
  let check (type s) (dt : (float, s) D.t) =
    equal (array float_exact) [| -0.; 0.; infinity; neg_infinity; nan |]
      (values Nx.asinh dt [| -0.; 0.; infinity; neg_infinity; nan |]);
    equal (array float_exact) [| 0.; nan; nan; infinity; nan |]
      (values Nx.acosh dt [| 1.; 0.5; neg_infinity; infinity; nan |]);
    equal (array float_exact)
      [| infinity; neg_infinity; nan; nan; -0.; 0.; nan; nan; nan |]
      (values Nx.atanh dt
         [| 1.; -1.; 1.5; -2.; -0.; 0.; infinity; neg_infinity; nan |]);
    equal (array float_exact) [| infinity; neg_infinity; nan |]
      (values Nx.rsqrt dt [| 0.; -0.; -1. |])
  in
  check Nx.float64;
  check Nx.float32

(* A format without infinities or NaN stores an infinite or NaN result as a
   cast does: float8_e4m3fn saturates, and float4_e2m1fn holds NaN as +0. *)
let no_infinities () =
  let e4m3 = Nx.float8_e4m3fn and e2m1 = Nx.float4_e2m1fn in
  equal (array float_exact) [| -448. |] (values Nx.log e4m3 [| 0. |]);
  equal (array float_exact) [| 448. |] (values Nx.atanh e4m3 [| 1. |]);
  equal (array float_exact) [| 448. |] (values Nx.exp e4m3 [| 100. |]);
  equal (array float_exact) [| -6.; 0. |] (values Nx.log e2m1 [| 0.; -1. |])

(* At float64 and float32, which compute it differently. *)
let hypot_specials () =
  let check (type s) (dt : (float, s) D.t) =
    let v a b = Nx.item [] (Nx.hypot (Nx.scalar dt a) (Nx.scalar dt b)) in
    equal float_exact infinity (v nan infinity);
    equal float_exact infinity (v neg_infinity nan);
    equal float_exact 0. (v 0. (-0.));
    equal float_exact 0. (v (-0.) (-0.));
    equal float_exact 3. (v (-3.) (-0.));
    equal float_exact infinity (v 2. neg_infinity);
    equal bool true (Float.is_nan (v nan 1.));
    equal float_exact 5. (v 3. 4.);
    equal float_exact 0x5p100 (v 0x3p100 0x4p100)
  in
  check Nx.float64;
  check Nx.float32

(* rsqrt's error at [x] in ulps of the exact [e = 1/√x], where [v = e (1 + d)]:
   [v² x - 1 = 2d + d²], which a square split exactly by fma and one fma
   give far below an ulp. [x] and [v] are first scaled by [2^-2k] and [2^k],
   which keeps [v² x] and puts [x] in [\[1/2, 2)], where [v²] cannot
   overflow. *)
let rsqrt_ulps dt x v =
  let _, ex = Float.frexp x in
  let k = if ex >= 0 then ex / 2 else (ex - 1) / 2 in
  let x = Float.ldexp x (-2 * k) and v = Float.ldexp v k in
  let p = v *. v in
  let pe = Float.fma v v (-.p) in
  let t = Float.fma p x (-1.) +. (pe *. x) in
  let d = t /. (1. +. Float.sqrt (1. +. t)) in
  let e = Float.ldexp (v /. (1. +. d)) (-k) in
  Float.abs d *. e /. ulp dt e

let law_rsqrt (type s) (dt : (float, s) D.t) (x : (float, s, Nx.host) Nx.t) =
  let xs = Nx.to_array x and vs = Nx.to_array (Nx.rsqrt x) in
  Array.iteri
    (fun i x ->
      let v = vs.(i) in
      if x > 0. && Float.is_finite x then begin
        let e = rsqrt_ulps dt x v in
        if not (e <= 2.) then
          failf "at %a: %a, %g ulps from 1/√x" float_pp x float_pp v e
      end
      else if not (same (1. /. Float.sqrt x) v) then
        failf "at %a: %a, not %a" float_pp x float_pp v float_pp
          (1. /. Float.sqrt x))
    xs

let rsqrt_ends (type s) (dt : (float, s) D.t) =
  let least = match dt with D.Float32 -> 0x1p-149 | _ -> 0x1p-1074 in
  let most =
    match dt with D.Float32 -> 0x1.fffffep127 | _ -> Float.max_float
  in
  law_rsqrt dt
    (Nx.create dt [| 7 |]
       [| least; 3. *. least; 0x1p-126; 0.5; 2.; 3.; most |])

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
    group "rsqrt"
      [
        prop "float64 within 2 ulps" (drawn D.Float64 samples)
          (law_rsqrt D.Float64);
        prop "float32 within 2 ulps" (drawn D.Float32 samples)
          (law_rsqrt D.Float32);
        test "at the ends of the range" (fun () ->
            rsqrt_ends D.Float64;
            rsqrt_ends D.Float32);
      ];
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
        test "a format without infinities or NaN stores as a cast"
          no_infinities;
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
