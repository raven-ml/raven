(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The float functions nx composes from the kernels' kinds: each within its
   stated ulps of the C library's at float32 and float64, over values drawn
   from bytes, its special values as stated, and a narrower float computed at
   float32 and rounded once. At float32, log is one of the two floats around
   the exact value and asinh within 2 ulps, each against the C library's
   float64 result. roundeven is exact: at every float32 that rounds to neither
   zero nor itself and a sample of the rest, and at float64 over drawn bits
   and halves, it is the float64 arithmetic's ties to even. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout

let invalid ~by f = raises_match (Exn.invalid_arg ~sub:(by ^ ": ")) f

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
    ~id:(fun (D.Any dt) -> D.name dt)
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

(* sinc *)

(* [π] less its float64 head. *)
let pi_tail = 0x1.1a62633145c07p-53

(* [sin (πx) / (πx)] within about an ulp of float64: [sin (πx)] is [±sin (πr)]
   for [r = x - k], [k] the nearest integer, exact; [πr] and [πx] are each a
   head and a tail by fma, and the quotient is corrected by its residual. *)
let sinc_ref x =
  if x = 0. then 1.
  else if not (Float.is_finite x) then Float.nan
  else if Float.is_integer x then 0.
  else
    let k = Float.round x in
    let times a =
      let h = Float.pi *. a in
      (h, Float.fma Float.pi a (-.h) +. (pi_tail *. a))
    in
    let sh, sl = times (x -. k) and dh, dl = times x in
    let s = Float.sin sh +. (sl *. Float.cos sh) in
    let q = s /. dh in
    let q = q +. ((Float.fma (-.q) dh s -. (q *. dl)) /. dh) in
    if Float.rem k 2. = 0. then q else -.q

(* Each [sinc x] is within [k] ulps of [dt] of the exact value, and [0], of
   either sign, where that is [0]. nx states no bound for sinc. Its parts give
   one of 8 ulps: [sin]'s stated 2, which is 4 relative at most, and one
   rounding each of [π], [πr], [πx] and the quotient, [π]'s entering once since
   [πr cot πr] lies in [[0, 1]]. The tolerances, 4 ulps at float32 and 6 at
   float64, are tighter: the largest errors measured over ten million draws, 2.9
   and 5 ulps against this reference, which is itself within an ulp, rounded up.
   They are a baseline that a less accurate form would break. *)
let law_sinc (type s) (dt : (float, s) D.t) k (x : (float, s, Nx.host) Nx.t) =
  let xs = Nx.to_array x and vs = Nx.to_array (Nx.sinc x) in
  Array.iteri
    (fun i x ->
      let r = sinc_ref x and v = vs.(i) in
      let ok =
        if Float.is_nan r then Float.is_nan v
        else if r = 0. then v = 0.
        else Float.abs (v -. r) <= k *. ulp dt r
      in
      if not ok then
        failf "at %a: %a, the exact value %a, beyond %g ulps" float_pp x
          float_pp v float_pp r k)
    xs

(* Values in [-8, 8], near [0], near the integers, where the zeros lie, and from
   bytes. *)
let sinc_values (type s) (dt : (float, s) D.t) =
  let open Gen in
  let near_int =
    let+ n = int_range (-6) 6 and+ e = int_range 1 40 and+ s = bool in
    Float.of_int n +. Float.ldexp (if s then 1. else -1.) (-e)
  in
  let one =
    frequency
      [
        (3, float_range (-8.) 8.);
        (1, map (fun e -> Float.ldexp 1. (-e)) (int_range 1 1000));
        (2, near_int);
      ]
  in
  let+ v = array ~size:(constant samples) one and+ b = drawn dt samples in
  Nx.concatenate ~axis:0 [ Nx.create dt [| samples |] v; b ]

let law_sinc_drawn dt k x =
  let has p = Array.exists p (Nx.to_array x) in
  cover "below 1/4, not 0" (has (fun x -> x <> 0. && Float.abs x < 0.25));
  cover "from 1/4 to 8"
    (has (fun x -> Float.abs x >= 0.25 && Float.abs x <= 8.));
  cover "near an integer"
    (has (fun x -> x <> Float.round x && Float.abs (x -. Float.round x) < 1e-3));
  law_sinc dt k x

(* In every float format: [1] at both zeros, NaN at NaN and the infinities, and
   [0] at the integers but [0], up to the largest finite value. *)
let sinc_specials () =
  let check (type s) (dt : (float, s) D.t) =
    let f = D.float_format dt in
    let small = Array.init 32 (fun i -> Float.of_int (i + 1)) in
    let large = [| 1000.; 12345.; 0x1p23 -. 1.; 0x1p52 +. 2.; f.max_finite |] in
    let ints = Array.append small large in
    let ints = Array.append ints (Array.map Float.neg ints) in
    Array.iteri
      (fun i v ->
        let x = D.of_float dt ints.(i) in
        if Float.is_integer x && x <> 0. && v <> 0. then
          failf "%s: sinc %h is %h, not 0" (D.name dt) x v)
      (values Nx.sinc dt ints);
    equal (array float_exact) [| 1.; 1. |] (values Nx.sinc dt [| 0.; -0. |]);
    if f.infinities then
      equal (array bool) [| true; true; true |]
        (Array.map Float.is_nan
           (values Nx.sinc dt [| nan; infinity; neg_infinity |]))
  in
  List.iter
    (fun (D.Any dt) -> match D.kind dt with D.Float -> check dt | _ -> ())
    D.all

(* float32 log and asinh *)

(* [e^-3] rounded to float32: its exact log lies 0.13 ulp below [-3.]. *)
let e_minus_3 = 0x3d4bed86l

(* Bit patterns of positive finite float32: subnormals, values near [1.], the
   neighbours of [e_minus_3], values from [2^28], where asinh changes form, and
   any. *)
let positive_bits =
  Gen.(
    frequency
      [
        (1, int32_range 1l 0x007f_ffffl);
        (1, int32_range 0x3f7f_0000l 0x3f81_0000l);
        (1, int32_range (Int32.sub e_minus_3 4l) (Int32.add e_minus_3 4l));
        (1, int32_range 0x4d80_0000l 0x7f7f_ffffl);
        (2, int32_range 1l 0x7f7f_ffffl);
      ])

let float32_values bits =
  let open Gen in
  let+ bits = array ~size:(constant 64) bits in
  Array.map Int32.float_of_bits bits

let covers xs =
  let has p = Array.exists p xs in
  cover "a subnormal" (has (fun x -> Float.abs x < 0x1p-126));
  cover "near 1" (has (fun x -> Float.abs (Float.abs x -. 1.) < 0x1p-7));
  cover "e^-3 rounded" (has (fun x -> x = Int32.float_of_bits e_minus_3));
  cover "from 2^28" (has (fun x -> Float.abs x >= 0x1p28))

(* A float32's place in the order of float32 bit patterns, [-0.] just below
   [+0.]. *)
let rank x =
  let b = Int32.to_int (Int32.bits_of_float x) in
  if b >= 0 then b else -(b land 0x7fff_ffff) - 1

(* [v] rounded to the nearest float32, and the ranks of the greatest float32
   at most [v] and the least at least [v]. *)
let round32 v = Int32.float_of_bits (Int32.bits_of_float v)

let below v =
  let f = round32 v in
  if f > v then rank f - 1 else rank f

let above v =
  let f = round32 v in
  if f < v then rank f + 1 else rank f

(* The C library's float64 [log] and [asinh] are taken within [2^-48] of
   the exact value, relatively: far below a float32 ulp, [2^-23] relatively. *)
let reference f x =
  let y = f x in
  (y, Float.abs y *. 0x1p-48)

(* Each result, of [x] and of [x] reversed, a view, lies within [ranks] of
   [low y] to [high y] for the exact value [y]. *)
let ranks_within ~name f libm ~low ~high xs =
  let x = Nx.create Nx.float32 [| Array.length xs |] xs in
  let got = Nx.to_array (f x) in
  let reversed = Nx.to_array (f (Nx.flip x)) in
  let n = Array.length xs in
  Array.iteri
    (fun i x ->
      let y, e = reference libm x in
      let lo = low (y -. e) and hi = high (y +. e) in
      let check r =
        if rank r < lo || rank r > hi then
          failf "%s %h is %h, the C library's float64 %h" name x r y
      in
      check got.(i);
      check reversed.(n - 1 - i))
    xs

(* The two floats around the exact value, or that value where it is a float:
   only [log 1.], whose reference is exactly [0.]. *)
let law_log xs =
  covers xs;
  ranks_within ~name:"log" Nx.log Stdlib.log ~low:below ~high:above xs

(* Within 2 ranks of the exact value rounded to the nearest float32. *)
let law_asinh xs =
  covers xs;
  ranks_within ~name:"asinh" Nx.asinh Float.asinh
    ~low:(fun v -> rank (round32 v) - 2)
    ~high:(fun v -> rank (round32 v) + 2)
    xs

let signed bits =
  let open Gen in
  let+ b = bits and+ negative = bool in
  if negative then Int32.logor b Int32.min_int else b

(* [1.], [e_minus_3], the least subnormal and the greatest float. *)
let edge_examples =
  [ Array.map Int32.float_of_bits [| 0x3f80_0000l; e_minus_3; 1l; 0x7f7f_ffffl |] ]

(* roundeven *)

(* Ties to even by the float64 arithmetic: below [2^52], [(|x| + 2^52) - 2^52]
   is [|x|] rounded to an integer, ties to even, and the sign is put back;
   from [2^52] every float is an integer. *)
let even x =
  let a = Float.abs x in
  if a < 0x1p52 then Float.copy_sign (a +. 0x1p52 -. 0x1p52) x else x

(* The float32 patterns [start], [start + step], ..., [count] of them, against
   [even]. *)
let roundeven_patterns ~start ~step count =
  let x =
    Nx.bitcast Nx.float32
      (Nx.arange Nx.int32 start (start + ((count - 1) * step) + 1) step)
  in
  let got =
    Bigarray.array1_of_genarray
      (Nx.to_bigarray Bigarray.int32 (Nx.bitcast Nx.int32 (Nx.roundeven x)))
  in
  for i = 0 to count - 1 do
    let x = Int32.float_of_bits (Int32.of_int (start + (i * step))) in
    let g = Int32.float_of_bits (Bigarray.Array1.unsafe_get got i) in
    if not (same (even x) g) then failf "at %h: %h, not %h" x g (even x)
  done

(* Every float32 of magnitude in [2^-3, 2^25), the exponents where a float
   rounds to neither zero nor itself, and their neighbours'; of each other
   exponent, of either sign, its first and last 4096 patterns and every
   257th: zeros, subnormals, integers from [2^25], infinities and NaNs. The
   patterns run [2^24] at a time, a sign and two exponents. *)
let roundeven_float32 () =
  let n = 1 lsl 24 and edge = 4096 in
  for c = 0 to 255 do
    let start = (c lsl 24) - (1 lsl 31) in
    let exponents = 2 * (c land 127) in
    if exponents >= 124 && exponents < 152 then
      roundeven_patterns ~start ~step:1 n
    else begin
      roundeven_patterns ~start ~step:257 (((n - 1) / 257) + 1);
      List.iter
        (fun first -> roundeven_patterns ~start:(start + first) ~step:1 edge)
        [ 0; (n / 2) - edge; n / 2; n - edge ]
    end
  done

(* Drawn bits, and halves, quarters and integers up to [2^53] with their
   neighbours, signed. *)
let roundeven_float64 () =
  let rng = Random.State.make [| 2026 |] in
  let below k = Int64.to_float (Random.State.int64 rng k) in
  let drawn =
    List.init (1 lsl 20) (fun _ ->
        Int64.float_of_bits (Random.State.bits64 rng))
  in
  let halves = List.init 65536 (fun _ -> below 0x10_0000_0000_0000L +. 0.5) in
  let quarters =
    List.init 65536 (fun _ -> ((2. *. below 0x8_0000_0000_0000L) +. 1.) /. 4.)
  in
  let small = List.init 8001 (fun i -> Float.of_int (i - 4000) /. 2.) in
  let ends =
    [
      0.; 0.49999999999999994; 0.5; 0x1p51 +. 0.5; 0x1p52 -. 0.5;
      0x1p52 -. 1.5; 0x1p52; 0x1p52 +. 1.; 0x1p53; 0x1p-1074; Float.max_float;
      infinity; nan;
    ]
  in
  let around x = [ Float.pred x; x; Float.succ x ] in
  let xs = drawn @ List.concat_map around (halves @ quarters @ small @ ends) in
  let xs = Array.of_list (xs @ List.map Float.neg xs) in
  let got = values Nx.roundeven Nx.float64 xs in
  Array.iteri
    (fun i x ->
      if not (same (even x) got.(i)) then
        failf "at %h: %h, not %h" x got.(i) (even x))
    xs

let roundeven_others () =
  let ints = Nx.create Nx.int32 [| 3 |] [| -7l; 0l; 9l |] in
  equal (array int32) [| -7l; 0l; 9l |] (Nx.to_array (Nx.roundeven ints));
  equal (array float_exact) [| 2.; 4.; -0.; -2.; 3. |]
    (values Nx.roundeven Nx.float32 [| 2.5; 3.5; -0.5; -2.5; 2.5000002 |])

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
    group "log"
      [
        prop ~examples:edge_examples
          "float32 is one of the two floats around the exact value"
          (float32_values positive_bits) law_log;
      ];
    group "asinh"
      [
        prop "float64 within 3 ulps" (drawn D.Float64 samples) (fun x ->
            law_unary D.Float64 (fun x -> Nx.asinh x) Float.asinh x);
        prop ~examples:edge_examples "float32 within 2 ulps"
          (float32_values (signed positive_bits)) law_asinh;
      ];
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
    group "sinc"
      [
        prop "float64 within 6 ulps" (sinc_values D.Float64)
          (law_sinc_drawn D.Float64 6.);
        prop "float32 within 4 ulps" (sinc_values D.Float32)
          (law_sinc_drawn D.Float32 4.);
        test "zeros, integers, infinities and NaN" sinc_specials;
      ];
    group "roundeven"
      [
        test "float32 against float64 ties to even" roundeven_float32;
        test "float64 against its own ties to even" roundeven_float64;
        test "integers and examples" roundeven_others;
      ];
    group "narrow floats"
      [
        narrow "roundeven" (fun x -> Nx.roundeven x);
        narrow "asinh" (fun x -> Nx.asinh x);
        narrow "acosh" (fun x -> Nx.acosh x);
        narrow "atanh" (fun x -> Nx.atanh x);
        narrow "rsqrt" (fun x -> Nx.rsqrt x);
        narrow "sinc" (fun x -> Nx.sinc x);
        narrow "hypot" (fun x -> Nx.hypot x (Nx.flip x));
      ];
    group "special values"
      [
        test "square wraps integers and squares complex numbers" test_square;
        test "zeros, ends of the domain, infinities and NaN" specials;
        test "a format without infinities or NaN stores as a cast"
          no_infinities;
        cases "a function of floats refuses another dtype"
          ~id:(fun (n, _) -> n)
          [
            ("Nx.rsqrt", fun () -> ignore (Nx.rsqrt (Nx.zeros Nx.int32 [| 1 |])));
            ("Nx.hypot", fun () ->
                let z = Nx.zeros Nx.complex64 [| 1 |] in
                ignore (Nx.hypot z z));
            ("Nx.asinh", fun () -> ignore (Nx.asinh (Nx.zeros Nx.bool [| 1 |])));
            ("Nx.acosh", fun () -> ignore (Nx.acosh (Nx.zeros Nx.uint8 [| 1 |])));
            ("Nx.atanh", fun () -> ignore (Nx.atanh (Nx.zeros Nx.int64 [| 1 |])));
            ("Nx.square", fun () -> ignore (Nx.square (Nx.zeros Nx.bool [| 1 |])));
            ("Nx.roundeven", fun () ->
                ignore (Nx.roundeven (Nx.zeros Nx.complex64 [| 1 |])));
          ]
          (fun (by, f) -> invalid ~by f);
      ];
  ]

let () = exit (run "nx math" math)
