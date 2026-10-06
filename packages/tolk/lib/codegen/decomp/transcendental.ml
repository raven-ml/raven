(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC AND SunPro
  ---------------------------------------------------------------------------*)

(* The coefficients of the sine, cosine and logarithm kernels, and the split of
   [1/ln 2] in [xlog2], are fdlibm's ([k_sin.c], [k_cos.c], [k_log.h],
   [e_log2.c], [e_log2f.c]), which carry this notice:

   Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved. Developed
   at SunPro, a Sun Microsystems, Inc. business. Permission to use, copy,
   modify, and distribute this software is freely granted, provided that this
   notice is preserved. *)

open Ops
open Shape

(* The floats the functions compute in, and those with an integer of their
   width, which the bit manipulations take. *)
let transcendental_dtypes = Dtype.[ Float32; Float64 ]
let widths = Dtype.[ Float16; Float32; Float64 ]
let float_like u x = const_like u (`Float x)
let int_like u n = const_like u (`Int (Bigint.of_int n))

let not_in what dt =
  invalid_arg (Format.asprintf "%a is not %s" Dtype.pp dt what)

let not_float = not_in "a float16, float32 or float64"
let not_int = not_in "an int16, int32 or int64"
let check_width d = if not (List.mem (dtype d) widths) then not_float (dtype d)

let check d =
  if not (List.mem (dtype d) transcendental_dtypes) then
    not_in "a float32 or float64" (dtype d)

let int_of_width : Dtype.t -> Dtype.t = function
  | Float64 -> Int64
  | Float32 -> Int32
  | Float16 -> Int16
  | dt -> not_float dt

(* Only frexp reads it, after checking its argument. *)
let uint_of_width : Dtype.t -> Dtype.t = function
  | Float64 -> Uint64
  | Float32 -> Uint32
  | _ -> Uint16

(* inf to inf, -inf to neg_inf, NaN to nan, anything else to ratio. *)
let lazy_map_numbers x ~inf ~neg_inf ~nan ratio =
  where
    O.(x <> float infinity)
    (where O.(x <> x) nan (where O.(x <> float neg_infinity) ratio neg_inf))
    inf

(* Helper functions for bit manipulation *)

let mantissa_bits dt = snd (Dtype.finfo dt)

let exponent_bias dt =
  let fnuz = List.mem dt Dtype.fp8_fnuz in
  (1 lsl (fst (Dtype.finfo dt) - 1)) - if fnuz then 0 else 1

let exponent_mask dt = (1 lsl fst (Dtype.finfo dt)) - 1

(* Utils *)

let pow2 y =
  if y < 0 then invalid_arg (Printf.sprintf "negative shift %d" y);
  const (`Int (Bigint.shift_left Bigint.one y))

let shr x y = O.(x // pow2 y)
let shl x y = O.(x * pow2 y)

let rintk d =
  let half = where O.(d < float 0.) (float_like d (-0.5)) (float_like d 0.5) in
  cast O.(d + half) (int_of_width (dtype d))

let pow2if q float_dtype =
  let out : Dtype.t =
    match dtype q with
    | Int64 -> Float64
    | Int32 -> Float32
    | Int16 -> float_dtype
    | dt -> not_int dt
  in
  bitcast (shl O.(q + int (exponent_bias out)) (mantissa_bits out)) out

let ilogb2k d =
  let dt = dtype d in
  let dint = bitcast d (int_of_width dt) in
  O.(
    (shr dint (mantissa_bits dt) land int (exponent_mask dt))
    - int (exponent_bias dt))

let ldexp3k d e =
  let dt = dtype d in
  let m2 = shl (cast e (int_of_width dt)) (mantissa_bits dt) in
  bitcast O.(bitcast d (int_of_width dt) + m2) dt

let ldexp2k d e =
  O.(d * pow2if (shr e 1) (dtype d) * pow2if (e - shr e 1) (dtype d))

let frexp v =
  check_width v;
  let dt = dtype v in
  (* m1 masks the sign and the mantissa, m2 sets the exponent that normalizes
     the mantissa into [0.5, 1). *)
  let m1, m2 =
    match dt with
    | Float64 -> (0x000FFFFFFFFFFFFF, 0x3FE0000000000000)
    | Float32 -> (0x807FFFFF, 0x3F000000)
    | _ -> (0x83FF, 0x3800)
  in
  let bits = bitcast v (uint_of_width dt) in
  let exponent = O.(shr bits (mantissa_bits dt) land int (exponent_mask dt)) in
  let mantissa = bitcast O.(bits land int m1 lor int m2) dt in
  (mantissa, O.(exponent - int (exponent_bias dt) + int 1))

(* Reduction algorithms for sine *)

(* 1/(2pi) in 32-bit words, the first its integer part: 1312 bits, as a
   float64's greatest exponent needs *)
let one_over_two_pi =
  [|
    0x00000000;
    0x28be60db;
    0x9391054a;
    0x7f09d5f4;
    0x7d4d3770;
    0x36d8a566;
    0x4f10e410;
    0x7f9458ea;
    0xf7aef158;
    0x6dc91b8e;
    0x909374b8;
    0x01924bba;
    0x82746487;
    0x3f877ac7;
    0x2c4a69cf;
    0xba208d7d;
    0x4baed121;
    0x3a671c09;
    0xad17df90;
    0x4e64758e;
    0x60d4ce7d;
    0x272117e2;
    0xef7e4a0e;
    0xc7fe25ff;
    0xf7816603;
    0xfbcbc462;
    0xd6829b47;
    0xdb4d9fb3;
    0xc9f2c26d;
    0xd3d18fd9;
    0xa797fa8b;
    0x5d49eeb1;
    0xfaf97c5e;
    0xcf41ce7d;
    0xe294a4ba;
    0x9afed7ec;
    0x47e35742;
    0x1580cc11;
    0xbf1edaea;
    0xfc33ef08;
    0x26bd0d87;
    0x6a78e458;
  |]

(* [turn_fraction d] is [(zh, zl, dt)]: the fraction of a turn in [d >= 1], to
   128 bits in the 64-bit words [zh] and [zl], and the type [dt] it is computed
   for, {!Dtype.Float32} for a {!Dtype.Float16} [d]. *)
let turn_fraction d =
  check_width d;
  let dt = dtype d in
  let intermediate_dtype : Dtype.t = if dt = Float16 then Float32 else dt in
  (* d = m * 2^(e - w), the integer m of w bits in k words of 32 bits *)
  let w, k, max_exp =
    if intermediate_dtype = Float64 then (53, 2, 1024) else (24, 1, 128)
  in
  let f, e = frexp d in
  let m =
    cast O.(cast f intermediate_dtype * float (Float.ldexp 1. w)) Uint64
  in
  let ms =
    List.filteri (fun i _ -> i < k) [ O.(m land int 0xffffffff); shr m 32 ]
  in
  (* d/(2pi) * 2^128 mod 2^128 comes from the words of 1/(2pi) at d's exponent,
     with g words below it for the carries *)
  let g = 2 in
  let l = 4 + k + g in
  let shift = 128 - w in
  let e = maximum O.(cast e Int32 + int shift) (int 0) in
  let i = O.(e // int 32) and s = cast O.(e land int 31) Uint64 in
  let table =
    Array.sub one_over_two_pi 0 (((max_exp - w + 128) / 32) - 3 + l + 1)
  in
  let word offset =
    let ret = ref (int ~dtype:Uint64 0) in
    for count = Array.length table - offset - 1 downto max 0 (-offset) do
      ret :=
        where O.(i <> int count) !ret (int_like !ret table.(count + offset))
    done;
    !ret
  in
  let a = Array.init (l + 1) (fun j -> word (j - 3)) in
  let b =
    List.init l (fun j ->
        let next = a.(j + 1) in
        O.((a.(j) lsl s) lor (next lsr (int 32 - s)) land int 0xffffffff))
  in
  (* columns of 32 bits, from g + 1 words below the binary point to the top
     word *)
  let terms = Array.make (g + 5) [] in
  let push c t = terms.(c + g + 1) <- terms.(c + g + 1) @ [ t ] in
  List.iteri
    (fun k mk ->
      List.iteri
        (fun j bj ->
          let p = O.(mk * bj) and c = k - j + 3 in
          if -g - 1 <= c && c <= 3 then push c O.(p land int 0xffffffff);
          if -g - 1 <= c + 1 && c + 1 <= 3 then push (c + 1) (shr p 32))
        b)
    ms;
  let z, _ =
    Array.fold_left
      (fun (z, carry) ts ->
        let col =
          match ts @ carry with
          | t :: rest -> List.fold_left add t rest
          | [] -> invalid_arg "an empty column"
        in
        (z @ [ O.(col land int 0xffffffff) ], [ shr col 32 ]))
      ([], []) terms
  in
  let zh, zl =
    match List.rev z with
    | z3 :: z2 :: z1 :: z0 :: _ -> (O.(shl z3 32 lor z2), O.(shl z1 32 lor z0))
    | _ -> invalid_arg "fewer than four columns"
  in
  (zh, zl, intermediate_dtype)

(* [round_to dt c] is [c] rounded to the float [dt]. *)
let round_to (dt : Dtype.t) c =
  if dt = Float64 then c else Int32.float_of_bits (Int32.bits_of_float c)

(* Dekker's factor, [2^ceil(p/2) + 1] for [p] bits of significand, splits a
   float into halves whose products are exact. *)
let splitter (dt : Dtype.t) = if dt = Float64 then 134217729. else 4097.

(* [2pi/2^64] to twice the precision of [dt]: its leading part [k] in halves
   [(k_hi, k_lo)], and the rest. *)
let turn_unit dt =
  let pi_lo = 1.2246467991473532e-16 in
  let k = round_to dt (Float.ldexp (2. *. Float.pi) (-64)) in
  let p = round_to dt (k *. splitter dt) in
  let k_hi = round_to dt (p -. round_to dt (p -. k)) in
  let rest = (2. *. Float.pi) -. Float.ldexp k 64 +. (2. *. pi_lo) in
  (k, k_hi, k -. k_hi, Float.ldexp rest (-64))

(* [fast_two_sum a b] is [(s, e)] with [s = a + b] rounded and [s + e] exact,
   for [a] zero or of an exponent at least [b]'s. *)
let fast_two_sum a b =
  let s = O.(a + b) in
  (s, O.(b - (s - a)))

(* [radians hi zl dt] is the angle of [hi + zl 2^-64] units of [2^-64] turn, for
   a signed [hi] below [2^62] in magnitude, as [(r, r_lo)]: [r] rounded once to
   [dt], and the rest. The fraction is summed in two floats, from parts that
   convert exactly and whose sums are each exact as a rounded sum and its error,
   then multiplied by [2pi/2^64] in two floats. *)
let radians hi zl (dt : Dtype.t) =
  let low n = O.(hi land int Stdlib.((1 lsl n) - 1)) in
  let parts =
    if dt = Float64 then [ (shr hi 32, 32); (low 32, 0); (shr zl 11, -53) ]
    else
      [
        (shr hi 42, 42);
        (O.(shr hi 21 land int 0x1fffff), 21);
        (low 21, 0);
        (shr zl 43, -43);
      ]
  in
  let term (c, e) = O.(cast c dt * float (Float.ldexp 1. e)) in
  let sum (s, e) part =
    let s, e' = fast_two_sum s (term part) in
    (s, O.(e + e'))
  in
  let s, e =
    match parts with
    | p :: ps ->
        let t = term p in
        List.fold_left sum (t, float_like t 0.) ps
    | [] -> invalid_arg "no parts"
  in
  let k, k_hi, k_lo, k_rest = turn_unit dt in
  let c = O.(s * float (splitter dt)) in
  let s_hi = O.(c - (c - s)) in
  let s_lo = O.(s - s_hi) in
  let p = O.(s * float k) in
  let p_err =
    O.(
      (s_hi * float k_hi)
      - p
      + (s_hi * float k_lo)
      + (s_lo * float k_hi)
      + (s_lo * float k_lo))
  in
  fast_two_sum p O.(p_err + (s * float k_rest) + (e * float k))

(* [payne_hanek d] is [(r, r_lo, q)]: {!payne_hanek_reduction}'s remainder in
   two parts, in float32 for a float16 [d]. *)
let payne_hanek d =
  let zh, zl, dt = turn_fraction d in
  (* The quotient rounds to the nearest quadrant: from half a quadrant up, the
     remainder is the fraction less a quadrant. *)
  let half = O.(shr zh 61 land int 1) in
  let q = cast O.(shr zh 62 + half) Int32 in
  let hi =
    O.(cast (zh land int 0x3fffffffffffffff) Int64 - shl (cast half Int64) 62)
  in
  let r, r_lo = radians hi zl dt in
  (r, r_lo, q)

let payne_hanek_reduction d =
  let r, _, q = payne_hanek d in
  (cast r (dtype d), q)

(* [cody_waite d] is [(r, r_lo, q)]: {!cody_waite_reduction}'s remainder in two
   parts, for a float32 or float64 [d]. [q pi/2] is subtracted in four parts:
   those of float32 have 8, 9, 9 and 24 bits, those of float64 25, 24, 25 and
   52, so that the products by a quotient below 2^15 are exact but the last, and
   so are the first two differences. The third rounds once the remainder is
   large enough; its error is carried to the last. *)
let cody_waite d =
  let a, b, c, last =
    if dtype d = Float64 then
      ( 1.5707963109016418457,
        1.5893254712295856735e-08,
        6.123233932053594251e-17,
        6.368317163510949908e-25 )
    else
      ( 1.5703125,
        0.00048351287841796875,
        3.13855707645416259765e-07,
        6.077100628276710381e-11 )
  in
  let quadrant = rintk O.(d * float 0.636619772367581343075535053490057448) in
  let q = cast quadrant (dtype d) in
  let x = O.((q * float (-.a)) + d) in
  let x = O.((q * float (-.b)) + x) in
  let x, e = fast_two_sum x O.(q * float (-.c)) in
  let r, r_lo = fast_two_sum x O.(e + (q * float (-.last))) in
  (r, r_lo, cast quadrant Int32)

let cody_waite_reduction d =
  if dtype d = Float16 then
    let r, _, q = cody_waite (cast d Float32) in
    (cast r Float16, q)
  else
    let r, _, q = cody_waite d in
    (r, q)

(* Sine and cosine of a remainder

   On [[-pi/4, pi/4]], the sine is [r + r^3 S(r^2)] and the cosine [1 - r^2/2 +
   r^4 C(r^2)], with fdlibm's minimax coefficients ([__kernel_sin],
   [__kernel_cos]), in float32 the leading ones rounded. The leading term of
   each is exact and the others small, so that only the last addition rounds at
   the result's scale; the cosine adds back the rounding of [1 - r^2/2] ([w]
   below), which is not small. Both take the remainder as [r + r_lo], its
   rounding [r_lo] first order: the remainder rounded alone costs up to an ulp
   of the result. *)

let poly_n x = function
  | c :: cs ->
      List.fold_left (fun acc c -> O.((acc * x) + float c)) (float c) cs
  | [] -> invalid_arg "a polynomial without coefficients"

let sin_kernel r r_lo z =
  let s =
    if dtype r = Float64 then
      [
        1.58969099521155010221e-10;
        -2.50507602534068634195e-08;
        2.75573137070700676789e-06;
        -1.98412698298579493134e-04;
        8.33333333332248946124e-03;
      ]
    else
      [
        2.75573137070700676789e-06;
        -1.98412698298579493134e-04;
        8.33333333332248946124e-03;
      ]
  in
  let v = O.(z * r) in
  O.(
    r
    - ((z * ((float 0.5 * r_lo) - (v * poly_n z s)))
      - r_lo
      - (v * float (-1.66666666666666324348e-01))))

let cos_kernel r r_lo z =
  let c =
    if dtype z = Float64 then
      [
        -1.13596475577881948265e-11;
        2.08757232129817482790e-09;
        -2.75573143513906633035e-07;
        2.48015872894767294178e-05;
        -1.38888888888741095749e-03;
        4.16666666666666019037e-02;
      ]
    else
      [
        -2.75573143513906633035e-07;
        2.48015872894767294178e-05;
        -1.38888888888741095749e-03;
        4.16666666666666019037e-02;
      ]
  in
  let h = O.(float 0.5 * z) in
  let w = O.(float 1. - h) in
  O.(w + (float 1. - w - h + ((z * z * poly_n z c) - (r * r_lo))))

(* Toplevel functions for xsin, xlog2 and xexp2 *)

(* The sine of [d = q pi/2 + r] is that of [r] or its cosine, by the parity of
   [q], negated in the lower half-turn. *)
let xsin ?(fast = false) ?(switch_over = 30.0) d =
  check d;
  let fast =
    fast
    || Dtype.Value.(
         `Float (-.switch_over) < vmin d && vmax d < `Float switch_over)
  in
  let zero = float_like d 0. in
  let x = lazy_map_numbers d ~inf:zero ~neg_inf:zero ~nan:zero d in
  let x_sign = where O.(x < int 0) (int_like x (-1)) (int_like x 1) in
  let x_abs = O.(x * x_sign) in
  let r, r_lo, q =
    let small = cody_waite x_abs in
    if fast then small
    else
      (* Payne-Hanek takes angles from 1 on; the smaller ones take Cody-Waite,
         precise below the switch-over. *)
      let r, r_lo, q = small and r', r_lo', q' = payne_hanek x_abs in
      let small = O.(x_abs < float switch_over) in
      (where small r r', where small r_lo r_lo', where small q q')
  in
  let z = O.(r * r) in
  let s =
    where O.(q land int 1 <> int 0) (cos_kernel r r_lo z) (sin_kernel r r_lo z)
  in
  let s = O.(s * x_sign) in
  let s = where O.(q land int 2 <> int 0) O.(-s) s in
  (* A zero is its own sine, of its sign. *)
  let s = where O.(x <> int 0) s x in
  let nan = float_like d Dtype.nan in
  lazy_map_numbers d ~inf:nan ~neg_inf:nan ~nan s

let xexp2 d =
  check d;
  let zero = float_like d 0. in
  let x = lazy_map_numbers d ~inf:zero ~neg_inf:zero ~nan:zero d in
  let q = rintk x in
  let s = O.(x - q) in
  (* A polynomial with 13 non-zero terms on [-(log 2)/2, (log 2)/2]. *)
  let u =
    if dtype d = Float64 then
      poly_n s
        [
          0.4434359082926529454e-9;
          0.7073164598085707425e-8;
          0.1017819260921760451e-6;
          0.1321543872511327615e-5;
          0.1525273353517584730e-4;
          0.1540353045101147808e-3;
          0.1333355814670499073e-2;
          0.9618129107597600536e-2;
          0.5550410866482046596e-1;
          0.2402265069591012214e+0;
          0.6931471805599452862e+0;
          0.1000000000000000000e+1;
        ]
    else
      poly_n s
        [
          0.1535920892e-3;
          0.1339262701e-2;
          0.9618384764e-2;
          0.5550347269e-1;
          0.2402264476e+0;
          0.6931471825e+0;
          1.0;
        ]
  in
  let u = ldexp2k u q in
  let upper, lower = if dtype d = Float64 then (1024, -2000) else (128, -150) in
  let u = where O.(d >= int upper) (float_like d infinity) u in
  let u = where O.(d < int lower) (float_like d 0.) u in
  where O.(d <> d) (float_like d Dtype.nan) u

(* The logarithm of [d = m 2^e], [m] in [[1/sqrt 2, sqrt 2)], is [e + log2 (1 +
   f)] for [f = m - 1], which is exact. [log (1 + f)] is [f - h + s (h + R)]
   with [h = f^2/2], [s = f / (2 + f)] and [R] a minimax polynomial in [s^2]
   that approximates [2 atanh s / s - 2] (fdlibm's [__kernel_log]): every term
   but [f] is small, so only the sum rounds at the result's scale. Base 2 then
   keeps [f - h] in two parts, the first of few enough bits that its product by
   the leading part of [1/ln 2] is exact, and adds [e] last, carrying the
   rounding of that sum. *)
let xlog2 d =
  check d;
  let dt = dtype d in
  let is64 = dt = Float64 in
  let least_normal = Float.ldexp 1. (1 - exponent_bias dt) in
  let is_denormal = O.(d < float least_normal) in
  let a = where is_denormal O.(d * float (Float.ldexp 1. 64)) d in
  let e = cast (ilogb2k O.(a * float (Float.sqrt 2.))) dt in
  let m = ldexp3k a O.(-e) in
  let e = where is_denormal O.(e - int 64) e in
  let f = O.(m - float 1.) in
  let s = O.(f / (float 2. + f)) in
  let z = O.(s * s) in
  let lg =
    if is64 then
      [
        1.479819860511658591e-01;
        1.531383769920937332e-01;
        1.818357216161805012e-01;
        2.222219843214978396e-01;
        2.857142874366239149e-01;
        3.999999999940941908e-01;
        6.666666666666735130e-01;
      ]
    else [ 0xf89e26.0p-26; 0x91e9ee.0p-25; 0xccce13.0p-25; 0xaaaaaa.0p-24 ]
  in
  let h = O.(float 0.5 * f * f) in
  let tail = O.(s * (h + (z * poly_n z lg))) in
  let bits = bitcast O.(f - h) (if is64 then Uint64 else Uint32) in
  let keep =
    if is64 then Bigint.of_string "0xffffffff00000000"
    else Bigint.of_int 0xfffff000
  in
  let hi = bitcast O.(bits land const_like bits (`Int keep)) dt in
  let lo = O.(f - hi - h + tail) in
  let ivln2_hi, ivln2_lo =
    if is64 then (1.44269504072144627571e+00, 1.67517131648865118353e-10)
    else (1.4428710938e+00, -1.7605285393e-04)
  in
  let val_hi = O.(hi * float ivln2_hi) in
  let val_lo = O.(((lo + hi) * float ivln2_lo) + (lo * float ivln2_hi)) in
  let w = O.(e + val_hi) in
  let r = O.(val_lo + (e - w + val_hi) + w) in
  let r = where O.(d <> float infinity) r (float_like r infinity) in
  let r = where O.(d <> float 0.0) r (float_like r neg_infinity) in
  (* Some targets do not find -0.0 equal to 0.0; its reciprocal is -inf. So is
     that of a negative number too small for its reciprocal to be finite, which
     the next select makes NaN. *)
  let r =
    where O.(reciprocal d <> float neg_infinity) r (float_like r neg_infinity)
  in
  let r = where O.(d < float (-0.0)) (float_like r Dtype.nan) r in
  where O.(d <> d) (float_like r Dtype.nan) r

let xpow base exponent =
  let ret =
    exp2 (mul (log2 (where O.(base < int 0) O.(-base) base)) exponent)
  in
  (* A negative base gives NaN for an exponent that is not an integer, and the
     sign of the exponent's parity otherwise; -inf is never NaN, its power stays
     |base| ** exponent. *)
  let non_int = O.(exponent <> cast (cast exponent Int32) (dtype exponent)) in
  let magnitude = where O.(exponent < int 0) O.(-exponent) exponent in
  let is_odd = cast (mod_ (cast magnitude Int32) (int 2)) Bool in
  let neg_base =
    where non_int
      (where O.(base <> float neg_infinity) (float_like ret Dtype.nan) ret)
      (where is_odd O.(-ret) ret)
  in
  (* x ** 0 is 1, 0 ** 0 and inf ** 0 included. *)
  where
    (eq exponent (int 0))
    (int_like ret 1)
    (where O.(base < int 0) neg_base ret)

let patterns ~force ops =
  let other_floats =
    List.filter (fun dt -> not (List.mem dt transcendental_dtypes)) Dtype.floats
  in
  let lowered op = force || not (Op.Set.mem op ops) in
  let d = Upat.var "d" in
  let decompose (op, f) =
    if not (lowered op) then []
    else
      Pattern_matcher.
        [
          rule (Upat.op ~dtype:transcendental_dtypes ~src:[ d ] op) (fun m ->
              Some (f (m "d")));
          rule (Upat.op ~dtype:other_floats ~src:[ d ] ~name:"x" op) (fun m ->
              let x = m "x" in
              Some (cast (alu (cast (m "d") Float32) (Ops.op x) []) (dtype x)));
        ]
  in
  let sqrt =
    if not (lowered Sqrt) then []
    else
      [
        Pattern_matcher.rule (Upat.op ~each:d Sqrt) (fun m ->
            Some (xpow (m "d") (float_like (m "d") 0.5)));
      ]
  in
  let transcendentals =
    [ (Op.Exp2, xexp2); (Log2, xlog2); (Sin, fun d -> xsin d) ]
  in
  Pattern_matcher.v (fun () -> List.concat_map decompose transcendentals @ sqrt)
