(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC AND SunPro
  ---------------------------------------------------------------------------*)

open Tolk_next

let dtype = Ops.dtype
let is_float u = Dtype.is_float (dtype u)
let is_signed u = List.exists (Dtype.equal (dtype u)) Dtype.sints
let where = Ops.where
let float u x = Ops.const_like u (`Float x)
let int u n = Ops.const_like u (`Int (Z.of_int n))
let isnan x = Ops.ne x x

(* Comparisons false on NaN: [x >= y] and [x <= y]. *)
let at_least x y = Ops.bitwise_or (Ops.lt y x) (Ops.eq x y)
let at_most x y = Ops.bitwise_or (Ops.lt x y) (Ops.eq x y)

(* Narrow floats

   nx computes on [float16], [bfloat16] and the 8-bit floats at [float32] and
   rounds each result once. *)

let narrow u = is_float u && Dtype.itemsize (dtype u) < 4
let widen x = if narrow x then Ops.cast x Float32 else x
let float1 f x = if narrow x then Ops.cast (f (widen x)) (dtype x) else f x

let float2 f x y =
  if narrow x then Ops.cast (f (widen x) (widen y)) (dtype x) else f x y

(* The unsigned integer of [x]'s width, [x] a float of at least 32 bits. *)
let bits x =
  Ops.bitcast x (if Dtype.itemsize (dtype x) = 8 then Dtype.Uint64 else Uint32)

(* The bits of the float [x] with the sign bit only, or all bits but it. *)
let sign_mask x = Z.shift_left Z.one ((8 * Dtype.itemsize (dtype x)) - 1)

let sign_bit x =
  Ops.ne
    (Ops.bitwise_and (bits x) (Ops.const_like (bits x) (`Int (sign_mask x))))
    (int (bits x) 0)

(* Modular integers

   nx's integers wrap. C, Metal and CUDA leave signed overflow undefined, and
   promote integers narrower than [int] to [int], whose products overflow: a
   wrapping operation computes on the unsigned bit pattern, at 32 bits for the
   narrow widths, and narrows back. *)

let unsigned dt = if Dtype.itemsize dt = 8 then Dtype.Uint64 else Dtype.Uint32

let lift x =
  let u = unsigned (dtype x) in
  if Dtype.itemsize (dtype x) < 4 then Ops.cast x u else Ops.bitcast x u

let back dt r =
  if Dtype.itemsize dt < 4 then Ops.cast r dt else Ops.bitcast r dt

let wrapping1 f x = back (dtype x) (f (lift x))
let wrapping2 f x y = back (dtype x) (f (lift x) (lift y))

(* Unary *)

let recip x =
  if is_float x then float1 Ops.reciprocal x
  else
    (* The integer quotient of 1: 1 at 1, -1 at -1, and 0 elsewhere, 0
       included. *)
    let minus_one =
      if is_signed x then Ops.eq x (int x (-1)) else Ops.bool false
    in
    where
      (Ops.eq x (int x 1))
      (int x 1)
      (where minus_one (int x (-1)) (int x 0))

let abs x =
  if is_float x then
    float1
      (fun x ->
        let magnitude = Ops.const_like (bits x) (`Int (Z.pred (sign_mask x))) in
        Ops.bitcast (Ops.bitwise_and (bits x) magnitude) (dtype x))
      x
  else if is_signed x then where (Ops.lt x (int x 0)) (wrapping1 Ops.neg x) x
  else x

let sign x =
  if is_float x then
    float1
      (fun x ->
        where (isnan x) x
          (where
             (Ops.lt (float x 0.) x)
             (float x 1.)
             (where (Ops.lt x (float x 0.)) (float x (-1.)) (float x 0.))))
      x
  else
    let pos = where (Ops.ne x (int x 0)) (int x 1) (int x 0) in
    if is_signed x then where (Ops.lt x (int x 0)) (int x (-1)) pos else pos

(* Half away from zero: the integer part, and one further out when the rest
   reaches a half. The rest [x - trunc x] is exact, so nothing rounds before the
   decision. *)
let round x =
  let t = Ops.trunc x in
  let rest = Ops.sub x t in
  let out =
    where
      (Ops.lt x (float x 0.))
      (Ops.sub t (float x 1.))
      (Ops.add t (float x 1.))
  in
  where
    (Ops.bitwise_or
       (at_least rest (float x 0.5))
       (at_most rest (float x (-0.5))))
    out t

let rounding f x = if is_float x then float1 f x else x

(* Accurate compositions

   The transcendental functions compute on [float32] or [float64] from the
   primitives [exp2], [log2], [sin] and [sqrt], which each target has natively
   or as tolk's polynomials. Constants are written in double precision and
   rounded to the computing dtype. *)

let is64 x = Dtype.itemsize (dtype x) = 8

(* The sine as the target computes it: accurate on a reduced argument. *)
let sine x = Ops.alu x Op.Sin []
let neg = Ops.neg
let ( +: ) = Ops.add
let ( -: ) = Ops.sub
let ( *: ) = Ops.mul
let ( /: ) x y = Ops.alu x Op.Fdiv [ y ]

(* [horner x cs] is the polynomial of coefficients [cs] at [x], the constant
   term first. *)
let horner x cs =
  match List.rev cs with
  | c :: rest ->
      List.fold_left (fun acc c -> float x c +: (x *: acc)) (float x c) rest
  | [] -> invalid_arg "a polynomial without coefficients"

(* [x] as [hi + lo], each with half of [x]'s significand, so that products of
   halves are exact. *)
let split x =
  let c = x *: float x (if is64 x then 134217729. else 4097.) in
  let hi = c -: (c -: x) in
  (hi, x -: hi)

(* [two_prod a b] is [(p, e)] with [p] the rounded product and [p + e] the exact
   one. *)
let two_prod a b =
  let p = a *: b in
  let ah, al = split a and bh, bl = split b in
  (p, (ah *: bh) -: p +: (ah *: bl) +: (al *: bh) +: (al *: bl))

(* Constants to twice the precision of a double: the double nearest, and the
   rest. *)
let ln2 = (0.6931471805599453, 2.3190468138462996e-17)
let log2e = (1.4426950408889634, 2.0355273740931033e-17)
let pi_2 = (1.5707963267948966, 6.123233995736766e-17)
let pi = (3.141592653589793, 1.2246467991473532e-16)

(* [parts x c] is [(hi, lo)], the constant [c] to twice [x]'s precision: [hi] is
   [c] rounded to [x]'s dtype, and [lo] the rest, rounded. *)
let parts x (c, rest) =
  let hi = if is64 x then c else Int32.float_of_bits (Int32.bits_of_float c) in
  (float x hi, float x (c -. hi +. rest))

(* [minus c x] is [c - x] for a constant [c] in two parts ({!parts}): the
   difference with the leading part is exact where [x] is near it, and the rest
   is added after. *)
let minus c x =
  let hi, lo = parts x c in
  hi -: x +: lo

(* Exponentials

   [exp x] is [2^(x log2 e)], whose argument is a product rounded to [x]'s
   precision: rounding it costs up to [|x|] ulps of the result. The product is
   kept in two parts [t + e], and [2^(t + e)] is [2^t (1 + e ln 2)]. [2^t] is
   taken of [t] moved towards zero by [k], and the [2^k] multiplied in last, so
   that a result near the extremes of the dtype neither overflows early nor
   rounds twice in the subnormals. *)

let exp_parts ?(times = 1.) x t e =
  let k, emax, emin =
    if is64 x then (512., 1024., -1076.) else (64., 128., -151.)
  in
  let up = Ops.lt (float x k) t and down = Ops.lt t (float x (-.k)) in
  let shift = where up (float x k) (where down (float x (-.k)) (float x 0.)) in
  let scale =
    where up
      (float x (Float.ldexp times (int_of_float k)))
      (where down
         (float x (Float.ldexp times (-int_of_float k)))
         (float x times))
  in
  let p = Ops.exp2 (t -: shift) in
  (* Past these, [2^(t - k)] leaves the dtype, and so does the result. *)
  where
    (Ops.lt (float x (k +. emax)) t)
    (float x Float.infinity)
    (where
       (Ops.lt t (float x (-.k +. emin)))
       (float x 0.)
       ((p +: (p *: (e *: fst (parts x ln2)))) *: scale))

(* [exp ~times x] is [times * e^x], [times] a power of two. *)
let exp ?times x =
  let hi, lo = parts x log2e in
  let t, e = two_prod x hi in
  exp_parts ?times x t (e +: (x *: lo))

(* Logarithm *)

let log x = Ops.log2 x *: Ops.float (fst ln2)

(* Trigonometry

   [sin x] and [cos x] reduce [|x|] by [pi/2] themselves, to [r] in [[-pi/4,
   pi/4]] and a quadrant, and take the sine or the cosine of [r]: [cos x] as
   [sin (pi/2 - x)] would round [pi/2 - x] first, and the sine of a large
   argument is only as good as its reduction. The cosine of [r] is [1 - 2 sin^2
   (r/2)], which does not cancel. *)

let copysign m x = where (sign_bit x) (neg m) m

let cos_reduced r =
  let s = sine (r *: float r 0.5) in
  float r 1. -: (float r 2. *: s *: s)

(* [pi/2] in parts whose products by a quotient below the limit are exact: of 12
   bits below [2^12] in [float32], of 30 bits below [2^22] in [float64]. *)
let pi_2_cw x =
  if is64 x then
    ( 0x1p22,
      [
        0x1.921fb54p+0; 0x1.10b46118p-30; 0x1.313198ap-61; 0x1.701b839a25204p-92;
      ] )
  else (0x1p12, [ 0x1.92p+0; 0x1.fb4p-12; 0x1.444p-24; 0x1.68c234p-39 ])

(* [quarter_turns a] is [(r, q)] with [a = q pi/2 + r] and [|r|] about [pi/4],
   for finite [a >= 0], [q] an [int32] node whose value modulo 4 is the
   quadrant: [q pi/2] subtracted in exact parts below the limit, and beyond it
   the bits of [2/pi] (Payne-Hanek). *)
let quarter_turns a =
  let limit, parts = pi_2_cw a in
  let near = Ops.lt a (float a limit) in
  let q = Ops.trunc ((a *: float a (2. /. Float.pi)) +: float a 0.5) in
  let r = List.fold_left (fun r p -> r -: (q *: float a p)) a parts in
  let far_r, far_q =
    Transcendental.payne_hanek_reduction (where near (float a limit) a)
  in
  (where near r far_r, where near (Ops.cast q Int32) far_q)

(* [by_quadrant x f] is [f] of the sine and cosine of the reduced [|x|] and of
   its quadrant, NaN at an infinity and at NaN. *)
let by_quadrant x f =
  let a = abs x in
  let finite = Ops.lt a (float x Float.infinity) in
  let r, q = quarter_turns (where finite a (float x 0.)) in
  let quadrant n = Ops.eq (Ops.bitwise_and q (int q 3)) (int q n) in
  where finite (f (sine r) (cos_reduced r) quadrant) (float x Float.nan)

let sin x =
  let s =
    by_quadrant x (fun s c quadrant ->
        where (quadrant 0) s
          (where (quadrant 1) c (where (quadrant 2) (neg s) (neg c))))
  in
  copysign s x

let cos x =
  by_quadrant x (fun s c quadrant ->
      where (quadrant 0) c
        (where (quadrant 1) (neg s) (where (quadrant 2) (neg c) s)))

let tan x = sin x /: cos x

(* Inverse trigonometry

   [asin_small t] is the arcsine of [|t| <= 0.71]: the approximation 4.4.46 of
   Abramowitz and Stegun, then Newton's steps on [sin y = t], each of which
   squares the error, with [sqrt (1 - t^2)] as the derivative. Below [2^-12]
   ([2^-27] in [float64]) the arcsine rounds to [t]. The other functions reduce
   to it where it is well conditioned. *)

let asin_poly =
  [
    Float.pi /. 2.;
    -0.2145988016;
    0.0889789874;
    -0.0501743046;
    0.0308918810;
    -0.0170881256;
    0.0066700901;
    -0.0012624911;
  ]

let asin_small t =
  let a = abs t in
  let y0 =
    copysign
      (float t (Float.pi /. 2.)
      -: (Ops.sqrt (float t 1. -: a) *: horner a asin_poly))
      t
  in
  let slope = Ops.sqrt (float t 1. -: (t *: t)) in
  let newton y = y -: ((sine y -: t) /: slope) in
  let y = newton y0 in
  let y = if is64 t then newton y else y in
  where (Ops.lt a (float t (if is64 t then 0x1p-27 else 0x1p-12))) t y

(* [half d] is [asin (sqrt (d / 2))] for [d = 1 - a]: [asin a] is [pi/2] less
   twice it, and [acos a] twice it. [d] is exact for [a >= 1/2]. *)
let half d = asin_small (Ops.sqrt (d *: float d 0.5))

let asin x =
  let a = abs x in
  copysign
    (where
       (at_most a (float x 0.5))
       (asin_small a)
       (minus pi_2 (float x 2. *: half (float a 1. -: a))))
    x

let acos x =
  let small = minus pi_2 (asin_small x) in
  let pos = float x 2. *: half (float x 1. -: x) in
  let negative = minus pi (float x 2. *: half (float x 1. +: x)) in
  where
    (at_most (abs x) (float x 0.5))
    small
    (where (Ops.lt (float x 0.) x) pos negative)

(* The arctangent of [z >= 0] as the arcsine of [z / sqrt (1 + z^2)], of its
   inverse past 1. *)
let atan_positive z =
  let inv = Ops.lt (float z 1.) z in
  let t = where inv (Ops.reciprocal z) z in
  let c = asin_small (t /: Ops.sqrt (float z 1. +: (t *: t))) in
  where inv (minus pi_2 c) c

let atan x = copysign (atan_positive (abs x)) x

let atan2 y x =
  let ay = abs y and ax = abs x in
  let inf = float x Float.infinity in
  let angle =
    where
      (Ops.bitwise_and (Ops.eq ax inf) (Ops.eq ay inf))
      (float x (Float.pi /. 4.))
      (where
         (Ops.bitwise_and (Ops.eq ax (float x 0.)) (Ops.eq ay (float x 0.)))
         (float x 0.)
         (atan_positive (ay /: ax)))
  in
  copysign (where (sign_bit x) (minus pi angle) angle) y

(* Hyperbolic functions

   [e^|x| / 2 +- e^-|x| / 2], with the halving in the exponential's last
   scaling, so that [sinh] and [cosh] overflow where their value does. Below 1,
   [sinh x] is its Taylor series, since the difference cancels. *)

(* [1/3!], [1/5!], ... to [n] terms. *)
let factorials n =
  List.init n (fun k ->
      1.
      /. List.fold_left ( *. ) 1.
           (List.init ((2 * k) + 3) (fun i -> Float.of_int (i + 1))))

let sinh x =
  let a = abs x in
  let z = x *: x in
  let series =
    x +: (x *: z *: horner z (factorials (if is64 x then 10 else 6)))
  in
  let large = copysign (exp ~times:0.5 a -: (exp (neg a) *: float x 0.5)) x in
  where (Ops.lt a (float x 1.)) series large

let cosh x =
  let a = abs x in
  exp ~times:0.5 a +: (exp (neg a) *: float x 0.5)

let tanh x =
  where
    (Ops.lt (float x 22.) (abs x))
    (copysign (float x 1.) x)
    (sinh x /: cosh x)

(* Error function

   The rational approximations of the error function by Sun Microsystems'
   fdlibm, on the intervals [0, 0.84375), [0.84375, 1.25), [1.25, 1/0.35) and
   [1/0.35, 6), from 6 on 1. The coefficients are fdlibm's double precision
   ones, rounded to the computing dtype; rounded to [float32] they are those of
   fdlibm's [erff]. They carry this notice:

   Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved. Developed
   at SunPro, a Sun Microsystems, Inc. business. Permission to use, copy,
   modify, and distribute this software is freely granted, provided that this
   notice is preserved. *)

let erx = 8.45062911510467529297e-01
let efx8 = 1.02703333676410069053e+00

let pp =
  [
    1.28379167095512558561e-01;
    -3.25042107247001499370e-01;
    -2.84817495755985104766e-02;
    -5.77027029648944159157e-03;
    -2.37630166566501626084e-05;
  ]

let qq =
  [
    1.;
    3.97917223959155352819e-01;
    6.50222499887672944485e-02;
    5.08130628187576562776e-03;
    1.32494738004321644526e-04;
    -3.96022827877536812320e-06;
  ]

let pa =
  [
    -2.36211856075265944077e-03;
    4.14856118683748331666e-01;
    -3.72207876035701323847e-01;
    3.18346619901161753674e-01;
    -1.10894694282396677476e-01;
    3.54783043256182359371e-02;
    -2.16637559486879084300e-03;
  ]

let qa =
  [
    1.;
    1.06420880400844228286e-01;
    5.40397917702171048937e-01;
    7.18286544141962662868e-02;
    1.26171219808761642112e-01;
    1.36370839120290507362e-02;
    1.19844998467991074170e-02;
  ]

let ra =
  [
    -9.86494403484714822705e-03;
    -6.93858572707181764372e-01;
    -1.05586262253232909814e+01;
    -6.23753324503260060396e+01;
    -1.62396669462573470355e+02;
    -1.84605092906711035994e+02;
    -8.12874355063065934246e+01;
    -9.81432934416914548592e+00;
  ]

let sa =
  [
    1.;
    1.96512716674392571292e+01;
    1.37657754143519042600e+02;
    4.34565877475229228821e+02;
    6.45387271733267880336e+02;
    4.29008140027567833386e+02;
    1.08635005541779435134e+02;
    6.57024977031928170135e+00;
    -6.04244152148580987438e-02;
  ]

let rb =
  [
    -9.86494292470009928597e-03;
    -7.99283237680523006574e-01;
    -1.77579549177547519889e+01;
    -1.60636384855821916062e+02;
    -6.37566443368389627722e+02;
    -1.02509513161107724954e+03;
    -4.83519191608651397019e+02;
  ]

let sb =
  [
    1.;
    3.03380607434824582924e+01;
    3.25792512996573918826e+02;
    1.53672958608443695994e+03;
    3.19985821950859553908e+03;
    2.55305040643316442583e+03;
    4.74528541206955367215e+02;
    -2.24409524465858183362e+01;
  ]

(* [a] with the low half of its significand cleared, so that [a * a] is
   exact. *)
let high_half a =
  let keep =
    if is64 a then Z.of_string "0xffffffff00000000" else Z.of_int 0xffffe000
  in
  Ops.bitcast
    (Ops.bitwise_and (bits a) (Ops.const_like (bits a) (`Int keep)))
    (dtype a)

let erf x =
  let a = abs x in
  let one = float x 1. in
  let z = a *: a in
  let small = a +: (a *: (horner z pp /: horner z qq)) in
  let tiny = float x 0.125 *: ((float x 8. *: a) +: (float x efx8 *: a)) in
  let s = a -: one in
  let near = float x erx +: (horner s pa /: horner s qa) in
  let w = Ops.reciprocal (a *: a) in
  let inner = Ops.lt a (float x (1. /. 0.35)) in
  let r = where inner (horner w ra) (horner w rb) in
  let q = where inner (horner w sa) (horner w sb) in
  let h = high_half a in
  let tail =
    exp (neg (h *: h) -: float x 0.5625)
    *: exp (((h -: a) *: (h +: a)) +: (r /: q))
    /: a
  in
  let m =
    where
      (Ops.lt a (float x 0.84375))
      (where (Ops.lt a (float x 0x1p-28)) tiny small)
      (where
         (Ops.lt a (float x 1.25))
         near
         (where (Ops.lt a (float x 6.)) (one -: tail) one))
  in
  where (isnan x) x (copysign m x)

(* Powers

   [x^y] is [2^(y log2 |x|)]. The product [y log2 |x|] carries [y] times the
   error of the logarithm, up to a few hundred ulps of the result, so the
   logarithm is taken to twice the precision: [|x| = m 2^e] with [m] in [[1/sqrt
   2, sqrt 2)], and [ln m = 2 atanh s] for [s = (m - 1) / (m + 1)], whose
   leading term [2 s] is kept in two parts and whose tail is below a hundredth
   of it. The product is then kept in two parts, as [exp] keeps its own. The
   special values are C's [pow]'s. *)

(* [two_sum a b] is [(s, e)] with [s] the rounded sum and [s + e] the exact
   one. *)
let two_sum a b =
  let s = a +: b in
  let bb = s -: a in
  (s, a -: (s -: bb) +: (b -: bb))

(* [1/3], [1/5], ... to [n] terms. *)
let odd_inverses n = List.init n (fun k -> 1. /. Float.of_int ((2 * k) + 3))

let log2_parts a =
  let u = unsigned (dtype a) in
  let mbits, bias = if is64 a then (52, 1023) else (23, 127) in
  (* Subnormals are scaled into the normals first. *)
  let tiny = Ops.lt a (float a (Float.ldexp 1. (1 - bias))) in
  let a = where tiny (a *: float a (Float.ldexp 1. (mbits + 1))) a in
  let b = Ops.bitcast a u in
  let e =
    Ops.cast (Ops.shr b (int b mbits)) (dtype a) -: float a (Float.of_int bias)
  in
  let e = where tiny (e -: float a (Float.of_int (mbits + 1))) e in
  let fraction =
    Ops.bitwise_and b
      (Ops.const_like b (`Int (Z.pred (Z.shift_left Z.one mbits))))
  in
  let m =
    Ops.bitcast
      (Ops.bitwise_or fraction
         (Ops.const_like b (`Int (Z.shift_left (Z.of_int bias) mbits))))
      (dtype a)
  in
  let high = Ops.lt (float a (Float.sqrt 2.)) m in
  let m = where high (m *: float a 0.5) m
  and e = where high (e +: float a 1.) e in
  (* [s = (m - 1) / (m + 1)] in two parts; [m - 1] is exact. *)
  let num = m -: float a 1. in
  let den, den_lo = two_sum m (float a 1.) in
  let s = num /: den in
  let p, p_lo = two_prod s den in
  let s_lo = (num -: p -: p_lo -: (s *: den_lo)) /: den in
  let z = s *: s in
  let tail =
    float a 2. *: s *: z *: horner z (odd_inverses (if is64 a then 10 else 4))
  in
  (* [ln m = 2 s + 2 s_lo + tail], then in base 2 by [log2 e] in two parts. *)
  let l, l_lo = two_sum (float a 2. *: s) ((float a 2. *: s_lo) +: tail) in
  let c, c_lo = parts a log2e in
  let q, q_lo = two_prod l c in
  let q_lo = q_lo +: (l *: c_lo) +: (l_lo *: c) in
  let hi, lo = two_sum e q in
  (hi, lo +: q_lo)

let pow_float x y =
  let a = abs x and ay = abs y in
  let inf = float x Float.infinity and zero = float x 0. and one = float x 1. in
  let finite v = Ops.lt (abs v) inf in
  let integral = Ops.bitwise_and (finite y) (Ops.eq (Ops.trunc y) y) in
  let half = y *: float y 0.5 in
  let odd = Ops.bitwise_and integral (Ops.ne (Ops.trunc half) half) in
  let l, l_lo =
    log2_parts (where (Ops.bitwise_and (finite a) (Ops.lt zero a)) a one)
  in
  let t, e = two_prod y l in
  let core = exp_parts x t (e +: (y *: l_lo)) in
  let negative_y = Ops.lt y zero in
  let magnitude =
    where (Ops.eq a zero)
      (where negative_y inf zero)
      (where (Ops.eq a inf)
         (where negative_y zero inf)
         (where (Ops.eq ay inf)
            (where (Ops.eq a one) one
               (where (Ops.eq (Ops.lt a one) negative_y) inf zero))
            core))
  in
  let signed =
    where (Ops.bitwise_and (sign_bit x) odd) (neg magnitude) magnitude
  in
  let undefined =
    Ops.bitwise_and (Ops.lt x zero)
      (Ops.bitwise_and (finite x)
         (Ops.bitwise_and (finite y) (Ops.logical_not integral)))
  in
  where (Ops.eq y zero) one
    (where (Ops.eq x one) one
       (where
          (Ops.bitwise_or (isnan x) (isnan y))
          (float x Float.nan)
          (where undefined (float x Float.nan) signed)))

(* An integer power by squaring, wrapping, over the bits of the exponent. A
   negative exponent has an integer power only for a base of 1 or -1; it is 0
   otherwise, as the quotient of 1 by any other integer. *)
let pow_int x y =
  let bits = 8 * Dtype.itemsize (dtype x) in
  let e = lift (where (Ops.lt y (int y 0)) (int y 0) y) in
  let power =
    let rec go k r b =
      if k = bits then r
      else
        let set =
          Ops.ne (Ops.bitwise_and (Ops.shr e (int e k)) (int e 1)) (int e 0)
        in
        go (k + 1) (where set (r *: b) r) (b *: b)
    in
    back (dtype x) (go 0 (int e 1) (lift x))
  in
  if not (is_signed x) then power
  else
    let odd = Ops.ne (Ops.bitwise_and y (int y 1)) (int y 0) in
    let inverse =
      where
        (Ops.eq x (int x 1))
        (int x 1)
        (where
           (Ops.eq x (int x (-1)))
           (where odd (int x (-1)) (int x 1))
           (int x 0))
    in
    where (Ops.lt y (int y 0)) inverse power

(* Remainder

   C's [fmod] is exact: [|x| mod |y|] with [x]'s sign. On the significands [mx]
   and [my] as integers, with [x]'s exponent [d] above [y]'s, it is [(mx * 2^d)
   mod my], reduced a few bits of [d] at a time so that each shifted remainder
   holds in 64 bits, then scaled by [y]'s exponent. *)

let fmod x y =
  let u = unsigned (dtype x) in
  let wide = Dtype.Uint64 in
  let mbits = if is64 x then 52 else 23 in
  let emax = if is64 x then 2046 else 254 in
  let chunk = 63 - mbits in
  let bits_of v = Ops.bitcast (abs v) u in
  let exponent b =
    Ops.cast (Ops.shr b (Ops.const_like b (`Int (Z.of_int mbits)))) Dtype.Int64
  in
  let significand b =
    let frac =
      Ops.bitwise_and b
        (Ops.const_like b (`Int (Z.pred (Z.shift_left Z.one mbits))))
    in
    let implicit = Ops.const_like b (`Int (Z.shift_left Z.one mbits)) in
    Ops.cast
      (where
         (Ops.eq (exponent b) (Ops.const_like (exponent b) (`Int Z.zero)))
         frac
         (Ops.bitwise_or frac implicit))
      wide
  in
  let bx = bits_of x and by = bits_of y in
  let normal e = Ops.maximum e (Ops.const_like e (`Int Z.one)) in
  let ex = normal (exponent bx) and ey = normal (exponent by) in
  let mx = significand bx and my = significand by in
  let my = where (Ops.eq my (int my 0)) (int my 1) my in
  let d = ex -: ey in
  let steps = (emax + chunk - 1) / chunk in
  let rec reduce k r =
    if k = steps then r
    else
      let left =
        Ops.maximum
          (Ops.minimum
             (d -: Ops.const_like d (`Int (Z.of_int (k * chunk))))
             (Ops.const_like d (`Int (Z.of_int chunk))))
          (Ops.const_like d (`Int Z.zero))
      in
      reduce (k + 1) (Ops.fmod (Ops.shl r (Ops.cast left wide)) my)
  in
  let r = reduce 0 (Ops.fmod mx my) in
  (* [r * 2^(ey - bias - mbits)], in two exact steps when the scale is below the
     normal range. *)
  let bias = if is64 x then 1023 else 127 in
  let k = ey -: Ops.const_like ey (`Int (Z.of_int (bias + mbits))) in
  let lowest = Ops.const_like k (`Int (Z.of_int (1 - bias))) in
  let k1 = Ops.maximum k lowest in
  let pow2 k =
    Ops.bitcast
      (Ops.shl
         (Ops.cast (k +: Ops.const_like k (`Int (Z.of_int bias))) u)
         (Ops.const_like (Ops.cast k u) (`Int (Z.of_int mbits))))
      (dtype x)
  in
  let m = Ops.cast r (dtype x) *: pow2 k1 *: pow2 (k -: k1) in
  let ax = abs x and ay = abs y in
  let inf = float x Float.infinity in
  let m = where (Ops.lt ax ay) ax m in
  where
    (Ops.bitwise_or
       (Ops.eq y (float y 0.))
       (Ops.bitwise_or (Ops.eq ax inf) (Ops.bitwise_or (isnan x) (isnan y))))
    (float x Float.nan)
    (where (Ops.eq ay inf) x (copysign m x))

(* Integer division

   An integer quotient or remainder by 0 is 0, and so is a remainder by -1; the
   quotient by -1 wraps. A kernel computes both sides of a selection, so the
   divisor is made safe before it divides. *)

let division x y =
  let zero = Ops.eq y (int y 0) in
  let minus_one =
    if is_signed y then Ops.eq y (int y (-1)) else Ops.bool false
  in
  (zero, minus_one, where (Ops.bitwise_or zero minus_one) (int y 1) y)

let idiv x y =
  if is_float x then float2 (fun x y -> Ops.trunc (x /: y)) x y
  else
    let zero, minus_one, safe = division x y in
    where zero (int x 0)
      (where minus_one (wrapping1 Ops.neg x) (Ops.div ~rounding:`Trunc x safe))

let rem x y =
  if is_float x then float2 fmod x y
  else
    let zero, minus_one, safe = division x y in
    where (Ops.bitwise_or zero minus_one) (int x 0) (Ops.fmod x safe)

(* Extremes

   IEEE 754-2019's maximum and minimum: NaN when either operand is, and [-0.]
   below [0.], read on the sign bit. *)

let extreme ~greater x y =
  if is_float x then
    float2
      (fun x y ->
        let tie = Ops.eq x y in
        let pick =
          if greater then
            Ops.bitwise_or (Ops.lt y x)
              (Ops.bitwise_and tie (Ops.logical_not (sign_bit x)))
          else Ops.bitwise_or (Ops.lt x y) (Ops.bitwise_and tie (sign_bit x))
        in
        where
          (Ops.bitwise_or (isnan x) (isnan y))
          (float x Float.nan) (where pick x y))
      x y
  else if Dtype.equal (dtype x) Bool then
    (if greater then Ops.bitwise_or else Ops.bitwise_and) x y
  else (if greater then Ops.maximum else Ops.minimum) x y

(* Rounding *)

let ceil x =
  let t = Ops.trunc x in
  where (Ops.lt t x) (t +: float x 1.) t

(* The kinds *)

let unary (k : Nx_backend.unary) x =
  match k with
  | Neg -> if is_float x then float1 Ops.neg x else wrapping1 Ops.neg x
  | Recip -> recip x
  | Abs -> abs x
  | Sqrt -> float1 Ops.sqrt x
  | Sign -> sign x
  | Exp -> float1 exp x
  | Log -> float1 log x
  | Sin -> float1 sin x
  | Cos -> float1 cos x
  | Tan -> float1 tan x
  | Asin -> float1 asin x
  | Acos -> float1 acos x
  | Atan -> float1 atan x
  | Sinh -> float1 sinh x
  | Cosh -> float1 cosh x
  | Tanh -> float1 tanh x
  | Trunc -> rounding Ops.trunc x
  | Ceil -> rounding ceil x
  | Floor -> rounding Ops.floor x
  | Round -> rounding round x
  | Erf -> float1 erf x

let binary (k : Nx_backend.binary) x y =
  let arith f = if is_float x then float2 f x y else wrapping2 f x y in
  match k with
  | Add -> arith Ops.add
  | Sub -> arith Ops.sub
  | Mul -> arith Ops.mul
  | Fdiv -> float2 ( /: ) x y
  | Idiv -> idiv x y
  | Mod -> rem x y
  | Pow -> if is_float x then float2 pow_float x y else pow_int x y
  | Atan2 -> float2 atan2 x y
  | Maximum -> extreme ~greater:true x y
  | Minimum -> extreme ~greater:false x y
  | And -> Ops.bitwise_and x y
  | Or -> Ops.bitwise_or x y
  | Xor -> Ops.bitwise_xor x y

let compare (k : Nx_backend.compare) x y =
  match k with
  | Equal -> Ops.eq x y
  | Not_equal -> Ops.ne x y
  | Less -> Ops.lt x y
  | Less_equal -> at_most x y

(* Conversions

   A float converted to an integer saturates at the integer's range, and NaN is
   0; the conversion itself sees only values in range, since C leaves the others
   undefined. *)

let saturate dt x =
  let x = widen x in
  let w = 8 * Dtype.itemsize dt in
  let signed = List.exists (Dtype.equal dt) Dtype.sints in
  let lo = if signed then -.Float.ldexp 1. (w - 1) else 0. in
  let hi = Float.ldexp 1. (if signed then w - 1 else w) in
  let low = at_most x (float x lo) and high = at_least x (float x hi) in
  let bound v = Ops.const ~dtype:dt (v :> Dtype.const) in
  let safe =
    where (Ops.bitwise_or (isnan x) (Ops.bitwise_or low high)) (float x 0.) x
  in
  where (isnan x)
    (bound (`Int Z.zero))
    (where low
       (bound (Dtype.min dt))
       (where high (bound (Dtype.max dt)) (Ops.cast safe dt)))

(* [x], a [float64], as the [float32] rounded to odd: truncated, with its last
   bit set when a discarded bit was. A narrower float rounded from it then
   rounds once, as from [x] itself. *)
let to_float32_odd x =
  let t = Ops.cast x Float32 in
  let back = Ops.cast t Float64 in
  let b = Ops.bitcast t Uint32 in
  let toward_zero = where (Ops.lt (abs x) (abs back)) (b -: int b 1) b in
  where (Ops.eq back x) t
    (Ops.bitcast (Ops.bitwise_or toward_zero (int b 1)) Float32)

let cast dt x =
  if is_float x && Dtype.is_int dt then saturate dt x
  else if is64 x && is_float x && Dtype.is_float dt && Dtype.itemsize dt < 4
  then Ops.cast (to_float32_odd x) dt
  else Ops.cast x dt

let bitcast dt x = Ops.bitcast x dt

(* Random bits

   A Threefry-2x32 word pair lies along the last axis, the low word first. The
   operation hashes the pair as one [uint64]. *)

let threefry key counter =
  let pack t =
    let word i =
      let s = Ops.shape t in
      let last = List.length s - 1 in
      Ops.reshape
        (Ops.shrink t
           (List.mapi
              (fun a _ ->
                if a = last then Some (Ops.Int i, Ops.Int (i + 1)) else None)
              s))
        (List.filteri (fun a _ -> a < last) s)
    in
    let wide w = Ops.cast (Ops.bitcast w Uint32) Uint64 in
    Ops.bitwise_or
      (Ops.shl (wide (word 1)) (int (wide (word 1)) 32))
      (wide (word 0))
  in
  let bits = Ops.alu (pack counter) Op.Threefry [ pack key ] in
  let word b =
    let w = Ops.bitcast (Ops.cast b Uint32) Int32 in
    Ops.reshape w (Ops.shape w @ [ Ops.Int 1 ])
  in
  let lo = word bits and hi = word (Ops.shr bits (int bits 32)) in
  Ops.cat ~axis:(-1) lo [ hi ]
