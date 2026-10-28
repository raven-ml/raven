(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC AND SunPro
  ---------------------------------------------------------------------------*)

(* The exact kinds are bit manipulations and IEEE's correctly rounded
   operations. The transcendental kinds compose tolk's exp2, log2 and sin with
   them, in the forms whose bounds [lower_real.mli] states. *)

open Tolk
open Lower_bits

(* Types

   A node is a [float32] or a [float64]; its bits are the unsigned integer of
   its width. A pair [(a, b)] is a constant's value at each: [a] at float32, [b]
   at float64. *)

let wide x = Dtype.equal (Ops.dtype x) Float64
let pick x (a, b) = if wide x then b else a
let width x = if wide x then 64 else 32
let num u v = Shape.const_like u (`Float v)
let int u n = Shape.const_like u (`Int (Bigint.of_int n))
let word u n = Shape.const_like u (`Int (Bigint.of_int64_unsigned n))

let u64 u n =
  Shape.const_like ~dtype:Uint64 u (`Int (Bigint.of_int64_unsigned n))

let con x p = num x (pick x p)

let of_bits u =
  Ops.bitcast u (if Dtype.equal (Ops.dtype u) Uint64 then Float64 else Float32)

(* Operations *)

let ( +: ) = Ops.add
let ( -: ) = Ops.sub
let ( *: ) = Ops.mul
let ( /: ) x y = Ops.alu x Op.Fdiv [ y ]
let where = Ops.where
let lt = Ops.lt
let gt x y = Ops.lt y x
let eq = Ops.eq
let ne = Ops.ne
let ( ||: ) = Ops.bitwise_or
let ( &&: ) = Ops.bitwise_and

(* [x <= y] and [x >= y], false where an operand is NaN. *)
let le x y = lt x y ||: eq x y
let ge x y = le y x
let band = Ops.bitwise_and
let bor = Ops.bitwise_or
let shl u n = Ops.shl u (int u n)
let shr u n = Ops.shr u (int u n)
let set u = ne u (int u 0)
let sin_ x = Ops.alu x Op.Sin []

(* Constants *)

let one x = num x 1.
let zero x = num x 0.
let half x = num x 0.5
let two x = num x 2.
let inf x = num x Float.infinity
let sign_word x = word (bits x) (Int64.shift_left 1L (width x - 1))
let ones x = word (bits x) (if wide x then -1L else 0xFFFFFFFFL)
let mag_word x = word (bits x) (Int64.pred (Int64.shift_left 1L (width x - 1)))
let bias = (127, 1023)
let mbits = (23, 52)
let mmask = (0x007FFFFFL, 0x000FFFFFFFFFFFFFL)
let min_normal_bits = (0x00800000L, 0x0010000000000000L)
let inf_bits = (0x7F800000L, 0x7FF0000000000000L)
let sub_scale = (0x1p23, 0x1p52)

(* From [integral] on every float is an integer. *)
let integral = (0x1p23, 0x1p52)
let ln2 = (0x1.62e43p-1, 0x1.62e42fefa39efp-1)
let log2e = (0x1.715476p+0, 0x1.71547652b82fep+0)
let log2e_lo = (0x1.4ae0cp-26, 0x1.777d0ffda0d24p-56)
let pio2_hi = (0x1.921fb6p+0, 0x1.921fb54442d18p+0)
let pio2_lo = (-0x1.777a5cp-25, 0x1.1a62633145c07p-54)
let pi_hi = (0x1.921fb6p+1, 0x1.921fb54442d18p+1)
let pi_lo = (-0x1.777a5cp-24, 0x1.1a62633145c07p-53)

(* Where cosh, sinh and tanh take their far forms: past it, e^-2|x| is below the
   unit roundoff of both types. *)
let hyp_far = 20.

(* Below it, the trigonometric kinds reduce by Cody and Waite; from it, by Payne
   and Hanek. *)
let trig_switch = 30.

let horner x = function
  | [] -> invalid_arg "Lower_real.horner: no coefficient"
  | c :: cs -> List.fold_left (fun p c -> mulacc p x (num x c)) (num x c) cs

(* Helpers *)

let copysign m s =
  of_bits (bor (band (bits m) (mag_word m)) (band (bits s) (sign_word s)))

(* m with its sign flipped where s's sign bit is set. *)
let flip m s = where (negative s) (negated m) m

(* 2^k for a normal power, [k] an int32 node. *)
let pow2 x k =
  let e = Ops.cast (Ops.add k (int k (pick x bias))) (unsigned (Ops.dtype x)) in
  of_bits (shl e (pick x mbits))

(* y 2^k rounded once: the first product is exact. *)
let scale y k =
  let k1 = shr k 1 in
  y *: pow2 y k1 *: pow2 y (Ops.sub k k1)

let two_sum a b =
  let hi = a +: b in
  let bb = hi -: a in
  (hi, a -: (hi -: bb) +: (b -: bb))

(* The exponent e of a finite x > 0, x = m 2^e with m in [1, 2). *)
let ilogb x =
  let sub = lt (bits x) (word (bits x) (pick x min_normal_bits)) in
  let s = where sub (x *: con x sub_scale) x in
  let e = Ops.cast (shr (bits s) (pick x mbits)) Int32 in
  let e = Ops.sub e (int e (pick x bias)) in
  Ops.sub e (where sub (int e (pick x mbits)) (int e 0))

(* Exact kinds *)

let sign a =
  let s =
    where (gt a (zero a)) (one a) (where (lt a (zero a)) (num a (-1.)) (zero a))
  in
  where (is_nan a) a s

let trunc a =
  let m = magnitude a in
  let c = con a integral in
  let t = m +: c -: c in
  let t = where (gt t m) (t -: one a) t in
  let r = of_bits (bor (bits t) (band (bits a) (sign_word a))) in
  where (lt m c) r a

let floor a =
  let t = trunc a in
  where (gt t a) (t -: one a) t

let ceil a =
  let t = trunc a in
  where (lt t a) (t +: one a) t

let round a =
  let t = trunc a in
  let d = magnitude (a -: t) in
  where (ge d (half a)) (where (lt a (zero a)) (t -: one a) (t +: one a)) t

(* The greater (lesser) operand, [a] where [a] is NaN, and of equal operands the
   AND (OR) of their bits. *)
let maximum a b =
  let g = where (is_nan a) a (where (gt a b) a b) in
  let ne = where (ne a b) (ones a) (word (bits a) 0L) in
  of_bits (band (bits g) (bor (bits a) ne))

let minimum a b =
  let g = where (is_nan a) a (where (lt a b) a b) in
  let eq = where (ne a b) (word (bits a) 0L) (ones a) in
  of_bits (bor (bits g) (band (bits a) eq))

(* fmod, exact. With |x| = mx 2^ex and |y| = my 2^ey, mx and my integers of the
   significand's bits, |x| mod |y| is (mx 2^(ex - ey) mod my) 2^ey: the
   remainder of mx by my, shifted left by the exponents' difference a chunk at a
   time and reduced again, each shift small enough that the shifted remainder
   holds in 64 bits. *)
let fmod x y =
  let ux = band (bits x) (mag_word x) and uy = band (bits y) (mag_word y) in
  let ib = word ux (pick x inf_bits) in
  let special =
    eq uy (word uy 0L) ||: Ops.logical_not (lt ux ib) ||: gt uy ib
  in
  let mb = pick x mbits and b = pick x bias in
  let field u = Ops.cast (shr u mb) Int32 in
  let normal e = where (lt e (int e 1)) (int e 1) e in
  let sig_ u =
    let f = band u (word u (pick x mmask)) in
    let lead = word u (Int64.shift_left 1L mb) in
    Ops.cast (where (eq (field u) (int (field u) 0)) f (bor f lead)) Uint64
  in
  let ex = normal (field ux) and ey = normal (field uy) in
  let mx = sig_ ux and my = sig_ uy in
  let my = where (eq my (u64 my 0L)) (u64 my 1L) my in
  let d = Ops.sub ex ey in
  let chunk = 63 - mb in
  let steps = ((2 * b) + chunk - 1) / chunk in
  let modu r = Ops.alu r Op.Cmod [ my ] in
  let r = ref (modu mx) in
  for k = 0 to steps - 1 do
    let left = Ops.sub d (int d (k * chunk)) in
    let left = where (lt left (int d 0)) (int d 0) left in
    let left = where (gt left (int d chunk)) (int d chunk) left in
    r := modu (Ops.shl !r (Ops.cast left Uint64))
  done;
  (* r 2^(ey - bias - mbits), in two exact products when the scale is below the
     normal range *)
  let e = Ops.sub ey (int ey (b + mb)) in
  let e1 = where (lt e (int e (1 - b))) (int e (1 - b)) e in
  let m = Ops.cast !r (Ops.dtype x) *: pow2 x e1 *: pow2 x (Ops.sub e e1) in
  let sx = band (bits x) (sign_word x) in
  let r = where (lt ux uy) x (of_bits (bor (bits m) sx)) in
  where special (nan x) r

(* Exponentials and logarithms *)

(* 2^(hi + lo) for |lo| at most an ulp of hi: 2^hi (1 + lo ln2). A positive hi
   takes 2^(hi - 1), doubled after the product, since hi may round up to where
   2^hi overflows while 2^(hi + lo) does not. Where 2^hi is infinite, zero or
   NaN, the product is NaN or 2^hi, and 2^hi is kept. *)
let exp2_split hi lo =
  let up = gt hi (zero hi) in
  let y = Ops.exp2 (where up (hi -: one hi) hi) in
  let r = mulacc y (lo *: con lo ln2) y in
  where (is_nan r) y r *: where up (two hi) (one hi)

(* x / ln2 as hi + lo: hi rounded, lo its error, exact from the fma, plus x
   times the error of 1 / ln2 rounded. *)
let exp_parts x =
  let hi = x *: con x log2e in
  (hi, mulacc x (con x log2e) (negated hi) +: (x *: con x log2e_lo))

let exp x =
  let hi, lo = exp_parts x in
  exp2_split hi lo

(* fdlibm's log (e_log.c, e_logf.c). With x = m 2^e, m in (1/sqrt 2, sqrt 2],
   log x is e ln2 + log(1 + f), f = m - 1 exact, and log(1 + f) is f - (h - s (h
   + R)) with h = f^2 / 2, s = f / (2 + f) and R a minimax polynomial in s^2 for
   2 atanh(s) / s - 2: every term but f is small, so only the last sum rounds at
   the result's scale. ln2 is in two parts, the first of few enough bits that
   its product by e is exact. It reads no log2: a product of one by ln 2 would
   add its rounding to log2's error. Every positive float32 is within one ulp of
   libm's log rounded, as are drawn float64 operands of libm's; on the host exp
   and cos are within two, sin within one, over the same operands. The
   coefficients and the parts carry this notice:

   Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved. Developed
   at SunPro, a Sun Microsystems, Inc. business. Permission to use, copy,
   modify, and distribute this software is freely granted, provided that this
   notice is preserved. *)

let log_poly =
  ( [ 0xf89e26.0p-26; 0x91e9ee.0p-25; 0xccce13.0p-25; 0xaaaaaa.0p-24 ],
    [
      1.479819860511658591e-01;
      1.531383769920937332e-01;
      1.818357216161805012e-01;
      2.222219843214978396e-01;
      2.857142874366239149e-01;
      3.999999999940941908e-01;
      6.666666666666735130e-01;
    ] )

let ln2_hi = (6.9313812256e-01, 6.93147180369123816490e-01)
let ln2_lo = (9.0580006145e-06, 1.90821492927058770002e-10)
let sqrt2 = (0x1.6a09e6p+0, 0x1.6a09e667f3bcdp+0)

(* (m, e) with a = m 2^e, m in (1/sqrt 2, sqrt 2] and e an integer of a's type,
   for a finite a > 0: a subnormal is scaled into the normals first. *)
let normalized a =
  let sub = lt (bits a) (word (bits a) (pick a min_normal_bits)) in
  let s = where sub (a *: con a sub_scale) a in
  let mb = pick a mbits in
  let one_bits = Int64.shift_left (Int64.of_int (pick a bias)) mb in
  let m =
    of_bits
      (bor
         (band (bits s) (word (bits s) (pick a mmask)))
         (word (bits s) one_bits))
  in
  let e = Ops.cast (ilogb a) (Ops.dtype a) in
  let high = gt m (con a sqrt2) in
  (where high (m *: half a) m, where high (e +: one a) e)

let log x =
  let positive = gt x (zero x) &&: lt x (inf x) in
  let m, e = normalized (where positive x (one x)) in
  let f = m -: one x in
  let s = f /: (two x +: f) in
  let z = s *: s in
  let h = half x *: f *: f in
  let r = z *: horner z (pick x log_poly) in
  let l =
    (e *: con x ln2_hi) -: (h -: ((s *: (h +: r)) +: (e *: con x ln2_lo)) -: f)
  in
  (* The sign is read from the bits: a target that flushes subnormals takes a
     negative one for -0. *)
  let below = negative x &&: ne (bits (magnitude x)) (word (bits x) 0L) in
  let special =
    where (eq x (zero x)) (negated (inf x)) (where (is_nan x) x (inf x))
  in
  where below (nan x) (where positive l special)

(* Kahan's expm1: with w = e^x rounded, (w - 1) x / log w. w - 1 is expm1 at log
   w exactly, and expm1(t) / t varies slowly, so w's error cancels to first
   order. *)
let expm1 x =
  let w = exp x in
  let wm = w -: one x in
  let v = wm *: (x /: log w) in
  let v = where (eq w (inf x)) w v in
  let v = where (eq wm (num x (-1.))) wm v in
  where (eq w (one x)) x v

(* Goldberg's log1p: with w = 1 + x rounded, log w x / (w - 1), w - 1 exact. *)
let log1p x =
  let w = one x +: x in
  let v = log w *: (x /: (w -: one x)) in
  let v = where (eq x (inf x)) x v in
  where (eq w (one x)) x v

(* Trigonometric kinds *)

(* |x| = q pi/2 + r, |r| <= pi/4, by tolk's reductions, each fed 1 on the lanes
   it does not take; r is NaN for an infinite or NaN x. *)
let reduce x =
  let a = magnitude x in
  let near = lt a (num x trig_switch) and finite = lt a (inf x) in
  let rn, qn = Transcendental.cody_waite_reduction (where near a (one x)) in
  let far = Ops.logical_not near &&: finite in
  let rf, qf = Transcendental.payne_hanek_reduction (where far a (one x)) in
  (where finite (where near rn rf) (nan x), where near qn qf)

(* sin r and cos r = 1 - 2 sin^2(r/2), which cancels nowhere. *)
let sin_cos r =
  let s = sin_ (r *: half r) in
  (sin_ r, mulacc (negated (s +: s)) s (one r))

let odd q = set (band q (int q 1))

(* cos x = cos(r + q pi/2): cos r, -sin r, -cos r, sin r by q mod 4. *)
let cos x =
  let r, q = reduce x in
  let s, c = sin_cos r in
  let v = where (odd q) s c in
  where (set (band (Ops.add q (int q 1)) (int q 2))) (negated v) v

(* tan |x| = tan r or -1 / tan r by q's parity, of x's sign. *)
let tan x =
  let r, q = reduce x in
  let s, c = sin_cos r in
  let o = odd q in
  flip (where o (negated c) s /: where o s c) x

(* Inverse trigonometric kinds *)

(* atan t for t in [0, 1]: at float32, a minimax polynomial; at float64, one on
   [0, 7/16], after the half-angle step atan t = 2 atan(t / (1 + sqrt(1 +
   t^2))), which maps [0, 1] into [0, tan(pi/8)]. *)
let atan_unit t =
  if not (wide t) then
    let z = t *: t in
    let a =
      horner z
        [
          0x1.7ed21ep-9;
          -0x1.0c2c0cp-6;
          0x1.61fdd4p-5;
          -0x1.3556b4p-4;
          0x1.b4e12ap-4;
          -0x1.230adcp-3;
          0x1.9978f4p-3;
          -0x1.5554dcp-2;
        ]
    in
    mulacc (t *: z) a t
  else
    let h = t /: (one t +: Ops.sqrt (mulacc t t (one t))) in
    let z = h *: h in
    let a =
      horner z
        [
          -0x1.0a7ed459d86b8p-6;
          0x1.2b17be1f3a52cp-5;
          -0x1.97a089c9a7af5p-5;
          0x1.ddddadf2f06c5p-5;
          -0x1.10d6017d18c7ap-4;
          0x1.3b0f205706eb3p-4;
          -0x1.745cdba3cb3p-4;
          0x1.c71c6fdb0e4afp-4;
          -0x1.2492491ff2fd8p-3;
          0x1.999999998e7dp-3;
          -0x1.555555555550bp-2;
        ]
    in
    two t *: (h +: (h *: z *: a))

(* atan(n / d) for n, d >= 0: the angle of the smaller over the larger, taken
   from pi/2 (in two parts) where n is the larger. *)
let angle n d =
  let big = gt n d in
  let p = atan_unit (where big (d /: n) (n /: d)) in
  where big (con n pio2_hi +: (con n pio2_lo -: p)) p

let atan x = copysign (angle (magnitude x) (one x)) x
let asin x = atan (x /: Ops.sqrt (mulacc (negated x) x (one x)))
let acos x = two x *: atan (Ops.sqrt ((one x -: x) /: (one x +: x)))

(* atan2: two zeros are the angle 0, two infinities pi/4; where x's sign bit is
   set the angle is taken from pi; y's sign is copied. *)
let atan2 y x =
  let ax = magnitude x and ay = magnitude y in
  let zeros = eq ax (zero x) &&: eq ay (zero x) in
  let infs = eq ax (inf x) &&: eq ay (inf x) in
  let n = where infs (one x) (where zeros (zero x) ay) in
  let d = where (zeros ||: infs) (one x) ax in
  let v = angle n d in
  let v = where (negative x) (con x pi_hi +: (con x pi_lo -: v)) v in
  copysign v y

(* Hyperbolic kinds: past hyp_far, cosh and |sinh| are e^|x| / 2 = 2^(|x| log2e
   - 1), the subtraction exact, so that nothing overflows before the result. *)

let half_exp a =
  let hi, lo = exp_parts a in
  exp2_split (hi -: one a) lo

let cosh x =
  let a = magnitude x in
  let e = exp a in
  where (lt a (num x hyp_far)) ((e +: (one x /: e)) *: half x) (half_exp a)

let sinh x =
  let a = magnitude x in
  let t = expm1 a in
  let near = (t +: (t /: (t +: one x))) *: half x in
  copysign (where (lt a (num x hyp_far)) near (half_exp a)) x

let tanh x =
  let a = magnitude x in
  let t = expm1 (two x *: a) in
  copysign (where (ge a (num x hyp_far)) (one x) (t /: (t +: two x))) x

(* pow: 2^(y log2|x|), the product split exactly by an fma, with C99 F.9.4.4's
   special values. *)
let pow x y =
  let l = Ops.log2 (magnitude x) in
  let h = y *: l in
  let v = exp2_split h (mulacc y l (negated h)) in
  let integer v = eq (trunc v) v in
  let odd = integer y &&: Ops.logical_not (integer (y *: half y)) in
  let v = where (odd &&: negative x) (negated v) v in
  let below = lt x (zero x) &&: gt x (negated (inf x)) in
  let v = where (below &&: Ops.logical_not (integer y)) (nan x) v in
  let v = where (eq x (num x (-1.)) &&: eq (magnitude y) (inf x)) (one x) v in
  where (eq x (one x) ||: eq y (zero x)) (one x) v

(* erf, a polynomial per range; at float64, past 2.5, 1 - e^(G(a) - a^2). *)

let erf32 x =
  let a = magnitude x in
  let z = a *: a in
  let e =
    horner z
      [
        -0x1.252b1cp-17;
        0x1.cb3dccp-14;
        -0x1.ba9afap-11;
        0x1.55fc8cp-8;
        -0x1.b8260ep-6;
        0x1.ce2ee6p-4;
        -0x1.812746p-2;
      ]
  in
  let tail =
    where
      (lt a (num x 0x1p-100))
      (zero x)
      (a *: mulacc e z (num x (-0x1.f7ac92p-25)))
  in
  let small = mulacc a (num x 0x1.20dd76p+0) tail in
  let four = num x 4. in
  let u = where (lt four a) four a -: num x 2.5 in
  let q =
    horner u
      [
        -0x1.e11c0ep-21;
        -0x1.11733cp-21;
        0x1.088b0ep-16;
        -0x1.28a63ap-16;
        -0x1.54d0a6p-14;
        0x1.1b8868p-12;
        -0x1.0f0962p-12;
        -0x1.25a286p-11;
        0x1.6ea774p-9;
        -0x1.906118p-8;
        0x1.1a8c54p-7;
        -0x1.11ab44p-7;
        0x1.64e34ap-8;
        -0x1.1d7fe4p-9;
        0x1.aab886p-12;
      ]
  in
  copysign (where (lt a (one x)) small (one x -: q)) x

let erf_near64 x =
  let a = magnitude x in
  let z = a *: a in
  let e =
    horner z
      [
        0x1.3bd95b2fcc4c8p-42;
        -0x1.b3fe40ecb1d83p-38;
        0x1.9a0cfad36fb97p-34;
        -0x1.517c99ae9a999p-30;
        0x1.fcbb28cc88fd2p-27;
        -0x1.5f73c3747d06bp-23;
        0x1.b9e6c38c36437p-20;
        -0x1.f4d25bf9c9292p-17;
        0x1.f9a326f7afd31p-14;
        -0x1.c02db400361ccp-11;
        0x1.565bcd0e6a301p-8;
        -0x1.b82ce31288b48p-6;
        0x1.ce2f21a042be2p-4;
        -0x1.812746b0379e7p-2;
      ]
  in
  let small =
    mulacc a
      (num x 0x1.20dd750429b6dp+0)
      (a *: mulacc e z (num x 0x1.1ae3a914fed8p-56))
  in
  let u = a -: num x 1.75 in
  let q =
    horner u
      [
        0x1.1952d5a4cae0ep-27;
        -0x1.28e112f89c851p-26;
        -0x1.4fdfcb701a293p-24;
        0x1.5c6e3c15a59e3p-22;
        0x1.7545709555e68p-23;
        -0x1.b74c8ca4aa58fp-19;
        0x1.367dce5756fd5p-18;
        0x1.294cd0fe59fcfp-16;
        -0x1.20080dba8f773p-14;
        0x1.6d715d76f41ap-18;
        0x1.d041f3b63da76p-12;
        -0x1.d4cf0b04a217cp-11;
        -0x1.7ec1d228fe1f5p-11;
        0x1.8d10fb6c23dd8p-8;
        -0x1.25b378c8cdfffp-7;
        -0x1.b22257dd0a3ddp-8;
        0x1.8a0da543053a6p-5;
        -0x1.7148c3d57c2d8p-4;
        0x1.7a4a8a2bdce2cp-4;
        -0x1.b055303221016p-5;
        0x1.b4be201caa4b3p-7;
      ]
  in
  copysign (where (lt a (one x)) small (one x -: q)) x

(* Past 2.5, a taken as 5.95 past it, where erf rounds to 1. *)
let erf_far64 x =
  let a = magnitude x in
  let cap = num x 5.95 in
  let b = where (lt cap a) cap a in
  let u = b -: num x 4.25 in
  let g =
    horner u
      [
        0x1.7742963aff69cp-27;
        0x1.54fd3a02fb6fcp-30;
        0x1.52f34766557c1p-22;
        -0x1.8a631a0b82da7p-20;
        0x1.71c1a68f670c6p-17;
        -0x1.2f3790e7befacp-14;
        0x1.f7e642e7cc5b2p-12;
        -0x1.abc91bf263532p-9;
        0x1.87d48af31b6e3p-6;
        -0x1.ca4aec3d2de83p-3;
        -0x1.05cb67754a622p+1;
      ]
  in
  copysign (one x -: exp (g -: (b *: b))) x

let erf x =
  if not (wide x) then erf32 x
  else where (gt (magnitude x) (num x 2.5)) (erf_far64 x) (erf_near64 x)

(* Kinds *)

let unary (k : Nx_kernel.Prog.unary) x =
  match k with
  | Neg -> negated x
  | Recip -> one x /: x
  | Abs -> magnitude x
  | Sign -> sign x
  | Sqrt -> Ops.sqrt x
  | Exp -> exp x
  | Exp2 -> Ops.exp2 x
  | Log -> log x
  | Log2 -> Ops.log2 x
  | Log1p -> log1p x
  | Expm1 -> expm1 x
  | Sin -> sin_ x
  | Cos -> cos x
  | Tan -> tan x
  | Asin -> asin x
  | Acos -> acos x
  | Atan -> atan x
  | Sinh -> sinh x
  | Cosh -> cosh x
  | Tanh -> tanh x
  | Erf -> erf x
  | Floor -> floor x
  | Ceil -> ceil x
  | Round -> round x
  | Trunc -> trunc x
  | Asinh -> invalid_arg "Lower_real.unary: floats do not take Asinh"
  | Acosh -> invalid_arg "Lower_real.unary: floats do not take Acosh"
  | Atanh -> invalid_arg "Lower_real.unary: floats do not take Atanh"

let binary (k : Nx_kernel.Prog.binary) x y =
  match k with
  | Add -> x +: y
  | Sub -> x -: y
  | Mul -> x *: y
  | Fdiv -> x /: y
  | Mod -> fmod x y
  | Pow -> pow x y
  | Atan2 -> atan2 x y
  | Maximum -> maximum x y
  | Minimum -> minimum x y
  | Idiv | And | Or | Xor | Threefry ->
      invalid_arg "Lower_real.binary: a kind floats do not take"

let compare (k : Nx_kernel.Prog.compare) x y =
  match k with
  | Equal -> eq x y
  | Not_equal -> ne x y
  | Less -> lt x y
  | Less_equal -> le x y

let fma = mulacc
