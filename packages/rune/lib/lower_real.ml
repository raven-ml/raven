(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The exact kinds are bit manipulations and IEEE's correctly rounded
   operations. Some transcendental kinds restate nx's kernel header,
   nx_kinds.h with nx_kinds_real.h, operation for operation: the same
   constants, the same order, an fma ([Mulacc], rounded once) where the C
   writes one and none elsewhere, so that they give eager's bits. A C branch
   is a selection, both sides computed, a side's operands made safe on the
   lanes it does not take. Each such function names the header's function it
   follows; when the header changes, it changes with it.

   The other transcendental kinds compose tolk's exp2 with the exact kinds and
   those of nx's sequence, in forms that cancel nowhere, whose bounds
   [lower_real.mli] states. Those bounds are derived to first order in the
   unit roundoff u. An error of n ulps is taken as a relative error of 2 n u,
   and a relative error d as d / u + 1/2 ulps after the last rounding. The
   bounds of erf's polynomials, of float32 atan and of the host's exp are
   measured. *)

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

(* Where the complex kinds take the hyperbolic functions' far forms: past it,
   e^-2|x| is below the unit roundoff of both types. *)
let hyp_far = 20.

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
  let modu r = Ops.alu r Op.Truncmod [ my ] in
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

(* Exponentials in two parts, of tolk's exp2, for pow and the complex kinds *)

(* 2^(hi + lo) for |lo| at most two ulps of hi: 2^hi (1 + lo ln2), whose
   relative error (lo ln2)^2 / 2 is below 2^-30 at float32 and 2^-80 at
   float64 where 2^hi is finite and non-zero. A positive hi takes 2^(hi - 1),
   doubled after the product, since hi may round up to where 2^hi overflows
   while 2^(hi + lo) does not. Where 2^hi is infinite, zero or NaN, the product
   is NaN or 2^hi, and 2^hi is kept. *)
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

(* Logarithms

   nx_kinds_real.h's: u = 2^k (1 + f) with 1 + f in [sqrt(1/2), sqrt 2), f
   exact, and with s = f / (2 + f), log(1 + f) = f - f^2/2 + s (f^2/2 + R),
   R = s^2 T(s^2). Constants are nx_kinds.h's macros of the same names, at
   float32 then float64, their text copied. *)

let sqrt_half_bits = (0x3F3504F3L, 0x3FE6A09E667F3BCDL)
let ln2_hi = (0x1.62e4p-1, 0x1.62e42fefa38p-1)
let ln2_lo = (0x1.7f7d1cp-20, 0x1.ef35793c7673p-45)
let ivln2_hi = (0x1.716p+0, 0x1.71547652p+0)
let ivln2_lo = (-0x1.7135a8p-13, 0x1.705fc2eefa2p-33)
let log2_hi_mask = (0xFFFFF000L, 0xFFFFFFFF00000000L)

(* nx_log_t_f32 and nx_log_t_f64, from the leading coefficient. *)
let log_t =
  ( [ 0x1.ab4e0cp-3; 0x1.257d1cp-2; 0x1.9996acp-2; 0x1.555556p-1 ],
    [
      0x1.0c1994438ce53p-3;
      0x1.0fbc6f8b23776p-3;
      0x1.3b1c4a71fa6cep-3;
      0x1.745cf89e46188p-3;
      0x1.c71c720256c1fp-3;
      0x1.2492492476379p-2;
      0x1.9999999999a3cp-2;
      0x1.5555555555555p-1;
    ] )

(* NX_ONE_BITS, 1's bits: the bias in the exponent field. *)
let one_bits x = Int64.shift_left (Int64.of_int (pick x bias)) (pick x mbits)

(* nx_log_reduce: (f, k) for u > 0, a subnormal scaled up first. *)
let log_reduce u =
  let b = bits u in
  let sub = lt b (word b (pick u min_normal_bits)) in
  let u = where sub (u *: con u sub_scale) u in
  let half = pick u sqrt_half_bits in
  let ix = Ops.add (bits u) (word b (Int64.sub (one_bits u) half)) in
  let e = Ops.cast (shr ix (pick u mbits)) Int32 in
  let e = Ops.sub e (int e (pick u bias)) in
  let k = Ops.sub e (where sub (int e (pick u mbits)) (int e 0)) in
  let m = Ops.add (band ix (word b (pick u mmask))) (word b half) in
  (of_bits m -: one u, Ops.cast k (Ops.dtype u))

(* nx_log_core: k ln2 + log(1 + f) + c. *)
let log_core (f, k) c =
  let s = f /: (two f +: f) in
  let z = s *: s in
  let hfsq = half f *: f *: f in
  let t = horner z (pick f log_t) in
  let sr = mulacc s (mulacc z t hfsq) (mulacc k (con f ln2_lo) c) in
  mulacc k (con f ln2_hi) (negated (hfsq -: sr -: f))

(* nx_log_special: the value at u = 1 + x outside (0, inf), and at a NaN
   x. *)
let log_special u x y =
  let y = where (eq u (inf u)) u y in
  let y = where (eq u (zero u)) (negated (inf u)) y in
  let y = where (lt u (zero u)) (nan u) y in
  where (is_nan x) x y

(* nx_log *)
let log x = log_special x x (log_core (log_reduce x) (zero x))

(* nx_log2: log(1 + f) as hi + lo, hi truncated so that its product by
   1/ln2's head is exact. *)
let log2 x =
  let f, k = log_reduce x in
  let s = f /: (two f +: f) in
  let z = s *: s in
  let hfsq = half f *: f *: f in
  let hi =
    of_bits (band (bits (f -: hfsq)) (word (bits f) (pick x log2_hi_mask)))
  in
  let t = horner z (pick f log_t) in
  let lo = mulacc s (mulacc z t hfsq) (f -: hi -: hfsq) in
  let v = mulacc lo (con x ivln2_hi) ((lo +: hi) *: con x ivln2_lo) in
  log_special x x (mulacc hi (con x ivln2_hi) v +: k)

(* nx_log1p: u = 1 + x rounded, c its exact error; log(u + c) = log u +
   c/u. *)
let log1p x =
  let uh, ul = two_sum (one x) x in
  let y = log_core (log_reduce uh) (ul /: uh) in
  log_special uh x (where (eq x (zero x)) x y)

(* (m, e) with a = m 2^e, m in (top / 2, top] and e an integer of a's type,
   for a finite a > 0 and top in (1, 2): a subnormal is scaled into the normals
   first. *)
let normalized ~top a =
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
  let high = gt m (con a top) in
  (where high (m *: half a) m, where high (e +: one a) e)

(* nx's sequences

   Helpers of nx_kinds_real.h, each named as there. *)

let u32 u n = Shape.const_like ~dtype:Uint32 u (`Int (Bigint.of_int n))

(* [bits_of x v] is the bits of the float [v] at [x]'s type, as a word. *)
let bits_of x v =
  let b =
    if wide x then Int64.bits_of_float v
    else Int64.logand (Int64.of_int32 (Int32.bits_of_float v)) 0xFFFFFFFFL
  in
  word (bits x) b

(* NX_SHIFT, 1.5 * 2^MBITS: adding it rounds a float below 2^(MBITS - 1) to
   the nearest integer. *)
let shifter x = num x (Float.ldexp 3. (pick x mbits - 1))

(* nx_fast_two_sum *)
let fast_two_sum hi lo =
  let h = hi +: lo in
  (h, lo -: (h -: hi))

(* nx_div_pair *)
let div_pair (nh, nl) (dh, dl) =
  let nh, nl = fast_two_sum nh nl in
  let dh, dl = fast_two_sum dh dl in
  let rd = one dh /: dh in
  let qh = nh *: rd in
  (qh, (mulacc (negated qh) dh nh +: mulacc (negated qh) dl nl) *: rd)

(* nx_nearest: from t = v + NX_SHIFT, the integer nearest v, as a float and
   as an int32, the low word of the difference of the bits. *)
let nearest t =
  let s = shifter t in
  (t -: s, Ops.cast (Ops.sub (bits t) (bits s)) Int32)

(* nx_nan1: r, or a where a is a NaN. *)
let nan1 a r = where (is_nan a) a r

(* Exponentials

   nx_kinds_real.h's: x = k ln2 + r with k the integer nearest x / ln2, and
   expm1 r = r + r^2 Q(r). Constants are nx_kinds.h's macros of the same
   names, at float32 then float64, their text copied. *)

let exp_hi = (89., 710.)
let exp_lo = (-104., -746.)
let exp2_hi = (129., 1025.)
let exp2_lo = (-151., -1076.)
let expm1_lo = (-26., -40.)
let expm1_kcap = (64, 128)
let expm1_tiny = (0x1p-25, 0x1p-54)
let hyp_hi = (90., 712.)
let hyp_kcap = (32, 64)
let tanh_small = (0.625, 0.625)
let tanh_hi = (9.5, 19.5)

(* nx_exp_q_f32 and nx_exp_q_f64, from the leading coefficient. *)
let exp_q =
  ( [
      0x1.6ac6e2p-10;
      0x1.123ddcp-7;
      0x1.555858p-5;
      0x1.55548cp-3;
      0x1.fffffcp-2;
    ],
    [
      0x1.ad7f55323502cp-26;
      0x1.28ad86039e73p-22;
      0x1.71df2581c2cafp-19;
      0x1.a01999f89494ap-16;
      0x1.a01a012a34bddp-13;
      0x1.6c16c18430f75p-10;
      0x1.1111111127c73p-7;
      0x1.555555555085fp-5;
      0x1.55555555554fap-3;
      0x1.000000000000ap-1;
    ] )

(* nx_exp2_poly_f32 and nx_exp2_poly_f64, to the last term, 1. *)
let exp2_p =
  ( [
      0x1.177586p-13;
      0x1.5ef2ep-10;
      0x1.3b6816p-7;
      0x1.c6af9ep-5;
      0x1.ebfb94p-3;
      0x1.62e43p-1;
      1.;
    ],
    [
      0x1.c2cea03f4f3ap-36;
      0x1.ea014ee31fd31p-32;
      0x1.e4d06c7074e68p-28;
      0x1.b524dd55482cfp-24;
      0x1.62c021e26f2c5p-20;
      0x1.ffcbfc74d611dp-17;
      0x1.430912f883cf5p-13;
      0x1.5d87fe78a26c9p-10;
      0x1.3b2ab6fba4e22p-7;
      0x1.c6b08d704a0cfp-5;
      0x1.ebfbdff82c58fp-3;
      0x1.62e42fefa39efp-1;
      1.;
    ] )

(* nx_tanh_t_f32 and nx_tanh_t_f64 *)
let tanh_t =
  ( [
      -0x1.750ce4p-8;
      0x1.51f8cep-6;
      -0x1.b835a4p-5;
      0x1.1106f6p-3;
      -0x1.555532p-2;
    ],
    [
      -0x1.0da727a627179p-16;
      0x1.4391214300e67p-14;
      -0x1.df5579a8b4a07p-13;
      0x1.336612e405191p-11;
      -0x1.7d68a32fdd9b9p-10;
      0x1.d6cf280ecda0fp-9;
      -0x1.226e162bf7b5p-7;
      0x1.664f477adac7ap-6;
      -0x1.ba1ba1b5064afp-5;
      0x1.111111110accdp-3;
      -0x1.55555555554ffp-2;
    ] )

(* nx_exp_reduce: (r, rc, k), rc r's rounding error. ln2's head has few
   enough bits that k ln2_hi is exact, so the first fma is. *)
let exp_reduce x =
  let kf, k = nearest (mulacc x (con x log2e) (shifter x)) in
  let m = negated kf in
  let rh = mulacc m (con x ln2_hi) x in
  let r = mulacc m (con x ln2_lo) rh in
  (r, mulacc m (con x ln2_lo) (rh -: r), k)

(* expm1 r = r + r^2 Q(r), and r^2 Q(r) + c. *)
let poly_q r c = mulacc (r *: r) (horner r (pick r exp_q)) c

(* nx_exp: x clamped to the range past which e^x rounds to 0 or
   overflows. *)
let exp x =
  let c = where (lt x (con x exp_hi)) x (con x exp_hi) in
  let c = where (gt c (con x exp_lo)) c (con x exp_lo) in
  let r, _, k = exp_reduce c in
  nan1 x (scale (one x +: poly_q r r) k)

(* nx_exp_near: for |x| <= NX_EXP_NEAR, k added to 1 + em's exponent. *)
let exp_near x =
  let r, _, k = exp_reduce x in
  let b = bits (one x +: poly_q r r) in
  let k = Ops.cast k (unsigned (Ops.dtype x)) in
  of_bits (Ops.add b (shl k (pick x mbits)))

(* nx_exp2: x = k + r, |r| <= 1/2, 2^r = 1 + r P(r). *)
let exp2 x =
  let x = where (gt x (con x exp2_hi)) (con x exp2_hi) x in
  let x = where (lt x (con x exp2_lo)) (con x exp2_lo) x in
  let kf, k = nearest (x +: shifter x) in
  nan1 x (scale (horner (x -: kf) (pick x exp2_p)) k)

(* nx_expm1: 2^k ((1 - 2^-k) + r + r^2 Q(r)), the bracket summed exactly but
   for the last rounding, r's rounding error rc added scaled by e^r. *)
let expm1 x =
  let c = where (gt x (con x exp_hi)) (con x exp_hi) x in
  let c = where (lt c (con x expm1_lo)) (con x expm1_lo) c in
  let r, rc, k = exp_reduce c in
  let h = poly_q r (mulacc rc r rc) in
  let cap = int k (pick x expm1_kcap) in
  let m = pow2 x (Ops.sub (int k 0) (where (gt k cap) cap k)) in
  let uh, ul = two_sum (one x) (negated m) in
  let sh, sl = two_sum uh r in
  let y = scale (sh +: (sl +: (ul +: h))) k in
  nan1 x (where (lt (magnitude x) (con x expm1_tiny)) x y)

(* nx_hyp_reduce: for a = k ln2 + r, em = expm1 r and w = 1 + em as pairs,
   and 2^-2k. *)
let hyp_reduce a =
  let a = where (gt a (con a hyp_hi)) (con a hyp_hi) a in
  let r, rc, k = exp_reduce a in
  let em_lo = poly_q r (mulacc rc r rc) in
  let wh, wl = two_sum (one a) r in
  let cap = int k (pick a hyp_kcap) in
  let m = pow2 a (Ops.mul (where (gt k cap) cap k) (int k (-2))) in
  ((r, em_lo), (wh, wl +: em_lo), m, k)

(* nx_sinh: 2^(k-1) (em + (em + 1 - 2^-2k) / w) *)
let sinh x =
  let (eh, el), w, m, k = hyp_reduce (magnitude x) in
  let nh, nl = two_sum (one x -: m) eh in
  let qh, ql = div_pair (nh, nl +: el) w in
  let vh, vl = two_sum eh qh in
  let y = vh +: (vl +: (el +: ql)) in
  nan1 x (copysign (scale y (Ops.sub k (int k 1))) x)

(* nx_cosh: 2^(k-1) (w + 2^-2k / w) *)
let cosh x =
  let _, (wh, wl), m, k = hyp_reduce (magnitude x) in
  let qh, ql = div_pair (m, zero x) (wh, wl) in
  let vh, vl = two_sum wh qh in
  let y = vh +: (vl +: (wl +: ql)) in
  nan1 x (scale y (Ops.sub k (int k 1)))

(* nx_tanh: below the split a + a^3 T(a^2); above, 1 - 2 / (w + 1) with
   w = e^2a, the quotient and the subtraction's roundings kept as tails; x's
   sign bit set on either. *)
let tanh x =
  let a = magnitude x in
  let z = a *: a in
  let small = mulacc (a *: z) (horner z (pick x tanh_t)) a in
  let hi = con x tanh_hi in
  let w = exp_near (two x *: where (lt a hi) a hi) in
  let dh, dl = fast_two_sum w (one x) in
  let qh = two x /: dh in
  let ql = (mulacc (negated qh) dh (two x) -: (qh *: dl)) *: (half x *: qh) in
  let s = one x -: qh in
  let y =
    where (lt a (con x tanh_small)) small (s +: (one x -: s -: qh -: ql))
  in
  let y = of_bits (bor (bits y) (band (bits x) (sign_word x))) in
  nan1 x y

(* Trigonometric kinds

   x = q pi/2 + hi + lo, q a uint32, by nx_kinds.h's reductions: Cody and
   Waite's below the switch, Payne and Hanek's from it, which the C branches
   to and the selection [rem] takes. Constants are nx_kinds.h's macros of the
   same names, at float32 then float64, their hexadecimal text copied. *)

let two_over_pi = (0x1.45f306p-1, 0x1.45f306dc9c883p-1)
let pio2_3 = (-0x1.ee59dap-50, -0x1.f1976b7ed8fbcp-110)
let pio2_big = (0x1p+15, 0x1p+28)
let trig_tiny = (0x1p-12, 0x1p-27)

(* nx_sincos_p_f32 and nx_sincos_p_f64, by cosine: S then C, each from its
   leading coefficient. *)
let sin_s =
  ( [ -0x1.9952fap-13; 0x1.110776p-7; -0x1.555546p-3 ],
    [
      0x1.5d8ee79c0623bp-33;
      -0x1.ae5e57812c7p-26;
      0x1.71de356408ad5p-19;
      -0x1.a01a019bf9b6p-13;
      0x1.111111110f7bp-7;
      -0x1.5555555555548p-3;
    ] )

let cos_c =
  ( [ 0x1.99e80cp-16; -0x1.6c0c28p-10; 0x1.55554ap-5 ],
    [
      -0x1.8fa3958a5c52ap-37;
      0x1.1ee9d5c91b1dfp-29;
      -0x1.27e4f7e80273p-22;
      0x1.a01a019c80966p-16;
      -0x1.6c16c16c14f6cp-10;
      0x1.555555555554bp-5;
    ] )

(* nx_rem_pio2_cw_f32 and nx_rem_pio2_cw_f64, q the uint32 of nx_nearest's
   int. *)
let rem_cw x =
  let qf, q = nearest (mulacc x (con x two_over_pi) (shifter x)) in
  let q = Ops.cast q Uint32 in
  let mq = negated qf in
  let a = mulacc mq (con x pio2_hi) x in
  if not (wide x) then
    let b = mulacc mq (con x pio2_lo) a in
    let hi = mulacc mq (con x pio2_3) b in
    (hi, mulacc mq (con x pio2_3) (b -: hi), q)
  else
    let p = qf *: con x pio2_lo in
    let pe = mulacc qf (con x pio2_lo) (negated p) in
    let bh, bl = two_sum a (negated p) in
    let lo = mulacc mq (con x pio2_3) (bl -: pe) in
    let hi, lo = fast_two_sum bh lo in
    (hi, lo, q)

(* nx_two_over_pi: the bits of 2/pi after the point, behind one zero word. *)
let two_over_pi_words =
  [|
    0x00000000L; 0xA2F9836EL; 0x4E441529L; 0xFC2757D1L; 0xF534DDC0L;
    0xDB629599L; 0x3C439041L; 0xFE5163ABL; 0xDEBBC561L; 0xB7246E3AL;
    0x424DD2E0L; 0x06492EEAL; 0x09D1921CL; 0xFE1DEB1CL; 0xB129A73EL;
    0xE88235F5L; 0x2EBB4484L; 0xE99C7026L; 0xB45F7E41L; 0x3991D639L;
    0x835339F4L; 0x9C845F8BL; 0xBDF9283BL; 0x1FF897FFL; 0xDE05980FL;
    0xEF2F118BL; 0x5A0A6D1FL; 0x6D367ECFL; 0x27CB09B7L; 0x4F463F66L;
    0x9E5FEA2DL; 0x7527BAC7L; 0xEBE5F17BL; 0x3D0739F7L; 0x8A5292EAL;
    0x6BFB5FB1L; 0x1F8D5D08L; 0x56033046L; 0xFC7B6BABL; 0xF0CFBC20L;
    0x9AF4361DL;
  |]

(* nx_bits_at for [n] runs of 32 bits of 2/pi from bits [g], [g + 32] and so
   on, as uint64s: words [w] to [w + n], [w = g >> 5] of [nb] bits, are moved
   into place by one selection per bit of [w], as a barrel shifter does, since
   a node cannot index a table. *)
let window g n nb =
  let t = two_over_pi_words in
  let w = shr g 5 in
  let sh = Ops.cast (band g (int g 31)) Uint64 in
  let at m = u64 g (if m < Array.length t then t.(m) else 0L) in
  let words = ref (Array.init (n + (1 lsl nb)) at) in
  for b = nb - 1 downto 0 do
    let step = 1 lsl b and prev = !words in
    let on = set (band w (int w step)) in
    words :=
      Array.init
        (Array.length prev - step)
        (fun m -> where on prev.(m + step) prev.(m))
  done;
  let ws = !words in
  Array.init n (fun j ->
      let pair = bor (shl ws.(j) 32) ws.(j + 1) in
      band (shr (Ops.shl pair sh) 32) (u64 g 0xFFFFFFFFL))

(* x's reduction negated where x's sign bit is set, q as a uint32. *)
let signed_rem x (hi, lo) q =
  let n = negative x in
  ( where n (negated hi) hi,
    where n (negated lo) lo,
    where n (Ops.sub (u32 q 0) q) q )

(* |x|'s bits, held at the switch's on the lanes below it, where the
   reduction past the switch is computed but not taken: its exponent then
   reads within the table. *)
let big_bits x =
  let ix = band (bits x) (mag_word x) in
  let big = bits_of x (pick x pio2_big) in
  where (lt ix big) big ix

(* nx_rem_pio2_big_f32, of nx_two_over_pi *)
let rem_big32 x =
  let ix = big_bits x in
  let e = Ops.cast (shr ix 23) Int32 in
  let e = Ops.sub e (int e 150) in
  let m =
    Ops.cast (bor (band ix (word ix 0x007FFFFFL)) (word ix 0x00800000L)) Uint64
  in
  let w = window (Ops.add e (int e 30)) 3 3 in
  let a0 = m *: w.(2) and a1 = m *: w.(1) and a2 = m *: w.(0) in
  let low u = band u (u64 u 0xFFFFFFFFL) in
  let t1 = Ops.add (shr a0 32) a1 in
  let t2 = Ops.add (shr t1 32) a2 in
  let f = bor (bor (shl t2 34) (shl (low t1) 2)) (shr (low a0) 30) in
  let top = shr f 63 in
  let q = Ops.add (Ops.cast (shr t2 30) Uint32) (Ops.cast top Uint32) in
  (* the signed fraction in three exact pieces, its value f 2^-64 *)
  let part u s = Ops.cast u Float32 *: num x s in
  let c2 = part (shr f 40) 0x1p-24 in
  let c1 = part (band (shr f 16) (u64 f 0xFFFFFFL)) 0x1p-48 in
  let c0 = part (band f (u64 f 0xFFFFL)) 0x1p-64 in
  let c2 = where (set top) (c2 -: one x) c2 in
  let vh, vl = two_sum c2 c1 in
  let vlo = vl +: c0 in
  (* times pi/2 *)
  let ph = vh *: con x pio2_hi in
  let pl = mulacc vh (con x pio2_hi) (negated ph) in
  let pl = mulacc vh (con x pio2_lo) (mulacc vlo (con x pio2_hi) pl) in
  let hi = ph +: pl in
  signed_rem x (hi, pl -: (hi -: ph)) q

(* nx_rem_pio2_big_f64, of nx_two_over_pi *)
let rem_big64 x =
  let ix = big_bits x in
  let e = Ops.cast (shr ix 52) Int32 in
  let e = Ops.sub e (int e 1075) in
  let m =
    bor (band ix (word ix 0x000FFFFFFFFFFFFFL)) (word ix 0x0010000000000000L)
  in
  let win = window (Ops.add e (int e 30)) 6 5 in
  let w = Array.init 6 (fun j -> win.(5 - j)) in
  let low u = band u (u64 u 0xFFFFFFFFL) in
  let ml = low m and mh = shr m 32 in
  let p = Array.make 8 (u64 m 0L) in
  let c = ref (u64 m 0L) in
  for j = 0 to 5 do
    let t = Ops.add (ml *: w.(j)) !c in
    p.(j) <- low t;
    c := shr t 32
  done;
  p.(6) <- !c;
  c := u64 m 0L;
  for j = 0 to 5 do
    let t = Ops.add (Ops.add (mh *: w.(j)) p.(j + 1)) !c in
    p.(j + 1) <- low t;
    c := shr t 32
  done;
  p.(7) <- !c;
  let mask30 u = band u (u64 u 0x3FFFFFFFL) in
  let fh = bor (bor (shl (mask30 p.(5)) 34) (shl p.(4) 2)) (shr p.(3) 30) in
  let fl = bor (bor (shl (mask30 p.(3)) 34) (shl p.(2) 2)) (shr p.(1) 30) in
  let q =
    Ops.add (Ops.cast (shr p.(5) 30) Uint32) (Ops.cast (shr fh 63) Uint32)
  in
  (* the signed fraction (fh, fl) 2^-128 in three exact pieces *)
  let c2 = Ops.cast (Ops.cast (shr fh 32) Int32) Float64 *: num x 0x1p-32 in
  let c1 = Ops.cast (low fh) Float64 *: num x 0x1p-64 in
  let c0 = Ops.cast (shr fl 11) Float64 *: num x 0x1p-117 in
  let sh, sl = two_sum c2 c1 in
  let vh, vl = fast_two_sum sh (sl +: c0) in
  let ph = vh *: con x pio2_hi in
  let pl = mulacc vh (con x pio2_hi) (negated ph) in
  let pl = mulacc vh (con x pio2_lo) (mulacc vl (con x pio2_hi) pl) in
  signed_rem x (fast_two_sum ph pl) q

(* nx_trig_big: |x| finite and at or past the switch. *)
let trig_big x =
  let a = band (bits x) (mag_word x) in
  Ops.logical_not (lt a (bits_of x (pick x pio2_big)))
  &&: lt a (word a (pick x inf_bits))

(* The C's branch in nx_sin, nx_cos and nx_tan. *)
let rem x =
  let big = trig_big x in
  let bh, bl, bq = if wide x then rem_big64 x else rem_big32 x in
  let nh, nl, nq = rem_cw x in
  (where big bh nh, where big bl nl, where big bq nq)

(* nx_sin_kernel *)
let sin_kernel hi lo =
  let z = hi *: hi in
  (hi, mulacc (hi *: z) (horner z (pick hi sin_s)) lo)

(* nx_cos_kernel *)
let cos_kernel hi lo =
  let z = hi *: hi in
  let zl = mulacc hi hi (negated z) in
  let hz = half hi *: z in
  let ch = one hi -: hz in
  let tail =
    mulacc (z *: z)
      (horner z (pick hi cos_c))
      (mulacc (negated hi) lo (num hi (-0.5) *: zl))
  in
  (ch, one hi -: ch -: hz +: tail)

(* nx_flip2: y with its sign flipped where bit 1 of k is set. *)
let flip2 y k =
  let b = Ops.cast (band k (u32 k 2)) (unsigned (Ops.dtype y)) in
  of_bits (Ops.bitwise_xor (bits y) (shl b (width y - 2)))

let odd q = set (band q (u32 q 1))

(* Where x is a NaN or below the cutoff, as !(|x| >= tiny). *)
let tiny x = Ops.logical_not (ge (magnitude x) (con x trig_tiny))

(* nx_sin_at *)
let sin x =
  let hi, lo, q = rem x in
  let sh, sl = sin_kernel hi lo and ch, cl = cos_kernel hi lo in
  let y = flip2 (where (odd q) (ch +: cl) (sh +: sl)) q in
  where (tiny x) x y

(* nx_cos_at *)
let cos x =
  let hi, lo, q = rem x in
  let sh, sl = sin_kernel hi lo and ch, cl = cos_kernel hi lo in
  let y = flip2 (where (odd q) (sh +: sl) (ch +: cl)) (Ops.add q (u32 q 1)) in
  where (is_nan x) x y

(* nx_tan_at: -cos / sin in odd quadrants, sin / cos in even. *)
let tan x =
  let hi, lo, q = rem x in
  let sh, sl = sin_kernel hi lo and ch, cl = cos_kernel hi lo in
  let o = odd q in
  let n = (where o (negated ch) sh, where o (negated cl) sl) in
  let d = (where o sh ch, where o sl cl) in
  let qh, ql = div_pair n d in
  where (tiny x) x (qh +: ql)

(* Inverse trigonometric kinds

   nx_kinds_real.h's asin and acos, and nx_kinds.h's atan and atan2, which
   differ between the types. Constants are nx_kinds.h's macros and
   coefficients of the same names, at float32 then float64, their text
   copied. *)

(* nx_asin_b_f32 and nx_asin_b_f64 *)
let asin_b =
  ( [
      0x1.595c8cp-5;
      0x1.8c3e2ap-6;
      0x1.747bbap-5;
      0x1.330204p-4;
      0x1.5555c8p-3;
    ],
    [
      0x1.056cbe7839745p-5;
      -0x1.09d10e2ce02f9p-6;
      0x1.3ff33c58c8071p-6;
      0x1.abd19915948edp-8;
      0x1.8ec2b52f17c83p-7;
      0x1.c6fdcd892b624p-7;
      0x1.1c6be0fccac8fp-6;
      0x1.6e89f44cb6e5ap-6;
      0x1.f1c72c38610a7p-6;
      0x1.6db6db427c841p-5;
      0x1.333333336e7cbp-4;
      0x1.555555555539p-3;
    ] )

(* The polynomials of nx_atan_kernel_f32 and nx_atan_kernel_f64. *)
let atan_a =
  ( [
      0x1.7ed21ep-9;
      -0x1.0c2c0cp-6;
      0x1.61fdd4p-5;
      -0x1.3556b4p-4;
      0x1.b4e12ap-4;
      -0x1.230adcp-3;
      0x1.9978f4p-3;
      -0x1.5554dcp-2;
    ],
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
    ] )

(* atan 1/2, pi/4 and atan 3/2 at float64, in two parts. *)
let atan_1_2_hi = 0x1.dac670561bb4fp-2
let atan_1_2_lo = 0x1.a2b7f222f65e2p-56
let pio4_hi = 0x1.921fb54442d18p-1
let pio4_lo = 0x1.1a62633145c07p-55
let atan_3_2_hi = 0x1.f730bd281f69bp-1
let atan_3_2_lo = 0x1.007887af0cbbdp-56

(* nx_nan2: r, or the first of a and b that is a NaN. *)
let nan2 a b r = where (is_nan a) a (nan1 b r)

(* nx_asin_reduce: (s, tail, big), asin |x| = s + tail at or below 1/2, else
   s = sqrt((1 - |x|)/2), whose asin the caller doubles. *)
let asin_reduce a =
  let big = gt a (half a) in
  let z = where big ((one a -: a) *: half a) (a *: a) in
  let s = where big (Ops.sqrt z) a in
  let c = mulacc (negated s) s z /: (s +: s) in
  let c = where (big &&: gt z (zero a)) c (zero a) in
  (s, mulacc (s *: z) (horner z (pick a asin_b)) c, big)

(* nx_asin: pi/2 - 2 asin s past 1/2. *)
let asin x =
  let a = magnitude x in
  let s, tail, big = asin_reduce a in
  let d = s +: s in
  let hi = con x pio2_hi -: d in
  let lo =
    con x pio2_hi -: hi -: d +: mulacc (num x (-2.)) tail (con x pio2_lo)
  in
  let y = where big (hi +: lo) (s +: tail) in
  let y = where (gt a (one x)) (nan x) y in
  nan1 x (copysign y x)

(* nx_acos: pi/2 - asin x within 1/2, 2 asin s above, pi - 2 asin s below,
   as a pair. *)
let acos x =
  let a = magnitude x in
  let s, tail, big = asin_reduce a in
  let n = negative x in
  let term = where big (s +: s) s in
  let tail = where big (tail +: tail) tail in
  let base_hi = where big (where n (con x pi_hi) (zero x)) (con x pio2_hi) in
  let base_lo = where big (where n (con x pi_lo) (zero x)) (con x pio2_lo) in
  let plus = ne big n in
  let term = where plus term (negated term) in
  let tail = where plus tail (negated tail) in
  let hh, hl = two_sum base_hi term in
  let y = hh +: (hl +: (base_lo +: tail)) in
  nan1 x (where (gt a (one x)) (nan x) y)

(* nx_atan_kernel_f32: t + t^3 A(t^2) on |t| <= 1. *)
let atan_k32 t =
  let z = t *: t in
  mulacc (t *: z) (horner z (fst atan_a)) t

(* nx_atan_kernel_f64: t^3 A(t^2) on |t| <= 7/16, the caller adding t. *)
let atan_k64 t =
  let z = t *: t in
  t *: z *: horner z (snd atan_a)

(* nx_atan_pos_f64: atan a, a >= 0, as a pair, a = c + (a - c) / (1 + c a),
   c the nearest of 0, 1/2, 1, 3/2 and inf; the quotient's error and the
   argument's tail al kept in tl. *)
let atan_pos a al =
  let n v = num a v in
  let b1 = ge a (n 0.4375) and b2 = ge a (n 0.6875) in
  let b3 = ge a (n 1.1875) and b4 = ge a (n 2.4375) in
  let c = where b3 (n 1.5) (where b2 (n 1.) (n 0.5)) in
  let num_ = where b4 (n (-1.)) (a -: c) in
  let den = where b4 a (mulacc c a (n 1.)) in
  let rden = n 1. /: den in
  let t = where b1 (num_ *: rden) a in
  let rest = mulacc (negated t) den num_ in
  let tl = where b1 (mulacc al (mulacc (negated t) c (n 1.)) rest *: rden) al in
  let tl = where b4 (rest *: rden) tl in
  let tl = where (lt a (inf a)) tl (n 0.) in
  let pick4 v4 v3 v2 v1 =
    where b4 (n v4) (where b3 (n v3) (where b2 (n v2) (where b1 (n v1) (n 0.))))
  in
  let ch = pick4 (snd pio2_hi) atan_3_2_hi pio4_hi atan_1_2_hi in
  let cl = pick4 (snd pio2_lo) atan_3_2_lo pio4_lo atan_1_2_lo in
  let p = atan_k64 t in
  let z = t *: t in
  let w = mulacc z (z -: n 1.) (n 1.) in
  let hh, hl = two_sum ch t in
  fast_two_sum hh (hl +: (cl +: mulacc tl w p))

(* nx_atan_f32 and nx_atan_f64 *)
let atan x =
  let a = magnitude x in
  if wide x then nan1 x (copysign (fst (atan_pos a (zero x))) x)
  else
    let big = gt a (one x) in
    let p = atan_k32 (where big (one x /: a) a) in
    let y = where big (con x pio2_hi +: (con x pio2_lo -: p)) p in
    nan1 x (copysign y x)

(* nx_atan2_f32 and nx_atan2_f64: the angle of (|x|, |y|) from atan of the
   smaller magnitude over the larger; by quadrant, added to or subtracted
   from 0, pi/2 or pi as a pair, then given y's sign. *)
let atan2 y x =
  let ax = magnitude x and ay = magnitude y in
  let swap = gt ay ax in
  let num_ = where swap ax ay and den = where swap ay ax in
  let both_inf = eq ax (inf x) &&: eq ay (inf x) in
  let xneg = negative x in
  let base p2 p = where swap (con x p2) (where xneg (con x p) (zero x)) in
  let base_hi = base pio2_hi pi_hi and base_lo = base pio2_lo pi_lo in
  let minus = ne swap xneg in
  let v =
    if not (wide x) then
      let t = num_ /: den in
      let t = where (eq den (zero x)) (zero x) t in
      let t = where both_inf (one x) t in
      let p = atan_k32 t in
      let p = where minus (negated p) p in
      let hh, hl = two_sum base_hi p in
      hh +: (hl +: base_lo)
    else
      (* 1/den overflows below 2^-1024: scaled, num/den is unchanged *)
      let s = where (lt den (num x 0x1p-960)) (num x 0x1p64) (one x) in
      let num_ = num_ *: s and den = den *: s in
      let rden = one x /: den in
      let t = num_ *: rden in
      let tl = mulacc (negated t) den num_ *: rden in
      let degenerate = eq den (zero x) ||: both_inf ||: eq den (inf x) in
      let t = where (eq den (zero x)) (zero x) t in
      let t = where both_inf (one x) t in
      let tl = where degenerate (zero x) tl in
      let ph, pl = atan_pos t tl in
      let ph = where minus (negated ph) ph
      and pl = where minus (negated pl) pl in
      let hh, hl = two_sum base_hi ph in
      hh +: (hl +: (base_lo +: pl))
  in
  nan2 y x (copysign v y)

(* log2 a as hi + lo for a finite a > 0, |lo| within half an ulp of hi, to a
   relative 2^-12 u. With a = m 2^e, m in (sqrt3 / 2, sqrt3] and b = 1.5 from
   sqrt1_5 on, else 1, log2 a is e + log2 b + log2 (m / b), and log2 (m / b) is
   2 log2e s A(s^2) for s = (m - b) / (m + b) and A(z) = atanh(s) / s =
   1 + z / 3 + z^2 R(z). m - b is exact, |s| at most 0.10103 and z at most
   0.0103. s, z, z / 3, A and the products by s and log2e are carried in two
   parts by fmas.
   z^2 R(z), at most 2^-15.5 of A, is the one term rounded in the working
   precision: its relative 4u, R a minimax fit within 0.28u at float32 and 0.5u
   at float64, gives 2^-13.5 u, and the low parts' roundings at most 2^-12.9 u.
   Drawn operands, subnormals included, measure 2^-13.4 u. *)

let sqrt3 = (0x1.bb67aep+0, 0x1.bb67ae8584caap+0)
let sqrt1_5 = (0x1.3988e2p+0, 0x1.3988e1409212ep+0)
let log2_1_5 = (0x1.2b8034p-1, 0x1.2b803473f7ad1p-1)
let log2_1_5_lo = (0x1.cfdeb4p-27, -0x1.817fd3b7d7e5dp-58)
let third = (0x1.555556p-2, 0x1.5555555555555p-2)
let third_lo = (-0x1.555556p-27, 0x1.5555555555555p-56)

let log2_poly =
  ( [ 0x1.ccebf6p-4; 0x1.248f68p-3; 0x1.99999ap-3 ],
    [
      0x1.f1bd4d19e411fp-5;
      0x1.10f421d252391p-4;
      0x1.3b13e56a89c64p-4;
      0x1.745d171591ad7p-4;
      0x1.c71c71c731ac1p-4;
      0x1.2492492492476p-3;
      0x1.999999999999ap-3;
    ] )

(* hi + lo times the float c + c_lo, in two parts. *)
let mul2 (hi, lo) c c_lo =
  let p = hi *: c in
  (p, mulacc hi c (negated p) +: ((hi *: c_lo) +: (lo *: c)))

let log2_parts a =
  let m, e = normalized ~top:sqrt3 a in
  let upper = ge m (con a sqrt1_5) in
  let b = where upper (num a 1.5) (one a) in
  (* m + b in two parts: b's exponent is at least m's *)
  let d = b +: m in
  let d_lo = m -: (d -: b) in
  let f = m -: b in
  let s = f /: d in
  let s_lo = (mulacc (negated s) d f -: (s *: d_lo)) /: d in
  let z = s *: s in
  let z_lo = mulacc s s (negated z) +: (two a *: s *: s_lo) in
  let t, t_lo = mul2 (z, z_lo) (con a third) (con a third_lo) in
  let zc = z +: z_lo in
  let w = zc *: zc *: horner zc (pick a log2_poly) in
  let p = one a +: t in
  let p_lo = t -: (p -: one a) +: (t_lo +: w) in
  let q, q_lo = mul2 (two a *: s, two a *: s_lo) p p_lo in
  let l, l_lo = mul2 (q, q_lo) (con a log2e) (con a log2e_lo) in
  let c = where upper (con a log2_1_5) (zero a) in
  let c_lo = where upper (con a log2_1_5_lo) (zero a) in
  let g, g_lo = two_sum c l in
  let hi, hi_lo = two_sum e g in
  (* lo carries z^2 R(z) at up to 2^-15.5 of hi: |lo| is brought within half
     an ulp of hi, so that h's low part is within 1.5 ulps of h *)
  let lo = hi_lo +: (g_lo +: (c_lo +: l_lo)) in
  let r = hi +: lo in
  (r, lo -: (r -: hi))

(* pow: 2^(y log2|x|), log2|x| in two parts and their product by y carried in
   two by an fma, with C99 F.9.4.4's special values. The parts' relative 2^-12 u
   moves h = y log2|x| by 2^-12 u |h|, which 2^h turns into a relative
   2^-12.5 |h| u: at most 0.18 ulp, since past |h| = 1075 the result overflows
   or rounds to 0. With exp2_split's 8.5 ulps, as exp's, pow is within 9. *)
let pow x y =
  let a = magnitude x in
  let finite = gt a (zero x) &&: lt a (inf x) in
  let l, l_lo = log2_parts (where finite a (one x)) in
  let special = where (eq a (zero x)) (negated (inf x)) a in
  let l = where finite l special in
  let h = y *: l in
  let h_lo = mulacc y l (negated h) +: where finite (y *: l_lo) (zero x) in
  let v = exp2_split h h_lo in
  let integer v = eq (trunc v) v in
  let odd = integer y &&: Ops.logical_not (integer (y *: half y)) in
  let v = where (odd &&: negative x) (negated v) v in
  let below = lt x (zero x) &&: gt x (negated (inf x)) in
  let v = where (below &&: Ops.logical_not (integer y)) (nan x) v in
  let v = where (eq x (num x (-1.)) &&: eq (magnitude y) (inf x)) (one x) v in
  where (eq x (one x) ||: eq y (zero x)) (one x) v

(* erf

   nx_kinds.h's: a polynomial per range; at float64, past 2.5, 1 -
   e^(G(a) - a^2). Coefficients are the header's, their text copied. *)

(* nx_erf_f32 *)
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
  nan1 x (copysign (where (lt a (one x)) small (one x -: q)) x)

(* nx_erf_near_f64 *)
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
  nan1 x (copysign (where (lt a (one x)) small (one x -: q)) x)

(* nx_erf_far_f64: past 2.5, a taken as 5.95 past it, where erf rounds to
   1. *)
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
  copysign (one x -: exp_near (g -: (b *: b))) x

(* nx_erf_f32, and nx_erf_f64's branch at nx_erf_big_f64 *)
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
  | Exp2 -> exp2 x
  | Log -> log x
  | Log2 -> log2 x
  | Log1p -> log1p x
  | Expm1 -> expm1 x
  | Sin -> sin x
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
