(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Each kind is nx's C ([NX_COMPLEX_KINDS] of [nx_kinds.h], and
   [nx_kinds_complex.h]) written over pairs of nodes, as Lower_real writes the
   float kinds: the same operations in the same order, an fma where the C writes
   one. A C branch is a selection of every branch's value, the first branch
   whose condition holds winning, as its early return does. *)

open Tolk
open Lower_bits
module R = Lower_real

(* Operations *)

let ( +: ) = Ops.add
let ( -: ) = Ops.sub
let ( *: ) = Ops.mul
let ( /: ) x y = Ops.alu x Op.Fdiv [ y ]
let ( ||: ) = Ops.bitwise_or
let ( &&: ) = Ops.bitwise_and
let where = Ops.where
let not_ = Ops.logical_not
let lt = Ops.lt
let gt x y = Ops.lt y x
let eq = Ops.eq

(* [x <= y] and [x >= y], false where an operand is NaN. *)
let le x y = lt x y ||: eq x y
let ge x y = le y x

(* Constants *)

let num x v = Shape.const_like x (`Float v)
let int k n = Shape.const_like k (`Int (Bigint.of_int n))
let zero x = num x 0.
let one x = num x 1.
let half x = num x 0.5
let inf x = num x Float.infinity
let nans x = (nan x, nan x)

(* ln2 rounded and its remainder; exp's near range; clamps past which e^x c and
   2^x c overflow for every finite non-zero c; where sinh and cosh are e^|x|/2,
   asin and acos their terms at infinity, and atanh is past its overflow
   guard. *)
let ln2 = (0x1.62e430p-1, 0x1.62e42fefa39efp-1)
let ln2_low = (-0x1.05c610p-29, 0x1.abc9e3b39803fp-56)
let exp_near = (86., 708.)
let cexp_hi = (194., 1456.)
let cexp2_hi = (280., 2101.)
let hyp_big = (12., 22.)
let asin_big = (0x1p13, 0x1p28)
let atanh_big = (0x1p62, 0x1p510)

(* Branches *)

(* The value of the first case whose condition holds, else [d]. *)
let rec cases l d =
  match l with
  | [] -> d
  | (c, (re, im)) :: rest ->
      let dr, di = cases rest d in
      (where c re dr, where c im di)

(* Float kinds *)

let log = R.unary Log
let log2 = R.unary Log2
let log1p = R.unary Log1p
let exp = R.unary Exp
let exp2 = R.unary Exp2
let sin = R.unary Sin
let cos = R.unary Cos
let tan = R.unary Tan
let asin = R.unary Asin
let acos = R.unary Acos
let atan = R.unary Atan
let sinh = R.unary Sinh
let cosh = R.unary Cosh
let tanh = R.unary Tanh
let atan2 = R.binary Atan2
let pow = R.binary Pow

(* Parts *)

(* v with its sign flipped where s's sign bit is set. *)
let flip v s = where (negative s) (negated v) v
let finite x = lt (magnitude x) (inf x)
let larger a b = where (gt a b) a b
let neg_int k = Ops.sub (int k 0) k

(* The exponent e of a finite x > 0, x = m 2^e with m in [1, 2). *)
let ilogb x =
  let word n = Shape.const_like (bits x) (`Int (Bigint.of_int64_unsigned n)) in
  let sub = lt (bits x) (word (R.pick x R.min_normal_bits)) in
  let s = where sub (x *: R.con x R.sub_scale) x in
  let mbits = R.pick x R.mbits in
  let e = Ops.cast (Ops.shr (bits s) (int (bits s) mbits)) Int32 in
  let e = Ops.sub e (int e (R.pick x R.bias)) in
  Ops.sub e (where sub (int e mbits) (int e 0))

(* y 2^k rounded once, for k up to twice the span scale takes. *)
let ldexp y k =
  let k1 = Ops.shr k (int k 1) in
  R.scale (R.scale y k1) (Ops.sub k k1)

(* a b - c d within 1.5 ulps: c d's rounding error, which an fma gives exactly,
   is added back (Kahan's algorithm). *)
let det a b c d =
  let w = c *: d in
  let e = mulacc (negated c) d w in
  mulacc a b (negated w) +: e

(* sqrt(x^2 + y^2) of finite x and y, scaled by a power of two so that the
   larger part is in [1, 2). *)
let hypot x y =
  let ax = magnitude x and ay = magnitude y in
  let a = larger ax ay and b = where (gt ax ay) ay ax in
  let k = ilogb a in
  let sa = R.scale a (neg_int k) and sb = R.scale b (neg_int k) in
  let h = R.scale (Ops.sqrt (mulacc sa sa (sb *: sb))) k in
  where (eq a (zero a)) a h

(* log |x + iy|: of the larger part a and the smaller b, log a where b is 0;
   half log1p of a^2 + b^2 - 1 where a is near 1, from exact products; else log
   a + log1p((b/a)^2) / 2. *)
let log_abs x y =
  let ax = magnitude x and ay = magnitude y in
  let a = larger ax ay and b = where (gt ax ay) ay ax in
  let am = a -: one a in
  let aph, apl = R.two_sum a (one a) in
  let h1 = am *: aph and h2 = b *: b in
  let l1 = mulacc am aph (negated h1) and l2 = mulacc b b (negated h2) in
  let hh, hl = R.two_sum h1 h2 in
  let d = hh +: (hl +: mulacc am apl (l1 +: l2)) in
  let near = half a *: log1p d in
  let q = b /: a in
  let far = mulacc (half a) (log1p (q *: q)) (log a) in
  let v = where (ge a (half a) &&: le a (num a 2.)) near far in
  let v = where (eq b (zero b)) (log a) v in
  let v = where (is_nan ax ||: is_nan ay) (nan x) v in
  where (eq ax (inf x) ||: eq ay (inf x)) (inf x) v

(* asinh t, odd: log1p(a + a^2 / (1 + sqrt(1 + a^2))) of a = |t|, and log a +
   ln2 past asin_big, where a^2 + 1 rounds to a^2. *)
let asinh_part t =
  let a = magnitude t in
  let a2 = a *: a in
  let near = log1p (a +: (a2 /: (one a +: Ops.sqrt (one a +: a2)))) in
  let far = log a +: R.con a R.ln2_hi +: R.con a R.ln2_lo in
  R.copysign (where (lt a (R.con a asin_big)) near far) t

(* e^x c, for x past exp_near up to the clamp cexp_hi: with x/2 = k ln2 + r and
   m = e^r, (m c) m 2^2k, so that neither e^x nor its product with a small c
   overflows first. [shift] halves the result [shift] times. *)
let exp_times x c shift =
  let h = where (lt x (R.con x cexp_hi)) x (R.con x cexp_hi) *: half x in
  let r, _, k = R.exp_reduce h in
  let m = one x +: mulacc (r *: r) (R.exp_q r) r in
  ldexp (m *: c *: m) (Ops.sub (Ops.mul k (int k 2)) (int k shift))

(* Exponentials *)

(* e^z = e^x cis y. *)
let exp_raw x y =
  let c = cos y and s = sin y in
  let e = exp x in
  let m = where (gt x (zero x)) x (zero x) in
  let unbounded =
    cases
      [
        (eq x (inf x), (x, nan x));
        (eq x (negated (inf x)), (zero x, R.copysign (zero y) y));
      ]
      (nans x)
  in
  cases
    [
      (eq y (zero y), (e, y));
      (is_nan x, nans x);
      (not_ (finite y), unbounded);
      (eq (magnitude x) (inf x), (m *: c, m *: s));
      (le x (R.con x exp_near), (e *: c, e *: s));
    ]
    (exp_times x c 0, exp_times x s 0)

(* 2^z = 2^x cis(y ln2), 2^x as 2^k m. Below the trigonometric kinds' switch to
   Payne and Hanek, the phase y ln2 is p + t, ln2 in two parts and t the
   rounding error of p, and cis p turns by cis t. Past it, y ln2 is reduced by
   pi/2 from the bits of 2 ln2 / pi. *)
let exp2_raw x y =
  let p = y *: R.con y ln2 in
  let r = R.rem_big Two_ln2_over_pi y in
  let t = mulacc y (R.con y ln2_low) (mulacc y (R.con y ln2) (negated p)) in
  let cp = cos p and sp = sin p and ct = cos t and st = sin t in
  let big = R.trig_big p in
  let c = where big (R.cos_at p r) (det cp ct sp st) in
  let s = where big (R.sin_at p r) (det sp ct (negated cp) st) in
  let m = where (gt x (zero x)) x (zero x) in
  let v = where (lt x (R.con x cexp2_hi)) x (R.con x cexp2_hi) in
  let v = where (gt v (R.con x R.exp2_lo)) v (R.con x R.exp2_lo) in
  let kf, k = R.nearest (v +: R.con x R.shifter) in
  let e = R.exp2_poly (v -: kf) in
  cases
    [
      (eq y (zero y), (exp2 x, y));
      (is_nan x ||: not_ (finite y), exp_raw x y);
      (not_ (finite x), (m *: c, m *: s));
    ]
    (ldexp (e *: c) k, ldexp (e *: s) k)

(* log z = log |z| + i arg z. *)
let log_raw x y =
  let re = log_abs x y in
  let re = where (eq x (zero x) &&: finite y) (log (magnitude y)) re in
  (re, atan2 y x)

(* log2 z = log z / ln2, 1/ln2 as a head and a tail; on the axes, the part
   type's log2 of the modulus. *)
let log2_raw x y =
  let lr, li = log_raw x y in
  let by_ln2 v = mulacc v (R.con v R.log2e) (v *: R.con v R.log2e_lo) in
  let ax = magnitude x and ay = magnitude y in
  let re = by_ln2 lr in
  let re = where (eq y (zero y) &&: eq ax ax) (log2 ax) re in
  let re = where (eq x (zero x) &&: finite y) (log2 ay) re in
  (re, by_ln2 li)

(* Square root (Kahan, 1987): t = sqrt((|x| + |z|) / 2) is the part of the
   larger magnitude, y / 2t the other; z is scaled by an even power of two so
   that its larger part is in [1, 4). *)
let sqrt_raw x y =
  let ax = magnitude x and ay = magnitude y in
  let e = ilogb (larger ax ay) in
  let j = Ops.shr e (int e 1) in
  let m2j = Ops.mul j (int j (-2)) in
  let sx = R.scale ax m2j and sy = R.scale ay m2j in
  let t = R.scale (Ops.sqrt ((sx +: hypot sx sy) *: half x)) j in
  let u = ay /: (t +: t) in
  let pos = ge x (zero x) in
  let r = Ops.sqrt ax in
  let y_nan = is_nan y in
  cases
    [
      (eq ay (inf y), (inf x, y));
      (is_nan x, nans x);
      (eq x (inf x), (x, where y_nan (nan x) (R.copysign (zero x) y)));
      ( eq x (negated (inf x)),
        (where y_nan (nan x) (zero x), R.copysign (inf x) y) );
      (y_nan, nans x);
      (eq y (zero y), (where pos r (zero x), where pos y (R.copysign r y)));
    ]
    (where pos t u, where pos (R.copysign u y) (R.copysign t y))

(* z^w = e^(w log z): w real or imaginary multiplies log z part by part, so that
   a zero part of w meets no infinite part of log z. 1 for w = 0, and the part
   type's pow on the positive real axis. *)
let pow_raw (zr, zi) (wr, wi) =
  let lr, li = log_raw zr zi in
  let w_real = eq wi (zero wi) and w_imag = eq wr (zero wr) in
  let pr =
    where w_real (wr *: lr)
      (where w_imag (negated (wi *: li)) (mulacc wr lr (negated (wi *: li))))
  in
  let pi =
    where w_real (wr *: li) (where w_imag (wi *: lr) (mulacc wr li (wi *: lr)))
  in
  cases
    [
      (w_imag &&: w_real, (one zr, zero zr));
      (eq zi (zero zi) &&: w_real &&: gt zr (zero zr), (pow zr wr, flip zi wr));
    ]
    (exp_raw pr pi)

(* sign z = z / |z|: z itself at 0, and at an infinity its direction among the
   eight a pair of parts names. z is scaled by a power of two first, so that |z|
   is no subnormal. *)
let sign_raw x y =
  let ax = magnitude x and ay = magnitude y in
  let x_inf = eq ax (inf x) and y_inf = eq ay (inf y) in
  let both =
    Ops.bitcast
      (Shape.const_like
         ~dtype:(unsigned (Ops.dtype x))
         x
         (`Int (Bigint.of_int64_unsigned (R.pick x R.sqrt_half_bits))))
      (Ops.dtype x)
  in
  let re = where x_inf (where y_inf both (one x)) (zero x) in
  let im = where y_inf (where x_inf both (one x)) (zero x) in
  let k = neg_int (ilogb (larger ax ay)) in
  let sx = R.scale x k and sy = R.scale y k in
  let h = hypot sx sy in
  cases
    [
      (is_nan ax ||: is_nan ay, nans x);
      (x_inf ||: y_inf, (R.copysign re x, R.copysign im y));
      (eq ax (zero x) &&: eq ay (zero y), (x, y));
    ]
    (sx /: h, sy /: h)

(* Hyperbolic functions

   sinh z = sinh x cos y + i cosh x sin y and cosh z = cosh x cos y + i sinh x
   sin y. Past hyp_big, sinh |x| and cosh |x| are e^|x| / 2 to the type's
   precision, taken as exp_times does so that a small cos y or sin y keeps the
   product finite. *)

let sinh_raw x y =
  let c = cos y and s = sin y in
  let ax = magnitude x in
  cases
    [
      (eq y (zero y), (sinh x, y));
      (is_nan x, nans x);
      ( not_ (finite y),
        cases [ (eq x (zero x) ||: not_ (finite x), (x, nan x)) ] (nans x) );
      (eq ax (inf x), (x *: c, ax *: s));
      (le ax (R.con x hyp_big), (sinh x *: c, cosh x *: s));
    ]
    (flip (exp_times ax c 1) x, exp_times ax s 1)

let cosh_raw x y =
  let c = cos y and s = sin y in
  let ax = magnitude x in
  let unbounded =
    cases
      [ (eq x (zero x), (nan x, flip x y)); (not_ (finite x), (inf x, nan x)) ]
      (nans x)
  in
  cases
    [
      (eq y (zero y), (cosh x, flip y x));
      (is_nan x, nans x);
      (not_ (finite y), unbounded);
      (eq ax (inf x), (ax *: c, x *: s));
      (le ax (R.con x hyp_big), (cosh x *: c, sinh x *: s));
    ]
    (exp_times ax c 1, flip (exp_times ax s 1) x)

(* tanh z (Kahan, 1987): with t = tan y, b = 1 + t^2, s = sinh x and c = cosh x,
   (b c s + i t) / (1 + b s^2), which cancels nowhere. Past hyp_big, 1 rounds
   the real part, and the imaginary part is 4 sin y cos y e^-2|x|. *)
let tanh_raw x y =
  let ax = magnitude x in
  let one_x = R.copysign (one x) x in
  let e = exp (negated ax) in
  let sc = sin y *: cos y in
  let t = tan y in
  let b = mulacc t t (one t) in
  let s = sinh x and c = cosh x in
  let d = mulacc (b *: s) s (one x) in
  cases
    [
      (eq y (zero y), (tanh x, y));
      (is_nan x, nans x);
      (eq ax (inf x) &&: not_ (finite y), (one_x, R.copysign (zero y) y));
      (not_ (finite y), nans x);
      (gt ax (R.con x hyp_big), (one_x, num x 4. *: sc *: e *: e));
    ]
    (b *: c *: s /: d, t /: d)

(* Inverse functions

   asin and acos (Kahan, 1987), from A = sqrt(1 - z) and B = sqrt(1 + z): asin z
   = atan(x / Re(A B)) + i asinh(Im(conj(A) B)) and acos z = 2 atan(Re A / Re B)
   + i asinh(Im(conj(B) A)). Past asin_big, asin z is atan2(x, y) + i (log|z| +
   ln2) in the first quadrant, its leading terms at infinity. On the real axis
   within [-1, 1], the part type's asin and acos. *)

let far_im ax ay = log_abs ax ay +: R.con ax R.ln2_hi +: R.con ax R.ln2_lo

let asin_raw x y =
  let ax = magnitude x and ay = magnitude y in
  let big = gt ax (R.con x asin_big) ||: gt ay (R.con x asin_big) in
  let real = eq ay (zero y) &&: le ax (one x) in
  let ar, ai = sqrt_raw (one x -: ax) (negated ay) in
  let br, bi = sqrt_raw (one x +: ax) ay in
  let re =
    where big (atan2 ax ay)
      (where real (asin ax) (atan (ax /: det ar br ai bi)))
  in
  let im =
    where big (far_im ax ay)
      (where real (zero x) (asinh_part (det ar bi ai br)))
  in
  let v = (flip re x, flip im y) in
  let nan_ = (where (eq x (zero x)) x (nan x), nan x) in
  cases [ (big, v); (is_nan ax ||: is_nan ay, nan_) ] v

let acos_raw x y =
  let ax = magnitude x and ay = magnitude y in
  let big = gt ax (R.con x asin_big) ||: gt ay (R.con x asin_big) in
  let real = eq ay (zero y) &&: le ax (one x) in
  let ar, ai = sqrt_raw (one x -: x) (negated ay) in
  let br, bi = sqrt_raw (one x +: x) ay in
  let re =
    where big (atan2 ay x) (where real (acos x) (num x 2. *: atan (ar /: br)))
  in
  let im =
    where big
      (negated (far_im ax ay))
      (where real (num x (-0.)) (asinh_part (det br ai bi ar)))
  in
  let v = (re, flip im y) in
  let nan_ = (where (eq x (zero x)) (R.con x R.pio2_hi) (nan x), nan x) in
  cases [ (big, v); (is_nan ax ||: is_nan ay, nan_) ] v

(* atanh z in the first quadrant (Kahan, 1987): log1p(4x / ((1 - x)^2 + y^2)) /
   4 + i atan2(2y, (1 - x)(1 + x) - y^2) / 2, with x = 1 apart, where the real
   part is log((4 + y^2)^(1/4) / sqrt y); past atanh_big, Re(1/z) + i pi/2. On
   the imaginary axis, i atan y. *)
let atanh_raw x y =
  let ax = magnitude x and ay = magnitude y in
  let pio2 = R.con x R.pio2_hi in
  let m = larger ax ay in
  let sx = ax /: m and sy = ay /: m in
  let far = sx /: mulacc sx sx (sy *: sy) /: m in
  let unit_re =
    log (Ops.sqrt (Ops.sqrt (mulacc ay ay (num x 4.))) /: Ops.sqrt ay)
  in
  let unit_im = half x *: (pio2 +: atan (half x *: ay) +: R.con x R.pio2_lo) in
  let u = one x -: ax in
  let near_re = num x 0.25 *: log1p (num x 4. *: ax /: mulacc u u (ay *: ay)) in
  let near_im =
    half x *: atan2 (ay +: ay) (mulacc (negated ay) ay (u *: (one x +: ax)))
  in
  let re, im =
    cases
      [
        ( eq ax (inf x) ||: eq ay (inf x),
          (zero x, where (is_nan ay) (nan x) pio2) );
        (eq ax (zero x), (zero x, atan ay));
        (eq ax (one x) &&: eq ay (zero y), (inf x, zero x));
        (gt ax (R.con x atanh_big) ||: gt ay (R.con x atanh_big), (far, pio2));
        (eq ax (one x), (unit_re, unit_im));
      ]
      (near_re, near_im)
  in
  let v = (flip re x, flip im y) in
  let early = eq ax (inf x) ||: eq ay (inf x) ||: eq ax (zero x) in
  cases [ (early, v); (is_nan ax ||: is_nan ay, nans x) ] v

(* Arithmetic

   Add, Sub and Neg are the parts' kinds. Mul is the textbook product, each part
   one fused multiply-add over the other product; Fdiv is Smith's algorithm, the
   divisor's larger part dividing its smaller. *)

(* z with each NaN part the quiet NaN of a clear sign bit. *)
let quiet (re, im) =
  let q v = where (is_nan v) (nan v) v in
  (q re, q im)

let mul (ar, ai) (br, bi) =
  let ii = ai *: bi and ir = ai *: br in
  quiet (mulacc ar br (negated ii), mulacc ar bi ir)

let fdiv (ar, ai) (br, bi) =
  let pb = magnitude br and pi = magnitude bi in
  let big = ge pb pi in
  let p = where big br bi and q = where big bi br in
  let x = where big ar ai and y = where big ai ar in
  let r = q /: p in
  let d = mulacc r q p in
  let im = where big (mulacc (negated r) x y) (mulacc r x (negated y)) in
  quiet (mulacc r y x /: d, im /: d)

let has_nan (re, im) = is_nan re ||: is_nan im

(* Whether [x] precedes [y] in the order with -0 below +0, neither NaN. *)
let before x y = lt x y ||: (eq x y &&: negative x &&: not_ (negative y))
let same x y = eq x y &&: eq (negative x) (negative y)
let precedes (ar, ai) (br, bi) = before ar br ||: (same ar br &&: before ai bi)

(* The first NaN operand, else [b] where [gives_b a b]. *)
let extreme gives_b a b =
  let c = not_ (has_nan a) &&: (has_nan b ||: gives_b a b) in
  cases [ (c, b) ] a

(* Kinds

   The trigonometric kinds are the hyperbolic ones turned by i: sin z = -i
   sinh(iz), cos z = cosh(iz), tan z = -i tanh(iz) and atan z = -i atanh(iz),
   where iz = -y + ix and -i(a + ib) = b - ia; asinh turns asin back, and acosh
   turns acos. Every NaN part of an operand is made quiet first, and every NaN
   part of a value last. *)

let turn (re, im) = quiet (im, negated re)

let refuse k =
  invalid_arg ("Lower_complex.unary: complex numbers do not take " ^ k)

(* acosh z = i acos z above the real axis, and its conjugate below. A NaN
   imaginary part beside a finite real one is no direction: NaN. *)
let acosh x y =
  let vr, vi = acos_raw x (magnitude y) in
  let v = quiet (negated vi, flip vr y) in
  cases [ (is_nan y &&: finite x, nans x) ] v

(* z with each NaN part's quiet bit set, its sign and payload kept, as each
   transcendental kind reads its operands. *)
let unsignalled (re, im) =
  let q v = where (is_nan v) (quieted v) v in
  (q re, q im)

let unary (k : Nx_kernel.Prog.unary) ((re, im) as z) =
  let x, y = unsignalled z in
  match k with
  | Neg -> (negated re, negated im)
  | Recip -> fdiv (one re, zero re) z
  | Sign -> quiet (sign_raw x y)
  | Sqrt -> quiet (sqrt_raw x y)
  | Exp -> quiet (exp_raw x y)
  | Exp2 -> quiet (exp2_raw x y)
  | Log -> quiet (log_raw x y)
  | Log2 -> quiet (log2_raw x y)
  | Sinh -> quiet (sinh_raw x y)
  | Cosh -> quiet (cosh_raw x y)
  | Tanh -> quiet (tanh_raw x y)
  | Asin -> quiet (asin_raw x y)
  | Acos -> quiet (acos_raw x y)
  | Atanh -> quiet (atanh_raw x y)
  | Sin -> turn (sinh_raw (negated y) x)
  | Cos -> quiet (cosh_raw (negated y) x)
  | Tan -> turn (tanh_raw (negated y) x)
  | Atan -> turn (atanh_raw (negated y) x)
  | Asinh ->
      (* asinh z = i asin(-iz), -iz = y - ix and i(a + ib) = -b + ia *)
      let vr, vi = asin_raw y (negated x) in
      quiet (negated vi, vr)
  | Acosh -> acosh x y
  | Abs -> refuse "Abs"
  | Log1p -> refuse "Log1p"
  | Expm1 -> refuse "Expm1"
  | Erf -> refuse "Erf"
  | Floor -> refuse "Floor"
  | Ceil -> refuse "Ceil"
  | Round -> refuse "Round"
  | Trunc -> refuse "Trunc"

let binary (k : Nx_kernel.Prog.binary) ((ar, ai) as a) ((br, bi) as b) =
  match k with
  | Add | Sub -> (R.binary k ar br, R.binary k ai bi)
  | Mul -> mul a b
  | Fdiv -> fdiv a b
  | Pow -> quiet (pow_raw (unsignalled a) (unsignalled b))
  | Maximum -> extreme precedes a b
  | Minimum -> extreme (fun a b -> precedes b a) a b
  | Idiv | Mod | Atan2 | And | Or | Xor | Threefry ->
      invalid_arg "Lower_complex.binary: a kind complex numbers do not take"

let compare (k : Nx_kernel.Prog.compare) ((ar, ai) as a) ((br, bi) as b) =
  let ordered = not_ (has_nan a ||: has_nan b) in
  match k with
  | Equal -> eq ar br &&: eq ai bi
  | Not_equal -> not_ (eq ar br &&: eq ai bi)
  | Less -> ordered &&: (lt ar br ||: (eq ar br &&: lt ai bi))
  | Less_equal -> ordered &&: (lt ar br ||: (eq ar br &&: le ai bi))
