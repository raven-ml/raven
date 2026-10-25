(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Each kind composes Lower_real's float kinds over pairs of nodes, in a form
   that cancels nowhere and overflows only where its result does. *)

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

(* Clamps past which e^x c and 2^x c overflow for every finite non-zero c, and
   underflow for every c of magnitude at most 1. *)
let cexp_hi = (194., 1456.)
let cexp2_hi = (280., 2101.)

(* Float kinds *)

let log = R.unary Log
let log1p = R.unary Log1p
let exp = R.unary Exp
let sin = R.unary Sin
let cos = R.unary Cos
let tan = R.unary Tan
let atan = R.unary Atan
let sinh = R.unary Sinh
let cosh = R.unary Cosh
let atan2 = R.binary Atan2
let round = R.unary Round

(* Parts *)

let larger a b = where (gt a b) a b
let smaller a b = where (gt a b) b a
let neg_int k = Ops.sub (int k 0) k
let clamp x c = where (gt x c) c (where (lt x (negated c)) (negated c) x)
let infinite x = eq (magnitude x) (inf x)

(* Whether [x] is infinite or NaN. *)
let wild x = not_ (lt (magnitude x) (inf x))

(* y 2^k rounded once, for k up to twice the span R.scale takes. *)
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
  let a = larger ax ay and b = smaller ax ay in
  let k = R.ilogb a in
  let sa = R.scale a (neg_int k) and sb = R.scale b (neg_int k) in
  let h = R.scale (Ops.sqrt (mulacc sa sa (sb *: sb))) k in
  where (eq a (zero a)) a h

(* log |x + iy|: of the larger part a and the smaller b, half log1p of a^2 + b^2
   - 1 where a is near 1, from exact products (a - 1 is exact, a + 1 and each
   product a head and a tail), so that it keeps its digits near the unit circle;
   else log a + log1p((b/a)^2) / 2; log a where b is 0; inf where a part is,
   NaN included. *)
let log_abs x y =
  let ax = magnitude x and ay = magnitude y in
  let a = larger ax ay and b = smaller ax ay in
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
  where (infinite x ||: infinite y) (inf x) v

(* asinh t, odd: log1p(a + a^2 / (1 + sqrt(1 + a^2))) of a = |t|; past 2^p,
   where a^2 + 1 rounds to a^2, log a + ln2. *)
let asinh_part t =
  let a = magnitude t in
  let a2 = a *: a in
  let near = log1p (a +: (a2 /: (one a +: Ops.sqrt (one a +: a2)))) in
  let far = log a +: R.con a R.ln2 in
  R.copysign (where (lt a (num a (R.pick a (0x1p24, 0x1p53)))) near far) t

(* 2^(hi + lo) (c + i s), for hi an exponent clamped so that the product's scale
   is an int32: with k the integer nearest hi, 2^(hi - k + lo), within [1/2, 2],
   times each part, then 2^k, so that no part overflows before its product. *)
let polar (hi, lo) c s =
  let k = round hi in
  let m = R.exp2_split (hi -: k) lo in
  let k = Ops.cast k Int32 in
  (ldexp (m *: c) k, ldexp (m *: s) k)

(* x / ln2 as R.exp_parts gives it, of x clamped where e^x c over- or
   underflows. *)
let exp_parts x = R.exp_parts (clamp x (R.con x cexp_hi))

(* Exponentials and logarithms *)

(* C99 G.6.3.1's values of e^z, and 2^z's, where y is infinite or NaN and x
   infinite: 0 at x = -inf, inf + i NaN at x = +inf. *)
let unbounded x y (re, im) =
  let low = eq x (negated (inf x)) &&: wild y in
  let high = eq x (inf x) &&: wild y in
  (where low (zero x) (where high x re), where low (zero y) im)

(* e^z = e^x cis y. *)
let exp_raw x y = unbounded x y (polar (exp_parts x) (cos y) (sin y))

(* 2^z = 2^x cis(y ln2). *)
let exp2_raw x y =
  let t = y *: R.con y R.ln2 in
  unbounded x y (polar (clamp x (R.con x cexp2_hi), zero x) (cos t) (sin t))

let log_raw x y = (log_abs x y, atan2 y x)

let log2_raw x y =
  let re, im = log_raw x y in
  (re *: R.con re R.log2e, im *: R.con im R.log2e)

(* Square root (Kahan, 1987): t = sqrt((|x| + |z|) / 2) is the part of the
   larger magnitude, y / 2t the other; z is scaled by an even power of two so
   that its larger part is in [1, 4). On the real axis, the float square root.
   Where a part is infinite, an infinite part of the root. *)
let sqrt_raw x y =
  let ax = magnitude x and ay = magnitude y in
  let e = R.ilogb (larger ax ay) in
  let j = Ops.shr e (int e 1) in
  let m2j = Ops.mul j (int j (-2)) in
  let sx = R.scale ax m2j and sy = R.scale ay m2j in
  let t = R.scale (Ops.sqrt ((sx +: hypot sx sy) *: half x)) j in
  let u = ay /: (t +: t) in
  let pos = ge x (zero x) in
  let r = Ops.sqrt ax in
  let real = (where pos r (zero x), where pos y (R.copysign r y)) in
  let v = (where pos t u, where pos (R.copysign u y) (R.copysign t y)) in
  let real_axis = eq y (zero y) in
  let re = where real_axis (fst real) (fst v) in
  let im = where real_axis (snd real) (snd v) in
  (* G.6.4.2: inf + i y at an infinite y; an infinite x makes the real part
     infinite at +inf, the imaginary part at -inf *)
  let up = infinite y ||: eq x (inf x) in
  let side = infinite y ||: eq x (negated (inf x)) in
  (where up (inf x) re, where side (R.copysign (inf y) y) im)

(* z^w = e^(w log z). A real w multiplies log z's parts alone, so that a zero
   part meets no infinite one; z^0 is 1. *)
let pow_raw (zr, zi) (wr, wi) =
  let lr, li = log_raw zr zi in
  let real = eq wi (zero wi) in
  let pr = where real (wr *: lr) (mulacc wr lr (negated (wi *: li))) in
  let pi = where real (wr *: li) (mulacc wr li (wi *: lr)) in
  let er, ei = exp_raw pr pi in
  let unit = real &&: eq wr (zero wr) in
  (where unit (one zr) er, where unit (zero zr) ei)

(* sign z = z / |z|, z scaled by a power of two first so that |z| is no
   subnormal; z itself at 0, and at an infinity without a NaN part its
   direction among the eight a pair of parts names. *)
let sign_raw x y =
  let a = larger (magnitude x) (magnitude y) in
  let k = neg_int (R.ilogb a) in
  let sx = R.scale x k and sy = R.scale y k in
  let h = hypot sx sy in
  let z = eq a (zero a) in
  let ix = infinite x and iy = infinite y in
  let far = (ix ||: iy) &&: not_ (is_nan x ||: is_nan y) in
  let toward v p q =
    R.copysign (where p (where q (num v Float.(sqrt 0.5)) (one v)) (zero v)) v
  in
  ( where far (toward x ix iy) (where z x (sx /: h)),
    where far (toward y iy ix) (where z y (sy /: h)) )

(* Hyperbolic kinds

   sinh z = sinh x cos y + i cosh x sin y and cosh z = cosh x cos y + i sinh x
   sin y. Past R.hyp_far, sinh |x| and cosh |x| are e^|x| / 2 = 2^(|x| / ln2 -
   1), taken as polar does so that a small cos y or sin y keeps the product
   finite. *)

let half_polar ax c s =
  let hi, lo = exp_parts ax in
  polar (hi -: one ax, lo) c s

let sinh_raw x y =
  let c = cos y and s = sin y in
  let ax = magnitude x in
  let fr, fi = half_polar ax c s in
  let near = lt ax (num x R.hyp_far) in
  let re = where near (sinh x *: c) (R.flip fr x) in
  (* G.6.2.5: x + i NaN where x is 0 or infinite and y infinite or NaN *)
  let kept = wild y &&: (eq x (zero x) ||: infinite x) in
  (where kept x re, where near (cosh x *: s) fi)

let cosh_raw x y =
  let c = cos y and s = sin y in
  let ax = magnitude x in
  let fr, fi = half_polar ax c s in
  let near = lt ax (num x R.hyp_far) in
  let im = where near (sinh x *: s) (R.flip fi x) in
  (* G.6.2.4: NaN + i0 at x = 0 and inf + i NaN at an infinite x, where y is
     infinite or NaN *)
  let re = where near (cosh x *: c) fr in
  ( where (wild y &&: infinite x) (inf x) re,
    where (wild y &&: eq x (zero x)) x im )

(* tanh z (Kahan, 1987): with t = tan y, b = 1 + t^2, s = sinh x and c = cosh x,
   (b c s + i t) / (1 + b s^2), which cancels nowhere. Past R.hyp_far, the real
   part rounds to ±1, and the imaginary part is 4 sin y cos y e^-2|x|. *)
let tanh_raw x y =
  let ax = magnitude x in
  let t = tan y in
  let b = mulacc t t (one t) in
  let s = sinh x and c = cosh x in
  let d = mulacc (b *: s) s (one x) in
  let far_im = num x 4. *: sin y *: cos y *: exp (num x (-2.) *: ax) in
  (* G.6.2.6: 1 + i0 at an infinite x where y is infinite or NaN, which a
     finite x makes NaN + i NaN *)
  let far = ge ax (num x R.hyp_far) &&: not_ (wild y &&: not_ (infinite x)) in
  let far_im = where (wild y) (zero y) far_im in
  ( where far (R.copysign (one x) x) (b *: c *: s /: d),
    where far far_im (t /: d) )

(* Inverse kinds

   asin and acos (Kahan, 1987), from A = sqrt(1 - z) and B = sqrt(1 + z): asin z
   = atan(x / Re(A B)) + i asinh(Im(conj(A) B)) and acos z = 2 atan(Re A / Re B)
   + i asinh(Im(conj(B) A)). Past asin_far, where A B can overflow before the
   result, asin z is atan2(x, y) + i (log |z| + ln2) in the first quadrant, its
   leading terms at infinity, whose next are below the unit roundoff. *)

let asin_far = (0x1p13, 0x1p28)

let far_parts x y =
  let ax = magnitude x and ay = magnitude y in
  let far = gt ax (R.con x asin_far) ||: gt ay (R.con x asin_far) in
  (far, atan2 ax ay, log_abs ax ay +: R.con x R.ln2)

let asin_raw x y =
  let ar, ai = sqrt_raw (one x -: x) (negated y) in
  let br, bi = sqrt_raw (one x +: x) y in
  let far, fr, fi = far_parts x y in
  ( where far (R.flip fr x) (atan (x /: det ar br ai bi)),
    where far (R.flip fi y) (asinh_part (det ar bi ai br)) )

let acos_raw x y =
  let ar, ai = sqrt_raw (one x -: x) (negated y) in
  let br, bi = sqrt_raw (one x +: x) y in
  let far, _, fi = far_parts x y in
  ( where far (atan2 (magnitude y) x) (num x 2. *: atan (ar /: br)),
    where far (negated (R.flip fi y)) (asinh_part (det br ai bi ar)) )

(* atanh z in the first quadrant (Kahan, 1987): log1p(4x / ((1 - x)^2 + y^2)) /
   4 + i atan2(y, ((1 - x)(1 + x) - y^2) / 2) / 2, with x = 1 and y < 1 apart,
   where the real part is log((4 + y^2)^(1/4) / sqrt y), so that y^2 cannot
   underflow there. The parts' signs are x's and y's. *)
let atanh_raw x y =
  let ax = magnitude x and ay = magnitude y in
  let u = one x -: ax in
  let re = num x 0.25 *: log1p (num x 4. *: (ax /: mulacc u u (ay *: ay))) in
  let unit_re =
    log (Ops.sqrt (Ops.sqrt (mulacc ay ay (num x 4.))) /: Ops.sqrt ay)
  in
  let re = where (eq ax (one x) &&: lt ay (one y)) unit_re re in
  let d = half x *: mulacc (negated ay) ay (u *: (one x +: ax)) in
  let im = half x *: atan2 ay d in
  (* G.6.2.3: 0 + i pi/2 where a part is infinite, i NaN where y is NaN *)
  let far = infinite x ||: infinite y in
  let re = where far (zero x) re in
  let im = where (far &&: not_ (is_nan y)) (num x (Float.pi /. 2.)) im in
  (R.flip re x, R.flip im y)

(* Arithmetic

   Mul is the textbook product, each part one fused multiply-add over the other
   product; Fdiv is Smith's algorithm, the divisor's larger part dividing its
   smaller. *)

let mul (ar, ai) (br, bi) =
  let ii = ai *: bi and ir = ai *: br in
  (mulacc ar br (negated ii), mulacc ar bi ir)

let fdiv (ar, ai) (br, bi) =
  let big = ge (magnitude br) (magnitude bi) in
  let p = where big br bi and q = where big bi br in
  let x = where big ar ai and y = where big ai ar in
  let r = q /: p in
  let d = mulacc r q p in
  let im = where big (mulacc (negated r) x y) (mulacc r x (negated y)) in
  (mulacc r y x /: d, im /: d)

let has_nan (re, im) = is_nan re ||: is_nan im

(* Whether [x] precedes [y] in the order with -0 below +0, neither NaN. *)
let before x y = lt x y ||: (eq x y &&: negative x &&: not_ (negative y))
let same x y = eq x y &&: eq (negative x) (negative y)
let precedes (ar, ai) (br, bi) = before ar br ||: (same ar br &&: before ai bi)

(* The first NaN operand, else [b] where [gives_b a b]. *)
let extreme gives_b ((ar, ai) as a) ((br, bi) as b) =
  let c = not_ (has_nan a) &&: (has_nan b ||: gives_b a b) in
  (where c br ar, where c bi ai)

(* Kinds

   The trigonometric kinds are the hyperbolic ones turned by i: sin z = -i
   sinh(iz), cos z = cosh(iz), tan z = -i tanh(iz) and atan z = -i atanh(iz),
   where iz = -y + ix and -i(a + ib) = b - ia; asinh turns asin back, and acosh
   turns acos. *)

let turn (re, im) = (im, negated re)

let refuse k =
  invalid_arg ("Lower_complex.unary: complex numbers do not take " ^ k)

let unary (k : Nx_kernel.Prog.unary) ((x, y) as z) =
  match k with
  | Neg -> (negated x, negated y)
  | Recip -> fdiv (one x, zero x) z
  | Sign -> sign_raw x y
  | Sqrt -> sqrt_raw x y
  | Exp -> exp_raw x y
  | Exp2 -> exp2_raw x y
  | Log -> log_raw x y
  | Log2 -> log2_raw x y
  | Sinh -> sinh_raw x y
  | Cosh -> cosh_raw x y
  | Tanh -> tanh_raw x y
  | Asin -> asin_raw x y
  | Acos -> acos_raw x y
  | Atanh -> atanh_raw x y
  | Sin -> turn (sinh_raw (negated y) x)
  | Cos -> cosh_raw (negated y) x
  | Tan -> turn (tanh_raw (negated y) x)
  | Atan -> turn (atanh_raw (negated y) x)
  | Asinh ->
      (* asinh z = i asin(-iz), -iz = y - ix and i(a + ib) = -b + ia *)
      let vr, vi = asin_raw y (negated x) in
      (negated vi, vr)
  | Acosh ->
      (* i acos z above the real axis, and its conjugate below *)
      let vr, vi = acos_raw x (magnitude y) in
      (negated vi, R.flip vr y)
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
  | Pow -> pow_raw a b
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
