(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Each kind composes Lower_real's float kinds over pairs of nodes. A kind of
   nx's sequence restates nx_kinds_complex.h's function operation for
   operation, as Lower_real does nx_kinds_real.h's, and names it; when the
   header changes, it changes with it. Each other kind takes a form that
   cancels nowhere and overflows only where its result does, its bound in
   [lower_complex.mli] derived to first order from Lower_real's. *)

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
let log2 = R.unary Log2
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

(* nx_finite: [|x|] below infinity, false at a NaN. *)
let finite x = lt (magnitude x) (inf x)

let nans x = (nan x, nan x)

(* Branches

   A C function's early returns are a selection of every return's value, the
   first whose condition holds winning: [first l d] is the value of the first
   case of [l] whose condition holds, else [d], and [cases] that of pairs.
   Every case computes on every element; a case's value on an element whose
   condition is false is never selected, whatever it is. *)
let rec first l d =
  match l with [] -> d | (c, v) :: rest -> where c v (first rest d)

let cases l d =
  ( first (List.map (fun (c, (re, _)) -> (c, re)) l) (fst d),
    first (List.map (fun (c, (_, im)) -> (c, im)) l) (snd d) )

(* nx_unsignalled: x with its quiet bit set where it is a NaN, its sign and
   payload kept. *)
let unsignalled x =
  let q = Int64.shift_left 1L (R.pick x R.mbits - 1) in
  let qb = R.of_word x (q, q) in
  where (is_nan x)
    (Ops.bitcast (Ops.bitwise_or (bits x) (bits qb)) (Ops.dtype x))
    x

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

(* nx_log_abs: log |x + iy|. Of the larger part a and the smaller b, half
   log1p of a^2 + b^2 - 1 where a is near 1, from exact products (a - 1 is
   exact, a + 1 and each product a head and a tail), so that it keeps its
   digits near the unit circle; else log a + log1p((b/a)^2) / 2; log a where b
   is 0; inf where a part is, NaN where a part is NaN otherwise. *)
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
  first
    [
      (eq ax (inf x) ||: eq ay (inf y), inf x);
      (is_nan ax ||: is_nan ay, nan x);
      (eq b (zero b), log a);
      (ge a (half a) &&: le a (num a 2.), near);
    ]
    far

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

(* nx_log_raw: log z = log |z| + i arg z, arg z = atan2(y, x) with its special
   values. *)
let log_raw x y = (log_abs x y, atan2 y x)

(* nx_log2_raw: log2 z = log z / ln2, 1/ln2 as a head and a tail; on the axes,
   the part type's log2 of the modulus. *)
let log2_raw x y =
  let lr, li = log_raw x y in
  let by_ln2 v = mulacc v (R.con v R.log2e) (v *: R.con v R.log2e_lo) in
  let ax = magnitude x and ay = magnitude y in
  let re = by_ln2 lr in
  let re = where (eq y (zero y) &&: eq ax ax) (log2 ax) re in
  let re = where (eq x (zero x) &&: finite y) (log2 ay) re in
  (re, by_ln2 li)

(* nx_sqrt_raw (Kahan, 1987): t = sqrt((|x| + |z|) / 2) is the part of the
   larger magnitude, y / 2t the other; z is scaled by an even power of two so
   that its larger part is in [1, 4), and y / 2t is taken of y itself. The
   general case's scaling reads a finite non-zero larger part on the elements
   it is selected for; on the others its exponent is garbage and its value
   unread. *)
let sqrt_raw x y =
  let ax = magnitude x and ay = magnitude y in
  let a = larger ax ay in
  let e = R.ilogb a in
  let j = Ops.shr e (int e 1) in
  let m2j = Ops.mul j (int j (-2)) in
  let xs = R.scale ax m2j and ys = R.scale ay m2j in
  let t = R.scale (Ops.sqrt ((xs +: hypot xs ys) *: half x)) j in
  let u = ay /: (t +: t) in
  let pos = ge x (zero x) in
  let root = Ops.sqrt ax in
  let y_nan = is_nan y in
  cases
    [
      (eq ay (inf y), (inf x, y));
      (is_nan x, nans x);
      (eq x (inf x), (x, where y_nan (nan x) (R.copysign (zero x) y)));
      ( eq x (negated (inf x)),
        (where y_nan (nan x) (zero x), R.copysign (inf x) y) );
      (y_nan, nans x);
      ( eq y (zero y),
        (where pos root (zero x), where pos y (R.copysign root y)) );
    ]
    (where pos t u, where pos (R.copysign u y) (R.copysign t y))

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

(* nx_sign_raw: z / |z|, z scaled by a power of two first so that |z| is no
   subnormal; z itself at 0, and at an infinity its direction among the eight
   a pair of parts names. As in sqrt_raw, the scaling is garbage and unread
   where a part is 0, infinite or NaN. *)
let sign_raw x y =
  let ax = magnitude x and ay = magnitude y in
  let ix = eq ax (inf x) and iy = eq ay (inf y) in
  let both = R.of_word x R.sqrt_half_bits in
  let re = where ix (where iy both (one x)) (zero x) in
  let im = where iy (where ix both (one x)) (zero x) in
  let k = neg_int (R.ilogb (larger ax ay)) in
  let xs = R.scale x k and ys = R.scale y k in
  let h = hypot xs ys in
  cases
    [
      (is_nan ax ||: is_nan ay, nans x);
      (ix ||: iy, (R.copysign re x, R.copysign im y));
      (eq ax (zero x) &&: eq ay (zero y), (x, y));
    ]
    (xs /: h, ys /: h)

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
   product; Fdiv is Smith's algorithm (1962), the divisor's smaller part divided
   by its larger. *)

let mul (ar, ai) (br, bi) =
  let ii = ai *: bi and ir = ai *: br in
  (mulacc ar br (negated ii), mulacc ar bi ir)

(* A dividend whose parts are below the least normal over u, 2^-102 (float32)
   or 2^-969 (float64), is scaled by 2^25 or 2^54 first, and the quotient back:
   unscaled, a fused multiply-add of its parts lands subnormal and drops bits,
   which a small divisor magnifies into a normal quotient. The scaling back is
   exact where the quotient is normal. *)
let tiny_dividend = (0x1p-102, 0x1p-969)
let dividend_up = (0x1p25, 0x1p54)
let dividend_down = (0x1p-25, 0x1p-54)

let smith (ar, ai) (br, bi) =
  let big = ge (magnitude br) (magnitude bi) in
  let p = where big br bi and q = where big bi br in
  let x = where big ar ai and y = where big ai ar in
  let r = q /: p in
  let d = mulacc r q p in
  let im = where big (mulacc (negated r) x y) (mulacc r x (negated y)) in
  (mulacc r y x /: d, im /: d)

let fdiv (ar, ai) b =
  let tiny =
    lt (magnitude ar) (num ar (R.pick ar tiny_dividend))
    &&: lt (magnitude ai) (num ai (R.pick ai tiny_dividend))
  in
  let scale v = where tiny (num ar (R.pick ar v)) (one ar) in
  let up = scale dividend_up and down = scale dividend_down in
  let re, im = smith (ar *: up, ai *: up) b in
  (re *: down, im *: down)

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
  | Sign -> sign_raw (unsignalled x) (unsignalled y)
  | Sqrt -> sqrt_raw (unsignalled x) (unsignalled y)
  | Exp -> exp_raw x y
  | Exp2 -> exp2_raw x y
  | Log -> log_raw (unsignalled x) (unsignalled y)
  | Log2 -> log2_raw (unsignalled x) (unsignalled y)
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
