(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Special functions, each a fixed program of nx's operations.

   A function's regions run on every element, each over its inputs clamped into
   the region by [where], and [where] selects the region's result, so that every
   intermediate is finite where its region is not selected, and the zero partial
   derivative [where] gives it stays zero. Where an identity maps the whole
   domain into one region (a reflection, a recurrence shifted by a masked
   count), the input is mapped and one formula runs. Edges and NaN are selected
   last. The tables come from [Special_tables], which test/gen/special.py
   writes. *)

open Frontend
module T = Special_tables

let lit x v = scalar_like x v
let lt x v = cmplt x (lit x v)
let ge x v = cmpge x (lit x v)
let is x v = cmpeq x (lit x v)

(* Horner from the leading coefficient. *)
let horner cs t =
  let acc = ref (lit t cs.(0)) in
  for i = 1 to Array.length cs - 1 do
    acc := add (mul !acc t) (lit t cs.(i))
  done;
  !acc

(* [x] where [inside], elsewhere the point [b] of the region. *)
let clamp inside x b = where inside x (lit x b)

let tables (type b) (x : (float, b) t) =
  match dtype x with Float64 -> T.float64 | _ -> T.float32

let log_sqrt_2pi = 0.5 *. Stdlib.log (2. *. Float.pi)

(* Error function *)

(* fdlibm's erfc: [1 - erf] from erf's P/Q below 0.84375, a P/Q about 1 to 1.25,
   and [exp (-x^2 - 0.5625 + R/S) / x] above, the tails' R/S in [1/x^2], to 28,
   past which it is 0 or 2 ([Special_tables] holds the boundaries). [x^2] is
   held exactly as [hi + lo] by [fma], so the exponential of its rounded part is
   exact in its argument. Its parts also give [erf x] below 1.25, to the
   relative precision of the rationals, and, from 1.25 up, [2/sqrt pi exp (-x^2)
   / erfc x = 2/sqrt pi x exp (0.5625 - R/S)], which divides no tiny number by
   another. *)
type 'b erfc_parts = {
  erfc : (float, 'b) t;
  erf : (float, 'b) t; (* [erf x] where [|x| < 1.25] *)
  upper : (bool, bool_elt) t; (* [x >= 1.25] *)
  ratio : (float, 'b) t; (* [2/sqrt pi exp (-x^2) / erfc x] where [upper] *)
}

let erfc_parts x =
  let a = abs x and positive = cmpgt x (zeros_like x) in
  let small = lt a T.erf_small_below and below_near = lt a T.erf_near_below in
  let below = lt a T.erfc_far_from in
  let near = logical_and (logical_not small) below_near in
  let tail = logical_and (logical_not below_near) below in
  let xs = clamp small x 0.5 in
  let z = mul xs xs in
  let y = div (horner T.erf_small_p z) (horner T.erf_small_q z) in
  let xy = mul xs y in
  let r_small =
    where (lt xs T.erfc_small_split)
      (rsub_s 1. (add xs xy))
      (rsub_s 0.5 (add xy (sub_s xs 0.5)))
  in
  let s = sub_s (clamp near a 1.) 1. in
  let pq = div (horner T.erf_near_p s) (horner T.erf_near_q s) in
  let r_near =
    where positive (rsub_s (1. -. T.erx) pq) (add_s (add_s pq T.erx) 1.)
  in
  let at = clamp tail a 2. in
  let w = recip (mul at at) in
  let mid = lt at T.erfc_mid_below in
  let rs =
    div
      (where mid (horner T.erfc_mid_p w) (horner T.erfc_far_p w))
      (where mid (horner T.erfc_mid_q w) (horner T.erfc_far_q w))
  in
  let hi = mul at at in
  let lo = fma at at (neg hi) in
  let e = mul (exp (neg hi)) (exp (sub (sub_s rs 0.5625) lo)) in
  let q = div e at in
  let r_tail = where positive q (rsub_s 2. q) in
  let r_far = where positive (zeros_like x) (lit x 2.) in
  let ratio =
    mul_s (mul at (exp (rsub_s 0.5625 rs))) (2. /. Float.sqrt Float.pi)
  in
  let erf_near = add_s pq T.erx in
  {
    erfc = where small r_small (where near r_near (where tail r_tail r_far));
    erf = where small (add xs xy) (where positive erf_near (neg erf_near));
    upper = logical_and (logical_not below_near) positive;
    ratio;
  }

let erfc_at x = (erfc_parts x).erfc

(* erfinv: Giles' guess, single precision at float32 and double at float64, a
   polynomial in [w = -log ((1 - p)(1 + p))] or [sqrt w] by region, its
   coefficients selected per element; then one Newton step, whose derivative is
   the implicit one: on [erf y = p] below [|p| = erf 1.25], and on [erfc |y| = 1
   - |p|] above, where [1 - |p|] is exact and [erf] would round to 1. One
   [erfc_parts] serves both. *)
let erfinv_at p =
  let t = tables p in
  let edge = cmpge (abs p) (lit p 1.) in
  let p = clamp (logical_not edge) p 0. in
  let w = neg (log (mul (rsub_s 1. p) (add_s p 1.))) in
  let central = lt w t.erfinv_central_below and far = ge w t.erfinv_far_from in
  (* [sqrt] of [w] floored at 1, where the tails' polynomials are not selected:
     its derivative is infinite at [w = 0]. *)
  let r = sqrt (clamp (logical_not central) w 1.) in
  let u =
    where central
      (sub_s w t.erfinv_central_shift)
      (sub r (where far (lit r t.erfinv_far_shift) (lit r t.erfinv_near_shift)))
  in
  let coefficient i =
    where central
      (lit u t.erfinv_central.(i))
      (where far (lit u t.erfinv_far.(i)) (lit u t.erfinv_near.(i)))
  in
  let acc = ref (coefficient 0) in
  for i = 1 to Array.length t.erfinv_central - 1 do
    acc := add (mul !acc u) (coefficient i)
  done;
  let y = mul p !acc in
  let inner = lt (abs p) t.erfinv_newton_split in
  let a = abs y in
  let parts = erfc_parts (where inner y a) in
  (* [erf' y = 2/sqrt pi exp (-y^2)], so each step multiplies by its
     reciprocal. *)
  let slope = mul_s (exp (mul y y)) (Float.sqrt Float.pi /. 2.) in
  let central_y = sub y (mul (sub parts.erf p) slope) in
  let tail_a = add a (mul (sub parts.erfc (rsub_s 1. (abs p))) slope) in
  let tail_y = where (lt p 0.) (neg tail_a) tail_a in
  where inner central_y tail_y

(* [-x/sqrt 2] as [th + tl], for finite [x]. *)
let split_sqrt1_2 x =
  let t = tables x in
  let c = lit x (-.t.sqrt1_2_hi) in
  let th = mul x c in
  (th, add (fma x c (neg th)) (mul_s x (-.t.sqrt1_2_lo)))

(* The standard normal's distribution function at [x], [erfc t / 2] at [t =
   -x/sqrt 2 = th + tl], to first order in [tl]: [erfc (th + tl) = erfc th (1 -
   g tl)] for [g = 2/sqrt pi exp (-th^2) / erfc th], which the parts of [erfc
   th] give in the upper tail. Elsewhere [th] is clamped to [-erfc_far_from],
   below which [exp (-th^2)] is 0 already, so that [th^2] and its partial
   derivative stay finite for every finite [x]. *)
let ndtr_of th tl parts =
  let e = parts.erfc in
  let far = -.T.erfc_far_from in
  let tc = clamp (logical_not parts.upper) (clamp (ge th far) th far) 0. in
  let g =
    div
      (mul_s (exp (neg (mul tc tc))) (2. /. Float.sqrt Float.pi))
      (where parts.upper (ones_like e) e)
  in
  let g = where parts.upper parts.ratio g in
  (* Where [erfc th] underflows to [+0], the correction would only flip its
     sign. *)
  where (is e 0.) e (mul (mul_s e 0.5) (rsub_s 1. (mul g tl)))

let ndtr_at x =
  let finite = isfinite x in
  let th, tl = split_sqrt1_2 (clamp finite x 0.) in
  let r = ndtr_of th tl (erfc_parts th) in
  where finite r (where (cmpgt x (zeros_like x)) (ones_like x) (zeros_like x))

(* [log (ndtr x)] below the table's threshold: [-x^2/2 - log (-x) - log (2 pi)/2
   + log1p (sum_k (-1)^k (2k - 1)!! / x^(2k))]. *)
let log_ndtr_far x =
  let t = tables x in
  let v = recip (mul x x) in
  add
    (sub_s (neg (add (mul (mul_s x 0.5) x) (log (neg x)))) log_sqrt_2pi)
    (log1p (mul v (horner t.log_ndtr_series v)))

(* [log (ndtr x)]: above 0 [log1p (-(ndtr (-x)))]; down to the table's
   threshold, where [ndtr] is still normal, its logarithm; below,
   [log_ndtr_far]. One [ndtr] runs, at [-|x|]. *)
let log_ndtr_at x =
  let t = tables x in
  let positive = cmpgt x (zeros_like x) in
  let below = lt x t.log_ndtr_below in
  let y =
    clamp (logical_not below) (where positive (neg x) x) t.log_ndtr_below
  in
  let n = ndtr_at y in
  let near =
    where positive (log1p (neg n)) (log (where positive (ones_like n) n))
  in
  let far = log_ndtr_far (clamp below x (2. *. t.log_ndtr_below)) in
  let far = where (is x Float.neg_infinity) x far in
  where below far near

(* AS 241's guess on the nearer tail [q], then one Newton step: on [erf (x/sqrt
   2)/2 = p - 1/2] where [|p - 1/2| <= 0.425], whose residual is relative to
   [x]; on [log_ndtr x = log q] in the tails, where [q] may be far below the
   least normal float. One [erfc_parts] serves both steps, at the central
   [x/sqrt 2] or at the tail's [-x/sqrt 2]. *)
let ndtri_at p =
  let t = tables p in
  let d = sub_s p 0.5 in
  let central = cmple (abs d) (lit p T.ndtri_central_below) in
  let dc = clamp central d 0. in
  let r = rsub_s T.ndtri_central_r (mul dc dc) in
  let xc =
    mul dc (div (horner t.ndtri_central_p r) (horner t.ndtri_central_q r))
  in
  let lower = lt p 0.5 in
  let q = where lower p (rsub_s 1. p) in
  let tail = logical_and (logical_not central) (cmpgt q (zeros_like q)) in
  let q = clamp tail q 0.01 in
  let log_q = log q in
  let s = sqrt (neg log_q) in
  (* AS 241's two tail rationals, one evaluated with each coefficient
     selected. *)
  let near = cmple s (lit s T.ndtri_near_below) in
  let u =
    where near
      (sub_s (clamp near s 3.) T.ndtri_near_shift)
      (sub_s (clamp (logical_not near) s 6.) T.ndtri_far_shift)
  in
  let selected near_cs far_cs =
    let acc = ref (where near (lit u near_cs.(0)) (lit u far_cs.(0))) in
    for i = 1 to Array.length near_cs - 1 do
      acc :=
        add (mul !acc u) (where near (lit u near_cs.(i)) (lit u far_cs.(i)))
    done;
    !acc
  in
  let guess =
    div
      (selected t.ndtri_near_p t.ndtri_far_p)
      (selected t.ndtri_near_q t.ndtri_far_q)
  in
  let xt = neg guess in
  let th, tl = split_sqrt1_2 xt in
  let parts = erfc_parts (where central (mul_s xc (1. /. Float.sqrt 2.)) th) in
  let far = lt xt t.log_ndtr_below in
  let n = ndtr_of th tl parts in
  let lf =
    where far
      (log_ndtr_far (clamp far xt (2. *. t.log_ndtr_below)))
      (log (where far (ones_like n) n))
  in
  (* Each step divides by the density, [exp (x^2/2) sqrt (2 pi)] centrally and
     [ndtr x / pdf x = exp (log_ndtr x + x^2/2 + log (2 pi)/2)] in the tails:
     one exponential serves both. *)
  let half_square x = mul (mul_s x 0.5) x in
  let e =
    exp
      (where central (half_square xc)
         (add_s (add lf (half_square xt)) log_sqrt_2pi))
  in
  let residual = sub (mul_s parts.erf 0.5) dc in
  let central_x =
    sub xc (mul (mul_s residual (Float.sqrt (2. *. Float.pi))) e)
  in
  let xt = sub xt (mul (sub lf log_q) e) in
  where central central_x (where lower xt (neg xt))

(* Gamma *)

(* [sin (pi f)] for [f] in [0, 1/2], a polynomial the generator fits, which asks
   nothing of [sin]'s reduction of large arguments. *)
let sinpi_half f =
  let t = tables f in
  mul f (horner t.sinpi (mul f f))

(* [x] reduced modulo 2 exactly, [|x|] folded into [0, 1/2] as [f], and whether
   it was folded from (1/2, 1]: [|sin (pi x)| = sin (pi f)]. *)
let fold_pi x =
  let reduced = sub x (mul_s (round (mul_s x 0.5)) 2.) in
  let m = abs reduced in
  let folded = cmpgt m (lit m 0.5) in
  (reduced, folded, where folded (rsub_s 1. m) m)

let sinpi_abs x =
  let _, _, f = fold_pi x in
  sinpi_half f

(* fdlibm's lgamma for [y > 0]: on (0, 2) polynomials about 1, the minimum and
   2, with [-log y] below 0.9; on [2, 8) its rational on [2, 3) after a shift
   down by a masked count; from 8 Stirling's series. *)
let lgamma_pos y =
  let below2 = lt y T.lgamma_shift_from in
  let below8 = lt y T.lgamma_stirling_from in
  let small = lt y T.lgamma_small in
  let ys = clamp below2 y 1.5 in
  let ym = clamp (logical_and (logical_not below2) below8) y 2.5 in
  let i = floor ym in
  let f = sub ym i in
  let z = ref (ones_like y) in
  for k = 2 to int_of_float T.lgamma_stirling_from - 2 do
    z :=
      where
        (cmpgt i (lit i (float_of_int k)))
        (mul !z (add_s f (float_of_int k)))
        !z
  done;
  let yb = clamp (logical_not below8) y 10. in
  (* One logarithm serves the three regions: of [y] below 0.9, of the shift's
     product on [2, 8) and of [y] from 8. *)
  let log_y = log (where small (clamp small ys 0.5) (where below8 !z yb)) in
  let r_log = where small (neg log_y) (zeros_like y) in
  let about_one_or_two = where small (ge ys T.lgamma_c) (ge ys T.lgamma_f) in
  let about_min =
    logical_and
      (logical_not about_one_or_two)
      (where small (ge ys T.lgamma_b) (ge ys T.lgamma_e))
  in
  let t0 =
    clamp about_one_or_two (where small (rsub_s 1. ys) (rsub_s 2. ys)) 0.1
  in
  let p0 = sub (mul t0 (horner T.lgamma_a t0)) (mul_s t0 0.5) in
  let t1 =
    clamp about_min
      (where small (sub_s ys (T.lgamma_tc -. 1.)) (sub_s ys T.lgamma_tc))
      0.
  in
  let p1 =
    add_s
      (sub_s (mul (mul t1 t1) (horner T.lgamma_t t1)) T.lgamma_tt)
      T.lgamma_tf
  in
  let t2 =
    clamp
      (logical_not (logical_or about_one_or_two about_min))
      (where small ys (sub_s ys 1.))
      0.1
  in
  let p2 =
    sub
      (div (mul t2 (horner T.lgamma_u t2)) (horner T.lgamma_v t2))
      (mul_s t2 0.5)
  in
  let r_small = add r_log (where about_one_or_two p0 (where about_min p1 p2)) in
  let pm =
    add (mul_s f 0.5) (div (mul f (horner T.lgamma_s f)) (horner T.lgamma_r f))
  in
  let r_mid = add pm log_y in
  let zz = recip yb in
  let w = add_s (mul zz (horner T.lgamma_w (mul zz zz))) T.lgamma_w0 in
  let r_big = add (mul (sub_s yb 0.5) (sub_s log_y 1.)) w in
  where below2 r_small (where below8 r_mid r_big)

let epsilon (type b) (x : (float, b) t) =
  match dtype x with Float64 -> epsilon_float | _ -> 0x1p-23

let euler_gamma = 0.57721566490153286061

(* [log |Gamma x|]: [-log |x| - gamma x] below [epsilon] in magnitude, where
   [sin (pi x)] would round [pi x]; below 0 the reflection [log pi - log |sin
   (pi x)| - lgamma (1 - x)]; +inf at the poles. *)
let lgamma_at x =
  let eps = epsilon x in
  let negative = lt x 0. in
  let regular = logical_and (isfinite x) (logical_not (is x 0.)) in
  let y = clamp regular (where negative (rsub_s 1. x) x) 1. in
  let r = lgamma_pos y in
  let s = sinpi_abs (clamp negative x (-0.5)) in
  let pole = is s 0. in
  let reflected =
    sub (rsub_s (Stdlib.log Float.pi) (log (where pole (ones_like s) s))) r
  in
  let inf = lit x Float.infinity in
  let r = where negative (where pole inf reflected) r in
  let tiny = logical_and regular (lt (abs x) eps) in
  let xt = clamp tiny x (eps /. 2.) in
  let r = where tiny (sub (neg (log (abs xt))) (mul_s xt euler_gamma)) r in
  where regular r (where (is x Float.neg_infinity) (lit x Float.nan) inf)

(* digamma for [y > 0]: on [1, 2] [(y - root) g (y - 3/2)]; below 1 and from 2
   to the table's threshold the recurrence [psi y = psi (y - 1) + 1/(y - 1)]
   maps [y] into [1, 2] by a masked count; from the threshold [log y - 1/(2y) -
   sum_k B_2k / (2k y^2k)]. *)
let digamma_pos y =
  let t = tables y in
  let below1 = lt y 1. and asymptotic = ge y t.digamma_from in
  let ym = clamp (logical_not asymptotic) y 1.5 in
  let n = where (ge ym 2.) (sub_s (floor ym) 1.) (zeros_like ym) in
  let c = where (lt ym 1.) (add_s ym 1.) (sub ym n) in
  let core =
    mul
      (sub_s (sub_s c t.digamma_root_hi) t.digamma_root_lo)
      (horner t.digamma_core (sub_s c 1.5))
  in
  let shift = ref (zeros_like ym) in
  for k = 1 to int_of_float t.digamma_from - 2 do
    let on = cmpge n (lit n (float_of_int k)) in
    let d = where on (sub_s ym (float_of_int k)) (ones_like ym) in
    shift := add !shift (where on (recip d) (zeros_like ym))
  done;
  let shift = where below1 (neg (recip (clamp below1 ym 0.5))) !shift in
  let ya = clamp asymptotic y (2. *. t.digamma_from) in
  let v = recip (mul ya ya) in
  let far =
    sub (sub (log ya) (div (lit ya 0.5) ya)) (mul v (horner t.digamma_series v))
  in
  where asymptotic far (add shift core)

(* digamma: [-1/x - gamma] below [epsilon] in magnitude, where the reflection's
   [cot] would form [1/sin^2] past the range in its derivative; below 0 the
   reflection [psi (1 - x) - pi cot (pi x)], NaN at the poles; [-inf] at [+0]
   and [+inf] at [-0]. *)
let digamma_at x =
  let eps = epsilon x in
  let negative = lt x 0. in
  let zero = is x 0. in
  let regular = logical_and (isfinite x) (logical_not zero) in
  let tiny = logical_and regular (lt (abs x) eps) in
  let y =
    clamp
      (logical_and regular (logical_not tiny))
      (where negative (rsub_s 1. x) x)
      1.
  in
  let r = digamma_pos y in
  let xn = clamp (logical_and negative (logical_not tiny)) x (-0.5) in
  let reduced, folded, f = fold_pi xn in
  let sin_f = sinpi_half f in
  let cos_f = sinpi_half (rsub_s 0.5 f) in
  let pole = is sin_f 0. in
  let cot = div cos_f (where pole (ones_like sin_f) sin_f) in
  let cot = where (logical_xor folded (lt reduced 0.)) (neg cot) cot in
  let reflected = sub r (mul_s cot Float.pi) in
  let r = where negative (where pole (lit x Float.nan) reflected) r in
  let xt = clamp tiny x (eps /. 2.) in
  let r = where tiny (sub_s (neg (recip xt)) euler_gamma) r in
  let edge =
    where zero
      (where
         (lt (recip x) 0.)
         (lit x Float.infinity) (lit x Float.neg_infinity))
      (where (is x Float.infinity) x (lit x Float.nan))
  in
  where regular r edge

(* fdlibm's [w] less its constant: Stirling's correction [s x = lgamma x - (x -
   1/2) log x + x - log (2 pi)/2], a polynomial in [1/x]. *)
let stirling x =
  let z = recip x in
  mul z (horner T.lgamma_w (mul z z))

(* [s q - s (q + p)], each power's difference written as [1/q^n - 1/(q + p)^n =
   p z1 z2 (z1^(n-1) + z1^(n-2) z2 + ... + z2^(n-1))] for [z1 = 1/q] and [z2 =
   1/(q + p)], so that its derivative in [q] is not lost where [q + p] rounds to
   [q]. *)
let stirling_difference q p =
  let z1 = recip q and z2 = recip (add q p) in
  (* [h] is the sum of the [m + 1] products of [m] factors [z1] or [z2]. *)
  let h = ref (ones_like q) and z1m = ref (ones_like q) in
  let terms = Array.length T.lgamma_w in
  let sum = ref (zeros_like q) in
  for m = 0 to (2 * terms) - 2 do
    if m > 0 then (
      z1m := mul !z1m z1;
      h := add (mul z2 !h) !z1m);
    if m mod 2 = 0 then
      (* [T.lgamma_w] is highest degree first: [w_k] for [n = 2k - 1] is at
         [terms - k]. *)
      sum := add !sum (mul_s !h T.lgamma_w.(terms - 1 - (m / 2)))
  done;
  mul (mul p (mul z1 z2)) !sum

(* [log B(a, b)], with [p <= q] the arguments, is [lgamma p - (lgamma (p + q) -
   lgamma q)]. The difference is written from [Q = q + n], [n] the masked count
   that takes [q] past 8, as Stirling's series at [Q], [p (log Q - 1) + (p + Q -
   1/2) log1p (p/Q)] and the corrections' difference, less the shift's [log1p]
   of [prod_k (1 + p/(q + k)) - 1], accumulated without cancelling. Each term is
   a function of [p/q] or of a difference written as one, so the derivative in
   either argument keeps its precision when [p] is far below [q]. From [p = 8],
   [lgamma p] is Stirling's series too: [(p - 1/2) log (p/(p + q)) + (q - 1/2)
   log (q/(p + q)) - log (p + q)/2 + log (2 pi)/2] and the corrections, no term
   of which overflows before the result does. *)
let lbeta_at a b =
  let le = cmple a b in
  let p0 = where le a b and q0 = where le b a in
  let valid = logical_and (cmpgt p0 (zeros_like p0)) (lt q0 Float.infinity) in
  let p = clamp valid p0 1. and q = clamp valid q0 1. in
  let large = ge p T.lgamma_stirling_from in
  let n =
    where
      (ge q T.lgamma_stirling_from)
      (zeros_like q)
      (rsub_s T.lgamma_stirling_from (floor q))
  in
  let big_q = add q n in
  let log_q = log big_q in
  (* [log1p h] for [h = p/Q] as [h r], with [r = log1p h / h = 1 - h/2] where
     [h^2] is below [epsilon], which keeps every intermediate normal where the
     result is. *)
  let h = div p big_q in
  let series = lt h (Float.sqrt (epsilon h)) in
  let hl = clamp (logical_not series) h 0.5 in
  let r = where series (rsub_s 1. (mul_s h 0.5)) (div (log1p hl) hl) in
  let l1 = mul h r in
  let corrections = stirling_difference big_q p in
  let e = ref (zeros_like q) in
  for k = 0 to int_of_float T.lgamma_stirling_from - 1 do
    let hk = div p (add_s q (float_of_int k)) in
    let hk = where (cmpgt n (lit n (float_of_int k))) hk (zeros_like hk) in
    e := add !e (add hk (mul !e hk))
  done;
  let shifts = log1p !e in
  let difference =
    sub
      (add
         (mul p (sub_s log_q 1.))
         (mul (mul p (add_s (div (sub_s p 0.5) big_q) 1.)) r))
      (add corrections shifts)
  in
  let small = sub (lgamma_pos (clamp (logical_not large) p 1.)) difference in
  let pl = clamp large p 10. in
  let large_r =
    add
      (sub
         (sub
            (mul (sub_s pl 0.5) (sub (log (div pl big_q)) l1))
            (mul (sub_s big_q 0.5) l1))
         (mul_s (add log_q l1) 0.5))
      (add_s (add (stirling pl) corrections) log_sqrt_2pi)
  in
  let r = where large large_r small in
  let nan = lit a Float.nan in
  let edge =
    where
      (cmpgt p0 (zeros_like p0))
      (lit a Float.neg_infinity)
      (where (is q0 Float.infinity) nan (lit a Float.infinity))
  in
  let r = where valid r edge in
  where (logical_or (logical_or (isnan a) (isnan b)) (lt p0 0.)) nan r

(* Modified Bessel functions *)

(* [sum_k cs.(k) T_k t] by Clenshaw's recurrence. *)
let clenshaw cs t =
  let n = Array.length cs in
  let two_t = add t t in
  let b1 = ref (lit t cs.(n - 1)) and b2 = ref (zeros_like t) in
  for k = n - 2 downto 1 do
    let b = sub (add (lit t cs.(k)) (mul two_t !b1)) !b2 in
    b2 := !b1;
    b1 := b
  done;
  sub (add (lit t cs.(0)) (mul t !b1)) !b2

(* [i0e] and [i1e] at [a = |x|], taken by selection so that [-0] takes the
   right-hand branch with [+0]. On [0, bessel_split], [near] is the series of
   [q] with [i0e a h = 1 + a q], so that [i0e 0] is 1, or of [i1e a / a h^2],
   for [h = 1 + bessel_weight a], the weight keeping the series' terms of the
   order of its value; above, [far] is the series of [sqrt a i0e a] or [sqrt a
   i1e a]. Each region runs on its clamped [a]. *)
let bessel_parts x near far =
  let a = where (lt x 0.) (neg x) x in
  let inside = cmple a (lit a T.bessel_split) in
  let an = clamp inside a (T.bessel_split /. 2.) in
  let h = add_s (mul_s an T.bessel_weight) 1. in
  let tn = sub_s (mul_s an (2. /. T.bessel_split)) 1. in
  let af = clamp (logical_not inside) a (2. *. T.bessel_split) in
  let tf = sub_s (div (lit af (2. *. T.bessel_split)) af) 1. in
  (inside, an, clenshaw near tn, h, div (clenshaw far tf) (sqrt af))

let i0e_at x =
  let t = tables x in
  let inside, a, q, h, far = bessel_parts x t.i0e_near t.i0e_far in
  where inside (div (add_s (mul a q) 1.) h) far

let i1e_at x =
  let t = tables x in
  let inside, _, near, h, far = bessel_parts x t.i1e_near t.i1e_far in
  where inside (mul x (div near (mul h h))) (where (lt x 0.) (neg far) far)

(* The functions *)

let nan_through x r = where (isnan x) x r
let erfc x = real_at_float32 { r = (fun x -> nan_through x (erfc_at x)) } x
let ndtr x = real_at_float32 { r = (fun x -> nan_through x (ndtr_at x)) } x

let log_ndtr x =
  real_at_float32 { r = (fun x -> nan_through x (log_ndtr_at x)) } x

let ndtri p =
  let ndtri p =
    let r = ndtri_at p in
    let r = where (is p 0.) (lit p Float.neg_infinity) r in
    let r = where (is p 1.) (lit p Float.infinity) r in
    where
      (logical_or (logical_or (lt p 0.) (cmpgt p (lit p 1.))) (isnan p))
      (lit p Float.nan) r
  in
  real_at_float32 { r = ndtri } p

let lgamma x = real_at_float32 { r = (fun x -> nan_through x (lgamma_at x)) } x

let digamma x =
  real_at_float32 { r = (fun x -> nan_through x (digamma_at x)) } x

let erfinv p =
  let erfinv p =
    let edge = cmpeq (abs p) (lit p 1.) in
    let inf =
      where (lt p 0.) (lit p Float.neg_infinity) (lit p Float.infinity)
    in
    let r = where edge inf (erfinv_at p) in
    where (logical_or (cmpgt (abs p) (lit p 1.)) (isnan p)) (lit p Float.nan) r
  in
  real_at_float32 { r = erfinv } p

let i0e x = real_at_float32 { r = (fun x -> nan_through x (i0e_at x)) } x
let i1e x = real_at_float32 { r = (fun x -> nan_through x (i1e_at x)) } x

let lbeta a b =
  let a, b = broadcasted a b in
  if narrow (dtype a) then
    let f32 = cast Nx_dtype.float32 in
    cast (dtype a) (lbeta_at (f32 a) (f32 b))
  else lbeta_at a b
