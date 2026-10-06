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

(* fdlibm's polynomials on (0, 2): [lgamma (2 - t)] about 2, [lgamma (tc + t)]
   about the minimum [tc] and [lgamma (1 + t)] about 1, each relative to the
   zero it starts from. *)
let lgamma_about_two t = sub (mul t (horner T.lgamma_a t)) (mul_s t 0.5)

let lgamma_about_min t =
  add_s (sub_s (mul (mul t t) (horner T.lgamma_t t)) T.lgamma_tt) T.lgamma_tf

let lgamma_about_one t =
  sub (div (mul t (horner T.lgamma_u t)) (horner T.lgamma_v t)) (mul_s t 0.5)

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
  let p0 = lgamma_about_two t0 in
  let t1 =
    clamp about_min
      (where small (sub_s ys (T.lgamma_tc -. 1.)) (sub_s ys T.lgamma_tc))
      0.
  in
  let p1 = lgamma_about_min t1 in
  let t2 =
    clamp
      (logical_not (logical_or about_one_or_two about_min))
      (where small ys (sub_s ys 1.))
      0.1
  in
  let p2 = lgamma_about_one t2 in
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

let min_normal (type b) (x : (float, b) t) =
  match dtype x with Float64 -> 0x1p-1022 | _ -> 0x1p-126

let max_float_of (type b) (x : (float, b) t) =
  match dtype x with Float64 -> max_float | _ -> 0x1.fffffep127

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

(* Incomplete gamma *)

(* [log Γ(1 + a) / a] for [a] in (0, 1], from fdlibm's polynomials: about 2 in
   [1 - a], about the minimum, and about 1 in [a], where it is a ratio of
   polynomials that holds its precision for a subnormal [a]. *)
let lgamma1p_ratio a =
  let two = ge a T.lgamma_c in
  let low = logical_not two in
  let about_min = logical_and low (ge a T.lgamma_b) in
  let about_one = logical_and low (logical_not about_min) in
  let a1 = clamp about_one a 0.1 in
  let near_one =
    sub_s (div (horner T.lgamma_u a1) (horner T.lgamma_v a1)) 0.5
  in
  let far =
    where two
      (lgamma_about_two (clamp two (rsub_s 1. a) 0.1))
      (lgamma_about_min (clamp about_min (sub_s a (T.lgamma_tc -. 1.)) 0.))
  in
  where about_one near_one (div far (clamp (logical_not about_one) a 0.5))

(* [S w = sum_j w^(j - 1) / (2j + 1)] to [n] terms: [atanh z = z (1 + z^2 S
   z^2)]. *)
let atanh_series (t : T.t) n w =
  let cs = t.igamma_atanh in
  horner (Array.sub cs (Array.length cs - n) n) w

(* Stirling's correction [s a = lgamma a - (a - 1/2) log a + a - log (2 pi)/2]
   for [a >= 1]: fdlibm's polynomial from 8, and below, shifted past 8 by the
   masked count [n]: [s a = s (a + n) + sum_(k < n) (s (a + k) - s (a + k +
   1))], each difference [z^2 S z^2] for [z = 1/(2 (a + k) + 1)], with no
   cancellation. *)
let stirling_from_one (t : T.t) a =
  let n = where (ge a 8.) (zeros_like a) (rsub_s 8. (floor a)) in
  let s = ref (stirling (add a n)) in
  Array.iteri
    (fun k terms ->
      let z = recip (add_s (mul_s a 2.) (float_of_int ((2 * k) + 1))) in
      let w = mul z z in
      let d = mul w (atanh_series t (int_of_float terms) w) in
      s := add !s (where (cmpgt n (lit n (float_of_int k))) d (zeros_like d)))
    t.igamma_shift_terms;
  !s

let log_2pi = Stdlib.log (2. *. Float.pi)

(* [erfc w e^(w^2)] for [w >= 0]: erf's rationals below 1.25, fdlibm's tails
   [exp (R/S - 0.5625) / w] up to [erfcx_far], past which they lose their
   precision, and the asymptotic series beyond. *)
let erfcx_at w =
  let t = tables w in
  let small = lt w 0.84375 and below = lt w 1.25 in
  let near = logical_and (logical_not small) below in
  let far = ge w T.erfcx_far in
  let tail = logical_and (logical_not below) (logical_not far) in
  let ws = clamp small w 0.5 in
  let z = mul ws ws in
  let y = div (horner T.erf_small_p z) (horner T.erf_small_q z) in
  let xy = mul ws y in
  let e_small =
    where (lt ws 0.25)
      (rsub_s 1. (add ws xy))
      (rsub_s 0.5 (add xy (sub_s ws 0.5)))
  in
  let s = sub_s (clamp near w 1.) 1. in
  let pq = div (horner T.erf_near_p s) (horner T.erf_near_q s) in
  let e_near = rsub_s (1. -. T.erx) pq in
  let wb = clamp below w 0.5 in
  let r_below = mul (exp (mul wb wb)) (where small e_small e_near) in
  let wt = clamp tail w 2. in
  let v = recip (mul wt wt) in
  let mid = lt wt (1. /. 0.35) in
  let rs =
    div
      (where mid (horner T.erfc_mid_p v) (horner T.erfc_far_p v))
      (where mid (horner T.erfc_mid_q v) (horner T.erfc_far_q v))
  in
  let r_tail = div (exp (sub_s rs 0.5625)) wt in
  let wf = clamp far w (2. *. T.erfcx_far) in
  let r_far =
    div
      (horner t.erfcx_series (recip (mul_s (mul wf wf) 2.)))
      (mul_s wf (Float.sqrt Float.pi))
  in
  where below r_below (where tail r_tail r_far)

(* Temme's sum [T = sum_k c_k(eta) u^k], [u = 1/a], each [c_k] a polynomial in
   [eta] to the degree the table gives. *)
let temme_sum (t : T.t) eta u =
  let cs = t.igamma_temme in
  let polys =
    let off = ref 0 in
    Array.map
      (fun d ->
        let n = int_of_float d + 1 in
        let c = horner (Array.sub cs !off n) eta in
        off := !off + n;
        c)
      t.igamma_temme_degrees
  in
  let k = Array.length polys in
  let acc = ref polys.(k - 1) in
  for i = k - 2 downto 0 do
    acc := add polys.(i) (mul u !acc)
  done;
  !acc

(* [log (1 - e^l)] for [l < 0]. *)
let log1mexp l =
  let ln2 = Stdlib.log 2. in
  let near = cmpgt l (lit l (-.ln2)) in
  where near
    (log (neg (expm1 (clamp near l (-1.)))))
    (log1p (neg (exp (clamp (logical_not near) l (-1.)))))

(* The incomplete gamma ratio [P(a, x)], or [Q(a, x) = 1 - P(a, x)] where
   [upper], in logs, for finite [a > 0] and [x >= 0], [log_x] being [log x],
   finite where [x] underflows; and [log (x^a e^-x / Γ(a + 1))], from which an
   inverse forms the density. After DiDonato and Morris's GRATIO (ACM TOMS 654),
   each region of which computes the tail it keeps precise:

   - a < 1, x < 1.1: [P = x^a / Γ(1 + a) (1 - j)] and [Q = e^c j - expm1 c], [c
   = a log x - log Γ(1 + a)], [j] a series in [x], both precise; - x below max
   (a, 1.1): [P] by its series; - beyond: [Q] by Legendre's continued fraction,
   evaluated backward; - a >= 20, |x/a - 1| <= 0.4: Temme's uniform expansion of
   the smaller tail, [e^-y (erfc w e^(w^2) / 2 ± T / sqrt (2 pi a))] for [y =
   w^2 = a eta^2 / 2], [eta = v sqrt (2 kappa)] analytic through [x = a].

   The other tail is [log (1 - e^l)] of the computed one, never above 0.9. The
   prefactor's exponent below [a = 1] is [a log x - x - log Γ(1 + a)]; from 1 it
   is [-bd0 - log (2 pi a)/2 - s a], [bd0 = a (mu - log1p mu)] for [mu = x/a -
   1] (Loader), which in [v = (x - a)/(x + a)] is [a v^2 kappa], [kappa = 2/(1 -
   v) - 2 v S v^2], up to [|v| = 1/2], and [x - a - a log (x/a)] beyond. *)
let igamma_core a x log_x upper =
  let t = tables a in
  let corner = logical_and (lt a 1.) (lt x T.igamma_corner) in
  let temme =
    logical_and (ge a T.igamma_temme_from)
      (cmple (abs (sub x a)) (mul_s a T.igamma_temme_width))
  in
  let below = logical_or (lt x T.igamma_corner) (cmplt x a) in
  let series = logical_and (logical_not temme) (logical_and (ge a 1.) below) in
  let cf = logical_not (logical_or (logical_or corner temme) series) in
  (* The prefactor. *)
  let small_a = lt a 1. in
  let a_lo = clamp small_a a 0.5 in
  let lg1p_ratio = lgamma1p_ratio a_lo in
  let direct = sub (mul a_lo (sub log_x lg1p_ratio)) x in
  let a_hi = clamp (logical_not small_a) a 1. in
  let v = div (sub x a_hi) (add x a_hi) in
  let near = cmple (abs v) (lit v T.igamma_near_v) in
  let vn = clamp near v 0. in
  let w = mul vn vn in
  let s = atanh_series t (Array.length t.igamma_atanh) w in
  let kappa = sub (div (lit vn 2.) (rsub_s 1. vn)) (mul_s (mul vn s) 2.) in
  let lam = div x a_hi in
  let tiny = lt lam (min_normal lam) in
  let log_lam =
    where tiny
      (sub log_x (log a_hi))
      (log (clamp (logical_not (logical_or near tiny)) lam 4.))
  in
  let bd0 =
    where near (mul (mul a_hi w) kappa) (sub (sub x a_hi) (mul a_hi log_lam))
  in
  let log_a = log a_hi in
  let lr =
    where small_a direct
      (neg
         (add
            (add bd0 (mul_s (add_s log_a log_2pi) 0.5))
            (stirling_from_one t a_hi)))
  in
  (* P's series. *)
  let a_s = clamp series a 2. and x_s = clamp series x 1. in
  let sum = ref (ones_like x) in
  for n = int_of_float t.igamma_series_terms downto 1 do
    sum := add_s (mul !sum (div x_s (add_s a_s (float_of_int n)))) 1.
  done;
  let lp_series = add lr (log !sum) in
  (* Q's continued fraction. *)
  let a_c = clamp cf a 1. and x_c = clamp cf x 2. in
  let xa = sub x_c a_c in
  let f = ref (zeros_like x) in
  for n = int_of_float t.igamma_cf_depth downto 1 do
    let k = float_of_int n in
    f := div (mul_s (rsub_s k a_c) k) (sub (add_s xa ((2. *. k) +. 1.)) !f)
  done;
  let lq_cf = sub (add lr (log a_c)) (log (sub (add_s xa 1.) !f)) in
  (* The corner. *)
  let x_k = clamp corner x 0.5 in
  let log_xk = clamp corner log_x (Stdlib.log 0.5) in
  let jsum = ref (zeros_like x) in
  let factorial = ref 1. in
  let terms = int_of_float t.igamma_corner_terms in
  for n = 1 to terms do
    factorial := !factorial *. float_of_int n
  done;
  for n = terms downto 1 do
    jsum :=
      sub
        (recip (mul_s (add_s a_lo (float_of_int n)) !factorial))
        (mul x_k !jsum);
    factorial := !factorial /. float_of_int n
  done;
  let xj = mul x_k !jsum in
  let c_a = sub log_xk lg1p_ratio in
  let c = mul a_lo c_a in
  let j = mul a_lo xj in
  (* Below [2^-60], where [Q] is at most about [745 a], [log P = a (c/a - x J)]
     and [Q / a = e^c x J - (c / a) (1 + c/2)], to within [a] relatively, keep
     their precision and sign as [a] becomes subnormal, and [log a] the
     derivative. *)
  let tiny_a = lt a_lo 0x1p-60 in
  let lp_corner =
    where tiny_a (mul a_lo (sub c_a xj)) (add c (log1p (neg j)))
  in
  let q_a = sub (mul (exp c) xj) (mul c_a (add_s (mul_s c 0.5) 1.)) in
  let lq_corner =
    where tiny_a
      (add (log a_lo) (log (where tiny_a q_a (ones_like q_a))))
      (log (sub (mul (exp c) j) (expm1 c)))
  in
  (* Temme's expansion. *)
  let eta = mul vn (sqrt (mul_s kappa 2.)) in
  let eta = clamp temme eta 0. and a_t = clamp temme a_hi 40. in
  let q_native = cmpge eta (zeros_like eta) in
  let wt = mul (where q_native eta (neg eta)) (sqrt (mul_s a_t 0.5)) in
  let tt = temme_sum t eta (recip a_t) in
  let scaled =
    add
      (mul_s (erfcx_at wt) 0.5)
      (div (where q_native tt (neg tt)) (sqrt (mul_s a_t (2. *. Float.pi))))
  in
  let lt_temme = sub (log scaled) (mul wt wt) in
  (* Each region's tail, and the other. *)
  let native = where temme lt_temme (where series lp_series lq_cf) in
  let native_upper = where temme q_native (logical_not series) in
  let r = where (logical_xor native_upper upper) (log1mexp native) native in
  (where corner (where upper lq_corner lp_corner) r, lr)

(* [log P(a, x)], or [log Q(a, x)] where [upper], with the domain's edges: [P]
   is 0 at [x = 0] and where [a = inf], 1 at [x = inf]; NaN outside [a > 0, x >=
   0], at a NaN and at [a = x = inf]. *)
let log_gammainc_at a x upper =
  let a_inf = is a Float.infinity and x_inf = is x Float.infinity in
  let zero = is x 0. in
  let nan =
    logical_or
      (logical_or (isnan a) (isnan x))
      (logical_or
         (logical_or (cmple a (zeros_like a)) (lt x 0.))
         (logical_and a_inf x_inf))
  in
  let regular =
    logical_not (logical_or (logical_or nan a_inf) (logical_or x_inf zero))
  in
  let a = clamp regular a 1. and x = clamp regular x 1. in
  let r, _ = igamma_core a x (log x) upper in
  let p_zero = logical_or a_inf zero in
  let edge =
    where (logical_xor p_zero upper) (lit r Float.neg_infinity) (zeros_like r)
  in
  where regular r (where nan (lit r Float.nan) edge)

(* [P] or [Q] as the exponential of its logarithm. At [x = 0], where [P(a, x) =
   x^a / Γ(a + 1) + O(x^(a + 1))], it is written so that its slope in [x] is
   that limit's: 1 at [a = 1] and 0 above. *)
let gammainc_at a x upper =
  let r = exp (log_gammainc_at a x upper) in
  let at_zero =
    logical_and (is x 0.) (logical_and (cmpgt a (zeros_like a)) (isfinite a))
  in
  let p = where (is a 1.) x (zeros_like x) in
  where at_zero (where upper (rsub_s 1. p) p) r

(* [log Γ(1 + a)] for [a >= 1], as precise as a guess needs. *)
let lgamma_a1 (t : T.t) a =
  add
    (sub (mul (add_s a 0.5) (log a)) a)
    (add_s (stirling_from_one t a) log_sqrt_2pi)

(* Horner on tensor coefficients, the leading one first. *)
let horner_t cs t =
  List.fold_left (fun acc c -> add (mul acc t) c) (List.hd cs) (List.tl cs)

(* GAMINV's asymptotic form for a large [x] from [y = -log (q Γ(a))]: [x = y +
   c1 + c2/y + ... + c5/y^4], each [c_k] a polynomial in [c1 = (a - 1) log y]
   whose coefficients are polynomials in [a]. *)
let gaminv_asymptotic a y =
  let s = rsub_s 1. a in
  let c1 = neg (mul s (log y)) in
  let p cs = horner cs a in
  let c2 = neg (mul s (add_s c1 1.)) in
  let c3 = mul s (horner_t [ lit a 0.5; rsub_s 2. a; p [| -1.5; 2.5 |] ] c1) in
  let c4 =
    neg
      (mul s
         (horner_t
            [
              lit a (1. /. 3.);
              p [| -1.5; 2.5 |];
              p [| 1.; -6.; 7. |];
              p [| 11. /. 6.; -46. /. 6.; 47. /. 6. |];
            ]
            c1))
  in
  let c5 =
    neg
      (mul s
         (horner_t
            [
              lit a (-0.25);
              p [| 11. /. 6.; -17. /. 6. |];
              p [| -3.; 13.; -13. |];
              p [| 1.; -12.5; 36.; -30.5 |];
              p [| 25. /. 12.; -195. /. 12.; 477. /. 12.; -379. /. 12. |];
            ]
            c1))
  in
  add (horner_t [ c5; c4; c3; c2; c1 ] (recip y)) y

(* DiDonato and Morris's initial approximation (GAMINV, TOMS 654) to [log x],
   for finite [a > 0], the smaller tail's logarithm [l] and which tail it is,
   [lp] and [lq] the logarithms of [P] and [Q]: within about 0.1 of [log x].
   Below [a = 1], forms in [B = q Γ(1 + a) / a] for a large [x], and a power
   [x^a / Γ(1 + a)] on [P] otherwise; from 1, a normal quantile's expansion
   about [a], then corrections in either tail. *)
let gaminv_guess t a l q_tail lp lq =
  let small = lt a 1. in
  let a_lo = clamp small a 0.5 in
  let lg1_lo = mul a_lo (lgamma1p_ratio a_lo) in
  (* Below 1, a large [x] where [B < 0.45], [y = -log B]. *)
  let y = neg (add lq (sub lg1_lo (log a_lo))) in
  let b = exp (neg y) in
  let large_x = logical_and small (lt b 0.45) in
  let yl = clamp large_x y 1. and al = clamp large_x a 0.5 in
  let bl = exp (neg yl) in
  let s = rsub_s 1. al in
  let tt = sub yl (mul s (log yl)) in
  let g_tiny_a =
    let e = neg (add_s bl euler_gamma) in
    add e (mul (exp e) (exp (exp e)))
  in
  let g_mid =
    log (sub (sub yl (mul s (log tt))) (log1p (div s (add_s tt 1.))))
  in
  let u =
    div
      (add
         (mul (add tt (mul_s (rsub_s 3. al) 2.)) tt)
         (mul (rsub_s 2. al) (rsub_s 3. al)))
      (add_s (mul (add tt (rsub_s 5. al)) tt) 2.)
  in
  let g_low_b = log (sub (sub yl (mul s (log tt))) (log u)) in
  let g_asym = log (gaminv_asymptotic al (clamp (lt bl 0.01) yl 100.)) in
  let g_large =
    where
      (logical_and (lt al 0.3) (ge bl 0.35))
      g_tiny_a
      (where (ge bl 0.15) g_mid (where (cmpgt bl (lit bl 0.01)) g_low_b g_asym))
  in
  (* Below 1 otherwise, the power form on [P], divided by [1 - x/(a + 1)]. *)
  let power =
    where
      (lt (sub lq y) (Stdlib.log 1e-8))
      (neg (add_s (div (exp lq) a_lo) euler_gamma))
      (div (add lp lg1_lo) a_lo)
  in
  let ratio = div (exp power) (add_s a_lo 1.) in
  let g_power =
    sub power (log1p (neg (where (lt ratio 0.9) ratio (lit ratio 0.9))))
  in
  let below_one = where large_x g_large g_power in
  (* From 1, the normal quantile's expansion. *)
  let ah = clamp (logical_not small) a 2. in
  let lg1 = lgamma_a1 t ah in
  let y = neg (add lq (sub lg1 (log ah))) in
  let r = sqrt (mul_s l (-2.)) in
  let n = sub r (div (horner T.igamma_guess_p r) (horner T.igamma_guess_q r)) in
  let n = where q_tail n (neg n) in
  let ra = sqrt ah and n2 = mul n n in
  let xc =
    add
      (add
         (add (add ah (mul n ra)) (mul_s (sub_s n2 1.) (1. /. 3.)))
         (div (mul n (sub_s n2 7.)) (mul_s ra 36.)))
      (sub
         (div
            (mul n (sub_s (mul (add_s (mul_s n2 9.) 256.) n2) 433.))
            (mul_s (mul ah ra) 38880.))
         (div (sub_s (mul (add_s (mul_s n2 3.) 7.) n2) 16.) (mul_s ah 810.)))
  in
  let xc = where (cmpgt xc (zeros_like xc)) xc (lit xc 1e-30) in
  let ap1 = add_s ah 1. in
  (* The upper tail beyond [3a]: [x = y + (a - 1) log x - log1p (-(a - 1)/(x +
     1))] twice from [xc], or, where [y] is large, the asymptotic form. *)
  let beyond = logical_and q_tail (cmpge xc (mul_s ah 3.)) in
  let yh = clamp beyond y 10. and xb = clamp beyond xc 10. in
  let ab = clamp beyond ah 2. in
  let am1 = sub_s ab 1. in
  let iterate x =
    sub (add yh (mul am1 (log x))) (log1p (neg (div am1 (add_s x 1.))))
  in
  let far =
    logical_and (ge yh 0.)
      (cmpge yh
         (mul_s
            (where (cmpgt (mul ab am1) (lit ab 2.)) (mul ab am1) (lit ab 2.))
            (Stdlib.log 10.)))
  in
  let g_beyond =
    where far
      (log (gaminv_asymptotic (clamp far ab 2.) (clamp far yh 100.)))
      (log (iterate (iterate (clamp (logical_not far) xb 10.))))
  in
  (* The lower tail below [0.7 (a + 1)]: [w = log p + log Γ(a + 1)], four
     fixed-point steps below [0.15 (a + 1)] and, where they end above [0.01 (a +
     1)] or [xc] is above [0.15 (a + 1)], one step on P's series. *)
  let low = logical_and (logical_not q_tail) (cmple xc (mul_s ap1 0.7)) in
  let w = add lp lg1 in
  let a_w = clamp low ah 2. and w = clamp low w (-1.) in
  let ap1 = add_s a_w 1. and ap2 = add_s a_w 2. and ap3 = add_s a_w 3. in
  let fixed x =
    div (add w (sub x (log1p (mul (div x ap1) (add_s (div x ap2) 1.))))) a_w
  in
  let x1 = exp (div w a_w) in
  let x2 = exp (fixed x1) in
  let x3 = exp (fixed x2) in
  let y4 =
    div
      (add w
         (sub x3
            (log1p
               (mul (div x3 ap1)
                  (add_s (mul (div x3 ap2) (add_s (div x3 ap3) 1.)) 1.)))))
      a_w
  in
  let x4 = exp y4 in
  let tiny_x = cmple xc (mul_s ap1 0.15) in
  let done_ = logical_and tiny_x (cmple x4 (mul_s ap1 0.01)) in
  let xs = where tiny_x x4 (clamp low xc 1.) in
  let xs = where (cmple xs (mul_s ap1 0.7)) xs (mul_s ap1 0.7) in
  let term = ref (ones_like xs) and sum = ref (ones_like xs) in
  for k = 1 to int_of_float T.igamma_guess_terms do
    term := mul !term (div xs (add_s a_w (float_of_int k)));
    sum := add !sum !term
  done;
  let tw = sub w (log !sum) in
  let x5 = exp (div (add xs tw) a_w) in
  let d = sub a_w x5 in
  let step =
    div
      (sub (sub (mul a_w (log x5)) x5) tw)
      (where (cmpgt (abs d) (lit d 1e-3)) d (lit d 1.))
  in
  let g_series =
    add (log x5)
      (log1p (neg (where (lt (abs step) 0.5) step (zeros_like step))))
  in
  let g_low = where done_ y4 g_series in
  let from_one = where beyond g_beyond (where low g_low (log xc)) in
  where small below_one from_one

(* The gamma's quantile at [p] from the lower tail, or from the upper where
   [upper], for finite [a > 0] and [p] in (0, 1): GAMINV's guess, then Halley
   steps in [y = log x] on the logarithm of the smaller tail, which stays finite
   where [x] underflows and is concave in [y], and one Newton step more in [x],
   skipped where [x] underflows, so that the result's derivative is the implicit
   one. Each step's slope [d log T / dy = ± x pdf(x) / T] is formed from the
   prefactor [igamma_core] computes. *)
let igamma_inverse a p upper =
  let t = tables a in
  let flip = cmpgt p (lit p 0.5) in
  let s = where flip (rsub_s 1. p) p in
  let q_tail = logical_xor upper flip in
  let l = log s and lc = log1p (neg s) in
  let lq = where q_tail l lc and lp = where q_tail lc l in
  let y = gaminv_guess t a l q_tail lp lq in
  let log_a = log a in
  let sign = where q_tail (lit a (-1.)) (ones_like a) in
  let y_hi = Stdlib.log (max_float_of a) in
  let y_lo =
    div (lit a (-.max_float_of a /. 4.)) (where (lt a 1.) (ones_like a) a)
  in
  (* [log T - l] at [y] and its slope in [y]. *)
  let residual y =
    let x = exp y in
    let lt_, lr = igamma_core a x y q_tail in
    (x, sub lt_ l, mul sign (exp (sub (add lr log_a) lt_)))
  in
  let halley y =
    let y =
      where (cmpgt y (lit y y_hi)) (lit y y_hi) (where (cmplt y y_lo) y_lo y)
    in
    let x, g, slope = residual y in
    let d = div g slope in
    let den = rsub_s 1. (div (mul g (sub (sub a x) slope)) (mul_s slope 2.)) in
    let d = where (cmpgt den (lit den 0.5)) (div d den) d in
    (* A step beyond 1 in [log x] leaves the guess's reach, which happens only
       where [x] near [a] is finer than [y]'s precision, past [a = 2^20]. *)
    sub y
      (where
         (cmple (abs d) (lit d 1.))
         d
         (where (lt d 0.) (lit d (-1.)) (ones_like d)))
  in
  let y = ref y in
  for _ = 1 to int_of_float t.igamma_halley_steps do
    y := halley !y
  done;
  let x, g, slope = residual !y in
  let d = div g slope in
  let polish = logical_and (ge x (min_normal x)) (lt (abs d) 0.5) in
  where polish (mul x (rsub_s 1. d)) x

(* The quantile with the domain's edges: 0 at the probability that is 0 at [x =
   0], [inf] at the other end and where [a = inf], NaN outside [a > 0] and [p]
   in [0, 1]. *)
let gammaincinv_at a p upper =
  let nan =
    logical_or
      (logical_or (isnan a) (isnan p))
      (logical_or
         (cmple a (zeros_like a))
         (logical_or (lt p 0.) (cmpgt p (lit p 1.))))
  in
  let zero = where upper (is p 1.) (is p 0.) in
  let inf =
    logical_or (where upper (is p 0.) (is p 1.)) (is a Float.infinity)
  in
  let regular = logical_not (logical_or nan (logical_or zero inf)) in
  let a = clamp regular a 1. and p = clamp regular p 0.5 in
  let r = igamma_inverse a p upper in
  where regular r
    (where nan (lit r Float.nan)
       (where zero (zeros_like r) (lit r Float.infinity)))

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

(* A function of two floats, its arguments broadcast, narrow floats at
   float32. *)
type real2 = { r2 : 'b. (float, 'b) t -> (float, 'b) t -> (float, 'b) t }

let real2_at_float32 c a b =
  let a, b = broadcasted a b in
  if narrow (dtype a) then
    let f32 = cast Nx_dtype.float32 in
    cast (dtype a) (c.r2 (f32 a) (f32 b))
  else c.r2 a b

let lbeta a b = real2_at_float32 { r2 = lbeta_at } a b

(* A tail as an operand: [upper] everywhere [x] is. *)
let tail x upper = full (Value.context x) Nx_dtype.bool (shape x) upper

let gammainc a x =
  real2_at_float32 { r2 = (fun a x -> gammainc_at a x (tail x false)) } a x

let gammaincc a x =
  real2_at_float32 { r2 = (fun a x -> gammainc_at a x (tail x true)) } a x

let log_gammainc a x =
  real2_at_float32 { r2 = (fun a x -> log_gammainc_at a x (tail x false)) } a x

let log_gammaincc a x =
  real2_at_float32 { r2 = (fun a x -> log_gammainc_at a x (tail x true)) } a x

let gammaincinv a p =
  real2_at_float32 { r2 = (fun a p -> gammaincinv_at a p (tail p false)) } a p

let gammainccinv a q =
  real2_at_float32 { r2 = (fun a q -> gammaincinv_at a q (tail q true)) } a q
