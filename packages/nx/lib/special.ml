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

(* [lgamma (q + p) - lgamma q] for [p, q > 0], written from [Q = q + n], [n] the
   masked count that takes [q] past 8, as Stirling's series at [Q], [p (log Q -
   1) + (p + Q - 1/2) log1p (p/Q)] and the corrections' difference, less the
   shift's [log1p] of [prod_k (1 + p/(q + k)) - 1], accumulated without
   cancelling. Each term is a function of [p/q] or of a difference written as
   one, so the derivative in either argument keeps its precision when [p] is far
   below [q]. [tail] is all but [p (log Q - 1)]. *)
type 'b shift = {
  big_q : (float, 'b) t;
  log_q : (float, 'b) t;
  l1 : (float, 'b) t; (* [log1p (p/Q)] *)
  corrections : (float, 'b) t;
  tail : (float, 'b) t;
}

let shift p q =
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
  let corrections = stirling_difference big_q p in
  let e = ref (zeros_like q) in
  for k = 0 to int_of_float T.lgamma_stirling_from - 1 do
    let hk = div p (add_s q (float_of_int k)) in
    let hk = where (cmpgt n (lit n (float_of_int k))) hk (zeros_like hk) in
    e := add !e (add hk (mul !e hk))
  done;
  let tail =
    sub
      (mul (mul p (add_s (div (sub_s p 0.5) big_q) 1.)) r)
      (add corrections (log1p !e))
  in
  { big_q; log_q; l1 = mul h r; corrections; tail }

(* [log B(a, b)], with [p <= q] the arguments, is [lgamma p - (lgamma (p + q) -
   lgamma q)]. From [p = 8], [lgamma p] is Stirling's series too: [(p - 1/2) log
   (p/(p + q)) + (q - 1/2) log (q/(p + q)) - log (p + q)/2 + log (2 pi)/2] and
   the corrections, no term of which overflows before the result does. *)
let lbeta_at a b =
  let le = cmple a b in
  let p0 = where le a b and q0 = where le b a in
  let valid = logical_and (cmpgt p0 (zeros_like p0)) (lt q0 Float.infinity) in
  let p = clamp valid p0 1. and q = clamp valid q0 1. in
  let large = ge p T.lgamma_stirling_from in
  let s = shift p q in
  let difference = add (mul p (sub_s s.log_q 1.)) s.tail in
  let small = sub (lgamma_pos (clamp (logical_not large) p 1.)) difference in
  let pl = clamp large p 10. in
  let large_r =
    add
      (sub
         (sub
            (mul (sub_s pl 0.5) (sub (log (div pl s.big_q)) s.l1))
            (mul (sub_s s.big_q 0.5) s.l1))
         (mul_s (add s.log_q s.l1) 0.5))
      (add_s (add (stirling pl) s.corrections) log_sqrt_2pi)
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
(* Q's continued fraction for [x >= max (a, igamma_corner)]: [x + 1 - a - f],
   whose reciprocal is [Q e^x x^-a Gamma(a)], Legendre's fraction evaluated
   backward from its fixed depth. *)
let igamma_cf (t : T.t) a x =
  let xa = sub x a in
  let f = ref (zeros_like x) in
  for n = int_of_float t.igamma_cf_depth downto 1 do
    let k = float_of_int n in
    f := div (mul_s (rsub_s k a) k) (sub (add_s xa ((2. *. k) +. 1.)) !f)
  done;
  sub (add_s xa 1.) !f

(* The corner's [x J], [J = sum_n (-x)^(n-1) / ((a + n) n!)], for [a < 1] and [x
   < igamma_corner]. *)
let corner_series (t : T.t) a x =
  let jsum = ref (zeros_like x) in
  let factorial = ref 1. in
  let terms = int_of_float t.igamma_corner_terms in
  for n = 1 to terms do
    factorial := !factorial *. float_of_int n
  done;
  for n = terms downto 1 do
    jsum :=
      sub (recip (mul_s (add_s a (float_of_int n)) !factorial)) (mul x !jsum);
    factorial := !factorial /. float_of_int n
  done;
  mul x !jsum

(* The corner's [c = a log x - log Gamma(1 + a)] and [j = a x J]: [Q = e^c j -
   expm1 c]. *)
let igamma_corner_parts (t : T.t) a x log_x =
  (mul a (sub log_x (lgamma1p_ratio a)), mul a (corner_series t a x))

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
  let lq_cf = sub (add lr (log a_c)) (log (igamma_cf t a_c x_c)) in
  (* The corner. *)
  let x_k = clamp corner x 0.5 in
  let log_xk = clamp corner log_x (Stdlib.log 0.5) in
  let xj = corner_series t a_lo x_k in
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

(* Where [a = 0] each tail is its limit: [P(a, x) = 1 - a E1(x) + O(a^2)] for [x
   > 0], so [P = 1] and [Q = 0]. [Q] is written there as [a] times [Q/a] at [a =
   2^-61], whose relative error from [E1(x)] is about [2^-61], so that its slope
   in [a] is [E1(x)]. *)
let limit_a = 0x1p-61

(* [log P(a, x)], or [log Q(a, x)] where [upper], with the domain's edges: [P]
   is 0 at [x = 0] and where [a = inf], 1 at [x = inf] and, as its limit, at [a
   = 0]; NaN outside [a >= 0, x >= 0], at a NaN, at [a = x = 0] and at [a = x =
   inf]. Also whether [a = 0] with [x] finite and positive, and [Q] there. *)
let igamma_edges a x upper =
  let a_inf = is a Float.infinity and x_inf = is x Float.infinity in
  let zero = is x 0. and a_zero = is a 0. in
  let nan =
    logical_or
      (logical_or (isnan a) (isnan x))
      (logical_or
         (logical_or (lt a 0.) (lt x 0.))
         (logical_or (logical_and a_inf x_inf) (logical_and a_zero zero)))
  in
  let limit = logical_and a_zero (logical_not (logical_or nan x_inf)) in
  let regular =
    logical_not
      (logical_or
         (logical_or nan (logical_or a_inf a_zero))
         (logical_or x_inf zero))
  in
  let a_c = where limit (lit a limit_a) (clamp regular a 1.) in
  let x_c = clamp (logical_or regular limit) x 1. in
  let r, _ = igamma_core a_c x_c (log x_c) (logical_or upper limit) in
  (* [a + 0] is [+0] at [-0], with [a]'s slope. *)
  let q =
    mul
      (add_s (clamp limit a 0.) 0.)
      (exp (where limit (sub r (log a_c)) (zeros_like r)))
  in
  let p_zero = logical_or a_inf zero in
  let edge =
    where (logical_xor p_zero upper) (lit r Float.neg_infinity) (zeros_like r)
  in
  (* [log Q] is [-inf] there, selected: a logarithm of [0] would make every
     slope through it NaN. *)
  let at_limit = where upper (lit q Float.neg_infinity) (log1p (neg q)) in
  let r =
    where regular r (where limit at_limit (where nan (lit r Float.nan) edge))
  in
  (r, limit, q)

let log_gammainc_at a x upper =
  let r, _, _ = igamma_edges a x upper in
  r

(* [P] or [Q] as the exponential of its logarithm, and [Q] itself at [a = 0]. At
   [x = 0], where [P(a, x) = x^a / Γ(a + 1) + O(x^(a + 1))], it is written so
   that its slope in [x] is that limit's: 1 at [a = 1] and 0 above. *)
let gammainc_at a x upper =
  let l, limit, q = igamma_edges a x upper in
  let r = where limit (where upper q (rsub_s 1. q)) (exp l) in
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
   0], [inf] at the other end and where [a = inf]; at [a = 0], where all the
   mass is at 0, the limit: 0 but at that other end. NaN outside [a >= 0] and
   [p] in [0, 1]. *)
let gammaincinv_at a p upper =
  let nan =
    logical_or
      (logical_or (isnan a) (isnan p))
      (logical_or (lt a 0.) (logical_or (lt p 0.) (cmpgt p (lit p 1.))))
  in
  let low = where upper (is p 1.) (is p 0.) in
  let high = where upper (is p 0.) (is p 1.) in
  let inf =
    logical_or high (logical_and (is a Float.infinity) (logical_not low))
  in
  let zero = logical_or low (logical_and (is a 0.) (logical_not high)) in
  let regular = logical_not (logical_or nan (logical_or zero inf)) in
  let a = clamp regular a 1. and p = clamp regular p 0.5 in
  let r = igamma_inverse a p upper in
  where regular r
    (where nan (lit r Float.nan)
       (where zero (zeros_like r) (lit r Float.infinity)))

(* Incomplete beta function

   TOMS 708 (DiDonato and Morris, ACM TOMS 18, 1992), in logarithms, but for
   APSER, whose region BPSER serves. After the swap [(a, b, x) -> (b, a, 1 - x)]
   each region computes one tail directly, as a logarithm [l], and the other is
   [log (1 - e^l)]. Per element one region applies, so the costly part every
   method shares, the logarithm of [x^a y^b / (a B(a, b))] for the arguments the
   region's method needs, runs once on arguments selected per element; each
   method's series runs on every element over its inputs clamped into its
   region. [x] is the only exact input: [log x], [log (1 - x)] and [a - (a + b)
   x] come from it to the dtype's precision. *)

let le x v = cmple x (lit x v)
let gt x v = cmpgt x (lit x v)

(* [(e - log1p e) / e^2] for [|e| <= rlog1_series], from [log1p e = 2 atanh r],
   [r = e / (e + 2)]: [1/(e + 2) - 2 e P(r^2) / (e + 2)^3], [P] the series of
   [sum_k r^2k / (2k + 3)]; no term cancels another. *)
let rlog1_ratio e =
  let t = tables e in
  let d = recip (add_s e 2.) in
  let r = mul e d in
  sub d (mul (mul_s (mul (mul r d) d) 2.) (horner t.rlog1 (mul r r)))

(* [a - (a + b) x] to the dtype's precision: [a + b] and its product with [x]
   are held in two parts, so that the difference keeps its precision near the
   mean [a / (a + b)], where the incomplete beta function is most sensitive to
   it. *)
let lambda_of a b x =
  let s = add a b in
  let bv = sub s a in
  let s_lo = add (sub a (sub s bv)) (sub b bv) in
  let p = mul s x in
  let p_lo = add (fma s x (neg p)) (mul s_lo x) in
  sub (sub a p) p_lo

let bcorr p q = add (stirling p) (stirling_difference q p)

(* The front [log (x^a y^b / (a B(a, b)))]. Below [beta_large] in either
   argument, with [p <= q] the arguments and [v] the variable of [p]: [q log w +
   p (log (v Q) - 1) + tail - lgamma p], [w] the variable of [q], so that
   neither [p log v] nor [lgamma (p + q) - lgamma q] cancels the other where [v
   Q] is near 1; [lgamma p] is [lgamma (1 + f) + log prod_k (f + k)] for [p = f
   + m], [f] in (0, 1], and [lgamma (1 + f)] from [lgamma1p_ratio], so that
   [lgamma p + log a] holds no [log a] to cancel where [p] is [a]. From
   [beta_large], [-(a u + b w)] for [u = e - log1p e], [e = -lam / a], and [w]
   likewise in [lam / b], with the corrections' difference: [x] and [y] enter
   through [lam = a - (a + b) x] alone near the mean.

   With it [log (x^a / (a B(a, b)))], BPSER's front, which below [beta_large]
   leaves [q log w] out where [p] is [a]: there the rest is O(a), and adding [b
   log y] to it to take it away would round away its digits. *)
let front a b x y lx ly lam =
  let large = logical_and (ge a T.beta_large) (ge b T.beta_large) in
  let small = logical_not large in
  let a_s = clamp small a 1. and b_s = clamp small b 1. in
  let a_le = cmple a_s b_s in
  let p = where a_le a_s b_s and q = where a_le b_s a_s in
  let v = where a_le x y and lv = where a_le lx ly in
  let lw = where a_le ly lx in
  let s = shift p q in
  let vq = mul v s.big_q in
  let normal = cmpge vq (lit vq (min_normal vq)) in
  let log_vq = where normal (log (clamp normal vq 1.)) (add lv s.log_q) in
  (* [p = f + m]: [m = ceil p - 1] shifts, [f] in (0, 1]. *)
  let m = sub_s (ceil p) 1. in
  let f = sub p m in
  let prod = ref (ones_like p) in
  for k = 1 to int_of_float T.beta_large - 2 do
    let fk = float_of_int k in
    prod := where (gt m fk) (mul !prod (add_s f fk)) !prod
  done;
  let unshifted = is m 0. in
  (* [-lgamma p - log a]: [-lgamma (1 + f) - log prod], less [log a], plus [log
     p] where [m = 0], those two cancelling where [p] is [a]. *)
  let log_a = log a_s in
  let rest =
    where
      (logical_and unshifted a_le)
      (zeros_like p)
      (sub (where unshifted (log p) (zeros_like p)) log_a)
  in
  let core =
    add
      (add (mul p (sub_s log_vq 1.)) s.tail)
      (sub (sub rest (mul f (lgamma1p_ratio f))) (log !prod))
  in
  let a_l = clamp large a 10. and b_l = clamp large b 10. in
  let lam_l = clamp large lam 0. in
  let part n e l other =
    let near = cmple (abs e) (lit e T.rlog1_series) in
    let en = clamp near e 0. in
    where near
      (mul (mul en en) (rlog1_ratio en))
      (sub e (add l (log1p (div other n))))
  in
  let u = part a_l (div (neg lam_l) a_l) lx b_l in
  let w = part b_l (div lam_l b_l) ly a_l in
  let al_le = cmple a_l b_l in
  let p_l = where al_le a_l b_l and q_l = where al_le b_l a_l in
  let phi_large =
    sub
      (sub
         (sub
            (mul_s (sub (log p_l) (log1p (div p_l q_l))) 0.5)
            (add (mul a_l u) (mul b_l w)))
         (add_s (log a_l) log_sqrt_2pi))
      (bcorr p_l q_l)
  in
  let phi = where large phi_large (add (mul q lw) core) in
  (phi, where (logical_and small a_le) core (sub phi (mul b ly)))

(* BPSER's series [sum_n c_n / (a + n)], [c_n = prod_k (1 - b/k) x]: [I_x(a, b)
   = x^a / (a B(a, b)) (1 + a sum)] for [b <= 1] or [b x <= 0.7]. *)
let bpser_sum a b x =
  let t = tables x in
  let c = ref (ones_like x) and sum = ref (zeros_like x) in
  for n = 1 to t.bpser_terms do
    let fn = float_of_int n in
    c := mul (mul !c (add_s (rsub_s 0.5 (mul_s b (1. /. fn))) 0.5)) x;
    sum := add !sum (div !c (add_s a fn))
  done;
  !sum

(* BUP's terms [x^(a+i) y^b / ((a + i) B(a + i, b))], [i < n] for a masked count
   [n <= beta_bup_most], each the one before it times [r_l = (a + b + l) x / (a
   + 1 + l)]: their sum relative to an anchor term, the first, or the last where
   every ratio is at least 1, so that no partial sum overflows; and the first
   term relative to the anchor, which may underflow to 0. [anchor_last] is
   whether the last is the anchor. *)
let bup_last_anchor a b x n =
  let two = ge n 2. in
  let l = clamp two (sub_s n 2.) 0. in
  let last = div (mul (add (add a b) l) x) (add_s (add a l) 1.) in
  logical_and two (ge last 1.)

let bup_sum a b x n anchor_last =
  let g = ref (ones_like x) and total = ref (ones_like x) in
  for i = 1 to T.beta_bup_most - 1 do
    let fi = float_of_int i in
    let active = gt n fi in
    let l =
      clamp active
        (where anchor_last (sub_s n (1. +. fi)) (lit n (fi -. 1.)))
        0.
    in
    let r = div (mul (add (add a b) l) x) (add_s (add a l) 1.) in
    g := where active (mul !g (where anchor_last (recip r) r)) !g;
    total := add !total (where active !g (zeros_like !g))
  done;
  (!total, where anchor_last !g (ones_like x))

(* The second BUP's sum, [n] terms relative to the first, every ratio below
   1. *)
let bup_sum_down a b x n =
  let d = ref (ones_like x) and total = ref (ones_like x) in
  for i = 1 to n - 1 do
    let l = float_of_int (i - 1) in
    d := mul !d (div (mul (add_s (add a b) l) x) (add_s a (l +. 1.)));
    total := add !total !d
  done;
  !total

(* BFRAC's continued fraction for [a, b > 1], [lam = a - (a + b) x >= 0]: [log
   (1 / (beta_0 + alpha_1 / (beta_1 + alpha_2 / ...)))], evaluated backward from
   its fixed depth; [I_x(a, b)] is it times [x^a y^b / B(a, b)]. With [s_k = a +
   2k - 1], DiDonato and Morris's terms are [alpha_k = (a + k - 1) (a + b + k -
   1) k (b - k) x^2 / s_k^2] and [beta_k = k + k (b - k) x / s_k + (a + k) (1 +
   lam + k (1 + y)) / (s_k + 2)], [beta_0 = (1 + lam) a / (a + 1)]. *)
let bfrac a b x y lam =
  let t = tables x in
  let c = add_s lam 1. and yp1 = add_s y 1. and apb = add a b in
  (* [(s_k, k (b - k) x)]. *)
  let term k = (add_s a ((2. *. k) -. 1.), mul_s (mul (sub_s b k) x) k) in
  let beta k (s, w) =
    add_s
      (add (div w s) (div (mul (add_s a k) (add c (mul_s yp1 k))) (add_s s 2.)))
      k
  in
  let alpha k (s, w) =
    div
      (mul (mul (add_s a (k -. 1.)) (add_s apb (k -. 1.))) (mul w x))
      (mul s s)
  in
  let depth = t.bfrac_depth in
  let current = ref (term (float_of_int depth)) in
  let f = ref (beta (float_of_int depth) !current) in
  for n = depth downto 1 do
    let k = float_of_int n in
    let alpha_k = alpha k !current in
    let below =
      if n = 1 then div (mul c a) (add_s a 1.)
      else (
        current := term (k -. 1.);
        beta (k -. 1.) !current)
    in
    f := add below (div alpha_k !f)
  done;
  neg (log !f)

(* BASYM: [log I_x(a, b)] for [a, b > beta_frac_a] near the mean by Temme's
   asymptotic expansion as DiDonato and Morris write it, [lam >= 0]. Its
   coefficients [d_i] are [sign (b - a)^i G_i(h) / (1 + h)^(i div 2)] for [h]
   the smaller argument over the larger, [G_i] from the tables. [f = lam^2 g]
   keeps [sqrt f] smooth where [lam] is 0. *)
let basym a b lam =
  let t = tables a in
  let a_le = cmple a b in
  let p = where a_le a b and q = where a_le b a in
  let h = div p q in
  let r0 = recip (add_s h 1.) in
  let sign = where (cmpge b a) (ones_like a) (neg (ones_like a)) in
  let w0 = mul sign (recip (sqrt (mul p (add_s h 1.)))) in
  let e1 = div (neg lam) a and e2 = div lam b in
  let g = add (div (rlog1_ratio e1) a) (div (rlog1_ratio e2) b) in
  let f = mul (mul lam lam) g in
  let z0 = mul lam (sqrt g) in
  let z2 = add f f in
  let c0 = 2. /. Float.sqrt Float.pi and c1 = 0.5 /. Float.sqrt 2. in
  let poly i = horner t.basym.(i - 1) h in
  let j0 = ref (mul_s (erfcx_at z0) (0.5 /. c0)) and j1 = ref (lit a c1) in
  let sum = ref (add !j0 (mul (mul_s (poly 1) c1) w0)) in
  let znm1 = ref (mul_s z0 (Float.sqrt 2.)) and zn = ref z2 in
  let rp = ref (ones_like a) and ww = ref w0 in
  let terms = Array.length t.basym - 1 in
  let n = ref 2 in
  while !n <= terms do
    let fn = float_of_int !n in
    j0 := add (mul_s !znm1 c1) (mul_s !j0 (fn -. 1.));
    j1 := add (mul_s !zn c1) (mul_s !j1 fn);
    znm1 := mul z2 !znm1;
    zn := mul z2 !zn;
    rp := mul !rp r0;
    ww := mul !ww w0;
    let t0 = mul (mul (poly !n) !rp) (mul !ww !j0) in
    ww := mul !ww w0;
    let t1 = mul (mul (poly (!n + 1)) !rp) (mul !ww !j1) in
    sum := add !sum (add t0 t1);
    n := !n + 2
  done;
  sub (sub (add_s (log !sum) (Stdlib.log c0)) f) (bcorr p q)

(* [Q(b, z) / (e^-z z^b / Gamma(b))] for [b <= 1], [z > 0], from the incomplete
   gamma's corner below [igamma_corner], [(e^c j - expm1 c) / (b e^(c - z))],
   and its continued fraction above. *)
let gamma_ratio b z =
  let t = tables z in
  let corner = lt z T.igamma_corner in
  let zc = clamp corner z 0.5 in
  let c, j = igamma_corner_parts t b zc (log zc) in
  let below = div (sub (mul (exp c) j) (expm1 c)) (mul b (exp (sub c zc))) in
  let zf = clamp (logical_not corner) z 2. in
  where corner below (recip (igamma_cf t b zf))

(* BGRAT's asymptotic expansion of [I_x(a0 + m, b)] for [a0 + m >= 15] and [b <=
   1], relative to [x^a0 y^b / B(a0, b)]: [U sum], [U = e^-z z^b Gamma(a + b) /
   (Gamma(b) Gamma(a) nu^b)] over that front is [(z / (nu y))^b x^(nu - a0)
   prod_k<m (a0 + b + k) / (a0 + k)], for [nu = a + (b - 1)/2], [z = -nu log x].
   Its coefficients [d_n(b)] come from the tables. *)
let bgrat a0 b x y lx m =
  let t = tables x in
  let a = add a0 m in
  let nu = add a (mul_s (sub_s b 1.) 0.5) in
  let z = mul (neg nu) lx in
  let prod = ref (ones_like a0) in
  for k = 0 to T.beta_bup_terms - 1 do
    let fk = float_of_int k in
    prod :=
      where (gt m fk)
        (mul !prod (div (add_s (add a0 b) fk) (add_s a0 fk)))
        !prod
  done;
  let u =
    mul (exp (add (mul b (log (div (neg lx) y))) (mul (sub nu a0) lx))) !prod
  in
  let v = div (lit a 0.25) (mul nu nu) and t2 = mul_s (mul lx lx) 0.25 in
  let j = ref (gamma_ratio b z) in
  let sum = ref !j and tt = ref (ones_like x) in
  Array.iteri
    (fun i d ->
      let bp2n = add_s b (2. *. float_of_int i) in
      j :=
        mul
          (add
             (mul (mul bp2n (add_s bp2n 1.)) !j)
             (mul (add (add_s bp2n 1.) z) !tt))
          v;
      tt := mul !tt t2;
      sum := add !sum (mul (horner d b) !j))
    t.bgrat;
  mul u !sum

(* [log I_x(a, b)], or [log (1 - I_x(a, b))] where [upper], for finite [a, b >
   0] and [0 < x < 1]. After the swap, the regions name the method for [(a0, b0,
   x0)]; the direct tail is the swapped problem's lower one [w] or its upper one
   [w1]. *)
let log_betainc_at upper a b x =
  let t = tables x in
  let eps = t.beta_eps in
  let ( &&& ) = logical_and and ( ||| ) = logical_or and no = logical_not in
  (* Everything comes from [s], the smaller of [x] and [1 - x], exact both ways,
     so that [(b, a, 1 - x)] computes what [(a, b, x)] does bit for bit where [x
     >= 1/2]: [y], the logarithms and [p - (p + q) v] for [v] either
     variable. *)
  let upper_half = ge x 0.5 in
  let y = rsub_s 1. x in
  let s = where upper_half y x in
  (* At [x = 1/2] both variables are 1/2: their logarithms are [log x] and [log
     (1 - x)], the same bits with each its own derivative, and [p - (p + q) v]
     is [(p - q)/2 + (p + q) (1/2 - v)], whose last term is an exact 0. *)
  let half = is x 0.5 in
  let ls = log s and l1s = log1p (neg s) in
  let lx = where half (log x) (where upper_half l1s ls) in
  let ly = where half (log y) (where upper_half ls l1s) in
  let lam_at p q v_is_x =
    let off = where v_is_x (rsub_s 0.5 x) (sub_s x 0.5) in
    where half
      (add (mul_s (sub p q) 0.5) (mul (add p q) off))
      (where
         (logical_xor v_is_x upper_half)
         (lambda_of p q s)
         (neg (lambda_of q p s)))
  in
  let lam = lam_at a b (ones_like upper_half) in
  let small = le a 1. ||| le b 1. in
  (* A tie swaps to put the larger argument second, as [(b, a, 1 - x)] does, and
     at [a = b] to compute the tail asked for directly, so that the two compute
     the same. *)
  let tie = where upper (cmpge a b) (cmpgt a b) in
  let swap =
    where small
      (gt x 0.5 ||| (is x 0.5 &&& tie))
      (lt lam 0. ||| (is lam 0. &&& tie))
  in
  let sel u v = where swap v u in
  let a0 = sel a b and b0 = sel b a and x0 = sel x y and y0 = sel y x in
  let lx0 = sel lx ly and ly0 = sel ly lx and lam0 = sel lam (neg lam) in
  (* Either argument at most 1. *)
  let fp = small &&& lt b0 eps &&& cmplt b0 (mul_s a0 eps) in
  (* Below [eps] in [a0] with [b0 x0 <= 1], where TOMS 708 takes APSER's
     complement, BPSER serves too: in logarithms each term of its front is
     relatively exact to O(a0), and [log1mexp] keeps the complement. *)
  let tiny_a =
    small &&& no fp &&& lt a0 eps
    &&& cmplt a0 (mul_s b0 eps)
    &&& le (mul b0 x0) 1.
  in
  let rest = small &&& no fp &&& no tiny_a in
  let both = le a0 1. &&& le b0 1. in
  let far = ge x0 T.beta_x_far in
  let low =
    ge a0 T.beta_a_small ||| cmpge a0 b0
    ||| le (mul a0 lx0) (Stdlib.log T.beta_power_x)
  in
  let near_pow =
    lt x0 T.beta_x_near
    &&& le (mul a0 (log (mul x0 b0))) (Stdlib.log T.beta_power_bx)
  in
  let bp_small =
    fp ||| tiny_a
    ||| (rest &&& both &&& low)
    ||| (rest &&& no both &&& (le b0 1. ||| (no far &&& near_pow)))
  in
  let bpy = rest &&& no bp_small &&& far in
  let grat_small = rest &&& no bp_small &&& no far in
  let alone = grat_small &&& no both &&& gt b0 T.beta_bgrat_b in
  let g20 = grat_small &&& no alone in
  (* Both above 1. *)
  let big = no small in
  let below40 = big &&& lt b0 T.beta_b_small in
  let bp_big = below40 &&& le (mul b0 x0) T.beta_bpser_bx in
  let bup_bp = below40 &&& no bp_big &&& le x0 T.beta_bup_x in
  let bup_g = below40 &&& no bp_big &&& no bup_bp in
  let bup2_on = bup_g &&& le a0 T.beta_bgrat_b in
  let beyond = big &&& no below40 in
  let frac_on =
    where (cmple a0 b0)
      (le a0 T.beta_frac_a ||| cmpgt lam0 (mul_s a0 T.beta_frac_lambda))
      (le b0 T.beta_frac_a ||| cmpgt lam0 (mul_s b0 T.beta_frac_lambda))
  in
  let frac = beyond &&& frac_on and asym = beyond &&& no frac_on in
  let bp = bp_small ||| bp_big in
  let shifted_b = bup_bp ||| bup_g in
  (* [b0 = n + bb], [bb] in (0, 1]; the first BUP runs on [(b0, a0, y0)] for 20
     terms or on [(bb, a0, y0)] for [n]. *)
  let n_fl = floor b0 in
  let whole = cmpeq n_fl b0 in
  let n_b = where whole (sub_s n_fl 1.) n_fl in
  let bb = sub b0 n_b in
  let bup_on = g20 ||| shifted_b in
  let terms = float_of_int T.beta_bup_terms in
  let u_a = clamp bup_on (where g20 b0 bb) 1. in
  let u_b = clamp bup_on a0 1. and u_x = clamp bup_on y0 0.5 in
  let u_n = clamp bup_on (where g20 (lit b0 terms) n_b) 1. in
  let last = bup_last_anchor u_a u_b u_x u_n in
  (* The front's arguments: [(a0, b0, x0)], [(b0, a0, y0)], or the first BUP's
     anchor term [(u_a + j, a0, y0)]. *)
  let on_y = bpy ||| g20 ||| alone ||| shifted_b in
  let f_a =
    where shifted_b
      (add u_a (where last (sub_s u_n 1.) (zeros_like u_n)))
      (where on_y b0 a0)
  in
  let f_b = where on_y a0 b0 in
  let f_on = no asym in
  let f_a = clamp f_on f_a 2. and f_b = clamp f_on f_b 2. in
  let f_x = clamp f_on (where on_y y0 x0) 0.5 in
  let f_y = clamp f_on (where on_y x0 y0) 0.5 in
  let f_lx = clamp f_on (where on_y ly0 lx0) (-.Stdlib.log 2.) in
  let f_ly = clamp f_on (where on_y lx0 ly0) (-.Stdlib.log 2.) in
  (* [f_x] is [x] itself where it is [x0] unswapped or [y0] swapped. *)
  let f_lam = lam_at f_a f_b (logical_not (logical_xor on_y swap)) in
  let fr, fr_x = front f_a f_b f_x f_y f_lx f_ly (clamp f_on f_lam 0.) in
  (* BPSER's series on [(a0, b0, x0)], [(b0, a0, y0)] or [(a0, bb, x0)]. *)
  let s_on = bp ||| bpy ||| bup_bp in
  let s_a = clamp s_on (where bpy b0 a0) 1. in
  let s_b = clamp s_on (where bpy a0 (where bup_bp bb b0)) 1. in
  let s_x = clamp s_on (where bpy y0 x0) 0.5 in
  let series = bpser_sum s_a s_b s_x in
  let l_bp = add fr_x (log1p (mul s_a series)) in
  (* The first BUP's terms, relative to the front; [g0] is its first term
     relative to the front, in whose units the other parts add. *)
  let total, g0 = bup_sum u_a u_b u_x u_n last in
  let total = where alone (zeros_like total) total in
  (* BPSER on [(a0, bb, x0)] in units of the first term [y0^bb x0^a0 / (bb B(bb,
     a0))]: [bb / (a0 y0^bb) (1 + a0 sum)]. *)
  let r_a = clamp bup_bp a0 2. and r_b = clamp bup_bp bb 0.5 in
  let bp_rel =
    mul
      (div (mul (exp (neg (mul r_b (clamp bup_bp ly0 (-0.5))))) r_b) r_a)
      (add_s (mul r_a series) 1.)
  in
  (* The second BUP, on [(a0, bb, x0)] for 20 terms: its first term is the first
     BUP's times [bb / a0]. *)
  let v_a = clamp bup2_on a0 2. and v_b = clamp bup2_on bb 0.5 in
  let bup2 =
    mul (div v_b v_a)
      (bup_sum_down v_a v_b (clamp bup2_on x0 0.5) T.beta_bup_terms)
  in
  let bup2 = where bup2_on bup2 (zeros_like bup2) in
  (* BGRAT on [(b0 (+ 20), a0, y0)] or [(a0 (+ 20), bb, x0)]: relative to [x^a
     y^b / B(a, b)] for its unshifted arguments, which is the first BUP term
     times [u_a]. *)
  let g_on = g20 ||| alone ||| bup_g in
  let g_a0 = clamp g_on (where bup_g a0 b0) 20. in
  let g_b = clamp g_on (where bup_g bb a0) 0.5 in
  let g_x = clamp g_on (where bup_g x0 y0) 0.9 in
  let g_y = clamp g_on (where bup_g y0 x0) 0.1 in
  let g_lx = clamp g_on (where bup_g lx0 ly0) (Stdlib.log 0.9) in
  let g_m = where (g20 ||| bup2_on) (lit b0 terms) (zeros_like b0) in
  let g_u = clamp g_on (where bup_g bb b0) 1. in
  let grat = mul g_u (bgrat g_a0 g_b g_x g_y g_lx g_m) in
  let extra =
    where bup_bp bp_rel
      (where (g20 ||| alone ||| bup_g) (add bup2 grat) (zeros_like grat))
  in
  let l_bup = add fr (log (add total (mul g0 extra))) in
  let l_frac =
    add (add fr (log f_a)) (bfrac f_a f_b f_x f_y (clamp frac lam0 1.))
  in
  let l_asym =
    basym (clamp asym a0 200.) (clamp asym b0 200.) (clamp asym lam0 0.)
  in
  let l =
    where (bp ||| bpy) l_bp
      (where (g20 ||| alone ||| shifted_b) l_bup (where frac l_frac l_asym))
  in
  let direct_upper = bpy ||| g20 ||| alone in
  (* The swapped problem's upper tail is the lower one asked for. *)
  let wanted_upper = logical_xor swap upper in
  where (logical_xor direct_upper wanted_upper) (log1mexp l) l

(* The argument a vanishing tail is measured at: its ratio to the argument there
   is the tail's slope at 0, to within its square. *)
let beta_vanishing = 0x1p-61

(* A tail on the whole domain: [a, b >= 0], [x] in [0, 1], as its logarithm and
   its value. The lower tail is 1 at [x = 1], at [a = 0] and at [b = +inf], 0 at
   [x = 0], at [b = 0] and at [a = +inf], and NaN where a 1 meets a 0. At [a =
   0], with [b] finite and [x] inside, the upper tail is [a C] for [C] its ratio
   to [a] at [beta_vanishing], so that its slope in [a] is [C]; at [b = 0] the
   lower tail likewise. *)
let beta_tail upper a b x =
  let positive v = gt v 0. in
  let finite v = logical_and (isfinite v) (positive v) in
  let inf v = is v Float.infinity in
  let zero v = is v 0. in
  let inside = logical_and (gt x 0.) (lt x 1.) in
  let interior = logical_and (logical_and (finite a) (finite b)) inside in
  let at_a0 = logical_and (logical_and (zero a) (finite b)) inside in
  let at_b0 = logical_and (logical_and (zero b) (finite a)) inside in
  let slope = logical_or at_a0 at_b0 in
  let run = logical_or interior slope in
  (* On those two edges the tail that vanishes, the upper at [a = 0]. *)
  let a_r = where at_a0 (lit a beta_vanishing) a in
  let b_r = where at_b0 (lit b beta_vanishing) b in
  let upper_r = where slope at_a0 upper in
  let r =
    log_betainc_at upper_r (clamp run a_r 1.) (clamp run b_r 1.)
      (clamp run x 0.5)
  in
  let n = clamp slope (where at_a0 a b) 0.5 in
  let log_c = clamp slope (sub_s r (Stdlib.log beta_vanishing)) 0. in
  (* [+ 0] writes the tail at [n = -0] as [+0]. *)
  let vanishing = add_s (mul n (exp log_c)) 0. in
  let wants = where at_a0 upper (logical_not upper) in
  (* The vanishing tail's logarithm takes [n] only where it is wanted, so that
     no [1 / n] at [n = 0] meets a zero cotangent. *)
  let l_slope =
    where wants (add (log (clamp wants n 0.5)) log_c) (log1p (neg vanishing))
  in
  let v_slope = where wants vanishing (rsub_s 1. vanishing) in
  let one = List.fold_left logical_or (is x 1.) [ zero a; inf b ] in
  let nil = List.fold_left logical_or (zero x) [ zero b; inf a ] in
  let is_one = logical_xor one upper in
  (* A tail of 1 for [x] inside is a limit, reached from below: its logarithm is
     [-0]. *)
  let log_one = where inside (lit x (-0.)) (zeros_like x) in
  let l_edge = where is_one log_one (lit x Float.neg_infinity) in
  let v_edge = where is_one (ones_like x) (zeros_like x) in
  let invalid =
    List.fold_left logical_or (isnan a)
      [
        isnan b;
        isnan x;
        lt a 0.;
        lt b 0.;
        lt x 0.;
        gt x 1.;
        logical_and one nil;
      ]
  in
  let nan = lit x Float.nan in
  let l = where interior r (where slope l_slope l_edge) in
  let v = where interior (exp r) (where slope v_slope v_edge) in
  (where invalid nan l, where invalid nan v)

type ternary = {
  f3 : 'c. (float, 'c) t -> (float, 'c) t -> (float, 'c) t -> (float, 'c) t;
}

let ternary_at_float32 { f3 } a b x =
  let a, b = broadcasted a b in
  let a, x = broadcasted a x in
  let b, x = broadcasted b x in
  if narrow (dtype a) then
    let f32 = cast Nx_dtype.float32 in
    cast (dtype a) (f3 (f32 a) (f32 b) (f32 x))
  else f3 a b x

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

let log_betainc a b x =
  ternary_at_float32
    { f3 = (fun a b x -> fst (beta_tail (tail x false) a b x)) }
    a b x

let log_betaincc a b x =
  ternary_at_float32
    { f3 = (fun a b x -> fst (beta_tail (tail x true) a b x)) }
    a b x

let betainc a b x =
  ternary_at_float32
    { f3 = (fun a b x -> snd (beta_tail (tail x false) a b x)) }
    a b x

let betaincc a b x =
  ternary_at_float32
    { f3 = (fun a b x -> snd (beta_tail (tail x true) a b x)) }
    a b x
