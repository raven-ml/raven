(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Elementwise

(* Vectors over lanes *)

let lanes v = Array.sub (Nx.shape v) 0 (Nx.ndim v - 1)

let rows v =
  let width = (Nx.shape v).(Nx.ndim v - 1) in
  Nx.reshape [| Array.fold_left ( * ) 1 (lanes v); width |] v

let width v = (Nx.shape v).(Nx.ndim v - 1)

(* A vector of one component is its own sum, which needs no reduction. *)
let dot u v =
  let uv = Nx.mul u v in
  if width uv = 1 then Nx.reshape (lanes uv) uv
  else Nx.reshape (lanes uv) (Num.sum_rows (rows uv))

(* Whether every component of each lane holds. *)
let every b =
  if width b = 1 then Nx.reshape (lanes b) b else Nx.all ~axes:[ -1 ] b

(* The largest magnitude is clamped to the normal floats so that a vector of
   zeros or with an infinite component divides by a finite, positive number. *)
let largest v =
  let dtype = Nx.dtype v in
  Nx.clamp ~min:(Num.tiny dtype) ~max:(Num.huge dtype)
    (Nx.max ~axes:[ -1 ] ~keepdims:true (Nx.abs v))

(* The squares are summed on the vector divided by its largest magnitude: the
   squares of components of 10⁻¹⁷⁰ are below the floats' range. *)
let norm v =
  match width v with
  | 0 -> Nx.zeros_at (Nx.beside v) (Nx.dtype v) (lanes v)
  | 1 -> Nx.reshape (lanes v) (Nx.abs v)
  | _ ->
      let m = largest v in
      let u = Nx.div v m in
      Nx.mul (Nx.reshape (lanes v) m) (Nx.sqrt (dot u u))

(* Each vector scaled by its largest magnitude, so a product neither underflows
   nor overflows. *)
let descends g d =
  let scaled v =
    let m = Nx.max ~axes:[ -1 ] ~keepdims:true (Nx.abs v) in
    Nx.div v (Nx.where (Nx.equal_s m 0.) (Nx.ones_like m) m)
  in
  Nx.less_s (dot (scaled g) (scaled d)) 0.

let hold m next last =
  let ones = Array.make (Nx.ndim next - Nx.ndim m) 1 in
  let m = Nx.reshape (Array.append (Nx.shape m) ones) m in
  Nx.where (Nx.broadcast_to (Nx.shape next) m) next last

(* Whether each lane's error [e], over its scale, has a root mean square of
   at most 1. *)
let within ~scale e =
  let ratio = Tol.scaled ~scale e in
  let r = if width e = 1 then ratio else Num.rms_rows (rows ratio) in
  Nx.less_equal_s (Nx.reshape (lanes e) r) 1.

let accepted tol ~e ~y = within ~scale:(Tol.scale tol y) e
let moves ~from x = Nx.logical_not (every (Nx.equal x from))

(* Backtracking

   The carry holds the trial length, the payload, whether each lane still
   searches and whether it accepted a trial, the trials each lane took and the
   count of trips. A trial at the estimate itself is no decrease and ends the
   search: every shorter step rounds to it too. *)

let backtrack p dtype ~trials ~running ~shrink trial last =
  let step (alpha, (payload, ((searching, found), (n, j)))) =
    let next, phi, accepted, moved = trial alpha in
    let n = Nx.add n (Nx.cast Nx.int32 searching) in
    let ok = Nx.logical_and searching (Nx.logical_and accepted moved) in
    let payload = Nx.Ptree.map2 p (fun _ a b -> hold ok a b) next payload in
    let searching =
      Nx.logical_and searching (Nx.logical_and (Nx.logical_not ok) moved)
    in
    let alpha = Nx.where searching (shrink alpha phi) alpha in
    ( alpha,
      (payload, ((searching, Nx.logical_or found ok), (n, Nx.add_s j 1l))) )
  in
  let lanes = Nx.shape running in
  let at = Nx.placement running in
  let _, (payload, ((_, found), (n, _))) =
    Rune.iterate
      Nx.Ptree.(
        pair tensor (pair p (pair (pair tensor tensor) (pair tensor tensor))))
      ~max:trials
      ~until:(fun (_, (_, ((searching, _), (_, j)))) ->
        Nx.logical_or
          (Nx.logical_not (Nx.any searching))
          (Nx.greater_equal_s j (Int32.of_int trials)))
      step
      ( Nx.ones_at at dtype lanes,
        ( last,
          ( (running, Nx.zeros_like running),
            (Nx.zeros_at at Nx.int32 lanes, Nx.scalar_at at Nx.int32 0l) ) ) )
  in
  (payload, found, n)

(* The sufficient-decrease constant: a step must take [c] of the decrease the
   slope predicts. *)
let c = 1e-4

(* Once [c α φ' 0] falls below [φ 0]'s rounding, the bound rounds to [φ 0] and a
   trial that leaves [φ] unchanged would meet it: a decrease must be strict,
   unless [φ α] is zero, as it is wherever [|f|²] underflows. *)
let armijo ~phi0 ~slope0 =
  let accept alpha phi =
    Nx.logical_and
      (Nx.logical_or (Nx.less phi phi0) (Nx.equal_s phi 0.))
      (Nx.less_equal phi (Nx.add phi0 (Nx.mul_s (Nx.mul alpha slope0) c)))
  in
  let shrink alpha phi =
    let excess = Nx.sub (Nx.sub phi phi0) (Nx.mul slope0 alpha) in
    let quadratic =
      Nx.div (Nx.neg (Nx.mul slope0 (Nx.square alpha))) (Nx.mul_s excess 2.)
    in
    let lo = Nx.mul_s alpha 0.1 and hi = Nx.mul_s alpha 0.5 in
    Nx.where (Nx.isfinite quadratic)
      (Nx.minimum hi (Nx.maximum lo quadratic))
      hi
  in
  (accept, shrink)

(* Strong Wolfe

   The bracketing and zoom of Nocedal and Wright (Numerical Optimization,
   Algorithms 3.5 and 3.6) as one loop over lanes. [lo] is the trial of least
   merit that meets the sufficient decrease, [hi] the bracket's other end.
   Before the bracket closes a trial that keeps decreasing doubles [α]; inside
   it the next [α] is the minimum of the quadratic through [lo]'s value and
   slope and [hi]'s value, kept to the middle 80% of the bracket, and the
   midpoint when the bracket has not halved over two trials. *)

let c2 = 0.9

type ('p, 'q, 'd) wolfe = {
  alpha : (float, 'q, 'd) Nx.t;
  lo : (float, 'q, 'd) Nx.t * ((float, 'q, 'd) Nx.t * (float, 'q, 'd) Nx.t);
  at : 'p;
  hi : (float, 'q, 'd) Nx.t * (float, 'q, 'd) Nx.t;
  zoom : 'd Nx.bool_t;
  searching : 'd Nx.bool_t;
  widths : (float, 'q, 'd) Nx.t * (float, 'q, 'd) Nx.t;
  tries : 'd Nx.int32_t;
  trip : 'd Nx.int32_t;
}

let wolfe p dtype ~trials ~running ~longest ~phi0 ~slope0 trial origin =
  let lanes = Nx.shape running in
  let at = Nx.placement running in
  let zero = Nx.zeros_at at dtype lanes
  and inf = Nx.full_at at dtype lanes Float.infinity in
  let rounding = Nx.mul_s (Nx.abs phi0) (sqrt (Num.eps dtype)) in
  let step s =
    let la, (lphi, lslope) = s.lo and ha, hphi = s.hi in
    let at, phi, slope = trial s.alpha in
    let tries = Nx.add s.tries (Nx.cast Nx.int32 s.searching) in
    (* Near a minimum the decrease a step predicts falls below the rounding of
       [φ], and the slope tells it instead: for a quadratic, the sufficient
       decrease is [φ' α ≤ (2c − 1) φ' 0]. A trial within [√ε |φ 0|] of [φ 0]
       passes on its slope (Hager and Zhang's approximate Wolfe conditions,
       2005). *)
    let decrease =
      Nx.logical_or
        (Nx.less_equal phi (Nx.add phi0 (Nx.mul_s (Nx.mul s.alpha slope0) c)))
        (Nx.logical_and
           (Nx.less_equal phi (Nx.add phi0 rounding))
           (Nx.less_equal slope (Nx.mul_s slope0 ((2. *. c) -. 1.))))
    in
    (* Inside the rounding of [φ 0] values do not order trials; slopes do. *)
    let blurred =
      Nx.logical_and
        (Nx.less_equal (Nx.abs (Nx.sub phi phi0)) rounding)
        (Nx.less_equal (Nx.abs (Nx.sub lphi phi0)) rounding)
    in
    let high =
      Nx.logical_or (Nx.logical_not decrease)
        (Nx.logical_and (Nx.greater_equal phi lphi) (Nx.logical_not blurred))
    in
    let flat = Nx.less_equal (Nx.abs slope) (Nx.mul_s slope0 (-.c2)) in
    let low = Nx.logical_and s.searching (Nx.logical_not high) in
    let rising =
      Nx.where s.zoom
        (Nx.greater_equal_s (Nx.mul slope (Nx.sub ha la)) 0.)
        (Nx.greater_equal_s slope 0.)
    in
    (* A trial at the longest step that still descends ends the bracketing: it
       cannot grow. *)
    let capped =
      Nx.logical_and (Nx.logical_not s.zoom)
        (Nx.logical_and (Nx.logical_not rising)
           (Nx.greater_equal s.alpha longest))
    in
    let accept = Nx.logical_and low (Nx.logical_or flat capped) in
    let flip =
      Nx.logical_and low (Nx.logical_and (Nx.logical_not flat) rising)
    in
    let raise_ = Nx.logical_and s.searching high in
    let pick m a b = Nx.where m a b in
    let ha = pick raise_ s.alpha (pick flip la ha)
    and hphi = pick raise_ phi (pick flip lphi hphi) in
    let lo =
      (pick low s.alpha la, (pick low phi lphi, pick low slope lslope))
    in
    let at = Nx.Ptree.map2 p (fun _ a b -> hold low a b) at s.at in
    let zoom = Nx.logical_or s.zoom (Nx.logical_or raise_ flip) in
    let searching = Nx.logical_and s.searching (Nx.logical_not accept) in
    (* The next trial. *)
    let la, (lphi, lslope) = lo in
    let w = Nx.sub ha la in
    let quadratic =
      Nx.sub la
        (Nx.div
           (Nx.mul lslope (Nx.square w))
           (Nx.mul_s (Nx.sub (Nx.sub hphi lphi) (Nx.mul lslope w)) 2.))
    in
    let near = Nx.add la (Nx.mul_s w 0.1)
    and far = Nx.sub ha (Nx.mul_s w 0.1) in
    let inside =
      Nx.logical_and
        (Nx.less_equal (Nx.minimum near far) quadratic)
        (Nx.less_equal quadratic (Nx.maximum near far))
    in
    let w1, w2 = s.widths in
    let halved = Nx.less_equal (Nx.abs w) (Nx.mul_s w2 0.5) in
    let inner =
      Nx.where
        (Nx.logical_and inside halved)
        quadratic
        (Nx.add la (Nx.mul_s w 0.5))
    in
    let next = Nx.where zoom inner (Nx.minimum longest (Nx.mul_s s.alpha 2.)) in
    let widths = (pick zoom (Nx.abs w) w1, pick zoom w1 w2) in
    {
      alpha = pick searching next s.alpha;
      lo;
      at;
      hi = (ha, hphi);
      zoom;
      searching;
      widths;
      tries;
      trip = Nx.add_s s.trip 1l;
    }
  in
  let ptree =
    Nx.Ptree.(
      pair
        (pair tensor (pair tensor (pair tensor tensor)))
        (pair p
           (pair (pair tensor tensor)
              (pair (pair tensor tensor)
                 (pair (pair tensor tensor) (pair tensor tensor))))))
  in
  let pack s =
    let la, (lphi, lslope) = s.lo and ha, hphi = s.hi in
    ( (s.alpha, (la, (lphi, lslope))),
      ( s.at,
        ((ha, hphi), ((s.zoom, s.searching), (s.widths, (s.tries, s.trip)))) )
    )
  in
  let unpack
      ( (alpha, (la, (lphi, lslope))),
        (at, ((ha, hphi), ((zoom, searching), (widths, (tries, trip))))) ) =
    {
      alpha;
      lo = (la, (lphi, lslope));
      at;
      hi = (ha, hphi);
      zoom;
      searching;
      widths;
      tries;
      trip;
    }
  in
  let s =
    Rune.iterate ptree ~max:trials
      ~until:(fun c ->
        let s = unpack c in
        Nx.logical_or
          (Nx.logical_not (Nx.any s.searching))
          (Nx.greater_equal_s s.trip (Int32.of_int trials)))
      (fun c -> pack (step (unpack c)))
      (pack
         {
           alpha = Nx.ones_at at dtype lanes;
           lo = (zero, (phi0, slope0));
           at = origin;
           hi = (zero, phi0);
           zoom = Nx.zeros_like running;
           searching = running;
           widths = (inf, inf);
           tries = Nx.zeros_at at Nx.int32 lanes;
           trip = Nx.scalar_at at Nx.int32 0l;
         })
    |> unpack
  in
  let la, _ = s.lo in
  let found =
    Nx.logical_and running
      (Nx.logical_or (Nx.logical_not s.searching) (Nx.greater_s la 0.))
  in
  (s.at, found, s.tries)


(* Iterations

   A Newton-type search takes undamped steps [δ x] and carries the estimate [x],
   the function it steps on at [x] ([f] for a system, the gradient for a
   minimum), the first end of the secant that reads the contraction of the
   undamped map [N x = x + δ x] and [N] there, the error estimate, the two
   newest settled readings, the status, the evaluations and the count of
   iterations, which every lane shares. *)

type ('q, 'd) state = {
  x : (float, 'q, 'd) Nx.t;
  fx : (float, 'q, 'd) Nx.t;
  base : (float, 'q, 'd) Nx.t;
  image : (float, 'q, 'd) Nx.t;
  e : (float, 'q, 'd) Nx.t;
  q : (float, 'q, 'd) Nx.t;
  q' : (float, 'q, 'd) Nx.t;
  st : 'd Nx.int32_t;
  n : 'd Nx.int32_t;
  k : 'd Nx.int32_t;
}

let finite v = every (Nx.isfinite v)
let column v = Nx.reshape (Array.append (Nx.shape v) [| 1 |]) v
let along alpha v = Nx.mul (column alpha) v

let start x0 fx =
  let lanes = lanes x0 in
  let at = Nx.beside x0 in
  let st =
    settle
      (Nx.full_at at Nx.int32 lanes running)
      (Nx.logical_not (finite fx))
      Not_finite
  in
  let none = Nx.full_at at (Nx.dtype x0) lanes Float.nan in
  {
    x = x0;
    fx;
    base = Nx.full_like x0 Float.nan;
    image = Nx.full_like x0 Float.nan;
    e = Nx.full_like x0 Float.infinity;
    q = none;
    q' = none;
    st;
    n = Nx.ones_at at Nx.int32 lanes;
    k = Nx.scalar_at at Nx.int32 0l;
  }

(* The error needs an overestimate of the rate (Kelley, Solving Nonlinear
   Equations with Iterative Methods, 2022, §1.5), and one reading can be low: a
   quasi-Newton method's rate varies from step to step, and one small ratio
   would promise steps that shrink faster than the next one does. *)
let contraction s = Nx.where (Nx.isnan s.q') s.q (Nx.maximum s.q s.q')

(* [tol]'s scale at [y], floored at the smallest normal float so that a scale
   that underflows divides by a positive number. *)
let floored tol y = Nx.clamp ~min:(Num.tiny (Nx.dtype y)) (Tol.scale tol y)

(* The reading [|N x − N x'| / |x − x'|] over the secant from [x' = s.base]
   measures [N]'s contraction whatever step moved the estimate: after a full
   step it is the ratio of the last two undamped steps, and the length of a
   shortened, damped or mixed step never enters it. Lengths are in [tol]'s
   scaled norm, the one in which a contraction bounds the error the tolerance
   accepts (Hairer and Wanner's RADAU5 and Deuflhard's NLEQ1 measure Newton's
   corrections in it): a Euclidean secant is dominated by the large components
   and understates a small one that contracts slowly. The scale is taken at the
   larger magnitude of the secant's ends, so a component that moved has a
   positive one.

   [N x] and [N x'] are each rounded, by up to half a unit in their last place,
   which leaves the reading uncertain by [r = ε (|N x| + |N x'|) / (2 |x − x'|)]
   either way; [q] is its upper end. A reading settles once the rounding
   decides whether the map contracts, [q < 1] or [q − 2r ≥ 1]. Over a step of
   [k] ulps [r] is about [1 / k], so the secant's first end advances only on a
   settled reading: until then the secant spans more steps and its rounding
   shrinks, so a map that contracts by 0.99 settles over a secant of about a
   hundred ulps, which one step near its zero never spans. An unsettled reading
   whose lower end [q − 2r] exceeds the contraction held refutes it: both
   readings are dropped, and the next settled one counts alone. *)
let read tol s ~map =
  let run = searching s.st in
  let image = Nx.add s.x map in
  let scale = floored tol (Nx.maximum (Nx.abs s.x) (Nx.abs s.base)) in
  let length v = norm (Nx.div v scale) in
  let moved = length (Nx.sub s.x s.base) in
  (* Before the first reading, or after a step that did not move the estimate,
     the secant has no length: the next one starts here. *)
  let fresh = Nx.logical_not (Nx.greater_s moved 0.) in
  let moved = Nx.where fresh (Nx.ones_like moved) moved in
  let half = Num.eps (Nx.dtype image) /. 2. in
  let r =
    Nx.div
      (Nx.mul_s (length (Nx.add (Nx.abs image) (Nx.abs s.image))) half)
      moved
  in
  let q = Nx.add (Nx.div (length (Nx.sub image s.image)) moved) r in
  let lower = Nx.sub q (Nx.mul_s r 2.) in
  let settled =
    Nx.logical_and (Nx.logical_not fresh)
      (Nx.logical_or (Nx.less_s q 1.) (Nx.greater_equal_s lower 1.))
  in
  let refuted =
    Nx.logical_and
      (Nx.logical_not (Nx.logical_or fresh settled))
      (Nx.greater lower (contraction s))
  in
  let push = Nx.logical_and run settled
  and drop = Nx.logical_and run refuted
  and advance = Nx.logical_and run (Nx.logical_or fresh settled) in
  let none = Nx.full_like s.q Float.nan in
  {
    s with
    base = hold advance s.x s.base;
    image = hold advance image s.image;
    q = Nx.where push q (Nx.where drop none s.q);
    q' = Nx.where push s.q (Nx.where drop none s.q');
  }

type ('q, 'd) source =
  | Paired of (float, 'q, 'd) Nx.t
  | Solved of (float, 'q, 'd) Nx.t

let gap q =
  let shrinking = Nx.logical_and (Nx.greater_equal_s q 0.) (Nx.less_s q 1.) in
  column (Nx.where shrinking (Nx.rsub_s 1. q) (Nx.zeros_like q))

(* Below the least normal float a step rounds by up to the subnormals' spacing
   whatever its size, and underflows to 0 short of the zero: a step there
   counts that spacing as well. A zero step counts zero only where it is
   resolved, and where the answer is not a subnormal, at which [f] itself may
   have underflowed: [x − x / 100] rounds both to 0 fifty spacings from its
   zero. A step paired with its equation is resolved where that equation's
   value is exactly 0; one solved from [f] by a linear system, where [f] is 0
   or did not underflow, its largest magnitude at least [tiny / ε], since its
   component [i] answers to every equation. The step's size and whether its
   zero is exact. *)
let size ~x ~f delta =
  let dtype = Nx.dtype delta in
  let tiny = Num.tiny dtype in
  let size = Nx.abs delta in
  let size =
    Nx.where (Nx.less_s size tiny) (Nx.add_s size (tiny *. Num.eps dtype)) size
  in
  let resolved =
    match f with
    | Paired f -> Nx.equal_s f 0.
    | Solved f ->
        let m = Nx.max ~axes:[ -1 ] ~keepdims:true (Nx.abs f) in
        Nx.broadcast_to (Nx.shape delta)
          (Nx.logical_or (Nx.equal_s m 0.)
             (Nx.greater_equal_s m (tiny /. Num.eps dtype)))
  in
  let answer = Nx.abs (Nx.add x delta) in
  let exact =
    Nx.logical_and
      (Nx.logical_and (Nx.equal_s delta 0.) resolved)
      (Nx.logical_or (Nx.equal_s answer 0.) (Nx.greater_equal_s answer tiny))
  in
  (size, exact)

let error q ~x ~f delta =
  let size, exact = size ~x ~f delta in
  Nx.where exact (Nx.zeros_like delta) (Nx.div size (gap q))

(* Whether each lane's step is within [ε |x|∞] in every component, the floats'
   resolution of a coupled problem's gradient, however finely its small
   components are represented. *)
let within_ulp x step =
  let top = Nx.max ~axes:[ -1 ] ~keepdims:true (Nx.abs x) in
  every (Nx.less_equal (Nx.abs step) (Nx.mul_s top (Num.eps (Nx.dtype x))))

(* A step at the floats' resolution: within one unit of [x]'s last place in
   every component, so it moves the estimate by rounding at most, as a step that
   flips between the floats either side of a zero does. *)
let still x step =
  let ulp = Nx.mul_s (Nx.abs x) (Num.eps (Nx.dtype x)) in
  every
    (Nx.logical_or
       (Nx.equal (Nx.add x step) x)
       (Nx.less_equal (Nx.abs step) ulp))

(* A quasi-Newton method's step ratios say nothing of its error: its rate jumps
   from step to step, so a lane can meet [tol] by the steps' contraction far
   from its minimum. Its gradient bounds the error instead, a test Kelley calls
   reasonable for any gradient method (Iterative Methods for Optimization,
   1999, §2.3.2).

   In [tol]'s scaled coordinates [u = x / σ], [σ] the scale at the answer,
   the gradient is [σ g], and it is [Ĥ] times [u − u*], [Ĥ] the mean of the
   Hessian in [u] on the segment from [u*] to [u]. Where [Ĥ]'s eigenvalues
   are at least [μ], [|u − u*|₂ ≤ |σ g|₂ / μ], and the root mean square [tol]
   measures, [|u − u*|₂ / √n], is at most [|σ g|₂ / (μ √n)]. Each component
   carries that times its scale, whose root mean square is the bound: zero
   where the gradient is, or before any step measured a curvature.

   [μ] is not known. It is estimated by the least curvature the accepted
   steps [d] measured, [yᵀd / |d / σ|²]: [y = Ĥ' d] over each step, [Ĥ'] the
   mean Hessian along it, so each is a Rayleigh quotient, at least the least
   eigenvalue there. The estimate is therefore at least [μ], and the bound at
   most the distance: it is the distance where the error lies along a
   direction of the least curvature measured, and understates it by up to
   [κ], the ratio of the largest curvature to the least, elsewhere. The
   method's step [δ] is its own estimate of [x* − x], so {!judge} converges a
   lane only once its search measured the curvature along the [δ] it judges:
   an error along a direction the path never sampled, as a soft component
   left after the stiff ones converged, then counts at its own curvature.
   That quotient is the curvature along [δ] alone. Where [δ] points mostly
   elsewhere, as at a stiff component still near [tol] beside a soft one
   whose curvature the method's model overstates, the bound still
   understates the error, by up to [κ].

   The scale enters as its shape [σ̂ = σ / m], [m] its largest component,
   and the bound is [|σ̂ g| / (μ̂ m)] with [μ̂ = μ / m²] measured in [x / σ̂]:
   at a minimum of 10⁻²⁸⁹ under [Tol.ulps 4], [σ] is about 10⁻³⁰⁴ and both
   [σ g] and [μ] are below the floats' range, where they would round the
   bound to 0. The magnitude [m] is the scale's at the answer, as the bound
   states. *)
let gradient_bound ~scale ~mu g =
  let m = largest scale in
  let size = norm (Nx.mul g (Nx.div scale m)) in
  let n = Float.of_int (width g) in
  let b =
    Nx.div_s
      (Nx.div (Nx.div size mu) (Nx.reshape (lanes scale) m))
      (Float.sqrt n)
  in
  let none = Nx.logical_or (Nx.isnan mu) (Nx.equal_s size 0.) in
  Nx.mul scale (column (Nx.where none (Nx.zeros_like b) b))

(* [(σ y)ᵀ ŵ / |w|] with [w = d / σ] and [ŵ] its direction, [σ] the scale's
   shape at the move's larger end, whose terms neither underflow nor
   overflow. *)
let quotient tol ~x ~x' y =
  let sigma = floored tol (Nx.maximum (Nx.abs x) (Nx.abs x')) in
  let sigma = Nx.div sigma (largest sigma) in
  let w = Nx.div (Nx.sub x' x) sigma in
  let size = norm w in
  let c = Nx.div (dot (Nx.mul sigma y) (Nx.div w (column size))) size in
  let measured = Nx.logical_and (Nx.isfinite c) (Nx.greater_s c 0.) in
  Nx.where measured c (Nx.full_like c Float.nan)

let least mu c =
  Nx.where (Nx.isnan c) mu (Nx.where (Nx.isnan mu) c (Nx.minimum mu c))

type ('q, 'd) limit = {
  error : (float, 'q, 'd) Nx.t;
  answer : (float, 'q, 'd) Nx.t;
  scale : (float, 'q, 'd) Nx.t;
}

(* A lane at the floats' limit gets no reading to replace an older one taken
   before the map settled, as the overshooting first step of a quasi-Newton
   method: its [q] is the newest settled reading. *)
let limit s l ~at =
  let run = Nx.logical_and (searching s.st) at in
  let converged = Nx.logical_and run (within ~scale:l.scale l.error) in
  let st = settle s.st converged Converged in
  {
    s with
    x = hold converged l.answer s.x;
    e = hold run l.error s.e;
    st = settle st run Stalled;
  }

(* The error is tested at the evaluated estimate's step and the answer is the
   point that step reaches, as Deuflhard's NLEQ1 does: if [N] reduces the error
   by [L] from [x], [|x − x*| ≤ |δ| / (1 − L)] by the triangle inequality and
   [|x + δ − x*| ≤ L |δ| / (1 − L)], within [e] wherever [L ≤ q] and within
   [tol] wherever that step at least halves the error. The error with the newest
   reading, which the floats' limit takes, shares all but the gap. *)
let judge tol s ~map ~mu ~resolution ~f delta =
  let run = searching s.st in
  let answer = Nx.add s.x delta in
  let scale = Tol.scale tol answer in
  let size, exact = size ~x:s.x ~f delta in
  let step = Nx.where exact (Nx.zeros_like delta) size in
  (* A quasi-Newton method's error is at least its gradient's bound [b], and
     at most [|δ| + b] whatever its readings: [b] bounds [x]'s distance to
     the zero and [δ] moves the answer from [x], so a map whose readings do
     not contract, as one whose curvature estimate overshoots a component,
     still converges once both are small. *)
  let bounded e =
    let e =
      match (mu, f) with
      | None, _ -> e
      | Some (mu, _), (Paired g | Solved g) ->
          let floor = Nx.clamp ~min:(Num.tiny (Nx.dtype g)) scale in
          let b = gradient_bound ~scale:floor ~mu g in
          Nx.minimum (Nx.maximum e b) (Nx.add step b)
    in
    match resolution with None -> e | Some r -> Nx.maximum e r
  in
  let error q =
    bounded (Nx.where exact (Nx.zeros_like delta) (Nx.div size (gap q)))
  in
  let newest = error s.q in
  (* Where the resolution alone fails [tol], a step within an ulp of the
     estimate's largest magnitude is at the floats' limit as well: no
     component is resolved closer than that, whatever the condition, and no
     later step can meet [tol]. Elsewhere such a step still converges. *)
  let stuck =
    match resolution with
    | None -> still s.x map
    | Some r ->
        Nx.logical_or (still s.x map)
          (Nx.logical_and
             (Nx.logical_not (within ~scale r))
             (within_ulp s.x map))
  in
  let e = Nx.where (column stuck) newest (error (contraction s)) in
  (* A quasi-Newton lane converges once the curvature along its step was
     measured, at the floats' limit, or where its step is an exact zero. *)
  let measured =
    match mu with
    | None -> run
    | Some (_, along) -> Nx.logical_or along (Nx.logical_or stuck (every exact))
  in
  let converged =
    Nx.logical_and (Nx.logical_and run measured) (within ~scale e)
  in
  let st = settle s.st converged Converged in
  ( {
      s with
      x = hold converged answer s.x;
      e = hold run e s.e;
      st = settle st stuck Stalled;
    },
    { error = newest; answer; scale } )

let test tol s ~f delta =
  fst
    (judge tol
       (read tol s ~map:delta)
       ~map:delta ~mu:None ~resolution:None ~f delta)

(* The Rayleigh quotient [yᵀs / sᵀs] of a move [s] that changed the gradient
   by [y], taken on [s / m], [m] its largest magnitude, so that its squares
   neither underflow nor overflow; NaN is none yet. *)
let curvature (c, c') ~step y =
  let m = largest step in
  let u = Nx.div step m in
  let r = Nx.div (dot y u) (Nx.mul (dot u u) (Nx.reshape (lanes step) m)) in
  let measured = Nx.logical_and (Nx.isfinite r) (Nx.greater_s r 0.) in
  (Nx.where measured r c, Nx.where measured c c')

(* The two newest curvatures, as the contraction takes its two newest
   readings: a path's earlier curvatures, far from the answer, say nothing of
   its condition there. *)
let condition (c, c') =
  let kappa = Nx.div (Nx.maximum c c') (Nx.minimum c c') in
  Nx.where (Nx.isfinite kappa) kappa (Nx.ones_like kappa)

let resolution ~kappa answer =
  let dtype = Nx.dtype answer in
  let top = Nx.max ~axes:[ -1 ] ~keepdims:true (Nx.abs answer) in
  Nx.mul (Nx.mul_s (column kappa) (Num.eps dtype)) top

let iterations ~budget extra step (s, aux) =
  let carry =
    Nx.Ptree.(
      pair
        (pair tensor (pair tensor (pair tensor tensor)))
        (pair
           (pair tensor (pair tensor tensor))
           (pair tensor (pair tensor tensor))))
  in
  let pack s =
    ( (s.x, (s.fx, (s.base, s.image))),
      ((s.e, (s.q, s.q')), (s.st, (s.n, s.k))) )
  in
  let unpack ((x, (fx, (base, image))), ((e, (q, q')), (st, (n, k)))) =
    { x; fx; base; image; e; q; q'; st; n; k }
  in
  let step (c, aux) =
    let s = unpack c in
    let s', aux = step s aux in
    let k = Nx.add_s s.k 1l in
    let spent = Nx.greater_equal_s k (Int32.of_int budget) in
    let st =
      settle s'.st (Nx.broadcast_to (Nx.shape s'.st) spent) Budget_spent
    in
    (pack { s' with st; k }, aux)
  in
  let c, aux =
    Rune.iterate
      Nx.Ptree.(pair carry extra)
      ~max:budget
      ~until:(fun ((_, (_, (st, _))), _) ->
        Nx.logical_not (Nx.any (searching st)))
      step
      (pack s, aux)
  in
  (unpack c, aux)
