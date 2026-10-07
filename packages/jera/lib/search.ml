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

let dot u v =
  let uv = Nx.mul u v in
  Nx.reshape (lanes uv) (Num.sum_rows (rows uv))

let norm v = Nx.sqrt (dot v v)

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

let accepted tol ~e ~y =
  let r = Num.rms_rows (rows (Tol.ratio tol ~e ~y)) in
  Nx.less_equal_s (Nx.reshape (lanes e) r) 1.

(* Contraction *)

let contraction delta ~q =
  let shrinking = Nx.logical_and (Nx.greater_equal_s q 0.) (Nx.less_s q 1.) in
  let factor =
    Nx.where shrinking
      (Nx.div q (Nx.rsub_s 1. q))
      (Nx.full_like q Float.infinity)
  in
  (* A zero component counts zero, whatever the factor. *)
  let e =
    Nx.mul (Nx.abs delta)
      (Nx.reshape (Array.append (Nx.shape factor) [| 1 |]) factor)
  in
  Nx.where (Nx.equal_s delta 0.) (Nx.zeros_like delta) e

(* Backtracking

   The carry holds the trial length, the payload, whether each lane still
   searches, the trials each lane took and the count of trips. *)

let backtrack p dtype ~trials ~running ~shrink trial last =
  let step (alpha, (payload, (searching, (n, j)))) =
    let next, phi, accepted = trial alpha in
    let n = Nx.add n (Nx.cast Nx.int32 searching) in
    let ok = Nx.logical_and searching accepted in
    let payload = Nx.Ptree.map2 p (fun _ a b -> hold ok a b) next payload in
    let searching = Nx.logical_and searching (Nx.logical_not ok) in
    let alpha = Nx.where searching (shrink alpha phi) alpha in
    (alpha, (payload, (searching, (n, Nx.add_s j 1l))))
  in
  let _, (payload, (searching, (n, _))) =
    Rune.iterate
      Nx.Ptree.(pair tensor (pair p (pair tensor (pair tensor tensor))))
      ~max:trials
      ~until:(fun (_, (_, (searching, (_, j)))) ->
        Nx.logical_or
          (Nx.logical_not (Nx.any searching))
          (Nx.greater_equal_s j (Int32.of_int trials)))
      ~f:step
      ( Nx.ones dtype (Nx.shape running),
        ( last,
          ( running,
            (Nx.zeros Nx.int32 (Nx.shape running), Nx.scalar Nx.int32 0l) ) ) )
  in
  (payload, Nx.logical_and running (Nx.logical_not searching), n)

(* The sufficient-decrease constant: a step must take [c] of the decrease the
   slope predicts. *)
let c = 1e-4

let armijo ~phi0 ~slope0 =
  let accept alpha phi =
    Nx.less_equal phi (Nx.add phi0 (Nx.mul_s (Nx.mul alpha slope0) c))
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

type ('p, 'd) wolfe = {
  alpha : (float, 'd) Nx.t;
  lo : (float, 'd) Nx.t * ((float, 'd) Nx.t * (float, 'd) Nx.t);
  at : 'p;
  hi : (float, 'd) Nx.t * (float, 'd) Nx.t;
  zoom : (bool, Nx.bool_elt) Nx.t;
  searching : (bool, Nx.bool_elt) Nx.t;
  widths : (float, 'd) Nx.t * (float, 'd) Nx.t;
  tries : (int32, Nx.int32_elt) Nx.t;
  trip : (int32, Nx.int32_elt) Nx.t;
}

let wolfe p dtype ~trials ~running ~phi0 ~slope0 trial origin =
  let lanes = Nx.shape running in
  let zero = Nx.zeros dtype lanes
  and inf = Nx.full dtype lanes Float.infinity in
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
    let accept = Nx.logical_and low flat in
    let rising =
      Nx.where s.zoom
        (Nx.greater_equal_s (Nx.mul slope (Nx.sub ha la)) 0.)
        (Nx.greater_equal_s slope 0.)
    in
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
    let next = Nx.where zoom inner (Nx.mul_s s.alpha 2.) in
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
      ~f:(fun c -> pack (step (unpack c)))
      (pack
         {
           alpha = Nx.ones dtype lanes;
           lo = (zero, (phi0, slope0));
           at = origin;
           hi = (zero, phi0);
           zoom = Nx.zeros Nx.bool lanes;
           searching = running;
           widths = (inf, inf);
           tries = Nx.zeros Nx.int32 lanes;
           trip = Nx.scalar Nx.int32 0l;
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
   minimum), the last point a step was tested at and the undamped map [N x = x +
   δ x] there, the error estimate and the contraction [q] of the last test, the
   status, the evaluations and the count of iterations, which every lane
   shares. *)

type 'd state = {
  x : (float, 'd) Nx.t;
  fx : (float, 'd) Nx.t;
  before : (float, 'd) Nx.t;
  mapped : (float, 'd) Nx.t;
  e : (float, 'd) Nx.t;
  q : (float, 'd) Nx.t;
  st : (int32, Nx.int32_elt) Nx.t;
  n : (int32, Nx.int32_elt) Nx.t;
  k : (int32, Nx.int32_elt) Nx.t;
}

let finite v = Nx.all ~axes:[ -1 ] (Nx.isfinite v)

let along alpha v =
  Nx.mul (Nx.reshape (Array.append (Nx.shape alpha) [| 1 |]) alpha) v

let start x0 fx =
  let lanes = lanes x0 in
  let st =
    settle
      (Nx.full Nx.int32 lanes running)
      (Nx.logical_not (finite fx))
      Not_finite
  in
  let nan = Nx.full_like x0 Float.nan in
  {
    x = x0;
    fx;
    before = nan;
    mapped = nan;
    e = Nx.full_like x0 Float.infinity;
    q = Nx.full (Nx.dtype x0) lanes Float.nan;
    st;
    n = Nx.ones Nx.int32 lanes;
    k = Nx.scalar Nx.int32 0l;
  }

(* [q] is the contraction of the undamped map over the last step, [|N x − N x'|
   / |x − x'|] from the last tested point [x']: when that step was taken in
   full, [x = N x'] and [q] is the ratio of the last two undamped steps, and
   after a shortened, damped or mixed step it is still [N]'s contraction, which
   the length of the step taken never enters. The error takes the larger of the
   last two contractions: a quasi-Newton method's rate varies from step to step,
   and one small ratio would promise steps that shrink faster than the next one
   does. *)
(* A step that did not move the estimate has no contraction: [q] is then
   infinite, so the error stays unbounded. *)
let secant s next =
  let moved = norm (Nx.sub s.x s.before) in
  let still = Nx.equal_s moved 0. in
  Nx.where still
    (Nx.full_like moved Float.infinity)
    (Nx.div
       (norm (Nx.sub next s.mapped))
       (Nx.where still (Nx.ones_like moved) moved))

let decide tol s ~map ~q delta =
  let run = searching s.st in
  let next = Nx.add s.x delta and mapped = Nx.add s.x map in
  let e = contraction delta ~q:(Nx.maximum q s.q) in
  let st = settle s.st (accepted tol ~e ~y:next) Converged in
  let converged =
    Nx.logical_and run (Nx.equal_s st (Solution.code Converged))
  in
  let st = settle st (Nx.all ~axes:[ -1 ] (Nx.equal mapped s.x)) Stalled in
  {
    s with
    x = hold converged next s.x;
    before = hold run s.x s.before;
    mapped = hold run mapped s.mapped;
    e = hold run e s.e;
    q = hold run q s.q;
    st;
  }

let test tol s delta =
  decide tol s ~map:delta ~q:(secant s (Nx.add s.x delta)) delta

let iterations ~budget extra step (s, aux) =
  let carry =
    Nx.Ptree.(
      pair
        (pair tensor (pair tensor (pair tensor tensor)))
        (pair (pair tensor tensor) (pair tensor (pair tensor tensor))))
  in
  let pack s =
    ((s.x, (s.fx, (s.before, s.mapped))), ((s.e, s.q), (s.st, (s.n, s.k))))
  in
  let unpack ((x, (fx, (before, mapped))), ((e, q), (st, (n, k)))) =
    { x; fx; before; mapped; e; q; st; n; k }
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
      ~f:step
      (pack s, aux)
  in
  (unpack c, aux)
