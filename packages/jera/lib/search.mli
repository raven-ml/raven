(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** What the Newton-type searches share: vectors over lanes, the contraction
    estimate and the backtracking line search.

    A vector over lanes is a tensor of shape [lanes @ [n]]: one vector of [n]
    components for each index of [lanes]. A problem that is one system has no
    lane axis, [lanes = [||]]. Sums over a vector add pairwise in a fixed order,
    so eager and compiled searches take the same decisions. *)

val dot : (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t
(** [dot u v] is each lane's inner product, of the lanes' shape. *)

val largest : (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t
(** [largest v] is each lane's largest magnitude, of the lanes' shape and an
    axis of one, clamped to the dtype's normal floats: the vector divided by it
    has a largest magnitude of one, so its squares neither underflow nor
    overflow. *)

val norm : (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t
(** [norm v] is each lane's Euclidean norm, of the lanes' shape. Its sum of
    squares is taken on the vector scaled to a largest magnitude of one, so the
    sum neither underflows nor overflows; the norm itself overflows past the
    largest float. A lane with an infinite component and no NaN has an infinite
    norm, and one with a NaN a NaN norm. *)

val descends : (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t -> 'd Nx.bool_t
(** [descends g d] is [true] in each lane where [gᵀ d < 0], decided on the
    vectors scaled to a largest magnitude of one, so it neither underflows nor
    overflows. *)

val hold :
  'd Nx.bool_t -> ('a, 'c, 'd) Nx.t -> ('a, 'c, 'd) Nx.t -> ('a, 'c, 'd) Nx.t
(** [hold m next last] is [next] in the lanes where [m], [last] elsewhere, for
    tensors whose shape starts with [m]'s. *)

val accepted :
  Tol.t -> e:(float, 'b, 'd) Nx.t -> y:(float, 'b, 'd) Nx.t -> 'd Nx.bool_t
(** [accepted tol ~e ~y] is [true] in each lane whose error [e] at [y] meets
    [tol]: the root mean square of the scaled components is at most [1]. *)

val moves : from:(float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t -> 'd Nx.bool_t
(** [moves ~from x] is [true] in each lane where [x] differs from [from] in
    some component. *)

val backtrack :
  ('p, 'd) Nx.Ptree.t ->
  (float, 'b) Nx.dtype ->
  trials:int ->
  running:'d Nx.bool_t ->
  shrink:((float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t) ->
  ((float, 'b, 'd) Nx.t ->
  'p * (float, 'b, 'd) Nx.t * 'd Nx.bool_t * 'd Nx.bool_t) ->
  'p ->
  'p * 'd Nx.bool_t * 'd Nx.int32_t
(** [backtrack p dtype ~trials ~running ~shrink trial last] searches from the
    step length [α = 1] of [dtype] in each running lane: [trial α] is a point's
    payload of structure [p], whose tensors' shapes start with the lanes', its
    merit [φ], whether the lane accepts it and whether the point differs from
    the search's start. A lane ends at the first accepted [α] whose point
    differs, ends without a step at a point that does not, since every shorter
    step rounds to it too, and otherwise moves to [shrink α φ], which is at
    most [α / 2], for at most [trials] trials. The result is the accepted
    payload, or [last] in a lane that accepted none, whether each lane
    accepted and the trials each lane took. *)

val c : float
(** [c] is the sufficient-decrease constant, [10⁻⁴]: a step must take [c] of the
    decrease its slope predicts. *)

val armijo :
  phi0:(float, 'b, 'd) Nx.t ->
  slope0:(float, 'b, 'd) Nx.t ->
  ((float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t -> 'd Nx.bool_t)
  * ((float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t)
(** [armijo ~phi0 ~slope0] is the sufficient-decrease test of a merit [φ] with
    [φ 0 = phi0] and [φ' 0 = slope0], [φ α ≤ φ 0 + c α φ' 0] with [c = 10⁻⁴] and
    [φ α < φ 0] or [φ α = 0], which the first implies unless [c α φ' 0] is below
    [φ 0]'s rounding, and its step: the minimum of the quadratic through [φ 0],
    [φ' 0] and [φ α], kept in [[α / 10, α / 2]]. *)

val wolfe :
  ('p, 'd) Nx.Ptree.t ->
  (float, 'b) Nx.dtype ->
  trials:int ->
  running:'d Nx.bool_t ->
  longest:(float, 'b, 'd) Nx.t ->
  phi0:(float, 'b, 'd) Nx.t ->
  slope0:(float, 'b, 'd) Nx.t ->
  ((float, 'b, 'd) Nx.t -> 'p * (float, 'b, 'd) Nx.t * (float, 'b, 'd) Nx.t) ->
  'p ->
  'p * 'd Nx.bool_t * 'd Nx.int32_t
(** [wolfe p dtype ~trials ~running ~longest ~phi0 ~slope0 trial origin]
    searches each running lane for a step length [α] that meets the strong Wolfe
    conditions on the merit [φ] with [φ 0 = phi0] and [φ' 0 = slope0 < 0]:
    [φ α ≤ φ 0 + c α φ' 0] with [c = 10⁻⁴], and [|φ' α| ≤ 0.9 |φ' 0|]; a trial
    within [√ε |φ 0|] of [φ 0] meets the first by its slope,
    [φ' α ≤ (2c − 1) φ' 0], which a decrease below [φ]'s rounding still shows.
    [trial α] is the point's payload of structure [p], [φ α] and [φ' α];
    bracketing never tries past [longest], per lane, and a decreasing trial
    there ends it; [origin] is the payload at [α = 0]. Bracketing doubles [α]
    from [1], then zooms by safeguarded quadratic steps, bisecting when the
    bracket has not halved over two trials, for at most [trials] trials. The
    result is the accepted payload or, when the trials ran out, the one of least
    merit that met the sufficient decrease; whether a lane found either; and the
    trials each lane took. *)

(** {1:iterations Iterations} *)

val finite : (float, 'b, 'd) Nx.t -> 'd Nx.bool_t
(** [finite v] is [true] in each lane whose components are all finite. *)

val along : (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t -> (float, 'b, 'd) Nx.t
(** [along alpha v] is [alpha v], [alpha] one length per lane. *)

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
(** The state of a Newton-type search: the estimate [x], the function the search
    steps on at [x] ([f] for a system, the gradient for a minimum), the first
    end [x'] of the secant that reads the contraction of the undamped map
    [N x = x + δ x] and [N x'], the error estimate, the newest settled reading
    of the contraction and the one before it, NaN while there is none, the
    status, the evaluations and the count of iterations, which every lane
    shares. *)

val start : (float, 'q, 'd) Nx.t -> (float, 'q, 'd) Nx.t -> ('q, 'd) state
(** [start x0 fx] is the state at [x0], where the function is [fx], after one
    evaluation: a lane where [fx] is not finite ends [Not_finite]. *)

val contraction : ('q, 'd) state -> (float, 'q, 'd) Nx.t
(** [contraction s] is the contraction the error takes, per lane: the larger of
    [s.q] and [s.q'], [s.q] while [s.q'] is NaN, and NaN before any reading. *)

val read : Tol.t -> ('q, 'd) state -> map:(float, 'q, 'd) Nx.t -> ('q, 'd) state
(** [read tol s ~map] reads the contraction of the map that sends [s.x] to
    [s.x + map] in each running lane: [q = |N x − N x'| / |x − x'| + r] over
    the secant from [x' = s.base], lengths in [tol]'s scaled norm at
    [max (|x|, |x'|)], with [r = ε (|N x| + |N x'|) / (2 |x − x'|)] the
    rounding of [N]'s two values. A reading settled by its rounding,
    [q < 1] or [q − 2r ≥ 1], becomes [s.q], the one before [s.q'], and the
    secant's first end moves to [s.x]; an unsettled one leaves the first end,
    so the next secant spans more steps, and drops both readings where its
    lower end [q − 2r] exceeds {!contraction}. A secant of no length, as before
    the first reading, reads nothing and starts at [s.x]. *)

type ('q, 'd) source =
  | Paired of (float, 'q, 'd) Nx.t
      (** The function's component [i] is the equation of unknown [i]: an
          elementwise solve, a fixed point, a gradient. *)
  | Solved of (float, 'q, 'd) Nx.t
      (** The step solves a linear system on the function, so each of its
          components answers to every equation. *)
(** The function a step is taken from, which decides when a zero step is a
    zero. *)

val error :
  (float, 'q, 'd) Nx.t ->
  x:(float, 'q, 'd) Nx.t ->
  f:('q, 'd) source ->
  (float, 'q, 'd) Nx.t ->
  (float, 'q, 'd) Nx.t
(** [error q ~x ~f delta] is [|delta| / (1 − q)] per component of the step
    [delta] from [x], [q] one contraction per lane: infinite unless
    [0 ≤ q < 1]. A component of [delta] below the dtype's least normal float
    counts the subnormals' spacing as well. A zero one counts zero where it is
    resolved and [x + delta] is not a subnormal there: for [Paired f] where
    [f] is zero there, for [Solved f] where [f] is zero or its largest
    magnitude is at least [tiny / ε]. *)

val least :
  Tol.t ->
  (float, 'q, 'd) Nx.t ->
  x:(float, 'q, 'd) Nx.t ->
  x':(float, 'q, 'd) Nx.t ->
  (float, 'q, 'd) Nx.t ->
  (float, 'q, 'd) Nx.t
(** [least tol mu ~x ~x' y] is [mu] lowered, per lane, to the curvature
    [yᵀd / |d / σ|²] of the move [d] from [x] to [x'] that changed the gradient
    by [y], [σ] [tol]'s scale at the move's larger end, where that curvature is
    positive; NaN [mu] is none yet. *)

type ('q, 'd) limit
(** What a lane at the floats' limit converges with: its error with the newest
    settled reading, its answer and the answer's scale. *)

val judge :
  Tol.t ->
  ('q, 'd) state ->
  map:(float, 'q, 'd) Nx.t ->
  mu:(float, 'q, 'd) Nx.t option ->
  resolution:(float, 'q, 'd) Nx.t option ->
  f:('q, 'd) source ->
  (float, 'q, 'd) Nx.t ->
  ('q, 'd) state * ('q, 'd) limit
(** [judge tol s ~map ~mu ~resolution ~f delta] tests the undamped step [delta]
    at [s.x] in each running lane, [f] the function the step is taken from: the
    lane converges at [s.x + delta] where its error meets [tol] there. The error
    is [error q ~x:s.x ~f delta] with [q] the {!contraction}; with [mu], the
    least curvature a quasi-Newton method's steps measured, at least the
    gradient's bound [b = |σ g| / (mu √n)] times each component's scale [σ] at
    [s.x + delta], [g] the gradient [f] holds, and at most [|delta| + b]; and
    with [resolution], at least it, per component. A lane whose [map], the step
    the method takes, cannot move its estimate, one within a unit in the last
    place of every component or, where [resolution] alone fails [tol], of the
    estimate's largest magnitude, is at the floats' limit, and ends as {!limit}
    says. The result holds what {!limit} needs for the lanes the method's search
    leaves there. *)

val limit : ('q, 'd) state -> ('q, 'd) limit -> at:'d Nx.bool_t -> ('q, 'd) state
(** [limit s l ~at] ends each running lane of [at] at the floats' limit, with
    the [l] of the {!judge} that led to [s]: it converges at its answer where
    its error with [q] the newest settled reading [s.q] meets the tolerance, and
    stalls otherwise. *)

val test :
  Tol.t -> ('q, 'd) state -> f:('q, 'd) source -> (float, 'q, 'd) Nx.t -> ('q, 'd) state
(** [test tol s ~f delta] is
    [fst (judge tol (read tol s ~map:delta) ~map:delta ~mu:None
     ~resolution:None ~f delta)], for a method whose map steps by its test
    step and that has no line search. *)

val curvature :
  (float, 'q, 'd) Nx.t * (float, 'q, 'd) Nx.t ->
  step:(float, 'q, 'd) Nx.t ->
  (float, 'q, 'd) Nx.t ->
  (float, 'q, 'd) Nx.t * (float, 'q, 'd) Nx.t
(** [curvature (c, c') ~step y] is, per lane, the two newest curvatures
    measured, NaN before any: the Rayleigh quotient [yᵀs / sᵀs] of the move
    [s = step] that changed the gradient by [y] and [c], where the quotient is
    positive, and [(c, c')] elsewhere. *)

val condition :
  (float, 'q, 'd) Nx.t * (float, 'q, 'd) Nx.t -> (float, 'q, 'd) Nx.t
(** [condition (c, c')] is, per lane, the larger of the two newest curvatures
    over the smaller, 1 before two: an estimate of the condition near the
    answer, where a path's earlier curvatures say nothing of it. *)

val resolution :
  kappa:(float, 'q, 'd) Nx.t -> (float, 'q, 'd) Nx.t -> (float, 'q, 'd) Nx.t
(** [resolution ~kappa answer] is [ε κ |answer|∞] per lane, of the lanes'
    shape and an axis of one: how far a coupled problem's zero is fixed, its
    function's rounding [ε] times its largest terms, amplified by the
    condition [κ = kappa]. *)

val iterations :
  budget:int ->
  ('a, 'd) Nx.Ptree.t ->
  (('q, 'd) state -> 'a -> ('q, 'd) state * 'a) ->
  ('q, 'd) state * 'a ->
  ('q, 'd) state * 'a
(** [iterations ~budget a step (s, aux)] applies [step] until no lane runs, with
    the method's own carry [aux] of structure [a]; a lane still running after
    [budget] iterations ends [Budget_spent]. *)
