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

val dot : (float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [dot u v] is each lane's inner product, of the lanes' shape. *)

val norm : (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [norm v] is each lane's Euclidean norm, of the lanes' shape. *)

val descends : (float, 'b) Nx.t -> (float, 'b) Nx.t -> (bool, Nx.bool_elt) Nx.t
(** [descends g d] is [true] in each lane where [gᵀ d < 0], decided on the
    vectors scaled to a largest magnitude of one, so it neither underflows nor
    overflows. *)

val hold :
  (bool, Nx.bool_elt) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t -> ('a, 'c) Nx.t
(** [hold m next last] is [next] in the lanes where [m], [last] elsewhere, for
    tensors whose shape starts with [m]'s. *)

val accepted :
  Tol.t -> e:(float, 'b) Nx.t -> y:(float, 'b) Nx.t -> (bool, Nx.bool_elt) Nx.t
(** [accepted tol ~e ~y] is [true] in each lane whose error [e] at [y] meets
    [tol]: the root mean square of the scaled components is at most [1]. *)

val contraction : (float, 'b) Nx.t -> q:(float, 'b) Nx.t -> (float, 'b) Nx.t
(** [contraction delta ~q] is the error of an iteration whose undamped step is
    [delta] and whose undamped map contracts by [q] per lane:
    [|delta| q / (1 − q)] per component. It is infinite unless [0 ≤ q < 1], and
    zero where [delta] is. *)

val backtrack :
  'p Nx.Ptree.t ->
  (float, 'b) Nx.dtype ->
  trials:int ->
  running:(bool, Nx.bool_elt) Nx.t ->
  shrink:((float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t) ->
  ((float, 'b) Nx.t -> 'p * (float, 'b) Nx.t * (bool, Nx.bool_elt) Nx.t) ->
  'p ->
  'p * (bool, Nx.bool_elt) Nx.t * (int32, Nx.int32_elt) Nx.t * (float, 'b) Nx.t
(** [backtrack p dtype ~trials ~running ~shrink trial last] searches from the
    step length [α = 1] of [dtype] in each running lane: [trial α] is a point's
    payload of structure [p], whose tensors' shapes start with the lanes', its
    merit [φ] and whether the lane accepts it. A lane ends at the first accepted
    [α], and otherwise moves to [shrink α φ], which is at most [α / 2], for at
    most [trials] trials. The result is the accepted payload, or [last] in a
    lane that accepted none, whether each lane accepted, the trials each lane
    took, and the merit of the full step. *)

val c : float
(** [c] is the sufficient-decrease constant, [10⁻⁴]: a step must take [c] of the
    decrease its slope predicts. *)

val armijo :
  phi0:(float, 'b) Nx.t ->
  slope0:(float, 'b) Nx.t ->
  ((float, 'b) Nx.t -> (float, 'b) Nx.t -> (bool, Nx.bool_elt) Nx.t)
  * ((float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t)
(** [armijo ~phi0 ~slope0] is the sufficient-decrease test of a merit [φ] with
    [φ 0 = phi0] and [φ' 0 = slope0], [φ α ≤ φ 0 + c α φ' 0] with [c = 10⁻⁴] and
    [φ α < φ 0] or [φ α = 0], which the first implies unless [c α φ' 0] is below
    [φ 0]'s rounding, and its step: the minimum of the quadratic through [φ 0],
    [φ' 0] and [φ α], kept in [[α / 10, α / 2]]. *)

val wolfe :
  'p Nx.Ptree.t ->
  (float, 'b) Nx.dtype ->
  trials:int ->
  running:(bool, Nx.bool_elt) Nx.t ->
  longest:(float, 'b) Nx.t ->
  phi0:(float, 'b) Nx.t ->
  slope0:(float, 'b) Nx.t ->
  ((float, 'b) Nx.t -> 'p * (float, 'b) Nx.t * (float, 'b) Nx.t) ->
  'p ->
  'p * (bool, Nx.bool_elt) Nx.t * (int32, Nx.int32_elt) Nx.t
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

val finite : (float, 'b) Nx.t -> (bool, Nx.bool_elt) Nx.t
(** [finite v] is [true] in each lane whose components are all finite. *)

val along : (float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [along alpha v] is [alpha v], [alpha] one length per lane. *)

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
(** The state of a Newton-type search: the estimate [x], the function the search
    steps on at [x] ([f] for a system, the gradient for a minimum), the last
    point a step was tested at, the undamped map [N x = x + δ x] there, the
    error estimate, the contraction over the last step the floats resolve, the
    status, the evaluations and the count of iterations, which every lane
    shares. *)

val start : (float, 'd) Nx.t -> (float, 'd) Nx.t -> 'd state
(** [start x0 fx] is the state at [x0], where the function is [fx], after one
    evaluation: a lane where [fx] is not finite ends [Not_finite]. *)

val test : Tol.t -> 'd state -> (float, 'd) Nx.t -> 'd state
(** [test tol s delta] tests the undamped step [delta] at [s.x]:
    [e = contraction delta ~q], with [q] the larger of the contraction of the
    undamped map over the last step, [|N x − N x'| / |x − x'|] from the point
    [x'] tested before [x], and the one the state holds. The state then holds
    the last step's contraction, unless that step lies within a unit in the last
    place of [x'], where the contraction is a ratio of rounding errors. A
    running lane whose error meets [tol] converges at [x + delta], and one whose
    step no longer moves its estimate ends as {!resolved} says. *)

val secant : 'd state -> (float, 'd) Nx.t -> (float, 'd) Nx.t
(** [secant s next] is the contraction [|next − N x'| / |x − x'|] of a map that
    sends [s.x] to [next], per lane. *)

val resolved :
  'd state ->
  delta:(float, 'd) Nx.t ->
  stuck:(bool, Nx.bool_elt) Nx.t ->
  'd state
(** [resolved s ~delta ~stuck] ends each running lane whose step [delta] cannot
    move its estimate, [stuck]: [Converged] with [e = |delta|] when the last
    step the floats resolve contracted, [0 ≤ s.q < 1], at its zero to the
    arithmetic's precision; [Stalled] otherwise. *)

val decide :
  Tol.t ->
  'd state ->
  map:(float, 'd) Nx.t ->
  q:(float, 'd) Nx.t ->
  (float, 'd) Nx.t ->
  'd state
(** [decide tol s ~map ~q delta] is {!test} for a method whose map steps by
    [map] and whose test step is [delta]: [q] is [map]'s contraction, the state
    records [x + map] as the map's value, a lane converges at [x + delta], and
    stalls when [map] no longer moves its estimate. [test] is [decide] with
    [map = delta]. *)

val iterations :
  budget:int ->
  'a Nx.Ptree.t ->
  ('d state -> 'a -> 'd state * 'a) ->
  'd state * 'a ->
  'd state * 'a
(** [iterations ~budget a step (s, aux)] applies [step] until no lane runs, with
    the method's own carry [aux] of structure [a]; a lane still running after
    [budget] iterations ends [Budget_spent]. *)
