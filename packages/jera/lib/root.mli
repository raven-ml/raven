(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Zeros of functions of one variable.

    Both methods are elementwise: every element of the input is its own problem,
    with its own status, and [f] must compute each element of its result from
    the same element of its argument alone. The answer is stated as a zero of
    [f] ({!Rune.root}), so its derivative is the implicit one, [−∂f/∂θ / ∂f/∂x]
    at a converged element, through every tracked value [f] reads, and zero at
    an element that did not converge. At a converged zero where [∂f/∂x] is [0]
    the derivative does not exist and is not finite.

    The derivative solves elementwise and checks its solution with one more
    product: where it misses, [f] read another element than its own, and the
    derivative raises [Invalid_argument] naming the entry point.

    {[
    (* Kepler's equation M = E − e sin E, for each epoch *)
    let eccentric_anomaly m e =
      Root.newton
        ~tol:(Tol.v ~rel:1e-12 ~abs:1e-15)
        ~budget:8
        ~slope:(fun x -> Nx.rsub_s 1. (Nx.mul e (Nx.cos x)))
        (fun x -> Nx.sub (Nx.sub x (Nx.mul e (Nx.sin x))) m)
        (Nx.add m (Nx.mul e (Nx.sin m)))
      |> Solution.get
    ]} *)

val bracket :
  tol:Tol.t ->
  ((float, 'b) Nx.t -> (float, 'b) Nx.t) ->
  lo:(float, 'b) Nx.t ->
  hi:(float, 'b) Nx.t ->
  (float, 'b) Nx.t Solution.t
(** [bracket ~tol f ~lo ~hi] is a zero of [f] between [lo] and [hi] in each
    element, given in either order, which broadcast together.

    {b Method.} ITP steps (Oliveira and Takahashi, 2020) interleaved with
    bisections of the floats' ordered-integer images, so an element ends within
    [2b + 2] evaluations for a [b]-bit dtype (130 in float64), whatever [f].
    {b Error.} [e] is half the final bracket and [y] its midpoint; an element
    also ends when no float lies strictly inside its bracket. It converges only
    if [|f|] at its estimate, the bracket's end of smaller [|f|], is at most the
    smaller [|f|] at the given ends, so a pole ends [Stalled]; a bounded jump
    looks like a steep zero at float resolution and converges at the jump, where
    the derivative does not exist. Ends of one sign end [Not_bracketed]; a
    non-finite value of [f] ends [Not_finite]. *)

val newton :
  tol:Tol.t ->
  budget:int ->
  slope:((float, 'b) Nx.t -> (float, 'b) Nx.t) ->
  ((float, 'b) Nx.t -> (float, 'b) Nx.t) ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t Solution.t
(** [newton ~tol ~budget ~slope f x0] is a zero of [f] near [x0] in each
    element, by Newton's method: [slope x] is [f]'s derivative, elementwise. It
    steers the steps only, so a wrong slope slows a solve and cannot end it
    early.

    {b Method.} Undamped Newton steps [δ = −f x / slope x], quadratic near a
    simple zero. {b Error.} [e = |δ| q / (1 − q)], with [q] the ratio of the
    last two steps, the distance left when the steps shrink by [q]: unbounded
    while [q ≥ 1] or before two steps; [y] is the estimate. A zero step is a
    zero. A zero or non-finite slope, or a step that no longer moves the
    estimate without meeting [tol], ends the element [Stalled]; a non-finite [f]
    ends it [Not_finite]; [budget] iterations end it [Budget_spent], so a budget
    of 1 converges only on a zero step. {b Cost.} One call of [f] and one of
    [slope] per iteration.

    Raises [Invalid_argument] if [budget < 1]. *)
