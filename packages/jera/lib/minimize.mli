(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Minima. *)

val bracket :
  tol:Tol.t ->
  ((float, 'b) Nx.t -> (float, 'b) Nx.t) ->
  lo:(float, 'b) Nx.t ->
  hi:(float, 'b) Nx.t ->
  (float, 'b) Nx.t Solution.t
(** [bracket ~tol f ~lo ~hi] is a minimum of [f] in each [[lo, hi]],
    elementwise: every element is its own problem, the ends broadcast together
    and come in either order, and [f] must compute each element of its result
    from the same element of its argument alone.

    {b Method.} Brent's method: parabolic steps through the three best points,
    forced to a golden-section step whenever the bracket has not shrunk by
    [0.618] over two evaluations, so an element ends within about [3b]
    evaluations for a [b]-bit dtype. On a function with several minima in the
    bracket it finds one of them. {b Error.} [e] is half the final bracket and
    [y] its midpoint; an element also ends when no float lies strictly inside
    its bracket. A non-finite value of [f] ends it [Not_finite]. {b Derivative.}
    A minimum inside the bracket is stated as a zero of rune's derivative of
    [f], so its derivative is [−∂²f/∂x∂θ / ∂²f/∂x²]; a minimum at an end is that
    end, with its derivative; a lane that did not converge has a zero
    derivative. *)
