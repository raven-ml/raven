(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Integrals.

    An integrand is an elementwise function: it receives points with jera's axes
    in front and the lanes' shape behind, and returns one value per point, of
    the points' shape. Every element of a range is its own integral, so the
    integrand must not reduce or mix along any axis.

    {[
    (* ∫₀¹ x^a dx for each a, by 10-point Gauss–Legendre *)
    let moments a =
      Quad.fixed (Quad.Rule.gauss 10)
        (fun x -> Nx.pow x a)
        (Quad.Range.v (Nx.zeros_like a) (Nx.ones_like a))
    ]}

    An integral of samples is the integral of their interpolant
    ({!Piecewise.integral}). *)

(** {1:rules Rules} *)

(** Quadrature rules on [[−1, 1]].

    A rule's tag says which integrals take it: [`Formula] rules sum their nodes
    in {!fixed} and {!cumulative}; [`Embedded] rules also estimate their error.
*)
module Rule : sig
  type 'k t
  (** The type for rules with tags ['k]. *)

  val gauss : int -> [ `Formula ] t
  (** [gauss n] is the [n]-point Gauss–Legendre rule, exact for polynomials of
      degree [2n − 1]. Its nodes and weights are computed on the host in float64
      by Newton's method on the Legendre polynomial.

      Raises [Invalid_argument] if [n < 1]. *)

  val kronrod : int -> [ `Formula | `Embedded ] t
  (** [kronrod n] is the [(2n + 1)]-point Gauss–Kronrod rule that extends
      {!gauss}[ n] (Piessens et al., QUADPACK, 1983): exact for polynomials of
      degree [3n + 1], with the [n] Gauss nodes among its own. [n] is [7] or
      [10].

      Raises [Invalid_argument] if [n] is neither. *)

  val nodes : _ t -> (float, 'b) Nx.dtype -> (float, 'b) Nx.t * (float, 'b) Nx.t
  (** [nodes r dtype] is [(x, w)], [r]'s nodes in increasing order and their
      weights, 1-D tensors of [dtype]: [Σ w f(x)] approximates [∫₋₁¹ f]. A sum
      in log space is [logsumexp (log w + ℓ x)]. *)
end

(** {1:ranges Ranges} *)

(** Ranges of integration, one per element. *)
module Range : sig
  type 'b t
  (** The type for ranges of dtype ['b]. *)

  val v : (float, 'b) Nx.t -> (float, 'b) Nx.t -> 'b t
  (** [v a b] is [[a, b]], [a] and [b] broadcast together. An integral over
      [[a, b]] with [b < a] is minus the integral over [[b, a]]. *)

  val from : (float, 'b) Nx.t -> 'b t
  (** [from a] is the half-line from [a] to [+∞], at unit scale: a rule's nodes spread over
      distances near [1] from [a]. Scale the variable so the integrand's width
      is near 1. *)

  val line : (float, 'b) Nx.t -> 'b t
  (** [line c] is the whole line, around [c] at unit scale. *)
end

type 'b integrand = (float, 'b) Nx.t -> (float, 'b) Nx.t
(** The type for integrands: [f x] is [f] at each point of [x], of [x]'s shape.
*)

(** {1:formulas Formulas} *)

val fixed :
  [> `Formula ] Rule.t -> 'b integrand -> 'b Range.t -> (float, 'b) Nx.t
(** [fixed r f range] is [r]'s sum for the integral of [f] over each element of
    [range], of [range]'s shape. [f] receives the points of shape [[m] @ shape],
    [m] the rule's nodes. A finite range maps the rule linearly; [from a] by
    [x = a + (1 + u) / (1 − u)] and [line c] by [x = c + u / (1 − u²)], so an
    infinite range needs an integrand that decays fast enough for the rule's
    degree to show.

    {b Error.} An [n]-point Gauss sum on a finite range is exact for polynomials
    of degree [2n − 1], and for a smooth [f] its error falls geometrically in
    [n]. {b Cost.} One call of [f] on [m] points per element. {b Derivative.}
    The sum's: in the integrand's parameters, in the ends and in the points [f]
    reads.

    Raises [Invalid_argument] if [f]'s result has another shape than its points.
*)

val cumulative :
  [> `Formula ] Rule.t -> 'b integrand -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [cumulative r f knots] is the integral of [f] from the first knot to each,
    by [r] on each interval between knots: for [knots] of shape [[n] @ shape], a
    result of the same shape whose row [0] is zero and row [i] the sum of the
    first [i] intervals' integrals. [f] receives points of shape
    [[m; n − 1] @ shape].

    Raises [Invalid_argument] if [knots] has no axis or no knot, or as {!fixed}
    does. *)
