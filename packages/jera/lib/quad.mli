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
  type -'k t
  (** The type for rules with tags ['k]. A rule with more tags is one with
      fewer: [(Quad.Rule.kronrod 7 :> [ `Formula ] Quad.Rule.t)]. *)

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
  (** [from a] is the half-line from [a] to [+∞], at unit scale: a rule's nodes
      spread over distances near [1] from [a]. Scale the variable so the
      integrand's width is near 1. *)

  val line : (float, 'b) Nx.t -> 'b t
  (** [line c] is the whole line, around [c] at unit scale. *)
end

(** Boxes of integration, one per lane. *)
module Box : sig
  type 'b t
  (** The type for boxes of dtype ['b]. *)

  val v : (float, 'b) Nx.t -> (float, 'b) Nx.t -> 'b t
  (** [v lo hi] is the box with corners [lo] and [hi], of shape [lanes @ [d]]: a
      point's coordinates are the last axis.

      Raises [Invalid_argument] if [lo] and [hi] differ in shape or are scalars.
  *)
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

(** {1:solves Solves}

    Each solve is elementwise: every element of the range is its own integral,
    with its own status. Its search runs on detached values; the answer of a
    converged element is its rule over its final decisions, tracked, so its
    derivative is that rule's, and an element that did not converge returns its
    detached best estimate. *)

val adaptive :
  [> `Embedded ] Rule.t ->
  tol:Tol.t ->
  budget:int ->
  'b integrand ->
  'b Range.t ->
  (float, 'b) Nx.t Solution.t
(** [adaptive r ~tol ~budget f range] is the integral of [f] over each element
    of [range] by the rule [r] on a partition it refines: it bisects the piece
    of largest error until the error meets [tol]. An infinite range is mapped to
    [[0, 1]] as {!fixed} maps it.

    {b Error.} [e] is the sum over the pieces of [|K − G|], the Kronrod sum's
    difference from its embedded Gauss sum, and [y] the integral. An element
    whose worst piece is at level 62, or holds no float strictly inside, ends
    [Stalled]; a non-finite sum ends it [Not_finite]; [budget] pieces end it
    [Budget_spent]. A feature narrower than the first rule's nodes can be
    invisible to every estimate, and an element can converge without it.
    {b Cost.} [2n + 1] points per piece, and the answer evaluates the final
    partition again, in chunks of 32 pieces under a {!Rune.scan}, so reverse
    mode keeps one chunk's values. {b Derivative.} The final partition's rule's:
    the pieces are integers [(level, index)] whose ends are
    [a + (b − a) index / 2^level], so it reaches the ends.

    Raises [Invalid_argument] if [budget < 1], or as {!fixed} does. *)

val tanh_sinh :
  tol:Tol.t -> 'b integrand -> 'b Range.t -> (float, 'b) Nx.t Solution.t
(** [tanh_sinh ~tol f range] is the integral of [f] over each element of [range]
    by a double-exponential rule: tanh-sinh on a finite range, exp-sinh on
    {!Range.from} and sinh-sinh on {!Range.line}. Its nodes crowd toward the
    ends double-exponentially, so it converges for an integrable singularity at
    an end, and on a half-line or the line for an integrand that decays. Scale
    the variable so the integrand's width is near 1.

    {b Method.} The trapezoidal rule in [t] after the change of variable, at
    steps [1, 1/2, 1/4, ...]: each level adds the odd multiples of its step, in
    fixed-size chunks, to the finest level the dtype calls for ([2^-7] in
    float64, [2^-6] in float32). The nodes stop where their numbers leave the
    dtype's normal floats; a node whose point rounds to an end is unused. Near
    an end [a = 0] a point is its own distance to the end, so the nodes reach
    the singularity in full precision; another end loses the digits [a]'s
    magnitude rounds away, unless the integrand computes the distance to the end
    from its argument. {b Error.} [e] is the difference of the last two levels
    and [y] the integral. A lane whose terms at the truncation do not fall below
    the tolerance, or whose finest level does not meet it, ends [Stalled]; a
    non-finite sum ends it [Not_finite]. {b Derivative.} That of the sum over
    the final level, through the ends and the integrand's parameters. *)

val cubature :
  tol:Tol.t ->
  budget:int ->
  'b integrand ->
  'b Box.t ->
  (float, 'b) Nx.t Solution.t
(** [cubature ~tol ~budget f box] is the integral of [f] over each lane's box,
    of [d] dimensions with [2 ≤ d ≤ 10]. [f] receives points of shape
    [[m; k] @ lanes @ [d]] and reduces only their last, coordinate axis.

    {b Method.} Genz and Malik's (1980) adaptive rule of degree 7 with an
    embedded degree 5, [2^d + 2d² + 2d + 1] points per box: it bisects the box
    of largest error across the axis of largest fourth difference. {b Error.}
    [e] is the sum over the boxes of the two rules' difference, and [y] the
    integral. A lane whose worst box is at level 62 along its axis, or holds no
    float strictly inside along it, ends [Stalled]; a non-finite sum ends it
    [Not_finite]; [budget] boxes end it [Budget_spent]. {b Derivative.} The
    final partition's rule's, through the corners and the integrand's
    parameters.

    Raises [Invalid_argument] if [d] is not in [[2, 10]], if [budget < 1], or if
    [f]'s result is not the points' shape without its last axis. *)

val qmc :
  Nx.Rng.t ->
  tol:Tol.t ->
  budget:int ->
  'b integrand ->
  'b Box.t ->
  (float, 'b) Nx.t Solution.t
(** [qmc key ~tol ~budget f box] is the integral of [f] over each lane's box, of
    any dimension [d] up to 1111, by randomised quasi-Monte Carlo. [f] receives
    points of shape [[c; 16] @ lanes @ [d]] and reduces only their last,
    coordinate axis.

    {b Method.} The mean of [f] over a Sobol sequence (Joe and Kuo's direction
    numbers) under 16 independent random digital shifts drawn from [key]. It
    adds the sequence in chunks of 64 points and tests at each power of two,
    where a Sobol prefix is balanced. Points are [(i + ½) / 2^k] after the
    shift, [k] the bits the dtype holds below 1 (32 in float64), so none lies on
    the box's boundary. {b Error.} [e] is the standard error of the mean over
    the shifts, an estimate of a standard deviation: the test is statistical.
    [y] is the integral. Each estimate at a fixed point count is unbiased, and
    the stopped one to within its standard error. [budget] chunks end a lane
    [Budget_spent]. {b Derivative.} The mean's over the final points: an
    estimate of the integral's derivative where the integrand is Lipschitz in
    the parameter.

    Raises [Invalid_argument] if [d] is above 1111, if [budget < 1], or if [f]'s
    result is not the points' shape without its last axis. *)
