(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tensor-product series over grids.

    A grid value is the tensor product of {!Piecewise}'s representation: along
    each of its [d] axes, breaks and a Chebyshev series per piece, and in
    between, the series of their products. Its values have shape [value] at each
    point; a point is the last axis of a tensor, its [d] coordinates.

    {[
    let table = Grid.cubic `Not_a_knot ~axes:[ temperature; pressure ] density
    let rho = Grid.eval table states (* states of shape [n; 2] *)
    ]}

    The domain is the closed box between each axis's first and last break, with
    {!Piecewise}'s rules along each axis: a point on a break lies in the piece
    that ends there, or the first piece at the first break; NaN evaluates to
    NaN, and any other point outside raises [Invalid_argument] through
    {!Nx.check}.

    {b Cost.} Evaluation searches each axis's breaks, gathers [Π (degree_k + 1)]
    coefficients per point, and runs Clenshaw's recurrence along each axis in
    turn. {b Derivative.} The composition's, in the values, the axes and the
    points. *)

type 'b t
(** The type for grid series over breaks of dtype ['b]. The coefficients have
    shape [[pieces_1; …; pieces_d; degree_1 + 1; …; degree_d + 1] @ value]. *)

(** {1:interpolants Interpolants}

    Each takes [d ≥ 1] axes of knots, each 1-D with at least two strictly
    increasing knots, and values of shape [[n_1; …; n_d] @ value], [n_k] the
    knots of axis [k], and passes through every value. Each raises
    [Invalid_argument] if an axis is not 1-D with two knots, if the values'
    leading axes do not match the axes, or, through {!Nx.check}, if an axis is
    not strictly increasing. *)

val linear : axes:(float, 'b) Nx.t list -> (float, 'b) Nx.t -> 'b t
(** [linear ~axes values] is the multilinear interpolant: degree 1 along each
    axis. *)

val cubic :
  [ `Natural | `Not_a_knot ] ->
  axes:(float, 'b) Nx.t list ->
  (float, 'b) Nx.t ->
  'b t
(** [cubic ends ~axes values] is the tensor-product cubic spline, with
    {!Piecewise.ends}' [ends] along every axis: degree 3 along each axis. *)

(** {1:fits Fits} *)

val chebyshev :
  degree:int ->
  pieces:int ->
  ((float, 'b) Nx.t -> (float, 'b) Nx.t) ->
  lo:(float, 'b) Nx.t ->
  hi:(float, 'b) Nx.t ->
  'b t
(** [chebyshev ~degree ~pieces f ~lo ~hi] interpolates [f] on the box from [lo]
    to [hi], both of shape [[d]], split into [pieces] equal pieces along each
    axis, at the tensor product of each piece's [degree + 1] Chebyshev points of
    the second kind. [f] receives points of shape [q @ [d]] and returns values
    of shape [q @ value]. It costs one call of [f] on
    [(pieces × (degree + 1))^d] points.

    Raises [Invalid_argument] if [degree < 0], [pieces < 1], if [lo] and [hi]
    are not of one shape [[d]] with [d ≥ 1], or if [f]'s result does not start
    with the points' shape without their last axis. *)

(** {1:eval Evaluation} *)

val eval : 'b t -> (float, 'b) Nx.t -> (float, 'b) Nx.t
(** [eval g x] is [g] at the points [x] of shape [q @ [d]]: of shape
    [q @ value].

    Raises [Invalid_argument] if [x]'s last axis is not [d], and through
    {!Nx.check} if a point that is not NaN lies outside the domain. *)

(** {1:calculus Calculus} *)

val derivative : axis:int -> 'b t -> 'b t
(** [derivative ~axis g] is the partial derivative of [g] along [axis], of one
    degree less along it.

    Raises [Invalid_argument] if [axis] is negative or not below [d]. *)

val integral : axis:int -> 'b t -> 'b t
(** [integral ~axis g] is the antiderivative of [g] along [axis], zero at that
    axis's first break, of one degree more along it.

    Raises [Invalid_argument] if [axis] is negative or not below [d]. *)

(** {1:access Access} *)

val ptree : (float, 'b) Nx.dtype -> 'b t Nx.Ptree.t
(** [ptree dtype] is the structure of grid values over breaks of [dtype]: the
    breaks of each axis as a list at [breaks], and the coefficients at
    [coefficients]. *)
