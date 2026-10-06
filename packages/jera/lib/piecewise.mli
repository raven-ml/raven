(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Piecewise Chebyshev series.

    One value serves every approximation of a function of one variable: splines
    and other interpolants of samples, fits of a function, and their derivatives
    and integrals. On piece [i], between the breaks [x_i] and [x_(i+1)], the
    value is [Σ_k c_k T_k(u)] with [u = 2 (x − x_i) / (x_(i+1) − x_i) − 1] in
    [[−1, 1]]. A value is plain tensors, so it is an argument, a result and a
    carry of every transformation, and closed under evaluation, differentiation
    and integration.

    {[
    let spline = Piecewise.cubic `Natural knots samples
    let slope = Piecewise.eval (Piecewise.derivative spline) x
    ]}

    {b Domain.} The domain is the closed interval from the first break to the
    last. A point on a break lies in the last piece of positive width that ends
    there, at [u = 1], and a point on the first break in the first piece. NaN
    evaluates to NaN. An infinite point raises [Invalid_argument] through
    {!Nx.check}, and so does any other point outside the domain unless the value
    is extended ({!extend}).

    {b Cost.} Evaluation is a binary search of the breaks ({!Nx.searchsorted}),
    a gather of [degree + 1] coefficients and Clenshaw's recurrence:
    [O(log pieces + degree)] per point. Every operation here compiles to a fixed
    graph of tensor operations, fits included.

    {b Derivative.} Every function is a composition of tensor operations, so it
    differentiates in the coefficients, the breaks, the samples and the points;
    the piece a point falls in carries no derivative. *)

type ('v, 'b) t
(** The type for piecewise series with values of structure ['v] over breaks of
    dtype ['b]: each float leaf of the coefficients has shape
    [[pieces; degree + 1] @ value], and evaluation at points of shape [q] gives
    it shape [q @ value]. Leaves may differ in degree and dtype.

    It has [pieces + 1] breaks, non-decreasing, the first below the last. A
    piece of zero width is empty: no point lies in it, and the end pieces are
    the first and last of positive width. *)

val v : 'v Nx.Ptree.t -> breaks:(float, 'b) Nx.t -> 'v -> ('v, 'b) t
(** [v s ~breaks c] is the series with [pieces + 1] [breaks] and the
    coefficients [c], each leaf of shape [[pieces; degree + 1] @ value].

    Raises [Invalid_argument] if [breaks] is not 1-D with at least two elements,
    if a leaf of [c] is not a float tensor of at least two axes with [pieces]
    rows, or, through {!Nx.check}, if [breaks] decreases or its first equals its
    last. *)

(** {1:interpolants Interpolants}

    Each takes knots [x] of shape [[n]], [n ≥ 2], strictly increasing, and
    samples of shape [[n] @ value], and passes through every sample. Each raises
    [Invalid_argument] if [x] is not 1-D with at least two knots, if the samples
    do not have [n] rows, or, through {!Nx.check}, if [x] is not strictly
    increasing. *)

type 'b ends =
  [ `Natural | `Not_a_knot | `Clamped of (float, 'b) Nx.t * (float, 'b) Nx.t ]
(** The type for a cubic spline's end conditions, which change it near the ends:
    - [`Natural]: zero second derivative at both ends;
    - [`Not_a_knot]: a continuous third derivative at the second and the
      second-last knots, so the first two and last two pieces are each one
      cubic; through three knots, the parabola;
    - [`Clamped (s0, s1)]: the slopes [s0] and [s1], of shape [value], at the
      ends. *)

val linear : (float, 'b) Nx.t -> (float, 'b) Nx.t -> ((float, 'b) Nx.t, 'b) t
(** [linear x y] is the broken line through the samples: degree 1. *)

val cubic :
  'b ends -> (float, 'b) Nx.t -> (float, 'b) Nx.t -> ((float, 'b) Nx.t, 'b) t
(** [cubic ends x y] is the cubic spline through the samples, twice continuously
    differentiable: degree 3. Its system is solved in parallel by
    {!Nx.associative_scan}, stable because it is diagonally dominant.

    Raises [Invalid_argument] also if a clamped slope's shape is not [value]. *)

val steffen : (float, 'b) Nx.t -> (float, 'b) Nx.t -> ((float, 'b) Nx.t, 'b) t
(** [steffen x y] is Steffen's (1990) interpolant, once continuously
    differentiable: degree 3. It has no extremum between knots that the data
    lack, so the interpolant of monotone samples is monotone. Its end slopes are
    the end intervals' secants. *)

val hermite :
  (float, 'b) Nx.t ->
  values:(float, 'b) Nx.t ->
  slopes:(float, 'b) Nx.t ->
  ((float, 'b) Nx.t, 'b) t
(** [hermite x ~values ~slopes] is the piecewise cubic with the given values and
    slopes at the knots: degree 3, once continuously differentiable.

    Raises [Invalid_argument] also if [slopes] has another shape than [values].
*)

(** {1:fits Fits} *)

val chebyshev :
  'v Nx.Ptree.t ->
  degree:int ->
  pieces:int ->
  ((float, 'b) Nx.t -> 'v) ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  ('v, 'b) t
(** [chebyshev s ~degree ~pieces f a b] interpolates [f] on [pieces] equal
    pieces of [[a, b]] at each piece's [degree + 1] Chebyshev points of the
    second kind, ends included ([degree = 0] takes the midpoint). [f] receives
    points of shape [[pieces; degree + 1]] and returns each leaf of shape
    [[pieces; degree + 1] @ value]; [a] and [b] are scalars.

    {b Error.} For an [f] analytic in an ellipse around each piece, the error
    falls geometrically in [degree]; for one with [k] continuous derivatives, as
    [degree^(−k)]. {b Cost.} One call of [f] on all [pieces × (degree + 1)]
    points, and a constant matrix per leaf.

    Raises [Invalid_argument] if [degree < 0], [pieces < 1], if [a] or [b] is
    not a scalar, or if [f]'s leaves do not start with the points' shape. *)

val adapt :
  'v Nx.Ptree.t ->
  degree:int ->
  tol:Tol.t ->
  budget:int ->
  ((float, 'b) Nx.t -> 'v) ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t ->
  ('v, 'b) t Solution.t
(** [adapt s ~degree ~tol ~budget f a b] matches [f] on [[a, b]] to [tol] by
    series of [degree], bisecting the piece whose tail is largest. It solves one
    problem; {!Rune.val-vmap} gives each lane its own.

    {b Error.} A piece's [e] is the larger of its last two coefficients in
    magnitude, and [y] its largest coefficient: the series' tail against its
    size. [budget] bounds the pieces, so the answer always holds [budget]
    pieces, the unused ones empty at [b] after the domain. A lane whose worst
    piece is at level 62, or holds no float strictly inside, ends [Stalled]; a
    non-finite coefficient ends it [Not_finite]; [budget] pieces end it
    [Budget_spent]. The error is a series of degree 0 on the answer's breaks,
    each piece's tail. {b Cost.} [degree + 1] evaluations of [f] per piece, and
    the answer evaluates [f] again at every piece's points. {b Derivative.} The
    final partition's interpolant's: its breaks are
    [a + (b − a) index / 2^level], so it reaches the ends.

    Raises [Invalid_argument] if [degree < 2], [budget < 1], if [a] or [b] is
    not a scalar, or if [f]'s leaves do not start with the points' shape. *)

(** {1:eval Evaluation} *)

val eval : ('v, 'b) t -> (float, 'b) Nx.t -> 'v
(** [eval p x] is [p] at each point of [x], of shape [q]: each leaf of shape
    [q @ value].

    Raises [Invalid_argument] through {!Nx.check} if a point is infinite, or if
    a point that is not NaN lies outside the domain of a value that is not
    extended. *)

val eval_at : ('v, 'b) t -> (int64, Nx.int64_elt) Nx.t -> (float, 'b) Nx.t -> 'v
(** [eval_at p i u] is the series of piece [i] at the local coordinate [u] in
    [[−1, 1]], [i] and [u] of one shape [q]: each leaf of shape [q @ value]. [u]
    outside [[−1, 1]] extrapolates the piece's series.

    Raises [Invalid_argument] if [i] and [u] differ in shape, and through
    {!Nx.check} if an index is not a piece's. *)

(** {1:calculus Calculus}

    Each keeps the breaks and the extension: outside the domain the result
    extends as {!extend} says, from its own end pieces. Outside the domain,
    then, [eval (derivative p)] is not the derivative of [eval p]: the
    derivative of a held series holds its end slopes, and the integral of a held
    series holds its end values. *)

val derivative : ('v, 'b) t -> ('v, 'b) t
(** [derivative p] is the derivative of [p] in [x] on each piece, of degree one
    less (and [0] from degree [0]). At a break where [p]'s derivative jumps it
    takes the value of the piece that ends there. *)

val integral : ('v, 'b) t -> ('v, 'b) t
(** [integral p] is the antiderivative of [p] that is zero at the first break
    and continuous across breaks, of degree one more. *)

val extend : [ `Hold | `Polynomial ] -> ('v, 'b) t -> ('v, 'b) t
(** [extend e p] is [p] defined at every finite point: [`Hold] takes the value
    at the nearest end, and [`Polynomial] continues the end pieces' series. It
    replaces [p]'s extension. An infinite point still raises. *)

(** {1:access Access} *)

val breaks : ('v, 'b) t -> (float, 'b) Nx.t
(** [breaks p] is [p]'s breaks, of shape [[pieces + 1]]. *)

val coefficients : ('v, 'b) t -> 'v
(** [coefficients p] is [p]'s coefficients, each leaf of shape
    [[pieces; degree + 1] @ value]. *)

val ptree : 'v Nx.Ptree.t -> (float, 'b) Nx.dtype -> ('v, 'b) t Nx.Ptree.t
(** [ptree s dtype] is the structure of series with values of structure [s] over
    breaks of [dtype]: the breaks at [breaks], the coefficients under
    [coefficients], and the extension reported at [extension] as one of the
    cases ["bounded"], ["hold"] and ["polynomial"]. *)
