(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Scalar fields sampled on grids, and their isolines and isobands.

    A field is a real function of the plane known by its samples on a
    rectangular grid, whose rows and columns may be unevenly spaced. The sample
    of row [i] and column [j] lies at [(xs.(j), ys.(i))]: rows follow y and
    columns x, so with the default coordinates a field lies in the plane like an
    image on screen, row [0] at the top.

    {!isoline} gives the curves where a field crosses a level, as a path, and
    {!isoband} the polygon where it lies between two levels. Both are marching
    squares on the field's domain, defined next.

    {1:semantics Domain and regions}

    A sample is {e missing} if it is NaN or infinite. A {e cell} is the
    rectangle between two adjacent rows and two adjacent columns, with a sample
    at each corner. A cell with no missing sample is {e whole}. A cell with
    exactly one missing sample keeps only the triangle of its other three, on
    which the field is read by linear interpolation. Other cells are left out.
    The {e domain} of the field is the union of its whole cells and triangles,
    its {e pieces}. Isolines and isobands lie in the grid's box and, up to the
    placement of crossings on diagonals, in the domain: missing samples cut
    holes in it, and rings close along its boundary, the border of the grid
    included.

    For a level [l], the {e region} [R l] approximates the part of the domain
    where the field is at least [l]. It is the union of the polygons made in
    each piece as follows:
    - A corner of a piece is {e in} if its sample is at least [l], and {e out}
      otherwise.
    - A side of a piece with a corner in and a corner out has a {e crossing},
      the point where linear interpolation between the side's two samples
      reaches [l]. It is the in corner if that corner's sample is [l]. On a
      triangle's diagonal, where it cannot lie exactly, the crossing is moved
      along the diagonal so that the crossings of two levels coincide or lie
      apart, and a crossing is an end of the diagonal or off the lines of the
      other sides. It moves by at most the fraction [256 ε (c + d) / d] of the
      diagonal, where [ε] is [epsilon_float], [d] is the cell's extent along an
      axis, [c] the larger magnitude of its two coordinates on that axis, and
      the axis the one that gives the larger fraction. This is negligible
      unless the coordinates are large against the grid's spacing, where a
      crossing can move to an end of the diagonal.
    - The polygon walks around the piece's boundary through its in corners and
      its crossings, in order. In a whole cell whose in corners are two opposite
      ones, this walk is a hexagon around the cell's centre if the mean of the
      cell's four samples, up to rounding, is at least [l]. Otherwise the
      polygon is the two triangles cut off at the in corners.
    - A polygon that encloses no area, such as the point or segment left where
      samples equal [l], is dropped.

    So no part of [R l] lacks area: samples equal to [l] with only lower samples
    around them, a peak or a ridge exactly at the level, are not in [R l]. On a
    triangle, the polygon is the part where the linear interpolation is at least
    [l], up to the placement of crossings on its diagonal, if that part has an
    area. The region of an infinite level is the domain for [neg_infinity] and
    empty for [infinity]. *)

open Hugin_next_gg

(** {1:types Types} *)

type t
(** The type for fields. Invariant: coordinates are finite, strictly increasing
    or strictly decreasing along each axis, and span a finite distance along
    each axis. *)

(** {1:constructors Constructors} *)

val v : ?xs:float array -> ?ys:float array -> ('a, 'b) Nx.t -> t
(** [v ~xs ~ys z] is the field whose sample at row [i] and column [j] is the
    element [(i, j)] of the two-dimensional tensor [z], at [(xs.(j), ys.(i))],
    where:
    - [xs] defaults to the column indices [[|0.; 1.; …|]];
    - [ys] defaults to the row indices.

    Elements are converted to floats, so an integer beyond 2{^ 53} is rounded.
    The elements of [z] are read to the host once and the field keeps its own
    copy of them and of the coordinates. A field with fewer than two rows or two
    columns has no cell and an empty domain.

    Raises [Invalid_argument] if [z] is not two-dimensional or has a complex or
    boolean dtype, if the length of [xs] is not the number of columns of [z] or
    that of [ys] the number of rows, if [xs] or [ys] holds a coordinate that is
    not finite, if either array is neither strictly increasing nor strictly
    decreasing, or if the difference between its first and last coordinates is
    not finite. *)

(** {1:contours Isolines and isobands}

    Both functions run in time linear in the number of samples. The order of
    their curves or rings, the first point of each, and where curves that touch
    are split depend only on their arguments. *)

val isoline : float -> t -> Path.t
(** [isoline l f] is the isoline of [f] at level [l]: the boundary of [R l] less
    its parts on the boundary of the domain, as curves. A curve that ends on the
    boundary of the domain is an open subpath of the result, and one that does
    not is a closed subpath.

    Curves run with [R l] on their right on screen, so closed curves around
    higher values are positively oriented. An open curve has at least two points
    and a closed one at least three. Consecutive points are distinct, and so are
    the last and first points of a closed curve. Curves neither cross nor
    overlap, and touch only at points of the grid, as where samples equal [l].
    The isoline at an infinite level is {!Hugin_next_gg.Path.empty}.

    Raises [Invalid_argument] if [l] is NaN. *)

val isoband : lo:float -> hi:float -> t -> Pgon2.t
(** [isoband ~lo ~hi f] is the polygon whose surface is [R lo] without [R hi]:
    where [f] is at least [lo] and below [hi]. An infinite bound bounds nothing,
    so [isoband ~lo:neg_infinity ~hi:infinity f] is the domain of [f].

    Its rings are the boundary of that surface, outer boundaries positively
    oriented and holes negatively. They wind [0] or [1] times around every point
    off them, so {!Pgon2.area} is the band's area. A ring has at least three
    points. Consecutive points are distinct, and so are the last and first
    points. Rings neither cross nor overlap, and touch only at points. Every
    segment of a ring lies on the boundary of the domain, on a segment of
    [isoline lo f] in the same direction, or on a segment of [isoline hi f] in
    the opposite direction.

    For levels [l0 < l1 < … < ln], the bands between consecutive levels do not
    overlap. With [l0 = neg_infinity] and [ln = infinity] they tile the domain,
    and {!Pgon2.mem} finds a point in at most one of them.

    These laws hold exactly except within the crossings' displacement of the
    diagonal of a triangle. There the bands bend at their crossings while the
    domain's boundary runs straight, so a band may leave the domain or leave
    uncovered a sliver of it that narrow; and where the crossings of two levels
    on a side of the triangle lie within rounding error of each other, rings
    may cross and bands overlap. The laws also assume that products of
    coordinate differences neither overflow nor underflow, as the
    {{!Hugin_next_gg_kit.section-conventions}conventions} state for
    {!Pgon2.mem}.

    Raises [Invalid_argument] if [lo] or [hi] is NaN or [lo > hi]. *)
