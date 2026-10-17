(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The exact area of a convex quadrilateral cell inside a shape centred on
    the origin, with fixed shapes and no data-dependent branch.

    Points are pairs [(x, y)] of tensors. A cell is its four corners
    [(xs.(k), ys.(k))], [k < 4], each coordinate of the batch shape, and must
    be convex for its weight to be its area's fraction inside the shape. The
    weight is the area inside over the cell's area, both signed by the
    cell's orientation, so either orientation gives the same weight. Each
    term is taken about the cell's first corner, so its size is the cell's.

    A disc clips each cell edge to the circle and adds, by Green's theorem,
    the segment inside and an arc of the circle for each part outside. A
    polygon is the signed sum of the triangles fanned from the cell's first
    corner, each clipped to the cell (Liang and Barsky). *)

type 'e t = {
  weight : (float, 'e) Nx.t;  (** The weight, in \[0, 1\]. *)
  area : (float, 'e) Nx.t;
      (** The cell's signed shoelace area, positive for a counter-clockwise
          cell. *)
  outside : Nx.bool_t;  (** Where the weight is exactly 0 by the predicate. *)
  full : Nx.bool_t;  (** Where the weight is exactly 1 by the predicate. *)
}
(** The type for a cell's overlap with a shape. *)

val disc :
  (float, 'a) Nx.t -> (float, 'a) Nx.t array -> (float, 'a) Nx.t array -> 'a t
(** [disc r xs ys] weighs the cells [xs], [ys] against the disc of radius [r]
    about the origin, [r] broadcast against the corners. A cell with every
    corner inside weighs exactly 1. A cell whose nearest point is at least
    [r] from the centre and that does not hold the centre weighs exactly 0,
    as does every cell for a zero radius. *)

val annulus :
  (float, 'a) Nx.t ->
  (float, 'a) Nx.t ->
  (float, 'a) Nx.t array ->
  (float, 'a) Nx.t array ->
  'a t
(** [annulus inner outer xs ys] is [disc outer xs ys]'s weight minus
    [disc inner xs ys]'s. A cell is exactly outside where it is outside the
    outer disc or wholly inside the inner one. *)

val ellipse :
  (float, 'a) Nx.t ->
  (float, 'a) Nx.t ->
  (float, 'a) Nx.t ->
  (float, 'a) Nx.t array ->
  (float, 'a) Nx.t array ->
  'a t
(** [ellipse a b angle xs ys] weighs the cells against the ellipse of
    semi-axis [a] along the direction at [angle] radians from +y toward +x
    and semi-axis [b] across it: {!disc} of radius [min a b] on the corners
    mapped by the affine map that takes the ellipse to that disc. The
    weights' rounding grows as the square of the axis ratio, to 1e-14 of the
    ellipse's area at a ratio of 100. A zero semi-axis weighs 0. *)

val polygon :
  ((float, 'a) Nx.t * (float, 'a) Nx.t) array ->
  (float, 'a) Nx.t array ->
  (float, 'a) Nx.t array ->
  'a t
(** [polygon vs xs ys] weighs the cells against the polygon of vertices
    [vs], in either orientation, each coordinate broadcast against the
    corners. A cell weighs exactly 1 when its corners are all inside and no
    edge crosses it, and exactly 0 when no corner is inside, no edge crosses
    it and no vertex is in it. *)

val crosses :
  (float, 'a) Nx.t * (float, 'a) Nx.t ->
  (float, 'a) Nx.t * (float, 'a) Nx.t ->
  (float, 'a) Nx.t * (float, 'a) Nx.t ->
  (float, 'a) Nx.t * (float, 'a) Nx.t ->
  Nx.bool_t
(** [crosses a b c d] is where the segments [a b] and [c d] cross at a point
    inside both. *)

val cross :
  ('a, 'b) Nx.t * ('a, 'b) Nx.t ->
  ('a, 'b) Nx.t * ('a, 'b) Nx.t ->
  ('a, 'b) Nx.t
(** [cross a b] is the planar cross product [a × b]. *)

val sub :
  ('a, 'b) Nx.t * ('c, 'd) Nx.t ->
  ('a, 'b) Nx.t * ('c, 'd) Nx.t ->
  ('a, 'b) Nx.t * ('c, 'd) Nx.t
(** [sub a b] is [a - b]. *)
