(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Paths.

    A path is a sequence of subpaths made of line and cubic Bézier segments.
    Build one segment by segment with {!move_to}, {!line_to}, {!curve_to} and
    {!close}, or from data with {!polyline} and {!polygon}, which store their
    points without per-point allocation. Paths are immutable and are consumed by
    {!Picture.fill}, {!Picture.stroke} and {!Picture.clip}.

    {!fold} visits a path segment by segment; {!flatten} does the same with
    curves replaced by chords, which is what renderers and {!bounds} use. *)

(** {1:types Types} *)

type t
(** The type for paths. *)

val empty : t
(** [empty] is the path with no subpaths. *)

val is_empty : t -> bool
(** [is_empty p] is [true] iff [p] has no subpaths. *)

(** {1:building Building}

    These functions take the path last so that they chain with [|>]. A subpath
    is {e open} until {!close}d; filling closes open subpaths implicitly,
    stroking does not. *)

val move_to : float -> float -> t -> t
(** [move_to x y p] is [p] with a new subpath started at [(x, y)]. *)

val line_to : float -> float -> t -> t
(** [line_to x y p] is [p] with a line from the current point to [(x, y)].
    Without a current point it is [move_to x y p]. *)

val curve_to : float -> float -> float -> float -> float -> float -> t -> t
(** [curve_to c1x c1y c2x c2y x y p] is [p] with a cubic Bézier from the current
    point to [(x, y)] with control points [(c1x, c1y)] and [(c2x, c2y)]. Without
    a current point it is [move_to x y p]. *)

val close : t -> t
(** [close p] is [p] with its current subpath closed by a line back to its
    start. It is [p] if the current subpath is empty or already closed. *)

val append : t -> t -> t
(** [append p q] is the subpaths of [p] followed by those of [q]. *)

val transform : Affine.t -> t -> t
(** [transform m p] is [p] with every point mapped through [m]. *)

(** {1:shapes Shapes} *)

val rect : float -> float -> float -> float -> t
(** [rect x y w h] is the closed rectangle with corner [(x, y)], width [w] and
    height [h]. *)

val circle : float -> float -> float -> t
(** [circle cx cy r] is the closed circle of radius [r] centred on [(cx, cy)],
    as four Bézier arcs. *)

val polyline : float array -> float array -> t
(** [polyline xs ys] is the open subpath through the points [(xs.(i), ys.(i))]
    in order. Fewer than two points give {!empty}.

    Raises [Invalid_argument] if [xs] and [ys] differ in length. *)

val polygon : float array -> float array -> t
(** [polygon xs ys] is [polyline xs ys] closed. *)

(** {1:traversing Traversing} *)

val fold :
  move:('a -> float -> float -> 'a) ->
  line:('a -> float -> float -> 'a) ->
  curve:('a -> float -> float -> float -> float -> float -> float -> 'a) ->
  close:('a -> 'a) ->
  'a ->
  t ->
  'a
(** [fold ~move ~line ~curve ~close acc p] folds over the segments of [p] in
    drawing order. [move acc x y] starts a subpath, [line acc x y] and
    [curve acc c1x c1y c2x c2y x y] extend it and [close acc] closes it, with
    the meaning of the building functions. Polylines and polygons are visited
    point by point. *)

val default_tolerance : float
(** [default_tolerance] is [0.1], the distance in path units by which {!flatten}
    lets a chord depart from its curve by default. *)

val flatten :
  ?tolerance:float ->
  Affine.t ->
  move:('a -> float -> float -> 'a) ->
  line:('a -> float -> float -> 'a) ->
  close:('a -> 'a) ->
  'a ->
  t ->
  'a
(** [flatten ~tolerance m ~move ~line ~close acc p] is {!fold} over [p] mapped
    through [m], with every curve replaced by line segments that stay within
    [tolerance] of it, measured after [m]. [tolerance] defaults to
    {!default_tolerance}. A segment with a non-finite endpoint ends the current
    subpath without closing it; the next finite point starts a new one. *)

val bounds : t -> Box.t option
(** [bounds p] is the box enclosing the points of [p] after flattening, or
    [None] if [p] has no finite points. *)

(** {1:formatting Formatting} *)

val pp : Format.formatter -> t -> unit
(** [pp fmt p] formats [p] as SVG path data, one segment per token: [M x y],
    [L x y], [C c1x c1y c2x c2y x y] and [Z]. *)
