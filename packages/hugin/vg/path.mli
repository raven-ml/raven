(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Paths.

    A path is a sequence of subpaths made of line and cubic Bézier segments.
    Paths are immutable; the bulk constructors {!polyline} and {!polygon} store
    their points without per-point allocation. *)

type t
(** The type for paths. *)

val empty : t
val is_empty : t -> bool

(** {1:building Building} *)

val move_to : float -> float -> t -> t
(** [move_to x y p] starts a new subpath at [(x, y)]. *)

val line_to : float -> float -> t -> t
(** [line_to x y p] appends a line to [(x, y)]. Without a current point it acts
    as {!move_to}. *)

val curve_to : float -> float -> float -> float -> float -> float -> t -> t
(** [curve_to c1x c1y c2x c2y x y p] appends a cubic Bézier to [(x, y)] with
    control points [(c1x, c1y)] and [(c2x, c2y)]. Without a current point it
    acts as {!move_to}. *)

val close : t -> t
(** [close p] closes the current subpath. Closing an empty or already closed
    subpath does nothing. *)

val append : t -> t -> t
(** [append p q] is the subpaths of [p] followed by those of [q]. *)

val transform : Affine.t -> t -> t

(** {1:shapes Shapes} *)

val rect : float -> float -> float -> float -> t
(** [rect x y w h] is the closed rectangle with corner [(x, y)], width [w] and
    height [h]. *)

val circle : float -> float -> float -> t
(** [circle cx cy r] is the closed circle of radius [r] centred on [(cx, cy)].
*)

val polyline : float array -> float array -> t
(** [polyline xs ys] is the open subpath through the points [(xs.(i), ys.(i))].
    Fewer than two points give {!empty}.

    Raises [Invalid_argument] if the arrays differ in length. *)

val polygon : float array -> float array -> t
(** [polygon xs ys] is [polyline xs ys] closed. *)

(** {1:traversal Traversal} *)

val fold :
  move:('a -> float -> float -> 'a) ->
  line:('a -> float -> float -> 'a) ->
  curve:('a -> float -> float -> float -> float -> float -> float -> 'a) ->
  close:('a -> 'a) ->
  'a ->
  t ->
  'a
(** [fold ~move ~line ~curve ~close acc p] folds over the segments of [p] in
    drawing order. [move] starts a subpath, [line] and [curve] extend it and
    [close] closes it, with the same conventions as the building functions.
    Polylines and polygons are visited point by point. *)

val default_tolerance : float
(** [default_tolerance] is a tenth of a unit, the flattening tolerance
    {!flatten} uses by default. *)

val flatten :
  ?tolerance:float ->
  Affine.t ->
  move:('a -> float -> float -> 'a) ->
  line:('a -> float -> float -> 'a) ->
  close:('a -> 'a) ->
  'a ->
  t ->
  'a
(** [flatten ~tolerance m ~move ~line ~close acc p] is like {!fold} over [p]
    mapped through [m], with every curve replaced by line segments staying
    within [tolerance] of it. [tolerance] defaults to {!default_tolerance}. A
    segment with a non-finite endpoint ends its subpath. *)

val bounds : t -> Box.t option
(** [bounds p] is the box enclosing [p], or [None] if [p] has no points. *)

(** {1:printing Printing} *)

val pp : Format.formatter -> t -> unit
(** [pp fmt p] prints [p] as SVG path data, one segment per token: [M x y],
    [L x y], [C c1x c1y c2x c2y x y] and [Z]. *)
