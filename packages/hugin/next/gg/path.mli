(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Paths.

    A path is a sequence of subpaths, each a start point followed by line and
    cubic Bézier segments, optionally closed. Build one segment by segment from
    {!empty}, or take a shape: {!rect}, {!circle}, and {!polyline} and
    {!polygon}, which store their points in two float arrays without a block per
    point. Quadratic Béziers and circular arcs are converted to cubic Béziers as
    they are added, exactly for quadratics and within a stated error for arcs,
    so every renderer draws the same curves.

    {!fold} visits what a path draws, segment by segment, with its
    {{!section-gaps}gaps} applied; {!flatten} does the same with curves replaced
    by chords. Renderers draw paths through them. *)

(** {1:types Types} *)

type t
(** The type for paths. *)

val empty : t
(** [empty] is the path with no subpaths. *)

val is_empty : t -> bool
(** [is_empty p] is [equal p empty]. A path of gaps only is not empty;
    [bounds p = None] tells whether [p] draws anything. *)

(** {1:building Building}

    A subpath is {e open} until {!close}d. Filling closes open subpaths
    implicitly; stroking does not. A line or curve added to a path without a
    current point, that is to {!empty} or right after {!close}, starts a subpath
    at its end point instead; an {!arc}, which carries its own start, starts a
    subpath there and is drawn. *)

val move_to : P2.t -> t -> t
(** [move_to pt p] is [p] with a new subpath started at [pt]. *)

val line_to : P2.t -> t -> t
(** [line_to pt p] is [p] with a line from the current point to [pt]. *)

val quad_to : P2.t -> P2.t -> t -> t
(** [quad_to c pt p] is [p] with a quadratic Bézier from the current point to
    [pt] with control point [c], stored as the cubic of the same curve. *)

val cubic_to : P2.t -> P2.t -> P2.t -> t -> t
(** [cubic_to c c' pt p] is [p] with a cubic Bézier from the current point to
    [pt] with control points [c] and [c']. *)

val arc : P2.t -> float -> start:float -> sweep:float -> t -> t
(** [arc c r ~start ~sweep p] is [p] with the circular arc of centre [c] and
    radius [r] from angle [start] through [sweep] radians, positive sweeps
    turning from the positive x axis towards the positive y axis. If [p] has a
    current point, a line joins it to the arc's start; otherwise the arc starts
    a subpath. The arc is stored as cubic Béziers, one per quarter turn or less,
    which depart from the circle by at most 0.03% of [r]. A [sweep] beyond a
    full turn retraces the circle.

    Raises [Invalid_argument] if [r] is negative, if [r], [start] or [sweep] is
    not finite, or if [sweep] exceeds a thousand turns in magnitude. *)

val close : t -> t
(** [close p] is [p] with its last subpath closed by a line back to its start,
    after which [p] has no current point. It is [p] if [p] is empty or its last
    subpath is closed or has no segment. *)

val append : t -> t -> t
(** [append q p] is the subpaths of [p] followed by those of [q]. The order is
    the opposite of [List.append], so that [p |> append q] appends [q]. *)

val transform : Affine.t -> t -> t
(** [transform m p] is [p] with every point mapped through [m]. Since curves are
    cubic Béziers, the image of a curve is exact. *)

(** {1:shapes Shapes} *)

val rect : Box2.t -> t
(** [rect b] is the closed subpath around [b] from its corner [(minx b, miny b)]
    along the x axis first, clockwise on screen. *)

val circle : P2.t -> float -> t
(** [circle c r] is the closed circle of centre [c] and radius [r], starting at
    angle [0.] and turning clockwise on screen, as {!arc} draws a full turn.
    Ellipses are the {!transform}s of circles.

    Raises [Invalid_argument] if [r] is negative or not finite. *)

val polyline : float array -> float array -> t
(** [polyline xs ys] is the open subpath through the points [(xs.(i), ys.(i))]
    in order, or {!empty} for fewer than two points. The arrays are copied.

    Raises [Invalid_argument] if [xs] and [ys] differ in length. *)

val polygon : float array -> float array -> t
(** [polygon xs ys] is [polyline xs ys] closed. *)

(** {1:traversing Traversing}

    {2:gaps Gaps}

    The points of a segment are the points it was built with: its end point and,
    for a curve, its control points. The point it starts from is not one of
    them. A segment with a non-finite point is a {e gap}, and so is a subpath
    start at a non-finite point. The subpath a gap would extend ends there,
    unclosed, and the next segment whose points are finite starts a subpath at
    its end point. A {!close} closes the subpath current at that moment, if that
    subpath has a segment. After a gap, that is the subpath the gap's successor
    started, and the {!close} does nothing if no such subpath has started. *)

val fold :
  move:('a -> float -> float -> 'a) ->
  line:('a -> float -> float -> 'a) ->
  cubic:('a -> float -> float -> float -> float -> float -> float -> 'a) ->
  close:('a -> 'a) ->
  'a ->
  t ->
  'a
(** [fold ~move ~line ~cubic ~close acc p] folds over what [p] draws, in drawing
    order: [move acc x y] starts a subpath, [line acc x y] and
    [cubic acc c1x c1y c2x c2y x y] extend it and [close acc] closes it.
    {{!section-gaps}Gaps} are applied, so every number given is finite: a
    segment after a gap is given as [move] to its end point. Polylines and
    polygons are visited point by point, quadratics and arcs as the cubics that
    store them. *)

val flatten :
  ?tolerance:float ->
  Affine.t ->
  move:('a -> float -> float -> 'a) ->
  line:('a -> float -> float -> 'a) ->
  close:('a -> 'a) ->
  'a ->
  t ->
  'a
(** [flatten ~tolerance m ~move ~line ~close acc p] is {!fold} over
    [transform m p], with every cubic replaced by line segments that stay within
    [tolerance] of it. [tolerance] defaults to [0.1]. Gaps are those of
    [transform m p], so every number given is finite whatever [m]. A cubic is
    cut into at most 65536 chords, which keeps it within [tolerance] if its
    control points lie within [1e9 *. tolerance] of each other.

    Raises [Invalid_argument] if [tolerance] is not positive. *)

val bounds : t -> Box2.t option
(** [bounds p] is the smallest box containing what {!fold} visits in [p], each
    curve bounded exactly rather than by its control points, or [None] if
    {!fold} visits nothing. *)

(** {1:comparing Comparing and formatting} *)

val equal : t -> t -> bool
(** [equal p q] is [true] iff [p] and [q] were built from the same segments with
    equal points, non-finite ones included, a polyline counting as its
    point-by-point segments. A polyline thus equals the same points built with
    {!move_to} and {!line_to}, and equal paths fold identically. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf p] formats the segments [p] was built from as SVG path data, one
    segment per token: [M x y], [L x y], [C c1x c1y c2x c2y x y] and [Z],
    numbers printed with [%g], non-finite ones included. *)
