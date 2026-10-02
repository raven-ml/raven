(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Curves.

    A curve says how a line passes through a sequence of points: straight from
    one to the next, in steps, or smoothly. {!path} draws a curve through
    points, as a line mark does. Points are given as arrays of x and y
    coordinates in the plane the path is drawn in, which for a line mark is that
    of the panel's normalised positions. Paths are made of lines and cubic
    Béziers, so every renderer draws a curve the same way.

    Drawing a curve then mapping the path by an affine map is, up to rounding,
    the same as mapping the points then drawing the curve: for {!linear},
    {!natural} and {!basis} under every affine map; for the steps and the
    monotone curves under maps that scale and translate each axis on its own;
    and for {!catmull_rom}, which depends on distances, under similarities.

    Curves other than {!linear} and the steps compute their cubics from
    differences and ratios of coordinates, so coordinates or differences of
    magnitude beyond about [1e300] or below about [1e-300] can give a cubic
    non-finite control points, which a path draws as a
    {{!Path.section-gaps}gap}.

    {1:runs Missing points}

    A point with a non-finite coordinate is {e missing}. Missing points cut the
    sequence into {e runs}, the maximal sequences of consecutive points that are
    not missing. A curve draws each run on its own, from the run's points alone,
    so a line breaks where a value is missing and smoothing never reaches across
    the break. A run of one point draws nothing. *)

open Hugin_next_gg

(** {1:curves Curves}

    Every curve but the steps draws a run of two points as the straight segment
    between them. *)

type t
(** The type for curves. *)

val linear : t
(** [linear] joins consecutive points with straight segments. *)

val step_after : t
(** [step_after] goes from each point parallel to the x axis as far as the next
    point's x, then parallel to the y axis to the next point: a value holds
    until the next one. *)

val step_before : t
(** [step_before] goes from each point parallel to the y axis as far as the next
    point's y, then parallel to the x axis to the next point: a value holds
    since the previous one. *)

val step_mid : t
(** [step_mid] goes from each point parallel to the x axis halfway to the next
    point's x, then parallel to the y axis to the next point's y, then parallel
    to the x axis to the next point: steps fall midway between points. *)

val monotone_x : t
(** [monotone_x] is Steffen's interpolant of y as a function of x. It passes
    through every point, rises or falls between two points only as they do, so
    that it never overshoots the data, and its slope is continuous. Between two
    consecutive points it is the cubic with the slopes below at its ends, whose
    Bézier control points lie a third and two thirds of the way along x. With
    [h] and [s] the difference of x and the slope of one segment, and [h'] and
    [s'] those of the next:
    - at the point between them, the slope is
      [(sign s + sign s') × min (|s|, |s'|, |p| / 2)], where
      [p = (s h' + s' h) / (h + h')] and [sign 0. = 0.];
    - at the first point of a piece, with [h] and [s] those of its first segment
      and [h'] and [s'] those of its second, the slope is
      [p = s (1 + h / (h + h')) - s' h / (h + h')], or [0.] if [p s <= 0.], or
      [2 s] if [|p| > 2 |s|]; at the last point, likewise from its last segment
      and the one before. A piece of two points is a straight segment.

    A {e piece} is a maximal sequence of consecutive points of a run whose x
    strictly increase or strictly decrease. A run is drawn piece by piece,
    consecutive pieces sharing their end point, and two consecutive points with
    equal x are joined by a straight segment, so a run whose x turn back or
    repeat is drawn too.

    Reference: M. Steffen.
    {e A simple method for monotonic interpolation in one dimension}. Astronomy
    and Astrophysics 239, 1990. *)

val monotone_y : t
(** [monotone_y] is {!monotone_x} with the roles of x and y exchanged: the
    interpolant of x as a function of y. *)

val natural : t
(** [natural] is the natural cubic spline through the points, taken
    parametrically: x and y are each the cubic spline of the point's index that
    passes through every point, has continuous first and second derivatives, and
    has a zero second derivative at both ends. Since the parameter steps evenly
    from point to point, unevenly spaced points can make the curve turn back in
    x; {!monotone_x} draws a function of x. *)

val catmull_rom : t
(** [catmull_rom] is the centripetal Catmull–Rom spline through the points: it
    passes through every point, and each segment is shaped by the points before
    and after it, the parameter advancing by the square root of the distance
    between consecutive points, which keeps segments free of cusps and
    self-intersections. The first and last segments take their end point as
    their missing neighbour, which puts the first control point of the first
    segment, and the last of the last, on the end point. Consecutive equal
    points count as one, and a run whose points are all equal is drawn as
    {!linear} draws it.

    Reference: Cem Yuksel, Scott Schaefer and John Keyser.
    {e Parameterization and applications of Catmull–Rom curves}. Computer-Aided
    Design 43(7), 2011. *)

val basis : t
(** [basis] is the uniform cubic B-spline whose control points are the points,
    the first and the last counted three times: it starts at the first point and
    ends at the last, has continuous curvature, and in between passes near the
    points rather than through them. *)

(** {1:drawing Drawing} *)

val path : t -> float array -> float array -> Path.t
(** [path c xs ys] is curve [c] through the points [(xs.(i), ys.(i))]: one open
    subpath per run of at least two points ({!section-runs}), in order, or
    {!Path.empty} if there is none.

    Raises [Invalid_argument] if [xs] and [ys] differ in length. *)

(** {1:comparing Comparing and formatting} *)

val equal : t -> t -> bool
(** [equal c c'] is [true] iff [c] and [c'] are the same curve. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf c] formats the name of [c], such as [monotone_x]. *)
