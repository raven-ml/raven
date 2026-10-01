(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Symbols.

    A symbol is a marker shape: the circle, square or star a dot mark draws at
    each point.

    Symbols are sized by their ink: a symbol of {e size} [a] lays as much ink as
    the circle of area [a] painted the same way. Filled, the region it encloses
    has area [a]; stroked, its outline is as long as that circle's
    circumference, [2 √(π a)], so that at one stroke width it lays as much ink
    as the circle. {!plus}, {!times} and {!asterisk} enclose nothing and are
    sized by length however they are painted. A symbol keeps its shape at every
    size: the symbol of size [a] is the symbol of size [1.] scaled by [√a], so a
    figure can draw one outline and stamp it at many sizes.

    Every symbol is symmetric under a rotation about the origin, so the
    centroids of the region it encloses and of its outline both lie at the
    origin: a triangle sits on its data point rather than above it. The plane is
    the one of {{!Hugin_next_gg.section-conventions}[hugin.next.gg]}, y pointing
    down, so a symbol that points up points towards negative y.

    Some symbols are designed to be filled ({!filled}), the others to be stroked
    ({!stroked}).

    Reference: Mike Bostock.
    {{:https://d3js.org/d3-shape/symbol}d3-shape symbols}, from which the
    shapes, the two sets and the sizing by ink are taken. *)

open Hugin_next_gg

(** {1:symbols Symbols} *)

type t
(** The type for symbols. *)

val circle : t
(** [circle] is a circle. *)

val square : t
(** [square] is a square with sides parallel to the axes. *)

val diamond : t
(** [diamond] is two equilateral triangles joined by a side parallel to the x
    axis: a rhombus standing on a corner, taller than wide. *)

val triangle : t
(** [triangle] is an equilateral triangle pointing up. *)

val cross : t
(** [cross] is a Greek cross: five equal squares, one at the centre and one on
    each of its sides. *)

val star : t
(** [star] is a regular five-pointed star pointing up, its inner corners on the
    lines that join its points. *)

val wye : t
(** [wye] is a Y: an equilateral triangle with a square on each of its sides,
    one of them below it. *)

val plus : t
(** [plus] is two strokes of equal length crossing at their middles, one
    parallel to each axis. *)

val times : t
(** [times] is {!plus} turned by an eighth of a turn. *)

val asterisk : t
(** [asterisk] is three strokes of equal length crossing at their middles, a
    sixth of a turn apart, one parallel to the y axis. *)

(** {1:sets Sets}

    Ordered sets of symbols for categories, one for marks that fill their
    symbols and one for marks that stroke them. *)

val filled : t list
(** [filled] is the symbols designed to be filled: {!circle}, {!cross},
    {!diamond}, {!square}, {!star}, {!triangle} and {!wye}. *)

val stroked : t list
(** [stroked] is the symbols designed to be stroked: {!circle}, {!plus},
    {!times}, {!triangle}, {!asterisk}, {!square} and {!diamond}. *)

(** {1:drawing Drawing} *)

type paint = [ `Fill | `Stroke ]
(** The type for the ways a symbol is painted, which decide how it is sized:
    [`Fill] by the area it encloses, [`Stroke] by the length of its outline. *)

val path : paint -> float -> t -> Path.t
(** [path paint a s] is the outline of [s] at size [a] painted with [paint], its
    centre at the origin. It is [path paint 1. s] scaled by [√a], up to
    rounding. Closed outlines are closed subpaths that turn clockwise on screen,
    as {!Path.rect} and {!Path.circle} do, and the circle is the one
    {!Path.circle} draws, whose area and length are those of a true circle up to
    the error of its arcs. {!plus}, {!times} and {!asterisk} are open subpaths
    of one segment per stroke, so filled they paint nothing. [path paint 0. s]
    has every point at the origin.

    Raises [Invalid_argument] if [a] is negative or not finite. *)

(** {1:comparing Comparing and formatting} *)

val equal : t -> t -> bool
(** [equal s s'] is [true] iff [s] and [s'] are the same symbol. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf s] formats the name of [s], such as [triangle]. *)
