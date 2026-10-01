(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Linear rings.

    A ring is a closed polygonal curve: points in order, each joined to the next
    by a straight segment and the last to the first. A ring may cross or touch
    itself. A ring of fewer than three points encloses nothing: its winding
    number is [0] around every point off it. Orientation, winding numbers and
    signed areas follow the
    {{!Hugin_next_gg_kit.section-conventions}conventions}. Rings store their
    coordinates in two float arrays, without a block per point. *)

open Hugin_next_gg

(** {1:types Types} *)

type t
(** The type for rings. Invariant: coordinates are finite. *)

(** {1:constructors Constructors} *)

val v : float array -> float array -> t
(** [v xs ys] is the ring through the points [(xs.(i), ys.(i))] in order. The
    arrays are copied.

    Raises [Invalid_argument] if [xs] and [ys] differ in length or a coordinate
    is not finite. *)

(** {1:accessors Accessors}

    Functions that take a point index raise [Invalid_argument] if it is not in
    \[[0];[length r - 1]\]. *)

val length : t -> int
(** [length r] is the number of points of [r]. *)

val x : t -> int -> float
(** [x r i] is the x coordinate of point [i] of [r]. *)

val y : t -> int -> float
(** [y r i] is the y coordinate of point [i] of [r]. *)

(** {1:measures Measures} *)

val area : t -> float
(** [area r] is the signed area of [r], computed with rounding: positive if [r]
    is positively oriented and negative if negatively, unless the area is within
    rounding error of zero, and [0.] if [r] has fewer than three points. It
    overflows to an infinity or NaN if [r] spans more than about 10{^ 154}. *)

val mem : P2.t -> t -> bool
(** [mem pt r] is [true] iff the winding number of [r] around [pt] is not [0], a
    point on [r] being decided as the
    {{!Hugin_next_gg_kit.section-conventions}conventions} state. It is [false]
    if a coordinate of [pt] is not finite. *)

val bounds : t -> Box2.t option
(** [bounds r] is the smallest box containing the points of [r], or [None] if
    [r] has no point. *)

(** {1:transforming Transforming and converting} *)

val reverse : t -> t
(** [reverse r] is [r] with its points in reverse order: point [i] of
    [reverse r] is point [length r - 1 - i] of [r]. Its orientation, signed area
    and winding numbers are those of [r] negated. *)

val to_path : t -> Path.t
(** [to_path r] is the closed subpath through the points of [r] in order,
    {!Hugin_next_gg.Path.polygon} of its coordinates. It is
    {!Hugin_next_gg.Path.empty} if [r] has fewer than two points. *)

(** {1:comparing Comparing and formatting} *)

val equal : t -> t -> bool
(** [equal r r'] is [true] iff [r] and [r'] have equal points in the same order
    from the same first point. A ring and its rotation, which draw the same
    curve, are not equal. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf r] formats the points of [r] as SVG path data, [M x y] for the first
    point, [L x y] for each other and a final [Z], with numbers printed with
    [%g]. A ring with no point formats as nothing. *)
