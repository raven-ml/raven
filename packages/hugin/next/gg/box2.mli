(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Axis-aligned boxes.

    A box is a closed rectangle of the plane with sides parallel to the axes,
    possibly of zero width or height. There is no empty box: a function whose
    result may enclose nothing, such as {!Hugin_next_gg.Path.bounds}, returns an
    option. Boxes are what bounds report and what layout places panels in. *)

(** {1:types Types} *)

type t
(** The type for boxes. Invariant: the corners are finite, [minx <= maxx] and
    [miny <= maxy]. *)

(** {1:constructors Constructors} *)

val v : float -> float -> float -> float -> t
(** [v x y w h] is the box with corner [(x, y)], width [w] and height [h]: it
    spans \[[x];[x +. w]\] by \[[y];[y +. h]\].

    Raises [Invalid_argument] if [w] or [h] is negative, or if [x], [y],
    [x +. w] or [y +. h] is not finite. *)

val of_pts : P2.t -> P2.t -> t
(** [of_pts p q] is the smallest box containing [p] and [q], which are any two
    opposite corners.

    Raises [Invalid_argument] if a coordinate of [p] or [q] is not finite. *)

(** {1:accessors Accessors} *)

val minx : t -> float
(** [minx b] is the smallest x of [b]. *)

val miny : t -> float
(** [miny b] is the smallest y of [b], its top edge on screen. *)

val maxx : t -> float
(** [maxx b] is the largest x of [b]. *)

val maxy : t -> float
(** [maxy b] is the largest y of [b], its bottom edge on screen. *)

val w : t -> float
(** [w b] is [maxx b -. minx b], possibly [infinity]. *)

val h : t -> float
(** [h b] is [maxy b -. miny b], possibly [infinity]. *)

val mid : t -> P2.t
(** [mid b] is the centre of [b]. *)

(** {1:combining Combining and transforming} *)

val union : t -> t -> t
(** [union a b] is the smallest box containing [a] and [b]. *)

val inter : t -> t -> t option
(** [inter a b] is the box of the points in both [a] and [b], or [None] if they
    have none. Boxes that touch along an edge or at a corner meet in a box of
    zero width or height. *)

val grow : float -> t -> t
(** [grow d b] is [b] with each side moved outwards by [d], inwards if [d] is
    negative.

    Raises [Invalid_argument] if a corner of the result is not finite or if [d]
    shrinks a side past the opposite one. *)

val transform : Affine.t -> t -> t
(** [transform m b] is the smallest box containing the images of [b]'s four
    corners under [m]. Under a rotation it is larger than the rotated box.

    Raises [Invalid_argument] if an image is not finite. *)

(** {1:comparing Comparing and formatting} *)

val equal : t -> t -> bool
(** [equal a b] is [true] iff [a] and [b] have equal corners. *)

val compare : t -> t -> int
(** [compare a b] orders boxes lexicographically by [minx], [miny], [maxx] and
    [maxy]. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf b] formats [b]'s corners for debugging. *)
