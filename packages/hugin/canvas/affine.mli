(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** 2D affine transforms. *)

type t = {
  xx : float;
  yx : float;
  xy : float;
  yy : float;
  x0 : float;
  y0 : float;
}
(** The type for affine transforms. A point [(x, y)] maps to
    [(xx *. x +. xy *. y +. x0, yx *. x +. yy *. y +. y0)]. *)

val id : t
val translate : float -> float -> t
val scale : float -> float -> t

val rotate : float -> t
(** [rotate a] rotates by [a] radians about the origin. Since y points down on
    the canvas, positive angles turn clockwise on screen. *)

val ( * ) : t -> t -> t
(** [a * b] is the transform applying [b] first, then [a]. *)

val apply : t -> float -> float -> float * float
(** [apply m x y] is the image of [(x, y)] under [m]. *)

val invert : t -> t
(** [invert m] is the inverse of [m].

    Raises [Invalid_argument] if [m] is singular. *)

val is_translation : t -> bool
(** [is_translation m] is [true] iff [m] only translates. *)
