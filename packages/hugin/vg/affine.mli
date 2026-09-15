(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** 2D affine transforms.

    Transforms are plain data. Compose them with {!( * )}, which applies its
    right operand first, so [translate x y * rotate a] rotates about the origin
    and then moves the result to [(x, y)]. *)

(** {1:types Types} *)

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

(** {1:constructors Constructors} *)

val id : t
(** [id] is the identity transform. *)

val translate : float -> float -> t
(** [translate dx dy] moves points by [(dx, dy)]. *)

val scale : float -> float -> t
(** [scale sx sy] scales x by [sx] and y by [sy] about the origin. A negative
    factor mirrors. *)

val rotate : float -> t
(** [rotate a] rotates by [a] radians about the origin. Since y points down,
    positive angles turn clockwise on screen. *)

(** {1:composing Composing} *)

val ( * ) : t -> t -> t
(** [a * b] is the transform applying [b] first, then [a]. *)

val invert : t -> t
(** [invert m] is the inverse of [m], so that [m * invert m] is {!id}.

    Raises [Invalid_argument] if [m] is singular. *)

(** {1:applying Applying} *)

val apply : t -> float -> float -> float * float
(** [apply m x y] is the image of [(x, y)] under [m]. *)
