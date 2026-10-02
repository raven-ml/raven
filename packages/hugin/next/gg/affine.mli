(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Affine maps of the plane.

    An affine map is six numbers and nothing constrains them, so the type is a
    plain record whose fields renderers write out as they are (SVG's
    [matrix(xx yx xy yy x0 y0)], PDF's [cm]). Compose maps with {!( * )}, which
    applies its right operand first: [translate x y * rotate a] rotates about
    the origin, then moves the result to [(x, y)]. Points, boxes and paths are
    mapped by [P2.transform], [Box2.transform] and [Path.transform]. *)

(** {1:types Types} *)

type t = {
  xx : float;
  yx : float;
  xy : float;
  yy : float;
  x0 : float;
  y0 : float;
}
(** The type for affine maps. The point [(x, y)] maps to
    [(xx *. x +. xy *. y +. x0, yx *. x +. yy *. y +. y0)]. *)

(** {1:constructors Constructors} *)

val id : t
(** [id] is the identity map. *)

val translate : float -> float -> t
(** [translate dx dy] moves points by [(dx, dy)]. *)

val scale : float -> float -> t
(** [scale sx sy] scales x by [sx] and y by [sy] about the origin. A negative
    factor mirrors. *)

val rotate : float -> t
(** [rotate a] turns points by [a] radians about the origin, from the positive x
    axis towards the positive y axis. *)

(** {1:composing Composing} *)

val ( * ) : t -> t -> t
(** [m * n] is the map applying [n] first, then [m]. *)

val invert : t -> t option
(** [invert m] is [Some m'], with [m * m'] the identity up to rounding, if the
    determinant [xx *. yy -. xy *. yx] of [m] is non-zero and every coefficient
    of [m'] is finite, and [None] otherwise. The determinant is computed on the
    coefficients [xx], [yx], [xy] and [yy] divided by the largest of their
    magnitudes, so it neither overflows nor underflows: [scale 1e-200 1e-200]
    inverts to [scale 1e200 1e200]. *)

(** {1:measures Measures} *)

val linear : t -> t
(** [linear m] is [m] without its translation: the map of vectors that [m]
    gives, [m] with [x0] and [y0] zero. *)

val stretch : t -> float
(** [stretch m] is the largest factor by which [m] scales a length, the largest
    singular value of [linear m]: a vector of length [l] maps to one of length
    at most [stretch m *. l]. It is computed so that it neither overflows nor
    underflows where its value is a finite float:
    [stretch (scale 1e-200 1e-200)] is [1e-200]. It is NaN if [linear m] is zero
    or a coefficient of it is not finite. *)

(** {1:comparing Comparing and formatting} *)

val equal : t -> t -> bool
(** [equal m n] is [true] iff the coefficients of [m] and [n] are pairwise
    equal. *)

val compare : t -> t -> int
(** [compare m n] orders maps lexicographically by [xx], [yx], [xy], [yy], [x0]
    and [y0]. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf m] formats the coefficients of [m] for debugging. *)
