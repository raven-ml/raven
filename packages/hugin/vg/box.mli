(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Axis-aligned boxes.

    Boxes are the extents {!Path.bounds}, {!Font.bounds} and {!Picture.bounds}
    report. Combine them with {!union} and map them through transforms with
    {!transform}. *)

type t = { x0 : float; y0 : float; x1 : float; y1 : float }
(** The type for boxes, from the corner [(x0, y0)] to [(x1, y1)]. Invariant:
    [x0 <= x1] and [y0 <= y1]. *)

val v : float -> float -> float -> float -> t
(** [v x0 y0 x1 y1] is the box with corners [(x0, y0)] and [(x1, y1)], given in
    either order. *)

val width : t -> float
(** [width b] is [b.x1 -. b.x0]. *)

val height : t -> float
(** [height b] is [b.y1 -. b.y0]. *)

val union : t -> t -> t
(** [union a b] is the smallest box enclosing both [a] and [b]. *)

val transform : Affine.t -> t -> t
(** [transform m b] is the smallest box enclosing the images of [b]'s four
    corners under [m]. Under a rotation it is larger than the rotated box. *)
