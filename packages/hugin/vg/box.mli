(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Axis-aligned boxes. *)

type t = { x0 : float; y0 : float; x1 : float; y1 : float }
(** The type for boxes, from the corner [(x0, y0)] to [(x1, y1)] with [x0 <= x1]
    and [y0 <= y1]. *)

val v : float -> float -> float -> float -> t
(** [v x0 y0 x1 y1] is the box with the given corners, in either order. *)

val width : t -> float
val height : t -> float

val union : t -> t -> t
(** [union a b] is the smallest box enclosing both. *)

val transform : Affine.t -> t -> t
(** [transform m b] is the box enclosing the image of [b]'s corners under [m].
*)
