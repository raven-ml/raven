(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** 3D kd-tree for nearest-neighbor queries.

    {b Note.} Private module. *)

type t
(** The type for a 3D kd-tree. *)

val build : float array -> float array -> float array -> t
(** [build xs ys zs] is a kd-tree over the points [(xs.(i), ys.(i), zs.(i))].
    The three arrays must have equal length.

    Raises [Invalid_argument] if the arrays differ in length. *)

val nearest : t -> float -> float -> float -> int * float
(** [nearest tree qx qy qz] is [(i, d2)] where [i] is the index of the nearest
    point to [(qx, qy, qz)] and [d2] is the squared Euclidean distance.

    Raises [Invalid_argument] if the tree is empty. *)

val within : t -> float -> float -> float -> float -> (int * float) list
(** [within tree qx qy qz max_d2] is the list of [(i, d2)] pairs for all points
    within squared Euclidean distance [max_d2] of [(qx, qy, qz)]. *)
