(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Points of the plane. *)

(** {1:types Types} *)

type t
(** The type for points. Coordinates are any floats; a non-finite coordinate is
    meaningful only where a function says so, as in a path's gaps. *)

(** {1:constructors Constructors and accessors} *)

val v : float -> float -> t
(** [v x y] is the point [(x, y)]. *)

val x : t -> float
(** [x p] is the x coordinate of [p]. *)

val y : t -> float
(** [y p] is the y coordinate of [p]. *)

(** {1:transforming Transforming} *)

val transform : Affine.t -> t -> t
(** [transform m p] is the image of [p] under [m]. *)

(** {1:comparing Comparing and formatting} *)

val equal : t -> t -> bool
(** [equal p q] is [true] iff [p] and [q] have equal coordinates. *)

val compare : t -> t -> int
(** [compare p q] orders points by x, then by y. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf p] formats [p] as [(x, y)] for debugging. *)
