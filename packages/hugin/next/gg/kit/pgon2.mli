(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Polygons.

    A polygon is a list of {!Ring2} rings. Its surface is the set of points
    around which the rings together wind a non-zero number of times, as the
    {{!Hugin_next_gg_kit.section-conventions}conventions} state. Outer
    boundaries are positively oriented and holes negatively: a square with a
    hole is a positively oriented square and, inside it, a negatively oriented
    ring. {!Field2.isoband} returns polygons of this form, whose rings neither
    cross nor overlap. *)

open Hugin_next_gg

(** {1:types Types} *)

type t
(** The type for polygons. *)

(** {1:constructors Constructors} *)

val v : Ring2.t list -> t
(** [v rs] is the polygon of the rings [rs], in order. *)

(** {1:accessors Accessors} *)

val rings : t -> Ring2.t list
(** [rings p] is the rings of [p], in order. *)

(** {1:measures Measures} *)

val area : t -> float
(** [area p] is the signed area of the rings of [p], the sum of their
    {!Ring2.area}s. It is the area of the surface of [p], up to rounding, when
    the rings of [p] wind [0] or [1] times around every point off them, as
    those of {!Field2.isoband} do. *)

val mem : P2.t -> t -> bool
(** [mem pt p] is [true] iff [pt] is in the surface of [p]: the winding numbers
    of the rings of [p] around [pt] do not sum to [0], a point on a ring being
    decided as the {{!Hugin_next_gg_kit.section-conventions}conventions} state.
    It is [false] if a coordinate of [pt] is not finite. *)

val bounds : t -> Box2.t option
(** [bounds p] is the smallest box containing the points of the rings of [p], or
    [None] if they have no point. *)

(** {1:converting Converting} *)

val to_path : t -> Path.t
(** [to_path p] is the subpaths {!Ring2.to_path} gives for the rings of [p], in
    order. Filled with the nonzero rule, it paints the surface of [p]. *)

(** {1:comparing Comparing and formatting} *)

val equal : t -> t -> bool
(** [equal p p'] is [true] iff [p] and [p'] have {!Ring2.equal} rings in the
    same order. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf p] formats the rings of [p] in order as {!Ring2.pp} does, separated
    by spaces. *)
