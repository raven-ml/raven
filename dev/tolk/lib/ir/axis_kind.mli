(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Kernel axis kinds.

    An axis kind classifies each loop axis in a kernel schedule. The kind
    determines how the axis maps to hardware execution dimensions (threads,
    workgroups, registers) or to compiler transforms (unroll, upcast, reduce).

    {b Note.} The variant declaration order is load-bearing: {!compare}
    delegates to {!Stdlib.compare}, so reordering variants changes the sort
    order used by the optimizer. *)

(** {1:types Types} *)

(** The type for axis kinds. *)
type t =
  | Global  (** Global work dimension. *)
  | Thread  (** Per-thread dimension. *)
  | Local  (** Workgroup-local dimension. *)
  | Warp  (** Warp-level dimension. *)
  | Loop  (** Software loop. *)
  | Group_reduce  (** Group-level reduction. *)
  | Reduce  (** Reduction axis. *)
  | Upcast  (** Vectorization (upcast) axis. *)
  | Unroll  (** Unrolled loop axis. *)
  | Placeholder  (** Placeholder for unassigned axes. *)

(** {1:predicates Predicates and comparisons} *)

val equal : t -> t -> bool
(** [equal a b] is [true] iff [a] and [b] are the same kind. *)

val compare : t -> t -> int
(** [compare a b] totally orders axis kinds by variant declaration order. *)

(** {1:fmt Formatting} *)

val to_string : t -> string
(** [to_string kind] is a lowercase string (e.g. ["global"], ["group_reduce"]). *)

val pp : Format.formatter -> t -> unit
(** [pp] formats an axis kind with {!to_string}. *)

(** {1:data Data definitions} *)

(* CR: that's a weird API, I suspect this is used to implement something that should be provided in axis_kind itself possibly? *)
val to_pos : t -> int
(** [to_pos kind] is the sorting priority for [kind]. *)

(* CR: what is this used for? if it's really useful we should document *)
val letter : t -> string
(** [letter kind] is a single-character label (e.g. ["g"], ["l"], ["R"]). *)

(* CR: is this even used? That's such a weird api, seems out of place? *)
val color : t -> string
(** [color kind] is a debug color name (e.g. ["blue"], ["cyan"]). *)
