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
  | Outer  (** Outer loop axis. *)
  | Placeholder  (** Placeholder for unassigned axes. *)

(** {1:predicates Predicates and comparisons} *)

val equal : t -> t -> bool
(** [equal a b] is [true] iff [a] and [b] are the same kind. *)

val compare : t -> t -> int
(** [compare a b] totally orders axis kinds by variant declaration order. *)

(** {1:fmt Formatting} *)

val pp : Format.formatter -> t -> unit
(** [pp] formats an axis kind as a lowercase string (e.g. ["global"],
    ["thread"], ["group_reduce"]). *)
