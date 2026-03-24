(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Backend-provided special dimensions.

    A special dimension identifies a hardware execution index supplied by the
    compute backend (e.g. OpenCL's [get_group_id], [get_local_id], or a fused
    global index). Each variant carries the axis number it refers to. *)

(** {1:types Types} *)

(** The type for special dimensions. *)
type t =
  | Group_id of int  (** Workgroup id on the given axis. *)
  | Local_id of int  (** Thread-local id within a workgroup. *)
  | Global_idx of int  (** Global thread index on the given axis. *)

(** {1:access Accessors} *)

val axis : t -> int
(** [axis d] is the axis number carried by [d]. *)

(** {1:predicates Predicates and comparisons} *)

val equal : t -> t -> bool
(** [equal a b] is [true] iff [a] and [b] denote the same special dimension and
    axis. *)

val compare : t -> t -> int
(** [compare a b] totally orders special dimensions. *)

(** {1:fmt Formatting} *)

val pp : Format.formatter -> t -> unit
(** [pp] formats a special dimension as a compact string (e.g. [gid0], [lid1],
    [idx2]). *)
