(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** What {!Heuristic}'s rules read of a kernel and apply to it. *)

module K = Postrange.Scheduler

val debug : unit -> int
(** [debug ()] is the level of {!Setting.debug}. *)

val split : ?top:bool -> K.t -> int -> int -> Opt.target -> Ops.t list
(** [split k axis amount target] applies a split the heuristic has checked and
    returns the axes it made, with [top] choosing the outer end. Raises
    [Invalid_argument] if {!Postrange.Scheduler.apply_opt} refuses it. *)

val try_split : ?top:bool -> K.t -> int -> int -> Opt.target -> bool
(** [try_split k axis amount target] is whether the split applied. *)

val divisible : Ops.sint -> int -> bool
(** [divisible s n] is [true] iff [s] is a known size that [n] divides. *)

val shape_at : K.t -> int -> Ops.sint
(** [shape_at k i] is the size of [k]'s axis [i]. *)

val size_at : K.t -> int -> Bigint.t
(** [size_at k i] is the size of [k]'s axis [i]. Raises [Invalid_argument] if
    it is not known. *)

val holds : Shape.Sint.cond -> bool
(** [holds c] is the truth of the comparison [c]. *)

val idx : Ops.t -> Ops.t
(** [idx b] is the index of the access [b]. *)

val indexes : Ops.t -> Ops.t -> bool
(** [indexes r b] is [true] iff the index of [b] computes from [r], [r] aside. *)

val reads : Ops.t -> Ops.t -> bool
(** [reads r b] is [true] iff the index of [b] is or computes from [r]. *)

val stored_addresses : K.t -> Ops.t list
(** [stored_addresses k] is the addresses [k] stores to. *)

val gcd : int -> int -> int
(** [gcd a b] is the greatest common divisor of [a] and [b]. *)

val gcd_with : int -> Bigint.t -> int
(** [gcd_with a n] is the greatest common divisor of [n] with the positive
    [a]. *)

val run : Ops.t -> Ops.t -> Bigint.t
(** [run o r] is the run of [o] along the range [r]: [d] if [o] reads [r] only
    as [r / d], 1 otherwise. *)

val upcast_max : int
(** [upcast_max] is the most values an upcast of one axis takes
    ({!Postrange.Scheduler.apply_opt}'s bound). *)

val stride : Ops.t -> Ops.t -> Bigint.t option
(** [stride r i] is the stride of the range [r] in the index [i], when a term
    of [i] is [r] or [r * c]: [1] or [c]. *)
