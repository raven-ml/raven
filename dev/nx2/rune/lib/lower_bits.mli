(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Nodes by their bits.

    The lowering states nx's numerics on a node's bits, read as the unsigned
    integer of its width: a sign, a NaN's payload or a quiet bit is a bit
    pattern there, which no target's float arithmetic changes. A float node is
    one of tolk's float types; each function keeps its operand's shape. *)

open Tolk

(** {1:types Types} *)

val unsigned : Dtype.t -> Dtype.t
(** [unsigned t] is the unsigned integer type of [t]'s width. *)

val signed : Dtype.t -> Dtype.t
(** [signed t] is the signed integer type of [t]'s width. *)

val extent : Ops.t -> int -> int
(** [extent u i] is the extent of [u]'s axis [i].

    Raises [Invalid_argument] if it is symbolic. *)

(** {1:bits Bits} *)

val bits : Ops.t -> Ops.t
(** [bits x] is [x]'s bits, a node of {!unsigned} of its type. *)

val negative : Ops.t -> Ops.t
(** [negative x] is whether the float [x]'s sign bit is set: true on [-0.] and
    on a NaN of negative sign. *)

val negated : Ops.t -> Ops.t
(** [negated x] is the float [x] with its sign bit flipped, every other bit
    kept, NaN included. *)

val magnitude : Ops.t -> Ops.t
(** [magnitude x] is the float [x] with its sign bit cleared, every other bit
    kept, NaN included. *)

(** {1:nan NaN} *)

val is_nan : Ops.t -> Ops.t
(** [is_nan x] is whether the float [x] is a NaN. *)

val nan : Ops.t -> Ops.t
(** [nan x] is the quiet NaN of a clear sign bit and no other fraction bit, of
    the float type and shape of [x]. *)

(** {1:fma Fused multiply-add} *)

val mulacc : Ops.t -> Ops.t -> Ops.t -> Ops.t
(** [mulacc a b c] is [a * b + c] of three floats of one type, rounded once by
    the target's fused multiply-add, NaN operands giving what it gives. *)
