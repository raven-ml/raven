(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Bit-level operations on tolk float nodes, and the integer types of a node's
    width.

    A {e float node} is a node of [Float16], [Bfloat16], [Float32] or [Float64].
    The module reads and sets a float node's sign, tests for and makes NaNs, and
    builds the target's fused multiply-add, where the {e target} is the device a
    lowered program is rendered for. It reads a node's bits as the unsigned
    integer of its width ({!bits}). On those bits a sign, a NaN's payload and
    its {e quiet bit}, the fraction's top bit, are kept on every target, also
    where the target's float operations would change them. Each function on
    nodes keeps its operand's shape. *)

open Tolk

(** {1:types Types and extents} *)

val unsigned : Dtype.t -> Dtype.t
(** [unsigned t] is the unsigned integer type of [t]'s width
    ({!Tolk.Dtype.itemsize}), [Uint8] for [Bool].

    Raises [Invalid_argument] unless that width is 1, 2, 4 or 8 bytes. *)

val signed : Dtype.t -> Dtype.t
(** [signed t] is the signed integer type of [t]'s width
    ({!Tolk.Dtype.itemsize}).

    Raises [Invalid_argument] unless that width is 1, 2, 4 or 8 bytes. *)

val extent : Ops.t -> int -> int
(** [extent u i] is the extent of [u]'s axis [i].

    Raises [Invalid_argument] if that extent is symbolic. *)

(** {1:bits Bits} *)

val bits : Ops.t -> Ops.t
(** [bits x] is [x]'s bits, a node of {!unsigned} of its type. *)

val negative : Ops.t -> Ops.t
(** [negative x] is the [Bool] node true where the float [x]'s sign bit is set,
    [-0.] and a NaN of negative sign included. *)

val negated : Ops.t -> Ops.t
(** [negated x] is the float [x] with its sign bit flipped, every other bit
    kept, NaN included. *)

val magnitude : Ops.t -> Ops.t
(** [magnitude x] is the float [x] with its sign bit cleared, every other bit
    kept, NaN included. *)

(** {1:nan NaN} *)

val is_nan : Ops.t -> Ops.t
(** [is_nan x] is the [Bool] node true where the float [x] is a NaN. *)

val nan : Ops.t -> Ops.t
(** [nan x] is a quiet NaN of [x]'s float type and shape. Its sign bit is clear
    and the quiet bit is its only set fraction bit. *)

(** {1:fma Fused multiply-add} *)

val mulacc : Ops.t -> Ops.t -> Ops.t -> Ops.t
(** [mulacc a b c] is [a * b + c] of three floats of one type, rounded once by
    the target's fused multiply-add. With a NaN operand, the result's sign and
    payload are what that fused multiply-add gives. *)
