(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Elementwise operations as UOps.

    Each function takes the nodes of an operation's operands, of one shape and
    dtype as nx gives them, and is the node that computes its result as nx
    documents it:

    - {b narrow floats} ([float16], [bfloat16], the 8-bit floats) compute at
      [float32] and round once;
    - {b integers} wrap: signed and narrow operations compute on the unsigned
      bit pattern of at least 32 bits, where overflow is defined;
    - {b extremes} follow IEEE 754-2019: NaN when an operand is NaN, and [-0.]
      below [0.];
    - {b the transcendental functions} are accurate compositions of [exp2],
      [log2], [sin] and [sqrt], within a few units in the last place of the
      correctly rounded result;
    - {b a float converted to an integer} saturates at the integer's range, and
      NaN is 0. *)

open Tolk_next

val widen : Ops.t -> Ops.t
(** [widen x] is [x] converted to [float32] if it is a narrow float, and [x]
    otherwise. *)

val unary : Nx_backend.unary -> Ops.t -> Ops.t
(** [unary k x] is [k] of each element of [x]. *)

val binary : Nx_backend.binary -> Ops.t -> Ops.t -> Ops.t
(** [binary k x y] is [k] of the elements of [x] and [y]. An integer quotient or
    remainder by 0 is 0, and so is a remainder by -1. *)

val compare : Nx_backend.compare -> Ops.t -> Ops.t -> Ops.t
(** [compare k x y] is the boolean comparison [k] of the elements of [x] and
    [y], false where either is NaN except for [Not_equal]. *)

val cast : Dtype.t -> Ops.t -> Ops.t
(** [cast dt x] is each element of [x] converted to [dt]. *)

val bitcast : Dtype.t -> Ops.t -> Ops.t
(** [bitcast dt x] is the bits of each element of [x] read as [dt], of the same
    width. *)

val threefry : Ops.t -> Ops.t -> Ops.t
(** [threefry key counter] is the Threefry-2x32 hash of each word pair of the
    [int32] node [counter] under the pair of [key] at the same position, the
    pairs along the last axis, the low word first. *)
