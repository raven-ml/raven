(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Multi-device collective reduction.

    Builds allreduce computation graphs using naive, ring, or all-to-all
    strategies depending on device count, element count, and the [RING],
    [ALL2ALL], and [RING_ALLREDUCE_THRESHOLD] context variables. *)

(** {1:encoding Shape encoding} *)

val emit_shape : int list -> Tolk_ir.Tensor.t
(** [emit_shape dims] encodes [dims] as a tensor shape node. A single
    dimension becomes a scalar constant; multiple dimensions become a
    {!Tolk_ir.Tensor.vectorize} of scalar constants. *)

val emit_pairs :
  (int * int) list -> Tolk_ir.Tensor.t * Tolk_ir.Tensor.t
(** [emit_pairs pairs] splits [(lo, hi)] int pairs into two shape
    nodes [(emit_shape los, emit_shape his)]. *)

(** {1:allreduce Allreduce} *)

val handle_allreduce :
  Tolk_ir.Tensor.t ->
  op:Tolk_ir.Op.reduce ->
  device:Tolk_ir.Tensor.t ->
  Tolk_ir.Tensor.t option
(** [handle_allreduce buf ~op ~device] builds a reduction graph that
    combines every shard of [buf] with [op] and places the result
    on [device].

    Returns [None] if [buf] is not on a multi-device. Raises
    [Failure] if [buf] has no concrete shape.

    The strategy is selected automatically:
    {ul
    {- {e Naive} when the device count is [<= 2] or the element
       count is below [RING_ALLREDUCE_THRESHOLD] (default 256k).}
    {- {e All-to-all} when [ALL2ALL >= 2], or [ALL2ALL >= 1] and
       the size exceeds the threshold with [> 2] devices.}
    {- {e Ring} when [RING >= 2], or [RING >= 1] and the size
       exceeds the threshold with [> 2] devices.}} *)

val create_allreduce_function :
  Tolk_ir.Tensor.t ->
  op:Tolk_ir.Op.reduce ->
  device:Tolk_ir.Tensor.t ->
  dtype:Tolk_ir.Dtype.t ->
  shape:int list ->
  ?output:Tolk_ir.Tensor.t ->
  unit ->
  Tolk_ir.Tensor.t option
(** [create_allreduce_function buf ~op ~device ~dtype ~shape ()]
    wraps {!handle_allreduce} into a precompiled [CALL] kernel with
    parameter and buffer setup.

    [output] defaults to a fresh contiguous buffer of the given
    [dtype], [shape], and [device].

    Returns [None] if [buf] is not on a multi-device. *)
