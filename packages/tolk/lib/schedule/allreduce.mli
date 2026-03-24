(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Multi-device collective reduction.

    Implements naive, ring, and all-to-all allreduce strategies for reducing
    buffers across multiple devices. *)

val emit_shape : Tolk_ir.Tensor.builder -> int list -> Tolk_ir.Tensor.id
(** [emit_shape b dims] encodes a concrete int list as a Tensor shape node. *)

val emit_pairs :
  Tolk_ir.Tensor.builder -> (int * int) list -> Tolk_ir.Tensor.id * Tolk_ir.Tensor.id
(** [emit_pairs b pairs] encodes [(before, after)] pairs as two shape nodes. *)

val handle_allreduce :
  Tolk_ir.Tensor.builder ->
  shapes:int list option array ->
  devices:Tolk_ir.Tensor.device option array ->
  buf:Tolk_ir.Tensor.id ->
  red_op:Tolk_ir.Op.reduce ->
  red_device_id:Tolk_ir.Tensor.id ->
  Tolk_ir.Tensor.id option
(** [handle_allreduce b ~shapes ~devices ~buf ~red_op ~red_device_id]
    builds the allreduce computation graph. Returns [None] if [buf] is not
    on a multi-device. *)

val create_allreduce_function :
  Tolk_ir.Tensor.builder ->
  shapes:int list option array ->
  devices:Tolk_ir.Tensor.device option array ->
  buf:Tolk_ir.Tensor.id ->
  red_op:Tolk_ir.Op.reduce ->
  red_device_id:Tolk_ir.Tensor.id ->
  red_dtype:Tolk_ir.Dtype.t ->
  red_shape:int list ->
  red_size:int ->
  ?output:Tolk_ir.Tensor.id ->
  unit ->
  Tolk_ir.Tensor.id option
(** [create_allreduce_function b ~shapes ~devices ~buf ~red_op ~red_device_id
    ~red_dtype ~red_shape ~red_size] wraps {!handle_allreduce} into a CALL
    kernel with proper PARAM/BUFFER setup. *)
