(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Core rangeify algorithm.

    Converts the high-level tensor graph into an indexed representation with
    explicit RANGE loops, BUFFERIZE nodes, and INDEX operations. *)

val is_always_contiguous : Tolk_ir.Tensor.view -> bool
(** [is_always_contiguous v] is [true] for ops that never need realization. *)

val is_assign_after : Tolk_ir.Tensor.t -> Tolk_ir.Tensor.view -> bool
(** [is_assign_after program v] is [true] when [v] is an [After] whose deps
    include a [Store] — the Store+After pattern encoding buffer assignment. *)

type realize_state =
  | Marked
  | Realized of int list

type indexing_context = {
  realize_map : realize_state option array;
  range_map :
    (Tolk_ir.Tensor.id list * Tolk_ir.Tensor.id list) option array;
  mutable range_idx : int;
  range_axes : (Tolk_ir.Tensor.id, int) Hashtbl.t;
}

val create_context : int -> indexing_context

val new_range :
  Tolk_ir.Tensor.builder ->
  indexing_context ->
  int ->
  kind:Tolk_ir.Axis_kind.t ->
  Tolk_ir.Tensor.id

val simplify_tensor_expr :
  Tolk_ir.Tensor.builder ->
  Tolk_ir.Tensor.t ->
  Tolk_ir.Tensor.id ->
  Tolk_ir.Tensor.id
(** [simplify_tensor_expr b program id] converts a Tensor.t arithmetic
    expression to Kernel.t, applies symbolic simplification, and converts back.
    Used for reshape index simplification and post-rangeify optimization. *)

val apply_movement_op :
  Tolk_ir.Tensor.builder ->
  Tolk_ir.Tensor.t ->
  shapes:int list option array ->
  Tolk_ir.Tensor.view ->
  Tolk_ir.Tensor.id list ->
  Tolk_ir.Tensor.id list
(** [apply_movement_op b program ~shapes op rngs] transforms ranges through a
    movement op. Handles SHRINK, PERMUTE, FLIP, EXPAND, PAD, RESHAPE. *)

val run_rangeify :
  Tolk_ir.Tensor.builder ->
  Tolk_ir.Tensor.t ->
  shapes:int list option array ->
  indexing_context * Tolk_ir.Tensor.t
(** [run_rangeify b program ~shapes] runs the core rangeify analysis
    (backward walk) and merges new RANGE nodes into the program.
    Returns the context and the extended program. *)

val apply_rangeify_pass :
  Tolk_ir.Tensor.t ->
  indexing_context ->
  shapes:int list option array ->
  devices:Tolk_ir.Tensor.device option array ->
  Tolk_ir.Tensor.t
(** [apply_rangeify_pass program ctx ~shapes ~devices] applies the rangeify
    transformation: REDUCE_AXIS→REDUCE, PAD→WHERE, creates BUFFERIZE+INDEX,
    removes movement ops. Uses a manual translation pass with correct
    old→new id mapping. *)
