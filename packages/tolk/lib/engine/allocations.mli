(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Buffer allocation for tensor graphs.

    Decides which tensor computations need explicit buffer allocations
    and transforms a lazy tensor-level {!Tolk_ir.Tensor.view.Sink}
    into a {!Tolk_ir.Tensor.view.Call} with allocated buffers.

    The transformation runs in three phases:
    {ol
    {- {e Tag.}  Identify nodes that need realization
       ({!Tolk_ir.Tensor.view.Contiguous},
       {!Tolk_ir.Tensor.view.After}+{!Tolk_ir.Tensor.view.Store},
       and non-trivial bases of the sink's children).}
    {- {e Allocate.}  Replace tagged nodes with explicit
       {!Tolk_ir.Tensor.view.Buffer} +
       {!Tolk_ir.Tensor.view.Store} +
       {!Tolk_ir.Tensor.view.After} sequences.  When movement ops
       on a buffer collapse to a contiguous range, a
       {!Tolk_ir.Tensor.view.Buffer_view} is used instead.}
    {- {e Finalize.}  Strip internal bookkeeping, collect the
       resulting stores, replace input buffers with
       {!Tolk_ir.Tensor.view.Param} nodes for cache-key
       normalisation, and wrap everything in a
       {!Tolk_ir.Tensor.view.Call}.}}

    The returned [buffer_map] tracks which original tensor nodes map
    to which allocated buffers, keyed by {!Tolk_ir.Tensor.tag}. *)

val transform_to_call :
  Tolk_ir.Tensor.t ->
  Tolk_ir.Tensor.t * (int, Tolk_ir.Tensor.t) Hashtbl.t
(** [transform_to_call big_sink] is [(call, buffer_map)].

    [big_sink] must be a {!Tolk_ir.Tensor.view.Sink} node
    representing the lazy tensor graph to be realized.

    [call] is a {!Tolk_ir.Tensor.view.Call} whose callee is a
    parameterised sink (input buffers replaced by
    {!Tolk_ir.Tensor.view.Param} nodes) and whose arguments are
    the original buffer and bind nodes.

    [buffer_map] maps original tensor nodes to their allocated
    buffers, keyed by {!Tolk_ir.Tensor.tag}.  Downstream scheduling
    uses this to resolve tensor references to concrete buffers. *)
