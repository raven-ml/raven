(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Schedule pipeline: tensor graph to kernel graph.

    Transforms a tensor-level SINK into a graph of CALL nodes wrapping
    {!Tolk_ir.Kernel.t} ASTs ready for codegen.  The pipeline has ten
    passes:

    {ol
    {- {e multi_pm} — multi-device rewriting.}
    {- {e fold_moved_after} — openpilot AFTER folding (when enabled).}
    {- {e earliest_rewrites} — syntactic sugar, movement ops,
       call resolution, allreduce, split-reduce, size-0 folding.}
    {- {e run_rangeify} — core range analysis (in {!Indexing}).}
    {- {e apply_rangeify} — bottom-up rewrite with rangeify context.}
    {- {e post-rangeify} — dead-axis cleanup, buffer folding, const
       folding, cost-based buffer removal.}
    {- {e limit_bufs} — insert BUFFERIZE when a kernel exceeds the
       device buffer limit.}
    {- {e add_buffers} — lower BUFFERIZE to STORE + BUFFER.}
    {- {e split_kernels} — convert STORE/END subtrees into
       CALL(kernel SINK).}
    {- {e WAR deps} — write-after-read dependency fixup.}} *)

val get_kernel_graph : Tolk_ir.Tensor.t -> Tolk_ir.Tensor.t
(** [get_kernel_graph sink] is the kernel graph for [sink].

    [sink] is a tensor-level SINK node.  The returned graph contains
    AFTER nodes whose deps are CALL nodes wrapping
    {!Tolk_ir.Kernel.t} ASTs, connected by WAR dependency edges. *)
