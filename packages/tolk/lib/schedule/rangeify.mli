(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Schedule pipeline: tensor graph to kernel graph.

    Transforms a tensor-level SINK into a graph of CALL nodes wrapping
    kernel ASTs ready for codegen.  The pipeline has ten passes:

    {ol
    {- {e multi_pm} — multi-device rewriting.}
    {- {e fold_moved_after} — openpilot AFTER folding (when enabled).}
    {- {e earliest_rewrites} — syntactic sugar, movement ops,
       call resolution, allreduce, split-reduce, size-0 folding.}
    {- {e run_rangeify} — core range analysis (in {!Indexing}).}
    {- {e apply_rangeify} — bottom-up rewrite with rangeify context.}
    {- {e post-rangeify} — dead-axis cleanup, buffer folding, const
       folding, cost-based buffer removal.}
    {- {e limit_bufs} — insert STAGE when a kernel exceeds the
       device buffer limit.}
    {- {e add_buffers} — lower STAGE to STORE + BUFFER.}
    {- {e split_kernels} — convert STORE/END subtrees into
       CALL(kernel SINK).}
    {- {e WAR deps} — write-after-read dependency fixup.}} *)

val get_kernel_graph : Tolk_uop.Uop.t -> Tolk_uop.Uop.t
(** [get_kernel_graph sink] is the kernel graph for [sink].

    [sink] is a tensor-level SINK node.  The returned graph contains
    AFTER nodes whose deps are CALL nodes wrapping kernel ASTs,
    connected by WAR dependency edges. *)

val early_movement_pass : Tolk_uop.Uop.t -> Tolk_uop.Uop.t
(** [early_movement_pass sink] runs the cleanup rewrites that the reference
    applies at the very top of codegen on a just-split kernel body: strip
    movement ops on [INDEX], push movement ops past [AFTER]/[END], merge
    nested [INDEX]es, and add explicit [RANGE] loops to any shaped [STORE].

    This must run before [full_rewrite_to_sink]'s optimize stage so a
    scalar [STORE(reshape(param)(1,))] is lifted into
    [STORE(param.index(r), value.index(r)).end(r)] and later passes see
    plain pointer-indexed form. *)

val rewrite_movement_ops : Tolk_uop.Uop.t -> Tolk_uop.Uop.t
(** [rewrite_movement_ops sink] pushes movement ops through [INDEX],
    [AFTER], and [END], and merges nested [INDEX] nodes. *)

val movement_ops : Tolk_uop.Uop.t -> Tolk_uop.Uop.t option
(** [movement_ops u] is the movement/index part of tinygrad's [pm_mops]:
    push movement ops through [INDEX], [AFTER], and [END]. *)

val mop_cleanup : Tolk_uop.Uop.t -> Tolk_uop.Uop.t option
(** [mop_cleanup u] performs the movement-op cleanup shared with codegen's
    remove-reduce pass. *)

val add_local_buffers : Tolk_uop.Uop.t -> Tolk_uop.Uop.t
(** [add_local_buffers sink] materializes local {!Tolk_uop.Ops.Stage}
    nodes inside a codegen kernel body, matching tinygrad's
    [pm_add_buffers_local + rangeify_codegen] pass. *)
