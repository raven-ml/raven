(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Schedule pipeline: tensor graph to kernel graph.

    Transforms a tensor-level SINK into a graph of CALL nodes wrapping
    kernel ASTs ready for codegen.  Preparation precedes range assignment. The indexing pipeline runs:

    {ol
    {- {e run_rangeify} — core range analysis (in {!Indexing}).}
    {- {e apply_rangeify} — bottom-up rewrite with rangeify context.}
    {- {e post-rangeify} — dead-axis cleanup, buffer folding, const
       folding, cost-based buffer removal.}
    {- {e limit_bufs} — insert STAGE when a kernel exceeds the
       device buffer limit.}
    {- {e add_buffers} — lower STAGE to STORE + ALLOC or local BUFFER.}
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

