(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Late reduction lowering and devectorization.

    Transforms Kernel IR from abstract buffer references into concrete
    {!Tolk_ir.Kernel.view.Load}/{!Tolk_ir.Kernel.view.Store} operations
    and scalarizes wide vector operations before linearization.

    The passes run in this order:
    + {!pm_reduce} — lower reductions to accumulator loops
    + {!pm_add_loads} — insert explicit loads
    + {!pm_devectorize} — split wide vector ops into scalar
    + {!pm_correct_load_store} — split oversized memory ops
    + {!pm_render} — prepare for rendering

    See also {!Expander}, {!Linearizer}, and {!Images}. *)

val pm_reduce : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t
(** [pm_reduce kernel] lowers all {!Tolk_ir.Kernel.view.Reduce}
    operations in [kernel] to explicit accumulator loops. Runs until
    fixpoint via {!Tolk_ir.Kernel.graph_rewrite}.

    The pass performs four rewrites:
    - {b Reduce lowering.} Transforms
      {!Tolk_ir.Kernel.view.Reduce} nodes into
      {!Tolk_ir.Kernel.view.Define_reg} accumulator initialization,
      explicit accumulation inside the range loop, and
      {!Tolk_ir.Kernel.view.End} to close the reduction scope.
    - {b Parallel-reduce merging.} Merges parallel reduces that share
      the same range into a single {!Tolk_ir.Kernel.view.End} via
      {!Tolk_ir.Kernel.view.Group}.
    - {b Wmma accumulate folding.} Folds
      [{!Tolk_ir.Kernel.view.Wmma}(...) + add] into
      [{!Tolk_ir.Kernel.view.Wmma}(a, b, c + add)] for tensor-core
      built-in accumulate.
    - {b Gep pushing.} Pushes {!Tolk_ir.Kernel.view.Gep} through
      {!Tolk_ir.Kernel.view.Vectorize} and
      {!Tolk_ir.Kernel.view.Const}. *)

val pm_add_loads : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t
(** [pm_add_loads kernel] inserts explicit {!Tolk_ir.Kernel.view.Load}
    instructions for {!Tolk_ir.Kernel.view.Index} nodes that are
    consumed as values (by ALU ops) rather than as pointers (by
    {!Tolk_ir.Kernel.view.Load}/{!Tolk_ir.Kernel.view.Store}).

    This is a direct pass (not rewrite-engine based) because it
    requires distinguishing pointer consumers from value consumers,
    which depends on the consuming instruction's structure, not just
    the referenced node. *)

val pm_devectorize_rule : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t option
(** Individual devectorize rule for composition. *)

val pm_devectorize : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t
(** [pm_devectorize kernel] scalarizes wide vector operations in
    [kernel] before final rendering.

    The pass performs:
    - {b ALU scalarization.} Splits vectorized ALU operations into
      per-lane scalar operations wrapped in
      {!Tolk_ir.Kernel.view.Vectorize}. For example,
      [Add(float4, float4)] becomes
      [Vectorize(Add(f32, f32), …, Add(f32, f32))].
    - {b Cast reordering.} Reorders {!Tolk_ir.Kernel.view.Cast}
      through {!Tolk_ir.Kernel.view.After}.
    - {b Wmma splitting.} Splits oversized
      {!Tolk_ir.Kernel.view.Wmma} results into multiple smaller
      instructions followed by
      {!Tolk_ir.Kernel.view.Vectorize}.
    - {b Local/register devectorization.} Scalarizes vectorized
      {!Tolk_ir.Kernel.view.Define_local}/
      {!Tolk_ir.Kernel.view.Define_reg} buffers by rewriting their
      vector loads and stores into per-lane scalar accesses.
    - {b Gate cleanup.} Drops trivially-true gates from
      {!Tolk_ir.Kernel.view.Index} nodes. *)

val pm_correct_load_store_rule : Renderer.t -> Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t option
(** Individual correct_load_store rule for composition. *)

val pm_correct_load_store : Renderer.t -> Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t
(** [pm_correct_load_store renderer kernel] splits oversized
    {!Tolk_ir.Kernel.view.Load}/{!Tolk_ir.Kernel.view.Store} operations
    into scalar operations when [renderer] does not support the full
    vector width (as reported by {!Renderer.load_store_widths}).

    Register-addrspace accesses are skipped (already handled by
    {!pm_devectorize}). Image fixup is not included; Tolk handles
    images separately via {!Images.rewrite}.

    Currently splits to scalar only; intermediate-width splitting is
    deferred until the Kernel IR gains a pointer-narrowing mechanism. *)

val load_store_folding_rule : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t option
(** Vectorized load/store restructuring: expand/fold INDEX on VECTORIZE,
    push GEP through LOAD/STORE, spread PTRCAT across LOAD/STORE.
    *)

val load_store_indexing_rule : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t option
(** Gate simplification on INDEX nodes (drop true gates).
    *)

val pm_render_rule : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t option
(** [pm_render_rule] is the individual rewrite rule for pm_render,
    suitable for composition with other rules in a single fixpoint. *)

val pm_render : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t
(** [pm_render kernel] performs final transformations before
    linearization:
    - Expands vector constants into explicit
      {!Tolk_ir.Kernel.view.Vectorize} of scalar constants.
    - Simplifies trivial {!Tolk_ir.Kernel.view.Gep} (single-element
      on scalar source).
    - Simplifies trivial {!Tolk_ir.Kernel.view.Vectorize} (single
      source).
    - Gives masked loads a default alt value of [0].
    - Folds [Where(cond, gated_load(cond), alt)] into a masked
      {!Tolk_ir.Kernel.view.Load} alt. *)
