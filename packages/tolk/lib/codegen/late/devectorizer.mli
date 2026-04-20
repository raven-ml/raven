(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Late reduction lowering and devectorization.

    Transforms Kernel IR from abstract buffer references into concrete
    {!Tolk_ir.Kernel.view.Load}/{!Tolk_ir.Kernel.view.Store} operations,
    scalarises wide vector operations, and folds load/store grouping before
    linearisation.

    Image-related passes are omitted; Tolk handles images separately via
    {!Images}.

    The passes run in this order (composed by {!Lowering.lower}):
    + {!pm_reduce} — lower reductions to accumulator loops.
    + {!pm_add_loads} — insert explicit loads.
    + {!pm_devectorize} — scalarise, fold, correct, and simplify.
    + {!pm_render} — prepare for rendering.

    See also {!Expander}, {!Linearizer}. *)

(** {1:passes Passes} *)

val pm_reduce : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t
(** [pm_reduce root] lowers {!Tolk_ir.Kernel.view.Reduce} nodes to
    explicit {!Tolk_ir.Kernel.view.Define_reg} accumulator loops with
    {!Tolk_ir.Kernel.view.End}. Parallel reductions that share the same
    range are merged into a single {!Tolk_ir.Kernel.view.End} via
    {!Tolk_ir.Kernel.view.Group}. Also folds
    [{!Tolk_ir.Kernel.view.Wmma} + add] into the WMMA accumulator.
    Includes GEP pushing ({!Symbolic.gep_pushing}) in the same fixpoint,
    matching tinygrad's [pm_reduce+gep_pushing] composition. *)

val pm_add_loads : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t
(** [pm_add_loads root] inserts explicit {!Tolk_ir.Kernel.view.Load}
    for value-typed {!Tolk_ir.Kernel.view.Index} nodes, and collapses
    [Store(Load(x), v)] to [Store(x, v)]. *)

val pm_devectorize : Renderer.t -> Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t
(** [pm_devectorize renderer root] runs a single fixpoint that:

    - Scalarises vectorised ALU, Cast, Bitcast, and WMMA.
    - Scalarises {!Tolk_ir.Kernel.view.Define_local}/
      {!Tolk_ir.Kernel.view.Define_reg} with vector base types.
    - Scalarises vector {!Tolk_ir.Kernel.view.Index} on local/reg
      memory (plain, broadcast, and GEP patterns).
    - Reorders {!Tolk_ir.Kernel.view.Cast} through
      {!Tolk_ir.Kernel.view.After}.
    - Expands and folds vectorised INDEX for load/store grouping
      (consecutive offsets share a single wide pointer).
    - Pushes {!Tolk_ir.Kernel.view.Gep} through Load/Store.
    - Spreads {!Tolk_ir.Kernel.view.Ptrcat} across Load/Store.
    - Splits oversized Load/Store for [renderer]
      (as reported by {!Renderer.supports_float4}).
    - Drops trivially-true gates from
      {!Tolk_ir.Kernel.view.Index}.
    - Applies symbolic simplification. *)

val load_store_indexing : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t option
(** [load_store_indexing node] simplifies INDEX validity gates:
    drops always-true gates, simplifies gated indexes via
    {!Symbolic.uop_given_valid}, and for image buffers drops redundant
    validity clauses proved by image dimension bounds. *)

val no_vectorized_alu : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t option
(** [no_vectorized_alu node] scalarizes a vectorized ALU, Cast, or Bitcast
    by extracting each lane via GEP, applying the scalar operation, and
    re-vectorizing.  Returns [None] for scalar nodes or image-index WHERE
    patterns.  Used by renderer [extra_pm] to devectorize bool-typed ops
    and WHERE in the final rewrite fixpoint. *)

(** {1:render Render preparation} *)

val pm_render_rule : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t option
(** [pm_render_rule node] is the individual render-preparation rewrite
    rule, suitable for composition with decomposition and renderer rules
    in a final fixpoint.

    Expands vector {!Tolk_ir.Kernel.view.Const},
    {!Tolk_ir.Kernel.view.Vconst}, and multi-element
    {!Tolk_ir.Kernel.view.Gep} to {!Tolk_ir.Kernel.view.Vectorize};
    removes trivial GEP and single-element Vectorize; gives gated loads
    a zero alt value; folds [Where(cond, gated_load, fallback)] into the
    load's alt. *)

val pm_render : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t
(** [pm_render root] runs {!pm_render_rule} to fixpoint. *)
