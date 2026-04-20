(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Late expansion of structured vector markers.

    This pass owns late vector-shape realisation between post-range
    optimisation and devectorisation.  It converts scheduler-introduced
    {!Tolk_ir.Kernel.view.Unroll}/{!Tolk_ir.Kernel.view.Contract}
    structure into ordinary late-kernel vector operations, rewrites
    grouped reductions through {!Tolk_ir.Kernel.view.Bufferize}, and
    consumes [Upcast]/[Unroll] ranges on {!Tolk_ir.Kernel.view.Reduce},
    {!Tolk_ir.Kernel.view.Store}, and {!Tolk_ir.Kernel.view.End} before
    later lowering.

    The result is still late Kernel IR, not render-ready IR.  Later
    stages may still see transient vectorised reduce sources or
    vectorised pointer indexing forms that are consumed by
    {!Devectorizer}.

    See also {!Devectorizer}, {!Linearizer}. *)

val expand : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t
(** [expand root] runs symbolic simplification, range-to-unroll
    conversion, group-for-reduce lowering, and the main
    expand/contract engine as a single fixpoint.

    Replaces structured expansion markers
    ({!Tolk_ir.Kernel.view.Unroll}, {!Tolk_ir.Kernel.view.Contract})
    with ordinary late-kernel vector structure while keeping range
    fields scalar-only. *)
