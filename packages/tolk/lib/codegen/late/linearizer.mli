(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Kernel-to-program linearization.

    Provides rewrite passes for splitting multi-range Ends and adding
    control-flow edges, plus the final priority-based toposort and
    Kernel → Program emission.

    The input kernel must be in late codegen form: exactly one
    {!Tolk_ir.Kernel.view.Sink}, no unlowered nodes
    ({!Tolk_ir.Kernel.view.Reduce}, {!Tolk_ir.Kernel.view.Bufferize},
    {!Tolk_ir.Kernel.view.Ptrcat}, {!Tolk_ir.Kernel.view.Vcat},
    {!Tolk_ir.Kernel.view.Unroll}, {!Tolk_ir.Kernel.view.Contract}),
    and gated loads must already carry an alternate value.

    See also {!Devectorizer}, {!Expander}. *)

val do_split_ends : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t option
(** [do_split_ends node] splits a multi-range End into nested single-range
    Ends. Returns [None] for non-End nodes or single-range Ends. *)

val pm_split_ends : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t
(** [pm_split_ends root] splits multi-range {!Tolk_ir.Kernel.view.End}
    nodes into nested single-range Ends, sorted by axis (descending). *)

val pm_add_control_flow : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t
(** [pm_add_control_flow sink] computes control-flow dependencies between
    sibling loops and adds ordering edges to {!Tolk_ir.Kernel.view.Range}
    nodes. *)

val linearize : Tolk_ir.Kernel.t -> Tolk_ir.Program.t
(** [linearize sink] performs priority-based topological ordering and
    emits a flat {!Tolk_ir.Program.t}.  Expects [pm_split_ends] and
    [pm_add_control_flow] to have already been applied.

    Raises [Failure] if [sink] contains unlowered nodes or has
    unclosed ranges after emission. *)
