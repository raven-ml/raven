(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Kernel-to-program linearization.

    Provides rewrite passes for splitting multi-range Ends and adding
    control-flow edges, plus the final priority-based toposort and
    Uop → Program emission.

    The input kernel must be in linearize-ready form: exactly one
    {!Tolk_uop.Ops.Sink}, no unlowered nodes
    ({!Tolk_uop.Ops.Reduce}, {!Tolk_uop.Ops.Stage}),
    gated loads must already carry an alternate value, and graph-stage
    {!Tolk_uop.Ops.If}/{!Tolk_uop.Ops.Endif} nodes are rejected.
    Gated stores are lowered after ordering into explicit
    [If; Store; Endif] program lines.

    See also {!Codegen_lower}. *)

val do_split_ends : Tolk_uop.Uop.t -> Tolk_uop.Uop.t option
(** [do_split_ends node] splits a multi-range End into nested single-range
    Ends. Returns [None] for non-End nodes or single-range Ends. *)

val pm_split_ends : Tolk_uop.Uop.t -> Tolk_uop.Uop.t
(** [pm_split_ends root] splits multi-range {!Tolk_uop.Ops.End}
    nodes into nested single-range Ends, sorted by full range argument
    (descending). *)

val pm_add_control_flow : Tolk_uop.Uop.t -> Tolk_uop.Uop.t
(** [pm_add_control_flow sink] computes control-flow dependencies between
    sibling loops and adds ordering edges to {!Tolk_uop.Ops.Range}
    nodes. *)

val linearize : Tolk_uop.Uop.t -> Program_spec.program
(** [linearize sink] performs priority-based topological ordering and
    emits a flat {!Program_spec.program}. Gated stores are rewritten to
    explicit {!Tolk_uop.Ops.If}, ungated {!Tolk_uop.Ops.Store}, and
    {!Tolk_uop.Ops.Endif} lines. Expects [pm_split_ends] and
    [pm_add_control_flow] to have already been applied.

    Raises [Failure] if [sink] contains unlowered nodes, gated loads without
    an alternate value, empty groups, or graph-stage
    {!Tolk_uop.Ops.If}/{!Tolk_uop.Ops.Endif} nodes. Full program validation is
    performed by {!Tolk_uop.Spec.verify_list} with
    {!Tolk_uop.Spec.program_spec}. *)
