(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Range simplification pattern matchers.

    These passes simplify range structure in Kernel IR before late lowering.

    Each function takes a Kernel root and returns the transformed root after
    running its pattern matcher(s) to fixpoint via
    {!Tolk_ir.Kernel.graph_rewrite}. *)

val node_vmin : Tolk_ir.Kernel.t -> int
(** [node_vmin n] is a lower bound on the integer value of [n].
    Used by reduce collapse to create symbolic placeholders with tight
    bounds. *)

val node_vmax : Tolk_ir.Kernel.t -> int
(** [node_vmax n] is an upper bound on the integer value of [n]. *)

val pm_flatten_range : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t
(** [pm_flatten_range root] toposorts the range children of
    {!Tolk_ir.Kernel.view.Reduce}, {!Tolk_ir.Kernel.view.Store}, and
    {!Tolk_ir.Kernel.view.End} nodes and reattaches them in sorted order. *)

val pm_split_ranges : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t
(** [pm_split_ranges root] splits ranges where [Range % Const] appears and
    the range size divides the constant. Each qualifying range [r] with
    divisor [c] is replaced with [Range(size/c) * c + Range(c)].

    Image stores are excluded: their ranges are not substituted. *)

val pm_simplify_ranges : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t
(** [pm_simplify_ranges root] performs two simplifications:

    - {b Merge adjacent ranges.} For {!Tolk_ir.Kernel.view.End} and
      {!Tolk_ir.Kernel.view.Reduce} nodes, tries merging pairs of ranges
      with the same kind into a single [Range(s0 * s1)], keeping the
      merge only if it does not increase the divmod count.
    - {b Shrink gated ranges.} From {!Tolk_ir.Kernel.view.Index} nodes,
      detects guards of the form [r < c] and shrinks the corresponding
      range bound. Reduce ranges are never shrunk. *)

val pm_reduce_unparented : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t
(** [pm_reduce_unparented root] removes reduce ranges that are not referenced
    in the reduce source. For [ADD] reduces, the removed range size is
    multiplied into the result. For [MUL], the range size is exponentiated. *)

val pm_reduce_simplify : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t
(** [pm_reduce_simplify root] combines {!pm_reduce_unparented} with symbolic
    reduce collapse: algebraic simplification that eliminates ranges from
    ADD reduces when possible (arange optimization, indexing). *)

val pm_load_collapse : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t
(** [pm_load_collapse root] extends reduce collapse with load-specific rules
    for collapsing reduces over gated loads (tensor indexing patterns).

    Also includes an undo rule to prevent math on loaded index values that
    could overflow. *)
