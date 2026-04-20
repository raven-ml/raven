(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Range simplification passes.

    Each pass takes a Kernel root and returns the transformed root after
    running its rewrite rules to fixpoint via {!Tolk_ir.Kernel.graph_rewrite}.

    Passes are composed in the codegen pipeline in this order:
    {!pm_load_collapse}, {!pm_split_ranges}, initial symbolic +
    {!flatten_range}, {!pm_simplify_ranges}. *)

val flatten_range : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t option
(** [flatten_range node] toposorts the range children of a Reduce, Store,
    or End node and reattaches them in sorted order. Returns [None] for
    other nodes or when already sorted. *)

val pm_flatten_range : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t
(** [pm_flatten_range root] applies {!flatten_range} to all nodes in the
    DAG. *)

val pm_split_ranges : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t
(** [pm_split_ranges root] splits ranges where [range % C] appears and
    the range size divides [C]. Each qualifying range with divisor [C]
    becomes [outer(size/C) * C + inner(C)].

    Image stores are excluded. *)

val pm_simplify_ranges : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t
(** [pm_simplify_ranges root] merges adjacent ranges with the same kind
    into a single range when it does not increase the divmod count, and
    shrinks gated ranges based on [r < C] guards extracted from Index
    nodes. Reduce ranges are never shrunk. *)

val pm_reduce_unparented : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t
(** [pm_reduce_unparented root] removes reduce ranges not referenced in
    the reduce source. For ADD reduces the removed range size is multiplied
    into the result; for MUL it is exponentiated. *)

val pm_reduce_simplify : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t
(** [pm_reduce_simplify root] combines {!pm_reduce_unparented} with
    reduce collapse: algebraic simplification that eliminates ranges from
    ADD reduces when possible. *)

val pm_load_collapse : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t
(** [pm_load_collapse root] collapses reduces over gated loads (tensor
    indexing patterns) and includes an undo rule to prevent arithmetic
    on loaded index values that could overflow. *)
