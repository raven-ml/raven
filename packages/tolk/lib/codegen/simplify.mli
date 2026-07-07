(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Range simplification passes.

    Each pass takes a Uop root and returns the transformed root after
    running its rewrite rules to fixpoint via {!Tolk_uop.Uop.graph_rewrite}.

    Passes are composed in the codegen pipeline in this order:
    {!load_collapse_all}, {!split_ranges}, initial symbolic +
    {!flatten_range}, {!simplify_ranges}. *)

val flatten_range : Tolk_uop.Uop.t -> Tolk_uop.Uop.t option
(** [flatten_range node] toposorts the range children of a Reduce or End
    node and reattaches them in sorted order. Returns [None] for
    other nodes or when already sorted. *)

val split_ranges : Tolk_uop.Uop.t -> Tolk_uop.Uop.t
(** [split_ranges root] splits ranges where [range floormod C] appears
    and [C] divides the range size. Each qualifying range with divisor
    [C] becomes [outer(size//C) * C + inner(C)]. C-style modulo is not a
    split trigger. *)

val simplify_ranges : Tolk_uop.Uop.t -> Tolk_uop.Uop.t
(** [simplify_ranges root] merges adjacent ranges with the same kind
    into a single range when it does not increase the divmod count, and
    shrinks gated ranges based on [r < C] guards extracted from Index
    nodes. Reduce ranges are never shrunk. *)

val reduce_unparented_all : Tolk_uop.Uop.t -> Tolk_uop.Uop.t
(** [reduce_unparented_all root] removes reduce ranges not referenced in
    the reduce source. For ADD reduces the removed range size is multiplied
    into the result; for MUL it is exponentiated. *)

val reduce_simplify_all : Tolk_uop.Uop.t -> Tolk_uop.Uop.t
(** [reduce_simplify_all root] removes unparented reduce ranges and
    algebraically collapses ADD reduces when possible. *)

val pm_reduce_simplify : Tolk_uop.Upat.Pattern_matcher.t
(** [pm_reduce_simplify] is the node-level matcher used by
    {!reduce_simplify_all}. *)

val load_collapse_all : Tolk_uop.Uop.t -> Tolk_uop.Uop.t
(** [load_collapse_all root] collapses reduces over gated loads and includes
    an undo rule to prevent arithmetic on loaded index values that could
    overflow. *)
