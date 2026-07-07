(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Rangeify: tensor graph to indexed representation.

    Converts the high-level graph (movement ops, REDUCE, etc.)
    into an indexed representation with explicit RANGE loops,
    STAGE nodes, and INDEX operations.

    The algorithm runs in three phases:

    {ol
    {- {b Realize map.}  Decide which nodes need their own buffer
       (realization boundary).  See {!generate_realize_map}.}
    {- {b Range propagation.}  Walk the graph root-to-leaf, assigning
       one range expression per axis to every node.  Realized nodes get
       fresh ranges; others inherit or merge from consumers.  Movement
       ops transform ranges instead of persisting as nodes.
       See {!run_rangeify}.}
    {- {b Apply.}  Bottom-up graph rewrite: REDUCE keeps explicit ranges,
       PAD becomes WHERE, realized sources are wrapped in
       STAGE + INDEX or END, and movement ops are removed.
       See {!apply_rangeify_pass}.}} *)

(** {1:predicates Predicates} *)

val always_contiguous : Tolk_uop.Ops.t -> bool
(** [always_contiguous op] is [true] for ops whose output is
    contiguous by definition (Contiguous, After, Copy, Buffer,
    Slice, Const, Bind, Device, Mselect, Mstack, Param,
    Load, Call).  Their consumers can
    index directly without realization. *)

(** {1:context Indexing context} *)

type realize_state =
  | Marked
      (** Pending realization — set during realize-map construction,
          before axis resolution. *)
  | Realized of int list
      (** Resolved — records which output axes were realized. *)
(** Realization state for a single node. *)

type indexing_context = {
  realize_map : (int, realize_state) Hashtbl.t;
  range_map : (int, Tolk_uop.Uop.t list * Tolk_uop.Uop.t list) Hashtbl.t;
      (** Maps {!Tolk_uop.Uop.tag} to [(input_ranges, output_ranges)]. *)
  nodes : (int, Tolk_uop.Uop.t) Hashtbl.t;
  mutable range_idx : int;
      (** Monotonic counter for fresh range axis indices. *)
}
(** Per-node state populated by {!run_rangeify}.  All maps are keyed
    by {!Tolk_uop.Uop.tag}. *)

val create_context : unit -> indexing_context
(** [create_context ()] is a fresh, empty context. *)

val new_range :
  indexing_context -> int -> ?kind:Tolk_uop.Axis_type.t -> unit ->
  Tolk_uop.Uop.t
(** [new_range ctx size ?kind ()] is a fresh RANGE node over
    \[[0];[size-1]\] with axis kind [kind] (default {!Tolk_uop.Axis_type.Loop}).
    Returns a constant [0] when [size] is [1]. *)

val new_range_expr :
  indexing_context ->
  Tolk_uop.Uop.t ->
  ?kind:Tolk_uop.Axis_type.t ->
  unit ->
  Tolk_uop.Uop.t
(** [new_range_expr ctx size ?kind ()] is like {!new_range}, but [size]
    is a symbolic integer expression. Returns [size] unchanged if it is
    already a {!Tolk_uop.Ops.Range}, and returns a constant [0] when [size]
    simplifies to the constant [1]. *)

(** {1:simplify Symbolic simplification} *)

val simplify_expr : Tolk_uop.Uop.t -> Tolk_uop.Uop.t
(** [simplify_expr e] applies {!Tolk_uop.Symbolic.sym} to [e] through a
    graph rewrite. *)

(** {1:movement Movement ops} *)

val apply_movement_op :
  ?shape_exprs:(Tolk_uop.Uop.t -> Tolk_uop.Uop.t list option) ->
  shapes:(Tolk_uop.Uop.t -> int list option) ->
  Tolk_uop.Uop.t ->
  Tolk_uop.Uop.t list ->
  Tolk_uop.Uop.t list
(** [apply_movement_op ?shape_exprs ~shapes node rngs] transforms [rngs]
    (output ranges) through a movement op, producing the corresponding input
    ranges. [shape_exprs], when supplied, preserves symbolic dimensions;
    otherwise concrete [shapes] are lifted to integer constants before
    trying the node's own symbolic shape. Handles Shrink, Permute, Flip,
    Expand, Pad, and Reshape.

    Raises [Assert_failure] if [view] is not a movement op. *)

(** {1:rangeify Rangeify passes} *)

val run_rangeify :
  ?shape_exprs:(Tolk_uop.Uop.t -> Tolk_uop.Uop.t list option) ->
  Tolk_uop.Uop.t ->
  shapes:(Tolk_uop.Uop.t -> int list option) ->
  indexing_context
(** [run_rangeify ?shape_exprs root ~shapes] builds the realize map, then
    walks the graph from roots to leaves assigning per-node ranges.
    [shape_exprs], if given, supplies symbolic axis sizes for range bounds.
    Returns a populated {!indexing_context} ready for
    {!apply_rangeify_pass}. *)

val apply_rangeify_pass :
  ?shape_exprs:(Tolk_uop.Uop.t -> Tolk_uop.Uop.t list option) ->
  indexing_context ->
  shapes:(Tolk_uop.Uop.t -> int list option) ->
  Tolk_uop.Uop.t ->
  Tolk_uop.Uop.t
(** [apply_rangeify_pass ?shape_exprs ctx ~shapes root] rewrites [root]
    bottom-up:

    {ul
    {- REDUCE keeps explicit range children.}
    {- PAD → WHERE guarded by the input ranges' validity.}
    {- Realized sources → STAGE + INDEX (or END for stores).}
    {- Direct buffer sources (Param, Buffer, Slice, …) → INDEX.}
    {- Movement ops → removed (their effect is in the range map).}} *)

(** {1:helpers Range helpers} *)

val get_idx : Tolk_uop.Uop.t -> Tolk_uop.Uop.t
(** [get_idx r] extracts the index value from a possibly-gated range.
    [where(valid, index, invalid)] yields [index]; [stack [r0; ...]]
    yields [stack [get_idx r0; ...]]; anything else yields [r]
    unchanged. *)

val get_valid : Tolk_uop.Uop.t -> Tolk_uop.Uop.t
(** [get_valid r] extracts the validity condition from a
    possibly-gated range.  [where(valid, _, invalid)] yields [valid];
    [stack [r0; ...]] yields [stack [get_valid r0; ...]]; [invalid]
    yields [false]; anything else yields [true]. *)
