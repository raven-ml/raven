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
    contiguous by definition (Stage, After, Copy, Buffer,
    Const, Mselect, Mstack, Param, Alloc, Load, Call).  Their consumers can index directly without
    realization. *)

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
  non_removable : (int, unit) Hashtbl.t;
      (** Sources of a custom kernel call: their stage must stay a buffer,
          since the kernel addresses it by slot. *)
  range_map : (int, Tolk_uop.Uop.t list * Tolk_uop.Uop.t list) Hashtbl.t;
      (** Maps {!Tolk_uop.Uop.tag} to [(input_ranges, output_ranges)].
          Only nodes of the graph {!run_rangeify} walked are present; a node
          rebuilt by {!apply_rangeify_pass} is deliberately absent. *)
  buf_cache : (int, Tolk_uop.Uop.t list) Hashtbl.t;
      (** Memoised reachable buffer-boundary nodes per node tag.  Shared
          across buffer-limiting rewrites so a subtree's reachable set is
          computed once. *)
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
    \[[0];[size-1]\] with axis kind [kind] (default {!Tolk_uop.Axis_type.Weak}).
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
  Tolk_uop.Uop.t -> Tolk_uop.Uop.t list -> Tolk_uop.Uop.t list
(** [apply_movement_op node rngs] transforms output ranges [rngs] through
    [node], using its symbolic shape to produce input ranges. Handles Shrink,
    Permute, Flip, Expand, Pad, and Reshape.

    Raises [Invalid_argument] if [node] is not a movement operation. *)

(** {1:rangeify Rangeify passes} *)

val run_rangeify : Tolk_uop.Uop.t -> indexing_context
(** [run_rangeify root] builds the realize map, then walks the graph from roots
    to leaves assigning ranges from each node's symbolic shape. Returns a
    context ready for {!apply_rangeify_pass}. *)

val apply_rangeify_pass :
  indexing_context -> Tolk_uop.Uop.t -> Tolk_uop.Uop.t
(** [apply_rangeify_pass ctx root] rewrites [root] bottom-up using the ranges
    recorded in [ctx] by {!run_rangeify}:

    {ul
    {- REDUCE keeps explicit range children.}
    {- PAD → WHERE guarded by the input ranges' validity.}
    {- STACK → nested WHERE selecting a source on the leading range.}
    {- Realized sources → STAGE + INDEX (or END for stores).}
    {- Direct buffer sources (Param, Buffer, Alloc, …) → INDEX.}
    {- Movement ops → removed (their effect is in the range map).}} *)

(** {1:helpers Range helpers} *)

val movement_ops : Tolk_uop.Uop.t -> Tolk_uop.Uop.t option
(** [movement_ops u] pushes movement through INDEX, AFTER and END. *)

val contiguous_view : Tolk_uop.Uop.t -> (Tolk_uop.Uop.t * int) option
(** [contiguous_view u] is the graph anchor and byte offset of a proven
    contiguous view. The anchor retains pending effects and may be a bitcast.
    It need not own allocated storage. Returns [None] when a constant offset
    cannot be proved, or the device does not support views.

    Raises [Invalid_argument] if the byte offset does not fit a host integer. *)

val storage_window : Tolk_uop.Uop.t -> (Tolk_uop.Uop.t * int) option
(** [storage_window u] is [contiguous_view u] when its anchor names storage:
    a BUFFER, ALLOC, PARAM, MSELECT, MSTACK or zero-coordinate STAGE, possibly
    through bitcasts and pending effects. A STAGE names its own future
    allocation. Arithmetic alone does not name storage. *)
