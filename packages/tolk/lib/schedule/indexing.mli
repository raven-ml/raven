(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Rangeify: tensor graph to indexed representation.

    Converts the high-level tensor graph (movement ops, REDUCE_AXIS,
    etc.) into an indexed representation with explicit RANGE loops,
    BUFFERIZE nodes, and INDEX operations.

    The algorithm runs in three phases:

    {ol
    {- {b Realize map.}  Decide which nodes need their own buffer
       (realization boundary).  See {!generate_realize_map}.}
    {- {b Range propagation.}  Walk the graph root-to-leaf, assigning
       one range expression per axis to every node.  Realized nodes get
       fresh ranges; others inherit or merge from consumers.  Movement
       ops transform ranges instead of persisting as nodes.
       See {!run_rangeify}.}
    {- {b Apply.}  Bottom-up graph rewrite: REDUCE_AXIS becomes REDUCE,
       PAD becomes WHERE, realized sources are wrapped in
       BUFFERIZE + INDEX or END, and movement ops are removed.
       See {!apply_rangeify_pass}.}} *)

(** {1:predicates Predicates} *)

val is_always_contiguous : Tolk_ir.Tensor.view -> bool
(** [is_always_contiguous v] is [true] for ops whose output is
    contiguous by definition (Contiguous, After, Copy, Buffer,
    Buffer_view, Const, Bind, Device, Mselect, Mstack, Param,
    Define_local, Call).  Their consumers can index directly
    without realization. *)

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
  range_map : (int, Tolk_ir.Tensor.t list * Tolk_ir.Tensor.t list) Hashtbl.t;
      (** Maps {!Tolk_ir.Tensor.tag} to [(input_ranges, output_ranges)]. *)
  mutable range_idx : int;
      (** Monotonic counter for fresh range axis indices. *)
}
(** Per-node state populated by {!run_rangeify}.  All maps are keyed
    by {!Tolk_ir.Tensor.tag}. *)

val create_context : unit -> indexing_context
(** [create_context ()] is a fresh, empty context. *)

val new_range :
  indexing_context -> int -> ?kind:Tolk_ir.Axis_kind.t -> unit ->
  Tolk_ir.Tensor.t
(** [new_range ctx size ?kind ()] is a fresh RANGE node over
    \[[0];[size-1]\] with axis kind [kind] (default {!Tolk_ir.Axis_kind.Loop}).
    Returns a constant [0] when [size] is [1]. *)

(** {1:simplify Symbolic simplification} *)

val simplify_tensor_expr : Tolk_ir.Tensor.t -> Tolk_ir.Tensor.t
(** [simplify_tensor_expr e] round-trips [e] through the Kernel IR,
    applies {!Tolk_ir.Symbolic.sym}, and converts back.  Only handles
    index-arithmetic nodes (Const, Range, Binary, Unary, Ternary,
    Invalid_index). *)

(** {1:movement Movement ops} *)

val apply_movement_op :
  shapes:(Tolk_ir.Tensor.t -> int list option) ->
  Tolk_ir.Tensor.view ->
  Tolk_ir.Tensor.t list ->
  Tolk_ir.Tensor.t list
(** [apply_movement_op ~shapes view rngs] transforms [rngs] (output
    ranges) through a movement op, producing the corresponding input
    ranges.  Handles Shrink, Permute, Flip, Expand, Pad, and Reshape.

    Raises [Assert_failure] if [view] is not a movement op. *)

(** {1:rangeify Rangeify passes} *)

val run_rangeify :
  Tolk_ir.Tensor.t ->
  shapes:(Tolk_ir.Tensor.t -> int list option) ->
  indexing_context
(** [run_rangeify root ~shapes] builds the realize map, then walks
    the graph from roots to leaves assigning per-node ranges.
    Returns a populated {!indexing_context} ready for
    {!apply_rangeify_pass}. *)

val apply_rangeify_pass :
  indexing_context ->
  devices:(Tolk_ir.Tensor.t -> Tolk_ir.Tensor.device option) ->
  Tolk_ir.Tensor.t ->
  Tolk_ir.Tensor.t
(** [apply_rangeify_pass ctx ~devices root] rewrites [root] bottom-up:

    {ul
    {- REDUCE_AXIS → REDUCE with explicit range children.}
    {- PAD → WHERE guarded by the input ranges' validity.}
    {- Realized sources → BUFFERIZE + INDEX (or END for stores).}
    {- Direct buffer sources (Param, Buffer_view, …) → INDEX.}
    {- Movement ops → removed (their effect is in the range map).}} *)

(** {1:helpers Range helpers} *)

val get_idx : Tolk_ir.Tensor.t -> Tolk_ir.Tensor.t
(** [get_idx r] extracts the index value from a possibly-gated range.
    [where(valid, index, invalid)] yields [index]; anything else
    yields [r] unchanged. *)

val get_valid : Tolk_ir.Tensor.t -> Tolk_ir.Tensor.t
(** [get_valid r] extracts the validity condition from a
    possibly-gated range.  [where(valid, _, invalid)] yields [valid];
    [invalid] yields [false]; anything else yields [true]. *)
