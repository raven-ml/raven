(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Post-range kernel optimisation scheduler.

    Wraps a late {!Tolk_ir.Kernel.t} AST and applies optimisation passes
    (upcast, local, unroll, group, tensor core, padding, swap) that
    reshape the iteration space before expansion and devectorisation.

    See also {!Tc}, {!Heuristic}, {!Search}. *)

(** {1:types Types} *)

type t
(** Mutable scheduler state wrapping a kernel AST, renderer, and
    optimisation history. *)

exception Opt_error of string
(** Raised when an optimisation precondition fails. *)

(** {1:lifecycle Lifecycle} *)

val create : Tolk_ir.Kernel.t -> Renderer.t -> t
(** [create ast ren] is a fresh scheduler for [ast] with renderer
    [ren]. *)

val copy : t -> t
(** [copy t] is a shallow copy with independent mutable state. *)

val apply_opt :
  ?append_opt:bool -> t -> Tolk_ir.Kernel.Opt.t ->
  (Tolk_ir.Kernel.t * Tolk_ir.Kernel.t) option
(** [apply_opt t opt] applies [opt] to [t] and returns
    [Some (replaced_rng, new_rng)] for shift_to opts, the first two
    TC axes as a pair for TC opts, or [None] otherwise.
    [append_opt] defaults to [true].

    Raises {!Opt_error} on precondition failure. *)

val get_optimized_ast :
  ?name_override:string -> t -> Tolk_ir.Kernel.t
(** [get_optimized_ast t] finalises [t]: flattens ranges, generates a
    debug name, and returns the AST with updated kernel info. *)

(** {1:pipeline Pipeline} *)

val apply_opts :
  ?beam_search:(t -> t) ->
  ?hand_coded_optimizations:(t -> t) ->
  Tolk_ir.Kernel.t -> Renderer.t -> Tolk_ir.Kernel.t
(** [apply_opts ast ren] optimises [ast] for [ren].  Returns [ast]
    unchanged if already tagged.  Strategy callbacks break circular
    dependencies with {!Search} and {!Heuristic}. *)

val bufs_from_ast : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t list
(** [bufs_from_ast ast] is the Param nodes of [ast] sorted by index. *)

(** {1:accessors Accessors} *)

val ast : t -> Tolk_ir.Kernel.t
(** [ast t] is [t]'s current kernel AST. *)

val ren : t -> Renderer.t
(** [ren t] is [t]'s renderer. *)

val applied_opts : t -> Tolk_ir.Kernel.Opt.t list
(** [applied_opts t] is the opts applied so far. *)

val tensor_core : t -> Tc.t option
(** [tensor_core t] is the active tensor core, if any. *)

(** {1:shape Shape queries} *)

val rngs : t -> Tolk_ir.Kernel.t list
(** Active ranges sorted by (axis kind, axis number). *)

val shape_len : t -> int
(** Number of active ranges. *)

val full_shape : t -> Tolk_ir.Kernel.t list
(** Size node of each active range. *)

val axis_types : t -> Tolk_ir.Axis_kind.t list
(** Axis kind of each active range. *)

val shape_str : t -> string list
(** Labelled axis names (["g0"; "l0"; "r0"; …]). *)

val shape_str_to_axis : t -> string list -> int list
(** Map axis label names to indices.  Raises [Failure] if not found. *)

val axes_of : t -> Tolk_ir.Axis_kind.t list -> int list
(** Indices of ranges whose kind is in the given list. *)

val ranges_of : t -> Tolk_ir.Axis_kind.t list -> Tolk_ir.Kernel.t list
(** Ranges whose kind is in the given list. *)

val upcastable_dims : t -> int list
(** Global/Local/Loop axes with constant size > 1. *)

val unrollable_dims : t -> int list
(** Group_reduce/Reduce axes with constant size > 1. *)

val upcast_size : t -> int
(** Product of Upcast and Unroll shape sizes. *)

val output_shape : t -> Tolk_ir.Kernel.t list
(** {!full_shape} with reduce/unroll/group axes replaced by [1]. *)

val upcasted : t -> int
(** Number of Upcast and Unroll axes. *)

val group_for_reduces : t -> int
(** Number of Group_reduce axes. *)

(** {1:queries Queries} *)

val reduceop : t -> Tolk_ir.Kernel.t option
(** First Reduce node in the AST, if any. *)

val bufs : t -> Tolk_ir.Kernel.t list
(** Index nodes in the AST, reversed. *)

val colored_shape : t -> string
(** Debug string of range sizes with axis-kind labels. *)

val range_int_size : Tolk_ir.Kernel.t -> int
(** Constant integer size of a range node, or [0]. *)

val real_axis :
  t -> Tolk_ir.Kernel.Opt.t -> int option -> int
(** [real_axis t op axis] resolves [axis] for [op] to a range index.
    Returns [-1] when [axis] is [None] or [op] is TC.

    Raises {!Opt_error} on invalid axis. *)

(** {1:transforms Transforms} *)

val convert_loop_to_global : t -> unit
(** Promote eligible Loop ranges to Global. *)

val shift_to :
  ?top:bool -> ?input_new_rng:Tolk_ir.Kernel.t ->
  t -> Tolk_ir.Kernel.t -> int -> Tolk_ir.Axis_kind.t ->
  Tolk_ir.Kernel.t * Tolk_ir.Kernel.t
(** [shift_to t rng amount kind] splits [rng] by [amount].  Returns
    [(replaced_rng, new_rng)].  [top] defaults to [false].
    Raises [Failure] if [amount] does not divide the range. *)
