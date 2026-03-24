(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Kernel optimization scheduler.

    This is the core kernel optimization engine. The {!type-t} type manages
    optimization state (current AST, renderer target, applied opts) and
    {!apply_opt} dispatches individual optimization actions.

    *)

(** {1:errors Errors} *)

exception Opt_error of string
(** Raised when an optimization is invalid or cannot be applied. *)

(** {1:types Types} *)

type t
(** Mutable optimization scheduler state. *)

(** {1:construction Construction} *)

val create : Tolk_ir.Kernel.t -> Renderer.t -> t
(** [create ast ren] builds a scheduler from a Sink-rooted kernel AST and a
    target renderer. Reads [kernel_info] from [ast] to recover
    [dont_use_locals] and [applied_opts]. *)

val copy : t -> t
(** [copy t] returns a shallow copy with an independent [opt_range] counter
    and a fresh [opt_range] iterator derived from the current AST. *)

(** {1:state State queries} *)

val ast : t -> Tolk_ir.Kernel.t
(** [ast t] is the current kernel AST root. *)

val ren : t -> Renderer.t
(** [ren t] is the target renderer. *)

val applied_opts : t -> Tolk_ir.Kernel.Opt.t list
(** [applied_opts t] is the list of optimizations applied so far. *)

val rngs : t -> Tolk_ir.Kernel.t list
(** [rngs t] returns all Range nodes with size > 1, sorted by
    [(axis_to_pos kind, axis)]. *)

val shape_len : t -> int
(** [shape_len t] is [List.length (rngs t)]. *)

val full_shape : t -> Tolk_ir.Kernel.t list
(** [full_shape t] returns the size node of each range. *)

val axis_types : t -> Tolk_ir.Axis_kind.t list
(** [axis_types t] returns the kind of each range. *)

val tensor_core : t -> Renderer.tensor_core option
(** [tensor_core t] is the active tensor core configuration, if any. *)

val shape_str : t -> string list
(** [shape_str t] returns axis labels with per-kind counters
    (e.g. [["g0"; "g1"; "l0"; "R0"; "r0"; "u0"]]). *)

val shape_str_to_axis : t -> string list -> int list
(** [shape_str_to_axis t nms] finds the index of each named axis in
    [shape_str t]. *)

(** {1:helpers Helpers} *)

val ranges_of : t -> Tolk_ir.Axis_kind.t list -> Tolk_ir.Kernel.t list
(** [ranges_of t kinds] returns ranges whose kind is in [kinds]. *)

val axes_of : t -> Tolk_ir.Axis_kind.t list -> int list
(** [axes_of t kinds] returns indices of ranges whose kind is in [kinds]. *)

val upcast_size : t -> int
(** [upcast_size t] is the product of UPCAST and UNROLL shape sizes. *)

val upcastable_dims : t -> int list
(** [upcastable_dims t] returns non-reduce axes with const shape > 1. *)

val unrollable_dims : t -> int list
(** [unrollable_dims t] returns GROUP_REDUCE/REDUCE axes with const shape > 1. *)

val real_axis : t -> Tolk_ir.Kernel.Opt.t -> int
(** [real_axis t opt] maps the optimization's axis to the actual range index. *)

(** {1:transforms Transforms} *)

val convert_loop_to_global : t -> unit
(** [convert_loop_to_global t] replaces globalizable LOOP ranges with GLOBAL
    kind. No-op if the renderer does not support locals. *)

val shift_to :
  ?top:bool ->
  ?input_new_rng:Tolk_ir.Kernel.t ->
  ?tags:int Tolk_ir.Kernel.Ref_tbl.t ->
  t ->
  Tolk_ir.Kernel.t ->
  int ->
  Tolk_ir.Axis_kind.t ->
  Tolk_ir.Kernel.t * Tolk_ir.Kernel.t
(** [shift_to ?top ?input_new_rng ?tags t rng amount new_kind] splits [rng]
    into two new ranges: one of [amount] elements with [new_kind], and the
    remainder.  Returns [(replaced_rng, new_rng)].

    When [tags] is provided, node tags are propagated through the underlying
    {!Tolk_ir.Kernel.substitute} call.

    Raises {!Opt_error} if [amount] does not divide the range size. *)

val apply_opt :
  ?append_opt:bool ->
  t ->
  Tolk_ir.Kernel.Opt.t ->
  (Tolk_ir.Kernel.t * Tolk_ir.Kernel.t) option
(** [apply_opt ?append_opt t opt] applies a single optimization to the
    scheduler state. Returns [Some (replaced, new_rng)] for shift-based opts,
    [None] otherwise.

    When [append_opt] is [true] (default), the opt is appended to
    [applied_opts]. *)

val get_optimized_ast :
  ?name_override:string -> t -> Tolk_ir.Kernel.t
(** [get_optimized_ast ?name_override t] finalizes the kernel: flattens ranges
    and returns the AST with updated kernel_info. *)

(** {1:utilities Utilities} *)

val bufs_from_ast : Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t list
(** [bufs_from_ast ast] extracts sorted Param nodes from the AST.

    Returns raw Param nodes rather than device buffers; the caller (Pipeline)
    constructs device buffers, keeping Postrange device-agnostic. *)

val reduceops : t -> Tolk_ir.Kernel.t list
(** [reduceops t] returns all Reduce nodes in the backward slice. *)

val reduceop : t -> Tolk_ir.Kernel.t option
(** [reduceop t] returns the first reduce node, if any. *)

val bufs : t -> Tolk_ir.Kernel.t list
(** [bufs t] returns INDEX nodes in reverse toposort order. *)

val output_shape : t -> Tolk_ir.Kernel.t list
(** [output_shape t] is [full_shape t] with REDUCE/UNROLL/GROUP_REDUCE sizes
    replaced by [1]. *)

val upcasted : t -> int
(** [upcasted t] is the count of UPCAST and UNROLL axes. *)

val group_for_reduces : t -> int
(** [group_for_reduces t] is the count of GROUP_REDUCE axes. *)

(** {1:display Display} *)

val colors : t -> string list
(** [colors t] returns a color name for each range, for debug display. *)

val colored_shape : t -> string
(** [colored_shape t] returns a debug string of range shapes with colors. *)

(** {1:dispatch Optimization dispatch} *)

val apply_opts :
  ?beam_search:(t -> t) ->
  ?hand_coded_optimizations:(t -> t) ->
  Tolk_ir.Kernel.t ->
  Renderer.t ->
  Tolk_ir.Kernel.t
(** [apply_opts ?beam_search ?hand_coded_optimizations ast ren] is the
    top-level optimization dispatch.

    Returns [ast] unchanged if the kernel is already optimized (non-empty
    [axis_kinds] in kernel_info).

    1. If [kernel_info.opts_to_apply] is set, applies those opts in order.
    2. Else if [beam_search] is [Some f], calls [f] on the scheduler.
    3. Else if [hand_coded_optimizations] is [Some f], no opts applied yet,
       and no BUFFERIZE nodes in the AST, calls [f].
    4. Otherwise returns the unoptimized AST.

    The strategy closures are passed by the caller (Pipeline) to break the
    circular module dependency. *)
