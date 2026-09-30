(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Applying kernel optimisations.

    A kernel's axes are its ranges ({!Op.Range}) with a positive count, except
    serial loops and device ranges, listed by the role that
    {!Ops.Axis_type.position} orders, then by identity. An optimisation
    ({!Opt.t}) names an axis by its position in that list, and rewrites the
    kernel: a split replaces an axis [r] of size [n] by [r'] of size [n / a] and
    a new axis [s] of size [a], and every use of [r] by [r' * a + s], or
    [s * (n / a) + r'] from the top; a pad grows an axis to a multiple and
    guards its accesses; a swap exchanges two global axes' identities; a tensor
    core replaces a reduction of products by a matrix multiply-accumulate
    ({!Op.Wmma}). *)

(** {1:schedulers Schedulers} *)

(** Schedulers.

    A scheduler holds a kernel and the optimisations applied to it so far. It is
    mutable: applying an optimisation rewrites its kernel in place. *)
module Scheduler : sig
  type t
  (** The type for schedulers. *)

  val v : Ops.t -> Renderer.t -> t
  (** [v ast ren] schedules the kernel [ast], the sink of a kernel
      ({!Ops.kernel_info}), for [ren]. Its applied optimisations are those
      [ast]'s argument lists. *)

  val copy : t -> t
  (** [copy k] is a scheduler of [k]'s kernel and optimisations, which applying
      an optimisation to either does not change in the other. *)

  val ast : t -> Ops.t
  (** [ast k] is [k]'s kernel. *)

  val ren : t -> Renderer.t
  (** [ren k] is the renderer [k] optimises for. *)

  val applied_opts : t -> Opt.t list
  (** [applied_opts k] is the optimisations applied to [k], in order. *)

  (** {1:axes Axes} *)

  val rngs : t -> Ops.t list
  (** [rngs k] is [k]'s axes. *)

  val shape_len : t -> int
  (** [shape_len k] is the number of [k]'s axes. *)

  val full_shape : t -> Ops.sint list
  (** [full_shape k] is the size of each of [k]'s axes, simplified. *)

  val axis_types : t -> Ops.Axis_type.t list
  (** [axis_types k] is the role of each of [k]'s axes. *)

  val ranges_of : t -> Ops.Axis_type.t list -> Ops.t list
  (** [ranges_of k types] is [k]'s axes of a role in [types]. *)

  val axes_of : t -> Ops.Axis_type.t list -> int list
  (** [axes_of k types] is the positions of [k]'s axes of a role in [types]. *)

  val reduce_axes : t -> int list
  (** [reduce_axes k] is the positions of [k]'s axes that a reduction
      ({!Op.Reduce}) reduces over. *)

  val upcast_size : t -> Ops.sint
  (** [upcast_size k] is the product of the sizes of [k]'s
      {!Ops.Axis_type.Upcast} and {!Ops.Axis_type.Unroll} axes. *)

  val upcastable_dims : t -> int list
  (** [upcastable_dims k] is the positions of [k]'s global, local and weak axes
      of a known size greater than [1]. *)

  val unrollable_dims : t -> int list
  (** [unrollable_dims k] is the positions of [k]'s reduced local and reduce
      axes of a known size greater than [1]. *)

  val upcasted : t -> int
  (** [upcasted k] is the number of [k]'s {!Ops.Axis_type.Upcast} and
      {!Ops.Axis_type.Unroll} axes. *)

  val group_for_reduces : t -> int
  (** [group_for_reduces k] is the number of [k]'s reduced axes that are warp or
      local axes. *)

  val reduceops : t -> Ops.t list
  (** [reduceops k] is [k]'s reductions, in {!Ops.backward_slice}'s order. *)

  val reduceop : t -> Ops.t option
  (** [reduceop k] is the first of [reduceops k]. *)

  val bufs : t -> Ops.t list
  (** [bufs k] is [k]'s indexed accesses ({!Op.Index}), in the reverse of
      {!Ops.backward_slice}'s order. *)

  val colored_shape : t -> string
  (** [colored_shape k] is the size of each of [k]'s axes, right-aligned in four
      columns and coloured by its role; a weak axis is black if it is not an
      output axis and white if it cannot be made global. *)

  (** {1:rewriting Rewriting} *)

  val convert_loop_to_global : t -> unit
  (** [convert_loop_to_global k] makes global each weak output axis of [k] that
      every buffered value ({!Op.Stage}) runs inside, if [k]'s renderer has
      local indices. *)

  val shift_to :
    ?top:bool ->
    ?new_rng:Ops.t ->
    t ->
    Ops.t ->
    int ->
    Opt.target ->
    Ops.t * Ops.t
  (** [shift_to k r amount target] splits the axis [r] into [(r', s)]: [r'] of
      [r]'s size divided by [amount], and [s], of size [amount] and [target]'s
      role, a new range unless [new_rng] is given. [r] becomes
      [r' * amount + s], or [s * size r' + r'] if [top] (default [false]).

      Raises [Invalid_argument] if [amount] is not greater than [1], if [r]'s
      role is not among [split_targets target], or if [amount] does not divide
      [r]'s size. *)

  val apply_opt : ?append_opt:bool -> t -> Opt.t -> (Ops.t list, string) result
  (** [apply_opt k opt] applies [opt] to [k], records it among [k]'s applied
      optimisations if [append_opt] (default [true]), and is [Ok] the axes it
      made: a split's [[r'; s]] ({!shift_to}), a pad's grown axis, a tensor
      core's three axes [N], [M] and [K], and [[]] for a swap.

      It is [Error] with the reason if [opt] does not apply, leaving [k] as it
      was unless a tensor core's reduction is found ambiguous after its axes are
      split. It does not apply when its axis is out of range, a split's target
      cannot come from its axis's role, or exceeds what the renderer allows (32
      unrolled or 16 upcast lanes, locals only with local indices, the
      workgroup's shared memory), a pad's axis is not of a constant size or
      would more than quadruple, or no tensor core fits. Raises
      [Invalid_argument] if the shared memory a split needs depends on a
      symbolic size and cannot be decided. *)

  val get_optimized_ast : ?name_override:string -> t -> Ops.t
  (** [get_optimized_ast ~name_override k] is [k]'s kernel with the ranges its
      reductions and ends close flattened ({!Simplify.pm_flatten_range}),
      argument listing the applied optimisations, and tagged [1]. It is named
      [name_override], or else ["r"] for a kernel that reduces and ["E"]
      otherwise, followed by the sizes of its hardware indices and axes,
      coloured. *)
end

(** {1:opts Optimising} *)

val split_targets : Opt.target -> Ops.Axis_type.t list
(** [split_targets target] is the roles of the axes a split can move to
    [target]: global, local and weak axes to {!Opt.Upcast}, reduce and local
    axes to {!Opt.Unroll}, and global, weak and reduce axes to {!Opt.Local}. *)

val apply_opts :
  ?beam:(Scheduler.t -> Scheduler.t) ->
  hand_coded:(Scheduler.t -> Scheduler.t) ->
  Ops.t ->
  Renderer.t ->
  Ops.t
(** [apply_opts ~beam ~hand_coded ast ren] is the kernel [ast] optimised for
    [ren] ({!Scheduler.get_optimized_ast}), named by its argument unless that is
    ["test"]. After making weak output axes global
    ({!Scheduler.convert_loop_to_global}), it applies the optimisations that
    [ast]'s argument asks for, or else searches with [beam] if given, or else,
    unless the setting {!Helpers.noopt} is set, [ast] already had optimisations
    applied, or it buffers values ({!Op.Stage}), applies [hand_coded]. A kernel
    [ast] that is tagged is returned as it is.

    Raises [Invalid_argument] if an optimisation asked for does not apply, with
    the reason {!Scheduler.apply_opt} gives. *)
