(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Applying kernel optimisations.

    A kernel's axes are its ranges ({!Op.Range}) with a positive count, except
    serial loops, listed by argument ({!Ops.compare_range_arg}), the weak and upcast ranges that a reduction
    closes last. A range's role in a reduction is that a reduction
    ({!Op.Reduce}) closes it: its axis is a reduce axis, and an upcast of it
    unrolls the reduction into the thread's lanes. An optimisation
    ({!Opt.t}) names an axis by its position in that list, and rewrites the
    kernel: a split replaces an axis [r] of size [n] by [r'] of size [n / a] and
    a new axis [s] of size [a], and every use of [r] by [r' * a + s], or
    [s * (n / a) + r'] from the top; a pad grows an axis to a multiple and
    guards every access along it, an access through storage ordered after the
    axis included, and an integer division whose divisor reads the axis
    divides by [1] on the padding, whose loads read zeros; a swap exchanges two
    global axes' identities; a tensor core replaces a reduction of products by
    a matrix multiply-accumulate ({!Op.Wmma}). *)

(** {1:schedulers Schedulers} *)

(** Schedulers.

    A scheduler holds a kernel and the optimisations applied to it so far. It is
    mutable: applying an optimisation rewrites its kernel in place. *)
module Scheduler : sig
  type t
  (** The type for schedulers. *)

  val v : Ops.t -> Renderer.t -> t
  (** [v ast ren] schedules the kernel [ast], the sink of a kernel
      ({!Ops.type-kernel_info}), for [ren], with the ranges its reductions and
      ends close flattened ({!Simplify.pm_flatten_range}). Its applied
      optimisations are those [ast]'s argument lists. *)

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

  val reduce_ranges : t -> Ops.t list
  (** [reduce_ranges k] is the ranges [k]'s reductions close, each once. *)

  val axes_of : ?reduce:bool -> t -> Ops.Axis_type.t list -> int list
  (** [axes_of ~reduce k types] is the positions of [k]'s axes of a role in
      [types], or of any role if [types] is [[]], that are reduce axes if
      [reduce] is [true] and are not if it is [false]. Without [reduce], any. *)

  val upcast_size : t -> Ops.sint
  (** [upcast_size k] is the product of the sizes of [k]'s
      {!Ops.Axis_type.Upcast} axes. *)

  val read_by_bounds : t -> Ops.t list
  (** [read_by_bounds k] is the ranges that the bound of one of [k]'s ranges
      reads, as [i] in a loop [j < x[i]]. A loop has one count of trips, so
      none of them can become a thread's lanes. *)

  val opaque : t -> Ops.t list
  (** [opaque k] is [k]'s opaque operations, but those of the bodies it calls:
      its calls ({!Ops.call}) and its code in the target's language
      ({!Op.Custom}). Each runs once for each point of the ranges its sources
      read, and takes one pointer or value for each source. *)

  val read_by_opaque : t -> Ops.t list
  (** [read_by_opaque k] is the ranges that a source of one of [k]'s {!opaque}
      operations, or of one of its expressions in the target's language
      ({!Op.Customi}), reads, as [i] in [f(&x[i])] or [popcount(x[i])]. Each
      takes one pointer or value for each source, so none of them can become
      a thread's lanes, which would pass it a vector, and none can be padded,
      which would run it on a source of no element. *)

  val unread_by_opaque : t -> Ops.t list
  (** [unread_by_opaque k] is [k]'s ranges that the sources of one of its
      {!opaque} operations do not read, as [j] for [f(&x[i])] beside a loop
      over [j]. Every thread of a launch runs each such operation, so none of
      them can take threads, which would run it once in each. *)

  val upcastable_dims : ?reduce:bool -> t -> int list
  (** [upcastable_dims ~reduce k] is the positions of [k]'s global, local and
      weak axes of a known size that no range's bound reads
      ({!read_by_bounds}) and no opaque operation reads ({!read_by_opaque}), that
      are reduce axes if [reduce] is [true] and are not if it is [false]
      (default). *)

  val group_for_reduces : t -> int
  (** [group_for_reduces k] is the number of [k]'s reduce axes that are warp or
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
      columns and coloured by its role ({!Ops.Axis_type.color}). *)

  (** {1:rewriting Rewriting} *)

  val convert_loop_to_global : t -> unit
  (** [convert_loop_to_global k] makes global each weak range that an end of
      [k]'s sink closes and no reduction does, whose bound a launch computes
      ({!Ops.known_at_launch}), that every buffered value ({!Op.Stage}) runs
      inside and every opaque operation reads ({!unread_by_opaque}), if [k]'s
      renderer has local indices. *)

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
      [r' * amount + s], or [s * size r' + r'] if [top] (default [false]), and
      the ranges reductions and ends close stay flat.

      Raises [Invalid_argument] if [amount] is not greater than [1], if [r]'s
      role is not among [split_targets target], if [r]'s trips are not known at
      launch ({!Ops.known_at_launch}), or if [amount] does not divide [r]'s
      size. *)

  val apply_opt : ?append_opt:bool -> t -> Opt.t -> (Ops.t list, string) result
  (** [apply_opt k opt] applies [opt] to [k], records it among [k]'s applied
      optimisations if [append_opt] (default [true]), and is [Ok] the axes it
      made: a split's [[r'; s]] ({!shift_to}), a pad's grown axis, a tensor
      core's three axes [N], [M] and [K], and [[]] for a swap.

      It is [Error] with the reason if [opt] does not apply, leaving [k] as it
      was unless a tensor core's reduction is found ambiguous after its axes are
      split. It does not apply when its axis is out of range, a split's target
      cannot come from its axis's role or its axis's trips are not known at
      launch ({!Ops.known_at_launch}), as [j < 4 (i + 1)]'s, or a split
      exceeds what the renderer allows (32 lanes of a reduce axis or 16 of
      another, locals only with local indices, the workgroup's shared memory),
      an upcast of an axis a range's bound reads ({!read_by_bounds}) or an
      opaque operation reads ({!read_by_opaque}), a local split of an axis an
      opaque operation does not read ({!unread_by_opaque}), a local split of a
      reduce axis where a range's bound reads an axis or whose reduction runs
      inside an unrolled reduce axis, a pad's axis is not
      of a constant size, would more than quadruple, or is read by an opaque
      operation, or no tensor core fits, as none does a kernel where a range's
      bound reads an axis, one with an opaque operation, or one where an
      expression in the target's language reads an axis. Raises
      [Invalid_argument] if the shared memory a split needs depends on a
      symbolic size and cannot be decided. *)

  val get_optimized_ast : ?name_override:string -> t -> Ops.t
  (** [get_optimized_ast ~name_override k] is [k]'s kernel with its argument
      listing the applied optimisations and marking it optimised
      ([optimized]). It is named
      [name_override], or else ["r"] for a kernel that reduces and ["E"]
      otherwise, followed by the sizes of its hardware indices and axes, each
      after an underscore. The name holds no colour. *)
end

(** {1:opts Optimising} *)

val split_targets : Opt.target -> Ops.Axis_type.t list
(** [split_targets target] is the roles of the axes a split can move to
    [target]: global, local and weak axes to {!Opt.Upcast}, and global and weak
    axes to {!Opt.Local}. *)

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
    unless the setting {!Setting.noopt} is set, [ast] already had optimisations
    applied, or it buffers values ({!Op.Stage}), applies [hand_coded]. A kernel
    one of whose loops, or of the loops of a body it calls, has a bound that
    loads from memory ({!Ops.bound_loads}) is not searched: its time depends on
    what memory holds, which a search's samples do not, so it takes
    [hand_coded] as without [beam]. A kernel whose builder fixed its threads is
    neither searched nor given [hand_coded]: it, or a body it calls, holds
    workgroup or register storage ({!Dtype.Local}, {!Dtype.Reg}), a barrier, a
    matrix multiply-accumulate or custom code, made for the threads and lanes
    its builder chose, which splitting its axes would change; the
    optimisations it asks for still apply. A kernel [ast] already optimised
    ([optimized]) is returned as it is.

    Before any optimisation, it refuses a kernel whose store breaks a weak
    range's promise visibly ({!Ops.Axis_type.Weak}): a store to global or
    workgroup memory that names fewer cells than it has points along one of
    its weak ranges, or along all of them together, where neither its validity
    nor its gate reads them, as [C[r // 8] += 1] over a weak [r < 16] or
    [C[r + s] += 1] over weak [r] and [s]. An offset's cells are counted as the
    multiples of the greatest common divisor of the constant factors of its
    terms that read the ranges, between those terms' least and greatest sums,
    over every value of the kernel's variables. Only a range whose bound is a
    constant or a variable is counted, its points at the bound's greatest
    value; a store inside a range of no trip is not counted, and a kernel
    already optimised is not checked.

    Raises [Invalid_argument] if a store breaks a weak range's promise
    visibly, naming the store and the ranges, or if an optimisation asked for
    does not apply, with the reason {!Scheduler.apply_opt} gives. *)
