(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Schedules: from tensor graphs to the calls that realize them.

    A schedule is an {!Op.Linear} of calls, run in order: kernels, copies
    between devices and functions compiled on their own. This module turns the
    sink of the values to realize into a schedule: it scopes the graph into a
    call of its own, lowers the call's body to kernels
    ({!Prepare.prepare_rangeify}, {!Rangeify.get_kernel_graph}), orders them,
    and binds their parameters to storage. *)

(** {1:linear Linearizing} *)

val create_schedule : Ops.t -> Ops.t
(** [create_schedule sched_sink] is the {!Op.Linear} of the kernel calls of the
    kernel graph [sched_sink] ({!Rangeify.get_kernel_graph}), in an order that
    runs each kernel after those that write the storage it reads, and before
    those that overwrite the storage it reads. Ready kernels run in the order
    the graph reaches them. Each call's arguments are the storage of its reads
    and writes, or the views of it that move with the loops the call runs in. A
    loop of a call, an {!Op.End} of the call over {!Ops.Axis_type.Loop} ranges,
    stays such an end around the call; the end of a call over device ranges is
    the call, its ranges bound at launch.

    Raises [Invalid_argument] if the kernels' dependencies form a cycle, or if
    an effect is not a call, an end of a call, a store or an {!Op.After}. *)

val pm_flatten_linear : (unit, Ops.t) Ops.Pattern_matcher.t
(** [pm_flatten_linear] inlines each {!Op.Linear} among the sources of an
    {!Op.Linear} into it. *)

(** {1:views Views of storage} *)

val contiguous_mops_to_view : Ops.t -> Ops.t -> Ops.t option
(** [contiguous_mops_to_view c src] is, for the copy, stage or store [c] of the
    movements and bitcasts [src] of a buffer, [c] of a view of the buffer's
    elements ({!Prepare.contiguous_view}) when they are contiguous, and that
    view for any other [c]; [None] when they are not contiguous or [src]'s shape
    is symbolic. The view is the buffer shrunk to the elements and bitcast to
    [src]'s type, reshaped to [c]'s shape. On several devices, it is the view of
    each shard of a value sharded on one axis, sharded again, in place of [c].
*)

val is_store_after : Ops.t -> bool
(** [is_store_after u] is [true] iff [u] is an {!Op.After} that orders a store
    into storage: storage other than call-local storage ({!Op.Alloc}), or any
    storage whose first effect is a store. *)

(** {1:schedules Schedules} *)

val create_linear_with_vars :
  ?capturing:bool -> Ops.t -> Ops.t * (string * int) list
(** [create_linear_with_vars ~capturing big_sink] is the schedule that realizes
    the values of [big_sink], whose results are stored into buffers
    ({!Op.Buffer}), with the value of each variable it reads, by name:

    + the stores of [big_sink] become the body of a call compiled on its own,
      whose arguments are the buffers, the contiguous views of buffers that
      copies and stores read ({!contiguous_mops_to_view}), and the bound
      variables, each replaced in the body by a parameter; call-local storage is
      numbered by its order in each body, so equal graphs schedule alike;
    + each such call's body is scheduled ({!create_schedule}), once per body
      when the setting {!Helpers.scache} is [1] or more. From [2], the default,
      schedules are also kept on disk ({!Helpers.Diskcache}, table
      ["schedule_cache"]) for later processes: a schedule is read back for the
      same body, the same values of the settings and environment variables that
      shape a schedule ({!Helpers.split_reduceop},
      {!Helpers.max_kernel_buffers}, {!Helpers.ring}, {!Helpers.all2all},
      {!Helpers.allreduce_cast}, {!Helpers.allreduce_node_ndevs},
      {!Helpers.default_float}, {!Helpers.default_int} and the environment
      variables in {!Helpers.variables}) and the same sources of this library;
      an entry that does not read as a schedule is made anew and replaced;
    + the schedule's parameters are bound to the call's arguments, and each
      call-local storage to a new buffer on its device;
    + a kernel that only copies a buffer between devices, or to or from a disk,
      becomes a call of a store;
    + unless [capturing] (default [false]), the schedule's buffers are placed in
      arenas ({!Memory.memory_plan_rewrite}), except those the call's arguments
      name, whose contents outlive the schedule. A jit capturing the schedule
      plans it with the schedules it captures along with it.

    With the setting {!Helpers.spec} at [1] or more, [big_sink] and each body
    are checked against {!Spec.tensor}. With {!Helpers.debug} at [1] or more,
    each schedule of more than one call prints its size and time on standard
    output.

    Raises [Invalid_argument] if two bound variables of one name have different
    values, if a kernel that is not a copy accesses buffers on several devices,
    or as {!create_schedule} does. *)
