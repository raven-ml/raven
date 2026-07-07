(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Tensor schedule linearization.

    This module mirrors [tinygrad.schedule]: it turns tensor-level
    {!Tolk_uop.Ops.Sink} graphs into program-level {!Tolk_uop.Ops.Linear}
    schedules and extracts the runtime variable values used by those
    schedules.

    The linearized schedule is executed by {!Realize.run_linear}; the buffer
    memory planner is embedded here as {!create_linear_with_vars}'s optional
    UOp rewrite. *)

val fresh_internal_buffer_slot : unit -> int
(** [fresh_internal_buffer_slot ()] draws the next slot for
    schedule-internal buffers from this process's counter. Slots are
    negative and strictly decreasing, disjoint from the non-negative
    user-facing buffer slots.

    Buffer nodes hash-cons on their slot, so a slot identifies a buffer:
    reusing one collapses two distinct buffers onto a single node,
    silently aliasing their storage. In particular, a graph imported from
    another process (see {!Tolk_uop.Uop.import}) may carry internal slots
    that collide with slots minted here; renumber each imported internal
    buffer with a slot from this counter before executing the graph. *)

val create_schedule : Tolk_uop.Uop.t -> Tolk_uop.Uop.t
(** [create_schedule sched_sink] builds the kernel dependency graph and
    topologically sorts it into a {!Tolk_uop.Ops.Linear} node.

    [sched_sink] is the output of {!Rangeify.get_kernel_graph}. For each
    reachable {!Tolk_uop.Ops.After}, children after [src.(0)] are partitioned
    like tinygrad: {!Tolk_uop.Ops.Call} and {!Tolk_uop.Ops.End} children are
    kernels, {!Tolk_uop.Ops.After} children are dependencies, and
    {!Tolk_uop.Ops.Store} children are ignored.

    Raises [Invalid_argument] if a kernel dependency is not an [After],
    [Buffer], [Param], [Mselect], [Mstack], or [Bind] after source
    unwrapping. *)

val lower_sink_to_linear :
  get_kernel_graph:(Tolk_uop.Uop.t -> Tolk_uop.Uop.t) ->
  Tolk_uop.Uop.t ->
  Tolk_uop.Uop.t option
(** [lower_sink_to_linear ~get_kernel_graph sink] lowers a tensor-level
    [Sink] into a [Linear] node via [get_kernel_graph] and
    {!create_schedule}.

    Returns [None] if [sink] is not a tensor-level sink (e.g. it
    already carries [kernel_info]). Results are cached by
    {!Tolk_uop.Uop.semantic_key} when [SCACHE] is enabled. *)

val memory_plan_rewrite :
  Tolk_uop.Uop.t -> Tolk_uop.Uop.t list -> Tolk_uop.Uop.t
(** [memory_plan_rewrite linear held_bufs] replaces the internal
    {!Tolk_uop.Ops.Buffer} nodes of the {!Tolk_uop.Ops.Linear} [linear] with
    {!Tolk_uop.Ops.Slice} views into per-device arena buffers sized by
    liveness analysis. Buffers in [held_bufs] (compared by physical identity)
    keep their identity and their own allocation: hold any buffer that
    outlives the schedule, such as external inputs and outputs.

    Disabled when the [NO_MEMORY_PLANNER] environment variable is set. *)

val create_linear_with_vars :
  get_kernel_graph:(Tolk_uop.Uop.t -> Tolk_uop.Uop.t) ->
  Tolk_uop.Uop.t ->
  Tolk_uop.Uop.t * (string * int) list
(** [create_linear_with_vars ~get_kernel_graph big_sink] runs the
    schedule creation pipeline on [big_sink] and returns the
    linearized schedule plus the values bound to runtime variables.

    [big_sink] is a raw {!Tolk_uop.Ops.Sink} or a {!Tolk_uop.Ops.Call}
    from allocations. Nested calls whose body is a {!Tolk_uop.Ops.Linear}
    are resolved by substituting parameter slots with call arguments
    and flattening the resulting linear schedule. Only binds for variables
    referenced by scheduled kernel bodies are returned.

    When {!Realize.capturing} is non-empty (and the [CAPTURING] environment
    variable is not [0]), the schedule is handed to the head capturer
    unplanned and an empty {!Tolk_uop.Ops.Linear} is returned: there is
    nothing to execute, and the capturer runs {!memory_plan_rewrite} once
    over the combined captured schedule. Otherwise the schedule is
    memory-planned with the graph's own buffer arguments held.

    Raises [Invalid_argument] if duplicate binds disagree or if the resolved
    result is not a {!Tolk_uop.Ops.Linear} node. *)
