(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Memory planning.

    A schedule allocates a buffer for each intermediate value, though most are
    needed by a few of its calls only. Planning places the buffers whose
    lifetimes do not overlap at shared addresses of a few large buffers, the
    arenas, so the schedule needs less memory. *)

val memory_plan_rewrite : ?held_bufs:Ops.t list -> Ops.t -> Ops.t
(** [memory_plan_rewrite ~held_bufs linear] is the schedule [linear], an
    {!Op.Linear} of calls run in order, with its buffers placed in arenas.

    A buffer is planned unless it is among [held_bufs] (default [[]]), whose
    contents outlive the schedule, lives on a disk, or is reached through a view
    ({!Op.Shrink}, {!Op.Bitcast}, {!Op.After}) by some call. A call takes the
    buffers of its arguments, an argument passing itself if it is a buffer and
    the buffers of each of its sources if it is an {!Op.Mselect} or an
    {!Op.Mstack}. A buffer lives from the first call that takes it to the last,
    a loop around calls ({!Op.End} or {!Op.Backedge}) counting as one call that
    takes the buffers of each of its calls and a back edge's flag. The buffers of copies, the calls
    whose body is an {!Op.Store}, are planned apart from the others on each
    device, so that planning adds no dependency between copies and kernels, and
    each lives for as many calls again after its last, so that a copy is not
    overwritten while it may still run.

    Each device and kind of buffer has one arena, an {!Op.Buffer} of
    {!Dtype.Int8} as large as its buffers need at once. A buffer's size and
    phase ({!Ops.type-param_arg}) together are rounded up to 256 bytes, and its
    place is found by a {!Support_memory.Tlsf_allocator}, the buffers ending at
    a call freed before those starting there are placed. Each planned buffer is
    replaced wherever [linear] reaches it outside call bodies by the bytes of
    its arena from its phase past its place ({!Shape.shrink}), viewed as its
    type ({!Ops.bitcast}), so that it keeps its phase. New arenas take
    the next slots ({!Ops.unique_num}).

    [linear] is itself if {!Setting.no_memory_planner} is set or no buffer is
    planned. With {!Setting.debug} at [1] or more, the memory saved is printed
    on standard output. *)
