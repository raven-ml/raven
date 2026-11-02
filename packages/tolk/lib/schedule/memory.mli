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
    contents outlive the schedule, or lives on a disk. A call takes the buffers
    of its arguments, an argument passing itself if it is a buffer and the
    buffers it views if it is a view ({!Op.Shrink}, {!Op.Bitcast},
    {!Op.After}). A buffer lives from the first call that takes it to the
    last, a loop around calls ({!Op.End} or {!Op.Backedge}) counting as one
    call that takes the buffers of each of its calls and a back edge's flag.
    The buffers of copies, the calls whose body is an {!Op.Store}, are planned
    apart from the others on each device, so that planning adds no dependency
    between copies and kernels. A buffer's bytes are free from the call after its last: an arena
    is one allocation, and the executor runs the work that writes an
    allocation after every earlier use of it, a copy's included.

    Each device and kind of buffer has one arena, an {!Op.Buffer} of
    {!Dtype.Int8} as large as its buffers need at once. A buffer's size and
    phase ({!Ops.type-param_arg}) together are rounded up to 256 bytes, and its
    place is found by a {!Support_memory.Tlsf_allocator}, the buffers ending at
    a call freed before those starting there are placed. Each planned buffer is
    replaced wherever [linear] reaches it outside call bodies, inside views
    included, by the bytes of its arena from its phase past its place
    ({!Shape.shrink}), viewed as its type ({!Ops.bitcast}), so that it keeps
    its phase. New arenas take the next slots ({!Ops.unique_num}).

    [linear] is itself if {!Setting.no_memory_planner} is set or no buffer is
    planned. With {!Setting.debug} at [1] or more, the memory saved is printed
    on standard output. *)

val taken : Ops.t -> Ops.t list
(** [taken si] is the storage that the entry [si] of a schedule takes, each
    once, in the order its calls take it: the buffer, call-local storage or
    parameter ({!Op.Buffer}, {!Op.Alloc}, {!Op.Param}) that each argument of its
    calls names or views ({!Ops.storage_base}). A loop around calls takes the
    storage of each of its calls and its back edge's flag, as
    {!memory_plan_rewrite} counts a buffer's uses. *)
