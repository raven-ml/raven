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
(** [memory_plan_rewrite ~held_bufs linear] is the schedule [linear],
    an {!Op.Linear} of calls run in order, with its buffers placed in arenas.

    A buffer is planned unless it is among [held_bufs] (default [[]]), whose
    contents outlive the schedule, or lives on a disk. A call takes the buffers
    of its arguments, an argument passing itself if it is a buffer and the
    buffers it views if it is a view ({!Op.Shrink}, {!Op.Bitcast},
    {!Op.After}). A buffer lives from the first call that takes it to the
    last, the calls of a loop around calls ({!Op.End} or {!Op.Backedge}) in
    the order of one trip. A buffer that only the calls of one loop take, as
    call-local storage of the loop's call is, lives within a trip, and the
    loop holds its trip's peak for its whole run. That holds only where no
    trip reads such a buffer that an earlier trip left, which [linear] must
    ensure and the plan does not check: a loop's state passes between trips
    in buffers its calls take beside a call outside the loop. A buffer that calls inside and outside
    a loop take lives for the whole loop, as if the loop were one call that
    takes it, and so does a back edge's flag. A buffer's bytes are free from
    the call after its last: an arena is one allocation, and the executor
    runs the work that writes an allocation after every earlier use of it, a
    copy's included.

    Each device has one arena, which its copies' buffers, those of the calls
    whose body is an {!Op.Store}, share with its kernels': the executor runs a
    device's work in order, so on one device planning copies apart would order
    nothing more, and two arenas, held at once, would hold both peaks. Across
    devices the executor orders work by allocation: where a device whose work
    the host runs reuses bytes a GPU's copy touched, its work, and the run
    that submits it, wait for the GPU's earlier work. The arena is an
    {!Op.Buffer} of {!Dtype.Int8} as large as its placed buffers reach. A
    buffer's size and phase ({!Ops.type-param_arg}) together are rounded up to
    256 bytes. Buffers are placed largest first, ties in the order they first
    appear: each at the start of the smallest gap that holds it between the
    buffers already placed whose lives meet its, else past the last byte of
    those, two lives meeting where one's first call is at or before the
    other's last; placing keeps the placed buffers sorted by offset, so a lane
    of [b] buffers costs [O(b²)]. Placed so, an arena is often exactly the
    bytes its buffers take at once, the least any placement needs, though
    never promised to be. Each planned buffer is replaced wherever [linear]
    reaches it outside call bodies, inside views
    included, by the bytes of its arena from its phase past its place
    ({!Shape.shrink}), viewed as its type ({!Ops.bitcast}), so that it keeps
    its phase. New arenas take the next slots ({!Ops.unique_num}).

    [linear] is itself if {!Setting.no_memory_planner} is set or no buffer is
    planned. With {!Setting.debug} at [1] or more, the memory saved is printed
    on standard output. *)

val buffers : Ops.t -> Ops.t list
(** [buffers si] is the buffers ({!Op.Buffer}) that the entry [si] of a
    schedule takes as {!memory_plan_rewrite} counts its uses: each that an
    argument of its calls is or views, a loop's calls' and its back edge's
    flag included, once each, in order. *)

val bytes : Ops.t -> int
(** [bytes b] is the bytes the buffer [b] takes of its arena once planned:
    its size and phase, rounded up to 256. *)

val taken : Ops.t -> Ops.t list
(** [taken si] is the storage that the entry [si] of a schedule takes, each
    once, in the order its calls take it: the buffer, call-local storage or
    parameter ({!Op.Buffer}, {!Op.Alloc}, {!Op.Param}) that each argument of its
    calls names or views ({!Ops.storage_base}). A loop around calls takes the
    storage of each of its calls and its back edge's flag, as
    {!memory_plan_rewrite} counts a buffer's uses. *)
