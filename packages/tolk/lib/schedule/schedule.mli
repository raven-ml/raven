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
(** [create_schedule sched_sink] is the {!Op.Linear} of the kernel
    calls of the kernel graph [sched_sink] ({!Rangeify.get_kernel_graph}), in
    an order that runs each kernel after those that write the storage it
    reads, and before those that overwrite the storage it reads. The order
    depends on the graph alone, the order of a node's sources included, and
    on nothing the process built before it. A kernel's number is its place in
    the order {!Ops.toposort} of the graph meets it, each node after its
    sources in their order. Of the ready kernels, one that frees storage,
    being the last to take storage a kernel wrote, runs first: the lowest
    numbered of those that make no storage, else of those that on no device
    make more storage than they free, bytes counted unrounded, storage living
    from the first kernel that takes it to the last ({!Memory.taken}).
    Otherwise the lowest numbered runs. So where the walk reaches a value's
    readers right after the value, it is made just before them, and the
    pieces of one value are each read before the next is made. The order is
    this rule, and no promise of the least storage. Each call's arguments
    are the storage of its reads and writes, or the views of it that move with
    the loops the call runs in. A loop of a call, an {!Op.End} of the call over {!Ops.Axis_type.Loop} ranges,
    stays such an end around the call, or around an {!Op.Linear} of the copies
    it makes as each trip starts and the call, and a back edge of a call
    ({!Ops.backedge}) stays a back edge, its condition the storage it reads.

    Each storage's states form a chain: an {!Op.After} of a state lists the
    kernels of the one write over it, which make the next state; a kernel
    listed again by an {!Op.After} of a later state of the same storage is no
    part of that write. A kernel reads the states its arguments name, a join
    ({!Ops.is_join}) being the state under it and a view of a storage
    ({!Ops.whole}) being that storage's state; through an argument it writes,
    it reads the state its write is over, and a view it stores through names
    the storage and no state. It runs after the kernels that write the states
    it reads and those the joins it reads through wait for, after the kernels
    of the write that made the state it writes over, and before every other
    kernel of each write over a state it reads, whichever elements each
    touches. A call that writes a state through an argument its body writes
    ({!Shape.written_params}) and reads it through another reads it as it was
    before the call. The kernels of a write marked disjoint
    ({!Ops.after_disjoint}) run in any order, whichever {!Op.After}s list
    each, none ordered before another by reading the state they write.

    Where these orders form a cycle, one read on the cycle reads instead a copy
    of its whole state, or of its own run for a view of the state, into
    call-local storage ({!Op.Alloc}) at the state's phase, which a kernel of
    its own makes before the state's overwrite; as [b + a * 3] does where [b]
    read [a] before [a] was assigned, or as two buffers swapped through values
    computed from each do. The read copied is one whose state is copied
    already, else the one of fewest bytes. A loop whose call writes a state
    through one argument and reads it through another copies it in the loop, as
    each trip starts, for a state the loop itself writes; any other copy is
    made once, before the loop. A copy starts as its state's storage does, or
    where new storage starts ({!Shape.allocated_start}) for a parameter its
    state's storage starts off: a call reads storage it writes off its
    parameter's start through a copy, made whether or not its orders form a
    cycle. A copy of storage of no device, a function's parameter, is
    call-local storage of no device, placed with the call's arguments. A graph
    whose kernels form no cycle gets no copy.

    Raises [Invalid_argument] if two {!Op.After}s of one state list different
    kernels (two writes over one state), if a call whose body does not show
    its writes takes one state at two arguments and writes it, if a cut graph
    still has a cycle, or if an effect is not a call, a loop of a call, a store
    or an {!Op.After}. *)

val pm_flatten_linear : (unit, Ops.t) Ops.Pattern_matcher.t
(** [pm_flatten_linear] inlines each {!Op.Linear} among the sources of an
    {!Op.Linear} into it. *)

(** {1:views Views of storage} *)

val contiguous_mops_to_view : Ops.t -> Ops.t -> Ops.t option
(** [contiguous_mops_to_view c src] is, for the copy, stage or store [c] of the
    movements and bitcasts [src] of a buffer, [c] of a view of the buffer's
    elements ({!Prepare.contiguous_view}) when they are contiguous, and that
    view for any other [c]; [None] when they are not contiguous or [src]'s shape
    is symbolic, for a copy or a stage when the elements start at bound
    variables, and for a store of a value that reads the buffer short of the
    stored states of other storage, which the kernel split then stages. A
    store's [src] views the buffer through whichever of its states it names:
    the states order the store, and the store writes the buffer as it runs.
    The view is the buffer shrunk to the elements and bitcast to [src]'s type,
    reshaped to [c]'s shape. *)

val is_store_after : Ops.t -> bool
(** [is_store_after u] is [true] iff [u] is an {!Op.After} that orders a store
    into storage: storage other than call-local storage ({!Op.Alloc}), or any
    storage whose first effect is a store. *)

(** {1:schedules Schedules} *)

(** The type for a graph numbered by its own order: a process names each
    range and each piece of storage from counters that depend on what it did
    before, so a key or a kept graph names them by their order instead. *)
type numbering = {
  key : string;  (** The graph's key under that numbering. *)
  place : Ops.t -> Ops.t;
      (** Each node of the graph's in another graph replaced by its
          placeholder. *)
  restore : Ops.t -> Ops.t;  (** The inverse of [place]. *)
  position : Ops.t -> int;
      (** A range's or a piece of storage's place in that order; [-1] for a
          node it does not number. *)
}

val numbered : Ops.t -> numbering
(** [numbered g] numbers the ranges and the storage [g] reaches, that of its
    calls' bodies too, by their order in [g]. Two graphs equal up to renaming
    of those numbers have the same [key], except where a range's bound or a
    piece of storage's size reads another that is numbered: such graphs may
    key apart, which costs a miss. Storage at a negative slot, a variable's, is
    left as it is. *)

val create_linear_with_vars :
  ?capturing:bool -> ?held:Ops.t list -> Ops.t -> Ops.t * (string * int) list
(** [create_linear_with_vars ~capturing ~held big_sink] is the schedule
    that realizes the values of [big_sink], whose results are stored into
    buffers ({!Op.Buffer}), with the value of each variable it reads, by name:

    + the stores of [big_sink] become the body of a call compiled on its own,
      whose arguments are the buffers, the contiguous views of buffers that
      copies read and stores store into ({!contiguous_mops_to_view}), and the
      bound variables, each replaced in the body by a parameter; call-local
      storage is numbered by its order in each body, so equal graphs schedule
      alike;
    + each such call's body is scheduled ({!create_schedule}), once per body
      when the setting {!Setting.scache} is [1] or more: a call that asks for a
      body another domain is scheduling waits for it. From [2], the default,
      schedules are also kept on disk ({!Helpers.Diskcache}, table
      ["schedule_cache"]) for later processes: a schedule is read back for the
      same body, the same values of the settings and variables that shape what
      compilation makes ({!Setting.shaping}), and the same sources of this
      library; a body scheduled under other values is scheduled again, in memory
      too; an entry that does not read as a schedule is made anew and replaced;
    + the schedule's parameters are bound to the call's arguments, and each
      call-local storage to a new buffer on its device. A kernel reads a
      variable by its name, a run binding it, except where [big_sink] binds
      variables of that name to several values: each kernel that reads one
      then takes it as a nameless parameter, its own call passing the bound
      variable, so that kernels which differ in that value alone are one
      program;
    + a kernel that only copies a buffer between devices, or to or from a disk,
      becomes a call of a store, between views of its arguments where it
      copies a run of one into a run of the other;
    + calls of one device that run in either order run as one launch
      ({!Shape.launch}), its members their kernels, whose work is laid end to
      end at once. A launch keeps no order among its members: a call joins
      only where it accesses no byte another member writes. It joins the
      earliest launch of its device that starts after the last call it
      interferes with, so it moves up only past calls whose bytes it does not
      touch, a view counting as the bytes of its run of storage and any other
      view as its whole storage, and a loop or copy writing its storage
      whole. Only where its order lets them, kernels of one write whose
      regions no byte test shows apart, as an assembly's columns are, join
      when their write is marked disjoint ({!Ops.after_disjoint}), which
      exempts them on the state they write, and only there. At the top level a
      call moves up only where no storage's life grows: past the first use of
      each storage it takes, unless it is [held] or holds one element; and a
      call of many elements joins only a launch that writes storage it
      writes. In a loop's body a call joins the earliest launch at whose
      start no step up to it would hold more bytes of the body's own storage
      ({!Memory.bytes}), from each buffer's first use to its last, than the
      body held at its peak before any call joined: storage that is [held]
      or that entries outside the body take does not count. A member is a
      call of a kernel whose ranges have constant sizes and whose storage
      lies on one device that is no disk.
      Limits of the target ({!Realize.lower_and_compile}) split a launch
      after;
    + unless [capturing] (default [false]), the schedule's buffers are placed in
      arenas ({!Memory.memory_plan_rewrite}), except those the call's arguments
      name or view, whose contents outlive the schedule. A jit capturing the schedule
      plans it with the schedules it captures along with it.

    [held] is the storage whose contents outlive the schedule, which the
    launches leave out of what they weigh: by default the buffers the call's
    arguments name, as the schedule's own plan holds them. A capturer that
    plans other buffers itself passes those it holds.

    With the setting {!Setting.spec} at [1] or more, [big_sink] and each body
    are checked against {!Spec.tensor}. With {!Setting.debug} at [1] or more,
    each schedule of more than one call prints its size and time on standard
    output.

    The values are those of the bound variables the kernels read by name or
    the calls pass, each name bound to one value.

    Raises [Invalid_argument] if a kernel that is not a copy accesses buffers
    on several devices, on two writes over one state, an effect listed over a
    state that needs it or a loop that runs anything but one call
    ({!Prepare.sequence}), on a store through a bitcast into bytes that are
    no run of its storage ({!Prepare.prepare_rangeify}), or as
    {!create_schedule} does. *)

(**/**)

val launches : held:Ops.t list -> Ops.t -> Ops.t
(** [launches ~held linear] is the schedule [linear] with the calls that run
    at once launched, as {!create_linear_with_vars} launches them, [held]
    being the storage the schedule is given. *)

val copies : unit -> int
(** [copies ()] is the number of copies of states {!create_schedule} has
    made, in every domain, since the program started. *)
