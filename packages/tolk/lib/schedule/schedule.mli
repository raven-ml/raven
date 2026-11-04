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

val create_schedule : ?by_storage:bool -> Ops.t -> Ops.t
(** [create_schedule ~by_storage sched_sink] is the {!Op.Linear} of the kernel
    calls of the kernel graph [sched_sink] ({!Rangeify.get_kernel_graph}), in
    an order that runs each kernel after those that write the storage it
    reads, and before those that overwrite the storage it reads. Ready kernels
    run in the order they become ready, unless [by_storage] (default [false]),
    for a schedule whose storage is made at each run: then a ready kernel runs
    in the order the graph reaches it, after any ready kernel that frees
    storage, being the last to take storage a kernel wrote: first one that
    makes no storage, then one that on no device makes more storage than it
    frees, storage living from the first kernel that takes it to the last
    ({!Memory.taken}). Each call's arguments are the storage of its
    reads and writes, or the views of it that move with the loops the call runs in. A
    loop of a call, an {!Op.End} of the call over {!Ops.Axis_type.Loop} ranges,
    stays such an end around the call, or around an {!Op.Linear} of the copies
    it makes as each trip starts and the call, and a back edge of a call
    ({!Ops.backedge}) stays a back edge, its condition the storage it reads.

    A kernel reads the states its arguments name, a join ({!Ops.is_join})
    being the state under it, and writes the states the {!Op.After}s that list
    it are over. It runs after the kernels that write the states it reads and
    those the joins it reads through wait for, and before the kernels of
    every write over a state it reads of which it is no part, whichever
    elements each touches. The kernels of one write, whichever {!Op.After}s
    list each, run in any order: one of them must read the elements another
    writes only where it writes their state too, which it then reads as it
    was. A call that writes a state through an argument its body writes
    ({!Shape.written_params}) and reads it through another reads it as it was
    before the call. Where these orders form a cycle, one read on the cycle
    reads instead a copy of its whole state into call-local storage
    ({!Op.Alloc}) at the state's phase, which a kernel of its own makes before
    the state's overwrite; as [b + a * 3] does where [b] read [a] before [a]
    was assigned, or as two buffers swapped through values computed from each
    do. The read copied is one whose state is copied already, else the one of
    fewest bytes. A loop whose call writes a state through one argument and
    reads it through another copies it in the loop, as each trip starts; any
    other copy is made once, before the loop. A write's prior state is the
    latest state of the {!Op.After}'s storage its call is given, and the state
    the {!Op.After} is after where it is given none: a loop after its carry's storage, given the carry's first
    state, writes over that state. A graph whose kernels form no cycle gets no
    copy.

    Raises [Invalid_argument] if two {!Op.After}s list different kernels over
    one state, which no storage holds both successors of, if a call whose body
    does not show its writes takes one state at two arguments and writes it,
    if a state a kernel reads a copy of has no device, if a cut graph still
    has a cycle, or if an effect is not a call, a loop of a call, a store or an
    {!Op.After}. *)

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
    variables, and for a store into a buffer's state after earlier stores or
    of a value that reads the buffer short of the stored states of other
    storage, which the kernel split then stages. The view is the buffer shrunk
    to the elements and bitcast to [src]'s type, reshaped to [c]'s shape. *)

val is_store_after : Ops.t -> bool
(** [is_store_after u] is [true] iff [u] is an {!Op.After} that orders a store
    into storage: storage other than call-local storage ({!Op.Alloc}), or any
    storage whose first effect is a store. *)

(** {1:schedules Schedules} *)

val create_linear_with_vars :
  ?capturing:bool -> ?by_storage:bool -> Ops.t -> Ops.t * (string * int) list
(** [create_linear_with_vars ~capturing ~by_storage big_sink] is the schedule
    that realizes the values of [big_sink], whose results are stored into
    buffers ({!Op.Buffer}), with the value of each variable it reads, by name:

    + the stores of [big_sink] become the body of a call compiled on its own,
      whose arguments are the buffers, the contiguous views of buffers that
      copies read and stores store into ({!contiguous_mops_to_view}), and the
      bound variables, each replaced in the body by a parameter; call-local
      storage is numbered by its order in each body, so equal graphs schedule
      alike;
    + each such call's body is scheduled ({!create_schedule}, [by_storage]
      default [false]), once per body and order
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
    + unless [by_storage], the kernels of one write, as the stores of one
      state into its regions are, run as one launch ({!Shape.launch}) whose
      members are their kernels: the kernels of one write run in any order
      ({!create_schedule}), so they can run at once. A schedule ordered by
      its storage takes none, since a launch holds every member's storage at
      once, where the order would make and free each in turn. A kernel
      joins where no order of the schedule leads from it to another member,
      through kernels that read what one writes or overwrite what one reads;
      the launch then runs where its members would. A member is a call of a kernel whose ranges have constant
      sizes and whose storage lies on one device that is no disk. A member
      that fixes its threads ({!Postrange.threads_fixed}) or reduces, and may
      then share values between a workgroup's threads through barriers, joins
      only members of the same kernel, so that each of the launch's
      workgroups has its threads, and on Metal, which declares workgroup
      memory only in a kernel function, runs alone; any other member joins
      every other such member of its device. A launch passes at most 512
      arguments, the 4 KiB of parameters CUDA takes, and with
      {!Setting.max_kernel_buffers} at [n] above [0] at most [n] buffers. A
      launch of one member is that member's call;
    + a kernel that only copies a buffer between devices, or to or from a disk,
      becomes a call of a store, between views of its arguments where it
      copies a run of one into a run of the other;
    + kernel calls that can run in either order run as one call: two kernels
      that reduce nothing, whose storage lies on one device, whose stores
      end over ranges of the same constant sizes and kinds by position, and
      neither
      of which reads or writes storage the other writes, a view counting as
      its buffer. A call joins the first call of its group, the earliest
      that starts after every call it conflicts with, and after the first
      use of each storage it takes that is neither given to the schedule nor
      of one element; in a loop's body, where storage lives for the whole
      loop, that last condition does not apply, and at the top level only
      calls of one element join. An argument several calls pass is passed
      once, so the merged kernel computes what they share once. With
      {!Setting.max_kernel_buffers} at [n] above [0], a merged call passes
      at most [n] buffers. The merged kernel is optimised as one;
    + unless [capturing] (default [false]), the schedule's buffers are placed in
      arenas ({!Memory.memory_plan_rewrite}), except those the call's arguments
      name or view, whose contents outlive the schedule. A jit capturing the schedule
      plans it with the schedules it captures along with it.

    With the setting {!Setting.spec} at [1] or more, [big_sink] and each body
    are checked against {!Spec.tensor}. With {!Setting.debug} at [1] or more,
    each schedule of more than one call prints its size and time on standard
    output.

    The values are those of the bound variables the kernels read by name or
    the calls pass, each name bound to one value.

    Raises [Invalid_argument] if a kernel that is not a copy accesses buffers on several devices,
    or as {!create_schedule} does. *)

(**/**)

val merge_calls : held:Ops.t list -> Ops.t -> Ops.t
(** [merge_calls ~held linear] is the schedule [linear] with the calls that
    can run as one merged, as {!create_linear_with_vars} merges them, [held]
    being the storage the schedule is given. *)

val copies : unit -> int
(** [copies ()] is the number of copies of states {!create_schedule} has
    made, in every domain, since the program started. *)
