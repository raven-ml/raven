(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Test drivers and probes for rig's suites. *)

(** A driver over host memory whose queue runs only when its {!run} or its sleep
    runs it: a wait that returns before it slept leaves work unrun. A sleep
    first runs the queues of the Polled devices whose words its first queued
    submission waits for and has not seen, as a device runs its own work while
    the host sleeps on another. Every driver call is logged. A device's lock
    counts among the locks its thread holds, as a driver's does
    ({!Rig_edge.section-drivers}).

    Polled checks that rig keeps the word promises of {!Rig_edge}: rig frees
    only a word outside its array, which is the driver's own, once, and then
    reads it no more, and a queue's wait reads a word that rig's array or a
    live mapping of the queue's device holds. A breach is recorded
    ({!violations}) and raises [Failure] where the call that saw it returns
    to OCaml: a [signaled] or a [free] of the word, or a {!run} or sleep that
    read the wait. A device that runs itself records it alone. A mapping of
    another device's word lies at that word's address, so a wait through a
    mapping rig gave back is seen only once the device holds no mapping of
    that word.

    Polled also checks that rig settles the value a time pair timed before it
    gives the pair to another value. A breach, which a submit sees with no
    OCaml call to return to, is only recorded. *)
module Polled : sig
  include Rig_edge.Driver

  val make :
    ?capacity:int ->
    ?grows:int ->
    ?most_parts:int ->
    ?copies:bool ->
    ?host_visible:bool ->
    ?addresses:bool ->
    ?transport:bool ->
    ?peers:bool ->
    ?maps_host:bool ->
    ?alignment:int ->
    ?budget:int ->
    ?memory:int ->
    ?window:int ->
    ?may_block:bool ->
    ?completion:[ `Host | `Store ] ->
    ?waits_on:[ `Store | `Host ] list ->
    ?max_waits:int ->
    ?answer:[ `Stopped | `Unknown ] ->
    ?runs:[ `When_slept | `Itself ] ->
    ?lag:int ->
    ?hang_ms:int ->
    ?capability:Rig_edge.capability ->
    unit ->
    t
  (** [make ()] is a device whose queue holds [capacity] parts (defaults to
      1024), the most one submission holds: beyond it [room] answers
      [`Later], or, with [may_block], submit waits for room, and it answers
      [`Never] for a submission of more. With [grows] it answers [RIG_GROW]
      for such a submission instead, and its [grow] makes the queue hold the
      parts of the submission whose [room] answered so on the calling
      thread, up to [grows] parts. For more, and where it refuses
      ({!fail_at}), [grow] answers [Short 4096]. Its facts state
      [most_parts] (defaults to [capacity]), at most [capacity], so that an
      idle queue runs that many parts. Without
      [copies] (defaults to [true]) it lists no copy queue. Its compute queue
      takes words, and does nothing for them.
      Without [host_visible] (defaults to [true]) the host does not address
      its [Device] memory. Without [addresses] (defaults to [true]) its
      [Device] memory has no address: its work names it by handle alone. With
      [transport] (defaults to [false]) the host does not address its word
      either, which is read through [signaled], as behind a transport. Its
      [host_addresses] fact holds with [host_visible] and without [transport].
      Without [peers] (defaults to [true]) it maps no memory of another device,
      and without [maps_host] (defaults to [true]) no host memory. Its facts
      state [alignment] (defaults to [4096]), and its regions start on a
      page, which may break a larger [alignment]. Its budget is
      [budget] (defaults to 1 GiB); it holds at most [memory] bytes of [Device]
      memory and [window] bytes of [Mapped] memory (default to [max_int]);
      [Pinned] memory is unbounded. Its word advances as [completion] says
      (defaults to [`Host]): with [`Store], as a queue's store writes it. Its
      queue waits for producers of the completions [waits_on] lists (defaults
      to none), at most [max_waits] of them per submission (defaults to
      [max_int]). With [answer] [`Stopped] (the default) its stop drops its
      queued work and writes the last value it received into its word, and frees
      what it ran work with: its queue, and the records {!last_waits},
      {!layout}, {!copy_sides}, {!copy_device_memory}, {!blocked} and
      {!launches} read, which then read none. With [`Unknown] it leaves all of
      it, as a driver whose work may still run. With [runs] [`Itself] (the
      default is [`When_slept]) a thread of the driver also runs its queue as
      work arrives, until the stop, as a device runs its own work: its work is
      done at no point a test chooses. With [lag] (defaults to [1]) it commits
      on its own once [lag] values are uncommitted, and before a submit waits
      for room: with [1] each hand-over commits its value. Its queue runs only
      committed submissions, and a sleep that finds only uncommitted ones
      queued, its word unmoved, raises [Failure "Polled: nothing committed"], as
      a wait for work nobody committed would hang. With [hang_ms] its facts
      bound hangs to that many milliseconds (defaults to no bound). Its facts
      declare [capability] (defaults to [()] under {!capability_key}).

      Raises [Invalid_argument] if [lag < 1] or [most_parts > capacity]. *)

  val bind : ?start:int -> t -> word:nativeint -> t
  (** [bind d ~word] is [d], whose word is now the 8 bytes at [word], as an
      opener's [make] does with the word rig gives it ({!Rig_edge.type-opener}):
      [d]'s values start after the value they hold. With [start] it first
      raises them to [start], as a [make] that raises its word does. A device
      made with [transport] keeps a word of its own, started at their value,
      as a device of another machine does. *)

  val opener : ?start:int -> string -> t -> t Rig_edge.opener
  (** [opener name d] opens [d] as the device [name], whose [make] binds [d]
      to its word ({!bind}). *)

  val open_ :
    ?capacity:int ->
    ?grows:int ->
    ?most_parts:int ->
    ?copies:bool ->
    ?host_visible:bool ->
    ?addresses:bool ->
    ?transport:bool ->
    ?peers:bool ->
    ?maps_host:bool ->
    ?alignment:int ->
    ?budget:int ->
    ?memory:int ->
    ?window:int ->
    ?may_block:bool ->
    ?completion:[ `Host | `Store ] ->
    ?waits_on:[ `Store | `Host ] list ->
    ?max_waits:int ->
    ?answer:[ `Stopped | `Unknown ] ->
    ?runs:[ `When_slept | `Itself ] ->
    ?lag:int ->
    ?hang_ms:int ->
    string ->
    Rig.t * t
  (** [open_ name] opens a fresh device named [name]. *)

  val run : t -> int
  (** [run d] runs the queued committed submissions whose waits hold: how many
      ran. *)

  val queued : t -> int
  val submits : t -> int

  val fail : t -> unit
  (** [fail d] makes [d]'s next submit fail. *)

  val fail_commit : t -> unit
  (** [fail_commit d] makes [d]'s next commit of uncommitted values fail. *)

  val fail_settle : t -> unit
  (** [fail_settle d] makes [d]'s next settle fail. Its other settles do
      nothing, since its queue's run writes a value's times before the word
      shows the value. *)

  val fault : t -> string -> unit
  (** [fault d why] makes [d]'s sleeps, allocations, mappings, loads and reads
      of its budget raise [Fault why] from now on, as a faulted device's do,
      and its frees raise it once they gave their region back. *)

  val before : t -> string -> (unit -> unit) -> unit
  (** [before d call f] runs [f] once, at the start of [d]'s next counted call
      or stop that {!log} names [call], such as ["entry"], of its next read
      of its word, ["signaled"], or of the function of the next [Place] its
      [image] answers, ["place"]: a point inside a
      call rig makes, where a test can change what rig was given, hold the
      call or raise from it. A free runs it once it passed the free gate,
      and a free of the word once it gave the word back. *)

  val fault_word : t -> string -> unit
  (** [fault_word d why] makes [d]'s reads of its word ([signaled]) raise
      [Fault why] from now on, as a transport's that lost its link. *)

  val break_word : t -> unit
  (** [break_word d] makes [d]'s next read of its word ([signaled]) raise
      [Sys.Break], as a Ctrl-C landing in it does. *)

  val fail_at : t -> int -> [ `Fault of string | `Refuse of int ] -> unit
  (** [fail_at d n how] fails [d]'s fallible calls from the [n]-th from now on,
      counting from [1]: its facts, its counted calls and its hand-over. With
      [`Fault why] that call and every later one raises [Fault why], and a
      hand-over among them fails with [why] (at most 63 bytes of it), as on a
      device that faulted; from then on a commit of uncommitted values fails
      with [why] too, uncounted. With [`Refuse k] that call and the [k - 1]
      after it refuse where they can, as on a device short of memory: [alloc],
      [map_peer] and [map_host] answer [None], [image] [Error], [entry]
      [No_memory entry_scratch] and [grow] [Short 4096]; other calls go on.
      [`Refuse 0] ends an earlier refusal. *)

  val steps : t -> int
  (** [steps d] is the number of fallible calls [d] received. *)

  val outstanding : t -> int list
  (** [outstanding d] is the bytes of each region [d] allocated or mapped and
      has not freed. *)

  val set_word : t -> int -> unit
  (** [set_word d v] writes [v] into [d]'s word. *)

  val capability_key : unit Type.Id.t
  (** [capability_key] is the key of Polled's capability, [()]. *)

  val entry_scratch : int
  (** [entry_scratch] is the bytes a refused [entry] says its function's
      launches need ({!fail_at}): 64 KiB. *)

  val stop_fault : t -> string option
  (** [stop_fault d] is the [fault] [d]'s stop was given, [None] before a stop.
  *)

  val stops : t -> int
  (** [stops d] is the number of calls of [d]'s stop, each counted before the
      stop allocates: also a call that an exception at its first allocation
      ended, which {!log} misses. *)

  val word_at : t -> int
  (** [word_at d] is the host address of [d]'s word. *)

  val log : t -> string list
  (** [log d] is [d]'s driver calls, oldest first: ["alloc"], ["free"] (of
      a region or of the word), ["map_host"], ["map_peer"], ["unmap"] (a
      mapping's free), ["sleep"],
      ["stop"], ["image"], ["entry"], ["unload"]. Consecutive sleeps are one
      ["sleep"]; {!sleeps} counts them. *)

  val sleeps : t -> int
  (** [sleeps d] is the number of [d]'s sleeps. *)

  val violations : unit -> string list
  (** [violations ()] is each breach of the word promises any Polled device
      saw, oldest first, the first 64. *)

  val frees : t -> (int * int) list
  (** [frees d] is the address of each region [d] freed, oldest first, with
      [d]'s word when it was freed. *)

  val word_frees : t -> int
  (** [word_frees d] is the number of times rig freed [d]'s word. *)

  val allocs : t -> (Rig_edge.memory * int * bool) list
  (** [allocs d] is each allocation asked of [d], oldest first: its kind, its
      bytes and whether [d] gave it. *)

  val host_maps : t -> int list
  (** [host_maps d] is the bytes of each host memory [d] mapped, oldest first.
  *)

  val allocated : t -> Rig_edge.memory -> int
  (** [allocated d kind] is the bytes of [kind] [d] holds allocated. *)

  val last_times : t -> int
  (** [last_times d] is the host address where the run of [d]'s last submit
      writes its times, [0] for nowhere. *)

  val last_waits : t -> (int * int) list
  (** [last_waits d] is the waits [d]'s last submit received, at most 8, as
      [(at, value)]. *)

  val layout : t -> (int * int list) list
  (** [layout d] is, for each part of the last submission [d]'s hand-over
      received, its queue's index and the parts its [after] lists (the first
      64 parts, and their first 8 afters). *)

  val copy_sides : t -> [ `None | `Src | `Dst ] list
  (** [copy_sides d] is the side of this process's memory that each copy [d] ran
      named ([copy_local] in [rig_edge.h]), the first 64, in the order run. *)

  val copy_device_memory : t -> bool list
  (** [copy_device_memory d] is the [copy.device_memory] fact of each copy [d]
      ran, including empty copies, the first 64 in the order run. *)

  val blocked : t -> int
  (** [blocked d] is the number of submits waiting for room in [d]'s [may_block]
      queue. *)

  (** {1:launches Launches}

      [d]'s queue ["COMPUTE:0"] runs launches; its ["COPY:0"] does not. An image
      of [d] is the binary ["code:N"], [N] bytes of code placed in [d]'s memory,
      or ["functions"], which the driver loads and places nowhere. Its functions
      are Polled's host functions, run once per group of the grid, x fastest:
      - ["main"] does nothing;
      - ["fill"] stores, as the 64-bit word of index [i], the group's index in
        the grid, the second parameter word plus [i] at the address the first
        holds;
      - ["copy"], in group 0 alone, copies as many bytes as the third word holds
        from the address the first holds to the one the second holds.

      Each allows 1024 threads per group and 48 KiB of shared memory, and grids
      of at most [2{^31} - 1] groups along x and 65535 along y and z. The
      driver states the threads, and the bytes of parameters of ["fill"], 16,
      and of ["copy"], 24. *)

  type launch = {
    groups : int * int * int;
    threads : int * int * int;
    shared : int;
    params : string;  (** As the function read them. *)
  }

  val launches : t -> launch list
  (** [launches d] is the first 8 launches [d] ran since the last call, oldest
      first, with their blocks as they ran. *)

  (** {1:sleeps Sleeps}

      Seams that hold a wait at a known point: each acts on the sleeps that
      follow, in the order a sleep checks them: the gate, a fault, an
      interruption, a stall. *)

  val gate : t -> unit
  (** [gate d] makes [d]'s sleeps and its stop block until {!open_gate}. *)

  val open_gate : t -> unit
  (** [open_gate d] lets [d]'s blocked sleeps and stop go on. *)

  val sleepers : t -> int
  (** [sleepers d] is the number of [d]'s calls blocked at its gate. *)

  val gate_frees : t -> unit
  (** [gate_frees d] makes [d]'s frees of regions and mappings, but not of its
      word, block before they are logged, until {!open_frees}. *)

  val open_frees : t -> unit
  (** [open_frees d] lets [d]'s blocked frees go on. *)

  val freers : t -> int
  (** [freers d] is the number of [d]'s frees blocked. *)

  val gate_locates : t -> unit
  (** [gate_locates d] makes locates of the mappings [d] made block until
      {!open_locates}: a probe after a map, outside any counted call, so
      [d]'s stop can return meanwhile. *)

  val open_locates : t -> unit
  (** [open_locates d] lets the blocked locates of [d]'s mappings go on. *)

  val locaters : t -> int
  (** [locaters d] is the number of locates of [d]'s mappings blocked. *)

  val gate_room : t -> unit
  (** [gate_room d] makes the [room] of [d], made with [may_block], block
      until {!open_room}: a submit there holds [d]'s turn, its waits
      checked and no value taken yet.

      Raises [Invalid_argument] if [d] was made without [may_block]. *)

  val open_room : t -> unit
  (** [open_room d] lets [d]'s blocked [room] calls go on. *)

  val in_room : t -> int
  (** [in_room d] is the number of [d]'s [room] calls under way, those
      blocked at the gate included. *)

  val gate_settle : t -> unit
  (** [gate_settle d] makes the [settle] of [d], made with [may_block], block
      until {!open_settle}: a profile's read of a value's times there holds
      [d]'s turn.

      Raises [Invalid_argument] if [d] was made without [may_block]. *)

  val open_settle : t -> unit
  (** [open_settle d] lets [d]'s blocked [settle] calls go on. *)

  val settlers : t -> int
  (** [settlers d] is the number of [d]'s [settle] calls under way, those
      blocked at the gate included. *)

  val interrupt : t -> unit
  (** [interrupt d] makes [d]'s next sleep raise SIGINT in its thread and return
      without running the queue. *)

  val stall : t -> int -> unit
  (** [stall d n] makes [d]'s next [n] sleeps return after their still interval
      without running the queue, as over work that runs long. *)
end

val noop : nativeint
(** [noop] is a fill that returns without reading its argument. *)

val touching :
  (int -> at:int -> Rig.Submission.ref) array -> Rig.Submission.part
(** [touching uses] is a fill of {!noop} whose ref [i] is [uses.(i) i ~at:(8 *
    i)], such as [Rig.Submission.read]: work that uses each slot and does
    nothing. *)

val bump : nativeint
(** [bump] is a fill adding 1 to the 64-bit word at the address in its argument.
*)

val waiting : nativeint
(** [waiting] is a fill that sets the second of the two 64-bit words at the
    address in its argument, then returns once the first is not zero, or
    after 30 s: work held running until the test releases it. *)

val poke : nativeint
(** [poke] is a fill storing the second 64-bit word of its argument at the
    address its first holds. *)

val far : nativeint
(** [far] is a fill of 8,192 bytes of argument storing the 64-bit word at its
    byte 4,096 at the address its last 8 bytes hold. *)

val countdown : nativeint
(** [countdown] is a fill taking one from the 64-bit word at the address in its
    argument, which fails if that makes the word zero. *)

val ready : nativeint
(** [ready] is the address of [void ready(void *word, uint64_t c)], which stores
    [c] at [word] with release order, as a rail's ready function does. *)

val carry : nativeint
(** [carry] is a fill copying bytes: its argument's three 64-bit words are the
    destination's address, the source's and the number of bytes. *)

val load : int -> int
(** [load a] reads the 64-bit word at the host address [a]. *)

val store : int -> int -> unit

val load64 :
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t ->
  int ->
  int64
(** [load64 a at] reads the 64-bit word at byte [at] of [a] with acquire order,
    as work that reads a rail's count does: the bytes written before a release
    store of the word are visible after it. *)

val move : dst:int -> src:int -> int -> unit
(** [move ~dst ~src n] copies the [n] bytes at host address [src] to host
    address [dst]. *)

val host_held : unit -> int
(** [host_held ()] is the bytes that count in the host's budget: host buffers
    not yet returned, and devices' pinned host memory. *)

val heap_bytes : unit -> int option
(** [heap_bytes ()] is the bytes the C heap holds allocated, as its allocator
    counts them, or [None] where it does not say. On glibc it first fills the
    calling thread's cache of freed chunks, which glibc counts as allocated, so
    two counts on one thread differ by what was allocated and not freed
    between them. *)

val descriptors : unit -> int option
(** [descriptors ()] is the number of file descriptors the process holds open,
    or [None] on Windows. *)

val needs_fork : unit -> unit
(** [needs_fork ()] skips the running test on Windows, which has no fork, and
    where the process runs more than one thread and GCC compiled its C with
    AddressSanitizer. GCC's runtime holds no allocator lock across fork, so a
    child can inherit one that another thread held, and hang. A test calls it
    before it forks. *)

val resident : unit -> int
(** [resident ()] is this process's resident memory in bytes: its working set
    on Windows. *)

val recv_timeout : Unix.file_descr -> float -> unit
(** [recv_timeout fd s] bounds each read on the socket [fd] to [s] seconds, [0.]
    for none, on every system: [Unix.setsockopt_float] hands Windows the bound
    in a unit it reads as milliseconds. A read past the bound raises
    [Unix.Unix_error] with [EAGAIN] or [EWOULDBLOCK], [ETIMEDOUT] on Windows. *)

val descriptor_limit : unit -> int
(** [descriptor_limit ()] is this process's soft limit of its descriptors:
    [max_int] if it has none, [-1] on Windows. *)

val set_descriptor_limit : int -> int
(** [set_descriptor_limit n] sets this process's soft limit of its
    descriptors to [n]: [0], the errno, or [-1] on Windows, which has no such
    limit. *)

val shares : ('a, 'b, 'c) Bigarray.Array1.t -> int
(** [shares ba] is how many holders share [ba]'s storage, as the runtime counts
    them on the proxy its arrays share: [0] if they share none. *)

val park : unit -> unit
(** [park ()] blocks the calling thread outside OCaml, holding no runtime lock,
    until {!unpark} or 5 s. One thread parks at a time. *)

val parked : unit -> bool
(** [parked ()] is [true] while a thread is in {!park}, outside OCaml. *)

val unpark : unit -> unit
(** [unpark ()] ends the thread's {!park}. *)

val sees_waits : bool
(** [sees_waits] is [true] where the system shows which call a thread is
    blocked in, and condition waits enter a call of their own: on macOS, and on
    Linux with glibc. *)

type thread
(** A thread, as the system names it. *)

val self : unit -> thread
(** [self ()] is the calling thread. *)

val waits_for_lock : thread -> bool
(** [waits_for_lock t] is whether [t] blocks taking a mutex, as a take of
    one of rig's locks does, from the system's view of it. A wait for a
    domain's own lock shows too, briefly. It raises [Failure] where not
    {!sees_waits}. *)

val waits_in_condition : thread -> bool
(** [waits_in_condition t] is whether [t] blocks in a condition wait, as a
    submit does while its device is held, from the system's view of it. A
    wait for a domain's own lock shows too. It raises [Failure] where not
    {!sees_waits}. *)

val interrupt : unit -> unit
(** [interrupt ()] raises SIGINT in the calling thread and returns with no
    poll point: the runtime records the signal, and the thread's next poll
    point runs its handler when the caller runs alone. With other domains,
    the first to poll runs it. *)

val interrupt_after : float -> unit
(** [interrupt_after s] starts a thread that runs no OCaml code and raises
    SIGINT [s] seconds later, as a Ctrl-C does: its handler runs on the next
    thread that returns to OCaml. One such thread runs until
    {!interrupt_join}. It raises [Failure] on Windows. *)

val interrupt_join : unit -> unit
(** [interrupt_join ()] waits until the thread {!interrupt_after} started,
    if any, has ended, then runs the handlers of the signals pending, so its
    SIGINT lands here if no poll point took it before. *)

val interrupt_in_wait : (unit -> 'a) -> 'a * bool
(** [interrupt_in_wait f] is [f ()], run while a new thread that runs no OCaml
    code raises SIGINT once the calling thread blocks in a condition wait, as a
    signal from outside the process does: its handler runs on the next thread
    that returns to OCaml. The boolean is whether the signal was raised. It
    raises [Failure] where not {!sees_waits}. *)

(** What lands at the allocation {!interrupted} counts to. *)
type landing =
  | Raise
      (** That allocation and every one after it until the function returns
          raise [Exit], as an asynchronous exception does, so handlers that
          allocate meet it again. *)
  | Signal
      (** That allocation raises SIGINT, which the next poll point turns into
          [Sys.Break], as a Ctrl-C does. A poll point that follows no
          allocation is reached so: a callback deferred from an allocation in
          C runs at a poll point, after the signals pending there. *)
  | Call of (unit -> unit)
      (** That allocation calls the function, which may raise, or interrupt
          ({!interrupt}) under a handler of the caller's. *)

val interrupted :
  ?from:(Gc.Memprof.allocation -> bool) ->
  ?landing:landing ->
  int ->
  (unit -> unit) ->
  bool
(** [interrupted ~from ~landing n f] runs [f ()] while a memory profile
    samples every allocation of the domain, and is whether [f] reached the
    [n]-th allocation of the calling thread among those [from] holds for
    (every one from [f]'s start by default), where [landing] (default
    {!Raise}) lands. Once it was reached,
    [Exit] and [Sys.Break] from [f] are the landing's and are dropped; any
    other exception passes through. Under {!Signal}, SIGINT's handler raises
    [Sys.Break] until [f] returns and a signal still pending has landed.
    Walking [n] from [1] until it is [false] lands at every
    allocation [f] makes. *)

val holds : unit -> int
(** [holds ()] is the number of locks the calling thread holds, rig's and the
    drivers' ([rig_holds] in [rig_edge.h]). *)

val inside_lock : (unit -> 'a) -> 'a
(** [inside_lock f] is [f ()] run while the calling thread counts one more
    lock, as a callback that runs inside a lock does. *)

type seen
(** The type for the landings of a walk, counted inside a lock and between
    locks. *)

val seen : unit -> seen
(** [seen ()] has counted no landing. *)

type landed
(** The type for what a call from a callback answered. *)

val landing : seen -> string -> (unit -> unit) -> landing * landed
(** [landing seen prefix call] is a {!Call} landing that runs [call ()],
    counts in [seen] whether the thread held a lock there, and records what
    [call] did: refused, for an [Invalid_argument] whose text starts with
    [prefix], returned, or another exception. *)

val answered : string -> landed -> bool option
(** [answered what l] fails, naming [what], unless the call was refused inside
    a lock and returned between locks, leaving its thread's locks as they
    were. It is [Some held], whether the thread held a lock, or [None] if the
    call did not run. *)

val covered : seen -> unit
(** [covered seen] fails unless [seen] counted landings both inside a lock and
    between locks. *)

val set_address_space : int -> int
(** [set_address_space n] sets the process's soft limit of its address space
    to [n] bytes: [0], the errno of a refusal, or [-1] off Linux, where the
    limit does not bound allocations. *)

val time : float -> float
(** [time s] is [s] seconds times the factor [RIG_TEST_TIME_SCALE] names, 1
    when it is unset. A profile whose build slows every test sets it, as the
    tsan profile does, and a test's [~timeout] is [time s].

    Raises [Invalid_argument] at the program's start if the factor is not a
    number of at least 1. *)

val await : string -> (unit -> bool) -> unit
(** [await what f] returns once [f ()] holds, yielding to other threads between
    checks. It raises [Failure] naming [what] after 10 s. *)

val allocated_bytes : (unit -> unit) -> int
(** [allocated_bytes f] is the bytes [f ()] adds to the OCaml heap, minor and
    major ({!Gc.allocated_bytes}). A function that allocates nothing has [0]
    for every number of calls in [f]. *)

val donated : Rig.Buffer.t -> Rig.Buffer.t option
(** [donated b] donates [b] alone with the reason ["donated"]: the buffer
    {!Rig.Claim.consume} returns where the donation held [b]'s memory
    exclusive, and [None], changing nothing, otherwise. *)

val io : string -> Rig.t
(** [io name] is the io device named [name], whose memory holds nothing, opened
    unless it is open. *)

val machine : string -> Rig.t
(** [machine m] is the host of the machine named [m], a {!Polled} device whose
    memory this process does not address and that maps no other memory, opened
    unless it is open: devices of [m] open beside it ({!Rig.open_}). *)

(** The C readers of [rig.h], called from C. *)
module Reader : sig
  val host : Rig.Buffer.t -> int
  (** [host b] is [rig_buffer_host b] as an integer, [0] for [NULL]. *)

  val bytes : Rig.Buffer.t -> int
  (** [bytes b] is [rig_buffer_bytes b]. *)

  val why : Rig.Buffer.t -> string option
  (** [why b] is [rig_buffer_why b], [None] for [NULL]. *)

  (** The answers of [rig_buffer_claim], in [enum rig_claim]'s order. *)
  type answer = Claimed | Wait | Lost | Dead | Exclusive | Read_only

  val pp_answer : Format.formatter -> answer -> unit

  val claim : Rig.Buffer.t -> Rig.access -> answer
  (** [claim b access] is [rig_buffer_claim b access]. *)

  val wait : Rig.Buffer.t -> Rig.access -> unit
  (** [wait b access] is [rig_buffer_wait b access], raising the exception it
      answers. *)

  val release : Rig.Buffer.t -> unit
  (** [release b] is [rig_buffer_release b]. *)

  (** The outcomes of [rig_buffers_claim]: every buffer claimed, with each
      one's answer, or the index of the first buffer refused and why, with
      none claimed. *)
  type outcome = All of answer array | Refused of int * answer

  val claim_all : Rig.Buffer.t array -> Rig.access array -> outcome
  (** [claim_all bs access] is [rig_buffers_claim] of [bs] for [access]. *)

  val release_all : Rig.Buffer.t array -> unit
  (** [release_all bs] is [rig_buffers_release] of [bs]. *)

  val read : Rig.Buffer.t -> unit
  (** [read b] takes a read claim on [b]'s memory with {!claim}, held until
      {!release}, without waiting. It raises [Invalid_argument] where the claim
      is refused, and [Rig.Lost], the claim ended, where the memory or the work
      it follows is a lost device's. *)

  val span : Rig.Buffer.t -> int * int
  (** [span b] is the space and first byte [rig_buffer_span b] gives. *)
end

val at : Rig.Buffer.t -> int
(** [at b] is where [b]'s first byte lies, to tell memory apart: its host
    address where this process addresses its memory ({!Reader.host}), and its
    driver's handle plus its offset otherwise, the address of a {!Polled}
    region. Raises as {!Rig.Buffer.handle} for memory with neither. *)
