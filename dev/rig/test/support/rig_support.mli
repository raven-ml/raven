(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Test drivers and probes for rig's suites. *)

(** A driver over host memory whose queue runs only when its {!run} or its sleep
    runs it: a wait that returns before it slept leaves work unrun. A sleep
    first runs the queues of the Polled devices whose words its first queued
    submission waits for and has not seen, as a device runs its own work while
    the host sleeps on another. Every driver call is logged. *)
module Polled : sig
  include Rig.Driver

  val make :
    ?capacity:int ->
    ?copies:bool ->
    ?host_visible:bool ->
    ?addresses:bool ->
    ?transport:bool ->
    ?peers:bool ->
    ?maps_host:bool ->
    ?budget:int ->
    ?memory:int ->
    ?window:int ->
    ?may_block:bool ->
    ?completion:[ `Host | `Object ] ->
    ?waits_on:[ `Store | `Object | `Host ] list ->
    ?max_waits:int ->
    ?answer:[ `Stopped | `Unknown ] ->
    ?runs:[ `When_slept | `Itself ] ->
    ?lag:int ->
    ?hang_ms:int ->
    unit ->
    t
  (** [make ()] is a device whose queue holds [capacity] parts (defaults to
      1024): beyond it [room] answers [`Later], or, with [may_block], submit
      waits for room. Without [copies] (defaults to [true]) it lists no copy
      queue. Without [host_visible] (defaults to [true]) the host does not
      address its [Device] memory. Without [addresses] (defaults to [true]) its
      [Device] memory has no address: its work names it by handle alone. With
      [transport] (defaults to [false]) the host does not address its word
      either, which is read through [signaled], as behind a transport. Its
      [host_addresses] fact holds with [host_visible] and without [transport].
      Without [peers] (defaults to [true]) it maps no memory of another device,
      and without [maps_host] (defaults to [true]) no host memory. Its budget is
      [budget] (defaults to 1 GiB); it holds at most [memory] bytes of [Device]
      memory and [window] bytes of [Mapped] memory (default to [max_int]);
      [Pinned] memory is unbounded. Its word advances as [completion] says
      (defaults to [`Host]): with [`Object], it is the driver's object, its
      handle the word's address. Its queue waits for producers of the
      completions [waits_on] lists (defaults to none), at most [max_waits] of
      them per submission (defaults to [max_int]). With [answer] [`Stopped] (the
      default) its stop drops its queued work and writes the last value it
      received into its word; with [`Unknown] it leaves both, as a driver whose
      work may still run. With [runs] [`Itself] (the default is [`When_slept]) a
      thread of the driver also runs its queue as work arrives, as a device runs
      its own work: its work is done at no point a test chooses. With [lag]
      (defaults to [1]) it commits on its own once [lag] values are uncommitted,
      and before a submit waits for room: with [1] each hand-over commits its
      value. Its queue runs only committed submissions, and a sleep that finds
      only uncommitted ones queued, its word unmoved, raises
      [Failure "Polled: nothing committed"], as a wait for work nobody committed
      would hang. With [hang_ms] its facts bound hangs to that many milliseconds
      (defaults to no bound).

      Raises [Invalid_argument] if [lag < 1]. *)

  val open_ :
    ?capacity:int ->
    ?copies:bool ->
    ?host_visible:bool ->
    ?addresses:bool ->
    ?transport:bool ->
    ?peers:bool ->
    ?maps_host:bool ->
    ?budget:int ->
    ?memory:int ->
    ?window:int ->
    ?may_block:bool ->
    ?completion:[ `Host | `Object ] ->
    ?waits_on:[ `Store | `Object | `Host ] list ->
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

  val fault : t -> string -> unit
  (** [fault d why] makes [d]'s sleeps, allocations, mappings, loads and reads
      of its budget raise [Fault why] from now on, as a faulted device's do. *)

  val before : t -> string -> (unit -> unit) -> unit
  (** [before d call f] runs [f] once, at the start of [d]'s next fallible call
      that {!log} names [call], such as ["entry"]: a point inside a call rig
      makes, where a test can change what rig was given. *)

  val fault_word : t -> string -> unit
  (** [fault_word d why] makes [d]'s reads of its word ([signaled]) raise
      [Fault why] from now on, as a transport's that lost its link. *)

  val fail_at : t -> int -> [ `Fault of string | `Refuse of int ] -> unit
  (** [fail_at d n how] fails [d]'s fallible calls from the [n]-th from now on,
      counting from [1]: its facts, its counted calls and its hand-over. With
      [`Fault why] that call and every later one raises [Fault why], and a
      hand-over among them fails with [why] (at most 63 bytes of it), as on a
      device that faulted; from then on a commit of uncommitted values fails
      with [why] too, uncounted. With [`Refuse k] that call and the [k - 1]
      after it refuse where they can, as on a device short of memory: [alloc],
      [map_peer] and [map_host] answer [None], [image] [Error] and [entry]
      [None]; other calls go on. [`Refuse 0] ends an earlier refusal. *)

  val steps : t -> int
  (** [steps d] is the number of fallible calls [d] received. *)

  val outstanding : t -> int list
  (** [outstanding d] is the bytes of each region [d] allocated or mapped and
      has not freed. *)

  val set_word : t -> int -> unit
  (** [set_word d v] writes [v] into [d]'s word. *)

  val capability_key : unit Type.Id.t
  (** [capability_key] is the key of Polled's capability, [()]. *)

  val stop_fault : t -> string option
  (** [stop_fault d] is the [fault] [d]'s stop was given, [None] before a stop.
  *)

  val word_at : t -> int
  (** [word_at d] is the host address of [d]'s word, which is also its object
      ([completion]). *)

  val log : t -> string list
  (** [log d] is [d]'s driver calls, oldest first: ["alloc"], ["free"],
      ["map_host"], ["map_peer"], ["unmap"] (a mapping's free), ["word"] (the
      timeline word's free), ["sleep"], ["stop"], ["image"], ["entry"],
      ["unload"]. *)

  val frees : t -> (int * int) list
  (** [frees d] is the address of each region [d] freed, oldest first, with
      [d]'s word when it was freed. *)

  val allocs : t -> (Rig_edge.memory * int * bool) list
  (** [allocs d] is each allocation asked of [d], oldest first: its kind, its
      bytes and whether [d] gave it. *)

  val host_maps : t -> int list
  (** [host_maps d] is the bytes of each host memory [d] mapped, oldest first.
  *)

  val allocated : t -> Rig_edge.memory -> int
  (** [allocated d kind] is the bytes of [kind] [d] holds allocated. *)

  val last_waits : t -> (int * int * int) list
  (** [last_waits d] is the waits [d]'s last submit received, at most 8, as
      [(kind, at, value)], [kind] one of {!rig_word} and {!rig_object}. *)

  val last_handles : t -> int list
  (** [last_handles d] is the handles [d]'s last submit received, at most 64, in
      the order received. A region's handle is its address. *)

  val copy_sides : t -> [ `None | `Src | `Dst ] list
  (** [copy_sides d] is the side of this process's memory that each copy [d] ran
      named ([copy_local] in [rig_edge.h]), the first 64, in the order run. *)

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
      of at most [2{^31} - 1] groups along x and 65535 along y and z. *)

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

  val interrupt : t -> unit
  (** [interrupt d] makes [d]'s next sleep raise SIGINT in its thread and return
      without running the queue. *)

  val stall : t -> int -> unit
  (** [stall d n] makes [d]'s next [n] sleeps return after their still interval
      without running the queue, as over work that runs long. *)
end

val rig_word : int
(** [rig_word] is [rig_edge.h]'s [RIG_WORD]. *)

val rig_object : int
(** [rig_object] is [rig_edge.h]'s [RIG_OBJECT]. *)

val bump : nativeint
(** [bump] is a fill adding 1 to the 64-bit word its argument points at. *)

val poke : nativeint
(** [poke] is a fill storing the second 64-bit word of its argument at the
    address its first holds. *)

val slow : nativeint
(** [slow] is a fill that runs for the milliseconds in the 64-bit word its
    argument points at, as a long kernel does. *)

val countdown : nativeint
(** [countdown] is a fill taking one from the 64-bit word its argument points
    at, which fails if that makes the word zero. *)

val ready : nativeint
(** [ready] is the address of [void ready(void *word, uint64_t c)], which
    stores [c] at [word] with release order, as a rail's ready function
    does. *)

val carry : nativeint
(** [carry] is a fill copying bytes: its argument's three 64-bit words are the
    destination's address, the source's and the number of bytes. *)

val load : int -> int
(** [load a] reads the 64-bit word at the host address [a]. *)

val store : int -> int -> unit

val move : dst:int -> src:int -> int -> unit
(** [move ~dst ~src n] copies the [n] bytes at host address [src] to host
    address [dst]. *)

val host_held : unit -> int
(** [host_held ()] is the bytes that count in the host's budget: host buffers
    not yet returned, and devices' pinned host memory. *)

val heap_bytes : unit -> int option
(** [heap_bytes ()] is the bytes the C heap holds allocated, as its allocator
    counts them, or [None] where it does not say. *)

val descriptors : unit -> int option
(** [descriptors ()] is the number of file descriptors the process holds open,
    or [None] on Windows. *)

val shares : ('a, 'b, 'c) Bigarray.Array1.t -> int
(** [shares ba] is how many holders share [ba]'s storage, as the runtime counts
    them on the proxy its arrays share: [0] if they share none. *)

val await : string -> (unit -> bool) -> unit
(** [await what f] returns once [f ()] holds, yielding to other threads between
    checks. It raises [Failure] naming [what] after 10 s. *)

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

  val claim : Rig.Buffer.t -> Rig.Buffer.access -> answer
  (** [claim b access] is [rig_buffer_claim b access]. *)

  val wait : Rig.Buffer.t -> Rig.Buffer.access -> unit
  (** [wait b access] is [rig_buffer_wait b access], raising the exception it
      answers. *)

  val release : Rig.Buffer.t -> unit
  (** [release b] is [rig_buffer_release b]. *)

  val span : Rig.Buffer.t -> int * int
  (** [span b] is the space and first byte [rig_buffer_span b] gives. *)
end
