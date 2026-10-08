(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Test drivers and probes for the core's suites. *)

(** A driver over host memory whose queue runs only when its {!run} or its sleep
    runs it: a wait that returns before it slept leaves work unrun. Every driver
    call is logged. *)
module Polled : sig
  include Rig.Driver

  val make :
    ?capacity:int ->
    ?copies:bool ->
    ?host_visible:bool ->
    ?transport:bool ->
    ?peers:bool ->
    ?budget:int ->
    ?memory:int ->
    ?window:int ->
    ?may_block:bool ->
    ?completion:[ `Host | `Object ] ->
    ?waits_on:[ `Store | `Object | `Host ] list ->
    ?max_waits:int ->
    ?answer:[ `Stopped | `Unknown ] ->
    unit ->
    t
  (** [make ()] is a device whose queue holds [capacity] parts (defaults to
      1024): beyond it [room] answers [`Later], or, with [may_block], submit
      waits for room. Without [copies] (defaults to [true]) it lists no copy
      queue. Without [host_visible] (defaults to [true]) the host does not
      address its [`Device] memory. With [transport] (defaults to [false]) the
      host does not address its word either, which is read through [signaled],
      as behind a transport. Without [peers] (defaults to [true]) it maps no
      memory of another device. Its budget is [budget] (defaults to 1 GiB); it
      holds at most [memory] bytes of [`Device] memory and [window] bytes of
      [`Mapped] memory (default to [max_int]); [`Pinned] memory is unbounded.
      Its word advances as [completion] says (defaults to [`Host]): with
      [`Object], it is the driver's object, its handle the word's address. Its
      queue waits for producers of the completions [waits_on] lists (defaults to
      none), at most [max_waits] of them per submission (defaults to [max_int]).
      With [answer] [`Stopped] (the default) its stop drops its queued work and
      writes the last value it received into its word; with [`Unknown] it leaves
      both, as a driver whose work may still run. *)

  val open_ :
    ?capacity:int ->
    ?copies:bool ->
    ?host_visible:bool ->
    ?transport:bool ->
    ?peers:bool ->
    ?budget:int ->
    ?memory:int ->
    ?window:int ->
    ?may_block:bool ->
    ?completion:[ `Host | `Object ] ->
    ?waits_on:[ `Store | `Object | `Host ] list ->
    ?max_waits:int ->
    ?answer:[ `Stopped | `Unknown ] ->
    string ->
    Rig.t * t
  (** [open_ name] opens a fresh device named [name]. *)

  val run : t -> int
  (** [run d] runs the queued submissions whose waits hold: how many ran. *)

  val queued : t -> int
  val submits : t -> int

  val fail : t -> unit
  (** [fail d] makes [d]'s next submit fail. *)

  val fault : t -> string -> unit
  (** [fault d why] makes [d]'s sleeps, allocations, mappings, loads and reads
      of its budget raise [Fault why] from now on, as a faulted device's do. *)

  val fault_word : t -> string -> unit
  (** [fault_word d why] makes [d]'s reads of its word ([signaled]) raise
      [Fault why] from now on, as a transport's that lost its link. *)

  val set_word : t -> int -> unit
  (** [set_word d v] writes [v] into [d]'s word. *)

  val log : t -> string list
  (** [log d] is [d]'s driver calls, oldest first: ["alloc"], ["free"],
      ["map_host"], ["unmap"] (a mapping's free), ["sleep"], ["stop"],
      ["image"], ["unload"]. *)

  val frees : t -> (int * int) list
  (** [frees d] is the address of each region [d] freed, oldest first, with
      [d]'s word when it was freed. *)

  val allocs : t -> ([ `Device | `Pinned | `Mapped ] * int * bool) list
  (** [allocs d] is each allocation asked of [d], oldest first: its kind, its
      bytes and whether [d] gave it. *)

  val host_maps : t -> int list
  (** [host_maps d] is the bytes of each host memory [d] mapped, oldest first.
  *)

  val allocated : t -> [ `Device | `Pinned | `Mapped ] -> int
  (** [allocated d kind] is the bytes of [kind] [d] holds allocated. *)

  val last_waits : t -> (int * int * int) list
  (** [last_waits d] is the waits [d]'s last submit received, at most 8, as
      [(kind, at, value)], [kind] one of {!rig_word} and {!rig_object}. *)

  val last_handles : t -> int list
  (** [last_handles d] is the handles [d]'s last submit received, at most 64, in
      the order received. A region's handle is its address. *)

  val blocked : t -> int
  (** [blocked d] is the number of submits waiting for room in [d]'s [may_block]
      queue. *)

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

val load : int -> int
(** [load a] reads the 64-bit word at the host address [a]. *)

val store : int -> int -> unit

val move : dst:int -> src:int -> int -> unit
(** [move ~dst ~src n] copies the [n] bytes at host address [src] to host
    address [dst]. *)

val host_kept : unit -> int
(** [host_kept ()] is the bytes the host keeps of collected buffers, for the
    next of their sizes. *)

val host_held : unit -> int
(** [host_held ()] is the bytes that count in the host's budget: host buffers
    not yet returned, and devices' pinned host memory. *)

val locked : (unit -> 'a) -> 'a
(** [locked f] is [f ()] run holding every lock of rig's core at once, as
    threads inside each would: what a fork during [f] leaves the child. [f] must
    not call rig, whose calls would wait for those locks. *)

val waiting : unit -> int
(** [waiting ()] is how many threads wait inside a lock of rig's core for
    another to change what it guards, such as a copy waiting for a staging slot.
*)

val shares : ('a, 'b, 'c) Bigarray.Array1.t -> int
(** [shares ba] is how many holders share [ba]'s storage, as the runtime counts
    them on the proxy its arrays share: [0] if they share none. *)

val await : string -> (unit -> bool) -> unit
(** [await what f] returns once [f ()] holds, yielding to other threads between
    checks. It raises [Failure] naming [what] after 10 s. *)

val machine : string -> Rig.t
(** [machine m] is the host of the machine named [m], an io device whose memory
    holds nothing, opened unless it is open: devices of [m] open once it is. *)

(** The C readers of [rig.h], called from C. *)
module Reader : sig
  val host : Rig.Buffer.t -> int
  (** [host b] is [rig_buffer_host b] as an integer, [0] for [NULL]. *)

  val bytes : Rig.Buffer.t -> int
  (** [bytes b] is [rig_buffer_bytes b]. *)

  val why : Rig.Buffer.t -> string option
  (** [why b] is [rig_buffer_why b], [None] for [NULL]. *)
end
