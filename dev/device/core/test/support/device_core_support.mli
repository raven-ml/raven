(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Test drivers and probes for the core's suites. *)

(** A driver over host memory whose queue runs only when its {!run} or its sleep
    runs it: a wait that returns before it slept leaves work unrun. Every driver
    call is logged. *)
module Polled : sig
  include Device_core.Driver

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
      none). Its stop answers [answer] (defaults to [`Stopped]). *)

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
    ?answer:[ `Stopped | `Unknown ] ->
    string ->
    Device_core.t * t
  (** [open_ name] opens a fresh device named [name]. *)

  val run : t -> int
  (** [run d] runs the queued submissions whose waits hold: how many ran. *)

  val queued : t -> int
  val submits : t -> int

  val fail : t -> unit
  (** [fail d] makes [d]'s next submit fail. *)

  val fault : t -> string -> unit
  (** [fault d why] makes [d]'s sleeps, allocations, mappings and loads raise
      [Fault why] from now on, as a faulted device's do. *)

  val set_word : t -> int -> unit
  (** [set_word d v] writes [v] into [d]'s word. *)

  val log : t -> string list
  (** [log d] is [d]'s driver calls, oldest first: ["alloc"], ["free"],
      ["map_host"], ["unmap"], ["sleep"], ["stop"], ["image"], ["unload"]. *)

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
      [(kind, at, value)], [kind] one of {!nx_word} and {!nx_object}. *)

  val blocked : t -> int
  (** [blocked d] is the number of submits waiting for room in [d]'s [may_block]
      queue. *)

  (** {1:sleeps Sleeps}

      Seams that hold a wait at a known point: each acts on the sleeps that
      follow, in the order a sleep checks them: the gate, a fault, an
      interruption, a stall. *)

  val gate : t -> unit
  (** [gate d] makes [d]'s sleeps block until {!open_gate}. *)

  val open_gate : t -> unit
  (** [open_gate d] lets [d]'s blocked sleeps go on. *)

  val sleepers : t -> int
  (** [sleepers d] is the number of [d]'s sleeps blocked at its gate. *)

  val interrupt : t -> unit
  (** [interrupt d] makes [d]'s next sleep raise SIGINT in its thread and return
      without running the queue. *)

  val stall : t -> int -> unit
  (** [stall d n] makes [d]'s next [n] sleeps return after their still interval
      without running the queue, as over work that runs long. *)
end

val nx_word : int
(** [nx_word] is [nx_edge.h]'s [NX_WORD]. *)

val nx_object : int
(** [nx_object] is [nx_edge.h]'s [NX_OBJECT]. *)

val bump : nativeint
(** [bump] is a fill adding 1 to the 64-bit word its argument points at. *)

val poke : nativeint
(** [poke] is a fill storing the second 64-bit word of its argument at the
    address its first holds. *)

val load : int -> int
(** [load a] reads the 64-bit word at the host address [a]. *)

val store : int -> int -> unit

val await : string -> (unit -> bool) -> unit
(** [await what f] returns once [f ()] holds, yielding to other threads between
    checks. It raises [Failure] naming [what] after 10 s. *)

(** The C readers of [device_core.h], called from C. *)
module Reader : sig
  val host : Device_core.Buffer.t -> int
  (** [host b] is [device_core_buffer_host b] as an integer, [0] for [NULL]. *)

  val bytes : Device_core.Buffer.t -> int
  (** [bytes b] is [device_core_buffer_bytes b]. *)

  val why : Device_core.Buffer.t -> string option
  (** [why b] is [device_core_buffer_why b], [None] for [NULL]. *)
end
