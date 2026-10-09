(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Memory records, their stamps, the devices' caches, release lists and drains,
    and the mappings of borrows.

    Any domain may call any function. A device's lock guards its cache and
    lists; no driver call and no read of a word runs under it, since a word
    behind a transport is read through its driver, whose fault stops the device,
    and a stop drains.

    Every object goes back to its driver through {!Dev.give}: counted on a live
    device, so it raises {!Dev.Lost} if the device is or becomes lost; on a
    lost device once it is Stopped, uncounted, dropping its failures; and on an
    orphaned device not at all. Lost devices whose stop returned
    ({!Dev.ended}) are walked by every drain, for the life of the process. *)

open Def

val page : int
(** [page] is the host's page size, in bytes. *)

(** {1:stamps Stamps}

    Stamps are the address of a C record ([struct rig_stamps] in [rig_stubs.h])
    as an int, [0] for none. They count references: each memory and hold that
    shares them holds one, so does each memory linked to a hold's, and the last
    {!stamps_unref} frees them. Nothing checks a use after that. A memory's
    points are its stamps' and, once it is in a hold, those of the hold's
    stamps, to which its own link. The stubs never block or release the
    runtime. *)

val stamps_new : unit -> int
(** [stamps_new ()] is new empty stamps with one reference. Raises
    [Stdlib.Out_of_memory] if memory runs out. *)

val stamps_ref : int -> unit
val stamps_unref : int -> unit

val stamps_hold : int -> int -> unit
(** [stamps_hold st h] links [st], the stamps of memory in no hold, to the
    hold's stamps [h]. *)

val held : int -> bool
(** [held st] is [true] iff [st] links to a hold's stamps. *)

val iter_points : (int -> unit) -> int -> unit
(** [iter_points f st] is [f] over the points of [st], the last write first,
    then those of the hold's stamps [st] links to, none for [0]. It allocates
    nothing. *)

val iter_write : (int -> unit) -> int -> unit
(** [iter_write f st] is [f] of [st]'s last write, if any. It allocates nothing.
*)

val check_points : int -> unit
(** [check_points st] raises {!Dev.Lost} if a point of [st] is on a lost device
    and not done. *)

val check_owner : memory -> unit
(** [check_owner m] raises {!Dev.Lost} if [m]'s memory is a lost device's, or
    [m] is a lost device's borrow: the memory a lost device owns or maps. *)

val check : memory -> unit
(** [check m] is {!check_owner} [m], then {!check_points} of [m]'s stamps: for
    a use of [m] that follows every point. *)

(** {1:records Records} *)

type bytes_ba =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

val ba_address : ('a, 'b, 'c) Bigarray.Array1.t -> int

val token : int -> released -> int -> int -> int -> token
(** [token list record bytes room live] puts [record] on the release list [list]
    once collected, in whichever domain sweeps it, pacing the collector by
    [bytes] out of [room] ([live] for memory the host shares, [-1] otherwise).
    [record] must not reach the token, which would then never be collected.
    Raises [Stdlib.Out_of_memory] if memory runs out. *)

val holds_list : int
(** [holds_list] is the release list of holds, which every drain reads. *)

val entry :
  ?region:region ->
  ?io_region:io_region ->
  ?access:access ->
  device ->
  memory_kind ->
  int ->
  int ->
  entry
(** [entry owner kind bytes st] is a new release record of [bytes] bytes of
    [owner]'s memory of [kind], whose stamps are [st], in no hold and mapped by
    no device. [access] defaults to [Read_write]. *)

val no_entry : entry
(** [no_entry] is the entry of host memory no device borrowed: no stamps, no
    mapping. It is shared: nothing writes it. *)

(** {1:claims Claim words}

    A claim's [count] is one word, so that a claim, a donation and an export are
    each one compare-and-set with one winner. From zero up, bit 0 ({!outside})
    is set once something outside the claims reaches the memory: the bigarray
    [Buffer.of_bigarray] was given, an io library's region, an array
    [Buffer.bigarray] made. Bit 1 ({!read_only}) is set for [Read] memory. The
    read claims count in steps of {!one_claim}. Only the word {!one_claim}, one
    claim and no bit, becomes {!exclusive}, then {!consumed} once the claims
    consumed the memory, then {!exported} once the consumer exported it. {!make}
    sets the bits. *)

val outside : int
val read_only : int
val one_claim : int
val exclusive : int
val consumed : int
val exported : int

val export : claim -> bool
(** [export c] marks [c]'s memory outside the claims for good, and is [false],
    changing nothing, while claims that have not consumed it hold it exclusive.
*)

val make :
  ?keep:keep ->
  ?host:int ->
  ?address:int ->
  ?handle:nativeint ->
  ?token:token ->
  device ->
  int ->
  entry ->
  memory
(** [make d n e] is a new memory record of [n] bytes on [d] over [e], its own
    root, with a claim of its own. [host] and [address] default to [-1],
    [handle] to [0n] and [token] to none. *)

val stamps : memory -> int
(** [stamps m] is the stamps of the memory [m] lies in, below its borrows. *)

val region_info : region -> int * nativeint * int
(** [region_info r] is [r]'s address, handle and host address, [-1] for none. *)

val note : device -> unit
(** [note d] records [d]'s allocated bytes in the profiles being taken. *)

val ensure_entry : memory -> unit
(** [ensure_entry m] gives [m], host memory with {!no_entry}, stamps and a
    token, under [m]'s device's lock, unless another call did. Once [m] is
    collected and its uses are reached, the token's release unmaps its mappings.
*)

val proxy : entry -> int
(** [proxy e] is the C proxy that bigarrays over [e]'s memory share, made at the
    first: [e]'s memory is neither reused nor freed while one is reachable. *)

(** {1:borrows Mappings and borrows} *)

val map_host_range : device -> int -> int -> region option
(** [map_host_range d p n] is [d]'s mapping of the [n] bytes of host memory at
    [p], or [None] where [d] maps none. *)

val map_peer_region : device -> region -> region option
(** [map_peer_region d r] is [d]'s mapping of [r], a region of another device of
    [d]'s driver, or [None] for another driver's region or where [d] maps none.
*)

val borrow : device -> memory -> memory option
(** [borrow d m] is [d]'s borrow of [m]'s memory, or [None] where [d] cannot map
    it. An io memory borrows through its pages, and raises what asking for them
    raises. *)

val maps : device -> memory -> bool
(** [maps d m] is whether [d] can borrow [m]'s memory, as {!borrow} says, making
    [d]'s mapping of it if it needs one. It builds no borrow. *)

val prefetch : device -> memory -> at:int -> len:int -> unit
(** [prefetch d m ~at ~len] asks the io device of [m] to read [len] bytes of it
    from [at] ahead, if [d] is no host. It raises nothing. *)

val of_io : device -> io_region -> access:access -> int -> memory
(** [of_io d r ~access n] is a memory record over [n] bytes of the region [r] an
    io library gave the io device [d], admitting [access], after draining [d].
*)

(** {1:allocation Allocation and drains} *)

val drain : device -> unit
(** [drain d] returns the memory of [d]'s collected buffers, runs the releases
    due on it and on lost devices, and the holds' releases due, raising the
    first exception a hold's release raised once all ran. It forgets the holds
    made before a fork. It does nothing on an orphaned device, whose frees would
    call its parent's driver. *)

val reclaiming : device -> pool:device -> int -> (unit -> 'a option) -> 'a
(** [reclaiming d ~pool n f] is the out-of-memory ladder for [n] bytes of [d]
    once [f], a try at them, answered [None]: [pool] is whose budget refused,
    the host for memory the host's budget counts, [d] otherwise. It runs a
    round, then [f], whose first [Some] answer is the result, up to three
    rounds. A round: every cached memory that counts in [pool]'s budget, on any
    device, returns once its device's handed work is done, and for the host its
    kept buffers too; every device drains, the host included; from the second
    round a full major collection runs and every device drains again. Raises
    {!Dev.Out_of_memory}[ (d, n)] when [f] answers [None] after the last round.
    A caller tries first and enters the ladder on a refusal, so a try that
    succeeds builds no closure. *)

val room : device -> int
(** [room d] is the bytes left in [d]'s budget, [0] if it holds more. *)

val free_entry : entry -> unit
(** [free_entry e] gives [e]'s region back to its driver, and its bytes to their
    keeper, once no other device maps its memory: each mapping is released once
    its mapper's work submitted until now is done, and the last release gives
    [e] back. *)

val retire : device -> entry -> unit
(** [retire d e] frees [e] once [d] reached the value it has submitted now, or,
    lost, once it counts as stopped. *)

val unload : device -> loaded -> unit
(** [unload d i] releases what [d]'s driver made for [i] ({!Dev.give}), once
    [d] stopped if it is lost and not stopped. *)

val kernel_entry : image -> string -> Rig_edge.entry option
(** [kernel_entry i f] is the driver's entry for [i]'s function [f], a counted
    call ({!Rig_edge.Driver.entry}). *)

val alloc_entry : device -> memory_kind -> int -> entry
(** [alloc_entry d kind n] allocates [n] bytes of [d]'s memory of [kind] on the
    allocation path: drains, the cache, the budget, the reclaim rounds, then
    {!Dev.Out_of_memory}. *)

val alloc : device -> memory_kind -> int -> memory
(** [alloc d kind n] is a memory record over [alloc_entry d kind n], with its
    token. *)

val host_memory : int -> memory
(** [host_memory n] is a new memory record of [n] bytes of the host's heap,
    counted in the host's budget, after draining the host. Raises
    {!Dev.Out_of_memory} once the host's reclaim rounds fail. *)

val set_budget : device -> int -> unit
(** {!Rig.set_budget}. *)

val free_cache : device -> unit
(** {!Rig.free_cache}. *)
