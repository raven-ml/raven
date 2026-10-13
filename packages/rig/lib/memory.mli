(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Memory keeps a record of each allocation, and returns the allocation
    to its driver once nothing uses it.

    Each allocation has one entry ({!Def.entry}), the record its owner gives
    back, which holds its stamps, its backing and the other devices'
    mappings of it. Each device that reaches the allocation does so through a
    memory record ({!Def.memory}): the owner's, which is its own root, and
    one per borrow, whose root is the owner's. A token in the root puts the
    entry on its device's release list once the root is collected. Host
    memory has no entry ({!no_entry}) until a device borrows it
    ({!ensure_entry}), and its bytes return to the host's budget when their
    bigarray is collected.

    This module implements {!Rig.section-reclaim}. The life of a device's
    allocation goes like this:
    + {!alloc_entry} drains the device, then takes the memory from its cache,
      or from its driver within its budget, or reclaims ({!reclaiming}).
    + {!alloc} wraps the entry in a memory record with its token.
    + A borrow on another device maps it ({!borrow}). The entry lists each
      mapping.
    + Once the record is collected, a {!drain} takes the entry off the
      release list. The entry enters the cache if it is a driver's region,
      its device is live and holds no more than its budget, every point of
      its stamps has settled, and no bigarray reads it. Otherwise it waits
      on the device's retiring list, which each later drain judges again.
    + {!free_entry} releases each mapping once its mapper's work submitted
      until then is done, then gives the region back to its driver and its
      bytes to the budget.

    Any domain may call any function. A device's lock guards its cache and
    lists. No driver call and no read of a word runs under it. A word with
    no host address is read through its driver, a fault there stops the
    device, and a stop drains.

    Every object goes back to its driver through {!Dev.give}. What a device
    lost and not yet Stopped would give back waits on its pending list
    ([pending] in {!Def.live}) until it is Stopped. A device's collected
    memory moves in a drain of that device: its own calls', its stop's, or
    a round of the ladder's ({!reclaiming}), which drains every device.
    {!Rig.section-reclaim} states the model a caller sees. *)

open Def

val page : int
(** [page] is the host's page size, in bytes. *)

(** {1:stamps Stamps}

    Stamps are the address of a C record ([struct rig_stamps] in
    [rig_stubs.h]) as an int, [0] for none. They hold, per device, the
    point of the allocation's last write, of its last use and of its last
    unordered use, which no access follows and the allocation's
    return waits for. A device's words pass to another device once its stop
    was claimed and their points are done, so the words, and a walk's cost,
    follow the devices whose work on the allocation may still matter. They
    count references: the entry or hold that owns them holds one, and the
    last {!stamps_unref} frees them. Nothing checks a use after that. The
    stubs never block or release the runtime. *)

val stamps_new : unit -> int
(** [stamps_new ()] is new empty stamps with one reference. Raises
    [Stdlib.Out_of_memory] if the C heap has no room for them. *)

val stamps_ref : int -> unit
(** [stamps_ref st] adds a reference to [st]. *)

val stamps_unref : int -> unit
(** [stamps_unref st] drops a reference to [st], and frees [st] with the
    last. *)

val iter_points : (int -> unit) -> int -> unit
(** [iter_points f st] is [f] over the points of [st], each device's last
    use, which a write also raises; none for [0]. It allocates nothing. *)

val wait_points : memory -> unit
(** [wait_points m] is {!Dev.wait_point} of each point of [m]'s stamps, as
    {!iter_points} walks them, for an access that follows every use. [m]
    stays reachable until it returns. *)

val wait_write : memory -> unit
(** [wait_write m] is {!Dev.wait_point} of each device's last write of [m]'s
    stamps, for an access that follows every last write. [m] stays
    reachable until it returns. *)

val check_points : int -> unit
(** [check_points st] raises {!Dev.Lost} if a point of [st] is on a lost
    device and not done. *)

val check_owner : memory -> unit
(** [check_owner m] raises {!Dev.Lost} if [m]'s device is lost, or, for a
    borrow, if the device that owns the allocation is lost. *)

val check : memory -> unit
(** [check m] is {!check_owner}[ m], then {!check_points} of [m]'s stamps,
    for a use of [m] that depends on every point. *)

(** {1:records Records} *)

type bytes_ba =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
(** The type for byte bigarrays. *)

val ba_address : ('a, 'b, 'c) Bigarray.Array1.t -> int
(** [ba_address a] is the address of [a]'s first byte. *)

val token : int -> released -> int -> int -> int -> token
(** [token list record bytes room live] puts [record] on the release list
    [list] once collected, in whichever domain sweeps it, pacing the
    collector by [bytes] out of [room] ([live] for memory the host shares,
    [-1] otherwise). [record] must not reach the token, which would then
    never be collected. It is [no_token], putting nothing anywhere, if the
    list is a device's whose slot was freed or is being freed: a token on it
    could outlive the device's record. Raises [Stdlib.Out_of_memory] if the C
    heap has no room for it. *)

val no_token : token
(** [no_token] is no token: one that puts nothing on any list. *)

val holds_list : int
(** [holds_list] is the release list of holds, which every drain reads. *)

val entry : ?access:access -> device -> backing -> int -> int -> entry
(** [entry owner backing bytes st] is a new entry of [bytes] bytes of
    [owner]'s memory, whose stamps are [st], mapped by no device. [access]
    defaults to [Read_write]. *)

val no_entry : entry
(** [no_entry] is the entry of host memory no device borrowed: no stamps,
    no mapping. It is shared, and nothing writes it. *)

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
    root, with a claim of its own whose bits come from [keep] and [e]. [host]
    and [address] default to [-1], [handle] to [0n] and [token] to none. *)

val root : memory -> memory
(** [root m] is [m]'s root: the owner's record for a borrow, [m] itself for
    the owner's. *)

val stamps : memory -> int
(** [stamps m] is the stamps of the allocation [m] lies in, read from its
    root's entry. *)

val locate : region -> Rig_edge.location
(** [locate r] is where [r] lies, as its driver states it. *)

val allocated : device -> int
(** [allocated d] is {!Rig.allocated}[ d]. *)

val note : device -> unit
(** [note d] records [d]'s allocated bytes in the profiles being taken. *)

val ensure_entry : memory -> unit
(** [ensure_entry m] gives [m], host memory with {!no_entry}, an entry with
    stamps and a token, under [m]'s device's lock, unless another call did.
    Once [m] is collected and its uses are settled, the token's release
    unmaps the entry's mappings. Raises {!Dev.Lost} if the device's slot was
    freed. *)

val proxy : entry -> int
(** [proxy e] is the C proxy that bigarrays over [e]'s allocation share,
    made at the first. The allocation is neither reused nor freed while one
    is reachable. *)

(** {1:claims Claim words}

    A claim's [count] is one word, so that a claim, a donation and an export
    are each one compare-and-set with one winner. A word from zero up holds
    two bits and a count. Bit 0 ({!outside}) is set once something outside
    the claims reaches the memory: the bigarray {!Rig.Buffer.of_bigarray}
    was given, an io library's region, an array {!Rig.Buffer.export} made,
    or a value {!Rig.Claim.share} marked. Bit 1 ({!read_only}) is set for
    memory that admits only reads. The read claims count in steps of
    {!one_claim}. Only the word {!one_claim}, one claim and no bit, may
    become {!exclusive}, and it is {!one_claim} again once the claims have
    consumed the memory. {!make} sets the bits. *)

val outside : int
(** [outside] is bit 0, set for memory outside the claims. *)

val read_only : int
(** [read_only] is bit 1, set for memory that admits only reads. *)

val one_claim : int
(** [one_claim] is the step of one read claim. *)

val exclusive : int
(** [exclusive] is the word of memory one {!Claim.with_} holds exclusive. *)

val mark_outside : claim -> bool
(** [mark_outside c] marks [c]'s memory outside the claims for good. It is
    [false], changing nothing, while claims hold the memory exclusive. *)

(** {1:borrows Mappings and borrows}

    A borrow of an allocation on a device is a memory record whose root is
    the owner's. Where the device needs a mapping, the first borrow makes it
    and later ones share it. The mapping is released when the entry is freed
    ({!free_entry}) or once the device is Stopped ({!drain}), whichever
    comes first, and the entry goes back to its driver after its last
    mapping is released. A device's mapping of another's word, which its
    queue waits on, is released once the device is Stopped. *)

val map_host_range : device -> int -> int -> region option
(** [map_host_range d p n] is [d]'s mapping of the [n] bytes of host memory
    at [p], or [None] where [d] maps none. *)

val map_peer_region : device -> region -> region option
(** [map_peer_region d r] is [d]'s mapping of [r], a region of another
    device of [d]'s driver, or [None] for another driver's region or where
    [d] maps none. *)

val unmap : device -> region -> unit
(** [unmap d r] gives back [d]'s mapping [r], which counts in no memory, now
    or once [d] stopped. *)

val borrow : device -> memory -> memory option
(** [borrow d m] is [d]'s borrow of [m]'s allocation, or [None] where [d]
    cannot map it. An io allocation borrows through its pages, and raises
    what asking for them raises. *)

val maps : device -> memory -> bool
(** [maps d m] is whether [d] can borrow [m]'s allocation, as {!borrow}
    says, making [d]'s mapping of it if it needs one. It builds no
    borrow. *)

val prefetch : device -> memory -> at:int -> len:int -> unit
(** [prefetch d m ~at ~len] asks the io device of [m] to read [len] bytes of
    it from [at] ahead, unless [d] is the host. It drops the io library's
    faults and [Sys_error]. *)

val of_io : device -> io_region -> access:access -> int -> memory
(** [of_io d r ~access n] is a memory record over [n] bytes of the region [r]
    an io library gave the io device [d], admitting [access], after draining
    [d]. Its memory starts outside the claims and counts in no budget.
    Raises {!Dev.Lost} if [d] is lost. *)

(** {1:allocation Allocation and drains}

    An allocation that its budget or its driver refuses enters the
    out-of-memory ladder ({!reclaiming}), which runs up to three rounds and
    tries again after each. In a round, on every device whose lock is free,
    cached memory that counts in the refusing budget returns once the work
    submitted on the device until then is done, waiting for that work. A
    device whose submitted work holds back the return of memory that counts
    in the budget is waited for too. For the host's budget, the host's kept
    buffers return as well. Then every device drains, and from the second
    round a full major collection runs and every device drains again. *)

val drain : device -> unit
(** [drain d] moves [d]'s collected memory whose points are settled to its
    cache or its driver, returns a Stopped device's cache and gives back its
    mappings of other devices' memory and words, runs the pending
    releases whose value [d] reached, and releases the holds that are due.
    In a forked child it forgets the holds made before the fork. It does
    nothing on an orphaned device, whose frees would call its parent's
    driver. It may raise {!Dev.Lost} when the read of a word with no host
    address loses its device. *)

val reclaiming : device -> pool:device -> int -> (unit -> 'a option) -> 'a
(** [reclaiming d ~pool n f] is the first [Some] result of [f], a try at [n]
    bytes of [d], after each round of the ladder, once a try has returned
    [None]. [pool] is the device whose budget refused: the host for memory
    the host's budget counts, [d] otherwise. A caller tries first and enters
    the ladder on a refusal, so a try that succeeds builds no closure.
    Raises {!Dev.Out_of_memory}[ (d, n)] when [f] returns [None] after the
    last round. *)

val room : device -> int
(** [room d] is the bytes left in [d]'s budget, [0] if it holds more. *)

val free_entry : entry -> unit
(** [free_entry e] gives [e]'s region back to its driver, and its bytes to
    their keeper, once no other device maps its allocation. Each mapping is
    released once its mapper's work submitted until now is done, and the
    last release gives [e] back. *)

val retire : device -> entry -> unit
(** [retire d e] frees [e] once [d] has reached the value it has submitted now,
    or, if [d] is lost, once it is Stopped. *)

val unload : device -> loaded -> unit
(** [unload d i] releases what [d]'s driver made for [i] ({!Dev.give}), once
    [d] is Stopped if it is lost and not Stopped. *)

val kernel_entry : image -> string -> Rig_edge.entry option
(** [kernel_entry i f] is the driver's entry for [i]'s function [f], a
    counted call ({!Rig_edge.Driver.entry}), or [None] if [i] has no [f]. It
    keeps [i] reachable until the driver returns, since [i]'s collection
    unloads the image. A driver with no memory for [f] is asked again after
    each round of the ladder ({!reclaiming}). Raises {!Dev.Out_of_memory} if
    it still has none after the last. *)

val alloc_entry : device -> Rig_edge.memory -> int -> entry
(** [alloc_entry d kind n] is a new entry of [n] bytes of [d]'s memory of
    [kind]. It drains [d], then takes the memory from [d]'s cache or from
    its driver within the budget the memory counts in, and on a refusal runs
    {!reclaiming}. [Mapped] memory larger than [d]'s budget, or that a try
    refuses, is [Pinned] memory instead. Raises {!Dev.Out_of_memory}
    immediately if [n] exceeds the budget the memory counts in. *)

val alloc : device -> Rig_edge.memory -> int -> memory
(** [alloc d kind n] is a memory record over [alloc_entry d kind n], with
    its token. Raises {!Dev.Lost}, giving the entry back, if [d]'s slot was
    freed meanwhile. *)

val host_memory : int -> memory
(** [host_memory n] is a new memory record of [n] bytes of the host's heap,
    counted in the host's budget, after draining the host. Raises
    {!Dev.Out_of_memory} once the host's reclaim rounds fail. *)

val set_budget : device -> int -> unit
(** [set_budget d n] is {!Rig.set_budget}[ d n]. *)

val free_cache : device -> unit
(** [free_cache d] is {!Rig.free_cache}[ d]. *)
