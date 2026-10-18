(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Memory keeps a record of each allocation, and returns the allocation
    to its driver once nothing uses it.

    Each allocation has one entry ({!Def.entry}), the record its owner gives
    back, which holds its stamps, its backing and the other devices'
    mappings of it. Each lifetime of the allocation has one {!Def.alloc},
    which holds its claim, and whose token puts the entry on its owner's
    release list once the lifetime is collected. Each device that reaches the
    allocation does so through a memory record ({!Def.memory}) over it: the
    owner's, and one per borrow. Host memory has no entry ({!no_entry})
    until a device borrows it ({!ensure_entry}), and its bytes return to the
    host's budget when their bigarray is collected.

    This module implements {!Rig.section-reclaim}. The life of a device's
    allocation goes like this:
    + {!alloc_entry} drains the device, then takes the memory from its cache,
      or from its driver within its budget, or reclaims ({!reclaiming}).
    + {!alloc} wraps the entry in a new lifetime, with its token, and the
      owner's memory record over it.
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
    count references: the entry that owns them holds one, and the
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
    {!iter_points} walks them, for an access that follows every use. [m]'s
    allocation stays reachable until it returns. *)

val wait_write : memory -> unit
(** [wait_write m] is {!Dev.wait_point} of each device's last write of [m]'s
    stamps, for an access that follows every last write. [m]'s allocation
    stays reachable until it returns. *)

val check_points : int -> unit
(** [check_points st] raises {!Dev.Lost} if a point of [st] is on a lost
    device and not done. *)

val check_owner : memory -> unit
(** [check_owner m] raises {!Dev.Lost} if [m]'s device or the device that
    owns the allocation is lost. *)

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
  device ->
  entry ->
  memory
(** [make d e] is the owner's memory record of a new lifetime of [e] on [d],
    with no token and a claim word whose bits come from [keep] and [e].
    [host] and [address] default to [-1], and [handle] to [0n]. *)

val size : alloc -> int
(** [size a] is the number of [a]'s bytes: its bigarray's for host memory,
    its entry's otherwise. *)

val locate : region -> Rig_edge.location
(** [locate r] is where [r] lies, as its driver states it. *)

val allocated : device -> int
(** [allocated d] is {!Rig.allocated}[ d]. *)

val account : alloc -> device
(** [account a] is the device in whose {!allocated} count [a]'s memory lies,
    or [a]'s owner for memory that counts in none, as a profile's copy names
    it ({!Rig.Profile.event}). *)

val note : device -> unit
(** [note d] records [d]'s allocated bytes in the profiles being taken. *)

val ensure_entry : alloc -> unit
(** [ensure_entry a] gives [a], host memory with {!no_entry}, an entry with
    stamps and a token, under its owner's lock, unless another call did.
    Once [a] is collected and its uses are settled, the token's release
    unmaps the entry's mappings. Raises {!Dev.Lost} if the owner's slot was
    freed. *)

val proxy : entry -> int
(** [proxy e] is the C proxy that bigarrays over [e]'s allocation share,
    made at the first. The allocation is neither reused nor freed while one
    is reachable. *)

(** {1:claims Claim words}

    An allocation's [count] is one word, so that a claim, a donation, a hold
    and an export are each one compare-and-set with one winner. A word from
    zero up holds three bits and two counts. Bit 0 ({!outside}) is set once
    something outside the claims reaches the memory: the bigarray
    {!Rig.Buffer.of_bigarray} was given, an io library's region, an array
    {!Rig.Buffer.export} made, or a value {!Rig.Claim.share} marked. Bit 1
    ({!read_only}) is set for memory that admits only reads. Bit 2 is set
    while a consumption of memory not held exclusive checks that no hold
    holds it and raises the generation, a few instructions in C, which a
    hold waits out. The read claims count in steps of {!one_claim} from bit
    3, and the holds ({!Rig.Claim.hold}) in steps of {!one_hold} from bit
    32. Only the word {!one_claim}, one claim and no bit or hold, may become
    {!exclusive}, and it is {!one_claim} again once the claims have consumed
    the memory. {!make} sets the bits. *)

val outside : int
(** [outside] is bit 0, set for memory outside the claims. *)

val admits : int -> access
(** [admits g] is what the buffers of the generation [g] admit: [Read] iff
    [g] is odd. The first generation of memory that admits only reads is 1,
    others' 0; a consumption raises the generation to the next even one
    where its claims held the memory exclusive, and to the next odd one
    otherwise ([caml_rig_claim_consume]). *)

val read_only : int
(** [read_only] is bit 1, set for memory that admits only reads. *)

val one_claim : int
(** [one_claim] is the step of one read claim. *)

val one_hold : int
(** [one_hold] is the step of one hold. *)

val held : alloc -> bool
(** [held a] is [true] iff a hold holds [a]'s memory. *)

val hold_token : holds -> token
(** [hold_token h] is a token that puts [h] on the list of holds once
    collected, whose holds the next {!drain} of any device then releases,
    those taken over memory still live. Raises [Stdlib.Out_of_memory] if the
    C heap has no room for it. *)

val no_hold : token
(** [no_hold] is the hold of no buffer, which releases nothing. *)

val exclusive : int
(** [exclusive] is the word of memory one {!Claim.with_} holds exclusive. *)

val claim_add : buffer -> bool
(** [claim_add b] adds a read claim to [b]'s memory, with no check of [b],
    unless the memory is held exclusive: whether it did. It tries one
    compare-and-set and retries in C, so in native code nothing in it polls
    or allocates, contended or not, and a sequence of takes needs no
    handler between them. *)

external claim_drop : buffer -> unit = "caml_rig_claim_drop" [@@noalloc]
(** [claim_drop b] ends a read claim on [b]'s memory, in C, so no poll point
    comes between a section's end and the claim's. *)

val mark_outside : alloc -> bool
(** [mark_outside a] marks [a]'s memory outside the claims for good. It is
    [false], changing nothing, while claims hold the memory exclusive. *)

(** {1:borrows Mappings and borrows}

    A borrow of an allocation on a device is a memory record of that device
    over the allocation's lifetime. Where the device needs a mapping, the
    first borrow makes it and later ones share it. The mapping is released
    when the entry is freed ({!free_entry}) or once the device is Stopped
    ({!drain}), whichever comes first, and the entry goes back to its driver
    after its last mapping is released. A device's mapping of rig's array,
    which its queue waits on, is released once the device is Stopped, and
    its mapping of another device's word once its queue passed the waits
    that may read it ({!section-words}). *)

val map_host_range : device -> int -> int -> region option
(** [map_host_range d p n] is [d]'s mapping of the [n] bytes of host memory
    at [p], or [None] where [d] maps none. *)

val map_peer_region : device -> region -> region option
(** [map_peer_region d r] is [d]'s mapping of [r], a region of another
    device of [d]'s driver, or [None] for another driver's region or where
    [d] maps none. *)

val unmap : device -> region -> Dev.given
(** [unmap d r] gives back [d]'s mapping [r], which counts in no memory, now,
    or later among [d]'s pending releases, once [d] stopped. It is what
    {!Dev.give} answered. *)

(** {2:words Mappings of words}

    A device counts the mappings of its word that other devices hold, in
    their pairs ({!Def.live.pairs}) or among their retired mappings
    ({!Def.live.retired}), plus one until its slot is freed
    ({!Def.live.word_unmaps}). A word of its driver's own goes back to the
    driver at the first drain ({!drain}) after the count reached [0], when
    no queue can wait on it: a submit gives back mappings of words, and must
    not wait for a driver's free. *)

val undecided : pair
(** [undecided] is the pair of no device: its [occupant] is [-1], it waits
    on the host and holds no mapping. *)

val hold_word : device -> bool
(** [hold_word p] counts a mapping of [p]'s word about to be made, and is
    [false], counting nothing, once [p]'s count reached [0]: [p] has
    stopped and its points are done. *)

val word_unmapped : device -> unit
(** [word_unmapped p] takes one off [p]'s count, and at [0], if [p]'s word
    is its driver's own, leaves the word to the next drain. It calls no
    driver. *)

val unmap_word : device -> region * device -> Dev.given
(** [unmap_word d (r, p)] gives back [d]'s mapping [r] of [p]'s word now, or
    later among [d]'s retired mappings, then {!word_unmapped}[ p]. It is
    what {!Dev.give} answered. *)

val retire_locked : device -> region * device -> unit
(** [retire_locked d m] puts [d]'s mapping [m] of a producer's word among
    [d]'s retired mappings at {!Dev.next_value}[ d]: [m] goes back once
    [d]'s queue ran every submit that may wait through it. The caller holds
    [d]'s lock. *)

val give_retired : device -> unit
(** [give_retired d] gives back [d]'s retired mappings whose value [d]
    reached, up to the first that does not go back. It frees no memory, and
    leaves a word that loses its last mapping to the next drain
    ({!word_unmapped}). *)

val borrow : device -> memory -> memory option
(** [borrow d m] is [d]'s borrow of [m]'s allocation, or [None] where [d]
    cannot map it. On the allocation's owner it is [m] if [m] is the owner's
    record, and otherwise a record with the coordinates the owner's record
    was made with. An io allocation borrows through its pages, and raises
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
(** [drain d] releases the collected holds ({!hold_token}), moves [d]'s
    collected memory whose points are settled to its cache or its driver,
    returns a Stopped device's cache and gives back its mappings of other
    devices' memory and words, and runs the pending releases whose value [d]
    reached, which let go of the values that [d]'s collected submissions
    kept. On an orphaned device it only routes [d]'s release list and lets
    go of the kept values pending on it, since its frees would call its
    parent's driver. It may raise {!Dev.Lost} when the read of a word with
    no host address loses its device. *)

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

val free_entry : entry -> Dev.given
(** [free_entry e] gives [e]'s region back to its driver, and its bytes to
    their keeper, once no other device maps its allocation. Each mapping is
    released once its mapper's work submitted until now is done, and the
    last release gives [e] back. A release that does not go back waits
    among its device's pending releases. *)

val retire : device -> entry -> unit
(** [retire d e] frees [e] once [d] has reached the value it has submitted now,
    or, if [d] is lost, once it is Stopped. *)

val unload : device -> loaded -> entry option -> unit
(** [unload d i code] releases what [d]'s driver made for [i] ({!Dev.give}),
    then gives back [i]'s code memory [code]. Both wait among [d]'s pending
    releases, as one, until [d] is Stopped if it is lost and not
    Stopped. *)

val grow_queues : device -> round:int -> lacked:int -> int * int
(** [grow_queues d ~round ~lacked] grows [d]'s queues for the parts its
    driver's room answered [RIG_GROW] for on this thread, a counted call
    ({!Rig_edge.Driver.grow}), as ask [round] of a submission's ladder
    ({!reclaiming}), [1] for its first, after the driver last lacked
    [lacked] bytes, [0] for none. It is the round and the lacked bytes of
    the submission's next ask: [round + 1] either way, after a round of the
    ladder for [d]'s budget and the host's when the driver had no memory for
    the growth. The caller asks room again after it. Raises
    {!Dev.Out_of_memory} once the driver has no memory for it at the last
    round, or if [round] is past the last, and [Invalid_argument] if [d] is
    the host or an io device. *)

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
(** [alloc d kind n] is the owner's memory record of a new lifetime of
    [alloc_entry d kind n], with its token. Raises {!Dev.Lost}, giving the
    entry back, if [d]'s slot was freed meanwhile. *)

val host_memory : int -> memory
(** [host_memory n] is a new memory record of [n] bytes of the host's heap,
    counted in the host's budget, after draining the host. Raises
    {!Dev.Out_of_memory} once the host's reclaim rounds fail. *)

val set_budget : device -> int -> unit
(** [set_budget d n] is {!Rig.set_budget}[ d n]. *)

val free_cache : device -> unit
(** [free_cache d] is {!Rig.free_cache}[ d]. *)
