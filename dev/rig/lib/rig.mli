(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Devices, their memory, and the order of work across them.

    A {e device} is memory and, for most, a processor that runs work on it: the
    {!host}, a GPU that a {e driver} opened ({!Driver}), or a store of bytes the
    host reads and writes, which runs no work ({!Io}). A device's work is one
    sequence of {e submissions}, numbered [1], [2], … : the {e values} of its
    {e timeline}. Value [v] is {e reached} once every submission up to [v]
    completed. A {e point} ({!Point}) names a device and a value.

    Memory returns by the timeline. Each memory records the point of its last
    write and, per device, the point of its last use: its {e stamps}. Work that
    reads memory waits for its last write; work that writes it, and a reuse of
    the memory, wait for every use. A wait on the work of the device itself is
    the device's own order; a wait on another device's work is a {e foreign}
    wait, which the device's queue makes itself where its driver can, and the
    host makes before the submission otherwise.

    {v
      Buffer.t ──view, borrow──> memory ──stamps──> points
          │                         ▲                  ▲
          │ submit ~reads ~writes   │ Hold.make        │ submit
          ▼                         │                  │
      Submission.t ──────────── one device ── Driver.submit_entry (C)
    v}

    A program opens devices ({!open_}), allocates {!Buffer}s, and copies between
    them ({!Buffer.copy}), which waits for the work that touched them. A library
    that runs compiled work on devices prepares {!Submission}s once and submits
    each run's buffers with them ({!submit}); it orders its own host access with
    {!Buffer.wait}. A vendor library matches {!Driver} without linking this
    library; a program passes it to {!open_} with the library's function that
    opens the hardware.

    {1:domains Domains}

    Every function may be called from any domain, at the same time as others. No
    function but {!open_} holds a lock while it waits: {!open_} holds its name's
    lock while the opener runs, so opens of one name run one at a time and opens
    of other names go on. A copy through the host's staging memory
    ({!Buffer.copy}) holds a staging slot while it waits for devices, and a copy
    that finds both slots held waits for one. A wait releases the domain lock
    while it blocks and returns to OCaml at least every still interval (200 ms),
    so a pending [Sys.Break] from Ctrl-C raises there; an interrupted wait loses
    no device and assigns no value.

    {1:loss Loss}

    A device whose driver reports a fault, or whose hand-over fails, is
    {e lost}, once and for good ({!Lost}). Work that waits in its queue on a
    lost device's unreached values is lost with it. Every later use of the
    device that needs its driver, and of memory whose stamps name it, raises
    {!Lost}; other devices go on. A lost device's facts still answer: {!name},
    {!arch}, {!budget} and {!capability} as before, {!submitted} the last value
    handed over, and {!signaled} the last value its word showed, which may stop
    moving. A lost device's memory returns only once its timeline word shows its
    last submitted value reached, which it may never do: such memory is kept for
    the life of the process. Opening the device again makes a new device.

    {1:reclaim Reclamation}

    Nothing frees a buffer by hand. Once a buffer and its views are unreachable,
    the collector hands its memory back to its device, even while the domain
    that made them is blocked. The device keeps it in a cache for reuse once
    every other device's last use of it is reached, and gives it back to its
    driver once its own work is done too. A {e drain} of a device takes back the
    memory of its buffers collected since, and runs the release of every hold
    that became due ({!Hold}). {!Buffer.create}, {!Buffer.of_io} and
    {!Buffer.copy} drain the devices they use first. An allocation of more than
    the device's {!budget} raises {!Out_of_memory} at once and keeps the cache.
    Another that the budget or the driver refuses waits for the device's
    submitted work, releases its cache, and for memory the host's budget counts
    every device's cached memory it counts, drains every other device, collects
    unreachable buffers from its second try on, and tries again, four tries in
    all, before it raises {!Out_of_memory}.

    The host keeps the memory of collected buffers of 64 KiB or more in a cache
    for the next buffers of their sizes. It returns what the cache holds beyond
    a major cycle's share of the program's memory, or beyond 32 MiB where that
    is more, as it keeps a buffer and at the end of each major cycle.

    The host memory of buffers of 64 KiB or more paces the collector's major
    cycles by the program's memory: its OCaml heap and the host memory its live
    buffers hold. A cycle is due once the memory allocated since the last one
    reaches [custom_major_ratio / 150] of it ({!Gc.control}), 29% by default.
    Unreachable buffers then hold at most three such shares, 88% of the
    program's memory by default. The memory of another device's buffers paces
    major cycles by the room left in that device's budget, the same share of it,
    and at least a page: a device that fills up runs cycles more often before it
    refuses an allocation. Smaller host buffers pace the collector as any
    bigarray does. *)

(** {1:devices Devices} *)

type t
(** The type for devices. A device stays the same value while it is open; an
    open of a device's name after its loss makes a new value, unequal to the
    lost one. *)

val host : t
(** [host] is this process's host, named ["CPU"]. Its memory is the process's
    heap; its work is the process's own code, which needs no submission. *)

val name : t -> string
(** [name d] is [d]'s name: the name its opener gave it, followed by
    ["@MACHINE"] for a device of another machine. It is for people; nothing
    parses it. *)

val host_of : t -> t
(** [host_of d] is the host of [d]'s machine: {!host} for this machine's
    devices, the device {!open_io} opened with [~host:true] for another
    machine's. A host is its own. *)

val arch : t -> string
(** [arch d] is the architecture of [d]'s processor, as its driver names it,
    such as ["gfx1100"], ["sm_89"] or ["Apple7"]; the machine's instruction set
    for a host, ["arm64"] or ["x86_64"]; and [""] for an {!Io} device. *)

val computes : t -> bool
(** [computes d] is [true] iff [d] runs work: a host, or a device a driver
    opened. It is [false] for an {!Io} device, whose memory is only read and
    written. *)

val runs_on_host : t -> bool
(** [runs_on_host d] is [true] iff [d]'s work is this process's code: [d] is
    {!host}, or a device {!memory_device} opened. The host computes on such a
    device's memory as on its own. *)

val shares_host_memory : t -> bool
(** [shares_host_memory d] is [true] iff this process addresses [d]'s memory as
    host memory: [d] and {!host} reach each other's memory,
    [reaches host d && reaches d host]. It holds for {!host}, a
    {!memory_device}, and a driver's device of this machine that runs no copy,
    such as Metal's; it is [false] for every device of another machine, that
    machine's host included. *)

val reaches : t -> t -> bool
(** [reaches d d'] is [true] iff [d]'s work addresses [d']'s own memory
    ({!Buffer.Device}) once [d] borrows it ({!Buffer.borrow}), as a compiler
    that places copies must know before any buffer exists. A device reaches its
    own memory. Of devices of one machine:
    - a host reaches the memory of a {!memory_device} and of a driver's device
      whose driver runs no copy ({!Driver.queues}): that memory is the host's;
    - a driver's device reaches its host's memory where it maps host memory
      ({!Driver.maps_host}), a memory device's, and that of a device of its own
      driver that it maps ({!Driver.peer}).

    Otherwise it is [false]: across machines, and between an {!Io} device and
    any other device. *)

val capability : t -> 'a Type.Id.t -> 'a option
(** [capability d k] is [Some c] if [d]'s driver declares its capability record
    under [k] ({!Driver.capability_key}), [c] being the record it filled when
    [d] opened, and [None] otherwise. *)

val equal : t -> t -> bool
(** [equal d d'] is [true] iff [d] and [d'] are the same device. *)

val pp : Format.formatter -> t -> unit
(** [pp] formats a device's {!name}. *)

(** {2:budgets Budgets} *)

val budget : t -> int
(** [budget d] is the most bytes [d] holds at once in the memory that counts in
    its budget ({!Buffer.memory}): live buffers, loaded programs' code and its
    cache. It starts at [max_int] for a host, {!Driver.budget} for a driver's
    device and {!Io.budget} for an io device; {!set_budget} changes it. *)

val set_budget : t -> int -> unit
(** [set_budget d n] sets [d]'s budget to [n], returning cached memory
    ({!free_cache}) until [d] holds at most [n] bytes or its cache is empty.
    Live buffers are never released.

    Raises [Invalid_argument] if [n < 0]. *)

val free_cache : t -> unit
(** [free_cache d] returns [d]'s cached memory to its driver; for a host, the
    memory it keeps of collected buffers for reuse. *)

exception Out_of_memory of t * int
(** [Out_of_memory (d, n)] is raised when [d] cannot allocate [n] bytes, at once
    or after the tries {!reclaim} describes, [d] being the device asked,
    whichever budget refused ({!Buffer.Pinned}). *)

(** {2:lost Lost devices} *)

exception Lost of t * string
(** [Lost (d, why)] is raised by every use of the lost device [d] and of memory
    whose stamps name [d], [why] being the driver's reason or, for a device lost
    because its queue waited on another lost device, ["NAME lost"] with that
    device's name. It prints as ["NAME lost: why"]. *)

val lost : t -> string option
(** [lost d] is [Some why] if [d] is lost, [why] being what its {!Lost} carries,
    and [None] otherwise. It raises nothing and waits for nothing. *)

(** {1:timeline Timeline} *)

(** Points on timelines.

    A point is a device and a value of its timeline, held in one word: an
    immediate value that a store into an array or a record never boxes. *)
module Point : sig
  type device := t

  type t [@@immediate]
  (** The type for points. *)

  val device : t -> device
  (** [device p] is [p]'s device. *)

  val value : t -> int
  (** [value p] is [p]'s value, [1] or more. *)

  val pp : Format.formatter -> t -> unit
  (** [pp] formats a point as [NAME:v]. *)
end

val submitted : t -> int
(** [submitted d] is the last value assigned on [d]'s timeline, [0] before any:
    by {!submit}, or by a {!Buffer.copy} or {!Program.load} that [d] runs. It
    never decreases. *)

val signaled : t -> int
(** [signaled d] is the last value [d]'s timeline word showed reached. Work up
    to it completed, and its writes are visible to the caller. *)

val wait : t -> int -> unit
(** [wait d v] returns once [d] reached [v]. For a driver whose host writes the
    word ([`Host] {!Driver.completion}), or whose word the host does not
    address, it blocks in the driver ({!Driver.sleep}) from the first read of
    the word. For another, it spins on the word, yielding the processor and with
    the domain lock released, and blocks in the driver between reads once the
    word stood still for the still interval. It waits however long the work
    runs: only [d]'s driver decides that work hung.

    Raises [Invalid_argument] if [v > submitted d], and {!Lost} if [d] is lost
    or is lost by the wait. *)

(** {1:buffers Buffers} *)

(** Buffers of device memory.

    A buffer is {!length} bytes in a range of one device's memory. What the
    bytes mean, elements of some type, is the caller's: this library moves,
    orders and returns bytes.

    A buffer is {e owned} when {!create} or {!of_string} made it, and
    {e borrowed} when it is over memory something else holds: a bigarray
    ({!of_bigarray}) or another device's memory ({!borrow}). A {!view} is as the
    buffer it views. *)
module Buffer : sig
  type device := t

  type t
  (** The type for buffers. *)

  (** The type for the memories of a device that {!create} allocates. *)
  type memory =
    | Device
        (** The device's own memory, which counts in the device's {!budget}. *)
    | Pinned
        (** Host memory that both the device's work and the host address,
            page-locked where the host pages: coherent, with no flush. It counts
            in the host's {!budget}, except on a device whose memory the host
            addresses, where it is the device's own memory and counts in the
            device's. *)
    | Mapped
        (** The device's own memory, which the host also addresses through a
            write-combined window, and which counts in the device's {!budget}. A
            host write is seen by work submitted after it; the host reads it
            slowly. Where the device has no such window, or the window or the
            budget cannot hold the buffer, it is [Pinned] memory, which keeps
            these promises; no cache is released for it. *)

  val create : ?memory:memory -> device -> int -> t
  (** [create d n] is an owned buffer of [n] bytes in [d]'s memory [memory]
      (defaults to [Device]), with unspecified contents. A buffer of no bytes
      allocates nothing. On a device whose memory the host addresses, every
      [memory] is the device's own. On a host, a buffer of 64 KiB or more (four
      pages, where pages are larger) starts on a page, so devices can {!borrow}
      it. On an {!Io} device it is memory the device's {!Io.alloc} makes.

      Raises [Invalid_argument] if [n < 0] or [d] is an io device that makes no
      memory of its own; {!Out_of_memory}; and {!Lost} if [d] is lost. *)

  (** The type for accesses to memory. *)
  type access =
    | Read  (** Reading it. *)
    | Read_write  (** Reading and writing it. *)

  val of_io : device -> 'r Type.Id.t -> 'r -> access:access -> int -> t
  (** [of_io d k r ~access n] is a buffer over the [n] bytes of [r], a region of
      the io device [d] whose library declares [k] ({!Io.region_key}), whose
      memory admits [access] ({!val-access}): [Read] for a region [d] holds for
      reading, such as a file opened read-only. It returns to [d]'s {!Io.free}
      once unreachable. The memory is outside the claims ({!Claim}).

      Raises [Invalid_argument] if [d] is no io device, its library's key is not
      [k], or [n < 0]. *)

  val io : t -> 'r Type.Id.t -> 'r option
  (** [io b k] is the region [b]'s memory lies in, if it is io memory of a
      library whose key is [k], and [None] otherwise.

      Raises [Invalid_argument] if [b] is dead ({!Claim.consume}). *)

  val of_bigarray : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> t
  (** [of_bigarray ba] is a borrowed buffer on {!host} over [ba]'s bytes,
      without a copy. It keeps [ba] reachable. The memory is outside the claims
      ({!Claim}). *)

  val of_string : string -> t
  (** [of_string s] is an owned buffer on {!host} holding [s]'s bytes, placed as
      {!create} places a host buffer. *)

  val borrow : device -> t -> t option
  (** [borrow d b] is [Some b'], a borrowed buffer on [d] over [b]'s memory,
      without a copy, of [b]'s length, or [None] where [d] cannot map it. It is
      [Some b] for [b] on [d]. [b'] keeps [b] reachable, and its stamps are
      [b]'s: work through [b'] is work on [b]'s memory.

      [d] maps host memory of its machine that starts on a page, where it maps
      host memory at all ({!Driver.maps_host}), and memory of a device of its
      own driver that its driver maps ({!Driver.map_peer}). Every borrow on [d]
      of one memory shares one mapping, made at the first borrow, which lasts
      while the memory lives and is released with it, once [d]'s work submitted
      until then is done: a borrow dropped and made again maps nothing. A borrow
      of a borrow maps the memory the first one maps. A {!memory_device} maps
      any host memory. Host memory that does not start on a page, such as a host
      buffer of fewer than 64 KiB, borrows only on hosts and memory devices. An
      io device's memory borrows through its pages ({!Io.pages}), as host
      memory, where its device maps them; a device other than the host asks the
      io device to read the borrowed bytes ahead ({!Io.prefetch}).

      Raises [Invalid_argument] if [b] is dead ({!Claim.consume}); {!Lost} if
      [d] or [b]'s device is lost or is lost by the borrow, or [b]'s stamps name
      a lost device; and [Sys_error] where asking an io device for [b]'s pages
      failed and may pass ({!Io.pages}). *)

  val wait : t -> access -> unit
  (** [wait b access] returns once the work on [b]'s memory that an access of
      the host must follow is done: for [Read], the work of its last write; for
      [Read_write], the work of every use, its own device's included. It waits
      for the work on all of [b]'s memory, through any view, for the work on the
      memory a borrow maps, and, for memory in a {!Hold}, for every point of the
      hold, whatever [access]. Host memory and an {!Io} device's memory have
      work to wait for only where a device borrowed them: an io device's memory,
      through its pages ({!Io.pages}).

      It waits for one device after another and holds no lock. Once every point
      is reached it reads one word per device and allocates nothing. Work
      submitted after it returns is the caller's to exclude, by a claim taken
      before the wait ({!Claim}) or a lock of its own.

      Raises [Invalid_argument] if [b] is dead, and {!Lost} if a point it waits
      for is on a lost device. *)

  val copy : src:t -> dst:t -> unit
  (** [copy ~src ~dst] copies [src]'s bytes into [dst] and returns once they are
      there. It orders itself as work that reads [src] and writes [dst]: it
      waits for [src]'s last write and for every use of [dst].

      Between memory the host addresses, the host copies. Otherwise a device
      with a copy queue copies, as work on its timeline: [dst]'s device when
      only [src] is host-addressable, [src]'s otherwise, directly between memory
      it addresses or maps, and through the host's {e staging memory} otherwise:
      two slots of 64 MiB of host memory, made at the first copy that needs them
      and kept for the life of the process ({!domains}). A device that runs no
      copy has memory the host addresses, which the host copies. An {!Io}
      device's memory, of any machine, is read and written by its {!Io.read} and
      {!Io.write}, through the staging memory when the host does not address the
      other side, except a copy into it from memory the host does not address,
      of a device with a copy queue that maps its pages ({!Io.pages}), which
      that device runs through them. Memory of a driver's device of another
      machine copies only directly, by [src]'s device, which no staging memory
      reaches.

      Staging memory that a device lost while it used it is replaced, so a loss
      reaches no other device's copies.

      Raises [Invalid_argument] if [src] and [dst] differ in size, overlap
      ({!overlaps}), or either is dead, [dst]'s memory is [Read]
      ({!val-access}), or one is memory of a driver's device of another machine
      and [src]'s device runs no copy or does not reach [dst]'s ({!reaches});
      {!Lost} if a device involved is lost or is lost by the copy, or a point it
      waits for is on a lost device; {!Out_of_memory} if a host cannot allocate
      its staging memory; and what an {!Io} device's read or write raises. *)

  val device : t -> device
  (** [device b] is the device [b] is on: [d] for a buffer that {!create},
      {!of_io} or {!borrow} made on [d], {!host} for one {!of_string} or
      {!of_bigarray} made, and its buffer's device for a {!view} and for
      {!Claim.consume}'s result. *)

  val length : t -> int
  (** [length b] is the number of [b]'s bytes. *)

  val is_borrowed : t -> bool
  (** [is_borrowed b] is [true] iff [b] is borrowed. *)

  val dead : t -> string option
  (** [dead b] is [Some why] if [b] is dead, [why] the reason its memory was
      consumed with ({!Claim.consume}), and [None] while [b] lives. A library
      that calls this one on its caller's buffer checks it first, to refuse a
      dead buffer under its own name. *)

  val access : t -> access
  (** [access b] is the accesses [b]'s memory admits, the same for each of its
      views and borrows: [Read_write] for memory {!create}, {!of_string} and
      {!of_bigarray} make, and what {!of_io} was given.

      [Read] memory is never written: a {!copy} into it raises, and so do a
      submission that writes it ({!Submission.make}, {!submit}) and a host claim
      for writing ([rig_buffer_claim] in [rig.h]), and it is never exclusive
      ({!Claim}). The host writes it only by breaking this, through {!bigarray},
      after which what reads of it see is unspecified. A library that writes its
      caller's buffer checks this first, to refuse a [Read] buffer under its own
      name. *)

  val view : t -> first:int -> length:int -> t
  (** [view b ~first ~length] is the [length] bytes of [b] from its byte [first]
      on, over [b]'s memory.

      Raises [Invalid_argument] if [b] is dead, [first] or [length] is negative,
      or the bytes do not lie inside [b]'s. *)

  val spans : t -> bool
  (** [spans b] is [true] iff [b]'s bytes are all of the memory it lies in,
      below its borrows: those of a buffer {!create}, {!of_string} or
      {!of_bigarray} made, and of a borrow of one, and not those of a {!view} of
      part of it. *)

  val overlaps : t -> t -> bool
  (** [overlaps b b'] is [true] iff [b] and [b'] share a byte of memory: through
      views of one memory, a borrow and the memory it maps, or two bigarrays
      over the same bytes. Buffers of no bytes overlap nothing. *)

  val bigarray :
    ('a, 'b) Bigarray.kind -> t -> ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
  (** [bigarray k b] is the bytes of the host buffer [b] read as elements of
      kind [k], without a copy: [length b / Bigarray.kind_size_in_bytes k] of
      them, in the host's byte order. Writing through it writes [b], whose
      memory must admit it ({!val-access}). It, and every array made from it,
      keeps [b]'s memory alive while reachable, memory a borrow on the host maps
      included: no buffer reuses it and its device does not free it until then.
      Access through it is the host's: {!wait} orders it after devices' work.
      From then on the memory is outside the claims for good, and is never
      exclusive again ({!Claim}).

      Raises [Invalid_argument] if [b] is dead or not on {!host}, [b]'s bytes
      are not a whole number of elements of [k] starting at a multiple of their
      size (of one component's for complex kinds), or [b]'s memory is held
      exclusive by claims that have not consumed it ({!Claim.consume}), and
      {!Lost} if [b]'s stamps name a lost device. *)

  val blit_from_string : string -> int -> t -> int -> int -> unit
  (** [blit_from_string s i b j n] copies the [n] bytes of [s] from [i] into the
      host buffer [b] from its byte [j], and returns once they are there. It
      waits as a {!wait} with [Read_write] does. It takes no claim: claims are
      the caller's, as for {!copy}.

      Raises [Invalid_argument] if the ranges are not valid, [b] is dead or not
      on {!host}, or its memory is [Read] ({!val-access}), and {!Lost} if a
      point it waits for is on a lost device. *)

  val blit_to_bytes : t -> int -> bytes -> int -> int -> unit
  (** [blit_to_bytes b i s j n] copies the [n] bytes of the host buffer [b] from
      its byte [i] into [s] from [j]. It waits as a {!wait} with [Read] does.

      Raises [Invalid_argument] if the ranges are not valid, or [b] is dead or
      not on {!host}, and {!Lost} if a point it waits for is on a lost device.
  *)

  (** {1:low Low level}

      For the libraries that submit work. C code reads a buffer's host address,
      its length and the reason it is dead with [rig_buffer_host],
      [rig_buffer_bytes] and [rig_buffer_why], and claims its memory with
      [rig_buffer_claim] and [rig_buffer_release], declared in [rig.h]. *)

  val address : t -> int
  (** [address b] is the address of [b]'s first byte as [b]'s device's work
      addresses it: its memory's {!Driver.address} plus {!offset} for a driver's
      memory, the host address for a host's. An empty buffer that {!create} made
      on a driver's device names no memory: its address is [0], as work of no
      bytes reads none. An address fits in the 62 bits of an [int]'s
      non-negative range.

      Raises [Invalid_argument] if [b] is dead, is an {!Io} device's memory, or
      is memory its driver names by handle only. *)

  val handle : t -> nativeint
  (** [handle b] is the driver's object for [b]'s memory ({!Driver.handle}),
      which [b] starts {!offset} bytes into, such as an [MTLBuffer]; [0n] for an
      empty buffer that {!create} made on a driver's device.

      Raises [Invalid_argument] if [b] is dead or is a host's or an {!Io}
      device's memory, which no driver object names. *)

  val offset : t -> int
  (** [offset b] is the number of bytes of [b]'s memory before [b]'s first byte.
  *)
end

(** {1:claims Claims} *)

(** Claims on memory.

    Every view and borrow of one memory shares one count of claims. A reader
    claims the memory while the host reads it, and a compiled call that writes
    over memory it was handed holds it exclusive, so that no reader sees the
    write. Claims never wait: a claim that cannot be had raises, or for an
    exclusive one, is refused. A host reader claims, then waits
    ({!Buffer.wait}): a wait before the claim does not cover a write in place
    that a donation on another domain makes between them.

    Memory that something outside this library reaches is {e outside the claims}
    and never exclusive: memory {!Buffer.of_bigarray} and {!Buffer.of_io} make,
    and memory {!Buffer.bigarray} exported. [Read] memory ({!Buffer.val-access})
    is never exclusive either.

    A claimed memory may be {e consumed}: every buffer over it made before
    becomes {e dead}, and reaching a dead buffer's bytes raises
    [Invalid_argument] with the consumption's reason. Consuming releases
    nothing: the memory lives while a buffer reaches it.

    Device work takes no claim. A {!submit} requires its caller's claims on the
    buffers it is passed until it returns. *)
module Claim : sig
  val read : Buffer.t -> unit
  (** [read b] claims [b]'s memory for reading, beside other readers.

      Raises [Invalid_argument] if [b] is dead or its memory is held exclusive,
      and {!Lost} if [b]'s stamps name a lost device. *)

  val release : Buffer.t -> unit
  (** [release b] ends a {!read} of [b]'s memory that the caller made. It
      accepts a dead [b].

      Raises [Invalid_argument] and changes nothing if the memory has no read
      claim of a {!read}, a {!with_} or [rig_buffer_claim]: a reader outside the
      claims is not one. *)

  type t
  (** The type for the claims of a {!with_}. *)

  val with_ : read:Buffer.t list -> donate:Buffer.t list list -> (t -> 'a) -> 'a
  (** [with_ ~read ~donate f] claims the memory of [read] and [donate] for
      reading, then tries each value of [donate] (its buffers, one per device)
      exclusive, and is [f] of the claims. A value is exclusive if each of its
      buffers {!Buffer.spans} its memory and the memory has no other claim;
      otherwise its buffers stay read. Every claim is released when [f] returns
      or raises.

      Raises [Invalid_argument] before [f], releasing what it claimed, if a
      buffer is dead, a memory is held exclusive, or a buffer of [donate]
      overlaps another of [read] or [donate], and {!Lost} likewise if a buffer's
      stamps name a lost device. *)

  val exclusive : t -> Buffer.t -> bool
  (** [exclusive c b] is [true] iff [c] holds [b]'s memory exclusive: the caller
      may write it in place, through the buffer {!consume} returns, and no
      reader sees the write. *)

  val consume : t -> why:string -> Buffer.t -> Buffer.t
  (** [consume c ~why b] consumes [b]'s memory with the reason [why]: [b] and
      every buffer over that memory made before are dead. It is a live buffer
      over the same memory, which the caller writes in place only if [c] holds
      it {!exclusive}.

      Raises [Invalid_argument] if [c] does not claim [b]'s memory, [b] is dead,
      or [b] does not {!Buffer.spans} its memory. *)
end

(** {1:holds Holds} *)

(** Memory kept across many submissions.

    A linked program keeps its fixed memory in one hold, and its programs and
    the driver objects its work uses reachable through the hold's release. A
    hold has one stamp per device, raised to the point of each submission that
    names it. Its memory is named by submissions only with the hold, and returns
    once the hold is unreachable and each of its stamps is reached. A hold is
    made before work names its memory: a submission that named the memory before
    is refused at its next {!submit}. *)
module Hold : sig
  type t
  (** The type for holds. *)

  val make : ?release:(unit -> unit) -> Buffer.t list -> t
  (** [make ~release bs] holds the memory of [bs]. A {!Buffer.wait} on that
      memory, whatever its access, waits for every stamp of the hold.

      [release] (defaults to doing nothing) frees what the hold's work uses
      beyond memory, such as a driver object that work runs. It runs once, after
      the hold is unreachable and, for each device the hold has a stamp of, that
      stamp is reached and, if the device is lost, its driver's {!Driver.stop}
      returned. It runs in the next {{!reclaim}drain} of any device, such as a
      {!Buffer.copy}'s, or as a lost device's stop returns. It holds no lock of
      this library, must not call it, and must not raise: an exception it raises
      is raised again by the call whose drain ran it. It counts as a call in
      flight on each device of the hold that is not lost, so no {!Driver.stop}
      of those devices runs beside it.

      Raises [Invalid_argument] if a buffer is dead, or its memory is already in
      a hold. *)
end

(** {1:submitting Submitting work} *)

(** Prepared submissions.

    A submission is work for one device: its {e parts}, each for one of the
    device's queues ({!Driver.queues}), and the parts each waits for within the
    submission. It is made once and run many times. What changes from one run to
    the next, the buffers its work reads and writes and the points it waits for,
    are arguments of {!submit}, which keeps none of them once it returns. Its
    prepared form lives outside the OCaml heap, so a submit allocates nothing.

    A submission keeps every buffer its parts name reachable while it is
    reachable itself, so their memory is never freed between {!make} and a
    submit: a part never hands its driver memory that was given back. Once the
    submission is unreachable, that memory returns by its stamps, after the work
    of the last submit. *)
module Submission : sig
  type device := t

  type t
  (** The type for prepared submissions. *)

  (** The type for the work of a part. Every buffer it names is used by the
      work: a copy's [dst] is written, everything else read. *)
  type work =
    | Words of Buffer.t
        (** The 32-bit words of the buffer, placed on the queue's ring, such as
            a run copy's indirect-buffer or GPFIFO entry. The buffer is host
            memory. *)
    | Fill of {
        fill : nativeint;
        arg : Buffer.t;
        ring_units : int;
        segment_bytes : int;
      }
        (** The C function at [fill], called inside the driver's hand-over with
            the queue's context, the host address of [arg] and the value
            ({!Driver.submit_entry}). A ring driver's fill writes at most
            [ring_units] entries and [segment_bytes] bytes of argument segment;
            a library driver's declares [0] of each. *)
    | Copy of { src : Buffer.t; dst : Buffer.t }
        (** A copy of [src]'s bytes into [dst], on a copy queue. *)

  type part = { queue : string; after : int array; work : work }
  (** The type for parts: [work] on [queue], one of the device's
      ({!Driver.queues}). A part runs after the parts of its submission before
      it on its queue, and after those whose indices [after] lists: [after]
      orders parts of different queues. *)

  val make :
    ?hold:Hold.t -> reads:int -> writes:int -> device -> part array -> t
  (** [make ~hold ~reads ~writes d parts] is a submission of [parts] on [d]
      whose every run reads [reads] buffers and writes [writes] buffers
      ({!submit}), a buffer counted as often as it is passed. [hold] names the
      memory of a {!Hold}: every submit of the submission raises the hold's
      stamp of [d], and the parts may name the hold's memory.

      Raises [Invalid_argument] if [d] is a host or an {!Io} device, which run
      no submitted work, [reads] or [writes] is negative, an index of a part's
      [after] is negative or not below its own, a queue is not one of [d]'s, a
      part's buffer is dead, a {!Words} or {!Fill} buffer is not host memory, a
      {!Copy}'s buffers differ in size or are not [d]'s memory or its [dst]'s
      memory is [Read] ({!Buffer.val-access}), [d]'s driver runs no copies (it
      lists no copy queue, {!Driver.queues}), or a part names memory of a hold
      other than [hold]; and {!Lost} if [d] is lost. A part [d]'s driver does
      not run is refused at {!submit}. *)
end

val submit :
  Submission.t ->
  reads:Buffer.t array ->
  writes:Buffer.t array ->
  waits:Point.t array ->
  Point.t
(** [submit s ~reads ~writes ~waits] runs [s]'s work once, reading the buffers
    of [reads] and writing those of [writes] after the points of [waits]. It
    hands the work to [s]'s device [d] as the value [v] it assigns, and is the
    point [(d, v)]. A buffer may appear more than once, in either array. The
    buffers are on [d]: [d]'s work reaches other memory through a
    {!Buffer.borrow}, an {!Io} device's through its pages. [submit] reads each
    element of [reads] and [writes] once and keeps the buffer reachable until it
    has raised its stamps. The caller holds its claims on them until [submit]
    returns. It:
    + Loads the points [s]'s work must follow: the last write of each buffer of
      [reads] and of each buffer its parts read, every use by another device of
      each buffer of [writes] and of each copy's [dst], and the points of
      [waits]. Each foreign point not yet reached is a wait in [d]'s queue if
      [d]'s driver waits on the producer's completion ({!Driver.waits_on}) and
      [d] maps the producer's timeline word, decided once per pair of devices,
      up to {!Driver.max_waits} waits; otherwise [submit] waits for it now,
      holding no lock.
    + Takes [d]'s {e turn}, the right to be [d]'s one submission between its
      room check and its hand-over, and asks [d]'s driver for room
      ({!Driver.room_entry}). Once the parts fit, it assigns [v], one more than
      {!submitted}[ d], hands the work over ({!Driver.submit_entry}), and raises
      the stamps of [reads], [writes], the parts' buffers and the hold to
      [(d, v)]. While they do not fit, it waits for [d]'s next value with the
      turn released, and tries again. No OCaml code runs between the assignment
      and the turn's release, so a value is handed over or [d] is lost.

    It allocates nothing unless it waits.

    Raises [Invalid_argument] if [reads] or [writes] holds another number of
    buffers than {!Submission.make} declared, a buffer of [reads], [writes] or a
    part is dead, a buffer of [reads] or [writes] is not on [d] or its memory is
    in a hold, the memory of a buffer of [writes] is [Read]
    ({!Buffer.val-access}), a part names memory of a hold other than the
    submission's, or the parts never fit [d]'s empty queues or name one its
    driver does not run; and {!Lost} if [d] is lost, [d]'s hand-over fails, a
    point [s] follows is on a lost device, or a producer [d]'s queue waits on is
    lost before the hand-over. A device lost after [v] was handed over raises
    {!Lost}, with [v]'s stamps naming it. *)

(** {1:programs Programs} *)

(** Code loaded on a device. *)
module Program : sig
  type device := t

  type t
  (** The type for loaded images: a binary in a device's format, whose functions
      that device's work runs. *)

  val load : device -> string -> (t, string) result
  (** [load d binary] loads [binary] on [d]: a code object for AMD, a cubin for
      NV, a CUDA module for CUDA, a metallib for Metal. Where [d]'s memory holds
      the code, [load] allocates it as {!Buffer.create} allocates [Device]
      memory, so the code counts in [d]'s {!budget}, then copies it there as a
      submission on [d], after [d]'s queued work, and waits for it. Each call
      loads anew. The image stays loaded while [t] is reachable, and is unloaded
      once it is not and the work [d] was handed until then is done; its memory
      then returns to [d].

      [Error why] if [d]'s driver rejects [binary], with its reason, which
      starts with [d]'s {!name}. [d] stays usable.

      Raises [Invalid_argument] if [d] loads no code (a host or an {!Io}
      device), {!Out_of_memory}, and {!Lost} if [d] is lost or is lost by the
      load. *)

  val device : t -> device
  (** [device p] is the device [p] is loaded on. *)

  val entry : t -> string -> int option
  (** [entry p f] is the driver's name for [p]'s function [f] ({!Driver.entry}),
      such as a kernel descriptor's address, a [CUfunction] or an
      [MTLComputePipelineState], or [None] if [p] has no function [f]. Work that
      runs [f] keeps [p] reachable until it is done, such as a {!Hold}'s release
      that holds it.

      Raises {!Lost} if [p]'s device is lost. *)
end

(** {1:profiles Profiles} *)

(** Profiles of devices' work.

    A profile of every device is taken while a function runs ({!take}). Profiles
    nest and overlap, in any domains: each holds the events recorded while it is
    taken. Every time is on the host clock ({!now}). When no profile is taken,
    recording reads one atomic word and allocates nothing. *)
module Profile : sig
  type device := t

  (** The type for profile events. Times are nanoseconds of the host clock. *)
  type event =
    | Span of {
        device : device;
        lane : string;  (** Its track within the device. *)
        name : string;
        start : int;
        stop : int;
      }
        (** Work that ran on a lane of a device. The host's lanes are its
            domains, ["domain 0"], ["domain 1"], …. *)
    | Allocation of { device : device; time : int; allocated : int }
        (** The bytes of memory [device] allocated from [time] on. *)
    | Load of { program : Program.t; binary : string; time : int }
        (** An image loaded. *)
    | Counters of {
        device : device;
        name : string;
        start : int;
        stop : int;
        counters : (string * int array) list;
            (** Each counter asked for and its count during the run, one per
                unit of the hardware that counts it. *)
      }  (** The counters of a run of a function. *)
    | Trace of {
        device : device;
        name : string;
        start : int;
        stop : int;
        part : int;  (** The part of the device that wrote it. *)
        data : string;  (** The trace as the device wrote it. *)
      }  (** The thread trace of a part of a device during a run. *)
    | Overwritten of { device : device; time : int; runs : int }
        (** Runs whose counters and traces the device overwrote unread. *)
    | Copy of {
        src : device;
        dst : device;
        bytes : int;
        start : int;
        stop : int;
      }
        (** A transfer of [bytes] from [src]'s memory to [dst]'s, by a
            {!Buffer.copy}, from when it was asked to when the host saw it done.
            A copy through staging memory records each piece into and out of it,
            with {!host} as one side, and no event of its own. *)

  val take :
    ?counters:string list -> ?trace:bool -> (unit -> 'a) -> 'a * event list
  (** [take f] is [f ()] and the events recorded while it ran, in time order
      and, at equal times, longest first, then in the order they were recorded.
      It asks the libraries that encode work to count [counters] (defaults to
      none) and, with [trace] (defaults to [false]), to trace ({!counters},
      {!traced}). Before it returns it waits for the points whose events are
      still to be read ({!after}); those of a device lost meanwhile are left
      out. If [f] raises, the events are dropped and the exception is raised
      again with its backtrace.

      Raises [Invalid_argument] if [counters] names a counter twice. *)

  val enabled : unit -> bool
  (** [enabled ()] is [true] iff some profile is being taken. *)

  val counters : unit -> string list
  (** [counters ()] is the counters the profiles being taken ask for, each once,
      in the order they started. A library that keeps encoded work keeps it for
      each value of [counters ()]. *)

  val traced : unit -> bool
  (** [traced ()] is [true] iff a profile being taken asks for traces. *)

  val now : unit -> int
  (** [now ()] is the host clock: nanoseconds of the system's monotonic clock,
      from an unspecified start. *)

  val span : string -> (unit -> 'a) -> 'a
  (** [span name f] is [f ()], recorded as a span named [name] on the calling
      domain's lane of {!host}, from the call until [f] returns or raises, in
      each profile taken when the call starts. *)

  val after : Point.t -> (unit -> event list) -> unit
  (** [after p f] records the events [f ()] in each profile being taken, once
      [p] is reached: [f] reads what [p]'s work wrote, such as its times or
      counters. [f] runs in the first wait that finds [p] reached, among the
      waits for [p]'s device ({!wait}, {!Buffer.wait}, {!Buffer.copy} and
      {!take}'s), before that wait returns: memory a {!Buffer.wait} returns for
      is read before its caller rewrites it. [f] holds no lock of this library,
      must not call it and must not raise: an exception it raises is raised
      again by the wait that ran it. It does nothing unless {!enabled}. *)

  val record : Point.t -> lane:string -> name:string -> Buffer.t -> unit
  (** [record p ~lane ~name stamps] is [after p] of a {!Span} of [p]'s device
      named [name] on [lane], whose start and stop are the unsigned 64-bit words
      at bytes 8 and 24 of the host memory [stamps], in the host's byte order
      and on the host clock: [p]'s work, or its driver, writes them, as
      {!timestamp} does.

      Raises [Invalid_argument] if [stamps] is not 32 bytes of host memory
      starting at a multiple of 8. *)

  val timestamp : nativeint
  (** [timestamp] is the address of the C function

      {v void rig_timestamp(void *word); v}

      which stores {!now} into the 64-bit word at [word] with one aligned atomic
      store, in the platform's C calling convention. It takes no lock, so a
      device library may call it from a completion path. *)

  val output_chrome_trace : out_channel -> event list -> unit
  (** [output_chrome_trace oc events] writes [events] to [oc] in Chrome's trace
      event format, JSON, which Perfetto and [chrome://tracing] load: a process
      per device, a thread per lane, a complete event per span, per copy (on its
      [src]'s process) and per run's counters (each summed over its units), a
      counter [memory] per allocation change, and instant events for loads,
      traces and overwritten runs. Times are microseconds from the earliest
      event. [oc] is neither flushed nor closed. *)
end

(** {1:drivers Drivers}

    For the vendor libraries that open devices. *)

(** Drivers.

    A driver runs one kind of hardware once it is open: it holds the device's
    memory, writes its queues, makes completion observable and reports faults.
    It matches this signature structurally, over standard types, without linking
    this library; a program passes it to {!open_}.

    A device's {e timeline word} is eight bytes the driver alone writes: the
    last value [v] such that every submission up to [v] completed. It never
    moves backwards, and work never writes it. A driver orders the work it is
    handed on each queue in the order handed ({e prefix order}), and lets the
    host learn that a prefix completed or that the device faulted
    ({e observable completion}).

    {b Work} crosses in C only: this library calls the driver's C room check and
    hand-over ({!room_entry}, {!submit_entry}), in the shapes [rig_edge.h]
    states, one at a time per device, under the device's turn. A driver's own
    OCaml forms of them, for a driver used alone, are no part of this signature.

    {b Calls.} {!address}, {!handle}, {!host} and {!peer} call nothing that may
    block or fault, and this library calls them at any time. Every other call
    this library makes on a device that is not lost is {e counted}: {!stop}
    waits for none of them. {!stop} runs once, with no counted call inside, and
    after it only {!free}, {!signaled} and holds' releases follow. A {!Fault}
    from a counted call, and a failed hand-over, lose the device. *)
module type Driver = sig
  type t
  (** The type for open devices of the driver. *)

  type region
  (** The type for memory of a device. *)

  type image
  (** The type for loaded code. *)

  type capability
  (** The type for what compiled code needs from a device, declared by the
      driver's ABI library. *)

  exception Fault of string
  (** [Fault why] reports that the device faulted. Any counted call may raise
      it. *)

  val key : t Type.Id.t
  (** [key] identifies the driver: two devices of one driver map each other's
      memory ({!map_peer}). *)

  (** {1:facts Facts}

      Read once, when the device opens: a {!Fault} here is the open's [Error].
  *)

  val arch : t -> string
  (** [arch d] is [d]'s architecture. *)

  val budget : t -> int
  (** [budget d] is the bytes of own memory [d] should hold at most. *)

  val queues : t -> string list
  (** [queues d] is [d]'s queues, ["COMPUTE:0"] first, then ["COPY:i"] for its
      copy queues. A part's queue is its index in the list. A driver that lists
      no copy queue runs no {!Submission.Copy}: every region it allocates has a
      host address, and the host copies it. *)

  val completion : t -> [ `Store | `Object of nativeint | `Host ]
  (** [completion d] is how [d]'s word advances: the queue stores it ([`Store]);
      an object of the driver's API completes, and the driver writes the word
      ([`Object h], [h] fitting in 62 bits, as it crosses the hand-over's waits
      as an integer); or the host writes it from the driver's handler or before
      the hand-over returns ([`Host]). *)

  val waits_on : t -> [ `Store | `Object | `Host ] -> bool
  (** [waits_on d c] is [true] iff [d]'s queues wait in the queue for a producer
      of completion [c]; for [`Object], a producer of [d]'s own driver, whose
      objects [d]'s queues know. *)

  val max_waits : t -> int
  (** [max_waits d] is the most waits in [d]'s queue ({!waits_on}) one
      submission carries, [0] or more: this library waits on the host for the
      others, before the hand-over. It is read only for completions {!waits_on}
      accepts. *)

  val blocks : t -> [ `Returns | `May_block ]
  (** [blocks d] is [`Returns] if the C room check and hand-over never block,
      and [`May_block] if they may block on [d]'s own earlier work, on its own
      transfers or on its library's back-pressure. *)

  val maps_host : t -> bool
  (** [maps_host d] is [true] iff [d] maps host memory of its machine that
      starts on a page ({!map_host}). Where it is [false], this library never
      calls {!map_host}. *)

  val capability : t -> capability
  (** [capability d] is [d]'s record, filled when [d] opened. *)

  val capability_key : capability Type.Id.t
  (** [capability_key] is the key the driver's ABI library declares. *)

  (** {1:memory Memory} *)

  val alloc : t -> [ `Device | `Pinned | `Mapped ] -> int -> region option
  (** [alloc d kind n] is [n] bytes of [d]'s memory of [kind]
      ({!Buffer.memory}), or [None] if [d] has none. [n] is positive. Counted.
  *)

  val free : t -> region -> unit
  (** [free d r] gives back [r], a region {!alloc} made or a mapping {!map_peer}
      or {!map_host} made. This library frees [r] once no work of [d] that uses
      it can run; it may free after {!stop}. *)

  val address : region -> int option
  (** [address r] is [r]'s address as [d]'s work addresses it, or [None] for
      memory named by its handle only. *)

  val handle : region -> nativeint
  (** [handle r] is the driver's object for [r]: a buffer object, a
      [CUdeviceptr], an [MTLBuffer]. *)

  val host : region -> int option
  (** [host r] is the host address of [r]'s first byte, if the host addresses
      it. *)

  val peer : t -> t -> bool
  (** [peer d d'] is [true] iff {!map_peer}[ d d'] maps [`Device] memory of
      [d'], a device of the same driver. It answers from what [d] and [d']
      learned when they opened, also once either is lost. *)

  val map_peer : t -> t -> region -> region option
  (** [map_peer d d' r] is a region of [d] over [r], any memory of [d'], a
      device of the same driver, or [None]. Counted. *)

  val map_host : t -> int -> int -> region option
  (** [map_host d p n] is a region of [d] over the [n] bytes of host memory at
      [p], or [None] if [d] does not map these pages, such as read-only ones.
      [p] starts a page and [n] is positive; it is called only where
      {!maps_host}. The memory stays mapped until the region is freed ({!free}).
      Counted. *)

  (** {1:code Code} *)

  val image :
    t ->
    string ->
    ( [ `Loaded of image | `Place of int * (region -> image * string) ],
      string )
    result
  (** [image d b] loads the binary [b]. It is [`Loaded i] where the driver's
      library places the code itself, and [`Place (n, lay)] where the code goes
      into [n] bytes of [d]'s [`Device] memory: [lay r] is the image over the
      region [r], which this library allocated with at least [n] bytes, and the
      bytes to place at [r]'s start, at most [n]. [image] makes nothing on [d]
      before [lay] is called, so a [`Place] whose function is never called
      leaves nothing to release; [lay] calls nothing that may block or fault,
      and raises nothing. [Error why] if [d] refuses [b]. Counted. *)

  val entry : image -> string -> int option
  (** [entry i f] is the driver's name for [i]'s function [f]: an address or an
      object. Counted. *)

  val unload : t -> image -> unit
  (** [unload d i] releases what {!image} made for [i], once no work of [d] that
      runs it can run. The region of a [`Place] is not [i]'s: this library frees
      it after. Counted. *)

  (** {1:timeline Timeline} *)

  val word : t -> region
  (** [word d] is [d]'s timeline word, never freed: other devices may map it,
      and it is read after a loss. It has a host address except behind a
      transport. *)

  val signaled : t -> int
  (** [signaled d] is the value in [d]'s word, read with acquire order. This
      library calls it only for a word with no host address, behind a transport,
      and may call it after {!stop}. Counted. *)

  val sleep : t -> seen:int -> still_ms:int -> unit
  (** [sleep d ~seen ~still_ms] blocks on the device's events until the word
      differs from [seen], returning at once if it already does, and returns at
      the latest after [still_ms] milliseconds. It may return earlier, on an
      event of other work or a timer of the driver's own: this library reads the
      word again after every return and sleeps again while it is unmoved. It
      raises {!Fault} once the device faulted. It may run beside the hand-over.
      Counted. *)

  val room_entry : nativeint
  (** [room_entry] is the address of [d]'s room check, in the shape
      [rig_room_fn] of [rig_edge.h]: whether parts fit [d]'s queues now, once
      one of [d]'s values is reached, or never, for parts that exceed [d]'s
      empty queues or name work [d] does not run. *)

  val submit_entry : nativeint
  (** [submit_entry] is the address of [d]'s hand-over, in the shape
      [rig_submit_fn] of [rig_edge.h]: it hands the parts to [d] as the work of
      the value after the last one it received, after the waits and [d]'s
      earlier work, and writes the value into the word once that work completed.
      When it returns nothing of the value is left uncommitted; a failure loses
      [d]. It calls no function of the OCaml runtime and reads no OCaml value.
  *)

  val self : t -> nativeint
  (** [self d] is the [self] argument of the C entries for [d], valid while the
      process runs. *)

  (** {1:stopping Stopping} *)

  val stop : t -> unit
  (** [stop d] stops [d] once it is lost, never waiting. The driver writes the
      last value its hand-over received into the word, with release order, once
      no work of [d] runs: before [stop] returns if none does. This library
      counts [d] as stopped once the word reads that value.

      [stop] also releases what {!image} made for each image not yet
      {!unload}ed: before it returns if no work of [d] runs, and otherwise once
      that work has ended, which may be at the device's next open. This library
      calls no {!unload} after [stop], and frees the region of a [`Place] itself
      ({!free}). *)
end

(** Devices of memory reached by reading and writing.

    An io device holds memory the host does not address and no queue runs:
    files, or another machine's host. Its reads and writes are synchronous, in
    the caller. *)
module type Io = sig
  type t
  (** The type for open io devices. *)

  type region
  (** The type for memory of an io device. *)

  exception Fault of string
  (** [Fault why] reports that the device failed, such as a closed connection.
  *)

  val region_key : region Type.Id.t
  (** [region_key] identifies the io library and its regions: {!Buffer.of_io}
      and {!Buffer.io} cast regions by it, and a device's name stays with the
      library that opened it. *)

  val budget : t -> int
  (** [budget d] is the bytes [d] should hold at most. *)

  val alloc : t -> int -> region option
  (** [alloc d n] is [n] new bytes of [d]'s memory, or [None] if [d] has not the
      room. [n] is positive.

      Raises [Invalid_argument] if [d] makes no memory of its own, which
      {!Buffer.create} raises in turn. *)

  val free : t -> region -> unit
  (** [free d r] gives [r] back. *)

  val read : t -> region -> at:int -> dst:int -> len:int -> unit
  (** [read d r ~at ~dst ~len] reads the [len] bytes at [at] in [r] into host
      memory at [dst]. A failure of the memory alone, such as a file truncated
      since it was opened, raises [Sys_error], which loses nothing; {!Fault}
      loses [d]. *)

  val write : t -> region -> at:int -> src:int -> len:int -> unit
  (** [write d r ~at ~src ~len] writes the [len] bytes of host memory at [src]
      at [at] in [r]. It raises as {!read}. *)

  val pages :
    t ->
    region ->
    (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
    option
  (** [pages d r] is [r]'s bytes as host memory, or [None] if [d] maps none. The
      mapping is [r]'s memory: {!read} and {!write} see writes through it, and
      it sees theirs. For [r] that [d] holds for reading ({!Buffer.val-access}),
      no write through the mapping reaches [r].

      A device other than the host maps the pages where its driver maps host
      memory ({!Driver.map_host}), and on Linux every such driver maps it for
      writing. NVIDIA's kernel driver, for CUDA and NV, pins the pages for
      writing for as long as they are mapped (its os-mlock.c), which the kernel
      refuses for a shared mapping of a file (mm/gup.c,
      [writable_file_mapping_allowed]): those devices map no file that [d] holds
      for writing, and its copies go through the staging memory. AMD's KFD
      faults the pages in for writing, and refuses pages mapped read-only
      (amdgpu_hmm.c), so a region [d] holds for reading is mapped as the
      process's own copy, as a private mapping of a file is.

      [None] is a fact of [r]: this library asks once per memory, at its first
      borrow that answers, and keeps the answer. A failure that may pass, of the
      memory alone, such as too many open files, raises [Sys_error], which loses
      nothing: the borrow raises it, and the next asks again. {!Fault} loses
      [d]. *)

  val prefetch : t -> region -> at:int -> len:int -> unit
  (** [prefetch d r ~at ~len] asks [d] to read the [len] bytes at [at] in [r]
      ahead, before a device other than the host reaches them through {!pages}.
      It is a hint: it raises nothing. *)

  val stop : t -> unit
  (** [stop d] ends [d] once it is lost. *)
end

(** {1:opening Opening} *)

val open_ :
  (module Driver with type t = 'a) ->
  ?machine:string ->
  name:string ->
  (unit -> ('a, string) result) ->
  (t, string) result
(** [open_ (module D) ~machine ~name make] is the open device named [name] on
    [machine] (defaults to this one), the machine whose hardware [make] opens. A
    machine's name names one machine for the life of the process: a library that
    reaches machines gives each one it makes a name of its own, so a second
    connection to one address is another machine, with devices of its own. If no
    device of that name is open there, [make ()] opens it, under the name's
    lock, so one name on one machine has one live device; its [Error] is the
    result, and an exception it raises is raised again, the name left unopened.
    Opens of other names go on meanwhile. A lost device's name opens again once
    its driver's {!Driver.stop} returned.

    The result is [Error why] if the name's device is lost and its stop has not
    returned, or if the process opened 65,535 devices already: device indices
    are never reused.

    Raises [Invalid_argument] if the open device of that name is another
    driver's, or [machine] names another machine whose host is not open
    ({!open_io}). *)

val open_io :
  (module Io with type t = 'a) ->
  ?machine:string ->
  ?host:bool ->
  name:string ->
  (unit -> ('a, string) result) ->
  (t, string) result
(** [open_io (module I) ~machine ~host ~name make] is {!open_} for an io device.
    With [host] (defaults to [false]), the device is [machine]'s host
    ({!host_of}), and devices of [machine] open once it is.

    Raises [Invalid_argument] as {!open_}, or if [host] and [machine] is this
    one, or [machine] has a host of another name, open and not lost or still
    opening. *)

val memory_device : string -> (t, string) result
(** [memory_device name] is {!open_} of the device named [name] whose memory is
    the host's and whose work this process runs, with a timeline of its own: it
    runs a submission's copies and fills before its submit returns. It stands
    for a device with a timeline in tests of what names several devices. *)
