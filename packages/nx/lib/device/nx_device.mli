(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Devices, their memory and their programs.

    A device is hardware with memory: the {!host}, the {!disk}, or a GPU that a
    vendor library such as [nx.metal.device], [nx.cuda.device], [nx.amd.device]
    or [nx.nv.device] opens. Memory is held in {!Buffer}s, a number of elements
    of one storage format ({!Nx_dtype.Scalar.t}) on one device, and copied
    between devices by {!Buffer.copy}. The host addresses the memory of some
    GPUs, such as Metal's, and not that of others, such as CUDA's, AMD's and
    NV's, whose device copies it. The disk's memory is files, which a copy reads
    and writes. A device also loads {!Program}s: the host calls its own, and the
    libraries that submit work to a GPU launch the GPU's.

    Work runs on a device asynchronously. Each device has a {e timeline}: the
    value its last submitted work signals when it completes. {!synchronize}
    waits for it, and for the work of other devices that touched the device's
    memory. Copies and the return of memory to the system synchronize the
    devices involved first. A {!Profile} records when that work ran, on the
    host's clock.

    Every function may be called from any domain. The operations on one device
    run one at a time: each takes the device for its whole duration, so they are
    ordered as they take it. An operation on several devices takes them in a
    fixed order.

    {b Machines.} A device is attached to a machine, whose host is a device too
    ({!host_of}): {!host} for this machine's devices, and the host of another
    machine, reached over the network, for that machine's devices, such as
    [nx.remote.device] connects to. What "the host addresses" below means for a
    device is what its machine's host addresses. A copy between machines moves
    the bytes over a link between them, such as two RDMA network adapters
    [nx.rdma.device] opened, and through the hosts otherwise.

    {b Reclamation.} Nothing frees a buffer by hand. Once a buffer and all its
    views are unreachable, the garbage collector hands its memory back to the
    device, which reclaims it at the start of its next operation, without
    waiting: a GPU keeps it in a cache for reuse once the work of other devices
    that touched it ({!submit}) is done, since its own later work is ordered
    after its earlier work; the host returns it to the heap; and a borrowing
    device unmaps a borrow once its work on it is done. Memory goes back to the
    system only once no work can still use it. If that work is a lost device's
    ({!Lost}), the memory is {e retained}: kept, and never freed or reused. An
    allocation that the device's {!budget} or its driver refuses first releases
    the cache to the system, then waits for the work of the memory its device
    released, then collects garbage with the device free for other domains,
    releases the borrows of its memory that idle devices still hold, and tries
    again, up to four times, and raises {!Out_of_memory} only after that. Memory
    returns one major cycle after the last value holding its buffers dies, in
    whichever domain runs the collection, even while the domain that made them
    is blocked. A value that a finaliser closure keeps dies only once that
    closure has run, a cycle after its own holder died, so a chain of such
    holders takes a cycle per link.

    {b Collection pace.} The host memory of buffers of 64 KiB or more (four
    pages, where pages are larger) paces the collector's major cycles by the
    program's memory: its OCaml heap and the host memory its live buffers hold.
    A cycle is due once the memory of those allocated since the last one reaches
    [custom_major_ratio / 150] of it ({!Gc.control}), 29% by default, the share
    the collector applies to its heap alone for other memory outside it. A
    buffer dropped just after a cycle marks it is found by the next cycle and
    freed while the one after that sweeps, so unreachable buffers hold at most
    three such shares, 88% of the program's memory by default; under steady
    allocation of buffers they measure about 0.75 times the memory of the live
    ones. A program whose OCaml heap is small and whose buffers are large thus
    runs a major cycle per share of the memory it holds. The live memory is
    measured once each cycle ends, at the end of the next major slice of any
    domain. Smaller buffers are paced as any bigarray: by the OCaml heap once
    they outlive the minor heap.

    The memory of another device's buffers paces the major cycles by the room
    left in that device's {!budget}: a cycle is due once the memory of the
    buffers allocated since the last one reaches the same share of that room,
    and at least a page. Unreachable buffers then hold at most three such shares
    of what the device could still allocate, and a device that fills up runs
    cycles more often instead of refusing allocations; a full collection
    ({!Out_of_memory}) is the last resort. A buffer small enough for the minor
    heap paces minor collections no faster than any custom block does.

    {b Hangs and faults.} {!synchronize} and {!Buffer.copy} wait for the work of
    the devices involved. A device that hangs or faults is lost for good
    ({!Lost}).

    {b Audiences.} Programs use the sections up to {!section-submitting}. The
    libraries that submit work to a device use {!section-submitting} and the
    buffers' low-level accessors, and the vendor libraries that open devices
    describe them with {!Driver}. *)

(** {1:devices Devices} *)

type t
(** The type for devices. There is one value per device: every open of a device
    returns the value its first open made. *)

val host : t
(** [host] is the host, named ["CPU"], with a budget of [max_int]. Its buffers
    are memory of the process's heap, which it does not cache. On x86_64 and
    arm64 it loads programs, which {!Program.call} runs; elsewhere it loads
    none. *)

val disk : t
(** [disk] is this machine's file system, named ["DISK"], with a budget of
    [max_int]. Its buffers are byte ranges of files ({!Buffer.of_file},
    {!Buffer.create_file}), which the host does not address: {!Buffer.copy}
    reads and writes them, and {!Buffer.borrow} maps them. It allocates no
    memory, runs no work, loads no programs and never fails. *)

val shares_host_memory : t -> bool
(** [shares_host_memory d] is [true] iff [d]'s memory and the host's are one:
    [d] is this machine's, the host addresses all of [d]'s memory and [d] maps
    the host's, as for the {!host}, Metal and the devices described as
    [Host_visible] with a mapping ({!Driver.memory}). It is [false] for the
    {!disk} and for GPUs whose own memory the host does not address, such as
    CUDA, AMD and NV GPUs. *)

val runs_on_host : t -> bool
(** [runs_on_host d] is [true] iff [d]'s work is this process's host's: [d] is
    the {!host}, or a device of this machine described as [Host_visible] that
    loads no programs ({!Driver.device}), such as the test devices over the
    host's memory. Such a device has no processor of its own: its memory is the
    host's, and the host computes on it as on its own. It is [false] for Metal,
    which runs programs, for CUDA, AMD and NV GPUs, for the {!disk} and for
    other machines' devices. *)

val reaches : t -> t -> bool
(** [reaches d d'] is [true] iff [d]'s work addresses the memory of [d'] once
    [d] borrows it ({!Buffer.borrow}), as a compiler that places copies must
    know before any buffer exists: [d]'s own; its machine's host's, when [d]
    maps host memory ({!Driver.mapping}); for a host, the memory of a device of
    its machine that the host addresses; for another device, memory of a device
    of its machine that the host addresses, when [d] maps host memory, and the
    memory of the devices its driver maps ({!Driver.device}'s [reaches]). It is
    [false] for devices of two machines, and between the {!disk} and another
    device. *)

val name : t -> string
(** [name d] is [d]'s name, [LOCAL] for the devices of this machine and
    [LOCAL@ADDRESS] for those of the machine at [ADDRESS]: ["CPU"] for the host,
    ["DISK"] for the disk, ["METAL"] for the Metal GPU, ["CUDA"], ["CUDA:1"],
    ... for CUDA GPUs, ["AMD"], ["AMD:1"], ... for AMD GPUs, ["NV"], ["NV:1"],
    ... for NVIDIA GPUs opened without CUDA, and ["RDMA"], ["RDMA:1"], ... for
    RDMA network adapters; ["CPU@10.0.0.2:6667"] and ["AMD:1@10.0.0.2:6667"] on
    another machine. The runtime composes it from the device's description
    ({!Driver}); it is for people, and nothing parses it. *)

val host_of : t -> t
(** [host_of d] is the host of the machine [d] is attached to: {!host} for the
    devices of this machine, and another machine's host for its devices. A host
    is its own. *)

val arch : t -> string
(** [arch d] is the architecture of [d]'s processor: the machine's instruction
    set for the host, such as ["arm64"] or ["x86_64"], the GPU family for Metal,
    such as ["Apple7"], the compute capability for CUDA and NV, such as
    ["sm_86"], the graphics target for AMD, such as ["gfx1100"], and [""] for
    the disk, which has no processor. *)

val equal : t -> t -> bool
(** [equal d d'] is [true] iff [d] and [d'] are the same device. *)

val compare : t -> t -> int
(** [compare] is a total order on devices, compatible with {!equal}: the order
    in which they were opened, the {!host} first. *)

val pp : Format.formatter -> t -> unit
(** [pp] formats a device's {!name}. *)

val synchronize : t -> unit
(** [synchronize d] returns once the work submitted to [d], and the work
    submitted to other devices that touched [d]'s memory, has completed. The
    work of a lost device is not waited for.

    Raises {!Lost} if [d] is lost, or is lost by the wait: [d] does not signal
    in time or its driver reports a fault. *)

(** {1:failures Failures} *)

exception Lost of t * string
(** [Lost (d, why)] is raised when [d]'s state is unknown and nothing recovers
    it: [d] did not signal within its {!timeout} ([why] is ["hang detected"]),
    its driver reported a fault or erred while work was enqueued on its queue
    ([why] is the driver's message), or the connection to its machine failed. It
    prints as ["NAME: why"], where [NAME] is [d]'s {!name}.

    [d] is then lost for good, and its loss is scoped to the memory it can
    reach: its own buffers, the host memory it borrowed, and memory another
    device owns that a copy of [d] was writing, or that work of [d] touched
    ({!submit}), and that [d] had not finished with when it was lost.
    - The operation that finds the loss raises [Lost (d, why)], and so does,
      with the same [why], every later operation that takes [d], and every
      {!Buffer.copy}, {!Buffer.bigarray} and {!Program.call} that reaches memory
      [d] can reach. {!Buffer.view} does not. {!stats} answers, as do the
      functions that do not take [d]: {!name}, {!arch}, {!budget}, {!submitted},
      {!signaled} and {!signal_word}.
    - Other devices do not wait for [d]'s work, and their other operations are
      unaffected.
    - [d] never reclaims memory again: its buffers, and the host memory it
      borrowed, stay allocated for the life of the process, including borrows it
      was unmapping when it was lost. Memory of another device that its copy was
      writing is never reused either. *)

exception Out_of_memory of t * int
(** [Out_of_memory (d, n)] is raised when [d] cannot allocate [n] bytes:
    - by {!Buffer.create}, at once if [n] exceeds [d]'s {!budget}, and otherwise
      if the budget or the driver refuses them after [d]'s cache was released
      and unreachable buffers collected;
    - by {!Buffer.copy}, when the host [d] of a machine cannot allocate the
      staging memory the copy goes through;
    - by {!Program.load}, when [d]'s driver cannot allocate the memory of a
      program's code after unreachable programs were collected. *)

(** {1:memory Memory} *)

val budget : t -> int
(** [budget d] is the most bytes of [d]'s own memory, its mapped memory included
    ({!Buffer.memory}), that [d] holds at once in live buffers, loaded programs'
    code and its cache together. Pinned memory, which is the host's, borrowed
    memory and the host's staging memory ({!Buffer.copy}) do not count. Mapped
    memory is also held within the window the host addresses it through. It is
    [max_int] for the host, and defaults to a device's recommended working set
    or memory size otherwise. *)

val set_budget : t -> int -> unit
(** [set_budget d n] sets [d]'s budget to [n], releasing cached memory to the
    system until [d] holds at most [n] bytes or its cache is empty. Live buffers
    are never released: an allocation fails until enough of them are collected.

    Raises [Invalid_argument] if [n < 0]. *)

val timeout : t -> int
(** [timeout d] is how long, in milliseconds, a wait for [d]'s work lasts before
    [d] is considered hung and lost for good. Every device starts at
    {!Driver.default_timeout}. *)

val set_timeout : t -> int -> unit
(** [set_timeout d ms] sets [d]'s {!timeout} to [ms], for the waits that start
    after it, from any domain at any time. Work that takes longer, such as a
    kernel that runs longer than [ms] without the device signaling, loses [d]
    for good, and the memory it can reach stays allocated: raise the timeout
    before submitting such work.

    Raises [Invalid_argument] if [ms <= 0]. *)

val free_cache : t -> unit
(** [free_cache d] returns all of [d]'s cached memory to the system. *)

(** {1:buffers Buffers} *)

(** Buffers of device memory. *)
module Buffer : sig
  type device := t

  type t
  (** The type for buffers: {!length} elements of format {!dtype} in a range of
      one device's memory. Their bytes are the elements in their storage
      representation, in order: [Nx_dtype.Scalar.bitsize s / 8] bytes each, and
      two per byte for [Int4] and [UInt4], the first in the low nibble.
      Multi-byte elements are little-endian, the byte order of the host on arm64
      and x86_64 and of Metal; a big-endian host holds its own byte order.

      A buffer is {e owned} when {!create} made it and {e borrowed} when it is
      over memory that something else holds: a bigarray ({!of_bigarray}),
      another device's memory ({!borrow}), a file ({!of_file}, {!create_file}),
      or memory a vendor library holds ({!Driver.buffer}). A {!view} is as the
      buffer it views. Owned memory returns to its device once the buffer and
      all its views are unreachable. Borrowed memory is never cached, and never
      counted in a device's budget or statistics. *)

  (** The type for the memories of a device that {!create} allocates. *)
  type memory =
    | Device  (** The device's own memory. *)
    | Pinned
        (** Memory that both the device's work and its host address: page-locked
            host memory on CUDA, AMD and NV. It is coherent: a write by either
            side is seen by the other once the work that wrote it has completed,
            with no flush. The libraries that submit work allocate their command
            buffers, queue words and volatile arguments this way. It is the
            host's memory, locked, and counts in no {!budget}: its driver's
            refusal is its only limit. *)
    | Mapped
        (** The device's own memory, which its work reads at the speed of its
            own and the host also addresses, through a write-combined window
            onto it (a BAR). A host write is seen by the work submitted after a
            full fence on the host and the vendor's flush of the host's path to
            the memory, which its library performs when it submits. A write by
            the device is seen by the host once the work that wrote it has
            completed. The host reads it uncached and slowly.

            Where the device has no such window, or the window has no room,
            mapped memory is pinned memory, which keeps the same promises more
            strongly. Nothing raises: the buffer's region
            ({!Driver.Region.of_buffer}) tells the device's library which it
            got. *)

  val create : ?memory:memory -> device -> Nx_dtype.Scalar.t -> int -> t
  (** [create d s n] is an owned buffer of [n] elements of format [s] on [d], in
      [d]'s memory [memory] (defaults to [Device]). Its contents are
      unspecified. A buffer of no bytes allocates nothing.

      On a device whose memory the host addresses, such as the {!host}, Metal
      and test devices over the host's memory, every [memory] is the device's
      own: [create ~memory d = create d]. Pinned and mapped memory count in
      [d]'s budget, [d] caches them, and copies between them and [d]'s memory
      need no staging.

      On the {!host}, buffers of at least 64 KiB (four pages where pages are
      larger) start on a page, so that devices can {!borrow} them.

      Raises [Invalid_argument] if [d] is {!disk}, whose buffers are files, if
      [n < 0] or if [n] elements of [s] take more than [max_int] bytes, and
      {!Out_of_memory} if [d] cannot allocate its bytes. *)

  val of_bigarray : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> t
  (** [of_bigarray ba] is a borrowed buffer on {!host} over [ba]'s elements,
      without a copy, of the format of [ba]'s kind: [Float16], [Float32],
      [Float64], [Int8], [UInt8] for [Int8_unsigned] and [Char], [Int16],
      [UInt16], [Int32], [Int64], [Complex64] for [Complex32] and [Complex128]
      for [Complex64]. A write through either is seen through the other. The
      buffer keeps [ba] reachable. When [ba] is over memory that OCaml does not
      manage, such as memory a C library allocated, its owner must keep that
      memory alive for as long as the buffer and its views are reachable.

      Raises [Invalid_argument] if [ba]'s kind is [Int] or [Nativeint], which
      are no storage format, or if [ba]'s first element does not lie at a
      multiple of its size (of one component for complex kinds), as a bigarray
      that [Unix.map_file] maps from an unaligned [pos] may not. *)

  val of_file : string -> (t, string) result
  (** [of_file path] is the bytes of the regular file [path] on {!disk}, as
      [UInt8] elements, one per byte, for reading: {!copy} reads them, and
      refuses to write them. Its length is the file's size when it is opened.

      The buffer and its views name the file that was opened. The disk keeps a
      descriptor of it open while it is among the 64 files it used last, and
      reads it through that descriptor even once [path] is renamed, removed or
      replaced. A descriptor it closed to open other files is opened again by
      [path], and a copy raises [Sys_error] naming the file if [path] no longer
      names that file, or names it changed by another writer since: renamed,
      removed, replaced, or changed in place. So a file the process keeps
      buffers of must keep its path, and change only through them. A file
      changed in place while its descriptor is open changes what they read, and
      a read past a new end of the file raises. The mapping of its pages lasts
      while its borrows ({!borrow}) are reachable, open or not; a borrow of its
      pages asks more of the file.

      [Error why] naming [path] if it cannot be opened for reading or is not a
      regular file. *)

  val create_file : string -> int -> (t, string) result
  (** [create_file path n] is the file at [path], created, or emptied if it
      exists, and sized to [n] bytes, as a buffer of [n] [UInt8] elements on
      {!disk} for reading and writing: {!copy} reads and writes them. Its bytes
      read as zero until they are written. It names the file as {!of_file} does,
      and its own writes do not change which file that is. A write reaches the
      file when {!copy} returns, and the storage once the system flushes the
      file, which a sync of the file forces.

      [Error why] naming [path] if it cannot be created.

      Raises [Invalid_argument] if [n < 0]. *)

  val borrow : device -> t -> (t, string) result
  (** [borrow d b] is a borrowed buffer on [d] over the memory of the buffer [b]
      of [d]'s machine, without a copy, of [b]'s format and length. A write
      through either is seen through the other, once the devices involved are
      synchronized. The result keeps [b] reachable. [borrow d b] is [Ok b] for
      [b] on [d], so [borrow host b] is [Ok b] for [b] on the host. Work that
      reads or writes through the result lists it in its {!submit}'s [touches],
      which reaches the memory's device too.

      A borrow of a borrow maps the memory the first one maps: [borrow d b] for
      [b] a borrow on another device of host memory is [d]'s borrow of that host
      memory.

      [d] maps [b]'s memory by where it lives:
      - a file's bytes on the {!disk}, through the file's pages, below;
      - system memory, which [d]'s mapping of host memory ({!Driver.mapping})
        maps: memory of [d]'s host ({!host_of}), of a device described as
        [Host_visible] ({!Driver.memory}), such as Metal and test devices over
        the host's memory, and the pinned memory of any device ({!memory}).
        Through an [Identity] mapping, such as the {!host}'s, the borrow is over
        the memory's host addresses, without a copy;
      - the own memory of a [Device_local] device, such as a CUDA, AMD or NV
        GPU's, which [d]'s driver maps ({!Driver.device}'s [peer]) once per
        region, even where a memory BAR gives it a host address. The mapping
        lasts until that device frees the memory, and [d] can reach it until
        then. The device's mapped memory ({!memory}) is the exception for its
        host, which borrows it over its host address, as it does pinned memory.

      Through a [Pages] mapping, [d] maps the whole host memory that [b] is a
      view of, once: the borrows on [d] of views of that memory share one
      mapping, which [d] releases once they are all unreachable. A mapping
      covers whole pages, so that memory must start on a page. Host buffers that
      {!create} makes start on one from 64 KiB (four pages where pages are
      larger), and smaller ones cannot be borrowed, wherever they start: {!copy}
      moves them through staging memory, and {!reach} stages them for a device's
      work. Memory-mapped files start on a page. Memory that {!of_bigarray}
      wraps borrows if it starts on one. On CUDA, mapping page-locks the memory,
      which must be writable: memory mapped read-only cannot be borrowed there.
      A buffer of no bytes always borrows, mapping nothing.

      A buffer [b] of a file on the {!disk} ({!of_file}) is borrowed from the
      file's pages: the disk maps the whole file into host memory at the first
      borrow of any of its buffers, and keeps the mapping while a borrow of it
      or the file's buffers are reachable. The mapping is copy-on-write: a write
      through a borrow changes the process's copy of a page, never the file, and
      is seen through the other borrows of that page. A device other than the
      host has the system read the borrowed bytes ahead of their use; the host
      reads them as it needs them. The file must not change while a borrow is
      reachable: a write to it, by a {!copy} or by another process, may show
      through pages the process has not written, and a truncation makes a read
      of the pages cut off kill the process. Only a device that shares the
      host's memory ({!shares_host_memory}) borrows a file's bytes; the others
      {!copy} them into their memory.

      [Error why] if [d] cannot map that kind of memory (a host maps no
      [Device_local] memory but mapped memory, and another machine's host none
      of its devices'), if, through a [Pages] mapping, [b] is a host buffer
      {!create} made of fewer than 64 KiB, the memory [b] is a view of does not
      start on a page, or [d]'s driver refuses to map it, with the driver's
      reason; for [b] on the disk, if [d] does not share the host's memory, if
      [b]'s first byte is not aligned to the size of one of its elements, or if
      the system cannot map the file, naming it.

      Raises [Invalid_argument] if [b] is on another machine or is dead
      ({!Claim.consume}), and {!Lost} if [d] is lost. *)

  (** The type for what a device's work does with a buffer it reaches. *)
  type access =
    | Read  (** The work reads the buffer. *)
    | Read_write  (** The work reads and writes it. *)

  val reach : device -> t -> access -> (t, string) result
  (** [reach d b access] is a buffer on [d] over [b]'s bytes, for the work of
      [d] that [access]es them: [borrow d b] where [d] maps [b]'s memory, and
      otherwise, for fewer than 64 KiB of memory of this machine's host
      ({!shares_host_memory}), a {e staged} buffer: memory of [d] of [b]'s
      format and length ({!is_staged}). A host buffer of 64 KiB or more starts
      on a page when {!create} makes it, and is borrowed.

      A {!submit} whose [touches] hold a staged buffer copies [b] into it, with
      its devices taken, after its waits and once the work that last used the
      staged memory is done: its work reads [b] as it was then. When [access] is
      [Read_write], [submit] then waits for its work, with no device taken, and
      copies the staged buffer back into [b] before it returns.

      [Error why] as {!borrow}, and, naming its size, if [b] is 64 KiB or more
      of host memory that [d] does not map, such as a bigarray's that does not
      start on a page: a copy on every submission would cost its size each time
      and hold it twice.

      Raises as {!borrow}. *)

  val is_staged : t -> bool
  (** [is_staged b] is [true] iff [b] is a staged buffer that {!reach} made. *)

  val device : t -> device
  (** [device b] is the device whose memory [b] is. *)

  val dtype : t -> Nx_dtype.Scalar.t
  (** [dtype b] is the format of [b]'s elements. *)

  val length : t -> int
  (** [length b] is the number of elements of [b]. *)

  val nbytes : t -> int
  (** [nbytes b] is [b]'s size in bytes. *)

  val is_borrowed : t -> bool
  (** [is_borrowed b] is [true] iff [b] is borrowed. *)

  val view : t -> offset:int -> Nx_dtype.Scalar.t -> int -> t
  (** [view b ~offset s n] is the [n] elements of format [s] from byte [offset]
      of [b] on. It shares [b]'s memory: a write through either is seen through
      the other.

      Raises [Invalid_argument] if [offset] or [n] is negative, if [n] elements
      of [s] take more than [max_int] bytes, if the view's bytes do not lie
      inside [b]'s, or if its first byte is not aligned to the size of one
      element of [s], except on {!disk}, whose files are read and written at any
      byte. *)

  val spans : t -> bool
  (** [spans b] is [true] iff [b]'s bytes are all of the memory it lies in,
      below its borrows, as those of a buffer {!create} or {!of_bigarray} made
      are, and of a {!borrow} of one, and not those of a {!view} of part of it.
      A borrow's mapping may hold more than the memory, such as the whole pages
      around it: [spans] judges the memory it maps. *)

  val overlaps : t -> t -> bool
  (** [overlaps b b'] is [true] iff [b] and [b'] share a byte of memory: through
      views of one memory, through a borrow and the memory it maps, or through
      two bigarrays over the same bytes. Buffers of no bytes overlap nothing, a
      file's bytes do not overlap the borrows of its pages, which are
      copy-on-write, and two opens of one file ({!of_file}, {!create_file}) are
      two memories. *)

  (** Claims on memory.

      Every view and borrow of one memory shares one count of claims. A reader
      claims the memory while it reads it on the host, and a compiled call that
      writes over memory it was handed holds it exclusive, so that no reader
      sees the write. Claims never wait: a claim that cannot be had raises, or
      for an exclusive one, reports [false].

      A claimed memory may be consumed: every buffer over it made before becomes
      dead, and reaching a dead buffer's bytes ({!address}, {!bigarray},
      {!copy}, {!borrow}, {!Program.call}, {!submit}, or a kernel reading it)
      raises [Invalid_argument] with the consumption's reason. A consumption
      releases nothing: the memory lives while a buffer reaches it.

      Device work takes no claim: it is ordered after the work queued before it.
      Memory reached outside the claims, by whoever holds the bigarray of
      {!of_bigarray} or by a holder {!Claim.export} names, is never exclusive. A
      raw {!address} or {!bigarray} is outside the claims, and its holder
      answers for it. *)
  module Claim : sig
    type buffer := t

    val read : buffer -> unit
    (** [read b] claims [b]'s memory for reading, beside other readers.

        Raises [Invalid_argument] if [b] is dead, or if the memory is held
        exclusive. *)

    val release : buffer -> unit
    (** [release b] ends a {!read} of [b]'s memory. It accepts a dead [b].

        Raises [Invalid_argument] and changes nothing if the memory has no read
        claim. *)

    val try_exclusive : buffer -> bool
    (** [try_exclusive b] turns the caller's read claim on [b]'s memory into an
        exclusive one. It is [false], and changes nothing, if the memory has
        other claims or is exported. *)

    val finish : buffer -> unit
    (** [finish b] turns the exclusive claim on [b]'s memory back into the read
        claim it came from, which {!release} then ends. It accepts a dead [b].

        Raises [Invalid_argument] if the memory is not exclusive. *)

    val export : buffer -> unit
    (** [export b] is a read claim on [b]'s memory for a holder outside the
        claims, such as a library given its bytes without a copy, which is never
        released: the memory is never exclusive again. A buffer {!of_bigarray}
        makes starts with one, for whoever holds the bigarray, and so does one
        {!of_file} or {!create_file} makes, for the file.

        Raises [Invalid_argument] as {!read} does. *)

    type t
    (** The type for the claims of a {!with_}. *)

    val with_ : read:buffer list -> donate:buffer list list -> (t -> 'a) -> 'a
    (** [with_ ~read ~donate f] claims the memory of [read] and of [donate] for
        reading, then tries each value of [donate] (its buffers, one per device)
        exclusive, and is [f] of the claims. A value is exclusive if each of its
        buffers {!spans} its memory and can be had exclusive; its buffers are
        otherwise left read. Every claim is released when [f] returns or raises.

        Raises [Invalid_argument] before [f], releasing what it claimed, if a
        buffer is dead, if a memory is held exclusive, or if a buffer of
        [donate] overlaps another of [read] or [donate]. *)

    val exclusive : t -> buffer -> bool
    (** [exclusive c b] is [true] iff [c] holds [b]'s memory exclusive: the
        caller may write it in place. *)

    val consume : t -> why:string -> buffer -> buffer
    (** [consume c ~why b] consumes [b]'s memory with the reason [why]: [b] and
        every buffer over the memory made before are dead. It is a buffer over
        the same memory, live, which the caller may write in place only if [c]
        holds it {!exclusive}; with a read claim only, it must not write it.

        Raises [Invalid_argument] if [c] does not claim [b]'s memory, if [b] is
        dead, or if [b] does not {!spans} its memory: consuming a window would
        kill the rest. *)
  end

  val copy : src:t -> dst:t -> unit
  (** [copy ~src ~dst] copies [src]'s bytes into [dst] and returns once they are
      there. It first synchronizes the devices of [src] and [dst], and the
      devices whose memory their borrows map. A copy between the memory of two
      devices counts in the [bytes_out] of the device whose memory [src] is and
      in the [bytes_in] of [dst]'s; a borrow's memory is its host's.

      Between memory that the host addresses, the host copies the bytes.
      Otherwise a device copies them on its copy queue, as work on its timeline:
      the device of [dst] when only [src] is host-addressable memory, the device
      of [src] otherwise. It copies directly between memory it addresses: its
      own, host memory it allocated or maps, and host memory another device
      allocated, which it maps for the copy. Other host memory goes through the
      host's staging memory, two 64 MiB slots of host memory made at the first
      such copy and kept for the life of the process, which each device maps at
      its first such copy. Between two devices whose memory the host does not
      address, it moves the bytes to the other device's memory when it can, and
      through the staging memory otherwise.

      On another machine, its host copies and stages as this one's does, in its
      own memory. Between two machines, the bytes cross over a link that an
      opened device carries between them, such as an RDMA network adapter on
      each, and otherwise in chunks of 64 MiB through the hosts: from memory the
      source's host addresses (the source, or its host's staging memory),
      through this process, to memory the destination's host addresses. A link
      that fails loses its devices, which may still write [dst].

      A copy from the {!disk} reads the file, and a copy to it writes the file:
      straight into or from memory that the host addresses, and otherwise
      through the staging memory, which the other buffer's device copies from or
      into while the host reads or writes the other slot. A copy from the disk
      to the disk goes through the staging memory too.

      Raises [Invalid_argument] if [src] and [dst] have different sizes in
      bytes, if they {!overlaps}, if [dst] is a buffer of {!of_file}, if one is
      on the disk and the other on another machine, or if the host does not
      address the memory of a device that has no copy queue; [Sys_error] naming
      the file if a read or a write of a file fails or a read reaches its end,
      as in a file truncated since it was opened, which loses no device; {!Lost}
      if a device involved is lost or is lost by the copy (it does not signal in
      time, its driver reports a fault or errs while the copy is enqueued, or
      the connection to its machine fails), or if a lost device can reach [src]
      or [dst]; {!Out_of_memory} if a host cannot allocate its staging memory;
      and [Failure] with the driver's reason if a device cannot map the staging
      memory, which loses no device. *)

  val bigarray :
    ('a, 'b) Bigarray.kind -> t -> ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
  (** [bigarray k b] is the bytes of the host buffer [b] read as elements of
      kind [k], without a copy: [nbytes b / Bigarray.kind_size_in_bytes k] of
      them, in the machine's byte order. Writing through it mutates [b]. It
      keeps [b]'s memory alive for as long as it is reachable, except memory
      that OCaml does not manage: the memory {!of_bigarray}'s caller keeps
      alive, and another device's memory that a borrow on the host maps
      ({!borrow}), which that borrow keeps alive.

      Access through the view is outside the devices' ordering: {!synchronize}
      {!host} first, and the device whose memory a borrow maps, to see the work
      that touched it.

      Formats with no kind of their own are read as their storage kind and
      decoded with {!Nx_dtype.Scalar.decode}: [BFloat16] as [Int16_unsigned],
      the float8 formats as [Int8_unsigned], [Int4] and [UInt4] as
      [Int8_unsigned] holding two per byte. With an odd number of 4-bit
      elements, the last byte's high nibble is not [b]'s, and a write to it may
      change memory outside [b].

      Raises [Invalid_argument] if [b] is not on {!host}, if [k] is [Int] or
      [Nativeint], which are no storage format, or if [b]'s bytes are not a
      whole number of elements of [k], aligned to the size of one element (of
      one component for complex kinds), and {!Lost} if a lost device can reach
      [b]. *)

  (** {2:low Low-level}

      For the libraries that submit work over buffers. C code reads the host
      address of a live buffer of this machine whose memory the host addresses,
      such as every buffer on {!host}, with [nx_device_buffer_host] from the
      header [nx_device.h], which [nx.device] installs: it reads the value's
      fields without allocating, and the address stays valid while the value is
      reachable. Reading another buffer's is undefined behaviour. *)

  val address : t -> nativeint
  (** [address b] is the address of [b]'s first byte as [b]'s device's work
      addresses it: the host address on a host, and the position of [b]'s first
      byte in its file on the {!disk}. It is
      [Driver.Region.address (Driver.Region.of_buffer b) + offset b].

      Raises [Invalid_argument] if [b] is dead. *)

  val offset : t -> int
  (** [offset b] is the number of bytes of [b]'s memory
      ({!Driver.Region.of_buffer}) before [b]'s first byte. A {!view} at
      [offset] is [offset b + offset] into the same memory. *)
end

(** {1:programs Programs} *)

(** Programs loaded on a device. *)
module Program : sig
  type device := t

  type t
  (** The type for programs: a function of a binary, loaded on a device. A
      program of the {!host} is run by {!call}; a GPU's are launched by the
      libraries that submit work to it. *)

  val load : device -> binary:string -> name:string -> (t, string) result
  (** [load d ~binary ~name] is the function [name] of [binary], a compiled
      library in [d]'s format: a metallib for Metal, a CUDA module (cubin,
      fatbin, or PTX, which the driver compiles) for CUDA, a code object for
      AMD, a cubin for NV.

      [d] loads [binary] once while it is reachable: while a program of it, or a
      buffer of its code ({!code}), is. Loading the same binary on [d] again
      until then finds the functions of that load, with the same {!handle}s.
      Once none is reachable, the binary is unloaded after the work [d]
      submitted until then is done, and loading it again loads it anew.

      For the {!host}, [binary] is a 64-bit little-endian ELF relocatable object
      for the machine's instruction set, as
      [clang -c -fPIC --target=ARCH-none-unknown-elf] makes it with [ARCH]
      [x86_64] or [arm64]. Its code may call the functions of the libraries the
      process has loaded, such as the C and math libraries, and of the
      compiler's runtime library ([libgcc_s] on Linux). The host has its own
      copies of the runtime's 16-bit float conversions, [__extendhfsf2],
      [__truncsfhf2] and [__truncsfbf2], which compilers call where the machine
      has no instruction for them, even in code that converts nothing. The code
      has no writable data, such as [.data] or [.bss]: the host loads it into
      memory that is executable and never writable.

      [Error why] if [d] loads no programs, or if its driver rejects [binary] or
      finds no function [name] in it, with the driver's reason (on the host, the
      reason it cannot load [binary], such as a symbol that no library defines).
      [why] starts with [d]'s {!name}. [d] stays usable.

      Raises {!Out_of_memory} if [d]'s driver has no memory for the code, and
      {!Lost} if [d] is lost, or is lost by the load. *)

  val device : t -> device
  (** [device p] is the device [p] is loaded on. *)

  val name : t -> string
  (** [name p] is the name of [p]'s function. *)

  val handle : t -> nativeint
  (** [handle p] is the driver's object for [p], such as a
      [MTLComputePipelineState], a [CUfunction], the address of an AMD kernel
      descriptor, or the address of an NV function's first instruction. On the
      host, it is the address of the function's first instruction. *)

  val code : t -> Buffer.t option
  (** [code p] is a buffer of the device memory [p]'s binary lies in, for a
      device whose driver loads code into its memory, such as AMD's and NV's.
      Work that launches [p] lists it among the buffers it touches, and what
      keeps the work, such as a compiled schedule, keeps the buffer: the binary
      stays loaded while the buffer is reachable, and its memory until that work
      is done. *)

  val keep : t -> Buffer.t -> Buffer.t
  (** [keep p b] is [b], as a buffer that also keeps [p]'s binary loaded while
      it is reachable, such as a word holding [p]'s {!handle} that a host
      program reads to launch it. *)

  type split = {
    extent : int;  (** The iterations, [0] to [extent - 1]. *)
    blocks : int;  (** The contiguous blocks they are cut into. *)
    lo : int;  (** The slot of [values] that takes a block's first. *)
    hi : int;  (** The slot of [values] that takes the one after its last. *)
  }
  (** The type for splits of a host program's work into blocks that the host's
      cores run at once. Block [i] runs the iterations [i * extent / blocks] to
      [(i + 1) * extent / blocks - 1]. *)

  val workers : unit -> int
  (** [workers ()] is the number of threads a split {!call} runs on: the host's
      cores that pay for compute-bound work, its performance cores where it also
      has slower ones. *)

  val call : ?split:split -> t -> Buffer.t array -> int array -> unit
  (** [call ~split p buffers values] runs the host program [p] in the calling
      domain and returns once it returns. [p] is called as the C function

      {v void f(void **buffers, const int64_t *values); v}

      given the host address of each buffer's first byte ({!Buffer.address} on
      the host) and each value as a 64-bit integer, in order. A value lies
      between -2{^ 62} and 2{^ 62} - 1, the range of an OCaml [int] on 64-bit
      platforms, which holds timeline values, sizes and device addresses. It
      follows the platform's C calling convention: on Windows, an object
      compiled for x86_64 ELF declares [__attribute__((ms_abi))] on its entry
      and on the library functions it calls, and code for arm64 leaves the
      register [x18] alone ([-ffixed-x18]), which macOS and Windows reserve.

      The OCaml runtime is released while [p] runs, so other threads and domains
      go on; the buffers stay reachable until it returns. The call is outside
      the devices' ordering, as an access through {!Buffer.bigarray} is: it does
      not take the host, and several domains may run programs at once.
      Synchronize the devices whose work touches [buffers] first. While a
      {!Profile} is taken, the call is a span of the calling domain's lane of
      the host, named after [p].

      A program of another machine's host runs there, on memory that host
      addresses, with the same ABI. The call is sent in order after the earlier
      operations on that machine and before the later ones, and returns once
      sent: the program has run when the next operation that waits for an answer
      from the machine, such as {!synchronize} of its host, returns. A program
      that fails there loses the host.

      With [split], [p] runs once per block, with [values.(split.lo)] and
      [values.(split.hi)] set to the block's iterations, on {!workers} threads
      of the host's pool, which nx.cpu's kernels share; the call returns once
      every block has. Blocks run in any order and at once, so a block must not
      read what another writes. A split call waits for the pool while another
      domain's work holds it. A program of another machine's host runs as one
      block.

      Only hosts run programs; a program of another device is launched by the
      libraries that submit work to it. [buffers] may be any buffers whose
      memory [p]'s host addresses: its own, and that of the devices of its
      machine whose memory it addresses, such as Metal's, CUDA's pinned memory
      and the test devices of {!Driver.host_memory}.

      Raises [Invalid_argument] if [p]'s device runs no programs or does not
      address the memory of a buffer of [buffers], if [split.extent < 0],
      [split.blocks < 1], [split.extent * split.blocks] exceeds [max_int],
      [split.lo] or [split.hi] is no slot of [values], or [split.lo = split.hi],
      and {!Lost} if a lost device can reach a buffer of [buffers]. *)
end

(** {1:stats Statistics} *)

(** Device statistics. *)
module Stats : sig
  type t
  (** The type for a snapshot of a device's statistics. *)

  val allocated : t -> int
  (** [allocated s] is the bytes of owned memory in buffers that
      {!Buffer.create} returned and that were not yet returned to the device,
      and in the code of loaded programs ({!Program.code}). *)

  val cached : t -> int
  (** [cached s] is the bytes in the device's cache: memory allocated from the
      driver and held for reuse. The host's cache holds the memory of collected
      buffers of 64 KiB or more for the next buffers of their sizes: up to a
      major cycle's share of the program's memory, or 32 MiB where that is less.
  *)

  val retained : t -> int
  (** [retained s] is the bytes of the device's own memory that it retains
      because the work that last used them could not be waited for. Those of its
      own and mapped memory count against its budget. Retained borrows are not
      counted. *)

  val bytes_in : t -> int
  (** [bytes_in s] is the bytes copied into the device from another one. *)

  val bytes_out : t -> int
  (** [bytes_out s] is the bytes copied from the device to another one. *)

  val diff : t -> t -> t
  (** [diff s s'] is what changed from [s] to [s']: each count of [s'] minus
      that of [s]. The [allocated] of a diff is the bytes allocated in between
      and not returned. *)
end

val stats : t -> Stats.t
(** [stats d] is a snapshot of [d]'s statistics. The memory of buffers collected
    before the call counts as returned, unless [d] is lost. It answers on a lost
    device, whose retained bytes it reports. *)

(** {1:profiling Profiling} *)

(** Profiles of devices' work.

    One profile of every device is taken at a time, between {!start} and
    {!stop}. While it is taken, the devices record {!event}s: {e spans} of work
    on a device, changes of its allocated memory, the programs it loads and, on
    request, the {e counters} and {e traces} of its programs' runs. Spans come
    from the host ({!span}), from the runtime's own copies and calls of host
    programs, and from the libraries that submit work ({!Submission.record}).
    Every time is on the host's clock, {!now}: the times a device stamps on its
    own clock are calibrated against it when the profile is stopped.
    {!output_chrome_trace} writes the events in Chrome's trace event format,
    which Perfetto ({{:https://ui.perfetto.dev}ui.perfetto.dev}) and
    [chrome://tracing] load.

    When no profile is taken, recording costs a read of one atomic value and
    allocates nothing. *)
module Profile : sig
  type device := t

  (** The type for profile events. Times are nanoseconds of the host clock,
      {!now}. *)
  type event =
    | Span of {
        device : device;  (** The device the work ran on. *)
        lane : string;  (** Its track within the device. *)
        name : string;  (** What the work was. *)
        start : int;  (** When it started, on the host clock. *)
        stop : int;  (** When it stopped, on the host clock. *)
      }
        (** Work that ran on a lane of a device. The host's lanes are its
            domains, ["domain 0"], ["domain 1"], ...; a device's copy queue runs
            the runtime's copies on its ["copy"] lane; the libraries that submit
            work name their own lanes. *)
    | Allocation of {
        device : device;
        time : int;
        allocated : int;
            (** The device's {!Stats.allocated} bytes from [time] on. *)
      }
        (** A change of the memory a device allocated. The {!host} records each
            allocation, and the return of the memory of the buffers it allocated
            while a profile was taken: its other buffers return theirs without a
            record, which its next record counts. *)
    | Load of {
        program : Program.t;
        binary : string;  (** The binary it was loaded from. *)
        time : int;  (** When it was loaded. *)
      }  (** A program loaded on its device. *)
    | Counters of {
        device : device;  (** The device the program ran on. *)
        name : string;  (** The name of the program's function. *)
        start : int;  (** When the counted run started. *)
        stop : int;  (** When it stopped. *)
        counters : (string * int array) list;
            (** Each counter the profile asks for ({!start}) and its count
                during the run, one count per unit of the device's hardware that
                counts it, in the order the device's library states. *)
      }
        (** The counters of a run of a program, read once the run completed. The
            run is timed by the device, within the span the libraries that
            submit work record for it, if they record one. *)
    | Trace of {
        device : device;  (** The device the program ran on. *)
        name : string;  (** The name of the program's function. *)
        start : int;  (** When the traced run started. *)
        stop : int;  (** When it stopped. *)
        part : int;
            (** The part of the device that wrote it, such as an AMD GPU's
                shader engine. *)
        data : string;
            (** The trace as the device wrote it, which the device's library
                reads. *)
      }
        (** The thread trace of a part of a device during a run of a program,
            read once the run completed and timed as {!Counters} are. A device
            that decodes its traces also reports the spans of their waves. *)
    | Overwritten of {
        device : device;  (** The device that lost them. *)
        time : int;  (** When the device found them overwritten. *)
        runs : int;  (** The runs whose counters and traces are lost. *)
      }
        (** Runs whose counters and traces the device overwrote before it could
            read them: a device keeps a bounded number of runs between two of
            its synchronizations. *)

  type t
  (** The type for profiles being taken. *)

  val start : ?counters:string list -> ?trace:bool -> unit -> t
  (** [start ()] starts taking a profile of every device, which only its holder
      stops. Each run of a program on a device that counts [counters] (defaults
      to none) has a {!Counters} event: their names are the device's, as its
      library lists them, and work on a device that has no counter of such a
      name raises [Invalid_argument] naming it when its library encodes the
      work. A device that counts nothing, such as the {!host}, has no
      [Counters]. With [trace] (defaults to [false]), each run of a program on a
      device that traces has a {!Trace} event of each part of the device that
      traced it.

      Raises [Invalid_argument] if a profile is being taken or if [counters]
      names a counter twice. *)

  val counters : unit -> string list
  (** [counters ()] is the counters the profile being taken asks for, if any.
      The libraries that encode work read it when they encode work, which then
      counts these counters on every run: work encoded under one profile does
      not count those of another, so a library that keeps encoded work keeps it
      for each value of [counters ()]. *)

  val traced : unit -> bool
  (** [traced ()] is [true] iff the profile being taken asks for traces. The
      libraries that encode work read it as they read {!counters}, and keep
      encoded work for each value of it. *)

  val stop : t -> event list
  (** [stop p] stops taking [p], and is its events, in time order and, at equal
      times, longest first, then in the order they were recorded. It first
      synchronizes the devices whose recorded spans are still to be read and, if
      [p] asks for counters or traces, those that report them, and calibrates
      the clocks of the devices that stamp times on their own. The unread spans
      of a device lost meanwhile are left out; its next operation raises
      {!Lost}.

      Raises [Invalid_argument] if [p] is not being taken: it was stopped
      already. *)

  val now : unit -> int
  (** [now ()] is the host clock: nanoseconds of the system's monotonic clock,
      which starts at an unspecified point. C code reads it with
      [nx_device_now_ns] from the header [nx_device.h]. On macOS it is the clock
      of Metal's command buffer times. *)

  val enabled : unit -> bool
  (** [enabled ()] is [true] iff a profile is being taken. The libraries that
      submit work read it to decide whether to stamp their work at all. *)

  val span : string -> (unit -> 'a) -> 'a
  (** [span name f] is [f ()]. While a profile is taken, it records a span named
      [name] on the lane of the calling domain of the {!host}, from the call
      until [f] returns or raises. *)

  val output_chrome_trace : out_channel -> event list -> unit
  (** [output_chrome_trace oc events] writes [events] to [oc] in Chrome's trace
      event format, JSON: a process for each device, named after it, with a
      thread for each of its lanes; a complete event for each span; a counter
      [memory] for each change of memory; an instant event for each program
      load, with the program's handle; a complete event for each run's counters
      ({!Counters}), with the sum of each counter as arguments, on the device's
      lane [counters]; an instant event for each trace ({!Trace}), with its part
      and its bytes, whose waves are spans of their own; and an instant event
      for overwritten runs. Times are microseconds from the earliest event.
      Malformed UTF-8 in names becomes U+FFFD. [oc] is neither flushed nor
      closed. *)
end

val staging : t -> Buffer.t
(** [staging h] is the host [h]'s staging memory: two slots of 64 MiB of host
    memory, made at the first staged copy or call and kept for the life of the
    process. {!Buffer.copy} stages through it the bytes no device of a copy can
    address. The libraries that submit work stage copies through it too: work
    that uses it touches it, and holds [h] taken while it writes or queues a use
    of a slot, which its {!submit} does by touching it. A host fill of a slot
    first waits for every queued use.

    Raises [Invalid_argument] if [h] is no host or is the {!disk}, and
    {!Out_of_memory} if [h] cannot allocate it. *)

(** {1:submitting Submitting work}

    For the libraries that submit work to devices. Work is submitted inside
    {!submit}, which takes the devices involved, gives each device's work the
    value it signals on completion, and says what the work must wait for. *)

(** Submissions in progress. *)
module Submission : sig
  type device := t

  type t
  (** The type for the submission {!submit} runs [f] with. *)

  val value : t -> device -> int
  (** [value s d] is the value [d]'s work in [s] signals on completion: one more
      than {!submitted}[ d].

      Raises [Invalid_argument] if [d] is not a device of [s]. *)

  val waits : t -> (device * int) list
  (** [waits s] is the work of other devices that [s]'s work must wait for
      before it touches its buffers: [(d', v)] is complete once [d'] signals
      [v], which its work stores into {!signal_word}[ d'].

      There is one pair for each device of [D], by the order the devices were
      opened. [D] is the devices of [s] and the devices of the buffers [s]
      touches ({!Buffer.device}), less those whose waits cannot be encoded: a
      device whose work does not store its values into its signal word
      ({!Driver.Signal}), and one whose signal word a device of [s] does not
      address, of another machine or with no mapping of host memory. Hosts and
      the disk are never in [D]. [D], and so the shape of [waits s], depends on
      these devices alone: a program built once for them takes the same
      arguments at every submission, and only the values change.

      [v] is the latest value of [d'] whose work touched the memory the buffers
      reach, or a value [d'] has signaled already, whose wait does nothing.
      Before [f] runs, {!submit} waits on the host for any other work that
      touched that memory: that of devices outside [D]. A device's own earlier
      work is ordered by its vendor's rule. *)

  val wait : t -> device -> int -> unit
  (** [wait s d v] blocks the host until [d] has signaled [v], for work that
      [s]'s devices cannot wait for on their queues. [d] may be any device [s]
      took, the devices of [s] included: a submitter that must see the previous
      use of its memory complete before it rewrites it, such as a program's
      arguments, waits for the value that use signals. It returns at once for a
      value a wait already saw signaled.

      Raises [Invalid_argument] if [s] did not take [d] or [v > submitted d],
      and {!Lost} if [d] does not signal [v] within its {!timeout}. *)

  val record : t -> device -> lane:string -> name:string -> Buffer.t -> unit
  (** [record s d ~lane ~name stamps] records a span of [d]'s work in [s] named
      [name] on [lane] of [d]. [stamps] is four [UInt64], two slots of 16 bytes:
      the work writes its start time into the second word and its stop time into
      the fourth, on [d]'s clock. The span is kept if [f] returns, and its
      stamps are read at [d]'s next synchronization, which waits for the work:
      they must stay the work's until then, and a later record of the same
      stamps before then replaces this one. It does nothing unless a profile is
      being taken ({!Profile.enabled}).

      Raises [Invalid_argument] if [d] is not a device of [s], or if [stamps] is
      not four [UInt64] that [d]'s host addresses. *)

  val copied : t -> src:device -> dst:device -> int -> unit
  (** [copied s ~src ~dst n] counts [n] bytes that [s]'s work copies from
      [src]'s memory into [dst]'s in [src]'s {!Stats.bytes_out} and [dst]'s
      {!Stats.bytes_in}. A copy within one device's memory counts nothing.

      Raises [Invalid_argument] if [s] did not take [src] or [dst], or if [n] is
      negative. *)
end

val submit : t list -> touches:Buffer.t list -> (Submission.t -> 'a) -> 'a
(** [submit ds ~touches f] is [f s], where [s] submits work to the devices of
    [ds] whose memory accesses are through the buffers of [touches]. [ds] is a
    set.

    It takes the devices of [ds], those of the buffers of [touches], and those
    whose memory the buffers reach: a borrow's device and the device of the
    memory it maps. It waits on the host for the work [s]'s work cannot wait for
    itself ({!Submission.waits}) and until each device of [ds] has room in its
    queues for a submission ({!Driver.device}'s [room]), then runs [f s]. For
    each device [d] of [ds], [f] enqueues work that:
    - completes after all of [d]'s earlier work;
    - waits for each pair of {!Submission.waits}[ s], on the device by reading
      {!signal_word}[ d'], or on the host with {!Submission.wait};
    - then signals {!Submission.value}[ s d], by storing it into
      {!signal_word}[ d] or in the vendor's own way ({!Driver.Signal}).

    A device's values thus complete in order, which {!signaled} and
    {!synchronize} rely on. How work waits for the value before its own is its
    vendor's encoding, which its library's low-level section states.

    When [f] returns, [s] commits: each value is its device's {!submitted}
    value, {!synchronize} on each device whose memory the buffers reach waits
    for the work, and the spans of {!Submission.record} are kept. The memory of
    [touches] is stamped with the values: once unreachable, it returns to its
    device, or to the system, only after the work is done. So [touches] lists
    every buffer the work reaches, its code and arguments included: a buffer it
    leaves out may be freed while the work still runs. If [f] raises, nothing is
    committed, so [f] may raise only before it enqueues any work. A {!Lost} that
    [f] raises loses its device, as a driver error after work was enqueued
    leaves the queue in an unknown state.

    A staged buffer of [touches] ({!Buffer.reach}) is filled from the memory it
    stands in for before [f] runs. [submit] returns once every call is queued,
    except when [touches] hold a staged buffer the work writes: [submit] then
    waits for the work and copies that buffer back before it returns.

    Inside [f], the devices are used only through {!Submission}, {!submitted},
    {!signaled}, {!signal_word}, the buffers' properties and low-level
    accessors, {!Program.handle}, {!Program.call} of a host program, and
    {!Profile.enabled}. Vendor setup that allocates, such as AMD's scratch
    memory, runs before [submit]. [submit] allocates no device memory.

    Raises [Invalid_argument] if [ds] is empty or has a host or the disk, which
    run no submitted work, or if a buffer of [touches] is on the disk or dead
    ({!Buffer.Claim.consume}), and {!Lost} if a device it takes is lost, if a
    lost device can reach a buffer of [touches], or if a device of [ds] has no
    room in its queues within its {!timeout}. *)

val submitted : t -> int
(** [submitted d] is the value [d]'s last submitted work signals, [0] before any
    submission. *)

val signaled : t -> int
(** [signaled d] is the last value [d] signaled. Work that signals
    [v <= signaled d] has completed. *)

val signal_word : t -> Buffer.t
(** [signal_word d] is one [UInt64] that [d]'s host and [d]'s work address, into
    which [d]'s work stores the values it signals, and which {!signaled} reads.
    It is [d]'s pinned memory when [d]'s memory is [Device_local]
    ({!Driver.memory}), such as page-locked memory on CUDA, and memory of its
    host ({!host_of}) otherwise, so the devices of its machine borrow it
    ({!Buffer.borrow}). A device that signals in its own way ({!Driver.Signal})
    leaves it at [0], as Metal's shared event does. *)

(** {1:drivers Drivers}

    For the vendor libraries that open devices. *)

(** Descriptions of devices.

    A library that opens devices of some kind describes each of them with
    {!device}, once, and returns that value from every later open. A library
    that reaches another machine describes that machine's host with {!host}, and
    the other devices of that machine with {!device} given that host.

    {b The callbacks} of a description run with the device taken, except [link]
    and [dma], which run with none; they must not use their device through this
    module. They report in four ways:
    - [None]: the driver has none;
    - [Error why]: it refuses;
    - [false] from {!signal}'s [wait]: the time ran out;
    - [Failure why]: the device faulted, which loses it ({!Lost}).

    A [finalize] that raises is printed and ignored at exit. A {!depends}
    function that raises [Failure] retains its memory. Blocking driver calls
    should release the OCaml runtime. *)
module Driver : sig
  type device := t

  (** Regions of a device's memory. *)
  module Region : sig
    type t
    (** The type for a range of a device's memory that its driver allocated or
        mapped. *)

    val v : ?host:nativeint -> ?handle:nativeint -> nativeint -> int -> t
    (** [v ?host ?handle address n] is the [n] bytes at [address] as the
        device's work addresses them. [host] is their address on the host of the
        device's machine, if that host addresses them. [handle] is the driver's
        object for them, such as a [MTLBuffer] or the start of a CUDA allocation
        (defaults to [0n]).

        Raises [Invalid_argument] if [n < 0]. *)

    val address : t -> nativeint
    (** [address r] is [r]'s first byte as the device's work addresses it. *)

    val host_address : t -> nativeint option
    (** [host_address r] is [r]'s first byte on the host of the device's
        machine, if that host addresses it. *)

    val handle : t -> nativeint
    (** [handle r] is the driver's object for [r]. *)

    val nbytes : t -> int
    (** [nbytes r] is [r]'s size in bytes. *)

    val of_buffer : Buffer.t -> t
    (** [of_buffer b] is the region [b] lies in; [b] starts {!Buffer.offset}[ b]
        bytes into it. Every view of [b] lies in the same region. *)
  end

  type allocator = {
    alloc : int -> Region.t option;
        (** [alloc n] is a region of [n > 0] bytes of new memory, or [None] if
            the driver has none. *)
    free : Region.t -> unit;  (** [free r] returns [r] to the driver. *)
  }
  (** The type for allocators of a device's memory. Memory the device frees is
      cached for reuse first, and the device synchronizes before it frees it to
      the driver, so no work still uses it. *)

  (** The type for how a device addresses host memory. *)
  type mapping =
    | Identity
        (** The device addresses host memory at its host addresses, as the
            {!host} and the test devices over its memory do. A borrow is the
            memory itself: it maps nothing and calls no driver. *)
    | Pages of {
        map : nativeint -> int -> (Region.t, string) result;
            (** [map a n] maps the [n] bytes of host memory at [a], which start
                on a page, for the device: a region whose host address is at or
                below [a], or [Error why] if the driver refuses. *)
        unmap : Region.t -> unit;
            (** [unmap r] releases a mapping [map] made, once the device
                synchronized. *)
      }
        (** The device's driver maps host memory into its address space, whole
            pages at a time, such as by page-locking it. *)

  type copy = dst:nativeint -> src:nativeint -> int -> signal:int -> unit
  (** The type for enqueueing copies. [copy ~dst ~src n ~signal] enqueues on the
      device's copy queue a copy of [n > 0] bytes from [src] to [dst], after the
      device's earlier work, and then the signal of [signal] once the copy is
      complete. It returns without waiting for either. It raises [Failure] with
      the driver's message if the driver errs, which loses the device. *)

  (** The type for the clocks of a device's timestamps. *)
  type clock =
    | Host_clock
        (** The device's timestamps are readings of the host clock,
            {!Profile.now}, such as the command buffer times of Metal or the
            stamps of host functions that a device's queue runs. *)
    | Device_clock of { hz : int }
        (** The device's timestamps count ticks of a clock of its own, [hz] per
            second. When a profile is stopped, the runtime calibrates it against
            the host clock with its queue's [stamp]. *)

  type queue = {
    copy : copy;
        (** Copies between the device's addresses: the memory it allocated or
            maps. *)
    transfer : device -> copy option;
        (** [transfer d'] copies from the device's memory to the memory of the
            device [d'], [None] if the device cannot. The destination address is
            that of the device's mapping of it ([peer]), or its own address on a
            device without [peer]. *)
    stamp : slot:nativeint -> signal:int -> unit;
        (** [stamp ~slot ~signal] enqueues, after the device's earlier work, the
            write of a timestamp of the device's clock into the second [UInt64]
            of the 16 bytes at the device address [slot], of which it may write
            the first too, and then the signal of [signal]. It returns without
            waiting and raises like [copy]. The runtime stamps its copies while
            a profile is taken, and calibrates a device's own clock with it. *)
    clock : clock;  (** The clock of the device's timestamps. *)
  }
  (** The type for a device's copy queue. *)

  (** The type for how a device's memory is copied, named by who copies it. *)
  type memory =
    | Host_visible of { memory : allocator; mapping : mapping option }
        (** The host addresses all of the device's memory, and copies it.
            [memory] serves {!Buffer.create}, whatever its [~memory]. With
            [mapping], the device borrows host memory ({!Buffer.borrow}); the
            device then shares the host's memory ({!shares_host_memory}). *)
    | Device_local of {
        memory : allocator;
        host_memory : allocator;
        mapped : (allocator * int) option;
        mapping : mapping;
        queue : timeline:Region.t -> queue;
      }
        (** The device's queue copies its own memory, staging through host
            memory it maps.
            - [memory] allocates its own memory, which the host may not address.
            - [host_memory] allocates its pinned memory ({!Buffer.memory}):
              coherent host memory that its work addresses, whose regions have a
              host address.
            - [mapped] is the allocator of its mapped memory ({!Buffer.memory})
              and the bytes of the window it lies in: its own memory that the
              host addresses through that window, whose regions have a host
              address. The window holds at most that many bytes of mapped memory
              and of loaded programs' code ({!Program.code}). With [None], or
              when the window has no room left, mapped memory is pinned memory,
              and code counts in pinned memory.
            - [mapping] maps host memory for {!Buffer.borrow} and for the host's
              staging memory.
            - [queue ~timeline] is its copy queue, given the region of its
              timeline, which the runtime allocates first from [host_memory]:
              its {!signal_word}, a word that aligns what follows, and two slots
              of 16 bytes for the copy queue's timestamps. *)

  type signal = {
    signaled : unit -> int;  (** The last value the device signaled. *)
    wait : int -> timeout_ms:int -> bool;
        (** [wait v ~timeout_ms] waits until the device signaled [v], for at
            most [timeout_ms] milliseconds; [false] if it did not. It raises
            [Failure] with the driver's message if the driver reports a fault,
            which loses the device. *)
  }
  (** The type for how a device signals completion and is waited for. *)

  (** The type for how a device's work completes. *)
  type completion =
    | Poll
        (** Work signals by storing its value into its {!signal_word}, and waits
            poll it. The timeout restarts whenever the word moves. *)
    | Sleep of (timeline:Region.t -> int -> unit)
        (** As [Poll], and, given the region of its timeline, [sleep ms] runs
            once a wait has seen the signal word stay still for 200
            milliseconds, and again each time it returns while the word stays
            still: it blocks for at most [ms] milliseconds, at most 200 and
            never past the timeout, on the device's interrupts or events, and
            raises [Failure] with the driver's message if the device reports a
            fault, which loses the device. Before a wait declares the device
            hung, it runs once more with [ms = 1], so a fault reported late
            still names its cause. *)
    | Signal of (timeline:Region.t -> signal)
        (** The device signals in its own way, given the region of its timeline,
            and never through its {!signal_word}, which stays [0]. The work of
            other devices cannot wait for it on their queues: {!submit} waits
            for it on the host. *)

  type dma = {
    bus : string;
        (** The PCI function that serves the memory, such as ["0000:03:00.0"],
            on its machine. *)
    pages : (int * int) list;
        (** The memory's bus address ranges, as (address, bytes), in order. *)
  }
  (** The type for how the other PCI functions of a machine reach a device's
      memory. *)

  type link = {
    through : device list;  (** The devices that carry the copy. *)
    move : src:Buffer.t -> dst:Buffer.t -> unit;
        (** [move ~src ~dst] copies [src]'s bytes into [dst], of the same size,
            and returns once they are there. It raises [Failure] if a device of
            [through] fails, which loses them all. *)
  }
  (** The type for links, which carry copies between the memory of devices of
      two machines. *)

  type io = {
    read : src:nativeint -> dst:nativeint -> int -> unit;
        (** [read ~src ~dst n] copies the host's [n] bytes at [src] into the
            process's memory at [dst]. *)
    write : dst:nativeint -> src:nativeint -> int -> unit;
        (** [write ~dst ~src n] copies the process's [n] bytes at [src] to the
            host's memory at [dst]. It may return before they land there, once
            [src] may be reused. *)
    copy : dst:nativeint -> src:nativeint -> int -> unit;
        (** [copy ~dst ~src n] copies [n] bytes within the host's memory. It may
            return before they land. *)
  }
  (** The type for how the process reaches the memory of another machine's host.
      Every call sees the bytes of the calls before it, and the work submitted
      to the machine's devices afterwards sees them too. Each raises [Failure]
      if the machine cannot be reached, which loses the host. *)

  type host_programs = {
    load :
      binary:string ->
      entry:string ->
      (nativeint * (unit -> unit), string) result;
        (** [load ~binary ~entry] loads the function [entry] of [binary], and is
            its handle and how it is unloaded, once it is unreachable, or
            [Error why] if the host refuses it. *)
    call : nativeint -> (nativeint * int) array -> int array -> unit;
        (** [call handle buffers values] runs the program with each buffer's
            address and size in bytes, with {!Program.call}'s ABI. *)
  }
  (** The type for how a host loads and calls programs. *)

  type image = {
    code : Region.t option;
        (** The device memory the binary lies in, if the driver loaded it there:
            {!Program.code} gives it to the libraries that launch its functions.
        *)
    entry : string -> (nativeint, string) result;
        (** [entry name] is the handle of the function [name]
            ({!Program.handle}), or [Error why] if the binary has none. It runs
            once per name, with the device taken. *)
    unload : unit -> unit;
        (** [unload ()] releases what the driver holds for the binary, its
            memory and the objects [entry] made, once nothing reaches the binary
            and the work the device submitted until then is done. It runs with
            the device taken. *)
  }
  (** The type for a binary as a driver loaded it. *)

  val default_timeout : int
  (** [default_timeout] is [30_000], the {!timeout} in milliseconds every device
      starts at. *)

  val host_memory : allocator
  (** [host_memory] allocates memory of this process's heap, as the {!host}
      does: regions of at least 64 KiB (four pages where pages are larger) start
      on a page. Its regions' addresses are host addresses. A device described
      with it and the [Identity] mapping shares the host's memory, such as the
      test devices ["CPU:1"], ["CPU:2"], ... of a program that runs multi-device
      work on the host. *)

  val host_programs : host_programs
  (** [host_programs] loads and calls programs on this machine's host, as
      {!Program.load} and {!Program.call} of {!host} do, for a server of this
      machine's programs to other processes. On a machine other than x86_64 and
      arm64, [load] gives [Error]. *)

  val name : ?host:device -> string -> string
  (** [name ~host local] is the name of the device of [host]'s machine (defaults
      to {!Nx_device.val-host}) described as [local]: [local] on this machine,
      and [local@ADDRESS] on the machine of the host at [ADDRESS]
      ({!Nx_device.val-name}). A vendor library names with it a device it could
      not open. *)

  val device :
    name:string ->
    arch:string ->
    budget:int ->
    ?host:device ->
    ?completion:completion ->
    ?load:(binary:string -> (image, string) result) ->
    ?peer:(device -> Region.t -> (Region.t * (unit -> unit), string) result) ->
    ?reaches:(device -> bool) ->
    ?link:(src:Buffer.t -> dst:Buffer.t -> link option) ->
    ?dma:(Region.t -> (dma, string) result) ->
    ?resolve:(nativeint -> unit) ->
    ?synchronized:(unit -> unit) ->
    ?report:(unit -> Profile.event list) ->
    ?room:(unit -> bool) ->
    ?finalize:(failed:bool -> unit) ->
    memory ->
    device
  (** [device ~name ~arch ~budget memory] is a new device whose memory [memory]
      describes, named [name] on its machine: the runtime names it [name] on
      this machine and [name@ADDRESS] on the machine of the host at [ADDRESS]
      ({!Nx_device.val-name}).
      - [host] is the host of the device's machine ({!host_of}):
        {!Nx_device.val-host} (the default), or another machine's host, which
        {!host} made. The host below is that host, and addresses are those of
        that machine.
      - [completion] is how its work completes. Defaults to [Poll].
      - [load ~binary] loads a binary of programs ({!Program.load}), or is
        [Error why] if the driver rejects it. It runs with the device taken. A
        driver that has no memory for the code raises {!Nx_device.Out_of_memory}
        with nothing changed, and the load is tried again once unreachable
        programs are collected; code that no collection could make room for is
        [Error why]. Without [load], the device loads no programs.
      - [peer d' r] maps the region [r] of the device [d'] of the same machine
        for the device: the region as the device's work addresses it, with the
        host address [r] has, if any, and how to unmap it, or [Error why] if the
        device cannot reach [d']'s memory. The runtime maps [r] once, for the
        device's borrows of it ({!Buffer.borrow}) and its copies into it, and
        unmaps it when [r]'s memory is released, once the device's work
        submitted until then is done. [peer] runs with the device taken, and the
        unmap with [d'] taken, which must not take the device. Without [peer],
        the device borrows no other device's memory.
      - [reaches d'] is [true] iff [peer] maps memory of the device [d'] of the
        same machine, as the machine's topology fixes, without trying
        ({!Nx_device.reaches}). Defaults to [false].
      - [link ~src ~dst] is how the device carries {!Buffer.copy} of [src] into
        [dst] when their devices are of two machines, if it does. Its [move]
        runs with the devices of [src], [dst] and [through] taken and those of
        [src] and [dst] synchronized.
      - [dma r] is how the other PCI functions of the device's machine reach its
        memory [r], or [Error why] ({!val-dma}). Without it, the device
        describes no memory.
      - [resolve a] runs once the work of a span that {!Submission.record}
        recorded on the device completed, before the runtime reads its stamps,
        with the host address [a] of the stamps' two slots: it writes the
        timestamps that the device's work does not write itself. It runs before
        [synchronized]. Defaults to doing nothing.
      - [synchronized ()] runs at the end of each synchronization of the device.
        Defaults to doing nothing.
      - [report ()] is the events of the runs of programs on the device that
        completed since its last report, in the order they ran, timed on the
        device's clock: the {!Profile.Counters} {!Profile.counters} asks for,
        the {!Profile.Trace}s {!Profile.traced} asks for and the spans the
        device decodes from them, and the {!Profile.Overwritten} runs it lost.
        It runs at each synchronization of the device while a profile that asks
        for counters or traces is taken, after [resolve] and before
        [synchronized], and when that profile stops. Without it, the device
        counts and traces nothing.
      - [room ()] is [true] iff each queue that submitted work writes has room
        for what one submission writes, as the device's library bounds it (its
        low-level section). {!submit} waits for it on each of its devices before
        it runs [f], for at most the device's {!timeout}, and loses the device
        if it stays [false]; a [Failure] it raises loses the device. Defaults to
        [true].
      - [finalize ~failed] runs once when the program exits, whether or not the
        device is lost: after the device synchronized if it was not, with
        [failed] telling whether it is lost by then. It leaves the hardware as
        the next open of it expects and, for a lost device, at least stops the
        device's access to the memory the process is about to release. Defaults
        to doing nothing.

      Raises [Invalid_argument] if [budget < 0], if [host] is no host, if the
      queue's clock is a [Device_clock] of no more than [0] Hz, or if
      [host_memory] gives memory the host does not address, and [Failure] if the
      device's machine has no memory for its timeline. *)

  val host :
    address:string ->
    arch:string ->
    ?programs:host_programs ->
    ?synchronized:(unit -> unit) ->
    ?finalize:(failed:bool -> unit) ->
    memory:allocator ->
    io ->
    device
  (** [host ~address ~arch ~memory io] is the host of the machine at [address],
      named ["CPU@ADDRESS"], with a budget of [max_int], as {!device} describes
      a device. [memory] allocates the machine's memory, which the host
      addresses and the process reaches only through [io]. Its timeline is
      memory of its own. With [programs], it loads and calls programs there
      ({!Program}): a call is sent in order after the machine's earlier
      operations. *)

  val buffer : device -> Region.t -> Nx_dtype.Scalar.t -> int -> Buffer.t
  (** [buffer d r s n] is a borrowed buffer of [n] elements of format [s] at the
      start of the region [r] of [d], which the vendor holds, such as a queue
      word or memory another library allocated. Nothing checks that [r] is [d]'s
      memory, and nothing frees it: its owner keeps it allocated for as long as
      the buffer and its views are reachable.

      Raises [Invalid_argument] if [d] is {!Nx_device.val-host}, whose memory
      {!Buffer.of_bigarray} borrows, or {!disk}, whose buffers are files, if
      [n < 0], or if [n] elements of [s] do not fit in [r]. *)

  val dma : Buffer.t -> (dma, string) result
  (** [dma b] is how the other PCI functions of its machine reach the region [b]
      lies in, such as a network adapter that reads and writes it, or
      [Error why] if [b]'s device does not describe its memory, or cannot for
      this memory. *)

  val depends : Buffer.t -> (unit -> unit) -> unit
  (** [depends b f] runs [f] once the memory [b] lies in is released: its
      buffers are unreachable and all work that may use it is done, that of
      [b]'s device included. An object that refers to the memory, such as
      another device's mapping of it or a command buffer that names it, is
      released in [f], which keeps what that object refers to. [f] must not
      reach [b], which would keep the memory forever. If [f] raises [Failure],
      the memory is retained instead of reused.

      [f] runs with [b]'s device taken. It must not take another device or wait
      on one; it may call another device's driver, which serialises the call
      with its own lock.

      Raises [Invalid_argument] if [b] is borrowed, empty, or on
      {!Nx_device.val-host}, whose memory the heap frees. *)
end
