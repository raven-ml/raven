(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Devices, their memory and their programs.

    A device is hardware with memory: the {!host}, or a GPU that a vendor
    library such as [nx.metal.device], [nx.cuda.device], [nx.amd.device] or
    [nx.nv.device] opens. Memory is held in {!Buffer}s, a number of elements of
    one storage format ({!Nx_dtype.Scalar.t}) on one device, and copied between
    devices by {!Buffer.copy}. The host addresses the memory of some GPUs, such
    as Metal's, and not that of others, such as CUDA's, AMD's and NV's, whose
    device copies it. A device also loads {!Program}s: the host calls its own,
    and the libraries that submit work to a GPU launch the GPU's.

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
    device, which reclaims it at the start of its next operation: a GPU keeps it
    in a cache for reuse, the host returns it to the heap, and a borrowing
    device unmaps a borrow once its work is done. Memory goes back to the system
    only once no work can still use it. If that work cannot be waited for, the
    memory is {e retained}: kept, and never freed or reused. An allocation that
    the device's {!budget} or its driver refuses first releases the cache to the
    system, then collects garbage and tries again, and raises {!Out_of_memory}
    only after that. The collector finalises a buffer in the domain that created
    it, so memory dropped by a domain that is not allocating returns when that
    domain next runs its finalisers.

    {b Hangs and faults.} {!synchronize} and {!Buffer.copy} wait for the work of
    the devices involved. A device that does not signal within its {!timeout},
    whose driver reports a fault, or whose driver errs while work is enqueued on
    its queue, is {e failed}: its state is unknown and nothing recovers it. Its
    failure is scoped to the memory it can reach: its own buffers, the host
    memory it borrowed, and memory another device owns that a copy of the failed
    device was writing when it failed:
    - The operation that finds the failure, waiting for the device's own work,
      raises [Failure "NAME hang detected"], where [NAME] is the device's name,
      or [Failure] with the driver's message.
    - Every later operation that takes the failed device raises that error at
      once. {!stats} answers, as do the functions that do not take the device:
      {!name}, {!arch}, {!budget}, {!submitted}, {!signaled} and {!timeline}.
    - {!Buffer.copy} from or to memory the failed device can reach, and
      {!Buffer.bigarray} of such memory, raise that error too. {!Buffer.view}
      does not.
    - Other devices do not wait for the failed device's work, and their other
      operations are unaffected.
    - A failed device never reclaims memory again: its buffers, and the host
      memory it borrowed, stay allocated for the life of the process, including
      borrows it was unmapping when it failed. Memory of another device that its
      copy was writing is never reused either. *)

(** {1:devices Devices} *)

type t
(** The type for devices. There is one value per device: every open of a device
    returns the value its first open made. *)

val host : t
(** [host] is the host, named ["CPU"], with a budget of [max_int]. Its buffers
    are memory of the process's heap, which it does not cache. On x86_64 and
    arm64 it loads programs, which {!Program.call} runs; elsewhere it loads
    none. *)

val name : t -> string
(** [name d] is [d]'s name: ["CPU"] for the host, ["METAL"] for the Metal GPU,
    ["CUDA"], ["CUDA:1"], ... for CUDA GPUs, ["AMD"], ["AMD:1"], ... for AMD
    GPUs, ["NV"], ["NV:1"], ... for NVIDIA GPUs opened without CUDA, and
    ["RDMA"], ["RDMA:1"], ... for RDMA network adapters. The devices of another
    machine are named so, followed by [@] and the machine as it was connected
    to, such as ["CPU@10.0.0.2:6667"] and ["AMD:1@10.0.0.2:6667"]. *)

val host_of : t -> t
(** [host_of d] is the host of the machine [d] is attached to: {!host} for the
    devices of this machine, and another machine's host for its devices. A host
    is its own. *)

val arch : t -> string
(** [arch d] is the architecture of [d]'s processor: the machine's instruction
    set for the host, such as ["arm64"] or ["x86_64"], the GPU family for Metal,
    such as ["Apple7"], the compute capability for CUDA and NV, such as
    ["sm_86"], and the graphics target for AMD, such as ["gfx1100"]. *)

val equal : t -> t -> bool
(** [equal d d'] is [true] iff [d] and [d'] are the same device. *)

val synchronize : t -> unit
(** [synchronize d] returns once the work submitted to [d], and the work
    submitted to other devices that touched [d]'s memory, has completed.

    Work of a failed device is not waited for. Raises
    [Failure "NAME hang detected"] if [d] does not signal in time, and [Failure]
    with the driver's message if its driver reports a fault; [d] is then failed.
    Raises [d]'s error at once if [d] has failed. *)

(** {1:memory Memory} *)

val budget : t -> int
(** [budget d] is the most bytes [d]'s allocator holds at once, in live buffers
    and in its cache together, of its own memory and of the host memory it
    allocates ({!Buffer.create}[ ~host:true]). Borrowed memory and the host's
    staging memory ({!Buffer.copy}) do not count. It is [max_int] for the host,
    and defaults to a device's recommended working set or memory size otherwise.
*)

val set_budget : t -> int -> unit
(** [set_budget d n] sets [d]'s budget to [n], releasing cached memory to the
    system until [d] holds at most [n] bytes or its cache is empty. Live buffers
    are never released: an allocation fails until enough of them are collected.

    Raises [Invalid_argument] if [n < 0]. *)

val timeout : t -> int
(** [timeout d] is how long, in milliseconds, a wait for [d]'s work lasts before
    [d] is considered hung and failed for good. It defaults to [30_000] unless
    [d]'s vendor library sets another. *)

val set_timeout : t -> int -> unit
(** [set_timeout d ms] sets [d]'s {!timeout} to [ms], for the waits that start
    after it, from any domain at any time. Work that takes longer, such as a
    kernel that runs longer than [ms] without the device signaling, fails [d]
    for good, and the memory it can reach stays allocated: raise the timeout
    before submitting such work.

    Raises [Invalid_argument] if [ms <= 0]. *)

val free_cache : t -> unit
(** [free_cache d] returns all of [d]'s cached memory to the system. *)

exception Out_of_memory of t * int
(** [Out_of_memory (d, n)] is raised by {!Buffer.create} when [d] cannot
    allocate [n] bytes: at once if [n] exceeds [d]'s {!budget}, and otherwise if
    the budget or the driver refuses them after [d]'s cache was released and
    unreachable buffers collected. *)

type dma = {
  bus : string;
      (** The PCI function that serves the memory, such as ["0000:03:00.0"], on
          its machine. *)
  pages : (int * int) list;
      (** The memory's bus address ranges, as (address, bytes), in order. *)
}
(** The type for how the other PCI functions of a machine reach a device's
    memory. *)

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
      over memory that something else holds ({!of_bigarray}, {!borrow}); a
      {!view} is as the buffer it views. Owned memory returns to its device once
      the buffer and all its views are unreachable. Borrowed memory is never
      cached, and never counted in a device's budget or statistics. *)

  val create : ?host:bool -> device -> Nx_dtype.Scalar.t -> int -> t
  (** [create d s n] is an owned buffer of [n] elements of format [s] on [d].
      Its contents are unspecified. A buffer of no bytes allocates nothing.

      With [~host:true] (defaults to [false]) the memory is host memory that
      [d]'s work addresses, and that the host reads and writes at
      {!host_address}; for a device of another machine, memory of that machine,
      which its host addresses. On a GPU whose own memory the host does not
      address, such as CUDA's, it is page-locked, and copies between it and
      [d]'s memory need no staging. On other devices it is [d]'s memory.

      On the {!host}, buffers of at least 64 KiB (four pages where pages are
      larger) start on a page, so that devices can {!borrow} them.

      Raises [Invalid_argument] if [n < 0] or if [n] elements of [s] take more
      than [max_int] bytes, and {!Out_of_memory} if [d] cannot allocate its
      bytes. *)

  type file = {
    path : string;  (** The absolute path the file was opened by. *)
    size : int;  (** Its size in bytes when it was mapped. *)
    mtime : float;  (** Its modification time when it was mapped. *)
    inode : int;  (** Its inode number when it was mapped, [0] where none. *)
  }
  (** The type for the identity of a mapped file. A path may name another file
      later: a reader of [path] checks that the file it opens has this size,
      modification time and inode. *)

  val of_bigarray :
    ?file:file -> ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> t
  (** [of_bigarray ?file ba] is a borrowed buffer on {!host} over [ba]'s
      elements, without a copy, of the format of [ba]'s kind: [Float16],
      [Float32], [Float64], [Int8], [UInt8] for [Int8_unsigned] and [Char],
      [Int16], [UInt16], [Int32], [Int64], [Complex64] for [Complex32] and
      [Complex128] for [Complex64]. A write through either is seen through the
      other. The buffer keeps [ba] reachable. When [ba] is over memory that
      OCaml does not manage, such as memory a C library allocated, its owner
      must keep that memory alive for as long as the buffer and its views are
      reachable.

      [file], if given, says that [ba] maps the whole of [file] from its first
      byte, as [Unix.map_file] at position [0] does: {!val-file} then answers
      for the result, its views and its borrows.

      Raises [Invalid_argument] if [ba]'s kind is [Int] or [Nativeint], which
      are no storage format, if [ba]'s first element does not lie at a multiple
      of its size (of one component for complex kinds), as a bigarray that
      [Unix.map_file] maps from an unaligned [pos] may not, or if [file] is
      given and its [size] is not [ba]'s size in bytes. *)

  val file : t -> (file * int) option
  (** [file b] is the file [b]'s memory maps and the offset in it of [b]'s first
      byte, if [b] is a buffer of {!of_bigarray} given a [file], or a view or a
      borrow of one. It is [None] for any other buffer. A copy of [b]'s bytes
      elsewhere, such as to a device, can read them from the file with ordinary
      reads instead of faulting them in through the mapping. *)

  val borrow : device -> t -> t
  (** [borrow d b] is a borrowed buffer on [d] over the memory of the buffer [b]
      of [d]'s host ({!host_of}), without a copy, of [b]'s format and length. A
      write through either is seen through the other, once the devices involved
      are synchronized. The result keeps [b] reachable. [borrow host b] is [b].
      Work that reads or writes through the result touches the host's memory:
      its {!submit} lists {!host} in [touches].

      [d] maps the whole host memory that [b] is a view of, once: the borrows on
      [d] of views of that memory share one mapping, which [d] releases once
      they are all unreachable. A mapping covers whole pages, so that memory
      must start on a page: host buffers of at least 64 KiB (four pages where
      pages are larger) and memory-mapped files do. Smaller host buffers cannot
      be borrowed; {!copy} moves them through staging memory. On CUDA, mapping
      page-locks the memory, which must be writable: memory mapped read-only
      cannot be borrowed there.

      Raises [Invalid_argument] if [b] is not on [d]'s host, if [d] cannot
      address the host's memory, if the memory [b] is a view of does not start
      on a page, or if [d]'s driver refuses to map it, with the driver's reason.
  *)

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
      element of [s]. *)

  val copy : src:t -> dst:t -> unit
  (** [copy ~src ~dst] copies [src]'s bytes into [dst] and returns once they are
      there. It first synchronizes the devices of [src] and [dst]. A copy
      between two devices counts in [src]'s [bytes_out] and in [dst]'s
      [bytes_in].

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
      that fails fails its devices, which may still write [dst].

      Raises [Invalid_argument] if [src] and [dst] have different sizes in
      bytes, if they overlap in the memory of one buffer, or if the host does
      not address the memory of a device that has no copy queue;
      [Failure "NAME hang detected"] if a device involved does not signal in
      time; [Failure] with the driver's message if a device's driver reports a
      fault or errs while the copy is enqueued, which fails that device;
      [Failure] with a failed device's error if that device can reach [src] or
      [dst]; [Failure] with the error of another machine's host that the network
      failed, which fails that host; and [Failure] if a device cannot map the
      staging memory, or [Stdlib.Out_of_memory] if the host cannot allocate it,
      which fail no device. *)

  val bigarray :
    ('a, 'b) Bigarray.kind -> t -> ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
  (** [bigarray k b] is the bytes of the host buffer [b] read as elements of
      kind [k], without a copy: [nbytes b / Bigarray.kind_size_in_bytes k] of
      them, in the machine's byte order. Writing through it mutates [b]. It
      keeps [b]'s memory alive for as long as it is reachable, except memory
      that OCaml does not manage, which {!of_bigarray}'s caller keeps alive.

      Access through the view is outside the devices' ordering: {!synchronize}
      {!host} first to see the work that touched it.

      Formats with no kind of their own are read as their storage kind and
      decoded with {!Nx_dtype.Scalar.decode}: [BFloat16] as [Int16_unsigned],
      the float8 formats as [Int8_unsigned], [Int4] and [UInt4] as
      [Int8_unsigned] holding two per byte. With an odd number of 4-bit
      elements, the last byte's high nibble is not [b]'s, and a write to it may
      change memory outside [b].

      Raises [Invalid_argument] if [b] is not on {!host}, if [k] is [Int] or
      [Nativeint], which are no storage format, or if [b]'s bytes are not a
      whole number of elements of [k], aligned to the size of one element (of
      one component for complex kinds), and [Failure] with a failed device's
      error if that device can reach [b]. *)

  (** {2:low Low-level}

      For the libraries that submit work over buffers. C code reads the host
      address of a buffer value with [nx_device_buffer_host] from the header
      [nx_device.h], which [nx.device] installs: it reads the value's fields
      without allocating, and the address stays valid while the value is
      reachable. Its buffer must be one the host addresses, one that
      {!host_address} answers for, such as every buffer on {!host}; reading any
      other is undefined behaviour. *)

  val address : t -> nativeint
  (** [address b] is the address of [b]'s first byte in its device's address
      space, as the device's work reads it. *)

  val host_address : t -> nativeint
  (** [host_address b] is the address of [b]'s first byte in the host's address
      space. Reading or writing it is outside the device's ordering: synchronize
      first.

      Raises [Invalid_argument] if the host does not address the memory of a
      nonempty [b], such as memory that {!create} allocated on a CUDA device
      without [~host:true], or memory of another machine. *)

  val handle : t -> nativeint
  (** [handle b] is the driver's object for the memory [b] lies in, such as a
      [MTLBuffer] or the start of a CUDA allocation, and [0n] on the host. [b]
      starts {!offset} bytes into it. *)

  val offset : t -> int
  (** [offset b] is the byte offset of [b]'s first byte in {!handle}[ b]. *)

  val dma : t -> dma
  (** [dma b] is how the other PCI functions of its machine reach the memory [b]
      lies in, from {!address}[ b - ]{!offset}[ b] on, such as a network adapter
      that reads and writes it.

      Raises [Invalid_argument] if [b]'s device does not describe its memory, or
      cannot for this memory, with the reason. *)

  val on_free : t -> (unit -> unit) -> unit
  (** [on_free b f] runs [f] once [b]'s device frees the memory [b] lies in to
      its driver, after the device synchronized, and before: another device
      mapped that memory, and [f] unmaps it. Memory kept in the device's cache
      stays mapped. If [f] raises [Failure], the memory is retained instead of
      freed. [f] runs with [b]'s device taken, and must not use it through this
      module.

      Raises [Invalid_argument] if [b] is borrowed, empty, or on {!host}, whose
      memory the heap frees. *)
end

(** {1:programs Programs} *)

(** Programs loaded on a device. *)
module Program : sig
  type device := t

  type t
  (** The type for programs: a function of a binary, loaded on a device. A
      program of the {!host} is run by {!call}; a GPU's are launched by the
      libraries that submit work to it. *)

  val load : device -> binary:string -> name:string -> t
  (** [load d ~binary ~name] is the function [name] of [binary], a compiled
      library in [d]'s format: a metallib for Metal, a CUDA module (cubin,
      fatbin, or PTX, which the driver compiles) for CUDA, a code object for
      AMD, a cubin for NV. Loading the same binary and name on [d] again returns
      the same program.

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
      memory that is executable and never writable. That memory is freed once
      the program is unreachable, so loading the same binary and name again
      returns the same program only while it is reachable.

      Raises [Invalid_argument] if [d] loads no programs, and [Failure] with the
      driver's message if it rejects [binary] or has no function [name]; on the
      host, with the reason it cannot load [binary], such as a symbol that no
      library defines. *)

  val device : t -> device
  (** [device p] is the device [p] is loaded on. *)

  val name : t -> string
  (** [name p] is the name of [p]'s function. *)

  val handle : t -> nativeint
  (** [handle p] is the driver's object for [p], such as a
      [MTLComputePipelineState], a [CUfunction], the address of an AMD kernel
      descriptor, or the address of an NV function's first instruction. On the
      host, it is the address of the function's first instruction. *)

  val call : t -> Buffer.t array -> int array -> unit
  (** [call p buffers values] runs the host program [p] in the calling domain
      and returns once it returns. [p] is called as the C function

      {v void f(void **buffers, const int64_t *values); v}

      given the host address of each buffer's first byte
      ({!Buffer.host_address}) and each value as a 64-bit integer, in order. It
      follows the platform's C calling convention: on Windows, an object
      compiled for x86_64 ELF declares [__attribute__((ms_abi))] on its entry
      and on the library functions it calls, and code for arm64 leaves the
      register [x18] alone ([-ffixed-x18]), which macOS and Windows reserve.

      The OCaml runtime is released while [p] runs, so other threads and domains
      go on; the buffers stay reachable until it returns. The call is outside
      the devices' ordering, as an access at {!Buffer.host_address} is: it does
      not take the host, and several domains may run programs at once.
      Synchronize the devices whose work touches [buffers] first. While a
      {!Profile} is taken, the call is a span of the calling domain's lane of
      the host, named after [p].

      A program of another machine's host runs there, on memory that host
      addresses, with the same ABI. The call is sent in order after the earlier
      operations on that machine and before the later ones, and returns once
      sent: the program has run when the next operation that waits for an answer
      from the machine, such as {!synchronize} of its host, returns. A program
      that fails there fails the host.

      Raises [Invalid_argument] if [p]'s device runs no programs or does not
      address the memory of a buffer of [buffers], and [Failure] with a failed
      device's error if that device can reach a buffer of [buffers]. *)
end

(** {1:stats Statistics} *)

(** Device statistics. *)
module Stats : sig
  type t
  (** The type for a snapshot of a device's statistics. *)

  val allocated : t -> int
  (** [allocated s] is the bytes of owned memory in buffers that
      {!Buffer.create} returned and that were not yet returned to the device. *)

  val cached : t -> int
  (** [cached s] is the bytes in the device's cache: memory allocated from the
      driver and held for reuse. *)

  val retained : t -> int
  (** [retained s] is the bytes of the device's own memory that it retains
      because the work that last used them could not be waited for. They count
      against its budget. Retained borrows are not counted. *)

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
    before the call counts as returned, unless [d] has failed. It answers on a
    failed device, whose retained bytes it reports. *)

(** {1:submitting Submitting work}

    For the libraries that submit work to a device. Work is submitted inside
    {!submit}, which orders it after the device's earlier work and gives it the
    value it must signal on completion. *)

val submit : t -> touches:t list -> (int -> 'a) -> 'a
(** [submit d ~touches f] is [f v], run with [d] and the devices of [touches]
    taken, where [v] is the next value of [d]'s timeline. [f] submits work to
    [d] that signals [v] when it completes, and whose memory accesses reach [d]
    and the devices of [touches]. Work through a buffer borrowed from the host
    ({!Buffer.borrow}) touches the host: list {!host} in [touches]. Once [f]
    returns, [v] is [d]'s submitted value, and {!synchronize} on each device of
    [touches] waits for [v] on [d].

    If [f] raises, nothing is recorded, so [f] may raise only before it commits
    any work: committed work that signals [v] would signal a value the next
    submission takes again.

    [f] must not use [d] or the devices of [touches] through this module, which
    they are taken by: it submits through the handles its vendor library gives.
*)

val submitted : t -> int
(** [submitted d] is the value [d]'s last submitted work signals, [0] before any
    submission. *)

val signaled : t -> int
(** [signaled d] is the last value [d] signaled. Work that signals
    [v <= signaled d] has completed. *)

val timeline : t -> Buffer.t
(** [timeline d] is a buffer of two [UInt64] that [d]'s host and [d]'s work
    address: a signal word, then [d]'s submitted value. It is [d]'s host memory
    when [d] allocates some, such as page-locked memory on CUDA, and memory of
    its host ({!host_of}) otherwise. Work signals by storing its value into the
    signal word, which {!signaled} reads, unless the device signals in its own
    way: Metal's shared event reports through {!signaled} alone and leaves the
    signal word at [0]. *)

(** {1:profiling Profiling} *)

(** Profiles of devices' work.

    While a profile is taken, between {!start} and {!stop}, the devices record
    {!event}s: {e spans} of work on a device, changes of its allocated memory,
    and the programs it loads. Spans come from the host ({!span}), from the
    runtime's own copies and calls of host programs, and from the libraries that
    submit work ({!record}). Every time is on the host's clock, {!now}: the
    times a device stamps on its own clock are calibrated against it when the
    profile is taken. {!output} writes a profile in Chrome's trace event format,
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
    | Memory of {
        device : device;
        time : int;
        allocated : int;
            (** The device's {!Stats.allocated} bytes from [time] on. *)
      }  (** A change of the memory a device allocated. *)
    | Program of {
        program : Program.t;
        binary : string;  (** The binary it was loaded from. *)
        time : int;  (** When it was loaded. *)
      }  (** A program loaded on its device. *)

  val now : unit -> int
  (** [now ()] is the host clock: nanoseconds of the system's monotonic clock,
      which starts at an unspecified point. C code reads it with
      [nx_device_now_ns] from the header [nx_device.h]. On macOS it is the clock
      of Metal's command buffer times. *)

  val start : unit -> unit
  (** [start ()] starts taking a profile of every device.

      Raises [Invalid_argument] if a profile is being taken. *)

  val stop : unit -> event list
  (** [stop ()] stops taking the profile {!start} started, and is its events, in
      time order and, at equal times, longest first. It first synchronizes the
      devices whose recorded spans are still to be read, and calibrates the
      clocks of the devices that stamp times on their own. The unread spans of a
      device that fails meanwhile are left out; its next operation raises its
      error.

      Raises [Invalid_argument] if no profile is being taken. *)

  val enabled : unit -> bool
  (** [enabled ()] is [true] iff a profile is being taken. The libraries that
      submit work read it to decide whether to stamp their work at all. *)

  val span : string -> (unit -> 'a) -> 'a
  (** [span name f] is [f ()]. While a profile is taken, it records a span named
      [name] on the lane of the calling domain of the {!host}, from the call
      until [f] returns or raises. *)

  val record : device -> lane:string -> name:string -> Buffer.t -> unit
  (** [record d ~lane ~name stamps] records a span of work named [name] on
      [lane] of [d], whose start and stop timestamps the work writes into the
      two elements of [stamps], on [d]'s clock. Call it once the {!submit} of
      that work returned. The stamps are read at [d]'s next synchronization,
      which waits for the work: they must stay the work's until then, and a
      later record of the same stamps before then replaces this one. It does
      nothing unless a profile is being taken.

      Raises [Invalid_argument] if [stamps] is not two [UInt64] that [d]'s host
      addresses. *)

  val output : out_channel -> event list -> unit
  (** [output oc events] writes [events] to [oc] in Chrome's trace event format,
      JSON: a process for each device, named after it, with a thread for each of
      its lanes; a complete event for each span, a counter [memory] for each
      change of memory, and an instant event for each program load, with the
      program's handle. Times are microseconds from the earliest event.
      Malformed UTF-8 in names becomes U+FFFD. [oc] is neither flushed nor
      closed. *)
end

(** {1:vendors Vendor runtimes}

    A library that opens devices of some kind makes each of them with {!make},
    once, and returns that value from every later open. A library that reaches
    another machine makes that machine's host with {!make_host}, and the other
    devices of that machine with {!make} given that host. *)

type memory = {
  host : nativeint option;
      (** The memory's first byte, as the device's host addresses it, if it
          does: in that host's address space, another machine's for a device of
          another machine. *)
  device : nativeint;  (** Its first byte, as the device's work addresses it. *)
  handle : nativeint;  (** The driver's object for it. *)
}
(** The type for memory that a driver allocated or mapped. *)

type allocator = {
  alloc : int -> memory option;
      (** [alloc n] is [n > 0] bytes of new memory, or [None] if the driver has
          none. *)
  free : memory -> unit;  (** [free m] returns [m] to the driver. *)
}
(** The type for allocators of a device's memory. *)

type mapping = {
  map : nativeint -> int -> (memory, string) result;
      (** [map a n] maps the [n] bytes of host memory at [a] for the device:
          memory whose [host] is at or below [a], or [Error why] if the driver
          refuses, saying why. *)
  unmap : memory -> unit;  (** [unmap m] releases a mapping [map] made. *)
}
(** The type for the mappings of host memory into a device's address space. *)

type copy = dst:nativeint -> src:nativeint -> int -> int -> unit
(** The type for enqueueing copies. [copy ~dst ~src n v] enqueues on the
    device's copy queue a copy of [n > 0] bytes from [src] to [dst], after the
    device's earlier work, and then the signal of [v] once the copy is complete.
    It returns without waiting for either. It raises [Failure] with the driver's
    message if the driver errs, which fails the device. *)

type copy_queue = {
  copy : copy;
      (** Copies between the device's addresses: [device] addresses of the
          memory it allocated or mapped. *)
  transfer : t -> copy option;
      (** [transfer d'] copies from the device's memory to the memory of the
          device [d'], [None] if the device cannot. *)
  stamp : slot:nativeint -> int -> unit;
      (** [stamp ~slot v] enqueues on the copy queue, after the device's earlier
          work, the write of a timestamp of the device's clock into the second
          [UInt64] of the 16 bytes at the device address [slot], of which it may
          write the first too, and then the signal of [v]. It returns without
          waiting and raises like [copy]. The runtime stamps its copies while a
          profile is taken, and calibrates a device's own clock with it. *)
}
(** The type for a device's copy queue. *)

(** The type for the clocks of a device's timestamps. *)
type clock =
  | Host_clock
      (** The device's timestamps are readings of the host clock,
          {!Profile.now}, such as the command buffer times of Metal or the
          stamps of host functions that a device's queue runs. *)
  | Device_clock of { hz : int }
      (** The device's timestamps count ticks of a clock of its own, [hz] per
          second. When a profile is taken, the runtime calibrates it against the
          host clock with its copy queue's [stamp]. *)

type signal = {
  signaled : unit -> int;  (** The last value the device signaled. *)
  wait : int -> timeout_ms:int -> bool;
      (** [wait v ~timeout_ms] waits until the device signaled [v], for at most
          [timeout_ms] milliseconds; [false] if it did not. It raises [Failure]
          with the driver's message if the driver reports a fault. *)
}
(** The type for how a device signals completion and is waited for. *)

type link = {
  through : t list;  (** The devices that carry the copy. *)
  move : src:Buffer.t -> dst:Buffer.t -> unit;
      (** [move ~src ~dst] copies [src]'s bytes into [dst], of the same size,
          and returns once they are there. It raises [Failure] if a device of
          [through] fails, which fails them all. *)
}
(** The type for links, which carry copies between the memory of devices of two
    machines. *)

val make :
  name:string ->
  arch:string ->
  budget:int ->
  memory:allocator ->
  ?host:t ->
  ?host_memory:allocator ->
  ?mapping:mapping ->
  ?copy_queue:(memory -> copy_queue) ->
  ?load:(binary:string -> name:string -> nativeint) ->
  ?link:(src:Buffer.t -> dst:Buffer.t -> link option) ->
  ?dma:(memory -> (dma, string) result) ->
  ?signal:(memory -> signal) ->
  ?sleep:(int -> unit) ->
  ?timeout_ms:int ->
  ?synchronized:(unit -> unit) ->
  ?finalize:(failed:bool -> unit) ->
  ?clock:clock ->
  ?resolve:(nativeint -> unit) ->
  unit ->
  t
(** [make ~name ~arch ~budget ~memory ?host ?host_memory ?mapping ?copy_queue
     ?load ?link ?dma ?signal ?sleep ?timeout_ms ?synchronized ?finalize ?clock
     ?resolve ()] is a new device:
    - [host] is the host of the device's machine ({!host_of}): {!host} (the
      default), or another machine's host, which {!make_host} made. The host
      below is that host, and addresses are those of that machine.
    - [memory] allocates the device's own memory, and [host_memory] the host
      memory that its work addresses, for {!Buffer.create}[ ~host:true]: memory
      the host addresses. Without [host_memory], [memory] serves both, and the
      host must address its memory. Memory the device frees is cached for reuse
      first, and the device synchronizes before it frees it to the driver, so no
      work still uses it.
    - [mapping] maps host memory for {!Buffer.borrow}, and unmaps it once the
      device synchronized. It is given memory that starts on a page. Without it,
      the device cannot borrow.
    - [copy_queue m] copies the memory that the host does not address, given
      [m], the memory of its {!timeline}; other host memory is staged through
      the host's staging memory, which it maps with [mapping]. Without it, the
      host must address all of the device's memory, and copies are host memory
      copies.
    - [load ~binary ~name] loads a program, raising [Failure] if the driver
      rejects it. The device keeps the programs it loads for its life. Without
      [load], it loads no programs.
    - [link ~src ~dst] is how the device carries {!Buffer.copy} of [src] into
      [dst] when their devices are of two machines, if it does. It is asked
      without any device taken; its [move] runs with the devices of [src], [dst]
      and [through] taken and those of [src] and [dst] synchronized.
    - [dma m] is how the other PCI functions of the device's machine reach its
      memory [m], or [Error why] ({!Buffer.dma}). It runs without the device
      taken. Without it, the device describes no memory.
    - [signal m] is how the device signals completion and is waited for, given
      [m], the memory of its {!timeline}. Without it, work signals by storing
      into the timeline's signal word, and waits poll it.
    - [sleep ms], without [signal], runs once a wait has seen the signal word
      stay still for 200 milliseconds, and again each time it returns while the
      word stays still: it blocks for at most [ms] milliseconds, at most 200 and
      never past the timeout, on the device's interrupts or events, and raises
      [Failure] with the driver's message if the device reports a fault, which
      fails the device. Before a wait declares the device hung, it runs once
      more with [ms = 1], so a fault reported late still names its cause.
      Without it, waits only poll.
    - [timeout_ms] is the device's initial {!timeout}. Defaults to [30_000].
      Without [signal], the timeout restarts whenever the signal word moves.
    - [synchronized ()] runs at the end of each synchronization of the device,
      and raises [Failure] with the driver's message if the device reports a
      fault, which fails the device. Defaults to doing nothing.
    - [finalize ~failed] runs once when the program exits, whether or not the
      device has failed: after the device synchronized if it had not, with
      [failed] telling whether it has failed by then. It leaves the hardware as
      the next open of it expects and, for a failed device, at least stops the
      device's access to the memory the process is about to release. An
      exception it raises is printed and ignored. Defaults to doing nothing.
    - [clock] is the clock of the device's timestamps. Defaults to
      {!Host_clock}.
    - [resolve a] runs once the work of a span that {!Profile.record} recorded
      on the device completed, before the runtime reads its stamps, with the
      host address [a] of the stamps: it writes the timestamps that the device's
      work does not write itself. It runs before [synchronized]. Defaults to
      doing nothing.

    These functions run while the device is taken, and must not use it through
    this module. Blocking driver calls should release the OCaml runtime.

    Raises [Invalid_argument] if [budget < 0], if [timeout_ms <= 0], if [host]
    is no host, if [copy_queue] is given without [mapping], if [sleep] is given
    with [signal], if [clock] is a {!Device_clock} of no more than [0] Hz or
    without [copy_queue], or if [host_memory] gives memory the host does not
    address, and [Failure] if [host_memory] or the host has no memory for the
    timeline. *)

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
    Every call sees the bytes of the calls before it, and the work submitted to
    the machine's devices afterwards sees them too. Each raises [Failure] if the
    machine cannot be reached, which fails the host. *)

val make_host :
  name:string ->
  arch:string ->
  budget:int ->
  memory:allocator ->
  io:io ->
  ?load:(binary:string -> name:string -> nativeint * (unit -> unit)) ->
  ?call:(nativeint -> (nativeint * int) array -> int array -> unit) ->
  ?timeout_ms:int ->
  ?synchronized:(unit -> unit) ->
  ?finalize:(failed:bool -> unit) ->
  unit ->
  t
(** [make_host ~name ~arch ~budget ~memory ~io ?load ?call ?timeout_ms
     ?synchronized ?finalize ()] is the host of another machine, as {!make}
    makes a device:
    - [memory] allocates the machine's memory, which the host addresses and the
      process reaches only through [io]. Its timeline is memory of its own.
    - [load ~binary ~name] loads a program there, and is its handle and how it
      is unloaded, once it is unreachable. [call handle buffers values] sends a
      call of the program ({!Program.call}), with each buffer's address and size
      in bytes, in order after the machine's earlier operations. Without them,
      it runs no programs.

    Raises [Invalid_argument] as {!make} does, and if only one of [load] and
    [call] is given. *)

val external_buffer : t -> memory -> Nx_dtype.Scalar.t -> int -> Buffer.t
(** [external_buffer d m s n] is a borrowed buffer of [n] elements of format [s]
    over the memory [m] of [d], which something outside this module allocated.
    Nothing frees [m]: its owner keeps it allocated for as long as the buffer
    and its views are reachable. Vendor libraries build their constructors of
    such buffers on it, and check that [m] is [d]'s memory first.

    Raises [Invalid_argument] if [d] is {!host}, whose memory
    {!Buffer.of_bigarray} borrows, if [n < 0], or if [n] elements of [s] take
    more than [max_int] bytes. *)
