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
    device owns that a copy of [d] was writing when [d] was lost.
    - The operation that finds the loss raises [Lost (d, why)], and so does,
      with the same [why], every later operation that takes [d], and every
      {!Buffer.copy}, {!Buffer.bigarray} and {!Program.call} that reaches memory
      [d] can reach. {!Buffer.view} does not. {!stats} answers, as do the
      functions that do not take [d]: {!name}, {!arch}, {!budget}, {!submitted},
      {!signaled} and {!timeline}.
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
      staging memory the copy goes through. *)

(** {1:memory Memory} *)

val budget : t -> int
(** [budget d] is the most bytes [d]'s allocator holds at once, in live buffers
    and in its cache together, of its own memory and of its pinned memory
    ({!Buffer.create}[ ~pinned:true]). Borrowed memory and the host's staging
    memory ({!Buffer.copy}) do not count. It is [max_int] for the host, and
    defaults to a device's recommended working set or memory size otherwise. *)

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

  val create : ?pinned:bool -> device -> Nx_dtype.Scalar.t -> int -> t
  (** [create d s n] is an owned buffer of [n] elements of format [s] on [d].
      Its contents are unspecified. A buffer of no bytes allocates nothing.

      With [~pinned:true] (defaults to [false]) the memory is [d]'s pinned
      memory, which both [d]'s work and [d]'s host address: page-locked host
      memory on CUDA, AMD and NV, and [d]'s own memory on the devices whose
      memory the host addresses. It is coherent: a write by either side is seen
      by the other once the work that wrote it has completed, with no flush. The
      libraries that submit work allocate their command buffers, queue words and
      volatile arguments this way. It counts in [d]'s budget, [d] caches it, and
      copies between it and [d]'s memory need no staging.

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

      The buffer holds the file open, and its views share it: they read the file
      that was opened, even once [path] is renamed, removed or replaced. The
      file is closed once the buffer and all its views are unreachable, at the
      disk's next operation, such as another {!of_file}, and its mapping once
      its borrows ({!borrow}) are unreachable too. A file changed in place while
      it is open changes what they read, and a read past a new end of the file
      raises; a borrow of its pages asks more of the file.

      [Error why] naming [path] if it cannot be opened for reading or is not a
      regular file. *)

  val create_file : string -> int -> (t, string) result
  (** [create_file path n] is the file at [path], created, or emptied if it
      exists, and sized to [n] bytes, as a buffer of [n] [UInt8] elements on
      {!disk} for reading and writing: {!copy} reads and writes them. Its bytes
      read as zero until they are written. It holds the file open as {!of_file}
      does. A write reaches the file when {!copy} returns, and the storage once
      the system flushes the file, which a sync of the file forces.

      [Error why] naming [path] if it cannot be created.

      Raises [Invalid_argument] if [n < 0]. *)

  val borrow : device -> t -> (t, string) result
  (** [borrow d b] is a borrowed buffer on [d] over the memory of the buffer [b]
      of [d]'s machine, without a copy, of [b]'s format and length. A write
      through either is seen through the other, once the devices involved are
      synchronized. The result keeps [b] reachable. [borrow d b] is [Ok b] for
      [b] on [d], so [borrow host b] is [Ok b] for [b] on the host. Work that
      reads or writes through the result touches the memory's device: its
      {!submit} lists it in [touches].

      A borrow of a borrow maps the memory the first one maps: [borrow d b] for
      [b] a borrow on another device of host memory is [d]'s borrow of that host
      memory.

      [d] maps [b]'s memory by where it lives:
      - a file's bytes on the {!disk}, through the file's pages, below;
      - system memory, which [d]'s mapping of host memory maps: memory of [d]'s
        host ({!host_of}), of a device described as [Host_visible]
        ({!Driver.memory}), such as Metal and test devices over the host's
        memory, and the pinned memory of any device ({!create});
      - the own memory of a [Device_local] device, such as a CUDA, AMD or NV
        GPU's, which [d]'s driver maps ({!Driver.device}'s [peer]) once per
        region, even where a memory BAR gives it a host address. The mapping
        lasts until that device frees the memory, and [d] can reach it until
        then.

      [d] maps the whole host memory that [b] is a view of, once: the borrows on
      [d] of views of that memory share one mapping, which [d] releases once
      they are all unreachable. A mapping covers whole pages, so that memory
      must start on a page. Host buffers that {!create} makes start on one from
      64 KiB (four pages where pages are larger), and smaller ones cannot be
      borrowed, wherever they start; {!copy} moves them through staging memory.
      Memory-mapped files start on a page. Memory that {!of_bigarray} wraps
      borrows if it starts on one. A buffer of no bytes always borrows, mapping
      nothing. On CUDA, mapping page-locks the memory, which must be writable:
      memory mapped read-only cannot be borrowed there.

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

      [Error why] if [d] cannot map that kind of memory (a host maps no other
      device's), if [b] is a host buffer {!create} made of fewer than 64 KiB, if
      the memory [b] is a view of does not start on a page, or if [d]'s driver
      refuses to map it, with the driver's reason; for [b] on the disk, if [d]
      does not share the host's memory, if [b]'s first byte is not aligned to
      the size of one of its elements, or if the system cannot map the file,
      naming it.

      Raises [Invalid_argument] if [b] is on another machine or is dead
      ({!consume}), and {!Lost} if [d] is lost. *)

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
  (** [spans b] is [true] iff [b]'s bytes are all of the memory it lies in, as
      those of a buffer {!create} or {!of_bigarray} made are, and not those of a
      {!view} of part of it: [offset b = 0] and [nbytes b] is the size of that
      memory ({!Driver.Region.nbytes}). *)

  val overlaps : t -> t -> bool
  (** [overlaps b b'] is [true] iff [b] and [b'] share a byte of memory: through
      views of one memory, through a borrow and the memory it maps, or through
      two bigarrays over the same bytes. Buffers of no bytes overlap nothing, a
      file's bytes do not overlap the borrows of its pages, which are
      copy-on-write, and two opens of one file ({!of_file}, {!create_file}) are
      two memories. *)

  val consume : why:string -> t -> t
  (** [consume ~why b] is a buffer over [b]'s memory, and kills every other
      buffer over that memory made before: [b] and its views. Reaching a dead
      buffer's bytes ({!address}, {!bigarray}, {!copy}, {!borrow},
      {!Program.call}, or a kernel reading it) raises [Invalid_argument why].
      The result, and the views made of it, are live; the memory stays owned or
      borrowed as [b]'s was.

      A library that takes over memory it was handed, such as a compiled call
      that writes its result over an argument, consumes the argument's buffer,
      so that no earlier handle observes the new contents.

      Raises [Invalid_argument] if [b] is dead or does not {!spans} its memory:
      consuming a window of it would kill the rest. *)

  val copy : src:t -> dst:t -> unit
  (** [copy ~src ~dst] copies [src]'s bytes into [dst] and returns once they are
      there. It first synchronizes the devices of [src] and [dst]. A copy
      between the memory of two devices counts in the [bytes_out] of the device
      whose memory [src] is and in the [bytes_in] of [dst]'s; a borrow's memory
      is its host's.

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

      [Error why] if [d] loads no programs, or if its driver rejects [binary] or
      finds no function [name] in it, with the driver's reason (on the host, the
      reason it cannot load [binary], such as a symbol that no library defines).
      [why] starts with [d]'s {!name}. [d] stays usable.

      Raises {!Lost} if [d] is lost, or is lost by the load. *)

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

      given the host address of each buffer's first byte ({!Buffer.address} on
      the host) and each value as a 64-bit integer, in order. It follows the
      platform's C calling convention: on Windows, an object compiled for x86_64
      ELF declares [__attribute__((ms_abi))] on its entry and on the library
      functions it calls, and code for arm64 leaves the register [x18] alone
      ([-ffixed-x18]), which macOS and Windows reserve.

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

      Only hosts run programs; a program of another device is launched by the
      libraries that submit work to it. [buffers] may be any buffers whose
      memory [p]'s host addresses: its own, and that of the devices of its
      machine whose memory it addresses, such as Metal's, CUDA's pinned memory
      and the test devices of {!Driver.host_memory}.

      Raises [Invalid_argument] if [p]'s device runs no programs or does not
      address the memory of a buffer of [buffers], and {!Lost} if a lost device
      can reach a buffer of [buffers]. *)
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
    before the call counts as returned, unless [d] is lost. It answers on a lost
    device, whose retained bytes it reports. *)

(** {1:profiling Profiling} *)

(** Profiles of devices' work.

    One profile of every device is taken at a time, between {!start} and
    {!stop}. While it is taken, the devices record {!event}s: {e spans} of work
    on a device, changes of its allocated memory, and the programs it loads.
    Spans come from the host ({!span}), from the runtime's own copies and calls
    of host programs, and from the libraries that submit work ({!record}). Every
    time is on the host's clock, {!now}: the times a device stamps on its own
    clock are calibrated against it when the profile is stopped.
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

  type t
  (** The type for profiles being taken. *)

  val start : unit -> t
  (** [start ()] starts taking a profile of every device, which only its holder
      stops.

      Raises [Invalid_argument] if a profile is being taken. *)

  val stop : t -> event list
  (** [stop p] stops taking [p], and is its events, in time order and, at equal
      times, longest first. It first synchronizes the devices whose recorded
      spans are still to be read, and calibrates the clocks of the devices that
      stamp times on their own. The unread spans of a device lost meanwhile are
      left out; its next operation raises {!Lost}.

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

  val output_chrome_trace : out_channel -> event list -> unit
  (** [output_chrome_trace oc events] writes [events] to [oc] in Chrome's trace
      event format, JSON: a process for each device, named after it, with a
      thread for each of its lanes; a complete event for each span, a counter
      [memory] for each change of memory, and an instant event for each program
      load, with the program's handle. Times are microseconds from the earliest
      event. Malformed UTF-8 in names becomes U+FFFD. [oc] is neither flushed
      nor closed. *)
end

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
    address: a signal word, then [d]'s submitted value. It is [d]'s pinned
    memory when [d]'s memory is [Device_local] ({!Driver.memory}), such as
    page-locked memory on CUDA, and memory of its host ({!host_of}) otherwise.
    Work signals by storing its value into the signal word, which {!signaled}
    reads, unless the device signals in its own way: Metal's shared event
    reports through {!signaled} alone and leaves the signal word at [0]. *)

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

    A [finalize] that raises is printed and ignored at exit. An {!on_free} hook
    that raises [Failure] retains its memory. Blocking driver calls should
    release the OCaml runtime. *)
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

  type mapping = {
    map : nativeint -> int -> (Region.t, string) result;
        (** [map a n] maps the [n] bytes of host memory at [a], which start on a
            page, for the device: a region whose host address is at or below
            [a], or [Error why] if the driver refuses. *)
    unmap : Region.t -> unit;
        (** [unmap r] releases a mapping [map] made, once the device
            synchronized. *)
  }
  (** The type for the mappings of host memory into a device's address space. *)

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
            device [d'], [None] if the device cannot. *)
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
            [memory] serves {!Buffer.create}, with and without [~pinned]. With
            [mapping], the device borrows host memory ({!Buffer.borrow}); the
            device then shares the host's memory ({!shares_host_memory}). *)
    | Device_local of {
        memory : allocator;
        host_memory : allocator;
        mapping : mapping;
        queue : timeline:Region.t -> queue;
      }
        (** The device's queue copies its own memory, staging through host
            memory it maps.
            - [memory] allocates its own memory, which the host may not address.
            - [host_memory] allocates its pinned memory
              ({!Buffer.create}[ ~pinned:true]): coherent host memory that its
              work addresses, whose regions have a host address.
            - [mapping] maps host memory for {!Buffer.borrow} and for the host's
              staging memory.
            - [queue ~timeline] is its copy queue, given the region of its
              {!timeline}, which the runtime allocates first from [host_memory].
        *)

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
        (** Work signals by storing its value into the {!timeline}'s signal
            word, and waits poll it. The timeout restarts whenever the word
            moves. *)
    | Sleep of (int -> unit)
        (** As [Poll], and [sleep ms] runs once a wait has seen the signal word
            stay still for 200 milliseconds, and again each time it returns
            while the word stays still: it blocks for at most [ms] milliseconds,
            at most 200 and never past the timeout, on the device's interrupts
            or events, and raises [Failure] with the driver's message if the
            device reports a fault, which loses the device. Before a wait
            declares the device hung, it runs once more with [ms = 1], so a
            fault reported late still names its cause. *)
    | Signal of (timeline:Region.t -> signal)
        (** The device signals in its own way, given the region of its
            {!timeline}. *)

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

  val default_timeout : int
  (** [default_timeout] is [30_000], the {!timeout} in milliseconds every device
      starts at. *)

  val host_memory : allocator
  (** [host_memory] allocates memory of this process's heap, as the {!host}
      does: regions of at least 64 KiB (four pages where pages are larger) start
      on a page. Its regions' addresses are host addresses. A device described
      with it, and a mapping that is the identity, shares the host's memory,
      such as the test devices ["CPU:1"], ["CPU:2"], ... of a program that runs
      multi-device work on the host. *)

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
    ?load:(binary:string -> entry:string -> (nativeint, string) result) ->
    ?peer:(device -> Region.t -> (Region.t, string) result) ->
    ?link:(src:Buffer.t -> dst:Buffer.t -> link option) ->
    ?dma:(Region.t -> (dma, string) result) ->
    ?resolve:(nativeint -> unit) ->
    ?synchronized:(unit -> unit) ->
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
      - [load ~binary ~entry] loads the function [entry] of a program, or is
        [Error why] if the driver rejects it. The device keeps the programs it
        loads for its life. Without [load], it loads no programs.
      - [peer d' r] maps the region [r] of the device [d'] of the same machine
        for the device, for {!Buffer.borrow}: the region as the device's work
        addresses it, with the host address [r] has, if any, or [Error why] if
        the device cannot reach [d']'s memory. The driver keeps the mapping
        until [d'] frees [r] ({!on_free}); the runtime asks for a region again
        once the borrows of it are unreachable, and [peer] gives the same
        mapping. It runs with the device taken and [d'] free. Without it, the
        device borrows no other device's memory.
      - [link ~src ~dst] is how the device carries {!Buffer.copy} of [src] into
        [dst] when their devices are of two machines, if it does. Its [move]
        runs with the devices of [src], [dst] and [through] taken and those of
        [src] and [dst] synchronized.
      - [dma r] is how the other PCI functions of the device's machine reach its
        memory [r], or [Error why] ({!val-dma}). Without it, the device
        describes no memory.
      - [resolve a] runs once the work of a span that {!Profile.record} recorded
        on the device completed, before the runtime reads its stamps, with the
        host address [a] of the stamps: it writes the timestamps that the
        device's work does not write itself. It runs before [synchronized].
        Defaults to doing nothing.
      - [synchronized ()] runs at the end of each synchronization of the device.
        Defaults to doing nothing.
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

  val on_free : Buffer.t -> (unit -> unit) -> unit
  (** [on_free b f] runs [f] once [b]'s device has synchronized and before it
      frees the region [b] lies in to its driver, so that another device that
      mapped the region unmaps it in [f]. Memory kept in the device's cache
      stays mapped. If [f] raises [Failure], the memory is retained instead of
      freed. [f] runs with [b]'s device taken.

      Raises [Invalid_argument] if [b] is borrowed, empty, or on
      {!Nx_device.val-host}, whose memory the heap frees. *)
end
