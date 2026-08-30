(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Hardware command-queue building blocks.

    Device-independent pieces for drivers that submit work to hardware
    through memory-mapped command queues: raw file and mapping primitives
    ({!File_io}), bounds-checked volatile access to mapped device memory
    ({!Mmio}), device-memory regions ({!Buffer}), command-stream
    accumulation ({!Q}), synchronization slots ({!Signal}) and kernel
    argument staging ({!Kernargs}). *)

(** Files and memory mappings. *)
module File_io : sig
  (** {1:files Files} *)

  val openfile : string -> flags:int -> int
  (** [openfile path ~flags] is a file descriptor open on [path] with
      [flags]; close-on-exec is always added. Raises [Failure] with the
      system error on failure. *)

  val close : int -> unit
  (** [close fd] closes [fd]. Raises [Failure] on failure. *)

  val o_rdonly : int
  (** Open read-only. *)

  val o_rdwr : int
  (** Open read-write. *)

  (** {1:mappings Mappings} *)

  val mmap :
    addr:nativeint ->
    size:int ->
    prot:int ->
    flags:int ->
    fd:int ->
    offset:int64 ->
    nativeint
  (** [mmap ~addr ~size ~prot ~flags ~fd ~offset] maps [size] bytes of
      [fd] at file offset [offset] into memory and is the address of the
      mapping. [addr] is the placement hint ([0n] lets the system
      choose); pass [~fd:(-1)] with {!map_anonymous} in [flags] for a
      mapping backed by fresh memory. Raises [Failure] with the system
      error on failure. *)

  val munmap : nativeint -> size:int -> unit
  (** [munmap addr ~size] unmaps the [size] bytes mapped at [addr].
      Raises [Failure] on failure. *)

  val prot_none : int
  (** Pages may not be accessed. *)

  val prot_read : int
  (** Pages may be read. *)

  val prot_write : int
  (** Pages may be written. *)

  val map_shared : int
  (** Writes are shared with other mappings of the object. *)

  val map_private : int
  (** Writes are private to this mapping. *)

  val map_anonymous : int
  (** The mapping is backed by fresh zeroed memory, not a file. *)

  val map_fixed : int
  (** Place the mapping exactly at [addr]. *)

  val map_noreserve : int
  (** Do not reserve swap space for the mapping; [0] on systems without
      the notion. *)
end

(** Bounds-checked access to a mapped device-memory region.

    All offsets and sizes are in bytes. Reads and writes are single
    volatile loads and stores of the given width, suitable for device
    registers and descriptors in mapped memory. Use {!Mmio.fence} to
    order them against other memory accesses observed by the device. *)
module Mmio : sig
  type t
  (** The type for mapped regions: a base address and a byte size. *)

  val make : addr:nativeint -> size:int -> t
  (** [make ~addr ~size] is the region of [size] bytes at virtual
      address [addr]. Raises [Invalid_argument] if [size] is negative. *)

  val addr : t -> nativeint
  (** [addr t] is the base address of the region. *)

  val size : t -> int
  (** [size t] is the size of the region in bytes. *)

  val view : t -> off:int -> ?size:int -> unit -> t
  (** [view t ~off ?size ()] is the subregion of [t] starting at byte
      offset [off] and covering [size] bytes (to the end of [t] by
      default). Raises [Invalid_argument] if the subregion does not fit
      in [t]. *)

  val read32 : t -> int -> int32
  (** [read32 t off] is the 32-bit value at byte offset [off]. Raises
      [Invalid_argument] if [off + 4] exceeds the region. *)

  val write32 : t -> int -> int32 -> unit
  (** [write32 t off v] stores the 32-bit value [v] at byte offset
      [off]. Raises [Invalid_argument] if [off + 4] exceeds the
      region. *)

  val read64 : t -> int -> int64
  (** [read64 t off] is the 64-bit value at byte offset [off]. Raises
      [Invalid_argument] if [off + 8] exceeds the region. *)

  val write64 : t -> int -> int64 -> unit
  (** [write64 t off v] stores the 64-bit value [v] at byte offset
      [off]. Raises [Invalid_argument] if [off + 8] exceeds the
      region. *)

  val blit_bytes : t -> off:int -> bytes -> unit
  (** [blit_bytes t ~off src] copies all of [src] into the region at
      byte offset [off]. Raises [Invalid_argument] if it does not
      fit. *)

  val read_bytes : t -> off:int -> len:int -> bytes
  (** [read_bytes t ~off ~len] is a copy of the [len] bytes at byte
      offset [off]. Raises [Invalid_argument] if the range exceeds the
      region. *)

  val fence : unit -> unit
  (** [fence ()] is a full memory barrier: memory accesses sequenced
      before it complete before any access sequenced after it. *)
end

(** Regions of device memory.

    A buffer names a range of the device's virtual address space,
    carries driver metadata, and optionally a CPU mapping of the range.
    Buffers are values: {!Buffer.offset} derives sub-buffers that share
    the parent's metadata and refer back to the root allocation through
    {!Buffer.base}. *)
module Buffer : sig
  type 'meta t
  (** The type for device-memory regions carrying metadata of type
      ['meta]. *)

  val make :
    va:nativeint -> size:int -> ?view:Mmio.t -> meta:'meta -> unit -> 'meta t
  (** [make ~va ~size ?view ~meta ()] is the region of [size] bytes at
      device virtual address [va]. [view] is a CPU mapping of the same
      bytes, when the region is CPU-accessible. Raises [Invalid_argument]
      if [size] is negative. *)

  val va : 'meta t -> nativeint
  (** [va t] is the device virtual address of the region. *)

  val size : 'meta t -> int
  (** [size t] is the size of the region in bytes. *)

  val view : 'meta t -> Mmio.t option
  (** [view t] is the CPU mapping of the region, if any. *)

  val cpu_view : 'meta t -> Mmio.t
  (** [cpu_view t] is the CPU mapping of the region. Raises
      [Invalid_argument] if the region has none. *)

  val meta : 'meta t -> 'meta
  (** [meta t] is the driver metadata of the region. Sub-buffers share
      the metadata of their parent. *)

  val base : 'meta t -> 'meta t
  (** [base t] is the root allocation [t] was derived from: the buffer
      itself unless it came from {!offset}. *)

  val offset : 'meta t -> off:int -> ?size:int -> unit -> 'meta t
  (** [offset t ~off ?size ()] is the sub-buffer of [t] starting at byte
      offset [off] and covering [size] bytes (to the end of [t] by
      default). Its view, when [t] has one, is narrowed to the same
      range. Raises [Invalid_argument] if the range does not fit in
      [t]. *)
end

(** Command-stream accumulation.

    A queue accumulates the 32-bit words of a hardware command stream in
    order. Values are stored exactly as pushed; {!Q.push} rejects
    anything that does not fit in 32 bits. *)
module Q : sig
  type t
  (** The type for dword streams. Mutable; grows as needed. *)

  val create : unit -> t
  (** [create ()] is an empty stream. *)

  val length : t -> int
  (** [length t] is the number of dwords accumulated. *)

  val push : t -> int -> unit
  (** [push t v] appends [v] to the stream. Raises [Invalid_argument]
      if [v] is negative or exceeds [0xFFFFFFFF]. *)

  val push64 : t -> int64 -> unit
  (** [push64 t v] appends the unsigned 64-bit value [v] as two dwords,
      low dword first. *)

  val get : t -> int -> int
  (** [get t i] is the [i]th dword of the stream. Raises
      [Invalid_argument] if [i] is out of bounds. *)

  val set : t -> int -> int -> unit
  (** [set t i v] replaces the [i]th dword of the stream with [v], for
      packets whose fields are only known once later dwords have been
      accumulated. Raises [Invalid_argument] if [i] is out of bounds, or
      if [v] is negative or exceeds [0xFFFFFFFF]. *)

  val dwords : t -> int array
  (** [dwords t] is a fresh array of the accumulated dwords, in push
      order. *)

  val clear : t -> unit
  (** [clear t] empties the stream. *)
end

(** Synchronization slots in shared memory.

    A signal is a 16-byte slot of CPU-mapped, device-visible memory: a
    64-bit value at byte offset [0] and a 64-bit timestamp at byte
    offset [8]. The device advances the value and records timestamps as
    work completes; the CPU polls the slot with {!Signal.wait}. *)
module Signal : sig
  type ('meta, 'dev) t
  (** The type for signals over a slot with metadata ['meta], optionally
      owned by a device of type ['dev]. *)

  exception Timeout of { timeout_ms : int; goal : int; value : int }
  (** Raised by {!wait} when the deadline expires: the signal stopped at
      [value] without reaching [goal]. *)

  val make :
    ?value:int ->
    ?is_timeline:bool ->
    ?timestamp_divider:float ->
    ?sleep:(int -> unit) ->
    ?owner:'dev ->
    'meta Buffer.t ->
    ('meta, 'dev) t
  (** [make buf] is a signal over the 16-byte slot at the start of
      [buf], with its value initialized to [value] (defaults to [0]).

      [is_timeline] marks the device's monotonic completion counter
      (defaults to [false]). [timestamp_divider] converts the raw
      timestamp counter to microseconds (defaults to [1000.], for a
      nanosecond counter). [sleep] is called on every {!wait} poll with
      the milliseconds elapsed since the last observed progress; it may
      yield the CPU to the driver and may raise to abort the wait
      (defaults to doing nothing). [owner] is the device that allocated
      the slot.

      Raises [Invalid_argument] if [buf] is smaller than 16 bytes or has
      no view. *)

  val buf : ('meta, 'dev) t -> 'meta Buffer.t
  (** [buf t] is the slot the signal lives in. *)

  val owner : ('meta, 'dev) t -> 'dev option
  (** [owner t] is the device that allocated the slot, if any. *)

  val is_timeline : ('meta, 'dev) t -> bool
  (** [is_timeline t] is [true] if the signal is a device's completion
      counter. *)

  val value_addr : ('meta, 'dev) t -> nativeint
  (** [value_addr t] is the device virtual address of the value. *)

  val timestamp_addr : ('meta, 'dev) t -> nativeint
  (** [timestamp_addr t] is the device virtual address of the
      timestamp. *)

  val value : ('meta, 'dev) t -> int
  (** [value t] is the current value, read with a single volatile load.
      The value is interpreted as a non-negative 63-bit integer. *)

  val set_value : ('meta, 'dev) t -> int -> unit
  (** [set_value t v] stores [v] as the current value with a single
      volatile store. *)

  val timestamp : ('meta, 'dev) t -> float
  (** [timestamp t] is the recorded timestamp in microseconds: the raw
      counter divided by the signal's timestamp divider. *)

  val wait : ('meta, 'dev) t -> ?timeout_ms:int -> int -> unit
  (** [wait t goal] polls until [value t >= goal]. Every observed change
      of the value resets the deadline, so only a stalled signal times
      out; the [sleep] hook of [t] runs on each poll. [timeout_ms]
      defaults to the [HCQDEV_WAIT_TIMEOUT_MS] environment variable, or
      30000. Raises {!Timeout} if the value stalls below [goal] for
      [timeout_ms] milliseconds. *)

  (** Slot pools.

      A pool carves pages of mapped memory into 16-byte signal slots and
      recycles them, most recently released first. *)
  module Pool : sig
    val slot_size : int
    (** [slot_size] is the size of a slot in bytes: 16. *)

    type 'meta t
    (** The type for slot pools over pages with metadata ['meta]. *)

    val create : alloc_page:(unit -> 'meta Buffer.t) -> 'meta t
    (** [create ~alloc_page] is an empty pool. [alloc_page] is called
        whenever no free slot remains; it must return a CPU-mapped
        buffer of at least {!slot_size} bytes, which the pool carves
        into as many slots as fit. *)

    val get : 'meta t -> 'meta Buffer.t
    (** [get t] is a free slot, allocating a fresh page if needed.
        Raises [Invalid_argument] if [alloc_page] returns a page smaller
        than {!slot_size}. *)

    val put : 'meta t -> 'meta Buffer.t -> unit
    (** [put t slot] returns [slot] to the pool for reuse. [slot] must
        have come from {!get} on [t] and no longer be in use. *)

    val pages : 'meta t -> 'meta Buffer.t list
    (** [pages t] is every page allocated so far, in allocation
        order. *)
  end
end

(** Kernel argument staging.

    A kernargs region hands out small slots of a CPU-mapped buffer for
    the arguments of individual kernel launches. The region recycles
    space by wrapping: slots are valid until the region wraps back
    around, which in-flight work must outlive by construction. *)
module Kernargs : sig
  type 'meta t
  (** The type for kernel-argument regions. Mutable. *)

  val create : 'meta Buffer.t -> 'meta t
  (** [create buf] is a region handing out slots of the CPU-mapped
      buffer [buf]. *)

  val alloc : 'meta t -> int -> 'meta Buffer.t
  (** [alloc t size] is a fresh 8-byte-aligned slot of [size] bytes,
      wrapping to the start of the region when the end is reached.
      Raises [Invalid_argument] if [size] exceeds the region. *)

  val write_args : 'meta Buffer.t -> bufs:nativeint array -> vals:int array -> unit
  (** [write_args slot ~bufs ~vals] lays out kernel arguments in
      [slot]: the addresses in [bufs] as 64-bit words from byte offset
      [0], then the values in [vals] as 32-bit words. Raises
      [Invalid_argument] if the layout does not fit in [slot], if a
      value does not fit in 32 bits, or if [slot] has no view. *)
end
