(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Buffer ownership independent of compilation and execution. *)

module Buffer_spec : sig
  type t = {
    uncached : bool;  (** [true] to request uncached memory. *)
    cpu_access : bool;  (** [true] to request CPU-accessible device memory. *)
    host : bool;  (** [true] to allocate in host memory. *)
    nolru : bool;  (** [true] to bypass the LRU allocator cache on free. *)
    external_ptr : nativeint option;
        (** External backing pointer, or [None] to let the allocator choose.
            Buffers with an external pointer bypass LRU caching on free. *)
  }
  (** Buffer allocation options. *)

  val default : t
  (** [default] is
      [{uncached = false; cpu_access = false; host = false; nolru = false;
       external_ptr = None}]. *)
end

(** {1:allocator Allocator} *)

(** Backend allocator interface.

    An allocator manages device buffer lifecycle: allocation, data transfer,
    addressing, and optional features such as offset views and device-to-device
    copies. The buffer type ['buf] is backend-specific and hidden behind
    {!packed} at the device level.

    Allocator wrappers can add caching without changing buffer ownership. *)
module Allocator : sig
  (** {1:types Types} *)

  type 'buf transfer = dest:'buf -> src:'buf -> int -> unit
  (** The type for device-to-device transfers. [transfer ~dest ~src nbytes]
      copies [nbytes] from [src] to [dest]. Both buffers belong to the same
      backend.

      The transfer carries no device identities, so it can only express copies
      a single execution context can order. Backends whose instances hold
      independent command queues (for example separate GPUs of one vendor)
      cannot serialise a cross-instance copy through this hook and must fall
      back to a host bounce. *)

  type host_view =
    (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
  (** The type for a buffer's bytes seen from the host. *)

  type 'buf t = {
    alloc : int -> Buffer_spec.t -> 'buf;
        (** [alloc nbytes spec] allocates a device buffer of [nbytes] bytes with
            options [spec]. *)
    free : 'buf -> int -> Buffer_spec.t -> unit;
        (** [free buf nbytes spec] releases [buf]. [nbytes] and [spec] must
            match the values passed to {!field-alloc}. *)
    copyin : 'buf -> bytes -> unit;
        (** [copyin buf src] copies [src] into [buf]. *)
    copyout : bytes -> 'buf -> unit;
        (** [copyout dst buf] copies [buf] into [dst]. *)
    as_buffer : ('buf -> int -> host_view) option;
        (** [as_buffer buf nbytes] is the [nbytes] bytes of [buf] as host memory,
            without a copy, on backends whose memory the host addresses, or
            [None]. The view aliases [buf]: it is valid while [buf] is
            allocated, and it races with queued work that uses [buf] unless the
            device is synchronized first. *)
    addr : 'buf -> nativeint;  (** [addr buf] is the device address of [buf]. *)
    offset : ('buf -> int -> int -> 'buf) option;
        (** [offset buf nbytes byte_offset] is a view into [buf] starting at
            [byte_offset] and spanning [nbytes], or [None] if the backend does
            not support offset views. *)
    transfer : 'buf transfer option;
        (** Device-to-device transfer, or [None] if unsupported. *)
    supports_transfer : bool;  (** [true] iff {!field-transfer} is [Some _]. *)
    copy_from_disk : ('buf -> 'buf -> int -> unit) option;
        (** Direct disk-to-device copy, or [None] if unsupported. *)
    supports_copy_from_disk : bool;
        (** [true] iff {!field-copy_from_disk} is [Some _]. *)
  }
  (** The type for backend allocators parameterised by the buffer representation
      ['buf]. *)

  type packed =
    | Pack : 'buf t -> packed
        (** Existential wrapper hiding the backend buffer type. *)
end

(** {1:types Types} *)

type t
(** The type for existentially-packed device buffers. *)

(** {1:constructors Constructors} *)

val create :
  device:string ->
  size:int ->
  dtype:Dtype.t ->
  ?spec:Buffer_spec.t ->
  Allocator.packed ->
  t
(** [create ~device ~size ~dtype ?spec allocator] is an unallocated base
    buffer for [size] elements of [dtype] on [device]. Raises
    [Invalid_argument] if [dtype] is weak, [size] is negative, or the byte size
    exceeds [max_int].

    [spec] defaults to {!Buffer_spec.default}. *)

val on_device :
  device:string -> size:int -> dtype:Dtype.t -> ?spec:Buffer_spec.t -> unit -> t
(** [on_device ~device ~size ~dtype ?spec ()] is an unallocated buffer whose
    allocator is resolved on first use. Constructing it does not open [device].
    Size validation and [spec] have the same contract as {!create}. *)

val install_allocator_resolver : (string -> Allocator.packed) -> unit
(** [install_allocator_resolver f] connects lazy buffers to the device registry.
    The runtime installs this once; graph construction does not need it. *)

val view : t -> size:int -> dtype:Dtype.t -> offset:int -> t
(** [view b ~size ~dtype ~offset] is a view into [b] starting at byte [offset]
    and spanning [size] elements of [dtype]. The view shares the base buffer's
    allocator and spec.

    An empty view may start at [nbytes b]. Raises [Invalid_argument] if
    [offset] is negative, past [nbytes b], at [nbytes b] for a nonempty view,
    if [size] is negative or its byte size exceeds [max_int], or if the
    resulting view extends past the root base buffer. *)

(** {1:identity Identity and metadata} *)

val id : t -> int
(** [id b] is [b]'s globally unique identifier. *)

val base_id : t -> int
(** [base_id b] is the unique identifier of [b]'s root base buffer. Equal to
    [id b] when [b] is itself a base buffer. *)

val device : t -> string
(** [device b] is the device name [b] is bound to. *)

val size : t -> int
(** [size b] is the element count. *)

val dtype : t -> Dtype.t
(** [dtype b] is the element dtype. *)

val spec : t -> Buffer_spec.t
(** [spec b] is the buffer specification. *)

val nbytes : t -> int
(** [nbytes b] is the size in bytes ([size b * Dtype.itemsize (dtype b)]). *)

val base : t -> t
(** [base b] is the root base buffer. If [b] is already a base buffer,
    [base b] is [b] itself. *)

val offset : t -> int
(** [offset b] is the byte offset into the base buffer. [0] for base buffers.
*)

(** {1:allocation Allocation} *)

val allocate : t -> unit
(** [allocate b] materialises backing storage for [b]. For views, ensures the
    base buffer is allocated first, then creates the offset view via the
    allocator. Empty buffers and views acquire a storage identity without
    calling the allocator or retaining a native allocation.

    Raises [Invalid_argument] if [b] is already allocated, or if [b] is a
    nonempty view and the allocator does not support {!Allocator.offset}. *)

val ensure_allocated : t -> unit
(** [ensure_allocated b] calls {!allocate} if [b] is not yet initialised.
    No-op otherwise. *)

val is_allocated : t -> bool
(** [is_allocated b] is [true] iff [b] has initialized empty storage or its
    base buffer's storage exists. *)

val is_initialized : t -> bool
(** [is_initialized b] is [true] iff this specific buffer or view has its own
    storage initialized, including empty storage without a pointer. A view
    can be uninitialised even when the base buffer is allocated. *)

val allocated_views : t -> int
(** [allocated_views b] is the number of nonempty allocated views of [b]'s
    root base buffer. {!deallocate} refuses a base buffer while it is positive. *)

val deallocate : t -> unit
(** [deallocate b] releases backing storage if allocated. For base buffers,
    frees via the allocator. For views, detaches from the base buffer. No-op
    if already deallocated.

    Raises [Invalid_argument] if [b] is a base buffer that still has allocated
    views. *)

val supports_offset : t -> bool
(** [supports_offset b] is [true] iff [b]'s allocator provides offset views.
*)

val supports_transfer : t -> t -> bool
(** [supports_transfer dst src] is [true] iff [dst]'s allocator provides
    native transfer and [dst] and [src] are on the same backend prefix
    (for example ["METAL"] for ["METAL:0"] and ["METAL:1"]). *)

val allocator : t -> Allocator.packed
(** [allocator b] is the allocator of [b]'s base buffer. *)

(** {1:refcount Reference counting} *)

val uop_refcount : t -> int
(** [uop_refcount b] is the base buffer's UOp reference count. *)

val add_ref : t -> int -> t
(** [add_ref b cnt] increments the base buffer's UOp reference count by [cnt]
    and returns [b]. *)

(** {1:data_transfer Data transfer}

    {!copy_from} is the canonical way to move data between buffers. The
    primitives below expose a buffer's allocator directly and exist for the
    execution engine to service copies; prefer {!copy_from} in application
    code. *)

val copyin : t -> bytes -> unit
(** [copyin b src] writes the raw bytes [src] into [b]'s backing store through
    its allocator. Low-level host-to-device primitive; application code should
    move data with {!copy_from}.

    Raises [Invalid_argument] if [Bytes.length src <> nbytes b] or if [b] is
    not allocated. *)

val copyout : t -> bytes -> unit
(** [copyout b dst] reads the raw bytes of [b] from its backing store into
    [dst] through its allocator. Low-level device-to-host primitive;
    application code should move data with {!copy_from}.

    Raises [Invalid_argument] if [Bytes.length dst <> nbytes b] or if [b] is
    not allocated. *)

val as_buffer : t -> Allocator.host_view option
(** [as_buffer b] is [b]'s bytes as host memory, without a copy, when its
    allocator has {!Allocator.field-as_buffer}, as tinygrad's zero-copy
    [as_memoryview]. The device is not synchronized: the caller waits for the
    work that writes [b] before reading, and the view must not outlive [b]'s
    allocation.

    Raises [Invalid_argument] if [b] is not allocated. *)

val as_bytes : t -> bytes
(** [as_bytes b] is a fresh [bytes] value containing the contents of [b].
    Equivalent to allocating [Bytes.create (nbytes b)] and calling {!copyout}.
*)

val transfer : dst:t -> src:t -> bool
(** [transfer ~dst ~src] copies [src] into [dst] through [dst]'s allocator
    device-to-device transfer hook when {!supports_transfer} is [true],
    returning [true] when the native transfer ran and [false] when no hook is
    available. Both buffers are allocated if the transfer runs. Low-level
    same-backend primitive that {!copy_from} uses as a fast path; application
    code should use {!copy_from}.

    Raises [Invalid_argument] if [dst] and [src] differ in size or dtype. *)

val copy_between : dst:t -> src:t -> unit
(** [copy_between ~dst ~src] copies the contents of [src] into [dst] via a
    host-memory bounce buffer. Both buffers are allocated if needed.

    Raises [Invalid_argument] if [size dst <> size src] or
    [dtype dst <> dtype src]. *)

val copy_from : dst:t -> src:t -> unit
(** [copy_from ~dst ~src] copies the contents of [src] into [dst], allocating
    either buffer as needed. This is the canonical way to move data between
    buffers, including across devices: the copy is scheduled and executed as a
    device operation rather than a host-side byte shuffle.

    Raises [Invalid_argument] if [dst] and [src] differ in size or dtype. *)

val install_copy_runner : (dst:t -> src:t -> unit) -> unit
(** [install_copy_runner f] provides the implementation used by {!copy_from}.
    The execution engine installs it once during initialization; until then
    {!copy_from} raises [Invalid_argument]. Not for application use. *)

val addr : t -> nativeint
(** [addr b] is the device address of [b], or [0n] for empty storage.
    Initializes [b] if needed. *)

(** {1:accounting Allocation accounting} *)

val mem_used : int ref
(** [mem_used] counts live internally allocated bytes, excluding disk storage. *)

val mem_used_per_device : (string, int) Hashtbl.t
(** [mem_used_per_device] contains the same accounting grouped by device. *)

(** {1 Serialization} *)

type snapshot
(** A portable storage description containing bytes and base/view relationships. *)

val snapshot : t list -> snapshot list
(** [snapshot buffers] captures allocation state and bytes, preserving shared
    bases across [buffers]. External storage is copied into owned bytes. *)

val of_snapshot : snapshot list -> t list
(** [of_snapshot snapshots] restores independent storage through the device
    registry. Shared bases within [snapshots] remain shared after restoration. *)
