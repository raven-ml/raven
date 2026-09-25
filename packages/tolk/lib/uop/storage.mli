(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Buffer ownership independent of compilation and execution. *)

type t
(** The type for device buffers and their owned storage. *)

exception Mapping_unavailable of string
(** A storage import is unsupported; callers may use an ordinary copy. *)

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

    An allocator manages allocation, addressing, mapping and offset views.
    Device execution owns transfers. The buffer type ['buf] is backend-specific
    and hidden behind
    {!packed} at the device level.

    Allocator wrappers can add caching without changing buffer ownership. *)
module Allocator : sig
  (** {1:types Types} *)

  type buffer = t
  (** An owned buffer that may be mapped by another allocator. *)

  type 'buf mapping = {
    map : buffer -> 'buf;
        (** [map source] maps the source allocation into this allocator's device.
            Raises {!Mapping_unavailable} when that storage cannot be imported.
            Raises [Fun.Finally_raised error] if rollback could not establish
            that the receiver no longer maps the source. This permanently
            retains the source allocation; explicit {!deallocate} also raises
            that exception. Other failures must leave no receiver mapping. *)
    unmap : 'buf -> unit;
        (** [unmap mapped] releases mapping metadata, without freeing source storage. *)
  }
  (** Cross-device mapping operations. *)

  type host_view =
    (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
  (** The type for a buffer's bytes seen from the host. *)

  type 'buf t = {
    host : 'buf -> nativeint option;
        (** Host address of the allocation, if CPU-accessible. *)
    mapping : 'buf mapping option;
        (** Mapping operations, absent when the device cannot map other storage. *)
    synchronize : unit -> unit;
        (** Waits for this allocator's device before host access or unmapping. *)
    kind : 'buf Type.Id.t;
        (** Identity of the backend buffer representation, shared by allocator
            instances whose buffers can be passed to the same runtime. *)
    alloc : int -> Buffer_spec.t -> 'buf;
        (** [alloc nbytes spec] allocates a device buffer of [nbytes] bytes with
            options [spec]. *)
    free : 'buf -> int -> Buffer_spec.t -> unit;
        (** [free buf nbytes spec] releases [buf]. [nbytes] and [spec] must
            match the values passed to {!field-alloc}. *)
    addr : ('buf -> nativeint) option;
        (** Device address access, absent for opaque buffer handles. *)
    offset : ('buf -> int -> int -> 'buf) option;
        (** [offset buf nbytes byte_offset] is a view into [buf] starting at
            [byte_offset] and spanning [nbytes], or [None] if the backend does
            not support offset views. *)
  }
  (** The type for backend allocators parameterised by the buffer representation
      ['buf]. *)

  type packed =
    | Pack : 'buf t -> packed
        (** Existential wrapper hiding the backend buffer type. *)
end

(** {1:types Types} *)

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

(** {1:operations Device operations} *)

val release : (unit -> unit) -> unit
(** [release action] calls [action ()] once. If it raises, [action] and the
    resources it owns are retained until process exit without automatic retry,
    and the original exception and backtrace are propagated. *)

val with_operation : (unit -> 'a) -> 'a
(** [with_operation f] is [f ()], deferring buffer finalizers on the current
    domain until the outermost operation returns. Deferred releases retain
    their buffers and use their allocators' normal synchronization. Storage
    operations already establish this scope; device runtimes also use it
    around setup, dispatch and submission.

    If [f] raises, pending releases wait for a later successful operation.
    A deferred release failure is propagated and its storage retained until
    process exit, without retrying uncertain teardown. Calls on different
    domains are not serialized. *)

(** {1:allocation Allocation} *)

val allocate : t -> unit
(** [allocate b] materialises backing storage for [b]. For views, ensures the
    base buffer is allocated first, then creates the offset view via the
    allocator. A view whose base was deallocated is refreshed against its
    current storage. Empty buffers and views acquire a storage identity without
    calling the allocator or retaining a native allocation.

    Raises [Invalid_argument] if [b] is already allocated, or if [b] is a
    nonempty view and the allocator does not support {!Allocator.offset}. *)

val ensure_allocated : t -> unit
(** [ensure_allocated b] calls {!allocate} if [b] is not yet initialised or
    its base storage has changed. No-op otherwise. *)

val is_allocated : t -> bool
(** [is_allocated b] is [true] iff [b] has initialized storage and, for a
    nonempty view, it belongs to the base's current allocation. Empty storage
    has no pointer. An unallocated view remains unallocated when only its base
    has storage. *)

val allocated_views : t -> int
(** [allocated_views b] is the number of initialized nonempty views of [b]'s
    root base buffer, including views awaiting refresh after base deallocation.
    Refreshing a view leaves this count unchanged. *)

val deallocate : t -> unit
(** [deallocate b] releases backing storage if allocated. For base buffers,
    frees via the allocator. For views, detaches from the base buffer. No-op
    if already deallocated. Live views become stale when their base is freed
    and refresh on their next access; they do not prevent deallocation.
    If an import rollback failed, re-raises its exception and keeps the source
    allocated, including through finalization. There is no automatic retry. *)

val supports_offset : t -> bool
(** [supports_offset b] is [true] iff [b]'s allocator provides offset views.
*)

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
    byte operations below synchronize host access or use owned host staging
    and scheduled device copies. *)

val copyin : t -> bytes -> unit
(** [copyin b src] writes [src] into [b] and waits for completion. Host-mapped
    storage is written directly after synchronization; other storage uses
    owned host staging and {!copy_from}. Staging is bounded to 64 MiB when
    the allocator supports offset views, otherwise it spans the whole buffer.

    A previously initialized view refreshes if its base storage has changed.

    Raises [Invalid_argument] if [Bytes.length src <> nbytes b] or if [b] has
    no initialized storage. *)

val copyout : t -> bytes -> unit
(** [copyout b dst] reads [b] into [dst] after synchronization, using direct
    host access or the same bounded staging as {!copyin}.

    A previously initialized view refreshes if its base storage has changed.

    Raises [Invalid_argument] if [Bytes.length dst <> nbytes b] or if [b] has
    no initialized storage. *)

val as_buffer : t -> Allocator.host_view option
(** [as_buffer b] is [b]'s bytes as host memory, without a copy, when its
    allocator exposes a {!Allocator.field-host} address, as tinygrad's zero-copy
    [as_memoryview]. The device is not synchronized: the caller waits for the
    work that writes [b] before reading, and the view must not outlive [b]'s
    allocation.

    A previously initialized view refreshes if its base storage has changed.

    Raises [Invalid_argument] if [b] has no initialized storage. *)

val as_bytes : t -> bytes
(** [as_bytes b] is a fresh [bytes] value containing the contents of [b].
    Equivalent to allocating [Bytes.create (nbytes b)] and calling {!copyout}.
*)

val copy_from : dst:t -> src:t -> unit
(** [copy_from ~dst ~src] copies the contents of [src] into [dst], allocating
    either buffer as needed. This is the canonical way to move data between
    buffers, including across devices: the copy is scheduled and executed as a
    device operation rather than a host-side byte shuffle.

    Raises [Invalid_argument] if [dst] and [src] differ in size or dtype. *)

val install_copy_runner : (dst:t -> src:t -> unit) -> unit
(** [install_copy_runner f] provides the implementation used by {!copy_from}.
    The code generator installs it once during initialization; until then
    {!copy_from} raises [Invalid_argument]. Not for application use. *)

val find_mapping : 'a Type.Id.t -> t -> 'a option
(** [find_mapping kind b] is an existing import of [b] with representation
    [kind], if any. It does not allocate or synchronize. Mapping callbacks
    use it to reuse a driver registration shared by several devices; the
    source owner retains imports and releases them in reverse creation order.
    Views preserve their byte offsets. The caller must retain [b]. *)

val get : ?device:string -> 'a Type.Id.t -> t -> 'a option
(** [get ?device kind b] initializes [b] and returns its backend buffer, or
    [None] for empty storage. [device] defaults to [b]'s device. Another device
    maps the base allocation once, then derives byte-offset views from that
    mapping. Mappings are retained by the source owner and unmapped before it
    is freed. Binding an existing mapping does not wait for its users: direct
    dispatch must call {!synchronize}, while compiled queues encode their own
    dependencies. Creating a new mapping may synchronize through the allocator.

    Raises [Invalid_argument] if [kind] differs from the target allocator's
    identity or it cannot map [b]. The caller must retain [b] while using the
    returned backend buffer. *)

val synchronize : ?device:string -> t -> unit
(** [synchronize ?device b] waits for other importing devices and, when [device]
    differs from [b]'s owner, for the owner as well. [device] defaults to [b]'s
    device, whose own dispatch order must be preserved by the caller. Use this
    before direct dispatch; compiled queue dependencies replace these waits. *)

val host_addr : t -> nativeint option
(** [host_addr b] initializes and synchronizes [b], then returns its host
    mapping, if any. The pointer is valid while [b] remains allocated. *)

val addr : ?device:string -> t -> nativeint
(** [addr ?device b] is the device address of [b], or [0n] for empty storage.
    Initializes or maps [b] as {!get} does, without waiting on an existing
    mapping. Raises [Invalid_argument] for opaque storage. *)

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

module Host_allocator : sig
  val kind : nativeint Type.Id.t
  (** [kind] identifies shared host-address storage. *)

  val make : synchronize:(unit -> unit) -> nativeint Allocator.t
  (** [make ~synchronize] allocates zeroed, page-backed host memory and maps
      CPU-accessible storage without copying. It accepts external pointers and byte views;
      host reads, writes and frees wait for [synchronize]. *)
end
