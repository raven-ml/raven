(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Device runtime abstraction.

    A {e device} bundles the pieces needed to run compiled kernels on a specific
    backend: an {!Allocator.packed} for buffer management, a {!Renderer_set.t}
    for renderer/compiler selection, a {!Queue.t} for kernel dispatch, and a
    preparation hook for device-specific program setup.

    {!Buffer.t} values are existentially packed so that the concrete backend
    buffer type does not leak into consumer code. {!Program.t} values carry
    compiled binaries together with their runtime metadata. Compiled programs
    are cached per device and compiler context. *)

(** {1:types Types} *)

type t
(** The type for compiled device runtimes. *)

type device = t
(** Alias for {!t}, used in signatures where [device] reads better than
    [Device.t]. *)

(** {1:buffer_spec Buffer specification} *)

(** Buffer allocation options.

    A {!t} describes allocation constraints for a device buffer: memory
    location, caching policy, and optional external backing. *)
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

    See {!Lru_allocator} for LRU caching on top of a raw allocator. *)
module Allocator : sig
  (** {1:types Types} *)

  type 'buf transfer = dest:'buf -> src:'buf -> int -> unit
  (** The type for device-to-device transfers. [transfer ~dest ~src nbytes]
      copies [nbytes] from [src] to [dest]. Both buffers belong to the same
      backend. *)

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

(** {1:lru_allocator LRU allocator} *)

(** LRU buffer reuse layer.

    Wraps a raw allocator so that freed buffers are cached by [(size, spec)] and
    reused on subsequent allocations. Buffers marked {!Buffer_spec.nolru} or
    carrying an {!Buffer_spec.external_ptr} bypass the cache and are freed
    immediately. When a fresh allocation fails, the entire cache is flushed and
    the allocation is retried once. *)
module Lru_allocator : sig
  val wrap : 'buf Allocator.t -> 'buf Allocator.t
  (** [wrap alloc] is [alloc] augmented with LRU buffer reuse. *)
end

(** {1:buffer Buffers} *)

(** Existentially-packed device buffers.

    A buffer is either a {e base buffer} (directly allocated) or a {e view} into
    a base buffer at a byte offset. Views share the base buffer's backing
    storage.

    Buffers start unallocated. Call {!allocate} or {!ensure_allocated} to
    materialise backing storage. Each buffer has a globally unique {!id}
    assigned at creation. A GC finaliser calls {!deallocate} when the buffer
    becomes unreachable.

    Reference counting ({!uop_refcount}, {!add_ref}) is managed externally by
    the compiler runtime and is not used for deallocation. *)
module Buffer : sig
  (** {1:types Types} *)

  type t
  (** The type for existentially-packed device buffers. *)

  (** {1:constructors Constructors} *)

  val create :
    device:string ->
    size:int ->
    dtype:Tolk_ir.Dtype.t ->
    ?spec:Buffer_spec.t ->
    Allocator.packed ->
    t
  (** [create ~device ~size ~dtype ?spec allocator] is an unallocated base
      buffer for [size] elements of [dtype] on [device].

      [spec] defaults to {!Buffer_spec.default}. *)

  val view : t -> size:int -> dtype:Tolk_ir.Dtype.t -> offset:int -> t
  (** [view b ~size ~dtype ~offset] is a view into [b] starting at byte [offset]
      and spanning [size] elements of [dtype]. The view shares the base buffer's
      allocator and spec.

      Raises [Invalid_argument] if [offset] is negative or [>= nbytes b]. *)

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

  val dtype : t -> Tolk_ir.Dtype.t
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
      allocator.

      Raises [Invalid_argument] if [b] is already allocated, or if [b] is a view
      and the allocator does not support {!Allocator.offset}. *)

  val ensure_allocated : t -> unit
  (** [ensure_allocated b] calls {!allocate} if [b] is not yet initialised.
      No-op otherwise. *)

  val is_allocated : t -> bool
  (** [is_allocated b] is [true] iff the base buffer's backing storage exists.
  *)

  val is_initialized : t -> bool
  (** [is_initialized b] is [true] iff this specific buffer or view has its own
      storage pointer set. A view can be uninitialised even when the base buffer
      is allocated. *)

  val deallocate : t -> unit
  (** [deallocate b] releases backing storage if allocated. For base buffers,
      frees via the allocator. For views, detaches from the base buffer. No-op
      if already deallocated.

      Raises [Invalid_argument] if [b] is a base buffer that still has allocated
      views. *)

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

  (** {1:data_transfer Data transfer} *)

  val copyin : t -> bytes -> unit
  (** [copyin b src] copies [src] into [b].

      Raises [Invalid_argument] if [Bytes.length src <> nbytes b] or if [b] is
      not allocated. *)

  val copyout : t -> bytes -> unit
  (** [copyout b dst] copies the contents of [b] into [dst].

      Raises [Invalid_argument] if [Bytes.length dst <> nbytes b] or if [b] is
      not allocated. *)

  val as_bytes : t -> bytes
  (** [as_bytes b] is a fresh [bytes] value containing the contents of [b].
      Equivalent to allocating [Bytes.create (nbytes b)] and calling {!copyout}.
  *)

  val copy_between : dst:t -> src:t -> unit
  (** [copy_between ~dst ~src] copies the contents of [src] into [dst] via a
      host-memory bounce buffer. Both buffers are allocated if needed.

      Raises [Invalid_argument] if [size dst <> size src] or
      [dtype dst <> dtype src]. *)

  val addr : t -> nativeint
  (** [addr b] is the device address of [b]. Allocates [b] if needed. *)
end

(** {1:prog Runtime program handle} *)

type prog = {
  call :
    nativeint array -> global:int array -> local:int array option ->
    vals:int64 array -> wait:bool -> timeout:int option -> float option;
  free : unit -> unit;
}
(** A device-specific dispatch handle. *)

type runtime = string -> bytes -> runtimevars:(string * int) list -> prog
(** [runtime name lib ~runtimevars] creates a dispatch handle for [lib]
    with entry point [name]. [runtimevars] maps variable names (e.g.
    ["core_id"]) to their index in the vals array. *)

(** {1:renderer_set Renderer selection} *)

(** Available renderers for a device.

    Each renderer carries its own {!Compiler.t} via {!Renderer.compiler}.
    The active renderer is chosen at {!Device.compile_program} time:
    explicit environment override takes priority, then forced entries
    ([ctrl = 1]), then the first non-disabled entry. *)
module Renderer_set : sig
  type t
  (** The type for renderer sets. *)

  val make :
    ?ctrl:string Helpers.Context_var.t ->
    (Renderer.t * int Helpers.Context_var.t option) list ->
    t
  (** [make ?ctrl entries] is a renderer set from [entries]. Each entry
      pairs a renderer with an optional environment variable control
      ([1] forces selection, [0] disables). [ctrl] is a global override
      that selects by compiler name (case-insensitive). *)
end

(** {1:device_operations Device operations} *)

val make :
  name:string ->
  allocator:Allocator.packed ->
  renderer_set:Renderer_set.t ->
  runtime:runtime ->
  synchronize:(unit -> unit) ->
  ?invalidate_caches:(unit -> unit) ->
  unit ->
  t
(** [make ~name ~allocator ~renderer_set ~runtime ~synchronize
    ?invalidate_caches ()] is a device runtime.

    [runtime name lib] loads a compiled binary and returns a dispatch handle.

    [synchronize ()] blocks until all pending work on the device completes. *)

val name : t -> string
(** [name d] is [d]'s device name. *)

val renderer : t -> Renderer.t
(** [renderer d] is the active renderer. *)

val runtime : t -> runtime
(** [runtime d] is [d]'s runtime factory. *)

val synchronize : t -> unit
(** [synchronize d] blocks until all pending work on [d] completes. *)

val compile_program :
  t ->
  ?name:string ->
  ?applied_opts:Tolk_ir.Kernel.Opt.t list ->
  ?estimates:Program_spec.Estimates.t ->
  Tolk_ir.Program.t ->
  Program_spec.t
(** [compile_program d ?name ?estimates program] renders and compiles [program]
    for [d], returning a prepared {!Program.t}.

    Results are cached by device name, compiler name, kernel content digest,
    renderer context, entry name, and estimates. Cached programs are cloned
    (entry address and cleanup cleared) before being returned.

    [name] defaults to ["kern"]. [estimates] defaults to
    {!Program_spec.Estimates.zero}. *)

val create_buffer :
  size:int -> dtype:Tolk_ir.Dtype.t -> ?spec:Buffer_spec.t -> t -> Buffer.t
(** [create_buffer ~size ~dtype ?spec d] is an unallocated buffer for [size]
    elements of [dtype] on [d].

    [spec] defaults to {!Buffer_spec.default}. *)

val invalidate_caches : t -> unit
(** [invalidate_caches d] flushes device caches (e.g., L2) if the device
    supports it. No-op if [~invalidate_caches] was not provided to {!make}.
    Called by beam search between timing runs for consistent measurements. *)

(** {1:multi_buffer Multi-device buffers} *)

(** Buffers spanning multiple devices.

    A multi-device buffer holds one {!Buffer.t} per device, all sharing the same
    size and dtype. Operations apply element-wise across the per-device buffers.
*)
module Multi_buffer : sig
  (** {1:types Types} *)

  type t
  (** The type for multi-device buffers. *)

  (** {1:constructors Constructors} *)

  val create :
    devices:device list ->
    size:int ->
    dtype:Tolk_ir.Dtype.t ->
    ?spec:Buffer_spec.t ->
    unit ->
    t
  (** [create ~devices ~size ~dtype ?spec ()] is a multi-device buffer with one
      underlying buffer per device in [devices].

      [spec] defaults to {!Buffer_spec.default}. The trailing [unit] argument is
      needed because [spec] is optional.

      Raises [Invalid_argument] if [devices] is empty. *)

  (** {1:accessors Accessors} *)

  val bufs : t -> Buffer.t list
  (** [bufs t] is the underlying per-device buffers, one per device in the order
      given to {!create}. *)

  val size : t -> int
  (** [size t] is the element count (same across all buffers). *)

  val dtype : t -> Tolk_ir.Dtype.t
  (** [dtype t] is the element dtype (same across all buffers). *)

  val is_allocated : t -> bool
  (** [is_allocated t] is [true] iff all underlying buffers are allocated. *)

  (** {1:operations Operations} *)

  val add_ref : t -> int -> t
  (** [add_ref t cnt] increments the UOp reference count on all underlying
      buffers by [cnt] and returns [t]. *)

  val copy_between : dst:t -> src:t -> unit
  (** [copy_between ~dst ~src] copies pairwise across the underlying buffers.

      Raises [Invalid_argument] if [dst] and [src] have different numbers of
      devices. *)
end
