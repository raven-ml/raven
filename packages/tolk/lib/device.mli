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
      backend.

      The transfer carries no device identities, so it can only express copies
      a single execution context can order. Backends whose instances hold
      independent command queues (for example separate GPUs of one vendor)
      cannot serialise a cross-instance copy through this hook and must fall
      back to a host bounce. *)

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
    dtype:Tolk_uop.Dtype.t ->
    ?spec:Buffer_spec.t ->
    Allocator.packed ->
    t
  (** [create ~device ~size ~dtype ?spec allocator] is an unallocated base
      buffer for [size] elements of [dtype] on [device].

      [spec] defaults to {!Buffer_spec.default}. *)

  val view : t -> size:int -> dtype:Tolk_uop.Dtype.t -> offset:int -> t
  (** [view b ~size ~dtype ~offset] is a view into [b] starting at byte [offset]
      and spanning [size] elements of [dtype]. The view shares the base buffer's
      allocator and spec.

      Raises [Invalid_argument] if [offset] is negative, [>= nbytes b], or if
      the resulting view extends past the root base buffer. *)

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

  val dtype : t -> Tolk_uop.Dtype.t
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
  (** [addr b] is the device address of [b]. Allocates [b] if needed. *)
end

(** {1:prog Runtime program handle} *)

type prog = {
  call :
    nativeint array -> global:int array -> local:int array option ->
    vals:int64 array -> wait:bool -> timeout:int option -> float option;
  free : unit -> unit;
  handle : nativeint;
      (** Backend kernel handle used to build {!Graph} nodes. [0n] when the
          backend has no addressable kernel object. *)
}
(** A device-specific dispatch handle. *)

type runtime = string -> bytes -> runtimevars:(string * int) list -> prog
(** [runtime name lib ~runtimevars] creates a dispatch handle for [lib]
    with entry point [name]. [runtimevars] maps variable names (e.g.
    ["core_id"]) to their index in the vals array. *)

(** {1:graph Batched dispatch graphs} *)

(** Backend interface for batched replay of a fixed call sequence.

    A graph records a sequence of kernel launches and buffer copies once and
    replays them with a single dispatch, eliminating per-call launch
    overhead. The engine builds the node list, tracks node dependencies, and
    patches per-replay state (rebound buffer arguments, variable values,
    launch dimensions) through {!exec} before each launch. *)
module Graph : sig
  type node =
    | Kernel of {
        handle : nativeint;  (** Kernel handle from {!prog.handle}. *)
        global : int array;  (** Global launch dimensions (3 entries). *)
        local : int array;  (** Local launch dimensions (3 entries). *)
        bufs : nativeint array;  (** Buffer argument addresses. *)
        vals : int array;  (** Scalar arguments. *)
        deps : int array;  (** Indices of nodes this node must wait on. *)
      }
    | Copy of {
        dest : nativeint;  (** Destination address. *)
        src : nativeint;  (** Source address. *)
        nbytes : int;  (** Copied byte count. *)
        deps : int array;  (** Indices of nodes this node must wait on. *)
      }  (** One recorded call. Node indices follow build order. *)

  type exec = {
    set_buf : int -> int -> nativeint -> unit;
        (** [set_buf node pos addr] stages buffer argument [pos] of [node] to
            [addr]. For {!constructor-Copy} nodes position [0] is the
            destination and position [1] the source. *)
    set_val : int -> int -> int -> unit;
        (** [set_val node idx v] stages scalar argument [idx] of [node]. *)
    set_launch_dims : int -> global:int array -> local:int array -> unit;
        (** [set_launch_dims node ~global ~local] stages new launch
            dimensions for kernel [node]. *)
    set_params : int -> unit;
        (** [set_params node] commits the staged state of [node] into the
            instantiated graph. *)
    launch : wait:bool -> float option;
        (** [launch ~wait] replays the graph. Returns the elapsed device time
            in seconds when [wait] is [true] and the backend supports
            timing. *)
  }
  (** An instantiated graph. *)

  type t = {
    supports_copy : bool;
        (** [true] iff {!constructor-Copy} nodes are supported, allowing the
            engine to batch buffer copies alongside kernels. *)
    build : node array -> exec;
        (** [build nodes] records and instantiates a graph over [nodes]. *)
  }
  (** The type for backend graph capabilities. *)
end

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
  ?graph:Graph.t ->
  unit ->
  t
(** [make ~name ~allocator ~renderer_set ~runtime ~synchronize
    ?invalidate_caches ?graph ()] is a device runtime.

    [runtime name lib] loads a compiled binary and returns a dispatch handle.

    [synchronize ()] blocks until all pending work on the device completes.

    [graph] is the batched-dispatch capability, or absent when the backend
    cannot replay call sequences as a single dispatch. *)

val name : t -> string
(** [name d] is [d]'s device name. *)

val renderer : t -> Renderer.t
(** [renderer d] is the active renderer. *)

val runtime : t -> runtime
(** [runtime d] is [d]'s runtime factory. *)

val synchronize : t -> unit
(** [synchronize d] blocks until all pending work on [d] completes. *)

val graph : t -> Graph.t option
(** [graph d] is [d]'s batched-dispatch capability, if any. *)

val compile_program :
  t ->
  ?name:string ->
  ?applied_opts:Tolk_uop.Uop.Opt.t list ->
  ?estimates:Program_spec.Estimates.t ->
  Program_spec.program ->
  Program_spec.t
(** [compile_program d ?name ?estimates program] renders and compiles [program]
    for [d], returning a prepared {!Program.t}.

    Results are cached by device name, compiler name, kernel content digest,
    renderer context, entry name, and estimates. Cached programs are cloned
    (entry address and cleanup cleared) before being returned.

    [name] defaults to ["kern"]. [estimates] defaults to
    {!Program_spec.Estimates.zero}. *)

val create_buffer :
  size:int -> dtype:Tolk_uop.Dtype.t -> ?spec:Buffer_spec.t -> t -> Buffer.t
(** [create_buffer ~size ~dtype ?spec d] is an unallocated buffer for [size]
    elements of [dtype] on [d].

    [spec] defaults to {!Buffer_spec.default}. *)

val invalidate_caches : t -> unit
(** [invalidate_caches d] flushes device caches (e.g., L2) if the device
    supports it. No-op if [~invalidate_caches] was not provided to {!make}.
    Called by beam search between timing runs for consistent measurements. *)

(** {1:registry Device registry}

    The registry maps canonical device names to opened device runtimes.
    Backends register an opener per name prefix (e.g. ["CPU"]); {!get} opens
    a device on first lookup and caches it, so every consumer of a device
    name shares one runtime instance per canonical name. *)

val canonicalize : string -> string
(** [canonicalize name] is [name] with its backend prefix uppercased and a
    trailing [":0"] instance suffix removed, so ["cpu:0"], ["CPU:0"] and
    ["CPU"] all name the same device. *)

val register : string -> (string -> t) -> unit
(** [register prefix opener] installs [opener] for device names whose backend
    prefix is [prefix] (case-insensitive). [opener name] must return the
    device runtime for the canonical [name]. *)

val get : string -> t
(** [get name] is the device runtime for the canonicalized [name], opened via
    its registered opener on first lookup and cached afterwards.

    Raises [Failure] if no opener is registered for [name]'s prefix or the
    opener fails. *)

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
    devices:string list ->
    size:int ->
    dtype:Tolk_uop.Dtype.t ->
    ?spec:Buffer_spec.t ->
    unit ->
    t
  (** [create ~devices ~size ~dtype ?spec ()] is a multi-device buffer with one
      underlying buffer per device name in [devices], each resolved through the
      device registry ({!get}).

      [spec] defaults to {!Buffer_spec.default}. The trailing [unit] argument is
      needed because [spec] is optional.

      Raises [Invalid_argument] if [devices] is empty and [Failure] if a device
      name cannot be opened. *)

  val of_bufs : Buffer.t list -> t
  (** [of_bufs bufs] is a multi-device buffer stacking the per-device buffers
      [bufs].

      Raises [Invalid_argument] if [bufs] is empty or the buffers disagree in
      size or dtype. *)

  val view : t -> size:int -> dtype:Tolk_uop.Dtype.t -> offset:int -> t
  (** [view t ~size ~dtype ~offset] is a multi-device buffer viewing each
      underlying buffer at byte [offset] for [size] elements of [dtype]. See
      {!Buffer.view}. *)

  (** {1:accessors Accessors} *)

  val bufs : t -> Buffer.t list
  (** [bufs t] is the underlying per-device buffers, one per device in the order
      given to {!create}. *)

  val size : t -> int
  (** [size t] is the element count (same across all buffers). *)

  val dtype : t -> Tolk_uop.Dtype.t
  (** [dtype t] is the element dtype (same across all buffers). *)

  val is_allocated : t -> bool
  (** [is_allocated t] is [true] iff all underlying buffers are allocated. *)

  (** {1:operations Operations} *)

  val add_ref : t -> int -> t
  (** [add_ref t cnt] increments the UOp reference count on all underlying
      buffers by [cnt] and returns [t]. *)
end
