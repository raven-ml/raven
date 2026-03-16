(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Device runtime abstraction.

    A {e device} bundles the pieces needed to run compiled kernels on a specific
    backend: an {!Allocator.packed} for buffer management, a {!Compiler.set} for
    rendering and compiling IR programs, a {!Queue.t} for kernel dispatch, and a
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

(** {1:context Context variables} *)

(** Environment-backed configuration variables.

    A {!var} is bound to an environment variable name and carries a default
    value and a parser. The variable is resolved lazily from {!Sys.getenv} on
    each call to {!get} or {!get_opt}. *)
module Context : sig
  (** {1:types Types} *)

  type 'a var
  (** The type for environment-backed variables of type ['a]. *)

  (** {1:constructors Constructors} *)

  val make : name:string -> default:'a -> parse:(string -> 'a option) -> 'a var
  (** [make ~name ~default ~parse] is a variable bound to the environment
      variable [name]. [parse] converts the raw string value; it must return
      [None] to reject a value (in which case {!get} falls back to [default]).
  *)

  val int : name:string -> default:int -> int var
  (** [int ~name ~default] is a variable that parses an integer. Leading and
      trailing whitespace is trimmed before parsing. *)

  val string : name:string -> default:string -> string var
  (** [string ~name ~default] is a variable that parses a non-empty string. The
      value is trimmed; empty or whitespace-only values are treated as unset. *)

  (** {1:accessors Accessors} *)

  val get : 'a var -> 'a
  (** [get v] is the parsed value of [v], or the default if the environment
      variable is unset, empty, or rejected by the parser. *)

  val get_opt : 'a var -> 'a option
  (** [get_opt v] is [Some value] if the environment variable is set and
      accepted by the parser, and [None] otherwise. *)
end

(** {1:buffer_spec Buffer specification} *)

(** Buffer allocation options.

    A {!t} describes allocation constraints for a device buffer: memory
    location, caching policy, and optional external backing. *)
module Buffer_spec : sig
  type t = {
    image : Tolk_ir.Dtype.t option;
        (** Image format hint, or [None] for plain buffers. *)
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
      [{image = None; uncached = false; cpu_access = false; host = false; nolru
       = false; external_ptr = None}]. *)
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

(** {1:program Compiled programs} *)

(** Compiled kernel programs with runtime metadata.

    A {!t} bundles a compiled binary, the rendered source, and the
    {!Program_spec.t} from which launch dimensions, variable bindings, and
    buffer indices are derived. An optional cleanup callback (registered via
    {!set_cleanup}) releases device-specific resources when {!release} is
    called. *)
module Program : sig
  (** {1:types Types} *)

  type t
  (** The type for compiled kernel programs. *)

  (** {1:constructors Constructors} *)

  val make : spec:Program_spec.t -> src:string -> binary:bytes -> t
  (** [make ~spec ~src ~binary] is a program with the given spec, rendered
      source, and compiled binary. The entry address and cleanup callback are
      initially unset. *)

  (** {1:accessors Accessors} *)

  val name : t -> string
  (** [name t] is the kernel name from the spec. *)

  val src : t -> string
  (** [src t] is the rendered source code. *)

  val binary : t -> bytes
  (** [binary t] is the compiled binary blob. *)

  val entry_name : t -> string
  (** [entry_name t] is the entry symbol name. Currently equal to {!name}. *)

  val entry_addr : t -> nativeint option
  (** [entry_addr t] is the cached entry address, or [None] if not yet resolved.
  *)

  val vars : t -> Program_spec.var list
  (** [vars t] is the scalar variable definitions in argument order. *)

  val outs : t -> int list
  (** [outs t] is the written buffer parameter indices. *)

  val ins : t -> int list
  (** [ins t] is the read buffer parameter indices. *)

  val core_id : t -> Program_spec.core_id option
  (** [core_id t] is the runtime-managed ["core_id"] metadata, if any. *)

  val launch_kind : t -> Program_spec.launch_kind
  (** [launch_kind t] is the kernel launch model. *)

  val estimates : t -> Program_spec.Estimates.t
  (** [estimates t] is the kernel cost estimates. *)

  val launch_dims : t -> int list -> int array * int array option
  (** [launch_dims t args] evaluates launch dimensions from scalar [args].
      Delegates to {!Program_spec.launch_dims}. *)

  (** {1:lifecycle Lifecycle} *)

  val set_entry_addr : t -> nativeint -> unit
  (** [set_entry_addr t addr] caches the resolved entry address. *)

  val set_cleanup : t -> (unit -> unit) -> unit
  (** [set_cleanup t f] registers a device-specific cleanup callback. Replaces
      any previously registered callback. *)

  val release : t -> unit
  (** [release t] invokes the cleanup callback (if any), then clears both the
      cleanup callback and the cached entry address. *)
end

(** {1:queue Execution queue} *)

(** Kernel execution queue.

    A queue serialises kernel dispatch and synchronisation for one device. The
    abstraction models device execution only; backend-specific host scheduling
    (e.g., CPU worker fan-out) does not belong here. *)
module Queue : sig
  (** {1:types Types} *)

  type t = {
    exec : Program.t -> Buffer.t list -> int list -> unit;
        (** [exec program bufs var_args] dispatches [program] with [bufs] as
            buffer arguments and [var_args] as scalar variable values. *)
    synchronize : unit -> unit;
        (** [synchronize ()] blocks until all queued work completes. *)
  }
  (** The type for kernel execution queues. *)

  (** {1:constructors Constructors} *)

  val make :
    exec:(Program.t -> Buffer.t list -> int list -> unit) ->
    synchronize:(unit -> unit) ->
    t
  (** [make ~exec ~synchronize] is a queue with the given dispatch and
      synchronisation functions. *)

  (** {1:operations Operations} *)

  val exec : t -> Program.t -> Buffer.t list -> int list -> unit
  (** [exec q program bufs var_args] dispatches [program] on [q]. *)

  val synchronize : t -> unit
  (** [synchronize q] blocks until all work queued on [q] completes. *)
end

(** {1:compiler Compiler} *)

(** Kernel compiler and renderer pairing.

    A {!Compiler.t} turns rendered source code into a compiled binary. A {!pair}
    associates a {!Renderer.t} with an optional compiler and an optional
    environment variable control. A {!set} collects available pairs and provides
    a global override variable.

    The active pair is chosen by {!Device.compile_program} at compile time:
    explicit environment override takes priority, then forced pairs
    ([ctrl = 1]), then the first non-disabled pair. *)
module Compiler : sig
  (** {1:types Types} *)

  type t = {
    name : string;  (** Compiler name (e.g., ["clang"], ["nvcc"]). *)
    compile : string -> bytes;
        (** [compile src] compiles source code [src] and returns the binary.
            Raises {!Compile_error} on failure. *)
  }
  (** The type for kernel compilers. *)

  exception Compile_error of string
  (** Raised by {!field-compile} when compilation fails. The payload is a
      human-readable error message. *)

  val make : name:string -> compile:(string -> bytes) -> t
  (** [make ~name ~compile] is a compiler with the given name and compilation
      function. *)

  type pair = {
    renderer : Renderer.t;  (** The renderer that produces source code. *)
    compiler : t option;
        (** The compiler that turns source into binary, or [None] for
            renderer-only backends (e.g., interpreter). *)
    ctrl : int Context.var option;
        (** Environment variable controlling this pair. [1] forces selection;
            [0] disables it. [None] means always eligible. *)
  }
  (** A renderer/compiler pair with optional environment control. *)

  type set = {
    pairs : pair list;
        (** Available renderer/compiler pairs, in priority order. *)
    ctrl : string Context.var option;
        (** Global override: when set, its value is matched against compiler
            names (case-insensitive) to select a pair directly. *)
  }
  (** A set of renderer/compiler pairs for a device. *)
end

(** {1:device_operations Device operations} *)

val make :
  name:string ->
  allocator:Allocator.packed ->
  compiler_set:Compiler.set ->
  queue:Queue.t ->
  prepare_program:(Program.t -> unit) ->
  t
(** [make ~name ~allocator ~compiler_set ~queue ~prepare_program] is a device
    runtime. [prepare_program] is called once on each freshly compiled or cloned
    {!Program.t} before it is cached and returned — backends use this hook to
    resolve entry addresses, load binaries, or perform device-specific setup. *)

val name : t -> string
(** [name d] is [d]'s device name. *)

val renderer : t -> Renderer.t
(** [renderer d] is the active renderer, selected from [d]'s compiler set. *)

val compile_program :
  t ->
  ?name:string ->
  ?estimates:Program_spec.Estimates.t ->
  Tolk_ir.Program.t ->
  Program.t
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

val queue : t -> Queue.t
(** [queue d] is [d]'s execution queue. *)

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
