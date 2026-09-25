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
module Buffer_spec = Tolk_uop.Storage.Buffer_spec

(** {1:allocator Allocator} *)

(** Backend allocator interface.

    An allocator manages device buffer lifecycle: allocation, data transfer,
    addressing, and optional features such as offset views and device-to-device
    copies. The buffer type ['buf] is backend-specific and hidden behind
    {!packed} at the device level.

    See {!Lru_allocator} for LRU caching on top of a raw allocator. *)
module Allocator = Tolk_uop.Storage.Allocator

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
module Buffer = Tolk_uop.Storage

(** {1:prog Runtime program handle} *)

type prog = {
  call :
    Buffer.t array -> global:int array -> local:int array option ->
    vals:int64 array -> wait:bool -> timeout:int option -> float option;
  free : unit -> unit;
  handle : nativeint;
      (** Backend kernel handle retained by linked command storage. [0n] when the
          backend has no addressable kernel object. *)
}
(** A device-specific dispatch handle. *)

type runtime = Tolk_uop.Tiny_elf.t -> prog
(** [runtime obj] creates a dispatch handle for [obj]. Scalar arguments use
    the order declared by its signature. *)

type queue = {
  host : string; (** Host device executing submission programs. *)
  copy : Tolk_uop.Uop.t -> bool; (** [copy call] whether this bulk-store call can use the device queue. *)
  encode : Tolk_uop.Uop.t -> Tolk_uop.Uop.t option;
      (** Rewrites a device submit function into host operations. *)
  lower : Tolk_uop.Uop.t -> Tolk_uop.Uop.t option;
      (** Lowers device-specific host accesses, such as timeline polling. *)
  compile : Tolk_uop.Uop.t -> Tolk_uop.Uop.t;
      (** Compiles the final host sink to a PROGRAM. *)
}
(** Device hooks for compiling queue submission through the shared UOp protocol. *)

(** {1:renderer_set Renderer selection} *)

(** Available renderers for a device. Each renderer carries its compiler.
    [DEV] selects a renderer by name, or initialization tries the factories
    in priority order. Successful selections are cached by target. *)
module Renderer_set : sig
  type t
  (** The type for renderer sets. *)

  val make :
    ?arch:string -> device:string ->
    (string * (Tolk_uop.Target.t -> Renderer.t)) list -> t
  (** [make ~device entries] lists named renderer factories for [device].
      Names are uppercase, for example ["CLANG"] or ["CUDA"]. [arch] supplies
      the detected architecture when [DEV] does not specify one.

      Factories run on first use with the resolved target. Failed factories
      fall through to the next candidate; an explicit renderer selection
      restricts candidates to that name. Deprecated renderer environment
      switches raise [Invalid_argument] with a [DEV] replacement. *)
end

(** {1:device_operations Device operations} *)

val make :
  name:string ->
  allocator:Allocator.packed ->
  renderer_set:Renderer_set.t ->
  runtime:runtime ->
  synchronize:(unit -> unit) ->
  ?invalidate_caches:(unit -> unit) ->
  ?queue:queue ->
  ?bufferize:(Tolk_uop.Uop.t -> Buffer.t option) ->
  unit ->
  t
(** [make ~name ~allocator ~renderer_set ~runtime ~synchronize
    ?invalidate_caches ?queue ?bufferize ()] is a device runtime, registered under its
    canonical [name] for graph-owned buffers to resolve their allocator.

    [runtime obj] loads a compiled binary and returns a dispatch handle.

    [synchronize ()] blocks until all pending work on the device completes.

    [queue] supplies host compilation hooks. [bufferize] resolves backend
    allocation descriptors during linking, returning [None] for generic storage. *)

val name : t -> string
(** [name d] is [d]'s device name. *)

val renderer : t -> Renderer.t
(** [renderer d] is the active renderer. *)

val runtime : t -> runtime
(** [runtime d obj] loads [obj] on [d]. Its dispatch handle checks buffer and
    scalar argument counts before entering the backend.

    Raises [Invalid_argument] if signature slots are not a permutation of
    buffers followed by scalars. *)

val synchronize : t -> unit
(** [synchronize d] blocks until all pending work on [d] completes. *)

val queue : t -> queue option
(** [queue d] is [d]'s compiled submission capability, if any. *)

val bufferize : t -> Tolk_uop.Uop.t -> Buffer.t option
(** [bufferize d placeholder] resolves a backend storage descriptor at link time. *)

val compile_program :
  t ->
  ?name:string ->
  ?applied_opts:Tolk_uop.Uop.Opt.t list ->
  ?estimates:Program_spec.Estimates.t ->
  Program_spec.program ->
  Program_spec.t
(** [compile_program d ?name ?applied_opts ?estimates program] renders and
    compiles [program] for [d], returning its {!Program_spec.t} description.
    Parameter declarations are grouped before the body, with buffers before
    scalars, preserving their relative order within each group.

    Compiled bytes use {!Compiler.compile_cached}, keyed by the actual source
    and the compiler's cache key. Device, optimization and execution metadata
    are built from this call's arguments.

    [name] defaults to ["kern"], [applied_opts] to [[]], and [estimates] to
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
