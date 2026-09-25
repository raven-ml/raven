(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Linear execution and kernel dispatch.

    A {e runner} ({!Runner.t}) is the common dispatch interface for
    executable operations: compiled kernels and buffer copies.
    {!Compiled_runner} compiles kernel programs and creates runners.
    {!run_linear} executes a {!Tolk_uop.Ops.Linear} node, resolving each
    call's buffer arguments through owned storage and parameter slots.

    See also {!Device.prog} for the low-level device dispatch
    handle. *)

(** {1:runners Runners} *)

(** Common dispatch interface.

    A runner wraps a single dispatchable operation (compiled kernel,
    buffer copy, view). Dispatch takes a list of buffers and
    name-keyed variable bindings and optionally returns execution
    time. *)
module Runner : sig
  type t
  (** The type for runners. *)

  val make :
    display_name:string ->
    device:Device.t ->
    ?estimates:Program_spec.Estimates.t ->
    (Device.Buffer.t list -> (string * int) list ->
     wait:bool -> timeout:int option -> float option) ->
    t
  (** [make ~display_name ~device ?estimates call] is a runner that
      dispatches via [call].

      [estimates] defaults to {!Program_spec.Estimates.zero}. *)

  val dev : t -> Device.t
  (** [dev t] is [t]'s device. *)

  val display_name : t -> string
  (** [display_name t] is [t]'s human-readable name for debug
      output. *)

  val estimates : t -> Program_spec.Estimates.t
  (** [estimates t] is [t]'s cost estimates. *)

  val call :
    t -> Device.Buffer.t list -> (string * int) list ->
    wait:bool -> timeout:int option -> float option
  (** [call t bufs var_vals ~wait ~timeout] dispatches the operation
      on [bufs] with variable bindings [var_vals].

      Returns [Some time] when [wait] is [true] and the backend
      supports timing, [None] otherwise. *)

  val exec :
    t -> Device.Buffer.t list -> ?var_vals:(string * int) list ->
    unit -> float option
  (** [exec t bufs ?var_vals ()] is {!call} with [~wait:false] and
      [~timeout:None]. Always returns [None].

      [var_vals] defaults to [[]]. *)
end

(** {1:compiled_runner Compiled runner} *)

(** Kernel compilation and dispatch.

    A compiled runner wraps a {!Program_spec.t}, compiles it if
    its {!Program_spec.lib} is [None], creates a {!Device.prog}
    handle via {!Device.runtime}, and dispatches kernels through
    it. *)
module Compiled_runner : sig
  type t
  (** The type for compiled runners. *)

  val create :
    device:Device.t ->
    ?prg:Device.prog ->
    Program_spec.t ->
    t
  (** [create ~device ?prg p] is a compiled runner for [p] on
      [device].

      When {!Program_spec.lib} [p] is [None], the source is
      compiled via the device's {!Renderer.compiler}.

      [prg] overrides the {!Device.prog} handle. When [None]
      (default), one is created via {!Device.runtime}.

      Raises [Invalid_argument] if the device has no compiler and
      [p] has no compiled binary. *)

  val p : t -> Program_spec.t
  (** [p t] is [t]'s program spec. *)

  val runner : t -> Runner.t
  (** [runner t] is [t]'s underlying runner. *)

  val call :
    t -> Device.Buffer.t list -> (string * int) list ->
    wait:bool -> timeout:int option -> float option
  (** [call t bufs var_vals ~wait ~timeout] dispatches the kernel
      on [bufs] with variable bindings [var_vals].

      See {!Runner.call} for the return value semantics. *)
end

(** {1:buffer_copy Buffer copy} *)

val buffer_copy :
  device:Device.t ->
  total_sz:int ->
  dest_device:string ->
  src_device:string ->
  Runner.t
(** [buffer_copy ~device ~total_sz ~dest_device ~src_device] is a
    runner that copies data between buffers. It uses the destination
    allocator's native transfer hook when {!Device.Buffer.supports_transfer}
    holds, otherwise it falls back to a host-memory bounce. Allocators with
    offset views stream large copies through 64 MiB chunks and synchronize
    each upload to bound native staging memory, preserving overlapping-view
    copy semantics. [dest_device]
    and [src_device] are device names used in the display string.

    Raises [Invalid_argument] if the two buffers differ in size or
    dtype, or if the argument list does not contain exactly two
    buffers. *)

(** {1:compile Kernel compilation} *)

val program_config : unit -> string
(** [program_config ()] renders the current values of the settings that change
    the program compiled from a fixed kernel: [NOOPT], [TC], [TC_SELECT],
    [TC_OPT], [IMAGE], [DISABLE_FAST_IDIV], [TRANSCENDENTAL], [ALLOW_TF32],
    [FLOAT16], the default float and int dtypes, the heuristic's [MV],
    [MV_BLOCKSIZE], [MV_THREADS_PER_ROW], [MV_ROWS_PER_THREAD] and
    [OCCUPANCY_FLOOR], memory coalescing's [DMC] and the startup value of
    [ALLOW_HALF8], and the C renderer's [EXPAND_SSA] and [ALIGNED]. Two
    compilations of one kernel on one renderer and compiler (whose cache key
    records the CPU's [CC]) are interchangeable exactly when their
    configurations are equal, so any cache of compiled programs must key on
    it. *)

val queue_config : ?profile:bool -> Device.t -> string
(** [queue_config ?profile d] renders the settings a queue compilation for [d]
    reads, as [KEY=value] pairs: [profile] (defaults to [DEBUG >= 2] or
    [PROFILE=1]), which adds queue timestamps, [ALL2ALL] and [HCQ_NUM_SDMA],
    which pick the copy queues, and [d]'s queue {!Device.queue.config}. Two
    compilations of one schedule for [d] are interchangeable when their configs
    are equal, so any cache of compiled schedules must key on it, with
    {!program_config} for its kernels. *)

val compile_linear :
  device:Device.t ->
  ?beam:int ->
  ?profile:bool ->
  to_program:(Device.t -> Tolk_uop.Uop.t -> Tolk_uop.Uop.t) ->
  Tolk_uop.Uop.t ->
  Tolk_uop.Uop.t
(** [compile_linear ~device ?beam ?profile ~to_program linear] rewrites every kernel
    {!Tolk_uop.Ops.Call} in [linear] whose body is a {!Tolk_uop.Ops.Sink}
    into a call whose body is the compiled {!Tolk_uop.Ops.Program} returned by
    [to_program execution_device sink]. The execution device comes from the
    call arguments, falling back to [device] for a kernel without placed
    arguments. The program cache uses that device and its selected renderer.
    {!Tolk_uop.Ops.Store} calls are left unchanged.

    [profile] adds queue timestamps and defaults to [true] when [DEBUG >= 2]
    or [PROFILE=1]. [wait] uses them for elapsed time; [PROFILE=1] also retains
    asynchronous timestamp records for {!Device.profile}.

    When [beam] is [b >= 1], every kernel sink that does not already carry a
    beam width (its {!Tolk_uop.Uop.kernel_info} has [beam = 0]) is stamped with [b]
    before compilation, enabling beam-search autotuning for it regardless of
    the [BEAM] environment variable. When omitted or [< 1], kernels compile
    under the ambient environment settings.

    Compiled programs are cached by the kernel's semantic key, the device, and
    {!program_config}, so kernels that differ only by diagnostic tags share one
    compilation; the stamped beam width is part of the key. *)

(** {1:capture Capture registry} *)

val capturing : (Tolk_uop.Uop.t -> (string * int) list -> unit) list ref
(** [capturing] is the schedule-capture registry. While non-empty,
    {!Schedule.create_linear_with_vars} hands each linearized schedule and its
    variable bindings to the head entry and returns an empty
    {!Tolk_uop.Ops.Linear} instead of planning the schedule for execution.
    {!Jit.call} installs its capturer here for the duration of the capture
    run. *)

(** {1:binding Buffer binding} *)

type buffer =
  | Single of Device.Buffer.t  (** A buffer on one device. *)
  | Multi of Device.Multi_buffer.t
      (** One buffer per device of a multi-device placement. *)
(** The type for concrete buffers named by call arguments. *)

type exec_context = {
  var_vals : (string * int) list;
  input_uops : Tolk_uop.Uop.t array;
  update_stats : bool;
  jit : bool;
  wait : bool;
  timeout : int option;
  cache : bool;
}
(** Execution context threaded through a LINEAR run: symbolic variable values,
    the input buffer nodes that {!Tolk_uop.Ops.Param} slots index into, whether
    calls are counted and reported, and the JIT and wait flags. *)

val exec_context :
  ?var_vals:(string * int) list ->
  ?input_uops:Tolk_uop.Uop.t array ->
  ?update_stats:bool ->
  ?jit:bool ->
  ?wait:bool ->
  ?timeout:int ->
  ?cache:bool ->
  unit ->
  exec_context
(** [exec_context ?var_vals ?input_uops ?update_stats ?jit ?wait ?timeout ?cache ()] builds a
    context. [update_stats] and [cache] default to [true]; the other fields
    default to empty or [false]. [timeout] is the device wait budget in
    milliseconds. [cache=false] releases each dispatch handle after its
    synchronous sample; failed drains retain the handle. *)

val resolve_buffer : exec_context -> Tolk_uop.Uop.t -> buffer
(** [resolve_buffer ctx node] is the concrete buffer named by call
    argument [node]: a {!Tolk_uop.Ops.Param} resolves through
    [ctx.input_uops]; contiguous movement and bitcast views alias their
    resolved storage at the byte offset from {!Tolk_uop.Uop.contiguous_view} (per underlying device when the source is multi-device);
    a {!Tolk_uop.Ops.Buffer} supplies its owned storage; a
    {!Tolk_uop.Ops.Mselect} indexes one shard of its multi-device source; a
    {!Tolk_uop.Ops.Mstack} joins its per-device sources into a multi-device
    buffer.

    Raises [Invalid_argument] on an unbound parameter, a symbolic slice offset,
    or a node that does not name a buffer. *)

val resolve : exec_context -> Tolk_uop.Uop.t -> Device.Buffer.t
(** [resolve ctx node] is {!resolve_buffer} for a node that names a
    single-device buffer.

    Raises [Invalid_argument] if [node] names a multi-device buffer, and in
    the {!resolve_buffer} failure cases. *)

val link_linear :
  ?ctx:exec_context -> ?allow_cache:bool ->
  Tolk_uop.Uop.t -> Tolk_uop.Uop.t
(** [link_linear ?ctx ?allow_cache linear] binds a compiled
    schedule's tagged storage placeholders and static address patches, retaining
    their storage in the returned schedule. Call bodies remain compiled and
    untagged parameters remain runtime-bound. See {!Link.run} for cache rules. *)

(** {1:run_linear Linear execution} *)

val run_linear :
  device:Device.t ->
  to_program:(Device.t -> Tolk_uop.Uop.t -> Tolk_uop.Uop.t) ->
  ?var_vals:(string * int) list ->
  ?input_uops:Tolk_uop.Uop.t array ->
  ?update_stats:bool ->
  ?jit:bool ->
  ?wait:bool ->
  Tolk_uop.Uop.t ->
  unit
(** [run_linear ~device ~to_program ?var_vals ?input_uops ?update_stats
    ?jit ?wait linear] executes each {!Tolk_uop.Ops.Call} in the {!Tolk_uop.Ops.Linear}
    [linear] in order.

    When [jit] is [false] (default), [linear] is first compiled with
    {!compile_linear}, turning each kernel {!Tolk_uop.Ops.Sink} body into a
    {!Tolk_uop.Ops.Program}, then linked with {!link_linear}; when [jit] is
    [true], [linear] is assumed already compiled and linked. Eager queue
    templates preserve buffer aliases and byte offsets without retaining input
    storage. Below [HCQ_CACHE_THRESH] calls (default [64]), linked command
    storage is reused with runtime address patches; larger schedules resolve
    their inputs at each link.

    Each call is then dispatched on its body: a
    {!Tolk_uop.Ops.Program} is launched with launch dimensions and scalar
    arguments read from its {!Tolk_uop.Uop.program_info} and a device handle
    built from its compiled binary. A program carrying queue metadata refreshes
    its address table and submits through its host device; [wait] also waits for
    the submitted devices. Before updating that table, replay rejects writable
    overlaps between unordered calls, including independently wrapped external
    pointers. FIFO order and transitive queue waits permit buffer donation;
    new read-only aliases and disjoint views are also allowed.
    Represent writable aliases through a shared root in the graph before
    compiling their queue dependencies. Unsupported copy imports prepare a
    cached schedule with two alternating 64 MiB host slots and queue fences
    before slot reuse. Staging memory belongs to that prepared schedule;
    runtime input buffers are not retained. If staging cannot be imported,
    the original calls execute in order. Preparation happens before the
    address table is updated or queue work is published. Allocation and device
    faults propagate. A
    {!Tolk_uop.Ops.Store} transfers between its
    resolved buffers. Buffer arguments are resolved with
    {!resolve_buffer}, so {!Tolk_uop.Ops.Param} slots index into
    [input_uops].

    A call whose arguments resolve to multi-device buffers executes once per
    device position: a kernel launches its one compiled program on each
    device with the device index bound to the [_device_num] variable, and a
    copy transfers each per-device pair (natively when the devices share a
    backend, through a host bounce otherwise).

    Unless [update_stats] is [false], every dispatched kernel, view, copy and
    queue submission is counted in {!Helpers.Global_counters} with its estimated
    operations and memory traffic. When [DEBUG >= 2] each also prints one line
    on standard error: device, running call count, name, argument count, device
    memory in use, and its time over the running total with the rates the
    estimates give. A call that measured no time of its own is timed by
    synchronizing the device after it. The header is magenta under [jit] and
    green the first time a program runs.

    [wait] is forced to [true] when [DEBUG >= 2]. *)

val queue_submissions : int ref
(** [queue_submissions] counts compiled host submissions dispatched through
    {!run_linear}. A cumulative observability counter for tests and debugging. *)


val time_call :
  device:Device.t ->
  to_program:(Device.t -> Tolk_uop.Uop.t -> Tolk_uop.Uop.t) ->
  ?var_vals:(string * int) list ->
  ?timeout:int ->
  ?clear_l2:bool ->
  Tolk_uop.Uop.t ->
  ((unit -> float) -> 'a) ->
  'a
(** [time_call ~device ~to_program ?var_vals ?timeout ?clear_l2 call f] compiles
    and links [call] with device timestamps, then calls [f sample]. Each
    [sample ()] executes the linked call synchronously and returns its longest
    host or device duration in seconds. [clear_l2] invalidates the device
    caches before each sample and defaults to [false]. [timeout] is forwarded
    to runtimes and recoverable device waits in milliseconds.

    Linked storage is shared between samples. Transient dispatch handles are
    released after successful draining, including when execution raises.
    [sample] must only be used within [f]; the device is drained when [f]
    returns or raises. Timing does not update execution statistics. *)
