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
    call's buffer arguments through a {!Buffers.t} binding.

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
    holds, otherwise it falls back to a host-memory bounce. [dest_device]
    and [src_device] are device names used in the display string.

    Raises [Invalid_argument] if the two buffers differ in size or
    dtype, or if the argument list does not contain exactly two
    buffers. *)

(** {1:compile Kernel compilation} *)

val program_config : unit -> string
(** [program_config ()] renders the current values of the settings that change
    the program compiled from a fixed kernel: [NOOPT], [TC],
    [IMAGE], [DISABLE_FAST_IDIV], [TRANSCENDENTAL], [ALLOW_TF32], and the
    default float and int dtypes. Two compilations of one kernel on one device
    are interchangeable exactly when their configurations are equal, so any
    cache of compiled programs must key on it. *)

val pm_compile :
  device:Device.t ->
  ?beam:int ->
  to_program:(Tolk_uop.Uop.t -> Tolk_uop.Uop.t) ->
  Tolk_uop.Uop.t ->
  Tolk_uop.Uop.t
(** [pm_compile ~device ?beam ~to_program linear] rewrites every kernel
    {!Tolk_uop.Ops.Call} in [linear] whose body is a {!Tolk_uop.Ops.Sink}
    into a call whose body is the compiled {!Tolk_uop.Ops.Program} returned by
    [to_program]. {!Tolk_uop.Ops.Copy} calls are left unchanged.

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

(** Resolves graph-owned buffers and caller-supplied bindings.

    A placed {!Tolk_uop.Ops.Buffer} retains its own storage across execution
    contexts. Seeding explicitly overrides that storage for the binding.
    Unowned nodes require an explicit binding; execution never allocates owners. *)
module Buffers : sig
  type t
  (** The type for buffer bindings. *)

  val create : unit -> t
  (** [create ()] is an empty set of caller-supplied buffer bindings. *)

  val seed : t -> Tolk_uop.Uop.t -> Device.Buffer.t -> unit
  (** [seed t node buf] binds [node] to [buf], overriding lazy allocation. *)

  val seed_multi : t -> Tolk_uop.Uop.t -> Device.Multi_buffer.t -> unit
  (** [seed_multi t node mbuf] binds [node] to the multi-device buffer [mbuf],
      overriding lazy allocation. *)

  val remove : t -> Tolk_uop.Uop.t -> unit
  (** [remove t node] drops [node]'s binding, if any. *)

  val mem : t -> Tolk_uop.Uop.t -> bool
  (** [mem t node] is [true] iff [node] is bound. *)

  val find_buffer : t -> Tolk_uop.Uop.t -> buffer option
  (** [find_buffer t node] is the buffer bound to [node], if any. *)

  val find_opt : t -> Tolk_uop.Uop.t -> Device.Buffer.t option
  (** [find_opt t node] is the single-device buffer bound to [node], if any.

      Raises [Invalid_argument] if [node] is bound to a multi-device
      buffer. *)

  val buffer_of_node : t -> Tolk_uop.Uop.t -> buffer
  (** [buffer_of_node t node] is the buffer backing the {!Tolk_uop.Ops.Buffer}
      [node], taken from an explicit binding or the node's storage owner.

      @raise Invalid_argument if neither exists. *)

  val of_buffer_node : t -> Tolk_uop.Uop.t -> Device.Buffer.t
  (** [of_buffer_node t node] is {!buffer_of_node} for a node backed by a
      single-device buffer.

      Raises [Invalid_argument] if [node] is backed by a multi-device
      buffer. *)

  val iter : t -> (buffer -> unit) -> unit
  (** [iter t f] applies [f] to every bound buffer. *)

  val clear : t -> unit
  (** [clear t] drops all bindings. *)
end

type exec_context = {
  var_vals : (string * int) list;
  input_uops : Tolk_uop.Uop.t array;
  update_stats : bool;
  jit : bool;
  wait : bool;
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
  unit ->
  exec_context
(** [exec_context ?var_vals ?input_uops ?update_stats ?jit ?wait ()] builds a
    context. [update_stats] defaults to [true]; the other fields default to
    empty or [false]. *)

val resolve_buffer : Buffers.t -> exec_context -> Tolk_uop.Uop.t -> buffer
(** [resolve_buffer binding ctx node] is the concrete buffer named by call
    argument [node]: a {!Tolk_uop.Ops.Param} resolves through
    [ctx.input_uops]; contiguous movement and bitcast views alias their
    resolved storage at the byte offset from {!Tolk_uop.Uop.contiguous_view} (per underlying device when the source is multi-device);
    a {!Tolk_uop.Ops.Buffer} is resolved through [binding]; a
    {!Tolk_uop.Ops.Mselect} indexes one shard of its multi-device source; a
    {!Tolk_uop.Ops.Mstack} joins its per-device sources into a multi-device
    buffer.

    Raises [Invalid_argument] on an unbound parameter, a symbolic slice offset,
    or a node that does not name a buffer. *)

val resolve : Buffers.t -> exec_context -> Tolk_uop.Uop.t -> Device.Buffer.t
(** [resolve binding ctx node] is {!resolve_buffer} for a node that names a
    single-device buffer.

    Raises [Invalid_argument] if [node] names a multi-device buffer, and in
    the {!resolve_buffer} failure cases. *)

(** {1:run_linear Linear execution} *)

val run_linear :
  device:Device.t ->
  to_program:(Tolk_uop.Uop.t -> Tolk_uop.Uop.t) ->
  Buffers.t ->
  ?var_vals:(string * int) list ->
  ?input_uops:Tolk_uop.Uop.t array ->
  ?update_stats:bool ->
  ?jit:bool ->
  ?wait:bool ->
  Tolk_uop.Uop.t ->
  unit
(** [run_linear ~device ~to_program binding ?var_vals ?input_uops ?update_stats
    ?jit ?wait linear] executes each {!Tolk_uop.Ops.Call} in the {!Tolk_uop.Ops.Linear}
    [linear] in order.

    When [jit] is [false] (default), [linear] is first compiled with
    {!pm_compile}, turning each kernel {!Tolk_uop.Ops.Sink} body into a
    {!Tolk_uop.Ops.Program}; when [jit] is [true], [linear] is assumed already
    compiled. Each call is then dispatched on its body: a
    {!Tolk_uop.Ops.Program} is launched with launch dimensions and scalar
    arguments read from its {!Tolk_uop.Uop.program_info} and a device handle
    built from its compiled binary; a {!Tolk_uop.Ops.Copy} transfers between its
    resolved buffers; a {!Tolk_uop.Ops.Custom_function} named ["graph"] records
    its LINEAR body into the device's {!Device.Graph} on first execution and
    replays that graph afterwards, patching per run every buffer argument
    whose resolution changed (arguments reaching a {!Tolk_uop.Ops.Param} slot
    or a binding reseeded through {!Buffers.seed}), variable values, and
    symbolic launch dimensions. Buffer arguments are resolved with
    {!resolve_buffer}, so {!Tolk_uop.Ops.Param} slots index into
    [input_uops].

    A call whose arguments resolve to multi-device buffers executes once per
    device position: a kernel launches its one compiled program on each
    device with the device index bound to the [_device_num] variable, and a
    copy transfers each per-device pair (natively when the devices share a
    backend, through a host bounce otherwise).

    Unless [update_stats] is [false], every dispatched kernel, view, copy and
    batched graph is counted in {!Helpers.Global_counters} with its estimated
    operations and memory traffic. When [DEBUG >= 2] each also prints one line
    on standard error: device, running call count, name, argument count, device
    memory in use, and its time over the running total with the rates the
    estimates give. A call that measured no time of its own is timed by
    synchronizing the device after it. The header is magenta under [jit] and
    green the first time a program runs.

    [wait] is forced to [true] when [DEBUG >= 2]. *)

val graph_launches : int ref
(** [graph_launches] counts batched graph launches dispatched through
    {!Device.Graph} execs, including each graph's recording launch. A
    cumulative observability counter for tests and debugging. *)

val graph_runners : unit -> int
(** [graph_runners ()] is the number of recorded graphs whose graph call is
    still reachable. A recorded graph keeps the buffers it addresses alive, and
    is dropped with the last linear that mentions it. An observability hook for
    tests and debugging. *)
