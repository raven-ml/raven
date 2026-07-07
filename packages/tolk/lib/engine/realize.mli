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

(** {1:local_size Local size optimization} *)

val optimize_local_size :
  device:Device.t ->
  Device.prog ->
  int array ->
  Device.Buffer.t list ->
  int array
(** [optimize_local_size ~device prg global_size rawbufs] finds the
    local workgroup size that minimises execution time for [prg]
    with [global_size].

    Enumerates all valid local sizes (each dimension drawn from
    powers of two up to [1024], total product at most [1024]),
    tries each twice in random order, and returns the fastest.

    When the first buffer in [rawbufs] also appears later in the
    list, a temporary buffer is allocated to avoid clobbering
    output during measurement.

    Raises [Invalid_argument] if every candidate fails. *)

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

val pm_compile :
  device:Device.t ->
  to_program:(Tolk_uop.Uop.t -> Tolk_uop.Uop.t) ->
  Tolk_uop.Uop.t ->
  Tolk_uop.Uop.t
(** [pm_compile ~device ~to_program linear] rewrites every kernel
    {!Tolk_uop.Ops.Call} in [linear] whose body is a {!Tolk_uop.Ops.Sink}
    into a call whose body is the compiled {!Tolk_uop.Ops.Program} returned by
    [to_program]. {!Tolk_uop.Ops.Slice} and {!Tolk_uop.Ops.Copy} calls are left
    unchanged.

    Compiled programs are cached by the kernel's semantic key and the device,
    so kernels that differ only by diagnostic tags share one compilation. *)

(** {1:capture Capture registry} *)

val capturing : (Tolk_uop.Uop.t -> (string * int) list -> unit) list ref
(** [capturing] is the schedule-capture registry. While non-empty,
    {!Schedule.create_linear_with_vars} hands each linearized schedule and its
    variable bindings to the head entry and returns an empty
    {!Tolk_uop.Ops.Linear} instead of planning the schedule for execution.
    {!Jit.call} installs its capturer here for the duration of the capture
    run. *)

(** {1:binding Buffer binding} *)

(** Maps buffer UOps to concrete device buffers.

    A {!Tolk_uop.Ops.Buffer} node is backed by a fresh device allocation the
    first time it is resolved, then cached by node identity; seeding a node
    with {!seed} overrides that allocation. *)
module Buffers : sig
  type t
  (** The type for buffer bindings. *)

  val create : device:Device.t -> t
  (** [create ~device] is an empty binding allocating on [device]. *)

  val seed : t -> Tolk_uop.Uop.t -> Device.Buffer.t -> unit
  (** [seed t node buf] binds [node] to [buf], overriding lazy allocation. *)

  val remove : t -> Tolk_uop.Uop.t -> unit
  (** [remove t node] drops [node]'s binding, if any. *)

  val mem : t -> Tolk_uop.Uop.t -> bool
  (** [mem t node] is [true] iff [node] is bound. *)

  val find_opt : t -> Tolk_uop.Uop.t -> Device.Buffer.t option
  (** [find_opt t node] is the buffer bound to [node], if any. *)

  val of_buffer_node : t -> Tolk_uop.Uop.t -> Device.Buffer.t
  (** [of_buffer_node t node] is the buffer backing the {!Tolk_uop.Ops.Buffer}
      [node], allocating and caching it on first use. *)

  val iter : t -> (Device.Buffer.t -> unit) -> unit
  (** [iter t f] applies [f] to every bound buffer. *)

  val clear : t -> unit
  (** [clear t] drops all bindings. *)
end

type exec_context = {
  var_vals : (string * int) list;
  input_uops : Tolk_uop.Uop.t array;
  jit : bool;
  wait : bool;
}
(** Execution context threaded through a LINEAR run: symbolic variable values,
    the input buffer nodes that {!Tolk_uop.Ops.Param} slots index into, and the
    JIT and wait flags. *)

val exec_context :
  ?var_vals:(string * int) list ->
  ?input_uops:Tolk_uop.Uop.t array ->
  ?jit:bool ->
  ?wait:bool ->
  unit ->
  exec_context
(** [exec_context ?var_vals ?input_uops ?jit ?wait ()] builds a context. All
    fields default to empty or [false]. *)

val resolve : Buffers.t -> exec_context -> Tolk_uop.Uop.t -> Device.Buffer.t
(** [resolve binding ctx node] is the concrete buffer named by call argument
    [node]: a {!Tolk_uop.Ops.Param} resolves through [ctx.input_uops]; a
    {!Tolk_uop.Ops.Slice} is an offset view of its resolved source; a
    {!Tolk_uop.Ops.Buffer} is resolved through [binding].

    Raises [Invalid_argument] on an unbound parameter, a symbolic slice offset,
    or a node that does not name a buffer. *)

(** {1:run_linear Linear execution} *)

val run_linear :
  device:Device.t ->
  to_program:(Tolk_uop.Uop.t -> Tolk_uop.Uop.t) ->
  Buffers.t ->
  ?var_vals:(string * int) list ->
  ?input_uops:Tolk_uop.Uop.t array ->
  ?jit:bool ->
  ?wait:bool ->
  Tolk_uop.Uop.t ->
  unit
(** [run_linear ~device ~to_program binding ?var_vals ?input_uops ?jit ?wait
    linear] executes each {!Tolk_uop.Ops.Call} in the {!Tolk_uop.Ops.Linear}
    [linear] in order.

    When [jit] is [false] (default), [linear] is first compiled with
    {!pm_compile}, turning each kernel {!Tolk_uop.Ops.Sink} body into a
    {!Tolk_uop.Ops.Program}; when [jit] is [true], [linear] is assumed already
    compiled. Each call is then dispatched on its body: a
    {!Tolk_uop.Ops.Program} is launched with launch dimensions and scalar
    arguments read from its {!Tolk_uop.Uop.program_info} and a device handle
    built from its compiled binary; a {!Tolk_uop.Ops.Slice} binds a view of its
    resolved source into [binding]; a {!Tolk_uop.Ops.Copy} transfers between its
    resolved buffers; a {!Tolk_uop.Ops.Custom_function} named ["graph"] records
    its LINEAR body into the device's {!Device.Graph} on first execution and
    replays that graph afterwards, patching per run every buffer argument
    whose resolution changed (arguments reaching a {!Tolk_uop.Ops.Param} slot
    or a binding reseeded through {!Buffers.seed}), variable values, and
    symbolic launch dimensions. Buffer arguments are resolved with {!resolve},
    so {!Tolk_uop.Ops.Param} slots index into [input_uops].

    [wait] is forced to [true] when [DEBUG >= 2]. *)

val graph_launches : int ref
(** [graph_launches] counts batched graph launches dispatched through
    {!Device.Graph} execs, including each graph's recording launch. A
    cumulative observability counter for tests and debugging. *)
