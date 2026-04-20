(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Schedule execution and kernel dispatch.

    A {e runner} ({!Runner.t}) is the common dispatch interface for
    all executable operations: compiled kernels, buffer copies, and
    views. {!Compiled_runner} compiles kernel programs and creates
    runners. {!Exec_item} pairs a runner with its buffers for
    scheduled execution. {!run_schedule} executes a list of items
    in order.

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

(** {1:view_op View operation} *)

val view_op : device:Device.t -> Device.Buffer.t -> Runner.t
(** [view_op ~device buf] is a runner that asserts [dst] and [src]
    share the same base buffer. No data is copied.

    Raises [Invalid_argument] if the buffers do not share a base
    or if the argument list does not contain exactly two buffers. *)

(** {1:buffer_copy Buffer copy} *)

val buffer_copy :
  device:Device.t ->
  total_sz:int ->
  dest_device:string ->
  src_device:string ->
  Runner.t
(** [buffer_copy ~device ~total_sz ~dest_device ~src_device] is a
    runner that copies data between buffers via a host-memory
    bounce. [dest_device] and [src_device] are device names used
    in the display string.

    Raises [Invalid_argument] if the two buffers differ in size or
    dtype, or if the argument list does not contain exactly two
    buffers. *)

(** {1:method_cache Method cache} *)

val get_runner :
  device:Device.t ->
  get_program:(Tolk_ir.Kernel.t -> Program_spec.t) ->
  Tolk_ir.Kernel.t ->
  Compiled_runner.t
(** [get_runner ~device ~get_program ast] is a compiled runner for
    [ast] on [device]. Returns a cached runner when available;
    on a miss, calls [get_program ast] to compile the kernel and
    caches the result. A base-device entry is shared across
    device instances with the same compiler and renderer. *)

(** {1:exec_item Execution items} *)

(** A scheduled execution step.

    An exec item pairs an AST reference with buffer arguments and
    fixed variable bindings. {!lower} resolves the AST to a
    {!Runner.t}; {!run} dispatches it. *)
module Exec_item : sig
  type t
  (** The type for execution items. *)

  val make :
    ast:Tolk_ir.Tensor.t ->
    bufs:Device.Buffer.t option list ->
    ?var_vals:(string * int) list ->
    ?prg:Runner.t ->
    unit ->
    t
  (** [make ~ast ~bufs ?var_vals ?prg ()] is an exec item.

      [ast] is the tensor graph node that describes the operation
      (kernel SINK, BUFFER_VIEW, COPY, etc.).

      [var_vals] defaults to [[]]. [prg] defaults to [None]. *)

  val ast : t -> Tolk_ir.Tensor.t
  (** [ast t] is the tensor graph node. *)

  val bufs : t -> Device.Buffer.t option list
  (** [bufs t] is the buffer argument list. *)

  val var_vals : t -> (string * int) list
  (** [var_vals t] is the fixed variable bindings. *)

  val lower :
    device:Device.t ->
    get_program:(Tolk_ir.Kernel.t -> Program_spec.t) ->
    t -> t
  (** [lower ~device ~get_program t] resolves [t]'s AST to a
      runner if not already set. Kernel SINKs are compiled via
      {!get_runner}; BUFFER_VIEWs become {!view_op}; COPYs become
      {!buffer_copy}.

      Returns [t] unchanged if the runner is already set. *)

  val run :
    t ->
    ?var_vals:(string * int) list ->
    ?wait:bool ->
    ?do_update_stats:bool ->
    unit ->
    float option
  (** [run t ?var_vals ?wait ?do_update_stats ()] dispatches [t]'s
      runner. Variable bindings are [t]'s fixed bindings merged
      with [var_vals]. [None] buffer slots are skipped; remaining
      buffers are allocated if needed.

      [var_vals] defaults to [[]]. [wait] defaults to [false]
      (forced to [true] when [DEBUG >= 2]). [do_update_stats]
      defaults to [true].

      Raises [Invalid_argument] if the runner has not been set. *)
end

(** {1:run_schedule Schedule execution} *)

val run_schedule :
  device:Device.t ->
  get_program:(Tolk_ir.Kernel.t -> Program_spec.t) ->
  Exec_item.t list ->
  ?var_vals:(string * int) list ->
  ?do_update_stats:bool ->
  unit ->
  unit
(** [run_schedule ~device ~get_program items ?var_vals
    ?do_update_stats ()] lowers and executes each item in order.

    [var_vals] defaults to [[]]. [do_update_stats] defaults to
    [true]. *)
