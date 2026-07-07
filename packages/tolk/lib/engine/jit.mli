(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** JIT compilation and replay.

    A {e JIT} ({!type:tiny_jit}) wraps a function and transparently captures
    its computation as a {!Tolk_uop.Ops.Linear} on the second call, then
    replays that linear on all subsequent calls. Three phases:

    {ul
    {- {e Warmup} (cnt=0): execute eagerly.}
    {- {e Capture} (cnt=1): install a capturer in {!Realize.capturing} so
       every schedule the function creates is recorded instead of executed,
       then lower the combined record for replay: substitute each input
       buffer node with a slotted {!Tolk_uop.Ops.Param}, plan intermediate
       buffer memory once over the combined linear, compile every kernel,
       and, when the device has a {!Device.Graph} capability, batch
       consecutive compatible calls into graph calls that replay as a single
       dispatch each ([JIT_BATCH_SIZE] caps the first batch and the cap
       doubles per batch; [JIT >= 2] disables batching).}
    {- {e Exec} (cnt>=2): validate the inputs against the capture and replay
       the compiled linear through {!Realize.run_linear}, passing the current
       input buffer nodes as [input_uops] and the per-call variable values as
       [var_vals].}}

    Non-input buffers (weights, outputs, held buffers) are bound once at
    capture through the caller's buffer resolver; planned intermediates live
    in arena buffers owned by the capture's persistent binding. *)

(** {1:exceptions Exceptions} *)

exception Jit_error of string
(** Raised for JIT-specific errors: nested capture, empty capture, or an
    input mismatch on replay. *)

(** {1:captured Captured schedule} *)

type 'a captured_jit
(** A compiled linear together with the buffer binding it replays through. *)

(** {1:tiny_jit TinyJit} *)

type 'a tiny_jit
(** The JIT wrapper. *)

val captured : 'a tiny_jit -> 'a captured_jit option
(** [captured t] is [t]'s captured schedule, or [None] before capture. *)

val create :
  device:Device.t ->
  to_program:(Tolk_uop.Uop.t -> Tolk_uop.Uop.t) ->
  ?fxn:(Tolk_uop.Uop.t array -> (string * int) list -> 'a) ->
  ?captured:'a captured_jit ->
  ?prune:bool ->
  unit ->
  'a tiny_jit
(** [create ~device ~to_program ?fxn ?captured ?prune ()] is a JIT wrapper.

    [to_program] compiles a kernel {!Tolk_uop.Ops.Sink} into an on-graph
    {!Tolk_uop.Ops.Program}; the captured linear is compiled with it once,
    then replayed.

    Provide either [fxn] (the function to JIT, taking the input buffer nodes
    and the variable values of the call) or [captured] (a pre-captured
    schedule); when [captured] is given, execution starts at the replay
    phase. [prune] is accepted for compatibility and currently ignored.

    Raises [Invalid_argument] if neither [fxn] nor [captured] is provided. *)

val reset : 'a tiny_jit -> unit
(** [reset t] returns [t] to the warmup phase, discarding any captured
    schedule.

    Raises [Invalid_argument] if [t] was created without a function. *)

val call :
  ?wait:bool ->
  ?held_buffers:(unit -> Tolk_uop.Uop.t list) ->
  'a tiny_jit ->
  Tolk_uop.Uop.t array ->
  (string * int) list ->
  buffers:(Tolk_uop.Uop.t -> Device.Buffer.t option) ->
  'a
(** [call ?wait ?held_buffers t input_uops var_vals ~buffers] runs [t] with
    the input buffer nodes [input_uops] and variable values [var_vals].

    {ul
    {- {e Warmup} (cnt=0): calls the wrapped function eagerly.}
    {- {e Capture} (cnt=1): calls the function under the capture handler,
       lowers the combined recorded schedule for replay, binds the non-input
       buffers it references through [buffers], and replays.}
    {- {e Exec} (cnt>=2): validates inputs against the capture and replays
       with the current [input_uops] and [var_vals].}}

    [buffers] maps buffer nodes to their concrete device buffers. At capture
    it binds the non-input buffers of the recorded schedule; at every replay
    it resolves the current [input_uops], so each input node must have a
    backing buffer when [call] runs.

    [held_buffers], evaluated once when capture completes, lists the buffer
    nodes that outlive the jitted computation — external outputs, and any
    buffer assigned inside the jit but read outside it (for example a cache
    updated in place across calls). Held buffers keep their identity and
    their own allocation; all other intermediate buffers are folded into
    per-device arenas by the memory planner and their contents do not survive
    the call. Defaults to holding nothing.

    Replay validates each input's element count, dtype, and device against
    the capture and raises {!Jit_error} on mismatch. A fuller check would
    also compare each input's movement view and its set of bound symbolic
    variables, catching inputs that alias the same buffer through a different
    layout or binding; inputs here are whole buffer nodes, so those checks
    have nothing further to compare today. *)
