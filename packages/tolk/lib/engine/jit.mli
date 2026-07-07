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
    {- {e Capture} (cnt=1): record the linears the function schedules, combine
       them, and store the result as a {!captured_jit}.}
    {- {e Exec} (cnt>=2): validate inputs, re-seed fresh input buffers, and
       replay the captured linear through {!Realize.run_linear}.}}

    Buffer arguments are resolved through a persistent {!Realize.Buffers.t}
    binding, seeded once from the capture-time resolver and re-seeded with the
    current inputs on every call. *)

(** {1:exceptions Exceptions} *)

exception Jit_error of string
(** Raised for JIT-specific errors: nested capture, empty capture, or an input
    mismatch on replay. *)

(** {1:capture Capture handler} *)

val is_capturing : unit -> bool
(** [is_capturing ()] is [true] iff a {!type:tiny_jit} capture is in progress. *)

val add_linear : Tolk_uop.Uop.t -> unit
(** [add_linear linear] records [linear] into the active capture.

    Raises [Failure] if no capture is in progress. *)

(** {1:captured Captured schedule} *)

type 'a captured_jit
(** A captured linear together with the buffer binding it replays through. *)

(** {1:tiny_jit TinyJit} *)

type 'a tiny_jit
(** The JIT wrapper. *)

val captured : 'a tiny_jit -> 'a captured_jit option
(** [captured t] is [t]'s captured schedule, or [None] before capture. *)

val create :
  device:Device.t ->
  to_program:(Tolk_uop.Uop.t -> Tolk_uop.Uop.t) ->
  ?fxn:(Device.Buffer.t array -> (string * int) list -> 'a) ->
  ?captured:'a captured_jit ->
  ?prune:bool ->
  unit ->
  'a tiny_jit
(** [create ~device ~to_program ?fxn ?captured ?prune ()] is a JIT wrapper.

    [to_program] compiles a kernel {!Tolk_uop.Ops.Sink} into an on-graph
    {!Tolk_uop.Ops.Program}; the captured linear is compiled with it once, then
    replayed.

    Provide either [fxn] (the function to JIT) or [captured] (a pre-captured
    schedule); when [captured] is given, execution starts at the replay phase.
    [prune] is accepted for compatibility and currently ignored.

    Raises [Invalid_argument] if neither [fxn] nor [captured] is provided. *)

val reset : 'a tiny_jit -> unit
(** [reset t] returns [t] to the warmup phase, discarding any captured
    schedule.

    Raises [Invalid_argument] if [t] was created without a function. *)

val call :
  ?wait:bool ->
  ?held_buffers:(unit -> Device.Buffer.t list) ->
  'a tiny_jit ->
  Device.Buffer.t array ->
  (string * int) list ->
  buffers:(Tolk_uop.Uop.t -> Device.Buffer.t option) ->
  'a
(** [call ?wait ?held_buffers t input_bufs var_vals ~buffers] runs [t] with
    [input_bufs] and variable bindings [var_vals].

    {ul
    {- {e Warmup} (cnt=0): calls the wrapped function eagerly.}
    {- {e Capture} (cnt=1): calls the function under the capture handler,
       combines the recorded linears, seeds the binding from [buffers], and
       replays.}
    {- {e Exec} (cnt>=2): validates inputs against the capture and replays.}}

    [buffers] maps scheduled buffer nodes to device buffers; it is consulted
    only on the first replay, to seed the binding. [held_buffers] is accepted
    for compatibility and currently ignored. *)
