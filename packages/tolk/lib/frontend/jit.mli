(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Capture-and-replay JIT for tensor functions.

    A {!t} wraps a function from input tensors (and per-call symbolic
    variable bindings) to a result. The first call runs the function
    eagerly. The second call runs it once more while recording every kernel
    it realizes, then compiles the record into a replayable program. Every
    later call skips the function body entirely — no graph construction, no
    scheduling, no compilation — and re-executes the compiled kernels on the
    current inputs and variable values.

    {2 Contract}

    {ul
    {- {e Realize the outputs.} The wrapped function must realize everything
       it wants computed (see {!Run.realize}) before returning: on replay
       only the recorded kernels run, so work left lazy at capture never
       executes again.}
    {- {e Same signature every call.} Inputs must keep their element count,
       dtype, and device across calls, and the same variables must be bound
       in the same order. {!Jit_error} is raised on mismatch.}
    {- {e Everything that varies flows through arguments.} Values that change
       between calls must enter either as input tensor data or as [vars]
       values; anything else the function reads is frozen into the capture.
       Tensors the function closes over (weights, caches) keep their storage
       across calls, and in-place assignments to them replay against that
       same storage.}
    {- {e Buffer lifetime.} Buffers read after a call — outputs, and any
       buffer backing a tensor still reachable by the program — keep their
       own allocation. Other intermediates are folded into arena memory that
       is reused within a call and does not survive it.}} *)

type 'a t
(** A JIT-wrapped tensor function. *)

exception Jit_error of string
(** Raised on misuse: unrealizable or duplicate inputs, malformed [vars], an
    input signature mismatch on replay, or a capture that recorded no
    kernels. *)

val create : (Tensor.t array -> vars:Tolk_uop.Uop.t array -> 'a) -> 'a t
(** [create fxn] wraps [fxn] for capture and replay. [fxn] receives the
    input tensors and the [vars] array of the current call unchanged; it
    should build views from the [vars] bind nodes (for example with
    {!Movement.symbolic_shrink}) so that one captured program serves every
    bound value. *)

val call : ?vars:Tolk_uop.Uop.t array -> 'a t -> Tensor.t array -> 'a
(** [call ?vars t tensors] runs [t] on [tensors].

    Each element of [vars] must be a {!Tolk_uop.Uop.bind} of a named
    {!Tolk_uop.Uop.variable} to an integer constant; the bound values are
    passed to the replayed kernels, so variable names and bounds must not
    change across calls. [vars] defaults to no variables.

    Unrealized input tensors are realized first. Each input must then be
    backed by its own buffer; on replay the current buffers are substituted
    for the captured ones, and the result is the value returned at capture,
    whose tensors now read from the freshly computed buffers.

    @raise Jit_error on invalid inputs or an input mismatch with the
    capture. *)

val captured : 'a t -> bool
(** [captured t] is [true] once [t] has recorded and compiled its program,
    i.e. after the second call. *)

val reset : 'a t -> unit
(** [reset t] discards the captured program; the next two calls warm up and
    capture again. *)
