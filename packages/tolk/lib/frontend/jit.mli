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
    {- {e Same signature every call.} Inputs must keep their normalized
       movement views, symbolic variable declarations, dtype, and device
       across calls. Equivalent compositions of movement operations are
       accepted. Bound variable values may vary; the order of explicit
       [vars] does not matter. {!Jit_error} is raised on mismatch.}
    {- {e Everything that varies flows through arguments.} Values that change
       between calls must enter as input tensor data, symbolic bindings in
       input views, or [vars] values; anything else the function reads is frozen into the capture.
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

val create :
  outputs:('a -> Tensor.t list) ->
  (Tensor.t array -> vars:Tolk_uop.Uop.t array -> 'a) -> 'a t
(** [create ~outputs fxn] wraps [fxn] for capture and replay.
    [outputs] enumerates the tensors in the returned value, including tensors
    nested in records or containers. For a single tensor result, pass
    [(fun tensor -> [tensor])]. Their symbolic views are rebound to the current
    input-view bindings and [vars] after replay; variables bound only inside
    [fxn] keep their captured values.

    [fxn] receives the
    input tensors and the [vars] array of the current call unchanged; it
    should build views from the [vars] bind nodes (for example with
    {!Movement.symbolic_shrink}) so that one captured program serves every
    bound value. *)

val call : ?vars:Tolk_uop.Uop.t array -> 'a t -> Tensor.t array -> 'a
(** [call ?vars t tensors] runs [t] on [tensors].

    Each element of [vars] must be a {!Tolk_uop.Uop.bind} of a named
    {!Tolk_uop.Uop.variable} to an integer constant; the bound values are
    passed to the replayed kernels. Bindings in symbolic input views are
    extracted automatically and merged with [vars]; conflicting values for
    the same name raise {!Jit_error}. Input-view variable names and bounds
    must not change across calls. [vars] defaults to no explicit bindings.

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
