(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Captured schedules.

    A captured schedule is every call a function makes, in order, over the
    buffers it was given. Lowering it once gives a schedule that runs again on
    other buffers: its inputs become parameters, which each run binds, its
    intermediates share a few arenas, and its kernels and batches are compiled.
    The engine links the result once and runs it on each call. *)

val jit_lower :
  ?beam:int ->
  ?search:(int -> Postrange.Scheduler.t -> Postrange.Scheduler.t) ->
  devices:(string -> Hcq2.device) ->
  held_bufs:Ops.t list ->
  inputs:Ops.t list ->
  Ops.t ->
  Ops.t
(** [jit_lower ~beam ~search ~devices ~held_bufs ~inputs linear] is the captured
    schedule [linear], an {!Op.Linear} of calls, ready to link:
    + the [i]th buffer of [inputs] is replaced wherever [linear] reaches it by
      the parameter of slot [i] ({!Ops.param}), of its type, device and size;
    + its buffers are placed in arenas ({!Memory.memory_plan_rewrite}), except
      [held_bufs], whose contents outlive a run, such as the buffers a caller
      keeps or that hold constants;
    + it is compiled ({!Hcq2.compile_linear}, with [search] and [devices]), with
      the beam width [beam], or else {!Helpers.jitbeam}'s, or else
      {!Helpers.beam}'s.

    Raises as {!Hcq2.compile_linear} does. *)
