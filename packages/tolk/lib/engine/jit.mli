(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Captured schedules.

    A captured schedule is every call a function makes, in order, over the
    buffers it was given. Lowering it once gives a schedule that runs again on
    other buffers: its inputs become parameters, which each run binds, its
    intermediates share a few arenas, and its kernels are compiled. The engine
    links the result once and runs it on each call. *)

val shaping : ?beam:int -> unit -> (string * string) list
(** [shaping ~beam ()] is {!Setting.shaping} as {!jit_lower} [~beam]
    compiles under it: with [BEAM] at the width the lowering searches, and
    [JITBEAM], which the lowering has read by then, unset
    ({!Setting.shaping_under}). Two lowerings whose [shaping] is equal compile
    alike, whatever [BEAM] and [JITBEAM] were. It binds no setting. *)

val jit_lower :
  ?beam:int ->
  ?search:(int -> Postrange.Scheduler.t -> Postrange.Scheduler.t) ->
  targets:(string -> Helpers.Target.t) ->
  held_bufs:Ops.t list ->
  inputs:Ops.t list ->
  Ops.t ->
  Ops.t
(** [jit_lower ~beam ~search ~targets ~held_bufs ~inputs linear] is the captured
    schedule [linear], an {!Op.Linear} of calls, ready to link:
    + the [i]th buffer of [inputs] is replaced wherever [linear] reaches it by
      the parameter of slot [i] ({!Call.param}), of its type, device, size,
      alignment and phase ({!Call.storage_phase});
    + its buffers are placed in arenas ({!Memory.memory_plan_rewrite}), except
      [held_bufs], whose contents outlive a run, such as the buffers a caller
      keeps or that hold constants;
    + it is compiled ({!Realize.compile_linear}, with [search] and [targets]),
      with the beam width [beam], or else {!Setting.jitbeam}'s, or else
      {!Setting.beam}'s, as [BEAM], and [JITBEAM] unset.

    Raises as {!Realize.compile_linear} does. *)
