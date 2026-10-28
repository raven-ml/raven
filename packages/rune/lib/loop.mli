(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Loops of a program, each a range of trips around one call of a body.

    A program that repeats a step as many times as its shapes fix holds the step
    in a body that a loop runs ({!repeat}). The loop is a range of {e trips}
    around one tolk call of the body, the loop's {e call}. In {!repeat}'s loop,
    each trip applies the step twice, and an odd count applies it once more
    before the loop, so compiling the program costs the same whatever the count.
    The call's {e own} parameters are those whose values change from trip to
    trip, such as the carries and the trip's index. {e Storage} is a buffer, a
    parameter in global memory, or a value after the work that writes it. The
    parts of the body that read storage but no own parameter are computed once,
    before the loop, except as {!cut} says. *)

open Tolk

val cut :
  ?hoist:bool ->
  own:unit Ops.Tbl.t ->
  (unit -> int) ->
  (int * Ops.t) list ref ->
  Ops.t ->
  Ops.t
(** [cut ~own next args body] is [body] with each part of its graph computed
    once, before the loop, where the part reaches none of the call's own
    parameters [own], runs inside no range that is open there, and reads
    storage. Each such part is replaced by a parameter of slot [next ()], which
    the call binds to the part, added to [args]. A movement stays in the body,
    and the part is what it moves. A range open at a part is one of the body's
    own loops, which the part moves with. Another call's parameter, such as an
    enclosing step's carry, is storage the call binds like any other. Storage
    that a call of the body that runs each trip writes varies as [own] does, so
    every part that reads it stays in the body. So does a part of no width (a
    weak dtype), which the body computes from what it reads.

    [hoist] defaults to [true]. With [~hoist:false], only a buffer or a
    parameter in global memory is replaced. Every part that reads it stays in
    the body, and so does a value after the work that writes it, so that a call
    a flag skips computes none of it. *)

val repeat :
  Ops.device ->
  int ->
  Ops.t list ->
  (Ops.t -> Ops.t list -> Ops.t list) ->
  Ops.t list
(** [repeat d n init step] is the values that [n] applications of [step] reach
    from [init], computed on [d]. [step i vs] maps values of [init]'s shapes and
    dtypes to values of the same, where [i] is the application's index, an
    [int32] scalar from [0]. [step] receives graph nodes, so it may read them at
    any index. Where [n >= 2], [step] is applied three times while the loop is
    built: once from [init] at index 0, whose result an even [n] drops, and
    twice to trace the body. It is applied once where [n = 1] or [n < 0], and
    never where [n = 0] or no value of [init] has an element. The call passes
    each trip its first application's index as a scalar argument, and the
    second's is one more. Where [n <= 0], the result is [init]. A value with no
    element stands in [step] as zeros, and is its initial value after the loop.
*)
