(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Loops of a program: a range of trips around one call of a body.

    A program that repeats a step as many times as its shapes fix holds the step
    once, as a body that a loop runs, so that compiling it costs one step
    whatever the count. *)

open Tolk

val cut :
  own:unit Ops.Tbl.t ->
  (unit -> int) ->
  (int * Ops.t) list ref ->
  Ops.t ->
  Ops.t
(** [cut ~own next args body] is [body] with each part of its graph that reaches
    none of the call's own parameters [own], runs inside no range that is open
    there, and reads storage replaced by a parameter of slot [next ()], which
    the call binds to the part, added to [args]: the part is computed once,
    before the loop. A movement stays in the body and the part is what it moves:
    a view costs nothing where it is read, while computing it once would store a
    transposed or broadcast copy that the loop then reads. A range open at a
    part is one of the body's own loops, which the part moves with. Another
    call's parameter, such as an enclosing step's carry, is storage the call
    binds like any other; so is storage that a call of the body that runs each
    trip writes. *)

val repeat :
  Ops.device ->
  int ->
  Ops.t list ->
  (Ops.t -> Ops.t list -> Ops.t list) ->
  Ops.t list
(** [repeat d n init step] is the values that [n] applications of [step] reach
    from [init], computed on [d]: [step i] maps values of [init]'s shapes and
    dtypes to values of the same, [i] the [int32] scalar index of the
    application, from [0], and is applied to the nodes it is given, so that it
    may read them at any index. The loop's call passes each trip its first
    step's index as a scalar argument: no kernel computes an index.

    A loop of [n / 2] trips runs [step] twice a trip, traced once as a call's
    body: from the carries' storage into storage of the body's own, then back.
    An odd [n] applies [step] once more before the loop. A value with no element
    stands in [step] as zeros, and is its initial value after the loop. The
    parts of the body that read storage but no carry are computed once before
    the loop ({!cut}). *)
