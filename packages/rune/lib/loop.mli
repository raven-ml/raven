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

val viewed : Ops.t -> (Ops.t * Ops.t list) option
(** [viewed u] is the storage [u] views where [u] reads no other: a buffer, a
    parameter in global memory or a state of one, under movements, bitcasts
    and gathers at positions that are functions of constants alone, as a
    strided argument's are;
    and the scalar parameters its bounds and positions read, such as an
    enclosing loop's trip. A call passed that storage and those scalars
    computes [u] over its parameters and copies nothing. [None] where [u]
    computes, reads several storages, or moves with a range. *)

val cut :
  ?hoist:bool ->
  host:(Ops.device -> bool) ->
  own:unit Ops.Tbl.t ->
  (unit -> int) ->
  (int * Ops.t) list ref ->
  Ops.t ->
  Ops.t
(** [cut ~host ~own next args body] is [body] with each part of its graph
    computed once, before the loop, where the part reaches none of the call's
    own parameters [own], runs inside no range that is open there, and reads
    storage. Each such part is replaced by a parameter of slot [next ()], which
    the call binds to the part, added to [args]. A view stays in the body, a
    movement or a gather at positions that are functions of constants alone,
    and the part is what it views. So does a cast to a type of more bytes that
    holds every value of its source ({!Ops.widens}), as bfloat16 to float32,
    and the part is its source, stored narrow, which its readers widen. The
    widened part is computed once on a device [d] where [host d] holds, whose
    kernels compile for the host's processors: there a product that widens
    such a value at each read costs more than one reading it stored widened.
    It is computed once too where a call of the body
    reads it, under views, which would otherwise store it widened at each
    trip. A scalar parameter of the body around the loop that a part left in
    the body reads, as a view's bound may read that body's trip, is passed
    too. A range open at a part is one of the body's own loops, which the
    part moves with. Another call's parameter, such as an enclosing step's
    carry, is storage the call binds like any other. Storage that a call of
    the body that runs each trip writes varies as [own] does, so every part
    that reads it stays in the body. So does a part of no width (a weak
    dtype), which the body computes from what it reads.

    [hoist] defaults to [true]. With [~hoist:false], only a buffer or a
    parameter in global memory is replaced. Every part that reads it stays in
    the body, and so does a value after the work that writes it, so that a call
    a flag skips computes none of it. *)

val repeat :
  host:bool ->
  Ops.device ->
  int ->
  Ops.t list ->
  (Ops.t -> Ops.t list -> Ops.t list) ->
  Ops.t list
(** [repeat ~host d n init step] is the values that [n] applications of [step]
    reach from [init], computed on [d], whose kernels compile for the host's
    processors where [host] holds ({!cut}). [step i vs] maps values of
    [init]'s shapes and dtypes to values of the same, where [i] is the
    application's index, an [int32] scalar from [0]. [step] receives graph
    nodes, so it may read them at any index. Where [n >= 2], [step] is applied
    twice while the loop is built, to trace the body, and once more before it,
    from [init] at index 0, where [n] is odd. It is applied once where
    [n = 1], and never where [n <= 0] or no value of [init] has an element.
    The call passes each trip its first application's index as a scalar
    argument, and the second's is one more. Where [n <= 0], the result is
    [init]. A value with no element stands in [step] as zeros, and is its
    initial value after the loop. *)

val until :
  host:bool ->
  Ops.device ->
  int ->
  Ops.t list ->
  stop:(Ops.t list -> Ops.t) ->
  (Ops.t -> Ops.t list -> Ops.t list) ->
  Ops.t list
(** [until ~host d n init ~stop step] is the values that applications of
    [step] reach from [init], computed on [d] as {!repeat}'s [host] says,
    applied while [stop] does not hold at the values reached, at most [n]
    times: [init] itself where [stop init] holds. [step] is as {!repeat}'s,
    and [stop vs] is a boolean of one element. [step] is applied once while
    the loop is built, to trace the body, and [stop] twice: to [init], before
    the loop, and to the body's values. The call passes each trip its
    application's index as a scalar argument. Where [n <= 0] or no value of
    [init] has an element, the result is [init]. A value with no element
    stands in [step] as zeros, and is its initial value after the loop. *)
