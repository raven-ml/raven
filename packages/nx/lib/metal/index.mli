(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Gathers and assemblies: plans of index.metal's moves.

    Each plans a call that nx.metal found nothing to refuse in. [ops] is the
    call's operands as the door numbers them, its destination first, by which a
    plan names their buffers ({!Plan.slot}). A call of no element is [Ok None];
    one where a unit's launch passes {!Plan.allowed}[ ()] is [Error Declined].
*)

val gather :
  ops:Nx_array.any array ->
  axis:int ->
  dst:('v, 's) Nx_array.t ->
  (int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  ('v, 's) Nx_array.t ->
  (Plan.t option, Nx_array.answer) result
(** [gather ~ops ~axis ~dst idx x] plans {!Nx_kernel.S.gather}: one move of
    [dst]'s units, a thread each, each from [x] at its position along [axis] or
    zero bits outside it. *)

val gather_ranges :
  ops:Nx_array.any array ->
  dst:('v, 's) Nx_array.t ->
  starts:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  ends:(int64, Nx_array.Dtype.int64_elt) Nx_array.t ->
  ('v, 's) Nx_array.t ->
  (Plan.t option, Nx_array.answer) result
(** [gather_ranges ~ops ~dst ~starts ~ends x] plans
    {!Nx_kernel.S.gather_ranges}: each row's running maxima of [ends] into the
    workspace, a launch of 4096 runs a group and, past them, levels of the
    groups' maxima and a launch that carries each level's into the one below;
    then, a chunk of positions at a time, each position's source position, a
    thread each finding its run by halving the maxima, into a region of the
    workspace, and a gather at them. *)

val assemble :
  ops:Nx_array.any array ->
  fill:'v ->
  dst:('v, 's) Nx_array.t ->
  (Nx_array.Move.range iarray * ('v, 's) Nx_array.t) iarray ->
  (Plan.t option, Nx_array.answer) result
(** [assemble ~ops ~fill ~dst pieces] plans {!Nx_kernel.S.assemble}: a fill of
    [fill] where the pieces do not cover [dst], of every element outside the
    largest box of a piece, then a move of each piece into its region, in order.
    A first piece that is [dst]'s own elements in place, of the whole of it,
    moves nothing. *)

val fold :
  ops:Nx_array.any array ->
  Nx_kernel.pad ->
  dst:('v, 's) Nx_array.t ->
  ('v, 's) Nx_array.t ->
  (Plan.t option, Nx_array.answer) result
(** [fold ~ops p ~dst x] plans {!Nx_kernel.S.fold}: a thread an element of
    [dst], adding the taps that read it in C order of their numbers, in launches
    of ranges of tap numbers where an element's taps would pass one launch's
    time, each range's sums onto the last's. *)

(**/**)

val bound : ?runs:int -> ?region:int -> ?narrow:int -> ?taps:int -> unit -> unit
(** [bound ~runs ~region ~narrow ~taps ()] is {!Nx_metal.bound_index}. *)

(**/**)
