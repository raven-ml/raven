(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Partitions: an operation over several devices as operations on one
    device each and moves between them.

    A value over a placement of several devices has a {e piece} on each of
    them: its window there ({!Devices.window}), as a value on that device
    alone. An operation's {e partition} computes each device's pieces of
    its results from its operands' pieces by operations whose operands and
    results lie on one device, views of pieces, and moves of pieces from
    one device to another. Eager execution runs an operation over several
    devices as its partition, and a compiler that traces the partition
    records the same operations and moves, so the two compute alike.

    An {e engine} gives the partition those steps. Each field names the
    operation it stands for, and an engine computes or records it. The
    partition decides every window, every device that computes, and the
    pieces each window comes from; an engine decides how a step runs, such
    as which of the pieces that hold a window a move reads, and whether it
    borrows their memory or copies it. An engine reads the windows it is
    given and never writes them: one window may stand for several. A device
    computes where one of its windows of the results holds elements. Where
    none does on any device, the first device computes, so that the kernels
    answer for the operation as on one device.

    This module lies above {!Prim} and {!Route}. {!Partial} partitions the
    folds over a cut axis by it; [Exec] gives the engine of eager
    execution. *)

(** The type for engines. *)
type engine = {
  run :
    'r 'd.
    by:string ->
    ?into:'d Value.any ->
    unit Devices.placement ->
    Nx_array.Layout.t array ->
    'r Value.prim ->
    'r;
      (** [run ~by ~into here layouts op] is [op] at the one-device
          placement [here], where every operand lies, its results laid out
          as [layouts] ({!Prim.val-layouts}). [into] is memory the first
          result may take, a value there of its dtype and layout; an engine
          may ignore it. *)
  piece :
    'v 's 'd.
    ('v, 's, 'd) Value.t -> int -> 'd Devices.placement -> ('v, 's, 'd) Value.t;
      (** [piece x j here] is [x]'s piece on the device at position [j] of
          its placement's grid, whose one-device placement is [here]: [x]
          itself on one device. A piece of an empty window has no
          elements. *)
  view :
    'v 's 'd. Nx_array.Move.t -> ('v, 's, 'd) Value.t -> ('v, 's, 'd) Value.t;
      (** [view mv x] is [Move (mv, x)] for [x] on one device, over [x]'s
          memory: a slice, a reshape of a contiguous value, or [x] itself for
          a slice of all of it. *)
  bring :
    'v 's 'd 'e.
    'e Devices.placement ->
    (('v, 's, 'd) Value.t * Nx_array.Move.range array) list ->
    ('v, 's, 'e) Value.t;
      (** [bring at ps] is [Place (at, v)] at the one-device placement [at]
          for the window [w] of a piece [x], [v] the view of [w] of [x], for
          the one [(x, w)] of [ps] the engine chooses. Each holds the same
          elements, on one device. *)
  gather :
    'v 's 'd 'e.
    'e Devices.placement ->
    ('v, 's) Value.dtype ->
    int iarray ->
    (Nx_array.Move.range array
    * ('v, 's, 'd) Value.t
    * Nx_array.Move.range array)
    list ->
    ('v, 's, 'e) Value.t;
      (** [gather at dt s boxes] is the value of shape [s] at the
          one-device placement [at] that holds, at each window [w] of
          [(w, x, b)] in [boxes], the window [b] of the piece [x], which
          lies on any device. The windows [w] tile [s]. *)
  assemble :
    'v 's 'd 'e.
    'e Devices.placement ->
    int array ->
    ('v, 's) Value.dtype ->
    ('v, 's, 'd) Value.t option array ->
    ('v, 's, 'e) Value.t;
      (** [assemble p s dt ps] is the value of shape [s] at [p] whose piece
          on the device at position [j] of [p]'s grid is [ps.(j)], or of no
          elements where [ps.(j)] is [None]. *)
}

val read :
  engine ->
  by:string ->
  ('v, 's, 'd) Value.t ->
  int ->
  Nx_array.Move.range array ->
  ('v, 's, 'd) Value.t
(** [read e ~by x k w] is the window [w] of [x]'s whole on its set's device
    [k], a value there alone: a view of [x]'s piece on [k], or {!engine}'s
    [assemble] of no piece for an empty [w].

    Raises [Invalid_argument] if no piece of [x] on [k] holds a nonempty
    [w]. *)

val windows :
  engine ->
  by:string ->
  ?into:'d Value.any ->
  unit Route.t ->
  Nx_array.Layout.t array ->
  'r Value.prim ->
  'r
(** [windows e ~by ~into route layouts op] is [op] computed by windows at
    [route]'s result, its results laid out as [layouts]
    ({!Prim.val-layouts}). On one device it is {!engine}'s [run] of [op]
    over each operand's piece there, its first result into [into] where it
    is given. Over several, each device computes its window [w] of the
    results by
    {!Prim.local}[ op w] over the windows of the operands it reads
    ({!Route.window}), at its own placement where [op] creates its
    values, its first result into its piece of [into], a value at the
    route's result, where it is given.

    Raises what [e]'s steps raise, and [Invalid_argument] as an internal
    fault for a route that computes by partials. *)

type ('v, 's, 'd) held
(** The type for a value's pieces with the windows they hold, from which
    {!take} takes windows. Each piece is made once, when a window first
    reads it. *)

val held : by:string -> ('v, 's, 'd) Value.t -> ('v, 's, 'd) held
(** [held ~by x] is [x]'s pieces and their windows. *)

val take :
  engine ->
  ('v, 's, 'd) held ->
  'e Devices.placement ->
  Nx_array.Move.range array ->
  ('v, 's, 'e) Value.t
(** [take e h here w] is the window [w] of the whole of [h]'s value at the
    one-device placement [here], as {!place} takes a device's window: from
    the pieces that hold all of it by {!engine}'s [bring], else gathered
    from the box each distinct window shares with [w], from a piece on that
    device where one holds the box's window. [w] holds elements. *)

val place :
  engine ->
  by:string ->
  'e Devices.placement ->
  ('v, 's, 'd) Value.t ->
  ('v, 's, 'e) Value.t
(** [place e ~by p x] is [x]'s elements at [p]: [x]'s pieces where [x] lies
    at [p]. Otherwise each device of [p] whose window [w] holds elements
    takes it from the pieces of [x] that hold all of [w], by {!engine}'s
    [bring] of their views, or, where none does, gathers it by {!engine}'s
    [gather] from the box each distinct window of [x]'s shares with [w],
    viewed from a piece on that device where one holds the box's window.

    Raises [Invalid_argument] naming [by] if [p] cuts an axis [x] lacks. *)
