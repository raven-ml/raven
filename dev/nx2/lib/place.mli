(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Placing values: a value's arrays moved to a placement.

    Placing reads and writes memory through nx.array and rig alone and runs no
    kernel, so a set without kernels places. Each device of the destination
    holds its window of the value and allocates at most the window's bytes.
    An empty window is an array of no bytes of its own.

    Where one source array holds the window, the device takes the array itself
    on the same device, a borrow where the host addresses both the array's
    memory and the device's ({!Rig.reaches}), or a copy of the window's
    elements ({!Nx_array.to_device}).

    Otherwise the device gathers its window into a fresh C-contiguous array:
    each distinct source window gives the box it shares with the destination's,
    from a source on the device where one holds it, each box stored by
    {!Nx_array.transfer}. An element narrower than a byte gathers on the host
    where the host does not address the device's memory, and the window is then
    brought to it. Nothing holds the whole value unless the destination does. *)

val value :
  by:string ->
  'e Devices.placement ->
  ('v, 's, 'd) Value.t ->
  ('v, 's, 'e) Value.t
(** [value ~by p x] is [x]'s elements at [p]: over [x]'s own arrays when [x]'s
    placement equals [p].

    Raises [Invalid_argument] naming [by] if [p] cuts an axis [x] lacks, and if
    [x] is a constant (the engine computes those), {!Rig.Lost} for a lost
    device, and what {!Rig.Buffer.copy} raises. *)

val view :
  by:string ->
  ('v, 's, 'd) Value.t ->
  int ->
  Nx_array.Move.range array ->
  ('v, 's) Nx_array.t
(** [view ~by x k w] is the window [w] of [x]'s whole, from [x]'s array on its
    set's device [k], without a copy.

    Raises [Invalid_argument] if [x] is a constant, or no array of [x] on [k]
    holds [w]. *)
