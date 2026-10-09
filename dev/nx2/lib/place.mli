(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Placing values: a value's arrays moved to a placement.

    Placing reads and writes memory through nx.array and rig alone and runs no
    kernel, so a set without kernels places. Each device of the destination
    takes its window of the value from the one source array that holds it: the
    array itself on the same device, a borrow where the device maps the array's
    memory ({!Nx_array.borrow}), and otherwise a copy of the bytes the window
    reaches ({!Nx_array.to_device}), keeping its layout. A window that no one
    source array holds, as a split changing axis gives, is taken from the value
    assembled on the host once per call: rig copies runs of whole bytes, and an
    element narrower than a byte is copied alone. That assembly runs no kernel
    either. *)

val value :
  by:string ->
  'e Devices.placement ->
  ('v, 's, 'd) Value.t ->
  ('v, 's, 'e) Value.t
(** [value ~by p x] is [x]'s elements at [p]: over [x]'s own arrays when [x]'s
    placement equals [p].

    Raises [Invalid_argument] naming [by] if [p]'s cuts do not divide [x]'s
    shape, and if [x] is a constant (the engine computes those), {!Rig.Lost} for
    a lost device, and what {!Rig.Buffer.copy} raises. *)

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
