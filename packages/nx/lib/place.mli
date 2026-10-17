(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Placing and reading values, which nx answers itself.

    Devices over one memory share its storage, so placing a value between
    them is a view. Between memories it is a copy, each device reading only
    its window, and reading a placed value copies its elements to the host.
    A value on the disk is read in its file's pages where the system maps
    them, and placed on devices whose memory the host addresses by
    borrowing them.

    {!move_to}, {!host_of} and {!read_elements} read a placed value under a
    read claim on its storage ({!Value.reading}), and raise as the claim
    does. A traced operand raises as {!Value.outside_trace}. *)

val move_to : Placement.t -> ('a, 'b) Value.t -> ('a, 'b) Value.t
(** [move_to p x] is [x] held at [p], with no interception. It is [x] itself
    if [x] is placed at [p], a view where [p]'s devices share [x]'s memories,
    and a copy otherwise. [x] must not be a host value with [p] the host's
    placement, which {!Entry.place} returns itself.

    Raises [Invalid_argument] as {!Nx.place} does. *)

val host_of : ('a, 'b) Value.t -> ('a, 'b) Nx_array.t
(** [host_of x] is [x] as a host array. It is [x]'s own if [x] is on the
    host, its file's pages if it lies on the disk at an aligned byte and the
    system maps them, and a copy of its view's elements otherwise. *)

val read_elements : ('a, 'b) Value.resident -> Rig.Buffer.t
(** [read_elements r] is a fresh host buffer of the elements of [r]'s whole
    view, in C order. *)

val run_in :
  ('a, 'b) Nx_dtype.t -> Rig.Buffer.t -> Nx_array.View.t ->
  Rig.Buffer.t option
(** [run_in dtype b v] is a view of [b] holding the elements of [dtype] at
    [v], in C order, if they are a contiguous run of [b] that starts on a
    byte and holds an element. It reads nothing and takes no claim. *)
