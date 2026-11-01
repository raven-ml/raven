(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Programs over several devices, split into operations on one device each and
    copies between them.

    A value over several devices has a {e piece} on each, its window there
    ({!Nx.Prim.pieces}). To {e split} a program is to trace it again with each
    operation computed as its partition ({!Nx.Prim.partition}), so that the
    program's operations are those eager computation runs over the pieces: each
    on one device, and each move between devices a placement of one piece onto
    another device. The split program's arguments are the pieces of the
    program's, in order, a value on one device being its own piece, and its
    outputs the pieces of the program's. This module lies above {!Trace};
    {!Compile} lowers the split program. *)

exception Declined of string
(** The exception for a program the compiler does not split, and the reason.
    {!Compile} raises it for programs it does not lower. *)

val device_of : 'd Nx.Placement.t -> Rig.t option
(** [device_of p] is the device [p] holds a whole value on, if it is one device
    alone. *)

val rig_of : 'd Nx.Placement.t -> Rig.t
(** [rig_of p] is the device of the one-device placement [p].

    Raises [Invalid_argument] if [p] is over several devices. *)

val needed : Program.t -> Program.form array -> bool
(** [needed p forms] is [true] iff [p] over arguments of [forms] has a value
    over several devices, or its arguments, the results it computes and its
    placements lie on several devices. A creation or a constant on another
    device than these is no such value: the program computes the one, and places
    or copies the other, where it computes. *)

(** {1:split Splitting} *)

(** Where a split program's outputs are among its program's. *)
type output =
  | One of int  (** On one device: the output at that index. *)
  | Over of Program.form * int array
      (** Over several devices, of the form: its pieces, one per device of its
          placement in order, the outputs at those indices. *)

type t = {
  program : Program.t;  (** The split program. *)
  forms : Program.form array;  (** Its arguments', the pieces' in order. *)
  args : int array;  (** The number of pieces of each argument, in order. *)
  outputs : output array;  (** Each output of the program split. *)
  moves : string list;
      (** Each move between devices, in order: the function that makes it, its
          bytes, and the devices it moves from and to. *)
}

val split : name:string -> home:Rig.t -> Program.t -> Program.form array -> t
(** [split ~name ~home p forms] is [p], over arguments of [forms], split, to
    compute on [home] its checks and an iterate's stop, which the split program
    copies there each trip where it lies only on another device. Its tracer is
    named [name] in messages.

    A scan or an iterate is the loop of its programs split, over its operands'
    pieces, each device reading its piece of a row from its piece of the rows.

    Raises [Declined] for a program that threads a total or emits, and one with
    a check, a construct other than [Detach], [Scan] and [Iterate], a scan over
    rows cut along axis 0, an iterate that stacks its steps, a loop whose step
    gives a carry another placement, a traced value placed onto another set, or
    an argument placed so that a device holds more of it, which a compiled call
    would hold at once with every piece of its closings; and what nx raises for
    an operation its partition refuses. *)
