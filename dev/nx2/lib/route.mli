(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Routes: where an operation reads its operands and where its results lie.

    A route is plain data from placements and shapes; it computes nothing. The
    engine takes placements from it alone, for traced results and eager ones, so
    a traced result lies where its eager twin does.

    The rule: every result lies at one placement, the {e target}. A constant's
    placement ({!Devices.anywhere}) joins the others: operands that are all
    constants give [anywhere]. Each other operand is first given the axes the
    operation reads whole on each device (an axis it acts along), and moved to
    the result's axes. The target is the last of these that cuts an axis; else,
    where some lie on one device, that device, which every such operand must
    share; else the first. Each operand is then read where it lies if that gives
    every device of the target the part of the operand the device's window of
    the result depends on: the same window, or the whole operand on that device;
    otherwise it is read at the target moved back to its axes, which the engine
    places it at.

    Two rules follow, with JAX as the reference for placements. Operands of one
    set never raise for where they lie: the brand already makes them one set,
    and moving data within a set is the route's, as JAX computes over operands
    sharded differently on one device set. Operands each alone on a different
    device raise, as JAX refuses operands committed to two devices: no
    arrangement holds both, and moving one is a choice the program makes with
    [Nx.place]. *)

(** The type for how an operation's results depend on its operands. *)
type rule =
  | Elementwise
      (** Operands and results of one shape; a result's element reads each
          operand at its own index. *)
  | Reduce of int array
      (** Operands of one shape; results drop these axes, each read whole. *)
  | Along of int array
      (** Operands and results of one rank, the operation acting along these
          axes, each read whole: a scan, a sort, a transform's axes, a
          factorisation's last two. *)
  | Gather of int
      (** Operands [idx; x] of one rank, and results of [idx]'s shape: [x] is
          read whole along this axis. *)
  | Into of int
      (** Operands of one rank, results of the last one's shape, written into it
          along this axis, which every operand reads whole: the target is the
          last operand's placement where it lies on the set. *)
  | Replicated  (** Results whole on every device of the set. *)

type 'd t = {
  operands : 'd Devices.placement array;
      (** Where each operand is read, in the operation's operand order. *)
  result : 'd Devices.placement;
}

val route :
  by:string -> rule -> 'd Devices.placement array -> int array array -> 'd t
(** [route ~by r ps shapes] routes an operation of rule [r] over operands at
    [ps] of [shapes].

    Raises [Invalid_argument] naming [by] if operands lie on two sets or on two
    devices alone, if a placement an operand is read at does not divide its
    shape ({!Grid.window}), if an axis of [r] is not an axis of an operand, and
    as an internal fault if [ps] and [shapes] differ in length or [ps] is empty.
*)
