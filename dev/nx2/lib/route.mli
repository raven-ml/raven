(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Routes: where an operation reads its operands and where its results lie.

    A route is plain data from placements and shapes; it computes nothing. The
    engine takes placements from it alone, for traced results and eager ones, so
    a traced result lies where its eager twin does.

    The rule: every result lies at one placement, the {e target}. An operand of
    every set has no placement and joins the others: operands that are all of
    every set give no route. Each other operand is first given the axes the
    operation reads whole on each device (an axis it acts along), and moved to
    the result's axes. The target is the last of these that cuts an axis; else,
    where some lie on one device, that device, which every such operand must
    share; else the first. Each operand is then read where it lies if that gives
    every device of the target the part of the operand the device's window of
    the result depends on, and lies on no other device: the same window, or the
    whole operand on each device of the target; otherwise it is read at the
    target moved back to its axes, which the engine places it at.

    Two rules follow, with JAX as the reference for placements. Operands of one
    set never raise for where they lie: the brand already makes them one set,
    and moving data within a set is the route's, as JAX computes over operands
    sharded differently on one device set. Operands each alone on a different
    device raise, as JAX refuses operands committed to two devices: no
    arrangement holds both, and moving one is a choice the program makes with
    [Nx.place]. *)

(** The type for how an operation's results depend on its operands. Each case
    gives every operand's window for a window of the results ({!window}). *)
type rule =
  | Elementwise
      (** Operands and results of one shape; a result's element reads each
          operand at its own index. *)
  | Reduce of { axes : int array; whole : int array }
      (** Operands of one shape, but for trailing axes [whole] holds; results
          drop [axes] and keep the others, [axes] and [whole] read whole: a
          reduction whose loads pad or window [whole]'s axes. *)
  | Whole of int array
      (** Operands and results that share their leading axes, these axes read
          whole: those the operation acts along (a scan's, a sort's, a
          transform's, a factorisation's last two), and those a padded load or
          a fold pads, crops or windows and those its windows append. An axis
          that not all of them have is one of these. *)
  | Gather of int
      (** Operands [idx; x] of one rank, and results of [idx]'s shape: [x] is
          read whole along this axis. *)
  | Into of int
      (** [Whole [| axis |]]'s windows, written into the last operand: operands
          of one rank, results of the last one's shape, and the target the last
          operand's placement where it lies on the set. *)
  | Contract of { batch : (int * int) array; contracting : (int * int) array }
      (** Operands [a; b], then possibly [init] of the result's shape, the
          result's axes the batch pairs', then [a]'s other axes, then [b]'s
          ({!Nx_kernel.Shape.contract}): the contracting pairs' axes are read
          whole. *)
  | Move of Nx_array.Move.t
      (** One operand, moved: an axis that keeps its elements in order keeps its
          cut, where it moves to (a permutation, a broadcast that does not
          stretch it, a reshape that keeps it with the product of the extents
          before it, a slice that keeps all of it); any other is read whole. *)
  | Replicated  (** Operands read whole, results whole on each device. *)

val moved : Nx_array.Move.t -> int array -> int -> int option
(** [moved m s a] is the axis that axis [a] of a value of shape [s] becomes
    under [m], where it keeps its elements in order: a permutation's, a
    broadcast that does not stretch it, a reshape that keeps its extent and the
    product of the extents before it, a slice that keeps all of it. [None] for
    any other. *)

val window :
  rule ->
  operand:int ->
  int array ->
  Nx_array.Move.range array ->
  Nx_array.Move.range array
(** [window r ~operand:i s w] is the window of operand [i], of shape [s], that
    the window [w] of an operation's results reads by [r]: whole along each axis
    [r] reads whole, [w]'s range along an axis the results keep. Where an
    operation has several results, [w] is a window of the first.

    The engine computes an operation one window at a time, each device over its
    operands' windows, under the {e window law}: for every window [w] of the
    results, the operation over each operand's [window r ~operand:i s w] equals
    the window [w] of the operation over the whole operands, bitwise. A kernel
    library meets it where an element's association, its algorithm and whether
    the library declines depend on the extents the element reads whole, the
    dtypes, the attributes, the library and the device's architecture, never on
    the extents its window cuts. *)

type 'd t = {
  operands : 'd Devices.placement array;
      (** Where each operand is read, in the operation's operand order. *)
  result : 'd Devices.placement;
}

(** The type for where operands lie together. *)
type 'd common =
  | Every_set  (** Every operand is of every set. *)
  | One of 'd Devices.placement
      (** Every operand not of every set lies at this placement. *)
  | Other  (** Any other arrangement. *)

val common : 'd Devices.placement option array -> 'd common
(** [common ps] is where operands at [ps] lie together, [None] for an operand of
    every set. Operands at [One p], [p] cutting no axis, are read at [p] and
    their results lie there, whatever the rule ({!route}). *)

val route :
  by:string ->
  rule ->
  'd Devices.placement option array ->
  int array array ->
  'd t option
(** [route ~by r ps shapes] routes an operation of rule [r] over operands at
    [ps] of [shapes], [None] for an operand of every set. It is [None] where
    every operand is: the operation is of every set too. An operand of every
    set is read at the target moved back to its axes.

    Raises [Invalid_argument] naming [by] if operands lie on two sets or on two
    devices alone, if a placement an operand is read at does not divide its
    shape ({!Grid.window}), if an axis of [r] is negative, a gather's or a
    scatter's axis is not an axis of an operand, operands are too many or too
    few for [r], or a movement does not fit its operand
    ({!Nx_array.Move.shape}), and as an internal fault if [ps] and [shapes]
    differ in length or [ps] is empty. *)

val keeps : rule -> int array array -> int -> bool
(** [keeps r shapes a] is whether the results of an operation of rule [r] over
    operands of [shapes] keep their axis [a] whole: every window of them that
    the engine computes holds all of it: an axis a reduction keeps of those it
    reads whole, an axis of [Whole] or [Into], an axis that no axis of a moved
    operand becomes in order ({!moved}), and every axis of [Replicated]. *)

val at : by:string -> rule -> int array array -> 'd Devices.placement -> 'd t
(** [at ~by r shapes p] routes an operation of rule [r] over operands of
    [shapes] whose results are wanted at [p]: the result lies at [p] with each
    axis [r] keeps whole in the results whole ({!keeps}), and each operand is
    read at that target moved back to its axes.

    Raises [Invalid_argument] naming [by] if a placement an operand is read at
    does not divide its shape. *)
