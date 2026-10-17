(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Routing: where an operation runs and where its result lives.

    Operands all on the host, or on the disk, run on the host. Placed
    operands must share their devices, and host operands join them. The
    route is decided before anything is read, so operands on two device
    lists raise [Invalid_argument] before any work. Views of whole shards of
    one split storage, each on its own device, meet as copies on the
    storage's devices.

    The result's placement follows the operation's {!rule}. An elementwise
    result keeps its operands' split, which must be the same for all of
    them. A reduction over a split axis holds a copy on every device. An
    operation along a split axis raises [Invalid_argument]. *)

type route = On_host | At of Placement.t  (** Where an operation runs. *)

(** How the axes of an operation's result derive from its operands'. *)
type rule =
  | Elementwise  (** The operands' shape. *)
  | Along of int list
      (** Acts along these axes, and along the others as elementwise. *)
  | Gather of int
      (** Reads its first operand along this axis at the positions its
          second holds, the other axes as elementwise. Along a split axis
          it is a reduction. *)
  | Reduce of { axes : int array; keepdims : bool }
  | Contract
      (** A product over the last axis of the first operand and the
          next-to-last of the second. *)
  | Into  (** The first operand, with the others written into it. *)

(** How an operation is routed. *)
type routing =
  | Computes of rule * Value.packed list
      (** A computation by a rule over its operands. *)
  | Moves of Value.packed * Move.t  (** A movement of one operand. *)
  | Places of Placement.t
  | Reads of Value.packed

val routing : 'r Op.t -> routing
(** [routing op] is how [op] is routed. *)

val placement_of : ('a, 'b) Value.t -> Placement.t option
(** [placement_of x] is [x]'s placement if [x] is placed off the disk, and
    [None] for a host value or one on the disk, which joins any placement.

    Raises as {!Value.outside_trace} if [x] is traced. *)

val route :
  (Value.packed -> Placement.t option) -> string -> rule ->
  Value.packed list -> route
(** [route where op rule xs] is where [op] runs over [xs], [where] giving
    each operand's placement, [None] for one that joins any.

    Raises [Invalid_argument] naming the function [op] if placed operands are
    on different devices, unless they are whole shards of one storage, if
    they are split differently, or if one is split along an axis [rule] acts
    along. *)

val placement : 'r Op.t -> Placement.t
(** {!Nx.Op.placement}. *)
