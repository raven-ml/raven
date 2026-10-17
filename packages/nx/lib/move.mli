(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Movements: views of a value's storage ({!Nx.Op.move}).

    A movement of a placed value is view arithmetic over its storage, the
    same on every shard. A split value moves shard by shard, so a movement
    must leave every element on its device. The split axis may move, stay
    whole, or be reshaped with the whole axes before it. A flip, a window or
    a cut of it across shards raises [Invalid_argument]. A cut inside one
    shard is a view of that shard on its device alone. *)

type t =
  | Reshape of int array
  | Expand of int array
  | Permute of int array
  | Shrink of (int * int) array
  | Flip of bool array
  | Window of { axis : int; size : int; step : int }
      (** {!Nx.Op.move}. *)

val view : Nx_array.View.t -> t -> Nx_array.View.t
(** [view v m] is the view [v] moved by [m]. Raises as the [Nx_array.View]
    function of [m] does. *)

val placement : Placement.t -> Nx_array.Shape.t -> t -> Placement.t
(** [placement p shape m] is the placement of a value of shape [shape] at [p]
    moved by [m].

    Raises [Invalid_argument] if [m] would move elements between devices, or
    if [p] is split and [m] reshapes to another number of elements. *)

val views : ('a, 'b) Value.t -> t -> bool
(** [views x m] is [true] iff [m] moves [x] as a view of its storage, which
    holds for every movement but a reshape that the strides of [x]'s view
    cannot express.

    Raises as {!placement} does, or as {!Value.outside_trace} if [x] is
    traced. *)

val apply : ('a, 'b) Value.t -> t -> ('a, 'b) Value.t
(** [apply x m] is [x] moved by [m], over [x]'s storage. A moved placed value
    gets a fresh id. [m] must move [x] as a view ({!views}).

    Raises as {!view} and {!placement} do, or as {!Value.outside_trace} if
    [x] is traced. *)
