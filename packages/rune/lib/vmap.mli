(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Vectorizing maps: lanes and the batching table.

    An installation ({!t}) owns the lanes it makes: traced values that stand for
    one row of a batched tensor, whose metadata is the row's. The interpreter
    turns an operation on its lanes into one operation on the batched tensors,
    with the map's axis in front, and forwards every other operation. A value
    the lanes share is a constant of the map. Maps nest: an operation one map
    issues on another's lanes is batched again by that one. *)

type t
(** The type for installations of maps. *)

val create : ?axis:Construct.axis -> string -> int -> t
(** [create ?axis entry n] is a fresh map of [n] lanes, distinct from every
    other, named [axis] if given, for the entry point [entry], which its errors
    name. *)

val lane : t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [lane m x] is the lane of [m] whose batched tensor is [x], of [m]'s length
    along its first axis. *)

val batched : t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [batched m x] is the batched tensor of [x] if it is a lane of [m], and [x]
    broadcast along a new first axis of [m]'s length otherwise: what the loop of
    the map stacks. *)

val install : t -> (unit -> 'a) -> 'a
(** [install m f] is [f ()] under [m]'s interpreter ({!Construct.install}).

    It answers the collectives that address [m]: [lanes] of the map named [m]'s
    axis is every lane's value stacked, a constant of [m]; the lane index of
    [m]'s axis, or of the innermost anonymous map, is a lane of [0] to [n - 1];
    and lanes of another map's axis keep [m]'s lanes. An addition to a total is
    the sum of its lanes', or a shared value times [n]. A custom call passes on
    as the call of its rule batched, and a remat as the remat of its function
    batched, each reinstalling [m] around the user's function, so that the lanes
    it captures are [m]'s again; the gradient of a custom call's argument the
    lanes share is the sum over the lanes. A loop until a stop the lanes do not
    share runs masked: a stopped lane holds its carry, runs the step at a
    running lane's point, and its additions are dropped. A root passes on as the
    root of its functions mapped; its linear solve applies the operator it
    receives at the map's level, and its residual may not gather [m]'s lanes. A
    call at a map's level ([At_map]) is answered by that map, or a held map
    inside it, and by a map between for its own lanes, mapping the function over
    its axis.

    Raises [Invalid_argument], at the operation, for a read of a lane. *)
