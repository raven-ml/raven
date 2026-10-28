(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The map interpretation, whose values stand for one value per lane.

    A map has [n] lanes and a name ({!Program.axis}). A {e lane value} is a
    value of the interpretation that holds a {e batch}, every lane's value
    stacked along a new axis 0 of length [n]. Its form is one lane's. Every
    other value is {e shared} by the lanes. A read of a lane value raises
    [Invalid_argument] naming the read and the map's entry point, as
    [Nx.item: the value is traced by Rune.vmap].

    The rule of each operation over lane values checks it as one lane's. It then
    issues the operation once outward, under the interpretations around this
    one, over the batches, with the operation's axes, pads, regions and
    coordinates moved past the lane axis. Its results have the forms one lane's
    operation gives, and are lane values except where every lane shares them, as
    below. A shared operand is broadcast along a new axis 0, except in a
    contraction.

    In a contraction ([Nx.Prim.Contract]), an operand or its scale with lanes
    gives the lane axis. Where both operands have lanes, or the offsets of its
    [segments] do, the lane axis is a batch axis of both, and a shared operand
    is broadcast. Otherwise it is a free axis of the operand with lanes, moved
    to the front of the result, or [a]'s where only [init] has lanes.

    A [Map] or [Reduce] output whose program reads some load ([Nx.Prim.load]),
    but none with lanes, is computed once, over the shared loads, and shared by
    every lane. A scan's output is shared too where neither its loads nor its
    carry have lanes. Where only tangents have lanes, as when a map runs forward
    mode over a batch of tangents, the primals are thus computed once. A [Place]
    onto the placement a lane already has passes the batch through, so a batch
    cut along its leading axis stays cut.

    The interpretation's handler of constructs ({!Program.construct}) lies just
    outside it. It answers {!Program.Bound} for its name, [Lanes] and
    [Lane_index] of its name, a call, an iterate or a switch whose programs name
    the map, every addition, and each other construct over lane values. It
    performs a construct outward over the batches with its programs batched over
    the lanes, a call's batched program derived at its site ({!Program.derive}).
    An addition adds the sum of its lanes' values outward once, so a shared
    value adds [n] copies. The handler answers {!Program.Running} with [None].
    Where an iterate or a switch runs the lanes apart, the additions of a lane
    that has stopped or taken another branch are dropped, and [Lanes] of the map
    raises [Invalid_argument]. *)

val run :
  name:string ->
  axis:Program.axis ->
  int ->
  (Nx.Prim.interpretation -> 'a) ->
  'a
(** [run ~name ~axis n f] is [f i], [i] a new map of [n] lanes whose name is
    [axis]. [name] names the entry point in messages. Raises what [f] raises. *)

val lane : Nx.Prim.interpretation -> ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t
(** [lane i b] is the lane value of [i] whose batch is [b]. Its form is [b]'s
    dtype, [b]'s layout without axis 0, and [b]'s placement without its leading
    axis ([Nx.Placement.without_leading_axis]). [b] must have rank at least one
    and the map's number of lanes along axis 0. *)

val batch :
  Nx.Prim.interpretation -> int -> ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t
(** [batch i n x] is [x]'s batch where [x] is a lane value of [i], and otherwise
    [x] broadcast along a new axis 0 of length [n]. [n] must be the map's number
    of lanes. *)
