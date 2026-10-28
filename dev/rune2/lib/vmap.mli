(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Maps: an interpretation whose values stand for one value per lane.

    A map has [n] lanes and a name ({!Program.axis}). A {e lane value} is a
    value of the interpretation holding its {e batch}: every lane's value
    stacked along a new axis 0 of length [n]. Its form is one lane's. A value
    the interpretation does not own is one every lane shares, a constant of the
    map. Nothing reads a lane value's elements.

    The rule of each operation over lane values checks it as one lane's, then
    issues it once outward, under the interpretations around this one, over the
    batches, with the operation's axes, pads, regions and coordinates moved past
    the lane axis. Its results have the forms eager execution gives one lane's,
    and are lane values but for the shared ones below. A constant operand is
    broadcast along a new axis 0, but for a contraction's: there the lane axis
    is a batch pair where both operands, or the segments' offsets, have lanes,
    and otherwise a free axis of the operand that has them, moved to the front
    of the result, as JAX's [dot_general] maps it. A loop's output whose program
    reads no load with lanes, and for a scan whose carry has none, is computed
    once, over the constant loads, and shared by every lane, so forward mode
    under a map of tangents computes its primal once. A [Place] onto the
    placement a lane already has passes the batch through, so a batch cut along
    its leading axis stays cut.

    The interpretation's handler of constructs ({!Program.construct}) lies just
    outside it. It answers {!Program.Bound} for its name; [Lanes] and
    [Lane_index] of its name, and [Lanes] of a lane value under another name;
    every addition, adding the sum of its lanes' values outward once; [Detach]
    of a lane value, detaching its batch outward; a custom rule or a linear call
    over lane values by performing it outward with each of its programs traced
    again at its signature over the batches and evaluated over their lanes, the
    additions of a pullback or a transpose summed over the lanes; a scan, a
    remat and a call over lane values, and a call whose program names the map,
    performing them outward with their programs mapped over the batches (a
    scan's carry has lanes where its init has them or a step gives them, its
    init broadcast until then, and its additions are every lane's; a call's map
    is derived once per derivation at its site, {!Program.derive}); and the
    other constructs over lane values by running their default where they were
    performed ({!Program.Evaluate}), so that their programs' operations reach the rules.
    It answers {!Program.Running} with [None]. *)

val run :
  name:string ->
  axis:Program.axis ->
  int ->
  (Nx.Prim.interpretation -> 'a) ->
  'a
(** [run ~name ~axis n f] is [f i], [i] a new map of [n] lanes named [axis],
    named [name] in messages. Raises what [f] raises. *)

val lane : Nx.Prim.interpretation -> ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t
(** [lane i b] is the lane value of [i] whose batch is [b], of rank at least
    one: of [b]'s dtype, its layout without axis 0, and [b]'s placement without
    its leading axis ([Nx.Placement.without_leading_axis]). *)

val batch :
  Nx.Prim.interpretation -> int -> ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t
(** [batch i n x] is [x]'s batch if [i] owns [x], and otherwise [x] broadcast
    along a new axis 0 of length [n], the map's number of lanes. *)
