(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Reverse mode: a function linearized into a tape, and the tape transposed.

    {e Linearizing} a function runs it under forward mode ({!Jvp}) whose
    tangents are the variables of a tape tracer ({!Trace.Tape}): the primal
    computes at the call, and the operations on tangents are recorded as a
    program linear in them, the {e tape}, whose constants are the primal values
    those operations read. A loop's tangent is recorded over its loads, so a
    map's {e residuals}, the values the tape keeps, are its loads: it computes
    its intermediates again.

    {e Transposing} the tape walks its equations backward, from cotangents of
    its outputs to cotangents of its tangent parameters, issuing each equation's
    transpose under whatever interpretations are around the call: eagerly an
    eager pullback, under another transformation one it transforms. The pairing
    is [<u, v> = Re Σ u v], unconjugated on complex values, so each transpose is
    the plain transpose of its operation's real-linear map in that pairing.

    A variable's cotangent is the sum of its contributions in the order the walk
    makes them, with no zero before the first: a single contribution is itself,
    sign of zero included. The contributions through a map, a reduction, a
    movement, a pad and a window keep a zero's sign; a gather's transpose adds
    to [+0], and a broadcast's sums its copies as [Nx.sum] does, from [+0]. *)

type t
(** The type for tapes. *)

val linearize :
  name:string ->
  Program.value array ->
  (Program.value array -> Program.value array) ->
  Program.value array * t
(** [linearize ~name xs f] is the primal outputs of [f xs], and the tape from
    the tangents of [xs]'s float and complex values, in order, to the tangents
    of [f]'s outputs. [name] names the entry point in messages. Raises what [f]
    raises. *)

type ct = {
  v : Program.value;
  fresh : bool;  (** Whether [v] is memory of its own, in C order. *)
}
(** The type for cotangents. *)

val transpose :
  ?witness:Program.value -> t -> Program.value option array -> ct option array
(** [transpose ~witness l cts] is the cotangent of each tangent parameter of
    [l], [None] where it receives no contribution, given [cts], the cotangent of
    each output of the linearized function, [None] for zero, each of its
    output's dtype and shape. [witness], a placed value of the brand of the
    function's arguments, gives the set of a cotangent of a value of every set
    that the function placed.

    A [Lanes] over tangents, of the map its axis names, gives the calling lane
    its row of the sum over the lanes of their cotangents.

    Raises [Invalid_argument] naming the tape's entry point for an equation that
    is not linear in its tangents, for another construct over tangents, and for
    a cotangent of a value of every set that lies on a set without [witness]. *)
