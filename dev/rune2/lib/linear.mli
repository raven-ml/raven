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

val trace :
  name:string ->
  Program.value array ->
  (Program.value array -> Program.value array) ->
  t
(** [trace ~name xs f] is [f] recorded as a tape over stand-ins of [xs]'s forms,
    in order: their values are not read, and the tape is [f] itself where [f] is
    linear in them. Raises what [f] raises. *)

val apply : t -> Program.value array -> Program.value option array
(** [apply l ts] evaluates [l] at the tangents [ts] of its parameters: the
    tangent of each output of the linearized function, [None] for zero.

    Raises [Invalid_argument] unless [ts] has one value per parameter. *)

type ct = {
  v : Program.value;
  fresh : bool;  (** Whether [v] is memory of its own, in C order. *)
}
(** The type for cotangents. *)

val transpose : t -> Program.value option array -> ct option array
(** [transpose l cts] is the cotangent of each tangent parameter of [l], [None]
    where it receives no contribution, given [cts], the cotangent of each output
    of the linearized function, [None] for zero, each of its output's dtype and
    shape. A value a transpose makes beside a cotangent, such as the zeros a
    gather's transpose scatters into, lies at its operand's placement, or whole
    on each device of the value it meets.

    A [Lanes] over tangents, of the map its axis names, gives the calling lane
    its row of the sum over the lanes of their cotangents. A scan over tangents
    transposes as a scan of its step's transpose over the rows in reverse, a
    remat by linearizing its function again at its constant arguments, and a
    call as the call of its program's transpose, derived once per derivation at
    its site ({!Program.derive}).

    Raises [Invalid_argument] naming the tape's entry point for an equation that
    is not linear in its tangents, and for an iterate over tangents. *)
