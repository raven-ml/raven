(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Reverse mode, which linearizes a function into a tape and transposes the
    tape. This module implements the derivatives of {!Rune2.section-reverse}.

    {e Linearizing} a function runs it under forward mode ({!Jvp}) whose
    tangents are the variables of a tape tracer ({!Trace.Tape}). The primal
    computes at the call, and the operations on tangents are recorded as a
    program linear in them, the {e tape}, whose constants, the primal values
    those operations read, are its {e residuals}. A [Map]'s tangent is recorded
    over the arrays the map reads, its loads ([Nx.Prim.load]), so its residuals
    are those arrays, and its transpose computes the map's intermediates again.

    {e Transposing} the tape walks its equations backward, from cotangents of
    its outputs to cotangents of its tangent parameters. It issues each
    equation's transpose under the interpretations around the call, so that a
    transformation around the call transforms the pullback. Each transpose is
    the plain transpose of its operation's real-linear map in the pairing of
    {!Rune2.section-derivatives}, unconjugated on complex values. A variable's
    cotangent sums its contributions in the order the walk makes them, by the
    rule of that section's sums.

    A tape is made by {!linearize} at values or by {!trace} over stand-ins. Its
    {e parameters} are the tangents it is over, one per float or complex value
    of {!linearize}'s [xs], or one per value of {!trace}'s [xs]. A tape is
    immutable. Where it was made outside every transformation, it can be applied
    ({!apply}) and transposed ({!transpose}) any number of times. *)

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
    in order. It reads none of [xs]'s values. Every operation of [f] is
    recorded, so [f] must be linear in its arguments. {!transpose} raises for an
    operation that is not linear in them, and drops a term that does not depend
    on them. Raises what [f] raises. *)

val apply : t -> Program.value array -> Program.value option array
(** [apply l ts] is the tangent of each output of the linearized function,
    [None] for zero, at the tangents [ts] of [l]'s parameters.

    Raises [Invalid_argument] unless [ts] has one value per parameter. *)

type ct = {
  v : Program.value;
  fresh : bool;  (** Whether [v] is memory of its own, in C order. *)
}
(** The type for cotangents. *)

val transpose : t -> Program.value option array -> ct option array
(** [transpose l cts] is the cotangent of each tangent parameter of [l], given
    [cts], the cotangent of each output of the linearized function. [None] in
    [cts] stands for zero, and [None] in the result for a parameter that
    receives no contribution. Each cotangent must have its output's dtype and
    shape. A value a transpose makes beside a cotangent lies at the placement of
    the operand it stands for, as the zeros a gather's transpose scatters into
    do, or else whole on each device of the cotangent, as the zeros an
    assembly's transpose puts over an overlapped region do.

    Raises [Invalid_argument] naming the tape's entry point for an equation that
    is not linear in its tangents, and for an iterate over tangents. *)
