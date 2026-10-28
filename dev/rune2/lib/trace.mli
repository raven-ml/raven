(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tracers: a function run once over variables, recorded as a program.

    A tracer is an [Nx.Prim] interpretation whose traced values are a program's
    variables ({!Program}), with a handler of constructs just outside it. The
    body and jit tracers run the function under [Nx.Rng.refusing], so a draw
    without a key raises naming the tracer.

    - The {e body} tracer ([Values]) traces a construct's functions. It receives
      the operations and constructs over its variables; one that reads none of
      them runs at the call, under the interpretations around the tracer, so a
      body holds only what depends on its parameters.
    - The {e jit} tracer ([Extent]) traces a compiled function. It receives
      every operation and construct of its extent, those over constants
      included, and answers {!Program.Staging}, so that a compiled call in its
      extent is inlined.
    - The {e tape} tracer ([Values]) records the linear part of a function whose
      other part computes at the call, as a forward-mode derivative over its
      variables does: of a loop it receives, the outputs that read none of its
      variables compute at the call, outward, over the loads they read, and the
      others are recorded over the loads they read, which hold the values they
      compute again. Of a scan over its variables, a known scan computes at the
      call what reads none of them, stacking those values the rest reads that
      are not its rows or captures, and the rest is recorded as a linear scan
      over them. It answers only the constructs over its variables, and raises
      [Invalid_argument] for an addition of one to a total. A draw without a key
      in its function is the scope's.

    The body and jit tracers turn each addition to a total made in their extent
    and not taken by a scope inside it into an output, and answer
    {!Program.Running} with [None]; the tape tracer refuses a {!Program.Replace}
    of one of its variables, as an addition. Each tracer leaves a call
    ({!Program.Call}) to its default, which issues the call's equations to it,
    and the tape tracer answers a custom rule ({!Program.Custom}) over its
    variables with {!Program.Evaluate}, so that its primal runs where it was
    performed and is recorded.

    {e Closure conversion.} A value of another interpretation that an equation
    or an output reads is a {e capture}: the body and tape tracers bind it to a
    parameter after the function's, once per value, and return it so the
    construct names it as an operand and its owner sees it. The jit tracer
    raises [Invalid_argument] for one. *)

type kind =
  | Body  (** Traces a construct's functions. *)
  | Jit  (** Traces a compiled function. *)
  | Tape  (** Records a linear part. *)

val trace :
  kind ->
  name:string ->
  Program.form array ->
  (Program.value array -> Program.value array) ->
  Program.t * Program.value array
(** [trace k ~name forms f] is the program of [f] over variables of [forms], and
    its captures, in order. The interpretation is named [name] in messages, as
    ["Rune.scan"]. Raises what [f] raises. *)
