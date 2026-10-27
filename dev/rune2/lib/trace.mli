(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tracers: a function run once over variables, recorded as a program.

    A tracer is an [Nx.Prim] interpretation whose traced values are a program's
    variables ({!Program}), with a handler of constructs just outside it. It
    runs the function under [Nx.Rng.refusing], so a draw without a key raises
    naming the tracer.

    - The {e body} tracer ([Values]) traces a construct's functions. It receives
      the operations and constructs over its variables; one that reads none of
      them runs at the call, under the interpretations around the tracer, so a
      body holds only what depends on its parameters.
    - The {e jit} tracer ([Extent]) traces a compiled function. It receives
      every operation and construct of its extent, those over constants
      included.

    Both turn each addition to a total made in their extent and not taken by a
    scope inside it into an output, and leave a call ({!Program.Call}) to its
    default, which issues the call's equations to them.

    {e Closure conversion.} A value of another interpretation that an equation
    or an output reads is a {e capture}: the body tracer binds it to a parameter
    after the function's, once per value, and returns it so the construct names
    it as an operand and its owner sees it. The jit tracer raises
    [Invalid_argument] for one. *)

type kind =
  | Body  (** Traces a construct's functions. *)
  | Jit  (** Traces a compiled function. *)

val trace :
  kind ->
  name:string ->
  Program.form array ->
  (Program.value array -> Program.value array) ->
  Program.t * Program.value array
(** [trace k ~name forms f] is the program of [f] over variables of [forms], and
    its captures, in order. The interpretation is named [name] in messages, as
    ["Rune.scan"]. Raises what [f] raises. *)
