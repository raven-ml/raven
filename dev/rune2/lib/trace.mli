(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Interpretations that record a function as a program.

    A {e tracer} is an [Nx.Prim] interpretation whose traced values are a
    program's variables ({!Program}), with a handler of constructs just outside
    it, so that what the handler computes goes to the interpretations around the
    tracer. {!trace} runs a function once over variables of the forms given and
    returns its program and its captures. The trace's {e extent} is the code
    that runs while the function runs. A tracer {e receives} an operation or a
    construct when it takes it instead of letting it run, to record it as an
    equation or, for a call or a custom rule, to run its default inside the
    trace. The four kinds ({!kind}) differ in what they receive.

    Every kind receives an operator ({!Program.Operator}), whatever it reads. A
    map's [Lanes] and [Lane_index] reach a tracer only inside that map. Outside
    it their default raises. The jit and staged tracers answer
    {!Program.Staging} with [true]. *)

type kind =
  | Body
      (** The tracer of a construct's functions. It receives the operations and
          constructs that read its variables. One that reads none runs at the
          call, under the interpretations around the tracer, so a body holds
          only what depends on its parameters. A call over its variables is
          recorded as an equation, which runs its default ({!Program.Call})
          where the program is evaluated. *)
  | Jit
      (** The tracer of a compiled function. It receives every operation and
          construct of its extent, those over constants included, and inlines
          each call. The call runs its default where it was performed
          ({!Program.Evaluate}), which issues the call's equations to the
          tracer. *)
  | Tape
      (** The tracer of the linear part of a function whose other part runs at
          the call, as linearization and reverse mode need. What reads none of
          its variables runs at the call, and the rest is recorded over the
          values it reads. A loop, a switch, a remat or a call over its
          variables is split in two the same way, a call's two parts derived at
          its site ({!Program.derive}). A custom rule over its variables runs
          its primal where it was performed, so that its operations are
          recorded. The tangents of a pullback ({!Program.Linear} without
          [apply]) that read none of its variables are zeros. *)
  | Staged
      (** The tracer of a function whose every operation is recorded, such as a
          remat's function, a switch's branches or a root's [solve]. It receives
          every operation and construct of its extent, as the jit tracer does,
          and records each call as an equation. *)

(** {1:tracing Tracing}

    {e Captures.} A value of another interpretation that an equation or an
    output reads is a {e capture}. The body, staged and tape tracers bind each
    to a parameter after the function's arguments, once per value, and return
    it, so that the construct names it as an operand and its owner sees it. The
    jit tracer raises [Invalid_argument] for one, since a compiled function must
    receive other transformations' values as its arguments.

    {e Totals.} The body, jit and staged tracers thread each total that their
    extent adds to and that no collect inside the extent takes ({!Program.t}).
    They answer {!Program.Running} with a variable holding the total's running
    value where the call is, the running value in being a parameter, and
    {!Program.Replace} by making the value given the running value, which the
    program returns once the function has returned. The tape tracer raises
    [Invalid_argument] where a running value would become one of its variables,
    as for a tangent added to a total.

    {e Draws.} The body, jit and staged tracers run the function under
    [Nx.Rng.refusing], so a draw without a key raises [Invalid_argument] with a
    message naming the tracer. A tape's draws are those of the code around it,
    since its known part runs at the call. *)

val trace :
  ?totals:Program.addition array ->
  kind ->
  name:string ->
  Program.form array ->
  (Program.value array -> Program.value array) ->
  Program.t * Program.value array
(** [trace ~totals k ~name forms f] is the program of [f] over variables of
    [forms], and its captures, in order. The interpretation is named [name] in
    messages, as ["Rune.scan"]. A total's running value in lies where [totals]
    places it. Without that, it lies where the first value asked of it
    ({!Program.Running}) or added to it lies. Raises what [f] raises. *)
