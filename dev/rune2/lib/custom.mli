(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Custom differentiation rules: the calls behind [Rune.custom_jvp],
    [Rune.custom_vjp] and [Rune.custom_transpose].

    A rule's call runs {e in place}, its user code applied to values where the
    call was performed, or {e traced} with the body tracer ({!Trace.Body}),
    which makes it a construct ({!Program.Custom}, {!Program.Linear}). A call
    that a tracer stages ({!Program.staging}) is always traced.

    A call of a rule with a derivative that no tracer stages first asks the
    innermost interpretation around it how the rule runs ({!Program.Rule}). A
    differentiation that tracks an argument, where no interpretation owns the
    arguments' primals ({!Nx.Prim.owner}), answers: the rule runs in place on
    the primals, and its derivative runs where the differentiation applies it
    ({!Jvp}): a tangent map at the call, on values under forward mode and traced
    under reverse mode, and a pullback in each backward pass. Where none answers
    and no interpretation owns an argument, the call is [fst (rule args)] and
    the derivative never runs. Either way the rule and its derivative run under
    [Nx.Rng.refusing], named by the entry point. Every other call is traced.

    A traced rule's {e primal} computes the results, and the derivative it
    returns is traced inside that trace, so that what the derivative reads of
    the primal's values are the primal's further outputs, its {e residuals}.
    With no differentiation around, the construct is its primal evaluated. A
    traced call runs the derivative's code once, whether or not a
    differentiation applies it.

    A linear call runs in place where no interpretation owns an argument: the
    function runs on the arguments and the transpose never runs. Every other
    linear call traces the function and the transpose apart.

    In every value below, [name] is the entry point, in messages. *)

open Program

val jvp_call :
  name:string ->
  value array ->
  (value array -> value array * (value array -> value array)) ->
  value array
(** [jvp_call ~name args rule] is [fst (rule args)], whose tangents are those
    [snd (rule args)] maps the arguments' tangents to, one per argument. Raises
    what [rule] raises, and what its tangent map raises, where each runs. *)

val vjp_call :
  name:string ->
  value array ->
  (value array -> value array * (value array -> value array)) ->
  value array
(** [vjp_call ~name args rule] is [fst (rule args)], whose cotangents
    [snd (rule args)] maps to the arguments', one per argument. Raises what
    [rule] raises, and what its pullback raises, where each runs: a pullback on
    values in the backward pass. *)

val transpose_call :
  name:string ->
  value array ->
  (value array -> value array * (value array -> value array)) ->
  value array
(** [transpose_call ~name args f] is the results of [fst (f args)], a function
    linear in [args] whose transpose is [snd (f args)], from a cotangent per
    result to one per argument. Where the call runs in place, as stated above,
    the function runs under [Nx.Rng.refusing] and the transpose never runs.
    Raises what they raise where they run, and [Invalid_argument] if either adds
    to a total: at the addition where the function runs in place, at the call
    where they are traced. *)

val residual_count : Program.t -> ny:int -> int
(** [residual_count p ~ny] is the number of residuals of a rule's primal [p] of
    [ny] results. *)
