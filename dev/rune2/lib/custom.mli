(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Custom differentiation rules: a function with the derivative its caller
    gives.

    A call that no tracer stages ({!Program.staging}) first asks the innermost
    interpretation around it how the rule runs ({!Program.Rule}). A
    differentiation that tracks one of the arguments, whose primals no
    interpretation traces, answers: the rule runs on those primals where the
    call was performed, and its derivative runs on values where that
    differentiation applies it ({!Jvp}): a tangent map at the call, a pullback
    in each backward pass. Where no interpretation answers
    so and none traces an argument, the call is [fst (rule args)], and the
    derivative never runs. The rule runs under [Nx.Rng.refusing], named by the
    entry point, in both cases.

    Every other call is {e traced} with the body tracer ({!Trace.Body}): the
    rule's {e primal} computes the results, and the derivative it returns, a
    tangent map or a pullback, is traced inside that trace, so that what it
    reads of the primal's values are the primal's further outputs, its
    {e residuals}. The call is then a construct ({!Program.Custom}); with no
    differentiation around, it is its primal evaluated. A traced call runs the
    derivative's code once, whether or not a differentiation applies it. *)

open Program

val jvp_call :
  name:string ->
  value array ->
  (value array -> value array * (value array -> value array)) ->
  value array
(** [jvp_call ~name args rule] is the results of [rule args], whose tangents are
    those [snd (rule args)] maps the arguments' tangents to, one per argument.
    [name] names the entry point. Raises what [rule] raises, and what its
    tangent map raises, where each runs. *)

val vjp_call :
  name:string ->
  value array ->
  (value array -> value array * (value array -> value array)) ->
  value array
(** [vjp_call ~name args rule] is the results of [rule args], whose cotangents
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
    result to one per argument. The two are traced apart, at the call. Raises
    what they raise, at the trace, and [Invalid_argument] if either adds to a
    total. *)

val residual_count : Program.t -> ny:int -> int
(** [residual_count p ~ny] is the number of residuals of a rule's primal [p] of
    [ny] results. *)
