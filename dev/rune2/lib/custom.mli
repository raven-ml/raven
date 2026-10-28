(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Custom differentiation rules: a function with the derivative its caller
    gives.

    A rule is traced at the call with the body tracer ({!Trace.Body}): its
    {e primal} computes the results, and the derivative it returns, a tangent
    map or a pullback, is traced inside that trace, so that what it reads of the
    primal's values are the primal's further outputs, its {e residuals}. The
    call is a construct ({!Program.Custom}); with no differentiation around, it
    is its primal evaluated. *)

open Program

val jvp_call :
  name:string ->
  value array ->
  (value array -> value array * (value array -> value array)) ->
  value array
(** [jvp_call ~name args rule] is the results of [rule args], whose tangents are
    those [snd (rule args)] maps the arguments' tangents to, one per argument.
    [name] names the entry point. Raises what [rule] and its tangent map raise,
    at the trace. *)

val vjp_call :
  name:string ->
  value array ->
  (value array -> value array * (value array -> value array)) ->
  value array
(** [vjp_call ~name args rule] is the results of [rule args], whose cotangents
    [snd (rule args)] maps to the arguments', one per argument. Raises as
    {!jvp_call} does. *)

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
