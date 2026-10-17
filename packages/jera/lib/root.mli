(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Zeros of scalar functions: the implementation of {!Jera.Root}, with the
    bracketing search that {!Ode.event} runs on its own values. *)

val bracket :
  tol:Tol.t ->
  ((float, 'b) Nx.t -> (float, 'b) Nx.t) ->
  lo:(float, 'b) Nx.t ->
  hi:(float, 'b) Nx.t ->
  (float, 'b) Nx.t Solution.t

val newton :
  tol:Tol.t ->
  budget:int ->
  slope:((float, 'b) Nx.t -> (float, 'b) Nx.t) ->
  ((float, 'b) Nx.t -> (float, 'b) Nx.t) ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t Solution.t

val locate :
  tol:Tol.t ->
  ((float, 'b) Nx.t -> (float, 'b) Nx.t) ->
  (float, 'b) Nx.t * (float, 'b) Nx.t ->
  (float, 'b) Nx.t * (float, 'b) Nx.t ->
  (((float, 'b) Nx.t * (float, 'b) Nx.t)
  * ((float, 'b) Nx.t * (float, 'b) Nx.t))
  * (int32, Nx.int32_elt) Nx.t
  * (int32, Nx.int32_elt) Nx.t
(** [locate ~tol f (a0, fa0) (b0, fb0)] is {!bracket}'s search on detached
    values, elementwise, from the ends [a0 <= b0] where [f] is [fa0] and
    [fb0]: the final bracket [(a, b)] and [f] at its ends, the status codes
    ({!Solution.code}) and the evaluations, the two given ends counted. An
    element still searching after [2 bits + 1] steps, [bits] the dtype's
    width, ends [Stalled]. *)
