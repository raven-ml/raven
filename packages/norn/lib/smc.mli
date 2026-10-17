(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The implementation of {!Norn.Smc}. *)

type move = Hmc | Slice

type ('u, 'f) state = private {
  particles : 'u;
  prior : (float, 'f) Nx.t;
  likelihood : (float, 'f) Nx.t;
  beta : (float, 'f) Nx.t;
  log_evidence : (float, 'f) Nx.t;
  variance : (float, 'f) Nx.t;
  steps : Nx.int32_t;
  step_size : (float, 'f) Nx.t;
  length : (float, 'f) Nx.t;
  length_moment : (float, 'f) Nx.t;
  resampled : int;
  move : move;
}

val ptree : 'u Nx.Ptree.t -> ('u, 'f) state Nx.Ptree.t

val init :
  'u Nx.Ptree.t ->
  ?move:move ->
  ?resampled:int ->
  prior:('u -> (float, 'f) Nx.t) ->
  likelihood:('u -> (float, 'f) Nx.t) ->
  'u ->
  ('u, 'f) state

val step :
  'u Nx.Ptree.t ->
  prior:('u -> (float, 'f) Nx.t) ->
  likelihood:('u -> (float, 'f) Nx.t) ->
  Nx.Rng.t ->
  ('u, 'f) state ->
  ('u, 'f) state

val evidence : ('u, 'f) state -> ('u, 'f) Evidence.t

val run :
  'u Nx.Ptree.t ->
  ?move:move ->
  ?resampled:int ->
  budget:int ->
  prior:('u -> (float, 'f) Nx.t) ->
  likelihood:('u -> (float, 'f) Nx.t) ->
  Nx.Rng.t ->
  'u ->
  ('u, 'f) Evidence.t
