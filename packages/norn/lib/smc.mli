(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The implementation of {!Norn.Smc}. *)

type move = Hmc | Slice

type ('u, 'f, 'd) state = private {
  particles : 'u;
  prior : (float, 'f, 'd) Nx.t;
  likelihood : (float, 'f, 'd) Nx.t;
  beta : (float, 'f, 'd) Nx.t;
  log_evidence : (float, 'f, 'd) Nx.t;
  variance : (float, 'f, 'd) Nx.t;
  steps : 'd Nx.int32_t;
  step_size : (float, 'f, 'd) Nx.t;
  length : (float, 'f, 'd) Nx.t;
  length_moment : (float, 'f, 'd) Nx.t;
  resampled : int;
  move : move;
}

val ptree : ('u, 'd) Nx.Ptree.t -> (('u, 'f, 'd) state, 'd) Nx.Ptree.t

val init :
  ('u, 'd) Nx.Ptree.t ->
  ?move:move ->
  ?resampled:int ->
  prior:('u -> (float, 'f, 'd) Nx.t) ->
  likelihood:('u -> (float, 'f, 'd) Nx.t) ->
  'u ->
  ('u, 'f, 'd) state

val step :
  ('u, 'd) Nx.Ptree.t ->
  prior:('u -> (float, 'f, 'd) Nx.t) ->
  likelihood:('u -> (float, 'f, 'd) Nx.t) ->
  'd Nx.Rng.key ->
  ('u, 'f, 'd) state ->
  ('u, 'f, 'd) state

val evidence : ('u, 'f, 'd) state -> ('u, 'f, 'd) Evidence.t

val run :
  ('u, 'd) Nx.Ptree.t ->
  ?move:move ->
  ?resampled:int ->
  budget:int ->
  prior:('u -> (float, 'f, 'd) Nx.t) ->
  likelihood:('u -> (float, 'f, 'd) Nx.t) ->
  'd Nx.Rng.key ->
  'u ->
  ('u, 'f, 'd) Evidence.t
