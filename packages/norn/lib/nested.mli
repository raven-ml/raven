(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The implementation of {!Norn.Nested}. *)

type ('u, 'f, 'd) state = private {
  live : 'u;
  prior : (float, 'f, 'd) Nx.t;
  likelihood : (float, 'f, 'd) Nx.t;
  rank : (float, 'f, 'd) Nx.t;
  lineage : 'd Nx.int32_t;
  dead : 'u;
  dead_likelihood : (float, 'f, 'd) Nx.t;
  shrinkage : (float, 'f, 'd) Nx.t;
  dead_lineage : 'd Nx.int32_t;
  deaths : 'd Nx.int32_t;
  log_volume : (float, 'f, 'd) Nx.t;
  log_evidence : (float, 'f, 'd) Nx.t;
  steps : 'd Nx.int32_t;
  tolerance : (float, 'f, 'd) Nx.t;
  batch : int;
  budget : int;
}

val ptree : ('u, 'd) Nx.Ptree.t -> (('u, 'f, 'd) state, 'd) Nx.Ptree.t

val init :
  ('u, 'd) Nx.Ptree.t ->
  ?batch:int ->
  ?tolerance:float ->
  budget:int ->
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

val evidence :
  ('u, 'd) Nx.Ptree.t -> 'd Nx.Rng.key -> ('u, 'f, 'd) state -> ('u, 'f, 'd) Evidence.t

val run :
  ('u, 'd) Nx.Ptree.t ->
  ?batch:int ->
  ?tolerance:float ->
  budget:int ->
  prior:('u -> (float, 'f, 'd) Nx.t) ->
  likelihood:('u -> (float, 'f, 'd) Nx.t) ->
  'd Nx.Rng.key ->
  'u ->
  ('u, 'f, 'd) Evidence.t
