(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The implementation of {!Norn.Nested}. *)

type ('u, 'f) state = private {
  live : 'u;
  prior : (float, 'f) Nx.t;
  likelihood : (float, 'f) Nx.t;
  rank : (float, 'f) Nx.t;
  lineage : Nx.int32_t;
  dead : 'u;
  dead_likelihood : (float, 'f) Nx.t;
  shrinkage : (float, 'f) Nx.t;
  dead_lineage : Nx.int32_t;
  deaths : Nx.int32_t;
  log_volume : (float, 'f) Nx.t;
  log_evidence : (float, 'f) Nx.t;
  steps : Nx.int32_t;
  tolerance : (float, 'f) Nx.t;
  batch : int;
  budget : int;
}

val ptree : 'u Nx.Ptree.t -> ('u, 'f) state Nx.Ptree.t

val init :
  'u Nx.Ptree.t ->
  ?batch:int ->
  ?tolerance:float ->
  budget:int ->
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

val evidence :
  'u Nx.Ptree.t -> Nx.Rng.t -> ('u, 'f) state -> ('u, 'f) Evidence.t

val run :
  'u Nx.Ptree.t ->
  ?batch:int ->
  ?tolerance:float ->
  budget:int ->
  prior:('u -> (float, 'f) Nx.t) ->
  likelihood:('u -> (float, 'f) Nx.t) ->
  Nx.Rng.t ->
  'u ->
  ('u, 'f) Evidence.t
