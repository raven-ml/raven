(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The implementation of {!Norn.Ensemble}. *)

type ('f, 'd) stats = { lp : (float, 'f, 'd) Nx.t; acceptance : (float, 'f, 'd) Nx.t }

type ('u, 'f, 'd) state = private {
  position : 'u;
  lp : (float, 'f, 'd) Nx.t;
  stats : ('f, 'd) stats;
  draw : 'd Nx.int32_t;
  ensembles : int;
}

val ptree : ('u, 'd) Nx.Ptree.t -> (('u, 'f, 'd) state, 'd) Nx.Ptree.t

val stats : (float, 'f) Nx.dtype -> (('f, 'd) stats, 'd) Nx.Ptree.t

val init :
  ('u, 'd) Nx.Ptree.t ->
  ?ensembles:int ->
  ('u -> (float, 'f, 'd) Nx.t) ->
  'u ->
  ('u, 'f, 'd) state

val step :
  ('u, 'd) Nx.Ptree.t ->
  ('u -> (float, 'f, 'd) Nx.t) ->
  'd Nx.Rng.key ->
  ('u, 'f, 'd) state ->
  ('u, 'f, 'd) state

val warmup :
  ('u, 'd) Nx.Ptree.t ->
  ('u -> (float, 'f, 'd) Nx.t) ->
  'd Nx.Rng.key ->
  steps:int ->
  ('u, 'f, 'd) state ->
  ('u, 'f, 'd) state

val sample :
  ('u, 'd) Nx.Ptree.t ->
  ('u -> (float, 'f, 'd) Nx.t) ->
  'd Nx.Rng.key ->
  draws:int ->
  ('u, 'f, 'd) state ->
  ('u, 'f, 'd) state * 'u Draws.t * ('f, 'd) stats Draws.t
