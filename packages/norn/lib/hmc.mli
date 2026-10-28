(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The implementation of {!Norn.Hmc}. *)

type ('u, 'f, 'd) state = private {
  position : 'u;
  lp : (float, 'f, 'd) Nx.t;
  grad : 'u;
  step_size : (float, 'f, 'd) Nx.t;
  length : (float, 'f, 'd) Nx.t;
  geometry : ('u, 'f, 'd) Gaussian.t;
  stats : ('f, 'd) Stats.t;
  draw : 'd Nx.int32_t;
  accept : (float, 'f, 'd) Nx.t;
}

type ('f, 'd) stats = ('f, 'd) Stats.t

val ptree : ('u, 'd) Nx.Ptree.t -> (('u, 'f, 'd) state, 'd) Nx.Ptree.t

val stats : (float, 'f) Nx.dtype -> (('f, 'd) stats, 'd) Nx.Ptree.t

val init :
  ('u, 'd) Nx.Ptree.t ->
  ?accept:float ->
  ?rank:int ->
  ?geometry:('u, 'f, 'd) Gaussian.t ->
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
