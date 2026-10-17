(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The implementation of {!Norn.Nuts}. *)

type ('u, 'f) state = private {
  position : 'u;
  lp : (float, 'f) Nx.t;
  grad : 'u;
  step_size : (float, 'f) Nx.t;
  geometry : ('u, 'f) Gaussian.t;
  stats : 'f Stats.t;
  draw : Nx.int32_t;
  accept : (float, 'f) Nx.t;
  max_depth : int;
}

val ptree : 'u Nx.Ptree.t -> ('u, 'f) state Nx.Ptree.t

val stats : (float, 'f) Nx.dtype -> 'f Stats.t Nx.Ptree.t

val init :
  'u Nx.Ptree.t ->
  ?max_depth:int ->
  ?accept:float ->
  ?rank:int ->
  ?geometry:('u, 'f) Gaussian.t ->
  ('u -> (float, 'f) Nx.t) ->
  'u ->
  ('u, 'f) state

val step :
  'u Nx.Ptree.t ->
  ('u -> (float, 'f) Nx.t) ->
  Nx.Rng.t ->
  ('u, 'f) state ->
  ('u, 'f) state

val warmup :
  'u Nx.Ptree.t ->
  ('u -> (float, 'f) Nx.t) ->
  Nx.Rng.t ->
  steps:int ->
  ('u, 'f) state ->
  ('u, 'f) state

val sample :
  'u Nx.Ptree.t ->
  ('u -> (float, 'f) Nx.t) ->
  Nx.Rng.t ->
  draws:int ->
  ('u, 'f) state ->
  ('u, 'f) state * 'u Draws.t * 'f Stats.t Draws.t
