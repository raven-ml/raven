(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Stochastic differential equations: the implementation of {!Jera.Sde}. *)

module Brownian : sig
  type ('b, 'd) t

  val v :
    'd Nx.Rng.key ->
    (float, 'b) Nx.dtype ->
    shape:int array ->
    t0:float ->
    t1:float ->
    depth:int ->
    ('b, 'd) t

  val increment :
    ('b, 'd) t ->
    (float, 'b, 'd) Nx.t ->
    (float, 'b, 'd) Nx.t ->
    (float, 'b, 'd) Nx.t * (float, 'b, 'd) Nx.t

  val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> ('f, 'd) t -> ('f, 'e) t
end

type t

val euler_maruyama : t
val milstein : t
val sra1 : t
val reversible_heun : t

val march :
  ('y, 'd) Nx.Ptree.t ->
  t ->
  steps:int ->
  drift:((float, 't, 'd) Nx.t -> 'y -> 'y) ->
  diffusion:((float, 't, 'd) Nx.t -> 'y -> (float, 't, 'd) Nx.t -> 'y) ->
  ('t, 'd) Brownian.t ->
  at:(float, 't, 'd) Nx.t ->
  'y ->
  'y
