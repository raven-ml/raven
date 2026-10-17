(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Stochastic differential equations: the implementation of {!Jera.Sde}. *)

module Brownian : sig
  type 'b t

  val v :
    Nx.Rng.t ->
    (float, 'b) Nx.dtype ->
    shape:int array ->
    t0:float ->
    t1:float ->
    depth:int ->
    'b t

  val increment :
    'b t ->
    (float, 'b) Nx.t ->
    (float, 'b) Nx.t ->
    (float, 'b) Nx.t * (float, 'b) Nx.t

  val ptree : (float, 'b) Nx.dtype -> 'b t Nx.Ptree.t
end

type t

val euler_maruyama : t
val milstein : t
val sra1 : t
val reversible_heun : t

val march :
  'y Nx.Ptree.t ->
  t ->
  steps:int ->
  drift:((float, 't) Nx.t -> 'y -> 'y) ->
  diffusion:((float, 't) Nx.t -> 'y -> (float, 't) Nx.t -> 'y) ->
  't Brownian.t ->
  at:(float, 't) Nx.t ->
  'y ->
  'y
