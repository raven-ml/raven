(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The implementation of {!Norn.Draws}. *)

type 'u t = private 'u

val v : 'u Nx.Ptree.t -> 'u -> 'u t

val ptree : 'u Nx.Ptree.t -> 'u t Nx.Ptree.t

val map : 'a Nx.Ptree.t -> 'b Nx.Ptree.t -> ('a -> 'b) -> 'a t -> 'b t

val simulate :
  'a Nx.Ptree.t ->
  'b Nx.Ptree.t ->
  (Nx.Rng.t -> 'a -> 'b) ->
  Nx.Rng.t ->
  'a t ->
  'b t

val append : 'u Nx.Ptree.t -> 'u t -> 'u t -> 'u t

val thin : 'u Nx.Ptree.t -> every:int -> 'u t -> 'u t
