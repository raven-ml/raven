(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The implementation of {!Norn.Draws}. *)

type 'u t = private 'u

val v : ('u, 'd) Nx.Ptree.t -> 'u -> 'u t

val ptree : ('u, 'd) Nx.Ptree.t -> ('u t, 'd) Nx.Ptree.t

val map : ('a, 'd) Nx.Ptree.t -> ('b, 'd) Nx.Ptree.t -> ('a -> 'b) -> 'a t -> 'b t

val simulate :
  ('a, 'd) Nx.Ptree.t ->
  ('b, 'd) Nx.Ptree.t ->
  ('d Nx.Rng.key -> 'a -> 'b) ->
  'd Nx.Rng.key ->
  'a t ->
  'b t

val append : ('u, 'd) Nx.Ptree.t -> 'u t -> 'u t -> 'u t

val thin : ('u, 'd) Nx.Ptree.t -> every:int -> 'u t -> 'u t
