(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Equal-weight draws from chains.

    Draws of a structure ['u] are a value of ['u] whose every tensor has two
    leading axes, [[chain; draw]], of one length each across tensors. A sampler
    returns them, and {!v} checks a value for them. Chain diagnostics
    ({!Norn.Diag}) take draws, so they apply to equal-weight chains only. *)

type 'u t = private 'u
(** The type for draws of ['u]. [(d :> 'u)] reads draws as a value. *)

val v : 'u Nx.Ptree.t -> 'u -> 'u t
(** [v u x] is [x] as draws.

    Raises [Invalid_argument] if a tensor of [x] has fewer than two axes, or if
    two tensors differ in the length of one of their first two axes, naming the
    path. *)

val ptree : 'u Nx.Ptree.t -> 'u t Nx.Ptree.t
(** [ptree u] is the structure of draws of [u]: [u]'s, with the same paths. *)

val map : 'a Nx.Ptree.t -> 'b Nx.Ptree.t -> ('a -> 'b) -> 'a t -> 'b t
(** [map a b f d] is [f] applied to every draw of [d], a value of ['a] without
    the two leading axes, the results stacked on them. [f] runs under
    {!Rune.val-vmap}, once for all draws. *)

val simulate :
  'a Nx.Ptree.t ->
  'b Nx.Ptree.t ->
  (Nx.Rng.t -> 'a -> 'b) ->
  Nx.Rng.t ->
  'a t ->
  'b t
(** [simulate a b f k d] is [map a b] of [f k_ij] over [d], where the draw [j]
    of chain [i] has the key [Nx.Rng.fold_in k (i * n + j)], [n] being the
    number of draws per chain. *)

val append : 'u Nx.Ptree.t -> 'u t -> 'u t -> 'u t
(** [append u d d'] is the draws of [d] followed, in every chain, by those of
    [d'].

    Raises [Invalid_argument] if [d] and [d'] differ in their visits
    ({!Nx.Ptree.visits}) or in their number of chains. *)

val thin : 'u Nx.Ptree.t -> every:int -> 'u t -> 'u t
(** [thin u ~every d] is every [every]-th draw of each chain of [d], from the
    first.

    Raises [Invalid_argument] if [every < 1]. *)
