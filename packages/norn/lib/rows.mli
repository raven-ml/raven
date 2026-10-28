(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Positions as rows of chains, and the densities over them.

    Every tensor of a position leads with the chain axis, of [c] rows. A
    per-chain scalar is a tensor of shape [[c]]. *)

val invalid_argf : ('a, unit, string, 'b) format4 -> 'a
val shape_string : int array -> string

val float_leaf : ('a, 'b, 'd) Nx.t -> bool
(** [float_leaf x] is [true] if [x] is a float tensor. *)

val element : ('a, 'b, 'd) Nx.t -> string
(** [element v] is the one element of [v], printed as {!Nx.Dtype.pp_value}
    prints it. *)

(** {1:arithmetic Arithmetic over chains} *)

val column : (float, 'f, 'd) Nx.t -> ('a, 'b, 'd) Nx.t -> ('a, 'b, 'd) Nx.t
(** [column h x] is the per-chain scalars [h] at [x]'s dtype, shaped to
    broadcast against [x]'s trailing axes. *)

val dot : ('u, 'd) Nx.Ptree.t -> (float, 'f, 'd) Nx.t -> 'u -> 'u -> (float, 'f, 'd) Nx.t
(** [dot u like a b] is each chain's inner product of [a] and [b] over every
    float tensor, at [like]'s dtype, shape [[c]]. *)

val axpy : ('u, 'd) Nx.Ptree.t -> (float, 'f, 'd) Nx.t -> 'u -> 'u -> 'u
(** [axpy u h x y] is [y + h x], [h] one scalar per chain. *)

val choose : ('u, 'd) Nx.Ptree.t -> 'd Nx.bool_t -> 'u -> 'u -> 'u
(** [choose u mask a b] is [a] in the chains where [mask] holds, [b] elsewhere.
*)

val beside :
  string -> what:string -> ('u, 'd) Nx.Ptree.t -> 'u -> 'd Nx.Placement.t
(** [beside context ~what u x] is {!Nx.beside} of [x]'s first tensor: where a
    value made for [x] lies.

    Raises [Invalid_argument] naming [context] and [what] if [x] has no
    tensor. *)

val whole_chains : string -> reads:string -> ('u, 'd) Nx.Ptree.t -> 'u -> unit
(** [whole_chains context ~reads u x] raises [Invalid_argument] naming
    [context] and the path of a tensor of [x] cut across its devices, saying
    that the sampler [reads] across chains. *)

val finite : ('u, 'd) Nx.Ptree.t -> 'u -> 'd Nx.bool_t
(** [finite u x] is whether every float element of each chain is finite. *)

val add : ('u, 'd) Nx.Ptree.t -> 'u -> 'u -> 'u
val zeros : ('u, 'd) Nx.Ptree.t -> 'u -> 'u

val elements : ('u, 'd) Nx.Ptree.t -> int -> 'u -> int
(** [elements u c x] is the number of float elements of one chain of [x], a
    position of [c] chains. *)

(** {1:flat Flat rows} *)

val ravel : ('u, 'd) Nx.Ptree.t -> (float, 'f, 'd) Nx.t -> 'u -> (float, 'f, 'd) Nx.t
(** [ravel u like x] is the float elements of each chain of [x], in walk order,
    as the rows of a [[c; d]] matrix at [like]'s dtype, [c] the length of
    [like]. *)

val unravel : ('u, 'd) Nx.Ptree.t -> 'u -> (float, 'f, 'd) Nx.t -> 'u
(** [unravel u x m] is the position of [x]'s structure whose float tensors hold
    the rows of [m], at their own dtypes, and whose other tensors are [x]'s. *)

(** {1:densities Densities} *)

val count : string -> ('u, 'd) Nx.Ptree.t -> 'u -> int
(** [count context u x] is the number of chains of [x].

    Raises [Invalid_argument] if [x] has no tensor. *)

val evaluate :
  ('u, 'd) Nx.Ptree.t -> ('u -> (float, 'f, 'd) Nx.t) -> 'u -> (float, 'f, 'd) Nx.t * 'u
(** [evaluate u lp x] is the density [lp] at [x] and its gradient, chain by
    chain. *)

val check_range : string -> (float, 'f, 'd) Nx.t -> unit
(** [check_range context l] raises through {!Nx.check} at a chain whose log
    density [l] is NaN or [+inf]. *)

val check_density :
  string ->
  ('u, 'd) Nx.Ptree.t ->
  ('u -> (float, 'f, 'd) Nx.t) ->
  'u ->
  (float, 'f, 'd) Nx.t ->
  unit
(** [check_density context u lp x l] raises [Invalid_argument] if [l], [lp] at
    [x], is not of shape [[c]], has a NaN or [+inf] row, or changes when the
    chains of [x] are reversed. *)
