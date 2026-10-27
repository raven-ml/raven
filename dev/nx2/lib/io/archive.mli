(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Archives of named arrays. {!Nx_io.Archive} documents the public part. *)

type 'd t

val empty : 'd t
(** [empty] has no entry. *)

val add : string -> 'd Nx.packed -> 'd t -> 'd t
(** [add name x a] is [a] with the entry [name]. [name] must be non-empty and
    not in [a]. *)

val bindings : 'd t -> (string * 'd Nx.packed) list
(** [bindings a] is [a]'s entries, in name order. *)

val of_list : (string * 'd Nx.packed) list -> 'd t
val union : 'd t list -> 'd t
val names : 'd t -> string list
val find : string -> 'd t -> 'd Nx.packed option

val tensor :
  shape:int array -> ('v, 's) Nx.dtype -> string -> 'd t -> ('v, 's, 'd) Nx.t

val float :
  shape:int array ->
  (float, 's) Nx.dtype ->
  string ->
  'd t ->
  (float, 's, 'd) Nx.t

val of_value : ?prefix:string -> ('s, 'd) Nx.Ptree.t -> 's -> 'd t
val to_value : ?prefix:string -> ('s, 'd) Nx.Ptree.t -> like:'s -> 'e t -> 's
