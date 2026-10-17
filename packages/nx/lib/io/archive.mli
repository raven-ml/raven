(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Archives, the implementation of {!Nx_io.Archive}.

    The formats build archives with {!empty} and {!add}, and must keep the
    names they add distinct and non-empty, which {!add} does not check.
    Every error names the public function, as [Nx_io.Archive.of_list]. *)

module Names : Map.S with type key = string
(** Maps keyed by entry names. *)

type t
(** {!Nx_io.Archive.t}. *)

val empty : t
(** [empty] is the archive with no entry. *)

val mem : string -> t -> bool
(** [mem name a] is [true] iff [a] has an entry [name]. *)

val add : string -> Nx.packed -> t -> t
(** [add name x a] is [a] with the entry [name] bound to [x], in place of
    any entry [name] of [a]. *)

val bindings : t -> (string * Nx.packed) list
(** [bindings a] is [a]'s entries, sorted by name. *)

val of_list : (string * Nx.packed) list -> t
val union : t list -> t
val names : t -> string list
val find : string -> t -> Nx.packed option

val tensor :
  shape:int array -> ('a, 'b) Nx.dtype -> string -> t -> ('a, 'b) Nx.t

val float :
  shape:int array -> (float, 'b) Nx.dtype -> string -> t -> (float, 'b) Nx.t

val of_value : 's Nx.Ptree.t -> 's -> t
val to_value : 's Nx.Ptree.t -> like:'s -> t -> 's
