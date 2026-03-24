(**************************************************************************)
(*                                                                        *)
(*  Copyright (C) Jean-Christophe Filliatre                               *)
(*                                                                        *)
(*  This software is free software; you can redistribute it and/or        *)
(*  modify it under the terms of the GNU Library General Public           *)
(*  License version 2.1, with the special exception on linking            *)
(*  described in file LICENSE.                                            *)
(*                                                                        *)
(*  This software is distributed in the hope that it will be useful,      *)
(*  but WITHOUT ANY WARRANTY; without even the implied warranty of        *)
(*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                  *)
(*                                                                        *)
(**************************************************************************)

(* Vendored from https://github.com/backtracking/hashcons
   Modifications:
   - Removed the non-functorial generic interface ([create], [hashcons],
     etc. at the top level) — we only use the [Make] functor.
   - Removed [Hmap] and [Hset] — we use [Ref_tbl] (keyed by integer tag)
     instead of Patricia-tree maps/sets. *)

(*s Hash tables for hash consing.

    The technique is described in this paper:
      Sylvain Conchon and Jean-Christophe Filliâtre.
      Type-Safe Modular Hash-Consing.
      In ACM SIGPLAN Workshop on ML, Portland, Oregon, September 2006.
      https://www.lri.fr/~filliatr/ftp/publis/hash-consing2.pdf

    Hash consed values are of the
    following type [hash_consed]. The field [tag] contains a unique
    integer (for values hash consed with the same table). The field
    [hkey] contains the hash key of the value (without modulo) for
    possible use in other hash tables (and internally when hash
    consing tables are resized). The field [node] contains the value
    itself.

    Hash consing tables are using weak pointers, so that values that are no
    more referenced from anywhere else can be erased by the GC. *)

type +'a hash_consed = private {
  hkey: int;
  tag : int;
  node: 'a;
}

val gentag_peek : unit -> int

(*s Functorial interface. *)

module type HashedType =
  sig
    type t
    val equal : t -> t -> bool
    val hash : t -> int
  end

module type S =
  sig
    type key
    type t
    val create : int -> t
    val clear : t -> unit
    val hashcons : t -> key -> key hash_consed
    val iter : (key hash_consed -> unit) -> t -> unit
    val stats : t -> int * int * int * int * int * int
  end

module Make(H : HashedType) : (S with type key = H.t)
