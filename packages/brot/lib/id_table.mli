(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tables indexed by token id.

    {b Internal module.} A table maps every id to a value: the value it was
    given for the ids it holds, and its {e absent} value for every other id.
    The ids of a vocabulary usually run from 0 with few gaps, but may leave any
    gaps. A table keeps its ids below its {!dense} length in an array, and the
    rest in sorted arrays searched by bisection, so its memory follows the
    number of ids it holds rather than the largest. The dense length is the
    largest n such that id n - 1 is held and \[0, n) holds at least n / 2
    ids, or 0 when no such n exists. A table holds no negative id.

    A vocabulary's ids lie in \[0, 2{^ 31}): brot hands ids out as [int32]
    values ([Brot.ids]). *)

type 'a t
(** The type for tables. *)

val of_list : absent:'a -> (int * 'a) list -> 'a t
(** [of_list ~absent pairs] holds the ids of [pairs], each with the last value
    [pairs] gives it, and [absent] for every other id. The ids must not be
    negative. *)

val in_range : int -> bool
(** [in_range id] is [true] iff [0 <= id < 2]{^ 31}. *)

val of_vocab : who:string -> (string, int) Hashtbl.t -> string t
(** [of_vocab ~who vocab] holds each id of [vocab] with its token, and [""]
    for the other ids. Raises [Invalid_argument], naming [who], if an id is
    not {!in_range}. *)

val of_array : absent:'a -> 'a array -> 'a t
(** [of_array ~absent a] holds the ids below [Array.length a], id [i] with
    [a.(i)]. It is [a] itself as its dense part: [a] must not be mutated. *)

val get : 'a t -> int -> 'a
(** [get t id] is the value of [id] in [t]. *)

val dense : 'a t -> 'a array
(** [dense t] is the value of each id below its length, absent ones included.
    Owned by [t]; do not mutate. *)

val mapi : absent:'b -> (int -> 'a -> 'b) -> 'a t -> 'b t
(** [mapi ~absent f t] holds the ids [t] holds, each id [i] of value [v] with
    [f i v], and [absent] for the others. The ids below [t]'s dense length count
    as held. *)

val iteri : (int -> 'a -> unit) -> 'a t -> unit
(** [iteri f t] calls [f id v] for each id below the dense length and each
    other id [t] holds, in increasing order of ids. *)

val to_list : 'a t -> (int * 'a) list
(** [to_list t] is the pairs {!iteri} visits, in its order. *)
