(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Values in a unit, as {!Ymir_units.Quantity} documents them. *)

type !'p t

val walk : ('a, 'b, 'd, 'e) Nx.Ptree.Walk.cursor -> 'a t -> 'b t

module Structure : Nx.Ptree.S with type ('a, 'd) t = 'a t
val v : Unit.t -> ('a, 'b, 'd) Nx.t -> ('a, 'b, 'd) Nx.t t
val unit : 'p t -> Unit.t
val value : Unit.t -> ('a, 'b, 'd) Nx.t t -> ('a, 'b, 'd) Nx.t
val convert : Unit.t -> ('a, 'b, 'd) Nx.t t -> ('a, 'b, 'd) Nx.t t
val times : Unit.t -> 'p t -> 'p t
val per : Unit.t -> 'p t -> 'p t
val map : (('a, 'b, 'e) Nx.t -> ('c, 'd, 'f) Nx.t) -> ('a, 'b, 'e) Nx.t t -> ('c, 'd, 'f) Nx.t t

val map2 :
  (('a, 'b, 'd) Nx.t -> ('a, 'b, 'd) Nx.t -> ('a, 'b, 'd) Nx.t) ->
  ('a, 'b, 'd) Nx.t t ->
  ('a, 'b, 'd) Nx.t t ->
  ('a, 'b, 'd) Nx.t t

val add : ('a, 'b, 'd) Nx.t t -> ('a, 'b, 'd) Nx.t t -> ('a, 'b, 'd) Nx.t t
val sub : ('a, 'b, 'd) Nx.t t -> ('a, 'b, 'd) Nx.t t -> ('a, 'b, 'd) Nx.t t
val mul : ('a, 'b, 'd) Nx.t t -> ('a, 'b, 'd) Nx.t t -> ('a, 'b, 'd) Nx.t t
val div : ('a, 'b, 'd) Nx.t t -> ('a, 'b, 'd) Nx.t t -> ('a, 'b, 'd) Nx.t t
val pow : int -> ('a, 'b, 'd) Nx.t t -> ('a, 'b, 'd) Nx.t t
val root : int -> (float, 'b, 'd) Nx.t t -> (float, 'b, 'd) Nx.t t
val pp : Format.formatter -> ('a, 'b, 'd) Nx.t t -> unit
