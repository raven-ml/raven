(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Values in a unit, as {!Ymir_units.Quantity} documents them. *)

type !'p t

val walk : ('a, 'b) Nx.Ptree.Walk.cursor -> 'a t -> 'b t
val v : Unit.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t t
val unit : 'p t -> Unit.t
val value : Unit.t -> ('a, 'b) Nx.t t -> ('a, 'b) Nx.t
val convert : Unit.t -> ('a, 'b) Nx.t t -> ('a, 'b) Nx.t t
val times : Unit.t -> 'p t -> 'p t
val per : Unit.t -> 'p t -> 'p t
val map : (('a, 'b) Nx.t -> ('c, 'd) Nx.t) -> ('a, 'b) Nx.t t -> ('c, 'd) Nx.t t

val map2 :
  (('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) ->
  ('a, 'b) Nx.t t ->
  ('a, 'b) Nx.t t ->
  ('a, 'b) Nx.t t

val add : ('a, 'b) Nx.t t -> ('a, 'b) Nx.t t -> ('a, 'b) Nx.t t
val sub : ('a, 'b) Nx.t t -> ('a, 'b) Nx.t t -> ('a, 'b) Nx.t t
val mul : ('a, 'b) Nx.t t -> ('a, 'b) Nx.t t -> ('a, 'b) Nx.t t
val div : ('a, 'b) Nx.t t -> ('a, 'b) Nx.t t -> ('a, 'b) Nx.t t
val pow : int -> ('a, 'b) Nx.t t -> ('a, 'b) Nx.t t
val root : int -> (float, 'b) Nx.t t -> (float, 'b) Nx.t t
val pp : Format.formatter -> ('a, 'b) Nx.t t -> unit
