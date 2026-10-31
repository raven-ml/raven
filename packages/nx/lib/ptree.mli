(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** [Nx.Ptree]'s walks: paths, cursors, descriptions and the operations derived
    from one walk. [Nx.Ptree] adds those that compute with arrays. This module
    lies above {!Prim}. *)

module Path : sig
  type seg = Field of string | Index of int
  type t

  val root : t
  val v : seg list -> t
  val add : seg -> t -> t
  val segments : t -> seg list
  val equal : t -> t -> bool
  val to_string : t -> string
  val pp : Format.formatter -> t -> unit
end

type report = Int of int | Case of string | Present of bool | Length of int
type visit = Leaf of Path.t | Report of Path.t * report

type ('d, 'e) visitor = {
  tensor : 'v 's. Path.t -> ('v, 's, 'd) Value.t -> ('v, 's, 'e) Value.t;
  report : Path.t -> report -> unit;
}
(** What a walk does at an array of a fixed type and at a report. *)

type ('s, 'd) t = ('d, 'd) visitor -> Path.t -> 's -> 's
(** A description is its walk: [w v p x] walks [x] at [p] with [v]. *)

module Walk : sig
  type ('a, 'b, 'd, 'e) cursor

  val field :
    ('a, 'b, 'd, 'e) cursor ->
    string ->
    (('a, 'b, 'd, 'e) cursor -> 'u -> 'w) ->
    'u ->
    'w

  val index :
    ('a, 'b, 'd, 'e) cursor ->
    int ->
    (('a, 'b, 'd, 'e) cursor -> 'u -> 'w) ->
    'u ->
    'w

  val leaf : ('a, 'b, 'd, 'e) cursor -> 'a -> 'b

  val tensor :
    ('a, 'b, 'd, 'e) cursor -> ('v, 's, 'd) Value.t -> ('v, 's, 'e) Value.t

  val int : ('a, 'b, 'd, 'e) cursor -> int -> int
  val case : ('a, 'b, 'd, 'e) cursor -> string -> unit

  val option :
    (('a, 'b, 'd, 'e) cursor -> 'u -> 'w) ->
    ('a, 'b, 'd, 'e) cursor ->
    'u option ->
    'w option

  val list :
    (('a, 'b, 'd, 'e) cursor -> 'u -> 'w) ->
    ('a, 'b, 'd, 'e) cursor ->
    'u list ->
    'w list

  val under :
    (('p, 'q, 'd, 'e) cursor -> 'u -> 'w) ->
    (('a, 'b, 'd, 'e) cursor -> 'p -> 'q) ->
    ('a, 'b, 'd, 'e) cursor ->
    'u ->
    'w

  val structure : ('s, 'd) t -> ('a, 'b, 'd, 'd) cursor -> 's -> 's
end

module type S = sig
  type ('a, 'd) t

  val walk : ('a, 'b, 'd, 'e) Walk.cursor -> ('a, 'd) t -> ('b, 'e) t
end

val walk :
  (module U : S) ->
  ('a -> 'b) -> ('d, 'e) visitor -> ('a, 'd) U.t -> ('b, 'e) U.t
(** [walk (module U) leaf v x] is [U.walk] of [x] from the root, each position
    of the parameter mapped by [leaf] and the rest visited by [v]. *)

val keep : ('d, 'd) visitor
(** [keep] returns each array and ignores each report. *)

(** {1:structures Structures} *)

val instantiate : (module U : S) -> ((('v, 's, 'd) Value.t, 'd) U.t, 'd) t
val nest : (module U : S) -> ('s, 'd) t -> (('s, 'd) U.t, 'd) t
val v : ('a 'b. ('a, 'b, 'd, 'd) Walk.cursor -> 's -> 's) -> ('s, 'd) t
val tensor : (('v, 's, 'd) Value.t, 'd) t
val unit : (unit, 'd) t
val pair : ('a, 'd) t -> ('b, 'd) t -> ('a * 'b, 'd) t
val option : ('a, 'd) t -> ('a option, 'd) t
val list : ('a, 'd) t -> ('a list, 'd) t

(** {1:derived Maps and folds} *)

val map :
  ('s, 'd) t ->
  ('v 'r. Path.t -> ('v, 'r, 'd) Value.t -> ('v, 'r, 'd) Value.t) ->
  's ->
  's

val zip :
  by:string ->
  ('s, 'd) t ->
  ('v 'r.
   Path.t ->
   ('v, 'r, 'd) Value.t ->
   ('v, 'r, 'd) Value.t ->
   ('v, 'r, 'd) Value.t) ->
  's ->
  's ->
  's
(** [zip ~by s f x y] is [Nx.Ptree.map2 s f x y], its messages naming [by]. *)

val map2 :
  ('s, 'd) t ->
  ('v 'r.
   Path.t ->
   ('v, 'r, 'd) Value.t ->
   ('v, 'r, 'd) Value.t ->
   ('v, 'r, 'd) Value.t) ->
  's ->
  's ->
  's

val fold :
  ('s, 'd) t ->
  ('v 'r. Path.t -> ('v, 'r, 'd) Value.t -> 'acc -> 'acc) ->
  's ->
  'acc ->
  'acc

val visits : ('s, 'd) t -> 's -> visit list
val pp_visit : Format.formatter -> visit -> unit

val pp_path : Format.formatter -> Path.t -> unit
(** [pp_path] formats a path, the root as [the root]. *)

(** {1:payload Payloads} *)

module Payload : sig
  val map :
    (module U : S) -> (Path.t -> 'a -> 'b) -> ('a, 'd) U.t -> ('b, 'd) U.t

  val map2 :
    (module U : S) ->
    (Path.t -> 'a -> 'b -> 'c) -> ('a, 'd) U.t -> ('b, 'd) U.t -> ('c, 'd) U.t

  val fold :
    (module U : S) ->
    (Path.t -> 'a -> 'acc -> 'acc) -> ('a, 'd) U.t -> 'acc -> 'acc
end

(** {1:signatures Signatures} *)

type (_, 'd) fn = private
  | Returns : ('a, 'd) t -> ('a, 'd) fn
  | Arg : ('a, 'd) t * ('b, 'd) fn -> ('a -> 'b, 'd) fn

val ( @-> ) : ('a, 'd) t -> ('b, 'd) fn -> ('a -> 'b, 'd) fn
val returns : ('a, 'd) t -> ('a, 'd) fn

(** {1:flattening Flattening} *)

type 'd packed = P : ('v, 's, 'd) Value.t -> 'd packed

module Skeleton : sig
  type t

  val equal : t -> t -> bool
  val hash : t -> int
  val diff : this:string -> t -> that:string -> t -> string option
end

val flatten : ('s, 'd) t -> 's -> 'd packed list * Skeleton.t
val rebuild : ('s, 'd) t -> like:'s -> 'd packed list -> 's
