(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Views: the values of the keys a figure reads. *)

module Scale := Hugin_next_kit.Scale

(** {1:keys Keys} *)

type _ sort =
  | Number : float sort
  | Choice : string sort
  | Interval : (float * float) option sort
  | Zoom : 'd Scale.kind -> ('d * 'd) option sort

type ident = User of string | Zoom_of of { scale : string; at : Common.id }
type 'a key = { ident : ident; sort : 'a sort; init : 'a }

val equal_ident : ident -> ident -> bool
val pp_ident : Format.formatter -> ident -> unit
val equal_sort : 'a sort -> 'b sort -> ('a, 'b) Type.eq option
val equal_key : 'a key -> 'b key -> ('a, 'b) Type.eq option
val number : string -> init:float -> float key
val choice : string -> init:string -> string key

val interval :
  string -> init:(float * float) option -> (float * float) option key

val zoom : ?at:Common.id -> 'd Scale.t -> ('d * 'd) option key

(** {1:views Views} *)

type value = V : 'a sort * 'a -> value

type t = (ident * value) list
(** Bindings in increasing order of their idents. *)

val empty : t
val set : 'a key -> 'a -> t -> t
val get : 'a key -> t -> 'a
val equal : t -> t -> bool
val pp : Format.formatter -> t -> unit
