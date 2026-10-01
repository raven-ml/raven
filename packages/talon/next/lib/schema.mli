(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Schemas: the names and types of a table's columns.

    [Talon_next.Schema] documents schemas. This interface adds {!names}. *)

type t

val v : (string * Type.any) list -> t
val columns : t -> (string * Type.any) list

val names : t -> string list
(** [names s] is the names of [s]'s columns, in order. *)

val find : t -> string -> Type.any option
val equal : t -> t -> bool

type change =
  | Added of string * Type.any
  | Removed of string * Type.any
  | Retyped of string * Type.any * Type.any

val diff : t -> t -> change list
val pp : Format.formatter -> t -> unit
