(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Record cells.

    [Talon_next.Record] documents record cells. This interface exposes their
    representation, {!Kind.record}. *)

type t = Kind.record

val kind : t Kind.t
val empty : t
val add : 'a Kind.t -> string -> 'a option -> t -> t
val field : 'a Kind.t -> string -> t -> 'a option
val names : t -> string list
