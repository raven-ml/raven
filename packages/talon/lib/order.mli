(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Sort keys.

    [Talon.Order] documents the keys that users write. This interface adds their
    representation, their check against a schema, and their equality and
    formatting. *)

type t = private {
  name : string;  (** The column. *)
  desc : bool;  (** [true] iff the order is descending. *)
  nulls_first : bool;  (** [true] iff nulls come before every value. *)
}

val asc : string -> t
val desc : string -> t
val nulls_first : t -> t

(** {1:checking Checking} *)

val check : t list -> Schema.t -> (t * Problem.t) list
(** [check ks s] is the problems of the keys [ks] over the columns [s], each
    with its key, in the order of [ks]: a key that names no column of [s]
    ({!Problem.missing}), a key that names the column of an earlier key, a key
    on an extension column, which orders only through its declaration, and a key
    on a column whose type contains an extension type ({!Type.has_ext}), which
    has no order. [Query.sort], [Expr.over]'s [~order] and [Source.v]'s
    [~sorted] check their keys with it. *)

(** {1:comparing Comparing and formatting} *)

val equal : t -> t -> bool
(** [equal k0 k1] is [true] iff [k0] and [k1] have the same name, direction and
    null placement. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf k] formats [k] as it is written: [asc "ts"], [desc "delay"],
    [nulls_first (desc "delay")]. *)
