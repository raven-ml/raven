(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** [Talon.Expr.hash]'s kernel.

    {!Talon.Expr.hash} documents the definition of the hash of each type.
    Binding admits only those types, and {!column} is undefined on the others.
*)

val column : seed:int -> Column.t -> Column.t
(** [column ~seed c] is the [int64] column of the hash of each row of [c]. It
    has no null.

    Raises [Invalid_argument] naming [Expr.hash] when nx cannot read a traced
    column's bytes ({!Talon.Query.section-compiled}). *)
