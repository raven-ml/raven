(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tables, as plans see them.

    A table is named, typed columns of equal length. Plans read a table's schema
    and row count when a verb is applied and compare tables with {!equal}; they
    never read its columns. [Talon_next] exports this type as [Talon_next.t].

    This module is internal. *)

type t
(** The type for tables. *)

val schema : t -> Schema.t
(** [schema t] is the names and types of [t]'s columns, in order. *)

val rows : t -> int
(** [rows t] is the number of rows of [t]. *)

val equal : t -> t -> bool
(** [equal t0 t1] is [true] iff [t0] and [t1] have equal schemas and the same
    keys row by row, by key identity ({!Type.compare_value}, null being one more
    key), whatever their batches. *)
