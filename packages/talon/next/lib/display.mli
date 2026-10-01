(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Display of tables.

    [Talon_next.pp_with] documents the display of tables. This module holds the
    layout only: the header, the elision of rows and columns, alignment and
    widths. The text of each cell is {!Form}'s. *)

type limits = { head : int; tail : int; columns : int; width : int }

val limits : limits
val pp : limits -> Format.formatter -> Table.t -> unit
