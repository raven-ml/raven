(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Text forms of values.

    The one home of the text of values. The parsers read a row's bytes in place,
    allocate nothing for a valid text except a float outside the exact fast
    path, and round floats correctly to [float32] and [float16]. *)

val parse : Type.any -> Column.t -> (Column.t, int * string) result
(** [parse ty c] is [Talon_next.Column.parse ty c]: the column of type [ty]
    whose rows are the values that [c]'s rows write, [c] a [string] or [binary]
    column, null where [c] is, or [Error (row, reason)] at the first non-null
    row that is not [ty]'s form or holds a value [ty] does not. [ty] is [bool],
    an integer, float or decimal type, [string] (the bytes, checked as UTF-8),
    [binary], a categorical, [date] or a [datetime].

    Raises [Invalid_argument] if [c] is not a [string] or [binary] column, or
    [ty] is another type. *)
