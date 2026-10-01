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

(** {1:print Printing} *)

val pp : 'a Type.t -> Format.formatter -> 'a -> unit
(** [pp ty ppf v] writes [v]'s canonical text, which {!parse} reads back to [v]:
    a float in the fewest significant digits that round to it at [ty]'s width,
    without an exponent from [1e-7] up to [1e21] ([150], [0.0015], [1e+21]), or
    [nan], [inf] or [-inf]; a decimal with [ty]'s scale of digits after the
    point; a datetime with the fewest fraction digits that are exact, and [Z]
    when [ty] has a zone. Text and byte strings are written as they are. Types
    that {!parse} does not read write as {!Type.pp_lit} writes their values. *)

val pp_cell : Column.t -> int -> Format.formatter -> unit
(** [pp_cell c i ppf] writes row [i] of [c] alone, as messages show a key: [∅]
    for a null, text and byte strings quoted as {!Type.pp_quoted} quotes them,
    other values in their canonical text, and a stored value outside the OCaml
    type as its storage: [int64] beyond OCaml's [int] in full, ticks outside
    {!Time}'s range as [datetime[ns] tick 42]. *)

val pp_floats : float array -> Format.formatter -> float -> unit
(** [pp_floats xs] is the printer of a table column that shows the floats [xs]:
    one number of decimals for all, the fewest, up to six, that give each of
    [xs] six significant digits, or scientific notation with five decimals when
    one of [xs] needs more than six decimals or is [10{^16}] or more in
    magnitude; [nan], [inf] and [-inf]. *)
