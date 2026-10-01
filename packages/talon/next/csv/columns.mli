(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A batch's fields read as columns, in Arrow's layouts.

    A reader reads one column of a {!Scan.t}'s batch, the same field of a range
    of its records, by its type's text form ({!Text}), into fresh nx buffers in
    the storage of the type. A field is null when it is unquoted and empty or
    equal to a null token. Text is checked as UTF-8 here, where a failure has a
    line and a column. *)

(** The type for columns. A row's value is zero, or the empty byte string, under
    a null. *)
type t =
  | Fixed of { valid : Nx.bool_t option; values : Nx.packed }
      (** One value per row, in the storage of the column's type: [bool],
          integers of the type's width, [float16] to [float64], [int64] unscaled
          decimals, [int32] days, [int64] ticks, [int32] codes of a categorical.
          [valid], a byte validity mask, is [true] at the rows that hold a
          value, and [None] when every row does. *)
  | Varsize of {
      valid : Nx.bool_t option;
      offsets : Nx.int64_t;
      data : Nx.uint8_t;
    }
      (** Byte strings, for [string] and [binary]: row [i] is [data] from
          [offsets.{i}] to [offsets.{i + 1}], and [offsets.{0}] is [0]. Quoted
          fields' doubled quotes are undoubled. *)

exception Invalid of { row : int; reason : string }
(** [Invalid {row; reason}] is raised for the field of record [row] of the batch
    that is not its type's text or holds a value the type does not, for
    {!Text.Invalid}'s [reason]. *)

val reads : Talon_next.Type.any -> bool
(** [reads t] is [true] iff CSV reads [t]: [bool], an integer, float or decimal
    type, [string], [binary], a categorical, [date] or a [datetime]. *)

val is_null : string list -> Scan.t -> int -> int -> bool
(** [is_null nulls s r j] is [true] iff field [j] of record [r] of [s]'s batch
    is null: unquoted, and empty or one of the null tokens [nulls]. *)

type reader
(** The type for column readers: a type, the dialect's quote and null tokens,
    and a categorical's index of its dictionary. *)

val reader : quote:char -> nulls:string list -> Talon_next.Type.any -> reader
(** [reader ~quote ~nulls t] reads fields as [t], with the null tokens [nulls],
    undoubling [quote] in quoted text. [reads t] holds. *)

val read : reader -> Scan.t -> int -> first:int -> rows:int -> t
(** [read c s j ~first ~rows] is field [j] of the records [first] to
    [first + rows - 1] of [s]'s batch, read by [c]. Each of these records has a
    field [j].

    Raises {!Invalid} at the first of these records whose field does not read as
    [c]'s type. *)
