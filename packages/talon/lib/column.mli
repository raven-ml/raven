(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Columns.

    [Talon.Column] documents columns and their layouts. This interface adds
    their {{!repr}representation}, the {{!codec}codec} that converts OCaml
    values to and from columns by recursion over their type, and the
    {{!structural}structural operations} that every verb uses. *)

type t

type layout =
  | Fixed of { validity : Nx.bit_t option; values : Nx.packed }
  | Varsize of { validity : Nx.bit_t option; offsets : Nx.int64_t; child : t }
  | Children of {
      validity : Nx.bit_t option;
      length : int;
      fields : (string * t) list;
    }

val type_ : t -> Type.any
val length : t -> int
val null_count : t -> int
val v : 'a Type.t -> 'a array -> t
val of_options : 'a Type.t -> 'a option array -> t
val values : 'a Kind.t -> t -> 'a array
val options : 'a Kind.t -> t -> 'a option array
val of_tensor : ?validity:Nx.bit_t -> ('a, 'b) Nx.t -> t
val to_tensor : ('a, 'b) Nx.dtype -> t -> ('a, 'b) Nx.t
val validity : t -> Nx.bit_t option
val ragged : ('a, 'b) Nx.dtype -> t -> ('a, 'b) Nx_ragged.t
val of_ragged : ?validity:Nx.bit_t -> ('a, 'b) Nx_ragged.t -> t
val layout : t -> layout
val of_layout : Type.any -> layout -> (t, int * string) result

(** {1:repr Representation} *)

(** The type for a column's values, by its type's storage:
    - [Fixed x] for [bool] ([Nx.bit], eight values to a byte), the integer and
      float types, categoricals ([int32] codes), dates ([int32] days), clocks,
      durations and datetimes ([int64] ticks), and tensors ([x] of shape
      [(length, …shape)]);
    - [Bytes r] for [string] and [binary], one row of [r] per row;
    - [List] for lists: row [i] is the child's rows [offsets.{i}] to
      [offsets.{i + 1} - 1];
    - [Fields] for records, one child per field in the type's order.

    An extension column has its storage's data. *)
type data =
  | Fixed of Nx.packed
  | Bytes of (int, Nx.uint8_elt) Nx_ragged.t
  | List of { offsets : Nx.int64_t; child : t }
  | Fields of t list

val data : t -> data
(** [data c] is [c]'s values. Under a null they are unspecified and
    deterministic: a function of the operation that made them and of its inputs.
    A reader that needs defined values masks them with {!validity}. *)

(** {2:nulls Nulls}

    A column's validity, when it has one, is its bits and the number of rows
    they leave null, read at the first {!null_count} and kept: a column that
    keeps a validity keeps its count, columns that share one count it once, and
    a column with new bits has a count not yet read. No operation that makes a
    column from other columns reads a count. *)

val known_zero : t -> bool
(** [known_zero c] is [true] iff [c] has no validity or its count was read as
    [0]. It reads nothing: fast paths consult it, and the absence of a validity
    implies no null, not the reverse. *)

val make : Type.any -> ?validity:Nx.bit_t -> length:int -> data -> t
(** [make ty ?validity ~length d] is the column of type [ty] with values [d] and
    a null wherever [validity] is clear, its count not yet read. It checks that
    [d] is [ty]'s storage and that every part has [length] rows, and nothing
    about values: the caller's data is held by [ty]. Every internal operation
    makes columns with it.

    Raises [Invalid_argument] if [d] is not [ty]'s storage, or if a part's
    length or [validity]'s is not [length]. *)

val with_data : Type.any -> data -> t -> t
(** [with_data ty d c] is the column of type [ty] with values [d] and [c]'s
    validity, which it shares, count included: [c]'s values cast to [ty], or
    read as another type of the same storage. As {!make}, it checks that [d] is
    [ty]'s storage for [c]'s rows, and nothing about values.

    Raises [Invalid_argument] if [d] is not [ty]'s storage for [length c] rows.
*)

type mask
(** The type for validities that columns share: bits, and their count read once
    for all of them. *)

val mask : Nx.bit_t -> mask
(** [mask b] is the validity [b], its count not yet read. *)

val restrict : mask -> t -> t
(** [restrict m c] is [c] where it has a validity, and otherwise [c] null where
    [m] is clear, sharing [m]'s count. A gather ({!gather}) clears the validity
    at the rows it pads, so [restrict found (gather indices c)] is null at each
    padded row of every column.

    Raises [Invalid_argument] if [m]'s length is not [c]'s. *)

(** {1:codec Codec}

    The two functions that interpret ['a Type.t]. Every conversion between OCaml
    values and columns goes through them. *)

val encode : 'a Type.t -> int -> (int -> 'a option) -> (t, int * string) result
(** [encode ty n f] is the column of type [ty] whose row [i] is [f i], null
    where it is [None], or [Error (row, reason)] at the first row whose value
    [ty] does not hold ({!Type.holds}). [f] is called once per row, in order, up
    to that row. [reason] is a phrase, as in [int8 does not hold 300]. *)

val decoder : 'a Type.t -> t -> (int -> 'a option, int * string) result
(** [decoder ty c] reads [c] row by row: [get i] is row [i]'s value, or [None]
    if it is null, for [Ok get]. The type is analysed and every row is checked
    once, when [decoder ty c] is applied, so [get] never fails on a row of [c].
    The result is [Error (row, reason)] at the first non-null row whose value is
    outside the OCaml type ['a]: an integer outside OCaml's [int], an instant or
    a span outside {!Time}'s range, a list with a null element, or any value of
    an extension type.

    Raises [Invalid_argument] if [c]'s type is not [ty]. *)

(** {1:structural Structural operations}

    They apply to every type, recursively, and keep the type. *)

val sub : t -> offset:int -> length:int -> t
(** [sub c ~offset ~length] is rows [offset] to [offset + length - 1] of [c],
    over [c]'s buffers.

    Raises [Invalid_argument] if the rows are not rows of [c]. *)

val gather : Nx.int64_t -> t -> t
(** [gather indices c] is the rows of [c] at the 1-D [indices], in order, with
    no range check: an index outside \[[0];[length c - 1]\] reads zero values, a
    clear validity bit, and an empty row of variable size. Its count is not yet
    read. *)

val permute : Nx.int64_t -> t -> t
(** [permute p c] is [gather p c] for a permutation [p] of [c]'s rows. Its null
    count is [c]'s, read or not. *)

val concat : t list -> t
(** [concat cs] is the rows of [cs], one column after the other: one column is
    itself, and several are contiguous from row [0], with the sum of their
    counts when every one is known. The columns have one type.

    Raises [Invalid_argument] if [cs] is empty. *)

val canonical : t -> t
(** [canonical c] is [c] with buffers that hold exactly its rows: offsets from
    [0], values exactly the rows', and every buffer of elements narrower than a
    byte from the first bit of its storage with every bit past its last element
    clear, at every depth. It is [c] itself when [c] is canonical, and one copy
    otherwise. It reads its null count, and drops a validity with no null. Two
    canonical columns whose rows hold the same bytes, under their nulls
    included, have the same layout, byte for byte. *)
