(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Columns.

    [Talon_next.Column] documents columns and their layouts. This interface adds
    their {{!repr}representation}, the {{!codec}codec} that converts OCaml
    values to and from columns by recursion over their type, and the
    {{!structural}structural operations} that every verb uses. *)

type t

type layout =
  | Fixed of { validity : Nx_bits.t option; values : Nx.packed }
  | Varsize of { validity : Nx_bits.t option; offsets : Nx.int64_t; child : t }
  | Children of {
      validity : Nx_bits.t option;
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
val of_tensor : ?validity:Nx_bits.t -> ('a, 'b) Nx.t -> t
val to_tensor : ('a, 'b) Nx.dtype -> t -> ('a, 'b) Nx.t
val validity : t -> Nx_bits.t option
val ragged : ('a, 'b) Nx.dtype -> t -> ('a, 'b) Nx_ragged.t
val of_ragged : ?validity:Nx_bits.t -> ('a, 'b) Nx_ragged.t -> t
val layout : t -> layout
val of_layout : Type.any -> layout -> (t, int * string) result

(** {1:repr Representation} *)

(** The type for a column's values, by its type's storage:
    - [Fixed x] for [bool] ([Nx.bool], one byte per value), the integer and
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
(** [data c] is [c]'s values. Under a null they are unspecified; talon's
    operations make zeros and empty rows. *)

val make : Type.any -> ?valid:Nx.bool_t -> length:int -> data -> t
(** [make ty ?valid ~length d] is the column of type [ty] with values [d] and a
    null wherever [valid] is [false]. It packs [valid] and counts its nulls (one
    read), and drops it when no row is null. It checks that [d] is [ty]'s
    storage and that every part has [length] rows, and nothing about values: the
    caller's data is held by [ty]. Every internal operation makes columns with
    it.

    Raises [Invalid_argument] if [d] is not [ty]'s storage, or if a part's
    length or [valid]'s is not [length]. *)

val with_data : Type.any -> data -> t -> t
(** [with_data ty d c] is the column of type [ty] with values [d] and [c]'s
    nulls, which it does not count again: [c]'s values cast to [ty], or read as
    another type of the same storage. As {!make}, it checks that [d] is [ty]'s
    storage for [c]'s rows, and nothing about values.

    Raises [Invalid_argument] if [d] is not [ty]'s storage for [length c] rows.
*)

val valid : t -> Nx.bool_t option
(** [valid c] is [c]'s validity as a byte mask, [None] iff [c] has no null. *)

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

val take : Nx.int64_t -> t -> t
(** [take indices c] is the rows of [c] at the 1-D [indices], in order. An index
    outside \[[0];[length c - 1]\] gives a null row of zeros or an empty row, so
    that joins pad with [-1]. *)

val permute : Nx.int64_t -> t -> t
(** [permute p c] is [take p c] for a permutation [p] of [c]'s rows. Its null
    count is [c]'s, so a fixed-width column is permuted without a read. *)

val concat : t list -> t
(** [concat cs] is the rows of [cs], one column after the other, canonical (see
    {!canonical}). The columns have one type.

    Raises [Invalid_argument] if [cs] is empty. *)

val canonical : t -> t
(** [canonical c] is [c] with buffers that hold exactly its rows: offsets from
    [0], values exactly the rows', a validity at bit offset [0] with no bit set
    past its length, at every depth. It is [c] itself when [c] is canonical, and
    one copy otherwise. Two canonical columns whose rows hold the same bytes,
    under their nulls included, have the same layout, byte for byte. *)
