(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Columns.

    [Talon_next.Column] documents columns and their layouts. This interface adds
    their {{!repr}representation} and the {{!codec}codec} that converts OCaml
    values to and from columns by recursion over their type. *)

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
val ragged : t -> (int, Nx.uint8_elt) Nx_ragged.t
val layout : t -> layout
val of_layout : Type.any -> layout -> (t, int * string) result

(** {1:repr Representation} *)

(** The type for a column's values, by its type's storage:
    - [Fixed x] for [bool] ([Nx.bool], one byte per value), the integer and
      float types, decimals ([int64] unscaled), categoricals ([int32] codes),
      dates ([int32] days), clocks, durations and datetimes ([int64] ticks), and
      tensors ([x] of shape [(length, …shape)]);
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
