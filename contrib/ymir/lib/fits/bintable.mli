(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The layout of a binary table (FITS 4.0 §7.3): rows of fields given by
    [TFORMn], then a heap of the arrays that P and Q descriptors point to.

    Functions that check the data raise {!Err.Fail} at the place that
    fails. Numbers in the table are big-endian. *)

(** {1:forms Forms} *)

(** The type for [TFORMn]'s type codes. *)
type elt = L | X | B | I | J | K | A | E | D | C | M

(** The type for where a field's elements are: in the row, or in the heap
    through a P (32-bit) or Q (64-bit) descriptor. *)
type heap = Row | P | Q

type form = { repeat : int; elt : elt; heap : heap }
(** The type for a [TFORMn]: the repeat, and for a heap field the code of
    the heap's elements. *)

val char_of_elt : elt -> char
(** [char_of_elt e] is [e]'s type code. *)

val size : elt -> int
(** [size e] is the bytes of an element of [e]; [1] for [X], which holds 8
    elements per byte. *)

(** {1:layout Layout} *)

type column = {
  number : int;  (** from 1 *)
  name : string;  (** TTYPEn, [""] when absent *)
  form : form;
  start : int;  (** byte offset in the row *)
}
(** The type for a binary table's columns. *)

type t = {
  place : Err.place;
  store : Hdu.store;
  rows : int;
  row_bytes : int;
  columns : column array;
  heap_start : int;  (** THEAP, from the data unit's start *)
  heap_size : int;
}
(** The type for a binary table's layout over its data unit. *)

val layout : Header.t -> Hdu.store -> t
(** [layout h store] is the layout that [h] gives the data unit [store].
    Raises {!Err.Fail} if a structural keyword is absent, does not read or
    is out of range, the fields' widths do not sum to [NAXIS1], or [THEAP]
    lies outside the rows' end and the data unit's end. *)

val find : t -> string -> column option
(** [find t name] is the one column named [name], or [None] if there is
    none or several. *)

val column_place : t -> column -> Err.place
(** [column_place t c] is the place of [c] in [t]. *)

(** {1:bytes Bytes} *)

val read_rows : t -> int -> int -> Checksum.bigbytes
(** [read_rows t a b] is a host copy of rows [a] to [b - 1]. *)

val descriptor :
  ?row:int -> t -> column -> Checksum.bigbytes -> int -> int * int
(** [descriptor ~row t c rows r] is the element count and heap offset of the
    heap column [c] in row [r] of the host rows [rows]; [row] is the table's
    row, for errors. A descriptor whose count is 0 is [(0, 0)]; any other
    one's bytes lie in the heap. Raises {!Err.Fail} if the count or the
    offset is negative or past an [int], or the bytes end past the heap, and
    [Invalid_argument] if [c] is not a heap column. *)

val heap : t -> int -> int -> Checksum.bigbytes
(** [heap t off n] is a host copy of [n] bytes of the heap from [off]. *)

val be_int : Checksum.bigbytes -> int -> int -> int
(** [be_int a off n] is the unsigned big-endian integer of the [n] bytes of
    [a] from [off], [n] at most 7. *)

val be_int32 : Checksum.bigbytes -> int -> int
(** [be_int32 a off] is the signed big-endian 32-bit integer at [off]. *)

val be_int64 : Checksum.bigbytes -> int -> int64
(** [be_int64 a off] is the big-endian 64-bit integer at [off]. *)

val float_at : Checksum.bigbytes -> int -> elt -> float
(** [float_at a off e] is the big-endian float of [E] or [D] at [off].
    Raises [Invalid_argument] for another [e]. *)
