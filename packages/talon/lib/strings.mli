(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernels over the bytes of text.

    What is not an array operation over text: UTF-8 validation, counting,
    slicing and matching scalar values, splitting and replacing. Each kernel
    reads its operand's bytes on the host, once, as a read named by its caller's
    function [by], and loops over them row by row. Comparisons and grouping of
    text compare codes elsewhere; {!compare} and {!member}, against a few
    texts, need none.

    Rows are those of an [Nx_ragged.t]; a row outside [mask], a bit mask, where
    one is given, is not read. *)

type bytes = (int, Nx.int8_unsigned_elt, Nx.host) Nx_ragged.t
(** The type for rows of bytes. *)

val reading :
  by:string ->
  ('a, 'b, Nx.host) Nx.t ->
  ('c, 'd) Bigarray.kind ->
  (('c, 'd, Bigarray.c_layout) Bigarray.Array1.t -> 'e) ->
  'e
(** [reading ~by x k f] is [f a] for [a] the bytes of a host buffer of [x]'s
    elements in C order, as elements of [k], under a read claim so that no
    donation writes their memory while [f] reads it ({!Nx.Repr.read}).

    Raises [Invalid_argument] starting with [by] if nx cannot read [x], as a
    traced [x]. *)

val fresh : ('v, 's) Nx.dtype -> int -> ('v, 's) Nx_array.t
(** [fresh dt n] is a host vector of [n] uninitialized elements of [dt]
    ({!Nx_array.host}). *)

val fill :
  ('v, 's) Nx_array.t ->
  (('v, 's, Bigarray.c_layout) Bigarray.Array1.t -> 'a) ->
  'a
(** [fill a f] is [f b], [b] the elements of the vector [a] as its dtype's
    Bigarray kind, under a write claim ({!Nx_array.write}). *)

val value : ('v, 's) Nx_array.t -> ('v, 's, Nx.host) Nx.t
(** [value a] is the host value over [a]'s memory, which the caller gives up:
    it must not read, write or keep [a] afterwards
    ({!Nx.Repr.unsafe_of_array}). *)

val utf_8 :
  by:string -> ?mask:Nx.host Nx.bit_t -> bytes -> (int * string) option
(** [utf_8 ~by b] is [None] if every row of [b] is valid UTF-8, and otherwise
    [Some (row, reason)] for the first row that is not, [reason] naming the
    first byte that starts no valid sequence, as in [invalid UTF-8 at byte 3].
    Overlong forms, surrogates, values past U+10FFFF and sequences cut short are
    invalid.

    Raises [Invalid_argument] starting with [by] if the bytes cannot be read, as
    under a compiled function. *)

val length : by:string -> ?mask:Nx.host Nx.bit_t -> bytes -> Nx.host Nx.int64_t
(** [length ~by b] is the number of Unicode scalar values of each row of [b],
    valid UTF-8, and [0] outside [mask]. *)

val slice :
  by:string ->
  ?mask:Nx.host Nx.bit_t ->
  offset:int ->
  length:int ->
  bytes ->
  bytes
(** [slice ~by ~offset ~length b] is the scalar values of each row of [b], valid
    UTF-8, at the positions [p] to [p + length - 1] that it has, [p] being
    [offset], or the row's length plus [offset] when [offset] is negative. A row
    outside [mask] is empty. *)

val trim : by:string -> ?mask:Nx.host Nx.bit_t -> bytes -> bytes
(** [trim ~by b] is each row of [b], valid UTF-8, without the scalar values of
    Unicode's White_Space property at its start and end. A row outside [mask] is
    empty. *)

(** The type for patterns that text matches. Each string is valid UTF-8 and not
    empty. *)
type pattern =
  | Literal of string  (** Anywhere in the text. *)
  | Prefix of string  (** At its start. *)
  | Suffix of string  (** At its end. *)
  | Pieces of string list  (** Anywhere, in order, without overlap. *)

val matches :
  by:string -> ?mask:Nx.host Nx.bit_t -> pattern -> bytes -> Nx.host Nx.bit_t
(** [matches ~by p b] is [true] where [p] matches the row of [b], and [false]
    outside [mask]. *)

val split :
  by:string ->
  ?mask:Nx.host Nx.bit_t ->
  string ->
  bytes ->
  Nx.host Nx.int64_t * bytes
(** [split ~by sep b] is [(lists, pieces)]: the pieces of each row of [b], valid
    UTF-8, between the matches of [sep], valid UTF-8 and not empty, left to
    right without overlap, so a row of k matches has k + 1 pieces; row [i]'s
    pieces are the rows [lists.{i}] to [lists.{i + 1} - 1] of [pieces]. A row
    outside [mask] has no piece. *)

val replace :
  by:string ->
  ?mask:Nx.host Nx.bit_t ->
  sub:string ->
  into:string ->
  bytes ->
  bytes
(** [replace ~by ~sub ~into b] is each row of [b], valid UTF-8, with each match
    of [sub], valid UTF-8 and not empty, replaced by [into], left to right
    without overlap. A row outside [mask] is empty. *)

val concat :
  by:string ->
  ?mask:Nx.host Nx.bit_t ->
  length:int ->
  string ->
  bytes list ->
  bytes
(** [concat ~by ~length sep parts] is the [length] rows whose row [i] is row [i]
    of each of [parts] joined by [sep], a part of one row giving its row [0] to
    every row. A row outside [mask] is empty. [parts] is not empty, and each
    part has one row or [length]. *)

val compare :
  by:string ->
  [ `Eq | `Ne | `Lt | `Le | `Gt | `Ge ] ->
  bytes ->
  bytes ->
  Nx.host Nx.bit_t
(** [compare ~by op b one] is [true] where the row of [b] compares by [op] with
    [one]'s row, [one] of one row: bytes compare as unsigned numbers, and a row
    orders before every row it is a prefix of. *)

val member : by:string -> bytes -> bytes -> Nx.host Nx.bit_t
(** [member ~by b texts] is [true] where the row of [b] has the bytes of a row
    of [texts]. *)
