(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernels over the bytes of text.

    What is not an array operation over text: UTF-8 validation, counting,
    slicing and matching scalar values. Each kernel reads its operand's bytes on
    the host, once, as a read named by its caller's function [by], and loops
    over them row by row. Comparisons and grouping of text compare codes
    elsewhere; {!compare}, against one text, needs none.

    Rows are those of an [Nx_ragged.t]; a row outside [mask], where one is
    given, is not read. *)

type bytes = (int, Nx.uint8_elt) Nx_ragged.t
(** The type for rows of bytes. *)

val reading : by:string -> ('a, 'b) Nx.t -> (Nx_device.Buffer.t -> 'c) -> 'c
(** [reading ~by x f] is [f b] for [b] a host buffer of [x]'s elements in C
    order, under a read claim so that no compiled call lends its memory while
    [f] reads it.

    Raises [Invalid_argument] starting with [by] if [x] cannot be read, as under
    a compiled function. *)

val utf_8 : by:string -> ?mask:Nx.bool_t -> bytes -> (int * string) option
(** [utf_8 ~by b] is [None] if every row of [b] is valid UTF-8, and otherwise
    [Some (row, reason)] for the first row that is not, [reason] naming the
    first byte that starts no valid sequence, as in [invalid UTF-8 at byte 3].
    Overlong forms, surrogates, values past U+10FFFF and sequences cut short are
    invalid.

    Raises [Invalid_argument] starting with [by] if the bytes cannot be read, as
    under a compiled function. *)

val length : by:string -> ?mask:Nx.bool_t -> bytes -> Nx.int64_t
(** [length ~by b] is the number of Unicode scalar values of each row of [b],
    valid UTF-8, and [0] outside [mask]. *)

val slice :
  by:string -> ?mask:Nx.bool_t -> offset:int -> length:int -> bytes -> bytes
(** [slice ~by ~offset ~length b] is the scalar values of each row of [b], valid
    UTF-8, at the positions [p] to [p + length - 1] that it has, [p] being
    [offset], or the row's length plus [offset] when [offset] is negative. A row
    outside [mask] is empty. *)

(** The type for patterns that text matches. Each string is valid UTF-8 and not
    empty. *)
type pattern =
  | Literal of string  (** Anywhere in the text. *)
  | Prefix of string  (** At its start. *)
  | Suffix of string  (** At its end. *)
  | Pieces of string list  (** Anywhere, in order, without overlap. *)

val matches : by:string -> ?mask:Nx.bool_t -> pattern -> bytes -> Nx.bool_t
(** [matches ~by p b] is [true] where [p] matches the row of [b], and [false]
    outside [mask]. *)

val compare : by:string -> bytes -> bytes -> Nx.int8_t
(** [compare ~by b one] is [-1], [0] or [1] where the row of [b] orders before,
    as or after [one]'s row, [one] of one row: bytes compare as unsigned
    numbers, and a row orders before every row it is a prefix of. *)
