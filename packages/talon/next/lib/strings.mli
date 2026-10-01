(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Kernels over the bytes of text.

    What is not an array operation over text: UTF-8 validation. Each kernel
    reads its operand's bytes on the host, once, as a read named by its caller's
    function [by], and loops over them row by row. Comparisons and grouping of
    text are not here.

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
