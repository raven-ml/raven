(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Snappy blocks.

    A
    {{:https://github.com/google/snappy/blob/main/format_description.txt}Snappy
     block} is the compressed form of a byte sequence, preceded by the
    sequence's length. Parquet's SNAPPY pages are such blocks. A block is
    decompressed and compressed whole, in memory; Snappy's framing format for
    streams is not supported. Snappy has no compression levels.

    {b Concurrency.} The module holds no global mutable state: calls may run on
    distinct domains at once. *)

(** {1:types Types} *)

type bigbytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
(** The type for byte arrays, such as files mapped in memory. *)

(** {1:decompress Decompressing} *)

val decompress : bigbytes -> bigbytes -> (unit, string) result
(** [decompress src dst] decompresses the block [src] into [dst]. It is
    [Error msg] if [src] is malformed or truncated, if its data is not as long
    as the length it declares, or if that length is not
    [Bigarray.Array1.dim dst]. [msg] states the offset in [src] at which
    decoding failed; [dst] is then partly written.

    Raises [Invalid_argument] if [src] and [dst] overlap. *)

(** {1:compress Compressing} *)

val max_compressed_length : int -> int
(** [max_compressed_length n] is the length of the longest block {!compress}
    writes for [n] bytes, [32 + n + n / 6].

    Raises [Invalid_argument] if [n] is negative or larger than [0xFFFFFFFF],
    the longest sequence a block holds. *)

val compress : bigbytes -> bigbytes -> int
(** [compress src dst] writes the block of the bytes of [src] at the start of
    [dst] and is the block's length. The block depends only on the bytes of
    [src].

    Raises [Invalid_argument] if [Bigarray.Array1.dim dst] is less than
    [max_compressed_length (Bigarray.Array1.dim src)], or if [src] and [dst]
    overlap. *)
