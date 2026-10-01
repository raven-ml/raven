(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Deflate, zlib and gzip streams.

    {{:https://www.rfc-editor.org/rfc/rfc1951}Deflate} is a compressed data
    format. {{:https://www.rfc-editor.org/rfc/rfc1950}zlib} and
    {{:https://www.rfc-editor.org/rfc/rfc1952}gzip} wrap deflate data in a
    header and a checksum. The modules {!Deflate}, {!Zlib} and {!Gzip} read and
    write the three formats with the same three functions:
    - [decompress_reads] decompresses sequentially, from a
      {!Bytesrw.Bytes.Reader.t}: a [.gz] file, a CSV file read through gzip.
    - [compress_writes] compresses sequentially, to a {!Bytesrw.Bytes.Writer.t}:
      a PDF stream, a ZIP entry.
    - [decompress] decompresses data held in memory whose decompressed length is
      known: a ZIP entry or a Parquet page read from a mapped file, the image
      data of a PNG file.

    A string held whole compresses with
    [Bytes.Writer.filter_string [Zlib.compress_writes ()] s] and decompresses
    with [Bytes.Reader.filter_string [Zlib.decompress_reads ()] s].

    {!Crc32} is the checksum of gzip, which ZIP, PNG and Parquet also use.

    {b Positions.} Readers and writers made by the filters of this module start
    at position [0] unless they are given [pos]. Errors raised by a reader are
    reported at a position of the reader it reads from.

    {b Slice lengths.} Readers made by [decompress_reads] return slices of at
    most [slice_length] bytes, which defaults to
    {!Bytesrw.Bytes.Slice.io_buffer_size}. Writers made by [compress_writes]
    hint [slice_length], which defaults to the slice length of the writer they
    write to, and write slices that respect the latter's hint.

    {b Compression.} The bytes written depend only on the bytes given and the
    level: not on how they are sliced, or on the machine.

    {b Concurrency.} The module holds no global mutable state. Distinct readers,
    writers and calls may run on distinct domains at once; one reader or writer
    is used by one domain at a time. *)

open Bytesrw

(** {1:types Types} *)

type bigbytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
(** The type for byte arrays, such as files mapped in memory. *)

type level = int
(** The type for compression levels, from [0] to [9]. [0] stores the data
    without compressing it. From [1] to [9], each level searches for matches at
    least as hard as the one below it, trading speed for smaller output; some
    neighbouring levels write the same stream. *)

(** {1:errors Errors} *)

(** The type for deflate, zlib and gzip stream errors.

    Readers made by the [decompress_reads] filters of this module raise
    {!Bytesrw.Bytes.Stream.Error} with this case when the stream they read is
    malformed, truncated, fails a checksum or has data after it; the error's
    format is ["deflate"], ["zlib"] or ["gzip"]. The [decompress] functions
    return the same messages as [Error]. *)
type Bytes.Stream.error += Error of string  (** *)

(** {1:formats Formats} *)

(** Raw {{:https://www.rfc-editor.org/rfc/rfc1951}deflate} streams, as ZIP
    entries hold them. *)
module Deflate : sig
  val decompress_reads : unit -> Bytes.Reader.filter
  (** [decompress_reads () r] is a reader of the data of the deflate stream that
      [r] reads. Its reads raise {!Bytesrw.Bytes.Stream.Error} with an {!Error}
      if the stream is malformed or truncated, or if [r] has data after the byte
      that ends the stream. *)

  val compress_writes : ?level:level -> unit -> Bytes.Writer.filter
  (** [compress_writes ~level () w ~eod] is a writer that compresses the bytes
      written to it to one deflate stream on [w], at level [level], which
      defaults to [6]. The stream ends when {!Bytesrw.Bytes.Slice.eod} is
      written. {!Bytesrw.Bytes.Slice.eod} is then written on [w] iff [eod] is
      [true]; otherwise [w] can be used again for other writes.

      The stream is at most [n + 5 * (n / 65535 + 1)] bytes long for [n] bytes
      written.

      Raises [Invalid_argument] if [level] is not in \[[0];[9]\]. *)

  val decompress : bigbytes -> bigbytes -> (unit, string) result
  (** [decompress src dst] decompresses the deflate stream [src] into [dst]. It
      is [Error msg] if [src] is malformed or truncated, if it has data after
      the byte that ends the stream, or if its data is not exactly
      [Bigarray.Array1.dim dst] bytes long. [msg] states the offset in [src] at
      which decoding failed; [dst] is then partly written.

      Raises [Invalid_argument] if [src] and [dst] overlap. *)
end

(** {{:https://www.rfc-editor.org/rfc/rfc1950}zlib} streams, as PNG image data
    and PDF's [FlateDecode] streams hold them. A zlib stream is a two-byte
    header, deflate data and the Adler-32 checksum of the data. Streams that
    need a preset dictionary are refused. *)
module Zlib : sig
  val decompress_reads : unit -> Bytes.Reader.filter
  (** [decompress_reads () r] is a reader of the data of the zlib stream that
      [r] reads. Its reads raise {!Bytesrw.Bytes.Stream.Error} with an {!Error}
      if the stream is malformed or truncated, needs a preset dictionary or
      fails its checksum, or if [r] has data after the stream. *)

  val compress_writes : ?level:level -> unit -> Bytes.Writer.filter
  (** [compress_writes ~level () w ~eod] is a writer that compresses the bytes
      written to it to one zlib stream on [w], with a 32 KiB window, at level
      [level], which defaults to [6]. The stream ends when
      {!Bytesrw.Bytes.Slice.eod} is written. {!Bytesrw.Bytes.Slice.eod} is then
      written on [w] iff [eod] is [true]; otherwise [w] can be used again for
      other writes.

      Raises [Invalid_argument] if [level] is not in \[[0];[9]\]. *)

  val decompress : bigbytes -> bigbytes -> (unit, string) result
  (** [decompress src dst] decompresses the zlib stream [src] into [dst]. It is
      [Error msg] if [src] is malformed or truncated, needs a preset dictionary
      or fails its checksum, if it has data after the stream, or if its data is
      not exactly [Bigarray.Array1.dim dst] bytes long. [msg] states the offset
      in [src] at which decoding failed; [dst] is then partly written.

      Raises [Invalid_argument] if [src] and [dst] overlap. *)
end

(** {{:https://www.rfc-editor.org/rfc/rfc1952}gzip} streams, as [.gz] files and
    Parquet's GZIP pages hold them. A gzip stream is a sequence of members; each
    is a header, deflate data, and the CRC-32 and length of the data. The data
    of a stream is that of its members, concatenated. *)
module Gzip : sig
  val decompress_reads : unit -> Bytes.Reader.filter
  (** [decompress_reads () r] is a reader of the data of the gzip members that
      [r] reads, concatenated, until [r] ends. A member's file name, comment,
      modification time and extra field are skipped, and its header checksum, if
      any, is checked. Its reads raise {!Bytesrw.Bytes.Stream.Error} with an
      {!Error} if [r] is empty, if a member is malformed or truncated or fails
      its CRC-32 or length, or if [r] has data that is not a member. *)

  val compress_writes : ?level:level -> unit -> Bytes.Writer.filter
  (** [compress_writes ~level () w ~eod] is a writer that compresses the bytes
      written to it to one gzip member on [w], at level [level], which defaults
      to [6]. The member's header has no file name, comment or extra field, a
      zero modification time and an unknown operating system. The member ends
      when {!Bytesrw.Bytes.Slice.eod} is written. {!Bytesrw.Bytes.Slice.eod} is
      then written on [w] iff [eod] is [true]; otherwise [w] can be used again
      for other writes.

      Raises [Invalid_argument] if [level] is not in \[[0];[9]\]. *)

  val decompress : bigbytes -> bigbytes -> (unit, string) result
  (** [decompress src dst] decompresses the gzip members of [src], concatenated,
      into [dst]. It is [Error msg] if [src] is empty, if a member is malformed
      or truncated or fails its CRC-32 or length, if [src] has data that is not
      a member, or if the data is not exactly [Bigarray.Array1.dim dst] bytes
      long. [msg] states the offset in [src] at which decoding failed; [dst] is
      then partly written.

      Raises [Invalid_argument] if [src] and [dst] overlap. *)
end

(** {1:checksums Checksums} *)

(** CRC-32 checksums.

    The checksum of gzip members, which ZIP entries, PNG chunks and Parquet
    pages also carry: the 32-bit cyclic redundancy check of polynomial
    [0x04C11DB7], with reflected bits and an initial value and a final exclusive
    or of [0xFFFFFFFF]. The checksum of the bytes of ["123456789"] is
    [0xCBF43926]. *)
module Crc32 : sig
  type t = int
  (** The type for checksums. An integer in the range \[[0];[0xFFFFFFFF]\]. *)

  val bigbytes : ?crc:t -> bigbytes -> t
  (** [bigbytes ~crc b] is the checksum of the bytes that [crc] checks followed
      by the bytes of [b]. [crc] defaults to [0], the checksum of no bytes, so
      [bigbytes ~crc:(bigbytes a) b] checks the bytes of [a] then those of [b].
  *)

  val slice : ?crc:t -> Bytes.Slice.t -> t
  (** [slice ~crc s] is like {!bigbytes} for the bytes in the range of [s]. With
      {!Bytesrw.Bytes.Reader.tap} or {!Bytesrw.Bytes.Writer.tap} it checks a
      stream as it is read or written. *)
end
