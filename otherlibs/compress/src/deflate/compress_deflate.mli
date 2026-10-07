(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Deflate, zlib and gzip streams.

    {{:https://www.rfc-editor.org/rfc/rfc1951}Deflate} is a compressed data
    format. {{:https://www.rfc-editor.org/rfc/rfc1950}zlib} and
    {{:https://www.rfc-editor.org/rfc/rfc1952}gzip} wrap deflate data in a
    header and a checksum. The modules {!Deflate}, {!Zlib} and {!Gzip} read and
    write the three formats with the same functions:
    - [compress] and [decompress] convert a string held whole: a PNG image's
      data, a PDF stream.
    - [encoder] and [decoder] make an {!Encoder.t} and a {!Decoder.t}, which
      convert a stream given in pieces, in bounded memory: a [.gz] file, a ZIP
      entry.
    - [decompress_into] decompresses data held in byte arrays whose decompressed
      length is known: a ZIP entry or a Parquet page of a mapped file.

    {!Crc32} is the checksum of gzip, which ZIP, PNG and Parquet also use.

    {b Compression.} The bytes written depend only on the bytes given and the
    level: not on how they are sliced, or on the machine.

    {b Concurrency.} The module shares no mutable state between domains:
    [compress] keeps each domain's encoder state for its next call. Distinct
    encoders, decoders and calls may run on distinct domains at once; one
    encoder or decoder is used by one domain at a time. *)

(** {1:types Types} *)

type bigbytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
(** The type for byte arrays, such as files mapped in memory. *)

type level = int
(** The type for compression levels, from [0] to [9]. [0] stores the data
    without compressing it. From [1] to [9], each level searches for matches at
    least as hard as the one below it, trading speed for smaller output; some
    neighbouring levels write the same stream. Functions that take a level
    default it to [6]. *)

(** {1:streams Streams}

    An encoder or a decoder is given its input with [src] and returns its output
    from [encode] or [decode], one slice at a time, until it awaits more input:

    {[
    let rec loop () =
      match Compress_deflate.Decoder.decode d with
      | `Await ->
          let n = read buf 0 (Bytes.length buf) in
          Compress_deflate.Decoder.src d buf 0 n;
          loop ()
      | `Data (b, first, length) ->
          write b first length;
          loop ()
      | `End -> Ok ()
      | `Error msg -> Error msg
    ]}

    Input of length [0] ends the stream. *)

(** Decompressing streams. *)
module Decoder : sig
  type t
  (** The type for decoders. A decoder holds about 170 KiB. *)

  val src : t -> bytes -> int -> int -> unit
  (** [src d s first length] gives [d] the [length] bytes of [s] starting at
      [first] to decode. A [length] of [0] ends the input. The bytes must not
      change until {!decode} returns [`Await].

      Raises [Invalid_argument] if [first] and [length] are not a range of [s],
      or if [d] does not await input. It awaits input until it is first given
      some and after each [decode] that returns [`Await]. *)

  val decode :
    t -> [ `Await | `Data of bytes * int * int | `End | `Error of string ]
  (** [decode d] is:
      - [`Await] if [d] needs more input, to give with {!src}.
      - [`Data (b, first, length)] for the next [length] bytes of data, at
        [first] in [b]. They are at most 64 KiB, and are valid until the next
        call to [decode].
      - [`End] if the input has ended after a whole stream, all of whose data
        was returned.
      - [`Error msg] if the stream is malformed or truncated, fails a check, or
        has data after it. [msg] states the position in the input at which
        decoding failed.

      Once [decode] is [`End] or [`Error msg], it stays so. *)
end

(** Compressing streams. *)
module Encoder : sig
  type t
  (** The type for encoders. An encoder holds about 1 MiB until it ends. *)

  val src : t -> bytes -> int -> int -> unit
  (** [src e s first length] gives [e] the [length] bytes of [s] starting at
      [first] to encode. A [length] of [0] ends the input. The bytes must not
      change until {!encode} returns [`Await].

      Raises [Invalid_argument] if [first] and [length] are not a range of [s],
      or if [e] does not await input. It awaits input until it is first given
      some and after each [encode] that returns [`Await]. *)

  val encode : t -> [ `Await | `Data of bytes * int * int | `End ]
  (** [encode e] is:
      - [`Await] if [e] needs more input, to give with {!src}.
      - [`Data (b, first, length)] for the next [length] bytes of the stream, at
        [first] in [b]. They are valid until the next call to [encode].
      - [`End] once the input has ended and the whole stream was returned.

      Once [encode] is [`End], it stays so. *)
end

(** {1:formats Formats} *)

(** Raw {{:https://www.rfc-editor.org/rfc/rfc1951}deflate} streams, as ZIP
    entries hold them. *)
module Deflate : sig
  val compress : ?level:level -> string -> string
  (** [compress ~level s] is the deflate stream of [s] at level [level]. It is
      at most [n + 5 * (n / 65535 + 1)] bytes long for the [n] bytes of [s].

      Raises [Invalid_argument] if [level] is not in \[[0];[9]\]. *)

  val decompress : string -> (string, string) result
  (** [decompress s] is the data of the deflate stream [s]. It is [Error msg] if
      [s] is malformed or truncated, or if [s] has data after the byte that ends
      the stream. *)

  val encoder : ?level:level -> unit -> Encoder.t
  (** [encoder ~level ()] encodes one deflate stream at level [level].

      Raises [Invalid_argument] if [level] is not in \[[0];[9]\]. *)

  val decoder : unit -> Decoder.t
  (** [decoder ()] decodes one deflate stream, as {!decompress} does. *)

  val decompress_into : bigbytes -> bigbytes -> (unit, string) result
  (** [decompress_into src dst] decompresses the deflate stream [src] into
      [dst]. It is [Error msg] as {!decompress} is, and if the data is not
      exactly [Bigarray.Array1.dim dst] bytes long. [msg] states the offset in
      [src] at which decoding failed; [dst] is then partly written.

      Raises [Invalid_argument] if [src] and [dst] overlap. *)
end

(** {{:https://www.rfc-editor.org/rfc/rfc1950}zlib} streams, as PNG image data
    and PDF's [FlateDecode] streams hold them. A zlib stream is a two-byte
    header, deflate data and the Adler-32 checksum of the data. Streams written
    have a 32 KiB window; streams that need a preset dictionary are refused. *)
module Zlib : sig
  val compress : ?level:level -> string -> string
  (** [compress ~level s] is the zlib stream of [s] at level [level].

      Raises [Invalid_argument] if [level] is not in \[[0];[9]\]. *)

  val decompress : string -> (string, string) result
  (** [decompress s] is the data of the zlib stream [s]. It is [Error msg] if
      [s] is malformed or truncated, needs a preset dictionary or fails its
      checksum, or if [s] has data after the stream. *)

  val encoder : ?level:level -> unit -> Encoder.t
  (** [encoder ~level ()] encodes one zlib stream at level [level].

      Raises [Invalid_argument] if [level] is not in \[[0];[9]\]. *)

  val decoder : unit -> Decoder.t
  (** [decoder ()] decodes one zlib stream, as {!decompress} does. *)

  val decompress_into : bigbytes -> bigbytes -> (unit, string) result
  (** [decompress_into src dst] decompresses the zlib stream [src] into [dst].
      It is [Error msg] as {!decompress} is, and if the data is not exactly
      [Bigarray.Array1.dim dst] bytes long. [msg] states the offset in [src] at
      which decoding failed; [dst] is then partly written.

      Raises [Invalid_argument] if [src] and [dst] overlap. *)
end

(** {{:https://www.rfc-editor.org/rfc/rfc1952}gzip} streams, as [.gz] files and
    Parquet's GZIP pages hold them. A gzip stream is a sequence of members; each
    is a header, deflate data, and the CRC-32 and length of the data. The data
    of a stream is that of its members, concatenated. *)
module Gzip : sig
  val compress : ?level:level -> string -> string
  (** [compress ~level s] is one gzip member of [s] at level [level]. Its header
      has no file name, comment or extra field, a zero modification time and an
      unknown operating system.

      Raises [Invalid_argument] if [level] is not in \[[0];[9]\]. *)

  val decompress : string -> (string, string) result
  (** [decompress s] is the data of the gzip members of [s], concatenated. A
      member's file name, comment, modification time and extra field are
      skipped, and its header checksum, if any, is checked. It is [Error msg] if
      [s] is empty, if a member is malformed or truncated or fails its CRC-32 or
      length, or if [s] has data that is not a member. *)

  val encoder : ?level:level -> unit -> Encoder.t
  (** [encoder ~level ()] encodes one gzip member at level [level], as
      {!compress} does.

      Raises [Invalid_argument] if [level] is not in \[[0];[9]\]. *)

  val decoder : unit -> Decoder.t
  (** [decoder ()] decodes gzip members until the input ends, as {!decompress}
      does. *)

  val decompress_into : bigbytes -> bigbytes -> (unit, string) result
  (** [decompress_into src dst] decompresses the gzip members of [src],
      concatenated, into [dst]. It is [Error msg] as {!decompress} is, and if
      the data is not exactly [Bigarray.Array1.dim dst] bytes long. [msg] states
      the offset in [src] at which decoding failed; [dst] is then partly
      written.

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

  val string : ?crc:t -> ?first:int -> ?length:int -> string -> t
  (** [string ~crc ~first ~length s] is like {!bigbytes} for the [length] bytes
      of [s] starting at [first]. [first] defaults to [0] and [length] to the
      bytes from [first] to the end of [s].

      Raises [Invalid_argument] if [first] and [length] are not a range of [s].
  *)
end
