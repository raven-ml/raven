(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** LZ4 blocks and frames.

    LZ4 has two formats.
    {{:https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md}Blocks} are
    compressed data alone, with their decompressed length known from elsewhere:
    Parquet's LZ4_RAW pages. A
    {{:https://github.com/lz4/lz4/blob/dev/doc/lz4_Frame_format.md}frame} wraps
    blocks in a header, an end mark and optional checksums: Arrow IPC's
    LZ4_FRAME buffers and [.lz4] files. {!Block} and {!Frame} decompress them;
    LZ4 compression is not supported.

    {b Concurrency.} The module holds no global mutable state: calls may run on
    distinct domains at once. *)

(** {1:types Types} *)

type bigbytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
(** The type for byte arrays, such as files mapped in memory. *)

(** {1:formats Formats} *)

(** LZ4 blocks. *)
module Block : sig
  val decompress_into : bigbytes -> bigbytes -> (unit, string) result
  (** [decompress_into src dst] decompresses the block [src] into [dst]. It is
      [Error msg] if [src] is malformed or truncated, or if its data is not
      exactly [Bigarray.Array1.dim dst] bytes long. [msg] states the offset in
      [src] at which decoding failed; [dst] is then partly written.

      Raises [Invalid_argument] if [src] and [dst] overlap. *)
end

(** LZ4 frames. A stream of frames has the data of its frames, concatenated;
    skippable frames have none. Frames that need a dictionary, and the legacy
    frame format, are refused. Block and content checksums are checked when a
    frame has them. *)
module Frame : sig
  val decompress_into : bigbytes -> bigbytes -> (unit, string) result
  (** [decompress_into src dst] decompresses the frames of [src], concatenated,
      into [dst]. It is [Error msg] if [src] is empty, if a frame is malformed,
      truncated, needs a dictionary or fails a checksum, if [src] has data that
      is not a frame, or if the data is not exactly [Bigarray.Array1.dim dst]
      bytes long. [msg] states the offset in [src] at which decoding failed;
      [dst] is then partly written.

      Raises [Invalid_argument] if [src] and [dst] overlap. *)
end
