(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Zstandard frames.

    {{:https://www.rfc-editor.org/rfc/rfc8878}Zstandard} compresses data as a
    stream of frames: Parquet's ZSTD pages, Arrow IPC's ZSTD buffers and [.zst]
    files. The data of a stream is that of its frames, concatenated; skippable
    frames have none. Frames that need a dictionary are refused, and content
    checksums are checked when a frame has one. Zstandard compression is not
    supported.

    {b Concurrency.} The module holds no global mutable state: calls may run on
    distinct domains at once. *)

(** {1:types Types} *)

type bigbytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
(** The type for byte arrays, such as files mapped in memory. *)

(** {1:decompress Decompressing} *)

val decompress_into : bigbytes -> bigbytes -> (unit, string) result
(** [decompress_into src dst] decompresses the frames of [src], concatenated,
    into [dst]. Any window is accepted: [dst] holds the data the frames refer
    back to. It is [Error msg] if [src] is empty, if a frame is malformed,
    truncated, needs a dictionary or fails its checksum, if [src] has data that
    is not a frame, or if the data is not exactly [Bigarray.Array1.dim dst]
    bytes long. [msg] states the offset in [src] at which decoding failed; [dst]
    is then partly written.

    Raises [Invalid_argument] if [src] and [dst] overlap. *)
