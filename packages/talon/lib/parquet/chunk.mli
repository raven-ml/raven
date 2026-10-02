(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Column chunks decoded into Arrow's layouts.

    A column chunk is one column's pages in one row group: at most one
    dictionary page, then data pages, version 1 or 2. Reading one decodes each
    data page's definition levels into one validity mask of the row group's
    rows, and its values, compacted, into one buffer of the column's Parquet
    values, each page at the position the previous one ended, after the
    dictionary page's values. The buffer is then converted once into the storage
    of the column's talon type, the values of dictionary-encoded pages are
    gathered from it, and the values are spread onto the rows once, zero under
    the nulls. Integers narrower than [int32] are checked and narrowed last, so
    that a value that does not fit is found at its row.

    Pages decompress with the [compress] libraries' [decompress], straight from
    the mapped file into a buffer reused across the chunk's pages, and a page
    with a CRC is checked against it first. The RLE and bit-packing hybrid, the
    delta encodings, [PLAIN] byte arrays and the gather of dictionary-encoded
    byte strings run in C ([talon_parquet_stubs.c]). Fixed-width gathers, the
    spread and dtype changes are nx operations; [BYTE_STREAM_SPLIT], [PLAIN]
    booleans, the conversions of [int96] and of decimals and categorical lookups
    are OCaml loops.

    Encodings read: [PLAIN], [PLAIN_DICTIONARY] and [RLE_DICTIONARY], [RLE] for
    booleans, [DELTA_BINARY_PACKED], [DELTA_LENGTH_BYTE_ARRAY],
    [DELTA_BYTE_ARRAY] and [BYTE_STREAM_SPLIT], each on the physical types the
    format allows it on. Definition levels are [RLE], or [BIT_PACKED] for a
    required column, which has none. *)

(** The type for decoded columns. A row's value is zero, or the empty byte
    string, under a null. *)
type t =
  | Fixed of { valid : Nx.bool_t option; values : Nx.packed }
      (** One value per row, in the storage of the column's type: [bool],
          integers of the type's width, [float16] to [float64], [int32] days,
          [int64] ticks, [int32] codes of a categorical. [valid] is [true] at
          the rows that hold a value, and [None] when every row does. *)
  | Varsize of {
      valid : Nx.bool_t option;
      offsets : Nx.int64_t;
      data : Nx.uint8_t;
    }
      (** Byte strings, for [string] and [binary]: row [i] is [data] from
          [offsets.{i}] to [offsets.{i + 1}], and [offsets.{0}] is [0]. Text is
          not validated as UTF-8 here: columns validate it when they are built.
      *)

val check : Meta.file -> row_group:int -> int -> Leaf.t -> unit
(** [check m ~row_group i l] checks that the chunk [i] of the row group
    [row_group] of [m] can be read as the leaf [l], as {!Talon_parquet.sniff}
    does for every chunk of a file.

    Raises {!Meta.Error}, with the row group and the bytes the chunk claims, if
    the chunk is encrypted, is in another file or has no metadata, if its path
    is not [[l.name]] or its physical type not [l]'s, if its number of values is
    not the row group's rows, if its pages do not lie between the file's magic
    and its footer, or if its codec is refused: LZO, Brotli, Hadoop's LZ4, or a
    value Parquet does not define. *)

val read :
  Meta.bigbytes ->
  Meta.file ->
  row_group:int ->
  int ->
  Leaf.t ->
  Talon.Type.any ->
  t
(** [read b m ~row_group i l t] decodes the chunk [i] of the row group
    [row_group] of the file [b], whose metadata are [m], as the leaf [l] read as
    [t]. The chunk has passed {!check}, and [Leaf.reads_as l t] holds.

    Raises {!Meta.Error}, with the row group and the bytes of the page that
    failed, if a page header does not decode or overruns the chunk, a page fails
    its CRC or its decompression, a page or its levels hold more or fewer values
    than their header says, an encoding is not one talon reads on [l]'s physical
    type, a value does not decode or does not fit [t] (an integer outside a
    narrower type, named with its row, an [int96] outside the unit's range or
    not a whole number of it, a decimal outside [int64], a byte string outside a
    categorical's dictionary), a dictionary index is out of the dictionary, the
    pages' rows do not add up to the row group's, or the chunk's values would
    take more bytes than an [int] counts. *)
