(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Parquet's metadata: the Thrift structures of a file's footer and of its page
    headers, with the fields talon reads.

    Field names follow
    {{:https://github.com/apache/parquet-format/blob/master/src/main/thrift/parquet.thrift}
     [parquet.thrift]}. Decoding checks structure only: required fields are
    present, enumerations talon interprets here have a defined value, and sizes
    and offsets are not negative. {!Leaf} gives the schema its meaning, and
    {!Chunk} checks the column chunks against it. *)

type bigbytes = Thrift.bigbytes

exception
  Error of {
    row_group : int option;
    bytes : (int * int) option;
    text : string option;
    msg : string;
  }
(** [Error {row_group; bytes; text; msg}] is raised by {!footer},
    {!Leaf.of_schema} and {!Chunk} on data that does not decode, or that talon
    refuses, for the reason [msg], found in the row group [row_group] at the
    bytes [bytes], [(first, last)], of the file, on the value [text], when they
    are known. *)

val fail :
  ?row_group:int ->
  ?bytes:int * int ->
  ?text:string ->
  ('a, unit, string, 'b) format4 ->
  'a
(** [fail ?row_group ?bytes ?text fmt] raises {!Error} with the message that
    [fmt] formats. *)

(** {1:schema Schema} *)

(** The type for physical types, [Type] in [parquet.thrift]. *)
type physical =
  | Boolean
  | Int32
  | Int64
  | Int96
  | Float
  | Double
  | Byte_array
  | Fixed_len_byte_array  (** Of the length {!element.length} gives. *)

(** The type for field repetitions, [FieldRepetitionType]. *)
type repetition = Required | Optional | Repeated

(** The type for logical types, the union [LogicalType]. A time unit is [Ms],
    [Us] or [Ns]. *)
type logical =
  | String
  | Map
  | List
  | Enum
  | Decimal of { precision : int; scale : int }
  | Date
  | Time of { unit_ : Talon_next.Type.unit_; utc : bool }
  | Timestamp of { unit_ : Talon_next.Type.unit_; utc : bool }
  | Integer of { bits : int; signed : bool }
  | Null  (** [UNKNOWN]: every value is null. *)
  | Json
  | Bson
  | Uuid
  | Float16
  | Variant
  | Geometry
  | Geography
  | Other of int
      (** A member talon does not know, by its field identifier, or a known
          member whose time unit talon does not know. *)

type element = {
  name : string;
  physical : physical option;  (** [None] for a group. *)
  length : int;  (** [type_length], or [0] when absent. *)
  repetition : repetition option;  (** [None] for the root only. *)
  children : int;  (** [num_children], or [0] when absent. *)
  converted : int option;
      (** [converted_type], as its value in [ConvertedType]. {!Leaf} reads the
          values it knows and ignores the others. *)
  precision : int;  (** [precision], or [0] when absent. *)
  scale : int;  (** [scale], or [0] when absent. *)
  logical : logical option;  (** [logicalType]. *)
}
(** The type for schema elements, [SchemaElement]: the schema is its elements in
    depth-first order, each group followed by its children. *)

(** {1:footer Footer} *)

(** The type for compression codecs, [CompressionCodec]. *)
type codec =
  | Uncompressed
  | Snappy
  | Gzip
  | Lzo
  | Brotli
  | Lz4  (** Hadoop's framing of LZ4 blocks. *)
  | Zstd
  | Lz4_raw
  | Unknown_codec of int  (** A value Parquet does not define. *)

type column_meta = {
  physical : physical;  (** [type]. *)
  path : string list;  (** [path_in_schema]. *)
  codec : codec;  (** [codec]. *)
  values : int;  (** [num_values]: values, nulls included. *)
  uncompressed_size : int;
      (** [total_uncompressed_size]: the bytes of the chunk's pages, headers
          included, once decompressed. *)
  compressed_size : int;
      (** [total_compressed_size]: the bytes of the chunk's pages, headers
          included. *)
  data_page : int;  (** [data_page_offset]. *)
  dictionary_page : int option;
      (** [dictionary_page_offset] when it is positive. Some writers write [0]
          for none. *)
}
(** The type for column chunk metadata, [ColumnMetaData]. *)

type chunk = {
  file_path : string option;
      (** [file_path]: the file that holds the chunk, when not this one. *)
  encrypted : bool;  (** [crypto_metadata] or [encrypted_column_metadata]. *)
  meta : column_meta option;  (** [meta_data], absent from encrypted chunks. *)
}
(** The type for column chunks, [ColumnChunk]. *)

type row_group = {
  rows : int;  (** [num_rows]. *)
  chunks : chunk array;  (** [columns], one per leaf of the schema. *)
}
(** The type for row groups, [RowGroup]. *)

type file = {
  rows : int;  (** [num_rows]. *)
  schema : element array;  (** [schema]. *)
  row_groups : row_group array;  (** [row_groups]. *)
  footer : int;
      (** The position of the footer's first byte: column chunks end before it.
      *)
}
(** The type for a file's metadata, [FileMetaData], and where it starts. *)

val footer : bigbytes -> file
(** [footer b] is the metadata in the footer of the Parquet file [b]: the
    [FileMetaData] that the last eight bytes, a length and the magic ["PAR1"],
    point to.

    Raises {!Error}, with the bytes that failed, if [b] does not start and end
    with ["PAR1"], if the length does not fit [b], if the metadata do not
    decode, if the row groups' rows do not add up to the file's, or if the file
    is encrypted: it ends with ["PARE"], or its metadata have an
    [encryption_algorithm]. *)

(** {1:pages Page headers} *)

(** The type for value encodings, [Encoding]. *)
type encoding =
  | Plain
  | Plain_dictionary
  | Rle
  | Bit_packed
  | Delta_binary_packed
  | Delta_length_byte_array
  | Delta_byte_array
  | Rle_dictionary
  | Byte_stream_split
  | Unknown_encoding of int  (** A value talon does not know. *)

(** The type for the pages of a column chunk, [PageType] with its header. *)
type page =
  | Data of { values : int; encoding : encoding; levels : encoding }
      (** [DATA_PAGE]: [num_values] values, nulls included, in [encoding];
          [levels] is the [definition_level_encoding]. *)
  | Data_v2 of {
      values : int;
      nulls : int;
      rows : int;
      encoding : encoding;
      definition_bytes : int;
      repetition_bytes : int;
      compressed : bool;
    }
      (** [DATA_PAGE_V2]: [num_values], [num_nulls], [num_rows], [encoding], the
          uncompressed bytes of the levels that start the page, and
          [is_compressed], which defaults to [true]. *)
  | Dictionary of { values : int; encoding : encoding }
      (** [DICTIONARY_PAGE]. *)
  | Other  (** [INDEX_PAGE], or a page type talon does not know. *)

type header = {
  page : page;
  uncompressed_size : int;  (** [uncompressed_page_size]. *)
  compressed_size : int;
      (** [compressed_page_size]: the bytes that follow the header. *)
  crc : int option;
      (** [crc], as an unsigned 32-bit value: the CRC-32 of the page's bytes as
          stored. *)
}
(** The type for page headers, [PageHeader]. *)

val page_header : bigbytes -> pos:int -> limit:int -> header * int
(** [page_header b ~pos ~limit] is the page header at [pos] in [b], and the
    position of the page's first byte after it, which is at most [limit].

    Raises {!Thrift.Error} if the header does not decode before [limit], or a
    size or a count in it is negative. *)
