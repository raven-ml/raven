(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Parquet files.

    A {{:https://parquet.apache.org/docs/file-format/}Parquet} file stores a
    table column by column, in row groups, and describes them in a footer at its
    end. Talon reads Parquet files mapped in memory: open one with
    {!Nx_device.Buffer.of_file}, then {!sniff} its footer into a {!type-format}.

    A format says how talon reads a file's columns: each column's name, its
    Parquet type, and the talon type it reads as. Print it with {!pp_format},
    and change the type a column reads as with {!with_type}.

    {1:types Types}

    A column reads as the talon type that holds its values with their meaning:
    - [boolean] as [bool], [float] as [float32] and [double] as [float64];
    - [int32] and [int64] as [int32] and [int64], or as the type their [INTEGER]
      annotation names, [int8] to [uint64];
    - [DECIMAL(p,s)] as [decimal[p, s]], whatever its physical type;
    - [DATE] as [date], [TIME] as [clock] of its unit, and [TIMESTAMP] as
      [datetime] of its unit, in the zone [UTC] when the timestamp is adjusted
      to UTC and in no zone otherwise;
    - [int96], the legacy timestamps, as [datetime[ns]];
    - [fixed_len_byte_array(2)] annotated [FLOAT16] as [float16];
    - [binary] annotated [STRING], [ENUM] or [JSON] as [string], and every other
      [binary] and [fixed_len_byte_array] column as [binary].

    An annotation that does not apply to its column's physical type, such as
    [UUID] on an [int32], or that talon does not know, is ignored, as Parquet
    asks of its readers: the column reads as its physical type.

    {1:refusals Refusals}

    {!sniff} refuses a file that talon cannot read in full:
    - a file whose footer or columns are encrypted;
    - a file whose column chunks are stored in other files;
    - a nested column: a group, such as a list or a record, or a repeated field.
      Talon reads flat files only;
    - a map, an interval, a variant, or a decimal of more than 18 digits;
    - a column chunk compressed with LZO, Brotli, Hadoop's LZ4 framing (the
      codec [LZ4]; [LZ4_RAW] is read) or a codec Parquet does not define. Talon
      reads uncompressed, Snappy, gzip, Zstandard and [LZ4_RAW] chunks. *)

(** {1:formats Formats} *)

type format
(** The type for Parquet formats: the columns of a file, in schema order, each
    with its name, its Parquet type and the talon type it reads as. A format
    describes columns, not where a file stores them, so it holds no row groups
    and no file. *)

val sniff : Nx_device.Buffer.t -> (format, Talon_next.Error.t) result
(** [sniff b] is the format of the Parquet file whose bytes are [b], read from
    its footer: each column of the file's schema, in order, reading as
    {{!types}its Parquet type} says. Nothing but the footer is read.

    [b] is a buffer of the {!Nx_device.disk}, as {!Nx_device.Buffer.of_file}
    makes, which [sniff] maps into host memory ({!Nx_device.Buffer.borrow}), or
    a buffer the {!Nx_device.host} addresses, such as one that
    {!Nx_device.Buffer.of_bigarray} makes.

    [Error e] if [b] cannot be borrowed on the host, if [b] is not a Parquet
    file or its footer is malformed, or if the file is one talon
    {{!refusals}refuses}. [e] gives the bytes of [b] that failed and, for a
    column chunk, its row group. *)

val with_type : string -> Talon_next.Type.any -> format -> format
(** [with_type name t f] is [f] with the column [name] read as [t]. Besides the
    type {!sniff} gives it, a column reads as:
    - [string], [binary] or a categorical, for a column of byte strings: a
      [binary] or [fixed_len_byte_array] column without a numeric annotation. As
      a [string], its values must be valid UTF-8, and as a categorical, in the
      categorical's dictionary;
    - a [datetime] of any unit without a zone, for an [int96] column. Its values
      must be whole numbers of the unit.

    Values that break these rules fail when the column is read. Any other type
    is a computation on the column once read, such as [Expr.cast].

    Raises [Invalid_argument] if [f] has no column [name], or if that column
    does not read as [t]; the message names the types it reads as. *)

val pp_format : Format.formatter -> format -> unit
(** [pp_format ppf f] formats [f] for people: a line that counts its columns,
    then one line per column, with its name and the type it reads as, as
    {!Talon_next.Schema.pp} formats a column, and its Parquet type, as Parquet's
    schema language writes it, aligned after an arrow:
    {v
    parquet (3 columns)
      id int32                       ← required int32
      name string                    ← optional binary (STRING)
      "departs at" datetime[us, UTC] ← optional int64 (TIMESTAMP(MICROS,true))
    v} *)

(**/**)

(** Not part of the API: talon's tests read decoded columns through it until
    sources read Parquet. *)
module Private : sig
  (** The type for columns in Arrow's layouts. A row's value is zero, or the
      empty byte string, under a null. *)
  type column =
    | Fixed of { valid : Nx.bool_t option; values : Nx.packed }
        (** One value per row, of the storage of the column's talon type.
            [valid] is [true] at the rows that hold a value, and [None] when
            every row does. *)
    | Varsize of {
        valid : Nx.bool_t option;
        offsets : Nx.int64_t;
        data : Nx.uint8_t;
      }
        (** Byte strings: row [i] is [data] from [offsets.{i}] to
            [offsets.{i + 1}]. *)

  val read_column :
    format ->
    Nx_device.Buffer.t ->
    row_group:int ->
    string ->
    (column, Talon_next.Error.t) result
  (** [read_column f b ~row_group name] is the column [name] of [f] in the row
      group [row_group] of the file [b], read as [f] says.

      [Error e] as for {!sniff}, if the file has no row group [row_group] or no
      column [name], if that column does not read as [f]'s type, or if its
      column chunk fails to decode.

      Raises [Invalid_argument] if [f] has no column [name]. *)
end
