(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Parquet files, read and written.

    A {{:https://parquet.apache.org/docs/file-format/}Parquet} file stores a
    table column by column, in {e row groups}, and describes them in a {e
    footer} at its end. Each row group holds a {e column chunk} per column, a
    sequence of {e pages} compressed with one {e codec}. A column has a {e
    physical type}, how its values are stored, and may carry an {e annotation},
    what they mean. A {e group} is a field with children.

    A {!type-format} says how talon reads a file's columns and how it writes
    them: each column's name, its Parquet type and the talon type
    ({!Talon.Type}) it reads as, with the codec and row-group size {!encode}
    writes with. A format describes columns only, so one format reads any file
    with those columns. {!pp_format} prints it.

    A read has three steps. Map the file with {!Rig_disk.of_file}, or take a
    buffer on the host. {!sniff} its footer into a format and declare what the
    file leaves open with {!with_type}. Read the rows with {!source}, a
    {!Talon.Source.t}, or with {!file} from a path. A write states a format for
    a query's schema with {!val-format}, and {!encode} writes the query's rows
    as that format says.

    Unless a value's doc says otherwise, a function that reads bytes returns
    [Error] for bytes it cannot read, and a function given a format or a name
    raises [Invalid_argument] for a misuse.

    This reads a file and writes its rows with another codec:
    {[
      let recompress w =
        let ( let* ) = Result.bind in
        let* source = Talon_parquet.file "events.parquet" in
        let q = Talon.Query.of_source source in
        Talon_parquet.encode
          (Talon_parquet.format ~compression:Talon_parquet.Zstd
             (Talon.Query.schema q))
          q w
    ]} *)

(** {1:types Types}

    {2:scalars Scalar columns}

    A column reads as the talon type that holds its values with their meaning:
    - [boolean] as [bool], [float] as [float32] and [double] as [float64];
    - [int32] and [int64] as [int32] and [int64], or as the type their
      [INTEGER] annotation names, [int8] to [uint64];
    - [DATE] as [date], [TIME] as [clock] of its unit, and [TIMESTAMP] as
      [datetime] of its unit, an instant ([datetime[u, UTC]]) when the
      timestamp is adjusted to UTC and a wall clock ([datetime[u]]) otherwise;
    - [int96], the legacy timestamps, as [datetime[ns]];
    - [fixed_len_byte_array(2)] annotated [FLOAT16] as [float16];
    - [binary] annotated [STRING], [ENUM] or [JSON] as [string], and every
      other [binary] and [fixed_len_byte_array] column as [binary].

    An annotation that does not apply to its column's physical type, such as
    [UUID] on an [int32], or that talon does not know, is ignored, as Parquet
    asks of its readers. The column reads as its physical type.

    {2:undeclared Undeclared decimals}

    A [DECIMAL(p,s)] column, the number [unscaled × 10{^-s}] of at most [p]
    digits, has no talon type until {!with_type} gives it one, and is
    {e undeclared} before then. A nested column that holds one is undeclared
    as a whole. A declared decimal reads as [float64], the float nearest each
    value, or, up to 18 digits, as [int64], each value's [unscaled] integer.

    {2:alternatives Alternative types}

    {!with_type} also reads:
    - a column of byte strings, a [binary] or [fixed_len_byte_array] column
      without a numeric annotation, as [string], [binary] or a categorical.
      As a [string] its values must be valid UTF-8, and as a categorical they
      must be in its dictionary;
    - an [int96] column as a wall-clock [datetime] of any unit. Its values
      must be whole numbers of the unit;
    - a {{!nested}nested column} as a list of what its element reads as, or a
      record of the same field names in order, each reading as its field does.

    Values that break these rules fail when the column is read.

    {1:nested Nested columns}

    A field of a group reads as the column its shape says, a list or a record
    of columns that each read by these same rules, and a null at any level of
    a file reads as a null of that level's column:
    - a group annotated [LIST] as a list, its element chosen by the
      backward-compatibility rules of Parquet's
      {{:https://github.com/apache/parquet-format/blob/master/LogicalTypes.md#lists}
       logical types}, so that a list of two levels and one of three read
      alike;
    - a group annotated [MAP] or [MAP_KEY_VALUE] as a list of records, the
      fields of its repeated group, a map's [key] and [value], or its [key]
      alone when the file holds no values. A map is its pairs, in the file's
      order, and two equal keys stay two elements;
    - any other group as a record of its fields, in order, and null where the
      file says the group is absent;
    - a [repeated] field that no such group holds as a list of its values,
      and that list is never null.

    A value below a null is unspecified, in a record's field as in a list's
    offsets.

    {1:refusals Refusals}

    {!sniff} refuses a file that talon cannot read in full:
    - a file whose footer or columns are encrypted;
    - a file whose column chunks are stored in other files;
    - an interval or a variant, at any level of a column;
    - a [repeated] group annotated [LIST] or [MAP] that is not the element of a
      list, a group annotated so that does not hold one [repeated] field, and a
      map whose [repeated] field is not a group;
    - a group of no fields, and a group with two fields of the same name;
    - a column chunk compressed with LZO, Brotli, Hadoop's LZ4 framing (the
      codec [LZ4]; [LZ4_RAW] is read) or a codec Parquet does not define. *)

(** {1:formats Formats} *)

type format
(** The type for Parquet formats: the columns of a file, in schema order, each
    with its name, its Parquet type and the talon type it reads as, and the
    codec and row-group size {!encode} writes with. One that {!sniff} gives
    writes with {!val-format}'s defaults. *)

val sniff : Rig.Buffer.t -> (format, Talon.Error.t) result
(** [sniff b] is the format of the Parquet file whose bytes are [b], read from
    its footer alone: each column of the file's schema, in order, reading as
    {{!types}its Parquet type} says.

    [b] must be a buffer of the disk ({!Rig_disk.device}), as
    {!Rig_disk.of_file} makes, which [sniff] maps into host memory
    ({!Rig.Buffer.borrow}), or a buffer the host addresses, as
    {!Rig.Buffer.of_bigarray} makes.

    [Error e] if [b] cannot be borrowed on the host, if [b] is not a Parquet
    file or its footer is malformed, or if the file is one talon
    {{!refusals}refuses}. When its Thrift encoding does not parse, [e] gives
    the bytes of [b] that failed. *)

val with_type : string -> Talon.Type.any -> format -> format
(** [with_type name t f] is [f] with the column [name] read as [t], the type
    {!sniff} gives it, a declaration if it is {{!undeclared}undeclared}, or
    one of its {{!alternatives}alternatives}. Any other type is a computation
    on the column once read, such as [Expr.cast].

    Raises [Invalid_argument] if [f] has no column [name], or if that column
    does not read as [t]; the message names the types it reads as. *)

val pp_format : Format.formatter -> format -> unit
(** [pp_format ppf f] formats [f] for people: a line that counts its columns,
    then one line per column, with its name and the type it reads as, as
    {!Talon.Schema.pp} formats a column, and its Parquet type, as Parquet's
    schema language writes it, aligned after an arrow. A column reads as
    [undeclared] until it is declared:
    {v
    parquet (3 columns)
      id int32         ← required int32
      name string      ← optional binary (STRING)
      price undeclared ← optional int64 (DECIMAL(12,2))
    v} *)

(** {1:reading Reading}

    A source decodes only the columns a request names and those of the
    conjuncts it applies ({!Talon.Source}), a nested column's every leaf (a
    column of values below its groups) when it is requested. The file's row
    groups are its parts, each one batch of the request's columns.

    {2:conjuncts Text conjuncts}

    The source answers {!Talon.Source.Exact} for a conjunct that reads one
    column of [string] or [binary], and applies it as it reads. On a chunk with
    a dictionary page, the conjunct is evaluated once for each value of the
    dictionary and of the chunk's other pages, and each row takes its value's
    result. On another chunk, it is evaluated for each row. A part of a request
    with such a conjunct leaves its number of rows unstated.

    {2:pruning Pruning}

    The source answers {!Talon.Source.Inexact} for another conjunct that
    row-group statistics can decide, and skips a row group whose statistics
    show that no row of it passes. A column of nulls passes no comparison. A
    minimum and a maximum bound a comparison or [In] on integers (except
    [uint64]), dates, datetimes, byte strings, and [float32] and [float64] in
    a row group the file states holds no NaN. A nested column's statistics say
    nothing of its rows, so no conjunct over a nested column prunes a row
    group.

    {2:failure Failure}

    A read fails with [Error e] as {!sniff} does, if the file has no column of
    the request or one that does not read as the format's type, or if a column
    chunk fails to decode. A failure in a column chunk names its row group. A
    page that does not decode also gives its bytes, and a value that a column's
    layout refuses ({!Talon.Column.of_layout}) names the column and the row of
    the row group. A failure in a nested column names the column or, for a
    leaf, its dotted path such as [tags.list.element], and a value's position
    among the leaf's values stands for its row. *)

val source : format -> Rig.Buffer.t -> Talon.Source.t
(** [source f b] is the source, named [parquet], of the rows of the Parquet
    file whose bytes are [b], read as [f] says. [b] is a buffer as {!sniff}
    takes. Each read of the source reads [b]'s footer again.

    Raises [Invalid_argument] if [f] leaves a column
    {{!undeclared}undeclared}; the message names it and the types it reads
    as. *)

val file : ?format:format -> string -> (Talon.Source.t, Talon.Error.t) result
(** [file ?format path] is the {!source} of the file [path], which it maps
    ({!Rig_disk.of_file}), read as [format], or as {!sniff} gives it when
    [format] is absent. It is named [parquet "path"] and states its number of
    rows, and its errors name the file.

    [Error e] if the file cannot be mapped, or as {!sniff}.

    Raises [Invalid_argument] as {!source} does, which a file with an
    {{!undeclared}undeclared} column does when [format] is absent. *)

(** {1:writing Writing}

    {!encode} writes every column [optional], its nulls as definition levels,
    and a list column in Parquet's three levels ([LIST], [list], [element]).

    {2:written Written types}

    - [bool], [float32] and [float64] write as [boolean], [float] and
      [double], and [float16] as [fixed_len_byte_array(2)] annotated [FLOAT16];
    - [int32] and [int64] write as themselves, and every other integer as
      [int32] or [int64] annotated [INTEGER];
    - [string] writes as [binary] annotated [STRING], and [binary] as [binary];
    - [date] writes as [DATE], and a [clock] as [TIME] of its unit, [MILLIS] on
      [int32] and [MICROS] or [NANOS] on [int64], with [isAdjustedToUTC]
      [false];
    - a [datetime] writes as [TIMESTAMP] of its unit, an instant with
      [isAdjustedToUTC] [true] and a wall clock with [false];
    - a [list] writes as a [LIST] group, a [record] as a group of its fields,
      and a [tensor] as nested lists, one per dimension, row-major.

    A type without a Parquet type of its own reads back as another. A
    [categorical] reads back as a [string], a [clock] or [datetime] of seconds
    as the same type of milliseconds, an extension as its storage, and a
    [duration] as its [int64] ticks, since the file records no unit.

    {2:statistics Statistics}

    Each column chunk records, as {!source} uses them, its nulls, its NaNs, and
    its minimum and maximum over the values that are not NaN, [-0.] as a zero
    minimum and [+0.] as a zero maximum. A chunk of only NaNs records no
    bounds. A byte string bound longer than 64 bytes is truncated to a prefix
    that ends at a whole UTF-8 character for text, and marked inexact. A
    truncated maximum is the next value past its prefix. When no value lies
    past it, as for a prefix of [0xFF] bytes, the chunk records no maximum. *)

type compression = Uncompressed | Snappy | Gzip | Zstd | Lz4_raw
(** The type for a file's compression, the codec of the pages of every column
    chunk. [Lz4_raw] is Parquet's [LZ4_RAW]. *)

val format :
  ?compression:compression -> ?row_group:int -> Talon.Schema.t -> format
(** [format ?compression ?row_group s] is the format of a file that {!encode}
    writes from a query of schema [s]. Each column has the type it writes and
    reads back as ({{!written}Written types}). [compression], the pages' codec,
    defaults to [Snappy]. [row_group], the number of rows of each row group but
    the last and one batch of {!source}, defaults to [1_048_576].

    Raises [Invalid_argument] if [s] has no column, if [row_group] is not
    positive, or if [s] holds a record without fields or a tensor whose dtype
    no talon scalar type stores, such as a complex one. *)

val encode :
  format ->
  Talon.Query.t ->
  Bytesrw.Bytes.Writer.t ->
  (unit, Talon.Error.t) result
(** [encode f q w] runs [q] with {!Talon.Query.fold} and writes its rows to [w]
    as a Parquet file of format [f], then [Bytesrw.Bytes.Slice.eod]. The bytes
    depend only on [q]'s rows, in order, and on [f], so [q]'s batches do not
    change them. It writes a row group as soon as it holds its rows, and
    holds no more than one row group and the batch that completes it.

    [Error e] where the run fails, as {!Talon.Query.fold} does, if writing to
    [w] raises [Bytesrw.Bytes.Stream.Error] or [Sys_error], or if a value does
    not write: a value that a page of Parquet's 2 GiB cannot hold, or a
    [datetime] of seconds that does not fit milliseconds in [int64]. [w] then
    holds an incomplete Parquet file.

    Raises [Invalid_argument] if [q]'s columns do not write as [f]'s, names and
    types, if [f] leaves a column {{!undeclared}undeclared}, or if a column of
    [f] has a Parquet type {!encode} does not write, such as a [required] one
    from {!sniff}. A format from {!val-format} writes. *)
