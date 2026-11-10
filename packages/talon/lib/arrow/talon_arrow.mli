(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Arrow IPC files and streams.

    The {{:https://arrow.apache.org/docs/format/Columnar.html}Arrow IPC format}
    stores a table as a schema and {e record batches}, each a set of columns of
    equal length. The bytes are framed as a {e file}, whose footer indexes its
    batches, or as a {e stream}, read from its start. Each talon reading
    function takes either framing.

    A {e dictionary-encoded} column holds per row an index into a
    {e dictionary} of values. Dictionary batches state dictionaries, each
    under an {e id}. A batch with the {e delta flag} appends to its id's
    dictionary. One without it replaces the dictionary for the record batches
    that follow, which only a stream may do.

    A {!type-format} says how talon reads a file and writes one: each column's
    name, its Arrow type and the talon type ({!Talon.Type}) it reads as, the
    custom metadata of the schema and of each field, and the framing,
    compression and batch size {!encode} writes with. {!pp_format} prints it.

    A read has three steps. Open the file with {!Rig_disk.of_file} and take
    its buffer ({!Rig_disk.buffer}), or take a buffer on the host. {!sniff}
    the bytes into a format and, where a column should read otherwise, change
    it with {!with_type}. Read the rows with {!source}, a {!Talon.Source.t},
    or with {!file} from a path. A write states a format for a query's schema
    with {!val-format}, and {!encode} writes the query's rows as that format
    says.

    Unless a value's doc says otherwise, a function that reads bytes returns
    [Error] for bytes it cannot read, and a function given a format or a name
    raises [Invalid_argument] for a misuse.

    This reads a file into a table:
    {[
      let events () =
        let ( let* ) = Result.bind in
        let* source = Talon_arrow.file "events.arrow" in
        Talon.Query.run (Talon.Query.of_source source)
    ]} *)

(** {1:types Types}

    {!sniff} reads a column as the talon type that holds its values with their
    meaning, as these sections say. {!with_type} changes the type of a
    [fixed_size_list] that reads as a tensor, and declares a decimal.

    {2:scalars Scalars and times}

    - Booleans, integers and floats read as themselves; [utf8], [large_utf8]
      and [utf8_view] as [string]; [binary], [large_binary], [binary_view] and
      a [fixed_size_binary] of at least one byte as [binary].
    - [date32] and [date64] read as [date], a [date64] that is not a midnight
      as the day that holds it. [time32] and [time64] read as [clock] of the
      same unit, and [duration] as [duration] of the same unit.
    - [timestamp] reads as [datetime] of its unit, an instant when the field
      states a time zone and a wall clock otherwise. Talon does not keep a
      zone's name.
    - Reading [utf8_view] or [binary_view] copies each row's value, so such a
      column takes the memory of its rows' values, whatever its views take.

    {2:undeclared Undeclared decimals}

    A [decimal] column, of scale [s], is {e undeclared} as {{!Talon.section-decimals}a
    file's decimal} is, until {!with_type} declares it, and a [list], [struct]
    or extension that holds one is undeclared as a whole. A [struct]'s other
    fields are read as they would be, a text dictionary among them as its value
    type. A scale may be negative, the unscaled integer then multiplied by
    [10{^-s}]. Each row converts when it is read, a dictionary's value per row
    that takes it.

    {2:dictionaries Dictionary-encoded columns}

    A dictionary-encoded field reads as a [categorical] when its values are
    strings, each distinct and none null, and each dictionary its id states
    begins with the one before it, as after a delta or a replacement that keeps
    every entry in place. Any other dictionary-encoded field reads as its value
    type, each record batch through the dictionary that stands for it, and a
    row at a null entry is null. A text dictionary nested in another
    dictionary's values reads as its value type too. Reading as the value type
    copies each row's value, so such a column takes the memory of its rows'
    values, whatever its indices take. Fields may share an id, and then one
    value type.

    {2:nested Lists and records}

    [list], [large_list] and [map] read as [list], and a map's elements are
    records of the file's key and value fields. [struct] reads as [record]. A
    [fixed_size_list] of numbers reads as a [tensor] of that dtype and the
    list's size, and any other as a [list]. A tensor's element is never null,
    so a null element under a row that is not null fails when the column is
    read. {!with_type} reads the column as a [list] instead.

    {2:extensions Extensions}

    A field naming [ARROW:extension:name] reads as an extension over the type
    its storage reads as, with its [ARROW:extension:metadata]. The canonical
    [arrow.fixed_shape_tensor] reads as the [tensor] of its [shape] when its
    storage is a [fixed_size_list] of numbers, the shape's product is the
    list's size and no [permutation] reorders the dimensions, and otherwise as
    that extension.

    {2:empty Columns of no bytes}

    A [fixed_size_binary(0)], a [fixed_size_list(0)] of elements that are not
    numbers, and a [fixed_size_list(w)] whose elements are columns of no bytes,
    a [fixed_size_list(0)] of numbers included, or records of them, read as a
    tensor of no elements of dtype [uint8]: [tensor[uint8, 0]], and
    [tensor[uint8, w×0]] for the last. An element's extension is dropped. A
    [fixed_size_list(0)] of numbers reads as a tensor of its dtype. A row of
    such a column may be null, which its validity bits record, and its values
    take no memory, so the column holds as many rows as a layout does
    ({!Nx_array.Layout.max_numel}). A null element under a row that is not null
    fails when the column is read.

    {1:refusals Refusals}

    {!sniff} returns [Error] for bytes that are not a file or a stream in the
    framing since Arrow 0.15, for a message whose metadata is older than V4,
    and for big-endian data. It also refuses these, naming the column:
    - a [null], [interval], [union], [run_end_encoded], [list_view] or
      [large_list_view] field, at any depth;
    - a dictionary whose values hold a decimal below a [list] or a [struct];
    - a decimal of a precision or width its Arrow type does not allow;
    - a file with two dictionary batches of one id without the delta flag,
      which the format forbids;
    - two fields of one id with different value types;
    - two fields of one name, at the top level or in a struct. *)

(** {1:formats Formats} *)

type format
(** The type for Arrow formats: the framing, compression and record-batch size
    {!encode} writes with, the schema's custom metadata, and each column's
    name, Arrow type, custom metadata and the talon type it reads as. *)

val sniff : Rig.Buffer.t -> (format, Talon.Error.t) result
(** [sniff b] is the format of the file or stream whose bytes are [b], every
    field of its schema, in order, read as {{!types}its Arrow type} says. It
    reads the footer, the headers of the messages and the dictionary batches,
    and no record batch.

    [b] must be a file's buffer ({!Rig_disk.buffer}), which [sniff] maps into
    host memory, or one the host addresses. [Error e] if [b] cannot be
    borrowed on the host, or is {{!refusals}refused}. *)

val with_type : string -> Talon.Type.any -> format -> format
(** [with_type name t f] is [f] with the column [name] read as [t], the type
    {!sniff} gives it, a declaration if it is {{!undeclared}undeclared}, or,
    for a [fixed_size_list] of numbers of at least one element that reads as a
    tensor, a [list] of its elements, which may be null. A column of another
    type is computed once read: with [Expr.cast], or from [binary] to [string]
    with {!Talon.Column.parse}.

    Raises [Invalid_argument] if [f] has no column [name], or if that column
    does not read as [t]; for an undeclared one the message names the types it
    reads as. *)

val metadata : ?column:string -> format -> (string * string) list
(** [metadata ?column f] is [f]'s schema-level custom metadata, or that of the
    field [column], in file order, without the extension keys its type holds.

    Raises [Invalid_argument] if [f] has no column [column]. *)

val with_metadata : ?column:string -> (string * string) list -> format -> format
(** [with_metadata ?column m f] is [f] with the custom metadata [m] on the
    schema, or on the field [column].

    Raises [Invalid_argument] if [f] has no column [column], or if a key of [m]
    starts with [ARROW:extension:]. *)

val pp_format : Format.formatter -> format -> unit
(** [pp_format ppf f] formats [f] for people: a line naming the framing and
    counting its columns, then one line per column, with its name and type, as
    {!Talon.Schema.pp} formats a column, and its Arrow type after an arrow, as
    in [a list[string] ← list<item: utf8>]. A column reads as [undeclared]
    until it is declared. *)

(** {1:reading Reading}

    A source reads one part per record batch ({!Talon.Source}), one batch of
    the request's columns.

    A column shares the file's bytes, except for what a read allocates: a
    compressed buffer, a buffer of [int32] offsets ([string] and [list] hold
    64-bit offsets), a [time32], [date64], [utf8_view] or [binary_view] column,
    a misaligned buffer, offsets for a [fixed_size_binary] or a
    [fixed_size_list] read as a [list], a dictionary's indices that are not
    [int32], and the rows of a dictionary-encoded column that reads as its
    values.

    A read fails with [Error e] as {!sniff} does for the file and the request's
    columns, if the file lacks a column of the request or holds one that does
    not read as the format's type, or if a record batch is malformed. A
    malformed record batch names itself and, for a value at fault, the column
    and row. *)

val source : format -> Rig.Buffer.t -> Talon.Source.t
(** [source f b] is the source, named [arrow], of the rows of the file or
    stream whose bytes are [b], read as [f] says. [b] is a buffer as {!sniff}
    takes. Each read opens the file as {!sniff} does.

    Raises [Invalid_argument] if [f] leaves a column
    {{!undeclared}undeclared}; the message names it and the types it reads as. *)

val file : ?format:format -> string -> (Talon.Source.t, Talon.Error.t) result
(** [file ?format path] is the {!source} of the file or stream at [path], which
    it maps ({!Rig_disk.of_file}), read as [format], or as {!sniff} gives it
    when [format] is absent. It is named [arrow "path"] and states its number
    of rows, and its errors name the file.

    [Error e] if the file cannot be mapped, or as {!sniff}.

    Raises [Invalid_argument] as {!source} does, which a file with an
    {{!undeclared}undeclared} column does when [format] is absent. *)

(** {1:writing Writing}

    {!encode} writes every field nullable, in little-endian IPC metadata V5.
    A value under a null is written zero, a null row of a list or string
    covers no element, and a null record's fields are written null.

    {2:written Written types}

    - Booleans, integers and floats write as themselves, [duration] as
      [duration] of its unit and [record] as [struct].
    - [string], [binary] and [list] write with 64-bit offsets ([large_utf8],
      [large_binary], [large_list]), which map back with no copy.
    - A [categorical] writes as a dictionary of [large_utf8] with [int32]
      indices, before the first record batch.
    - A [clock] writes as [time32] of seconds or milliseconds and [time64] of
      the finer units, a [date] as [date32], and a [datetime] as [timestamp]
      of its unit, zoned [UTC] for an instant.
    - A [tensor] writes as the canonical [arrow.fixed_shape_tensor], and
      another extension as its storage's Arrow type with
      [ARROW:extension:name] and [ARROW:extension:metadata]. *)

type framing = File | Stream
(** The type for framings. A {!File} has a footer and random access, and a
    {!Stream} is read from its start. *)

type compression = Uncompressed | Lz4_frame | Zstd
(** The type for the compression of each buffer. *)

val format :
  ?framing:framing ->
  ?compression:compression ->
  ?batch:int ->
  ?metadata:(string * string) list ->
  Talon.Schema.t ->
  format
(** [format ?framing ?compression ?batch ?metadata s] is the format of what
    {!encode} writes from a query of schema [s]. [framing] defaults to {!File},
    [compression] to {!Uncompressed}, which a reader maps, [batch], the rows of
    every record batch but the last, to [65_536], and [metadata], the schema's
    custom metadata, to none.

    Raises [Invalid_argument] if [batch] is not positive, if a key of
    [metadata] starts with [ARROW:extension:], or if [s] holds a tensor whose
    dtype is not one of Arrow's integers or floats, or a tensor under an
    extension. *)

val encode :
  format ->
  Talon.Query.t ->
  Bytesrw.Bytes.Writer.t ->
  (unit, Talon.Error.t) result
(** [encode f q w] runs [q] with {!Talon.Query.fold} and writes its rows to [w]
    as [f] says, then [Bytesrw.Bytes.Slice.eod]. The bytes depend only on
    [q]'s rows and on [f], so [q]'s batches do not change them. It holds the
    rows of one record batch and the query batch that completes them.

    [Error e] where the run fails, as {!Talon.Query.fold} does, or if writing
    to [w] raises [Bytesrw.Bytes.Stream.Error] or [Sys_error]. [w] then holds
    an incomplete file.

    Raises [Invalid_argument] if [q]'s columns do not write as [f]'s, if [f]
    leaves a column {{!undeclared}undeclared}, or if a column of [f] has an
    Arrow type {!encode} does not write, such as [utf8] or [decimal]. A format
    from {!val-format} writes. *)
