(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Arrow IPC files and streams.

    The {{:https://arrow.apache.org/docs/format/Columnar.html}Arrow IPC format}
    stores a table as a schema and record batches of equal length columns,
    framed as a {e file}, with a footer that indexes its batches, or as a
    {e stream}, read from its start. Talon reads both from memory mapped with
    {!Rig_disk.of_file}: {!sniff} a schema into a {!type-format}, change how a
    column reads with {!with_type}, and read it with {!source} or {!file}.
    {!encode} writes a query's rows in either framing.

    A format states each column's name, Arrow type and talon type, printed by
    {!pp_format}, and carries the custom metadata of schema and field
    ({!metadata}).

    {b Copies.} A column's buffers map the file with no copy, except for one
    copy of: a compressed buffer, an [int32] offsets buffer ([string] and
    [list] hold 64-bit offsets), [time32], [date64], [utf8_view], [binary_view],
    dictionary indices that are not [int32] and a misaligned buffer.

    {1:types Types}

    A column reads as the type that holds its values with their meaning:
    - booleans, integers and floats as themselves; [utf8], [large_utf8] and
      [utf8_view] as [string]; [binary], [large_binary], [binary_view] and
      [fixed_size_binary] as [binary];
    - [date32] and [date64] as [date], a [date64] that is not a midnight as the
      day that holds it, and [time32] and [time64] as [clock] and [duration] as
      [duration], each keeping its unit;
    - [timestamp] as [datetime] of its unit: an instant when the field states
      a time zone, whose name talon does not keep, a wall clock otherwise;
    - a dictionary-encoded field as a [categorical] when its values are
      strings, distinct and not null, in one dictionary for the file, deltas
      included;
    - [list], [large_list] and [map] as [list], a map being its list of [key]
      and [value] records, and [struct] as [record];
    - [fixed_size_list] of numbers as a [tensor] of that dtype and the list's
      size, and otherwise as a [list]. A tensor's element is never null: a null
      element under a row that is not null fails when the column is read;
    - a field naming [ARROW:extension:name] as an extension over the type its
      storage reads as, with its [ARROW:extension:metadata]. The canonical
      [arrow.fixed_shape_tensor] reads as the [tensor] of its [shape], unless a
      [permutation] reorders the dimensions, when it is that extension.

    {1:refusals Refusals}

    {!sniff} refuses bytes that are neither a file nor a stream in the framing
    since Arrow 0.15, metadata older than V4 and big-endian data, and, naming
    the column:
    - a [null], [decimal], [interval], [union], [run_end_encoded], [list_view]
      or [large_list_view] field, at any depth (a decimal is refused for now);
    - a field dictionary-encoded over anything but strings, or whose dictionary
      changes between batches, holds a null or holds a string twice;
    - two fields of one name, at the top level or in a struct.

    A read refuses the zero-width columns (a [fixed_size_binary(0)] or a
    non-numeric [fixed_size_list(0)]) that state more rows than eight per byte
    of their record batch's body, which no other column bounds.

    {1:written Written files}

    {!encode} writes every field nullable, in little-endian IPC metadata V5:
    [string], [binary] and [list] with 64-bit offsets ([large_utf8],
    [large_binary], [large_list]), which map back with no copy; a [categorical]
    as a dictionary of [large_utf8] with [int32] indices, before the first
    record batch; a [clock] as [time32] of seconds or milliseconds and [time64]
    of the finer units; a [date] as [date32]; a [datetime] as [timestamp] of its
    unit, zoned [UTC] for an instant; a [tensor] as the canonical
    [arrow.fixed_shape_tensor]; another extension as its storage's Arrow type
    with [ARROW:extension:name] and [ARROW:extension:metadata].

    A value under a null is written zero, a null row of a list or string covers
    no element, and a null record's fields are written null, so a file depends
    only on the rows and the format. *)

(** {1:formats Formats} *)

type format
(** The type for Arrow formats: the framing, compression and record-batch size
    {!encode} writes with, the schema's custom metadata, and each column's name,
    Arrow type, custom metadata and the talon type it reads as. *)

val sniff : Rig.Buffer.t -> (format, Talon.Error.t) result
(** [sniff b] is the format of the file or stream whose bytes are [b]: every
    field of its schema, in order, reading as {{!types}its Arrow type} says. It
    reads the footer or the message headers and the dictionary batches, and no
    record batch.

    [b] is a buffer of the disk ({!Rig_disk.of_file}), which [sniff] maps into
    host memory, or one the host addresses. [Error e] if [b] cannot be borrowed
    on the host, or is {{!refusals}refused}. *)

val with_type : string -> Talon.Type.any -> format -> format
(** [with_type name t f] is [f] with the column [name] read as [t]: the type
    {!sniff} gives it or, for a column of [fixed_size_list] that reads as a
    tensor, a [list] of its elements, which may be null. Any other type is a
    computation on the column once read, such as [Expr.cast].

    Raises [Invalid_argument] if [f] has no column [name], or if that column
    does not read as [t]. *)

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
    {!Talon.Schema.pp} formats a column, and its Arrow type after an arrow:
    [a list[string] ← list<item: utf8>]. *)

(** {1:reading Reading} *)

val source : format -> Rig.Buffer.t -> Talon.Source.t
(** [source f b] is the source, named [arrow], of the rows of the file or stream
    whose bytes are [b], read as [f] says. [b] is a buffer as {!sniff} takes.
    Each read opens the file as {!sniff} does. Each record batch is one part, of
    one batch of the request's columns.

    A read fails with [Error e] as {!sniff} does, if the file lacks a column of
    the request or holds one that does not read as [f]'s type, or if a record
    batch is malformed: [e] names the record batch and, for a value at fault,
    the column and row. *)

val file : ?format:format -> string -> (Talon.Source.t, Talon.Error.t) result
(** [file ?format path] maps [path] ({!Rig_disk.of_file}) and is its
    {!source}, read as [format] or as {!sniff} gives when [format] is absent,
    named [arrow "path"], with its number of rows. Its errors name the file.

    [Error e] if the file cannot be mapped, or as {!sniff}. *)

(** {1:writing Writing} *)

(** The type for framings: a {!File} has a footer and random access, a {!Stream}
    is read from its start. *)
type framing = File | Stream

(** The type for the compression of each buffer. *)
type compression = Uncompressed | Lz4_frame | Zstd

val format :
  ?framing:framing ->
  ?compression:compression ->
  ?batch:int ->
  ?metadata:(string * string) list ->
  Talon.Schema.t ->
  format
(** [format ?framing ?compression ?batch ?metadata s] is the format of what
    {!encode} writes from a query of schema [s]: [framing] defaults to {!File};
    [compression] to {!Uncompressed}, which a reader maps; [batch], the rows of
    every record batch but the last, to [65_536]; [metadata], the schema's
    custom metadata, to none.

    Raises [Invalid_argument] if [batch] is not positive, if a key of
    [metadata] starts with [ARROW:extension:], or if [s] holds a tensor whose
    dtype Arrow does not store or a tensor under an extension. *)

val encode :
  format ->
  Talon.Query.t ->
  Bytesrw.Bytes.Writer.t ->
  (unit, Talon.Error.t) result
(** [encode f q w] runs [q], folding its batches ({!Talon.Query.fold}), and
    writes its rows to [w] as [f] says, then [Bytesrw.Bytes.Slice.eod]. A file's
    bytes depend only on [q]'s rows and on [f]: batches never change them. It
    holds one record batch while it writes it.

    [Error e] where the run fails, as {!Talon.Query.fold} does, or if writing
    to [w] raises [Bytesrw.Bytes.Stream.Error] or [Sys_error]. [w] then holds
    the bytes written so far, and no complete file.

    Raises [Invalid_argument] if [q]'s columns do not write as [f]'s, or if a
    column of [f] has an Arrow type {!encode} does not write, such as [utf8]
    from {!sniff}; {!val-format} states one that it does. *)
