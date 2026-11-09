(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** JSON and NDJSON files.

    A JSON file holds a table as a sequence of objects, one per row, each
    object's members the row's columns. Talon reads two framings of that
    sequence, from a [Bytesrw.Bytes.Reader.t]: {!Lines}, one object per line
    ({{:https://github.com/ndjson/ndjson-spec}NDJSON}), and {!Array}, one JSON
    array of objects ({{:https://www.rfc-editor.org/rfc/rfc8259}RFC 8259}). To
    read compressed text, apply a filter of [Lz_bytesrw] to the reader.

    A {!type-format} says how talon reads a file: its framing and its columns,
    each a name and the talon type its values read as. {!sniff} infers a format
    from a file's first rows, {!val-format} states one, {!with_type} changes the
    type a column reads as, and {!pp_format} prints a format.

    Reading streams. Each framing is cut into batches of whole rows, about one
    MiB of input each, without holding the input: the first bytes of a large
    top-level array yield rows before its last bytes are read, and an array
    that never closes fails only when its end is reached. Only a batch and the
    row it ends in are in memory, so a single row longer than a batch is held
    whole.

    {1:syntax Syntax}

    The input is UTF-8, and every value is RFC 8259 JSON: its bytes are
    checked, including in a value no column reads, so that a read's errors do
    not depend on the columns it requests. A UTF-8 byte order mark at the start
    of the input is not part of the first value, and line 1 starts after it.

    - A string holds any Unicode scalar value. Unpaired surrogate escapes are
      refused. Control characters must be escaped.
    - A number has no limit on digits or exponent, except that its column's type
      must hold it.
    - Values nest to any depth. A {!type-format} that declares no column at a
      depth reads past it without using the call stack.
    - An object's names are compared byte for byte, after decoding escapes. An
      object that names a key the format reads twice fails the read: RFC 8259
      says the result is unpredictable, and a silent choice would be a wrong
      value. A key the format does not read is skipped without this check.

    {!Lines} is one JSON value per line, each line ended by a line feed or the
    end of the input; a carriage return before a line feed, and blank lines, are
    allowed, and a line break inside a value is not, as NDJSON says. An
    {!Array} is one array whose elements are the rows; only whitespace may
    follow it.

    {1:values Values}

    A row is an object. A key the format does not name is skipped, and a key
    the format names that a row lacks reads as null, like the JSON [null]. By
    type:
    - [bool] reads [true] and [false];
    - the integer types read a number with neither fraction nor exponent, in
      the type's range;
    - the float types read a number, the nearest value of the type, ties to
      even. A number beyond the type's range fails instead of reading as an
      infinity;
    - [string] and [binary] read a string as its decoded text, and any other
      value as its JSON text from the input, exactly as written;
    - a categorical, [date] or [datetime] reads a string, as
      {!Talon.Column.parse} reads the type's text;
    - a list reads an array, element by element, each as the list's element
      type, and a record reads an object as a row is read: its keys are the
      record's fields, in any order, an absent field is null.

    A value of another JSON type than the column's fails the read, with its
    line, column and byte, its text and the fix.

    {1:sniffing_rules Sniffing}

    {!sniff} reads a file's first rows to infer its format. A [\[] as the first
    value picks {!Array}, and a [{] picks {!Lines}. A column's type is inferred
    from every value it has in those rows, each key's own column:
    - [bool] if every value is [true] or [false];
    - [int64] if every value is an integer that [int64] holds;
    - [float64] if every value is a number and at least one is not such an
      integer, none being an integer beyond [int64];
    - a list if every value is an array, of the type of its elements taken
      together, and [string] for an empty list's elements;
    - a record if every value is an object, of the fields of its objects in the
      order they first appear, each field its own column;
    - [string] otherwise: strings, integers beyond [int64], a column whose
      values are of several JSON types, and a column of no value.

    Sniffing never infers a date, a datetime or a categorical: declare those
    with {!with_type}. A value more than 64 levels of nesting deep, the row's
    object being the first level, sniffs as [string], holding its JSON text. *)

(** {1:formats Formats} *)

type framing =
  | Lines  (** One object per line, as NDJSON. *)
  | Array  (** One array of objects, as RFC 8259 writes a table. *)

(** The type for how a file holds its rows. *)

type format
(** The type for JSON formats: a {!type-framing} and the file's columns, in
    order, each with its name and the type its values read as. A format that
    {!sniff} made records how many rows it inferred its types from and which
    types it inferred, so that a value that fails to read says so. *)

val format : ?framing:framing -> (string * Talon.Type.any) list -> format
(** [format ?framing columns] is the format of a file whose rows hold [columns],
    in order, each a name and the type its values read as. [framing] defaults
    to {!Lines}.

    Raises [Invalid_argument] if [columns] is empty, if two columns have the
    same name or a name is not valid UTF-8, or if a type is not one JSON reads
    ({{!values}values}). *)

val with_type : string -> Talon.Type.any -> format -> format
(** [with_type name t f] is [f] with the column [name] read as [t]. The type is
    declared: a value that fails to read as it no longer cites sniffing.

    Raises [Invalid_argument] if [f] has no column [name], or if [t] is not a
    type JSON reads ({{!values}values}). *)

val pp_format : Format.formatter -> format -> unit
(** [pp_format ppf f] formats [f] for people: a line with its number of columns
    and its framing, followed, when {!sniff} made [f], by the number of rows its
    types were sniffed from; then one line per column, with its name and its
    type as {!Talon.Schema.pp} formats a column, and, aligned after them,
    [declared] for a column whose type {!with_type} changed in a sniffed format:
    {v
    json (2 columns, lines), types sniffed from 2 rows
      id int64
      name string
    v} *)

(** {1:sniffing Sniffing} *)

val sniff :
  ?rows:int -> Bytesrw.Bytes.Reader.t -> (format, Talon.Error.t) result
(** [sniff ?rows r] is the format of the JSON text that [r] reads, inferred from
    its first rows as {{!sniffing_rules}sniffing} says. [rows] is the number of
    rows types are inferred from. Defaults to [16384].

    [r] is left as it was: the bytes [sniff] reads are pushed back on [r], which
    then reads its whole input from where it was, unless reading it failed.

    [Error e] if [r] reads no row, if a sampled row is not an object or an
    object sampled for columns names a key twice, if a sampled row breaks the
    {{!syntax}syntax}, if every sampled row is an empty object, or if reading
    [r] raises [Bytesrw.Bytes.Stream.Error] or [Sys_error]. When the input fails
    at a byte, [e] gives its line, column and position.

    Raises [Invalid_argument] if [rows < 1]. *)

(** {1:reading Reading} *)

val decode : format -> Bytesrw.Bytes.Reader.t -> (Talon.t, Talon.Error.t) result
(** [decode f r] is the table of the rows that [r] reads, read as [f] says, in
    batches of about one MiB of input. The batches are fixed by the bytes of
    [r]'s input, never by how [r] slices them.

    [Error e] at the earliest value that fails: a row that breaks the
    {{!syntax}syntax}, that is not an object, or that names a key the format
    reads twice, a value that does not read as its column's type, or a read of
    [r] that raises [Bytesrw.Bytes.Stream.Error] or [Sys_error]. {!Lines} reads
    an empty input as no row, and {!Array} fails on it. When the input fails at
    a byte, [e] gives its line, column and position. *)

val source : format -> (unit -> Bytesrw.Bytes.Reader.t) -> Talon.Source.t
(** [source f open_] is the source, named [json], of the rows that the readers
    [open_ ()] read, as [f] says, in {!decode}'s batches. Each read of the
    source calls [open_] once. It keeps the request's columns only, stops after
    its limit, and applies no conjunct. Its errors are {!decode}'s, for the rows
    it reads, which checks each row's syntax whatever the request names. *)

val file :
  ?format:format -> string -> (Talon.Source.t, Talon.Error.t) result
(** [file ?format path] is the {!source} of the JSON file [path], named
    [json "path"], whose errors name [path]: as [format] says, or as {!sniff}
    infers from the file when [format] is absent. Each read opens the file
    again.

    [Error e] if the file cannot be opened, or as {!sniff}. *)

(** {1:writing Writing} *)

val encode :
  ?eod:bool ->
  format ->
  Talon.Query.t ->
  Bytesrw.Bytes.Writer.t ->
  (unit, Talon.Error.t) result
(** [encode ?eod f q w] runs [q] and writes its rows to [w] as the rows of a
    file of format [f], which {!decode}[ f] reads back to [q]'s rows. A row is
    an object of every column, null columns included. Text is written with its
    bytes and the escapes JSON requires, a float in the fewest digits that read
    back to it with [.0] when they would read as an integer, and a date or
    datetime as text. A {!Lines} row ends with a line feed. An {!Array} writes
    its rows between [\[] and [\]], one per line. [eod] indicates whether
    [Bytesrw.Bytes.Slice.eod] is written on [w] after the rows, when they are
    all written: the caller decides when [w] ends. Defaults to [false].

    [Error e] if running [q] fails, if a float is an infinity or NaN, which
    JSON has no text for, or a [binary] value is not valid UTF-8, at the line
    the row is on, or if writing to [w] raises [Bytesrw.Bytes.Stream.Error] or
    [Sys_error]. The rows before the failure are written.

    Raises [Invalid_argument] if [q]'s columns are not [f]'s, names and types,
    in order. *)
