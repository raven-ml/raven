(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** CSV files.

    A CSV file is text: records on lines, each record's fields separated by a
    separator. Talon reads it as
    {{:https://www.rfc-editor.org/rfc/rfc4180}RFC 4180} writes it, strictly,
    from a {!Bytesrw.Bytes.Reader.t}.

    A {!type-format} says how talon reads a file: its dialect (the separator,
    the quote, whether a header names the columns, and the text that means null)
    and its columns, each a name and the talon type its fields read as. {!sniff}
    infers a format from a file's first records, {!format} states one,
    {!with_type} changes the type a column reads as, and {!pp_format} prints a
    format.

    {1:syntax Syntax}

    - A record ends at a line feed, or at a carriage return followed by a line
      feed. The last record may end at the end of the input instead.
    - An empty line, with no byte between its line breaks, is not a record: it
      is skipped, wherever it is. A one-column file therefore writes a null as a
      null token, never as an empty line.
    - Every record has one field per column, the header included.
    - A field that starts with the quote is {e quoted}. It ends at the next
      quote that is not doubled, which a separator, a line break or the end of
      the input must follow. Between its quotes it holds any byte, line breaks
      included, and a doubled quote stands for one quote.
    - A field that does not start with the quote holds every byte up to the next
      separator or line break, spaces included. It holds no quote, and no
      carriage return.
    - A UTF-8 byte order mark at the start of the input is not part of the first
      field, and line 1 starts after it.

    {1:values Values}

    An unquoted empty field is null, and so is an unquoted field equal to one of
    the format's null tokens. A quoted field is never null: [""] is the empty
    string. Every other field's text, without its quotes and with its doubled
    quotes undoubled, reads as its column's type as {!Talon_next.Column.parse}
    reads it. CSV reads [bool], the integer, float and decimal types, [string],
    [binary], categoricals, [date] and [datetime]: read a time of day as
    [string], then convert it with [Expr.Temporal.parse].

    A field that does not read as its type fails the read, with its line and
    column, its text and the fix.

    {1:sniffing Sniffing}

    {!sniff} reads a file's first records to infer its format, with the quote
    ['"'] and the null tokens it is given:
    {ul
     {- {b The separator}: [','], a tab, [';'] or ['|'], whichever splits every
        record into the same number of fields, more than one, and into the most
        fields; the first of that list on a tie. A file that none of them splits
        has one column, and its separator is [','].
     }
     {- {b The header}: the first record names the columns, unless at least one
        column has a type other than [string], inferred from the other records,
        and the first record's field in every such column is null or reads as
        that type. Without a header, columns are named [column_1], [column_2],
        ….
     }
     {- {b The types}, from the non-null fields of the records after the header:
        - [bool] if every value is [true] or [false];
        - [int64] if every value is a decimal integer that [int64] holds;
        - [float64] if every value is a number and at least one is not such an
          integer;
        - [date] if every value is a date;
        - [datetime[us]] if every value is a datetime without an offset, and
          [datetime[us, UTC]] if every value is one with an offset, in
          nanoseconds ([ns]) instead when a value is not a whole number of
          microseconds and every value is in the nanoseconds' range;
        - [string] otherwise: for text, for integers that [int64] does not hold,
          and for a column with no value.
     }
    }

    A number with a leading zero, such as [007] or [01.5], is text: it writes an
    identifier, whose zeros a number would lose. Sniffing never infers a
    categorical. *)

(** {1:formats Formats} *)

type format
(** The type for CSV formats: a dialect, and the columns of the file's records,
    in order, each with its name and the type its fields read as. A format that
    {!sniff} made records from how many rows it inferred its types and which
    types it inferred, so that a field that fails to read says so. *)

val format :
  ?sep:char ->
  ?quote:char ->
  ?header:bool ->
  ?nulls:string list ->
  (string * Talon_next.Type.any) list ->
  format
(** [format ?sep ?quote ?header ?nulls columns] is the format of a file whose
    records hold [columns], in order, each a name and the type its fields read
    as, with:
    - [sep], the separator. Defaults to [','].
    - [quote], the quote. Defaults to ['"'].
    - [header], [true] if the first record names the columns. It must then name
      [columns] in order, byte for byte. Defaults to [true].
    - [nulls], the null tokens: an unquoted field equal to one of them is null.
      Defaults to [[]], so that only unquoted empty fields are null.

    Raises [Invalid_argument] if [columns] is empty, if two columns have the
    same name or a name is not valid UTF-8, if a type is not one CSV reads
    ({{!values}values}), if [sep] or [quote] is a line feed, a carriage return
    or a byte outside ASCII, if [sep] equals [quote], or if a null token holds
    the separator, the quote, a line feed or a carriage return. *)

val with_type : string -> Talon_next.Type.any -> format -> format
(** [with_type name t f] is [f] with the column [name] read as [t]. The type is
    declared: a field that fails to read as it no longer cites sniffing.

    Raises [Invalid_argument] if [f] has no column [name], or if [t] is not a
    type CSV reads ({{!values}values}). *)

val pp_format : Format.formatter -> format -> unit
(** [pp_format ppf f] formats [f] for people: a line with the number of columns
    and the dialect, followed, when {!sniff} made [f], by the number of rows its
    types were sniffed from; then one line per column, with its name and its
    type as {!Talon_next.Schema.pp} formats a column, and, aligned after them,
    [declared] for a column whose type {!with_type} changed in a sniffed format:
    {v
    csv (3 columns, separator ',', quote '"', header), types sniffed from 16,384 rows
      carrier string
      dep_delay float32        declared
      "departs at" datetime[us]
    v}
    The dialect lists the null tokens, as in [nulls ["NA"]], when there are
    some. *)

(** {1:sniffing Sniffing} *)

val sniff :
  ?rows:int ->
  ?nulls:string list ->
  Bytesrw.Bytes.Reader.t ->
  (format, Talon_next.Error.t) result
(** [sniff ?rows ?nulls r] is the format of the CSV text that [r] reads,
    inferred from its first records as {{!sniffing}sniffing} says, with:
    - [rows], the number of records after the header that types are inferred
      from. Defaults to [16384].
    - [nulls], the format's null tokens, whose fields types are not inferred
      from, as from unquoted empty fields. Defaults to [[]].

    [r] is left as it was: the bytes [sniff] reads are pushed back on [r], which
    then reads its whole input from where it was, unless reading it failed.

    [Error e] if [r] reads no record, if a separator splits a sampled record but
    none splits every sampled record into the same number of fields, if a
    sampled record breaks the {{!syntax}syntax}, if the header names two columns
    the same or a name is not valid UTF-8, or if reading [r] raises
    {!Bytesrw.Bytes.Stream.Error} or [Sys_error]. [e] gives the line and column
    where the input fails.

    Raises [Invalid_argument] if [rows < 1], or if a null token holds the quote,
    a line break or one of the separators sniffing chooses from. *)

(** {1:reading Reading} *)

val decode :
  format -> Bytesrw.Bytes.Reader.t -> (Talon_next.t, Talon_next.Error.t) result
(** [decode f r] is the table of the records that [r] reads, read as [f] says,
    in batches of about one MiB of input. The batches are fixed by the bytes of
    [r]'s input, never by how [r] slices them.

    [Error e] at the earliest record, and its first field, that fails: a header
    that does not name [f]'s columns, a record that breaks the {{!syntax}syntax}
    or has another number of fields, a field that does not read as its column's
    type, or a read of [r] that raises {!Bytesrw.Bytes.Stream.Error} or
    [Sys_error]. An empty input fails when [f] has a header. [e] gives the line
    and column. *)

val source : format -> (unit -> Bytesrw.Bytes.Reader.t) -> Talon_next.Source.t
(** [source f open_] is the source, named [csv], of the records that the readers
    [open_ ()] read, as [f] says, in {!decode}'s batches. Each read of the
    source calls [open_] once. It reads the request's columns only, stops after
    its limit, and applies no conjunct. Its errors are {!decode}'s, for the
    records it reads. *)

val file :
  ?format:format ->
  ?nulls:string list ->
  string ->
  (Talon_next.Source.t, Talon_next.Error.t) result
(** [file ?format ?nulls path] is the {!source} of the CSV file [path], named
    [csv "path"], whose errors name [path]: as [format] says, or as {!sniff}
    [?nulls] infers from the file when [format] is absent. Each read opens the
    file again.

    [Error e] if the file cannot be opened, or as {!sniff}.

    Raises [Invalid_argument] if both [format] and [nulls] are given, since a
    format holds its null tokens ({!format}), or as {!sniff} does on [nulls]. *)
