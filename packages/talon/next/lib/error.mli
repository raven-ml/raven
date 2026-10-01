(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Failures found in data and in the environment.

    Reading a file that is missing or malformed, or data that breaks a query's
    contract, is not a programming error, so talon returns it as [Error e],
    where [e] says what failed and where it was found: in which file, at which
    line and column of its text, in which row group, at which bytes, and on
    which raw text. Formats and sources build errors with {!v}; programs print
    them with {!pp}. *)

type t
(** The type for errors. An error is a message and the places it was found at,
    each of which may be unknown. *)

val v :
  ?file:string ->
  ?line:int ->
  ?column:int ->
  ?row_group:int ->
  ?bytes:int * int ->
  ?text:string ->
  string ->
  t
(** [v ?file ?line ?column ?row_group ?bytes ?text msg] is the error [msg],
    found:
    - in [file], a path as the program named it;
    - at [line] of [file]'s text and at [column] of that line, both counted from
      [1], the column in bytes;
    - in the row group [row_group] of a Parquet file, counted from [0];
    - at the bytes [bytes] of [file], [(first, last)], the zero-based positions
      of the first and the last byte of the range, both included;
    - on [text], the raw bytes that failed to read, which need not be valid
      UTF-8.

    [msg] says what failed and, when there is one, how to fix it, in one or more
    sentences: [cannot read as float64. Declare the null token (~nulls).]

    Raises [Invalid_argument] if [line < 1], [column < 1], [column] is given
    without [line], [row_group < 0], or [bytes] is given and [first < 0] or
    [last < first]. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf e] formats [e] for people: the places [e] was found at, coarsest
    first, then its message, separated by [": "]:
    - the file, followed by [:line] and [:line:column] as compilers write them,
      [flights.csv:48213:12]; without a file, [line 48213] or
      [line 48213, column 12];
    - the row group, [row group 3];
    - the bytes, [bytes 106-113], or [byte 106] for a range of one byte;
    - the text, between double quotes, with double quotes and backslashes
      preceded by a backslash. Each byte of a control character (U+0000 to
      U+001F, U+007F to U+009F), of a bidirectional formatting control (U+202A
      to U+202E, U+2066 to U+2069), which could reorder the text around it, and
      each byte that is not part of valid UTF-8, is written as [\x] and two
      hexadecimal digits. A text longer than 64 bytes is cut before the first
      byte past its 64th, or before a valid UTF-8 sequence that this byte would
      split, and an ellipsis follows the closing quote: ["aaaa"…].

    As in [flights.csv:48213:12: "NA": cannot read as float64.] and
    [zoneinfo/Europe/Paris: bytes 106-113: transition 1 is not after the
     previous one].

    A failure that a run finds in the data has none of these places: its message
    starts with the plan step that found it, as [Query.pp] prints it, and the
    row of the step's input it was found at, counted from [0]:
    [filter (cast int32 x > 0): row 48212: cannot cast 3.5 to int32.] *)

val get_ok : ('a, t) result -> 'a
(** [get_ok r] is [v] if [r] is [Ok v].

    Raises [Failure] with [e] formatted by {!pp} if [r] is [Error e]. *)
