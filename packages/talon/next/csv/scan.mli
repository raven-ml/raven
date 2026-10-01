(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** RFC 4180 records read from a byte stream.

    A scanner reads a {!Bytesrw.Bytes.Reader.t} in batches of records, and
    splits each batch into records and fields by the syntax that
    {!Talon_next_csv} documents, skipping empty lines. A batch's fields stay
    where they are in the batch's bytes, and the scanner reuses its buffers from
    one batch to the next, so scanning allocates nothing per record or field.

    A batch is found in two passes. The first reads the input until a line feed
    outside quotes, counting every quote as opening or closing quoted text, ends
    the batch's input at or past 1 MiB: in input that follows the syntax, that
    is where a record ends. The second splits the batch exactly, and reports the
    first byte that breaks the syntax. Batch boundaries therefore depend on the
    input's bytes only, never on the slices the reader returns them in.

    Lines and columns count from [1], columns in bytes from the line's first
    byte. A UTF-8 byte order mark at the start of the input is skipped, and line
    1 starts after it. *)

exception Error of { line : int; column : int; msg : string }
(** [Error {line; column; msg}] is raised for input that breaks the syntax, at
    the byte [column] of the line [line], for the reason [msg]. *)

(** {1:scanners Scanners} *)

type t
(** The type for scanners: a reader, a separator and a quote, and the batch the
    scanner is at. *)

val make : sep:char -> quote:char -> Bytesrw.Bytes.Reader.t -> t
(** [make ~sep ~quote r] scans the input that [r] reads from its current
    position. [sep] and [quote] are distinct bytes other than a line feed and a
    carriage return. Nothing else reads [r] while the scanner does. *)

val next : t -> bool
(** [next s] moves [s] to its next batch of records, and is [false] when no
    record is left. The previous batch's positions are then invalid.

    A batch ends before the first record that breaks the syntax, and the next
    call raises {!Error} at the byte that breaks it, or at the opening quote of
    a quoted field that does not end. A batch has at least one record. Raises
    what reading the reader raises. *)

(** {1:fields Records and fields}

    Records count from [0] in the batch, and so do the fields of a record. *)

val rows : t -> int
(** [rows s] is the number of records of [s]'s batch. *)

val fields : t -> int -> int
(** [fields s r] is the number of fields of record [r]. *)

val bytes : t -> Bytes.t
(** [bytes s] holds the batch's input, at the positions the functions below
    give. *)

val quoted : t -> int -> int -> bool
(** [quoted s r j] is [true] iff field [j] of record [r] is quoted. *)

val pos : t -> int -> int -> int
(** [pos s r j] is the position of the first byte of the text of field [j] of
    record [r]: after the opening quote of a quoted field. *)

val len : t -> int -> int -> int
(** [len s r j] is the length of the text of field [j] of record [r]: between
    the quotes of a quoted field, whose doubled quotes are still doubled. *)

val raw : t -> int -> int -> string
(** [raw s r j] is the text of field [j] of record [r] as written: its bytes
    from {!pos}, of length {!len}. *)

val text : t -> int -> int -> string
(** [text s r j] is the value of field [j] of record [r]: {!raw}, with doubled
    quotes undoubled. *)

val start : t -> int -> int -> int
(** [start s r j] is the position of the first byte of field [j] of record [r]:
    its opening quote when it is quoted. *)

val stop : t -> int -> int -> int
(** [stop s r j] is the position after the last byte of field [j] of record [r]:
    after its closing quote when it is quoted. *)

val locate : t -> int -> int * int
(** [locate s i] is the line and the column of the byte at position [i] of
    [bytes s], which is in the batch or ends it. *)

(** {1:sampling Sampling} *)

val sample : quote:char -> records:int -> Bytesrw.Bytes.Reader.t -> string
(** [sample ~quote ~records r] is the input that [r] reads, up to the end of its
    [records]-th record, or all of it if it has fewer. Records end as {!next}'s
    first pass finds them, at line feeds that end a line with a byte other than
    a carriage return. [sample] reads those bytes, then pushes back on [r] every
    byte it read, which [r] reads again. [records] is positive.

    Raises what reading [r] raises, after which [r] has lost the bytes it read.
*)
