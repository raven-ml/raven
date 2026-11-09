(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** RFC 8259 tokens read from a byte stream, a batch of records at a time.

    A scanner reads a {!Bytesrw.Bytes.Reader.t} in two passes. The first finds a
    batch of whole records of about 1 MiB: for {!Lines}, the input up to a line
    feed past 1 MiB, and for {!Array}, up to a comma between elements past 1
    MiB, found by counting brackets outside strings. Only the first pass reads
    the reader, and its cuts depend on the input's bytes alone, so that a record
    is never split across batches and the reader's slices do not change a batch.

    The second pass walks the batch's records with the functions below and
    checks every token. A batch's bytes stay where they are in {!bytes}.

    A failure raises {!Error} at the byte that fails, with its line and column
    counted from [1], in bytes from the line's first byte, and its position in
    the input. A UTF-8 byte order mark at the start of the input is skipped, and
    line 1 starts after it. *)

exception
  Error of {
    line : int;
    column : int;
    byte : int;
    text : string option;
    msg : string;
  }

type framing = Lines | Array

(** {1:scanners Scanners} *)

type t
(** The type for scanners: a reader, a framing, a batch and a position in it. *)

val make : ?sampled:bool -> framing -> Bytesrw.Bytes.Reader.t -> t
(** [make ?sampled framing r] scans the input that [r] reads from its current
    position. Nothing else reads [r] while the scanner does. [sampled] says the
    input was cut after a whole record, so an {!Array} may end before its
    closing bracket. Defaults to [false]. *)

val next : t -> bool
(** [next s] moves [s] to its next batch, and is [false] when no input is left.
    Raises {!Error} when {!Array}'s input ends before its closing bracket, and
    raises what reading the reader raises. *)

val record : t -> bool
(** [record s] moves to the next record of the batch, and is [false] when the
    batch has no record left. {!Lines} skips blank lines. {!Array} reads the
    opening bracket and the commas. Raises {!Error} on bytes after a closed
    array or the array's opening bracket being absent. *)

val end_record : t -> unit
(** [end_record s] checks the record that was just read ends: its line ends for
    {!Lines}, and a comma or the closing bracket follows for {!Array}. *)

(** {1:tokens Tokens}

    Each of these reads at the current position, within the current record:
    past the record's line, or past the batch, the input ends. *)

val pos : t -> int
(** [pos s] is the current position in {!bytes}. *)

val bytes : t -> Bytes.t
(** [bytes s] holds the batch, at the positions the functions give. *)

val peek : t -> char
(** [peek s] is the byte at the position after whitespace, and ['\000'] at the
    end of the record. *)

val expect : t -> char -> unit
(** [expect s c] moves past [c], the next byte after whitespace. *)

val fail : t -> int -> ?text:string -> string -> 'a
(** [fail s p ?text msg] raises {!Error} at the position [p] of the batch. *)

val unexpected : t -> string -> 'a
(** [unexpected s what] raises {!Error} at the current position, saying [what]
    the byte there should have been, or that the record ends there. *)

val literal : t -> string -> unit
(** [literal s lit] moves past [true], [false] or [null], or fails. *)

val number : t -> bool
(** [number s] moves past an RFC 8259 number, and is [true] iff it has neither a
    fraction nor an exponent. *)

val string : t -> bool
(** [string s] moves past a string whose bytes it checks: escapes, control
    characters, UTF-8 and surrogate pairs. It is [true] iff the string holds an
    escape. {!text_pos} and {!text_len} give what lies between its quotes. *)

val text_pos : t -> int
val text_len : t -> int

val enter : t -> char -> bool
(** [enter s close] moves past a container's opening bracket and, if it is
    empty, its closing byte [close], and is [true] iff an item follows. *)

val continue_ : t -> char -> bool
(** [continue_ s close], after an item, moves past a comma and is [true], or
    past the closing byte [close] and is [false]. *)

val member : t -> bool
(** [member s] moves past the name and the colon of an object member, the
    string's bytes checked as by {!string}, and is {!string}'s result.
    {!name_pos} is where the name's opening quote is. *)

val name_pos : t -> int

val unescape : t -> Bytes.t -> int -> int
(** [unescape s b at] writes the text of the string {!string} last read to [b]
    from [at], escapes decoded, and returns its length. It writes at most
    {!text_len} bytes. *)

val skip_value : t -> unit
(** [skip_value s] moves past one value, however deeply it nests, checking its
    tokens. *)

(** {1:sampling Sampling} *)

val sample :
  records:int -> Bytesrw.Bytes.Reader.t -> framing * string
(** [sample ~records r] is the framing of the input that [r] reads, from its
    first byte after whitespace ([\[] for {!Array}, and otherwise {!Lines}), and
    the input up to the end of its first [records] records. It reads those bytes
    and pushes every byte it read back on [r], which reads them again. [records]
    is positive. Raises what reading [r] raises, after which [r] has lost the
    bytes it read. *)
