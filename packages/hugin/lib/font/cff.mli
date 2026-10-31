(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** CFF and CFF2 outlines: the ['CFF '] and ['CFF2'] tables of OpenType, their
    Type 2 charstrings run to paths, and the ['CFF '] table of a subset.

    References: Adobe Technical Note #5176, {e The Compact Font Format
    Specification}; Adobe Technical Note #5177, {e The Type 2 Charstring
    Format}; the OpenType specification's {e CFF2 — Compact Font Format
    (CFF) Version 2} table.

    Coordinates are in font units, y up, as charstrings give them. Decoding and
    running raise {!Malformed} or {!Unsupported} on data they cannot read, and
    nothing else: every read is checked against the table's end, the argument
    stack holds at most 48 numbers in CFF and 513 in CFF2, subroutines nest at
    most 10 deep, and a glyph runs at most 65535 bytes of charstring and
    subroutine code. A variable CFF2 font runs as its default instance, where
    every [blend] gives its default values. *)

exception Malformed of string
exception Unsupported of string

type t
(** The type for decoded CFF and CFF2 tables. *)

val decode : cff2:bool -> glyphs:int -> string -> off:int -> len:int -> t
(** [decode ~cff2 ~glyphs s ~off ~len] is the ['CFF '] table, or the ['CFF2']
    table if [cff2], at [off] in [s], [len] bytes long, whose CharStrings must
    number [glyphs]. It reads the structure; {!run} reads a glyph's
    charstring. *)

(** The events a charstring draws, in order: contours start with [move] and
    end with [close], which also closes a contour by a line back to its start
    where it does not end there. *)
type sink = {
  move : float -> float -> unit;
  line : float -> float -> unit;
  curve : float -> float -> float -> float -> float -> float -> unit;
  close : unit -> unit;
}

val run : t -> int -> sink -> unit
(** [run c g sink] runs the charstring of glyph [g], which must be one of [c],
    giving [sink] what it draws. *)

val charstring : t -> int -> string
(** [charstring c g] is the outline of glyph [g] as one Type 2 charstring
    without hints, width or subroutines: the events {!run} gives, written with
    [rmoveto], [rlineto], [rrcurveto] and [endchar], each delta a number the
    charstring holds exactly. A delta beyond the range of a charstring number
    is written as a sum with [add]. Running it gives the same events.

    Raises {!Unsupported} if it would take more than 65535 bytes. *)

val check : t -> int -> sink -> unit
(** [check c g sink] is [run c g sink], then raises as {!charstring} would. *)

val table : name:string -> string array -> string
(** [table ~name charstrings] is a CID-keyed ['CFF '] table of one font named
    [name] whose glyph [i], CID [i], is [charstrings.(i)]: registry [Adobe],
    ordering [Identity], one font DICT with an empty Private DICT, and no
    subroutines. [charstrings] must not be empty. *)
