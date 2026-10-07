(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Structures in memory around values.

    A launch descriptor ({!Qmd}) is a structure in memory: bytes whose fields
    hold integers the layout knows, and values of the caller. Each field a value
    fills is a {e hole}. As with {!Packet}, the caller interprets the
    description: {!encode} with integers, a compiler as its own nodes. Every
    structure comes from {!Qmd.structure}. *)

type 'v hole = private {
  at : int;  (** The offset of the field's first byte. *)
  bits : int;
      (** The field's width, from bit [0] of that byte, in \[[1];[64]\]. *)
  value : 'v Packet.term;  (** The term whose low [bits] bits fill the field. *)
}
(** The type for holes. A hole's {e word} is the little-endian word of the
    narrowest of 1, 2, 4 and 8 bytes that holds its [bits], at [at]. Filling the
    hole replaces the word's low [bits] bits with the term's, and keeps the
    word's other bits, which belong to the fields beside it.

    For example, a 24-bit field at offset [8] has the 4-byte word at [8]: its
    value's low 24 bits go in bytes [8] to [10], and byte [11] keeps the field
    that follows. *)

type 'v t = private {
  bytes : string;
      (** The structure's bytes: its known fields, and zeros in its holes'
          fields. *)
  holes : 'v hole list;
      (** Its holes, by increasing [at]. Their words lie in [bytes] and do not
          overlap. *)
}
(** The type for structures around values of type ['v]. *)

val encode : ('v -> int64) -> 'v t -> string
(** [encode value s] is [s.bytes] with each hole filled by its term, each value
    [v] taken as the 64-bit unsigned integer [value v].

    Raises [Invalid_argument] if a shift is outside \[[0];[63]\]. *)
