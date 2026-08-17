(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Character classes.

    The classes pre-tokenizers split on, as a code point indexed table filled
    from {!Uucp} on first touch. The ASCII half is a separate 128 byte table, so
    text that stays in ASCII never allocates the Unicode one. A class test is
    then a byte load.

    Filling is idempotent, so classifying the same code point on two domains at
    once is harmless. *)

(** {1 Categories} *)

val other : int
(** [other] is the category of everything the three below do not cover. *)

val whitespace : int
(** [whitespace] is the category of the White_Space property. *)

val letter : int
(** [letter] is the category of the general categories [Lu], [Ll], [Lt], [Lm]
    and [Lo], that is of the regular expression class [\p{L}]. *)

val numeric : int
(** [numeric] is the category of the general categories [Nd], [Nl] and [No],
    that is of the regular expression class [\p{N}]. *)

val category : int -> int
(** [category cp] is the category of code point [cp]. A value that is not a
    Unicode scalar value — a surrogate, or a value above [0x10FFFF] — is
    {!other}. *)

(** {1 Characters in a string} *)

val at : string -> int -> stop:int -> int
(** [at s i ~stop] is the class of the character starting at byte [i] of [s]
    packed with the byte length of that character. Take it apart with {!at_len},
    {!at_category}, {!at_is_punctuation} and {!at_is_word}.

    No byte at or after [stop] is read and the length is always at least one, so
    a walk over [at] terminates on any string. A UTF-8 sequence cut short by
    [stop], a byte that cannot lead a sequence, and a sequence whose
    continuation bytes are not continuation bytes are each one byte of category
    {!other}. *)

val at_len : int -> int
(** [at_len d] is the byte length of the character in the result [d] of {!at}.
*)

val at_category : int -> int
(** [at_category d] is the category of the character in the result [d] of {!at}.
*)

val at_is_punctuation : int -> bool
(** [at_is_punctuation d] is [true] iff the character in the result [d] of {!at}
    is punctuation: a printable ASCII character that is neither a letter nor a
    digit, or one of the general categories [Pc], [Pd], [Pe], [Pf], [Pi], [Po]
    and [Ps]. *)

val at_is_word : int -> bool
(** [at_is_word d] is [true] iff the character in the result [d] of {!at} is a
    word character, that is of the regular expression class [\w]: alphabetic
    characters, the marks, the decimal digits, connector punctuation and the
    joiners. *)
