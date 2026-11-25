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

val decode : string -> int -> stop:int -> int
(** [decode s i ~stop] is the character starting at byte [i] of [s], packed as
    [(u lsl 3) lor n] for its scalar value [u] and byte length [n], or [-1] if
    the byte at [i] opens no character. A character is a sequence RFC 3629
    allows that ends before [stop]: an overlong form, a surrogate, a value above
    U+10FFFF, a lead byte without the continuation bytes it asks for and a
    continuation byte with no lead open none. Every reader of brot's text takes
    a byte that opens no character as a character of one byte. *)

val at : string -> int -> stop:int -> int
(** [at s i ~stop] is the class of the character starting at byte [i] of [s]
    packed with the byte length of that character. Take it apart with {!at_len},
    {!at_category}, {!at_is_punctuation} and {!at_is_word}.

    No byte at or after [stop] is read and the length is always at least one, so
    a walk over [at] terminates on any string. A byte that opens no character,
    as {!decode} reads it, is one byte of category {!other} that is neither
    punctuation, a word character, a mark nor a letter of either case. *)

val length : string -> int -> stop:int -> int
(** [length s i ~stop] is the byte length of the character starting at byte [i]
    of [s], as {!at} reads it, without its class: the length {!decode} gives, or
    [1] for a byte that opens no character. *)

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

val at_is_mark : int -> bool
(** [at_is_mark d] is [true] iff the character in the result [d] of {!at} is a
    combining mark, that is of the regular expression class [\p{M}]: the general
    categories [Mc], [Me] and [Mn]. *)

val at_is_upper : int -> bool
(** [at_is_upper d] is [true] iff the character in the result [d] of {!at} is
    an uppercase or titlecase letter: the general categories [Lu] and [Lt]. *)

val at_is_lower : int -> bool
(** [at_is_lower d] is [true] iff the character in the result [d] of {!at} is
    a lowercase letter: the general category [Ll]. *)

(** {1:internals Internals} *)

val unicode_table : unit -> Bytes.t
(** [unicode_table ()] is the Unicode class table as it stands: byte [cp - 128]
    holds the packed properties of code point [cp], [0] when it has not been
    classified yet. Empty until the first non-ASCII code point is classified.
    The C kernels read it and hand back any code point whose byte is [0];
    {!category} fills it, from OCaml alone. *)
