(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Value fields: the implementation of {!Ymir_fits.Fits.Value}, with the
    grammars of FITS 4.0 §4.2 for reading and the fixed and free formats of
    §4.2.2 to §4.2.4 for printing.

    A value reads in two steps. {!split} cuts a record's value field into its
    token and comment, and {!read} parses a defined token in a {!base}
    grammar. A {!t} adds the conversions {!map} composes. *)

(** {1:fields Fields} *)

type field = { token : string; comment : string option }
(** The type for a value field split as written: the value's text, trimmed,
    and the comment after its slash, one leading space and trailing spaces
    dropped. A string's text runs from its opening quote to its closing one.
    [token] is [""] for an undefined value. *)

val split : string -> field
(** [split s] reads the value field [s], the bytes after the value
    indicator. A quote that does not close makes the rest of the field the
    token, which the string grammar then refuses. *)

val is_printable : char -> bool
(** [is_printable c] is [true] iff [c] is in ASCII 32 to 126. *)

val trim_right : string -> string
(** [trim_right s] is [s] without its trailing spaces. *)

(** {1:strings Strings} *)

val unquote : string -> string option
(** [unquote t] is the text of the string token [t], quotes removed, doubled
    quotes made single and trailing spaces kept, or [None] if [t] is not a
    quoted string. *)

val string_value : string -> string
(** [string_value s] is the string [s] holds (§4.2.1.1): trailing spaces
    dropped, and a string of spaces one space. *)

val quote : string -> string
(** [quote s] is [s] between quotes, its quotes doubled. *)

val is_value : string -> bool
(** [is_value t] is [true] iff [t] is one FITS value as written in a value
    field: a logical, a real, a complex or a quoted string. *)

(** {1:numbers Numbers} *)

val real_grammar : string -> bool
(** [real_grammar t] is [true] iff [t] is real text (§4.2.4), integer text
    included, with an exponent letter [E] or [D] in either case. *)

val float_of_token : string -> float option
(** [float_of_token t] is the float nearest the real text [t], correctly
    rounded, or [None] if [t] is not real text or is past float64's
    range. *)

val print_float : float -> string
(** [print_float x] is {!Ymir_fits.Fits.Value.float}'s printing of the
    finite [x]: positional when it fits twenty bytes, as [1288.4], and as
    [6.02214076E+23] otherwise. *)

(** {1:grammars Grammars} *)

(** The type for the grammars a token reads in. *)
type _ base =
  | Bool : bool base
  | Int : int base
  | Float : float base
  | String : string base
  | Text : string base

(** {!Ymir_fits.Fits.Value.t}: a grammar, the conversion of what it reads,
    and the conversion to what it prints. *)
type 'a t = V : 'b base * ('b -> ('a, string) result) * ('a -> 'b) -> 'a t

val bool : bool t
val int : int t
val float : float t
val string : string t
val text : string t
val map : ('a -> ('b, string) result) -> ('b -> 'a) -> 'a t -> 'b t

val base_equal : 'b base -> 'b -> 'b -> bool
(** [base_equal b x y] is [true] iff [x] and [y] are equal values of [b]. *)

val base_to_string : 'b base -> 'b -> string
(** [base_to_string b x] is [x] as a value field writes it. *)

val read : 'b base -> string -> string option -> ('b, string) result
(** [read b token joined] reads the defined [token] in [b]. [joined] is the
    string that a string token and its [CONTINUE] records hold, [None] when
    the token is not a string; [String] reads it in place of [token]. *)

val is_string_base : 'b base -> bool
(** [is_string_base b] is [true] iff [b] is [String]. *)
