(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** WKT 1 text as trees, for {!Crs.of_wkt}.

    The grammar is OGC 18-010r7's: Annex C's objects over the characters of
    §6.3. An object is a keyword and its attributes between delimiters, brackets
    or parentheses, one form for the whole text (§6.4). An attribute is an
    object, quoted text (a doubled quote stands for one), a number of §6.3.2, or
    a bare enumeration word such as [NORTH]. Padding with white space between
    tokens is ignored. *)

type t =
  | Object of string * t list  (** A keyword and its attributes. *)
  | Text of string  (** Quoted text, unquoted. *)
  | Number of float
  | Word of string  (** An enumeration. *)

val parse : string -> (t, string) result
(** [parse s] is the object [s] holds, the whole of [s]. [Error] names the byte
    offset of the first character that breaks the grammar. *)

val equal : t -> t -> bool
(** [equal a b] is [true] iff [a] and [b] have the same shape, keywords, texts
    and words equal up to ASCII case, and numbers equal as floats. *)
