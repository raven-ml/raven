(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Headers: the implementation of {!Ymir_fits.Fits.Header}.

    A header holds its records as the file held them, an index from each
    keyword to the records that give it a value, and the place that starts
    its errors. An edit returns a header with the same place. Records
    are numbered from 0 here; error messages count cards from 1. *)

type t
(** {!Ymir_fits.Fits.Header.t}. *)

val record_size : int
(** [record_size] is [80], the bytes of a record. *)

val block_size : int
(** [block_size] is [2880], the bytes of a FITS block. *)

(** {1:records Records} *)

val keyword : string -> (string * int) option
(** [keyword r] is the keyword the 80-byte record [r] gives a value to, and
    the offset of its value field, or [None] for commentary. A hierarchical
    keyword's tokens are joined by single spaces. *)

val is_end : string -> bool
(** [is_end r] is [true] iff the record [r] is [END]. *)

val span_of : string array -> int -> int
(** [span_of rs i] is the number of records from [rs.(i)] that hold its
    value: one, plus for a string the [CONTINUE] records that continue
    it. *)

val of_records : ?place:Err.place -> string array -> t
(** [of_records ~place rs] is the header of the 80-byte records [rs], [END]
    excluded, whose errors start at [place] ({!Err.nowhere} by default). The
    header keeps [rs], which must not change afterwards. *)

val place : t -> Err.place
(** [place h] is where [h]'s errors start. *)

val with_place : Err.place -> t -> t
(** [with_place p h] is [h] with its errors starting at [p]. *)

val empty : t
val records : t -> string list
val equal : t -> t -> bool
val to_string : t -> string
val pp : Format.formatter -> t -> unit
val of_string : ?name:string -> string -> (t, string) result

(** {1:values Values} *)

val cards : t -> string -> int list
(** [cards h k] is the indices of the records giving [k] a value, in
    order. *)

val card_place : t -> int -> string -> Err.place
(** [card_place h i k] is the place of [h]'s record [i], keyword [k]. *)

val find : 'a Value.t -> string -> t -> ('a option, string) result
val get : 'a Value.t -> string -> t -> ('a, string) result

val find_struct : 'a Value.t -> string -> t -> ('a option, string) result
(** [find_struct] is {!find} without the check of the keyword's name, for
    the structural keywords. *)

val set : ?comment:string -> 'a Value.t -> string -> 'a -> t -> t
val remove : string -> t -> t
val commentary : string -> t -> string list
val add_commentary : string -> string -> t -> t
