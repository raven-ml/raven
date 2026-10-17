(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Accessors and codecs for untyped [Jsont.json] values.

    The accessors are total. A value of the wrong shape reads as {!null},
    [None] or [[]], so a decoder skips what it does not recognise. *)

val null : Jsont.json
(** [null] is the JSON [null], with no source location. *)

val json_obj : (string * Jsont.json) list -> Jsont.json
(** [json_obj pairs] is the object with the members [pairs], in order. *)

val json_mem : string -> Jsont.json -> Jsont.json
(** [json_mem name json] is the member [name] of the object [json]. It is
    {!null} if [json] is not an object or has no such member. *)

val json_string : Jsont.json -> string option
val json_number : Jsont.json -> float option
val json_bool : Jsont.json -> bool option

val json_string_list : Jsont.json -> string list
(** [json_string_list json] is the items of the array [json] that are
    strings. It is [[]] if [json] is not an array. *)

val json_assoc : Jsont.json -> (string * Jsont.json) list
(** [json_assoc json] is the members of the object [json], in order. It is
    [[]] if [json] is not an object. *)

val json_of_string : string -> Jsont.json
(** [json_of_string text] decodes [text] as one JSON value.

    Raises [Failure] with the decoder's message if [text] is not one JSON
    value. *)

val json_to_string : ?pretty:bool -> Jsont.json -> string
(** [json_to_string json] encodes [json] on one line, or indented if
    [pretty] is [true] (defaults to [false]). The text ends with no newline.

    Raises [Failure] with the encoder's message if it cannot encode
    [json]. *)
