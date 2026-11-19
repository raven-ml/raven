(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Reading the oracle's JSON. Each accessor raises [Failure] naming what it
    expected when the value is not of its kind. *)

val read : string -> Jsont.json
(** [read path] is the JSON value of the file [path]. *)

val of_string : string -> Jsont.json
(** [of_string s] is the JSON value of [s]. *)

val mem : string -> Jsont.json -> Jsont.json
(** [mem name v] is the member [name] of the object [v]. *)

val mems : Jsont.json -> (string * Jsont.json) list
(** [mems v] is the members of the object [v], in order. *)

val is_null : Jsont.json -> bool
(** [is_null v] is [true] iff [v] is [null]. *)

val list : Jsont.json -> Jsont.json list
(** [list v] is the elements of the array [v]. *)

val int : Jsont.json -> int
(** [int v] is the integer [v]. *)

val float : Jsont.json -> float
(** [float v] is the number [v]. *)

val bool : Jsont.json -> bool
(** [bool v] is the boolean [v]. *)

val string : Jsont.json -> string
(** [string v] is the string [v]. *)

val ints : Jsont.json -> int array
(** [ints v] is the array of integers [v]. *)
