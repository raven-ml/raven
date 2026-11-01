(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Scalar values for parameters, summaries, and metadata. *)

type t = [ `Bool of bool | `Int of int | `Float of float | `String of string ]
(** The type for scalar values. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf v] pretty-prints [v], a float to six significant digits as C's
    [%g] writes it, the same text on every system: [1e+06], [0.333333]. *)

val to_float : t -> float option
(** [to_float v] extracts a float. [`Int] values are promoted. *)

val to_int : t -> int option
(** [to_int v] extracts an int. [`Float] values are truncated if integral. *)

val to_string : t -> string option
(** [to_string v] extracts a string. *)

val to_bool : t -> bool option
(** [to_bool v] extracts a bool. *)

val float_name : float -> string option
(** [float_name x] is the string that stands for [x] in munin's JSON where
    JSON has no number for it: ["NaN"], ["Infinity"] or ["-Infinity"], as
    Protocol Buffers' JSON mapping writes them. It is [None] for a finite
    [x]. *)

(**/**)

val float_to_json : float -> Jsont.json
(** [float_to_json x] is [x] as a JSON number where it is finite, and the
    string {!float_name}[ x] otherwise. *)

val float_of_json : Jsont.json -> float option
(** [float_of_json json] is the float {!float_to_json} writes as [json]: a
    number, or exactly one of {!float_name}'s strings. *)

val to_json : t -> Jsont.json
(** [to_json v] is [v] as JSON: a boolean, a number, a string, or for a float
    JSON has no number for, [{"float": "NaN"}], ["Infinity"] or
    ["-Infinity"]. *)

val of_json : Jsont.json -> t option
(** [of_json json] is the value {!to_json} writes as [json], or [None] for
    any other JSON. A number with an integer value is an [`Int]. *)

val assoc_of_json : Jsont.json -> (string * t) list
(** [assoc_of_json json] is the members of the object [json] that {!of_json}
    reads, in order. A decoder skips what it does not recognise: a member
    that is not a value is left out, as the [null] munin wrote for a NaN
    before, and a [json] that is not an object is [[]]. *)

(**/**)
