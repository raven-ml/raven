(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Scalar values for parameters, summaries, and metadata. *)

type t = [ `Bool of bool | `Int of int | `Float of float | `String of string ]
(** The type for scalar values. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf v] pretty-prints [v]. *)

val to_float : t -> float option
(** [to_float v] extracts a float. [`Int] values are promoted. *)

val to_int : t -> int option
(** [to_int v] extracts an int. [`Float] values are truncated if integral. *)

val to_string : t -> string option
(** [to_string v] extracts a string. *)

val to_bool : t -> bool option
(** [to_bool v] extracts a bool. *)

(**/**)

val to_json : t -> Jsont.json
(** [to_json v] is [v] as JSON: a boolean, a number, a string, or for a float
    JSON has no number for, [{"float": "NaN"}], ["Infinity"] or
    ["-Infinity"]. *)

val of_json : Jsont.json -> t option
(** [of_json json] is the value {!to_json} writes as [json], or [None] for
    any other JSON. A number with an integer value is an [`Int]. *)

val assoc_of_json : Jsont.json -> (string * t) list option
(** [assoc_of_json json] is the members of the object [json] read by
    {!of_json}, or [None] if [json] is not an object or a member is not a
    value. *)

(**/**)
