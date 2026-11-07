(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Values of params, summaries and metadata.

    A value is a boolean, an integer, a float or a string. A value written
    to a store reads back equal, with its constructor: [`Float 1.] reads
    back as [`Float 1.], [`Int 1] as [`Int 1], and a NaN as a NaN
    ({!Store_format.section-values}). A string must be UTF-8. *)

type t = [ `Bool of bool | `Int of int | `Float of float | `String of string ]
(** The type for values. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf v] prints [v] for people and for tab-separated output: a
    boolean as [true] or [false], an integer in decimal, a float in the
    fewest significant digits that read back to it, with a fraction or an
    exponent when it is integral, or as [nan], [inf] or [-inf], and a
    string with each backslash and control character escaped as a JSON
    string escapes it, so that a tab prints as a backslash and [t]. The
    text is the same on every system and holds no tab or line feed. *)

val to_float : t -> float option
(** [to_float v] is [f] for [`Float f], the float nearest [n] for
    [`Int n], and [None] otherwise. *)

val to_int : t -> int option
(** [to_int v] is [n] for [`Int n], and [None] otherwise. *)

val to_string : t -> string option
(** [to_string v] is [s] for [`String s], and [None] otherwise. *)

val to_bool : t -> bool option
(** [to_bool v] is [b] for [`Bool b], and [None] otherwise. *)
