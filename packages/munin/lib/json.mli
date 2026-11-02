(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** JSON as the store's files write it.

    The one codec of every file {!Store_format} specifies. A number keeps its
    text, so that a reader tells [1] from [1.0]
    ({!Store_format.section-values}). Decoding is strict RFC 8259: one value,
    UTF-8, no member name twice in an object. *)

type t =
  | Null
  | Bool of bool
  | Number of string  (** The number's text, as written. *)
  | String of string
  | Array of t list
  | Object of (string * t) list  (** Members in the order written. *)

val of_string : string -> (t, string) result
(** [of_string s] is the JSON value [s] holds, surrounded by optional white
    space, or [Error why]. *)

val utf8 : string -> bool
(** [utf8 s] is [true] iff [s] is valid UTF-8. *)

(** {1:writing Writing} *)

val add_string : Buffer.t -> string -> unit
(** [add_string b s] adds [s] as a JSON string. [s] must be UTF-8. *)

val add_float : Buffer.t -> float -> unit
(** [add_float b x] adds [x] as {!Store_format.section-values} says. *)

val add_int : Buffer.t -> int -> unit
(** [add_int b n] adds [n], tagged when its magnitude exceeds 2{^ 53}. *)

val add_value : Buffer.t -> Value.t -> unit
(** [add_value b v] adds [v]. *)

val add_object : Buffer.t -> (string * (Buffer.t -> unit)) list -> unit
(** [add_object b members] adds an object of [members], each written by its
    function. *)

(** {1:reading Reading members} *)

val float : t -> float option
(** [float j] is the float [j] writes: any number, or a tagged non-finite float.
*)

val int : t -> int option
(** [int j] is the integer [j] writes: a number without fraction or exponent
    that fits an [int], or a tagged integer. *)

val value : t -> Value.t option
(** [value j] is the value [j] writes. *)

val values : t -> (string * Value.t) list option
(** [values j] is the object of values [j] writes. *)

val mem : string -> t -> t option
(** [mem name j] is member [name] of the object [j]. *)

val string : t -> string option

val field : string -> (t -> 'a option) -> t -> 'a option
(** [field name f j] is [f] of member [name] of [j], if it has one. *)

val opt : string -> (t -> 'a option) -> t -> 'a option option
(** [opt name f j] is [Some None] if [j] lacks member [name], and
    [Some (Some v)] if [f] reads [v] from it. *)

val list : (t -> 'a option) -> t -> 'a list option
val assoc : (t -> 'a option) -> t -> (string * 'a) list option
