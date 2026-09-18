(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Step metadata dictionaries.

    Info dictionaries carry auxiliary data returned by {!Env.reset} and
    {!Env.step}. Keys are strings and values are {!Value.t}. *)

(** {1:types Types} *)

type t
(** The type for info dictionaries. *)

(** {1:constructors Constructors} *)

val empty : t
(** [empty] is the empty dictionary. *)

val of_list : (string * Value.t) list -> t
(** [of_list kvs] is a dictionary from the given key-value pairs. *)

(** {1:predicates Predicates} *)

val is_empty : t -> bool
(** [is_empty t] is [true] iff [t] has no bindings. *)

(** {1:ops Operations} *)

val set : string -> Value.t -> t -> t
(** [set k v t] is [t] with [k] bound to [v]. *)

val find : string -> t -> Value.t option
(** [find k t] is the value bound to [k] in [t], if any. *)

val find_exn : string -> t -> Value.t
(** [find_exn k t] is the value bound to [k] in [t].

    Raises [Invalid_argument] if [k] is not present. *)

val remove : string -> t -> t
(** [remove k t] is [t] without the binding for [k]. *)

val merge : t -> t -> t
(** [merge a b] is the union of [a] and [b]. When both have a binding for the
    same key, the value from [b] wins. *)

(** {1:converting Converting} *)

val to_list : t -> (string * Value.t) list
(** [to_list t] is the bindings of [t] in key order. *)

val to_value : t -> Value.t
(** [to_value t] is [t] as a {!Value.Dict}. *)

(** {1:fmt Formatting} *)

val pp : Format.formatter -> t -> unit
(** [pp] formats an info dictionary for debugging. *)

(** {1:convenience Convenience value constructors} *)

val null : Value.t
(** [null] is {!Value.Null}. *)

val bool : bool -> Value.t
(** [bool b] is [Value.Bool b]. *)

val int : int -> Value.t
(** [int i] is [Value.Int i]. *)

val float : float -> Value.t
(** [float f] is [Value.Float f]. *)

val string : string -> Value.t
(** [string s] is [Value.String s]. *)

val int_array : int array -> Value.t
(** [int_array arr] is [Value.Int_array (Array.copy arr)]. *)

val float_array : float array -> Value.t
(** [float_array arr] is [Value.Float_array (Array.copy arr)]. *)

val bool_array : bool array -> Value.t
(** [bool_array arr] is [Value.Bool_array (Array.copy arr)]. *)
