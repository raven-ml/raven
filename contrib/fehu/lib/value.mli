(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Universal value type.

    Values represent heterogeneous data flowing through spaces and info
    dictionaries. Each variant wraps one kind of scalar, array, or composite
    datum. *)

(** {1:types Types} *)

(** The type for universal values. *)
type t =
  | Null  (** No value. *)
  | Bool of bool  (** A boolean. *)
  | Int of int  (** An integer. *)
  | Float of float  (** A float. *)
  | String of string  (** A string. *)
  | Int_array of int array  (** An integer array. *)
  | Float_array of float array  (** A float array. *)
  | Bool_array of bool array  (** A boolean array. *)
  | List of t list  (** A heterogeneous list. *)
  | Dict of (string * t) list  (** A string-keyed association list. *)

(** {1:predicates Predicates} *)

val equal : t -> t -> bool
(** [equal a b] is [true] iff [a] and [b] are structurally equal. *)

(** {1:fmt Formatting} *)

val pp : Format.formatter -> t -> unit
(** [pp] formats a value for debugging. *)

val to_string : t -> string
(** [to_string v] is [v] formatted as a string via {!pp}. *)
