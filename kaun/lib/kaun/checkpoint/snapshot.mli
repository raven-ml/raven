(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Generic tree structure capable of storing heterogeneous tensors and scalar
    metadata. *)

module Record : module type of Map.Make (String)

(** Existential wrapper around any Rune tensor. *)
type tensor = Pack : ('a, 'layout) Rune.t -> tensor

(** Scalar metadata payloads that can accompany tensors in snapshots. *)
type scalar =
  | Bool of bool
  | Int of int
  | Float of float
  | String of string
  | Json of Yojson.Basic.t

(** Heterogeneous snapshot tree. *)
type t =
  | Tensor of tensor
  | Scalar of scalar
  | List of t list
  | Record of t Record.t

(** {2 Constructors} *)

val tensor : ('a, 'layout) Rune.t -> t
val bool : bool -> t
val int : int -> t
val float : float -> t
val string : string -> t
val json : Yojson.Basic.t -> t
val list : t list -> t
val scalar : scalar -> t
val record : (string * t) list -> t
val rng : Rune.Rng.key -> t
val ptree : Ptree.t -> t

(** {2 Accessors} *)

val is_tensor : t -> bool
val is_scalar : t -> bool
val is_list : t -> bool
val is_record : t -> bool
val get_tensor : t -> tensor option
val get_scalar : t -> scalar option
val get_list : t -> t list option
val get_record : t -> t Record.t option

(** {2 Traversal Helpers} *)

val iter :
  ?on_tensor:(tensor -> unit) -> ?on_scalar:(scalar -> unit) -> t -> unit

val map_tensors : (tensor -> tensor) -> t -> t
val map_scalars : (scalar -> scalar) -> t -> t
val fold_tensors : ('a -> tensor -> 'a) -> 'a -> t -> 'a
val fold_scalars : ('a -> scalar -> 'a) -> 'a -> t -> 'a

(** {2 Flattening Utilities} *)

val flatten_tensors : ?prefix:string -> t -> (string * tensor) list
(** Return dotted paths and tensor leaves. Lists use the notation [name[0]]. *)

val flatten_scalars : ?prefix:string -> t -> (string * scalar) list

(** {2 Conversion Helpers} *)

val scalar_to_yojson : scalar -> Yojson.Basic.t
val scalar_of_yojson : Yojson.Basic.t -> scalar

(** {2 Interoperability} *)

val to_ptree : t -> (Ptree.t, string) result
(** Convert a snapshot tree back into a parameter tree. Returns an error if the
    tree contains scalar nodes. *)
