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

val tensor : ('a, 'layout) Rune.t -> t
(** {2 Constructors} *)

val scalar_bool : bool -> t
val scalar_int : int -> t
val scalar_float : float -> t
val scalar_string : string -> t
val scalar_json : Yojson.Basic.t -> t
val list_of : t list -> t
val record_of : (string * t) list -> t

val is_tensor : t -> bool
(** {2 Accessors} *)

val is_scalar : t -> bool
val get_tensor : t -> tensor option
val get_scalar : t -> scalar option
val to_list : t -> t list option
val to_record : t -> t Record.t option

val iter :
  ?on_tensor:(tensor -> unit) -> ?on_scalar:(scalar -> unit) -> t -> unit
(** {2 Traversal Helpers} *)

val map_tensors : (tensor -> tensor) -> t -> t
val map_scalars : (scalar -> scalar) -> t -> t
val fold_tensors : ('a -> tensor -> 'a) -> 'a -> t -> 'a
val fold_scalars : ('a -> scalar -> 'a) -> 'a -> t -> 'a

(** {2 Flattening Utilities} *)
val flatten_tensors : ?prefix:string -> t -> (string * tensor) list
(** Return dotted paths and tensor leaves. Lists use the notation [name[0]]. *)

val flatten_scalars : ?prefix:string -> t -> (string * scalar) list

val scalar_to_yojson : scalar -> Yojson.Basic.t
(** {2 Conversion Helpers} *)

val scalar_of_yojson : Yojson.Basic.t -> scalar

(** {2 Interoperability} *)
val of_ptree : Ptree.t -> t
(** Convert a parameter tree into a snapshot tree. *)

val to_ptree : t -> (Ptree.t, string) result
(** Convert a snapshot tree back into a parameter tree. Returns an error if the
    tree contains scalar nodes. *)
