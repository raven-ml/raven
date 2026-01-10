(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Heterogeneous parameter tree structure. *)

(** Path submodule for advanced use *)
module Path : sig
  type t
  (** Path segments. *)

  val root : t
  (** Root path. *)

  val of_string : string -> t
  (** Parse string to path. *)

  val to_string : t -> string
  (** Convert path to string. *)

  val key : string -> t -> t
  (** Add key segment. *)

  val index : int -> t -> t
  (** Add index segment. *)
end

type tensor =
  | P : ('a, 'layout) Rune.t -> tensor
      (** Existential wrapper for tensor tensors. *)

type t =
  | Tensor of tensor
  | List of t list
  | Dict of (string * t) list  (** Parameter tree: tensors or containers. *)

(** Tensor utilities *)
module Tensor : sig
  val dtype : tensor -> Nx_core.Dtype.packed
  (** Get the dtype of the tensor. *)

  val shape : tensor -> int array
  (** Get the shape of the tensor. *)

  val numel : tensor -> int
  (** Get the number of elements in the tensor. *)

  val to_typed : ('a, 'l) Rune.dtype -> tensor -> ('a, 'l) Rune.t option
  (** [to_typed dtype t] returns [Some x] when the packed tensor [t] has the
      given [dtype], with [x] the typed tensor. Returns [None] if the dtype does
      not match. *)

  val to_typed_exn : ('a, 'l) Rune.dtype -> tensor -> ('a, 'l) Rune.t
  (** [to_typed_exn dtype t] returns the typed tensor when the dtype matches, or
      raises [Invalid_argument] on mismatch. *)
end

module List : sig
  val items_exn : ?ctx:string -> t -> t list
  (** Extract list items or fail with a helpful [ctx]-prefixed message. *)
end

module Dict : sig
  type fields = (string * t) list

  val fields_exn : ?ctx:string -> t -> fields
  (** Extract dict fields or fail with a helpful [ctx]-prefixed message. *)

  val find : string -> fields -> t option
  val find_exn : ?ctx:string -> string -> fields -> t
  val set : string -> t -> fields -> fields
  val update : (t -> t) -> string -> fields -> fields
  val mem : string -> fields -> bool

  val get_tensor :
    fields -> name:string -> ('a, 'l) Nx_core.Dtype.t -> ('a, 'l) Rune.t option

  val get_tensor_exn :
    fields -> name:string -> ('a, 'l) Nx_core.Dtype.t -> ('a, 'l) Rune.t
end

(** Builders *)

val tensor : ('a, 'layout) Rune.t -> t
(** Create a tensor node from a tensor. *)

val list : t list -> t
(** Create a list container. *)

val dict : (string * t) list -> t
(** Create a dict container from key-value pairs. Keys must be unique. *)

(* *)

type 'r tensor_handler = { run : 'a 'layout. ('a, 'layout) Rune.t -> 'r }

val with_tensor : tensor -> 'a tensor_handler -> 'a

(* *)

val as_tensor : t -> tensor option
(** Extract tensor if the tree is a single leaf, else None. *)

val as_tensor_exn : ?ctx:string -> t -> tensor
(** Extract tensor or raise. Optional context for error message. *)

(** Walking / zipping *)

val map : (('a, 'l) Rune.t -> ('a, 'l) Rune.t) -> t -> t
(** Typed map over tensors. Result dtype must equal input dtype. *)

val map2 :
  (('a, 'l) Rune.t -> ('a, 'l) Rune.t -> ('a, 'l) Rune.t) -> t -> t -> t
(** Typed zip-with over tensors. Structures must match; dtype per-pair must
    match. *)

val map_packed : (tensor -> tensor) -> t -> t
(** Packed map over tensors (escape hatch if types are dynamic). *)

val iter : (tensor -> unit) -> t -> unit
(** Iterate over tensors. *)

val fold : ('acc -> tensor -> 'acc) -> 'acc -> t -> 'acc
(** Fold over tensors. *)

(** Flatten & rebuild *)

val flatten : t -> tensor list * (tensor list -> t)
(** Flatten to tensors and a rebuilder function. *)

(** Path access *)

val get : path:Path.t -> t -> t option
(** Get subtree at path. *)

val get_exn : path:Path.t -> t -> t
(** Get subtree or raise. *)

val set : path:Path.t -> value:t -> t -> t
(** Set subtree at path. *)

val update : path:Path.t -> (t -> t) -> t -> t
(** Update subtree at path with function. *)

val mem : path:Path.t -> t -> bool
(** Check if path exists. *)

(** Typed path access for tensors *)

val get_tensor :
  path:Path.t -> t -> ('a, 'l) Rune.dtype -> ('a, 'l) Rune.t option
(** Get typed tensor at path, checking dtype. *)

val get_tensor_exn : path:Path.t -> t -> ('a, 'l) Rune.dtype -> ('a, 'l) Rune.t
(** Get typed tensor or raise. *)

(** Flatten with paths *)

val flatten_with_paths : t -> (Path.t * tensor) list
(** Flatten to (path, tensor) pairs. *)

val filter_tensors : t -> (Path.t -> tensor -> bool) -> (Path.t * tensor) list
(** Filter tensors by predicate on path and tensor. *)

(** Float dtype discovery *)

type float_dtype =
  | F : (float, 'l) Rune.dtype -> float_dtype
      (** Witness that the dtype is a floating-point dtype. Encodes
          [('a = float)] at the type level to avoid enumerating float
          constructors at call sites. *)

val first_float_dtype : t -> float_dtype option
(** Find the first floating-point tensor in the tree and return a float dtype
    witness, if any. Floating-point dtypes include float32/float64/float16,
    bfloat16, and float8 variants. *)

val first_float_dtype_exn : t -> float_dtype
(** Like {!first_float_dtype} but raises if no floating-point tensors are
    present. *)

(** Convenience *)

val zeros_like : t -> t
(** Create tree with zeros_like tensors. *)

val copy : t -> t
(** Deep copy the tree. *)

val count_tensors : t -> int
(** Count number of tensors. *)

val count_parameters : t -> int
(** Total elements across all tensors. *)

val pp : Format.formatter -> t -> unit
(** Printing *)

val to_string : t -> string
