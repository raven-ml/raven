(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Heterogeneous parameter trees.

    A parameter tree is a finite tree with tensor leaves and container nodes.
    Leaves are packed tensors ({!tensor}), and containers are either ordered
    lists ([List]) or string-keyed dicts ([Dict]). *)

type tensor =
  | P : ('a, 'layout) Rune.t -> tensor
      (** A packed tensor. The wrapper hides dtype and layout parameters. *)

type t =
  | Tensor of tensor  (** A tensor leaf. *)
  | List of t list  (** An ordered list branch. *)
  | Dict of (string * t) list  (** A dict branch. Keys are strings. *)

(** {1:constructors Constructors} *)

val tensor : ('a, 'layout) Rune.t -> t
(** [tensor x] is [Tensor (P x)]. *)

val list : t list -> t
(** [list xs] is [List xs]. *)

val dict : (string * t) list -> t
(** [dict kvs] is [Dict kvs] with key validation.

    Raises [Invalid_argument] if a key is empty, duplicated, or contains ['.'],
    ['\['], or ['\]']. *)

val empty : t
(** [empty] is [List []]. Canonical value for "no parameters" or "no state". *)

(** {1:tensor Tensor Inspection} *)

module Tensor : sig
  val dtype : tensor -> Nx_core.Dtype.packed
  (** [dtype t] is [t]'s dtype. *)

  val shape : tensor -> int array
  (** [shape t] is [t]'s shape. *)

  val numel : tensor -> int
  (** [numel t] is the number of elements in [t]. *)

  val to_typed : ('a, 'l) Rune.dtype -> tensor -> ('a, 'l) Rune.t option
  (** [to_typed dtype t] is [Some x] iff [t] has dtype [dtype], with [x] the
      typed tensor. It is [None] on dtype mismatch. *)

  val to_typed_exn : ('a, 'l) Rune.dtype -> tensor -> ('a, 'l) Rune.t
  (** [to_typed_exn dtype t] is the typed tensor in [t].

      Raises [Invalid_argument] if [t]'s dtype is not [dtype]. *)
end

(** {1:dict Dict Access} *)

module Dict : sig
  type fields = (string * t) list
  (** The type for dict fields. *)

  val fields_exn : ?ctx:string -> t -> fields
  (** [fields_exn ?ctx t] is [t]'s fields.

      Raises [Invalid_argument] if [t] is not [Dict _]. The optional [ctx] is
      prefixed to the error message. *)

  val find : string -> fields -> t option
  (** [find name fields] is [Some v] if [name] is bound in [fields], and [None]
      otherwise. *)

  val find_exn : ?ctx:string -> string -> fields -> t
  (** [find_exn ?ctx name fields] is [name]'s value in [fields].

      Raises [Invalid_argument] if [name] is missing. The optional [ctx] is
      prefixed to the error message. *)

  val get_tensor_exn :
    fields -> name:string -> ('a, 'l) Nx_core.Dtype.t -> ('a, 'l) Rune.t
  (** [get_tensor_exn fields ~name dtype] is the typed tensor in [fields] under
      [name].

      Raises [Invalid_argument] if [name] is missing, [name] is not a tensor, or
      the tensor dtype differs from [dtype]. *)
end

(** {1:list List Access} *)

module List : sig
  val items_exn : ?ctx:string -> t -> t list
  (** [items_exn ?ctx t] is [t]'s items.

      Raises [Invalid_argument] if [t] is not [List _]. The optional [ctx] is
      prefixed to the error message. *)
end

(** {1:leaf Leaf Access} *)

type 'r tensor_handler = { run : 'a 'layout. ('a, 'layout) Rune.t -> 'r }
(** Rank-2 handler for unpacking {!tensor}. *)

val with_tensor : tensor -> 'a tensor_handler -> 'a
(** [with_tensor t h] applies [h.run] to the unpacked tensor in [t]. *)

val as_tensor_exn : ?ctx:string -> t -> tensor
(** [as_tensor_exn ?ctx t] is [t]'s packed tensor.

    Raises [Invalid_argument] if [t] is not [Tensor _]. The optional [ctx] is
    prefixed to the error message. *)

(** {1:functional Functional Operations} *)

type map_handler = {
  run : 'a 'layout. ('a, 'layout) Rune.t -> ('a, 'layout) Rune.t;
}
(** Rank-2 tensor mapper used by {!map}. *)

val map : map_handler -> t -> t
(** [map f t] maps [f.run] over tensor leaves and preserves tree structure. *)

type map2_handler = {
  run :
    'a 'layout.
    ('a, 'layout) Rune.t -> ('a, 'layout) Rune.t -> ('a, 'layout) Rune.t;
}
(** Rank-2 tensor zipper used by {!map2}. *)

val map2 : map2_handler -> t -> t -> t
(** [map2 f a b] zips [a] and [b] and applies [f.run] to paired tensor leaves.

    Lists are matched by position. Dict nodes are matched by key using [a]'s key
    order.

    Raises [Invalid_argument] on structure mismatch, list or dict size mismatch,
    missing keys in [b], or paired dtype mismatch. *)

val iter : (tensor -> unit) -> t -> unit
(** [iter f t] applies [f] to each leaf tensor in depth-first order.

    Leaves are visited left-to-right in list order and dict field order. *)

val fold : ('acc -> tensor -> 'acc) -> 'acc -> t -> 'acc
(** [fold f acc t] folds leaf tensors in the same traversal order as {!iter}. *)

(** {1:flatten Flatten and Rebuild} *)

val flatten : t -> tensor list * (tensor list -> t)
(** [flatten t] is [(leaves, rebuild)] where:
    - [leaves] are [t]'s leaf tensors in depth-first order;
    - [rebuild new_leaves] rebuilds [t]'s structure with [new_leaves].

    [rebuild] raises [Invalid_argument] if [new_leaves] has a different length
    than [leaves]. *)

val flatten_with_paths : t -> (string * tensor) list
(** [flatten_with_paths t] returns [(path, tensor)] pairs where paths are
    dot-separated strings. Dict keys become path segments; list indices become
    decimal segments (e.g. ["layers.0.weight"]).

    If [t] is a tensor leaf, its path is the empty string.

    The path encoding is injective for trees built with {!dict}, because {!dict}
    rejects keys containing ['.'], ['\['], or ['\]']. *)

(** {1:utils Utilities} *)

val zeros_like : t -> t
(** [zeros_like t] has the same structure as [t], with each tensor replaced by
    [Rune.zeros_like]. *)

val count_parameters : t -> int
(** [count_parameters t] is the sum of {!Tensor.numel} over all leaf tensors. *)

val pp : Format.formatter -> t -> unit
(** [pp] formats trees for debugging. *)
