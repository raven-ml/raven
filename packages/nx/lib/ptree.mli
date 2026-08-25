(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Parameter trees: structures with tensor leaves.

    {!module-type:S} is the traversal interface the Raven ecosystem shares:
    autodiff transformations, structural optimizers and checkpointing all
    operate on any structure implementing it — typically a record of tensors
    with hand-written one-line traversals. The concrete {!type:t} is the stock
    instance for structures only known at runtime, and {!module-Make} derives an
    instance from any payload-generic structure ({!module-type:Uniform}). *)

(** Structures whose tensor leaves can be traversed.

    Implementations must satisfy:
    - [map f t] applies [f] to every tensor leaf of [t] exactly once and
      preserves the structure of [t].
    - [map2 f a b] applies [f] to corresponding leaves of [a] and [b], which
      must be structurally equal.
    - [iter f t] applies [f] to every tensor leaf of [t] exactly once, in a
      stable order. *)
module type S = sig
  type t
  (** The structure type. *)

  val map : ('a 'b. ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t) -> t -> t
  (** [map f t] is [t] with [f] applied to every tensor leaf. *)

  val map2 :
    ('a 'b. ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t) ->
    t ->
    t ->
    t
  (** [map2 f a b] is [a] and [b] combined leafwise with [f].

      Raises [Invalid_argument] if [a] and [b] are not structurally equal. *)

  val iter : ('a 'b. ('a, 'b) Nx_effect.t -> unit) -> t -> unit
  (** [iter f t] applies [f] to every tensor leaf of [t]. *)
end

type tensor =
  | P : ('a, 'b) Nx_effect.t -> tensor
      (** A packed tensor. The wrapper hides dtype and layout parameters. *)

(** Structures with a uniform payload: the traversal core.

    An ['a t] satisfying {!module-type:Traverse} is a structure whose payload
    positions all have type ['a]. Instantiating the payload turns one
    declaration into a family of trees — the model itself and any
    parameter-shaped data (masks, symmetries, per-leaf learning rates) — with
    type-changing traversals like
    [map2 : ('a -> 'b -> 'c) -> 'a t -> 'b t -> 'c t] connecting them.

    {!module-type:S} is the same structure specialised to packed tensor leaves;
    {!module-Make} and {!val-typed} convert any {!module-type:Traverse} into an
    {!S}. {!module-type:Uniform} extends the core with leaf paths.

    Implementations must satisfy:
    - [map f t] applies [f] to every payload of [t] exactly once and preserves
      the structure of [t].
    - [map2 f a b] applies [f] to corresponding payloads of [a] and [b], which
      must be structurally equal.
    - [iter f t] applies [f] to every payload of [t] exactly once.
    - Every traversal visits the payloads of a value in the same order: [map]
      and [iter] agree on the payload sequence, and [map2] visits corresponding
      pairs in that order. *)
module type Traverse = sig
  type 'a t
  (** The structure type, parameterised by the uniform payload type. *)

  val map : ('a -> 'b) -> 'a t -> 'b t
  (** [map f t] is [t] with [f] applied to every payload leaf. *)

  val map2 : ('a -> 'b -> 'c) -> 'a t -> 'b t -> 'c t
  (** [map2 f a b] is [a] and [b] combined leafwise with [f].

      Raises [Invalid_argument] if [a] and [b] are not structurally equal. *)

  val iter : ('a -> unit) -> 'a t -> unit
  (** [iter f t] applies [f] to every payload leaf of [t]. *)
end

(** Structures with a uniform payload and leaf paths.

    {!module-type:Uniform} extends {!module-type:Traverse} with path-threading
    reductions; it is the contract checkpointing consumes and the one
    [\[@@deriving ptree\]] derives for payload-generic types. Additional laws:
    - [fold f acc t] visits every payload in the traversal order; the [string]
      argument is the path to the current position.
    - [fold2] is like [fold] but across two structurally equal trees.
    - [names t] is [t] with every payload replaced by its path, so
      [names (map f t)] is [names t] and [fold] pairs with any traversal
      positionally. *)
module type Uniform = sig
  include Traverse

  val fold : (string -> 'acc -> 'a -> 'acc) -> 'acc -> 'a t -> 'acc
  (** [fold f acc t] reduces [t] to an accumulator, threading the path of each
      payload leaf. A path is the dot-joined sequence of record field names and
      container indices from the root to the leaf, matching the checkpoint
      naming convention:

      - A record field contributes its name, a list or array element its
        zero-based index; segments are joined with ["."] (e.g. ["layers.0.w"]).
      - [Some x] contributes nothing; [None] skips the payload.
      - A payload at the root has the empty path. *)

  val fold2 :
    (string -> 'acc -> 'a -> 'b -> 'acc) -> 'acc -> 'a t -> 'b t -> 'acc
  (** [fold2 f acc a b] is like {!fold} but across two structurally equal trees.

      Raises [Invalid_argument] if [a] and [b] are not structurally equal. *)

  val names : 'a t -> string t
  (** [names t] is [t] with every payload replaced by its path (see {!fold} for
      the path convention). *)
end

(** [Make (U)] is the parameter tree over [U] with packed tensor leaves: its [t]
    is [tensor U.t], and its traversals check leafwise that dtypes agree. *)
module Make (U : Traverse) : S with type t = tensor U.t

val typed :
  (module U : Traverse) -> (module S with type t = ('a, 'b) Nx_effect.t U.t)
(** [typed (module U)] is the parameter tree over [U] at a single tensor type:
    the structure instantiated at typed leaves, satisfying {!S} with no packing.
    Bind it once per structure and pass it wherever a first-class {!S} module is
    expected:

    {[
      let mlp = Nx.Ptree.typed (module Mlp) in
      let grads = grad mlp loss params
    ]} *)

val unpack :
  ?at:string -> ('a, 'b) Nx_core.Dtype.t -> tensor -> ('a, 'b) Nx_effect.t
(** [unpack ?at dt p] unpacks the packed tensor [p] and returns it with the
    dtype [dt]; raises [Invalid_argument] if the dtype does not match. The
    optional [~at] string, if non-empty, is included in the error message as the
    location of the mismatch. *)

(** The stock dynamic tree: ordered lists and string-keyed dicts, generic in
    the payload. It satisfies {!module-type:Uniform}, with zero-based list
    positions and dict keys as path segments. *)
module Tree : sig
  type 'a t =
    | Leaf of 'a  (** A payload leaf. *)
    | List of 'a t list  (** An ordered list branch. *)
    | Dict of (string * 'a t) list  (** A dict branch. Keys are strings. *)

  include Uniform with type 'a t := 'a t
end

type t = tensor Tree.t
(** The stock dynamic parameter tree — {!Tree} with packed tensor leaves;
    itself satisfies {!S}. *)

val tensor : ('a, 'b) Nx_effect.t -> t
(** [tensor x] is [Tree.Leaf (P x)]. *)

val list : t list -> t
(** [list ts] is [Tree.List ts]. *)

val dict : (string * t) list -> t
(** [dict kvs] is [Tree.Dict kvs]. *)

val map : ('a 'b. ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t) -> t -> t
(** [map f t] is [t] with [f] applied to every tensor leaf. *)

val map2 :
  ('a 'b. ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t) ->
  t ->
  t ->
  t
(** [map2 f a b] is [a] and [b] combined leafwise with [f].

    Raises [Invalid_argument] if [a] and [b] differ in structure, dict keys, or
    leaf dtype. *)

val iter : ('a 'b. ('a, 'b) Nx_effect.t -> unit) -> t -> unit
(** [iter f t] applies [f] to every tensor leaf of [t]. *)
