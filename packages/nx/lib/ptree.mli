(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Parameter trees: structures with tensor leaves.

    {!module-type:S} is the traversal interface the Raven ecosystem shares:
    autodiff transformations, structural optimizers and checkpointing all
    operate on any structure implementing it — typically a record of tensors
    with hand-written one-line traversals. The concrete {!type:t} is the stock
    instance for structures only known at runtime. *)

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

(** Uniform homomorphic structures.

    A [('a, Gtree).t] is a structure whose payload positions are all of type
    ['a] — the "higher-kinded data" (HKD) shape that enables polymorphic
    traversals like [map2 : ('a -> 'b -> 'c) -> 'a t -> 'b t -> 'c t].

    The classic {!module-type:S} is the rank-2 bridge over packed tensor leaves;
    {!module-Tensor_tree} converts any {!Gtree} into an {!S}.

    Implementations must satisfy:
    - [map f t] applies [f] to every payload of [t] exactly once and preserves
      the structure of [t].
    - [map2 f a b] applies [f] to corresponding payloads of [a] and [b], which
      must be structurally equal.
    - [iter f t] applies [f] to every payload of [t] exactly once, in a stable
      order.
    - [fold f acc t] visits every payload in stable order; the [string] argument
      is the dotted path to the current position.
    - [fold2] is like {!fold} but across two structurally equal trees. *)
module type Gtree = sig
  type 'a t
  (** The structure type, parameterised by the uniform payload type. *)

  val map : ('a -> 'b) -> 'a t -> 'b t
  (** [map f t] is [t] with [f] applied to every payload leaf. *)

  val map2 : ('a -> 'b -> 'c) -> 'a t -> 'b t -> 'c t
  (** [map2 f a b] is [a] and [b] combined leafwise with [f].

      Raises [Invalid_argument] if [a] and [b] are not structurally equal. *)

  val iter : ('a -> unit) -> 'a t -> unit
  (** [iter f t] applies [f] to every payload leaf of [t]. *)

  val fold : (string -> 'acc -> 'a -> 'acc) -> 'acc -> 'a t -> 'acc
  (** [fold f acc t] reduces [t] to an accumulator, threading a dotted path to
      each payload leaf. The path convention (stable for JIT caching):

      - Root starts at [""].
      - Record field [w] appends [".w"] under the parent path.
      - List / array at index [i] appends [".i"].
      - Option [Some x] keeps the path unchanged; [None] skips. *)

  val fold2 :
    (string -> 'acc -> 'a -> 'b -> 'acc) -> 'acc -> 'a t -> 'b t -> 'acc
  (** [fold2 f acc a b] is like {!fold} but across two structurally equal trees.

      Raises [Invalid_argument] if [a] and [b] are not structurally equal. *)
end

(** [Tensor_tree (U)] bridges a uniform {!Gtree} to the rank-2 {!S} interface by
    packing every payload leaf as {!tensor}. The resulting [t] is [tensor U.t].
*)
module Tensor_tree (U : Gtree) : S with type t = tensor U.t

val unpack :
  ?at:string -> ('a, 'b) Nx_core.Dtype.t -> tensor -> ('a, 'b) Nx_effect.t
(** [unpack ?at dt p] unpacks the packed tensor [p] and returns it with the
    dtype [dt]; raises [Invalid_argument] if the dtype does not match. The
    optional [~at] string, if non-empty, is included in the error message as the
    location of the mismatch. *)

(** The stock dynamic parameter tree; itself satisfies {!S}. *)
type t =
  | Tensor of tensor  (** A tensor leaf. *)
  | List of t list  (** An ordered list branch. *)
  | Dict of (string * t) list  (** A dict branch. Keys are strings. *)

val tensor : ('a, 'b) Nx_effect.t -> t
(** [tensor x] is [Tensor (P x)]. *)

val list : t list -> t
(** [list ts] is [List ts]. *)

val dict : (string * t) list -> t
(** [dict kvs] is [Dict kvs]. *)

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
