(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Parameter trees: structures with tensor leaves.

    {!module-type:S} is the traversal interface the Raven ecosystem shares:
    autodiff transformations, structural optimizers and checkpointing all
    operate on any structure implementing it — typically a record of tensors
    with hand-written one-line traversals. The concrete {!type:t} is the
    stock instance for structures only known at runtime. *)
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
    ('a 'b.
     ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t -> ('a, 'b) Nx_effect.t) ->
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

    Raises [Invalid_argument] if [a] and [b] differ in structure, dict keys,
    or leaf dtype. *)

val iter : ('a 'b. ('a, 'b) Nx_effect.t -> unit) -> t -> unit
(** [iter f t] applies [f] to every tensor leaf of [t]. *)
