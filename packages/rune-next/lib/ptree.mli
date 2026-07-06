(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Dynamically-typed parameter trees.

    A parameter tree is a finite tree with packed tensor leaves and container
    nodes. Use it when the parameter structure is only known at runtime (for
    example, models assembled from a configuration). When the structure is known
    statically, prefer a record implementing {!Rune_next.Differentiable}: it is
    fully typed and needs no runtime dtype checks. *)

type tensor =
  | P : ('a, 'b) Nx.t -> tensor
      (** A packed tensor. The wrapper hides dtype and layout parameters. *)

type t =
  | Tensor of tensor  (** A tensor leaf. *)
  | List of t list  (** An ordered list branch. *)
  | Dict of (string * t) list  (** A dict branch. Keys are strings. *)

(** {1:constructors Constructors} *)

val tensor : ('a, 'b) Nx.t -> t
(** [tensor x] is [Tensor (P x)]. *)

val list : t list -> t
(** [list ts] is [List ts]. *)

val dict : (string * t) list -> t
(** [dict kvs] is [Dict kvs]. *)

(** {1:traversals Traversals} *)

val map : ('a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) -> t -> t
(** [map f t] is [t] with [f] applied to every tensor leaf. *)

val map2 :
  ('a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) -> t -> t -> t
(** [map2 f a b] is [a] and [b] combined leafwise with [f].

    Raises [Invalid_argument] if [a] and [b] differ in structure, dict keys, or
    leaf dtype. *)

val iter : ('a 'b. ('a, 'b) Nx.t -> unit) -> t -> unit
(** [iter f t] applies [f] to every tensor leaf of [t]. *)
