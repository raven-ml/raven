(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Identity-keyed tensor maps.

    Tensors are keyed by physical identity: every Nx operation allocates a fresh
    tensor, so a tensor value identifies a node of the computation graph.
    Entries store the key's dtype, and lookups recover the static type through a
    dtype witness — an entry is only ever stored under the key of the tensor
    whose dtype it records, so the witness check cannot fail.

    This module is the single home of that pattern; the reverse tape, the
    forward tangent store and jit's constant tables build on it. A key hashes by
    [Nx_effect.identity_hash]: a placed or traced value by its id, so keying one
    never reads it. *)

type key = Key : ('a, 'b) Nx_effect.t -> key

module Tbl : Hashtbl.S with type key = key
(** Tables keyed by tensor identity. *)

type t
(** A map from tensors to same-typed tensors. *)

val create : unit -> t
(** [create ()] is an empty map. *)

val find : t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t option
(** [find m x] is the tensor bound to [x], if any. *)

val set : t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> unit
(** [set m x v] binds [x] to [v], replacing any previous binding. [v] must have
    [x]'s dtype and shape. *)

(** Identity sets of tensors. *)
module Ids : sig
  type t

  val create : unit -> t
  val add : t -> ('a, 'b) Nx.t -> unit
  val mem : t -> ('a, 'b) Nx.t -> bool
end
