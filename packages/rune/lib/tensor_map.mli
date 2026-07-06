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

    This module is the single home of that pattern; the reverse tape and the
    forward tangent store both build on it. Maintainers must preserve the
    invariant that keyed tensors are never mutated in place while a map is live
    — the differentiation handlers reject mutation. *)

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
