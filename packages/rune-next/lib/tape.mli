(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Reverse-mode tape.

    A tape records the differentiated subgraph of a traced computation: which
    tensors are reachable from the registered inputs ({e tracked}), the pull
    thunks that map an operation's output cotangent to input contributions, and
    the cotangents accumulated during the backward pass.

    Tensors are keyed by physical identity: every Nx operation allocates a fresh
    tensor, so a tensor value identifies a node of the computation graph.
    Cotangent entries store the tensor's dtype, and lookups recover the static
    type through a dtype witness — an entry is only ever stored under the key of
    the tensor whose dtype it records, so the witness check cannot fail.

    Maintainers must preserve two invariants:
    - tracked tensors are never mutated in place while the tape is live (the
      reverse handler rejects mutation during differentiation);
    - [accumulate] is only called with a contribution of the key's exact shape
      and dtype. *)

type t
(** A tape. Create one per differentiation, discard after {!backward}. *)

val create : unit -> t
(** [create ()] is an empty tape. *)

(** {1:tracking Forward pass} *)

val track : t -> ('a, 'b) Nx.t -> unit
(** [track tape x] marks [x] as reachable from the differentiated inputs. *)

val tracked : t -> ('a, 'b) Nx.t -> bool
(** [tracked tape x] is [true] iff [x] was marked with {!track}. *)

val record : t -> (unit -> unit) -> unit
(** [record tape pull] appends [pull] to the tape in forward order. [pull] runs
    during {!backward}; it reads its output cotangents with {!find} and
    contributes to its inputs with {!accumulate}. *)

(** {1:backward Backward pass} *)

val backward : t -> unit
(** [backward tape] runs the recorded pull thunks in reverse order. Call after
    seeding the output cotangent with {!accumulate}. *)

val find : t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t option
(** [find tape x] is the cotangent accumulated for [x], if any. *)

val accumulate : t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> unit
(** [accumulate tape x g] adds [g] to the cotangent accumulated for [x]. *)

val cotangent : t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [cotangent tape x] is the cotangent accumulated for [x], or an all-zero
    tensor like [x] if none was. *)
