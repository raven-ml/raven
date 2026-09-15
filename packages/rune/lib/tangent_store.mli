(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Ephemeron-keyed store of batched forward-mode tangents.

    The batched forward-mode handler maps every tensor to its tangent batch: [k]
    lanes stacked on a leading axis of the tensor's own shape. Under a long
    unrolled computation — a recurrence stepped over its inputs, a solver loop —
    the primals of past steps die as the loop moves past them, but a
    strong-keyed table (like {!Tensor_map}) would keep every entry alive
    together with the [k]-lane tangent it holds: the store itself would grow
    without bound, a [k]-fold multiplier on the memory the loop was trying to
    bound.

    Keying on ephemerons ties each binding's lifetime to its primal's: once the
    surrounding computation drops a tensor, the binding — and the tangent it
    retains — becomes collectable at the next major collection. The store holds
    the live working set of the differentiation, nothing more. {!live_entries}
    counts those live bindings, which makes the bounded-memory property
    observable. *)

type t
(** A map from tensors to their tangent batches. *)

val create : k:int -> t
(** [create ~k] is an empty store for tangent batches of [k] lanes. *)

val k : t -> int
(** [k t] is the lane count every stored tangent batches over. *)

val find : t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t option
(** [find t x] is the tangent batch bound to [x], if any. A tensor absent from
    the store is a constant of the differentiation. *)

val set : t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> unit
(** [set t x v] binds the tangent batch [v] to [x]. [v] must have [k t] lanes
    stacked on a leading axis of [x]'s shape.

    Raises [Invalid_argument] if [v]'s shape is not [k :: shape x]: a
    lane-stacked tangent can only be combined by operations that keep the lane
    axis first, so a mismatch means some enclosing transformation batched the
    computation around the tangent axis. Batch dimensions belong inside it. *)

val live_entries : t -> int
(** [live_entries t] is the number of bindings whose primals are still
    reachable. Bindings for unreachable primars are dropped by the garbage
    collector; the count walks the table, so it is not free. *)

val shape_of : ('a, 'b) Nx.t -> int array
(** [shape_of x] is [x]'s physical shape, read without performing the [E_view]
    effect and without forcing a deferred tensor (an unread jit output): an
    enclosing transformation may present a transformed view — vmap shows batched
    tensors without their batch axis — while the store's invariant is about the
    tensors as they physically are. *)
