(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Storage identity: whether a compiled call's result took a consumed
    argument's memory. *)

val buffers : ('a, 'b) Nx.t -> Rig.Buffer.t list
(** [buffers x] is each device's buffer of [x]'s storage, in the order of
    [x]'s placement's devices: the same memory for a result that took a
    consumed value's storage and that value, read before the call.

    Raises [Invalid_argument] if [x] is traced or consumed. *)

val same : Rig.Buffer.t -> Rig.Buffer.t -> bool
(** [same b b'] is [true] iff [b] and [b'] start at the same byte of one
    memory. *)

val storage : Rig.Buffer.t list Windtrap.testable
(** [storage] compares lists of buffers by {!same}, printing where each
    starts. *)
