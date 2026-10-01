(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Values made once, from any domain.

    A compiled call makes its programs, links and device descriptions the first
    time a key meets them, and reads them without a lock afterwards. *)

val assoc : ('k * 'v) list Atomic.t -> Mutex.t -> 'k -> (unit -> 'v) -> 'v
(** [assoc cell latch k make] is [k]'s value in [cell], keys compared
    physically, made by [make] and added the first time under [latch]. [cell] is
    read without [latch], which only guards its additions. *)

(** Tables whose values are made once per key. *)
module Make (K : Hashtbl.HashedType) : sig
  type 'v t
  (** The type for tables from [K.t] to values of type ['v]. *)

  val create : unit -> 'v t
  (** [create ()] is an empty table. *)

  val find : 'v t -> K.t -> miss:(unit -> unit) -> (unit -> 'v) -> 'v
  (** [find t k ~miss make] is [k]'s value in [t]. The first time [t] meets [k],
      [miss ()] runs under [t]'s lock, and [k] is added only if it returns; the
      first call that reads [k]'s value then makes it with [make ()] under [k]'s
      own latch, so that the values of distinct keys are made concurrently and
      each key's once. If [make] raises, the next call for [k] makes it again.
  *)
end
