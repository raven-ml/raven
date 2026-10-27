(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Values made once per key, from any domain.

    A compiled function makes its program the first time it meets a key, and
    reads it without making it again. *)

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
      each key's once. A call on the domain making [k]'s value, from inside
      [make] or from a fiber the domain resumes meanwhile, makes a value with
      [make ()] for itself, which [t] does not keep, where waiting would wait
      for itself. If [make] raises, the next call for [k] makes it again. *)
end
