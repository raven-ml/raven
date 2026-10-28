(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Values made once per key, from any domain.

    A table makes a key's value the first time a call reads it, and later calls
    from every domain read that value.

    A table's lock guards its keys, and each key has a {e latch}, a lock held
    while the key's value is made, so the values of distinct keys may be made
    concurrently. Other calls for a key block until its value is made. A call
    for a key whose value its own domain is making, from inside [make] or from a
    fiber the domain resumes meanwhile, returns [make ()] without waiting. The
    table does not keep that value. *)

(** Tables whose values are made once per key. *)
module Make (K : Hashtbl.HashedType) : sig
  type 'v t
  (** The type for tables from [K.t] to values of type ['v]. *)

  val create : unit -> 'v t
  (** [create ()] is an empty table. *)

  val find : 'v t -> K.t -> miss:(unit -> unit) -> (unit -> 'v) -> 'v
  (** [find t k ~miss make] is [k]'s value in [t]. The first call for [k] runs
      [miss ()] under [t]'s lock and adds [k] to [t] only if [miss] returns. The
      value is made with [make ()] by the first call that takes [k]'s latch
      while [k] has none: that first call, unless another call for [k] takes the
      latch first. A call made while its own domain makes [k]'s value runs
      [make ()] itself, and that value is not kept. If [make] raises, the
      exception reaches its caller, and the next call for [k], a waiting one
      included, makes the value again. *)
end
