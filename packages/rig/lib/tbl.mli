(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Hash tables that an exception inside a change leaves whole.

    A change allocates what it needs first and then publishes it with one
    store, and growing builds the new buckets whole before one store
    replaces the old ones. An exception raised where a change allocates or
    polls, such as [Sys.Break] or [Out_of_memory], leaves the table as it
    was before the change or as the change made it. The stdlib's
    [Hashtbl] stores its new bucket array before it fills it, so an
    exception while it grows loses the table.

    rig keeps in these the tables whose loss breaks a later call: the names
    of devices, the machines and the hosts opening on them, the
    devices whose values dropped, the stages of copies, and each device's
    cache and the memory it maps. A table takes no lock: its users hold
    one. *)

module Make (K : sig
  type t

  val equal : t -> t -> bool
  val hash : t -> int
end) : sig
  type key = K.t

  type 'a t
  (** The type for tables from [key] to ['a]. *)

  val create : int -> 'a t
  (** [create n] is an empty table sized for [n] bindings. *)

  val length : 'a t -> int
  (** [length t] is the number of bindings of [t]. *)

  val find : 'a t -> key -> 'a
  (** [find t k] is [k]'s value in [t]. It allocates nothing beyond [K.hash] and
      [K.equal]. Raises
      [Not_found] if [t] binds no [k]. *)

  val find_opt : 'a t -> key -> 'a option
  (** [find_opt t k] is [k]'s value in [t], if any. *)

  val mem : 'a t -> key -> bool
  (** [mem t k] is [true] iff [t] binds [k]. *)

  val replace : 'a t -> key -> 'a -> unit
  (** [replace t k v] binds [k] to [v] in [t], in place of its binding. Its
      last step is the store that binds [k], so a store the caller makes
      right after it, with nothing that allocates between, goes with the
      binding. *)

  val remove : 'a t -> key -> unit
  (** [remove t k] removes [k]'s binding from [t], if any. *)

  val fold : (key -> 'a -> 'b -> 'b) -> 'a t -> 'b -> 'b
  (** [fold f t acc] folds [f] over the bindings of [t], in no particular
      order. [f] must not change [t]. *)

  val reset : 'a t -> unit
  (** [reset t] removes every binding of [t], and gives [t] 8 buckets. *)

  val trim : 'a t -> unit
  (** [trim t] is [reset t] if [t] holds no binding and has more than 8
      buckets, and does nothing otherwise: an emptied table keeps no buckets
      for the most it held, and a small one allocates nothing. *)
end
