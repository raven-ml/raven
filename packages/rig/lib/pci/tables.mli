(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Hash tables keyed by addresses.

    An address lies on a large alignment, so a key's hash mixes every bit into
    the low ones a table indexes by, and integer keys compare as integers. *)

(** Tables keyed by an address, which an exception inside a change leaves
    whole.

    A change allocates what it needs first and then publishes it with one
    store, and growing builds the new buckets whole before one store
    replaces the old ones. An exception raised where a change allocates or
    polls, such as [Sys.Break] or [Out_of_memory], leaves the table as it
    was before the change or as the change made it. The stdlib's [Hashtbl]
    stores its new bucket array before it fills it, so an exception while it
    grows loses the table. A table takes no lock: its users hold one. *)
module Address : sig
  type key = int

  type 'a t
  (** The type for tables from addresses to ['a]. *)

  val create : int -> 'a t
  (** [create n] is an empty table sized for [n] bindings. *)

  val length : 'a t -> int
  (** [length t] is the number of bindings of [t]. *)

  val find : 'a t -> key -> 'a
  (** [find t k] is [k]'s value in [t]. It allocates nothing. Raises
      [Not_found] if [t] binds no [k]. *)

  val find_opt : 'a t -> key -> 'a option
  (** [find_opt t k] is [k]'s value in [t], if any. *)

  val mem : 'a t -> key -> bool
  (** [mem t k] is [true] iff [t] binds [k]. *)

  val replace : 'a t -> key -> 'a -> unit
  (** [replace t k v] binds [k] to [v] in [t], in place of its binding. Its
      last step is the store that binds [k], and it allocates nothing when
      [t] binds [k] already. *)

  val remove : 'a t -> key -> unit
  (** [remove t k] removes [k]'s binding from [t], if any. It allocates
      nothing, and unbinds [k] with one store. *)

  val fold : (key -> 'a -> 'b -> 'b) -> 'a t -> 'b -> 'b
  (** [fold f t acc] folds [f] over the bindings of [t], in no particular
      order. [f] must not change [t]. *)
end

module Range : Hashtbl.S with type key = int * int
(** Tables keyed by a range: its address and its bytes. *)

module Window : Hashtbl.S with type key = Window.t
(** Tables keyed by a window, hashed by its address. *)
