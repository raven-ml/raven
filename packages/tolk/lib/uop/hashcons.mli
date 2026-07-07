(**************************************************************************)
(*                                                                        *)
(*  Copyright (C) Jean-Christophe Filliatre                               *)
(*                                                                        *)
(*  This software is free software; you can redistribute it and/or        *)
(*  modify it under the terms of the GNU Library General Public           *)
(*  License version 2.1, with the special exception on linking            *)
(*  described in file LICENSE.                                            *)
(*                                                                        *)
(*  This software is distributed in the hope that it will be useful,      *)
(*  but WITHOUT ANY WARRANTY; without even the implied warranty of        *)
(*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                  *)
(*                                                                        *)
(**************************************************************************)

(** Hash-consing tables.

    Hash-consing maintains at most one representative per class of
    structurally equal values, so that equality on interned values reduces
    to pointer identity. Tables hold their entries through weak pointers:
    representatives that become otherwise unreachable may be reclaimed by
    the GC, and the next call to {!S.hashcons} on an equal value will
    produce a fresh representative with a new {!field-tag}.

    The technique is described in Sylvain Conchon and Jean-Christophe
    Filliâtre, {e Type-Safe Modular Hash-Consing}, ACM SIGPLAN Workshop on
    ML, Portland, Oregon, September 2006.

    {b Origin.} Vendored from
    {{:https://github.com/backtracking/hashcons}backtracking/hashcons}.
    The generic top-level interface and the [Hmap]/[Hset] modules have
    been removed; only the {!Make} functor is kept. *)

(** {1:types Types} *)

type +'a hash_consed = private {
  hkey : int;
      (** Hash of {!field-node}, masked to a non-negative integer. Exposed
          so clients can reuse it in other hash tables keyed on the
          underlying value, avoiding recomputation. *)
  tag : int;
      (** Unique identifier assigned at interning time. Tags are drawn from
          a single monotonically increasing global counter shared by all
          tables produced by {!Make}, so tags are unique across tables as
          well as within a table. *)
  node : 'a;  (** The underlying value. *)
}
(** The type for hash-consed values. Within a given table, two values are
    physically equal iff they have the same {!field-tag} iff their
    {!field-node}s are structurally equal (per the [equal] of the
    {!HashedType} used to build the table). Physical equality is therefore
    a sound substitute for structural equality on interned values. *)

val gentag_peek : unit -> int
(** [gentag_peek ()] is the current value of the global tag counter,
    without incrementing it. Useful for diagnostics and for sizing
    secondary structures indexed by tag. The counter is shared by every
    table built with {!Make}. *)

(** {1:functorial Functorial interface} *)

(** The input signature of {!Make}. *)
module type HashedType = sig
  type t
  (** The type of values to hash-cons. *)

  val equal : t -> t -> bool
  (** [equal a b] decides whether [a] and [b] must be interned as the same
      representative. It must be a true equivalence relation on [t] and
      must be compatible with {!hash}: [equal a b] implies [hash a = hash
      b]. *)

  val hash : t -> int
  (** [hash a] is a hash of [a] compatible with {!equal}. Any integer is
      accepted; the implementation masks it to a non-negative value. *)
end

(** The output signature of {!Make}. *)
module type S = sig
  type key
  (** The type of values being hash-consed. *)

  type t
  (** The type of hash-cons tables over {!key}. *)

  val create : int -> t
  (** [create n] is a fresh, empty table initially sized for about [n]
      entries. [n] is clamped to the closed interval
      \[[7]; {!Sys.max_array_length}\]; values outside that range are
      silently adjusted. *)

  val clear : t -> unit
  (** [clear t] removes every entry from [t] and resets its capacity.
      Previously returned representatives remain valid values but lose
      their association with [t]: a subsequent {!hashcons} of an equal
      value will allocate a new representative with a fresh tag. *)

  val hashcons : t -> key -> key hash_consed
  (** [hashcons t v] is the representative of [v] in [t]. If [t] already
      holds a live entry [r] with [H.equal r.node v], then [r] is
      returned; otherwise a fresh representative is interned, assigned
      the next global tag, and returned.

      Amortised expected cost is constant, assuming {!H.hash} distributes
      values well. The returned representative keeps its entry alive in
      [t] for as long as it is itself reachable. *)

  val iter : (key hash_consed -> unit) -> t -> unit
  (** [iter f t] applies [f] to every live representative currently held
      in [t]. Iteration order is unspecified. Entries whose weak pointer
      has been cleared by the GC are skipped. *)

  val stats : t -> int * int * int * int * int * int
  (** [stats t] is [(buckets, entries, total_slots, min_len, median_len,
      max_len)] where [buckets] is the number of bucket arrays in [t],
      [entries] is the count of live entries, [total_slots] is the sum of
      bucket capacities, and [min_len], [median_len], [max_len] summarise
      bucket lengths. Intended for diagnostics and tuning. *)
end

module Make (H : HashedType) : S with type key = H.t
(** [Make (H)] builds a hash-cons table type for values of type [H.t],
    using [H.equal] and [H.hash] to decide when two values share a
    representative. *)
