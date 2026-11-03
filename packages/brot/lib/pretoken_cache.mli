(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Pretoken caches.

    {b Internal module.} A cache maps a pretoken of at most {!max_key_len} bytes
    to the at most four ids a model encodes it to, so that a pretoken seen
    before is not encoded again. A small direct-mapped front table of 4096
    entries (128 KB) is probed first, then a two-way set-associative back
    table; a back-table hit is promoted to the front, and a miss is stored into
    the back table, evicting the older entry of its set.

    An entry is 32 bytes of four native-endian words: the pretoken's first eight
    bytes in little-endian order, zero-padded; its next seven bytes, likewise,
    with its length in the top byte; and two words of two 24-bit ids each, at
    bits 0 and 32, the first also holding the id count in bits 24 to 31. The
    byte-level and SentencePiece C kernels read and write the same tables.

    A table is read and written only by the thread holding the claim on it. *)

type t = private {
  front : Bytes.t;
  fmask : int;
  mutable table : Bytes.t;
  mutable mask : int;
  limit : int;
  mutable live : int;
}
(** The type for caches. [front] holds [fmask + 1] entries and [table] holds
    [mask + 1] sets of two, [live] of whose entries the OCaml side has filled;
    a [mask] below zero is no cache. The back table grows to at most [limit]
    sets. *)

val none : t
(** [none] is no cache: nothing is found in it or stored into it. *)

val create : ?grow:bool -> int -> t
(** [create capacity] is an empty cache of at most [capacity] back entries,
    rounded up to a power of two, or no cache when [capacity] is zero or less.
    With [grow] (default [true]) the back table starts at 512 entries (16 KB)
    and doubles once half its entries are live; without it
    the table is allocated at its full size, as the C kernels, which store
    without growing it, need. The front table is allocated either way. A full
    table takes 32 bytes per entry. *)

val default_capacity : int
(** [default_capacity] is the capacity models default to: 262144 entries,
    8 MB once full. *)

val max_key_len : int
(** [max_key_len] is the length of the longest pretoken a cache holds: 15. *)

type encoder = Ints.t -> opaque:Ints.t -> string -> pos:int -> len:int -> unit
(** The type for span encoders: [encode ids ~opaque text ~pos ~len] appends the
    ids of [text.\[pos..pos+len)] to [ids], and records in [opaque] the ids
    that do not stand for exactly the bytes their length table gives. *)

val encode : t -> len_table:int array -> encoder -> encoder
(** [encode t ~len_table miss ids ~opaque text ~pos ~len] is
    [miss ids ~opaque text ~pos ~len] through [t]: a pretoken found in [t]
    appends its ids, and one that is not is encoded by [miss] and kept when it
    can be handed back as ids alone. That is when [miss] records nothing in
    [opaque] and appends one to four ids below [2]{^ 24} whose lengths in
    [len_table] add up to [len]. [miss] must be a function of the pretoken's
    bytes alone. *)

val cached : t -> len_table:int array -> encoder -> encoder
(** [cached t ~len_table miss] is [encode t ~len_table miss], or [miss] when [t]
    is no cache, as one closure. *)

val key_words : string -> int64 * int64
(** [key_words key] is the two words the pretoken [key] of 1 to {!max_key_len}
    bytes is keyed by. *)

val hash : int64 -> int64 -> int
(** [hash k0 k1] is the hash the key words [k0] and [k1] are indexed by: the
    back set is its low bits, and so is the front slot. *)

val set_index : string -> sets:int -> int
(** [set_index key ~sets] is the back set, among [sets], a power of two, that
    the pretoken [key] of 1 to {!max_key_len} bytes is stored in. *)

val seed : Buffer.t -> string -> int array -> int -> unit
(** [seed buf text ids count] appends to [buf] the entry mapping [text] to the
    first [count] of [ids], if a cache can hold it. *)

val load_seeds : t -> Bytes.t -> unit
(** [load_seeds t seeds] stores into [t] the entries {!seed} appended to
    [seeds]. *)

(** {1:claims Claims} *)

val stamp : unit -> int
(** [stamp ()] is a number no other call returns, which names a model's slots.
*)

type 'a slot = private { value : 'a; busy : bool Atomic.t }
(** The type for a domain's slot of a model: a value, such as a cache, and
    whether a thread of the domain holds it. *)

type 'a registry
(** The type for per-domain slots of ['a] values. *)

val registry : unit -> 'a registry
(** [registry ()] is a registry holding no slot. *)

val slot : 'a registry -> stamp:int -> ('b -> 'a) -> 'b -> 'a slot
(** [slot r ~stamp make arg] is the calling domain's slot in [r] of the model
    [stamp] names, made with [make arg] the first time. A domain keeps the slots
    of the last eight models it began encoding with: past that, the slot it
    made longest ago is dropped, whether or not it was used since, and its
    memory goes with it. *)

val take : 'a slot -> bool
(** [take s] claims [s] and is [true], or is [false] if a thread holds it. *)

val give : 'a slot -> unit
(** [give s] gives back the claim {!take} made. *)
