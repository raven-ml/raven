(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Pretoken caches.

    {b Internal module.} A cache maps a pretoken of at most {!max_key_len} bytes
    to the at most four ids a model encodes it to, so that a pretoken seen
    before is not encoded again. A small direct-mapped front table of 4096
    entries (128 KB), when there is one, is probed first, then a two-way
    set-associative back table, whose sets pair up as the two halves of 128
    bytes: a key missing from its own set is looked for in the other one,
    which then trades it back for the older entry of its own set. A hit is
    promoted to the front, and a miss is stored into its set, whose older entry
    moves on into the other set, evicting that set's older entry.

    An entry is 32 bytes of four native-endian words: the pretoken's first eight
    bytes in little-endian order, zero-padded; its next seven bytes, likewise,
    with its length in the top byte; and two words of two 24-bit ids each, at
    bits 0 and 32, the first also holding the id count in bits 24 to 31. The
    byte-level and SentencePiece C kernels read and write the same tables.

    A table is read and written by one encoding at a time, the one holding
    the {{!states}state} it belongs to. *)

type t = private {
  front : Bytes.t;
  fmask : int;
  mutable table : Bytes.t;
  mutable base : int;
  mutable mask : int;
  limit : int;
  mutable live : int;
}
(** The type for caches. [front] holds [fmask + 1] entries, none when [fmask]
    is below zero, and [table] holds [mask + 1] sets of two from [base], on a
    64-byte boundary where the C kernels run, [live] of whose entries the OCaml
    side has filled; a [mask] below zero is no cache. The back table grows to
    at most [limit] sets. *)

val none : t
(** [none] is no cache: nothing is found in it or stored into it. *)

val create : ?front:bool -> int -> t
(** [create capacity] is an empty cache of at most [capacity] back entries,
    rounded up to a power of two, or no cache when [capacity] is zero or less.
    The back table starts at 512 entries (16 KB) and doubles once half its
    entries are live, up to [capacity]: an entry stored without evicting
    another makes one more live, whether {!encode} or a C kernel stored it
    ({!stored}). Doubling moves the table, so a kernel's tables are taken from
    [t] again after anything that stores. With [front] (default [true]) the
    front table is allocated; the C kernels, which prefetch the back sets of
    the spans they walk ahead, have none. A full table takes 32 bytes per
    entry. *)

val stored : t -> int -> unit
(** [stored t n] counts [n] entries a C kernel stored into [t] without evicting
    one, and doubles [t]'s back table when {!encode} would have. *)

val default_capacity : int
(** [default_capacity] is the capacity models default to: 4194304 entries, 128
    MB once full, which a cache reaches only for the million or so distinct
    pretokens of about a gigabyte of web text. *)

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

(** {1:states Encoder states} *)

type 'a states
(** The type for a model's encoder states: values, such as a cache with the
    buffers that fill it, each written by one encoding at a time. *)

val states : unit -> 'a states
(** [states ()] holds no state. *)

val with_state :
  'a states -> make:('b -> 'a) -> spare:('b -> 'a) -> 'b -> ('a -> 'c) -> 'c
(** [with_state s ~make ~spare arg f] is [f st] for a state [st] taken from
    [s], or [make arg] when [s] holds none free and fewer than
    [Domain.recommended_domain_count ()] states exist; [st] goes back to [s]
    when [f] returns. Past that count, [st] is [spare arg], which [s] never
    keeps, so [s] never holds more states than that, in use or free. When [f]
    raises, [st] is dropped: the exception may have left it half-written.
    [st] must not escape [f]. *)
