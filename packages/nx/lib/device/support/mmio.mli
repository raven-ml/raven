(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Memory the process maps from a device.

    A range of addresses in the process: a device's registers or memory mapped
    through one of its PCI BARs, system memory a device reads and writes, or a
    driver's mapping of either. Every access is a volatile load or store of
    exactly its width, so reads and writes of registers happen once each and as
    written; the compiler keeps them in program order, and {!barrier} orders
    them for the processor and the bus. Nothing is checked beyond the range: the
    caller keeps the mapping alive. Values are little-endian.

    A range may also be another machine's, reached through an {!access}: such as
    the registers of a GPU that a server of that machine maps ({!Remote}). Its
    reads are round trips; its writes are sent in order and may complete after
    they return, but before any later access through the same {!access}. *)

type t
(** The type for mapped ranges. *)

val v : nativeint -> int -> t
(** [v a n] is the [n] bytes mapped at [a] in the process.

    Raises [Invalid_argument] if [n < 0]. *)

type access = {
  read : nativeint -> int -> string;
      (** [read a n] is the [n] bytes at [a]: one access of their width when [n]
          is 1, 2, 4 or 8 and [a] is aligned to it. *)
  write : nativeint -> string -> unit;
      (** [write a s] stores [s] at [a], as [read] reads. *)
}
(** The type for the accesses to another machine's addresses. *)

val remote : access -> nativeint -> int -> t
(** [remote acc a n] is the [n] bytes at [a] in another machine's address space,
    reached through [acc].

    Raises [Invalid_argument] if [n < 0]. *)

val is_remote : t -> bool
(** [is_remote m] is [true] iff [m] is another machine's. *)

val address : t -> nativeint
(** [address m] is the address of [m]'s first byte, in the address space of the
    machine it is in. *)

val length : t -> int
(** [length m] is [m]'s size in bytes. *)

val sub : t -> int -> int -> t
(** [sub m off n] is the [n] bytes of [m] from byte [off] on.

    Raises [Invalid_argument] if they do not lie in [m]. *)

val get8 : t -> int -> int
(** [get8 m off] is the byte at [off] of [m].

    Raises [Invalid_argument] if it does not lie in [m]. *)

val set8 : t -> int -> int -> unit
(** [set8 m off b] stores the low 8 bits of [b] at [off] of [m].

    Raises [Invalid_argument] as {!get8} does. *)

val get32 : t -> int -> int
(** [get32 m off] is the unsigned 32-bit word at byte [off] of [m].

    Raises [Invalid_argument] if [off] is not 4-byte aligned or the word does
    not lie in [m]. *)

val set32 : t -> int -> int -> unit
(** [set32 m off w] stores the low 32 bits of [w] at byte [off] of [m].

    Raises [Invalid_argument] as {!get32} does. *)

val get64 : t -> int -> int64
(** [get64 m off] is the 64-bit word at byte [off] of [m], read in one access.

    Raises [Invalid_argument] if [off] is not 8-byte aligned or the word does
    not lie in [m]. *)

val set64 : t -> int -> int64 -> unit
(** [set64 m off w] stores [w] at byte [off] of [m] in one access.

    Raises [Invalid_argument] as {!get64} does. *)

val read : t -> int -> int -> string
(** [read m off n] is the [n] bytes of [m] from byte [off], read in 32-bit words
    where aligned.

    Raises [Invalid_argument] if they do not lie in [m]. *)

val write : t -> int -> string -> unit
(** [write m off s] stores [s] at byte [off] of [m], in 32-bit words where
    aligned.

    Raises [Invalid_argument] if [s] does not fit in [m] from [off]. *)

val fill : t -> int -> int -> char -> unit
(** [fill m off n c] stores [n] bytes [c] from byte [off] of [m].

    Raises [Invalid_argument] if they do not lie in [m]. *)

val bigarray :
  t -> (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
(** [bigarray m] is [m]'s bytes as a bigarray, without a copy. The caller keeps
    the mapping alive for as long as the bigarray is reachable.

    Raises [Invalid_argument] if [m] is another machine's. *)

val barrier : unit -> unit
(** [barrier ()] orders every memory access before it before every access after
    it, as seen by devices as well as by other processors. *)
