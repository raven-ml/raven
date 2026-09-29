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
    caller keeps the mapping alive. Values are little-endian. *)

type t
(** The type for mapped ranges. *)

val v : nativeint -> int -> t
(** [v a n] is the [n] bytes mapped at [a].

    Raises [Invalid_argument] if [n < 0]. *)

val address : t -> nativeint
(** [address m] is the address of [m]'s first byte. *)

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

val barrier : unit -> unit
(** [barrier ()] orders every memory access before it before every access after
    it, as seen by devices as well as by other processors. *)
