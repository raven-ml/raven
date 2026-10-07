(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Ranges of a machine's addresses that the process reads and writes.

    A window is a range of addresses of the machine a GPU is in: a PCI
    function's registers or memory behind one of its BARs, or system memory the
    function reaches. On this machine the range is mapped into the process; on
    another machine it is reached through a {{!transports}transport}, and each
    domain's accesses complete in the order it makes them.

    No access raises for the world. Through a transport whose machine failed
    ({!Machine.failed}), a read gives all ones and a write is dropped, as on a
    function that left the bus ({{!Device_pci.errors}errors}).

    Values are little-endian. Nothing is checked beyond the range: the caller
    keeps the mapping alive, and an access after it is unmapped is undefined.

    {b C.} A driver's C code accesses windows through [device_pci.h], which
    states its contract. *)

(** {1:windows Windows} *)

type t
(** The type for windows. Windows are values: two windows are equal iff they are
    the same bytes of the same machine. *)

val address : t -> int
(** [address w] is the address of [w]'s first byte on its machine. For a mapped
    window it is the process's address. *)

val length : t -> int
(** [length w] is [w]'s size in bytes. *)

val mapped : t -> bool
(** [mapped w] is [true] iff [w] is mapped into the process, so that {!address}
    is an address of the process. *)

val sub : t -> int -> int -> t
(** [sub w off n] is the [n] bytes of [w] from byte [off] on.

    Raises [Invalid_argument] if they do not lie in [w]. *)

(** {1:accesses Accesses}

    {!get32}, {!set32}, {!get64} and {!set64} are register accesses: one access
    of exactly that width, on this machine and through a transport, so a
    register is read and written once each and in program order. {!read},
    {!blit_string}, {!write} and {!fill} copy memory, at widths they choose;
    touch registers only with the accesses above. They are for device memory and
    windows through a transport. On a mapped window they go a 32-bit word at a
    time, which memory behind a BAR accepts, and a long one lets the domain's
    other threads and other domains' collections run meanwhile; a long read or
    write pays for that with a copy through a buffer. Bulk copies of this
    process's memory go through {!bigarray}. {!barrier} orders accesses for the
    processor and the bus.

    Each raises [Invalid_argument] if the bytes it accesses do not lie in the
    window, a count below zero included, or if a 32- or 64-bit access is at an
    address not aligned to its width. *)

val get8 : t -> int -> int
(** [get8 w off] is the byte at [off] of [w]. *)

val set8 : t -> int -> int -> unit
(** [set8 w off b] stores the low 8 bits of [b] at [off] of [w]. *)

val get32 : t -> int -> int
(** [get32 w off] is the unsigned 32-bit word at byte [off] of [w]. *)

val set32 : t -> int -> int -> unit
(** [set32 w off x] stores the low 32 bits of [x] at byte [off] of [w]. *)

val get64 : t -> int -> int64
(** [get64 w off] is the 64-bit word at byte [off] of [w]. *)

val set64 : t -> int -> int64 -> unit
(** [set64 w off x] stores [x] at byte [off] of [w]. *)

val read : t -> int -> int -> string
(** [read w off n] is a copy of the [n] bytes of [w] from byte [off]. *)

val blit_string : string -> int -> t -> int -> int -> unit
(** [blit_string s soff w off n] copies the [n] bytes of [s] from byte [soff] to
    byte [off] of [w].

    Raises [Invalid_argument] also if the bytes do not lie in [s]. *)

val write : t -> int -> string -> unit
(** [write w off s] is [blit_string s 0 w off (String.length s)]. *)

val fill : t -> int -> int -> char -> unit
(** [fill w off n c] stores [n] bytes [c] from byte [off] of [w]. *)

val barrier : unit -> unit
(** [barrier ()] orders every access before it before every access after it, as
    devices see them as well as other processors. On arm64 this is the
    full-system barrier, which stores to a BAR need. *)

val bigarray :
  t -> (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
(** [bigarray w] is [w]'s bytes as a bigarray, without a copy. The caller keeps
    the mapping alive while the bigarray is reachable.

    Raises [Invalid_argument] if [w] is not {!mapped}. *)

(** {1:transports Transports}

    For the libraries that reach another machine. A transport is a
    [struct device_pci_transport] of C functions that read and write the
    machine's addresses, declared in [device_pci.h], which states their
    contract. *)

type transport
(** The type for transports. *)

val unsafe_transport : int -> transport
(** [unsafe_transport p] is the transport whose [struct device_pci_transport] is
    at address [p] of the process. Unsafe: nothing checks [p], which must point
    to such a structure that stays valid and unchanged while a window or a
    machine uses it. *)

val through : transport -> int -> int -> t
(** [through tr a n] is the [n] bytes at [a] of the machine [tr] reaches.

    Raises [Invalid_argument] if [n < 0] or [tr] is [unsafe_transport 0]. *)

(**/**)

val v : int -> int -> t
(* [v a n] is the [n] bytes mapped at [a] in the process, for the library's own
   mappings. Raises [Invalid_argument] if [n < 0]. *)
