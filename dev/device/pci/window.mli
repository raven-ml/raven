(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Ranges of a machine's addresses that the process reads and writes.

    A window is a range of addresses of the machine a GPU is in: a PCI
    function's registers or memory behind one of its BARs, or system memory the
    function reaches. On this machine the range is mapped into the process and
    each access is one volatile load or store of exactly its width, so a
    register is read and written once each and in program order; {!barrier}
    orders accesses for the processor and the bus. On another machine the range
    is reached through a {{!transports}transport}, and each domain's accesses
    complete in the order it makes them. An access through a transport whose
    machine failed raises [Failure] with {!Machine.failed}'s reason
    ({{!Device_pci.errors}errors}).

    Values are little-endian. Nothing is checked beyond the range: the caller
    keeps the mapping alive, and an access after it is unmapped is undefined.

    {b C.} A driver's C code reaches windows through [device_pci.h]:
    [device_pci_window_of] reads a window into a [struct device_pci_window], and
    [device_pci_store32], [device_pci_store64], [device_pci_load32],
    [device_pci_load64] and [device_pci_write] access it without calling OCaml.
    On a mapped window they are the stores and loads; through a transport they
    call its C functions, so a submission that writes a queue stays one section
    of C on every machine. They return [0], or [-1] once the transport failed.
*)

(** {1:windows Windows} *)

type t
(** The type for windows. *)

val v : int -> int -> t
(** [v a n] is the [n] bytes mapped at [a] in the process.

    Raises [Invalid_argument] if [n < 0]. *)

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

    Each raises [Invalid_argument] if the bytes it accesses do not lie in the
    window, or if a 32- or 64-bit access is not aligned to its width. *)

val get8 : t -> int -> int
(** [get8 w off] is the byte at [off] of [w]. *)

val set8 : t -> int -> int -> unit
(** [set8 w off b] stores the low 8 bits of [b] at [off] of [w]. *)

val get32 : t -> int -> int
(** [get32 w off] is the unsigned 32-bit word at byte [off] of [w]. *)

val set32 : t -> int -> int -> unit
(** [set32 w off x] stores the low 32 bits of [x] at byte [off] of [w]. *)

val get64 : t -> int -> int64
(** [get64 w off] is the 64-bit word at byte [off] of [w], read in one access.
*)

val set64 : t -> int -> int64 -> unit
(** [set64 w off x] stores [x] at byte [off] of [w] in one access. *)

val read : t -> int -> int -> string
(** [read w off n] is the [n] bytes of [w] from byte [off], read a 32-bit word
    at a time where the window's side is aligned: memory behind a BAR need not
    accept other widths. *)

val write : t -> int -> string -> unit
(** [write w off s] stores [s] at byte [off] of [w], as {!read} reads. *)

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
    contract. The library calls them without holding the OCaml runtime. *)

type transport
(** The type for transports. *)

val transport : int -> transport
(** [transport p] is the transport whose [struct device_pci_transport] is at
    address [p] of the process. The structure stays valid and unchanged while a
    window or a machine uses it. *)

val through : transport -> int -> int -> t
(** [through tr a n] is the [n] bytes at [a] of the machine [tr] reaches.

    Raises [Invalid_argument] if [n < 0]. *)
