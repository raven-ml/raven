(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Ranges of a machine's addresses that the process reads and writes.

    A window is a range of addresses of the machine a GPU is in: a PCI
    function's registers or memory behind one of its BARs, or system memory the
    function reaches. On this machine the range is mapped into the process. On
    another machine it is reached through a {{!transports}transport}, and each
    domain's accesses complete in the order it makes them.

    No access raises when its machine fails. Through a transport whose machine
    has failed ({!Machine.failed}), a read gives all ones and a write is
    dropped, as on a function that has left the bus
    ({{!Rig_pci.errors}errors}).

    Values are little-endian. Nothing is checked beyond the window's range. The
    mapping must stay alive while the window is used: an access after it is
    unmapped is undefined.

    A driver's C code accesses windows through [rig_pci.h], which states its
    contract. *)

(** {1:windows Windows} *)

type t
(** The type for windows. Windows are values. Two windows are equal iff they
    have the same {!address}, {!length} and machine, and both combine or
    neither does. *)

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

    {!get8}, {!set8}, {!get32}, {!set32}, {!get64} and {!set64} are register
    accesses. Each is one access of exactly that width, on this machine and
    through a transport, so a register is read and written once each and in
    program order. A driver must touch registers only with these accesses.

    A window {e combines} if {!Function.map} made it combining, where the
    machine allows it. Through such a window the accesses above are memory
    accesses. Stores may merge and reach the function in another order, and
    loads may be served early, until a {!barrier} orders them. A driver puts a
    barrier between a load of a word the function wrote and the loads that
    word guards.

    A driver flushes each window it wrote ({!flush}) before the store that has
    the function read what it wrote, such as a doorbell.

    {!read}, {!blit_string}, {!write} and {!fill} copy memory, at widths they
    choose. On a mapped window they move a 32-bit word at a time, which memory
    behind any BAR accepts, or 16 bytes at a time through a window that
    combines. A long copy lets the domain's other threads and other domains'
    collections run meanwhile, and a long read or write pays for that with a
    copy through a buffer.

    Each access raises [Invalid_argument] if the bytes it accesses do not lie
    in the window, a count below zero included, or if a 32- or 64-bit access is
    at an address not aligned to its width. *)

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

    Raises [Invalid_argument] also if those bytes do not lie in [s]. *)

val write : t -> int -> string -> unit
(** [write w off s] is [blit_string s 0 w off (String.length s)]. *)

val fill : t -> int -> int -> char -> unit
(** [fill w off n c] stores [n] bytes [c] from byte [off] of [w]. *)

val barrier : unit -> unit
(** [barrier ()] orders every access before it before every access after it,
    as devices and other processors see them. On arm64 this is the full-system
    barrier, which stores to a BAR need. *)

val flush : t -> unit
(** [flush w] makes the stores made through [w] before it reach the function
    before any access made after it, through any window. It is a {!barrier},
    then a 32-bit load of [w]'s first word, which the bus answers only after
    the stores before it. Through a transport the load waits for the machine,
    so {!Machine.failed} read after it covers the stores before it.

    Raises [Invalid_argument] if [w] is shorter than 4 bytes or its first byte
    is not on a 4-byte boundary. *)

(** {1:transports Transports}

    This section is for the libraries that reach another machine. A transport
    is a [struct rig_pci_transport] of C functions that read and write the
    machine's addresses. [rig_pci.h] declares it and states the functions'
    contract. *)

type transport
(** The type for transports. *)

val unsafe_transport : int -> transport
(** [unsafe_transport p] is the transport whose [struct rig_pci_transport] is
    at address [p] of the process. Nothing checks [p]. It must point to such a
    structure, which must stay valid and unchanged while a window or a machine
    uses it. *)

val through : transport -> int -> int -> t
(** [through tr a n] is the [n] bytes at [a] of the machine [tr] reaches.

    Raises [Invalid_argument] if [n < 0] or [tr] is [unsafe_transport 0]. *)

(**/**)

val v : ?combines:bool -> int -> int -> t
(* [v a n] is the [n] bytes mapped at [a] in the process, for the library's own
   mappings, which combine if [combines] (defaults to [false]). Raises
   [Invalid_argument] if [n < 0]. *)
