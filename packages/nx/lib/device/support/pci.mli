(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** PCI functions driven by the process itself.

    A runtime that drives a GPU without its kernel driver takes the GPU's PCI
    function: it detaches the function from its kernel driver, then reads and
    writes its configuration space and maps its BARs, the windows through which
    the process reaches the function's registers and memory.

    This works through Linux's [/sys/bus/pci] and, for interrupts, VFIO, and
    needs the privileges to write there: root, or file permissions and
    capabilities granted for it. The process takes nothing it lacks the rights
    for, and asks for none: a missing privilege raises [Failure] naming the file
    and the command that grants it. On other systems {!scan} finds nothing and
    {!take} raises. *)

type t
(** The type for PCI functions the process has taken. *)

val scan : vendor:int -> ?class_:int -> (int * int list) list -> string list
(** [scan ~vendor ids] is the bus addresses, such as ["0000:03:00.0"] and in
    their order, of the functions of [vendor] whose device id, masked by [m], is
    in [l] for some [(m, l)] of [ids], and whose base class is [class_] if
    given. It is [[]] where the system has no [/sys/bus/pci]. *)

val take : lock:string -> string -> t
(** [take ~lock bus] takes the function at [bus]: it locks it for this process
    through the file [LOCK_BUS.lock] of the temporary directory, which every
    process driving such a GPU takes, removes the other functions of its device,
    such as its audio function, and enables it. A function bound to [vfio-pci]
    in VFIO's no-IOMMU mode stays bound, and delivers its interrupts to
    {!wait_interrupt}; any other kernel driver is detached.

    Raises [Failure] if another process holds the function, if the process may
    not detach or enable it, or if a driver stays bound to it, each naming what
    to change. *)

val bus : t -> string
(** [bus p] is [p]'s bus address. *)

val read_config : t -> int -> int -> int
(** [read_config p off n] is the [n]-byte little-endian value at byte [off] of
    [p]'s configuration space, [n] being 1, 2 or 4. *)

val write_config : t -> int -> int -> int -> unit
(** [write_config p off n v] writes the low [n] bytes of [v] at byte [off] of
    [p]'s configuration space, then reads them back so the write is complete
    when it returns. *)

val bar : t -> int -> int * int
(** [bar p i] is the bus address and size in bytes of [p]'s BAR [i].

    Raises [Failure] if [p] has no BAR [i]. *)

val map_bar : ?offset:int -> ?length:int -> t -> int -> Mmio.t
(** [map_bar p i] maps [length] bytes (defaults to the rest of the BAR) of [p]'s
    BAR [i] from byte [offset] (defaults to [0]) into the process, for the life
    of the process. Child processes do not inherit the mapping. *)

val unmap_bar : Mmio.t -> unit
(** [unmap_bar m] unmaps [m], which {!map_bar} mapped. *)

val resize_bar : t -> int -> unit
(** [resize_bar p i] makes [p]'s BAR [i] as large as the function allows, so
    that it covers all of a GPU's memory where the platform permits.

    Raises [Failure] if the system refuses. *)

val reset : t -> unit
(** [reset p] resets [p] with the reset Linux has for it, and waits, for at most
    a second, until it answers again: it clears the state a previous driver left
    in it.

    Raises [Failure] naming the file if the process may not reset it or Linux
    has no reset for it, and if [p] does not answer in time. *)

val wait_interrupt : t -> int -> bool
(** [wait_interrupt p ms] waits at most [ms] milliseconds for an interrupt of
    [p], releasing the OCaml runtime, and is [true] iff one arrived. Without
    VFIO it returns [false] at once. *)

val release : t -> unit
(** [release p] gives [p] back: it closes the process's files for it and unlocks
    it. Its BAR mappings stay. *)
