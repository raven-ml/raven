(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** PCI functions driven by the process itself.

    A runtime that drives a GPU without its kernel driver takes the GPU's PCI
    function, once {!detach} has detached it from its kernel driver: it reads
    and writes its configuration space and maps its BARs, the windows through
    which the process reaches the function's registers and memory. Taking a
    function changes nothing on the machine; {!detach} and {!attach} do, and
    what they change persists after the process.

    This works through Linux's [/sys/bus/pci] and, for interrupts, VFIO, and
    needs the privileges to write there: root, or file permissions and
    capabilities granted for it. The process takes nothing it lacks the rights
    for, and asks for none: a missing privilege raises [Failure] naming the file
    and the command that grants it. On other systems {!scan} finds nothing, and
    {!detach}, {!attach} and {!take} raise.

    A function may also be another machine's, taken through a {!Remote}
    connection to that machine's server, which does the same there. Its BARs and
    the system memory of its machine are then that machine's addresses, reached
    through the connection, and it has no interrupts. *)

(** {1:addresses Bus addresses}

    A function's bus address names it on its machine, as Linux spells it:
    ["DDDD:BB:DD.F"], its domain, bus, device and function numbers in lowercase
    hexadecimal, the domain in at least four digits. *)

val address : domain:int -> bus:int -> device:int -> fn:int -> string
(** [address ~domain ~bus ~device ~fn] is the bus address of function [fn] of
    device [device] on bus [bus] of domain [domain], such as ["0000:03:00.0"].
    Kernel drivers report their devices by these numbers. *)

val compare_address : string -> string -> int
(** [compare_address a b] orders bus addresses in bus order: by domain, then
    bus, device and function, each as a number.

    Raises [Invalid_argument] if [a] or [b] is no bus address. *)

(** {1:functions Functions} *)

type t
(** The type for PCI functions the process has taken. *)

val scan :
  ?remote:Remote.t ->
  vendor:int ->
  ?class_:int ->
  (int * int list) list ->
  string list
(** [scan ~vendor ids] is the bus addresses, in bus order ({!compare_address}),
    of the functions of [vendor] whose device id, masked by [m], is in [l] for
    some [(m, l)] of [ids], and whose base class is [class_] if given, on the
    machine of [remote] if given, on this one otherwise. It is [[]] where the
    system has no [/sys/bus/pci]. *)

val detach : string -> unit
(** [detach bus] detaches the function at [bus] of this machine, so that a
    process can take it ({!take}): it unbinds the function's kernel driver,
    unless that is [vfio-pci], removes the other functions of its device, such
    as its audio function, and enables it. Its kernel driver's users, a display
    among them, lose it. This persists after the process, until {!attach} or a
    reboot. It needs root, or write access to the files under [/sys/bus/pci] it
    writes.

    Raises [Failure] if [bus] is no function of this machine, if a process has
    it taken, if the process may not write a file, naming it, or if the driver
    stays bound. *)

val attach : string -> unit
(** [attach bus] gives the function at [bus] of this machine back to its kernel
    driver: Linux rescans the PCI bus, which brings back the functions {!detach}
    removed, and binds the function's driver. This persists after the process.
    It needs root, or write access to [/sys/bus/pci/rescan] and
    [/sys/bus/pci/drivers_probe].

    Raises [Failure] if [bus] is no function of this machine, if a process has
    it taken, if it is bound to [vfio-pci], or if no driver takes it, such as
    when the driver's module is not loaded. *)

val detached : string -> (unit, string) result
(** [detached bus] is [Ok ()] if the function at [bus] of this machine is as
    {!detach} leaves it: bound to no kernel driver but [vfio-pci], alone on its
    device, and enabled unless [vfio-pci] holds it. [Error why] says what
    differs, such as ["0000:01:00.0 is bound to the driver nvidia"]. *)

val take : ?remote:Remote.t -> lock:string -> string -> t
(** [take ~lock bus] takes the function at [bus], of the machine of [remote] if
    given, which must be {!detached} there: it locks it for this process through
    the files [nx_BUS.lock], which every process of this library takes, and
    [LOCK_BUS.lock], which every process driving such a GPU takes, in the
    temporary directory. A lock file that is a link or no regular file is
    refused. A function bound to [vfio-pci] in VFIO's no-IOMMU mode delivers its
    interrupts to {!wait_interrupt}. Taking it changes nothing on the machine.

    Raises [Failure] if the function is not {!detached}, saying why, if another
    process holds it, if a lock file cannot be opened, such as one another user
    created, or if the process may not access it, each naming what to change. *)

val bus : t -> string
(** [bus p] is [p]'s bus address on its machine. *)

val remote : t -> Remote.t option
(** [remote p] is the connection to [p]'s machine, if [p] is another machine's.
*)

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
(** [unmap_bar m] unmaps [m], which {!map_bar} mapped. Another machine's BARs
    stay mapped until their function is released. *)

val resize_bar : t -> int -> unit
(** [resize_bar p i] makes [p]'s BAR [i] as large as the function allows, so
    that it covers all of a GPU's memory where the platform permits. The size
    persists after the process.

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
    VFIO, or on another machine, it returns [false] at once. *)

val release : t -> unit
(** [release p] gives [p] back: it closes the process's files for it and unlocks
    it. Its BAR mappings stay, except on another machine, where they go with it.
*)

(** {1:sysmem System memory of the function's machine}

    {!Sysmem}'s functions, on the machine of the function: a GPU's system memory
    must be memory of the machine it is in. *)

val page : t -> int
(** [page p] is the page size of [p]'s machine. *)

val reserve : t -> base:int -> int -> unit
(** [reserve p ~base n] is {!Sysmem.reserve} on [p]'s machine. *)

val alloc_sysmem : t -> ?contiguous:bool -> ?va:int -> int -> Mmio.t * int list
(** [alloc_sysmem p ?contiguous ?va n] is {!Sysmem.alloc} on [p]'s machine. *)

val free_sysmem : t -> Mmio.t -> unit
(** [free_sysmem p m] is {!Sysmem.free} on [p]'s machine. *)

val pin : t -> nativeint -> int -> int list
(** [pin p a n] is {!Sysmem.pin} on [p]'s machine. *)

val unpin : t -> nativeint -> int -> unit
(** [unpin p a n] is {!Sysmem.unpin} on [p]'s machine. *)
