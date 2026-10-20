(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A machine's PCI functions, through its [sys/bus/pci].

    Reading changes nothing. A write the process may not make raises
    {!Fail.Failed}, naming the file and the privilege it needs. *)

type t
(** The type for a machine's files. *)

val v : ?kernel:(string -> string -> unit) -> string -> t
(** [v root] is the machine whose files are under the directory [root]: [/] for
    this machine, a fixture's directory in tests. [kernel file s] is called
    at each write of [s] to an attribute [file] of a driver, the bus or a
    function other than its [config], once [file] is open and before [s] is
    written, to act on it as Linux does inside the write: a fixture's kernel.
    It may raise [Unix.Unix_error] as Linux's refusal of the write, and [s]
    is then not written. The open truncated [file], so a fixture's kernel
    that refuses restores [file] itself. It defaults to doing nothing, as for
    this machine, whose kernel acts itself. *)

(** {1:locks Locks}

    A process takes or changes a function only while it holds the lock on the
    function's configuration file, and reads the function's binding, access and
    BARs only then. The lock's holder alone changes them, so what it read holds
    until it acts. *)

type 's lock
(** The type for the lock on a function's configuration file, held while ['s]
    is in scope. *)

val locked :
  t -> string -> ('s. 's lock -> ('a, string) result) -> ('a, string) result
(** [locked m bus f] is [f l] while the process holds the lock [l] on the
    function at [bus], released once [f] returns or raises. It is [Error why]
    without calling [f] if another open file holds the lock, in this process or
    another, or if the configuration file cannot be opened. *)

val files : 's lock -> t
(** [files l] is the files of the function [l] locks. *)

val bus : 's lock -> string
(** [bus l] is the bus address of the function [l] locks. *)

val keep : 's lock -> Unix.file_descr
(** [keep l] is a new descriptor that shares [l], so the lock outlives
    {!locked} until the descriptor is closed. A physical take keeps its
    function's lock this way until its release. *)

(** {1:identity Functions} *)

val exists : t -> string -> bool
(** [exists m bus] is [true] iff [m] has a function at [bus]. *)

val functions : t -> (Ops.id list, string) result
(** [functions m] is [m]'s functions, [Ok []] without [sys/bus/pci], or
    [Error why] if its directory cannot be read. *)

val vfio_pci : string
(** [vfio_pci] is VFIO's driver, ["vfio-pci"]. *)

val driver : 's lock -> string option
(** [driver l] is the kernel driver bound to the function [l] locks. *)

val group : t -> string -> string option
(** [group m bus] is the IOMMU group of the function at [bus]. *)

val bars : 's lock -> (int * int) option array
(** [bars l] is, for each of the six BARs of the function [l] locks, its bus
    address, from its register, and its size, or [None] where there is no BAR,
    such as at the upper index of a 64-bit BAR. Raises {!Fail.Failed} if a file
    cannot be read. *)

val root : t -> string
(** [root m] is the directory {!v} took. *)

val path : t -> string -> string -> string
(** [path m bus file] is the file [file] of the function at [bus]. *)

val vfio_file : t -> string -> string
(** [vfio_file m name] is VFIO's file [name] of [m], under [dev/vfio]. *)

(** {1:access How a function is taken} *)

val access : 's lock -> (Ops.addressing, string) result
(** [access l] is how the function [l] locks is taken, or why it cannot be.
    - It is [Iommu] for a function bound to [vfio-pci] behind an IOMMU, with
      the other functions of its group.
    - It is [Physical] for a function bound to [vfio-pci] without an IOMMU, or
      unbound and enabled under no IOMMU or an identity one, with the kernel
      not locked down. Either way it must be alone on its device if Linux
      resets it through its bus: the first method of its [reset_method] is
      [bus] or [cxl_bus], or the kernel lists none and it has a [reset].
    - It is [Error why] otherwise, naming the binding to [vfio-pci] where that
      would do, or {!detach} where it would. *)

val noiommu_file : t -> string -> string
(** [noiommu_file m g] is the file of group [g] in VFIO's no-IOMMU mode. *)

val group_holders : t -> string -> (string * string) list
(** [group_holders m g] is the functions of IOMMU group [g] that a driver VFIO
    refuses holds, with that driver. *)

(** {1:changes Changes} *)

val keep_off : 's lock -> remove:bool -> unit
(** [keep_off l ~remove] keeps kernel drivers off the function [l] locks,
    unless it is bound to [vfio-pci], by setting its [driver_override] to no
    driver. Behind an IOMMU that translates its addresses, it sets the
    function's [driver_override] to [vfio-pci] instead, if it is not on
    [vfio-pci]. Raises {!Fail.Failed}, having written nothing, if the
    function would not be takeable once unbound and enabled, alone if
    [remove], or once on [vfio-pci] behind such an IOMMU, or if a file cannot
    be written. Without [remove] it raises for a function whose device has
    other functions and that Linux resets through its bus, naming each with
    its driver. *)

val detach : 's lock -> remove:bool -> unit
(** [detach l ~remove] makes the function [l] locks takeable, after
    {!keep_off} with the same [remove]. It unbinds its driver, unless that is
    [vfio-pci], removes the other functions of its device if [remove] and,
    once unbound, enables it and turns its bus mastering off. Behind an IOMMU
    that translates, it sets the [driver_override] of the other functions of
    the device to [vfio-pci] if [remove] and unbinds their drivers, then the
    function's, which {!bind} then binds. Raises {!Fail.Failed} if the
    function is still not takeable, or a file cannot be written. *)

val bind : 's lock -> remove:bool -> unit
(** [bind l ~remove] binds to [vfio-pci], behind an IOMMU that translates, the
    function [l] locks and, if [remove], the other functions of its device,
    each not on [vfio-pci] yet: it sets each one's [driver_override] to
    [vfio-pci] and has Linux probe it. Elsewhere it does nothing. Raises
    {!Fail.Failed} if [vfio-pci] does not take one, saying where its reason
    is and how to proceed, or if a file cannot be written. *)

val attach : 's lock -> siblings:bool -> unit
(** [attach l ~siblings] gives the function [l] locks back to its kernel
    driver. It unbinds [vfio-pci], disables the function, rescans the bus,
    clears its [driver_override] and probes drivers. A function bound to
    another driver stays bound, and the bus is rescanned and its
    [driver_override] cleared if that is set. With [siblings], each other
    function of its device that [vfio-pci] holds, or whose [driver_override]
    names it, has its [driver_override] set to [vfio-pci] and is unbound
    from it first, and has its [driver_override] cleared and its drivers
    probed last. Raises {!Fail.Failed} if [vfio-pci] stays bound to such
    another one, or if no driver takes the function or such another one. *)

val reset : t -> string -> unit
(** [reset m bus] resets [bus] with the reset Linux has for it. *)

val resize : 's lock -> int -> unit
(** [resize l i] makes BAR [i] of the unbound function the largest size it
    supports that its bridge takes. A bound function keeps its size, and so
    does one whose resize the kernel refuses for another reason than room or
    permission, so its BAR may stay small. Raises {!Fail.Failed}, naming the
    file, if the process may not write it. *)

(** {1:peers Peer requests} *)

type route
(** The type for the way from a function up to its machine's root complex, as
    Linux's p2pdma reads it to decide whether the bus carries peer requests. *)

val route : t -> string -> route option
(** [route m bus] is the way from the function at [bus] up to its host bridge,
    read from the function's directory under [/sys/devices], each device's
    ACS controls in its [config], the host bridge's root bus and
    [proc/cpuinfo]. A device whose ACS controls cannot be read, as by a
    process without [CAP_SYS_ADMIN], counts as redirecting peer requests. It
    is [None] if the files cannot be read. *)

val forwards : provider:route -> client:route -> bool
(** [forwards ~provider ~client] is [true] iff the bus carries the requests
    of [client]'s function to a BAR of [provider]'s, as {!Memory.reaches}
    states Linux's p2pdma decides. *)

(** {1:open Open devices} *)

val held : t -> string -> string list -> string option
(** [held m bus nodes] is the file of a character device that this process
    holds open or mapped and that serves the function at [bus]. Such a device's
    number is given by a [dev] file under the function's directory, as for a
    DRM node, or is the number of a file of [nodes], paths from [m]'s root.
    Descriptors are read from [m]'s [proc/self/fd], and mapped files from
    [proc/self/map_files]. Raises {!Fail.Failed} if [proc/self/fd] is missing,
    or if it, [map_files], a directory of the function or a [dev] file cannot
    be read. *)

val drm_clients : t -> string -> (int * string) list
(** [drm_clients m bus] is the files of the DRM device of the function at
    [bus], each by the id and command of the process that opened it, read from
    debugfs's [dri/BUS/clients] under [m]'s root. It is [[]] if the function
    has no DRM device. A file stays listed until its last reference goes: its
    descriptors closed, its mappings gone and the references the kernel took to
    it dropped. A file opened in another PID namespace is listed with id [0].
    Raises {!Fail.Failed}, naming the file and the cause, if the function has a
    DRM device and the list cannot be read. *)

val held_elsewhere : t -> string -> string list -> (int * string) option
(** [held_elsewhere m bus nodes] is a process other than this one, by id, and
    the file of a device of {!held}'s that it holds open or mapped, read from
    [m]'s [proc/PID]: the processes of this PID namespace. A process gone since
    the listing is left out. Raises {!Fail.Failed} as {!held}, or if a process's
    [fd] or [map_files] cannot be read for another reason, naming it. *)

val command : t -> int -> string option
(** [command m pid] is the command of process [pid] of [m], from its
    [proc/PID/comm], or [None] if that cannot be read. *)

val refusal : 's lock -> remove:bool -> string option
(** [refusal l ~remove] is why a process could not take the function [l] locks
    once detached from its kernel driver with [remove], if it could not, such
    as an IOMMU group that holds other devices' functions bound to other
    drivers, or [vfio-pci] not loaded behind an IOMMU that translates. *)
