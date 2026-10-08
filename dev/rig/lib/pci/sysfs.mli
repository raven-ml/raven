(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A machine's PCI functions, through its [sys/bus/pci].

    Reading changes nothing. A write the process may not make raises
    {!Fail.Failed} naming the file and the privilege it needs. *)

type t
(** The type for a machine's files. *)

val v : string -> t
(** [v root] is the machine whose files are under the directory [root]: [/] for
    this machine, a fixture's directory in tests. *)

(** {1:identity Functions} *)

val exists : t -> string -> bool
(** [exists m bus] is [true] iff [m] has a function at [bus]. *)

val functions : t -> Ops.id list
(** [functions m] is [m]'s functions, [[]] without [sys/bus/pci]. *)

val vfio_pci : string
(** [vfio_pci] is VFIO's driver, ["vfio-pci"]. *)

val driver : t -> string -> string option
(** [driver m bus] is the kernel driver bound to the function at [bus]. *)

val group : t -> string -> string option
(** [group m bus] is the IOMMU group of the function at [bus]. *)

val bars : t -> string -> (int * int) option array
(** [bars m bus] is, for each of the six BARs of the function at [bus], its bus
    address, from its register, and its size; [None] where there is no BAR, such
    as at the upper index of a 64-bit BAR. Raises {!Fail.Failed} if a file
    cannot be read. *)

val root : t -> string
(** [root m] is the directory {!v} took. *)

val path : t -> string -> string -> string
(** [path m bus file] is the file [file] of the function at [bus]. *)

val vfio_file : t -> string -> string
(** [vfio_file m name] is VFIO's file [name] of [m], under [dev/vfio]. *)

(** {1:access How a function is taken} *)

val access : t -> string -> (Ops.addressing, string) result
(** [access m bus] is how the function at [bus] is taken, or why it cannot be:
    - bound to [vfio-pci] behind an IOMMU, siblings and all: [Iommu];
    - bound to [vfio-pci] without one, or unbound and enabled under none or an
      identity one, alone on its device and the kernel not locked down:
      [Physical];
    - otherwise [Error why], naming the binding to [vfio-pci] where that would
      do, or {!detach} where it would. *)

val noiommu_file : t -> string -> string
(** [noiommu_file m g] is the file of group [g] in VFIO's no-IOMMU mode. *)

val group_holders : t -> string -> (string * string) list
(** [group_holders m g] is the functions of IOMMU group [g] that a driver VFIO
    refuses holds, with that driver. *)

(** {1:changes Changes} *)

val detach : t -> string -> unit
(** [detach m bus] makes [bus] takeable and keeps kernel drivers off it, unless
    it is bound to [vfio-pci]: it sets its [driver_override] to no driver,
    unbinds its driver, removes its siblings and, unbound, enables it. Raises
    {!Fail.Failed}, having written nothing, if it would not be takeable once
    unbound, alone and enabled, and if it is still not takeable after. *)

val attach : t -> string -> unit
(** [attach m bus] gives [bus] back to its kernel driver, unbinding [vfio-pci]
    and clearing its [driver_override]. A function bound to another driver is
    left as it is. Raises {!Fail.Failed} if no driver takes it. *)

val reset : t -> string -> unit
(** [reset m bus] resets [bus] with the reset Linux has for it. *)

val resize : t -> string -> int -> unit
(** [resize m bus i] makes BAR [i] of the unbound function the largest size it
    supports that its bridge takes. A bound function keeps its size, and so does
    one whose resize the kernel refuses for another reason than room: its BAR
    may stay small. *)

(** {1:open Open devices} *)

val held : t -> string -> string list -> string option
(** [held m bus nodes] is the file of a character device this process holds open
    that serves the function at [bus]: one whose number a [dev] file under the
    function's directory gives, such as a DRM node's, or the number of a file of
    [nodes], paths from [m]'s root. Descriptors are read from [m]'s
    [proc/self/fd], and mapped files from [proc/self/map_files]. Raises
    {!Fail.Failed} if [proc/self/fd] is missing, or if it, [map_files], a
    directory of the function or a [dev] file cannot be read. *)

val drm_clients : t -> string -> (int * string) list
(** [drm_clients m bus] is the files of the DRM device of the function at [bus],
    each by the id and command of the process that opened it, from debugfs's
    [dri/BUS/clients] under [m]'s root: [[]] if the function has no DRM device.
    A file stays listed until its last reference goes, its descriptors closed,
    its mappings gone and the references the kernel took to it dropped; a file
    opened in another PID namespace is listed with id [0]. Raises {!Fail.Failed}
    if the function has a DRM device and the list cannot be read, naming the
    file and the cause. *)

val held_elsewhere : t -> string -> string list -> (int * string) option
(** [held_elsewhere m bus nodes] is a process other than this one, by id, and
    the file of a device of {!held}'s that it holds open or mapped, read from
    [m]'s [proc/PID]: the processes of this PID namespace. A process gone since
    the listing is left out. Raises {!Fail.Failed} as {!held}, or if a process's
    [fd] or [map_files] cannot be read for another reason, naming it. *)

val refusal : t -> string -> string option
(** [refusal m bus] is why a process could not take the function at [bus] once
    detached from its kernel driver, as an IOMMU translating its addresses
    without [vfio-pci], if it could not. *)
