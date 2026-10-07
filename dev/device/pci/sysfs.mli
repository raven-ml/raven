(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** This machine's PCI functions, through [/sys/bus/pci] (private).

    Reads what a function is and how it is held, decides how the process takes
    it, and makes the changes to the machine that {!Gpus} offers. Reading
    changes nothing. A write the process may not make raises [Failure] naming
    the file and the privilege it needs. *)

(** {1:identity Functions} *)

val exists : string -> bool
(** [exists bus] is [true] iff this machine has a function at [bus]. *)

val functions : unit -> Ops.id list
(** [functions ()] is this machine's functions in bus order, [[]] without
    [/sys/bus/pci]. *)

val driver : string -> string option
(** [driver bus] is the kernel driver bound to the function at [bus]. *)

val group : string -> string option
(** [group bus] is the IOMMU group of the function at [bus]. *)

val bar : string -> int -> (int * int) option
(** [bar bus i] is the bus address of BAR [i] of the function at [bus], from its
    BAR register, and its size, from the kernel's resources. [None] if it has no
    BAR [i], such as the upper index of a 64-bit BAR. *)

val path : string -> string -> string
(** [path bus file] is the file [file] of the function at [bus]. *)

(** {1:access How a function is taken} *)

(** The type for what an IOMMU does with a function's DMA while no process holds
    it. *)
type iommu =
  | No_iommu  (** There is none, or VFIO's no-IOMMU mode stands in for one. *)
  | Identity  (** It passes physical addresses through. *)
  | Translating  (** It translates them. *)

type state = {
  driver : string option;
  iommu : iommu;
  siblings : string list;  (** The other functions of its device. *)
  enabled : bool;
  locked_down : bool;  (** Whether the kernel refuses mappings of BARs. *)
}
(** The type for the state of a function. *)

val state : string -> state
(** [state bus] is the state of the function at [bus]. *)

val access : string -> state -> (Ops.addressing, string) result
(** [access bus s] is how a function at [bus] in state [s] is taken, or why it
    cannot be:
    - bound to [vfio-pci] behind an IOMMU, siblings and all: [Iommu];
    - bound to [vfio-pci] without one, or unbound and enabled under none or an
      identity one, alone on its device and the kernel not locked down:
      [Physical];
    - otherwise [Error why], naming the [driverctl] command that binds it to
      [vfio-pci] where that would do. *)

val bind_vfio : string -> string
(** [bind_vfio bus] is the command that binds the function at [bus] to
    [vfio-pci]. *)

(** {1:changes Changes} *)

val detach : string -> unit
(** [detach bus] makes the function at [bus] takeable, unless {!access} already
    takes it: it unbinds its driver unless that is [vfio-pci], removes its
    siblings and, unbound, enables it.

    Raises [Failure] if a write is refused or the function is still not
    takeable, saying why. *)

val attach : string -> unit
(** [attach bus] gives the function at [bus] back to its kernel driver.

    Raises [Failure] if it is bound to [vfio-pci], if a write is refused, or if
    no driver takes it. *)

val reset : string -> unit
(** [reset bus] resets the function at [bus] with the reset Linux has for it.

    Raises [Failure] if the write is refused. *)

val resize : string -> int -> unit
(** [resize bus i] makes BAR [i] of the unbound function at [bus] the largest
    size it supports that its bridge takes, trying sizes from the largest down.
    It does nothing where the function lists no sizes or is bound to a driver,
    which keeps its BAR's size. *)
