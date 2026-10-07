(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** This machine's PCI functions, through [/sys/bus/pci] (private).

    Reading changes nothing. A write the process may not make raises
    {!Fail.Failed} naming the file and the privilege it needs. *)

(** {1:identity Functions} *)

val exists : string -> bool
(** [exists bus] is [true] iff this machine has a function at [bus]. *)

val functions : unit -> Ops.id list
(** [functions ()] is this machine's functions, [[]] without [/sys/bus/pci]. *)

val driver : string -> string option
(** [driver bus] is the kernel driver bound to the function at [bus]. *)

val group : string -> string option
(** [group bus] is the IOMMU group of the function at [bus]. *)

val header : int
(** [header] is the bytes of configuration space every reader sees. *)

val bar : string -> int -> (int * int) option
(** [bar bus i] is BAR [i]'s bus address, from its register, and its size;
    [None] if there is no BAR [i], such as the upper index of a 64-bit BAR. *)

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
      [vfio-pci] where that would do, or {!detach} where it would. *)

val bind_vfio : string -> string
(** [bind_vfio bus] is the command that binds [bus] to [vfio-pci]. *)

val noiommu_file : string -> string
(** [noiommu_file g] is the file of group [g] in VFIO's no-IOMMU mode. *)

val group_holders : string -> (string * string) list
(** [group_holders g] is the functions of IOMMU group [g] that a driver VFIO
    refuses holds, with that driver. *)

(** {1:changes Changes} *)

val detach : string -> unit
(** [detach bus] makes [bus] takeable, unless {!access} takes it: it unbinds its
    driver unless that is [vfio-pci], removes its siblings and, unbound, enables
    it. Raises {!Fail.Failed} if it is still not takeable. *)

val attach : string -> unit
(** [attach bus] gives [bus] back to its kernel driver. Raises {!Fail.Failed} if
    it is bound to [vfio-pci] or no driver takes it. *)

val reset : string -> unit
(** [reset bus] resets [bus] with the reset Linux has for it. *)

val resize : string -> int -> unit
(** [resize bus i] makes BAR [i] of the unbound function the largest size it
    supports that its bridge takes. A bound function keeps its size, and so does
    one whose resize the kernel refuses for another reason than room: its BAR
    may stay small. *)
