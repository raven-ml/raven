(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* VFIO's requests (device_pci_vfio.c), over descriptors as ints. A refused
   request raises Unix.Unix_error with its errno; without Linux, Failure. *)

(* The IOMMU models, as the C side numbers them. *)
type model = Type1v2 | No_iommu

let model = function Type1v2 -> 0 | No_iommu -> 1

external open_ : string -> int = "caml_device_pci_vfio_open"
external supports : int -> int -> bool = "caml_device_pci_vfio_supports"
external viable : int -> bool = "caml_device_pci_vfio_viable"

external set_container : int -> int -> unit
  = "caml_device_pci_vfio_set_container"

external set_iommu : int -> int -> unit = "caml_device_pci_vfio_set_iommu"
external device : int -> string -> int = "caml_device_pci_vfio_device"

external region :
  int -> int -> bool -> int * int * bool * (int * int) list option
  = "caml_device_pci_vfio_region"

external msi : int -> int -> unit = "caml_device_pci_vfio_msi"
external reset : int -> unit = "caml_device_pci_vfio_reset"

external iommu : int -> int * (int * int) list * int option
  = "caml_device_pci_vfio_iommu"

external map : int -> int -> int -> int -> unit = "caml_device_pci_vfio_map"
external unmap : int -> int -> int -> unit = "caml_device_pci_vfio_unmap"
external eventfd : unit -> int = "caml_device_pci_eventfd"
external wait : int -> int -> bool = "caml_device_pci_wait"
