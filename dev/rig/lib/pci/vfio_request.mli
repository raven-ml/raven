(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** VFIO's requests, as Linux's [<linux/vfio.h>] defines them: their numbers,
    their parameters laid out as the kernel reads them, and the readers of what
    it writes back. *)

type params =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
(** The type for a request's bytes, outside the OCaml heap, which the kernel
    reads and writes while the runtime is released. *)

(** {1:numbers Numbers} *)

val api_version : int
(** [api_version] is the version of VFIO's requests. *)

val config_region : int
(** [config_region] is the index of a function's region of configuration space.
*)

val type1v2_iommu : int
(** [type1v2_iommu] is the IOMMU model that translates through mappings. *)

val noiommu_iommu : int
(** [noiommu_iommu] is VFIO's no-IOMMU model, which does not translate. *)

(** The requests' numbers: [get_api_version] is VFIO_GET_API_VERSION, and so on.
*)

val get_api_version : int
val check_extension : int
val set_iommu : int
val group_get_status : int
val group_set_container : int
val group_get_device_fd : int
val device_get_region_info : int
val device_set_irqs : int
val device_reset : int
val iommu_get_info : int
val iommu_map_dma : int
val iommu_unmap_dma : int

(** {1:params Parameters} *)

val group_status : unit -> params
(** [group_status ()] asks for a group's status. *)

val viable : params -> bool
(** [viable p] is [true] iff the {!group_status} [p] says the group is viable.
*)

val int : int -> params
(** [int n] is the C int [n], as a request that takes a pointer to a descriptor
    reads it. *)

val name : string -> params
(** [name s] is [s] with a NUL byte after it. *)

val region_info : int -> size:int -> params
(** [region_info i ~size] asks for region [i] of a function, in [size] bytes, or
    the region's own if fewer: more holds its capabilities. *)

val needs : params -> int
(** [needs p] is the bytes the kernel's answer to [p] needs, its capabilities
    included. *)

val region : params -> int * int * bool * (int * int) list option
(** [region p] is the region the {!region_info} [p] answered: its size, its
    offset in the function's file, whether it maps, and [Some areas] if only
    those [(offset, bytes)] parts of it map. *)

val msi : int -> params
(** [msi efd] routes a function's first MSI vector to the eventfd [efd]. *)

val iommu_info : size:int -> params
(** [iommu_info ~size] asks for what a container's IOMMU maps, in [size] bytes,
    or the answer's own if fewer: more holds its capabilities. *)

val iommu : params -> int * (int * int) list
(** [iommu p] is what the {!iommu_info} [p] answered: the page sizes the IOMMU
    maps as a bitmap, 0 if it does not say, and the [(first, last)] device
    address ranges it maps, [[]] if it does not say. *)

val dma_map : va:int -> iova:int -> bytes:int -> params
(** [dma_map ~va ~iova ~bytes] maps, readable and writable, the [bytes] bytes at
    [va] of the process at the device address [iova]. *)

val dma_unmap : iova:int -> bytes:int -> params
(** [dma_unmap ~iova ~bytes] unmaps the [bytes] bytes at the device address
    [iova]. *)

(** Every 64-bit value a reader answers is saturated at [max_int]: a range may
    end at 2{^ 64}-1. *)
