(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Linux's VFIO interface, as data.

    VFIO hands a PCI function to a process through three files: a container,
    [/dev/vfio/vfio], which holds an IOMMU context; the function's IOMMU group,
    [/dev/vfio/N]; and the function itself, which the group opens. The process
    drives them with [ioctl] requests whose arguments are the structures below,
    in the machine's byte order. Behind an IOMMU, the function reaches system
    memory only at the device addresses (IOVAs) the process maps in its
    container.

    This module encodes and decodes those requests and structures, and allocates
    device addresses; {!Pci} makes the calls. *)

(** {1:requests Requests} *)

(** The type for the requests {!Pci} makes. *)
type request =
  | Get_api_version  (** Of the container: its API version. *)
  | Check_extension  (** Of the container: whether an IOMMU model is there. *)
  | Set_iommu  (** Of the container: the IOMMU model it uses. *)
  | Group_get_status  (** Of a group: {!group_status}. *)
  | Group_set_container  (** Of a group: adds it to a container. *)
  | Group_get_device_fd  (** Of a group: opens one of its functions. *)
  | Device_get_region_info  (** Of a function: {!region_info}. *)
  | Device_set_irqs  (** Of a function: routes its interrupts ({!msi}). *)
  | Device_reset  (** Of a function: resets it. *)
  | Iommu_get_info  (** Of the container: {!iommu_info}. *)
  | Iommu_map_dma  (** Of the container: {!map_dma}. *)
  | Iommu_unmap_dma  (** Of the container: {!unmap_dma}. *)

val request : request -> int
(** [request r] is the [ioctl] request number of [r]. *)

val api_version : int
(** [api_version] is the API version {!Get_api_version} answers. *)

(** The type for IOMMU models. *)
type iommu =
  | Type1v2
      (** The IOMMU translates the function's addresses, through mappings the
          process makes and removes whole. *)
  | No_iommu
      (** No IOMMU: the function reaches physical addresses. Linux offers it
          only when its [enable_unsafe_noiommu_mode] parameter is set. *)

val iommu : iommu -> int
(** [iommu m] is [m]'s number, for {!Check_extension} and {!Set_iommu}. *)

(** {1:structures Structures} *)

val argsz : bytes -> int
(** [argsz b] is the size in bytes the kernel needs for the structure [b], whose
    first word gives it. A structure with capabilities is asked for again with
    that size when it exceeds [Bytes.length b]. *)

(** {2:groups Groups} *)

val group_status : unit -> bytes
(** [group_status ()] is the argument of {!Group_get_status}. *)

val viable : bytes -> bool
(** [viable b] is [true] iff the group status [b] says every function of the
    group is held by VFIO or by no driver, so that the group can be used. *)

(** {2:regions Regions} *)

val config_region : int
(** [config_region] is the index of a PCI function's configuration space among
    its regions. Its BARs are regions [0] to [5]. *)

type region = {
  size : int;  (** Its size in bytes. *)
  offset : int;  (** Where it starts in the function's file. *)
  readable : bool;
  writable : bool;
  mappable : bool;  (** Whether the process may map it. *)
  areas : (int * int) list option;
      (** The (offset, bytes) parts of it the process may map, when only those
          may be. *)
}
(** The type for regions of a function. *)

val region_info : ?argsz:int -> int -> bytes
(** [region_info i] is the argument of {!Device_get_region_info} for region [i],
    of [argsz] bytes (defaults to the structure's size). *)

val region : bytes -> region
(** [region b] is the region the answer [b] to {!Device_get_region_info}
    describes.

    Raises [Failure] if [b] is too short or its capability chain leaves it. *)

(** {2:interrupts Interrupts} *)

val msi : int -> bytes
(** [msi fd] is the argument of {!Device_set_irqs} that makes the function's
    first MSI vector signal the eventfd [fd]. *)

(** {2:containers Containers} *)

type iommu_info = {
  page_sizes : int;  (** The page sizes it maps, a bit per size. *)
  ranges : (int * int) list;
      (** The device addresses it maps, as (first, last) inclusive ranges, in
          order: [[]] if the kernel does not say, and every address is. *)
  mappings : int option;  (** How many more mappings it allows, if it says. *)
}
(** The type for what a container's IOMMU maps. *)

val iommu_info : ?argsz:int -> unit -> bytes
(** [iommu_info ()] is the argument of {!Iommu_get_info}, of [argsz] bytes
    (defaults to the structure's size). *)

val iommu_of : bytes -> iommu_info
(** [iommu_of b] is what the answer [b] to {!Iommu_get_info} describes.

    Raises [Failure] if [b] is too short or its capability chain leaves it. *)

val map_dma : va:nativeint -> iova:int -> int -> bytes
(** [map_dma ~va ~iova n] is the argument of {!Iommu_map_dma} that maps the [n]
    bytes of the process at [va] for reading and writing at the device address
    [iova]. *)

val unmap_dma : iova:int -> int -> bytes
(** [unmap_dma ~iova n] is the argument of {!Iommu_unmap_dma} that removes the
    mapping of [n] bytes at [iova]. *)

(** {1:iova Device addresses} *)

(** Device addresses of a container.

    A function behind an IOMMU reaches the system memory a process maps for it
    at addresses the process chooses. They are taken from 4 GiB up to 1 TiB,
    which every GPU reaches and where an address truncated to 32 bits points
    nowhere, from the largest part of that range the IOMMU maps. Memory of 2 MiB
    or more gets an address on 2 MiB, so that a GPU may map it with large pages;
    other memory, an address on a page. Not synchronized. *)
module Iova : sig
  type t
  (** The type for the device addresses of a container. *)

  val create : page:int -> (int * int) list -> t
  (** [create ~page ranges] is the device addresses of a container whose IOMMU
      maps pages of [page] bytes at the (first, last) inclusive [ranges], every
      address if [ranges] is [[]].

      Raises [Failure] if no page from 4 GiB up to 1 TiB is in [ranges], and
      [Invalid_argument] if [page] is not a positive power of two. *)

  val window : t -> int * int
  (** [window a] is the first address and the size of the range [a] takes
      addresses from. *)

  val alloc : t -> int -> int option
  (** [alloc a n] is the first of [n] free device addresses, rounded up to a
      page, or [None] if they do not fit. They fit while a free range of
      [2 * (n + align)] addresses is left, [align] being their alignment.

      Raises [Invalid_argument] if [n <= 0]. *)

  val free : t -> int -> unit
  (** [free a x] frees the addresses {!alloc} returned at [x].

      Raises [Invalid_argument] if none were. *)
end
