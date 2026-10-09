(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The values and layouts are Linux's, from include/uapi/linux/vfio.h, which the
   64-bit Linux ABIs lay out alike. They are written here by hand, where the
   other drivers generate theirs from excerpts of their headers: vfio.h is under
   GPL-2.0 WITH Linux-syscall-note only, and only these numbers are taken from
   it. test_vfio_request checks them against the header where it is at hand. *)

type params =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external get16 : params -> int -> int = "%caml_bigstring_get16"
external get32 : params -> int -> int32 = "%caml_bigstring_get32"
external get64 : params -> int -> int64 = "%caml_bigstring_get64"
external set32 : params -> int -> int32 -> unit = "%caml_bigstring_set32"
external set64 : params -> int -> int64 -> unit = "%caml_bigstring_set64"

let params n =
  let p = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n in
  Bigarray.Array1.fill p '\000';
  p

let u32 p at = Int32.to_int (get32 p at) land 0xffff_ffff
let set_u32 p at v = set32 p at (Int32.of_int v)
let set_u64 p at v = set64 p at (Int64.of_int v)

(* A 64-bit field saturated at the largest int. *)
let u64 p at =
  let x = get64 p at in
  if Int64.unsigned_compare x (Int64.of_int max_int) > 0 then max_int
  else Int64.to_int x

(* Numbers *)

let api_version = 0
let config_region = 7 (* VFIO_PCI_CONFIG_REGION_INDEX *)
let msi_irq = 1 (* VFIO_PCI_MSI_IRQ_INDEX *)
let type1v2_iommu = 3
let noiommu_iommu = 8

(* _IO(VFIO_TYPE, VFIO_BASE + n): VFIO's requests carry no size. *)
let request n = (Char.code ';' lsl 8) lor (100 + n)
let get_api_version = request 0
let check_extension = request 1
let set_iommu = request 2
let group_get_status = request 3
let group_set_container = request 4
let group_get_device_fd = request 6
let device_get_region_info = request 8
let device_set_irqs = request 10
let device_reset = request 11
let iommu_get_info = request 12
let iommu_map_dma = request 13
let iommu_unmap_dma = request 14

(* Every request's parameters start with argsz, their bytes, then flags. *)
let argsz = 0
let flags = 4

(* Groups *)

let group_status_bytes = 8
let group_flags_viable = 1

let group_status () =
  let p = params group_status_bytes in
  set_u32 p argsz group_status_bytes;
  p

let viable p = u32 p flags land group_flags_viable <> 0

let int n =
  let p = params 4 in
  set_u32 p 0 n;
  p

let name s =
  let p = params (String.length s + 1) in
  String.iteri (Bigarray.Array1.set p) s;
  p

(* Capabilities: a chain of headers, each its id (16 bits), its version (16
   bits) and the offset of the next (32 bits, 0 at the end). Offsets only grow,
   so a walk ends. *)
let cap_header_bytes = 8

let capability p ~first id =
  let size = Bigarray.Array1.dim p in
  let rec walk off =
    if off = 0 || off + cap_header_bytes > size then None
    else if get16 p off = id then Some off
    else
      let next = u32 p (off + 4) in
      if next <= off then None else walk next
  in
  walk first

(* The [n] pairs of 64-bit values from [at]. *)
let pairs p at n =
  List.init n (fun i -> (u64 p (at + (16 * i)), u64 p (at + (16 * i) + 8)))

(* Regions *)

(* struct vfio_region_info: argsz, flags, index, cap_offset (32 bits each), size
   and offset (64 bits each). *)
let region_info_bytes = 32
let region_index = 8
let region_cap_offset = 12
let region_size = 16
let region_offset = 24
let region_flag_mmap = 1 lsl 2
let region_flag_caps = 1 lsl 3

(* struct vfio_region_info_cap_sparse_mmap: the header, nr_areas, a reserved
   word, then areas of offset and size. *)
let cap_sparse_mmap = 1
let sparse_nr_areas = 8
let sparse_areas = 16

let region_info i ~size =
  let p = params (Int.max size region_info_bytes) in
  set_u32 p argsz (Bigarray.Array1.dim p);
  set_u32 p region_index i;
  p

let needs p = u32 p argsz

let region p =
  let f = u32 p flags in
  let sparse =
    if f land region_flag_caps = 0 then None
    else capability p ~first:(u32 p region_cap_offset) cap_sparse_mmap
  in
  let areas =
    Option.map
      (fun at -> pairs p (at + sparse_areas) (u32 p (at + sparse_nr_areas)))
      sparse
  in
  (u64 p region_size, u64 p region_offset, f land region_flag_mmap <> 0, areas)

(* Interrupts *)

(* struct vfio_irq_set: argsz, flags, index, start, count, then the data, here
   one eventfd. *)
let irq_set_bytes = 24
let irq_index = 8
let irq_start = 12
let irq_count = 16
let irq_data = 20
let irq_data_eventfd = 1 lsl 2
let irq_action_trigger = 1 lsl 5

let msi efd =
  let p = params irq_set_bytes in
  set_u32 p argsz irq_set_bytes;
  set_u32 p flags (irq_data_eventfd lor irq_action_trigger);
  set_u32 p irq_index msi_irq;
  set_u32 p irq_start 0;
  set_u32 p irq_count 1;
  set_u32 p irq_data efd;
  p

(* IOMMUs *)

(* struct vfio_iommu_type1_info: argsz, flags, iova_pgsizes (64 bits),
   cap_offset and a pad. *)
let iommu_info_bytes = 24
let iommu_pgsizes = 8
let iommu_cap_offset = 16
let iommu_info_pgsizes = 1
let iommu_info_caps = 1 lsl 1

(* struct vfio_iommu_type1_info_cap_iova_range: the header, nr_iovas, a reserved
   word, then ranges of start and end. *)
let cap_iova_range = 1
let range_nr_iovas = 8
let range_iovas = 16

let iommu_info ~size =
  let p = params (Int.max size iommu_info_bytes) in
  set_u32 p argsz (Bigarray.Array1.dim p);
  p

let iommu p =
  let f = u32 p flags in
  let page_sizes =
    if f land iommu_info_pgsizes = 0 then 0 else u64 p iommu_pgsizes
  in
  let ranges =
    if f land iommu_info_caps = 0 then None
    else capability p ~first:(u32 p iommu_cap_offset) cap_iova_range
  in
  let ranges =
    match ranges with
    | None -> []
    | Some at -> pairs p (at + range_iovas) (u32 p (at + range_nr_iovas))
  in
  (page_sizes, ranges)

(* struct vfio_iommu_type1_dma_map: argsz, flags, then vaddr, iova and size (64
   bits each); struct vfio_iommu_type1_dma_unmap: argsz, flags, iova and
   size. *)
let dma_map_bytes = 32
let dma_map_vaddr = 8
let dma_map_iova = 16
let dma_map_size = 24
let dma_unmap_bytes = 24
let dma_unmap_iova = 8
let dma_unmap_size = 16
let dma_map_flag_read = 1
let dma_map_flag_write = 1 lsl 1

let dma_map ~va ~iova ~bytes =
  let p = params dma_map_bytes in
  set_u32 p argsz dma_map_bytes;
  set_u32 p flags (dma_map_flag_read lor dma_map_flag_write);
  set_u64 p dma_map_vaddr va;
  set_u64 p dma_map_iova iova;
  set_u64 p dma_map_size bytes;
  p

let dma_unmap ~iova ~bytes =
  let p = params dma_unmap_bytes in
  set_u32 p argsz dma_unmap_bytes;
  set_u64 p dma_unmap_iova iova;
  set_u64 p dma_unmap_size bytes;
  p
