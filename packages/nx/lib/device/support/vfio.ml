(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Requests *)

(* A VFIO request is _IO(';', 100 + k): no direction and no size, so its number
   is the type in the second byte and the request in the first. *)
let vfio_type = Char.code ';'
let vfio_base = 100

type request =
  | Get_api_version
  | Check_extension
  | Set_iommu
  | Group_get_status
  | Group_set_container
  | Group_get_device_fd
  | Device_get_region_info
  | Device_set_irqs
  | Device_reset
  | Iommu_get_info
  | Iommu_map_dma
  | Iommu_unmap_dma

let request r =
  let k =
    match r with
    | Get_api_version -> 0
    | Check_extension -> 1
    | Set_iommu -> 2
    | Group_get_status -> 3
    | Group_set_container -> 4
    | Group_get_device_fd -> 6
    | Device_get_region_info -> 8
    | Device_set_irqs -> 10
    | Device_reset -> 11
    | Iommu_get_info -> 12
    | Iommu_map_dma -> 13
    | Iommu_unmap_dma -> 14
  in
  (vfio_type lsl 8) lor (vfio_base + k)

let api_version = 0

type iommu = Type1v2 | No_iommu

let iommu = function Type1v2 -> 3 | No_iommu -> 8

(* Structures *)

let u32 b off = Int32.to_int (Bytes.get_int32_ne b off) land 0xffff_ffff

(* A 64-bit field, saturated at [max_int]: the last address of a range may be
   2^64 - 1. *)
let u64 b off =
  let v = Bytes.get_int64_ne b off in
  if Int64.compare v 0L < 0 || Int64.compare v (Int64.of_int max_int) > 0 then
    max_int
  else Int64.to_int v

let set_u32 b off v = Bytes.set_int32_ne b off (Int32.of_int v)
let set_u64 b off v = Bytes.set_int64_ne b off (Int64.of_int v)

(* A structure of [n] bytes whose first word is its size. *)
let structure n =
  let b = Bytes.make n '\000' in
  set_u32 b 0 n;
  b

let argsz b = u32 b 0

let need b n what =
  if Bytes.length b < n then
    failwith
      (Printf.sprintf "VFIO %s: %d bytes, %d expected" what (Bytes.length b) n)

(* The capabilities of [b] from [first]: (id, offset of the body) for each
   header { u16 id; u16 version; u32 next }, whose [next] is an offset in [b].
   Offsets only grow, so the chain ends. *)
let capabilities what b first =
  let rec go off acc =
    if off = 0 then List.rev acc
    else begin
      if off < 0 || off + 8 > Bytes.length b then
        failwith
          (Printf.sprintf "VFIO %s: a capability leaves the structure" what);
      let id = Bytes.get_uint16_ne b off and next = u32 b (off + 4) in
      if next <> 0 && next <= off then
        failwith (Printf.sprintf "VFIO %s: the capabilities loop" what);
      go next ((id, off + 8) :: acc)
    end
  in
  go first []

(* Groups: { u32 argsz; u32 flags } *)

let group_status () = structure 8
let flag_viable = 1

let viable b =
  need b 8 "group status";
  u32 b 4 land flag_viable <> 0

(* Regions: { u32 argsz; u32 flags; u32 index; u32 cap_offset; u64 size; u64
   offset } *)

let config_region = 7
let region_size = 32

type region = {
  size : int;
  offset : int;
  readable : bool;
  writable : bool;
  mappable : bool;
  areas : (int * int) list option;
}

let region_info ?(argsz = region_size) i =
  let b = structure (max argsz region_size) in
  set_u32 b 8 i;
  b

let region_read = 1
let region_write = 2
let region_mmap = 4
let region_caps = 8
let cap_sparse_mmap = 1

(* { u32 nr_areas; u32 reserved; { u64 offset; u64 size } areas[] } *)
let sparse b body =
  need b (body + 8) "sparse mapping";
  let n = u32 b body in
  need b (body + 8 + (16 * n)) "sparse mapping";
  List.init n (fun i ->
      let at = body + 8 + (16 * i) in
      (u64 b at, u64 b (at + 8)))

let region b =
  need b region_size "region";
  let flags = u32 b 4 in
  let caps =
    if flags land region_caps = 0 then []
    else capabilities "region" b (u32 b 12)
  in
  {
    size = u64 b 16;
    offset = u64 b 24;
    readable = flags land region_read <> 0;
    writable = flags land region_write <> 0;
    mappable = flags land region_mmap <> 0;
    areas =
      List.find_map
        (fun (id, body) ->
          if id = cap_sparse_mmap then Some (sparse b body) else None)
        caps;
  }

(* Interrupts: { u32 argsz; u32 flags; u32 index; u32 start; u32 count; u8
   data[] } *)

let irq_data_eventfd = 4
let irq_action_trigger = 32
let msi_index = 1

let msi fd =
  let b = structure (20 + 4) in
  set_u32 b 4 (irq_data_eventfd lor irq_action_trigger);
  set_u32 b 8 msi_index;
  set_u32 b 12 0;
  set_u32 b 16 1;
  set_u32 b 20 fd;
  b

(* Containers: { u32 argsz; u32 flags; u64 iova_pgsizes; u32 cap_offset; u32 pad
   } *)

type iommu_info = {
  page_sizes : int;
  ranges : (int * int) list;
  mappings : int option;
}

let iommu_size = 24
let info_pgsizes = 1
let info_caps = 2
let cap_iova_range = 1
let cap_dma_avail = 3
let iommu_info ?(argsz = iommu_size) () = structure (max argsz iommu_size)

(* { u32 nr_iovas; u32 reserved; { u64 start; u64 end } ranges[] } *)
let iova_ranges b body =
  need b (body + 8) "IOVA ranges";
  let n = u32 b body in
  need b (body + 8 + (16 * n)) "IOVA ranges";
  List.init n (fun i ->
      let at = body + 8 + (16 * i) in
      (u64 b at, u64 b (at + 8)))

let iommu_of b =
  need b iommu_size "IOMMU information";
  let flags = u32 b 4 in
  let caps =
    if flags land info_caps = 0 then [] else capabilities "IOMMU" b (u32 b 16)
  in
  let find id f =
    List.find_map (fun (i, body) -> if i = id then Some (f body) else None) caps
  in
  {
    page_sizes = (if flags land info_pgsizes = 0 then 0 else u64 b 8);
    ranges = Option.value ~default:[] (find cap_iova_range (iova_ranges b));
    mappings =
      find cap_dma_avail (fun body ->
          need b (body + 4) "DMA mappings";
          u32 b body);
  }

(* { u32 argsz; u32 flags; u64 vaddr; u64 iova; u64 size } *)
let dma_read = 1
let dma_write = 2

let map_dma ~va ~iova n =
  let b = structure 32 in
  set_u32 b 4 (dma_read lor dma_write);
  Bytes.set_int64_ne b 8 (Int64.of_nativeint va);
  set_u64 b 16 iova;
  set_u64 b 24 n;
  b

(* { u32 argsz; u32 flags; u64 iova; u64 size } *)
let unmap_dma ~iova n =
  let b = structure 24 in
  set_u64 b 8 iova;
  set_u64 b 16 n;
  b

(* Device addresses *)

module Iova = struct
  type t = { tlsf : Tlsf.t; page : int }

  (* Every GPU reaches 40 bits. An address a device truncates to 32 bits falls
     below 4 GiB, where nothing is mapped, and faults. *)
  let low = 1 lsl 32
  let high = 1 lsl 40
  let large = 2 lsl 20
  let round_up n a = (n + a - 1) / a * a
  let round_down n a = n / a * a

  let create ~page ranges =
    if page <= 0 || page land (page - 1) <> 0 then
      invalid_arg (Printf.sprintf "Vfio.Iova.create: a page of %d bytes" page);
    let ranges = if ranges = [] then [ (0, max_int) ] else ranges in
    let pieces =
      List.filter_map
        (fun (first, last) ->
          let a = round_up (min high (max first low)) page in
          let b =
            if last >= high - 1 then high else round_down (last + 1) page
          in
          if a < b then Some (a, b - a) else None)
        ranges
    in
    match
      List.fold_left
        (fun best (a, n) ->
          match best with Some (_, m) when m >= n -> best | _ -> Some (a, n))
        None pieces
    with
    | None ->
        failwith "the IOMMU maps no device addresses between 4 GiB and 1 TiB"
    | Some (base, n) -> { tlsf = Tlsf.create ~block:page ~base n; page }

  let window a = (Tlsf.base a.tlsf, Tlsf.length a.tlsf)

  let alloc a n =
    if n <= 0 then invalid_arg (Printf.sprintf "Vfio.Iova.alloc: %d bytes" n);
    let n = round_up n a.page in
    let align = if n >= large then large else a.page in
    Tlsf.alloc ~align a.tlsf n

  let free a x =
    try Tlsf.free a.tlsf x
    with Invalid_argument _ ->
      invalid_arg (Printf.sprintf "Vfio.Iova.free: no addresses at 0x%x" x)
end
