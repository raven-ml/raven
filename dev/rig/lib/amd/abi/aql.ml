(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Packet
module P = Defs.Dispatch

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* Every packet waits for the ones before it and is coherent across the system
   before and after it. *)
let header =
  (1 lsl Defs.hsa_packet_header_barrier)
  lor Defs.hsa_fence_scope_system
      lsl Defs.hsa_packet_header_scacquire_fence_scope
  lor Defs.hsa_fence_scope_system
      lsl Defs.hsa_packet_header_screlease_fence_scope

(* The words of the [P.sizeof] bytes [b], each [(offset, word)] of [holes] in
   place of the bytes it covers. *)
let words b holes =
  let rec go off =
    if off >= P.sizeof then []
    else
      match List.assoc_opt off holes with
      | Some (W64 _ as w) -> w :: go (off + 8)
      | Some w -> w :: go (off + 4)
      | None ->
          Dword (Int32.to_int (Bytes.get_int32_le b off) land 0xffff_ffff)
          :: go (off + 4)
  in
  go 0

let set b (off, width) v =
  match width with
  | 2 -> Bytes.set_uint16_le b off v
  | _ -> Bytes.set_int32_le b off (Int32.of_int v)

(* A dispatch's three dimensions, and the largest workgroup side, a 16-bit
   field. *)
let dimensions = 3
let max_threads = 0xffff

let dispatch (k : Code_object.kernel) ~descriptor ~args ~threads:(tx, ty, tz)
    ~grid:(gx, gy, gz) =
  let check t =
    if t < 1 || t > max_threads then
      invalid_argf
        "Aql.dispatch: %d threads in a dimension, expected 1 to 65535" t
  in
  check tx;
  check ty;
  check tz;
  let b = Bytes.make P.sizeof '\000' in
  set b P.header
    (header
    lor (Defs.hsa_packet_type_kernel_dispatch lsl Defs.hsa_packet_header_type));
  set b P.setup (dimensions lsl Defs.hsa_kernel_dispatch_packet_setup_dimensions);
  set b P.workgroup_size_x tx;
  set b P.workgroup_size_y ty;
  set b P.workgroup_size_z tz;
  set b P.private_segment_size k.private_segment;
  set b P.group_segment_size k.group_segment;
  words b
    [
      (fst P.grid_size_x, W32 (Value gx));
      (fst P.grid_size_y, W32 (Value gy));
      (fst P.grid_size_z, W32 (Value gz));
      (fst P.kernel_object, W64 (Value descriptor));
      (fst P.kernarg_address, W64 (Value args));
    ]

type field = Workgroup_size of Code_object.axis | Group_segment_size

let offset = function
  | Workgroup_size X -> fst P.workgroup_size_x
  | Workgroup_size Y -> fst P.workgroup_size_y
  | Workgroup_size Z -> fst P.workgroup_size_z
  | Group_segment_size -> fst P.group_segment_size

(* The vendor packet of PM4 commands, amd_aql_pm4_ib in ROCR-Runtime's
   amd_aql_queue.cpp:1521-1547 (rocm-systems cccc350d): its format,
   AMD_AQL_FORMAT_PM4_IB, in the vendor header that follows the 16-bit AQL
   header, and dw_cnt_remain, the words left after its four of PM4. *)
let format_pm4_ib = 1
let vendor_header_shift = 16
let dw_count_remain = 10

(* IB_SIZE is 20 bits; bit 20 is CHAIN. *)

let indirect_buffer addr ~dwords =
  let hdr =
    header
    lor (Defs.hsa_packet_type_vendor_specific lsl Defs.hsa_packet_header_type)
    lor (format_pm4_ib lsl vendor_header_shift)
  in
  (Dword hdr :: Pm4.indirect_buffer addr ~dwords)
  @ (Dword dw_count_remain :: List.init dw_count_remain (fun _ -> Dword 0))
