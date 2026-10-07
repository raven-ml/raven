(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Packet

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* A field's value, from its (mask, shift). *)
let field (mask, shift) v = (v land mask) lsl shift

(* The bytes one linear copy moves: its count is 30 bits wide from SDMA 4.4.2
   below 5 and from 5.2, 22 bits otherwise. *)
let max_copy (g : Gpu.t) =
  let v = g.sdma in
  if
    (compare (4, 4, 2) v <= 0 && compare v (5, 0, 0) < 0)
    || compare v (5, 2, 0) >= 0
  then 1 lsl 30
  else 1 lsl 22

let copy_header =
  Defs.sdma_op_copy
  lor field Defs.sdma_pkt_copy_linear_header_sub_op Defs.sdma_subop_copy_linear

let copy g ~dst ~src n =
  if n < 0 then invalid_argf "Sdma.copy: %d bytes, expected 0 or more" n;
  let max = max_copy g in
  let piece i =
    let off = i * max in
    let at v = if off = 0 then Value v else Add (Value v, Int64.of_int off) in
    [
      Dword copy_header;
      Dword (Int.min max (n - off) - 1);
      Dword 0;
      W64 (at src);
      W64 (at dst);
    ]
  in
  List.concat (List.init ((n + max - 1) / max) piece)

(* The interval and retries of a poll, as the kernel driver sets them. *)
let interval = 0x04
let retries = 0xfff

(* SDMA compares as PM4's WAIT_REG_MEM does. *)
let function_of = function
  | Equal -> Defs.packet3_wait_reg_mem__function__equal_to_the_reference_value
  | Greater_equal ->
      Defs.packet3_wait_reg_mem__function__greater_than_or_equal_reference_value

let poll addr cmp v ?(mask = 0xffff_ffff) () =
  [
    Dword
      (Defs.sdma_op_poll_regmem
      lor field Defs.sdma_pkt_poll_regmem_header_func (function_of cmp)
      lor field Defs.sdma_pkt_poll_regmem_header_mem_poll 1);
    W64 (Value addr);
    W32 (Value v);
    Dword mask;
    Dword
      (field Defs.sdma_pkt_poll_regmem_dw5_interval interval
      lor field Defs.sdma_pkt_poll_regmem_dw5_retry_count retries);
  ]

(* The uncached memory type, MTYPE_UC. *)
let mtype_uc = 3

let fence (g : Gpu.t) addr v =
  let mtype =
    if compare g.sdma Defs.sdma_fence_mtype_from >= 0 then
      field Defs.sdma_pkt_fence_header_mtype mtype_uc
    else 0
  in
  [ Dword (Defs.sdma_op_fence lor mtype); W64 (Value addr); W32 (Value v) ]

let trap = [ Dword Defs.sdma_op_trap; Dword 0 ]

let timestamp addr =
  [
    Dword
      (Defs.sdma_op_timestamp
      lor field Defs.sdma_pkt_timestamp_get_global_header_sub_op
            Defs.sdma_subop_timestamp_get_global);
    W64 (Value addr);
  ]
