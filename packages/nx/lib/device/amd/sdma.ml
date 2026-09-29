(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The runtime's own copies on an SDMA queue: wait for the device's previous
   timeline value, copy, signal the next value, and interrupt. *)

module D = Amd_defs
module Mmio = Nx_device_support.Mmio

type queue = {
  ring : Mmio.t;
  read_ptr : Mmio.t; (* 8 bytes, written back by the engine *)
  write_ptr : Mmio.t; (* 8 bytes, polled by the engine *)
  put : Mmio.t; (* 8 bytes *)
  doorbell : Mmio.t; (* 8 bytes *)
}

let field (mask, shift) v = (v land mask) lsl shift
let lo32 v = v land 0xffff_ffff
let hi32 v = (v lsr 32) land 0xffff_ffff

(* The packets of a copy of [n] bytes from [src] to [dst] as timeline work of
   value [v], whose signal word is at [signal]. Fences write uncached, on the
   engines that take a memory type. *)
let packets ~family ~max ~signal ~dst ~src n v =
  let module P = (val D.sdma family : D.SDMA) in
  (* Values complete in order and only [v]'s work writes [v], so the signal word
     is at most [v - 1] when the engine reaches this: equality of the low words
     is exact, and never wraps. *)
  let poll =
    [
      P.op_poll_regmem
      lor field P.poll_regmem_header_func 3
      lor field P.poll_regmem_header_mem_poll 1;
      lo32 signal;
      hi32 signal;
      lo32 (v - 1);
      0xffff_ffff;
      field P.poll_regmem_dw5_interval 0x04
      lor field P.poll_regmem_dw5_retry_count 0xfff;
    ]
  in
  let copies =
    List.concat
      (List.init
         ((n + max - 1) / max)
         (fun i ->
           let off = i * max in
           let len = Int.min max (n - off) in
           [
             P.op_copy lor field P.copy_linear_header_sub_op P.subop_copy_linear;
             len - 1;
             0;
             lo32 (src + off);
             hi32 (src + off);
             lo32 (dst + off);
             hi32 (dst + off);
           ]))
  in
  (* The high word changes only when the low one wraps to 0, and is then written
     after it: mid-write, the word reads no higher than before, so no wait
     passes early. A high word written late carries the value every later writer
     has too, so it never takes the word back. *)
  let fence addr data =
    let mtype =
      match P.fence_header_mtype with Some f -> field f 3 | None -> 0
    in
    [ P.op_fence lor mtype; lo32 addr; hi32 addr; data ]
  in
  let high = if lo32 v = 0 then fence (signal + 4) (hi32 v) else [] in
  poll @ copies @ fence signal (lo32 v) @ high @ [ P.op_trap; 0 ]

(* Appends [words] to [q]'s ring and rings its doorbell. Positions count bytes;
   packets never wrap, so a submission that does not fit before the ring's end
   zeroes the rest of it and starts at its beginning. Waits for the engine to
   leave room, for at most [timeout_ms]. *)
let submit q ~timeout_ms words =
  let ring = Mmio.length q.ring in
  let size = 4 * List.length words in
  if size >= ring then invalid_arg "Sdma.submit: larger than the ring";
  let put = Int64.to_int (Mmio.get64 q.put 0) in
  let tail = put mod ring in
  let zero = if size <= ring - tail then 0 else ring - tail in
  let need = zero + size in
  let start = Amdev.now_ms () in
  let rec room () =
    let used = (put - Int64.to_int (Mmio.get64 q.read_ptr 0)) land (ring - 1) in
    if ring - used <= need then begin
      if Amdev.now_ms () - start > timeout_ms then
        failwith "the SDMA ring stayed full";
      Domain.cpu_relax ();
      room ()
    end
  in
  room ();
  if zero > 0 then Mmio.fill q.ring tail zero '\000';
  let at = if zero > 0 then 0 else tail in
  List.iteri (fun i w -> Mmio.set32 q.ring (at + (4 * i)) w) words;
  let next = Int64.of_int (put + need) in
  Mmio.barrier ();
  Mmio.set64 q.write_ptr 0 next;
  Mmio.set64 q.put 0 next;
  Mmio.barrier ();
  Mmio.set64 q.doorbell 0 next
