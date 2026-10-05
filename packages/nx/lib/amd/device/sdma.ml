(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The runtime's own work on an SDMA queue, copies and timestamps: wait for the
   device's previous timeline value, work, signal the next value, and
   interrupt. *)

module P = Nx_amd_packet
module Mmio = Nx_device_support.Mmio

type queue = {
  ring : Mmio.t;
  read_ptr : Mmio.t; (* 8 bytes, written back by the engine *)
  write_ptr : Mmio.t; (* 8 bytes, polled by the engine *)
  put : Mmio.t; (* 8 bytes *)
  doorbell : Mmio.t; (* 8 bytes *)
}

let lo32 v = v land 0xffff_ffff
let hi32 v = (v lsr 32) land 0xffff_ffff

(* The words of the work of timeline value [v] on an engine of version [sdma],
   whose signal word is at [signal]: a wait for [v - 1], [body], the signal of
   [v], and an interrupt. *)
let work ~sdma ~signal v body =
  (* Values complete in order and only [v]'s work writes [v], so the signal word
     is at most [v - 1] when the engine reaches this: equality of the low words
     is exact, and never wraps. *)
  let poll = P.Sdma.poll signal Equal (v - 1) ~mask:0xffff_ffff in
  (* The high word changes only when the low one wraps to 0, and is then written
     after it: mid-write, the word reads no higher than before, so no wait
     passes early. A high word written late carries the value every later writer
     has too, so it never takes the word back. *)
  let high =
    if lo32 v = 0 then P.Sdma.fence ~sdma (signal + 4) (hi32 v) else []
  in
  P.dwords (poll @ body @ P.Sdma.fence ~sdma signal v @ high @ P.Sdma.trap)

(* A copy of [n] bytes from [src] to [dst] as timeline work of value [v]. *)
let packets ~sdma ~signal ~dst ~src n v =
  work ~sdma ~signal v (P.Sdma.copy ~sdma ~dst ~src n)

(* The global timestamp, 100 MHz ticks, written into the second word of the 16
   bytes at [slot] as timeline work of value [v]. *)
let stamp ~sdma ~signal ~slot v =
  work ~sdma ~signal v (P.Sdma.timestamp (slot + 8))

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
