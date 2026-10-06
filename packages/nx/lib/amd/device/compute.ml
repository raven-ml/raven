(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The runtime's own work on a PM4 compute queue, launches: wait for the
   device's previous timeline value, invalidate the caches, run each kernel once
   the one before has completed, and signal the next value, as tolk's batches on
   the same queue do. *)

module P = Nx_amd_packet
module Code_object = Nx_amd_code_object
module Mmio = Nx_device_support.Mmio

(* A kernel run: its descriptor's fields, where its code and arguments are, and
   its grid. *)
type run = {
  kernel : Code_object.kernel;
  entry : int; (* the address of its first instruction *)
  args : int; (* the address of its arguments *)
  packet : int; (* the address of its dispatch packet, if it reads one *)
  scratch : int; (* the address of the device's scratch memory *)
  groups : int * int * int;
  threads : int * int * int;
}

(* A dword's bits. *)
let mask32 = 0xffff_ffff

(* The poll interval of a wait, as tolk sets it. *)
let wait_interval = 4

(* The packets of [r], on a GPU of graphics block [gc] whose scratch for kernels
   of [n] bytes per lane takes the COMPUTE_TMPRING_SIZE word [tmpring n], before
   the packets [rest]. *)
let run ~gc ~tmpring r rest =
  P.Pm4.acquire_mem ~gc Data_caches
  :: P.Pm4.dispatch ~gc r.kernel ~program:r.entry ~scratch:r.scratch
       ~packet:r.packet ~args:r.args
       ~tmpring:(tmpring r.kernel.private_segment)
       ~limits:0 ~threads:r.threads ~groups:r.groups
  :: P.Pm4.event_write Cs_partial_flush
  :: rest

(* The words of the work of timeline value [v] running [runs], whose signal word
   is at [signal], as a list of packets' words. Values complete in order and
   only [v]'s work writes [v], so the low word is at most [v - 1] when the queue
   reaches the wait: the wait for equality is exact. The value is written whole,
   in one 64-bit write. *)
let work ~gc ~tmpring ~signal v runs =
  P.Pm4.wait ~gc (Memory signal) Equal (v - 1) ~mask:0xffff_ffff
    ~interval:wait_interval
  :: P.Pm4.acquire_mem ~gc All_caches
  :: List.fold_right (run ~gc ~tmpring) runs
       [ P.Pm4.release_mem ~gc signal (Data_64 v) ]

(* Appends the words of [packets] to the PM4 queue [q] and rings its doorbell.
   Positions count dwords, and packets wrap around the ring's end. The caller
   has room for half the ring, which [Nx_device.submit] waits for. *)
let submit (q : Sdma.queue) packets =
  let ring = Mmio.length q.ring / 4 in
  let n =
    List.fold_left
      (List.fold_left (fun n -> function P.W64 _ -> n + 2 | _ -> n + 1))
      0 packets
  in
  if 2 * n > ring then
    invalid_arg "Nx_amd_device.launch: work larger than half the ring";
  let put = Int64.to_int (Mmio.get64 q.put 0) in
  let at = ref put in
  let emit w =
    Mmio.set32 q.ring (4 * (!at mod ring)) (w land mask32);
    incr at
  in
  List.iter
    (List.iter (function
      | P.Dword w -> emit w
      | P.W32 t -> emit (Int64.to_int (P.eval t))
      | P.W64 t ->
          let w = P.eval t in
          emit (Int64.to_int w);
          emit (Int64.to_int (Int64.shift_right_logical w 32))))
    packets;
  let next = Int64.of_int (put + n) in
  Mmio.barrier ();
  Mmio.set64 q.write_ptr 0 next;
  Mmio.set64 q.put 0 next;
  Mmio.barrier ();
  Mmio.set64 q.doorbell 0 next

(* Arguments *)

(* A ring of host memory the GPU reads arguments from, each launch's in a
   segment of its own. *)
(* A segment: its bytes [first, last) and the timeline value of its work. *)
type segment = { first : int; last : int; value : int }

type args = {
  mem : Mmio.t; (* as the host writes it *)
  gpu : int; (* its address for the GPU *)
  mutable head : int;
  segments : segment Queue.t; (* those whose bytes no later one took *)
}

let args mem ~gpu = { mem; gpu; head = 0; segments = Queue.create () }

(* The alignment of a segment, and of each kernel's arguments in it. *)
let segment_align = 256
let align n = (n + segment_align - 1) / segment_align * segment_align

(* The address of a new segment of [n] bytes for the work of timeline value [v].
   Segments are taken in order around the ring, so the oldest are first in
   [segments]: those the new one overlaps, which [wait] waits for (values
   complete in order, so waiting for the latest is enough), and, when the ring
   wraps, those past its head, which a segment of the lap after them overlaps
   first. *)
let segment r ~wait v n =
  let n = align n in
  if n > Mmio.length r.mem then
    invalid_arg "Nx_amd_device.launch: arguments larger than their ring";
  let first_is p =
    (not (Queue.is_empty r.segments)) && p (Queue.peek r.segments)
  in
  if r.head + n > Mmio.length r.mem then begin
    while first_is (fun s -> s.first >= r.head) do
      ignore (Queue.pop r.segments)
    done;
    r.head <- 0
  end;
  let first = r.head and last = r.head + n in
  let latest = ref 0 in
  while first_is (fun s -> s.first < last && first < s.last) do
    latest := Int.max !latest (Queue.pop r.segments).value
  done;
  if !latest > 0 then wait !latest;
  Queue.push { first; last; value = v } r.segments;
  r.head <- last;
  r.gpu + first

(* Writes [data] at the segment's address [at], and [n] zero bytes. *)
let write r ~at data = Mmio.write r.mem (at - r.gpu) data
let fill r ~at n = Mmio.fill r.mem (at - r.gpu) n '\000'
