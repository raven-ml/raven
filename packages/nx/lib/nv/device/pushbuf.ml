(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The runtime's own work on the GPU's channels: channel setup, local memory and
   copies, as methods in command segments of a ring of its own, which the
   channels' GPFIFOs point to. *)

module P = Nx_nv_packet
module M = P.Methods
module Mmio = Nx_device_support.Mmio

(* Methods *)

let words = P.dwords
let acquire addr v = words (M.acquire addr v)
let release addr v = words (M.release addr v)

(* The copy engine writes 32 bits per semaphore, so the 64-bit [v] goes as its
   low word, then its high word when the low one wrapped to 0: mid-write, the
   word reads no higher than before, so no wait passes early. A high word
   written late carries the value every later writer has too, so it never takes
   the word back, as rewriting it on every release would once another channel's
   release lands between the two words. *)
let copy_release addr v =
  let high =
    if v land 0xffff_ffff = 0 then M.copy_release (addr + 4) (v lsr 32) else []
  in
  words (M.copy_release addr v @ high)

(* Channels *)

type channel = {
  ring : Mmio.t;
  entries : int;
  gp_get : Mmio.t;
  gp_put : Mmio.t;
  put : Mmio.t; (* the 64-bit count of entries written *)
  doorbell : Mmio.t;
  token : int;
}

(* Appends the segment of [words] words at [addr] to [ch], after waiting until
   the GPU has fetched enough of its entries to make room, for at most
   [timeout_ms]. *)
let submit ch ~timeout_ms addr words =
  if addr >= 1 lsl 40 then failwith "a command segment above 2^40";
  let put = Int64.to_int (Mmio.get64 ch.put 0) in
  let unfetched () =
    (put - Mmio.get32 ch.gp_get 0 + ch.entries) mod ch.entries
  in
  Nvdev.wait_until ~timeout_ms "room in the GPU channel" (fun () ->
      unfetched () < ch.entries - 1);
  Mmio.set64 ch.ring
    (8 * (put mod ch.entries))
    (P.eval (P.Gpfifo.entry addr ~offset:0 ~words));
  Mmio.barrier ();
  Mmio.set64 ch.put 0 (Int64.of_int (put + 1));
  Mmio.set32 ch.gp_put 0 ((put + 1) mod ch.entries);
  Mmio.barrier ();
  Mmio.set32 ch.doorbell 0 ch.token

(* The runtime's segments *)

(* A segment: its bytes [first, last) and the timeline value of its work. *)
type segment = { first : int; last : int; value : int }

type ring = {
  mem : Mmio.t; (* as the host writes it *)
  gpu : int; (* its address for the GPU *)
  mutable head : int;
  segments : segment Queue.t; (* those whose bytes no later one took *)
}

let ring mem ~gpu = { mem; gpu; head = 0; segments = Queue.create () }

(* Writes [words] into a new segment for the work of timeline value [v], and is
   its address. Segments are taken in order around the ring, so the oldest are
   first in [segments]: those the new one overlaps, whose work it waits for, as
   [signaled] reads (values complete in order, so waiting for the latest is
   enough), and, when the ring wraps, those past its head, which a segment of
   the lap after them overlaps first. *)
let segment r ~signaled ~timeout_ms v words =
  let n = 4 * List.length words in
  if n > Mmio.length r.mem then
    failwith "a command segment larger than the ring";
  let first_is p =
    (not (Queue.is_empty r.segments)) && p (Queue.peek r.segments)
  in
  if r.head + n > Mmio.length r.mem then begin
    while first_is (fun s -> s.first >= r.head) do
      ignore (Queue.pop r.segments)
    done;
    r.head <- 0
  end;
  let start = r.head and stop = r.head + n in
  let latest = ref 0 in
  while first_is (fun s -> s.first < stop && start < s.last) do
    latest := Int.max !latest (Queue.pop r.segments).value
  done;
  Nvdev.wait_until ~timeout_ms "a command segment the GPU still reads"
    (fun () -> signaled () >= !latest);
  Queue.push { first = start; last = stop; value = v } r.segments;
  List.iteri (fun i w -> Mmio.set32 r.mem (start + (4 * i)) w) words;
  r.head <- stop;
  r.gpu + start
