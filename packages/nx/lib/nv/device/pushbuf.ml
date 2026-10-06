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
  error : Mmio.t; (* the notification RM writes when it stops the channel *)
}

(* Appends the segment of [words] words at [addr] to [ch], after waiting with
   [wait] for the GPU to fetch enough of its entries to make room. *)
let submit ch ~wait addr words =
  if addr >= 1 lsl 40 then failwith "a command segment above 2^40";
  let put = Int64.to_int (Mmio.get64 ch.put 0) in
  let unfetched () =
    (put - Mmio.get32 ch.gp_get 0 + ch.entries) mod ch.entries
  in
  wait (fun () -> unfetched () < ch.entries - 1);
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
   first in [segments]: when the ring wraps, those past its head, which a
   segment of the lap after them overlaps first, then those the new one
   overlaps, whose work [wait] waits for (values complete in order, so waiting
   for the latest is enough). Nothing changes before the wait, which Ctrl-C may
   end. *)
let segment r ~wait v words =
  let n = 4 * List.length words in
  if n > Mmio.length r.mem then
    failwith "a command segment larger than the ring";
  let wraps = r.head + n > Mmio.length r.mem in
  let start = if wraps then 0 else r.head in
  let stop = start + n in
  let past_head s = wraps && s.first >= r.head in
  let overlaps s = s.first < stop && start < s.last in
  (* The segments the new one replaces, and the latest value of those it
     overlaps. *)
  let rec overlapped k latest seq =
    match seq () with
    | Seq.Cons (s, seq) when overlaps s ->
        overlapped (k + 1) (Int.max latest s.value) seq
    | Seq.Cons _ | Seq.Nil -> (k, latest)
  in
  let rec passed k seq =
    match seq () with
    | Seq.Cons (s, seq') when past_head s -> passed (k + 1) seq'
    | _ -> overlapped k 0 seq
  in
  let replaced, latest = passed 0 (Queue.to_seq r.segments) in
  if latest > 0 then wait latest;
  for _ = 1 to replaced do
    ignore (Queue.pop r.segments)
  done;
  Queue.push { first = start; last = stop; value = v } r.segments;
  List.iteri (fun i w -> Mmio.set32 r.mem (start + (4 * i)) w) words;
  r.head <- stop;
  r.gpu + start
