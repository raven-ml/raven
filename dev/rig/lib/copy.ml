(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Copies between buffers: by the host, by a device's copy queue, through an io
   device's reads and writes, or through the host's staging memory. *)

open Def

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

external memmove : int -> int -> int -> unit = "caml_rig_memmove"

let fn = "Buffer.copy"
let local m = m.dev.machine = None && m.host >= 0
let host_address b = b.mem.host + b.offset

let record src dst bytes start =
  if Prof.enabled () then
    Prof.record (Copy { src; dst; bytes; start; stop = Prof.now () })

(* The host's staging memory: two slots, made at the first copy that needs them,
   each copied through by one copy at a time. A slot whose stamps name a lost
   device is replaced, so a loss reaches no other copy. *)
let slot_bytes = 64 * 1024 * 1024
let slots = [| None; None |]
let slots_lock = Lock.create ()
let in_use = [| false; false |]

let names_lost b =
  let lost = ref false in
  Memory.iter_points
    (fun p -> if Dev.is_lost (Dev.of_index (Point.index p)) then lost := true)
    (Memory.stamps b.mem);
  !lost

let take_slot () =
  Lock.protect slots_lock (fun () ->
      let rec free () =
        if not in_use.(0) then 0
        else if not in_use.(1) then 1
        else (
          Lock.wait slots_lock;
          free ())
      in
      let i = free () in
      in_use.(i) <- true;
      i)

let give_slot i =
  Lock.protect slots_lock (fun () ->
      in_use.(i) <- false;
      Lock.broadcast slots_lock)

let slot i =
  match slots.(i) with
  | Some b when not (names_lost b) -> b
  | _ ->
      let b = Buffer.create Dev.host slot_bytes in
      slots.(i) <- Some b;
      b

(* Whether [m] is a staging slot's memory: a copy through it that no device runs
   goes no further. *)
let is_slot m =
  Array.exists (function Some b -> b.mem.root == m.root | None -> false) slots

(* A copy on [d]'s copy queue between buffers [d] maps, waited for. *)
let on_queue d src dst =
  let queue = Array.find_opt (String.starts_with ~prefix:"COPY:") d.queues in
  match (queue, Memory.borrow d src.mem, Memory.borrow d dst.mem) with
  | Some queue, Some s, Some t ->
      let src = { src with mem = s } and dst = { dst with mem = t } in
      let part =
        { Submission.queue; after = [||]; work = Submission.Copy { src; dst } }
      in
      let s = Submission.make ~reads:0 ~writes:0 ~waits:0 d [| part |] in
      let p = Submission.submit s in
      Dev.wait d (Point.value p);
      true
  | _ -> false

let io_read src dst n =
  match src.mem.entry.io_region with
  | Some (Io_region { m; h; r }) ->
      let module I = (val m) in
      Dev.counted src.mem.dev (fun () ->
          I.read h r ~at:src.offset ~dst:(host_address dst) ~len:n)
  | None -> invalid_argf "Rig.%s: the source is no io memory" fn

let io_write src dst n =
  match dst.mem.entry.io_region with
  | Some (Io_region { m; h; r }) ->
      let module I = (val m) in
      Dev.counted dst.mem.dev (fun () ->
          I.write h r ~at:dst.offset ~src:(host_address src) ~len:n)
  | None -> invalid_argf "Rig.%s: the destination is no io memory" fn

let rec copy ~src ~dst =
  Buffer.check_live fn src;
  Buffer.check_live fn dst;
  let n = Buffer.length src in
  if n <> Buffer.length dst then
    invalid_argf "Rig.%s: %d bytes into %d" fn n (Buffer.length dst);
  if Buffer.overlaps src dst then invalid_argf "Rig.%s: the buffers overlap" fn;
  let sd = src.mem.dev and dd = dst.mem.dev in
  if Dev.is_lost sd then Dev.raise_lost sd;
  if Dev.is_lost dd then Dev.raise_lost dd;
  Memory.drain sd;
  if dd != sd then Memory.drain dd;
  if n > 0 then begin
    let start = Prof.now () in
    route src dst n;
    (* A read, a write or a move hands over addresses only: the buffers stay
       reachable until it returned, or a collection could free their memory
       under it. *)
    ignore (Sys.opaque_identity src);
    ignore (Sys.opaque_identity dst);
    record sd dd n start
  end

and route src dst n =
  let sd = src.mem.dev and dd = dst.mem.dev in
  if local src.mem && local dst.mem then begin
    Buffer.wait src Buffer.Read;
    Buffer.wait dst Buffer.Read_write;
    memmove (host_address dst) (host_address src) n
  end
  else if Dev.is_io sd && local dst.mem then begin
    Buffer.wait src Buffer.Read;
    Buffer.wait dst Buffer.Read_write;
    io_read src dst n
  end
  else if Dev.is_io dd && local src.mem then begin
    Buffer.wait src Buffer.Read;
    Buffer.wait dst Buffer.Read_write;
    io_write src dst n
  end
  else
    let runner = if local src.mem then dd else sd in
    let copied = (not (Dev.is_io runner)) && on_queue runner src dst in
    if copied then ()
    else if is_slot src.mem || is_slot dst.mem then
      invalid_argf "Rig.%s: no device copies between %s and %s" fn sd.name
        dd.name
    else staged src dst n

(* Through the host's staging memory, a slot at a time. *)
and staged src dst n =
  let i = take_slot () in
  Fun.protect
    ~finally:(fun () -> give_slot i)
    (fun () ->
      let rec go at =
        if at < n then begin
          let len = Int.min slot_bytes (n - at) in
          let s = Buffer.view (slot i) ~first:0 ~length:len in
          let piece b = Buffer.view b ~first:at ~length:len in
          copy ~src:(piece src) ~dst:s;
          copy ~src:s ~dst:(piece dst);
          go (at + len)
        end
      in
      go 0)
