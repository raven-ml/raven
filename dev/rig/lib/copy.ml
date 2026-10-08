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
let local m = Option.is_none m.dev.machine && m.host >= 0
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

(* A copy on [d]'s copy queue, its point. *)
let submit_copy d queue ~src ~dst =
  let part =
    { Submission.queue; after = [||]; work = Submission.Copy { src; dst } }
  in
  let s = Submission.make ~reads:0 ~writes:0 ~waits:0 d [| part |] in
  Submission.submit s

let queued d queue ~src ~dst =
  Dev.wait d (Point.value (submit_copy d queue ~src ~dst))

(* A copy on [d]'s copy queue between buffers [d] maps, waited for when
   [wait]. *)
let on_queue ~wait d src dst =
  match (d.copy_queue, Memory.borrow d src.mem, Memory.borrow d dst.mem) with
  | Some queue, Some s, Some t ->
      let src = { src with mem = s } and dst = { dst with mem = t } in
      if wait then queued d queue ~src ~dst
      else ignore (submit_copy d queue ~src ~dst);
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

(* Copies [n] bytes from [src] to [dst] in one transfer, where one runs it: the
   host between memory it addresses or with an io device's, or a device's copy
   queue between memory it maps. A copy on a queue returns at once unless
   [wait]: what reads or writes its buffers later waits for it by their stamps.
   Whether it ran. *)
let direct ~wait src dst n =
  let sd = src.mem.dev and dd = dst.mem.dev in
  if local src.mem && local dst.mem then begin
    Buffer.wait src Buffer.Read;
    Buffer.wait dst Buffer.Read_write;
    memmove (host_address dst) (host_address src) n;
    true
  end
  else if Dev.is_io sd && local dst.mem then begin
    Buffer.wait src Buffer.Read;
    Buffer.wait dst Buffer.Read_write;
    io_read src dst n;
    true
  end
  else if Dev.is_io dd && local src.mem then begin
    Buffer.wait src Buffer.Read;
    Buffer.wait dst Buffer.Read_write;
    io_write src dst n;
    true
  end
  else
    let runner = if local src.mem then dd else sd in
    (not (Dev.is_io runner)) && on_queue ~wait runner src dst

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
    route ~wait:true src dst n;
    (* A read, a write or a move hands over addresses only: the buffers stay
       reachable until it returned, or a collection could free their memory
       under it. *)
    ignore (Sys.opaque_identity src);
    ignore (Sys.opaque_identity dst)
  end

(* Copies [n] bytes from [src] to [dst], directly or through the staging memory,
   recording each transfer that ran. *)
and route ~wait src dst n =
  let sd = src.mem.dev and dd = dst.mem.dev in
  let start = Prof.now () in
  if direct ~wait src dst n then record sd dd n start
  else if is_slot src.mem || is_slot dst.mem then
    invalid_argf "Rig.%s: no device copies between %s and %s" fn sd.name dd.name
  else staged src dst n

(* Through the host's staging memory: the copy holds a slot, whose two halves
   take the pieces in turn, so one half fills while the other drains. A leg on a
   device's queue runs without the host waiting for it: the next leg that reads
   or writes its half waits for it by the half's stamps. The device's leg into
   the slot runs one piece ahead of the host's out of it; the host's leg into
   the slot fills one half while the device drains the other. *)
and staged src dst n =
  let i = take_slot () in
  let half = slot_bytes / 2 in
  let pieces = (n + half - 1) / half in
  let len k = Int.min half (n - (k * half)) in
  let run slot =
    let into k =
      leg
        ~src:(Buffer.view src ~first:(k * half) ~length:(len k))
        ~dst:(Buffer.view slot ~first:(k land 1 * half) ~length:(len k))
    in
    let out_of k =
      leg
        ~src:(Buffer.view slot ~first:(k land 1 * half) ~length:(len k))
        ~dst:(Buffer.view dst ~first:(k * half) ~length:(len k))
    in
    let ahead = not (local src.mem || Dev.is_io src.mem.dev) in
    if ahead then into 0;
    for k = 0 to pieces - 1 do
      if ahead then (if k + 1 < pieces then into (k + 1)) else into k;
      out_of k
    done;
    Buffer.wait dst Buffer.Read_write
  in
  (* No leg outlives the copy: the slot is given back unused, also when making
     its memory raised. *)
  let settle () =
    Option.iter (fun b -> Buffer.wait b Buffer.Read_write) slots.(i)
  in
  Fun.protect
    ~finally:(fun () ->
      (try settle () with Dev.Lost _ -> ());
      give_slot i)
    (fun () -> run (slot i))

(* One leg of a staged copy, which a device's queue runs without the host
   waiting. *)
and leg ~src ~dst = route ~wait:false src dst (Buffer.length src)
