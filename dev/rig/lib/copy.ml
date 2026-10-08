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

(* A driver's device of another machine, which no staging memory reaches. *)
let far d = Option.is_some d.machine && not (Dev.is_io d)

(* Records the transfer of [bytes] from [src]'s memory to [dst]'s asked at
   [start], which the host sees done now. *)
let record src dst bytes start =
  if Prof.enabled () then
    Prof.record (Copy { src; dst; bytes; start; stop = Prof.now () })

(* The host's staging memory: two slots, made at the first copy that needs them,
   each copied through by one copy at a time. A slot is two halves, each a
   memory of its own: stamps order the uses of a whole memory, so halves of one
   memory would order every leg after the other half's last one. A half whose
   stamps name a lost device's work that is not done is replaced, so a loss
   reaches no other copy. *)
let half_bytes = 32 * 1024 * 1024
let slots = [| [| None; None |]; [| None; None |] |]
let slots_lock = Lock.create ()
let in_use = [| false; false |]

let names_lost b =
  match Memory.check_points (Memory.stamps b.mem) with
  | () -> false
  | exception Dev.Lost _ -> true

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

let half i h =
  match slots.(i).(h) with
  | Some b when not (names_lost b) -> b
  | _ ->
      let b = Buffer.create Dev.host half_bytes in
      slots.(i).(h) <- Some b;
      b

(* Maps the staging half [h] on [d], which runs legs through it. A map [d]'s
   driver refuses is memory [d] cannot give: the out-of-memory ladder reclaims
   for it, then raises [Out_of_memory]. *)
let map_half d h =
  match Memory.borrow d h.mem with
  | Some _ -> ()
  | None ->
      ignore
        (Memory.reclaiming d ~pool:d half_bytes (fun () ->
             Memory.borrow d h.mem))

(* Whether [m] is a staging slot's memory: a copy through it that no device runs
   goes no further. *)
let is_slot m =
  Array.exists
    (Array.exists (function Some b -> b.mem.root == m.root | None -> false))
    slots

let queued d queue ~src ~dst =
  Dev.wait d (Point.value (Submission.copy ~hold_stamps:0 d queue ~src ~dst))

(* The stamps of the hold a copy between [src] and [dst] names, [0] for none:
   held memory's stamps are its hold's. [None] for memory of two holds, which no
   one submission names: the copy stages, each leg naming one hold's memory. *)
let hold_stamps src dst =
  let e = src.mem.root.entry and e' = dst.mem.root.entry in
  match (e.held, e'.held) with
  | true, true when e.stamps <> e'.stamps -> None
  | true, _ -> Some e.stamps
  | false, true -> Some e'.stamps
  | false, false -> Some 0

(* [m] as [d]'s copy queue takes it: a device of another machine takes this
   process's memory as it is, its driver carrying the bytes; another device
   maps it. A function of its own, so a copy builds no closure. *)
let mine d m =
  if Option.is_some d.machine && local m then Some m else Memory.borrow d m

(* A copy of [n] bytes on [d]'s copy queue between buffers [d] maps, asked at
   [start] and waited for when [wait]. Unwaited, it is recorded once a wait
   sees it done. *)
let on_queue ~wait d src dst n start =
  match (d.copy_queue, hold_stamps src dst, mine d src.mem, mine d dst.mem) with
  | Some queue, Some hold_stamps, Some s, Some t ->
      let v =
        Point.value
          (Submission.copy ~hold_stamps d queue ~src:{ src with mem = s }
             ~dst:{ dst with mem = t })
      in
      if wait then begin
        Dev.wait d v;
        record src.mem.dev dst.mem.dev n start
      end
      else if Prof.enabled () then
        Dev.after d v (fun () -> record src.mem.dev dst.mem.dev n start);
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

(* How the host moves bytes: between memory it addresses, or by an io device's
   read or write. A constant, so a host copy allocates nothing. *)
type transfer = Move | Io_read | Io_write

(* The host's copy of [n] bytes from [src] to [dst], after their devices' work,
   asked at [start]. *)
let on_host transfer src dst n start =
  Buffer.wait_points src Buffer.Read;
  Buffer.wait_points dst Buffer.Read_write;
  (match transfer with
  | Move -> memmove (host_address dst) (host_address src) n
  | Io_read -> io_read src dst n
  | Io_write -> io_write src dst n);
  record src.mem.dev dst.mem.dev n start;
  true

(* Copies [n] bytes from [src] to [dst] in one transfer, where one runs it: the
   host between memory it addresses or with an io device's, or a device's copy
   queue between memory it maps. A copy on a queue returns at once unless
   [wait]: what reads or writes its buffers later waits for it by their stamps.
   Records the transfer, and is whether it ran. *)
let direct ~wait src dst n =
  let sd = src.mem.dev and dd = dst.mem.dev in
  let start = Prof.now () in
  if local src.mem && local dst.mem then on_host Move src dst n start
  else if Dev.is_io sd && local dst.mem then on_host Io_read src dst n start
  else if Dev.is_io dd && local src.mem then on_host Io_write src dst n start
  else
    let side = if local src.mem then dst else src in
    (* A borrow on a device that runs no copy copies by its memory's own
       device. *)
    let runner =
      match side.mem.dev.copy_queue with
      | Some _ -> side.mem.dev
      | None -> side.mem.root.dev
    in
    (not (Dev.is_io runner)) && on_queue ~wait runner src dst n start

let rec copy ~src ~dst =
  Buffer.check_live fn src;
  Buffer.check_live fn dst;
  let n = Buffer.length src in
  if n <> Buffer.length dst then
    invalid_argf "Rig.%s: %d bytes into %d" fn n (Buffer.length dst);
  if Buffer.overlaps src dst then invalid_argf "Rig.%s: the buffers overlap" fn;
  if Buffer.access dst = Read then
    invalid_argf "Rig.%s: the destination's memory admits only reads" fn;
  let sd = src.mem.dev and dd = dst.mem.dev in
  Memory.check_owner src.mem;
  Memory.check_owner dst.mem;
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
  if direct ~wait src dst n then ()
  else if is_slot src.mem || is_slot dst.mem || far sd || far dd then
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
  let pieces = (n + half_bytes - 1) / half_bytes in
  let len k = Int.min half_bytes (n - (k * half_bytes)) in
  let run h0 h1 =
    let half k =
      Buffer.view (if k land 1 = 0 then h0 else h1) ~first:0 ~length:(len k)
    in
    let into k =
      leg
        ~src:(Buffer.view src ~first:(k * half_bytes) ~length:(len k))
        ~dst:(half k)
    in
    let out_of k =
      leg ~src:(half k)
        ~dst:(Buffer.view dst ~first:(k * half_bytes) ~length:(len k))
    in
    (* A device of this machine that runs legs maps the halves they use first;
       another machine's reaches no staging memory. *)
    let runs b = not (local b.mem || Dev.is_io b.mem.dev) in
    let maps b =
      if runs b && Option.is_none b.mem.dev.machine then begin
        map_half b.mem.dev h0;
        if pieces > 1 then map_half b.mem.dev h1
      end
    in
    maps src;
    maps dst;
    let ahead = runs src in
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
    for h = 0 to 1 do
      match slots.(i).(h) with
      | Some b -> ( try Buffer.wait b Buffer.Read_write with Dev.Lost _ -> ())
      | None -> ()
    done;
    give_slot i
  in
  match run (half i 0) (half i 1) with
  | () -> settle ()
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      settle ();
      Printexc.raise_with_backtrace e bt

(* One leg of a staged copy, which a device's queue runs without the host
   waiting. *)
and leg ~src ~dst = route ~wait:false src dst (Buffer.length src)
