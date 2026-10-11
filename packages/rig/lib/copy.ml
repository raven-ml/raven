(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The routes are copy.mli's. Where two devices could copy, [direct] picks
   [dst]'s device when only [src] is memory the host addresses, and [src]'s
   otherwise. *)

open Def

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

external memmove : int -> int -> int -> unit = "caml_rig_memmove"

let fn = "Buffer.copy"
let local m = Dev.same_machine m.dev Dev.host && m.host >= 0
let host_address b = b.mem.host + b.offset

(* A driver's device of another machine, which no staging memory reaches. *)
let far d = (not (Dev.same_machine d Dev.host)) && not (Dev.is_io d)

(* Records the host's transfer of [bytes] from [src] to [dst] asked at [start],
   which it sees done now, between the devices of their allocations, as a submit
   records a copy part. *)
let record src dst bytes start =
  if Prof.enabled () then
    Prof.record
      (Copy
         {
           src = (Memory.root src.mem).dev;
           dst = (Memory.root dst.mem).dev;
           bytes;
           start;
           stop = Prof.now ();
         })

(* Staging memory, as copy.mli states it, made at the first copy that needs
   it. Halves of one allocation would order every leg after the other half's
   last one, so each half is an allocation of its own. *)
let half_bytes = 32 * 1024 * 1024

type stage = {
  owner : device; (* the host, or the device whose Pinned memory it is *)
  halves : buffer option array array;
  in_use : bool array;
}

let stage owner =
  {
    owner;
    halves = [| [| None; None |]; [| None; None |] |];
    in_use = [| false; false |];
  }

(* The host's stage, made at load: its halves wait for a copy, and a lazy
   that two domains' first copies force at once raises in one of them. *)
let host_stage = stage Dev.host

(* The stages of devices that map no host memory, by device index. *)
module Stages = Tbl.Make (Int)

let device_stages : stage Stages.t = Stages.create 4
let slots_lock = Lock.create ()

let names_lost b =
  match Memory.check_points (Memory.stamps b.mem) with
  | () -> false
  | exception Dev.Lost _ -> true

(* The stage of [d], made at its first staged copy; the stages of lost devices
   go then, so that their memory returns. *)
let find_stage d =
  let lost i st is = if Dev.is_lost st.owner then i :: is else is in
  List.iter (Stages.remove device_stages) (Stages.fold lost device_stages []);
  match Stages.find_opt device_stages d.index with
  | Some st -> st
  | None ->
      let st = stage d in
      Stages.replace device_stages d.index st;
      st

let stage_of d = Lock.protect slots_lock find_stage d

(* Waits for a free slot of [st] and marks it used. *)
let rec claim_slot st =
  if not st.in_use.(0) then begin
    st.in_use.(0) <- true;
    0
  end
  else if not st.in_use.(1) then begin
    st.in_use.(1) <- true;
    1
  end
  else begin
    Lock.wait slots_lock;
    claim_slot st
  end

let take_slot st = Lock.protect slots_lock claim_slot st

(* Allocates nothing, so no asynchronous exception keeps the slot. *)
let give_slot st i =
  Lock.hold slots_lock;
  st.in_use.(i) <- false;
  Lock.broadcast slots_lock;
  Lock.release slots_lock

let half st i h =
  match st.halves.(i).(h) with
  | Some b when not (names_lost b) -> b
  | _ ->
      let b =
        if Dev.is_host st.owner then Buffer.create st.owner half_bytes
        else Buffer.create ~memory:Pinned st.owner half_bytes
      in
      st.halves.(i).(h) <- Some b;
      b

(* Maps the staging half [h] of [st] on [d], which runs legs through it. A map
   of the host's half that [d]'s driver refuses is memory [d] cannot give: the
   out-of-memory ladder reclaims for it, then raises [Out_of_memory]. A device's
   half is its own Pinned memory, which another device maps as a peer's or not
   at all: a refusal is no copy between the two. *)
let map_half st d h =
  if not (Memory.maps d h.mem) then
    if Dev.is_host st.owner then
      Memory.reclaiming d ~pool:d half_bytes (fun () ->
          if Memory.maps d h.mem then Some () else None)
    else
      invalid_argf "Rig.%s: no device copies between %s and %s" fn d.name
        st.owner.name

(* Whether a device of this machine runs the legs of a staged copy to or from
   [b]; another machine's reaches no staging memory. *)
let runs b = not (local b.mem || Dev.is_io b.mem.dev)

(* Maps the halves [h0] and [h1] of [st] for a staged copy of [pieces] on the
   device that runs [b]'s legs, if one does, before they run. *)
let map_halves st b ~pieces h0 h1 =
  let d = b.mem.dev in
  if runs b && Dev.same_machine d Dev.host then begin
    map_half st d h0;
    if pieces > 1 then map_half st d h1
  end

(* Whether [m] is the memory of a half of [st]: no closure, so the host's check
   allocates nothing. Racy: another thread changes only the halves of a slot it
   took, which [m] is not, so a stale half answers the same. *)
let holds_half st m i h =
  match st.halves.(i).(h) with Some b -> Memory.root b.mem == Memory.root m | None -> false

let holds st m =
  holds_half st m 0 0 || holds_half st m 0 1 || holds_half st m 1 0
  || holds_half st m 1 1

let staged_in m =
  Stages.fold (fun _ st r -> r || holds st m) device_stages false

(* Whether [m] is a staging slot's memory: a copy through it that no device runs
   goes no further. *)
let is_slot m =
  holds host_stage m
  || Stages.length device_stages > 0
     && Lock.protect slots_lock staged_in m

let queued d ~src ~dst =
  Dev.wait d (Point.value (Submission.run_copy d ~src ~dst))

(* [m] as [d]'s copy engine takes it: a device of another machine takes this
   process's memory as it is, its driver carrying the bytes; another device
   maps it. A function of its own, so a copy builds no closure. *)
let mine d m =
  if (not (Dev.same_machine d Dev.host)) && local m then Some m
  else Memory.borrow d m

(* A copy on [d]'s copy engine between buffers [d] maps, waited for when [wait].
   Its submit records it. *)
let on_queue ~wait d src dst =
  match ((Dev.live d).copy_engine, mine d src.mem, mine d dst.mem) with
  | Some _, Some s, Some t ->
      let p =
        Submission.run_copy d ~src:{ src with mem = s }
          ~dst:{ dst with mem = t }
      in
      if wait then Dev.wait d (Point.value p);
      true
  | _ -> false

let io_read src dst n =
  match src.mem.entry.backing with
  | Io_memory { region = Io_region { m; h; r }; _ } ->
      let module I = (val m) in
      Dev.counted src.mem.dev (fun () ->
          I.read h r ~at:src.offset ~dst:(host_address dst) ~len:n)
  | Driver_memory _ | Kept _ ->
      invalid_argf "Rig.%s: the source is no io memory" fn

let io_write src dst n =
  match dst.mem.entry.backing with
  | Io_memory { region = Io_region { m; h; r }; _ } ->
      let module I = (val m) in
      Dev.counted dst.mem.dev (fun () ->
          I.write h r ~at:dst.offset ~src:(host_address src) ~len:n)
  | Driver_memory _ | Kept _ ->
      invalid_argf "Rig.%s: the destination is no io memory" fn

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
  record src dst n start;
  true

(* Copies [n] bytes from [src] to [dst] in one transfer, where one runs it: the
   host between memory it addresses or with an io device's, or a device's copy
   queue between memory it maps. A copy on a queue returns at once unless
   [wait]: what reads or writes its buffers later waits for it by their stamps.
   The host records its transfer, a submit a queued one. Is whether it ran. *)
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
      match (Dev.live side.mem.dev).copy_engine with
      | Some _ -> side.mem.dev
      | None -> (Memory.root side.mem).dev
    in
    (not (Dev.is_io runner)) && on_queue ~wait runner src dst

(* The device that runs the legs of a staged copy to or from [b]: its device's
   copy engine, or, for a borrow on a device that runs no copy, its memory's own
   device's, as [direct] picks it. *)
let runner b =
  match (Dev.live b.mem.dev).copy_engine with
  | Some _ -> b.mem.dev
  | None -> (Memory.root b.mem).dev

(* Whether [b]'s legs run on a device of this machine that maps no host memory,
   which then stages through its own memory. *)
let needs_own b =
  let d = runner b in
  runs b && Dev.same_machine d Dev.host && not d.maps_host

(* The stage a transfer between the host's values and [b] goes through: its
   device's own where it needs one, and the host's otherwise. *)
let stage_with b =
  if needs_own b then stage_of (runner b) else host_stage

(* The stage a copy between [src] and [dst] goes through. Two devices that need
   their own share no staging memory. *)
let stage_for src dst =
  if needs_own src && needs_own dst && runner src != runner dst then
    invalid_argf "Rig.%s: no device copies between %s and %s" fn
      src.mem.dev.name dst.mem.dev.name;
  if needs_own src then stage_with src else stage_with dst

let rec wait_halves st i h =
  if h < 2 then begin
    (match st.halves.(i).(h) with
    | Some b -> ( try Buffer.wait b Buffer.Read_write with Dev.Lost _ -> ())
    | None -> ());
    wait_halves st i (h + 1)
  end

(* Waits for the legs through slot [i]'s halves, so that no leg outlives its
   transfer, then gives the slot back, also when a wait raises, such as
   [Sys.Break]: the halves' stamps order their next use, and lost halves are
   replaced. *)
let settle st i =
  match wait_halves st i 0 with
  | () -> give_slot st i
  | exception e ->
      give_slot st i;
      Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())

(* Moves [n] bytes through a slot of [st], whose two halves take the pieces in
   turn, so one half fills while the other drains: [into h at] fills the half
   [h] with the bytes from byte [at], and [out_of h at] drains it. The halves
   are first mapped on the devices that run legs of [sides]. A leg on a
   device's queue runs without the host waiting for it: the next leg that reads
   or writes its half waits for it by the half's stamps. With [ahead], the leg
   into the slot runs one piece ahead of the leg out of it, as when a device
   fills and the host drains; otherwise the host's leg into the slot fills one
   half while a device drains the other. *)
let through st n ~sides ~ahead ~into ~out_of =
  let pieces = (n + half_bytes - 1) / half_bytes in
  let len k = Int.min half_bytes (n - (k * half_bytes)) in
  let run i =
    let h0 = half st i 0 and h1 = half st i 1 in
    let half k =
      Buffer.view (if k land 1 = 0 then h0 else h1) ~first:0 ~length:(len k)
    in
    let into k = into (half k) (k * half_bytes) in
    let out_of k = out_of (half k) (k * half_bytes) in
    List.iter (fun b -> map_halves st b ~pieces h0 h1) sides;
    if ahead then into 0;
    for k = 0 to pieces - 1 do
      if ahead then (if k + 1 < pieces then into (k + 1)) else into k;
      out_of k
    done
  in
  (* Nothing allocates between taking the slot and the handlers that give it
     back, so no asynchronous exception keeps it. *)
  let i = take_slot st in
  match run i with
  | () -> settle st i
  | exception e -> (
      match Printexc.get_raw_backtrace () with
      | bt ->
          settle st i;
          Printexc.raise_with_backtrace e bt
      | exception x ->
          give_slot st i;
          raise x)

let rec copy ~src ~dst =
  Buffer.check_live fn src;
  Buffer.check_live fn dst;
  let n = Buffer.length src in
  if n <> Buffer.length dst then
    invalid_argf "Rig.%s: %d bytes into %d" fn n (Buffer.length dst);
  if Buffer.overlaps src dst then invalid_argf "Rig.%s: the buffers overlap" fn;
  (* An io device's write of its memory from a borrow of its own pages, or its
     read into them, can wait on itself in the system for good: a file's write
     from its own mapping on macOS's APFS. *)
  if
    Dev.is_io src.mem.dev <> Dev.is_io dst.mem.dev
    && Memory.root src.mem == Memory.root dst.mem
  then
    invalid_argf "Rig.%s: one buffer is a borrow of the other's io memory" fn;
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

(* Through staging memory: where a device runs the legs out of [src], each runs
   one piece ahead. *)
and staged src dst n =
  let piece b at h = Buffer.view b ~first:at ~length:(Buffer.length h) in
  through (stage_for src dst) n ~sides:[ src; dst ] ~ahead:(runs src)
    ~into:(fun h at -> leg ~src:(piece src at h) ~dst:h)
    ~out_of:(fun h at -> leg ~src:h ~dst:(piece dst at h));
  Buffer.wait dst Buffer.Read_write

(* One leg of a staged copy, which a device's queue runs without the host
   waiting. *)
and leg ~src ~dst = route ~wait:false src dst (Buffer.length src)

(* The host's values *)

let writable fn b =
  Buffer.check_live fn b;
  if Buffer.access b = Read then
    invalid_argf "Rig.%s: the buffer's memory admits only reads" fn

(* The bytes of [dst], which the host does not address, through staging
   memory: [put h at] writes the bytes from [dst]'s byte [at] into the half
   [h], after the half's last leg. *)
let upload dst put =
  Memory.check_owner dst.mem;
  Memory.drain dst.mem.dev;
  let n = Buffer.length dst in
  if n > 0 then begin
    through (stage_with dst) n ~sides:[ dst ] ~ahead:false ~into:put
      ~out_of:(fun h at ->
        leg ~src:h ~dst:(Buffer.view dst ~first:at ~length:(Buffer.length h)));
    Buffer.wait dst Buffer.Read_write
  end

(* The bytes of [src], which the host does not address, through staging
   memory: [take h at] reads the bytes from [src]'s byte [at] out of the half
   [h], after the leg that wrote it. *)
let download src take =
  Memory.check_owner src.mem;
  Memory.drain src.mem.dev;
  let n = Buffer.length src in
  if n > 0 then
    through (stage_with src) n ~sides:[ src ] ~ahead:true
      ~into:(fun h at ->
        leg ~src:(Buffer.view src ~first:at ~length:(Buffer.length h)) ~dst:h)
      ~out_of:take

let blit_from_string s i b j n =
  let fn = "Buffer.blit_from_string" in
  Buffer.check_range fn "string" (String.length s) i n;
  Buffer.check_range fn "buffer" (Buffer.length b) j n;
  writable fn b;
  if Buffer.host_addressed b then begin
    Buffer.wait b Buffer.Read_write;
    Buffer.blit_string s i b j n
  end
  else
    upload (Buffer.view b ~first:j ~length:n) (fun h at ->
        Buffer.wait h Buffer.Read_write;
        Buffer.blit_string s (i + at) h 0 (Buffer.length h))

let blit_to_bytes b i s j n =
  let fn = "Buffer.blit_to_bytes" in
  Buffer.check_range fn "buffer" (Buffer.length b) i n;
  Buffer.check_range fn "bytes" (Bytes.length s) j n;
  Buffer.check_live fn b;
  if Buffer.host_addressed b then begin
    Buffer.wait b Buffer.Read;
    Buffer.blit_bytes b i s j n
  end
  else
    download (Buffer.view b ~first:i ~length:n) (fun h at ->
        Buffer.wait h Buffer.Read;
        Buffer.blit_bytes h 0 s (j + at) (Buffer.length h))

let fill b c =
  writable "Buffer.fill" b;
  if Buffer.host_addressed b then begin
    Buffer.wait b Buffer.Read_write;
    Buffer.set_bytes b 0 (Buffer.length b) c
  end
  else
    (* The halves keep [c] once the first two pieces filled them. *)
    upload b (fun h at ->
        if at < 2 * half_bytes then begin
          Buffer.wait h Buffer.Read_write;
          Buffer.set_bytes h 0 (Buffer.length h) c
        end)
