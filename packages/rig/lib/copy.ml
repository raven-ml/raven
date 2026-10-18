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
external reraise : exn -> 'a = "%reraise"

let fn = "Buffer.copy"
let local m = Dev.same_machine m.dev Dev.host && m.host >= 0
let host_address b = b.mem.host + b.offset

(* A driver's device of another machine, which no staging memory reaches. *)
let far d = (not (Dev.same_machine d Dev.host)) && not (Dev.is_io d)

(* Records the host's transfer of [bytes] from [src] to [dst] asked at [start],
   which it sees done now, between the devices their memory counts in, as a
   submit records a copy part. *)
let record src dst bytes start =
  if Profile.enabled () then
    Profile.record
      (Copy
         {
           src = Memory.account src.mem.alloc;
           dst = Memory.account dst.mem.alloc;
           bytes;
           point = None;
           start;
           stop = Profile.now ();
         })

(* Staging memory, as copy.mli states it, each half made at the first piece
   that needs it. Halves of one allocation would order every leg after the
   other half's last one, so each half is an allocation of its own. *)
let half_bytes = 32 * 1024 * 1024

let names_lost b =
  match Memory.check_points b.mem.alloc.entry.stamps with
  | () -> false
  | exception Dev.Lost _ -> true

(* Staging memory lives in its device's [live] ([slots]), under the device's
   lock: the host's, or that of a device that maps no host memory. Memory
   returns its free halves under pressure, at [free_cache] and [set_budget],
   and once the device is lost and has stopped ([Memory.release_staging]). *)

(* Under [d]'s lock: the first slot of [d]'s staging memory from [i] that no
   copy holds, held now, or a new one once every slot is held, so no copy
   waits for a slot. A new slot is published by the last store, so an
   asynchronous exception before it changes nothing. It makes no half, whose
   creation can run the out-of-memory ladder, which takes every device's
   lock. *)
let rec claim_from d i =
  let l = d.live in
  if i = Array.length l.slots then begin
    let s =
      { halves = [| None; None |]; held = true; last = l.claims; owner = d }
    in
    l.slots <- Array.append l.slots [| s |];
    s
  end
  else if l.slots.(i).held then claim_from d (i + 1)
  else begin
    l.slots.(i).held <- true;
    l.slots.(i).last <- l.claims;
    l.slots.(i)
  end

let claim_slot d =
  d.live.claims <- d.live.claims + 1;
  claim_from d 0

let take_slot d = Dev.protect d claim_slot d

let release_slot s = s.held <- false

(* Allocates nothing, so no asynchronous exception keeps the slot. *)
let give_slot s = Dev.protect s.owner release_slot s

let rec idle_halves hs h (s : slot) =
  if h = 2 then hs
  else
    match s.halves.(h) with
    | Some b ->
        s.halves.(h) <- None;
        idle_halves (b :: hs) (h + 1) s
    | None -> idle_halves hs (h + 1) s

(* Under [d]'s lock: the halves of each free slot of [d] from [i] that no
   claim took within the last claims, as many as [d] has slots, taken out of
   them. *)
let rec take_idle d i hs =
  let l = d.live in
  if i = Array.length l.slots then hs
  else
    let s = l.slots.(i) in
    if (not s.held) && l.claims - s.last >= Array.length l.slots then
      take_idle d (i + 1) (idle_halves hs 0 s)
    else take_idle d (i + 1) hs

let take_idle_halves d = take_idle d 0 []

(* Returns the halves of [d]'s idle slots as memory returns free halves
   ([Memory.return_half]), after a give, outside the lock. Copies that keep a
   slot busy, one after another or in lockstep on several threads, claim it
   within every window and keep its halves; the extra slots of a burst go
   idle once as many later claims as there are slots have passed them by. An
   exception meanwhile leaves the halves it took to their collection. *)
let age d = List.iter Memory.return_half (Dev.protect d take_idle_halves d)

(* The half [h] of [s], made where it is missing or names a lost device's
   work, outside every lock. *)
let half s h =
  match s.halves.(h) with
  | Some b when not (names_lost b) -> b
  | _ ->
      let d = s.owner in
      let b =
        if Dev.is_host d then Buffer.create d half_bytes
        else Buffer.create ~memory:Pinned d half_bytes
      in
      s.halves.(h) <- Some b;
      b

(* Maps the staging half [h] of the slot [s] on [d], which runs legs through
   it. A map of the host's half that [d]'s driver refuses is memory [d] cannot
   give: the out-of-memory ladder reclaims for it, then raises
   [Out_of_memory]. A device's half is its own Pinned memory, which another
   device maps as a peer's or not at all: a refusal is no copy between the
   two. *)
let map_half s d h =
  if not (Memory.maps d h.mem) then
    if Dev.is_host s.owner then
      Memory.reclaiming d ~pool:d half_bytes (fun () ->
          if Memory.maps d h.mem then Some () else None)
    else
      invalid_argf "Rig.%s: no device copies between %s and %s" fn d.name
        s.owner.name

(* Whether a device of this machine runs the legs of a staged copy to or from
   [b]; another machine's reaches no staging memory. *)
let runs b = not (local b.mem || Dev.is_io b.mem.dev)

(* Maps the halves [h0] and [h1] of [s] for a staged copy of [pieces] on the
   device that runs [b]'s legs, if one does, before they run. *)
let map_halves s b ~pieces h0 h1 =
  let d = b.mem.dev in
  if runs b && Dev.same_machine d Dev.host then begin
    map_half s d h0;
    if pieces > 1 then map_half s d h1
  end

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
  match (d.live.copy_engine, mine d src.mem, mine d dst.mem) with
  | Some _, Some s, Some t ->
      let p =
        Submission.run_copy d ~src:{ src with mem = s }
          ~dst:{ dst with mem = t }
      in
      if wait then Dev.wait d (Point.value p);
      true
  | _ -> false

let io_read src dst n =
  match src.mem.alloc.entry.backing with
  | Io_memory { region = Io_region { m; h; r }; _ } ->
      let module I = (val m) in
      Dev.counted src.mem.dev Dev.Faults (fun () ->
          I.read h r ~at:src.offset ~dst:(host_address dst) ~len:n)
  | Driver_memory _ | Kept _ ->
      invalid_argf "Rig.%s: the source is no io memory" fn

let io_write src dst n =
  match dst.mem.alloc.entry.backing with
  | Io_memory { region = Io_region { m; h; r }; _ } ->
      let module I = (val m) in
      Dev.counted dst.mem.dev Dev.Faults (fun () ->
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
  let start = Profile.now () in
  if local src.mem && local dst.mem then on_host Move src dst n start
  else if Dev.is_io sd && local dst.mem then on_host Io_read src dst n start
  else if Dev.is_io dd && local src.mem then on_host Io_write src dst n start
  else
    let side = if local src.mem then dst else src in
    (* A borrow on a device that runs no copy copies by its memory's own
       device. *)
    let runner =
      match side.mem.dev.live.copy_engine with
      | Some _ -> side.mem.dev
      | None -> side.mem.alloc.owner
    in
    (not (Dev.is_io runner)) && on_queue ~wait runner src dst

(* The device that runs the legs of a staged copy to or from [b]: its device's
   copy engine, or, for a borrow on a device that runs no copy, its memory's own
   device's, as [direct] picks it. *)
let runner b =
  match b.mem.dev.live.copy_engine with
  | Some _ -> b.mem.dev
  | None -> b.mem.alloc.owner

(* Whether [b]'s legs run on a device of this machine that maps no host memory,
   which then stages through its own memory. *)
let needs_own b =
  let d = runner b in
  runs b && Dev.same_machine d Dev.host && not d.maps_host

(* The device whose staging memory a transfer between the host's values and
   [b] goes through: [b]'s runner where it needs its own, and the host
   otherwise. *)
let stage_with b = if needs_own b then runner b else Dev.host

(* The device whose staging memory a copy between [src] and [dst] goes
   through, where one serves both. *)
let stage_for src dst = if needs_own src then stage_with src else stage_with dst

(* Whether a staged copy between [src] and [dst] goes through two stages: two
   devices that need their own share no staging memory. *)
let two_stages src dst =
  needs_own src && needs_own dst && runner src != runner dst

let rec wait_halves s h =
  if h < 2 then begin
    (match s.halves.(h) with
    | Some b -> ( try Buffer.wait b Buffer.Read_write with Dev.Lost _ -> ())
    | None -> ());
    wait_halves s (h + 1)
  end

(* Waits for the legs through the slot [s]'s halves, so that no leg outlives
   its transfer, then gives the slot back, also when a wait raises, such as
   [Sys.Break]: the halves' stamps order their next use, and lost halves are
   replaced. *)
let settle s =
  match wait_halves s 0 with
  | () ->
      give_slot s;
      age s.owner
  | exception e ->
      give_slot s;
      Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())

(* Waits for the legs through the slots [a] and [b], then gives both back, as
   [settle] does. *)
let settle_two a b =
  match
    wait_halves a 0;
    wait_halves b 0
  with
  | () ->
      give_slot a;
      give_slot b;
      age a.owner;
      age b.owner
  | exception e ->
      give_slot a;
      give_slot b;
      Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())

let pieces n = (n + half_bytes - 1) / half_bytes
let piece_bytes n k = Int.min half_bytes (n - (k * half_bytes))

(* Moves [n] bytes through a slot of [d]'s staging memory, whose two halves
   take the pieces in turn, so one half fills while the other drains:
   [into h at] fills the half [h] with the bytes from byte [at], and
   [out_of h at] drains it. The halves are first mapped on the devices that
   run legs of [sides]. A leg on a device's queue runs without the host
   waiting for it: the next leg that reads or writes its half waits for it by
   the half's stamps. With [ahead], the leg into the slot runs one piece ahead
   of the leg out of it, as when a device fills and the host drains;
   otherwise the host's leg into the slot fills one half while a device
   drains the other. *)
let through d n ~sides ~ahead ~into ~out_of =
  let pieces = pieces n in
  let len k = piece_bytes n k in
  let run s =
    let h0 = half s 0 in
    (* One piece goes through the first half alone. *)
    let h1 = if pieces > 1 then half s 1 else h0 in
    let half k =
      Buffer.view (if k land 1 = 0 then h0 else h1) ~first:0 ~length:(len k)
    in
    let into k = into (half k) (k * half_bytes) in
    let out_of k = out_of (half k) (k * half_bytes) in
    List.iter (fun b -> map_halves s b ~pieces h0 h1) sides;
    if ahead then into 0;
    for k = 0 to pieces - 1 do
      if ahead then (if k + 1 < pieces then into (k + 1)) else into k;
      out_of k
    done
  in
  (* Nothing allocates between taking the slot and the handlers that give it
     back, so no asynchronous exception keeps it. *)
  let s = take_slot d in
  match run s with
  | () -> settle s
  | exception e -> (
      match Printexc.get_raw_backtrace () with
      | bt ->
          settle s;
          Printexc.raise_with_backtrace e bt
      | exception x ->
          give_slot s;
          raise x)

(* The bytes of [b] from [at] that go through the half [h]. *)
let piece b at h = Buffer.view b ~first:at ~length:(Buffer.length h)

let rec copy ~src ~dst =
  Dev.check_held fn;
  let n = Buffer.length src in
  if n <> Buffer.length dst then
    invalid_argf "Rig.%s: %d bytes into %d" fn n (Buffer.length dst);
  if Buffer.overlaps src dst then invalid_argf "Rig.%s: the buffers overlap" fn;
  (* An io device's write of its memory from a borrow of its own pages, or its
     read into them, can wait on itself in the system for good: a file's write
     from its own mapping on macOS's APFS. *)
  if
    Dev.is_io src.mem.dev <> Dev.is_io dst.mem.dev
    && src.mem.alloc == dst.mem.alloc
  then
    invalid_argf "Rig.%s: one buffer is a borrow of the other's io memory" fn;
  if Buffer.access dst = Read then
    invalid_argf "Rig.%s: the destination admits only reads" fn;
  Claim.reading2 fn src dst claimed n

(* [copy] under read claims on both sides, which keep a donation from writing
   either while the copy waits and runs. The claims' release after it keeps the
   buffers reachable: a read, a write or a move hands over addresses only, and a
   collection could otherwise free their memory under it. *)
and claimed src dst n =
  let sd = src.mem.dev and dd = dst.mem.dev in
  Memory.check_owner src.mem;
  Memory.check_owner dst.mem;
  Memory.drain sd;
  if dd != sd then Memory.drain dd;
  if n > 0 then route ~wait:true src dst n

(* Copies [n] bytes from [src] to [dst], directly or through the staging memory,
   recording each transfer that ran. *)
and route ~wait src dst n =
  let sd = src.mem.dev and dd = dst.mem.dev in
  if direct ~wait src dst n then ()
  else if far sd || far dd then
    invalid_argf "Rig.%s: no device copies between %s and %s" fn sd.name
      dd.name
  else staged src dst n

(* Through staging memory. Where a device runs the legs out of [src], each runs
   one piece ahead. *)
and staged src dst n =
  if two_stages src dst then between src dst n
  else
    through (stage_for src dst) n ~sides:[ src; dst ] ~ahead:(runs src)
      ~into:(fun h at -> leg ~src:(piece src at h) ~dst:h)
      ~out_of:(fun h at -> leg ~src:h ~dst:(piece dst at h));
  Buffer.wait dst Buffer.Read_write

(* Between two devices that each stage through their own memory: [src]'s
   device fills a half of a slot of its stage, the host moves the half's bytes
   into a half of a slot of [dst]'s device's stage, and that device drains it.
   The leg into the next half runs while the host moves the last. *)
and between src dst n =
  let a = runner src and b = runner dst in
  let pieces = pieces n in
  let run sa sb =
    let a0 = half sa 0 and b0 = half sb 0 in
    let a1 = if pieces > 1 then half sa 1 else a0 in
    let b1 = if pieces > 1 then half sb 1 else b0 in
    map_halves sa src ~pieces a0 a1;
    map_halves sb dst ~pieces b0 b1;
    let half h0 h1 k =
      Buffer.view (if k land 1 = 0 then h0 else h1) ~first:0
        ~length:(piece_bytes n k)
    in
    let into k =
      let h = half a0 a1 k in
      leg ~src:(piece src (k * half_bytes) h) ~dst:h
    in
    into 0;
    for k = 0 to pieces - 1 do
      if k + 1 < pieces then into (k + 1);
      let ha = half a0 a1 k and hb = half b0 b1 k in
      leg ~src:ha ~dst:hb;
      leg ~src:hb ~dst:(piece dst (k * half_bytes) hb)
    done
  in
  (* Nothing allocates between taking a slot and the handler that gives it
     back. *)
  let sa = take_slot a in
  match take_slot b with
  | exception e ->
      give_slot sa;
      reraise e
  | sb -> (
      match run sa sb with
      | () -> settle_two sa sb
      | exception e -> (
          match Printexc.get_raw_backtrace () with
          | bt ->
              settle_two sa sb;
              Printexc.raise_with_backtrace e bt
          | exception x ->
              give_slot sa;
              give_slot sb;
              raise x))

(* One leg of a staged copy, which a device's queue runs without the host
   waiting. A leg runs in one transfer by construction: one that finds none,
   such as into a device that runs no copy, goes no further. *)
and leg ~src ~dst =
  if not (direct ~wait:false src dst (Buffer.length src)) then
    invalid_argf "Rig.%s: no device copies between %s and %s" fn
      src.mem.dev.name dst.mem.dev.name

(* The host's values *)

let writable fn b =
  if Buffer.access b = Read then
    invalid_argf "Rig.%s: the buffer admits only reads" fn

(* The bytes of [dst], which the host does not address, through staging
   memory, holding a read claim on it as a copy does: [put h at] writes the
   bytes from [dst]'s byte [at] into the half [h], after the half's last
   leg. *)
let upload fn dst put =
  Claim.reading fn dst (fun () ->
      Memory.check_owner dst.mem;
      Memory.drain dst.mem.dev;
      let n = Buffer.length dst in
      if n > 0 then begin
        through (stage_with dst) n ~sides:[ dst ] ~ahead:false ~into:put
          ~out_of:(fun h at ->
            leg ~src:h
              ~dst:(Buffer.view dst ~first:at ~length:(Buffer.length h)));
        Buffer.wait dst Buffer.Read_write
      end)

(* The bytes of [src], which the host does not address, through staging
   memory, holding a read claim on it: [take h at] reads the bytes from
   [src]'s byte [at] out of the half [h], after the leg that wrote it. *)
let download fn src take =
  Claim.reading fn src (fun () ->
      Memory.check_owner src.mem;
      Memory.drain src.mem.dev;
      let n = Buffer.length src in
      if n > 0 then
        through (stage_with src) n ~sides:[ src ] ~ahead:true
          ~into:(fun h at ->
            leg ~src:(Buffer.view src ~first:at ~length:(Buffer.length h))
              ~dst:h)
          ~out_of:take)

(* Where the host addresses the buffer, the blits and [fill] claim and wait,
   move the bytes, and drop the claim. The move neither raises nor polls in
   native code, so no handler is needed and no asynchronous exception keeps
   the claim. *)
let blit_from_string s i b j n =
  let fn = "Buffer.blit_from_string" in
  Dev.check_held fn;
  Buffer.check_range fn "string" (String.length s) i n;
  Buffer.check_range fn "buffer" (Buffer.length b) j n;
  writable fn b;
  if Buffer.host_addressed b then begin
    Claim.wait fn Buffer.Read_write b;
    Buffer.blit_string s i b j n;
    Memory.claim_drop b
  end
  else begin
    (* The view would otherwise name itself for a dead buffer; the check that
       counts runs under the claim. *)
    Buffer.check_live fn b;
    upload fn (Buffer.view b ~first:j ~length:n) (fun h at ->
        Buffer.wait h Buffer.Read_write;
        Buffer.blit_string s (i + at) h 0 (Buffer.length h))
  end

let blit_to_bytes b i s j n =
  let fn = "Buffer.blit_to_bytes" in
  Dev.check_held fn;
  Buffer.check_range fn "buffer" (Buffer.length b) i n;
  Buffer.check_range fn "bytes" (Bytes.length s) j n;
  if Buffer.host_addressed b then begin
    Claim.wait fn Buffer.Read b;
    Buffer.blit_bytes b i s j n;
    Memory.claim_drop b
  end
  else begin
    Buffer.check_live fn b;
    download fn (Buffer.view b ~first:i ~length:n) (fun h at ->
        Buffer.wait h Buffer.Read;
        Buffer.blit_bytes h 0 s (j + at) (Buffer.length h))
  end

let fill b c =
  let fn = "Buffer.fill" in
  Dev.check_held fn;
  writable fn b;
  if Buffer.host_addressed b then begin
    Claim.wait fn Buffer.Read_write b;
    Buffer.set_bytes b 0 (Buffer.length b) c;
    Memory.claim_drop b
  end
  else
    (* The halves keep [c] once the first two pieces filled them. *)
    upload fn b (fun h at ->
        if at < 2 * half_bytes then begin
          Buffer.wait h Buffer.Read_write;
          Buffer.set_bytes h 0 (Buffer.length h) c
        end)
