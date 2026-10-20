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

(* Whether [m] is memory an io device owns: its own, or a borrow of its
   pages. *)
let io_owned m = Dev.is_io m.alloc.owner

let host_address b = b.mem.host + b.offset

(* A driver's device of another machine, which has no staging memory of its
   own: it reaches the host's only from a side the host moves ([hosted]). *)
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

(* Staging memory

   As copy.mli states it. A half's [cursor] is the byte from which no
   unfinished work uses its buffer, so the host writes a piece there without
   waiting, and the leg that drains it follows the host's write by the
   stamps, as work after a host write does. A device's write of a half
   follows every use of the whole allocation (rule 3), so a piece a device
   writes takes a half whose uses are done, or a new one: packed, a device's
   leg into a half would follow another device's drain of it, and the two
   devices' copies would chain. Pieces start at multiples of [piece_align]
   bytes in their half. *)

let half_bytes = 32 * 1024 * 1024
let piece_align = 256
let round n = (n + piece_align - 1) land lnot (piece_align - 1)

(* Who writes the pieces of a staged copy into its halves: the host, or a
   device's leg. *)
type writer = Host | Device

let names_lost b =
  match Memory.check_points b.mem.alloc.entry.stamps with
  | () -> false
  | exception Dev.Lost _ -> true

(* Under [l]'s lock: the index of the free half that [ok] takes, the one
   [better] prefers among them, from [i], or [-1]. *)
let rec find l ok better i found =
  if i = Array.length l.halves then found
  else
    let h = l.halves.(i) in
    let found =
      if (not h.held) && ok h && (found < 0 || better h l.halves.(found)) then
        i
      else found
    in
    find l ok better (i + 1) found

let has_room need h = h.buffer <> None && h.cursor + need <= half_bytes

let uses_done h =
  match h.buffer with None -> true | Some b -> Memory.uses_done b.mem

let later a b = a.lap > b.lap
let earlier a b = a.lap < b.lap
let buffered a b = a.buffer <> None && b.buffer = None
let some_buffer h = h.buffer <> None

(* Under [l]'s lock: [h] held from now, its lap restarted at byte 0 if
   [restart]. *)
let hold l ~restart h =
  h.held <- true;
  h.last <- l.takes;
  if restart then begin
    l.laps <- l.laps + 1;
    h.lap <- l.laps;
    h.cursor <- 0
  end;
  h

(* Under [d]'s lock: a half of [d]'s staging memory for the pieces [writer]
   writes, held now. A piece of [need] bytes that the host writes, [need]
   positive, takes the free half with room after its cursor whose lap started
   last. Otherwise a half is taken at its start: a free one whose uses are
   done, one with a buffer first; for the host, else the free half whose lap
   started first, whose uses its holder waits for; else a new half, so no copy
   waits for a hold. A new half is published by the last store, so an
   asynchronous exception before it changes nothing. It makes no buffer, whose
   creation can run the out-of-memory ladder, which takes every device's
   lock. *)
let take_locked (d, writer, need) =
  let l = d.live in
  l.takes <- l.takes + 1;
  let packed =
    if writer = Host && need > 0 then find l (has_room need) later 0 (-1)
    else -1
  in
  if packed >= 0 then hold l ~restart:false l.halves.(packed)
  else
    let i = find l uses_done buffered 0 (-1) in
    let i =
      if i < 0 && writer = Host then find l some_buffer earlier 0 (-1) else i
    in
    if i >= 0 then hold l ~restart:true l.halves.(i)
    else begin
      l.laps <- l.laps + 1;
      let h =
        {
          buffer = None;
          held = true;
          cursor = 0;
          lap = l.laps;
          last = l.takes;
          stage = d;
        }
      in
      l.halves <- Array.append l.halves [| h |];
      h
    end

let take d writer need = Dev.protect d take_locked (d, writer, need)
let release h = h.held <- false

(* Gives [h] back, its cursor set by its holder, which reads it no more.
   Allocates nothing, so no asynchronous exception keeps the half. *)
let give h = Dev.protect h.stage release h

(* Gives [h] back with its uses unknown: no piece is packed into it until its
   lap restarts. *)
let abandon h =
  h.cursor <- half_bytes;
  give h

(* Under [d]'s lock: the buffers of [d]'s free halves that no take took within
   the last takes, as many as [d] has halves, taken out of them, the list made
   before the stores. *)
let take_idle d =
  let l = d.live in
  let n = Array.length l.halves in
  Array.fold_left
    (fun bs h ->
      match h.buffer with
      | Some b when (not h.held) && l.takes - h.last >= n ->
          let bs = b :: bs in
          h.buffer <- None;
          h.cursor <- 0;
          bs
      | Some _ | None -> bs)
    [] l.halves

(* Returns the buffers of [d]'s idle halves as memory returns free halves
   ([Memory.return_half]), after a give, outside the lock. Copies that keep a
   half busy, one after another or in lockstep on several threads, take it
   within every window and keep its buffer; the extra halves of a burst go
   idle once as many later takes as there are halves have passed them by. An
   exception meanwhile leaves the buffers it took to their collection. *)
let age d = List.iter Memory.return_half (Dev.protect d take_idle d)

(* The buffer of [h], made where it is missing or names a lost device's work,
   outside every lock. A half the host writes from its start waits for its
   uses first, which a half taken at its start only has if its lap is the
   oldest; a loss meanwhile replaces it. *)
let ready h writer =
  let make () =
    let d = h.stage in
    let b =
      if Dev.is_host d then Buffer.create d half_bytes
      else Buffer.create ~memory:Pinned d half_bytes
    in
    h.buffer <- Some b;
    b
  in
  match h.buffer with
  | Some b when not (names_lost b) -> (
      if writer = Device || h.cursor > 0 then b
      else
        match Buffer.wait b Buffer.Read_write with
        | () -> b
        | exception Dev.Lost _ -> make ())
  | _ -> make ()

(* Maps the buffer [b] of a half of [stage]'s staging memory on [d], which
   runs legs through it. A map of the host's half that [d]'s driver refuses is
   memory [d] cannot give: the out-of-memory ladder reclaims for it, then
   raises [Out_of_memory]. A device's half is its own Pinned memory, which
   another device maps as a peer's or not at all: a refusal is no copy between
   the two. *)
let map_half stage d b =
  if not (Memory.maps d b.mem) then
    if Dev.is_host stage then
      Memory.reclaiming d ~pool:d half_bytes (fun () ->
          if Memory.maps d b.mem then Some () else None)
    else
      invalid_argf "Rig.%s: no device copies between %s and %s" fn d.name
        stage.name

(* Whether the host moves [b]'s bytes into or out of staging memory: memory
   it addresses or an io device's. Only such a side reaches a device of
   another machine, through the host's staging memory. *)
let hosted b = local b.mem || io_owned b.mem

(* Whether a device of this machine runs the legs of a staged copy to or from
   [b]; another machine's reaches no staging memory. *)
let runs b = not (local b.mem || Dev.is_io b.mem.dev)

(* Maps the buffers [b0] and [b1] of [stage]'s halves on the device that runs
   [b]'s legs, if one does, before they run. *)
let map_halves stage b b0 b1 =
  let d = b.mem.dev in
  if runs b && Dev.same_machine d Dev.host then begin
    map_half stage d b0;
    if b1 != b0 then map_half stage d b1
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

(* An io device's read or write of memory it owns, a borrow of its pages
   included, is a counted call of that device, so its fault loses the io
   device. *)
let io_read src dst n =
  match src.mem.alloc.entry.backing with
  | Io_memory { region = Io_region { m; h; r }; _ } ->
      let module I = (val m) in
      Dev.counted src.mem.alloc.owner Dev.Faults (fun () ->
          I.read h r ~at:src.offset ~dst:(host_address dst) ~len:n)
  | Driver_memory _ | Kept _ ->
      invalid_argf "Rig.%s: the source is no io memory" fn

let io_write src dst n =
  match dst.mem.alloc.entry.backing with
  | Io_memory { region = Io_region { m; h; r }; _ } ->
      let module I = (val m) in
      Dev.counted dst.mem.alloc.owner Dev.Faults (fun () ->
          I.write h r ~at:dst.offset ~src:(host_address src) ~len:n)
  | Driver_memory _ | Kept _ ->
      invalid_argf "Rig.%s: the destination is no io memory" fn

(* How the host moves bytes: between memory it addresses, or by an io device's
   read or write. A constant, so a host copy allocates nothing. *)
type transfer = Move | Io_read | Io_write

(* The host's copy of [n] bytes from [src] to [dst], after their devices' work,
   asked at [start]. Into a staging half ([into_half]) it waits for nothing on
   [dst]: no unfinished work uses the bytes it writes ([through]). *)
let on_host ~into_half transfer src dst n start =
  Buffer.wait_points src Buffer.Read;
  if not into_half then Buffer.wait_points dst Buffer.Read_write;
  (match transfer with
  | Move -> memmove (host_address dst) (host_address src) n
  | Io_read -> io_read src dst n
  | Io_write -> io_write src dst n);
  record src dst n start;
  true

(* Copies [n] bytes from [src] to [dst] in one transfer, where one runs it: the
   host between memory it addresses or with an io device's, or a device's copy
   queue between memory it maps. Memory an io device owns moves only through
   that device's read or write, never through a borrow of its pages, so a
   file cut short raises in the read where a read of its pages would end the
   process, and no device maps the pages for a copy. A copy on a queue
   returns at once unless [wait]: what reads or writes its buffers later
   waits for it by their stamps. The host records its transfer, a submit a
   queued one. Is whether it ran. *)
let direct ~wait ~into_half src dst n =
  let si = io_owned src.mem and di = io_owned dst.mem in
  let start = Profile.now () in
  if local src.mem && local dst.mem && not (si || di) then
    on_host ~into_half Move src dst n start
  else if si && (not di) && local dst.mem then
    on_host ~into_half Io_read src dst n start
  else if di && (not si) && local src.mem then
    on_host ~into_half Io_write src dst n start
  else if si || di then false
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

let pieces n = (n + half_bytes - 1) / half_bytes
let piece_bytes n k = Int.min half_bytes (n - (k * half_bytes))

(* The bytes of a half from its piece's start that a copy of [n] bytes uses:
   those of its one piece, or the whole half for more. With a half's base, its
   cursor once the copy gives it back. *)
let extent n = if pieces n = 1 then round n else half_bytes

(* [f h0 h1] on two halves of [d]'s staging memory that [writer] writes, or
   one, [h0] and [h1] the same, unless [two]; the halves given back once it
   returns or raises. A half [f] returns normally has its cursor set by [f].
   Nothing allocates between a take and the handler that gives it back. *)
let holding d writer need ~two f =
  let h0 = take d writer need in
  match if two then take d writer 0 else h0 with
  | exception e ->
      abandon h0;
      reraise e
  | h1 -> (
      match f h0 h1 with
      | () ->
          give h0;
          if two then give h1;
          age d
      | exception e ->
          abandon h0;
          if two then abandon h1;
          reraise e)

(* Moves [n] bytes through halves of [d]'s staging memory: [into h at] fills
   the half's view [h] with the bytes from byte [at], and [out_of h at] drains
   it. One piece goes through one half, from its cursor where the host writes
   it; more alternate between two halves from their start, so one fills while
   the other drains, and the host waits for a half's last drain before it
   fills it again. The halves are first mapped on the devices that run legs
   of [sides]. A leg on a device's queue runs without the host waiting for it:
   the next leg that reads or writes its half follows it by the half's stamps.
   With [ahead], the leg into a half runs one piece ahead of the leg out of
   it, as when a device fills and the host drains; otherwise the host's leg
   into a half fills one while a device drains the other. *)
let through d n ~writer ~sides ~ahead ~into ~out_of =
  let pieces = pieces n in
  let need = if pieces = 1 && writer = Host then extent n else 0 in
  holding d writer need ~two:(pieces > 1) (fun h0 h1 ->
      let b0 = ready h0 writer in
      let b1 = if pieces > 1 then ready h1 writer else b0 in
      let base0 = h0.cursor in
      List.iter (fun b -> map_halves d b b0 b1) sides;
      let half k =
        Buffer.view
          (if k land 1 = 0 then b0 else b1)
          ~first:(if k = 0 then base0 else 0)
          ~length:(piece_bytes n k)
      in
      let into k =
        if writer = Host && k >= 2 then Buffer.wait (half k) Buffer.Read_write;
        into (half k) (k * half_bytes)
      in
      let out_of k = out_of (half k) (k * half_bytes) in
      if ahead then into 0;
      for k = 0 to pieces - 1 do
        if ahead then (if k + 1 < pieces then into (k + 1)) else into k;
        out_of k
      done;
      h0.cursor <- base0 + extent n;
      h1.cursor <- base0 + extent n)

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
  if direct ~wait ~into_half:false src dst n then ()
  else if (far sd || far dd) && not (hosted src || hosted dst) then
    invalid_argf "Rig.%s: no device copies between %s and %s" fn sd.name
      dd.name
  else staged src dst n

(* Through staging memory. Where a device runs the legs out of [src], it
   writes the halves, each piece one ahead. *)
and staged src dst n =
  if two_stages src dst then between src dst n
  else
    through (stage_for src dst) n
      ~writer:(if runs src then Device else Host)
      ~sides:[ src; dst ] ~ahead:(runs src)
      ~into:(fun h at -> leg ~into_half:true ~src:(piece src at h) ~dst:h)
      ~out_of:(fun h at -> leg ~into_half:false ~src:h ~dst:(piece dst at h));
  Buffer.wait dst Buffer.Read_write

(* Between two devices that each stage through their own memory: [src]'s
   device fills a half of its stage, the host moves the half's bytes into a
   half of [dst]'s device's stage, and that device drains it. The leg into the
   next half runs while the host moves the last. *)
and between src dst n =
  let pieces = pieces n in
  let need = if pieces = 1 then extent n else 0 in
  holding (runner src) Device 0 ~two:(pieces > 1) (fun a0 a1 ->
      holding (runner dst) Host need ~two:(pieces > 1) (fun b0 b1 ->
          let ha0 = ready a0 Device and hb0 = ready b0 Host in
          let ha1 = if pieces > 1 then ready a1 Device else ha0 in
          let hb1 = if pieces > 1 then ready b1 Host else hb0 in
          let base0 = b0.cursor in
          map_halves a0.stage src ha0 ha1;
          map_halves b0.stage dst hb0 hb1;
          let half h0 h1 base k =
            Buffer.view
              (if k land 1 = 0 then h0 else h1)
              ~first:(if k = 0 then base else 0)
              ~length:(piece_bytes n k)
          in
          let into k =
            let h = half ha0 ha1 0 k in
            leg ~into_half:true ~src:(piece src (k * half_bytes) h) ~dst:h
          in
          into 0;
          for k = 0 to pieces - 1 do
            if k + 1 < pieces then into (k + 1);
            let ha = half ha0 ha1 0 k and hb = half hb0 hb1 base0 k in
            if k >= 2 then Buffer.wait hb Buffer.Read_write;
            leg ~into_half:true ~src:ha ~dst:hb;
            leg ~into_half:false ~src:hb ~dst:(piece dst (k * half_bytes) hb)
          done;
          a0.cursor <- extent n;
          a1.cursor <- extent n;
          b0.cursor <- base0 + extent n;
          b1.cursor <- base0 + extent n));
  Buffer.wait dst Buffer.Read_write

(* One leg of a staged copy, which a device's queue runs without the host
   waiting. A leg runs in one transfer by construction: one that finds none,
   such as into a device that runs no copy, goes no further. *)
and leg ~into_half ~src ~dst =
  if not (direct ~wait:false ~into_half src dst (Buffer.length src)) then
    invalid_argf "Rig.%s: no device copies between %s and %s" fn
      src.mem.dev.name dst.mem.dev.name

(* The host's values *)

let writable fn b =
  if Buffer.access b = Read then
    invalid_argf "Rig.%s: the buffer admits only reads" fn

(* The bytes of [dst], which the host does not address, through staging
   memory, holding a read claim on it as a copy does: [put h at] writes the
   bytes from [dst]'s byte [at] into the half [h], which no unfinished work
   uses. *)
let upload fn dst put =
  Claim.reading fn dst (fun () ->
      Memory.check_owner dst.mem;
      Memory.drain dst.mem.dev;
      let n = Buffer.length dst in
      if n > 0 then begin
        through (stage_with dst) n ~writer:Host ~sides:[ dst ] ~ahead:false
          ~into:put ~out_of:(fun h at ->
            leg ~into_half:false ~src:h
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
        through (stage_with src) n ~writer:Device ~sides:[ src ] ~ahead:true
          ~into:(fun h at ->
            leg ~into_half:true
              ~src:(Buffer.view src ~first:at ~length:(Buffer.length h))
              ~dst:h)
          ~out_of:take)

(* Whether the host moves the bytes of [b] for a blit or a fill itself: [b]'s
   memory is addressed by the host and no io device owns it, whose memory
   moves through its read or write, as in [direct]. *)
let host_moved b = Buffer.host_addressed b && not (io_owned b.mem)

(* Where the host moves the buffer's bytes, the blits and [fill] claim and
   wait, move the bytes, and drop the claim. The move neither raises nor polls
   in native code, so no handler is needed and no asynchronous exception keeps
   the claim. *)
let blit_from_string s i b j n =
  let fn = "Buffer.blit_from_string" in
  Dev.check_held fn;
  Buffer.check_range fn "string" (String.length s) i n;
  Buffer.check_range fn "buffer" (Buffer.length b) j n;
  writable fn b;
  if host_moved b then begin
    Claim.wait fn Buffer.Read_write b;
    Buffer.blit_string s i b j n;
    Memory.claim_drop b
  end
  else begin
    (* The view would otherwise name itself for a dead buffer; the check that
       counts runs under the claim. *)
    Buffer.check_live fn b;
    upload fn (Buffer.view b ~first:j ~length:n) (fun h at ->
        Buffer.blit_string s (i + at) h 0 (Buffer.length h))
  end

let blit_to_bytes b i s j n =
  let fn = "Buffer.blit_to_bytes" in
  Dev.check_held fn;
  Buffer.check_range fn "buffer" (Buffer.length b) i n;
  Buffer.check_range fn "bytes" (Bytes.length s) j n;
  if host_moved b then begin
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

(* The bytes an io device's fill writes at a time, from host memory holding
   the byte. *)
let io_fill_bytes = 1 lsl 20

(* Fills [b], memory an io device owns, with [c] through the io device's
   writes, after the work it follows. *)
let fill_io b c =
  let n = Buffer.length b in
  let k = Int.min n io_fill_bytes in
  let h = Buffer.create Dev.host k in
  Buffer.set_bytes h 0 k c;
  Buffer.wait_points b Buffer.Read_write;
  let rec go at =
    if at < n then begin
      let len = Int.min k (n - at) in
      io_write h (Buffer.view b ~first:at ~length:len) len;
      go (at + len)
    end
  in
  go 0

(* Fills [b] with [c] on the copy engine of the device that runs its legs
   ([runner]), reading no memory, and waits for it. *)
let fill_on_engine fn b c =
  let d = runner b in
  match (d.live.copy_engine, mine d b.mem) with
  | Some _, Some m when not (Dev.is_io d) ->
      Dev.wait d (Point.value (Submission.run_set d ~dst:{ b with mem = m } c))
  | _ -> invalid_argf "Rig.%s: no device sets the bytes of %s" fn b.mem.dev.name

(* Where the host addresses [b], the host sets its bytes; where [b] is an io
   device's memory, the io device writes them; otherwise a copy engine sets
   them. Each holds a read claim on [b] while it runs. *)
let fill b c =
  let fn = "Buffer.fill" in
  Dev.check_held fn;
  writable fn b;
  if host_moved b then begin
    Claim.wait fn Buffer.Read_write b;
    Buffer.set_bytes b 0 (Buffer.length b) c;
    Memory.claim_drop b
  end
  else begin
    Buffer.check_live fn b;
    Claim.reading fn b (fun () ->
        Memory.check_owner b.mem;
        Memory.drain b.mem.dev;
        if Buffer.length b > 0 then
          if io_owned b.mem then fill_io b c else fill_on_engine fn b c)
  end
