(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The routes are copy.mli's. Where two devices could copy, [direct] picks
   [dst]'s device when only [src] is memory the host addresses, and [src]'s
   otherwise. A route returns the point of its last transfer, [0] where the
   host made it, which the public calls turn into a point of the host. *)

open Def

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

external memmove : int -> int -> int -> unit = "caml_rig_memmove"
external reraise : exn -> 'a = "%reraise"

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

(* The points of [waits], waited for on the host. *)
let wait_all waits =
  for i = 0 to Array.length waits - 1 do
    Dev.wait_point 0 true (Array.unsafe_get waits i)
  done

(* Boxes *)

let row ?(src_offset = 0) ?(dst_offset = 0) n =
  { src_offset; dst_offset; width = n; rows = 1; src_pitch = n; dst_pitch = n }

let has_bytes x = x.width > 0 && x.rows > 0
let box_bytes x = x.width * x.rows
let total boxes = Array.fold_left (fun n x -> n + box_bytes x) 0 boxes

(* Whether the rows of [x] are one run of bytes on both sides. *)
let one_run x = x.rows = 1 || (x.src_pitch = x.width && x.dst_pitch = x.width)

(* Whether the last row of [x], from [off] at [pitch] on a side of [len]
   bytes, ends within it, without overflow: [pitch] is at least the width,
   which is positive. *)
let ends_within x ~off ~pitch len =
  off <= len && x.width <= len - off
  && x.rows - 1 <= (len - off - x.width) / pitch

(* The boxes of [boxes] that have bytes, read once, or [Invalid_argument]
   for one that does not fit [src] and [dst]. *)
let fitting fn ~src ~dst boxes =
  let fits i x =
    if
      x.src_offset < 0 || x.dst_offset < 0 || x.width < 0 || x.rows < 0
      || x.src_pitch < x.width || x.dst_pitch < x.width
    then
      invalid_argf "Rig.%s: box %d has a negative field or a pitch under its \
                    width"
        fn i;
    if
      has_bytes x
      && not
           (ends_within x ~off:x.src_offset ~pitch:x.src_pitch
              (Buffer.length src)
           && ends_within x ~off:x.dst_offset ~pitch:x.dst_pitch
                (Buffer.length dst))
    then invalid_argf "Rig.%s: box %d ends past a buffer" fn i
  in
  let boxes = Array.to_list boxes in
  List.iteri fits boxes;
  Array.of_list (List.filter has_bytes boxes)

(* The bytes of [b] from [at], [n] of them: [b] itself when they are all of
   it. *)
let sub b at n =
  if at = 0 && n = Buffer.length b then b else Buffer.view b ~first:at ~length:n

(* Transfers *)

(* An io device's read or write of memory it owns, a borrow of its pages
   included, is a counted call of that device, so its fault loses the io
   device. Each moves [n] bytes between byte [s] of [src] and byte [t] of
   [dst]. *)
let io_read fn src s dst t n =
  match src.mem.alloc.entry.backing with
  | Io_memory { region = Io_region { m; h; r }; _ } ->
      let module I = (val m) in
      Dev.counted src.mem.alloc.owner Dev.Faults (fun () ->
          I.read h r ~at:(src.offset + s) ~dst:(host_address dst + t) ~len:n)
  | Driver_memory _ | Kept _ ->
      invalid_argf "Rig.%s: the source is no io memory" fn

let io_write fn src s dst t n =
  match dst.mem.alloc.entry.backing with
  | Io_memory { region = Io_region { m; h; r }; _ } ->
      let module I = (val m) in
      Dev.counted dst.mem.alloc.owner Dev.Faults (fun () ->
          I.write h r ~at:(dst.offset + t) ~src:(host_address src + s) ~len:n)
  | Driver_memory _ | Kept _ ->
      invalid_argf "Rig.%s: the destination is no io memory" fn

(* How the host moves bytes: between memory it addresses, or by an io device's
   read or write. *)
type host = Move | Io_read | Io_write

(* How one transfer copies between two buffers: the host, a device's copy
   engine ([Queued]), or none, where an io device owns one side and the other
   is memory the host does not address ([Staged]). Each is a constant, so
   choosing allocates nothing. *)
type transfer = Host of host | Queued | Staged

(* Memory an io device owns moves only through that device's read or write,
   never through a borrow of its pages, so a file cut short raises in the read
   where a read of its pages would end the process, and no device maps the
   pages for a copy. *)
let transfer src dst =
  let si = io_owned src.mem and di = io_owned dst.mem in
  if local src.mem && local dst.mem && not (si || di) then Host Move
  else if si && (not di) && local dst.mem then Host Io_read
  else if di && (not si) && local src.mem then Host Io_write
  else if si || di then Staged
  else Queued

let move_run fn transfer src s dst t n =
  match transfer with
  | Move -> memmove (host_address dst + t) (host_address src + s) n
  | Io_read -> io_read fn src s dst t n
  | Io_write -> io_write fn src s dst t n

(* The host's move of [x]'s rows, in one call where they are one run. *)
let move_box fn transfer src dst x =
  if one_run x then
    move_run fn transfer src x.src_offset dst x.dst_offset (box_bytes x)
  else
    for i = 0 to x.rows - 1 do
      move_run fn transfer src
        (x.src_offset + (i * x.src_pitch))
        dst
        (x.dst_offset + (i * x.dst_pitch))
        x.width
    done

(* The host's copy of [boxes] from [src] to [dst], after the points of
   [waits] and the work it follows, asked at [start]. Into a staging half
   ([into_half]) it waits for nothing on [dst]: no unfinished work uses the
   bytes it writes ([through]). *)
let on_host fn ~waits ~into_half transfer src dst boxes start =
  wait_all waits;
  Buffer.wait_points src Buffer.Read;
  if not into_half then Buffer.wait_points dst Buffer.Read_write;
  for i = 0 to Array.length boxes - 1 do
    move_box fn transfer src dst (Array.unsafe_get boxes i)
  done;
  record src dst (total boxes) start;
  Some 0

(* [m] as [d]'s copy engine takes it: a device of another machine takes this
   process's memory as it is, its driver carrying the bytes; another device
   maps it. *)
let mine d m =
  if (not (Dev.same_machine d Dev.host)) && local m then Some m
  else Memory.borrow d m

(* A copy of [boxes] on [d]'s copy engine between buffers [d] maps, after the
   points of [waits], as one submission. A box of one run goes as [d]'s
   cached copy of the run. Its submit records it. *)
let on_queue ~waits d src dst boxes =
  match (d.live.copy_engine, mine d src.mem, mine d dst.mem) with
  | Some _, Some s, Some t -> (
      let src = { src with mem = s } and dst = { dst with mem = t } in
      match boxes with
      | [| x |] when one_run x ->
          let n = box_bytes x in
          Some
            (Submission.run_copy d ~waits ~src:(sub src x.src_offset n)
               ~dst:(sub dst x.dst_offset n))
      | _ -> Some (Submission.run_boxes d ~waits ~src ~dst boxes))
  | _ -> None

(* The device whose copy engine runs a transfer to or from [b]: [b]'s
   device's, or, for a borrow on a device that runs no copy, its memory's own
   device's. *)
let runner b =
  match b.mem.dev.live.copy_engine with
  | Some _ -> b.mem.dev
  | None -> b.mem.alloc.owner

(* The side of a queued copy between [src] and [dst] whose [runner] runs it. *)
let queue_side src dst = if local src.mem then dst else src

(* Copies [boxes] from [src] to [dst] in one transfer, where one runs it: the
   host between memory it addresses or with an io device's, or a device's copy
   queue between memory it maps ([transfer]). What reads or writes the
   buffers later follows a queued copy by their stamps. The host records its
   transfer, a submit a queued one. Is the transfer's point, [0] for the
   host's, or [None] where no transfer runs it. *)
let direct fn ~waits ~into_half src dst boxes =
  let start = Profile.now () in
  match transfer src dst with
  | Host t -> on_host fn ~waits ~into_half t src dst boxes start
  | Staged -> None
  | Queued ->
      let d = runner (queue_side src dst) in
      if Dev.is_io d then None else on_queue ~waits d src dst boxes

(* One leg of a staged copy, which a device's queue runs without the host
   waiting. A leg runs in one transfer by construction: one that finds none,
   such as into a device that runs no copy, goes no further. *)
let leg fn ~waits ~into_half src dst boxes =
  match direct fn ~waits ~into_half src dst boxes with
  | Some p -> p
  | None ->
      invalid_argf "Rig.%s: no device copies between %s and %s" fn
        src.mem.dev.name dst.mem.dev.name

(* Staging memory

   As copy.mli states it. A half's [cursor] is the byte from which no
   unfinished work uses its buffer, so the host writes a piece there without
   waiting, and the leg that drains it follows the host's write by the
   stamps, as work after a host write does. A device's write of a half
   follows every use of the whole allocation (rule 3), so a piece a device
   writes takes a half whose uses are done, or a new one: packed, a device's
   leg into a half would follow another device's drain of it, and the two
   devices' copies would chain. Nor is a piece the host writes packed behind
   a device's write: the host's write adds no point to the stamps, so the
   half's last write stays the device's leg, and the drain of the host's
   piece would follow that unrelated work, and its device's loss. A half a
   device wrote goes back with its cursor at its end, and its lap restarts at
   its start once its work is done, which asks nothing more of a take. Pieces
   start at multiples of [piece_align] bytes in their half. *)

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

let has_room need h =
  h.buffer <> None && h.cursor + need <= half_bytes

let uses_done h =
  match h.buffer with None -> true | Some b -> Memory.uses_done b.mem

let later a b = a.lap > b.lap
let earlier a b = a.lap < b.lap
let buffered a b = a.buffer <> None && b.buffer = None
let some_buffer h = h.buffer <> None

(* [find] of the free half with room for [need] bytes after its cursor whose
   lap started last, a loop of its own so that a take makes no closure. *)
let rec find_room l need i found =
  if i = Array.length l.halves then found
  else
    let h = l.halves.(i) in
    let found =
      if
        (not h.held) && has_room need h
        && (found < 0 || later h l.halves.(found))
      then i
      else found
    in
    find_room l need (i + 1) found

(* Under [l]'s lock: [h] held from now, its lap restarted at byte 0 if
   [restart], its holder waiting for its uses first if [waits]. *)
let hold l ~restart ~waits h =
  h.held <- true;
  h.last <- l.takes;
  h.waits <- waits;
  if restart then begin
    l.laps <- l.laps + 1;
    h.lap <- l.laps;
    h.cursor <- 0
  end;
  h

(* Under [d]'s lock: a half of [d]'s staging memory for the pieces [writer]
   writes, held now. A piece of [need] bytes that the host writes, [need]
   positive, takes the free half with room after its cursor whose lap started
   last, from its start if its uses are seen done: no work uses any of it, and
   the last copy left those bytes in the caches, where past the cursor each of a
   run of copies would write bytes that no copy touched for a lap of the half.
   Otherwise a half is taken at its start: a free one whose uses are done, one
   with a buffer first; for the host, else the free half whose lap started
   first, whose uses its holder waits for; else a new half, so no copy waits for
   a hold. A new half is published by the last store, so an asynchronous
   exception before it changes nothing. It makes no buffer, whose creation can
   run the out-of-memory ladder, which takes every device's lock. *)
let take_locked d writer need =
  let l = d.live in
  l.takes <- l.takes + 1;
  let packed =
    if writer = Host && need > 0 then find_room l need 0 (-1) else -1
  in
  if packed >= 0 then
    let h = l.halves.(packed) in
    hold l ~restart:(uses_done h) ~waits:false h
  else
    let i = find l uses_done buffered 0 (-1) in
    if i >= 0 then hold l ~restart:true ~waits:false l.halves.(i)
    else
      let i = if writer = Host then find l some_buffer earlier 0 (-1) else -1 in
      if i >= 0 then hold l ~restart:true ~waits:true l.halves.(i)
      else begin
        l.laps <- l.laps + 1;
        let h =
          {
            buffer = None;
            held = true;
            cursor = 0;
            lap = l.laps;
            last = l.takes;
            waits = false;
            stage = d;
          }
        in
        l.halves <- Array.append l.halves [| h |];
        h
      end

(* [take_locked] under [d]'s lock, its arguments in no tuple. *)
let take d writer need =
  let lock = d.live.lock in
  Lock.hold lock;
  match take_locked d writer need with
  | h ->
      Lock.release lock;
      h
  | exception e ->
      Lock.release lock;
      reraise e

let release h = h.held <- false

(* Gives [h] back, its cursor set by its holder, which reads it no more.
   Allocates nothing, so no asynchronous exception keeps the half. *)
let give h = Dev.protect h.stage release h

(* Gives [h] back with its uses unknown: no piece is packed into it until its
   lap restarts. *)
let abandon h =
  h.cursor <- half_bytes;
  give h

(* Under [d]'s lock: gives [h0] and [h1] back, as [give] does, then takes
   out the buffers of [d]'s free halves that no take took within the last
   takes, as many as [d] has halves, each list cell made before the stores.
   Nothing allocates until the halves are given. *)
let give_locked d h0 h1 =
  release h0;
  release h1;
  let l = d.live in
  let n = Array.length l.halves in
  let bs = ref [] in
  for i = 0 to n - 1 do
    let h = l.halves.(i) in
    match h.buffer with
    | Some b when (not h.held) && l.takes - h.last >= n ->
        bs := b :: !bs;
        h.buffer <- None;
        h.cursor <- 0
    | Some _ | None -> ()
  done;
  !bs

(* Gives [h0] and [h1] back, the same half for one, and returns the buffers
   of [d]'s idle halves as memory returns free halves ([Memory.return_half]),
   outside the lock, in one hold of it. Copies that keep a half busy, one
   after another or in lockstep on several threads, take it within every
   window and keep its buffer; the extra halves of a burst go idle once as
   many later takes as there are halves have passed them by. An exception
   meanwhile leaves the buffers it took to their collection. *)
let give_back d h0 h1 =
  let lock = d.live.lock in
  Lock.hold lock;
  match give_locked d h0 h1 with
  | bs ->
      Lock.release lock;
      List.iter Memory.return_half bs
  | exception e ->
      Lock.release lock;
      reraise e

(* The buffer of [h], made where it is missing or names a lost device's work,
   outside every lock. A half whose take did not see its uses done ([waits])
   waits for them first; a loss meanwhile replaces it. The uses a take saw
   done stay so: no use of a free half is added until a holder gives it
   back. *)
let ready h =
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
      if not h.waits then b
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
let map_half fn stage d b =
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
let map_halves fn stage b b0 b1 =
  let d = b.mem.dev in
  if runs b && Dev.same_machine d Dev.host then begin
    map_half fn stage d b0;
    if b1 != b0 then map_half fn stage d b1
  end

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

(* Pieces

   A staged copy's boxes go through staging memory in pieces of at most a
   half. A piece holds whole rows, of one box or of several, packed one after
   the other, where a row fits a half, and a range of one row's bytes
   otherwise, so each leg of a piece is one box per part of a box it holds.
   Each part is a box of the copy's geometry, with the byte of the piece
   where its rows start. *)
type piece = { bytes : int; parts : (box * int) list }

let pieces boxes =
  let out = ref [] and parts = ref [] and fill = ref 0 in
  let close () =
    if !parts <> [] then begin
      out := { bytes = !fill; parts = List.rev !parts } :: !out;
      parts := [];
      fill := 0
    end
  in
  let add x =
    parts := (x, !fill) :: !parts;
    fill := !fill + box_bytes x
  in
  let wide x =
    close ();
    for i = 0 to x.rows - 1 do
      let c = ref 0 in
      while !c < x.width do
        let w = Int.min half_bytes (x.width - !c) in
        add
          {
            src_offset = x.src_offset + (i * x.src_pitch) + !c;
            dst_offset = x.dst_offset + (i * x.dst_pitch) + !c;
            width = w;
            rows = 1;
            src_pitch = w;
            dst_pitch = w;
          };
        close ();
        c := !c + w
      done
    done
  in
  let rows x =
    let r = ref 0 in
    while !r < x.rows do
      let fit = (half_bytes - !fill) / x.width in
      if fit = 0 then close ()
      else begin
        let k = Int.min fit (x.rows - !r) in
        add
          {
            x with
            src_offset = x.src_offset + (!r * x.src_pitch);
            dst_offset = x.dst_offset + (!r * x.dst_pitch);
            rows = k;
          };
        r := !r + k
      end
    done
  in
  Array.iter (fun x -> if x.width > half_bytes then wide x else rows x) boxes;
  close ();
  Array.of_list (List.rev !out)

(* The boxes of the leg that moves [p] from the copy's source into a half
   from byte [base], and out of it into the destination. *)
let into_half base p =
  Array.of_list
    (List.map
       (fun (x, at) -> { x with dst_offset = base + at; dst_pitch = x.width })
       p.parts)

let out_of_half base p =
  Array.of_list
    (List.map
       (fun (x, at) -> { x with src_offset = base + at; src_pitch = x.width })
       p.parts)

(* The bytes of a half from its piece's start past which [writer]'s pieces
   [ps] leave nothing to pack behind: those of one piece the host writes, and
   the whole half for more pieces or a device's. With a half's base, its
   cursor once the copy gives it back. *)
let extent writer ps =
  if writer = Host && Array.length ps = 1 then round ps.(0).bytes
  else half_bytes

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
      | p ->
          give_back d h0 h1;
          p
      | exception e ->
          abandon h0;
          if two then abandon h1;
          reraise e)

(* Moves [ps] through halves of [d]'s staging memory: [into b base p] moves
   the piece [p] into the half buffer [b] from byte [base], and [out_of b base
   p] out of it, each the point of its transfer. One piece goes through one
   half, from its cursor where the host writes it; more alternate between two
   halves from their start, so one fills while the other drains, and the host
   waits for a half's last drain before it fills it again. The halves are
   first mapped on the devices that run legs of [sides]. With [ahead], the leg
   into a half runs one piece ahead of the leg out of it, as when a device
   fills and the host drains; otherwise the host's leg into a half fills one
   while a device drains the other. Is the point of the last leg out. *)
let through fn d ps ~writer ~sides ~ahead ~into ~out_of =
  let n = Array.length ps in
  let need = if n = 1 && writer = Host then extent Host ps else 0 in
  holding d writer need ~two:(n > 1) (fun h0 h1 ->
      let b0 = ready h0 in
      let b1 = if n > 1 then ready h1 else b0 in
      let base0 = h0.cursor in
      List.iter (fun b -> map_halves fn d b b0 b1) sides;
      let half k = if k land 1 = 0 then b0 else b1 in
      let base k = if k = 0 then base0 else 0 in
      let fill k =
        if writer = Host && k >= 2 then Buffer.wait (half k) Buffer.Read_write;
        ignore (into (half k) (base k) ps.(k) : int)
      in
      let last = ref 0 in
      if ahead then fill 0;
      for k = 0 to n - 1 do
        if ahead then (if k + 1 < n then fill (k + 1)) else fill k;
        last := out_of (half k) (base k) ps.(k)
      done;
      h0.cursor <- base0 + extent writer ps;
      h1.cursor <- base0 + extent writer ps;
      !last)

(* Copies *)

(* Through staging memory, from a device's or the host's side into the
   other's. *)
let rec staged fn ~waits src dst boxes =
  let ps = pieces boxes in
  if two_stages src dst then between fn ~waits src dst ps
  else
    through fn (stage_for src dst) ps
      ~writer:(if runs src then Device else Host)
      ~sides:[ src; dst ] ~ahead:(runs src)
      ~into:(fun h base p ->
        leg fn ~waits ~into_half:true src h (into_half base p))
      ~out_of:(fun h base p ->
        leg fn ~waits ~into_half:false h dst (out_of_half base p))

(* Between two devices that each stage through their own memory: [src]'s
   device fills a half of its stage, the host moves the half's bytes into a
   half of [dst]'s device's stage, and that device drains it. The leg into the
   next half runs while the host moves the last. *)
and between fn ~waits src dst ps =
  let n = Array.length ps in
  let need = if n = 1 then extent Host ps else 0 in
  holding (runner src) Device 0 ~two:(n > 1) (fun a0 a1 ->
      holding (runner dst) Host need ~two:(n > 1) (fun b0 b1 ->
          let ha0 = ready a0 and hb0 = ready b0 in
          let ha1 = if n > 1 then ready a1 else ha0 in
          let hb1 = if n > 1 then ready b1 else hb0 in
          let base0 = b0.cursor in
          map_halves fn a0.stage src ha0 ha1;
          map_halves fn b0.stage dst hb0 hb1;
          let half h0 h1 k = if k land 1 = 0 then h0 else h1 in
          let base k = if k = 0 then base0 else 0 in
          let into k =
            ignore
              (leg fn ~waits ~into_half:true src (half ha0 ha1 k)
                 (into_half 0 ps.(k))
                : int)
          in
          let last = ref 0 in
          into 0;
          for k = 0 to n - 1 do
            if k + 1 < n then into (k + 1);
            let ha = half ha0 ha1 k and hb = half hb0 hb1 k in
            if k >= 2 then Buffer.wait hb Buffer.Read_write;
            ignore
              (leg fn ~waits ~into_half:true ha hb
                 [| row ~dst_offset:(base k) ps.(k).bytes |]
                : int);
            last := leg fn ~waits ~into_half:false hb dst (out_of_half (base k) ps.(k))
          done;
          a0.cursor <- half_bytes;
          a1.cursor <- half_bytes;
          b0.cursor <- base0 + extent Host ps;
          b1.cursor <- base0 + extent Host ps;
          !last))

(* Copies [boxes] from [src] to [dst] through the staging memory, where no
   transfer copies them directly, recording each transfer that ran, and is
   the point of the last. *)
let undirected fn ~waits src dst boxes =
  let sd = src.mem.dev and dd = dst.mem.dev in
  if (far sd || far dd) && not (hosted src || hosted dst) then
    invalid_argf "Rig.%s: no device copies between %s and %s" fn sd.name
      dd.name
  else staged fn ~waits src dst boxes

(* [undirected]'s copy, directly where one transfer runs it. *)
let route fn ~waits src dst boxes =
  match direct fn ~waits ~into_half:false src dst boxes with
  | Some p -> p
  | None -> undirected fn ~waits src dst boxes

(* The checks of a copy that need no claim. An io device's write of its memory
   from a borrow of its own pages, or its read into them, can wait on itself
   in the system for good: a file's write from its own mapping on macOS's
   APFS. *)
let check fn src dst =
  if Buffer.overlaps src dst then invalid_argf "Rig.%s: the buffers overlap" fn;
  if
    Dev.is_io src.mem.dev <> Dev.is_io dst.mem.dev
    && src.mem.alloc == dst.mem.alloc
  then
    invalid_argf "Rig.%s: one buffer is a borrow of the other's io memory" fn;
  if Buffer.access dst = Read then
    invalid_argf "Rig.%s: the destination admits only reads" fn

(* Under read claims on both sides, which keep a donation from writing either
   while the copy routes and the host moves bytes. The claims' release after
   it keeps the buffers reachable: a read, a write or a move hands over
   addresses only, and a collection could otherwise free their memory under
   it. Is the copy's last point, [0] for the host's. *)
let prepare src dst =
  let sd = src.mem.dev and dd = dst.mem.dev in
  Memory.check_owner src.mem;
  Memory.check_owner dst.mem;
  Memory.drain sd;
  if dd != sd then Memory.drain dd

let claimed fn ~waits src dst boxes =
  prepare src dst;
  if Array.length boxes = 0 then begin
    wait_all waits;
    0
  end
  else route fn ~waits src dst boxes

(* A point of the host for [p], the point of a copy's last transfer, [0] for
   the host's, whose copy is done. *)
let point p = if p = 0 then Submission.mark () else p

let queue fn src dst (waits, boxes) = claimed fn ~waits src dst boxes

let submit_copy ?(waits = [||]) ~src ~dst boxes =
  let fn = "Buffer.submit_copy" in
  Dev.check_held fn;
  let boxes = fitting fn ~src ~dst boxes in
  check fn src dst;
  point (Claim.reading2 fn src dst (queue fn) (waits, boxes))

(* [copy]'s transfer of its [n] bytes on [d]'s queue, as [direct] picks
   [d], as [d]'s cached copy of one run, or through staging memory where [d]
   does not map both sides. *)
let queued_run fn d src dst n =
  match (d.live.copy_engine, mine d src.mem, mine d dst.mem) with
  | Some _, Some s, Some t ->
      Submission.run_copy d ~waits:[||] ~src:{ src with mem = s }
        ~dst:{ dst with mem = t }
  | _ -> undirected fn ~waits:[||] src dst [| row n |]

(* [copy] under its claims, through its wait. The host's copy of one run
   builds no box, so it allocates nothing, and a queue's builds no box
   either. *)
let copied src dst n =
  let fn = "Buffer.copy" in
  prepare src dst;
  if n > 0 then
    match transfer src dst with
    | Host t ->
        let start = Profile.now () in
        Buffer.wait_points src Buffer.Read;
        Buffer.wait_points dst Buffer.Read_write;
        move_run fn t src 0 dst 0 n;
        record src dst n start
    | Queued ->
        let d = runner (queue_side src dst) in
        let p =
          if Dev.is_io d then undirected fn ~waits:[||] src dst [| row n |]
          else queued_run fn d src dst n
        in
        if p <> 0 then Dev.wait_point 0 true p
    | Staged ->
        let p = route fn ~waits:[||] src dst [| row n |] in
        if p <> 0 then Dev.wait_point 0 true p

let copy ~src ~dst =
  let fn = "Buffer.copy" in
  Dev.check_held fn;
  let n = Buffer.length src in
  if n <> Buffer.length dst then
    invalid_argf "Rig.%s: %d bytes into %d" fn n (Buffer.length dst);
  check fn src dst;
  Claim.reading2 fn src dst copied n

let queued d ~src ~dst =
  Dev.wait d (Point.value (Submission.run_copy d ~waits:[||] ~src ~dst))

(* The host's values *)

let writable fn b =
  if Buffer.access b = Read then
    invalid_argf "Rig.%s: the buffer admits only reads" fn

(* The bytes of [dst], which the host does not address, through staging
   memory, holding a read claim on it as a copy does, and waiting for its
   last leg: [put h base at len] writes the [len] bytes from [dst]'s byte [at]
   into the half [h] from byte [base]. *)
let upload fn dst put =
  Claim.reading fn dst (fun () ->
      Memory.check_owner dst.mem;
      Memory.drain dst.mem.dev;
      let n = Buffer.length dst in
      if n > 0 then
        let p =
          through fn (stage_with dst) (pieces [| row n |]) ~writer:Host
            ~sides:[ dst ] ~ahead:false
            ~into:(fun h base p ->
              List.iter (fun (x, at) -> put h (base + at) x.dst_offset x.width)
                p.parts;
              0)
            ~out_of:(fun h base p ->
              leg fn ~waits:[||] ~into_half:false h dst (out_of_half base p))
        in
        if p <> 0 then Dev.wait_point 0 true p)

(* The bytes of [src], which the host does not address, through staging
   memory, holding a read claim on it: [take h base at len] reads the [len]
   bytes from [src]'s byte [at] out of the half [h] from byte [base], after
   the leg that wrote it. *)
let download fn src take =
  Claim.reading fn src (fun () ->
      Memory.check_owner src.mem;
      Memory.drain src.mem.dev;
      let n = Buffer.length src in
      if n > 0 then
        ignore
          (through fn (stage_with src) (pieces [| row n |]) ~writer:Device
             ~sides:[ src ] ~ahead:true
             ~into:(fun h base p ->
               leg fn ~waits:[||] ~into_half:true src h (into_half base p))
             ~out_of:(fun h base p ->
               Buffer.wait h Buffer.Read;
               List.iter
                 (fun (x, at) -> take h (base + at) x.src_offset x.width)
                 p.parts;
               0)
            : int))

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
    upload fn (Buffer.view b ~first:j ~length:n) (fun h base at len ->
        Buffer.blit_string s (i + at) h base len)
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
    download fn (Buffer.view b ~first:i ~length:n) (fun h base at len ->
        Buffer.blit_bytes h base s (j + at) len)
  end

(* Fills *)

(* The bytes an io device's fill writes at a time, from host memory holding
   the byte. *)
let io_fill_bytes = 1 lsl 20

(* Fills [b], memory an io device owns, with [c] through the io device's
   writes, after the work it follows. *)
let fill_io fn b c =
  let n = Buffer.length b in
  let k = Int.min n io_fill_bytes in
  let h = Buffer.create Dev.host k in
  Buffer.set_bytes h 0 k c;
  Buffer.wait_points b Buffer.Read_write;
  let rec go at =
    if at < n then begin
      let len = Int.min k (n - at) in
      io_write fn h 0 b at len;
      go (at + len)
    end
  in
  go 0

(* Fills [b] with [c] on the copy engine of the device that runs its legs
   ([runner]), reading no memory, after the points of [waits]: the fill's
   point. *)
let fill_on_engine fn ~waits b c =
  let d = runner b in
  match (d.live.copy_engine, mine d b.mem) with
  | Some _, Some m when not (Dev.is_io d) ->
      Submission.run_set d ~waits ~dst:{ b with mem = m } c
  | _ -> invalid_argf "Rig.%s: no device sets the bytes of %s" fn b.mem.dev.name

(* A fill of [b], which the host does not move, under a read claim on it: the
   io device writes an io device's memory, and a copy engine sets any other.
   Is its point, [0] for the host's. *)
let filled fn ~waits b c =
  Memory.check_owner b.mem;
  Memory.drain b.mem.dev;
  if Buffer.length b > 0 && not (io_owned b.mem) then
    fill_on_engine fn ~waits b c
  else begin
    wait_all waits;
    if Buffer.length b > 0 then fill_io fn b c;
    0
  end

(* Where the host addresses [b], the host sets its bytes after the points of
   [waits] and the work it follows, holding a read claim on [b]. *)
let fill_host fn ~waits b c =
  wait_all waits;
  Claim.wait fn Buffer.Read_write b;
  Buffer.set_bytes b 0 (Buffer.length b) c;
  Memory.claim_drop b

let submit_fill ?(waits = [||]) b c =
  let fn = "Buffer.submit_fill" in
  Dev.check_held fn;
  writable fn b;
  if host_moved b then begin
    fill_host fn ~waits b c;
    Submission.mark ()
  end
  else begin
    Buffer.check_live fn b;
    point (Claim.reading fn b (fun () -> filled fn ~waits b c))
  end

let fill b c =
  let fn = "Buffer.fill" in
  Dev.check_held fn;
  writable fn b;
  if host_moved b then fill_host fn ~waits:[||] b c
  else begin
    Buffer.check_live fn b;
    Claim.reading fn b (fun () ->
        let p = filled fn ~waits:[||] b c in
        if p <> 0 then Dev.wait_point 0 true p)
  end
