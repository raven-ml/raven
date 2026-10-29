(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Def
module Cache = Tbl.Make (Int)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* Stamps *)

external stamps_new : unit -> int = "caml_rig_stamps_new"
external stamps_ref : int -> unit = "caml_rig_stamps_ref" [@@noalloc]
external stamps_unref : int -> unit = "caml_rig_stamps_unref" [@@noalloc]
external stamps_followed : int -> int -> bool -> int
  = "caml_rig_stamps_followed"
[@@noalloc]

external stamps_keep : int -> int -> int = "caml_rig_stamps_keep" [@@noalloc]
external stamps_next : int -> int = "caml_rig_stamps_next" [@@noalloc]

(* The point folded into the stamps' head that every access follows, 0 for
   none: a lost write point of an earlier device of a slot, which a walk
   reads after the words. *)
external stamps_folded : int -> int = "caml_rig_stamps_folded"
[@@noalloc]
external stamps_clear : int -> unit = "caml_rig_stamps_clear" [@@noalloc]
external stamps_uses : unit -> int = "caml_rig_stamps_uses"

(* The number of use words, of write words and of keep words in a chunk of
   stamps. A walk reads each chunk's words from the first chunk, the stamps
   themselves, on, and ends where [stamps_followed] answers [ended]; a point
   0 is a word with no point. *)
let uses = stamps_uses ()
let ended = min_int

(* [f] over the use points of the chunk [c] from its [i]th word on, and of
   the chunks after it. *)
let rec iter_from f c i =
  if c <> 0 then
    if i = uses then iter_from f (stamps_next c) 0
    else
      let p = stamps_followed c i true in
      if p <> ended then begin
        if p <> 0 then f p;
        iter_from f c (i + 1)
      end

let iter_points f st =
  iter_from f st 0;
  let p = stamps_folded st in
  if p <> 0 then f p

(* Waits for the use points of [a]'s stamps if [every], their write points
   otherwise, from the chunk [c]'s [i]th word on, with no closure: the walk
   is a host access's every wait. [a], whose collection frees the stamps, is
   an argument of every step, so it stays reachable through every wait and
   every read. *)
let rec wait_from (a : alloc) every c i =
  if c <> 0 && i = uses then wait_from a every (stamps_next c) 0
  else
    let p = if c = 0 then ended else stamps_followed c i every in
    if p <> ended then begin
      if p <> 0 then Dev.wait_point a.entry.stamps every p;
      wait_from a every c (i + 1)
    end

(* The walk, then the point folded into the head. *)
let wait_all (a : alloc) every =
  let st = a.entry.stamps in
  wait_from a every st 0;
  let p = stamps_folded st in
  if p <> 0 then Dev.wait_point a.entry.stamps every p;
  ignore (Sys.opaque_identity a)

let wait_points m = wait_all m.alloc true
let wait_write m = wait_all m.alloc false

(* Whether the points [wait_from] waits for are done, from the chunk [c]'s
   [i]th word on, and [ok]: each point is asked, so that each device's work is
   committed and a point a lost device did not reach raises whatever point
   comes before it. *)
let rec reached_from (a : alloc) every c i ok =
  if c = 0 then ok
  else if i = uses then reached_from a every (stamps_next c) 0 ok
  else
    let p = stamps_followed c i every in
    if p = ended then ok
    else
      let ok = (p = 0 || Dev.reached_point p) && ok in
      reached_from a every c (i + 1) ok

let reached_all (a : alloc) every =
  let st = a.entry.stamps in
  let ok = reached_from a every st 0 true in
  let p = stamps_folded st in
  let ok = (p = 0 || Dev.reached_point p) && ok in
  ignore (Sys.opaque_identity a);
  ok

let reached_points m = reached_all m.alloc true
let reached_write m = reached_all m.alloc false

(* [reached_all]'s walk, asking [Dev.seen_done], which commits nothing and
   calls no driver. *)
let rec done_from c i =
  c = 0
  ||
  if i = uses then done_from (stamps_next c) 0
  else
    let p = stamps_followed c i true in
    p = ended || ((p = 0 || Dev.seen_done p) && done_from c (i + 1))

let uses_done m =
  let st = m.alloc.entry.stamps in
  let p = stamps_folded st in
  done_from st 0 && (p = 0 || Dev.seen_done p)

(* Visit every point even after a refusal: reading a later transport word may
   discover its device's loss. The walk carries no allocated state. *)
let rec all_from f c i ok =
  if c = 0 then ok
  else if i = uses then all_from f (stamps_next c) 0 ok
  else
    let p = stamps_followed c i true in
    if p = ended then ok
    else
      let ok = (p = 0 || f p) && ok in
      all_from f c (i + 1) ok

let for_all_points f st = all_from f st 0 true

let check_points st = iter_points Dev.check st

(* Raises [Lost] if [m]'s memory, or [m] itself as a borrow, is a lost
   device's. *)
let[@inline] check_owner m =
  let d = m.dev in
  if Dev.is_lost d then Dev.raise_lost d;
  let o = m.alloc.owner in
  if o != d && Dev.is_lost o then Dev.raise_lost o

(* Raises [Lost] for a use of [m] that depends on every point of its stamps. *)
let[@inline] check m =
  check_owner m;
  check_points m.alloc.entry.stamps

(* Whether [f] holds of each keep point of the chunk [c] from its [i]th word
   on, and of the chunks after it: the points of work that uses the memory
   unordered, which no access follows and the memory's return waits for. *)
let rec all_kept f c i =
  c = 0
  ||
  if i = uses then all_kept f (stamps_next c) 0
  else
    let p = stamps_keep c i in
    (p = 0 || f p) && all_kept f c (i + 1)

(* Whether every point of [st] has settled, its keep points included. *)
let reached st = for_all_points Dev.settled st && all_kept Dev.settled st 0

(* Tokens and release lists *)

(* A token on a release list, or [no_token] if the list is closed: its
   device's slot was freed, or is being freed. *)
external token : int -> released -> int -> int -> int -> token
  = "caml_rig_token"

external no_token : unit -> token = "%identity"

(* Under the device's lock: the node at the front of a list's hand, which a
   drain routes next, taking the list into the hand when it is empty, or [0]
   for none; and the front node's free, once its record is routed, which
   counts it routed: a device's slot frees once every token of its list is
   ({!Dev.free_slot}). A node keeps its record reachable until then. *)
external release_front : int -> int = "caml_rig_release_front" [@@noalloc]
external release_pop : int -> unit = "caml_rig_release_pop" [@@noalloc]
external node_record : int -> released = "caml_rig_node_record" [@@noalloc]
external released_any : int -> bool = "caml_rig_released_any" [@@noalloc]

(* Whether a token was collected onto the list of a device whose stop
   returned, or a stop returned, since the last call that answered [true]. *)
external ended_linked : unit -> bool = "caml_rig_ended_linked" [@@noalloc]

(* Asks the next drain to drain those devices again. *)
external ended_rearm : unit -> unit = "caml_rig_ended_rearm" [@@noalloc]
external page_size : unit -> int = "caml_rig_page_size"

let no_token = no_token ()
let page = page_size ()

(* The host's heap *)

external heap_init : int -> unit = "caml_rig_heap_init"
external heap_reserve : int -> int -> bool = "caml_rig_heap_reserve" [@@noalloc]

type bytes_ba =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external heap_bytes : int -> bytes_ba = "caml_rig_heap_bytes"
external heap_footprint : int -> int = "caml_rig_heap_footprint" [@@noalloc]
external heap_drop : unit -> unit = "caml_rig_heap_drop"
external heap_trim : int -> unit = "caml_rig_heap_trim"
external heap_release : int -> unit = "caml_rig_heap_release" [@@noalloc]
external heap_held : unit -> int = "caml_rig_heap_held" [@@noalloc]

external ba_address : ('a, 'b, 'c) Bigarray.Array1.t -> int
  = "caml_rig_bigarray_address"
[@@noalloc]

(* Host buffers of at least this many bytes start on a page, so devices can map
   them: a mapping takes whole pages, which another buffer's memory must not
   share. Aligning costs up to a page of slack, at most a quarter of the buffer.
   The heap keeps these for reuse once collected. *)
let aligned_from = Int.max (64 * 1024) (4 * page)
let () = heap_init aligned_from

(* Smaller ones of a line or more start on one (rig_heap.c), so threads that
   write pieces of whole lines share none: on fudge (Core Ultra 5 235) a 64³
   float32 product, one 16-column sliver a thread, took 2.4 us with its
   output 16 bytes past a line against 1.7 on one. *)
let heap_bytes n =
  if n > max_int - page then raise Stdlib.Out_of_memory else heap_bytes n

(* Memory records *)

let is_io_memory (e : entry) =
  match e.backing with
  | Io_memory _ -> true
  | Driver_memory _ | Kept _ -> false

let entry_ids = Atomic.make 0

let entry ?(access = Read_write) owner backing bytes stamps =
  {
    id = Atomic.fetch_and_add entry_ids 1;
    owner;
    backing;
    bytes;
    access;
    stamps;
    maps = [];
    unmaps = 1;
    pages = Unasked;
    proxy = 0;
    reached = false;
  }

let no_entry = entry Dev.host (Kept Nothing) 0 0

(* Claim words *)

let outside = 1
let read_only = 2

(* A generation's low bit is set iff its buffers admit only reads: the first
   one of memory that admits only reads, and the one a consumption of memory
   not held exclusive raises to. Each buffer carries its generation, so it
   carries its access in no word of its own. *)
let admits g : access = if g land 1 = 1 then Read else Read_write
let first_generation (e : entry) = if e.access = Read then 1 else 0

let one_claim = 8
let one_hold = 1 lsl 32
let exclusive = -1

(* Bit 2 is set while a consumption of memory that is not held exclusive
   checks for holds and raises the generation, in C. *)
let held (a : alloc) = Atomic.Loc.get [%atomic.loc a.count] >= one_hold

external claim_retry : buffer -> bool = "caml_rig_claim_add" [@@noalloc]

(* One compare-and-set in OCaml: with one domain running, the runtime's is a
   plain store, which a C take's locked one is not. A retry, after another
   domain's claim beat it, loops in C, since an OCaml loop's back edge
   polls. *)
let claim_add b =
  let a = b.mem.alloc in
  let w = Atomic.Loc.get [%atomic.loc a.count] in
  w >= 0
  && (Atomic.Loc.compare_and_set [%atomic.loc a.count] w (w + one_claim)
     || claim_retry b)

external claim_drop : buffer -> unit = "caml_rig_claim_drop" [@@noalloc]

(* The claim word of a new lifetime of [e] that keeps [keep]: outside the
   claims for the caller's bigarray and an io library's region, read-only for
   memory that admits only reads. *)
let claim_word keep (e : entry) =
  let reached =
    match keep with
    | Bigarray _ -> true
    | Nothing | Heap _ -> (
        match e.backing with
        | Io_memory { origin = Given; _ } -> true
        | Io_memory { origin = Made; _ } | Driver_memory _ | Kept _ -> false)
  in
  (if reached then outside else 0)
  lor if e.access = Read then read_only else 0

let rec mark_outside (a : alloc) =
  let w = Atomic.Loc.get [%atomic.loc a.count] in
  w >= 0
  && (w land outside <> 0
     || Atomic.Loc.compare_and_set [%atomic.loc a.count] w (w lor outside)
     || mark_outside a)

(* An owner's record of a new lifetime of [entry], with no token. Every
   argument is given, so a host buffer's records allocate nothing beside
   them. *)
let own ~keep ~host ~address ~handle dev entry =
  let count = claim_word keep entry in
  let alloc =
    {
      owner = dev;
      host;
      count;
      generation = first_generation entry;
      why = "";
      entry;
      token = no_token;
      keep;
    }
  in
  { dev; host; address; handle; alloc }

let make ?(keep = Nothing) ?(host = -1) ?(address = -1) ?(handle = 0n) dev
    entry =
  own ~keep ~host ~address ~handle dev entry

let memory_of alloc dev ~host ~address ~handle =
  { dev; host; address; handle; alloc }

(* Host memory's size is its bigarray's, other memory's its entry's:
   [no_entry], of no bytes, for memory of no bytes. *)
let size (a : alloc) =
  match a.keep with
  | Heap ba -> Bigarray.Array1.dim ba
  | Bigarray ba -> Bigarray.Array1.size_in_bytes ba
  | Nothing -> a.entry.bytes

let locate (Region { m; r; _ }) =
  let module D = (val m) in
  D.locate r

(* Allocation events *)

(* The host's buffers reserve against its budget in the heap, which counts
   them; a device's count is its own. *)
let allocated d = if d == Dev.host then heap_held () else d.live.used

let note d =
  if Profile.enabled () then
    Profile.record
      (Allocation
         { device = d; time = Profile.now (); allocated = allocated d })

(* Driver calls *)

type memory_kind = Rig_edge.memory = Device | Pinned | Mapped

(* Gives back a region [d] allocated or mapped, an io region or an image
   ([Dev.give]). *)
let free_region d (Region { m; h; r; _ }) =
  let module D = (val m) in
  Dev.give d Dev.Any (fun () -> D.free h r)

let free_io d (Io_region { m; h; r }) =
  let module I = (val m) in
  Dev.give d Dev.Faults (fun () -> I.free h r)

(* Linked code is unmapped once its record is collected. *)
let unload_now d = function
  | Loaded { m; h; i } ->
      let module D = (val m) in
      Dev.give d Dev.Any (fun () -> D.unload h i)
  | Linked _ -> Dev.Given

(* The budget memory of [kind] on [d] counts in. Pinned memory is [d]'s own
   where the host addresses [d]'s memory, and host memory elsewhere. *)
type budget = Device_budget | Host_budget | No_budget

let budget_of d = function
  | Device | Mapped -> Device_budget
  | Pinned -> if Dev.reaches Dev.host d then Device_budget else Host_budget

(* The budget [e] counts in: io memory its device made counts in the device's,
   memory a library or a keeper frees in none. *)
let entry_budget (e : entry) =
  match e.backing with
  | Driver_memory { memory; _ } -> budget_of e.owner memory
  | Io_memory { origin = Made; _ } -> Device_budget
  | Io_memory { origin = Given; _ } | Kept _ -> No_budget

let owns (e : entry) = entry_budget e = Device_budget

let account (a : alloc) =
  match entry_budget a.entry with
  | Host_budget -> Dev.host
  | Device_budget | No_budget -> a.owner

let take_device_room (d, n) =
  let l = d.live in
  l.used + n <= d.budget
  && begin
    l.used <- l.used + n;
    true
  end

(* Takes [n] bytes of the room of [budget] on [d], if it has them. *)
let take_room d budget n =
  match budget with
  | Device_budget -> Dev.protect d take_device_room (d, n)
  | Host_budget -> heap_reserve n Dev.host.budget
  | No_budget -> true

let give_room d budget n =
  match budget with
  | Device_budget ->
      let l = d.live in
      Lock.hold l.lock;
      l.used <- l.used - n;
      Lock.release l.lock
  | Host_budget -> heap_release n
  | No_budget -> ()

(* Releases *)

(* The queue [q] of releases with [p] at the value [v], after the releases
   of [v] or less. A release at the end, as one at the submitted value
   nearly always is, takes a cons. One before others, such as a release at
   the submitted value after an unmap of a word at the next value
   ([retire_locked]), rebuilds the queue. *)
let insert (front, back) v p =
  let rec last = function
    | [ (v, _) ] -> v
    | _ :: rest -> last rest
    | [] -> min_int
  in
  let latest = match back with (v, _) :: _ -> v | [] -> last front in
  if front == [] then ([ (v, p) ], [])
  else if v >= latest then (front, (v, p) :: back)
  else
    let before, after =
      List.partition (fun (v', _) -> v' <= v) (front @ List.rev back)
    in
    (before @ ((v, p) :: after), [])

(* Puts [p] after the work that may use it without naming it. Reading the
   submitted value under the lock keeps the queue's values ordered. *)
let append_pending (d, p) =
  let l = d.live in
  l.pending <- insert l.pending (Dev.submitted d) p

let defer d p = Dev.protect d append_pending (d, p)

let retire d e = defer d (Free e)

(* [waits d p given] is [given], [d]'s release [p] having gone back. One that
   did not waits among [d]'s pending releases, after the others, until [d]
   stopped ([Dev.give]). A caller that makes [p] makes it only for a release
   that did not go back, so a free allocates none. *)
let waits d p (given : Dev.given) =
  (match given with Given -> () | Later -> defer d p);
  given

(* Gives back [d]'s mapping [r] now, or later ([waits]). *)
let unmap d r : Dev.given =
  match free_region d r with
  | Given -> Given
  | Later -> waits d (Unmap (r, None)) Later

(* The first address of [l] off a multiple of [align], if any. *)
let misplaced align (l : Rig_edge.location) =
  let off = function
    | Some a when a land (align - 1) <> 0 -> Some a
    | _ -> None
  in
  match off l.address with Some a -> Some a | None -> off l.host

(* Why the [n]-byte region at [l] breaks the facts [d]'s driver states, if it
   does. *)
let breaks d n (l : Rig_edge.location) =
  if d.host_addresses && l.host = None then
    Some
      "a region without a host address, where its driver states the host \
       addresses its memory"
  else
    match misplaced d.alignment l with
    | Some a ->
        Some
          (Printf.sprintf
             "%d bytes placed at 0x%x, off the alignment of %d its driver \
              states"
             n a d.alignment)
    | None -> None

(* [n] new bytes of [d]'s memory of [kind] as a release record, or [None] if [d]
   has not the room. An io device's memory has no kind. A region that breaks the
   facts its driver states is freed and faults the device, as a fault of its
   driver's [alloc] does. *)
let new_entry d (kind : memory_kind) n =
  match d.kind with
  | Driver { m; h; rid } -> (
      let module D = (val m) in
      let alloc () =
        match D.alloc h kind n with
        | Some r as some -> (
            match breaks d n (D.locate r) with
            | None -> some
            | Some why ->
                D.free h r;
                raise (D.Fault why))
        | None -> None
      in
      match Dev.counted d Dev.Any alloc with
      | None -> None
      | Some r ->
          let region = Region { m; h; r; rid } in
          Some
            (entry d (Driver_memory { memory = kind; region }) n (stamps_new ())))
  | Io { m; h } -> (
      let module I = (val m) in
      match Dev.counted d Dev.Faults (fun () -> I.alloc h n) with
      | None -> None
      | Some r ->
          let region = Io_region { m; h; r } in
          Some (entry d (Io_memory { origin = Made; region }) n (stamps_new ())))
  | Host -> None

let drop_stamps (e : entry) = if e.stamps <> 0 then stamps_unref e.stamps

(* Words *)

(* Takes one of [p]'s word unmaps for a mapping about to be made: false once
   [p]'s slot is freed and no mapping is left, when its points are done and
   its word may be gone. *)
let rec hold_word p =
  let n = Atomic.Loc.get [%atomic.loc p.live.word_unmaps] in
  n > 0
  && (Atomic.Loc.compare_and_set [%atomic.loc p.live.word_unmaps] n (n + 1)
     || hold_word p)

(* The devices whose words of their drivers' own lost their last mapping,
   which the next drain gives back: a submit gives back mappings of words,
   and must not wait for a free, which CUDA's does for the GPU's running
   work. A device enters once, after its slot was freed, and a drain that
   raises nothing empties the list. *)
let unmapped : device list Atomic.t = Atomic.make []

let rec push_unmapped p =
  let l = Atomic.get unmapped in
  if not (Atomic.compare_and_set unmapped l (p :: l)) then push_unmapped p

let word_unmapped p =
  if
    Atomic.Loc.fetch_and_add [%atomic.loc p.live.word_unmaps] (-1) = 1
    && p.live.word = 0
  then push_unmapped p

(* Gives back the unmapped words one at a time. A device whose slot was
   freed has stopped, so [Dev.give] calls its driver at once. A word is
   taken off the list before its free, so a free that raises reaches the
   drain's caller and the word is not freed again ({!Rig_edge.Driver}); the
   rest wait for the next drain. *)
let rec give_words () =
  match Atomic.get unmapped with
  | [] -> ()
  | p :: rest as l ->
      (if Atomic.compare_and_set unmapped l rest then
         match p.live.word_region with
         | Some r -> ignore (free_region p r : Dev.given)
         | None -> ());
      give_words ()

let retire_locked d m =
  let l = d.live in
  l.pending <- insert l.pending (Dev.next_value d) (Unmap_word m)

external reraise : exn -> 'a = "%reraise"

(* Gives back [d]'s mapping [r] of [p]'s word now, or later ([waits]). A
   free that raises is not run again ([Dev.give]), so [p]'s count drops
   before the exception passes on, with nothing allocated first. *)
let unmap_word d ((r, p) as m) : Dev.given =
  match free_region d r with
  | Given ->
      word_unmapped p;
      Given
  | Later -> waits d (Unmap_word m) Later
  | exception e ->
      word_unmapped p;
      reraise e

(* Drops [d]'s own unmap once this call freed its slot. The lock is passed
   first: a refresh that began while [d] held its slot reads the word under
   it ({!Dev.stopped}), and one after reads none. *)
let let_go_word d =
  Lock.hold d.live.lock;
  Lock.release d.live.lock;
  word_unmapped d

(* Gives [e]'s region back to its driver, and its bytes to their keeper; or
   later ([waits]). *)
let give_back (e : entry) : Dev.given =
  let d = e.owner in
  let given =
    match e.backing with
    | Driver_memory { region; _ } -> free_region d region
    | Io_memory { region; _ } -> free_io d region
    | Kept _ -> Dev.Given
  in
  match given with
  | Given ->
      drop_stamps e;
      let budget = entry_budget e in
      give_room d budget e.bytes;
      if budget = Device_budget then note d;
      Given
  | Later -> waits d (Give e) Later

(* Releases the mapping [mp] now if its mapper ran all the work submitted so
   far, which is then all that could use it, and is [Later] if not. An
   orphaned device maps its parent's copy of the memory, never the child's:
   the child is done with it without calling its driver. *)
let unmap_now mp : Dev.given =
  let d = mp.on in
  let v = Dev.submitted d in
  if Dev.orphaned d then Given
  else if (Dev.is_lost d && not (Dev.stopped d)) || Dev.word d < v then Later
  else free_region d mp.map

(* Notes one of [e]'s unmaps done, and gives [e] back after the last. *)
let entry_unmapped (e : entry) : Dev.given =
  if Atomic.Loc.fetch_and_add [%atomic.loc e.unmaps] (-1) = 1 then give_back e
  else Given

(* Gives back [d]'s mapping [r] of [e] now, or later ([waits]). Memory counts
   its mappings and goes back to its driver at the last unmap: until then a
   mapper's driver may still name its pages, and a driver that maps host
   memory by its address would hand them to new memory at that address. *)
let give d r e : Dev.given =
  match free_region d r with
  | Given -> entry_unmapped e
  | Later -> waits d (Unmap (r, Some e)) Later

(* Takes the note [id] out of [d]'s table, which, emptied, keeps no buckets
   for the most [d] ever mapped. *)
let forget_note (d, id) =
  let l = d.live in
  Cache.remove l.mapped id;
  Cache.trim l.mapped

(* Releases [maps], the mappings of [e] that [free_entry] took, each once its
   mapper ran the work that could use it, then takes [e] out of the mapper's
   table. The note stays while the unmap runs: a drain that finds it leaves
   the mapping, which [e.maps] no longer holds, to this free. A loop of its
   own, so a free builds no closure. [e] outlives the loop: [free_entry]
   holds its own unmap. *)
let rec release_maps (e : entry) = function
  | [] -> ()
  | mp :: maps ->
      (match unmap_now mp with
      | Given -> ignore (entry_unmapped e : Dev.given)
      | Later ->
          ignore (waits mp.on (Unmap (mp.map, Some e)) Later : Dev.given));
      Dev.protect mp.on forget_note (mp.on, e.id);
      release_maps e maps

(* A mapping belongs to whichever call takes it out of [e.maps] under the
   owner's lock: [free_entry] takes them all, a Stopped mapper's drain its own
   ([take_back]). *)
let free_entry (e : entry) =
  (* No mapping is made of dead memory: one seen gone stays gone. *)
  let maps =
    if e.maps == [] then []
    else begin
      let l = e.owner.live in
      Lock.hold l.lock;
      let maps = e.maps in
      e.maps <- [];
      Lock.release l.lock;
      maps
    end
  in
  release_maps e maps;
  entry_unmapped e

let key n kind =
  let k = match kind with Device -> 0 | Pinned -> 1 | Mapped -> 2 in
  (n * 4) + k

(* Memory the cache never takes: io memory, host memory a device borrowed,
   which returns to its keeper once its uses are reached, and memory another
   PCI function reaches, which may still write it. *)
let uncached (e : entry) =
  e.reached
  ||
  match e.backing with
  | Driver_memory _ -> false
  | Io_memory _ | Kept _ -> true

let mark_reached (e : entry) =
  Dev.protect e.owner (fun (e : entry) -> e.reached <- true) e

let cache_key (e : entry) =
  match e.backing with
  | Driver_memory { memory; _ } -> key e.bytes memory
  | Io_memory _ | Kept _ -> invalid_arg "Rig: io or kept memory in a cache"

(* Bigarrays over memory *)

external proxy_new : unit -> int = "caml_rig_proxy_new"
external proxy_drop : int -> bool = "caml_rig_proxy_drop" [@@noalloc]

let make_proxy (e : entry) =
  if e.proxy = 0 then e.proxy <- proxy_new ();
  e.proxy

let proxy (e : entry) = Dev.protect e.owner make_proxy e

(* Whether a bigarray over [e]'s memory may be reachable, which keeps the memory
   from reuse and from its free. Once none is, the proxy is freed. *)
let viewed (e : entry) =
  if e.proxy = 0 then false
  else if proxy_drop e.proxy then begin
    e.proxy <- 0;
    false
  end
  else true

(* Puts [e] in its owner's cache, for reuse. A key's array allocates only as
   it grows past the most entries it held, so a drain of a size cached before
   allocates nothing. Each step allocates before it stores. The owner's lock
   is held. *)
let cache (e : entry) =
  let l = e.owner.live and k = cache_key e in
  let c =
    match Cache.find l.cache k with
    | c -> c
    | exception Not_found ->
        let c = { entries = [||]; n = 0 } in
        Cache.replace l.cache k c;
        c
  in
  let n = c.n in
  if n = Array.length c.entries then begin
    let a = Array.make (Int.max 8 (2 * n)) no_entry in
    Array.blit c.entries 0 a 0 n;
    c.entries <- a
  end;
  c.entries.(n) <- e;
  c.n <- n + 1;
  l.cached <- l.cached + e.bytes

(* Under the owner's lock: the latest entry of [c], which holds one, taken
   out of the cache. Its key stays bound. *)
let uncache l (c : cached) =
  let n = c.n - 1 in
  let e = c.entries.(n) in
  c.entries.(n) <- no_entry;
  c.n <- n;
  l.cached <- l.cached - e.bytes;
  e

(* Puts [e] among its owner's retiring memory. The owner's lock is held. *)
let retire_entry (e : entry) =
  let l = e.owner.live in
  let n = l.nretiring in
  if n = Array.length l.retiring then begin
    let a = Array.make (Int.max 8 (2 * n)) no_entry in
    Array.blit l.retiring 0 a 0 n;
    l.retiring <- a
  end;
  l.retiring.(n) <- e;
  l.nretiring <- n + 1

let to_cache (e : entry) = Dev.protect e.owner cache e

(* Whether [d] holds more than its budget in the memory that [e]'s counts in:
   [e] then returns to the driver, so a later allocation the budget refuses
   cannot reuse it. It skips the cache, never the wait: the driver gets [e]
   once every point of it is reached, [d]'s own included. *)
let over_budget d (e : entry) = owns e && d.live.used > d.budget

(* Routing collected records

   A drain routes the records of its device's list one at a time from the
   list's hand ({!release_front}), and judges its retiring memory one entry
   at a time. A record or entry is judged, stored in its home and taken out
   of the hand or the retiring memory in one section under the device's
   lock, so no other drain reads or frees it meanwhile. Reading a word
   behind a transport calls its driver, which runs no stop there. The store
   allocates first, so an exception leaves the record or the entry for the
   next drain. *)

(* Whether [d]'s collected [e] enters the cache now. *)
let cacheable d (e : entry) =
  (not (Dev.is_lost d))
  && (not (uncached e))
  && reached e.stamps
  && (not (viewed e))
  && not (over_budget d e)

(* Under [d]'s lock: routes the record at the front of [d]'s hand to its
   home and pops it: whether there was one. *)
let route_one d =
  let l = d.live in
  let n = release_front l.release in
  n <> 0
  && begin
    (match node_record n with
    | Memory e when cacheable d e -> cache e
    | Memory e -> retire_entry e
    | Image (loaded, code) -> append_pending (d, Unload (loaded, code))
    | Release h -> append_pending (d, Let_go h)
    | Stamps st -> stamps_unref st);
    release_pop l.release;
    true
  end

let rec route_released d = if Dev.protect d route_one d then route_released d

(* Holds

   The holds of {!Rig.Claim.hold} go on one list of no device, so that any
   drain releases those collected, under [hold_lock]. Releasing them waits
   for no work: work holds no claim. *)

external release_list : unit -> int = "caml_rig_release_list"

external hold_token : int -> holds -> int -> int -> int -> token
  = "caml_rig_token"

external hold_record : int -> holds = "caml_rig_node_record" [@@noalloc]

(* Releases the hold [i] of [taken] on [a]'s memory if it is taken, in one
   step, so releases that run concurrently, on several domains or in a
   callback that interrupts another at [Weak.get]'s allocation, and a release
   that an asynchronous exception interrupts and a later drain runs again,
   release each hold once. *)
external unhold : Bytes.t -> int -> alloc -> unit = "caml_rig_claim_unhold"
[@@noalloc]

let hold_list = release_list ()
let hold_lock = Lock.create ()
let hold_token h = hold_token hold_list h 0 max_int (-1)

(* The hold of no buffer, which has nothing to release. *)
let no_hold = no_token

let unhold_all (h : holds) =
  for i = 0 to Weak.length h.allocs - 1 do
    match Weak.get h.allocs i with Some a -> unhold h.taken i a | None -> ()
  done

external token_link : token -> unit = "caml_rig_token_link" [@@noalloc]

(* The holds released, the token's record goes on the list now, and the
   next drain only frees its node. It takes no lock, so a callback may run it
   inside one. *)
let release_hold_now t h =
  unhold_all h;
  token_link t

(* Early release *)

(* Under [a]'s owner's lock, which a borrow's first entry takes too: puts
   [a]'s entry on the release list as [a]'s collection would, and lets go of
   the host memory [a] keeps, which returns once the entry no longer keeps it
   and the collector frees it. [kept] becomes [a]'s token: it releases the
   reference [a] took on the stamps once [a] is collected. Every reader of
   [a]'s entry checks its buffer live first, and [a]'s buffers are dead, but
   a walk of the stamps that began before keeps reading them. An entry of
   host memory keeps its bytes, so [a] lets it go; another entry stays, so
   that a dead buffer keeps its place ({!Rig.Buffer.overlaps}). *)
let release_locked ((a : alloc), kept) =
  token_link a.token;
  a.token <- kept;
  a.keep <- Nothing;
  match a.entry.backing with
  | Kept _ -> a.entry <- no_entry
  | Driver_memory _ | Io_memory _ -> ()

(* The token that releases the stamps [st] once collected, or [no_token] if
   [a]'s owner's list is closed: [a]'s memory then returns at its
   collection. *)
let stamps_token (a : alloc) st =
  stamps_ref st;
  match token a.owner.live.release (Stamps st) 0 max_int (-1) with
  | t ->
      if t == no_token then stamps_unref st;
      t
  | exception e ->
      stamps_unref st;
      raise e

let release_early (a : alloc) =
  let st = a.entry.stamps in
  if st = 0 then Dev.protect a.owner release_locked (a, no_token)
  else
    let kept = stamps_token a st in
    if kept != no_token then Dev.protect a.owner release_locked (a, kept)

(* Staging memory

   A device's staging halves are copy's, in its [live]; memory returns the
   buffers of free halves where it returns cached memory, as collected memory
   returns. *)

external buffer_free : buffer -> string -> int = "caml_rig_buffer_free"

(* Under [l]'s lock: the buffers of [l]'s free halves, taken out of them,
   each half's cursor back at its start, and the list made before the
   stores. *)
let take_free_halves l =
  Array.fold_left
    (fun bs (h : half) ->
      match h.buffer with
      | Some b when not h.held ->
          let bs = b :: bs in
          h.buffer <- None;
          h.cursor <- 0;
          bs
      | Some _ | None -> bs)
    [] l.halves

(* Whether a half of [l] has a buffer, which keeps [l]'s device's slot. *)
let has_halves l = Array.exists (fun (h : half) -> h.buffer <> None) l.halves

let return_half b = if buffer_free b "freed" = 1 then release_early b.mem.alloc

(* Returns [d]'s free staging halves ([return_half]). *)
let release_staging d =
  if Array.length d.live.halves > 0 then
    List.iter return_half (Dev.protect d take_free_halves d.live)

(* Under [hold_lock]: releases the holds at the front of the list's hand whose
   memory is live and pops them: whether there were any. *)
let release_hold () =
  let n = release_front hold_list in
  n <> 0
  && begin
    unhold_all (hold_record n);
    release_pop hold_list;
    true
  end

let rec release_holds () =
  if Lock.protect hold_lock release_hold () then release_holds ()

type fate = Stays | Cached | Freed

(* What becomes of the retiring [e] of [d]. *)
let fate d ~lost ~free_lost (e : entry) =
  if viewed e then Stays
  else if lost then if free_lost && reached e.stamps then Freed else Stays
  else if uncached e || over_budget d e then
    if reached e.stamps then Freed else Stays
  else if reached e.stamps then Cached
  else Stays

(* Under [d]'s lock: from the retiring entry [judged] on, caches each whose
   uses are reached and takes out the first to free, moving the last entry
   into its place: it, or [no_entry]. Its judgement reads [freeing] at each
   entry, and [d]'s loss, a call, once per hold of the lock ([take_freed]):
   both only ever become true, and a drain of another domain that restarts
   the pass meanwhile makes it judge entries again, never wrongly. A loss
   after the read is as one after a judgement, which the entry's free or
   its device's cache then meets. *)
let rec take_freed_from d lost =
  let l = d.live in
  let i = l.judged in
  if i >= l.nretiring then no_entry
  else
    let e = l.retiring.(i) in
    match fate d ~lost ~free_lost:l.freeing e with
    | Stays ->
        l.judged <- i + 1;
        take_freed_from d lost
    | (Cached | Freed) as f ->
        if f = Cached then cache e;
        let last = l.nretiring - 1 in
        l.retiring.(i) <- l.retiring.(last);
        l.retiring.(last) <- no_entry;
        l.nretiring <- last;
        if f = Freed then e else take_freed_from d lost

let take_freed d = take_freed_from d (Dev.is_lost d)

(* Under [l]'s lock: a pass starts at the first retiring entry, the second
   on a device found lost and Stopped. *)
let start_pass l = l.judged <- 0

let start_freeing l =
  l.freeing <- true;
  l.judged <- 0

(* Frees [d]'s retiring memory that is due, one entry at a time: an entry
   leaves as its free starts, so an exception leaks that one at most. A
   pass's state is [d]'s, so the pass allocates nothing. *)
let rec judge d =
  let e = Dev.protect d take_freed d in
  if e != no_entry then begin
    ignore (free_entry e : Dev.given);
    judge d
  end

(* Releasing the cache *)

let cache_keys l = Cache.fold (fun k _ ks -> k :: ks) l.cache []

(* Under [l]'s lock: takes the first entry of [l]'s cache under the key [k]
   if [pick] holds of it. Entries of one key share their owner and their
   kind of memory, so [pick] answers alike for each. *)
let take_picked (l, k, pick) =
  match Cache.find l.cache k with
  | c when c.n = 0 ->
      Cache.remove l.cache k;
      None
  | c when pick c.entries.(c.n - 1) -> Some (uncache l c)
  | _ | (exception Not_found) -> None

(* Waits for [d]'s value [v] on behalf of memory, never of [d]'s caller: a loss
   the wait finds is [d]'s alone, whose memory then returns once it stopped. *)
let wait_for d v = try Dev.wait d v with Dev.Lost (d', _) when d' == d -> ()

(* Whether [d]'s work up to [v] is done, or [d] is lost: its frees then wait for
   its stop themselves. *)
let reached_or_lost d v =
  Dev.is_lost d || match Dev.word d >= v with r -> r | exception Dev.Lost _ -> true

(* Frees the cached memory of [d] that [pick] holds of, one entry at a time,
   once [d]'s work up to [v] is done, and defers each free otherwise: an
   exception leaks the entry in hand at most. *)
let release_picked d pick v =
  let l = d.live in
  let rec take k =
    match Dev.protect d take_picked (l, k, pick) with
    | None -> ()
    | Some e when not (reached_or_lost d v) ->
        defer d (Free e);
        take k
    | Some e ->
        ignore (free_entry e : Dev.given);
        take k
  in
  List.iter take (Dev.protect d cache_keys l)

let rec picked pick (c : cached) i =
  i < c.n && (pick c.entries.(i) || picked pick c (i + 1))

let any_picked (l, pick) =
  Cache.fold (fun _ c found -> found || picked pick c 0) l.cache false

(* Frees what [pick] holds of in [d]'s cache once the work [d] was handed
   until now is done: at once if it is, after waiting for it when [wait] and
   [pick] holds of some, and otherwise each free is deferred until it is.
   The wait comes before any entry leaves the cache, so an exception in it,
   such as [Sys.Break], loses none. A lost device's cache waits for it to
   count as stopped. *)
let release_cache_if ~wait d pick =
  if (not (Dev.is_lost d)) || Dev.stopped d then begin
    let v = Dev.submitted d in
    if wait && (not (Dev.is_lost d)) && Dev.protect d any_picked (d.live, pick)
    then wait_for d v;
    release_picked d pick v
  end

let every (_ : entry) = true

(* [d]'s cached memory, or with [upto] only memory that counts in [d]'s
   budget until [d] holds at most [upto] bytes, freed as [release_cache_if]
   does. A count of what is left goes with the picks, so [upto] is for a
   release that never waits. *)
let release_cache ?upto ~wait d =
  match upto with
  | None -> release_cache_if ~wait d every
  | Some upto ->
      let held = ref d.live.used in
      release_cache_if ~wait d (fun e ->
          owns e && !held > upto
          && begin
            held := !held - e.bytes;
            true
          end)

(* Ends [l]'s way to rig's array, and takes its mapping of the array. *)
let pop_words_map l =
  l.words <- -1;
  match l.word_maps with
  | r :: rest ->
      l.word_maps <- rest;
      Some r
  | [] -> None

let undecided = { occupant = -1; way = -1; word_map = None }

(* The first of [pairs] from [i] that holds a mapping, or [-1]. *)
let rec first_mapped pairs i =
  if i >= Array.length pairs then -1
  else if Option.is_some (Array.unsafe_get pairs i).word_map then i
  else first_mapped pairs (i + 1)

(* Takes the mapping of the first of [l]'s pairs from [i] that holds one,
   with its index, emptying its cell, or empties [l]'s pairs after the
   last. *)
let pop_pair_map (l, i) =
  match first_mapped l.pairs i with
  | -1 ->
      l.pairs <- [||];
      None
  | i ->
      let taken = Some (i, Option.get l.pairs.(i).word_map) in
      l.pairs.(i) <- undecided;
      taken

let mapped_ids l = Cache.fold (fun id _ ids -> id :: ids) l.mapped []

(* Takes the entry [id] out of those [d] maps, with [d]'s mapping of it, if
   it is there. *)
let pop_mapped (d, id) =
  match Cache.find_opt d.live.mapped id with
  | Some _ as found ->
      forget_note (d, id);
      found
  | None -> None

(* Takes [mp] back out of [e]'s mappings: whether it was there. *)
let take_back ((e : entry), mp) =
  List.memq mp e.maps
  && begin
    e.maps <- List.filter (fun mp' -> mp' != mp) e.maps;
    true
  end

(* Gives back the Stopped [d]'s mappings of words and of memory that
   outlives it, one at a time: an exception loses at most the one in hand,
   and the next drain finds the rest. A Stopped device's pairs change no
   more ({!Submission}), so the walk over them ends. *)
let give_maps d =
  let l = d.live in
  let rec words () =
    match Lock.protect l.lock pop_words_map l with
    | None -> ()
    | Some r ->
        ignore (unmap d r : Dev.given);
        words ()
  in
  let rec pairs i =
    match Lock.protect l.lock pop_pair_map (l, i) with
    | None -> ()
    | Some (i, m) ->
        ignore (unmap_word d m : Dev.given);
        pairs (i + 1)
  in
  let mapped id =
    match Lock.protect l.lock pop_mapped (d, id) with
    | Some ((e, mp) as taken) when Dev.protect e.owner take_back taken ->
        ignore (give d mp.map e : Dev.given)
    | Some _ | None -> ()
  in
  words ();
  pairs 0;
  List.iter mapped (Lock.protect l.lock mapped_ids l)

(* Moves what became due: retiring memory no bigarray reads and whose uses
   are reached enters the cache, or is freed if [d] is lost and counts as
   stopped, with its mappings and its cache. *)
let due d =
  let free_lost = Dev.is_lost d && Dev.stopped d in
  let l = d.live in
  if l.nretiring > 0 then begin
    Dev.protect d (if free_lost then start_freeing else start_pass) l;
    judge d
  end;
  if free_lost then begin
    give_maps d;
    release_picked d every (Dev.submitted d)
  end

(* Under [l]'s lock: takes the first of [l]'s pending releases if [w] reached
   its value. *)
let take_due (l, w) =
  match l.pending with
  | (v, p) :: front, back when v <= w ->
      let taken = Some p in
      l.pending <- (if front == [] then (List.rev back, []) else (front, back));
      taken
  | _ -> None

(* The value up to which [d]'s releases are due: [d]'s word, none while it
   is lost and not Stopped, and all once it is Stopped, since it runs
   nothing, also no submit it never gave a value. *)
let due_at d =
  if not (Dev.is_lost d) then Dev.word d
  else if Dev.stopped d then max_int
  else -1

(* Under [l]'s lock: takes the first of [l]'s retired mappings among its
   pending releases whose value [w] reached, and leaves the others in
   order. *)
let take_word (l, w) =
  let front, back = l.pending in
  let rec take before = function
    | (v, _) :: _ when v > w -> None
    | (_, Unmap_word m) :: rest ->
        let taken = Some m in
        l.pending <- (List.rev_append before rest, []);
        taken
    | p :: rest -> take (p :: before) rest
    | [] -> None
  in
  take [] (if back == [] then front else front @ List.rev back)

(* Gives back [d]'s retired mappings of words whose waits its queue passed,
   one at a time, as [give_maps] does, and runs none of its other releases.
   A mapping that does not go back ends the walk: it waits again, since [d]
   was lost after [w] was read and nothing more of it goes back until its
   stop. *)
let give_retired d =
  let l = d.live in
  if fst l.pending != [] then
    let w = due_at d in
    let rec give () =
      match Lock.protect l.lock take_word (l, w) with
      | None -> ()
      | Some m -> (
          match unmap_word d m with Given -> give () | Later -> ())
    in
    give ()

(* A release that waits again ([waits]) waits whole: an unload's code memory
   waits inside it for the unload. *)
let run_pending d : pending -> Dev.given = function
  | Free e -> free_entry e
  | Give e -> give_back e
  | Unmap (r, after) as p -> (
      match (waits d p (free_region d r), after) with
      | Given, Some e -> (
          (* A [Later] from [e]'s owner leaves [e] on the owner's queue, so
             [d]'s walk goes on. *)
          ignore (entry_unmapped e : Dev.given);
          Given)
      | Given, None -> Given
      | Later, _ -> Later)
  | Unload (loaded, code) as p -> (
      match (waits d p (unload_now d loaded), code) with
      | Given, Some e when Dev.is_lost d || over_budget d e -> free_entry e
      | Given, Some e ->
          to_cache e;
          Given
      | Given, None -> Given
      | Later, _ -> Later)
  | Unmap_word m -> unmap_word d m
  | Let_go _ -> Given

(* Unloads [d]'s image [l] now, then gives back its code memory [code], or
   does both later ([waits]). *)
let unload d l code = ignore (run_pending d (Unload (l, code)) : Dev.given)

(* Runs [d]'s pending releases up to the value [w], one at a time: each
   leaves the queue as it runs, so an exception leaks that one at most. A
   release that does not go back ends the walk: it waits for [d]'s stop, as
   does the rest, since [d] was lost after [w] was read. It may be due at
   [w] again, and the walk would then take it back forever. *)
let rec run_due d w =
  match Dev.protect d take_due (d.live, w) with
  | None -> ()
  | Some p -> ( match run_pending d p with Given -> run_due d w | Later -> ())

(* Runs the pending releases whose value [d] reached. *)
let due_pending d =
  match d.live.pending with
  | [], _ -> ()
  | (v, _) :: _, _ ->
      let w = due_at d in
      if w >= v then run_due d w

(* Drains [d]'s own list and what became due on it. *)
let drain_own d =
  if released_any d.live.release then route_released d;
  due d;
  due_pending d

let is_let_go = function _, Let_go _ -> true | _ -> false

let drop_let_go l =
  let front, back = l.pending in
  l.pending <-
    (List.filter (fun p -> not (is_let_go p)) (front @ List.rev back), [])

(* A forked child keeps nothing for an orphaned device: it routes the records
   the device's list gave and lets go of the values pending on it. Nothing
   else of the device runs, since it would call the parent's driver. *)
let forget d =
  let l = d.live in
  if released_any l.release then route_released d;
  let front, back = l.pending in
  if List.exists is_let_go front || List.exists is_let_go back then
    Dev.protect d drop_let_go l

(* Whether [l] has nothing to drain: no collected memory, nothing waiting for a
   value. A drain of an idle device allocates nothing. *)
let idle l =
  l.nretiring = 0
  && fst l.pending == []
  && not (released_any l.release)

let kept_locked l =
  l.cached > 0 || l.word_maps != []
  || fst l.pending != []
  || Cache.length l.mapped > 0
  || first_mapped l.pairs 0 >= 0
  || has_halves l

(* Whether the lost [d] keeps what returns once it is Stopped: its cache,
   its mappings and its pending releases. Read under the lock: a pair's
   mapping that a decision retires is in the cell or among the pending
   releases. *)
let kept d = Dev.protect d kept_locked d.live

(* A lost device's memory returns once it has stopped. A device whose stop
   returned frees its slot once its last memory and mapping went back and
   its last kept value went, then lets go of its word. Every drain gives
   back the words that lost their last mapping, and first releases the holds
   collected and drains the devices whose stop returned and whose list took
   records: no call on them is due, as a program makes none on a device it
   closed. *)
let rec drain d =
  if released_any hold_list then release_holds ();
  if ended_linked () then drain_every_ended ();
  if Dev.orphaned d then forget d
  else begin
    if Dev.is_lost d && Dev.stopped d then release_staging d;
    if (not (idle d.live)) || (Dev.is_lost d && kept d) then drain_own d;
    if Dev.stop_returned d && (not (kept d)) && Dev.free_slot d then
      let_go_word d;
    give_words ()
  end

and drain_ended d =
  if Dev.stop_returned d && released_any d.live.release then drain d

(* The request was taken before the walk, so a drain that raises, at an
   allocation's asynchronous exception or a driver's free, asks again for
   the devices it leaves. The request goes first, as it allocates nothing. *)
and drain_every_ended () =
  match Dev.iter drain_ended with
  | () -> ()
  | exception e ->
      ended_rearm ();
      Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())

let () = Dev.answered := drain

(* The cache *)

let take_cached_key (l, k) =
  match Cache.find l.cache k with
  | c when c.n = 0 -> None
  | c ->
      let e = uncache l c in
      (* Every old use has settled. The new buffer has no dependencies on
         its memory's previous lifetime, an unordered use included. *)
      stamps_clear e.stamps;
      Some e
  | exception Not_found -> None

let take_cached d kind n =
  let l = d.live in
  Lock.protect l.lock take_cached_key (l, key n kind)

(* The out-of-memory ladder *)

(* Four tries with a round ([reclaim_round]) between each two, as rig.mli's
   Reclamation states, and the size check before any round in [alloc_entry].
   The collector meanwhile runs at a pace set by each allocation's token,
   against the room left in its device's budget ([pace] in rig_heap.c). *)
let rounds = 4

(* Whether the memory [e] counts in the budget of [pool], the host or a
   device. *)
let charged pool (e : entry) =
  match entry_budget e with
  | Device_budget -> e.owner == pool
  | Host_budget -> Dev.is_host pool
  | No_budget -> false

(* Under [l]'s lock: whether its device holds memory that counts in
   [pool]'s budget back from its return until work is done: collected memory
   outside the cache, and returns deferred until the device's handed work is
   done. *)
let holds_back (pool, l) =
  Array.exists (charged pool) (Array.sub l.retiring 0 l.nretiring)
  ||
  let holds = function
    | _, (Free e | Give e | Unmap (_, Some e) | Unload (_, Some e)) ->
        charged pool e
    | _, (Unmap (_, None) | Unload (_, None) | Unmap_word _ | Let_go _) ->
        false
  in
  let front, back = l.pending in
  List.exists holds front || List.exists holds back

(* Drains every device, the host included. *)
let drain_all () =
  drain Dev.host;
  Dev.iter drain

let () = Dev.drain_all := drain_all

(* A round of the ladder for [pool]'s budget: every cached memory on any
   device that counts in it returns once its device's handed work is done,
   waited for; for the host, its kept buffers too. A device whose handed
   work holds back the return of memory that counts in it is waited for
   too, so that memory returns and a loss shows as one. Then every device
   drains, which returns collected memory and runs the unmaps that hold it;
   from the second round a collection finds the memory unreachable since,
   and every device drains again. *)
let reclaim_round pool round =
  if Dev.is_host pool then heap_drop ();
  Dev.iter (fun d ->
      if Dev.protect d holds_back (pool, d.live) && not (Dev.is_lost d) then
        wait_for d (Dev.submitted d);
      release_cache_if ~wait:true d (charged pool));
  drain_all ();
  if round >= 2 then begin
    release_staging Dev.host;
    Dev.iter release_staging;
    Gc.full_major ();
    drain_all ()
  end

(* A round may find [d] lost, such as by a fault its wait finds. Its memory
   returns only once its close has stopped it, so the allocation raises the
   loss. *)
let rec reclaim_from d pool n f round =
  if round >= rounds then raise (Dev.Out_of_memory (d, n));
  reclaim_round pool round;
  if Dev.is_lost d then Dev.raise_lost d;
  match f () with Some x -> x | None -> reclaim_from d pool n f (round + 1)

let reclaiming d ~pool n f = reclaim_from d pool n f 1

(* A driver's entry raises [Invalid_argument] for a function that needs more
   than the device offers, which loses nothing. *)
let entry_scope = Dev.Any_but [%extension_constructor Invalid_argument]

(* [image] stays reachable until the driver's call returns: its collection
   unloads [i], whose entry a driver may still be compiling, as Metal does. A
   function the device has no memory for is asked again after each round of the
   ladder. *)
let kernel_entry image f =
  match image.loaded with
  | Linked c ->
      (* The host knows neither a function's parameters nor a limit on its
         threads. *)
      Option.map
        (fun code ->
          { Rig_edge.code; launch = 0n; params = None; threads = None })
        (Host_code.entry c f)
  | Loaded { m; i; _ } ->
      let module D = (val m) in
      let d = image.idev in
      let ask () = Dev.counted d entry_scope (fun () -> D.entry i f) in
      let again () =
        match ask () with
        | Rig_edge.Entry e -> Some (Some e)
        | Absent -> Some None
        | No_memory _ -> None
      in
      let e =
        match ask () with
        | Rig_edge.Entry e -> Some e
        | Absent -> None
        | No_memory n -> reclaiming d ~pool:d n again
      in
      ignore (Sys.opaque_identity image);
      e

(* The growth of [d]'s queues a submission asks at round [round] of the
   ladder, after a Short of [lacked] bytes, or 0. Every answer takes a round,
   so a submission asks at most [rounds] times whatever its room answers. A
   round reclaims the host's budget too: a driver may grow its queues in
   pinned memory, which counts there. Only a driver's room answers that its
   queues grow. *)
let grow_queues d ~round ~lacked =
  if round > rounds then raise (Dev.Out_of_memory (d, lacked));
  match d.kind with
  | Driver { m; h; _ } -> (
      let module D = (val m) in
      match Dev.counted d Dev.Any (fun () -> D.grow h) with
      | Rig_edge.Grown -> (round + 1, lacked)
      | Short n ->
          if round >= rounds then raise (Dev.Out_of_memory (d, n));
          reclaim_round d round;
          reclaim_round Dev.host round;
          (round + 1, n))
  | Io _ | Host -> invalid_argf "Rig: %s has no queues to grow" d.name

(* Allocation *)

let room d = Int.max 0 (d.budget - d.live.used)

(* A fresh memory record over the entry [e] of [d], with its token. *)
let of_entry d e =
  let m =
    match e.backing with
    | Driver_memory { region; _ } ->
        let l = locate region in
        make ?host:l.host ?address:l.address ~handle:l.handle d e
    | Io_memory _ | Kept _ -> make d e
  in
  let l = d.live in
  let live = if m.host >= 0 then l.used else -1 in
  m.alloc.token <- token l.release (Memory e) e.bytes (room d) live;
  if m.alloc.token == no_token then begin
    (* A refused token means [d] has stopped: its driver takes [e] back
       now. *)
    ignore (free_entry e : Dev.given);
    Dev.raise_lost d
  end;
  m

(* [new_entry] within the budget memory of [kind] counts in: its room is taken
   before the driver is asked, so two domains' allocations never both fit the
   last room, and given back if the driver refuses. *)
let new_counted d kind n =
  let budget = budget_of d kind in
  if not (take_room d budget n) then None
  else
    match new_entry d kind n with
    | Some _ as e -> e
    | None ->
        give_room d budget n;
        None
    | exception x ->
        give_room d budget n;
        raise x

(* One try at [n] bytes of [d]'s memory of [kind]: from its cache, else from its
   driver within the budget. *)
let try_entry d kind n () =
  match take_cached d kind n with
  | Some _ as e -> e
  | None -> (
      match new_counted d kind n with
      | Some _ as e ->
          note d;
          e
      | None -> None)

(* [n] bytes of [d]'s memory of [kind], through the ladder of the budget it
   counts in once refused. *)
let laddered d kind n =
  match try_entry d kind n () with
  | Some e -> e
  | None ->
      let pool = if budget_of d kind = Host_budget then Dev.host else d in
      reclaiming d ~pool n (try_entry d kind n)

let alloc_entry d kind n =
  let kind = if kind = Mapped && n > d.budget then Pinned else kind in
  let limit =
    match budget_of d kind with
    | Device_budget -> d.budget
    | Host_budget -> Dev.host.budget
    | No_budget -> max_int
  in
  if n > limit then raise (Dev.Out_of_memory (d, n));
  drain d;
  if kind <> Mapped then laddered d kind n
  else
    (* Mapped memory the window or the budget cannot hold is pinned memory,
       which keeps its promises, and the cache stays. *)
    match try_entry d Mapped n () with
    | Some e -> e
    | None -> laddered d Pinned n

let alloc d kind n = of_entry d (alloc_entry d kind n)
let reserve n () = if heap_reserve n Dev.host.budget then Some () else None

(* Reserves what a host buffer of [n] bytes takes of the heap; a refusal
   names [n], the allocation's bytes. *)
let heap_reserved n =
  let footprint = heap_footprint n in
  match reserve footprint () with
  | Some () -> ()
  | None -> reclaiming Dev.host ~pool:Dev.host n (reserve footprint)

(* [n] reserved bytes of the heap. A refusal of the C library runs the ladder,
   and after its rounds gives the reservation back. *)
let bytes n () =
  match heap_bytes n with
  | ba -> Some ba
  | exception Stdlib.Out_of_memory -> None

let host_bytes n =
  match heap_bytes n with
  | ba -> ba
  | exception Stdlib.Out_of_memory -> (
      match reclaiming Dev.host ~pool:Dev.host n (bytes n) with
      | ba -> ba
      | exception (Dev.Out_of_memory _ as x) ->
          heap_release (heap_footprint n);
          raise x)

(* What every host buffer of no bytes keeps: nothing to free. *)
let empty = Bigarray.Array1.create Bigarray.char Bigarray.c_layout 0
let empty_keep = Heap empty

let host_memory n =
  drain Dev.host;
  if n = 0 then
    let addr = ba_address empty in
    own ~keep:empty_keep ~host:addr ~address:addr ~handle:0n Dev.host no_entry
  else begin
    heap_reserved n;
    let ba = host_bytes n in
    let addr = ba_address ba in
    own ~keep:(Heap ba) ~host:addr ~address:addr ~handle:0n Dev.host no_entry
  end

(* Borrows *)

let make_entry (a : alloc) =
  if a.entry == no_entry then begin
    let d = a.owner in
    let e = entry d (Kept a.keep) (size a) (stamps_new ()) in
    let t = token d.live.release (Memory e) 0 max_int (-1) in
    if t == no_token then begin
      stamps_unref e.stamps;
      Dev.raise_lost d
    end;
    a.entry <- e;
    a.token <- t
  end

let ensure_entry (a : alloc) = Dev.protect a.owner make_entry a

(* [d]'s mapping among [maps], if any: a loop of its own, so looking builds
   no closure. *)
let rec map_on d = function
  | [] -> None
  | mp :: maps -> if mp.on == d then Some mp else map_on d maps

(* [d]'s mapping of [e], if any. *)
let find_map ((e : entry), d) = map_on d e.maps

(* Counts [mp] in [e], and adds it to [e]'s mappings, unless its device
   mapped [e] meanwhile: that mapping, then. *)
let add_map ((e : entry), mp) =
  Atomic.Loc.incr [%atomic.loc e.unmaps];
  match find_map (e, mp.on) with
  | Some other -> Some other
  | None ->
      e.maps <- mp :: e.maps;
      None

(* Notes [e] among the entries [d] maps, with [d]'s mapping [mp] of it, unless
   [d]'s stop returned, whose drains may have given back its mappings
   already: whether it noted it. *)
let note_mapped (d, (e : entry), mp) =
  (not (Dev.stop_returned d))
  && begin
    Cache.replace d.live.mapped e.id (e, mp);
    true
  end

let map_host_range d start n =
  match d.kind with
  | Driver _ when not d.maps_host -> None
  | Driver { m = dm; h; rid } -> (
      let module D = (val dm) in
      match Dev.counted d Dev.Any (fun () -> D.map_host h start n) with
      | Some r -> Some (Region { m = dm; h; r; rid })
      | None -> None)
  | _ -> None

(* The region's module types the mapping; the keys' equality types [d]'s
   handle. *)
let map_peer_region d (Region { m = om; h = oh; r; rid }) =
  match d.kind with
  | Driver { m = dm; h; _ } -> (
      let module D = (val dm) in
      let module O = (val om) in
      match Type.Id.provably_equal D.key O.key with
      | Some Type.Equal -> (
          match Dev.counted d Dev.Any (fun () -> O.map_peer h oh r) with
          | Some r -> Some (Region { m = om; h; r; rid })
          | None -> None)
      | None -> None)
  | _ -> None

let map_peer d (a : alloc) =
  match a.entry.backing with
  | Driver_memory { region; _ } -> map_peer_region d region
  | Io_memory _ | Kept _ -> None

(* Whether [d] maps [a] as a peer's memory: memory of a device of [d]'s
   driver. *)
let peer_memory d (a : alloc) =
  match a.owner.kind with
  | Driver _ -> a.owner.key = d.key
  | Host | Io _ -> false

(* [d]'s new mapping of the allocation [a], whose host address is [host] or
   [-1]. It counts in the memory and is recorded there first, then noted on
   [d] unless [d]'s stop returned: a drain of [d] that came first found no
   note, so the mapping is taken back out of the memory and given back here,
   and [d] is lost. A drain that comes later finds the note. *)
let map_new d (a : alloc) host =
  (* Memory of a device of [d]'s driver maps as a peer's; other memory the
     host addresses, as host memory. *)
  let made =
    if peer_memory d a then map_peer d a
    else if host >= 0 then map_host_range d host (size a)
    else map_peer d a
  in
  match made with
  | None -> None
  | Some r -> (
      let e = a.entry in
      let loc =
        match locate r with
        | loc -> loc
        | exception x ->
            ignore (unmap d r : Dev.given);
            Printexc.raise_with_backtrace x (Printexc.get_raw_backtrace ())
      in
      let at = Option.value ~default:(-1) loc.address in
      let mp = { on = d; map = r; at; by = loc.handle } in
      (* A mapping that lost the race counts too, until it is given back, so
         the memory outlives it. *)
      match Dev.protect a.owner add_map (e, mp) with
      | Some other ->
          ignore (give d r e : Dev.given);
          Some other
      | None ->
          if Dev.protect d note_mapped (d, e, mp) then Some mp
          else begin
            if Dev.protect a.owner take_back (e, mp) then
              ignore (give d r e : Dev.given);
            Dev.raise_lost d
          end)

(* [d]'s mapping of the allocation [a], whose host address is [host] or [-1],
   made at the first borrow and shared by the later ones. *)
let mapping d (a : alloc) host =
  match Dev.protect a.owner find_map (a.entry, d) with
  | Some _ as found -> found
  | None -> map_new d a host

(* Keeps [got] as [e]'s pages, unless a call kept some first. *)
let set_pages ((e : entry), got) =
  match e.pages with Unasked -> e.pages <- got | Pages _ | No_pages -> ()

(* The host address of the io memory [a]'s pages, or [-1] if it maps none:
   the first result from its device, kept. A call that raises leaves it to
   ask again. Which pages a driver refuses to map is its own: each driver's
   .mli states it, from NVIDIA's os-mlock.c and Linux's mm/gup.c
   ([writable_file_mapping_allowed]) for CUDA and NV, and from amdgpu_hmm.c
   for AMD. *)
let pages (a : alloc) =
  let e = a.entry in
  (match e.pages with
  | Unasked -> (
      match e.backing with
      | Io_memory { region = Io_region { m = im; h; r }; _ } ->
          let module I = (val im) in
          let got =
            match Dev.counted a.owner Dev.Faults (fun () -> I.pages h r) with
            | Some ba -> Pages ba
            | None -> No_pages
          in
          Dev.protect a.owner set_pages (e, got)
      | Driver_memory _ | Kept _ -> ())
  | Pages _ | No_pages -> ());
  match e.pages with Pages ba -> ba_address ba | Unasked | No_pages -> -1

(* How [d] reaches the allocation [a] whose host address is [host]: constant
   answers, so asking builds nothing. *)
type reach =
  | Cannot (* [d] cannot map it *)
  | Itself (* [a] is [d]'s *)
  | At_host (* [d] is the host, which addresses it *)
  | Addressed (* [d] addresses host memory as the host does *)
  | Empty (* no bytes to map *)
  | Map (* through [d]'s mapping of it *)

let[@inline] reach d (a : alloc) host =
  let o = a.owner in
  if o == d then Itself
  else if Dev.is_io d || (Dev.is_io o && host < 0) then Cannot
  else if (not (Dev.same_machine o d)) && not (Dev.is_io o) then Cannot
  else if size a = 0 then
    (* Before the host's answer: memory of no bytes has no host address. *)
    Empty
  else if Dev.is_host d then if host >= 0 then At_host else Cannot
  else if d.memory_device && host >= 0 then Addressed
  else if host >= 0 && host mod page <> 0 && not (peer_memory d a) then
    (* [mapping] maps it as host memory, which a driver maps from a page
       ([map_host]). *)
    Cannot
  else Map

(* Whether [a] is io memory of no bytes, which every other device borrows over
   no pages. *)
let[@inline] empty_io d (a : alloc) =
  is_io_memory a.entry && size a = 0 && a.owner != d

let[@inline] host_of (a : alloc) =
  if is_io_memory a.entry then pages a else a.host

(* [a]'s owner's record, for a borrow on the owner of another device's: where
   its driver locates the memory, host memory at its host address, and no
   address for io memory and [0] for no bytes, as their records were made. *)
let owned (a : alloc) =
  let dev = a.owner and host = a.host in
  match a.entry.backing with
  | Driver_memory { region; _ } ->
      let l = locate region in
      let address = Option.value ~default:(-1) l.address in
      memory_of a dev ~host ~address ~handle:l.handle
  | Io_memory _ -> memory_of a dev ~host ~address:(-1) ~handle:0n
  | Kept _ ->
      memory_of a dev ~host ~address:(if host >= 0 then host else 0)
        ~handle:0n

let borrow d m =
  let a = m.alloc in
  if empty_io d a then
    let at = ba_address empty in
    Some (memory_of a d ~host:at ~address:at ~handle:0n)
  else
    let host = host_of a in
    match reach d a host with
    | Cannot -> None
    | Itself -> Some (if m.dev == d then m else owned a)
    | At_host -> Some (memory_of a d ~host ~address:host ~handle:0n)
    | Addressed ->
        Some (memory_of a d ~host ~address:host ~handle:(Nativeint.of_int host))
    | Empty -> Some (memory_of a d ~host ~address:0 ~handle:0n)
    | Map -> (
        if a.entry == no_entry then ensure_entry a;
        (* A borrow of host memory keeps its host address: it is host memory,
           which the host copies. *)
        match mapping d a host with
        | None -> None
        | Some mp -> Some (memory_of a d ~host ~address:mp.at ~handle:mp.by))

(* Whether one of [maps] is [d]'s. A loop of its own, so asking builds no
   closure. *)
let rec mapped_on d = function
  | [] -> false
  | mp :: maps -> mp.on == d || mapped_on d maps

let maps d m =
  let a = m.alloc in
  empty_io d a
  ||
  let host = host_of a in
  match reach d a host with
  | Cannot -> false
  | Itself | At_host | Addressed | Empty -> true
  | Map ->
      if a.entry == no_entry then ensure_entry a;
      let l = a.owner.live in
      Lock.hold l.lock;
      let maps = a.entry.maps in
      Lock.release l.lock;
      mapped_on d maps || Option.is_some (mapping d a host)

(* Asks an io device to read ahead the bytes that a borrow on [d], a device
   other than the host, reaches through its pages. It is a hint, so its
   faults and system errors are dropped. *)
let prefetch d (m : memory) ~at ~len =
  match m.alloc.entry.backing with
  | Io_memory { region = Io_region { m = im; h; r }; _ } when not (Dev.is_host d)
    -> (
      let module I = (val im) in
      try Dev.masked (fun () -> I.prefetch h r ~at ~len)
      with I.Fault _ | Sys_error _ -> ())
  | _ -> ()

let of_io d r ~access n =
  if Dev.is_lost d then Dev.raise_lost d;
  drain d;
  let e = entry ~access d (Io_memory { origin = Given; region = r }) n (stamps_new ()) in
  let m = make d e in
  m.alloc.token <- token d.live.release (Memory e) n max_int (-1);
  if m.alloc.token == no_token then begin
    stamps_unref e.stamps;
    Dev.raise_lost d
  end;
  m

let set_device_budget (d, n) = d.budget <- n

let set_budget d n =
  Dev.check_held "set_budget";
  if n < 0 then invalid_argf "Rig.set_budget: budget %d is negative" n;
  Dev.protect d set_device_budget (d, n);
  if Dev.is_host d then heap_trim n else release_cache ~upto:n ~wait:false d;
  if allocated d > n then release_staging d

(* [d]'s free staging halves go first, then its drain returns what was
   collected since its last call, the halves among it once their legs are
   done, and a Stopped device's cache with it. *)
let free_cache d =
  Dev.check_held "free_cache";
  release_staging d;
  drain d;
  if Dev.is_host d then heap_drop () else release_cache ~wait:false d
