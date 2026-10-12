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
external stamps_write : int -> int = "caml_rig_stamps_write" [@@noalloc]
external stamps_use : int -> int -> int = "caml_rig_stamps_use" [@@noalloc]
external stamps_keep : int -> int -> int = "caml_rig_stamps_keep" [@@noalloc]
external stamps_next : int -> int = "caml_rig_stamps_next" [@@noalloc]
external stamps_clear : int -> unit = "caml_rig_stamps_clear" [@@noalloc]
external stamps_uses : unit -> int = "caml_rig_stamps_uses"

(* The number of use words, and of keep words, in a chunk of stamps. A walk
   reads the last write, then each chunk's words from the first chunk, the
   stamps themselves, on; a point 0 is an empty word. *)
let uses = stamps_uses ()

(* [f] over the use points of the chunk [c] from its [i]th word on, and of
   the chunks after it. *)
let rec iter_from f c i =
  if c <> 0 then
    if i = uses then iter_from f (stamps_next c) 0
    else begin
      let p = stamps_use c i in
      if p <> 0 then f p;
      iter_from f c (i + 1)
    end

let iter_points f st =
  if st <> 0 then begin
    let p = stamps_write st in
    if p <> 0 then f p;
    iter_from f st 0
  end

let iter_write f st =
  if st <> 0 then
    let p = stamps_write st in
    if p <> 0 then f p

(* Waits for the use points of [m]'s stamps from the chunk [c]'s [i]th word
   on, with no closure: the walk is a host access's every wait. [m], whose
   collection frees the stamps, is an argument of every step, so it stays
   reachable through every wait and every read. *)
let rec wait_from m c i =
  if c = 0 then ignore (Sys.opaque_identity m)
  else if i = uses then wait_from m (stamps_next c) 0
  else begin
    let p = stamps_use c i in
    if p <> 0 then Dev.wait_point m true p;
    wait_from m c (i + 1)
  end

let wait_points m =
  let st = m.root.entry.stamps in
  if st <> 0 then begin
    let p = stamps_write st in
    if p <> 0 then Dev.wait_point m true p;
    wait_from m st 0
  end

let wait_write m =
  let st = m.root.entry.stamps in
  if st <> 0 then
    let p = stamps_write st in
    if p <> 0 then Dev.wait_point m false p

(* Visit every point even after a refusal: reading a later transport word may
   discover a loss and stop its device. The walk carries no allocated state. *)
let rec all_from f c i ok =
  if c = 0 then ok
  else if i = uses then all_from f (stamps_next c) 0 ok
  else
    let p = stamps_use c i in
    let ok = (p = 0 || f p) && ok in
    all_from f c (i + 1) ok

let for_all_points f st =
  st = 0
  ||
  let p = stamps_write st in
  let ok = p = 0 || f p in
  all_from f st 0 ok

let check_points st = iter_points Dev.check st

(* Raises [Lost] if [m]'s memory, or [m] itself as a borrow, is a lost
   device's. *)
let[@inline] check_owner m =
  let d = m.dev in
  if Dev.is_lost d then Dev.raise_lost d;
  if m.root != m then begin
    let o = m.root.dev in
    if o != d && Dev.is_lost o then Dev.raise_lost o
  end

(* Raises [Lost] for a use of [m] that depends on every point of its stamps. *)
let[@inline] check m =
  check_owner m;
  check_points m.root.entry.stamps

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

external token : int -> released -> int -> int -> int -> token
  = "caml_rig_token"

external no_token : unit -> token = "%identity"
external released : int -> released list = "caml_rig_released"
external released_any : int -> bool = "caml_rig_released_any" [@@noalloc]
external release_close : int -> bool = "caml_rig_release_close" [@@noalloc]
external page_size : unit -> int = "caml_rig_page_size"

let no_token = no_token ()
let page = page_size ()

(* The host's heap *)

external heap_init : int -> unit = "caml_rig_heap_init"
external heap_reserve : int -> int -> bool = "caml_rig_heap_reserve" [@@noalloc]

type bytes_ba =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external heap_bytes : int -> int -> bytes_ba = "caml_rig_heap_bytes"
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

let heap_bytes n =
  if n < aligned_from then heap_bytes 1 n
  else if n > max_int - page then raise Stdlib.Out_of_memory
  else heap_bytes page n

(* Memory records *)

let is_io_memory (e : entry) =
  match e.backing with
  | Io_memory _ -> true
  | Driver_memory _ | Kept _ -> false

let entry_ids = Atomic.make 0

(* Counts an entry of [d]'s memory until it is given back: [d]'s resources
   outlive it. *)
let count_entry d n =
  if d != Dev.host then
    let l = Dev.live d in
    ignore (Atomic.Loc.fetch_and_add [%atomic.loc l.entries] n : int)

let entry ?(access = Read_write) owner backing bytes stamps =
  count_entry owner 1;
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
  }

let no_entry = entry Dev.host (Kept Nothing) 0 0

(* Claim words *)

let outside = 1
let read_only = 2
let one_claim = 4
let exclusive = -1
let consumed = -2
let exported = -3

let new_claim keep (e : entry) =
  let reached =
    match keep with
    | Bigarray _ -> true
    | Nothing | Heap _ -> (
        match e.backing with
        | Io_memory { origin = Given; _ } -> true
        | Io_memory { origin = Made; _ } | Driver_memory _ | Kept _ -> false)
  in
  let word =
    (if reached then outside else 0)
    lor if e.access = Read then read_only else 0
  in
  { count = word; generation = 0; why = "" }

let rec export (c : claim) =
  let w = Atomic.Loc.get [%atomic.loc c.count] in
  if w = exclusive then false
  else if w = exported || (w >= 0 && w land outside <> 0) then true
  else
    let w' = if w = consumed then exported else w lor outside in
    Atomic.Loc.compare_and_set [%atomic.loc c.count] w w' || export c

(* The root a record holds until it is set to the record itself: a record made
   recursively would be made twice. *)
let rec unrooted =
  {
    dev = Dev.host;
    bytes = 0;
    host = -1;
    address = -1;
    handle = 0n;
    claim = new_claim Nothing no_entry;
    entry = no_entry;
    token = no_token;
    keep = Nothing;
    root = unrooted;
  }

(* An owned memory. Every argument is given, so a host buffer's record allocates
   nothing beside it. *)
let own ~keep ~host ~address ~handle ~token dev bytes entry =
  let m =
    {
      dev;
      bytes;
      host;
      address;
      handle;
      claim = new_claim keep entry;
      entry;
      token;
      keep;
      root = unrooted;
    }
  in
  m.root <- m;
  m

let make ?(keep = Nothing) ?(host = -1) ?(address = -1) ?(handle = 0n)
    ?(token = no_token) dev bytes entry =
  own ~keep ~host ~address ~handle ~token dev bytes entry

let borrow_of (root : memory) dev ~host ~address ~handle =
  {
    dev;
    bytes = root.bytes;
    host;
    address;
    handle;
    claim = root.claim;
    entry = root.entry;
    token = no_token;
    keep = Nothing;
    root;
  }

let stamps m = m.root.entry.stamps

let locate (Region { m; r; _ }) =
  let module D = (val m) in
  D.locate r

(* Allocation events *)

(* The host's buffers reserve against its budget in the heap, which counts
   them; a device's count is its own. *)
(* A device whose resources are freed holds nothing. *)
let allocated d =
  if d == Dev.host then heap_held ()
  else match d.live with Some l -> l.used | None -> 0

let note d =
  if Prof.enabled () then
    Prof.record
      (Allocation { device = d; time = Prof.now (); allocated = allocated d })

(* Driver calls *)

type memory_kind = Rig_edge.memory = Device | Pinned | Mapped

(* [n] new bytes of [d]'s memory of [kind] as a release record, or [None] if [d]
   has not the room. An io device's memory has no kind. *)
let new_entry d (kind : memory_kind) n =
  match d.kind with
  | Driver { m; h; rid } -> (
      let module D = (val m) in
      match Dev.counted d (fun () -> D.alloc h kind n) with
      | None -> None
      | Some r when d.host_addresses && (D.locate r).host = None ->
          ignore (Dev.give d (fun () -> D.free h r) : bool);
          invalid_argf
            "Rig: %s's driver states the host addresses its memory, and \
             allocated a region without a host address"
            d.name
      | Some r ->
          let region = Region { m; h; r; rid } in
          Some
            (entry d (Driver_memory { memory = kind; region }) n (stamps_new ())))
  | Io { m; h } -> (
      let module I = (val m) in
      match Dev.counted d (fun () -> I.alloc h n) with
      | None -> None
      | Some r ->
          let region = Io_region { m; h; r } in
          Some (entry d (Io_memory { origin = Made; region }) n (stamps_new ())))
  | Host -> None
  | Ended -> Dev.raise_lost d

(* Gives back a region [d] allocated or mapped, an io region or an image:
   whether it did, which a device lost and not stopped does not ([Dev.give]). *)
let free_region d (Region { m; h; r; _ }) =
  let module D = (val m) in
  Dev.give d (fun () -> D.free h r)

let free_io d (Io_region { m; h; r }) =
  let module I = (val m) in
  Dev.give d (fun () -> I.free h r)

(* Linked code is unmapped once its record is collected. *)
let unload_now d = function
  | Loaded { m; h; i } ->
      let module D = (val m) in
      Dev.give d (fun () -> D.unload h i)
  | Linked _ -> true

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

let take_device_room (d, n) =
  let l = Dev.live d in
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
      let l = Dev.live d in
      Lock.hold l.lock;
      l.used <- l.used - n;
      Lock.release l.lock
  | Host_budget -> heap_release n
  | No_budget -> ()

(* Releases *)

(* Appends [p] after the work that may use it without naming it. Reading the
   submitted value under the append lock keeps the queue's values ordered. *)
let append_pending (d, p) =
  let v = Dev.submitted d in
  let l = Dev.live d in
  let front, back = l.pending in
  l.pending <-
    (if front == [] then ([ (v, p) ], []) else (front, (v, p) :: back))

let defer d p = Dev.protect d append_pending (d, p)

let retire d e = defer d (Free e)

(* Gives back [d]'s mapping [r] now, or once [d] stopped. *)
let unmap d r = if not (free_region d r) then defer d (Unmap (r, None))

(* Unloads [d]'s image [l] now, or once [d] stopped. *)
let unload d l = if not (unload_now d l) then defer d (Unload (l, None))

let drop_stamps (e : entry) = if e.stamps <> 0 then stamps_unref e.stamps

(* Gives [e]'s region back to its driver, and its bytes to their keeper; once
   its device stopped if it is lost and not stopped. *)
let give_back (e : entry) =
  let d = e.owner in
  let given =
    match e.backing with
    | Driver_memory { region; _ } -> free_region d region
    | Io_memory { region; _ } -> free_io d region
    | Kept _ -> true
  in
  if not given then defer d (Give e)
  else begin
    drop_stamps e;
    let budget = entry_budget e in
    give_room d budget e.bytes;
    if budget = Device_budget then note d;
    count_entry d (-1)
  end

(* Releases the mapping [mp] now if its mapper ran all the work submitted so
   far, which is then all that could use it, and is whether it did. An orphaned
   device maps its parent's copy of the memory, never the child's: the child is
   done with it without calling its driver. *)
let unmap_now mp =
  let d = mp.on in
  let v = Dev.submitted d in
  Dev.orphaned d
  || ((not (Dev.is_lost d)) || Dev.stopped d)
     && Dev.word d >= v
     && free_region d mp.map

(* Drains in flight on a device: each holds its resources, and the last to
   leave a device that holds nothing more frees them. A device holds nothing
   once its word is given back, its memory and its images have returned and
   it maps no other memory. Entering and leaving allocate nothing. *)
let enter l =
  Lock.hold l.lock;
  let entered = l.drains >= 0 in
  if entered then l.drains <- l.drains + 1;
  Lock.release l.lock;
  entered

(* A given word means a Stopped device. The list is closed last. *)
let spent l =
  l.drains = 0 && l.word_end == Given
  && Atomic.Loc.get [%atomic.loc l.entries] = 0
  && l.retiring == [] && fst l.pending == [] && l.cached = 0
  && l.pair_maps == [] && Cache.length l.mapped = 0
  && Atomic.Loc.get [%atomic.loc l.word_unmaps] = 0
  && Atomic.Loc.get [%atomic.loc l.entry_maps] = 0
  && release_close l.release

(* Whether the drain leaving [l] is the last of a device that holds nothing
   more, which then frees its resources. *)
let left l =
  l.drains <- l.drains - 1;
  let free = spent l in
  if free then l.drains <- -1;
  free

let leave d l = if Lock.protect l.lock left l then Dev.free_live d

(* Notes one of [e]'s unmaps done, and gives [e] back after the last. *)
let entry_unmapped (e : entry) =
  if Atomic.Loc.fetch_and_add [%atomic.loc e.unmaps] (-1) = 1 then give_back e

(* Notes one unmap of [d]'s word done, and gives the word back after the
   last. It holds [d]'s drains meanwhile: the count may reach [0] in another
   device's drain, and [d]'s resources must outlive the give whichever drain
   finds [d] spent. A count above [0] keeps [d] from being spent, so the enter
   holds. *)
let word_unmapped d =
  let l = Dev.live d in
  if enter l then
    match
      if Atomic.Loc.fetch_and_add [%atomic.loc l.word_unmaps] (-1) = 1 then
        match l.word_region with Some r -> unmap d r | None -> ()
    with
    | () -> leave d l
    | exception e ->
        leave d l;
        Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())

let unmapped = function Entry e -> entry_unmapped e | Word d -> word_unmapped d

(* Notes that [d]'s driver took back its mapping of [m]: [d]'s count of
   memory mappings drops, and [m]'s. *)
let unmapped_by d m =
  (match m with
  | Entry _ -> Atomic.Loc.decr [%atomic.loc (Dev.live d).entry_maps]
  | Word _ -> ());
  unmapped m

(* Gives back [d]'s mapping [r] of [m] now, or once [d] stopped. Memory and
   words count their mappings and go back to their driver at the last unmap,
   whichever side, the mapper's end or their own, takes the mapping: until
   then a mapper's driver may still name their pages, and a driver that maps
   host memory by its address would hand them to new memory at that
   address. *)
let give d r m =
  if free_region d r then unmapped_by d m else defer d (Unmap (r, Some m))

(* [d] never used the mapping, so the count goes back before the region: an
   exception in the driver's free, which allocates, leaves the region alone
   behind, and [p] still ends. *)
let drop_word_map d r p =
  word_unmapped p;
  unmap d r

(* Counts a mapping of [d]'s word about to be made while its readers still
   read it: whether it counted. Once the word moves, the end's walk may have
   passed the mappings already, so a new one would hold the word for
   good. *)
let count_word_map l =
  l.word_end = Read
  && begin
    Atomic.Loc.incr [%atomic.loc l.word_unmaps];
    true
  end

let hold_word d =
  let l = Dev.live d in
  Lock.protect l.lock count_word_map l

let drop_mapped (l, id) = Cache.remove l.mapped id

(* [e]'s mappings are taken under its owner's lock, as a mapper's word end
   takes its own: each is released once. *)
let rec forget (e : entry) = function
  | [] -> ()
  | mp :: maps ->
      (let l = Dev.live mp.on in
       Lock.protect l.lock drop_mapped (l, e.id));
      forget e maps

let free_entry (e : entry) =
  (* No mapping is made of dead memory: one seen gone stays gone. *)
  let maps =
    if e.maps == [] then []
    else begin
      let l = Dev.live e.owner in
      Lock.hold l.lock;
      let maps = e.maps in
      e.maps <- [];
      Lock.release l.lock;
      maps
    end
  in
  forget e maps;
  List.iter
    (fun mp ->
      if unmap_now mp then unmapped_by mp.on (Entry e)
      else defer mp.on (Unmap (mp.map, Some (Entry e))))
    maps;
  entry_unmapped e

let key n kind =
  let k = match kind with Device -> 0 | Pinned -> 1 | Mapped -> 2 in
  (n * 4) + k

(* Memory the cache never takes: io memory, and host memory a device borrowed,
   which returns to its keeper once its uses are reached. *)
let uncached (e : entry) =
  match e.backing with Driver_memory _ -> false | Io_memory _ | Kept _ -> true

let cache_key (e : entry) =
  match e.backing with
  | Driver_memory { memory; _ } -> key e.bytes memory
  | Io_memory _ | Kept _ -> invalid_arg "Rig: io or kept memory in a cache"

(* Holds whose release is still to run. *)
let holds_lock = Lock.create ()
let holds : hold_release list ref = ref []

(* A hold's release is due once each of its points is reached on a live device,
   and each of its lost devices is Stopped. *)
let hold_due (Hold_release { stamps = st; _ }) =
  for_all_points
    (fun p ->
      let d = Dev.of_index (Point.index p) in
      if Dev.is_lost d then Dev.stopped d else Dev.settled p)
    st

(* A hold made before a fork is forgotten in the child: its release would call
   the parent's drivers. *)
let forgotten (Hold_release { stamps; generation; _ }) =
  generation <> Dev.generation ()
  && begin
    stamps_unref stamps;
    true
  end

(* Lets go of a due hold's value: the hold no longer keeps it reachable. *)
let run_release (Hold_release { stamps; _ }) = stamps_unref stamps

let take_holds () =
  let l = !holds in
  holds := [];
  l

let put_holds l = holds := l @ !holds
let add_hold h = holds := h :: !holds

(* Takes the holds whose release is due. Their points are read with no lock
   held: a word behind a transport is read through its driver, whose fault stops
   its device, and a stop drains. *)
let take_due_holds () =
  let all = Lock.protect holds_lock take_holds () in
  let put_back l = Lock.protect holds_lock put_holds l in
  let all = List.filter (fun h -> not (forgotten h)) all in
  match List.partition hold_due all with
  | due, later ->
      put_back later;
      due
  | exception e ->
      put_back all;
      raise e

let drain_holds () =
  if !holds != [] then List.iter run_release (take_due_holds ())

let holds_list = Dev.release_list ()

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

(* Puts [e] in its owner's cache, for reuse. The owner's lock is held. *)
let cache (e : entry) =
  let l = Dev.live e.owner and k = cache_key e in
  let es = Option.value ~default:[] (Cache.find_opt l.cache k) in
  Cache.replace l.cache k (e :: es);
  l.cached <- l.cached + e.bytes

(* Puts [e] among its owner's retiring memory. The owner's lock is held. *)
let retire_entry (e : entry) =
  let l = Dev.live e.owner in
  l.retiring <- e :: l.retiring

let to_cache (e : entry) = Dev.protect e.owner cache e

(* Whether [d] holds more than its budget in the memory that [e]'s counts in:
   [e] then returns to the driver, so a later allocation the budget refuses
   cannot reuse it. It skips the cache, never the wait: the driver gets [e]
   once every point of it is reached, [d]'s own included. *)
let over_budget d (e : entry) = owns e && (Dev.live d).used > d.budget

(* Routes a record [d]'s release list gave. *)
let route d = function
  | Memory e ->
      let cached =
        (not (Dev.is_lost d))
        && (not (uncached e))
        && reached e.stamps
        && (not (viewed e))
        && not (over_budget d e)
      in
      Dev.protect e.owner (if cached then cache else retire_entry) e
  | Image (loaded, code) -> defer d (Unload (loaded, code))
  | Release h -> Lock.protect holds_lock add_hold h

type fate = Stays | Cached | Freed

(* What becomes of the retiring [e] of [d]. It reads words, so no lock is held:
   a word behind a transport is read through its driver, whose fault stops its
   device, and a stop drains. A taken entry is the one drain's that took it. *)
let fate d ~lost ~free_lost (e : entry) =
  if viewed e then Stays
  else if lost then if free_lost && reached e.stamps then Freed else Stays
  else if uncached e || over_budget d e then
    if reached e.stamps then Freed else Stays
  else if reached e.stamps then Cached
  else Stays

let put_retiring (d, es) =
  let l = Dev.live d in
  l.retiring <- es @ l.retiring

(* Judges each retiring entry of [l] with no lock held, puts back those that
   stay or enter the cache, and is those to free, consed onto [freed]. *)
let rec judge d ~lost ~free_lost freed = function
  | [] -> freed
  | e :: l -> (
      match fate d ~lost ~free_lost e with
      | Freed -> judge d ~lost ~free_lost (e :: freed) l
      | Stays ->
          Dev.protect d retire_entry e;
          judge d ~lost ~free_lost freed l
      | Cached ->
          to_cache e;
          judge d ~lost ~free_lost freed l
      | exception x ->
          Dev.protect d put_retiring (d, e :: l);
          raise x)

(* Empties [l]'s cache: its entries, consed onto [freed]. *)
let take_all_cached (l, freed) =
  let freed = Cache.fold (fun _ es freed -> es @ freed) l.cache freed in
  Cache.reset l.cache;
  l.cached <- 0;
  freed

(* Takes what became due: retiring memory no bigarray reads and whose uses are
   reached enters the cache, or is freed if [d] is lost and counts as stopped;
   pending releases whose value [d] reached. Is the memory to free. *)
let due d =
  let lost = Dev.is_lost d in
  let free_lost = lost && Dev.stopped d in
  let l = Dev.live d in
  Lock.hold l.lock;
  let retiring = l.retiring in
  l.retiring <- [];
  Lock.release l.lock;
  let freed = judge d ~lost ~free_lost [] retiring in
  if not free_lost then freed
  else Lock.protect l.lock take_all_cached (l, freed)

(* Takes [l]'s pending releases up to the value [w]. *)
let take_pending (l, w) =
  let[@tail_mod_cons] rec take front back =
    match front with
    | (v, p) :: rest when v <= w -> p :: take rest back
    | [] when back <> [] -> take (List.rev back) []
    | _ ->
        l.pending <- (front, back);
        []
  in
  let front, back = l.pending in
  take front back

(* Takes the pending releases whose value [d] reached. *)
let due_pending d =
  let l = Dev.live d in
  match l.pending with
  | [], _ -> []
  | (v, _) :: _, _ ->
      let lost = Dev.is_lost d in
      let w = if lost && not (Dev.stopped d) then -1 else Dev.word d in
      if w < v then [] else Lock.protect l.lock take_pending (l, w)

(* A release that [d], lost and not stopped since, cannot run waits again: its
   code memory only after the image's unload. *)
let run_pending d = function
  | Free e -> free_entry e
  | Give e -> give_back e
  | Unmap (r, after) as p ->
      if free_region d r then Option.iter (unmapped_by d) after else defer d p
  | Unload (loaded, code) as p -> (
      if not (unload_now d loaded) then defer d p
      else
        match code with
        | Some e when Dev.is_lost d || over_budget d e -> free_entry e
        | Some e -> to_cache e
        | None -> ())

let rec route_all d = function
  | [] -> ()
  | r :: l ->
      route d r;
      route_all d l

let rec free_all = function
  | [] -> ()
  | e :: l ->
      free_entry e;
      free_all l

let rec run_all d = function
  | [] -> ()
  | p :: l ->
      run_pending d p;
      run_all d l

(* Drains [d]'s own list and what became due on it. *)
let drain_own d =
  let l = Dev.live d in
  if released_any l.release then route_all d (released l.release);
  free_all (due d);
  run_all d (due_pending d)

(* Whether [l] has nothing to drain: no collected memory, nothing waiting for a
   value. A drain of an idle device allocates nothing. *)
let idle l =
  l.retiring == [] && fst l.pending == [] && not (released_any l.release)

let mapped_ids l = Cache.fold (fun id _ ids -> id :: ids) l.mapped []

(* Takes the entry [id] out of those [l] maps, if it is there. *)
let pop_mapped (l, id) =
  match Cache.find_opt l.mapped id with
  | Some e ->
      Cache.remove l.mapped id;
      Some e
  | None -> None

(* Takes [d]'s mapping out of [e]'s, if any. *)
let take_map ((e : entry), d) =
  match List.find_opt (fun mp -> mp.on == d) e.maps with
  | Some mp ->
      e.maps <- List.filter (fun mp' -> mp' != mp) e.maps;
      Some mp
  | None -> None

(* Releases [d]'s mappings of the memory that outlives it, once its word shows
   its last value: each is taken out of its memory's [maps] under the memory's
   owner's lock, which [free_entry] takes, so whichever comes first releases
   it, and the memory goes back at its last unmap ([give]). One mapping is
   taken out at a time and given back before the next: an exception loses at
   most the one in hand, and a later end finds the rest. *)
let unmap_all d =
  let l = Dev.live d in
  List.iter
    (fun id ->
      match Lock.protect l.lock pop_mapped (l, id) with
      | None -> ()
      | Some e -> (
          match Dev.protect e.owner take_map (e, d) with
          | Some mp -> give d mp.map (Entry e)
          | None -> ()))
    (Lock.protect l.lock mapped_ids l)

(* Takes out of [l]'s pair maps one mapping of [d]'s word, if any. *)
let pop_pair_map (l, d) =
  match List.partition (fun (p, _) -> p == d) l.pair_maps with
  | [], _ -> None
  | (_, r) :: rest, others ->
      l.pair_maps <- rest @ others;
      Some r

(* Gives back [c]'s mappings of [d]'s word, one at a time, each while it
   holds [c]'s drains: [c]'s resources then outlive the mapping in hand. A
   [c] whose drains are closed has freed its resources, and has no mapping
   left. *)
let rec give_word_maps d c =
  match c.live with
  | Some l when enter l -> (
      match
        match Lock.protect l.lock pop_pair_map (l, d) with
        | Some r ->
            give c r (Word d);
            true
        | None -> false
      with
      | more ->
          leave c l;
          if more then give_word_maps d c
      | exception e ->
          leave c l;
          Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ()))
  | _ -> ()

(* Whether this drain gives the word back: no other drain took that step. *)
let give_word l =
  match l.word_end with
  | Moved ->
      l.word_end <- Given;
      true
  | Read | Given -> false

(* Takes one of [l]'s mappings of other devices' words, if any. *)
let pop_own_word_map l =
  match l.pair_maps with
  | pm :: rest ->
      l.pair_maps <- rest;
      Some pm
  | [] -> None

(* Gives back [d]'s mappings of other devices' words, one at a time. *)
let rec give_own_word_maps d =
  let l = Dev.live d in
  match Lock.protect l.lock pop_own_word_map l with
  | Some (p, r) ->
      give d r (Word p);
      give_own_word_maps d
  | None -> ()

(* The minor count at which [l]'s word moved, read under the lock that
   orders it after the move. *)
let moved_at l =
  Lock.hold l.lock;
  let n = l.moved_at in
  Lock.release l.lock;
  n

(* Gives back the stopped [d]'s timeline word once nothing reads it: its
   readers move to the C line's copy, then, once every domain passed a minor
   collection since, the driver gets the word back at the last unmap of it,
   this one's or a mapper's end's ([give]). [d]'s own mappings of other
   devices' words go then too: its queue reads them no more.

   [move_word] answers [true] once, so the move is recorded with stores
   alone, and read back under the same lock. While the word reads [Moved],
   no new mapping of it counts, and the mappings are taken out and given
   back one at a time: an exception loses at most the one in hand, and the
   next drain finds the rest. The word's own count goes last, with the step
   to [Given] that only one drain takes. *)
let end_word d =
  let l = Dev.live d in
  match l.word_end with
  | Given -> ()
  | Read ->
      if Dev.stopped d && Dev.move_word d then begin
        let n = Dev.minors () in
        Lock.hold l.lock;
        l.moved_at <- n;
        l.word_end <- Moved;
        Lock.release l.lock
      end
  | Moved ->
      if Dev.minors () > moved_at l then begin
        Dev.iter (give_word_maps d);
        give_own_word_maps d;
        unmap_all d;
        if Lock.protect l.lock give_word l then word_unmapped d
      end

(* Drains the devices whose stop returned ({!Dev.next_ended}), from the index
   [i] on, other than [d], and ends their words: every drain drains those
   that hold memory or have a release waiting, and reads the word of those
   whose work may still run. It allocates nothing for an idle one, as every
   drain walks them all. *)
let rec drain_lost d i =
  match Dev.next_ended i with
  | 0 -> ()
  | i ->
      let e = Dev.of_index i in
      (match e.live with
      | Some l when (not (Dev.orphaned e)) && enter l ->
          (match
             if l.word_end != Given then end_word e;
             if e != d && not (idle l && l.cached = 0) then drain_own e
           with
          | () -> leave e l
          | exception x ->
              leave e l;
              raise x)
      | _ -> ());
      drain_lost d (i + 1)

(* An orphaned device's objects are forgotten: nothing of it drains. *)
let drain d =
  if not (Dev.orphaned d) then begin
    (match d.live with
    | Some l when (not (idle l)) && enter l -> (
        match drain_own d with
        | () -> leave d l
        | exception x ->
            leave d l;
            raise x)
    | _ -> ());
    drain_lost d 1;
    if released_any holds_list then List.iter (route d) (released holds_list);
    if !holds != [] then drain_holds ()
  end

let () = Dev.answered := drain

(* The cache *)

let take_cached_key (l, k) =
  match Cache.find_opt l.cache k with
  | Some (e :: rest) ->
      (match rest with
      | [] -> Cache.remove l.cache k
      | _ -> Cache.replace l.cache k rest);
      l.cached <- l.cached - e.bytes;
      (* Every old use has settled. The new buffer has no dependencies on
         its memory's previous lifetime, an unordered use included. *)
      stamps_clear e.stamps;
      Some e
  | _ -> None

let take_cached d kind n =
  let l = Dev.live d in
  Lock.protect l.lock take_cached_key (l, key n kind)

(* Takes the cached memory of [l] that [take] picks out of its cache. The
   changes are worked out before any is made, and each key's list changes
   with the count. *)
let take_picked (l, take) =
  let pick k es picked =
    match List.partition take es with
    | [], _ -> picked
    | taken, kept ->
        let n = List.fold_left (fun n (e : entry) -> n + e.bytes) 0 taken in
        (k, taken, kept, n) :: picked
  in
  List.fold_left
    (fun all (k, taken, kept, n) ->
      (match kept with
      | [] -> Cache.remove l.cache k
      | _ -> Cache.replace l.cache k kept);
      l.cached <- l.cached - n;
      List.rev_append taken all)
    []
    (Cache.fold pick l.cache [])

(* Takes the cached memory of [d] that [take] picks out of its cache, in no
   particular order: none once [d]'s resources are freed. *)
let take_cache_if d take =
  match d.live with
  | None -> []
  | Some l -> Lock.protect l.lock take_picked (l, take)

(* Takes cached memory out of [d]'s cache: all of it, or with [upto] only memory
   that counts in [d]'s budget, until [d] holds at most [upto] bytes. *)
let take_cache ?upto d =
  match upto with
  | None -> take_cache_if d (fun _ -> true)
  | Some upto ->
      let held = ref (Dev.live d).used in
      take_cache_if d (fun e ->
          owns e && !held > upto
          && begin
            held := !held - e.bytes;
            true
          end)

(* Frees what [take_cache] takes once the work [d] was handed until now is done:
   at once if it is, after waiting for it when [wait], and otherwise each free
   is deferred until it is. A lost device's cache waits for it to count as
   stopped. *)
(* Waits for [d]'s value [v] on behalf of memory, never of [d]'s caller: a loss
   the wait finds is [d]'s alone, whose memory then returns once it stopped. *)
let wait_for d v = try Dev.wait d v with Dev.Lost (d', _) when d' == d -> ()

(* Whether [d]'s work up to [v] is done, or [d] is lost: its frees then wait for
   its stop themselves. *)
let reached_or_lost d v =
  Dev.is_lost d || match Dev.word d >= v with r -> r | exception Dev.Lost _ -> true

let release_taken ~wait d take =
  if (not (Dev.is_lost d)) || Dev.stopped d then
    match take d with
    | [] -> ()
    | taken ->
        let v = Dev.submitted d in
        if wait && not (Dev.is_lost d) then wait_for d v;
        if reached_or_lost d v then List.iter free_entry taken
        else List.iter (fun e -> defer d (Free e)) taken

(* Whether a call holds [d]'s lock, or its resources are freed: either way a
   walk over every device passes it by. *)
let busy d = match d.live with Some l -> Lock.busy l.lock | None -> true

let release_cache ?upto ~wait d = release_taken ~wait d (take_cache ?upto)

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

(* Whether [d] holds memory that counts in [pool]'s budget back from its return
   until work is done: collected memory outside the cache, and returns deferred
   until [d]'s handed work is done. Only the entries' plain fields are read: a
   drain may free them meanwhile. *)
let holds_back pool l =
  List.exists (charged pool) l.retiring
  ||
  let holds = function
    | _, (Free e | Give e | Unmap (_, Some (Entry e)) | Unload (_, Some e)) ->
        charged pool e
    | _, (Unmap (_, (None | Some (Word _))) | Unload (_, None)) -> false
  in
  let front, back = l.pending in
  List.exists holds front || List.exists holds back

(* Drains every device, the host included, skipping one whose lock another call
   holds. *)
let drain_all () =
  drain Dev.host;
  Dev.iter (fun d -> if not (busy d) then drain d)

(* A round of the ladder for [pool]'s budget: every cached memory on any device
   that counts in it returns once its device's handed work is done, waited for;
   for the host, its kept buffers too. A device whose handed work holds back the
   return of memory that counts in it is waited for too, so that memory returns
   and a loss shows as one. Then every device drains, which returns collected
   memory and runs the unmaps that hold it; from the second round a collection
   finds the memory unreachable since, and every device drains again. *)
let reclaim_round pool round =
  if Dev.is_host pool then heap_drop ();
  Dev.iter (fun d ->
      match d.live with
      | Some l when not (Lock.busy l.lock) ->
          if holds_back pool l && not (Dev.is_lost d) then
            wait_for d (Dev.submitted d);
          release_taken ~wait:true d (fun d -> take_cache_if d (charged pool))
      | _ -> ());
  drain_all ();
  if round >= 2 then begin
    Gc.full_major ();
    drain_all ()
  end

let rec reclaim_from d pool n f round =
  if round >= rounds then raise (Dev.Out_of_memory (d, n));
  reclaim_round pool round;
  match f () with Some x -> x | None -> reclaim_from d pool n f (round + 1)

let reclaiming d ~pool n f = reclaim_from d pool n f 1

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
      let ask () = Dev.counted d (fun () -> D.entry i f) in
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

(* Allocation *)

let room d = Int.max 0 (d.budget - (Dev.live d).used)

(* A fresh memory record over the entry [e] of [d], with its token. *)
let of_entry d e =
  let m =
    match e.backing with
    | Driver_memory { region; _ } ->
        let l = locate region in
        make ?host:l.host ?address:l.address ~handle:l.handle d e.bytes e
    | Io_memory _ | Kept _ -> make d e.bytes e
  in
  let l = Dev.live d in
  let live = if m.host >= 0 then l.used else -1 in
  m.token <- token l.release (Memory e) e.bytes (room d) live;
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

let heap_reserved n =
  match reserve n () with
  | Some () -> ()
  | None -> reclaiming Dev.host ~pool:Dev.host n (reserve n)

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
          heap_release n;
          raise x)

(* What every host buffer of no bytes keeps: nothing to free. *)
let empty = Bigarray.Array1.create Bigarray.char Bigarray.c_layout 0
let empty_keep = Heap empty

let host_memory n =
  drain Dev.host;
  if n = 0 then
    let addr = ba_address empty in
    own ~keep:empty_keep ~host:addr ~address:addr ~handle:0n ~token:no_token
      Dev.host 0 no_entry
  else begin
    heap_reserved n;
    let ba = host_bytes n in
    let addr = ba_address ba in
    own ~keep:(Heap ba) ~host:addr ~address:addr ~handle:0n ~token:no_token
      Dev.host n no_entry
  end

(* Borrows *)

let make_entry m =
  if m.entry == no_entry then begin
    let d = m.dev in
    let e = entry d (Kept m.keep) m.bytes (stamps_new ()) in
    m.entry <- e;
    m.token <- token (Dev.live d).release (Memory e) 0 max_int (-1)
  end

let ensure_entry m = Dev.protect m.dev make_entry m

(* [d]'s mapping of [e], if any. *)
let find_map ((e : entry), d) = List.find_opt (fun mp -> mp.on == d) e.maps

(* Notes [e] among the entries [l] maps while [l]'s word is read: whether it
   noted it. *)
let note_mapped (l, (e : entry)) =
  l.word_end = Read
  && begin
    Cache.replace l.mapped e.id e;
    true
  end

(* Counts [mp] in [e] and in its mapper, and adds it to [e]'s mappings,
   unless its device mapped [e] meanwhile: that mapping, then. *)
let add_map ((e : entry), mp) =
  Atomic.Loc.incr [%atomic.loc e.unmaps];
  Atomic.Loc.incr [%atomic.loc (Dev.live mp.on).entry_maps];
  match find_map (e, mp.on) with
  | Some other -> Some other
  | None ->
      e.maps <- mp :: e.maps;
      None

(* Takes [mp] back out of [e]'s mappings: whether it was there. *)
let take_back ((e : entry), mp) =
  List.memq mp e.maps
  && begin
    e.maps <- List.filter (fun mp' -> mp' != mp) e.maps;
    true
  end

let map_host_range d start n =
  match d.kind with
  | Driver _ when not d.maps_host -> None
  | Driver { m = dm; h; rid } -> (
      let module D = (val dm) in
      match Dev.counted d (fun () -> D.map_host h start n) with
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
          match Dev.counted d (fun () -> O.map_peer h oh r) with
          | Some r -> Some (Region { m = om; h; r; rid })
          | None -> None)
      | None -> None)
  | _ -> None

let map_peer d m =
  match m.entry.backing with
  | Driver_memory { region; _ } -> map_peer_region d region
  | Io_memory _ | Kept _ -> None

(* Whether [d] maps [m] as a peer's memory: memory of a device of [d]'s
   driver. *)
let peer_memory d (m : memory) =
  match m.dev.kind with
  | Driver _ -> m.dev.key = d.key
  | Host | Io _ | Ended -> false

(* [d]'s new mapping of the memory [m] owns, whose host address is [host] or
   [-1]. It counts in the memory and is recorded there first, then noted on
   [d] while [d]'s word is read: a word end of [d] that came first found no
   note, so the mapping is taken back out of the memory and given back here,
   and [d], whose word ended, is lost. A word end that comes later finds the
   note. *)
let map_new d m host =
  (* Memory of a device of [d]'s driver maps as a peer's; other memory the
     host addresses, as host memory. *)
  let made =
    if peer_memory d m then map_peer d m
    else if host >= 0 then map_host_range d host m.bytes
    else map_peer d m
  in
  match made with
  | None -> None
  | Some r -> (
      let e = m.entry in
      let loc =
        match locate r with
        | loc -> loc
        | exception x ->
            unmap d r;
            Printexc.raise_with_backtrace x (Printexc.get_raw_backtrace ())
      in
      let at = Option.value ~default:(-1) loc.address in
      let mp = { on = d; map = r; at; by = loc.handle } in
      (* A mapping that lost the race counts too, until it is given back, so
         the memory outlives it. *)
      let raced = Dev.protect m.dev add_map (e, mp) in
      match raced with
      | Some other ->
          give d r (Entry e);
          Some other
      | None ->
          let l = Dev.live d in
          let noted = Lock.protect l.lock note_mapped (l, e) in
          if noted then Some mp
          else begin
            let mine = Dev.protect m.dev take_back (e, mp) in
            if mine then give d r (Entry e);
            Dev.raise_lost d
          end)

(* [d]'s mapping of the memory [m] owns, whose host address is [host] or [-1],
   made at the first borrow and shared by the later ones. A borrow maps outside
   any counted call on [d], so [d]'s drains are held across the map and its
   record: a close on another domain cannot free [d]'s resources in between. *)
let mapping d m host =
  let found = Dev.protect m.dev find_map (m.entry, d) in
  match found with
  | Some mp -> Some mp
  | None -> (
      let l = Dev.live d in
      if not (enter l) then Dev.raise_lost d
      else
        match map_new d m host with
        | made ->
            leave d l;
            made
        | exception x ->
            leave d l;
            Printexc.raise_with_backtrace x (Printexc.get_raw_backtrace ()))

(* Keeps [got] as [e]'s pages, unless a call kept some first. *)
let set_pages ((e : entry), got) =
  match e.pages with Unasked -> e.pages <- got | Pages _ | No_pages -> ()

(* The host address of the io memory [m]'s pages, or [-1] if it maps none:
   the first result from its device, kept. A call that raises leaves it to
   ask again. Which pages a driver refuses to map is its own: each driver's
   .mli states it, from NVIDIA's os-mlock.c and Linux's mm/gup.c
   ([writable_file_mapping_allowed]) for CUDA and NV, and from amdgpu_hmm.c
   for AMD. *)
let pages (m : memory) =
  let e = m.entry in
  (match e.pages with
  | Unasked -> (
      match e.backing with
      | Io_memory { region = Io_region { m = im; h; r }; _ } ->
          let module I = (val im) in
          let got =
            match Dev.counted m.dev (fun () -> I.pages h r) with
            | Some ba -> Pages ba
            | None -> No_pages
          in
          Dev.protect m.dev set_pages (e, got)
      | Driver_memory _ | Kept _ -> ())
  | Pages _ | No_pages -> ());
  match e.pages with Pages ba -> ba_address ba | Unasked | No_pages -> -1

(* How [d] reaches the root memory [m] whose host address is [host]: constant
   answers, so asking builds nothing. *)
type reach =
  | Cannot (* [d] cannot map it *)
  | Itself (* [m] is [d]'s *)
  | At_host (* [d] is the host, which addresses it *)
  | Addressed (* [d] addresses host memory as the host does *)
  | Empty (* no bytes to map *)
  | Map (* through [d]'s mapping of it *)

let[@inline] reach d (m : memory) host =
  if m.dev == d then Itself
  else if Dev.is_io d || (Dev.is_io m.dev && host < 0) then Cannot
  else if (not (Dev.same_machine m.dev d)) && not (Dev.is_io m.dev) then Cannot
  else if Dev.is_host d then if host >= 0 then At_host else Cannot
  else if d.memory_device && host >= 0 then Addressed
  else if m.bytes = 0 then Empty
  else if host >= 0 && host mod page <> 0 && not (peer_memory d m) then
    (* [mapping] maps it as host memory, which a driver maps from a page
       ([map_host]). *)
    Cannot
  else Map

(* Whether [m] is io memory of no bytes, which every other device borrows over
   no pages. *)
let[@inline] empty_io d (m : memory) =
  is_io_memory m.entry && m.bytes = 0 && m.dev != d

let[@inline] host_of (m : memory) =
  if is_io_memory m.entry then pages m else m.host

let borrow d m =
  let m = m.root in
  if empty_io d m then
    let at = ba_address empty in
    Some (borrow_of m d ~host:at ~address:at ~handle:0n)
  else
    let host = host_of m in
    match reach d m host with
    | Cannot -> None
    | Itself -> Some m
    | At_host -> Some (borrow_of m d ~host ~address:host ~handle:0n)
    | Addressed ->
        Some (borrow_of m d ~host ~address:host ~handle:(Nativeint.of_int host))
    | Empty -> Some (borrow_of m d ~host ~address:0 ~handle:0n)
    | Map -> (
        if m.entry == no_entry then ensure_entry m;
        (* A borrow of host memory keeps its host address: it is host memory,
           which the host copies. *)
        match mapping d m host with
        | None -> None
        | Some mp -> Some (borrow_of m d ~host ~address:mp.at ~handle:mp.by))

(* Whether one of [maps] is [d]'s. A loop of its own, so asking builds no
   closure. *)
let rec mapped_on d = function
  | [] -> false
  | mp :: maps -> mp.on == d || mapped_on d maps

let maps d m =
  let m = m.root in
  empty_io d m
  ||
  let host = host_of m in
  match reach d m host with
  | Cannot -> false
  | Itself | At_host | Addressed | Empty -> true
  | Map ->
      if m.entry == no_entry then ensure_entry m;
      let l = Dev.live m.dev in
      Lock.hold l.lock;
      let maps = m.entry.maps in
      Lock.release l.lock;
      mapped_on d maps || Option.is_some (mapping d m host)

(* Asks an io device to read ahead the bytes that a borrow on [d], a device
   other than the host, reaches through its pages. It is a hint, so its
   failures are dropped. *)
let prefetch d (m : memory) ~at ~len =
  match m.root.entry.backing with
  | Io_memory { region = Io_region { m = im; h; r }; _ } when not (Dev.is_host d)
    -> (
      let module I = (val im) in
      try I.prefetch h r ~at ~len with I.Fault _ | Sys_error _ -> ())
  | _ -> ()

let of_io d r ~access n =
  drain d;
  let e = entry ~access d (Io_memory { origin = Given; region = r }) n (stamps_new ()) in
  let m = make d n e in
  m.token <- token (Dev.live d).release (Memory e) n max_int (-1);
  m

(* A device whose resources are freed keeps the budget set last, and holds no
   cache. So does {!Dev.forgotten}, which every point of a collected device
   names: a budget set on it is the one they all answer. *)
let set_device_budget (d, n) = d.budget <- n

let set_budget d n =
  if n < 0 then invalid_argf "Rig.set_budget: budget %d is negative" n;
  match d.live with
  | None -> d.budget <- n
  | Some l ->
      Lock.protect l.lock set_device_budget (d, n);
      if Dev.is_host d then heap_trim n
      else release_cache ~upto:n ~wait:false d

let free_cache d =
  if Dev.is_host d then heap_drop ()
  else if not (Dev.freed d) then release_cache ~wait:false d
