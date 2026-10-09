(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Def
module Cache = Hashtbl.Make (Int)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* Stamps *)

external stamps_new : unit -> int = "caml_rig_stamps_new"
external stamps_ref : int -> unit = "caml_rig_stamps_ref" [@@noalloc]
external stamps_unref : int -> unit = "caml_rig_stamps_unref" [@@noalloc]
external stamps_get : int -> int -> int = "caml_rig_stamps_get" [@@noalloc]
external stamps_hold : int -> int -> unit = "caml_rig_stamps_hold"
[@@noalloc]

external stamps_held : int -> bool = "caml_rig_stamps_held" [@@noalloc]
external stamps_keep : int -> int -> unit = "caml_rig_stamps_keep" [@@noalloc]

(* [f] over the points of the stamps [st] from the [k]th on. *)
let rec iter_from f st k =
  let p = stamps_get st k in
  if p <> -1 then begin
    if p <> 0 then f p;
    iter_from f st (k + 1)
  end

let iter_points f st = if st <> 0 then iter_from f st 0
let held st = st <> 0 && stamps_held st

let iter_write f st =
  if st <> 0 then
    let p = stamps_get st 0 in
    if p > 0 then f p

let for_all_points f st =
  let ok = ref true in
  iter_points (fun p -> if not (f p) then ok := false) st;
  !ok

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

(* Raises [Lost] for a use of [m] that follows every point of its stamps. *)
let[@inline] check m =
  check_owner m;
  check_points m.root.entry.stamps

(* Whether every point of [st] other than [except]'s has settled. *)
let reached ?(except = -1) st =
  for_all_points (fun p -> Point.index p = except || Dev.settled p) st

(* Tokens and release lists *)

external token : int -> released -> int -> int -> int -> token
  = "caml_rig_token"

external no_token : unit -> token = "%identity"
external released : int -> released list = "caml_rig_released"
external released_any : int -> bool = "caml_rig_released_any" [@@noalloc]
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
  match e.memory with
  | Io_made | Io_given -> true
  | Device | Pinned | Mapped | Host_kept -> false

let entry_ids = Atomic.make 0

let entry ?region ?io_region ?(access = Read_write) owner memory bytes stamps =
  {
    id = Atomic.fetch_and_add entry_ids 1;
    owner;
    memory;
    bytes;
    region;
    io_region;
    access;
    stamps;
    maps = [];
    unmaps = 0;
    pages = Unasked;
    proxy = 0;
    kept = Nothing;
  }

let no_entry = entry Dev.host Host_kept 0 0

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
    | Nothing | Heap _ -> e.memory = Io_given
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

let region_info (Region { m; r; _ }) =
  let module D = (val m) in
  let l = D.locate r in
  let address = Option.value ~default:(-1) l.address in
  let host = Option.value ~default:(-1) l.host in
  (address, l.handle, host)

(* Allocation events *)

let note d =
  if Prof.enabled () then
    Prof.record
      (Allocation { device = d; time = Prof.now (); allocated = d.used })

(* Driver calls *)

let edge_memory : memory_kind -> Rig_edge.memory = function
  | Pinned -> Pinned
  | Mapped -> Mapped
  | Device | Host_kept | Io_made | Io_given -> Device

(* [n] new bytes of [d]'s memory of [kind] as a release record, or [None] if [d]
   has not the room. *)
let new_entry d kind n =
  match d.kind with
  | Driver { m; h; rid } -> (
      let module D = (val m) in
      match Dev.counted d (fun () -> D.alloc h (edge_memory kind) n) with
      | None -> None
      | Some r ->
          Some
            (entry ~region:(Region { m; h; r; rid }) d kind n (stamps_new ())))
  | Io { m; h } -> (
      let module I = (val m) in
      match Dev.counted d (fun () -> I.alloc h n) with
      | None -> None
      | Some r ->
          Some
            (entry
               ~io_region:(Io_region { m; h; r })
               d Io_made n (stamps_new ())))
  | Host -> None

(* Gives back a region [d] allocated or mapped, an io region or an image:
   whether it did, which a device lost and not stopped does not ([Dev.give]). *)
let free_region d (Region { m; h; r; _ }) =
  let module D = (val m) in
  Dev.give d (fun () -> D.free h r)

let free_io d (Io_region { m; h; r }) =
  let module I = (val m) in
  Dev.give d (fun () -> I.free h r)

let unload_now d (Loaded { m; h; i }) =
  let module D = (val m) in
  Dev.give d (fun () -> D.unload h i)

let kernel_entry image f =
  let (Loaded { m; i; _ }) = image.loaded in
  let module D = (val m) in
  Dev.counted image.idev (fun () -> D.entry i f)

(* The budget memory of [kind] on [d] counts in. Pinned memory is [d]'s own
   where the host addresses [d]'s memory, and host memory elsewhere. *)
type budget = Device_budget | Host_budget | No_budget

let budget_of d = function
  | Device | Mapped | Io_made -> Device_budget
  | Pinned -> if Dev.reaches Dev.host d then Device_budget else Host_budget
  | Host_kept | Io_given -> No_budget

let owns d kind = budget_of d kind = Device_budget

(* Takes [n] bytes of the room of [budget] on [d], if it has them. *)
let take_room d budget n =
  match budget with
  | Device_budget ->
      Dev.protect d (fun () ->
          d.used + n <= d.budget
          && begin
            d.used <- d.used + n;
            true
          end)
  | Host_budget -> heap_reserve n Dev.host.budget
  | No_budget -> true

let give_room d budget n =
  match budget with
  | Device_budget ->
      Dev.hold d;
      d.used <- d.used - n;
      Dev.release d
  | Host_budget -> heap_release n
  | No_budget -> ()

(* Releases *)

(* Puts [p] on [d]'s pending list, due once [d] reached the value it has
   submitted now: the work that may use it without naming it. *)
let defer d p =
  let v = Dev.submitted d in
  Dev.protect d (fun () -> d.pending <- (v, p) :: d.pending)

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
    (match e.region with Some r -> free_region d r | None -> true)
    && match e.io_region with Some r -> free_io d r | None -> true
  in
  if not given then defer d (Free e)
  else begin
    drop_stamps e;
    let budget = budget_of d e.memory in
    give_room d budget e.bytes;
    if budget = Device_budget then note d
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

(* [e] waits for its last unmap: until then a mapper's driver may still name the
   memory's pages, and a driver that maps host memory by its address would hand
   them to new memory at that address. Its mappings are taken under its owner's
   lock, as a mapper's word end takes its own: each is released once. *)
let rec forget (e : entry) = function
  | [] -> ()
  | mp :: maps ->
      Dev.hold mp.on;
      Cache.remove mp.on.mapped e.id;
      Dev.release mp.on;
      forget e maps

let free_entry (e : entry) =
  (* No mapping is made of dead memory: one seen gone stays gone. *)
  let maps =
    if e.maps == [] then []
    else begin
      Dev.hold e.owner;
      let maps = e.maps in
      e.maps <- [];
      Dev.release e.owner;
      maps
    end
  in
  forget e maps;
  let later = List.filter (fun mp -> not (unmap_now mp)) maps in
  match later with
  | [] -> give_back e
  | _ ->
      Atomic.Loc.set [%atomic.loc e.unmaps] (List.length later);
      List.iter (fun mp -> defer mp.on (Unmap (mp.map, Some e))) later

(* Notes one of [e]'s unmaps done, and gives [e] back after the last. *)
let unmapped (e : entry) =
  if Atomic.Loc.fetch_and_add [%atomic.loc e.unmaps] (-1) = 1 then give_back e

let key n kind =
  let k =
    match kind with
    | Device -> 0
    | Pinned -> 1
    | Mapped -> 2
    | Host_kept -> 3
    | Io_made -> 4
    | Io_given -> 5
  in
  (n * 8) + k

let cache_key (e : entry) = key e.bytes e.memory

(* Holds whose release is still to run: their stamps, release and the
   generation they were made in. *)
let holds_lock = Lock.create ()
let holds : (int * (unit -> unit) * int) list ref = ref []

(* A hold's release is due once each of its points is reached on a live device,
   and each of its lost devices is Stopped. *)
let hold_due (st, _, _) =
  for_all_points
    (fun p ->
      let d = Dev.of_index (Point.index p) in
      if Dev.is_lost d then Dev.stopped d else Dev.settled p)
    st

(* A hold made before a fork is forgotten in the child: its release would call
   the parent's drivers. *)
let forgotten (st, _, generation) =
  generation <> Dev.generation ()
  && begin
    stamps_unref st;
    true
  end

(* Runs [release] as a call in flight on each device of [st] that is not
   lost. *)
let run_release (st, release, _) =
  let devices = ref [] in
  iter_points
    (fun p ->
      let d = Dev.of_index (Point.index p) in
      if (not (Dev.is_lost d)) && not (List.memq d !devices) then
        devices := d :: !devices)
    st;
  let rec go = function
    | [] -> release ()
    | d :: ds -> Dev.counted d (fun () -> go ds)
  in
  Fun.protect ~finally:(fun () -> stamps_unref st) (fun () -> go !devices)

(* Takes the holds whose release is due. Their points are read with no lock
   held: a word behind a transport is read through its driver, whose fault stops
   its device, and a stop drains. *)
let take_due_holds () =
  let all =
    Lock.protect holds_lock (fun () ->
        let l = !holds in
        holds := [];
        l)
  in
  let put_back l = Lock.protect holds_lock (fun () -> holds := l @ !holds) in
  let all = List.filter (fun h -> not (forgotten h)) all in
  match List.partition hold_due all with
  | due, later ->
      put_back later;
      due
  | exception e ->
      put_back all;
      raise e

let drain_holds () =
  if !holds != [] then begin
    let first = ref None in
    List.iter
      (fun h ->
        try run_release h with e -> if !first = None then first := Some e)
      (take_due_holds ());
    Option.iter raise !first
  end

let holds_list = Dev.release_list ()

(* Bigarrays over memory *)

external proxy_new : unit -> int = "caml_rig_proxy_new"
external proxy_drop : int -> bool = "caml_rig_proxy_drop" [@@noalloc]

let proxy (e : entry) =
  Dev.protect e.owner (fun () ->
      if e.proxy = 0 then e.proxy <- proxy_new ();
      e.proxy)

(* Whether a bigarray over [e]'s memory may be reachable, which keeps the memory
   from reuse and from its free. Once none is, the proxy is freed. *)
let viewed (e : entry) =
  if e.proxy = 0 then false
  else if proxy_drop e.proxy then begin
    e.proxy <- 0;
    false
  end
  else true

(* Puts the entry [e] of [d] in [d]'s cache, for reuse. [d]'s lock is held. *)
let cache d e =
  let k = cache_key e in
  let l = Option.value ~default:[] (Cache.find_opt d.cache k) in
  Cache.replace d.cache k (e :: l);
  d.cached <- d.cached + e.bytes

let to_cache d e = Dev.protect d (fun () -> cache d e)

(* Memory the cache never takes: io memory, and host memory a device borrowed,
   which returns to its keeper once its uses are reached. *)
let uncached (e : entry) = e.memory = Host_kept || is_io_memory e

(* Whether [d] holds more than its budget in the memory that [e]'s counts in:
   [e] then returns to the driver, so a later allocation the budget refuses
   cannot reuse it. It skips the cache, never the wait: the driver gets [e]
   once every point of it is reached, [d]'s own included. *)
let over_budget d (e : entry) = owns d e.memory && d.used > d.budget

(* Routes a record [d]'s release list gave. *)
let route d = function
  | Memory e ->
      let cached =
        (not (Dev.is_lost d))
        && (not (uncached e))
        && reached ~except:d.index e.stamps
        && (not (viewed e))
        && not (over_budget d e)
      in
      Dev.hold d;
      if cached then cache d e else d.retiring <- e :: d.retiring;
      Dev.release d
  | Image (loaded, code) -> defer d (Unload (loaded, code))
  | Release { stamps; release; generation } ->
      Lock.protect holds_lock (fun () ->
          holds := (stamps, release, generation) :: !holds)

type fate = Stays | Cached | Freed

(* What becomes of the retiring [e] of [d]. It reads words, so no lock is held:
   a word behind a transport is read through its driver, whose fault stops its
   device, and a stop drains. A taken entry is the one drain's that took it. *)
let fate d ~lost ~free_lost (e : entry) =
  if viewed e then Stays
  else if lost then if free_lost && reached e.stamps then Freed else Stays
  else if uncached e || over_budget d e then
    if reached e.stamps then Freed else Stays
  else if reached ~except:d.index e.stamps then Cached
  else Stays

(* Judges each retiring entry of [l] with no lock held, puts back those that
   stay or enter the cache, and is those to free, consed onto [freed]. *)
let rec judge d ~lost ~free_lost freed = function
  | [] -> freed
  | e :: l -> (
      match fate d ~lost ~free_lost e with
      | Freed -> judge d ~lost ~free_lost (e :: freed) l
      | Stays ->
          Dev.hold d;
          d.retiring <- e :: d.retiring;
          Dev.release d;
          judge d ~lost ~free_lost freed l
      | Cached ->
          Dev.hold d;
          cache d e;
          Dev.release d;
          judge d ~lost ~free_lost freed l
      | exception x ->
          Dev.hold d;
          d.retiring <- (e :: l) @ d.retiring;
          Dev.release d;
          raise x)

(* Takes what became due: retiring memory no bigarray reads and whose foreign
   uses are reached enters the cache, or is freed if [d] is lost and counts as
   stopped; pending releases whose value [d] reached. Is the memory to free. *)
let due d =
  let lost = Dev.is_lost d in
  let free_lost = lost && Dev.stopped d in
  Dev.hold d;
  let retiring = d.retiring in
  d.retiring <- [];
  Dev.release d;
  let freed = judge d ~lost ~free_lost [] retiring in
  if not free_lost then freed
  else begin
    Dev.hold d;
    let freed = Cache.fold (fun _ l freed -> l @ freed) d.cache freed in
    Cache.reset d.cache;
    d.cached <- 0;
    Dev.release d;
    freed
  end

(* Takes the pending releases whose value [d] reached. *)
let due_pending d =
  if d.pending == [] then []
  else
    let lost = Dev.is_lost d in
    let w = if lost && not (Dev.stopped d) then -1 else Dev.word d in
    Dev.protect d (fun () ->
        let pending, later =
          List.partition (fun (v, _) -> w >= 0 && v <= w) d.pending
        in
        d.pending <- later;
        List.map snd pending)

(* A release that [d], lost and not stopped since, cannot run waits again: its
   code memory only after the image's unload. *)
let run_pending d = function
  | Free e -> free_entry e
  | Unmap (r, after) as p ->
      if free_region d r then Option.iter unmapped after else defer d p
  | Unload (loaded, code) as p -> (
      if not (unload_now d loaded) then defer d p
      else
        match code with
        | Some e when Dev.is_lost d || over_budget d e -> free_entry e
        | Some e -> to_cache d e
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
  if released_any d.release then route_all d (released d.release);
  free_all (due d);
  run_all d (due_pending d)

(* Whether [d] has nothing to drain: no collected memory, nothing waiting for a
   value. A drain of an idle device allocates nothing. *)
let idle d = d.retiring == [] && d.pending == [] && not (released_any d.release)

(* Releases [d]'s mappings of the memory that outlives it, once its word shows
   its last value: each is taken out of its memory's [maps] under the memory's
   owner's lock, which [free_entry] takes, so whichever comes first releases
   it. *)
let unmap_all d =
  let mapped =
    Dev.protect d (fun () ->
        let l = Cache.fold (fun _ e l -> e :: l) d.mapped [] in
        Cache.reset d.mapped;
        l)
  in
  List.iter
    (fun (e : entry) ->
      let mine =
        Dev.protect e.owner (fun () ->
            match List.find_opt (fun mp -> mp.on == d) e.maps with
            | Some mp ->
                e.maps <- List.filter (fun mp' -> mp' != mp) e.maps;
                Some mp
            | None -> None)
      in
      Option.iter (fun mp -> unmap d mp.map) mine)
    mapped

(* Gives back the stopped [d]'s timeline word once nothing reads it: its
   readers move to the C record's copy, then, once every domain passed a minor
   collection since, the driver gets the word back, after every other device's
   mapping of it. One drain takes each step. *)
let end_word d =
  match d.word_end with
  | Given -> ()
  | Read ->
      if Dev.stopped d && Dev.move_word d then begin
        let n = Dev.minors () in
        Dev.protect d (fun () -> d.word_end <- Moved n)
      end
  | Moved n ->
      let mine =
        Dev.minors () > n
        && Dev.protect d (fun () ->
            match d.word_end with
            | Moved m when m = n ->
                d.word_end <- Given;
                true
            | _ -> false)
      in
      if mine then begin
        let maps_of c =
          Dev.protect c (fun () ->
              let of_d, others =
                List.partition (fun (i, _) -> i = d.index) c.pair_maps
              in
              c.pair_maps <- others;
              of_d)
        in
        Dev.iter (fun c -> List.iter (fun (_, r) -> unmap c r) (maps_of c));
        unmap_all d;
        Option.iter (unmap d) d.word_region
      end

(* Drains the devices of [l] whose stop returned, other than [d], and ends
   their words: every drain drains those that hold memory or have a release
   waiting, and reads the word of those whose work may still run. It allocates
   nothing for an idle one, as every drain walks them all. *)
let rec drain_lost d = function
  | [] -> ()
  | e :: l ->
      if not (Dev.orphaned e) then begin
        if e.word_end != Given then end_word e;
        if e != d && not (idle e && e.cached = 0) then drain_own e
      end;
      drain_lost d l

(* An orphaned device's objects are forgotten: nothing of it drains. *)
let drain d =
  if not (Dev.orphaned d) then begin
    if not (idle d) then drain_own d;
    drain_lost d (Dev.ended ());
    if released_any holds_list then List.iter (route d) (released holds_list);
    if !holds != [] then drain_holds ()
  end

let () = Dev.answered := drain

(* The cache *)

let take_cached d kind n =
  Dev.protect d (fun () ->
      let k = key n kind in
      match Cache.find_opt d.cache k with
      | Some (e :: rest) ->
          (match rest with
          | [] -> Cache.remove d.cache k
          | _ -> Cache.replace d.cache k rest);
          d.cached <- d.cached - e.bytes;
          (* A new buffer waits for nothing of the memory's other devices: its
             cache reached their points, a lost device's included. *)
          stamps_keep e.stamps d.index;
          Some e
      | _ -> None)

(* Takes the cached memory of [d] that [take] picks out of its cache, in no
   particular order. *)
let take_cache_if d take =
  Dev.protect d (fun () ->
      let taken = ref [] in
      Cache.filter_map_inplace
        (fun _ l ->
          let l =
            List.filter
              (fun (e : entry) ->
                if take e then begin
                  taken := e :: !taken;
                  d.cached <- d.cached - e.bytes;
                  false
                end
                else true)
              l
          in
          match l with [] -> None | _ -> Some l)
        d.cache;
      !taken)

(* Takes cached memory out of [d]'s cache: all of it, or with [upto] only memory
   that counts in [d]'s budget, until [d] holds at most [upto] bytes. *)
let take_cache ?upto d =
  match upto with
  | None -> take_cache_if d (fun _ -> true)
  | Some upto ->
      let held = ref d.used in
      take_cache_if d (fun e ->
          owns d e.memory && !held > upto
          && begin
            held := !held - e.bytes;
            true
          end)

(* Frees what [take_cache] takes once the work [d] was handed until now is done:
   at once if it is, after waiting for it when [wait], and otherwise each free
   is deferred until it is. A lost device's cache waits for it to count as
   stopped. *)
let release_taken ~wait d take =
  if (not (Dev.is_lost d)) || Dev.stopped d then
    match take d with
    | [] -> ()
    | taken ->
        let v = Dev.submitted d in
        if wait && not (Dev.is_lost d) then Dev.wait d v;
        if Dev.is_lost d || Dev.word d >= v then List.iter free_entry taken
        else List.iter (fun e -> defer d (Free e)) taken

let release_cache ?upto ~wait d = release_taken ~wait d (take_cache ?upto)

(* The out-of-memory ladder *)

let rounds = 4

(* Whether [d]'s memory [e] counts in the budget of [pool], the host or a
   device. *)
let charged pool d (e : entry) =
  match budget_of d e.memory with
  | Device_budget -> d == pool
  | Host_budget -> Dev.is_host pool
  | No_budget -> false

(* Whether [d] holds memory that counts in [pool]'s budget back from its return
   until work is done: collected memory outside the cache, and returns deferred
   until [d]'s handed work is done. Only the entries' plain fields are read: a
   drain may free them meanwhile. *)
let holds_back pool d =
  List.exists (charged pool d) d.retiring
  || List.exists
       (function
         | _, Free e | _, Unmap (_, Some e) | _, Unload (_, Some e) ->
             charged pool e.owner e
         | _, (Unmap (_, None) | Unload (_, None)) -> false)
       d.pending

(* Drains every device, the host included, skipping one whose lock another call
   holds. *)
let drain_all () =
  drain Dev.host;
  Dev.iter (fun d -> if not (Dev.busy d) then drain d)

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
      if not (Dev.busy d) then begin
        if holds_back pool d && not (Dev.is_lost d) then
          Dev.wait d (Dev.submitted d);
        release_taken ~wait:true d (fun d -> take_cache_if d (charged pool d))
      end);
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

(* Allocation *)

let room d = Int.max 0 (d.budget - d.used)

(* A fresh memory record over the entry [e] of [d], with its token. *)
let of_entry d e =
  let address, handle, host =
    match e.region with Some r -> region_info r | None -> (-1, 0n, -1)
  in
  let live = if host >= 0 then d.used else -1 in
  let m = make ~host ~address ~handle d e.bytes e in
  m.token <- token d.release (Memory e) e.bytes (room d) live;
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

let ensure_entry m =
  let d = m.dev in
  Dev.protect d (fun () ->
      if m.entry == no_entry then begin
        let e = entry d Host_kept m.bytes (stamps_new ()) in
        e.kept <- m.keep;
        m.entry <- e;
        m.token <- token d.release (Memory e) 0 max_int (-1)
      end)

let find_map (e : entry) d = List.find_opt (fun mp -> mp.on == d) e.maps

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
  match m.entry.region with Some r -> map_peer_region d r | None -> None

(* [d]'s mapping of the memory [m] owns, whose host address is [host] or [-1],
   made at the first borrow and shared by the later ones. *)
let mapping d m host =
  let found = Dev.protect m.dev (fun () -> find_map m.entry d) in
  match found with
  | Some mp -> Some mp
  | None -> (
      (* Memory of a device of [d]'s driver maps as a peer's; other memory the
         host addresses, as host memory. *)
      let made =
        match m.dev.kind with
        | Driver _ when m.dev.key = d.key -> map_peer d m
        | _ when host >= 0 -> map_host_range d host m.bytes
        | _ -> map_peer d m
      in
      match made with
      | None -> None
      | Some r -> (
          let at, by, _ = region_info r in
          let mp = { on = d; map = r; at; by } in
          (* Noted on [d] before the memory has it: a word end of [d] that takes
             [d]'s notes between the two leaves the mapping in the memory's
             [maps], which [free_entry] releases. *)
          Dev.protect d (fun () -> Cache.replace d.mapped m.entry.id m.entry);
          let raced =
            Dev.protect m.dev (fun () ->
                match find_map m.entry d with
                | Some other -> Some other
                | None ->
                    m.entry.maps <- mp :: m.entry.maps;
                    None)
          in
          match raced with
          | None -> Some mp
          | Some other ->
              defer d (Unmap (r, None));
              Some other))

(* The host address of the io memory [m]'s pages, or [-1] if it maps none: the
   first answer of its device, kept. A call that raises leaves it to ask
   again. *)
let pages (m : memory) =
  let e = m.entry in
  (match e.pages with
  | Unasked -> (
      match e.io_region with
      | Some (Io_region { m = im; h; r }) ->
          let module I = (val im) in
          let got =
            match Dev.counted m.dev (fun () -> I.pages h r) with
            | Some ba -> Pages ba
            | None -> No_pages
          in
          Dev.protect m.dev (fun () ->
              match e.pages with
              | Unasked -> e.pages <- got
              | Pages _ | No_pages -> ())
      | None -> ())
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
  else if host >= 0 && m.entry.region = None && host mod page <> 0 then Cannot
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
      Dev.hold m.dev;
      let made = mapped_on d m.entry.maps in
      Dev.release m.dev;
      made || Option.is_some (mapping d m host)

let prefetch d (m : memory) ~at ~len =
  match m.root.entry.io_region with
  | Some (Io_region { m = im; h; r }) when not (Dev.is_host d) -> (
      let module I = (val im) in
      try I.prefetch h r ~at ~len with I.Fault _ | Sys_error _ -> ())
  | _ -> ()

let of_io d r ~access n =
  drain d;
  let e = entry ~io_region:r ~access d Io_given n (stamps_new ()) in
  let m = make d n e in
  m.token <- token d.release (Memory e) n max_int (-1);
  m

let set_budget d n =
  if n < 0 then invalid_argf "Rig.set_budget: budget %d is negative" n;
  Dev.protect d (fun () -> d.budget <- n);
  if Dev.is_host d then heap_trim n else release_cache ~upto:n ~wait:false d

let free_cache d =
  if Dev.is_host d then heap_drop () else release_cache ~wait:false d
