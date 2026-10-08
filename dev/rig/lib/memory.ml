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
external stamps_absorb : int -> int -> unit = "caml_rig_stamps_absorb"
external stamps_keep : int -> int -> unit = "caml_rig_stamps_keep" [@@noalloc]

(* [f] over the points of the stamps [st] from the [k]th on. *)
let rec iter_from f st k =
  let p = stamps_get st k in
  if p <> -1 then begin
    if p <> 0 then f p;
    iter_from f st (k + 1)
  end

let iter_points f st = if st <> 0 then iter_from f st 0

let iter_write f st =
  if st <> 0 then
    let p = stamps_get st 0 in
    if p > 0 then f p

let for_all_points f st =
  let ok = ref true in
  iter_points (fun p -> if not (f p) then ok := false) st;
  !ok

let check_points st =
  iter_points
    (fun p ->
      let d = Dev.of_index (Point.index p) in
      if Dev.is_lost d then Dev.raise_lost d)
    st

(* Whether [p] is reached, as a release judges it: a device lost by the read of
   its word has not reached it, and its memory waits for its stop. *)
let settled p =
  match Dev.point_reached p with r -> r | exception Dev.Lost _ -> false

(* Whether every point of [st] other than [except]'s is reached. A point of a
   device a forked child inherited names its parent's work, on its parent's copy
   of the memory: it holds back nothing in the child. *)
let reached ?(except = -1) st =
  for_all_points
    (fun p ->
      let i = Point.index p in
      i = except || Dev.inherited (Dev.of_index i) || settled p)
    st

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

let entry ?region ?io_region ?(access = Read_write) owner memory bytes stamps =
  {
    owner;
    memory;
    bytes;
    region;
    io_region;
    access;
    stamps;
    own = stamps;
    maps = [];
    unmaps = 0;
    held = false;
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
  let address = Option.value ~default:(-1) (D.address r) in
  let host = Option.value ~default:(-1) (D.host r) in
  (address, D.handle r, host)

(* Allocation events *)

let note d =
  if Prof.enabled () then
    Prof.record
      (Allocation { device = d; time = Prof.now (); allocated = d.used })

(* Driver calls *)

let poly = function
  | Pinned -> `Pinned
  | Mapped -> `Mapped
  | Device | Host_kept | Io_made | Io_given -> `Device

(* [n] new bytes of [d]'s memory of [kind] as a release record, or [None] if [d]
   has not the room. *)
let new_entry d kind n =
  match d.kind with
  | Driver { m; h; rid } -> (
      let module D = (val m) in
      match Dev.counted d (fun () -> D.alloc h (poly kind) n) with
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

(* After [d]'s loss, its memory and mappings are freed only once its stop
   returned, and uncounted: a failure of the lost device's free, its fault or
   its memory's, frees nothing more. *)
let call d f =
  if not (Dev.is_lost d) then Dev.counted d f
  else
    match f () with
    | () -> ()
    | exception Sys_error _ -> ()
    | exception e when Option.is_some (d.fault e) -> ()

(* Gives back a region [d] allocated or mapped. *)
let free_region d (Region { m; h; r; _ }) =
  let module D = (val m) in
  call d (fun () -> D.free h r)

let free_io d (Io_region { m; h; r }) =
  let module I = (val m) in
  call d (fun () -> I.free h r)

let unload d (Image { m; h; i }) =
  let module D = (val m) in
  if not (Dev.is_lost d) then Dev.counted d (fun () -> D.unload h i)

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

let drop_stamps (e : entry) =
  if e.held then stamps_unref e.stamps;
  if e.own <> 0 then stamps_unref e.own

(* Gives [e]'s region back to its driver, and its bytes to their keeper. *)
let give_back (e : entry) =
  let d = e.owner in
  (match e.region with Some r -> free_region d r | None -> ());
  (match e.io_region with Some r -> free_io d r | None -> ());
  drop_stamps e;
  let budget = budget_of d e.memory in
  give_room d budget e.bytes;
  if budget = Device_budget then note d

(* Releases the mapping [mp] now if its mapper ran all the work submitted so
   far, which is then all that could use it, and is whether it did. A device a
   forked child inherited maps its parent's copy of the memory, never the
   child's: the child is done with it without calling its driver. *)
let unmap_now mp =
  let d = mp.on in
  let v = Dev.submitted d in
  Dev.inherited d
  || (not (Dev.is_lost d))
     && Dev.word d >= v
     && begin
       free_region d mp.map;
       true
     end

(* [e] waits for its last unmap: until then a mapper's driver may still name the
   memory's pages, and a driver that maps host memory by its address would hand
   them to new memory at that address. *)
let free_entry (e : entry) =
  let later = List.filter (fun mp -> not (unmap_now mp)) e.maps in
  e.maps <- [];
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

(* Holds whose release is still to run: their stamps and release. *)
let holds_lock = Lock.create ()
let holds : (int * (unit -> unit)) list ref = ref []

(* A hold's release is due once each of its points is reached and each of its
   lost devices' stop returned. *)
let hold_due (st, _) =
  for_all_points
    (fun p ->
      let d = Dev.of_index (Point.index p) in
      settled p && ((not (Dev.is_lost d)) || Dev.stop_returned d))
    st

(* Runs [release] as a call in flight on each device of [st] that is not
   lost. *)
let run_release (st, release) =
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

let any_held = Atomic.make false
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

(* Memory the cache never takes: held memory, io memory, and host memory a
   device borrowed, which returns to its keeper once its uses are reached. *)
let uncached (e : entry) = e.held || e.memory = Host_kept || is_io_memory e

(* Whether [d] holds more than its budget in the memory that [e]'s counts in:
   [e] then returns to the driver, so a later allocation the budget refuses
   cannot reuse it. *)
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
  | Program (image, code) -> defer d (Unload (image, code))
  | Release { stamps; release } ->
      Lock.protect holds_lock (fun () -> holds := (stamps, release) :: !holds)

type fate = Stays | Cached | Freed

(* What becomes of the retiring [e] of [d]. It reads words, so no lock is held:
   a word behind a transport is read through its driver, whose fault stops its
   device, and a stop drains. A taken entry is the one drain's that took it. *)
let fate d ~lost ~free_lost (e : entry) =
  if viewed e then Stays
  else if lost then if free_lost && reached e.stamps then Freed else Stays
  else if uncached e then if reached e.stamps then Freed else Stays
  else if reached ~except:d.index e.stamps then
    if over_budget d e then Freed else Cached
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

let run_pending d = function
  | Free e -> free_entry e
  | Unmap (r, after) ->
      free_region d r;
      Option.iter unmapped after
  | Unload (image, code) -> (
      unload d image;
      match code with
      | Some e when Dev.is_lost d || over_budget d e -> free_entry e
      | Some e -> to_cache d e
      | None -> ())

(* Lost devices whose stop returned: every drain drains those that hold memory
   or have a release waiting, and reads the word of those whose work may still
   run. A device stays for good, since its memory and its mappings of host
   memory may be dropped long after its stop: their frees and unmaps still go to
   its driver. *)
let lost_devices : device list Atomic.t = Atomic.make []

let rec add_lost d =
  let l = Atomic.get lost_devices in
  if not (Atomic.compare_and_set lost_devices l (d :: l)) then add_lost d

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

(* Drains the lost devices of [l] other than [d]. It allocates nothing for an
   idle one, as every drain walks them all. *)
let rec drain_lost d = function
  | [] -> ()
  | e :: l ->
      if e != d && (not (Dev.inherited e)) && not (idle e && e.cached = 0) then
        drain_own e;
      drain_lost d l

let drain d =
  if not (Dev.inherited d) then begin
    if not (idle d) then drain_own d;
    drain_lost d (Atomic.get lost_devices);
    if released_any holds_list then List.iter (route d) (released holds_list);
    if !holds != [] then drain_holds ()
  end

let () =
  Dev.answered :=
    fun d ->
      add_lost d;
      drain d

(* Drains every other device, skipping one whose lock another call holds, and
   reads the word of each lost device that answered [Unknown]. *)
let drain_others d =
  Dev.iter (fun e -> if e != d && not (Dev.busy e) then drain e)

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

(* Takes cached memory out of [d]'s cache: all of it, or with [upto] only memory
   that counts in [d]'s budget, until [d] holds at most [upto] bytes, in no
   particular order. *)
let take_cache ?upto d =
  Dev.protect d (fun () ->
      let taken = ref [] and held = ref d.used in
      let take (e : entry) =
        match upto with
        | None -> true
        | Some upto -> owns d e.memory && !held > upto
      in
      Cache.filter_map_inplace
        (fun _ l ->
          let l =
            List.filter
              (fun (e : entry) ->
                if take e then begin
                  taken := e :: !taken;
                  if owns d e.memory then held := !held - e.bytes;
                  d.cached <- d.cached - e.bytes;
                  false
                end
                else true)
              l
          in
          match l with [] -> None | _ -> Some l)
        d.cache;
      !taken)

(* Frees what [take_cache] takes once the work [d] was handed until now is done:
   at once if it is, after waiting for it when [wait], and otherwise each free
   is deferred until it is. A lost device's cache waits for it to count as
   stopped. *)
let release_cache ?upto ~wait d =
  if (not (Dev.is_lost d)) || Dev.stopped d then begin
    let taken = take_cache ?upto d in
    let v = Dev.submitted d in
    if wait && not (Dev.is_lost d) then Dev.wait d v;
    if Dev.is_lost d || Dev.word d >= v then List.iter free_entry taken
    else List.iter (fun e -> defer d (Free e)) taken
  end

(* Allocation *)

let rounds = 4

let reclaim d round =
  release_cache ~wait:true d;
  drain_others d;
  if round >= 2 then Gc.full_major ();
  drain d

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

let rec alloc_entry d kind n round =
  match take_cached d kind n with
  | Some e -> e
  | None -> (
      match new_counted d kind n with
      | Some e ->
          note d;
          e
      (* Mapped memory the window or the budget cannot hold is pinned memory,
         which keeps its promises, and the cache stays. *)
      | None when kind = Mapped -> alloc_entry d Pinned n round
      | None when round < rounds ->
          reclaim d round;
          alloc_entry d kind n (round + 1)
      | None -> raise (Dev.Out_of_memory (d, n)))

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
  alloc_entry d kind n 1

let alloc d kind n = of_entry d (alloc_entry d kind n)

(* A round of the host's reclaim: its kept buffers, every other device's drain,
   from the second round a collection, then the host's own drain, which frees
   the host memory devices borrowed that the collection found unreachable and
   whose uses are reached: the next round's collection returns its bytes. *)
let reclaim_host round =
  heap_drop ();
  drain_others Dev.host;
  if round >= 2 then Gc.full_major ();
  drain Dev.host

let rec heap_reserved n round =
  if heap_reserve n Dev.host.budget then ()
  else if round < rounds then begin
    reclaim_host round;
    heap_reserved n (round + 1)
  end
  else raise (Dev.Out_of_memory (Dev.host, n))

(* [n] reserved bytes of the heap. A refusal of the C library runs the host's
   reclaim, and after its rounds gives the reservation back and raises the
   host's [Out_of_memory]. *)
let rec host_bytes n round =
  match heap_bytes n with
  | ba -> ba
  | exception Stdlib.Out_of_memory when round < rounds ->
      reclaim_host round;
      host_bytes n (round + 1)
  | exception Stdlib.Out_of_memory ->
      heap_release n;
      raise (Dev.Out_of_memory (Dev.host, n))

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
    heap_reserved n 1;
    let ba = host_bytes n 1 in
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
      let made =
        if host >= 0 then map_host_range d host m.bytes else map_peer d m
      in
      match made with
      | None -> None
      | Some r -> (
          let at, by, _ = region_info r in
          let mp = { on = d; map = r; at; by } in
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

let borrow d m =
  let m = m.root in
  if is_io_memory m.entry && m.bytes = 0 && m.dev != d then
    (* No bytes to map: the borrow is over no pages. *)
    let at = ba_address empty in
    Some (borrow_of m d ~host:at ~address:at ~handle:0n)
  else
    let host = if is_io_memory m.entry then pages m else m.host in
    if m.dev == d then Some m
    else if Dev.is_io d || (Dev.is_io m.dev && host < 0) then None
    else if (not (Dev.same_machine m.dev d)) && not (Dev.is_io m.dev) then None
    else if Dev.is_host d then
      if host >= 0 then Some (borrow_of m d ~host ~address:host ~handle:0n)
      else None
    else if d.memory_device && host >= 0 then
      Some (borrow_of m d ~host ~address:host ~handle:(Nativeint.of_int host))
    else if m.bytes = 0 then
      (* No bytes to map: the borrow names no memory. *)
      Some (borrow_of m d ~host ~address:0 ~handle:0n)
    else if host >= 0 && m.entry.region = None && host mod page <> 0 then None
    else begin
      if m.entry == no_entry then ensure_entry m;
      (* A borrow of host memory keeps its host address: it is host memory,
         which the host copies. *)
      match mapping d m host with
      | None -> None
      | Some mp -> Some (borrow_of m d ~host ~address:mp.at ~handle:mp.by)
    end

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
