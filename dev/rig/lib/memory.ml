(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Def

let strf = Printf.sprintf

(* Stamps *)

external stamps_new : unit -> int = "caml_rig_stamps_new"
external stamps_ref : int -> unit = "caml_rig_stamps_ref" [@@noalloc]
external stamps_unref : int -> unit = "caml_rig_stamps_unref" [@@noalloc]
external stamps_get : int -> int -> int = "caml_rig_stamps_get" [@@noalloc]
external stamps_absorb : int -> int -> unit = "caml_rig_stamps_absorb"

(* [f] over the points of the stamps [st], the last write first; [write] only
   that one. *)
let rec iter_from f st k =
  let p = stamps_get st k in
  if p <> -1 then begin
    if p <> 0 then f p;
    iter_from f st (k + 1)
  end

(* [f] over the points of the stamps [st], the last write first. Neither it nor
   [iter_write] allocates. *)
let iter_points f st = if st <> 0 then iter_from f st 0

(* [f] of the last write of [st], if any. *)
let iter_write f st =
  if st <> 0 then
    let p = stamps_get st 0 in
    if p > 0 then f p

let for_all_points f st =
  let ok = ref true in
  iter_points (fun p -> if not (f p) then ok := false) st;
  !ok

(* Raises [Lost] if a point of [st] is on a lost device. *)
let check_points st =
  iter_points
    (fun p ->
      let d = Dev.of_index (Point.index p) in
      if Dev.is_lost d then Dev.raise_lost d)
    st

(* Whether every point of [st] other than [except]'s is reached. A point of a
   device a forked child inherited names its parent's work, on its parent's copy
   of the memory: it holds back nothing in the child. *)
let reached ?(except = -1) st =
  for_all_points
    (fun p ->
      let i = Point.index p in
      i = except || Dev.inherited (Dev.of_index i) || Dev.point_reached p)
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

external heap_init : unit -> unit = "caml_rig_heap_init"
external heap_reserve : int -> int -> bool = "caml_rig_heap_reserve"
external heap_token : int -> token = "caml_rig_heap_token"

type bytes_ba =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external heap_alloc : int -> bytes_ba = "caml_rig_heap_alloc"
external heap_aligned : int -> int -> bytes_ba option = "caml_rig_heap_aligned"
external heap_drop : unit -> unit = "caml_rig_heap_drop"

external ba_address : ('a, 'b, 'c) Bigarray.Array1.t -> int
  = "caml_rig_bigarray_address"

let () = heap_init ()

(* Host buffers of at least this many bytes start on a page, so devices can map
   them: a mapping takes whole pages, which another buffer's memory must not
   share. Aligning costs up to a page of slack, at most a quarter of the buffer.
   Smaller buffers are the runtime's bigarrays, which the minor heap holds and
   where most die: [caml_alloc_custom] would pace minor collections by the bound
   it paces major cycles by. *)
let aligned_from = Int.max (64 * 1024) (4 * page)

let heap_bytes n =
  if n < aligned_from then
    Bigarray.Array1.create Bigarray.char Bigarray.c_layout n
  else
    match heap_aligned page n with
    | Some ba -> ba
    | None ->
        if n > max_int - page then raise Stdlib.Out_of_memory;
        let ba = heap_alloc (n + page - 1) in
        let skip = (page - (ba_address ba mod page)) mod page in
        Bigarray.Array1.sub ba skip n

(* Memory records *)

let new_claim () = { count = 0; generation = 0; why = "" }

let is_io_memory (e : entry) =
  match e.memory with
  | Io_made | Io_given -> true
  | Device | Pinned | Mapped | Host_kept -> false

let entry ?region ?io_region owner memory bytes stamps =
  {
    owner;
    memory;
    bytes;
    region;
    io_region;
    stamps;
    own = stamps;
    maps = [];
    held = false;
    pages = Unasked;
    proxy = 0;
  }

(* The entry of host memory no device borrowed: no stamps, no mapping. *)
let no_entry = entry Dev.host Host_kept 0 0

(* The root a record holds until it is set to the record itself: a record made
   recursively would be made twice. *)
let rec unrooted =
  {
    dev = Dev.host;
    bytes = 0;
    host = -1;
    address = -1;
    handle = 0n;
    claim = new_claim ();
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
      claim = new_claim ();
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
   returned, and uncounted. *)
let call d f =
  if Dev.is_lost d then try f () with _ -> () else Dev.counted d f

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

(* Whether memory of [kind] counts in its device's budget. *)
let owns = function
  | Device | Mapped | Io_made -> true
  | Pinned | Host_kept | Io_given -> false

(* Releases *)

(* Puts [p] on [d]'s pending list, due once [d] reached the value it has
   submitted now: the work that may use it without naming it. *)
let defer d p =
  let v = Dev.submitted d in
  Dev.protect d (fun () -> d.pending <- (v, p) :: d.pending)

(* Unmaps the other devices' mappings of [e]'s memory, each once its mapper's
   work submitted until now is done. *)
let unmap_all (e : entry) =
  List.iter (fun mp -> defer mp.on (Unmap mp.map)) e.maps;
  e.maps <- []

let drop_stamps (e : entry) =
  if e.held then stamps_unref e.stamps;
  if e.own <> 0 then stamps_unref e.own

(* Gives [e]'s region back to its driver. *)
let free_entry (e : entry) =
  let d = e.owner in
  unmap_all e;
  Option.iter (free_region d) e.region;
  Option.iter (free_io d) e.io_region;
  drop_stamps e;
  if owns e.memory then begin
    Dev.protect d (fun () -> d.used <- d.used - e.bytes);
    note d
  end

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
      Dev.point_reached p && ((not (Dev.is_lost d)) || Dev.stop_returned d))
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

let drain_holds () =
  if !holds <> [] then begin
    let due =
      Lock.protect holds_lock (fun () ->
          let due, later = List.partition hold_due !holds in
          holds := later;
          due)
    in
    let first = ref None in
    List.iter
      (fun h ->
        try run_release h with e -> if !first = None then first := Some e)
      due;
    Option.iter raise !first
  end

(* The release list of holds, which every drain reads. *)
let holds_list = Dev.release_list ()

(* Bigarrays over memory *)

external proxy_new : unit -> int = "caml_rig_proxy_new"
external proxy_drop : int -> bool = "caml_rig_proxy_drop" [@@noalloc]

(* The proxy that bigarrays over [e]'s memory share, made at the first. *)
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
  let l = Option.value ~default:[] (Hashtbl.find_opt d.cache k) in
  Hashtbl.replace d.cache k (e :: l);
  d.cached <- d.cached + e.bytes

let to_cache d e = Dev.protect d (fun () -> cache d e)

(* Routes a record [d]'s release list gave. *)
let route d = function
  | Memory ({ memory = Host_kept; _ } as e) ->
      unmap_all e;
      drop_stamps e
  | Memory e ->
      let cached =
        (not (Dev.is_lost d))
        && (not e.held)
        && (not (is_io_memory e))
        && reached ~except:d.index e.stamps
        && not (viewed e)
      in
      Dev.protect d (fun () ->
          if cached then cache d e else d.retiring <- e :: d.retiring)
  | Program (image, code) -> defer d (Unload (image, code))
  | Release { stamps; release } ->
      Lock.protect holds_lock (fun () -> holds := (stamps, release) :: !holds)

(* Takes what became due: retiring memory no bigarray reads and whose foreign
   uses are reached enters the cache, or is freed if [d] is lost and counts as
   stopped; pending releases whose value [d] reached. *)
let due d =
  let lost = Dev.is_lost d in
  let free_lost = lost && Dev.stopped d in
  let w = if lost && not free_lost then -1 else Dev.word d in
  Dev.protect d (fun () ->
      let frees = ref [] in
      d.retiring <-
        List.filter
          (fun e ->
            if viewed e then true
            else if lost then
              if free_lost && reached e.stamps then (
                frees := e :: !frees;
                false)
              else true
            else if
              (not e.held)
              && (not (is_io_memory e))
              && reached ~except:d.index e.stamps
            then begin
              cache d e;
              false
            end
            else if (e.held || is_io_memory e) && reached e.stamps then (
              frees := e :: !frees;
              false)
            else true)
          d.retiring;
      if free_lost then begin
        Hashtbl.iter (fun _ l -> frees := l @ !frees) d.cache;
        Hashtbl.reset d.cache;
        d.cached <- 0
      end;
      let pending, later =
        List.partition (fun (v, _) -> w >= 0 && v <= w) d.pending
      in
      d.pending <- later;
      (!frees, List.map snd pending))

let run_pending d = function
  | Free e -> free_entry e
  | Unmap r -> free_region d r
  | Unload (image, code) -> (
      unload d image;
      match code with
      | Some e when Dev.is_lost d -> free_entry e
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

(* Drains [d]'s own list and what became due on it. *)
let drain_own d =
  if released_any d.release then List.iter (route d) (released d.release);
  let frees, pending = due d in
  List.iter free_entry frees;
  List.iter (run_pending d) pending

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

(* Drains [d], then the lost devices that hold memory, then the holds. A device
   a forked child inherited is never drained: its frees would call its
   driver. *)
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
  Array.iteri
    (fun i e ->
      if i = e.index && e != d && not (Dev.is_host e) then
        if not (Dev.busy e) then drain e)
    (Dev.all ())

(* The cache *)

let take_cached d kind n =
  Dev.protect d (fun () ->
      let k = key n kind in
      match Hashtbl.find_opt d.cache k with
      | Some (e :: rest) ->
          if rest = [] then Hashtbl.remove d.cache k
          else Hashtbl.replace d.cache k rest;
          d.cached <- d.cached - e.bytes;
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
        | Some upto -> owns e.memory && !held > upto
      in
      Hashtbl.filter_map_inplace
        (fun _ l ->
          let l =
            List.filter
              (fun (e : entry) ->
                if take e then begin
                  taken := e :: !taken;
                  if owns e.memory then held := !held - e.bytes;
                  d.cached <- d.cached - e.bytes;
                  false
                end
                else true)
              l
          in
          if l = [] then None else Some l)
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

(* The rounds an allocation that its budget or driver refuses runs before it
   raises: release the cache, drain every other device, collect. *)
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

let rec alloc_entry d kind n round =
  match take_cached d kind n with
  | Some e -> e
  | None -> (
      let fits =
        (not (owns kind)) || Dev.protect d (fun () -> d.used + n <= d.budget)
      in
      match if fits then new_entry d kind n else None with
      | Some e ->
          if owns e.memory then Dev.protect d (fun () -> d.used <- d.used + n);
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
  if owns kind && n > d.budget then raise (Dev.Out_of_memory (d, n));
  drain d;
  alloc_entry d kind n 1

let alloc d kind n = of_entry d (alloc_entry d kind n)

let rec heap_reserved n round =
  if heap_reserve n Dev.host.budget then ()
  else if round < rounds then begin
    heap_drop ();
    drain_others Dev.host;
    if round >= 2 then Gc.full_major ();
    heap_reserved n (round + 1)
  end
  else raise (Dev.Out_of_memory (Dev.host, n))

(* What every host buffer of no bytes keeps: nothing to free. *)
let empty = Bigarray.Array1.create Bigarray.char Bigarray.c_layout 0
let empty_keep = Heap (empty, no_token)

let host_memory n =
  drain Dev.host;
  if n = 0 then
    let addr = ba_address empty in
    own ~keep:empty_keep ~host:addr ~address:addr ~handle:0n ~token:no_token
      Dev.host 0 no_entry
  else begin
    heap_reserved n 1;
    let ba = heap_bytes n in
    let addr = ba_address ba in
    own
      ~keep:(Heap (ba, heap_token n))
      ~host:addr ~address:addr ~handle:0n ~token:no_token Dev.host n no_entry
  end

(* Borrows *)

(* Gives host memory that a device maps its stamps and a token, which unmaps its
   mappings once it is collected. *)
let ensure_entry m =
  let d = m.dev in
  Dev.protect d (fun () ->
      if m.entry == no_entry then begin
        let e = entry d Host_kept m.bytes (stamps_new ()) in
        m.entry <- e;
        m.token <- token d.release (Memory e) 0 max_int (-1)
      end)

let find_map (e : entry) d = List.find_opt (fun mp -> mp.on == d) e.maps

let map_host_range d start n =
  match d.kind with
  | Driver { m = dm; h; rid } -> (
      let module D = (val dm) in
      match Dev.counted d (fun () -> D.map_host h start n) with
      | Some r -> Some (Region { m = dm; h; r; rid })
      | None -> None)
  | _ -> None

(* [d]'s mapping of the region of another device of [d]'s driver. The region's
   module types it; the keys' equality types [d]'s handle. *)
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
              defer d (Unmap r);
              Some other))

(* The host address of the io memory [m]'s pages, asked of its device at the
   first borrow, or [-1] if it maps none. *)
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
          Dev.protect m.dev (fun () -> if e.pages = Unasked then e.pages <- got)
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
    else if m.dev.machine <> d.machine && not (Dev.is_io m.dev) then None
    else if Dev.is_host d then
      if host >= 0 then Some (borrow_of m d ~host ~address:host ~handle:0n)
      else None
    else if d.memory_device && host >= 0 then
      Some (borrow_of m d ~host ~address:host ~handle:(Nativeint.of_int host))
    else if host >= 0 && m.entry.region = None && host mod page <> 0 then None
    else begin
      if m.entry == no_entry then ensure_entry m;
      match mapping d m host with
      | None -> None
      | Some mp -> Some (borrow_of m d ~host:(-1) ~address:mp.at ~handle:mp.by)
    end

(* Asks the io device of [m] to read the bytes of [m] from [at] ahead, for a
   device other than the host that borrowed them. A hint: it raises nothing. *)
let prefetch d (m : memory) ~at ~len =
  match m.root.entry.io_region with
  | Some (Io_region { m = im; h; r }) when not (Dev.is_host d) -> (
      let module I = (val im) in
      try I.prefetch h r ~at ~len with _ -> ())
  | _ -> ()

(* A memory record over [n] bytes of the region [r] an io library gave the io
   device [d], which [d]'s free gives back once unreachable. *)
let of_io d r n =
  drain d;
  let e = entry ~io_region:r d Io_given n (stamps_new ()) in
  let m = make d n e in
  m.token <- token d.release (Memory e) n max_int (-1);
  m

(* Releases the device's cache down to its budget. *)
let trim d = release_cache ~upto:d.budget ~wait:false d

let set_budget d n =
  if n < 0 then invalid_arg (strf "Rig.set_budget: budget %d is negative" n);
  Dev.protect d (fun () -> d.budget <- n);
  trim d

let free_cache d = release_cache ~wait:false d
