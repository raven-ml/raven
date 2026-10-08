(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Def

let strf = Printf.sprintf

(* Stamps *)

external stamps_new : unit -> int = "caml_device_core_stamps_new"
external stamps_ref : int -> unit = "caml_device_core_stamps_ref" [@@noalloc]

external stamps_unref : int -> unit = "caml_device_core_stamps_unref"
[@@noalloc]

external stamps_reserve : int -> int -> unit = "caml_device_core_stamps_reserve"

external stamps_get : int -> int -> int = "caml_device_core_stamps_get"
[@@noalloc]

external stamps_absorb : int -> int -> unit = "caml_device_core_stamps_absorb"

(* [f] over the points of the stamps [st], the last write first; [write] only
   that one. *)
let iter_points ?(write = false) f st =
  if st <> 0 then begin
    let rec go k =
      let p = stamps_get st k in
      if p <> -1 then begin
        if p <> 0 then f p;
        if not (write && k = 0) then go (k + 1)
      end
    in
    go 0
  end

let for_all_points ?write f st =
  let ok = ref true in
  iter_points ?write (fun p -> if not (f p) then ok := false) st;
  !ok

(* Raises [Lost] if a point of [st] is on a lost device. *)
let check_points st =
  iter_points
    (fun p ->
      let d = Dev.of_index (Point.index p) in
      if Dev.is_lost d then Dev.raise_lost d)
    st

(* Whether every point of [st] other than [except]'s is reached. *)
let reached ?(except = -1) st =
  for_all_points (fun p -> Point.index p = except || Dev.point_reached p) st

(* Tokens and release lists *)

external token : int -> released -> int -> int -> int -> token
  = "caml_device_core_token"

external no_token : unit -> token = "%identity"
external released : int -> released list = "caml_device_core_released"

external released_any : int -> bool = "caml_device_core_released_any"
[@@noalloc]

external page_size : unit -> int = "caml_device_core_page_size"

let no_token = no_token ()
let page = page_size ()

(* The host's heap *)

external heap_init : unit -> unit = "caml_device_core_heap_init"
external heap_reserve : int -> int -> bool = "caml_device_core_heap_reserve"
external heap_token : int -> token = "caml_device_core_heap_token"

type bytes_ba =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external heap_alloc : int -> bytes_ba = "caml_device_core_heap_alloc"

external heap_aligned : int -> int -> bytes_ba option
  = "caml_device_core_heap_aligned"

external heap_drop : unit -> unit = "caml_device_core_heap_drop"

external ba_address : ('a, 'b, 'c) Bigarray.Array1.t -> int
  = "caml_device_core_bigarray_address"

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

(* Kinds: [Buffer.memory]'s, then the host's heap, then memory its maker
   keeps. *)
let device_kind = 0
let pinned_kind = 1
let mapped_kind = 2
let heap_kind = 3
let kept_kind = 4

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
  }

(* The entry of host memory no device borrowed: no stamps, no mapping. *)
let no_entry = entry Dev.host heap_kind 0 0

let make ?(keep = Nothing) ?(host = -1) ?(address = -1) ?(handle = 0n)
    ?(token = no_token) dev bytes entry =
  let rec m =
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
      root = m;
    }
  in
  m

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
  let host = match D.host r with Some p -> Nativeint.to_int p | None -> -1 in
  (address, D.handle r, host)

(* Allocation events *)

let note d =
  if Prof.enabled () then
    Prof.record
      (Allocation { device = d; time = Prof.now (); allocated = d.used })

(* Driver calls *)

let poly kind =
  if kind = pinned_kind then `Pinned
  else if kind = mapped_kind then `Mapped
  else `Device

let driver_alloc d kind n =
  match d.kind with
  | Driver { m; h } -> (
      let module D = (val m) in
      match Dev.counted d (fun () -> D.alloc h (poly kind) n) with
      | None -> None
      | Some r -> Some (Region { m; h; r }))
  | _ -> None

(* After [d]'s loss, its memory is freed, unmapped and unloaded only once its
   stop answered, and uncounted. *)
let call d f =
  if Dev.is_lost d then try f () with _ -> () else Dev.counted d f

let free_region d (Region { m; h; r }) =
  let module D = (val m) in
  call d (fun () -> D.free h r)

let unmap_region d (Region { m; h; r }) =
  let module D = (val m) in
  call d (fun () -> D.unmap h r)

let unload d (Image { m; h; i }) =
  let module D = (val m) in
  if not (Dev.is_lost d) then Dev.counted d (fun () -> D.unload h i)

let owns kind = kind = device_kind || kind = mapped_kind

(* Releases *)

(* Puts [p] on [d]'s pending list, due once [d] reached the value it has
   submitted now: the work that may use it without naming it. *)
let defer d p =
  let v = Dev.submitted d in
  Mutex.protect d.lock (fun () -> d.pending <- (v, p) :: d.pending)

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
  drop_stamps e;
  if owns e.memory then begin
    Mutex.protect d.lock (fun () -> d.used <- d.used - e.bytes);
    note d
  end

let cache_key (e : entry) = (e.bytes * 4) + e.memory

(* Whether [d]'s loss is answered: its memory may be freed. *)
let answered d =
  Dev.is_lost d && (Dev.answer d = Dev.answer_stopped || Dev.upgrade d)

(* Holds whose release is still to run: their stamps and release. *)
let holds_lock = Mutex.create ()
let holds : (int * (unit -> unit)) list ref = ref []

(* A hold's release is due once each of its points is reached and each of its
   lost devices answered. *)
let hold_due (st, _) =
  for_all_points
    (fun p ->
      let d = Dev.of_index (Point.index p) in
      Dev.point_reached p
      && ((not (Dev.is_lost d))
         || (Dev.answer d <> 0 && Dev.answer d <> Dev.answer_stopping)))
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
      Mutex.protect holds_lock (fun () ->
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

(* Routes a record [d]'s release list gave. *)
let route d = function
  | Memory e when e.memory = heap_kind || e.memory = kept_kind ->
      unmap_all e;
      drop_stamps e
  | Memory e ->
      let cache =
        (not (Dev.is_lost d))
        && (not e.held)
        && reached ~except:d.index e.stamps
      in
      Mutex.protect d.lock (fun () ->
          if cache then begin
            let k = cache_key e in
            let l = Option.value ~default:[] (Hashtbl.find_opt d.cache k) in
            Hashtbl.replace d.cache k (e :: l);
            d.cached <- d.cached + e.bytes
          end
          else d.retiring <- e :: d.retiring)
  | Program (image, bytes) -> defer d (Unload (image, bytes))
  | Release { stamps; release } ->
      Mutex.protect holds_lock (fun () -> holds := (stamps, release) :: !holds)

(* Takes what became due: retiring memory whose foreign uses are reached enters
   the cache, or is freed if [d] is lost and answered; pending releases whose
   value [d] reached. *)
let due d =
  let lost = Dev.is_lost d in
  let free_lost = lost && answered d in
  let w = if lost && not free_lost then -1 else Dev.word d in
  Mutex.protect d.lock (fun () ->
      let frees = ref [] in
      d.retiring <-
        List.filter
          (fun e ->
            if lost then
              if free_lost && reached e.stamps then (
                frees := e :: !frees;
                false)
              else true
            else if (not e.held) && reached ~except:d.index e.stamps then begin
              let k = cache_key e in
              let l = Option.value ~default:[] (Hashtbl.find_opt d.cache k) in
              Hashtbl.replace d.cache k (e :: l);
              d.cached <- d.cached + e.bytes;
              false
            end
            else if e.held && reached e.stamps then (
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
  | Unmap r -> unmap_region d r
  | Unload (image, bytes) ->
      unload d image;
      Mutex.protect d.lock (fun () -> d.used <- d.used - bytes);
      note d

(* Lost devices whose answer was recorded and that still hold memory: every
   drain drains them, and reads the word of those that answered [Unknown]. *)
let lost_devices : device list Atomic.t = Atomic.make []

let rec change_lost f =
  let l = Atomic.get lost_devices in
  if not (Atomic.compare_and_set lost_devices l (f l)) then change_lost f

let empty d =
  Mutex.protect d.lock (fun () ->
      d.retiring = [] && d.pending = [] && Hashtbl.length d.cache = 0)

(* Drains [d]'s own list and what became due on it. A forked child calls no
   driver. *)
let drain_own d =
  if released_any d.release then List.iter (route d) (released d.release);
  let frees, pending = due d in
  List.iter free_entry frees;
  List.iter (run_pending d) pending

(* Drains [d], then the lost devices that hold memory, then the holds. *)
let drain d =
  if not (Dev.forked ()) then begin
    drain_own d;
    List.iter
      (fun e ->
        if e != d then begin
          drain_own e;
          if Dev.answer e = Dev.answer_stopped && empty e then
            change_lost (List.filter (fun e' -> e' != e))
        end)
      (Atomic.get lost_devices);
    if released_any holds_list then List.iter (route d) (released holds_list);
    drain_holds ()
  end

let () =
  Dev.answered :=
    fun d ->
      change_lost (fun l -> d :: l);
      drain d

(* Drains every other device, skipping one whose lock another call holds, and
   reads the word of each lost device that answered [Unknown]. *)
let drain_others d =
  Array.iteri
    (fun i e ->
      if i = e.index && e != d && not (Dev.is_host e) then
        if Mutex.try_lock e.lock then begin
          Mutex.unlock e.lock;
          drain e
        end)
    (Dev.all ())

(* The cache *)

let take_cached d kind n =
  Mutex.protect d.lock (fun () ->
      let k = (n * 4) + kind in
      match Hashtbl.find_opt d.cache k with
      | Some (e :: rest) ->
          if rest = [] then Hashtbl.remove d.cache k
          else Hashtbl.replace d.cache k rest;
          d.cached <- d.cached - e.bytes;
          Some e
      | _ -> None)

(* Takes cached memory out of [d]'s cache until [d] holds at most [upto] bytes
   or the cache is empty, in no particular order. *)
let take_cache ~upto d =
  Mutex.protect d.lock (fun () ->
      let taken = ref [] and held = ref d.used in
      Hashtbl.filter_map_inplace
        (fun _ l ->
          let l =
            List.filter
              (fun (e : entry) ->
                if !held > upto then begin
                  taken := e :: !taken;
                  held := !held - e.bytes;
                  d.cached <- d.cached - e.bytes;
                  false
                end
                else true)
              l
          in
          if l = [] then None else Some l)
        d.cache;
      !taken)

(* Frees [d]'s cache down to [upto] bytes once the work [d] was handed until now
   is done: waiting for it when [wait], deferring each free otherwise. *)
let release_cache ?(upto = 0) ~wait d =
  let taken = take_cache ~upto d in
  if wait then begin
    if not (Dev.is_lost d) then Dev.wait d (Dev.submitted d);
    List.iter free_entry taken
  end
  else List.iter (fun e -> defer d (Free e)) taken

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
        (not (owns kind))
        || Mutex.protect d.lock (fun () -> d.used + n <= d.budget)
      in
      let r = if fits then driver_alloc d kind n else None in
      match r with
      | Some r ->
          if owns kind then
            Mutex.protect d.lock (fun () -> d.used <- d.used + n);
          note d;
          entry ~region:r d kind n (stamps_new ())
      | None when round < rounds ->
          reclaim d round;
          alloc_entry d kind n (round + 1)
      | None when kind = mapped_kind -> alloc_entry d pinned_kind n 1
      | None -> raise (Dev.Out_of_memory (d, n)))

let alloc d kind n =
  let kind = if kind = mapped_kind && n > d.budget then pinned_kind else kind in
  if owns kind && n > d.budget then raise (Dev.Out_of_memory (d, n));
  drain d;
  of_entry d (alloc_entry d kind n 1)

let rec heap_reserved n round =
  if heap_reserve n Dev.host.budget then ()
  else if round < rounds then begin
    heap_drop ();
    drain_others Dev.host;
    if round >= 2 then Gc.full_major ();
    heap_reserved n (round + 1)
  end
  else raise (Dev.Out_of_memory (Dev.host, n))

let host_memory n =
  drain Dev.host;
  heap_reserved n 1;
  let ba = heap_bytes n in
  let addr = ba_address ba in
  make
    ~keep:(Heap (ba, heap_token n))
    ~host:addr ~address:addr Dev.host n no_entry

(* Borrows *)

(* Gives host memory that a device maps its stamps and a token, which unmaps its
   mappings once it is collected. *)
let ensure_entry m =
  let d = m.dev in
  Mutex.protect d.lock (fun () ->
      if m.entry == no_entry then begin
        let e = entry d heap_kind m.bytes (stamps_new ()) in
        m.entry <- e;
        m.token <- token d.release (Memory e) 0 max_int (-1)
      end)

let find_map (e : entry) d = List.find_opt (fun mp -> mp.on == d) e.maps

let map_host_range d start n =
  match d.kind with
  | Driver { m = dm; h } -> (
      let module D = (val dm) in
      let p = Nativeint.of_int start in
      match Dev.counted d (fun () -> D.map_host h p n) with
      | Some r -> Some (Region { m = dm; h; r })
      | None -> None)
  | _ -> None

let map_host d m = map_host_range d m.host m.bytes

(* [d]'s mapping of the region of another device of [d]'s driver. The region's
   module types it; the keys' equality types [d]'s handle. *)
let map_peer_region d (Region { m = om; h = oh; r }) =
  match d.kind with
  | Driver { m = dm; h } -> (
      let module D = (val dm) in
      let module O = (val om) in
      match Type.Id.provably_equal D.key O.key with
      | Some Type.Equal -> (
          match Dev.counted d (fun () -> O.map_peer h oh r) with
          | Some r -> Some (Region { m = om; h; r })
          | None -> None)
      | None -> None)
  | _ -> None

let map_peer d m =
  match m.entry.region with Some r -> map_peer_region d r | None -> None

(* [d]'s mapping of the memory [m] owns, made at the first borrow and shared by
   the later ones. *)
let mapping d m =
  let found = Mutex.protect m.dev.lock (fun () -> find_map m.entry d) in
  match found with
  | Some mp -> Some mp
  | None -> (
      let made = if m.host >= 0 then map_host d m else map_peer d m in
      match made with
      | None -> None
      | Some r -> (
          let at, by, _ = region_info r in
          let mp = { on = d; map = r; at; by } in
          let raced =
            Mutex.protect m.dev.lock (fun () ->
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

let borrow d m =
  let m = m.root in
  if m.dev == d then Some m
  else if m.dev.machine <> d.machine || Dev.is_io d || Dev.is_io m.dev then None
  else if Dev.is_host d then
    if m.host >= 0 then
      Some (borrow_of m d ~host:m.host ~address:m.host ~handle:0n)
    else None
  else if d.memory_device && m.host >= 0 then
    Some
      (borrow_of m d ~host:m.host ~address:m.host
         ~handle:(Nativeint.of_int m.host))
  else if m.host >= 0 && m.entry.region = None && m.host mod page <> 0 then None
  else begin
    if m.entry == no_entry then ensure_entry m;
    match mapping d m with
    | None -> None
    | Some mp -> Some (borrow_of m d ~host:(-1) ~address:mp.at ~handle:mp.by)
  end

(* Releases the device's cache down to its budget. *)
let trim d = release_cache ~upto:d.budget ~wait:false d

let set_budget d n =
  if n < 0 then
    invalid_arg (strf "Device_core.set_budget: budget %d is negative" n);
  Mutex.protect d.lock (fun () -> d.budget <- n);
  trim d

let free_cache d = release_cache ~wait:false d
