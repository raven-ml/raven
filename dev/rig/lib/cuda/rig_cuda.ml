(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Any domain may call any function, as the interface states. The loader, the
   table of GPUs and the page-lock registry are the process's, each behind a
   mutex of its own; a region or an image ends once, by compare-and-set; a
   device's C state is written only by [submit] and [stop], which their caller
   serialises. *)

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let ( let* ) = Result.bind

exception Fault of string

(* The CUDA library. A stub with a numeric result returns it as a non-negative
   int, or CUDA's status negated. *)

external load : unit -> int * string = "caml_rig_cuda_load"
external error : int -> string = "caml_rig_cuda_error"
external symbol : string -> int = "caml_rig_cuda_symbol"
external page_size : unit -> int = "caml_rig_cuda_page_size"
external driver_version : unit -> int = "caml_rig_cuda_driver_version"
external device_count : unit -> int = "caml_rig_cuda_count"
external device : int -> int = "caml_rig_cuda_device"
external attribute : int -> int -> int = "caml_rig_cuda_attribute"
external total_memory : int -> int = "caml_rig_cuda_total_memory"

(* From cuda.h *)

let cuda_error_peer_access_already_enabled = 704
let attribute_unified_addressing = 41
let attribute_compute_capability_major = 75
let attribute_compute_capability_minor = 76
let attribute_pci_bus_id = 33
let attribute_pci_device_id = 34
let attribute_pci_domain_id = 50
let attribute_can_use_64_bit_stream_mem_ops = 122

(* A CUDA failure with [status], in the step form "[step]: NAME: text". *)
let fault step status = raise (Fault (strf "%s: %s" step (error status)))
let get step r = if r < 0 then fault step (-r) else r

external failed : int -> int = "caml_rig_cuda_failed"

(* [refused step self x] is [x], the answer to CUDA's refusal of a call's
   arguments, unless the context of the device [self] failed: CUDA answers its
   error to every call, and [refused] raises it. *)
let refused step self x = match failed self with 0 -> x | e -> fault step e

(* A GPU's holder: [unheld]; [taken], while a device of it is open or being
   opened; or the C state, negated, of a device stopped while its work still
   ran, whose modules [left] holds until that work ends. *)
type gpus = {
  devices : int array;
  held : int Atomic.t array;
  left : int list Atomic.t array;
}

let unheld = 0
let taken = 1
let lock = Mutex.create ()
let gpus = ref None

let discover () =
  let* () =
    match load () with
    | 0, _ -> Ok ()
    | -1, name ->
        Error
          (strf
             "no CUDA library (%s) on the library search path; install the \
              NVIDIA driver"
             name)
    | -2, name ->
        Error
          (strf "the CUDA library lacks %s; install a newer NVIDIA driver" name)
    | s, _ -> Error ("initialising CUDA: " ^ error s)
  in
  (* GPU [i] is the [i]th by PCI address, whatever order CUDA_DEVICE_ORDER gives
     CUDA's ordinals. *)
  let n = max 0 (device_count ()) in
  let address d =
    List.map
      (fun a -> get "finding CUDA's GPUs" (attribute d a))
      [ attribute_pci_domain_id; attribute_pci_bus_id; attribute_pci_device_id ]
  in
  match
    List.init n (fun k ->
        let d = get "finding CUDA's GPUs" (device k) in
        (address d, d))
  with
  | exception Fault why -> Error why
  | ds ->
      let devices = Array.of_list (List.map snd (List.sort compare ds)) in
      let each x = Array.map (fun _ -> Atomic.make x) devices in
      Ok { devices; held = each unheld; left = each [] }

(* Loads the library and finds its GPUs at the first call that needs them, until
   they are found: a failed load is tried again by the next call, so a driver
   installed meanwhile is found. *)
let find_gpus () =
  Mutex.protect lock @@ fun () ->
  match !gpus with
  | Some g -> Ok g
  | None ->
      let r = discover () in
      Result.iter (fun g -> gpus := Some g) r;
      r

(* Memory. Page-locking is the process's: a registration serves every device,
   and ends at the last free of a region over it, from any device. *)

type registration = {
  start : int;
  bytes : int;
  address : int; (* where CUDA's work reaches [start] *)
  mutable maps : int;
  mutable stuck : bool; (* CUDA refused its unregistration *)
}

type kind =
  | Device
  | Host
  | Word
  | Locked of registration option (* [None]: another owner's *)
  | Peer of kind

type region = {
  owner : int; (* the device's C state *)
  kind : kind;
  address : int;
  handle : int;
  home : int; (* the device whose GPU holds GPU memory *)
  live : bool Atomic.t; (* taken once by free *)
}

let region owner kind ~address ~handle =
  { owner; kind; address; handle; home = owner; live = Atomic.make true }

let rec on_host = function
  | Device -> false
  | Host | Word | Locked _ -> true
  | Peer k -> on_host k

(* Opening *)

type image = { owner : int; m : int; loaded : bool Atomic.t }

type t = {
  self : int;
  arch : string;
  budget : int;
  word : region;
  held : int Atomic.t;
  left : int list Atomic.t;
  images : image list Atomic.t; (* loaded, for stop to unload *)
}

external open_device : int -> int = "caml_rig_cuda_open"
external stop_device : int -> bool = "caml_rig_cuda_stop"
external unload_module : int -> int -> int = "caml_rig_cuda_unload"
external word_address : int -> int = "caml_rig_cuda_word" [@@noalloc]

let count () =
  match find_gpus () with Ok g -> Array.length g.devices | Error _ -> 0

let device_name i =
  if i < 0 then invalid_argf "Rig_cuda.device_name: GPU %d is negative" i;
  if i = 0 then "CUDA" else strf "CUDA:%d" i

let driver () =
  match driver_version () with
  | v when v < 0 -> "a CUDA driver of unknown version"
  | v -> strf "the CUDA %d.%d driver" (v / 1000) (v mod 1000 / 10)

(* Unloads the modules [ms] of the device [self]. CUDA's answers are dropped:
   after a fault the modules stay with the context, which the process keeps. *)
let unload_all self ms = List.iter (fun m -> ignore (unload_module self m)) ms

(* Takes a GPU's [held] for a new device: unheld, or held by a stopped device
   whose work has since ended, which is then stopped for good and its [left]
   modules unloaded. *)
let claim held left =
  let p = Atomic.get held in
  if p > 0 || not (Atomic.compare_and_set held p taken) then
    Error "the GPU has a device open; stop it first"
  else if p = unheld then Ok ()
  else if stop_device (-p) then Ok (unload_all (-p) (Atomic.exchange left []))
  else begin
    Atomic.set held p;
    Error "the GPU still runs the work of a stopped device"
  end

let open_ i =
  if i < 0 then invalid_argf "Rig_cuda.open_: GPU %d is negative" i;
  let* g = find_gpus () in
  let n = Array.length g.devices in
  if i >= n then
    Error
      (if n = 0 then "no such GPU; CUDA sees none"
       else strf "no such GPU; CUDA sees %d" n)
  else
    let d = g.devices.(i) in
    match
      let a = attribute d in
      let get = get "reading the GPU's facts" in
      ( get (a attribute_can_use_64_bit_stream_mem_ops),
        get (a attribute_unified_addressing),
        get (a attribute_compute_capability_major),
        get (a attribute_compute_capability_minor),
        get (total_memory d) )
    with
    | exception Fault why -> Error why
    | 0, _, _, _, _ ->
        Error
          (strf "the GPU lacks 64-bit stream memory operations under %s"
             (driver ()))
    | _, 0, _, _, _ ->
        Error (strf "the GPU lacks unified addressing under %s" (driver ()))
    | _, _, major, minor, budget ->
        let held = g.held.(i) and left = g.left.(i) in
        let* () = claim held left in
        let self = open_device d in
        if self < 0 then begin
          Atomic.set held unheld;
          Error
            ("opening the GPU's primary context and streams: " ^ error (-self))
        end
        else begin
          let w = word_address self in
          let word = region self Word ~address:w ~handle:w in
          let arch = strf "sm_%d%d" major minor in
          let images = Atomic.make [] in
          Ok { self; arch; budget; word; held; left; images }
        end

(* Facts *)

let key = Type.Id.make ()
let arch g = g.arch
let budget g = g.budget
let queues _ = [ "COMPUTE:0"; "COPY:0" ]
let completion _ = `Store
let waits_on _ = function `Store | `Host -> true | `Object -> false
let blocks _ = `May_block

type capability = Rig_cuda_abi.t

let functions =
  let symbol name =
    if String.contains name '\000' then None
    else match symbol name with 0 -> None | a -> Some (Nativeint.of_int a)
  in
  { Rig_cuda_abi.symbol }

let capability _ = functions
let capability_key = Rig_cuda_abi.key
let self g = Nativeint.of_int g.self

(* Memory *)

external alloc_memory : int -> bool -> int -> int = "caml_rig_cuda_alloc"
external free_memory : int -> bool -> int -> int = "caml_rig_cuda_free"
external mapped : int -> int -> int = "caml_rig_cuda_mapped"
external allocation : int -> int -> int = "caml_rig_cuda_allocation"
external lock : int -> bool -> int -> int -> int = "caml_rig_cuda_lock"
external enable_peer : int -> int -> int = "caml_rig_cuda_peer"

let alloc g kind n =
  if n < 1 then invalid_argf "Rig_cuda.alloc: %d bytes, expected at least 1" n;
  let host = match kind with `Device -> false | `Pinned | `Mapped -> true in
  match alloc_memory g.self host n with
  | a when a >= 0 ->
      Some (region g.self (if host then Host else Device) ~address:a ~handle:a)
  | _ -> refused (strf "allocating %d bytes" n) g.self None

let address r = Some r.address
let handle r = Nativeint.of_int r.handle
let host r = if on_host r.kind then Some r.handle else None

(* Whether [self]'s GPU addresses the GPU memory of [home]'s, enabling the
   access. *)
let reaches self home =
  match enable_peer self home with
  | 1 -> true
  | 0 -> false
  | s when -s = cuda_error_peer_access_already_enabled -> true
  | _ -> refused "enabling peer access" self false

let peer g g' =
  if g.self = g'.self then invalid_arg "Rig_cuda.peer: the two devices are one";
  reaches g.self g'.self

let map_peer g g' (r : region) =
  if g.self = g'.self then
    invalid_arg "Rig_cuda.map_peer: the two devices are one";
  if r.owner <> g'.self || not (Atomic.get r.live) then
    invalid_arg
      "Rig_cuda.map_peer: the region is no live region of the second device";
  let kind = match r.kind with Peer k -> k | k -> k in
  if not (on_host kind || reaches g.self r.home) then None
  else Some { r with owner = g.self; kind = Peer kind; live = Atomic.make true }

let registry : registration list ref = ref []
let registry_lock = Mutex.create ()
let page = page_size ()
let pages a n = (a / page * page, (a + n + page - 1) / page * page)
let locked g e a address = region g.self (Locked e) ~address ~handle:a
let page_locking n a = strf "page-locking %d bytes at 0x%x" n a

(* A range CUDA did not page-lock is registered. One it did, for another owner,
   is mapped as it is if both its ends lie in one allocation: two allocations
   may have unlocked pages between them. *)
let page_lock g a n =
  let first = mapped g.self a and last = mapped g.self (a + n - 1) in
  if first >= 0 && last >= 0 then
    let start = allocation g.self a in
    if start >= 0 && start = allocation g.self (a + n - 1) then
      Some (locked g None a first)
    else None
  else if first >= 0 || last >= 0 then None
  else if lock g.self true a n <> 0 then refused (page_locking n a) g.self None
  else
    match mapped g.self a with
    | address when address < 0 -> fault (page_locking n a) (-address)
    | address ->
        let e = { start = a; bytes = n; address; maps = 1; stuck = false } in
        registry := e :: !registry;
        Some (locked g (Some e) a address)

let map_host g a n =
  if n < 1 then
    invalid_argf "Rig_cuda.map_host: %d bytes, expected at least 1" n;
  let lo, hi = pages a n in
  let inside e = e.start <= a && a + n <= e.start + e.bytes in
  let shares e =
    let lo', hi' = pages e.start e.bytes in
    lo < hi' && lo' < hi
  in
  Mutex.protect registry_lock @@ fun () ->
  match List.find_opt inside !registry with
  | Some e when e.stuck -> None
  | Some e ->
      e.maps <- e.maps + 1;
      Some (locked g (Some e) a (e.address + (a - e.start)))
  | None -> if List.exists shares !registry then None else page_lock g a n

let free g (r : region) =
  (match r.kind with
  | Word -> invalid_arg "Rig_cuda.free: the region is a timeline word"
  | _ when r.owner <> g.self ->
      invalid_arg "Rig_cuda.free: the region is another device's"
  | _ -> ());
  if not (Atomic.compare_and_set r.live true false) then
    invalid_arg "Rig_cuda.free: the region was freed";
  match r.kind with
  | Device | Host ->
      (* CUDA's answer is dropped: after a fault the memory stays with the
         context, which the process keeps. *)
      ignore (free_memory g.self (r.kind = Host) r.address)
  | Locked (Some e) ->
      Mutex.protect registry_lock @@ fun () ->
      e.maps <- e.maps - 1;
      if e.maps = 0 && not e.stuck then
        if lock g.self false e.start 0 = 0 then
          registry := List.filter (fun e' -> e' != e) !registry
        else e.stuck <- true
  | Locked None | Peer _ | Word -> ()

(* Images *)

external load_module : int -> string -> int = "caml_rig_cuda_load_module"
external get_function : int -> int -> string -> int = "caml_rig_cuda_function"

let rec update a f =
  let x = Atomic.get a in
  if not (Atomic.compare_and_set a x (f x)) then update a f

let image g bin =
  match load_module g.self bin with
  | m when m >= 0 ->
      let i = { owner = g.self; m; loaded = Atomic.make true } in
      update g.images (List.cons i);
      Ok (`Loaded i)
  | s ->
      let step = "loading the image" in
      refused step g.self (Error (strf "%s: %s" step (error (-s))))

let entry (m : image) f =
  if not (Atomic.get m.loaded) then
    invalid_arg "Rig_cuda.entry: the image was unloaded";
  if String.contains f '\000' then None
  else
    match get_function m.owner m.m f with
    | h when h >= 0 -> Some h
    | _ -> refused (strf "finding kernel %S" f) m.owner None

let unload g (m : image) =
  if m.owner <> g.self then
    invalid_arg "Rig_cuda.unload: the image is another device's";
  if not (Atomic.compare_and_set m.loaded true false) then
    invalid_arg "Rig_cuda.unload: the image was unloaded";
  update g.images (List.filter (fun i -> i != m));
  match unload_module g.self m.m with
  | 0 -> ()
  | s -> fault "unloading the image" s

(* Work *)

external room_entry : unit -> int = "caml_rig_cuda_room_entry"
external submit_entry : unit -> int = "caml_rig_cuda_submit_entry"

let room_entry = Nativeint.of_int (room_entry ())
let submit_entry = Nativeint.of_int (submit_entry ())

(* Timeline *)

external signaled : int -> int = "caml_rig_cuda_signaled" [@@noalloc]
external sleep : int -> int -> int -> int = "caml_rig_cuda_sleep"

let word g = g.word
let signaled g = signaled g.self

let sleep g ~seen ~still_ms =
  match sleep g.self seen still_ms with
  | 0 -> ()
  | s -> fault "the GPU's work failed" s

(* Loss *)

let stop g =
  let stopped = stop_device g.self in
  let loaded (i : image) =
    if Atomic.compare_and_set i.loaded true false then Some i.m else None
  in
  let ms = List.filter_map loaded (Atomic.exchange g.images []) in
  if stopped then unload_all g.self ms else Atomic.set g.left ms;
  Atomic.set g.held (if stopped then unheld else -g.self)
