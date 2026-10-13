(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Any domain may call any function, as the interface states. The loader, the
   table of GPUs and the page-lock registry are the process's, each behind a
   mutex of its own; a device's C state is written only by [submit] and [stop],
   which their caller serialises, and freed with its word. Compiled code calls a
   device's graph maker beside every other call, so native construction and
   [stop] exclude each other under the device's [guard]. Memory, modules and
   graphs are the GPU's primary context's, which the process keeps: their calls
   take the context, so they also serve after the device's C state is gone. *)

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let ( let* ) = Result.bind

exception Fault of string

(* The CUDA library. A stub with a numeric result returns it as a non-negative
   int, or CUDA's status negated. *)

external load : unit -> int * string = "caml_rig_cuda_load"
external error : int -> string = "caml_rig_cuda_error"
external symbol : string -> int = "caml_rig_cuda_symbol"
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
let attribute_host_register_supported = 99
let attribute_kernel_exec_timeout = 17

(* Work that no display's watchdog bounds is hung once it has not advanced for
   30 s, the bound of a GPU driven without a kernel driver. *)
let hang_ms = 30_000

(* A CUDA failure with [status], in the step form "[step]: NAME: text". *)
let fault step status = raise (Fault (strf "%s: %s" step (error status)))
let get step r = if r < 0 then fault step (-r) else r

external failed : int -> int = "caml_rig_cuda_failed"

(* [refused step self x] is [x], the answer to CUDA's refusal of a call's
   arguments, unless the context of the device [self] failed: CUDA answers its
   error to every call, and [refused] raises it. *)
let refused step self x = match failed self with 0 -> x | e -> fault step e

(* A collected graph's destruction, by the process that made it. *)
type graph_cleanup = { pid : int; dispose : unit -> unit }

(* A GPU's holder: [unheld]; [taken], while a device of it is open or being
   opened; or the C edge, negated, of a device stopped while its work still
   ran. [first] keeps its primary context. Its pending graph destruction
   survives close and is drained only by that GPU's later blocking calls. *)
type gpus = {
  devices : int array;
  held : int Atomic.t array;
  first : int option array;
  reached : int list Atomic.t array;
  pending : graph_cleanup list Atomic.t array;
}

let unheld = 0
let taken = 1
let lock = Mutex.create ()

(* Held by a claim of a GPU. *)
let held_lock = Mutex.create ()
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
      let first = Array.map (fun _ -> None) devices in
      Ok
        {
          devices;
          held = each unheld;
          first;
          reached = each [];
          pending = each [];
        }

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

type registration = { start : int; bytes : int; mutable maps : int }

type kind =
  | Device
  | Host
  | Word
  | Locked of registration option (* [None]: another owner's *)
  | Peer of kind

type region = {
  kind : kind;
  address : int;
  handle : int;
  home : int; (* the GPU that holds GPU memory, by number *)
}

let region home kind ~address ~handle = { kind; address; handle; home }

let rec on_host = function
  | Device -> false
  | Host | Word | Locked _ -> true
  | Peer k -> on_host k

(* Opening *)

(* An image's [entries] are those [entry] made, each once, under [lock]. *)
type image = {
  owner : int; (* the C state of the device that loaded it *)
  context : int;
  m : int;
  lock : Mutex.t;
  mutable entries : (string * Rig_edge.entry) list; [@atomic]
  (* twice the references, plus one after unload *)
  mutable graphs : int; [@atomic]
}

type t = {
  self : int;
  context : int; (* its GPU's primary context *)
  gpu : int; (* its GPU's number *)
  reached : int list Atomic.t; (* its GPU's [reached] *)
  facts : region Rig_edge.facts;
  held : int Atomic.t;
  images : image list Atomic.t; (* loaded, whose functions a graph may run *)
  pending : graph_cleanup list Atomic.t;
  guard : Mutex.t; (* held by a graph call and by stop *)
  stopped : bool Atomic.t; (* stop began *)
  cap : Rig_cuda_abi.t;
}

external open_device : int -> int = "caml_rig_cuda_open"
external context_of : int -> int = "caml_rig_cuda_context" [@@noalloc]
external stop_device : int -> bool = "caml_rig_cuda_stop"
external ended : int -> int = "caml_rig_cuda_ended"
external unload_module : int -> int -> int = "caml_rig_cuda_unload"
external word_address : int -> int = "caml_rig_cuda_word" [@@noalloc]
external enable_peer : int -> int -> int -> int -> int = "caml_rig_cuda_peer"

let count () =
  match find_gpus () with Ok g -> Array.length g.devices | Error _ -> 0

let device_name i =
  if i < 0 then invalid_argf "Rig_cuda.gpu: GPU %d is negative" i;
  if i = 0 then "CUDA" else strf "CUDA:%d" i

let driver () =
  match driver_version () with
  | v when v < 0 -> "a CUDA driver of unknown version"
  | v -> strf "the CUDA %d.%d driver" (v / 1000) (v mod 1000 / 10)

let find_symbol name =
  if String.contains name '\000' then None
  else match symbol name with 0 -> None | a -> Some (Nativeint.of_int a)

(* Graphs *)

external make_graph :
  int ->
  int array ->
  int array ->
  string array ->
  int * nativeint * nativeint * nativeint array = "caml_rig_cuda_graph"

external release_graph : int -> nativeint -> nativeint -> unit
  = "caml_rig_cuda_graph_release"

external process : unit -> int = "caml_rig_cuda_process" [@@noalloc]

let rec push pending jobs =
  let before = Atomic.get pending in
  if not (Atomic.compare_and_set pending before (List.rev_append jobs before))
  then push pending jobs

let rec dispose pending = function
  | [] -> ()
  | job :: jobs when job.pid <> process () -> dispose pending jobs
  | job :: rest as jobs -> (
      match job.dispose () with
      | () -> dispose pending rest
      | exception e ->
          push pending jobs;
          raise e)

(* Runs the queued cleanups. A drainer takes them all at once, so another never
   waits for one running. An exception at an OCaml poll puts the cleanup it
   interrupted back with those after it, owned by this GPU; a cleanup resumes
   where it stopped. A child forgets its parent's. *)
let drain pending =
  match Atomic.get pending with
  | [] -> ()
  | _ -> dispose pending (Atomic.exchange pending [])

(* CUDA takes each size as a 32-bit unsigned integer. *)
let size_max = 0xffff_ffff

(* The grid, block and shared memory sizes of kernel [i], checked. *)
let sizes i (k : Rig_cuda_abi.kernel) =
  let gx, gy, gz = k.grid and bx, by, bz = k.block in
  let within lo x = lo <= x && x <= size_max in
  if not (List.for_all (within 1) [ gx; gy; gz; bx; by; bz ]) then
    invalid_argf
      "Rig_cuda_abi.graph: kernel %d has grid %dx%dx%d and block %dx%dx%d, \
       expected each from 1 to %d"
      i gx gy gz bx by bz size_max;
  if not (within 0 k.shared) then
    invalid_argf
      "Rig_cuda_abi.graph: kernel %d has %d bytes of shared memory, expected 0 \
       to %d"
      i k.shared size_max;
  [| gx; gy; gz; bx; by; bz; k.shared |]

(* The image whose entry gave [f], while it is available to new graphs. *)
let image_of images f =
  let gave (i : image) =
    List.exists (fun (_, (e : Rig_edge.entry)) -> e.code = f) i.entries
  in
  List.find_opt gave (Atomic.get images)

let unloading = 1
let graph_ref = 2

let rec keep_image (m : image) =
  let n = m.graphs in
  n land unloading = 0
  && (Atomic.Loc.compare_and_set [%atomic.loc m.graphs] n (n + graph_ref)
     || keep_image m)

let release_image (m : image) =
  let n = Atomic.Loc.fetch_and_add [%atomic.loc m.graphs] (-graph_ref) in
  if n = unloading + graph_ref then
    ignore (unload_module m.context m.m : int)

let graph context guard stopped images pending
    (ks : Rig_cuda_abi.kernel array) =
  let sizes = Array.concat (Array.to_list (Array.mapi sizes ks)) in
  drain pending;
  Mutex.protect guard @@ fun () ->
  if Atomic.get stopped then Error "making the graph: the device was stopped"
  else begin
    let refuse i =
      invalid_argf
        "Rig_cuda_abi.graph: kernel %d's function is of no image the device \
         loaded"
        i
    in
    let owners =
      Array.mapi
        (fun i (k : Rig_cuda_abi.kernel) ->
          match image_of images k.func with Some m -> m | None -> refuse i)
        ks
    in
    let funcs = Array.map (fun (k : Rig_cuda_abi.kernel) -> k.func) ks in
    let args = Array.map (fun (k : Rig_cuda_abi.kernel) -> k.args) ks in
    let left = ref 0 in
    let release_images () =
      while !left > 0 do
        let i = !left - 1 in
        release_image owners.(i);
        left := i
      done
    in
    let empty = (0, Nativeint.zero, Nativeint.zero, [||]) in
    let native = ref empty in
    let dispose () =
      if !native != empty then begin
        let _, handle, source, _ = !native in
        release_graph context handle source;
        native := empty
      end;
      release_images ()
    in
    let pid = process () in
    let job = { pid; dispose } in
    let collect () = if pid = process () then push pending [ job ] in
    (* This job owns every acquired reference, including failed construction.
       The ABI constructor transfers it to the graph on success. *)
    match
      for i = 0 to Array.length owners - 1 do
        if not (keep_image owners.(i)) then refuse i;
        left := i + 1
      done;
      let (s, _, _, _ as made) = make_graph context funcs sizes args in
      if s = 0 then native := made;
      made
    with
    | exception e ->
        collect ();
        raise e
    | s, _, _, _ when s <> 0 ->
        collect ();
        Error ("making the graph: " ^ error s)
    | _, handle, _, nodes -> Ok (Rig_cuda_abi.graph ~handle ~nodes ~collect)
  end

(* Takes a GPU's [held] for a new device: unheld, or held by a stopped device
   whose work has since ended, by completing or by a fault. A GPU has one
   device at a time: two would wait on each other's words through the
   context's shared hardware queues, an order CUDA does not see. *)
let claim held =
  Mutex.protect held_lock @@ fun () ->
  let p = Atomic.get held in
  if p > 0 || not (Atomic.compare_and_set held p taken) then
    Error "the GPU has a device open; stop it first"
  else if p = unheld then Ok ()
  else
    match ended (-p) with
    | 1 -> Ok ()
    | answer ->
        Atomic.set held p;
        if answer = 0 then
          Error "the GPU still runs the work of a stopped device"
        else
          Error
            "the GPU's context cannot be made current to ask whether a \
             stopped device's work has ended"

(* Whether GPU [i] of [g], of primary context [c], addresses the GPU memory of
   GPU [j], of primary context [c'], after enabling the access. A failure is a
   refusal, such as CUDA's limit on a GPU's peers. *)
let reaches (g : gpus) i c j c' =
  match enable_peer g.devices.(i) c g.devices.(j) c' with
  | 1 -> true
  | s -> -s = cuda_error_peer_access_already_enabled

(* Enables access between GPU [i], of primary context [c], and every GPU opened
   before it, both ways, recording the GPUs each reaches. *)
let enable_peers (g : gpus) i c =
  Mutex.protect lock @@ fun () ->
  let add i j = Atomic.set g.reached.(i) (j :: Atomic.get g.reached.(i)) in
  let enable j = function
    | None -> ()
    | Some c' ->
        if reaches g i c j c' then add i j;
        if reaches g j c' i c then add j i
  in
  match g.first.(i) with
  | Some _ -> ()
  | None ->
      Array.iteri enable g.first;
      g.first.(i) <- Some c

(* Each stream runs fills and copies, and COMPUTE:0 launches. *)
let compute = { Rig_edge.name = "COMPUTE:0"; runs = [ Fill; Copy; Launch ] }
let copy = { Rig_edge.name = "COPY:0"; runs = [ Fill; Copy ] }

let open_ i =
  if i < 0 then invalid_argf "Rig_cuda.gpu: GPU %d is negative" i;
  let* g = find_gpus () in
  let n = Array.length g.devices in
  if i >= n then
    Error
      (if n = 0 then "no such GPU; CUDA sees none"
       else strf "no such GPU; CUDA sees %d" n)
  else
    let () = drain g.pending.(i) in
    let d = g.devices.(i) in
    match
      let a = attribute d in
      let get = get "reading the GPU's facts" in
      ( get (a attribute_can_use_64_bit_stream_mem_ops),
        get (a attribute_unified_addressing),
        get (a attribute_compute_capability_major),
        get (a attribute_compute_capability_minor),
        get (a attribute_host_register_supported),
        get (a attribute_kernel_exec_timeout),
        get (total_memory d) )
    with
    | exception Fault why -> Error why
    | 0, _, _, _, _, _, _ ->
        Error
          (strf "the GPU lacks 64-bit stream memory operations under %s"
             (driver ()))
    | _, 0, _, _, _, _, _ ->
        Error (strf "the GPU lacks unified addressing under %s" (driver ()))
    | _, _, major, minor, registers, watchdog, budget ->
        let held = g.held.(i) in
        let* () = claim held in
        let self = open_device d in
        if self < 0 then begin
          Atomic.set held unheld;
          Error
            ("opening the GPU's primary context and streams: " ^ error (-self))
        end
        else begin
          let context = context_of self in
          enable_peers g i context;
          let w = word_address self in
          let word = region i Word ~address:w ~handle:w in
          let arch = strf "sm_%d%d" major minor in
          let images = Atomic.make [] in
          let pending = g.pending.(i) in
          let guard = Mutex.create () and stopped = Atomic.make false in
          let graph = graph context guard stopped images pending in
          let cap = { Rig_cuda_abi.symbol = find_symbol; graph } in
          let facts =
            {
              Rig_edge.arch;
              budget;
              queues = [ compute; copy ];
              copy_engine = Some 1;
              completion = Store;
              (* A submission's waits go to its stream in batches, as many
                 as it carries. *)
              waits =
                {
                  stores = true;
                  hosts = true;
                  objects = false;
                  most = max_int;
                };
              most_parts = max_int;
              may_block = true;
              hang_ms = (if watchdog = 0 then Some hang_ms else None);
              maps_host = registers <> 0;
              host_addresses = false;
              capability = Rig_cuda_abi.declare cap;
              word;
              edge = Nativeint.of_int self;
            }
          in
          let reached = g.reached.(i) in
          Ok
            {
              self;
              context;
              gpu = i;
              reached;
              facts;
              held;
              images;
              pending;
              guard;
              stopped;
              cap;
            }
        end

(* Facts *)

let key = Type.Id.make ()
let facts g = g.facts
let capability g = g.cap

(* Memory *)

external alloc_memory : int -> bool -> int -> int = "caml_rig_cuda_alloc"
external free_memory : int -> bool -> int -> int = "caml_rig_cuda_free"
external mapped : int -> int -> int = "caml_rig_cuda_mapped"
external allocation : int -> int -> int = "caml_rig_cuda_allocation"
external lock : int -> bool -> int -> int -> int = "caml_rig_cuda_lock"

let allocated g kind n =
  match alloc_memory g.context (kind = Host) n with
  | a when a >= 0 -> Some (region g.gpu kind ~address:a ~handle:a)
  | _ -> refused (strf "allocating %d bytes" n) g.self None

(* CUDA maps no GPU memory for the host: [Mapped] is [None]. *)
let alloc g (memory : Rig_edge.memory) n =
  if n < 1 then invalid_argf "Rig_cuda.alloc: %d bytes, expected at least 1" n;
  match memory with
  | Device -> allocated g Device n
  | Pinned -> allocated g Host n
  | Mapped -> None

let locate r =
  {
    Rig_edge.address = Some r.address;
    host = (if on_host r.kind then Some r.handle else None);
    handle = Nativeint.of_int r.handle;
  }

(* Whether [g]'s work addresses the GPU memory of GPU [i], as learned when the
   later of the two GPUs opened. *)
let addresses g i = i = g.gpu || List.mem i (Atomic.get g.reached)
let peer g g' = addresses g g'.gpu

let map_peer g _ (r : region) =
  let kind = match r.kind with Peer k -> k | k -> k in
  if not (on_host kind || addresses g r.home) then None
  else Some { r with kind = Peer kind }

(* The registrations [map_host] made, with the regions over each. *)
let registry : registration list ref = ref []
let registry_lock = Mutex.create ()
let page_locking n a = strf "page-locking %d bytes at 0x%x" n a

(* A region over the locked memory at [a], at the address CUDA's work reaches
   it by. *)
let locked g e a =
  match mapped g.context a with
  | address when address < 0 -> Error (-address)
  | address -> Ok (region g.gpu (Locked e) ~address ~handle:a)

(* A range inside a registration [map_host] made counts against it. Any other
   is registered; if CUDA refuses, it is mapped as it is when both its ends lie
   in one allocation of CUDA's, which another owner locked: two allocations
   may have unlocked pages between them. *)
let map_host g a n =
  if n < 1 then
    invalid_argf "Rig_cuda.map_host: %d bytes, expected at least 1" n;
  if not g.facts.maps_host then None
  else
    Mutex.protect registry_lock @@ fun () ->
    let inside e = e.start <= a && a + n <= e.start + e.bytes in
    match List.find_opt inside !registry with
    | Some e -> (
        match locked g (Some e) a with
        | Ok r ->
            e.maps <- e.maps + 1;
            Some r
        | Error s -> fault (page_locking n a) s)
    | None when lock g.context true a n = 0 -> (
        let e = { start = a; bytes = n; maps = 1 } in
        match locked g (Some e) a with
        | Ok r ->
            registry := e :: !registry;
            Some r
        | Error s ->
            ignore (lock g.context false a 0 : int);
            fault (page_locking n a) s)
    | None -> (
        let start = allocation g.context a in
        if start < 0 || start <> allocation g.context (a + n - 1) then
          refused (page_locking n a) g.self None
        else
          match locked g None a with
          | Ok r -> Some r
          | Error s -> fault (page_locking n a) s)

let free g (r : region) =
  match r.kind with
  | Device | Host ->
      (* CUDA's answer is dropped: after a fault the memory stays with the
         context, which the process keeps. *)
      ignore (free_memory g.context (r.kind = Host) r.address)
  | Locked (Some e) ->
      Mutex.protect registry_lock @@ fun () ->
      e.maps <- e.maps - 1;
      if e.maps = 0 then begin
        registry := List.filter (fun e' -> e' != e) !registry;
        ignore (lock g.context false e.start 0 : int)
      end
  | Word | Locked None | Peer _ -> ()

(* Images *)

external load_module : int -> string -> int = "caml_rig_cuda_load_module"

(* A function's C record, which a launch reads: its CUfunction and limits. *)
external make_entry : int -> int -> string -> int = "caml_rig_cuda_entry"
external entry_code : int -> int = "caml_rig_cuda_code" [@@noalloc]
external entry_params : int -> int = "caml_rig_cuda_params" [@@noalloc]
external entry_threads : int -> int = "caml_rig_cuda_threads" [@@noalloc]
external forget : int -> unit = "caml_rig_cuda_forget" [@@noalloc]

let rec update a f =
  let x = Atomic.get a in
  if not (Atomic.compare_and_set a x (f x)) then update a f

let image g bin =
  drain g.pending;
  match load_module g.self bin with
  | m when m >= 0 ->
      let i =
        {
          owner = g.self;
          context = g.context;
          m;
          lock = Mutex.create ();
          entries = [];
          graphs = 0;
        }
      in
      update g.images (List.cons i);
      Ok (Rig_edge.Loaded i)
  | s ->
      let step = "loading the image" in
      refused step g.self (Error (strf "%s: %s" step (error (-s))))

let entry (m : image) f =
  if String.contains f '\000' then Rig_edge.Absent
  else
    Mutex.protect m.lock @@ fun () ->
    match List.assoc_opt f m.entries with
    | Some e -> Rig_edge.Entry e
    | None -> (
        match make_entry m.context m.m f with
        | p when p >= 0 ->
            let e =
              {
                Rig_edge.code = entry_code p;
                launch = Nativeint.of_int p;
                params = Some (entry_params p);
                threads = Some (entry_threads p);
              }
            in
            m.entries <- (f, e) :: m.entries;
            Entry e
        | _ -> refused (strf "finding kernel %S" f) m.owner Rig_edge.Absent)

let unload g (m : image) =
  drain g.pending;
  update g.images (List.filter (fun i -> i != m));
  let unloaded =
    if Atomic.Loc.fetch_and_add [%atomic.loc m.graphs] unloading = 0 then
      unload_module g.context m.m
    else 0
  in
  List.iter
    (fun (_, (e : Rig_edge.entry)) -> forget (Nativeint.to_int e.launch))
    m.entries;
  if unloaded <> 0 then fault "unloading the image" unloaded

(* Timeline *)

external signaled : int -> int = "caml_rig_cuda_signaled" [@@noalloc]

let signaled g = signaled g.self

(* The host's clock, [rig_now_ns] of [rig_edge.h]. *)
external now : unit -> (int[@untagged])
  = "caml_rig_edge_now_byte" "caml_rig_edge_now"
[@@noalloc]

(* A sleep asks CUDA each [nap_s] whether a stream failed, and naps between the
   questions: a fault is found within a nap of CUDA's report, and the wake
   latency falls on waits rig has already spun on, which are long. *)
let nap_s = 0.001

let sleep g ~seen ~still_ms =
  let until = now () + (still_ms * 1_000_000) in
  let status = ref 0 and more = ref true in
  while !more && signaled g = seen do
    status := failed g.self;
    if !status <> 0 || now () >= until then more := false
    else Unix.sleepf nap_s
  done;
  if !status <> 0 then fault "the GPU's work failed" !status

(* Loss *)

let stop g ~fault:_ =
  Mutex.protect g.guard @@ fun () ->
  Atomic.set g.stopped true;
  let stopped = stop_device g.self in
  Atomic.set g.held (if stopped then unheld else -g.self)

module Driver = struct
  type nonrec t = t
  type nonrec region = region
  type nonrec image = image

  exception Fault = Fault

  let key = key
  let facts = facts
  let alloc = alloc
  let free = free
  let locate = locate
  let peer = peer
  let map_peer = map_peer
  let map_host = map_host
  let image = image
  let entry = entry
  let unload = unload
  let signaled = signaled
  let sleep = sleep
  let stop = stop
end

let gpu i : t Rig_edge.opener =
  {
    name = device_name i;
    driver = (module Driver);
    make = (fun ~word:_ -> open_ i);
  }
