(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Any domain may call any function, as the interface states. The loader, the
   table of GPUs and the page-lock registry are the process's, each behind a
   mutex of its own; a device's C state is written by the C functions, which
   rig serialises under the device's turn, by [grow], [unload] and [image],
   which reach its chains under its C mutex, and by [stop], which runs with no
   other call in flight and frees it. A word of the device's own goes back
   when rig frees it, after the stop. Memory and modules are the GPU's primary
   context's, which the process keeps: their calls take the context, so they
   also serve after the device's C state is gone.

   Every lock is a [Mutex.t] that [protect] holds, counted among the locks its
   thread holds (rig_hold in rig_edge.h). No section takes a second lock, so
   locks never nest. A function that a caller outside rig makes and that may
   take a lock, [count] and an opener's [make], runs inside [guarded].

   The locks are held across calls of the CUDA driver that may wait for the
   GPU: the library's load, a host range's registration, a function's lookup.
   Each must see its table or its device whole, and the driver gives no way to
   end such a call early, so a call of rig's from a callback during one
   raises. *)

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let ( let* ) = Result.bind

exception Fault of string

(* Locks *)

external reraise : exn -> 'a = "%reraise"

(* [take m] takes [m], counted from before the take, so that a signal handler
   that runs inside a contended take sees it held. An exception that ends the
   take gives the count back first, having taken nothing: one a handler
   raised, or the error-checking [Mutex]'s [Sys_error] for a take by the
   thread that holds [m], which only a call that skipped its check makes and
   which raises [Invalid_argument]. *)
let take m =
  Rig_edge_primitive.hold ();
  match Mutex.lock m with
  | () -> ()
  | exception e -> (
      Rig_edge_primitive.unhold ();
      match e with
      | Sys_error why when String.starts_with ~prefix:"Mutex.lock:" why ->
          invalid_arg
            "Rig_cuda: a lock taken again by the thread that holds it, from a \
             signal handler, finaliser or Gc.Memprof callback"
      | e -> reraise e)

(* Gives [m] back, then its count, with nothing between that allocates or
   polls. *)
let give m =
  Mutex.unlock m;
  Rig_edge_primitive.unhold ()

(* [protect m f] is [f ()] holding [m], taken as the last step before the
   handler and given back as the first step of both exits, where nothing
   allocates or polls, so an exception anywhere in [f], an asynchronous one
   too, leaves [m] free and the count right. *)
let protect m f =
  take m;
  match f () with
  | x ->
      give m;
      x
  | exception e ->
      give m;
      reraise e

(* A function a caller outside rig makes, and that may take a lock, runs as
   [guarded fn f]: inside a lock it raises [Invalid_argument] naming [fn],
   having done nothing, since the call comes from a signal handler,
   finaliser or Gc.Memprof callback inside that lock and would wait for it or
   see its section half done. Otherwise it runs [f] with the thread's signals
   blocked ([rig_mask] and [rig_unmask] in [rig_edge.h]), which only the
   driver's waits unblock. *)
let guarded fn f =
  if Rig_edge_primitive.holds () <> 0 then
    invalid_arg
      (fn
     ^ ": called from a signal handler, finaliser or Gc.Memprof callback \
        inside a lock of the calling thread's");
  Rig_edge_primitive.mask ();
  match f () with
  | r ->
      Rig_edge_primitive.unmask ();
      r
  | exception e ->
      Rig_edge_primitive.unmask ();
      reraise e

(* The CUDA library. A stub with a numeric result returns it as a non-negative
   int, or CUDA's status negated. *)

external load : unit -> int * string = "caml_rig_cuda_load"
external error : int -> string = "caml_rig_cuda_error"
external symbol : string -> int = "caml_rig_cuda_symbol"
external driver_version : unit -> int = "caml_rig_cuda_driver_version"
external device_count : unit -> int = "caml_rig_cuda_count"
external device : int -> int = "caml_rig_cuda_device"
external attribute : int -> int -> int = "caml_rig_cuda_attribute"
external bus : int -> bytes -> int = "caml_rig_cuda_bus" [@@noalloc]

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
let attribute_l2_cache_size = 38

(* The most parameter bytes of a kernel CUDA launches: 32,764 on Volta
   (compute capability 7.0) and later, 4,096 below. NVIDIA, "CUDA 12.1
   Supports Large Kernel Parameters": "The parameter limit remains at 4,096
   bytes on architectures below NVIDIA Volta". The larger bound also needs a
   12.1 driver, which the 12.4 library rig requires implies. ptxas refuses
   a kernel past these, so an image of one fails to load first and
   [entry]'s refusal guards what no test reaches. *)
let large_params = 32764
let small_params = 4096
let large_params_major = 7

(* A CUDA failure with [status], in the step form "[step]: NAME: text". *)
let fault step status = raise (Fault (strf "%s: %s" step (error status)))
let get step r = if r < 0 then fault step (-r) else r

external failed : int -> int = "caml_rig_cuda_failed"

(* [refused step self x] is [x], the answer to CUDA's refusal of a call's
   arguments, unless the context of the device [self] failed: CUDA answers its
   error to every call, and [refused] raises it. *)
let refused step self x = match failed self with 0 -> x | e -> fault step e

(* A GPU's holder: [unheld]; [taken], while a device of it is open or being
   opened; or the C edge, negated, of a device stopped while its work still
   ran. [first] keeps its primary context. [placeholder] is the module and
   function of its chains' placeholder kernel, once loaded, which the process
   keeps. *)
type gpus = {
  devices : int array;
  buses : string list;
  held : int Atomic.t array;
  first : int option array;
  reached : int list Atomic.t array;
  placeholders : (int * int) option Atomic.t array;
}

let unheld = 0
let taken = 1
let lock = Mutex.create ()

(* Held by a claim of a GPU. *)
let held_lock = Mutex.create ()
let gpus = ref None

(* A GPU's bus address as Linux's sysfs writes it, such as "0000:01:00.0".
   cuDeviceGetPCIBusId writes that form, in hexadecimal digits, and needs 13
   bytes with its NUL for a 16-bit domain; a wider domain takes more. *)
let bus_id_bytes = 32

let bus_address d =
  let id = Bytes.make bus_id_bytes '\000' in
  ignore (get "finding CUDA's GPUs" (bus d id));
  let n = Option.value (Bytes.index_opt id '\000') ~default:bus_id_bytes in
  String.lowercase_ascii (Bytes.sub_string id 0 n)

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
  | ds -> (
      let devices = Array.of_list (List.map snd (List.sort compare ds)) in
      match Array.to_list (Array.map bus_address devices) with
      | exception Fault why -> Error why
      | buses ->
      let each x = Array.map (fun _ -> Atomic.make x) devices in
      let first = Array.map (fun _ -> None) devices in
      Ok
        {
          devices;
          buses;
          held = each unheld;
          first;
          reached = each [];
          placeholders = each None;
        })

(* Loads the library and finds its GPUs at the first call that needs them, until
   they are found: a failed load is tried again by the next call, so a driver
   installed meanwhile is found. *)
let find_gpus () =
  protect lock @@ fun () ->
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
  maps : int Atomic.t; (* the regions over it *)
  context : int; (* the one it was made in, which ends it *)
  mutable registered : bool; (* until its end, under [registry_lock] *)
}

type kind =
  | Device
  | Host
  | Word (* rig's, in its registered array *)
  | Own_word (* the device's own: its GPU registers no host memory, or the
               word it was given lies outside rig's array *)
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
  | Device | Own_word -> false
  | Host | Word | Locked _ -> true
  | Peer k -> on_host k

(* Opening *)

(* The CUDA C++ Programming Guide, Device Memory Accesses: "Any address of a
   variable residing in global memory or returned by one of the memory
   allocation routines from the driver or runtime API is always aligned to at
   least 256 bytes." [cuMemAlloc] and [cuMemHostAlloc] are such routines. *)
let allocation_alignment = 256

(* The units in which CUDA takes memory, as measured (rig_cuda.mli): an
   allocation of more than 1 MiB in whole blocks of 2 MiB, a smaller one in
   blocks it shares, of which it takes at least its aligned bytes. *)
let granules =
  [ (0, allocation_alignment); ((1 lsl 20) + 1, 2 lsl 20) ]

(* An image's [entries] are those [entry] made, each once, under [lock]. *)
type image = {
  owner : int; (* the C state of the device that loaded it *)
  context : int;
  m : int;
  lock : Mutex.t;
  mutable entries : (string * Rig_edge.entry) list; [@atomic]
  params_bound : int; (* its device's [params_bound] *)
}

type t = {
  self : int;
  context : int; (* its GPU's primary context *)
  gpu : int; (* its GPU's number *)
  device : int; (* its CUdevice *)
  reached : int list Atomic.t; (* its GPU's [reached] *)
  placeholder : (int * int) option Atomic.t; (* its GPU's [placeholder] *)
  facts : region Rig_edge.facts;
  held : int Atomic.t;
  cap : Rig_cuda_abi.t;
  params_bound : int; (* the most parameter bytes of a kernel it launches *)
}

external open_device : int -> nativeint -> bool -> int = "caml_rig_cuda_open"
external context_of : int -> int = "caml_rig_cuda_context" [@@noalloc]
external total_of : int -> int = "caml_rig_cuda_total" [@@noalloc]
external stop_device : int -> bool = "caml_rig_cuda_stop"
external ended : int -> int = "caml_rig_cuda_ended"
external unload_module : int -> int -> int = "caml_rig_cuda_unload"
external word_address : int -> int = "caml_rig_cuda_word" [@@noalloc]
external word_at : int -> int = "caml_rig_cuda_word_at" [@@noalloc]
external keep_word : int -> int = "caml_rig_cuda_keep_word" [@@noalloc]
external enable_peer : int -> int -> int -> int -> int = "caml_rig_cuda_peer"

let count () =
  guarded "Rig_cuda.count" @@ fun () ->
  Result.map (fun g -> Array.length g.devices) (find_gpus ())

let buses () =
  guarded "Rig_cuda.buses" @@ fun () ->
  Result.map (fun g -> g.buses) (find_gpus ())

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

(* Takes a GPU's [held] for a new device: unheld, or held by a stopped device
   whose work has since ended, by completing or by a fault. A GPU has one
   device at a time: two would wait on each other's words through the
   context's shared hardware queues, an order CUDA does not see. *)
let claim held =
  protect held_lock @@ fun () ->
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
  protect lock @@ fun () ->
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

external lock : int -> bool -> int -> int -> int = "caml_rig_cuda_lock"

(* The registrations [map_host] made, with the regions over each. One whose
   last region was freed stays at 0 until the next registration drops it.
   [registry_lock] guards the list and the ends of registrations; a free drops
   its count without it. *)
let registry : registration list ref = ref []
let registry_lock = Mutex.create ()

(* Ends [e]'s registration, whose regions were all freed, under
   [registry_lock]. *)
let unregister e =
  if e.registered then begin
    ignore (lock e.context false e.start 0 : int);
    e.registered <- false
  end

(* Ends each registration of [l] whose regions were all freed, under
   [registry_lock]. *)
let rec end_freed = function
  | [] -> ()
  | e :: rest ->
      if Atomic.get e.maps = 0 then unregister e;
      end_freed rest

(* Whether a free left a registration to end because it found
   [registry_lock] held: set before the free tries the lock, cleared by the
   one that then ends the registrations, under it. *)
let owed = Atomic.make false

(* Ends what frees left, while they leave some and the lock is free. Every
   holder of [registry_lock] drains as it gives the lock back, and a free
   whose try finds the lock held sets [owed] before the try, so that one of
   the two ends the registration: no registration outlives its last free
   past the release of the lock. *)
let rec end_owed () =
  if Atomic.get owed then begin
    Rig_edge_primitive.hold ();
    if Mutex.try_lock registry_lock then begin
      Atomic.set owed false;
      (match end_freed !registry with
      | () -> give registry_lock
      | exception e ->
          give registry_lock;
          reraise e);
      end_owed ()
    end
    else Rig_edge_primitive.unhold ()
  end

(* [protect registry_lock f], then [end_owed ()]. *)
let registered f =
  match protect registry_lock f with
  | r ->
      end_owed ();
      r
  | exception e ->
      end_owed ();
      reraise e

(* Notes the registration of rig's array of words, which the first device
   that opened made and the process keeps: a range [map_host] maps inside it
   counts against it, and it is never given back. *)
let note_words () =
  let start, bytes = Rig_edge_primitive.words () in
  registered @@ fun () ->
  if not (List.exists (fun e -> e.start = start) !registry) then
    registry :=
      { start; bytes; maps = Atomic.make 1; context = 0; registered = true }
      :: !registry

let open_ ~word i =
  guarded "Rig_cuda.gpu" @@ fun () ->
  if i < 0 then invalid_argf "Rig_cuda.gpu: GPU %d is negative" i;
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
        get (a attribute_host_register_supported),
        get (a attribute_l2_cache_size) )
    with
    | exception Fault why -> Error why
    | 0, _, _, _, _, _ ->
        Error
          (strf "the GPU lacks 64-bit stream memory operations under %s"
             (driver ()))
    | _, 0, _, _, _, _ ->
        Error (strf "the GPU lacks unified addressing under %s" (driver ()))
    | _, _, major, minor, registers, l2 -> (
        let held = g.held.(i) in
        let opened = ref (-1) in
        (* An exception after the claim gives it back, stopping the device if
           one was made: a claim kept would refuse the GPU for good. Nothing
           allocates between the claim and the handler, between the device's
           making and its note in [opened], or in the handler. *)
        match claim held with
        | Error why -> Error why
        | Ok () -> (
            match
              let start, bytes = Rig_edge_primitive.words () in
              let w = Nativeint.to_int word in
              let own = registers = 0 || w < start || w + 8 > start + bytes in
              let self = open_device d word own in
              opened := self;
              if self < 0 then begin
                let e =
                  Error
                    ("opening the GPU's primary context and streams: "
                    ^ error (-self))
                in
                Atomic.set held unheld;
                e
              end
              else begin
                let context = context_of self in
                enable_peers g i context;
                if not own then note_words ();
                let kind = if own then Own_word else Word in
                let word =
                  region i kind ~address:(word_at self)
                    ~handle:(word_address self)
                in
                let arch = strf "sm_%d%d" major minor in
                let budget = total_of self in
                let params_bound =
                  if major >= large_params_major then large_params
                  else small_params
                in
                let cap = { Rig_cuda_abi.symbol = find_symbol } in
                let facts =
                  {
                    Rig_edge.arch;
                    budget;
                    queues = [ compute; copy ];
                    copy_engine = Some 1;
                    completion = Store;
                    (* A submission's waits go to its stream in batches, as
                       many as it carries. *)
                    waits = { stores = true; hosts = true };
                    may_block = true;
                    hang_ms = None;
                    maps_host = registers <> 0;
                    host_addresses = false;
                    mapped = false;
                    alignment = allocation_alignment;
                    granules;
                    cache_bytes = (if l2 > 0 then Some l2 else None);
                    counters = [];
                    capability = Rig_cuda_abi.declare cap;
                    word;
                    bus = List.nth_opt g.buses i;
                    edge = Nativeint.of_int self;
                  }
                in
                let reached = g.reached.(i) in
                Ok
                  {
                    self;
                    context;
                    gpu = i;
                    device = d;
                    reached;
                    placeholder = g.placeholders.(i);
                    facts;
                    held;
                    cap;
                    params_bound;
                  }
              end
            with
            | r -> r
            | exception e ->
                let self = !opened in
                if self < 0 || stop_device self then Atomic.set held unheld
                else Atomic.set held (-self);
                reraise e))

(* Facts *)

let key = Type.Id.make ()
let facts g = g.facts
let capability g = g.cap

(* Memory *)

external alloc_memory : int -> bool -> int -> int = "caml_rig_cuda_alloc"
external free_memory : int -> bool -> int -> int = "caml_rig_cuda_free"
external mapped : int -> int -> int = "caml_rig_cuda_mapped"
external allocation : int -> int -> int = "caml_rig_cuda_allocation"

let allocated g kind n =
  match alloc_memory g.context (kind = Host) n with
  | a when a >= 0 -> Some (region g.gpu kind ~address:a ~handle:a)
  | _ -> refused (strf "allocating %d bytes" n) g.self None

(* cuMemGetInfo's pair, read together in the device's context. *)
external memory_info : int -> int array -> int = "caml_rig_cuda_memory"

let capacity g =
  let free_total = [| 0; 0 |] in
  match memory_info g.context free_total with
  | 0 -> Some { Rig_edge.total = free_total.(1); free = free_total.(0) }
  | s -> fault "reading the GPU's memory" s

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

let map_peer g _ (r : region) : region Rig_edge.mapping =
  let kind = match r.kind with Peer k -> k | k -> k in
  if not (on_host kind || addresses g r.home) then Never
  else Maps { r with kind = Peer kind }

let page_locking n a = strf "page-locking %d bytes at 0x%x" n a

(* A region over the locked memory at [a], at the address CUDA's work reaches
   it by. *)
let locked g e a =
  match mapped g.context a with
  | address when address < 0 -> Error (-address)
  | address -> Ok (region g.gpu (Locked e) ~address ~handle:a)

(* Ends every registration whose regions were all freed and that a free left
   registered, then drops those ended, under [registry_lock]: whether it ended
   one. *)
let settle () =
  let ended = ref false in
  List.iter
    (fun e ->
      if Atomic.get e.maps = 0 && e.registered then begin
        unregister e;
        ended := true
      end)
    !registry;
  registry :=
    List.filter (fun e -> Atomic.get e.maps > 0 || e.registered) !registry;
  !ended

(* Counts a region over [e] unless its last was freed. *)
let rec counted e =
  let k = Atomic.get e.maps in
  k > 0 && (Atomic.compare_and_set e.maps k (k + 1) || counted e)

(* CUDA's code for memory it could not lock or map, the only refusal that
   returning memory may undo: it refuses read-only, file-backed and unmapped
   ranges with CUDA_ERROR_INVALID_VALUE. *)
let cuda_error_out_of_memory = 2

(* The answer to CUDA's refusal [s] of a range to lock. *)
let refusal s : _ Rig_edge.mapping =
  if s = cuda_error_out_of_memory then No_room else Never

(* A range inside a registration [map_host] made counts against it. Any other
   is registered; if CUDA refuses, it is mapped as it is when both its ends lie
   in one allocation of CUDA's, which another owner locked: two allocations
   may have unlocked pages between them. A registration a free left to end,
   which may hold the range, ends first. *)
let map_host g a n : region Rig_edge.mapping =
  if n < 1 then
    invalid_argf "Rig_cuda.map_host: %d bytes, expected at least 1" n;
  if not g.facts.maps_host then Never
  else
    registered @@ fun () ->
    let inside e =
      Atomic.get e.maps > 0 && e.start <= a && a + n <= e.start + e.bytes
    in
    (* A count or a registration is recorded with nothing allocated after it,
       and a registration not recorded is ended on any exception: either
       would keep the range locked for good, and a range locked but not
       recorded is mapped uncounted, as another owner's. A count fails once
       a free, which takes no lock, dropped the registration's last. *)
    let rec map () =
      match List.find_opt inside !registry with
      | Some e -> (
          match locked g (Some e) a with
          | Ok r ->
              let some = Rig_edge.Maps r in
              if counted e then some
              else begin
                ignore (settle () : bool);
                map ()
              end
          | Error s -> fault (page_locking n a) s)
      | None -> (
          match lock g.context true a n with
          | 0 -> fresh ()
          | s -> if settle () then map () else other s)
    and fresh () =
      match
        let e =
          {
            start = a;
            bytes = n;
            maps = Atomic.make 1;
            context = g.context;
            registered = true;
          }
        in
        match locked g (Some e) a with
        | Ok r ->
            let some = Rig_edge.Maps r
            and entries =
              e
              :: List.filter
                   (fun e -> Atomic.get e.maps > 0 || e.registered)
                   !registry
            in
            registry := entries;
            some
        | Error s -> fault (page_locking n a) s
      with
      | some -> some
      | exception x ->
          ignore (lock g.context false a 0 : int);
          reraise x
    and other s =
      let start = allocation g.context a in
      if start < 0 || start <> allocation g.context (a + n - 1) then
        refused (page_locking n a) g.self (refusal s)
      else
        match locked g None a with
        | Ok r -> Maps r
        | Error s -> fault (page_locking n a) s
    in
    ignore (settle () : bool);
    map ()

(* Another function's DMA

   cuMemGetHandleForAddressRange exports a range of one allocation as a
   dma-buf, where the driver supports it (CUDA driver API, Memory
   Management): NVIDIA's open kernel modules on Linux. It needs the range on
   the host's page, which cuMemAlloc does not promise, and the export is of
   the whole allocation, so a region's offset in it is its distance from the
   allocation's start. With flags 0 the kernel maps the memory for the
   importer as its topology allows, through BAR1 or a coherent link alike. *)

external range_size : int -> int -> int = "caml_rig_cuda_range_size"
external export_dmabuf : int -> int -> int -> int = "caml_rig_cuda_dmabuf"
external host_page : unit -> int = "caml_rig_cuda_page" [@@noalloc]

let cuda_error_not_supported = 801

(* cuda.h's: whether the device exports its own memory, and memory
   cuMemHostAlloc gave, as dma-bufs. CUDA before 13 lacks the second, which
   it then answers with an error: page-locked memory exports only on Tegra
   there. *)
let attribute_dma_buf_supported = 124
let attribute_host_alloc_dma_buf_supported = 146
let supports g a = attribute g.device a > 0

(* A bus address of this machine, as Linux spells it: "DDDD:BB:DD.F" in
   hexadecimal, the domain in four digits or more, under /sys/bus/pci. *)
let hex n s =
  String.length s >= n
  && String.for_all (function '0' .. '9' | 'a' .. 'f' -> true | _ -> false) s

let bus_function fn =
  match String.split_on_char ':' fn with
  | [ domain; bus; slot ] -> (
      match String.split_on_char '.' slot with
      | [ device; f ] ->
          hex 4 domain
          && String.length bus = 2
          && hex 2 bus
          && String.length device = 2
          && hex 2 device
          && String.length f = 1
          && hex 1 f
      | _ -> false)
  | _ -> false

let known fn =
  if not (bus_function fn) then
    Error (strf "%S is no PCI bus address, expected DDDD:BB:DD.F" fn)
  else if not (Sys.file_exists (Filename.concat "/sys/bus/pci/devices" fn)) then
    Error (strf "%s names no PCI function of this machine" fn)
  else Ok ()

(* CUDA ties the export of page-locked host memory to an attribute of its
   own, and states no other cause of refusing GPU memory than the device's
   attribute: a refusal names the attribute that decides it when that
   attribute is false, and CUDA's status otherwise. *)
let not_supported g (r : region) =
  let attribute =
    match r.kind with
    | Host -> attribute_host_alloc_dma_buf_supported
    | Device | Word | Own_word | Locked _ | Peer _ ->
        attribute_dma_buf_supported
  in
  if supports g attribute then
    "CUDA refused to export the memory as a dma-buf: CUDA_ERROR_NOT_SUPPORTED"
  else if r.kind = Host then
    "CUDA exports no dma-buf of page-locked host memory on this GPU \
     (CU_DEVICE_ATTRIBUTE_HOST_ALLOC_DMA_BUF_SUPPORTED is false, or unknown \
     before CUDA 13)"
  else
    "CUDA exports no dma-buf of this GPU's memory here \
     (CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED is false): that needs Linux and \
     NVIDIA's open kernel modules"

(* The whole allocation [r] lies in, as one dma-buf. *)
let export g (r : region) =
  let step = "exporting GPU memory as a dma-buf" in
  let base = get step (allocation g.context r.address) in
  let size = get step (range_size g.context r.address) in
  let page = host_page () in
  if base mod page <> 0 || size mod page <> 0 then
    Error
      (strf
         "the allocation of %d bytes at 0x%x is off the host's page of %d \
          bytes, which CUDA's export needs"
         size base page)
  else
    match export_dmabuf g.context base size with
    | fd when fd >= 0 -> Ok (Rig_edge.Dmabuf { fd; offset = r.address - base })
    | s when -s = cuda_error_not_supported ->
        refused step g.self (Error (not_supported g r))
    | s -> refused step g.self (Error (strf "%s: %s" step (error (-s))))

let dma g (r : region) ~fn =
  let* () = known fn in
  match r.kind with
  | Host when not (supports g attribute_host_alloc_dma_buf_supported) ->
      Error (not_supported g r)
  | Device | Host -> export g r
  | Word | Own_word | Locked _ | Peer _ ->
      invalid_arg "Rig_cuda.dma: a region alloc did not make"

(* CUDA makes no [Mapped] memory. *)
let flush _ = ()

let free g (r : region) =
  match r.kind with
  | Device | Host ->
      (* CUDA's answer is dropped: after a fault the memory stays with the
         context, which the process keeps. *)
      ignore (free_memory g.context (r.kind = Host) r.address)
  | Locked (Some e) ->
      (* The count drops as the free's first step and nothing after it
         allocates or polls, so no exception lands in the free: rig never
         frees a region again after its free raised. The last region's free
         ends the registration if it takes [registry_lock] without waiting:
         a wait for it would run a pending signal's handler on Windows, where
         no mask blocks a Ctrl-C. Otherwise the lock's holder ends it as it
         gives the lock back ([end_owed]). Only an exception in that drain,
         after the region is freed, lands in the free. The entry stays in the
         registry at 0, for [map_host] to drop. *)
      if Atomic.fetch_and_add e.maps (-1) = 1 then begin
        Atomic.set owed true;
        end_owed ()
      end
  | Own_word ->
      (* The edge keeps the word's value for later claims of the GPU, which
         read it under [held_lock]; the free runs outside it, as CUDA may
         wait for the GPU. *)
      let w = protect held_lock (fun () -> keep_word g.self) in
      ignore (free_memory g.context true w)
  | Word | Locked None | Peer _ -> ()

(* Images *)

external load_module : int -> string -> int = "caml_rig_cuda_load_module"

(* A function's C record, which a launch reads: its CUfunction and limits. *)
external make_entry : int -> int -> string -> int = "caml_rig_cuda_entry"
external entry_code : int -> int = "caml_rig_cuda_code" [@@noalloc]
external entry_params : int -> int = "caml_rig_cuda_params" [@@noalloc]
external entry_threads : int -> int = "caml_rig_cuda_threads" [@@noalloc]
external forget : int -> unit = "caml_rig_cuda_forget" [@@noalloc]

external give_placeholder : int -> int -> int -> unit
  = "caml_rig_cuda_placeholder"

(* The kernel a chain's nodes are made from and return to, which does nothing:
   PTX that CUDA compiles for every GPU a device opens. *)
let placeholder_name = "rig_chain_placeholder"

let placeholder_ptx =
  String.concat "\n"
    [
      ".version 7.0";
      ".target sm_50";
      ".address_size 64";
      strf ".visible .entry %s() { ret; }" placeholder_name;
      "";
    ]

(* Loads the placeholder of [g]'s GPU, unless the process holds it, and gives
   it to [g]'s chains: [Ok ()], or [Error status] with CUDA's refusal. A
   module's load waits for the context's queued work, so the GPU's images
   load it, where a load already waits: a growth runs inside a submit,
   whose earlier work may wait on a word that only the caller's later work
   raises. Two loads that race keep one placeholder, and the other module
   goes. *)
let rec placeholder g =
  match Atomic.get g.placeholder with
  | Some (m, f) ->
      give_placeholder g.self m f;
      Ok ()
  | None -> (
      match load_module g.self placeholder_ptx with
      | m when m < 0 -> Error (-m)
      | m -> (
          match make_entry g.context m placeholder_name with
          | p when p < 0 ->
              ignore (unload_module g.context m : int);
              Error (-p)
          | p ->
              let f = entry_code p in
              forget p;
              if not (Atomic.compare_and_set g.placeholder None (Some (m, f)))
              then ignore (unload_module g.context m : int);
              placeholder g))

(* A placeholder CUDA refuses while the context stands, for lack of memory or
   of the driver's PTX compiler, leaves [g] without chains, and the next image
   tries again: chains only speed runs up, and a cubin needs no compiler. *)
let image g bin =
  (match placeholder g with
  | Ok () -> ()
  | Error _ -> refused "loading the chains' placeholder kernel" g.self ());
  match load_module g.self bin with
  | m when m >= 0 ->
      let i =
        {
          owner = g.self;
          context = g.context;
          m;
          lock = Mutex.create ();
          entries = [];
          params_bound = g.params_bound;
        }
      in
      Ok (Rig_edge.Loaded i)
  | s ->
      refused "loading the image" g.self
        (Error (strf "loading the image: %s" (error (-s))))

let entry (m : image) f =
  if String.contains f '\000' then Rig_edge.Absent
  else
    protect m.lock @@ fun () ->
    match List.assoc_opt f m.entries with
    | Some e -> Rig_edge.Entry e
    | None -> (
        match make_entry m.context m.m f with
        | p when p >= 0 && entry_params p > m.params_bound ->
            let n = entry_params p in
            forget p;
            invalid_argf
              "Rig_cuda.entry: %s takes %d bytes of parameters, more than the \
               %d this GPU and driver launch"
              f n m.params_bound
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

external repoint : int -> int -> int = "caml_rig_cuda_repoint"

(* The nodes of [g]'s chains that may hold [m]'s functions return to the
   placeholder before the module goes, so that no chain names an unloaded
   function. A module CUDA could not repoint stays loaded. *)
let unload g (m : image) =
  let repointed = repoint g.self m.m in
  let unloaded = if repointed = 0 then unload_module g.context m.m else 0 in
  List.iter
    (fun (_, (e : Rig_edge.entry)) -> forget (Nativeint.to_int e.launch))
    m.entries;
  if repointed <> 0 then
    fault "returning a chain's nodes to the placeholder" repointed;
  if unloaded <> 0 then fault "unloading the image" unloaded

(* Timeline *)

external signaled : int -> int = "caml_rig_cuda_signaled" [@@noalloc]

let signaled g = signaled g.self

(* A sleep asks CUDA each [nap_s] whether a stream failed, and naps between the
   questions: a fault is found within a nap of CUDA's report, and the wake
   latency falls on waits rig has already spun on, which are long. *)
let nap_s = 0.001

(* A nap with the thread's signals unblocked, so that a signal's handler runs
   there. *)
let nap () =
  Rig_edge_primitive.unblock ();
  match Unix.sleepf nap_s with
  | () -> Rig_edge_primitive.block ()
  | exception e ->
      Rig_edge_primitive.block ();
      reraise e

let sleep g ~seen ~still_ms =
  let until = Rig_edge_primitive.now () + (still_ms * 1_000_000) in
  let status = ref 0 and more = ref true in
  while !more && signaled g = seen do
    status := failed g.self;
    if !status <> 0 || Rig_edge_primitive.now () >= until then more := false
    else nap ()
  done;
  if !status <> 0 then fault "the GPU's work failed" !status

(* Loss *)

let stop g ~fault:_ =
  let stopped = stop_device g.self in
  Atomic.set g.held (if stopped then unheld else -g.self)

(* Replay *)

external grow_chains : int -> int = "caml_rig_cuda_grow"
external replays_c : int -> int * int = "caml_rig_cuda_replays"
external made_c : int -> int = "caml_rig_cuda_made" [@@noalloc]

(* CUDA takes a graph's device memory in chunks of 2 MiB (measured on an RTX
   5000 Ada), the most one chain's growth asks of the device. *)
let chunk = 2 * 1024 * 1024

let grow g =
  match grow_chains g.self with
  | 0 -> Rig_edge.Grown
  | s when s = cuda_error_out_of_memory -> Short chunk
  | s -> fault "making a chain of kernels" s

let replays g = replays_c g.self
let made g = made_c g.self

(* The load placed every function, and an entry makes nothing slow. *)
let prepare _ _ ~still_ms:_ = true

module Driver = struct
  type nonrec t = t
  type nonrec region = region
  type nonrec image = image

  exception Fault = Fault

  let key = key
  let facts = facts
  let alloc = alloc
  let free = free
  let capacity = capacity
  let locate = locate
  let peer = peer
  let map_peer = map_peer
  let map_host = map_host
  let dma = dma
  let flush = flush
  let image = image
  let entry = entry
  let prepare = prepare
  let unload = unload
  let signaled = signaled
  let sleep = sleep
  let stop = stop
  let grow = grow
end

(* Packed once: a pack copies Driver unless it lists the signature's items
   exactly, in order. *)
let packed = (module Driver : Rig_edge.Driver with type t = t)

let gpu i : t Rig_edge.opener =
  {
    name = device_name i;
    driver = packed;
    make = (fun ~word -> open_ ~word i);
  }
