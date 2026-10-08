(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The process's KFD state (its file, each GPU's address space, the event page)
   is made under [opening]; a device's queues and events are its own. Any domain
   may call the path's functions. *)

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let ( let* ) = Result.bind

module Amd = Rig_amd

external linux : unit -> bool = "caml_rig_amd_amdgpu_linux"
external strerror : int -> string = "caml_rig_amd_amdgpu_strerror"
external open_file : string -> int = "caml_rig_amd_amdgpu_open"
external close_file : int -> unit = "caml_rig_amd_amdgpu_close"
external reserve : int -> int = "caml_rig_amd_amdgpu_reserve"

external map_file : int -> int -> int -> bytes -> int
  = "caml_rig_amd_amdgpu_map"

external unmap_mem : int -> int -> unit = "caml_rig_amd_amdgpu_unmap"
external store64 : int -> int -> unit = "caml_rig_amd_amdgpu_store64"
external version : int -> int = "caml_rig_amd_amdgpu_version"

external acquire_vm : int -> int -> int -> int
  = "caml_rig_amd_amdgpu_acquire_vm"

external runtime_enable : int -> int = "caml_rig_amd_amdgpu_runtime_enable"

external kfd_alloc : int -> int -> int -> int -> int -> bytes -> int
  = "caml_rig_amd_amdgpu_alloc_byte" "caml_rig_amd_amdgpu_alloc"

external kfd_free : int -> int -> int = "caml_rig_amd_amdgpu_free"

external map_gpu : int -> int -> int -> bool -> int
  = "caml_rig_amd_amdgpu_map_gpu"

external event : int -> int -> int -> int = "caml_rig_amd_amdgpu_event"

external destroy_event : int -> int -> int
  = "caml_rig_amd_amdgpu_destroy_event"

external kfd_queue : int -> int -> int array -> bytes -> int
  = "caml_rig_amd_amdgpu_queue"

external destroy_queue : int -> int -> int
  = "caml_rig_amd_amdgpu_destroy_queue"

external wait : int -> int array -> int -> int -> int array -> int
  = "caml_rig_amd_amdgpu_wait"

external device_info : int -> int array -> int
  = "caml_rig_amd_amdgpu_device_info"

external stable_pstate : int -> int = "caml_rig_amd_amdgpu_stable_power"
external pid : unit -> int = "caml_rig_amd_amdgpu_pid"

(* Errors *)

let eacces = -13
let ebusy = -16
let enomem = -12
let einval = -22
let fault step e = raise (Amd.Fault (strf "%s: %s" step (strerror (-e))))
let check step e = if e < 0 then fault step e
let page = 4096
let round_up n a = (n + a - 1) / a * a

let opened path what e =
  if e >= 0 then Ok e
  else if e = eacces then
    Error
      (strf
         "opening %s: %s; %s needs read and write access, which members of the \
          render group have"
         path (strerror (-e)) what)
  else Error (strf "opening %s: %s" path (strerror (-e)))

(* The process's KFD *)

type gpu = {
  node : Topology.node;
  drm : int; (* its render node *)
  hdp : int option; (* the register whose store flushes the HDP *)
  clock_khz : int;
  cus : int array; (* its active compute units, as cu_bitmap lays them out *)
  mutable doorbells : (int64 * int) option; (* the page: offset, address *)
  mutable events_mapped : bool;
  mutable stable : bool;
  faulted : string option Atomic.t;
      (* the first fault the kernel driver reported in the address space *)
  links : int list; (* the topology nodes this GPU has a link to *)
  mutable live : int array list;
      (* the exception events, memory and hardware, of its open devices *)
}

let opening = Mutex.create ()

(* The process that opened the file, and the file. *)
let kfd : (int * int) option ref = ref None

(* The GPUs whose address space the process acquired, by GPU id: the kernel
   driver lets a process acquire one once. *)
let acquired : (int, gpu) Hashtbl.t = Hashtbl.create 4

(* The process's event page: its handle, host address and GPU. *)
let event_page : (int * int * int) option ref = ref None

(* Views of another GPU's memory, by its handle and the viewing GPU's id: KFD
   maps a memory once per GPU, so the last view's free unmaps it. *)
let views : (int * int, int) Hashtbl.t = Hashtbl.create 16
let views_lock = Mutex.create ()

(* KFD serves a file to the process that opened it only: a forked child opens
   its own, and the parent's address spaces, event page and views stay the
   parent's. *)
let kfd_fd () =
  match !kfd with
  | Some (p, fd) when p = pid () -> Ok fd
  | _ ->
      Hashtbl.reset acquired;
      Hashtbl.reset views;
      event_page := None;
      let* fd =
        opened "/dev/kfd" "the compute interface" (open_file "/dev/kfd")
      in
      kfd := Some (pid (), fd);
      Ok fd

(* Memory *)

type kind = Own | Borrowed | Peer | View

type mem = {
  handle : int;
  bytes : int;
  at : int; (* the process's addresses, unmapped at the free *)
  kind : kind;
  owner : gpu; (* the GPU it was allocated for *)
}

let get64 b i = Int64.to_int (Bytes.get_int64_le b (8 * i))
let offset b = Bytes.sub b 8 8
let userptr = 3
let mmio = 4
let alloc_kind = function `Gpu -> 0 | `Bar -> 1 | `System -> 2

(* [n] bytes of [kind] at addresses reserved in the process, mapped for the GPU,
   and for the host unless they are [`Gpu]. *)
let alloc fd g kind n : mem Amd.memory option =
  if kind = `Bar && g.node.visible = 0 then None
  else
    let n = round_up n page in
    let at = reserve n in
    check "reserving GPU addresses" at;
    let b = Bytes.create 16 in
    match kfd_alloc fd g.node.gpu_id at n (alloc_kind kind) b with
    | e when e = enomem || (e = einval && kind = `Bar) ->
        unmap_mem at n;
        None
    | e when e < 0 ->
        unmap_mem at n;
        fault (strf "allocating %d bytes of GPU memory" n) e
    | _ ->
        let handle = get64 b 0 in
        let undo step e =
          ignore (kfd_free fd handle);
          unmap_mem at n;
          fault step e
        in
        let e = if kind = `Gpu then 0 else map_file g.drm at n (offset b) in
        if e < 0 then undo "mapping GPU memory" e;
        let e = map_gpu fd handle g.node.gpu_id true in
        if e < 0 then undo "mapping memory for the GPU" e;
        let host = if kind = `Gpu then None else Some at in
        let data = { handle; bytes = n; at; kind = Own; owner = g } in
        Some { Amd.address = at; host; data }

(* The kernel driver lets go of memory before the process unmaps it: unmapping
   host memory the driver still maps makes it evict every queue of the process
   and restore them later, which stalls the next work 5 to 10 ms. *)
let free fd g (m : mem Amd.memory) =
  let p = m.data in
  match p.kind with
  | View -> ()
  | Peer ->
      let key = (p.handle, g.node.gpu_id) in
      let last =
        Mutex.protect views_lock (fun () ->
            let n = Hashtbl.find views key - 1 in
            if n = 0 then Hashtbl.remove views key
            else Hashtbl.replace views key n;
            n = 0)
      in
      if last then
        check "unmapping another GPU's memory"
          (map_gpu fd p.handle g.node.gpu_id false)
  | Own | Borrowed ->
      ignore (map_gpu fd p.handle g.node.gpu_id false);
      check "freeing GPU memory" (kfd_free fd p.handle);
      if p.kind = Own then unmap_mem p.at p.bytes

let map_host fd g a n : mem Amd.memory option =
  let base = a land lnot (page - 1) in
  let bytes = round_up (a + n - base) page in
  let b = Bytes.create 16 in
  if kfd_alloc fd g.node.gpu_id base bytes userptr b < 0 then None
  else
    let handle = get64 b 0 in
    if map_gpu fd handle g.node.gpu_id true < 0 then begin
      ignore (kfd_free fd handle);
      None
    end
    else
      let data = { handle; bytes; at = base; kind = Borrowed; owner = g } in
      Some { Amd.address = a; host = Some a; data }

(* Memory of another device: of the same GPU, in this address space already; of
   a GPU the topology links this one to, mapped for it. *)
let linked g (n : Topology.node) =
  n.gpu_id = g.node.gpu_id || List.mem n.index g.links

let map_peer fd g (m : mem Amd.memory) =
  let o = m.data.owner in
  if o.node.gpu_id = g.node.gpu_id then
    Some { m with data = { m.data with kind = View } }
  else if not (linked g o.node) then None
  else
    let key = (m.data.handle, g.node.gpu_id) in
    Mutex.protect views_lock @@ fun () ->
    let n = Option.value ~default:0 (Hashtbl.find_opt views key) in
    if n = 0 && map_gpu fd m.data.handle g.node.gpu_id true < 0 then None
    else begin
      Hashtbl.replace views key (n + 1);
      Some { m with data = { m.data with kind = Peer } }
    end

(* Opening a GPU *)

(* The page of registers the kernel driver remaps for the process, whose first
   word flushes the host data path, or [None] if it refuses it. *)
let remap_hdp fd gpu_id =
  let at = reserve page in
  let b = Bytes.create 16 in
  if at < 0 then None
  else if kfd_alloc fd gpu_id at page mmio b < 0 then begin
    unmap_mem at page;
    None
  end
  else
    let p = map_file fd at page (offset b) in
    if p >= 0 then Some p
    else begin
      ignore (kfd_free fd (get64 b 0));
      unmap_mem at page;
      None
    end

(* KFD 1.14 asks a process to enable its runtime before using queues. *)
let runtime_from = 1014

(* The machine's AMD GPUs in bus order, with their topology nodes, read once:
   the kernel driver's topology does not change while the process runs. *)
let machine = ref None
let machine_lock = Mutex.create ()

let topology () =
  Mutex.protect machine_lock @@ fun () ->
  match !machine with
  | Some t -> t
  | None ->
      let t =
        if not (linux ()) then [||]
        else
          Array.of_list
            (List.map
               (fun bus -> (bus, Topology.node "/" bus))
               (Topology.gpus "/"))
      in
      machine := Some t;
      t

let acquire fd (node : Topology.node) =
  match Hashtbl.find_opt acquired node.gpu_id with
  | Some g -> Ok g
  | None ->
      let path = strf "/dev/dri/renderD%d" node.render in
      let* drm = opened path "the GPU's render node" (open_file path) in
      let ok step e =
        if e >= 0 then Ok e
        else begin
          close_file drm;
          Error (strf "%s: %s" step (strerror (-e)))
        end
      in
      let acquiring = "acquiring the GPU's address space" in
      let* v = ok "reading KFD's version" (version fd) in
      let* _ = ok acquiring (acquire_vm fd drm node.gpu_id) in
      let* _ =
        if v < runtime_from then Ok 0 else ok acquiring (runtime_enable fd)
      in
      let cus = Array.make 16 0 in
      let* khz = ok "reading the GPU's facts" (device_info drm cus) in
      let links =
        Array.to_list (topology ())
        |> List.filter_map (function
          | _, Ok (n : Topology.node)
            when Topology.linked "/" node.index n.index ->
              Some n.index
          | _ -> None)
      in
      let g =
        {
          node;
          drm;
          hdp = remap_hdp fd node.gpu_id;
          clock_khz = khz;
          cus;
          doorbells = None;
          events_mapped = false;
          stable = false;
          faulted = Atomic.make None;
          links;
          live = [];
        }
      in
      Hashtbl.replace acquired node.gpu_id g;
      Ok g

(* Events. On an interrupt from the GPU, the kernel driver sets a signal event
   only if the event's slot of the event page holds a value other than all ones,
   and puts all ones back. The host arms the slot, a signal event's id, when the
   event is made and after each wait, before the caller reads the timeline word
   again: an interrupt then finds the slot armed, or comes between a wait and
   its arm, before a read that sees the word the work wrote. *)

let event_page_bytes = 0x8000
let signal = 0
let memory_exception = 1
let hardware_exception = 2

let arm id =
  match !event_page with
  | Some (_, host, _) -> store64 (host + (8 * id)) id
  | None -> ()

(* The process's event page, made with a signal event of its own, which no
   device uses, so that device events have ids above 0, and mapped for [g]. *)
let page_for fd g =
  match !event_page with
  | Some (handle, _, owner) ->
      if owner <> g.node.gpu_id && not g.events_mapped then
        check "mapping the event page" (map_gpu fd handle g.node.gpu_id true);
      g.events_mapped <- true
  | None -> (
      match alloc fd g `System event_page_bytes with
      | None -> raise (Amd.Fault "no memory for the event page")
      | Some m ->
          check "making the event page" (event fd signal m.data.handle);
          event_page := Some (m.data.handle, m.address, g.node.gpu_id);
          g.events_mapped <- true)

let make_events fd g =
  page_for fd g;
  let made = ref [] in
  let make k =
    let id = event fd k 0 in
    if id < 0 then begin
      List.iter (fun id -> ignore (destroy_event fd id)) !made;
      fault "making a KFD event" id
    end;
    made := id :: !made;
    id
  in
  let ids = Array.map make [| signal; memory_exception; hardware_exception |] in
  arm ids.(0);
  ids

(* A device *)

type device = {
  fd : int;
  gpu : gpu;
  events : int array;
  mutable events_held : bool; (* until destroyed, once *)
  mutable queues : (int * mem Amd.memory list) list; (* id, its memory *)
}

(* Event ids are the process's: one destroyed twice may already name another
   device's event, which the second destroy would take from it. *)
let drop_events d =
  if d.events_held then begin
    d.events_held <- false;
    Mutex.protect opening (fun () ->
        d.gpu.live <- List.filter (fun e -> e != d.events) d.gpu.live);
    Array.iter (fun id -> ignore (destroy_event d.fd id)) d.events
  end

let eop_bytes = 0x1000

(* The waves the context save area holds, as the kernel driver counts them
   (kfd_queue.c, kfd_queue_ctx_save_restore_size): 32 per compute unit from GFX
   10.1, before it 40 per compute unit up to 512 per shader engine. *)
let waves (n : Topology.node) =
  let g = n.gpu in
  if compare g.target (10, 1, 0) < 0 then
    Int.min (g.compute_units * 40) (g.shader_engines * g.xccs * 512)
  else g.compute_units * 32

(* A compute queue's context save area: each die's, and the debugger's 32 bytes
   per wave after it. *)
let save_bytes (n : Topology.node) =
  let debug = round_up (waves n * 32) 64 in
  round_up ((n.cwsr + debug) * n.gpu.xccs) page

let queue_kind = function `Pm4 -> 0 | `Aql -> 1 | `Sdma -> 2

(* The doorbell at [off] in the process's doorbell page of the GPU, which KFD
   maps from the page's own offset. *)
let doorbell d off =
  let g = d.gpu in
  let base = Int64.logand off (Int64.lognot 0x1fffL) in
  let page_at =
    match g.doorbells with
    | Some (_, at) -> at
    | None ->
        let b = Bytes.create 8 in
        Bytes.set_int64_le b 0 base;
        let at = map_file d.fd 0 0x2000 b in
        check "mapping the doorbell page" at;
        g.doorbells <- Some (base, at);
        at
  in
  page_at + Int64.to_int (Int64.sub off base)

let queue d kind ~ring ~bytes ~read ~write =
  let compute = kind <> `Sdma in
  let gpu_mem what n =
    if not compute then Ok None
    else
      match alloc d.fd d.gpu `Gpu n with
      | Some m -> Ok (Some m)
      | None -> Error (strf "no GPU memory for a queue's %s" what)
  in
  let address = function Some (m : _ Amd.memory) -> m.address | None -> 0 in
  let node = d.gpu.node in
  let* eop = gpu_mem "end-of-pipe buffer" eop_bytes in
  let* save =
    let give_back () = Option.iter (free d.fd d.gpu) eop in
    match gpu_mem "context save area" (save_bytes node) with
    | Ok _ as s -> s
    | Error _ as e ->
        give_back ();
        e
    | exception (Amd.Fault _ as e) ->
        give_back ();
        raise e
  in
  let taken = List.filter_map Fun.id [ eop; save ] in
  let args =
    [|
      node.gpu_id;
      ring;
      bytes;
      address eop;
      (if compute then eop_bytes else 0);
      address save;
      (if compute then node.cwsr else 0);
      (if compute then node.ctl_stack else 0);
      write;
      read;
    |]
  in
  let b = Bytes.create 16 in
  match kfd_queue d.fd (queue_kind kind) args b with
  | e when e < 0 ->
      List.iter (free d.fd d.gpu) taken;
      Error (strf "making a KFD queue: %s" (strerror (-e)))
  | _ ->
      d.queues <- (get64 b 0, taken) :: d.queues;
      Ok (Mutex.protect opening (fun () -> doorbell d (Bytes.get_int64_le b 8)))

(* The fault [wait]'s answer [e] reports, with its data [r]. A fault leaves the
   process's queues on the GPU unscheduled, those made after it too, so the GPU
   keeps the first for every later sleep and open. *)
let report g e r =
  let why =
    match e with
    | 1 ->
        Some
          (strf
             "memory fault at 0x%x (not present %d, read-only %d, no execute \
              %d, imprecise %d, error type %d)"
             r.(0) r.(1) r.(2) r.(3) r.(4) r.(5))
    | 2 ->
        Some
          (strf
             "hardware exception (reset type %d, reset cause %d, memory lost \
              %d)"
             r.(0) r.(1) r.(2))
    | _ -> None
  in
  match why with
  | Some _ -> ignore (Atomic.compare_and_set g.faulted None why)
  | None -> ()

let raise_fault g =
  match Atomic.get g.faulted with
  | Some why -> raise (Amd.Fault why)
  | None -> ()

(* Allocates the stub's answer alone, as a wait that reported no fault does. *)
let sleep d ~ms =
  let g = d.gpu in
  raise_fault g;
  let r = Array.make 6 0 in
  let e = wait d.fd d.events g.node.gpu_id ms r in
  arm d.events.(0);
  if e < 0 then fault "waiting for KFD events" e;
  if e > 0 then begin
    report g e r;
    raise_fault g
  end

(* Asks the exception events of the GPU's open devices, without waiting, for a
   fault no sleep has read yet. The events stay set: they reset only by hand. *)
let poll_faults fd g =
  let r = Array.make 6 0 in
  List.iter
    (fun ev ->
      if Atomic.get g.faulted = None then
        report g (wait fd [| ev.(1); ev.(2) |] g.node.gpu_id 0 r) r)
    g.live

let stable_power g () =
  Mutex.protect opening @@ fun () ->
  match g.node.gpu.target with
  | 9, _, _ -> Ok ()
  | _ when g.stable -> Ok ()
  | _ -> (
      match stable_pstate g.drm with
      | 0 ->
          g.stable <- true;
          Ok ()
      | e when e = ebusy ->
          Error "another process holds the GPU's stable power state"
      | e ->
          Error
            (strf "the amdgpu driver refused the GPU's stable power state: %s"
               (strerror (-e))))

(* Each queue destroyed, and the memory of those destroyed given back. A queue
   the kernel driver kept may still run and raise the device's interrupt, so its
   memory and the events stay. *)
let stop d () =
  let gone, kept =
    List.partition (fun (id, _) -> destroy_queue d.fd id = 0) d.queues
  in
  let give_back m = try free d.fd d.gpu m with Amd.Fault _ -> () in
  List.iter (fun (_, mems) -> List.iter give_back mems) gone;
  d.queues <- kept;
  if kept <> [] then `Unknown
  else begin
    drop_events d;
    `Stopped
  end

(* The work-group processors that run work in each shader array, engines
   numbered across dies: on GFX10 on, the pairs of compute units both active; on
   GFX9, the compute units. [cus] is the render node's bitmap, which holds the
   first die's alone: AMDGPU_INFO_DEV_INFO copies [cu_info.bitmap[0]] of the
   kernel's per-die bitmaps (amdgpu_kms.c), and neither KFD's topology nor its
   debugger snapshot holds another. A later die's arrays therefore have every
   processor of their [per_array] compute units set; a counter of one that runs
   no work stays 0. *)
let wgps_of (gpu : Rig_amd_abi.Gpu.t) ~arrays ~per_array cus =
  let gfx9 = match gpu.target with 9, _, _ -> true | _ -> false in
  let pairs c =
    let m = ref 0 in
    for w = 0 to 15 do
      if (c lsr (2 * w)) land 3 = 3 then m := !m lor (1 lsl w)
    done;
    !m
  in
  let units = if gfx9 then per_array else per_array / 2 in
  let all = (1 lsl units) - 1 in
  let engines = gpu.shader_engines in
  Array.init (engines * gpu.xccs) (fun e ->
      Array.init arrays (fun a ->
          if e >= engines then all
          else
            let c = cus.((4 * (e mod 4)) + a + (e / 4 * arrays)) in
            if gfx9 then c else pairs c))

let key : mem Type.Id.t = Type.Id.make ()

let path d ~index : mem Amd.path =
  let g = d.gpu and fd = d.fd in
  {
    key;
    index;
    gpu = g.node.gpu;
    waves = g.node.waves_per_cu;
    lds = g.node.lds;
    clock_hz = g.clock_khz * 1000;
    mec = g.node.mec;
    wgps =
      wgps_of g.node.gpu ~arrays:g.node.arrays ~per_array:g.node.cu_per_array
        g.cus;
    budget = g.node.budget;
    alloc = alloc fd g;
    map_host = map_host fd g;
    reaches =
      (let reached =
         Array.mapi
           (fun j (_, n) ->
             j = index || match n with Ok n -> linked g n | Error _ -> false)
           (topology ())
       in
       fun j -> j >= 0 && j < Array.length reached && reached.(j));
    map_peer = map_peer fd g;
    free = free fd g;
    queue = queue d;
    hdp = g.hdp;
    interrupt = d.events.(0);
    hang_ms = None;
    sleep = sleep d;
    stable_power = stable_power g;
    stop = stop d;
  }

(* Numbering *)

let gpus_at = Topology.gpus

let gpu_at root bus =
  Result.map (fun (n : Topology.node) -> n.gpu) (Topology.node root bus)

let save_area_at root bus = Result.map save_bytes (Topology.node root bus)
let count () = Array.length (topology ())

let device_name i =
  if i < 0 then
    invalid_argf "Rig_amd_amdgpu.device_name: GPU %d is negative" i;
  if i = 0 then "AMD" else strf "AMD:%d" i

let open_ i =
  if i < 0 then invalid_argf "Rig_amd_amdgpu.open_: GPU %d is negative" i;
  let gpus = topology () in
  if i >= Array.length gpus then
    Error (strf "no GPU %d; the machine has %d AMD GPUs" i (Array.length gpus))
  else
    let* node = snd gpus.(i) in
    let* fd, g, events =
      Mutex.protect opening (fun () ->
          let* fd = kfd_fd () in
          let* g = acquire fd node in
          poll_faults fd g;
          match Atomic.get g.faulted with
          | Some why ->
              Error
                (strf
                   "GPU %d faulted in this process (%s); a new process opens it"
                   i why)
          | None -> (
              match make_events fd g with
              | events -> Ok (fd, g, events)
              | exception Amd.Fault why -> Error why))
    in
    let d = { fd; gpu = g; events; events_held = true; queues = [] } in
    (* The events stay while a queue the path could not stop may still raise the
       interrupt. *)
    let give_back () = if d.queues = [] then drop_events d in
    match Amd.make (path d ~index:i) with
    | Ok _ as r ->
        Mutex.protect opening (fun () -> g.live <- events :: g.live);
        r
    | Error _ as e ->
        give_back ();
        e
    | exception Amd.Fault why ->
        give_back ();
        Error why
