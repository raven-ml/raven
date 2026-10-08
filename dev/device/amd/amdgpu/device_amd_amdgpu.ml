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

module Amd = Device_amd

external linux : unit -> bool = "caml_device_amd_amdgpu_linux"
external strerror : int -> string = "caml_device_amd_amdgpu_strerror"
external open_file : string -> int = "caml_device_amd_amdgpu_open"
external close_file : int -> unit = "caml_device_amd_amdgpu_close"
external reserve : int -> int = "caml_device_amd_amdgpu_reserve"

external map_file : int -> int -> int -> bytes -> int
  = "caml_device_amd_amdgpu_map"

external unmap_mem : int -> int -> unit = "caml_device_amd_amdgpu_unmap"
external store64 : int -> int -> unit = "caml_device_amd_amdgpu_store64"
external version : int -> int = "caml_device_amd_amdgpu_version"

external acquire_vm : int -> int -> int -> int
  = "caml_device_amd_amdgpu_acquire_vm"

external runtime_enable : int -> int = "caml_device_amd_amdgpu_runtime_enable"

external kfd_alloc : int -> int -> int -> int -> int -> bytes -> int
  = "caml_device_amd_amdgpu_alloc_byte" "caml_device_amd_amdgpu_alloc"

external kfd_free : int -> int -> int = "caml_device_amd_amdgpu_free"

external map_gpu : int -> int -> int -> bool -> int
  = "caml_device_amd_amdgpu_map_gpu"

external event : int -> int -> int -> int = "caml_device_amd_amdgpu_event"

external destroy_event : int -> int -> int
  = "caml_device_amd_amdgpu_destroy_event"

external kfd_queue : int -> int -> int array -> bytes -> int
  = "caml_device_amd_amdgpu_queue"

external destroy_queue : int -> int -> int
  = "caml_device_amd_amdgpu_destroy_queue"

external wait : int -> int array -> int -> int -> int array -> int
  = "caml_device_amd_amdgpu_wait"

external device_info : int -> int array -> int
  = "caml_device_amd_amdgpu_device_info"

external stable_pstate : int -> int = "caml_device_amd_amdgpu_stable_power"

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
  hdp : nativeint option; (* the register whose store flushes the HDP *)
  clock_khz : int;
  cus : int array; (* its active compute units, as cu_bitmap lays them out *)
  mutable doorbells : (int64 * int) option; (* the page: offset, address *)
  mutable events_mapped : bool;
  mutable stable : bool;
}

let opening = Mutex.create ()
let kfd = ref None

(* The GPUs whose address space the process acquired, by GPU id: the kernel
   driver lets a process acquire one once. *)
let acquired : (int, gpu) Hashtbl.t = Hashtbl.create 4

(* The process's event page: its handle, host address and GPU. *)
let event_page : (int * int * int) option ref = ref None

let kfd_fd () =
  match !kfd with
  | Some fd -> Ok fd
  | None ->
      let* fd =
        opened "/dev/kfd" "the compute interface" (open_file "/dev/kfd")
      in
      kfd := Some fd;
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
      if kind <> `Gpu then
        check "mapping GPU memory" (map_file g.drm at n (offset b));
      check "mapping memory for the GPU" (map_gpu fd handle g.node.gpu_id true);
      let host = if kind = `Gpu then None else Some (Nativeint.of_int at) in
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
      check "unmapping another GPU's memory"
        (map_gpu fd p.handle g.node.gpu_id false)
  | Own | Borrowed ->
      ignore (map_gpu fd p.handle g.node.gpu_id false);
      check "freeing GPU memory" (kfd_free fd p.handle);
      if p.kind = Own then unmap_mem p.at p.bytes

let map_host fd g a n : mem Amd.memory option =
  let base = Nativeint.to_int a land lnot (page - 1) in
  let bytes = round_up (Nativeint.to_int a + n - base) page in
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
      Some { Amd.address = Nativeint.to_int a; host = Some a; data }

(* Memory of another device: of the same GPU, in this address space already; of
   a GPU the topology links this one to, mapped for it. *)
let map_peer fd g (m : mem Amd.memory) =
  let o = m.data.owner in
  if o.node.gpu_id = g.node.gpu_id then
    Some { m with data = { m.data with kind = View } }
  else if not (Topology.linked "/" g.node.index o.node.index) then None
  else if map_gpu fd m.data.handle g.node.gpu_id true < 0 then None
  else Some { m with data = { m.data with kind = Peer } }

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
    if p < 0 then None else Some (Nativeint.of_int p)

(* KFD 1.14 asks a process to enable its runtime before using queues. *)
let runtime_from = 1014

let acquire fd (node : Topology.node) =
  match Hashtbl.find_opt acquired node.gpu_id with
  | Some g -> Ok g
  | None ->
      let path = strf "/dev/dri/renderD%d" node.render in
      let* drm = opened path "the GPU's render node" (open_file path) in
      let fail step e =
        close_file drm;
        Error (strf "%s: %s" step (strerror (-e)))
      in
      let v = version fd in
      let e = acquire_vm fd drm node.gpu_id in
      let e = if e = 0 && v >= runtime_from then runtime_enable fd else e in
      let cus = Array.make 16 0 in
      let khz = if e < 0 then e else device_info drm cus in
      if v < 0 then fail "reading KFD's version" v
      else if e < 0 then fail "acquiring the GPU's address space" e
      else if khz < 0 then fail "reading the GPU's facts" khz
      else
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
  let ids =
    Array.map
      (fun k ->
        let id = event fd k 0 in
        check "making a KFD event" id;
        id)
      [| signal; memory_exception; hardware_exception |]
  in
  arm ids.(0);
  ids

(* A device *)

type device = {
  fd : int;
  gpu : gpu;
  events : int array;
  mutable queues : (int * mem Amd.memory list) list; (* id, its memory *)
}

let eop_bytes = 0x1000

(* The waves the context save area holds, as the kernel driver counts them
   (kfd_queue.c, kfd_queue_ctx_save_restore_size): 32 per compute unit from
   GFX 10.1, before it 40 per compute unit up to 512 per shader engine. *)
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
  Nativeint.of_int (page_at + Int64.to_int (Int64.sub off base))

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
  let* save = gpu_mem "context save area" (save_bytes node) in
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

let sleep d ~ms =
  let r = Array.make 6 0 in
  let e = wait d.fd d.events d.gpu.node.gpu_id ms r in
  arm d.events.(0);
  match e with
  | 0 -> ()
  | 1 ->
      raise
        (Amd.Fault
           (strf
              "memory fault at 0x%x (not present %d, read-only %d, no execute \
               %d, imprecise %d, error type %d)"
              r.(0) r.(1) r.(2) r.(3) r.(4) r.(5)))
  | 2 ->
      raise
        (Amd.Fault
           (strf
              "hardware exception (reset type %d, reset cause %d, memory lost \
               %d)"
              r.(0) r.(1) r.(2)))
  | e -> fault "waiting for KFD events" e

let stable_power g () =
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

(* Each queue destroyed: once none runs, its memory goes back. *)
let stop d () =
  let destroyed (id, _) = destroy_queue d.fd id = 0 in
  if not (List.for_all destroyed d.queues) then `Unknown
  else begin
    List.iter (fun (_, mems) -> List.iter (free d.fd d.gpu) mems) d.queues;
    d.queues <- [];
    Array.iter (fun id -> ignore (destroy_event d.fd id)) d.events;
    `Stopped
  end

(* The active work-group processors of each shader array: on GFX10 on, the pairs
   of compute units both active; on GFX9, the compute units. *)
let wgps (g : gpu) =
  let n = g.node in
  let engines = n.gpu.shader_engines * n.gpu.xccs in
  let cus e a = g.cus.((4 * (e mod 4)) + a + (e / 4 * n.arrays)) in
  let pairs c =
    let m = ref 0 in
    for w = 0 to 15 do
      if (c lsr (2 * w)) land 3 = 3 then m := !m lor (1 lsl w)
    done;
    !m
  in
  let gfx9 = match n.gpu.target with 9, _, _ -> true | _ -> false in
  Array.init engines (fun e ->
      Array.init n.arrays (fun a -> if gfx9 then cus e a else pairs (cus e a)))

let key : mem Type.Id.t = Type.Id.make ()

let path d : mem Amd.path =
  let g = d.gpu and fd = d.fd in
  {
    key;
    gpu = g.node.gpu;
    lds = g.node.lds;
    clock_hz = g.clock_khz * 1000;
    mec = g.node.mec;
    wgps = wgps g;
    budget = g.node.budget;
    alloc = alloc fd g;
    map_host = map_host fd g;
    map_peer = map_peer fd g;
    free = free fd g;
    queue = queue d;
    hdp = g.hdp;
    interrupt = d.events.(0);
    sleep = sleep d;
    stable_power = stable_power g;
    stop = stop d;
  }

(* Numbering *)

let gpus_at = Topology.gpus

let gpu_at root bus =
  Result.map (fun (n : Topology.node) -> n.gpu) (Topology.node root bus)

let count () = if linux () then List.length (gpus_at "/") else 0

let device_name i =
  if i < 0 then
    invalid_argf "Device_amd_amdgpu.device_name: GPU %d is negative" i;
  if i = 0 then "AMD" else strf "AMD:%d" i

let open_ i =
  if i < 0 then invalid_argf "Device_amd_amdgpu.open_: GPU %d is negative" i;
  let gpus = if linux () then gpus_at "/" else [] in
  match List.nth_opt gpus i with
  | None ->
      Error (strf "no GPU %d; the machine has %d AMD GPUs" i (List.length gpus))
  | Some bus -> (
      let* node = Topology.node "/" bus in
      let* fd, g, events =
        Mutex.protect opening (fun () ->
            let* fd = kfd_fd () in
            let* g = acquire fd node in
            match make_events fd g with
            | events -> Ok (fd, g, events)
            | exception Amd.Fault why -> Error why)
      in
      let d = { fd; gpu = g; events; queues = [] } in
      let drop () =
        Array.iter (fun id -> ignore (destroy_event fd id)) events
      in
      match Amd.make (path d) with
      | Ok _ as r -> r
      | Error _ as e ->
          drop ();
          e
      | exception Amd.Fault why ->
          drop ();
          Error why)
