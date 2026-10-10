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

external ioctl : int -> int -> Request.params -> int
  = "caml_rig_amd_amdgpu_ioctl"

external map_file : int -> int -> int -> int64 -> int
  = "caml_rig_amd_amdgpu_map"

external unmap_mem : int -> int -> unit = "caml_rig_amd_amdgpu_unmap"
external store64 : int -> int -> unit = "caml_rig_amd_amdgpu_store64"
external pid : unit -> int = "caml_rig_amd_amdgpu_pid"

(* Makes the request [r] of the file [fd]: 0, or -errno. *)
let call fd (r : Request.t) = ioctl fd r.number r.params

(* Makes the request [r] of [fd] and gives [r] back: 0, or -errno. *)
let call_once fd r =
  let e = call fd r in
  Request.give r;
  e

(* Errors *)

let eacces = -13
let ebusy = -16
let enomem = -12
let einval = -22
let eio = -5
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
   maps a memory once per GPU, so the first view maps it and the last view's
   free unmaps it. A count changes with its MAP or UNMAP, under [views_lock],
   so that a free never unmaps a mapping a new view counts on. *)
let views : (int * int, int) Hashtbl.t = Hashtbl.create 16
let views_lock = Mutex.create ()

(* Host memory mapped for a GPU. A process has one address space per GPU,
   which every device it opens on the GPU shares, and KFD maps a host page in
   it once: a mapping serves every device of the GPU, and ends at the last free
   of a region over it, from any device, stopped or not. *)
type registration = {
  gpu_id : int;
  start : int;
  bytes : int;
  handle : int;
  mutable maps : int;
}

let registry : registration list ref = ref []
let registry_lock = Mutex.create ()

(* KFD serves a file to the process that opened it only: a forked child opens
   its own, and the parent's address spaces, event page, views and host
   mappings stay the parent's. The process's first open makes it from
   [dev/kfd], and later opens take it whatever their [dev]. *)
let kfd_fd ~dev =
  match !kfd with
  | Some (p, fd) when p = pid () -> Ok fd
  | _ ->
      Hashtbl.reset acquired;
      Hashtbl.reset views;
      registry := [];
      event_page := None;
      let path = Filename.concat dev "kfd" in
      let* fd = opened path "the compute interface" (open_file path) in
      kfd := Some (pid (), fd);
      Ok fd

(* Memory *)

type kind = Own | Borrowed of registration | Peer | View

type mem = {
  handle : int;
  bytes : int;
  at : int; (* the process's addresses, unmapped at the free *)
  kind : kind;
  owner : gpu; (* the GPU it was allocated for *)
}

(* Allocates [bytes] bytes of [m] for GPU [gpu] at [va] with the request [r]: 0,
   with the answer in [r], or -errno. *)
let kfd_alloc fd r ~gpu ~va ~bytes m =
  Request.alloc r ~gpu ~va ~bytes m;
  call fd r

let kfd_free fd handle =
  let r = Request.take () in
  Request.free r handle;
  call_once fd r

(* Maps or unmaps [handle] for GPU [gpu]: 0, or -errno, -EIO if KFD did not
   reach the GPU. *)
let map_gpu fd handle gpu map =
  let r = Request.take () in
  if map then Request.map r ~gpu handle else Request.unmap r ~gpu handle;
  let e = call fd r in
  let reached = Request.mapped r in
  Request.give r;
  if e = 0 && reached <> 1 then eio else e

(* Counts a view of [handle] for GPU [gpu], mapping it at the first: whether
   the GPU maps it. *)
let take_view fd handle gpu =
  let key = (handle, gpu) in
  Mutex.protect views_lock @@ fun () ->
  let n = Option.value ~default:0 (Hashtbl.find_opt views key) in
  if n = 0 && map_gpu fd handle gpu true < 0 then false
  else begin
    Hashtbl.replace views key (n + 1);
    true
  end

(* Ends a view of [handle] for GPU [gpu], unmapping it at the last: 0, or
   UNMAP's -errno. A failed UNMAP leaves no count, so that the next view maps
   again. *)
let give_view fd handle gpu =
  let key = (handle, gpu) in
  Mutex.protect views_lock @@ fun () ->
  match Hashtbl.find views key - 1 with
  | 0 ->
      Hashtbl.remove views key;
      map_gpu fd handle gpu false
  | n ->
      Hashtbl.replace views key n;
      0

(* Gives back the KFD memory [handle] and the [n] addresses at [at] reserved for
   it, then raises the failure of [step]. *)
let undo fd handle at n step e =
  ignore (kfd_free fd handle);
  unmap_mem at n;
  fault step e

(* [n] bytes of [kind] at addresses reserved in the process, mapped for the GPU,
   and for the host unless they are [`Gpu]. *)
let alloc fd g (kind : [ `Gpu | `Bar | `System ]) n : mem Amd.memory option =
  if kind = `Bar && g.node.visible = 0 then None
  else
    let n = round_up n page in
    let at = reserve n in
    check "reserving GPU addresses" at;
    let r = Request.take () in
    let e =
      kfd_alloc fd r ~gpu:g.node.gpu_id ~va:at ~bytes:n (kind :> Request.memory)
    in
    let handle = Request.handle r in
    let offset = if kind = `Gpu then 0L else Request.mmap_offset r in
    Request.give r;
    if e = enomem || (e = einval && kind = `Bar) then begin
      unmap_mem at n;
      None
    end
    else if e < 0 then begin
      unmap_mem at n;
      fault (strf "allocating %d bytes of GPU memory" n) e
    end
    else
      let e = if kind = `Gpu then 0 else map_file g.drm at n offset in
      if e < 0 then undo fd handle at n "mapping GPU memory" e;
      let e = map_gpu fd handle g.node.gpu_id true in
      if e < 0 then undo fd handle at n "mapping memory for the GPU" e;
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
      check "unmapping another GPU's memory"
        (give_view fd p.handle g.node.gpu_id)
  | Own ->
      ignore (map_gpu fd p.handle g.node.gpu_id false);
      check "freeing GPU memory" (kfd_free fd p.handle);
      unmap_mem p.at p.bytes
  | Borrowed e ->
      Mutex.protect registry_lock @@ fun () ->
      e.maps <- e.maps - 1;
      if e.maps = 0 then begin
        registry := List.filter (fun e' -> e' != e) !registry;
        ignore (map_gpu fd e.handle e.gpu_id false);
        check "freeing GPU memory" (kfd_free fd e.handle)
      end

(* Pages within a mapping of the GPU share it; pages that only partly overlap
   one KFD refuses. *)
let map_host fd g a n : mem Amd.memory option =
  let start = a land lnot (page - 1) in
  let bytes = round_up (a + n - start) page in
  let gpu_id = g.node.gpu_id in
  let inside e =
    e.gpu_id = gpu_id && e.start <= start && start + bytes <= e.start + e.bytes
  in
  let region (e : registration) =
    let data =
      { handle = e.handle; bytes; at = start; kind = Borrowed e; owner = g }
    in
    Some { Amd.address = a; host = Some a; data }
  in
  Mutex.protect registry_lock @@ fun () ->
  match List.find_opt inside !registry with
  | Some e ->
      e.maps <- e.maps + 1;
      region e
  | None ->
      let r = Request.take () in
      let e = kfd_alloc fd r ~gpu:gpu_id ~va:start ~bytes `Userptr in
      let handle = Request.handle r in
      Request.give r;
      if e < 0 then None
      else if map_gpu fd handle gpu_id true < 0 then begin
        ignore (kfd_free fd handle);
        None
      end
      else
        let e = { gpu_id; start; bytes; handle; maps = 1 } in
        registry := e :: !registry;
        region e

(* Memory of another device: of the same GPU, in this address space already; of
   a GPU the topology links this one to, mapped for it. *)
let linked g (n : Topology.node) =
  n.gpu_id = g.node.gpu_id || List.mem n.index g.links

let map_peer fd g (m : mem Amd.memory) =
  let o = m.data.owner in
  if o.node.gpu_id = g.node.gpu_id then
    Some { m with data = { m.data with kind = View } }
  else if not (linked g o.node) then None
  else if take_view fd m.data.handle g.node.gpu_id then
    Some { m with data = { m.data with kind = Peer } }
  else None

(* Opening a GPU *)

(* The page of registers the kernel driver remaps for the process, whose first
   word flushes the host data path, or [None] if it refuses it. *)
let remap_hdp fd gpu_id =
  let at = reserve page in
  if at < 0 then None
  else
    let r = Request.take () in
    let e = kfd_alloc fd r ~gpu:gpu_id ~va:at ~bytes:page `Mmio in
    let handle = Request.handle r and offset = Request.mmap_offset r in
    Request.give r;
    if e < 0 then begin
      unmap_mem at page;
      None
    end
    else
      let p = map_file fd at page offset in
      if p >= 0 then Some p
      else begin
        ignore (kfd_free fd handle);
        unmap_mem at page;
        None
      end

(* KFD 1.14 asks a process to enable its runtime before using queues, and
   refuses a second enable as busy. *)
let runtime_from = 1014

let runtime_enable fd =
  let r = Request.take () in
  Request.runtime_enable r;
  match call_once fd r with e when e = ebusy -> 0 | e -> e

(* A machine's AMD GPUs in bus order, read once, with the topology node of each
   the kernel driver holds. A GPU the driver did not hold at the last look is
   looked at again at each use: a process may start while a driver-less session
   holds it detached, and open it once the driver holds it again. A node once
   found is kept, as the process's KFD state is tied to it. *)
type machine = {
  root : string;
  gpus : (string * (Topology.node, string) result) array;
  look : Mutex.t;
}

let machine_at root =
  let node bus = (bus, Topology.node root bus) in
  {
    root;
    gpus = Array.of_list (List.map node (Topology.gpus root));
    look = Mutex.create ();
  }

let nodes m =
  Mutex.protect m.look @@ fun () ->
  Array.iteri
    (fun i (bus, node) ->
      if Result.is_error node then m.gpus.(i) <- (bus, Topology.node m.root bus))
    m.gpus;
  Array.copy m.gpus

let gpus_of m =
  Array.to_list
    (Array.map
       (fun (bus, node) ->
         (bus, Result.map (fun (n : Topology.node) -> n.gpu) node))
       (nodes m))

(* The machine as each root shows it, read at the root's first use. *)
let machines : machine list ref = ref []
let machines_lock = Mutex.create ()

let topology root =
  let m =
    Mutex.protect machines_lock @@ fun () ->
    match List.find_opt (fun m -> m.root = root) !machines with
    | Some m -> m
    | None ->
        let m =
          if linux () then machine_at root
          else { root; gpus = [||]; look = Mutex.create () }
        in
        machines := m :: !machines;
        m
  in
  nodes m

let acquire fd ~root (node : Topology.node) =
  match Hashtbl.find_opt acquired node.gpu_id with
  | Some g -> Ok g
  | None ->
      let path = Filename.concat root (strf "dev/dri/renderD%d" node.render) in
      let* drm = opened path "the GPU's render node" (open_file path) in
      let ok step e =
        if e >= 0 then Ok e
        else begin
          close_file drm;
          Error (strf "%s: %s" step (strerror (-e)))
        end
      in
      let acquiring = "acquiring the GPU's address space" in
      let r = Request.take () in
      let facts =
        Request.version r;
        let* _ = ok "reading KFD's version" (call fd r) in
        let version = Request.version_of r in
        Request.acquire_vm r ~drm ~gpu:node.gpu_id;
        let* _ = ok acquiring (call fd r) in
        let* _ =
          if version < runtime_from then Ok 0
          else ok acquiring (runtime_enable fd)
        in
        Request.device_info r;
        let* _ = ok "reading the GPU's facts" (call drm r) in
        Ok (Request.clock_khz r, Request.compute_units r)
      in
      Request.give r;
      let* clock_khz, cus = facts in
      let links =
        Array.to_list (topology root)
        |> List.filter_map (function
          | _, Ok (n : Topology.node)
            when Topology.linked root node.index n.index ->
              Some n.index
          | _ -> None)
      in
      let g =
        {
          node;
          drm;
          hdp = remap_hdp fd node.gpu_id;
          clock_khz;
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

(* A new event of kind [k] on the event page [page]: its id, or -errno. *)
let event fd k ~page =
  let r = Request.take () in
  Request.event r k ~page;
  let e = call fd r in
  let id = Request.event_id r in
  Request.give r;
  if e < 0 then e else id

let destroy_event fd id =
  let r = Request.take () in
  Request.destroy_event r id;
  call_once fd r

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
          check "making the event page" (event fd `Signal ~page:m.data.handle);
          event_page := Some (m.data.handle, m.address, g.node.gpu_id);
          g.events_mapped <- true)

let make_events fd g =
  page_for fd g;
  let made = ref [] in
  let make k =
    let id = event fd k ~page:0 in
    if id < 0 then begin
      List.iter (fun id -> ignore (destroy_event fd id)) !made;
      fault "making a KFD event" id
    end;
    made := id :: !made;
    id
  in
  let ids = Array.map make [| `Signal; `Memory; `Hardware |] in
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

(* The doorbell at [off] in the process's doorbell page of the GPU, which KFD
   maps from the page's own offset. *)
let doorbell d off =
  let g = d.gpu in
  let base = Int64.logand off (Int64.lognot 0x1fffL) in
  let page_at =
    match g.doorbells with
    | Some (_, at) -> at
    | None ->
        let at = map_file d.fd 0 0x2000 base in
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
  let r = Request.take () in
  Request.queue r kind ~gpu:node.gpu_id ~ring ~ring_bytes:bytes
    ~eop:(address eop)
    ~eop_bytes:(if compute then eop_bytes else 0)
    ~save:(address save)
    ~save_bytes:(if compute then node.cwsr else 0)
    ~ctl_stack:(if compute then node.ctl_stack else 0)
    ~write ~read;
  let e = call d.fd r in
  let id = Request.queue_id r and doorbell_offset = Request.doorbell_offset r in
  Request.give r;
  if e < 0 then begin
    List.iter (free d.fd d.gpu) taken;
    Error (strf "making a KFD queue: %s" (strerror (-e)))
  end
  else begin
    d.queues <- (id, taken) :: d.queues;
    Ok (Mutex.protect opening (fun () -> doorbell d doorbell_offset))
  end

(* Resets the exception event [id] of GPU [gpu] unless [gpu] is [me] or none:
   set by another GPU's fault, it does not reset itself, and would end every
   later wait at once. *)
let reset_other fd ~me id gpu =
  if gpu <> 0 && gpu <> me then begin
    let r = Request.take () in
    Request.reset_event r id;
    ignore (call_once fd r)
  end

(* Waits at most [ms] for the events [ids] ({!Request.wait}): the fault one of
   their exception events reports of GPU [g], if any, or -errno. *)
let wait fd ids g ms =
  let r = Request.take () in
  Request.wait r ids ~ms;
  let e = call fd r in
  let me = g.node.gpu_id in
  let memory = Request.exception_gpu r `Memory in
  let hardware = Request.exception_gpu r `Hardware in
  let why =
    if e < 0 then None
    else if memory = me then Some (Request.fault r `Memory)
    else if hardware = me then Some (Request.fault r `Hardware)
    else None
  in
  let memory_id = Request.exception_id r `Memory in
  let hardware_id = Request.exception_id r `Hardware in
  Request.give r;
  if e < 0 then Error e
  else begin
    reset_other fd ~me memory_id memory;
    reset_other fd ~me hardware_id hardware;
    (* A constant [Ok None]: a wait that reports no fault allocates nothing. *)
    match why with
    | None -> Ok None
    | Some _ -> Ok why
  end

(* Records the fault [why]. A fault leaves the process's queues on the GPU
   unscheduled, those made after it too, so the GPU keeps the first for every
   later sleep and open. *)
let report g why = ignore (Atomic.compare_and_set g.faulted None (Some why))

let raise_fault g =
  match Atomic.get g.faulted with
  | Some why -> raise (Amd.Fault why)
  | None -> ()

let sleep d ~ms =
  let g = d.gpu in
  raise_fault g;
  let r = wait d.fd d.events g ms in
  arm d.events.(0);
  match r with
  | Error e -> fault "waiting for KFD events" e
  | Ok None -> ()
  | Ok (Some why) ->
      report g why;
      raise_fault g

(* Asks the exception events of the GPU's open devices, without waiting, for a
   fault no sleep has read yet. The events stay set: they reset only by hand. *)
let poll_faults fd g =
  List.iter
    (fun ev ->
      if Atomic.get g.faulted = None then
        match wait fd [| ev.(1); ev.(2) |] g 0 with
        | Ok (Some why) -> report g why
        | Ok None | Error _ -> ())
    g.live

(* Holds the GPU in its stable power state with a new context on the render node
   [drm], until the process closes it: 0, or -errno. *)
let stable_pstate drm =
  let r = Request.take () in
  Request.alloc_context r;
  let e = call drm r in
  let e =
    if e < 0 then e
    else
      let id = Request.context r in
      Request.stable_pstate r id;
      let e = call drm r in
      if e < 0 then begin
        Request.free_context r id;
        ignore (call drm r)
      end;
      e
  in
  Request.give r;
  e

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
   memory and the events stay. The kernel driver keeps nothing of a fault for
   later opens. *)
let stop d ~fault:_ =
  let gone, kept =
    List.partition
      (fun (id, _) ->
        let r = Request.take () in
        Request.destroy_queue r id;
        call_once d.fd r = 0)
      d.queues
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

let path d ~root ~index : mem Amd.path =
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
    map_host = Some (map_host fd g);
    reaches =
      (let reached =
         Array.mapi
           (fun j (_, n) ->
             j = index || match n with Ok n -> linked g n | Error _ -> false)
           (topology root)
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

let buses ?(root = "/") () = Topology.gpus root

let gpu_at root bus =
  Result.map (fun (n : Topology.node) -> n.gpu) (Topology.node root bus)

let save_area_at root bus = Result.map save_bytes (Topology.node root bus)
let count ?root () = List.length (buses ?root ())

let device_name i =
  if i < 0 then invalid_argf "Rig_amd_amdgpu.device_name: GPU %d is negative" i;
  if i = 0 then "AMD" else strf "AMD:%d" i

let open_ ?(root = "/") i =
  if i < 0 then invalid_argf "Rig_amd_amdgpu.open_: GPU %d is negative" i;
  let gpus = topology root in
  if i >= Array.length gpus then
    Error (strf "no GPU %d; the machine has %d AMD GPUs" i (Array.length gpus))
  else
    let* node = snd gpus.(i) in
    let* fd, g, events =
      Mutex.protect opening (fun () ->
          let* fd = kfd_fd ~dev:(Filename.concat root "dev") in
          let* g = acquire fd ~root node in
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
    match Amd.make (path d ~root ~index:i) with
    | Ok _ as r ->
        Mutex.protect opening (fun () -> g.live <- events :: g.live);
        r
    | Error _ as e ->
        give_back ();
        e
    | exception Amd.Fault why ->
        give_back ();
        Error why
