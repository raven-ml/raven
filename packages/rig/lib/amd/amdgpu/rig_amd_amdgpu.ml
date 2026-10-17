(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Sources: the Linux kernel's include/uapi/linux/kfd_ioctl.h and
   include/uapi/drm/amdgpu_drm.h for the compute interface and the render
   node's queries, and drivers/gpu/drm/amd/amdkfd for signal events and their
   slots (kfd_events.c), the sizes of a queue's context save area
   (kfd_queue.c) and the topology under
   /sys/devices/virtual/kfd/kfd/topology. *)

(* The process's KFD state (its file, each GPU's address space, the event page)
   is made under [opening]; a device's queues and events are its own. Any domain
   may call the path's functions. Its locks ([Rig_amd.Lock]) hold no other but
   [opening], which [stable_power] takes under a device's guard. *)

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let ( let* ) = Result.bind

module Amd = Rig_amd

(* The stubs raise [Unix.Unix_error] where a system call fails. *)

external open_file : string -> int = "caml_rig_amd_amdgpu_open"
external close_file : int -> unit = "caml_rig_amd_amdgpu_close"
external reserve : int -> int = "caml_rig_amd_amdgpu_reserve"

external map_file : int -> int -> int -> int64 -> int
  = "caml_rig_amd_amdgpu_map"

external unmap_mem : int -> int -> unit = "caml_rig_amd_amdgpu_unmap"
external store64 : int -> int -> unit = "caml_rig_amd_amdgpu_store64"
external clock : int -> int -> int = "caml_rig_amd_amdgpu_clock"

(* Errors *)

let message = Unix.error_message
let fault step e = raise (Amd.Fault (strf "%s: %s" step (message e)))

(* [check step f] is [f ()], raising [Amd.Fault] if [step] fails. *)
let check step f = try f () with Unix.Unix_error (e, _, _) -> fault step e

(* [quietly f] is [f ()], whose failure leaves nothing to undo. *)
let quietly f = try f () with Unix.Unix_error _ -> ()
let page = 4096
let round_up n a = (n + a - 1) / a * a

(* The kernel driver bounds no work on a user queue: it resets a queue only when
   the queue fails to stop, and a running kernel stops by saving its waves. Work
   that has not advanced for 30 s is hung, the bound of a GPU driven without a
   kernel driver. *)
let hang_ms = 30_000

let opened path what =
  match open_file path with
  | fd -> Ok fd
  | exception Unix.Unix_error (EACCES, _, _) ->
      Error
        (strf
           "opening %s: %s; %s needs read and write access, which members of \
            the render group have"
           path (message EACCES) what)
  | exception Unix.Unix_error (e, _, _) ->
      Error (strf "opening %s: %s" path (message e))

(* The process's KFD *)

type gpu = {
  node : Topology.node;
  drm : int; (* its render node *)
  hdp : int option; (* the register whose store flushes the HDP *)
  clock_khz : int;
  clock : int; (* its clock's reader, for the process *)
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

let opening = Amd.Lock.create ()

(* The process that opened the file, and the file. *)
let kfd : (int * int) option ref = ref None

(* The process's tables are immutable maps, each changed by one store of a map
   made first, so an exception leaves them whole. *)
module Ids = Map.Make (Int)

module Pairs = Map.Make (struct
  type t = int * int

  let compare = compare
end)

(* The GPUs whose address space the process acquired, by GPU id: the kernel
   driver lets a process acquire one once. *)
let acquired : gpu Ids.t ref = ref Ids.empty

(* The process's event page: its handle, host address and GPU. *)
let event_page : (int * int * int) option ref = ref None

(* Views of another GPU's memory, by its handle and the viewing GPU's id: KFD
   maps a memory once per GPU, so the first view maps it and the last view's
   free unmaps it. A count changes with its MAP or UNMAP, under [views_lock], so
   that a free never unmaps a mapping a new view counts on. *)
let views : int Pairs.t ref = ref Pairs.empty
let views_lock = Amd.Lock.create ()

(* Host memory mapped for a GPU. A process has one address space per GPU, which
   every device it opens on the GPU shares, and KFD maps a host page in it once:
   a mapping serves every device of the GPU, and ends at the last free of a
   region over it, from any device, stopped or not. *)
type registration = {
  gpu_id : int;
  start : int;
  bytes : int;
  handle : int;
  mutable maps : int;
}

let registry : registration list ref = ref []
let registry_lock = Amd.Lock.create ()

(* KFD serves a file to the process that opened it only: a forked child opens
   its own, and the parent's address spaces, event page, views and host mappings
   stay the parent's. The process's first open makes it from [dev/kfd], and
   later opens take it whatever their [dev]. *)
let kfd_fd ~dev =
  match !kfd with
  | Some (p, fd) when p = Unix.getpid () -> Ok fd
  | _ ->
      acquired := Ids.empty;
      views := Pairs.empty;
      registry := [];
      event_page := None;
      let path = Filename.concat dev "kfd" in
      let* fd = opened path "the compute interface" in
      kfd := Some (Unix.getpid (), fd);
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

let kfd_free fd handle = Request.free fd handle

(* Maps or unmaps [handle] for GPU [gpu]. Raises [Unix.Unix_error], [EIO] if
   KFD did not reach the GPU. *)
let map_gpu fd handle gpu map =
  if Request.map_gpu fd handle gpu map <> 1 then
    raise (Unix.Unix_error (EIO, "ioctl", ""))

(* Counts a view of [handle] for GPU [gpu], mapping it at the first: whether the
   GPU maps it. The counted table is made before the MAP, so that nothing
   allocates between the MAP and its count. *)
let take_view fd handle gpu =
  let key = (handle, gpu) in
  Amd.Lock.protect views_lock @@ fun () ->
  let n = Option.value ~default:0 (Pairs.find_opt key !views) in
  let counted = Pairs.add key (n + 1) !views in
  match if n = 0 then map_gpu fd handle gpu true with
  | () ->
      views := counted;
      true
  | exception Unix.Unix_error _ -> false

(* Ends a view of [handle] for GPU [gpu], unmapping it at the last. A failed
   UNMAP raises and leaves no count, so that the next view maps again. The
   count goes once the UNMAP was made, failed or not, so that an exception
   before it, such as a signal handler's as the request releases the runtime,
   leaves the count and the mapping in step. *)
let give_view fd handle gpu =
  let key = (handle, gpu) in
  Amd.Lock.protect views_lock @@ fun () ->
  match Pairs.find key !views - 1 with
  | 0 -> (
      let removed = Pairs.remove key !views in
      match map_gpu fd handle gpu false with
      | () -> views := removed
      | exception (Unix.Unix_error _ as e) ->
          views := removed;
          raise e)
  | n -> views := Pairs.add key n !views

(* Gives back the KFD memory [handle] and the [n] addresses at [at] reserved for
   it, then raises the failure of [step]. *)
let undo fd handle at n step e =
  quietly (fun () -> kfd_free fd handle);
  unmap_mem at n;
  fault step e

(* [n] bytes of [kind] at addresses reserved in the process, mapped for the GPU,
   and for the host unless they are [Device]. *)
let alloc fd g (kind : Rig_edge.memory) n : mem Amd.memory option =
  if kind = Mapped && g.node.visible = 0 then None
  else
    let n = round_up n page in
    let at = check "reserving GPU addresses" (fun () -> reserve n) in
    let request : Request.memory =
      match kind with Device -> Gpu | Mapped -> Bar | Pinned -> System
    in
    match Request.alloc fd g.node.gpu_id at n request with
    | exception Unix.Unix_error (e, _, _) ->
        unmap_mem at n;
        if e = ENOMEM || (e = EINVAL && kind = Mapped) then None
        else fault (strf "allocating %d bytes of GPU memory" n) e
    | handle, offset ->
        (try if kind <> Device then ignore (map_file g.drm at n offset : int)
         with Unix.Unix_error (e, _, _) ->
           undo fd handle at n "mapping GPU memory" e);
        (try map_gpu fd handle g.node.gpu_id true
         with Unix.Unix_error (e, _, _) ->
           undo fd handle at n "mapping memory for the GPU" e);
        let host = if kind = Device then None else Some at in
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
      check "unmapping another GPU's memory" (fun () ->
          give_view fd p.handle g.node.gpu_id)
  | Own ->
      quietly (fun () -> map_gpu fd p.handle g.node.gpu_id false);
      check "freeing GPU memory" (fun () -> kfd_free fd p.handle);
      unmap_mem p.at p.bytes
  | Borrowed e ->
      Amd.Lock.protect registry_lock @@ fun () ->
      e.maps <- e.maps - 1;
      if e.maps = 0 then begin
        registry := List.filter (fun e' -> e' != e) !registry;
        quietly (fun () -> map_gpu fd e.handle e.gpu_id false);
        check "freeing GPU memory" (fun () -> kfd_free fd e.handle)
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
  Amd.Lock.protect registry_lock @@ fun () ->
  match List.find_opt inside !registry with
  | Some e ->
      e.maps <- e.maps + 1;
      region e
  | None -> (
      match Request.alloc fd gpu_id start bytes Userptr with
      | exception Unix.Unix_error _ -> None
      | handle, _ -> (
          match map_gpu fd handle gpu_id true with
          | exception Unix.Unix_error _ ->
              quietly (fun () -> kfd_free fd handle);
              None
          | () ->
              let e = { gpu_id; start; bytes; handle; maps = 1 } in
              registry := e :: !registry;
              region e))

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
    (* Made before the view counts, so that nothing allocates after. *)
    let view = Some { m with data = { m.data with kind = Peer } } in
    if take_view fd m.data.handle g.node.gpu_id then view else None

(* Opening a GPU *)

(* The page of registers the kernel driver remaps for the process, whose first
   word flushes the host data path, or [None] if it refuses it. *)
let remap_hdp fd gpu_id =
  match reserve page with
  | exception Unix.Unix_error _ -> None
  | at -> (
      match Request.alloc fd gpu_id at page Mmio with
      | exception Unix.Unix_error _ ->
          unmap_mem at page;
          None
      | handle, offset -> (
          match map_file fd at page offset with
          | p -> Some p
          | exception Unix.Unix_error _ ->
              quietly (fun () -> kfd_free fd handle);
              unmap_mem at page;
              None))

(* KFD 1.14 asks a process to enable its runtime before using queues, and
   refuses a second enable as busy. *)
let runtime_from = 1014

let runtime_enable fd =
  try Request.runtime_enable fd with Unix.Unix_error (EBUSY, _, _) -> ()

(* A machine's AMD GPUs in bus order, read once, with the topology node of each
   the kernel driver holds. A GPU the driver did not hold at the last look is
   looked at again at each use: a process may start while a driver-less session
   holds it detached, and open it once the driver holds it again. A node once
   found is kept, as the process's KFD state is tied to it. *)
type machine = {
  root : string;
  gpus : (string * (Topology.node, string) result) array;
  look : Amd.Lock.t;
}

let machine_at root =
  let node bus = (bus, Topology.node root bus) in
  let* buses = Topology.gpus root in
  Ok
    {
      root;
      gpus = Array.of_list (List.map node buses);
      look = Amd.Lock.create ();
    }

let nodes m =
  Amd.Lock.protect m.look @@ fun () ->
  Array.iteri
    (fun i (bus, node) ->
      if Result.is_error node then m.gpus.(i) <- (bus, Topology.node m.root bus))
    m.gpus;
  Array.copy m.gpus

let gpus_of m =
  Amd.Lock.check "Rig_amd_amdgpu.gpus_of";
  Array.to_list
    (Array.map
       (fun (bus, node) ->
         (bus, Result.map (fun (n : Topology.node) -> n.gpu) node))
       (nodes m))

(* The machine as each root shows it, read at the root's first use that reads
   it: a read that fails is not kept, so the next use reads again. *)
let machines : machine list ref = ref []
let machines_lock = Amd.Lock.create ()

let topology root =
  let* m =
    Amd.Lock.protect machines_lock @@ fun () ->
    match List.find_opt (fun m -> m.root = root) !machines with
    | Some m -> Ok m
    | None ->
        let* m = machine_at root in
        machines := m :: !machines;
        Ok m
  in
  Ok (nodes m)

(* [gpus] is the machine's topology, as [topology] gave it. *)
let acquire fd ~root ~gpus (node : Topology.node) =
  match Ids.find_opt node.gpu_id !acquired with
  | Some g -> Ok g
  | None ->
      let path = Filename.concat root (strf "dev/dri/renderD%d" node.render) in
      let* drm = opened path "the GPU's render node" in
      let ok step f =
        match f () with
        | v -> Ok v
        | exception Unix.Unix_error (e, _, _) ->
            close_file drm;
            Error (strf "%s: %s" step (message e))
      in
      let acquiring = "acquiring the GPU's address space" in
      let* version = ok "reading KFD's version" (fun () -> Request.version fd) in
      let* () =
        ok acquiring (fun () -> Request.acquire_vm fd drm node.gpu_id)
      in
      let* () =
        if version < runtime_from then Ok ()
        else ok acquiring (fun () -> runtime_enable fd)
      in
      let* clock_khz, cus =
        ok "reading the GPU's facts" (fun () -> Request.device_info drm)
      in
      let links =
        Array.to_list gpus
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
          clock = clock fd node.gpu_id;
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
      let ok = Ok g and table = Ids.add node.gpu_id g !acquired in
      acquired := table;
      ok

(* Events. On an interrupt from the GPU, the kernel driver sets a signal event
   only if the event's slot of the event page holds a value other than all ones,
   and puts all ones back. The host arms the slot, a signal event's id, when the
   event is made and after each wait, before the caller reads the timeline word
   again: an interrupt then finds the slot armed, or comes between a wait and
   its arm, before a read that sees the word the work wrote. *)

let event_page_bytes = 0x8000

(* A new event of kind [k] on the event page [page]: its id. *)
let event fd k ~page = Request.event fd k page
let destroy_event fd id = Request.destroy_event fd id

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
        check "mapping the event page" (fun () ->
            map_gpu fd handle g.node.gpu_id true);
      g.events_mapped <- true
  | None -> (
      match alloc fd g Pinned event_page_bytes with
      | None -> raise (Amd.Fault "no memory for the event page")
      | Some m ->
          check "making the event page" (fun () ->
              ignore (event fd Request.Signal ~page:m.data.handle : int));
          event_page := Some (m.data.handle, m.address, g.node.gpu_id);
          g.events_mapped <- true)

let make_events fd g =
  page_for fd g;
  let made = ref [] in
  let make k =
    match event fd k ~page:0 with
    | id ->
        made := id :: !made;
        id
    | exception Unix.Unix_error (e, _, _) ->
        List.iter (fun id -> quietly (fun () -> destroy_event fd id)) !made;
        fault "making a KFD event" e
  in
  let ids = Array.map make [| Request.Signal; Memory; Hardware |] in
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
    Amd.Lock.protect opening (fun () ->
        d.gpu.live <- List.filter (fun e -> e != d.events) d.gpu.live);
    Array.iter (fun id -> quietly (fun () -> destroy_event d.fd id)) d.events
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
        let at =
          check "mapping the doorbell page" (fun () ->
              map_file d.fd 0 0x2000 base)
        in
        g.doorbells <- Some (base, at);
        at
  in
  page_at + Int64.to_int (Int64.sub off base)

let queue d kind ~ring ~bytes ~read ~write =
  let compute = kind <> `Sdma in
  let gpu_mem what n =
    if not compute then Ok None
    else
      match alloc d.fd d.gpu Device n with
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
  let kind : Request.queue =
    match kind with `Pm4 -> Pm4 | `Aql -> Aql | `Sdma -> Sdma
  in
  match
    Request.queue d.fd kind node.gpu_id ring bytes (address eop)
      (if compute then eop_bytes else 0)
      (address save)
      (if compute then node.cwsr else 0)
      (if compute then node.ctl_stack else 0)
      write read
  with
  | exception Unix.Unix_error (e, _, _) ->
      List.iter (free d.fd d.gpu) taken;
      Error (strf "making a KFD queue: %s" (message e))
  | id, doorbell_offset ->
      d.queues <- (id, taken) :: d.queues;
      Ok (Amd.Lock.protect opening (fun () -> doorbell d doorbell_offset))

(* Records the fault [why]. rig refuses the GPU for the rest of the process
   once its work faulted, since by default (amdgpu's
   no_queue_eviction_on_vm_fault=0) the kernel driver schedules none of the
   process's queues on it again, those made after the fault too. The GPU keeps
   the first fault for every later sleep and open. *)
let report g why = ignore (Atomic.compare_and_set g.faulted None (Some why))

let raise_fault g =
  match Atomic.get g.faulted with
  | Some why -> raise (Amd.Fault why)
  | None -> ()

let sleep d ~ms =
  let g = d.gpu in
  raise_fault g;
  let f =
    match Request.wait d.fd d.events ms g.node.gpu_id with
    | f -> f
    | exception Unix.Unix_error (e, _, _) ->
        arm d.events.(0);
        fault "waiting for KFD events" e
  in
  arm d.events.(0);
  match f with
  | None -> ()
  | Some f ->
      report g (Request.fault_message f);
      raise_fault g

(* Asks the exception events of the GPU's open devices, without waiting, for a
   fault no sleep has read yet. The events stay set: they reset only by hand. *)
let poll_faults fd g =
  List.iter
    (fun ev ->
      if Atomic.get g.faulted = None then
        match Request.wait fd [| ev.(1); ev.(2) |] 0 g.node.gpu_id with
        | Some f -> report g (Request.fault_message f)
        | None | (exception Unix.Unix_error _) -> ())
    g.live

let stable_power g () =
  Amd.Lock.protect opening @@ fun () ->
  match g.node.gpu.target with
  | 9, _, _ -> Ok ()
  | _ when g.stable -> Ok ()
  | _ -> (
      match Request.stable_pstate g.drm with
      | () ->
          g.stable <- true;
          Ok ()
      | exception Unix.Unix_error (EBUSY, _, _) ->
          Error "another process holds the GPU's stable power state"
      | exception Unix.Unix_error (e, _, _) ->
          Error
            (strf "the amdgpu driver refused the GPU's stable power state: %s"
               (message e)))

(* Each queue destroyed, and the memory of those destroyed given back. A queue
   the kernel driver kept may still run and raise the device's interrupt, so its
   memory and the events stay. The kernel driver keeps nothing of a fault for
   later opens. *)
let stop d ~fault:_ =
  let gone, kept =
    List.partition
      (fun (id, _) ->
        match Request.destroy_queue d.fd id with
        | () -> true
        | exception Unix.Unix_error _ -> false)
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

let path d ~gpus ~index : mem Amd.path =
  let g = d.gpu and fd = d.fd in
  {
    key;
    index;
    gpu = g.node.gpu;
    waves = g.node.waves_per_cu;
    lds = g.node.lds;
    clock_hz = g.clock_khz * 1000;
    clock = Source g.clock;
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
           gpus
       in
       fun j -> j >= 0 && j < Array.length reached && reached.(j));
    map_peer = map_peer fd g;
    free = free fd g;
    queue = queue d;
    hdp = g.hdp;
    interrupt = d.events.(0);
    hang_ms = Some hang_ms;
    sleep = sleep d;
    stable_power = stable_power g;
    stop = stop d;
  }

(* Numbering *)

let buses ?(root = "/") () = Result.value (Topology.gpus root) ~default:[]

let gpu_at root bus =
  Result.map (fun (n : Topology.node) -> n.gpu) (Topology.node root bus)

let save_area_at root bus = Result.map save_bytes (Topology.node root bus)
let count ?root () = List.length (buses ?root ())

let device_name i =
  if i < 0 then invalid_argf "Rig_amd_amdgpu.gpu: GPU %d is negative" i;
  if i = 0 then "AMD" else strf "AMD:%d" i

let open_ ?(root = "/") ~word i =
  Amd.Lock.check "Rig_amd_amdgpu.gpu";
  if i < 0 then invalid_argf "Rig_amd_amdgpu.gpu: GPU %d is negative" i;
  let* gpus =
    Result.map_error
      (fun why -> "reading the machine's GPUs: " ^ why)
      (topology root)
  in
  if i >= Array.length gpus then
    Error (strf "no GPU %d; the machine has %d AMD GPUs" i (Array.length gpus))
  else
    let* node = snd gpus.(i) in
    let* fd, g, events =
      Amd.Lock.protect opening (fun () ->
          let* fd = kfd_fd ~dev:(Filename.concat root "dev") in
          let* g = acquire fd ~root ~gpus node in
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
    match Amd.make ~word (path d ~gpus ~index:i) with
    | Ok _ as r ->
        Amd.Lock.protect opening (fun () -> g.live <- events :: g.live);
        r
    | Error _ as e ->
        give_back ();
        e
    | exception Amd.Fault why ->
        give_back ();
        Error why

let gpu ?root i : Rig_amd.t Rig_edge.opener =
  {
    name = device_name i;
    driver = (module Rig_amd);
    make = (fun ~word -> open_ ?root ~word i);
  }
