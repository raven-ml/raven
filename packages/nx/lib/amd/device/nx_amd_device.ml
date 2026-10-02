(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Amd_defs
module Mmio = Nx_device_support.Mmio
module Pci = Nx_device_support.Pci
module Pci_memory = Nx_device_support.Pci_memory
module Page_table = Nx_device_support.Page_table
module Remote = Nx_device_support.Remote
module Driver = Nx_device.Driver
module Region = Driver.Region

external linux : unit -> bool = "caml_nx_amd_linux"

type interface = Kernel | Pci

type queue = {
  ring : Nx_device.Buffer.t;
  read_ptr : Nx_device.Buffer.t;
  write_ptr : Nx_device.Buffer.t;
  put : Nx_device.Buffer.t;
  doorbell : Nx_device.Buffer.t;
}

(* A queue's words, as the host and the GPU address them: the ring and its size,
   and the 64-bit words. *)
type words = {
  ring : nativeint * int;
  read_ptr : nativeint;
  write_ptr : nativeint;
  put : nativeint;
  doorbell : nativeint;
}

type props = {
  target : int * int * int;
  gc : int * int * int;
  sdma : int * int * int;
  nbio : int * int * int;
  xccs : int;
  shader_engines : int;
  compute_units : int;
  waves_per_cu : int;
  lds_bytes : int;
  scratch_slots_per_cu : int;
}

type kernel = {
  code : Nx_device.Buffer.t;
  descriptor : nativeint;
  private_segment : int;
}

(* A kernel of a loaded code object, by its descriptor's address: its name and
   scratch bytes per lane, and the load it belongs to, whose unload removes it
   and nothing a later load put at the same address. *)
type entry = { name : string; scratch : int; image : unit ref }

type counter = {
  name : string;
  block : string;
  event : int;
  register : int;
  instances : int;
  engines : int;
  arrays : int;
  wgps : int;
  offset : int;
}

type counting = {
  samples : Nx_device.Buffer.t;
  counters : counter list;
  size : int;
  wgp_active : engine:int -> array:int -> wgp:int -> bool;
}

type tracing = {
  traces : Nx_device.Buffer.t;
  ends : Nx_device.Buffer.t;
  window : int;
  engines : int;
}

type profiling = {
  slots : int;
  log : Nx_device.Buffer.t;
  counting : counting option;
  tracing : tracing option;
}

type u64s = (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t
type u32s = (int32, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t

(* A device's trace buffers, with the host's views of them. Every profile that
   traces writes them, each run in its log's slot: a profile is taken at a time
   and its stop reads its runs, so the runs of two logs never share a slot
   unread. *)
type traces = {
  tracing : tracing;
  host : Nx_device.Buffer.t; (* the host's borrow of the traces *)
  end_words : u32s;
}

(* The profiling of a profile's counters and traces, with the host's views of
   its buffers, and the runs of the log read so far. *)
type profile = {
  profiling : profiling;
  log_words : u64s;
  sample_words : u64s;
  traces : traces option;
  mutable read : int;
}

(* The GPU behind a device, through its interface. *)
type gpu =
  | Kfd_gpu of Kfd.t
  | Am_gpu of { am : Am.t; memory : Pci_memory.t; pci : Pci.t }

type mem = Kfd_mem of Kfd.mem | Am_mem of Pci_memory.memory

module Int_map = Map.Make (Int)

type t = {
  index : int;
  machine : Nx_device.t; (* the host of the GPU's machine *)
  gpu : gpu;
  hw : Mutex.t; (* the GPU's registers and page tables *)
  mutable allocs : mem Int_map.t; (* the device's own memory, by address *)
  borrows : (int, mem) Hashtbl.t; (* host memory mapped for borrows *)
  reach : (int, bool) Hashtbl.t; (* whether it reaches a peer, by index *)
  kernels : (nativeint, entry) Hashtbl.t; (* by descriptor address, under hw *)
  props : props;
  cu_per_array : int; (* the compute units of a shader array *)
  scratch_lock : Mutex.t;
  mutable scratch : (Nx_device.Buffer.t * int) option;
  profile_lock : Mutex.t;
  profiles : (string list * bool, profile) Hashtbl.t;
      (* by the counters and traces asked for: work encoded for them writes
         there *)
  mutable traces : traces option; (* made by the first profile that traces *)
  mutable aql_desc : Mmio.t option; (* the AQL queue's descriptor *)
  mutable queues : (queue * bool * queue list) option;
      (* the compute queue, whether it takes AQL packets, the SDMA queues *)
  mutable dev : Nx_device.t option;
}

let va = function Kfd_mem m -> m.va | Am_mem m -> m.mapping.va
let size = function Kfd_mem m -> m.size | Am_mem m -> m.mapping.size
let host_view = function Kfd_mem m -> m.host | Am_mem m -> m.host
let with_hw a f = Mutex.protect a.hw f
let round_up n a = (n + a - 1) / a * a

(* Memory *)

type kind = Vram | Visible | Host | Uncached

let alloc_mem a kind n =
  with_hw a (fun () ->
      match a.gpu with
      | Kfd_gpu k ->
          let kind =
            match kind with
            | Vram -> Kfd.Vram
            | Visible -> Kfd.Visible
            | Host -> Kfd.Host
            | Uncached -> Kfd.Uncached
          in
          Option.map (fun m -> Kfd_mem m) (Kfd.alloc k kind n)
      | Am_gpu g ->
          let m =
            match kind with
            | Vram -> Pci_memory.alloc g.memory n
            | Visible -> Pci_memory.alloc ~cpu_access:true g.memory n
            | Host | Uncached ->
                Pci_memory.alloc ~host:true ~uncached:true g.memory n
          in
          Option.map (fun m -> Am_mem m) m)

(* Unmaps another GPU's memory [mem] from [a]. *)
let unmap_from a mem =
  with_hw a (fun () ->
      match (a.gpu, mem) with
      | Kfd_gpu k, Kfd_mem m -> Kfd.unmap_peer k m
      | Am_gpu g, Am_mem m -> Pci_memory.unmap g.memory { m with source = Peer }
      | _ -> invalid_arg "a peer of another interface")

let free_mem a mem =
  with_hw a (fun () ->
      match (a.gpu, mem) with
      | Kfd_gpu k, Kfd_mem m -> Kfd.free k m
      | Am_gpu g, Am_mem m -> Pci_memory.free g.memory m
      | _ -> invalid_arg "memory of another interface")

let register a mem = a.allocs <- Int_map.add (va mem) mem a.allocs

(* Frees [mem], which no peer maps any more: the runtime unmaps them first. *)
let release a mem =
  a.allocs <- Int_map.remove (va mem) a.allocs;
  free_mem a mem

(* The allocation of [a] that holds the address [x]. *)
let find a x =
  match Int_map.find_last_opt (fun s -> s <= x) a.allocs with
  | Some (s, mem) when x < s + size mem -> Some mem
  | _ -> None

(* The [n] bytes of [mem] from its start, at the same address for the host and
   the GPU when the host addresses them. *)
let region_of mem n =
  let va = Nativeint.of_int (va mem) in
  Region.v ?host:(Option.map Mmio.address (host_view mem)) ~handle:va va n

let allocator a kind =
  let alloc n =
    Option.map
      (fun mem ->
        register a mem;
        region_of mem n)
      (alloc_mem a kind n)
  in
  let free r =
    Option.iter (release a)
      (Int_map.find_opt (Nativeint.to_int (Region.handle r)) a.allocs)
  in
  { Driver.alloc; free }

(* Borrows: host memory the GPU addresses at its own address. *)
let mapping a =
  let map x n =
    with_hw a (fun () ->
        let r =
          match a.gpu with
          | Kfd_gpu k -> Result.map (fun m -> Kfd_mem m) (Kfd.map_host k x n)
          | Am_gpu g ->
              Result.map (fun m -> Am_mem m) (Pci_memory.map_host g.memory x n)
        in
        Result.map
          (fun mem ->
            Hashtbl.replace a.borrows (va mem) mem;
            Region.v ~host:x ~handle:x x n)
          r)
  in
  let unmap r =
    let x = Nativeint.to_int (Region.handle r) in
    match Hashtbl.find_opt a.borrows x with
    | Some mem ->
        Hashtbl.remove a.borrows x;
        with_hw a (fun () ->
            match (a.gpu, mem) with
            | Kfd_gpu k, Kfd_mem m -> Kfd.unmap_host k m
            | Am_gpu g, Am_mem m -> Pci_memory.unmap g.memory m
            | _ -> ())
    | None -> ()
  in
  Driver.Pages { map; unmap }

(* Queues *)

(* The range at [x] of the GPU's machine. *)
let range a x n =
  match a.gpu with
  | Am_gpu { pci; _ } -> (
      match Pci.remote pci with
      | Some r -> Mmio.remote (Remote.access r) x n
      | None -> Mmio.v x n)
  | Kfd_gpu _ -> Mmio.v x n

let sdma_queue a (q : queue) =
  let word b = range a (Nx_device.Buffer.address b) 8 in
  {
    Sdma.ring =
      range a (Nx_device.Buffer.address q.ring) (Nx_device.Buffer.nbytes q.ring);
    read_ptr = word q.read_ptr;
    write_ptr = word q.write_ptr;
    put = word q.put;
    doorbell = word q.doorbell;
  }

let sdma_family (props : props) =
  Am_reg.family "sdma_pkt"
    (if compare props.sdma (6, 0, 0) < 0 then props.sdma else (6, 0, 0))

(* The largest linear copy the engine takes at once. *)
let max_copy (props : props) =
  let v = props.sdma in
  if
    (compare (4, 4, 2) v <= 0 && compare v (5, 0, 0) < 0)
    || compare v (5, 2, 0) >= 0
  then 0x40000000
  else 0x400000

(* The opened GPUs, by the host of their machine and bus address there. *)
let opened : ((Nx_device.t * string) * t) list Atomic.t = Atomic.make []

let amd_of d =
  List.find_map
    (fun (_, a) ->
      match a.dev with Some d' when Nx_device.equal d d' -> Some a | _ -> None)
    (Atomic.get opened)

(* Whether [a]'s copy engine reaches the memory of [peer], which the topology
   fixes: over a link the driver reports, or through [peer]'s memory BAR when it
   is large, as mapping [peer]'s memory requires, on the same machine. *)
let reaches a peer =
  with_hw a (fun () ->
      match Hashtbl.find_opt a.reach peer.index with
      | Some r -> r
      | None ->
          let r =
            match (a.gpu, peer.gpu) with
            | Kfd_gpu k, Kfd_gpu k' -> Kfd.reaches k.node k'.node
            | Am_gpu _, Am_gpu g' ->
                a.machine == peer.machine
                && not (Pci_memory.small_bar g'.memory)
            | _ -> false
          in
          Hashtbl.replace a.reach peer.index r;
          r)

(* Borrows of and copies into another AMD GPU's memory that [a] reaches. *)
let peer a d' r =
  match amd_of d' with
  | None -> Error "memory of another vendor"
  | Some peer when not (reaches a peer) ->
      Error "the GPUs do not reach each other's memory"
  | Some peer -> (
      match find peer (Nativeint.to_int (Region.address r)) with
      | None -> Error "no memory of the other GPU"
      | Some mem ->
          let mapped =
            with_hw a (fun () ->
                match (a.gpu, peer.gpu, mem) with
                | Kfd_gpu k, _, Kfd_mem m -> (
                    match Kfd.map_peer k m with
                    | () -> Ok ()
                    | exception Failure why -> Error why)
                | Am_gpu g, Am_gpu g', Am_mem m ->
                    Result.map ignore (Pci_memory.map_peer g.memory g'.memory m)
                | _ -> Error "a peer of another interface")
          in
          Result.map (fun () -> (r, fun () -> unmap_from a mem)) mapped)

let flush_hdp a =
  match a.gpu with
  | Kfd_gpu k -> Kfd.flush_hdp k
  | Am_gpu g -> with_hw a (fun () -> Am.flush_hdp g.am.d)

let queue a ~timeline =
  let signal = Nativeint.to_int (Region.address timeline) in
  let props = a.props in
  let family = sdma_family props and max = max_copy props in
  let enqueue words =
    let q =
      match a.queues with
      | Some (_, _, sdma) -> sdma_queue a (List.hd sdma)
      | None -> failwith "the SDMA queue is not set up"
    in
    let timeout_ms =
      Option.fold ~none:Driver.default_timeout ~some:Nx_device.timeout a.dev
    in
    (* The copy engine reads what the host wrote to mapped memory. *)
    flush_hdp a;
    Sdma.submit q ~timeout_ms words
  in
  let submit ~dst ~src n ~signal:v =
    enqueue
      (Sdma.packets ~family ~max ~signal ~dst:(Nativeint.to_int dst)
         ~src:(Nativeint.to_int src) n v)
  in
  let stamp ~slot ~signal:v =
    enqueue (Sdma.stamp ~family ~signal ~slot:(Nativeint.to_int slot) v)
  in
  (* A transfer writes the destination through this GPU's mapping of it. *)
  let transfer d' =
    match amd_of d' with
    | Some peer when reaches a peer -> Some submit
    | _ -> None
  in
  {
    Driver.copy = submit;
    transfer;
    stamp;
    clock = Device_clock { hz = 100_000_000 };
  }

(* How the other functions of the GPU's machine reach its memory: its own
   through the memory BAR, even in a hive whose GPUs reach each other over XGMI,
   and system memory at its pages. *)
let dma a r =
  match (a.gpu, find a (Nativeint.to_int (Region.address r))) with
  | Kfd_gpu _, _ -> Error "the kernel driver's memory is not described"
  | Am_gpu _, None -> Error "no allocation of this GPU"
  | Am_gpu { memory; pci; _ }, Some (Am_mem pm) -> (
      let map = pm.mapping in
      match map.space with
      | Page_table.Sys -> Ok { Driver.bus = Pci.bus pci; pages = map.pages }
      | Phys when Pci_memory.small_bar memory ->
          Error "the memory BAR is too small for other functions to reach it"
      | Phys ->
          let start = fst (Pci.bar pci 0) in
          Ok
            {
              Driver.bus = Pci.bus pci;
              pages = List.map (fun (p, n) -> (p + start, n)) map.pages;
            }
      | Peer -> Error "memory of another GPU")
  | Am_gpu _, Some (Kfd_mem _) -> Error "memory of another interface"

(* Programs *)

(* The most memory the host can map for code: the part of the GPU's memory it
   maps under the kernel driver; under PCI the memory BAR, unless it is small
   and memory the host maps is system memory. *)
let code_window a =
  match a.gpu with
  | Kfd_gpu k -> k.visible
  | Am_gpu g ->
      if Pci_memory.small_bar g.memory then max_int else snd (Pci.bar g.pci 0)

(* The code object [binary], relocated and uploaded to memory of the device the
   host writes, which it frees once unloaded. A code object the device cannot
   run, or larger than that memory could ever hold, is refused, and the device
   stays usable. *)
let load a ~binary =
  match Code_object.image binary with
  | exception Failure why -> Error why
  | obj, img -> (
      let bytes = String.length img in
      match alloc_mem a Visible bytes with
      | None when bytes > code_window a ->
          Error
            (Printf.sprintf
               "%d bytes of code, more than the %d the host can map (enable \
                Resizable BAR in the firmware settings)"
               bytes (code_window a))
      | None -> raise (Nx_device.Out_of_memory (Option.get a.dev, bytes))
      | Some mem ->
          register a mem;
          Mmio.write (Option.get (host_view mem)) 0 img;
          Mmio.barrier ();
          let image = ref () and found = ref [] in
          let entry name =
            match
              Code_object.kernel obj img ~name
                ~lds_kib:(a.props.lds_bytes / 1024)
            with
            | exception Failure why -> Error why
            | k ->
                let descriptor = Nativeint.of_int (va mem + k.descriptor) in
                with_hw a (fun () ->
                    Hashtbl.replace a.kernels descriptor
                      { name; scratch = k.private_segment; image });
                found := descriptor :: !found;
                Ok descriptor
          in
          let unload () =
            with_hw a (fun () ->
                List.iter
                  (fun d ->
                    match Hashtbl.find_opt a.kernels d with
                    | Some e when e.image == image -> Hashtbl.remove a.kernels d
                    | Some _ | None -> ())
                  !found);
            release a mem
          in
          Ok { Driver.code = Some (region_of mem bytes); entry; unload })

(* Scratch *)

let scratch_gpu (p : props) =
  {
    Scratch.gc = p.gc;
    compute_units = p.compute_units;
    slots = p.scratch_slots_per_cu;
    shader_engines = p.shader_engines;
    xccs = p.xccs;
  }

(* Points the AQL descriptor's scratch at [b], for [n] bytes per lane. *)
let aql_scratch a desc b n =
  let g = scratch_gpu a.props in
  let base = Nativeint.to_int (Nx_device.Buffer.address b) in
  let open D.Amd_queue in
  Mmio.set64 desc (fst scratch_backing_memory_location) (Int64.of_int base);
  Mmio.set32 desc (fst scratch_wave64_lane_byte_size) n;
  List.iteri
    (fun i w -> Mmio.set32 desc (fst scratch_resource_descriptor + (4 * i)) w)
    (Scratch.descriptor g ~base (Nx_device.Buffer.nbytes b));
  Mmio.set32 desc (fst compute_tmpring_size) (Scratch.tmpring_size g n)

let scratch a n =
  let d = Option.get a.dev in
  Mutex.protect a.scratch_lock (fun () ->
      let n = Int.max n 128 in
      match a.scratch with
      | Some (b, have) when have >= n -> b
      | _ ->
          let b =
            Nx_device.Buffer.create d Nx_dtype.Scalar.UInt8
              (Scratch.bytes (scratch_gpu a.props) n)
          in
          Option.iter (fun desc -> aql_scratch a desc b n) a.aql_desc;
          a.scratch <- Some (b, n);
          b)

(* Opening *)

(* GPU [i]'s name through [iface]: ["AMD:i"], and ["AMD-PCI:i"] for GPUs taken
   from their kernel driver, here or on another machine, so that the name says
   which interface reaches the GPU. *)
let name iface i =
  let kind = match iface with Kernel -> "AMD" | Pci -> "AMD-PCI" in
  if i = 0 then kind else Printf.sprintf "%s:%d" kind i

let gpu_name a =
  let iface = match a.gpu with Kfd_gpu _ -> Kernel | Am_gpu _ -> Pci in
  name iface a.index

let target_of v =
  let v = if v = 90403 then 90402 else v in
  (v / 10000, v / 100 mod 100, v mod 100)

let arch (a, b, c) = Printf.sprintf "gfx%d%x%x" a b c

let supported ((major, _, _) as t) =
  if not (List.mem t [ (9, 4, 2); (9, 5, 0) ] || major = 11 || major = 12) then
    failwith (Printf.sprintf "%s GPUs are not supported" (arch t))

let props ~target ~ip ~xccs ~shader_engines ~compute_units ~waves_per_cu
    ~lds_kib ~slots =
  let v hwip = Option.value ~default:(0, 0, 0) (List.assoc_opt hwip ip) in
  {
    target;
    gc = v D.gc_hwip;
    sdma = v D.sdma0_hwip;
    nbio = v D.nbif_hwip;
    xccs;
    shader_engines;
    compute_units;
    waves_per_cu;
    lds_bytes = lds_kib * 1024;
    scratch_slots_per_cu = slots;
  }

(* The waves a GPU runs at once, which its context save area holds. *)
let wave_count (p : props) =
  let major, _, _ = p.target in
  if major <> 9 then p.compute_units * p.waves_per_cu
  else Int.min (p.compute_units * 40) (p.shader_engines * p.xccs * 512)

let page = 4096

type spec = {
  ring_bytes : int;
  eop_bytes : int;
  save : int;
  ctl_stack : int;
  debug : int;
}

(* The sizes of the compute queue's context save area, as the kernel driver
   computes them (kfd_queue.c). *)
let compute_spec a ~saves =
  let p = a.props in
  let major, minor, _ = p.target in
  let waves = wave_count p in
  let lds = if (major, minor) = (9, 5) then p.lds_bytes else 0x10000 in
  let vgpr =
    if
      List.mem p.target
        [ (11, 0, 0); (11, 0, 1); (11, 5, 1); (12, 0, 0); (12, 0, 1) ]
    then 0x60000
    else if major = 9 then 0x80000
    else 0x40000
  in
  let wg_data =
    round_up ((vgpr + 0x4000 + lds + 0x1000) * p.compute_units) page
  in
  let ctl_stack =
    round_up (((if major <> 9 then 12 else 8) * waves) + 8 + 40) page
  in
  {
    ring_bytes = 16 lsl 20;
    eop_bytes = 0x1000;
    save = (if saves then wg_data + ctl_stack else 0);
    ctl_stack;
    debug = round_up (waves * 32) 64;
  }

let sdma_spec =
  { ring_bytes = 16 lsl 20; eop_bytes = 0; save = 0; ctl_stack = 0; debug = 0 }

let need what = function
  | Some m -> m
  | None -> failwith ("no GPU memory for the " ^ what)

let addr mem = Nativeint.of_int (va mem)

(* Creates a hardware queue of [kind] (a KFD queue type) with its memory. *)
let create_queue a ~kind spec ~idx =
  let p = a.props in
  let ring = need "ring" (alloc_mem a Uncached spec.ring_bytes) in
  let gart = need "queue pointers" (alloc_mem a Uncached 0x100) in
  let gart_view = Mmio.sub (Option.get (host_view gart)) 0 0x100 in
  Mmio.fill gart_view 0 0x100 '\000';
  let aql = kind = D.kfd_ioc_queue_type_compute_aql in
  if aql then begin
    let open D.Amd_queue in
    Mmio.set32 gart_view (fst queue_properties)
      (D.amd_queue_properties_is_ptr64
     lor D.amd_queue_properties_enable_profiling);
    Mmio.set32 gart_view
      (fst read_dispatch_id_field_base_byte_offset)
      (fst read_dispatch_id);
    Mmio.set32 gart_view (fst max_cu_id) ((p.compute_units * p.xccs) - 1);
    Mmio.set32 gart_view (fst max_wave_id) (p.waves_per_cu - 1)
  end;
  let eop =
    if spec.eop_bytes > 0 then
      Some (need "EOP buffer" (alloc_mem a Vram spec.eop_bytes))
    else None
  in
  let save_bytes =
    if spec.save > 0 then round_up ((spec.save + spec.debug) * p.xccs) page
    else 0
  in
  let save =
    if save_bytes > 0 then
      Some (need "context save area" (alloc_mem a Vram save_bytes))
    else None
  in
  let rptr =
    Nativeint.add (addr gart)
      (Nativeint.of_int (fst D.Amd_queue.read_dispatch_id))
  in
  let wptr =
    Nativeint.add (addr gart)
      (Nativeint.of_int (fst D.Amd_queue.write_dispatch_id))
  in
  let opt = function Some m -> addr m | None -> 0n in
  let doorbell =
    match a.gpu with
    | Kfd_gpu k ->
        Kfd.create_queue k
          {
            Kfd.ring = addr ring;
            ring_bytes = spec.ring_bytes;
            gpu = 0;
            kind;
            eop = opt eop;
            eop_bytes = spec.eop_bytes;
            cwsr = opt save;
            cwsr_bytes = spec.save;
            ctl_stack_bytes = spec.ctl_stack;
            wptr;
            rptr;
          }
    | Am_gpu g ->
        let index =
          with_hw a (fun () ->
              if kind = D.kfd_ioc_queue_type_sdma then
                Am.setup_sdma_ring g.am ~ring:(va ring)
                  ~ring_bytes:spec.ring_bytes ~rptr:(Nativeint.to_int rptr)
                  ~wptr:(Nativeint.to_int wptr) ~idx
              else
                Am.setup_compute_ring g.am ~ring:(va ring)
                  ~ring_bytes:spec.ring_bytes ~rptr:(Nativeint.to_int rptr)
                  ~wptr:(Nativeint.to_int wptr)
                  ~eop:(va (Option.get eop))
                  ~eop_bytes:spec.eop_bytes ~idx ~aql)
        in
        Nativeint.add
          (Mmio.address g.am.d.doorbells)
          (Nativeint.of_int (index * 8))
  in
  let put = need "put word" (alloc_mem a Host 8) in
  Mmio.set64 (Option.get (host_view put)) 0 0L;
  let q =
    {
      ring = (addr ring, spec.ring_bytes);
      read_ptr = rptr;
      write_ptr = wptr;
      put = addr put;
      doorbell;
    }
  in
  (q, if aql then Some gart_view else None)

(* Whether each queue has room for half its ring, the most a submission writes
   into it. Positions count dwords on a PM4 queue, 64-byte packets on an AQL
   queue and bytes on an SDMA queue. *)
let room a () =
  match a.queues with
  | None -> true
  | Some (compute, aql, sdma) ->
      let half unit (q : queue) =
        let word b =
          Int64.to_int (Mmio.get64 (range a (Nx_device.Buffer.address b) 8) 0)
        in
        let ring = Nx_device.Buffer.nbytes q.ring in
        2 * ((word q.put - word q.read_ptr) * unit land (ring - 1)) <= ring
      in
      half (if aql then 64 else 4) compute && List.for_all (half 1) sdma

(* Profiling *)

(* The runs whose counters a device keeps until it reads them. *)
let count_slots = 32

(* The bytes of a shader engine's traces over the runs a device keeps, as
   tinygrad keeps them. *)
let trace_bytes = 256 lsl 20

let counters a names =
  let p = a.props in
  let major, _, _ = p.target in
  let table =
    D.counters
      (if major = 9 then arch p.target else Printf.sprintf "gfx%d" major)
  in
  let registers = Hashtbl.create 4 and offset = ref 0 in
  let counter name =
    match List.find_opt (fun (n, _, _) -> n = name) table with
    | None ->
        invalid_arg
          (Printf.sprintf "%s counts no %s; it counts %s" (arch p.target) name
             (String.concat ", " (List.map (fun (n, _, _) -> n) table)))
    | Some (_, block, event) ->
        let register =
          Option.value ~default:0 (Hashtbl.find_opt registers block)
        in
        Hashtbl.replace registers block (register + 1);
        let instances, engines, arrays, wgps =
          match block with
          | "GRBM" -> (1, 1, 1, 1)
          | "GL2C" -> (32, 1, 1, 1)
          | "TCC" -> (16, 1, 1, 1)
          | _ when major = 9 -> (1, p.shader_engines, 1, 1)
          | _ -> (1, p.shader_engines, 2, a.cu_per_array / 2)
        in
        let c =
          {
            name;
            block;
            event;
            register;
            instances;
            engines;
            arrays;
            wgps;
            offset = !offset;
          }
        in
        offset := !offset + (p.xccs * instances * engines * arrays * wgps * 8);
        c
  in
  List.map counter names

let values (p : props) c = p.xccs * c.instances * c.engines * c.arrays * c.wgps

(* Under the amdgpu driver, counts and traces are only stable in the GPU's
   stable power state, which a GFX9 GPU does not need. *)
let check_power a =
  match a.gpu with
  | Kfd_gpu k when match a.props.target with 9, _, _ -> false | _ -> true ->
      let level =
        try
          String.trim
            (In_channel.with_open_text
               (k.sysfs ^ "/power_dpm_force_performance_level")
               In_channel.input_all)
        with Sys_error _ -> "unknown"
      in
      if level <> "profile_standard" then
        failwith
          (Printf.sprintf
             "%s: profiling needs the GPU's stable power state, not %s: run \
              `amd-smi set -l stable_std`"
             (gpu_name a) level)
  | Kfd_gpu _ | Am_gpu _ -> ()

let wgp_active a =
  match a.gpu with
  | Am_gpu _ -> fun ~engine:_ ~array:_ ~wgp:_ -> true
  | Kfd_gpu k ->
      let bitmap = Kfd.cu_bitmap k in
      fun ~engine ~array ~wgp ->
        (bitmap.(engine mod 4).(array + (engine / 4 * 2)) lsr (2 * wgp)) land 3
        = 3

(* A run's entry in the log: its kernel descriptor's address, then when the run
   started and when it stopped, on the GPU's clock. *)
let entry_words = 3

(* [n] elements of [kind] in [d]'s [memory], and the host's view of them. *)
let host_words d memory scalar kind n =
  let b = Nx_device.Buffer.create ~memory d scalar n in
  match Nx_device.Buffer.borrow Nx_device.host b with
  | Error why -> failwith why
  | Ok h -> (b, Nx_device.Buffer.bigarray kind h)

let empty kind = Bigarray.Array1.create kind Bigarray.c_layout 0

(* The device's trace buffers, made once: a window of each shader engine per
   slot. *)
let traces a d =
  match a.traces with
  | Some t -> t
  | None ->
      let engines = a.props.shader_engines * a.props.xccs
      and window = trace_bytes / count_slots in
      let traces =
        Nx_device.Buffer.create ~memory:Mapped d UInt8
          (window * count_slots * engines)
      in
      let host =
        match Nx_device.Buffer.borrow Nx_device.host traces with
        | Ok h -> h
        | Error why -> failwith why
      in
      let ends, end_words =
        host_words d Pinned Int32 Bigarray.int32 (count_slots * engines)
      in
      Bigarray.Array1.fill end_words 0l;
      let t =
        { tracing = { traces; ends; window; engines }; host; end_words }
      in
      a.traces <- Some t;
      t

let profiling a =
  let asked = (Nx_device.Profile.counters (), Nx_device.Profile.traced ()) in
  match asked with
  | [], false -> None
  | names, trace ->
      Mutex.protect a.profile_lock (fun () ->
          match Hashtbl.find_opt a.profiles asked with
          | Some p -> Some p.profiling
          | None ->
              let counters = counters a names in
              check_power a;
              let d = Option.get a.dev in
              let log, log_words =
                host_words d Pinned UInt64 Bigarray.int64
                  (1 + (entry_words * count_slots))
              in
              Bigarray.Array1.fill log_words 0L;
              let counting, sample_words =
                match counters with
                | [] -> (None, empty Bigarray.int64)
                | _ ->
                    let size =
                      List.fold_left
                        (fun n c -> n + (values a.props c * 8))
                        0 counters
                    in
                    let samples, words =
                      host_words d Pinned UInt64 Bigarray.int64
                        (count_slots * size / 8)
                    in
                    Bigarray.Array1.fill words 0L;
                    ( Some { samples; counters; size; wgp_active = wgp_active a },
                      words )
              in
              let traces = if trace then Some (traces a d) else None in
              let tracing = Option.map (fun t -> t.tracing) traces in
              let profiling = { slots = count_slots; log; counting; tracing } in
              Hashtbl.replace a.profiles asked
                { profiling; log_words; sample_words; traces; read = 0 };
              Some profiling)

(* The values of the counters of the run in [slot]. *)
let counted a p (c : counting) slot =
  let base = slot * c.size / 8 in
  List.map
    (fun ct ->
      ( ct.name,
        Array.init (values a.props ct) (fun j ->
            Int64.to_int p.sample_words.{base + (ct.offset / 8) + j}) ))
    c.counters

(* The trace of shader engine [se] of the run in [slot]: the bytes up to the
   engine's write pointer, which counts 32-byte units from the trace's start, or
   from address 0 on GFX 11.0. A GFX9 trace starts with a header its GPU does
   not write. *)
let traced a (s : traces) ~slot ~se =
  let t = s.tracing in
  let off = ((se * count_slots) + slot) * t.window in
  let units w = w land 0x1FFF_FFFF * 32 in
  let wptr = units (Int32.to_int s.end_words.{(slot * t.engines) + se}) in
  let wptr =
    match a.props.target with
    | 11, 0, _ ->
        let start =
          Nativeint.to_int (Nx_device.Buffer.address t.traces) + off
        in
        wptr - units (start / 32)
    | _ -> wptr
  in
  if wptr < 0 || wptr > t.window then
    failwith
      (Printf.sprintf "%s: the trace of shader engine %d ends at %d of %d bytes"
         (gpu_name a) se wptr t.window);
  let data =
    Mmio.read
      (Mmio.v
         (Nx_device.Buffer.address s.host)
         (Nx_device.Buffer.nbytes s.host))
      off wptr
  in
  match a.props.target with
  | 9, _, _ ->
      let header = Bytes.create 8 in
      Bytes.set_int64_le header 0
        (Int64.of_int (0x11 lor (4 lsl 13) lor (0xf lsl 16) lor (se lsl 24)));
      Bytes.to_string header ^ data
  | _ -> data

(* The spans of the waves of a trace, on the GPU's clock, each on a lane of its
   shader engine, compute unit, SIMD and slot. *)
let wave_spans device ~name ~se data =
  match Thread_trace.clock data with
  | None -> []
  | Some clock ->
      List.map
        (fun (w : Thread_trace.wave) ->
          Nx_device.Profile.Span
            {
              device;
              lane =
                Printf.sprintf "SE %d CU %d SIMD %d wave %d" se w.cu w.simd
                  w.slot;
              name;
              start = clock w.start;
              stop = clock w.stop;
            })
        (Thread_trace.waves data)

(* The counters and traces of the runs a log took since its last report, timed
   on the GPU's clock, and the runs it took over before they were read. *)
let read_runs a (p : profile) =
  let device = Option.get a.dev in
  let n = Int64.to_int p.log_words.{0} and slots = p.profiling.slots in
  let first = Int.max p.read (n - slots) in
  let lost =
    if first > p.read then
      [
        Nx_device.Profile.Overwritten
          { device; time = Nx_device.Profile.now (); runs = first - p.read };
      ]
    else []
  in
  let run k =
    let slot = k mod slots in
    let word i = Int64.to_int p.log_words.{1 + (entry_words * slot) + i} in
    let handle = Nativeint.of_int (word 0) in
    let name =
      match with_hw a (fun () -> Hashtbl.find_opt a.kernels handle) with
      | Some e -> e.name
      | None -> Printf.sprintf "0x%nx" handle
    in
    let start = word 1 and stop = word 2 in
    let counters =
      match p.profiling.counting with
      | None -> []
      | Some c ->
          [
            Nx_device.Profile.Counters
              { device; name; start; stop; counters = counted a p c slot };
          ]
    in
    let traces =
      match p.traces with
      | None -> []
      | Some s ->
          List.concat
            (List.init s.tracing.engines (fun se ->
                 let data = traced a s ~slot ~se in
                 Nx_device.Profile.Trace
                   { device; name; start; stop; part = se; data }
                 :: wave_spans device ~name ~se data))
    in
    counters @ traces
  in
  let events =
    List.concat (List.init (Int.max 0 (n - first)) (fun i -> run (first + i)))
  in
  p.read <- n;
  lost @ events

let report a () =
  Mutex.protect a.profile_lock (fun () ->
      Hashtbl.fold (fun _ p events -> events @ read_runs a p) a.profiles [])

let make_device a ~budget ~sleep ?finalize () =
  let dev =
    Driver.device ~name:(gpu_name a) ~arch:(arch a.props.target) ~host:a.machine
      ~budget
      ~completion:(Sleep (fun ~timeline:_ -> sleep))
      ~load:(load a) ~peer:(peer a)
      ~reaches:(fun d' ->
        match amd_of d' with Some peer -> reaches a peer | None -> false)
      ~dma:(dma a) ~room:(room a) ~report:(report a) ?finalize
      (Device_local
         {
           memory = allocator a Vram;
           host_memory = allocator a Host;
           mapped =
             (match a.gpu with
             | Am_gpu g when Pci_memory.small_bar g.memory -> None
             | Kfd_gpu k when not (Kfd.flushes_hdp k) -> None
             | _ -> Some (allocator a Visible, code_window a));
           mapping = mapping a;
           queue = queue a;
         })
  in
  a.dev <- Some dev;
  dev

let setup a ~saves ~sdma_queues =
  let aql = a.props.xccs > 1 in
  let compute, desc =
    create_queue a
      ~kind:
        (if aql then D.kfd_ioc_queue_type_compute_aql
         else D.kfd_ioc_queue_type_compute)
      (compute_spec a ~saves) ~idx:0
  in
  let sdma =
    List.init sdma_queues (fun idx ->
        fst (create_queue a ~kind:D.kfd_ioc_queue_type_sdma sdma_spec ~idx))
  in
  a.aql_desc <- desc;
  (compute, aql, sdma)

let record ~machine ~index ~gpu ~props ~cu_per_array =
  {
    index;
    machine;
    gpu;
    hw = Mutex.create ();
    allocs = Int_map.empty;
    borrows = Hashtbl.create 16;
    reach = Hashtbl.create 4;
    kernels = Hashtbl.create 16;
    props;
    cu_per_array;
    scratch_lock = Mutex.create ();
    scratch = None;
    profile_lock = Mutex.create ();
    profiles = Hashtbl.create 2;
    traces = None;
    aql_desc = None;
    queues = None;
    dev = None;
  }

(* The words of a queue, as buffers of [dev] at addresses the host and the GPU
   share. *)
let queue_buffers dev (w : words) =
  let buffer (x, n) =
    Driver.buffer dev (Region.v ~host:x x n) Nx_dtype.Scalar.UInt8 n
  in
  let word x =
    Nx_device.Buffer.view (buffer (x, 8)) ~offset:0 Nx_dtype.Scalar.UInt64 1
  in
  ({
     ring = buffer w.ring;
     read_ptr = word w.read_ptr;
     write_ptr = word w.write_ptr;
     put = word w.put;
     doorbell = word w.doorbell;
   }
    : queue)

let finish a dev (compute, aql, sdma) =
  let q = queue_buffers dev in
  a.queues <- Some (q compute, aql, List.map q sdma)

let open_kfd ~index bus =
  let k = Kfd.open_gpu bus in
  let pr = Kfd.prop k in
  let target = target_of (pr "gfx_target_version") in
  supported target;
  let xccs = Option.value ~default:1 (List.assoc_opt "num_xcc" k.props) in
  let props =
    props ~target ~ip:k.ip_ver ~xccs
      ~shader_engines:(pr "array_count" / pr "simd_arrays_per_engine" / xccs)
      ~compute_units:(pr "simd_count" / pr "simd_per_cu" / xccs)
      ~waves_per_cu:(pr "max_waves_per_simd" * pr "simd_per_cu")
      ~lds_kib:(pr "lds_size_in_kb")
      ~slots:(pr "max_slots_scratch_cu")
  in
  let a =
    record ~machine:Nx_device.host ~index ~gpu:(Kfd_gpu k) ~props
      ~cu_per_array:(pr "cu_per_simd_array")
  in
  let compute, aql, sdma = setup a ~saves:true ~sdma_queues:1 in
  let dev = make_device a ~budget:k.vram ~sleep:(fun ms -> Kfd.sleep k ms) () in
  finish a dev (compute, aql, sdma);
  a

(* Opens the device of [am], a GPU booted over [pci]. *)
let open_booted ~machine ~buses ~index pci (am : Am.t) =
  let d = am.d in
  let peer =
    if Amdev.is_hive d then
      Some
        (fun ranges ->
          ( List.map (fun (p, n) -> (Amdev.paddr2xgmi d p, n)) ranges,
            Page_table.Peer ))
    else None
  in
  let memory = Pci_memory.create ?peer pci am.mm ~bar:0 in
  let g = d.gc_info in
  let a, b, c = Amdev.version d D.gc_hwip in
  let props =
    props
      ~target:(target_of ((a * 10000) + (b * 100) + c))
      ~ip:d.ip_ver ~xccs:d.xccs ~shader_engines:g.num_se
      ~compute_units:(g.cu_per_sa * g.sh_per_se * g.num_se)
      ~waves_per_cu:(g.max_waves_per_simd * 2) ~lds_kib:g.lds_size
      ~slots:g.max_scratch_slots_per_cu
  in
  supported props.target;
  let a =
    record ~machine ~index
      ~gpu:(Am_gpu { am; memory; pci })
      ~props ~cu_per_array:g.cu_per_sa
  in
  let compute, aql, sdma =
    setup a ~saves:false ~sdma_queues:(if d.is_vf then Int.min buses 8 else 1)
  in
  (* A PF resets a VF that holds its access for long: every queue is set up, so
     the access goes back. *)
  if d.is_vf then Amdev.release_vf_access d;
  let dev =
    make_device a ~budget:(Page_table.memory am.mm)
      ~sleep:(fun ms -> with_hw a (fun () -> Am.sleep am ms))
      ~finalize:(fun ~failed -> with_hw a (fun () -> Am.fini am ~failed))
      ()
  in
  finish a dev (compute, aql, sdma);
  a

(* Opens the GPU of [pci], which the process took. A GPU booted by an open that
   then fails is stopped as a failed device is at exit, and its next boot is a
   full one. *)
let open_taken ?firmware ~machine ~buses ~index pci =
  Pci.reserve pci
    ~base:(Page_table.Space.base Am.space)
    (Page_table.Space.length Am.space);
  let am = Am.boot ?firmware pci in
  match open_booted ~machine ~buses ~index pci am with
  | a -> a
  | exception e ->
      (try Am.fini am ~failed:true with Failure _ -> ());
      raise e

(* A failed open gives the function back, so that a later one can take it. *)
let open_am ?firmware ~machine ~index bus =
  let remote = Nx_remote_device.remote machine in
  let supported = Am.buses ?remote () in
  if not (List.mem bus supported) then
    failwith (bus ^ " is of no GPU family the PCI interface supports");
  (match remote with
  | Some _ -> ()
  | None -> (
      match Pci.detached bus with
      | Ok () -> ()
      | Error why ->
          failwith
            (Printf.sprintf "%s; Nx_amd_device.detach %d detaches it" why index)
      ));
  let pci = Pci.take ?remote ~lock:"am" bus in
  match
    open_taken ?firmware ~machine ~buses:(List.length supported) ~index pci
  with
  | a -> a
  | exception e ->
      Pci.release pci;
      raise e

(* The interface of the process, fixed by its first successful open. *)
let chosen = ref None
let lock = Mutex.create ()

(* Raises [Lost] for [host] if its machine can no longer be reached: the
   synchronization of [host] meets the failed connection, which loses it. *)
let check_reach host =
  match Nx_remote_device.remote host with
  | Some r when Remote.failed r <> None -> Nx_device.synchronize host
  | Some _ | None -> ()

(* The machine's AMD GPUs, its display controllers and processing accelerators
   (such as the Instinct GPUs) of AMD's functions, in bus order: GPU [i] is the
   [i]th under both interfaces, whatever driver holds it. *)
let gpus ?remote () =
  let scan class_ = Pci.scan ?remote ~vendor:0x1002 ~class_ [ (0, [ 0 ]) ] in
  List.merge Pci.compare_address (scan 0x03) (scan 0x12)

(* The GPUs of the machine of [host]. *)
let gpus_of host =
  match Nx_remote_device.remote host with
  | Some remote -> (
      try gpus ~remote ()
      with Failure _ as e ->
        check_reach host;
        raise e)
  | None when not (linux ()) -> []
  | None -> gpus ()

let count ?(host = Nx_device.host) () = List.length (gpus_of host)
let interface_name = function Kernel -> "the kernel driver" | Pci -> "PCI"

(* Why GPU [i] of the machine of [machine] cannot be opened. *)
let refuse ~machine iface i why =
  check_reach machine;
  Error (Driver.name ~host:machine (name iface i) ^ ": " ^ why)

(* Opens [i] through [iface] on the machine of [machine], once: an open is keyed
   by the GPU's bus address. *)
let open_gpu ~machine ~iface ?firmware i =
  let gpus = gpus_of machine in
  match List.nth_opt gpus i with
  | None ->
      refuse ~machine iface i
        (Printf.sprintf "no GPU %d; there are %d AMD GPUs" i (List.length gpus))
  | Some bus -> (
      match List.assoc_opt (machine, bus) (Atomic.get opened) with
      | Some a -> Ok (Option.get a.dev)
      | None -> (
          match
            match iface with
            | Kernel -> open_kfd ~index:i bus
            | Pci -> open_am ?firmware ~machine ~index:i bus
          with
          | a ->
              if machine == Nx_device.host then chosen := Some iface;
              Atomic.set opened (((machine, bus), a) :: Atomic.get opened);
              Ok (Option.get a.dev)
          | exception (Failure why | Sys_error why | Invalid_argument why) ->
              refuse ~machine iface i why
          | exception Am.Booted bus ->
              refuse ~machine iface i
                (Printf.sprintf
                   "%s was booted by its kernel driver or another process; \
                    Nx_amd_device.reset %d resets it"
                   bus i)
          | exception Amdev.Missing_firmware path ->
              refuse ~machine iface i
                (Printf.sprintf
                   "no firmware image %s in %s/lib/firmware or the cache; \
                    Nx_amd_device.fetch_firmware %d fetches it"
                   path
                   (match firmware with Some dir -> dir ^ ", " | None -> "")
                   i)
          | exception Unix.Unix_error (e, fn, arg) ->
              refuse ~machine iface i
                (Printf.sprintf "%s %s: %s" fn arg (Unix.error_message e))
          | exception Not_found ->
              refuse ~machine iface i "opening failed: Not_found"))

let get ?(host = Nx_device.host) ~interface ?firmware i =
  if i < 0 then invalid_arg (Printf.sprintf "Nx_amd_device.get: %d < 0" i);
  let refuse = refuse ~machine:host interface i in
  Mutex.protect lock (fun () ->
      match (Nx_remote_device.remote host, interface) with
      | Some _, Kernel -> refuse "another machine's GPUs are reached over PCI"
      | Some _, Pci -> open_gpu ~machine:host ~iface:Pci ?firmware i
      | None, _ when not (linux ()) -> refuse "AMD GPUs need Linux"
      | None, _ -> (
          match !chosen with
          | Some c when c <> interface ->
              refuse
                (Printf.sprintf
                   "this process reaches AMD GPUs through %s, not %s"
                   (interface_name c) (interface_name interface))
          | _ -> open_gpu ~machine:host ~iface:interface ?firmware i))

(* Changes to the machine *)

(* [f bus] for GPU [i] of this machine, which no device of the process holds, or
   why not, starting with the GPU's name through [iface]. *)
let on_gpu iface i f =
  if i < 0 then invalid_arg (Printf.sprintf "Nx_amd_device: %d < 0" i);
  let fail why = Error (name iface i ^ ": " ^ why) in
  Mutex.protect lock @@ fun () ->
  if not (linux ()) then fail "AMD GPUs need Linux"
  else
    let gpus = gpus () in
    match List.nth_opt gpus i with
    | None ->
        fail
          (Printf.sprintf "no GPU %d; there are %d AMD GPUs" i
             (List.length gpus))
    | Some bus when List.mem_assoc (Nx_device.host, bus) (Atomic.get opened) ->
        fail (bus ^ " is open in this process")
    | Some bus -> (
        match f bus with
        | r -> Result.map_error (fun why -> name iface i ^ ": " ^ why) r
        | exception (Failure why | Sys_error why) -> fail why
        | exception Unix.Unix_error (e, fn, arg) ->
            fail (Printf.sprintf "%s %s: %s" fn arg (Unix.error_message e)))

(* With BAR0 as large as the platform allows, the host maps all of the GPU's
   memory, and the runtime keeps no page tables in system memory. *)
let detach i =
  on_gpu Pci i @@ fun bus ->
  Pci.detach bus;
  let p = Pci.take ~lock:"am" bus in
  Fun.protect
    ~finally:(fun () -> Pci.release p)
    (fun () -> try Pci.resize_bar p 0 with Failure _ -> ());
  Ok ()

let attach i = on_gpu Kernel i @@ fun bus -> Ok (Pci.attach bus)

let reset i =
  on_gpu Pci i @@ fun bus ->
  match Pci.detached bus with
  | Error why ->
      Error (Printf.sprintf "%s; Nx_amd_device.detach %d detaches it" why i)
  | Ok () ->
      let p = Pci.take ~lock:"am" bus in
      Fun.protect
        ~finally:(fun () -> Pci.release p)
        (fun () ->
          Pci.reserve p
            ~base:(Page_table.Space.base Am.space)
            (Page_table.Space.length Am.space);
          Ok (Am.reset p))

(* The versions of the blocks the GPU's firmware images are named by, which the
   amdgpu driver lists while it holds the GPU. *)
let block_versions bus =
  let dir hwip =
    Printf.sprintf "/sys/bus/pci/devices/%s/ip_discovery/die/0/%d/0" bus
      (List.assoc hwip D.hw_id_map)
  in
  let version hwip =
    let read f =
      In_channel.with_open_text
        (Filename.concat (dir hwip) f)
        In_channel.input_all
      |> String.trim |> int_of_string
    in
    (hwip, (read "major", read "minor", read "revision"))
  in
  List.map version D.[ gc_hwip; sdma0_hwip; mp0_hwip; mp1_hwip ]

let fetch_firmware i =
  on_gpu Pci i @@ fun bus ->
  if not (List.mem bus (Am.buses ())) then
    Error (bus ^ " is of no GPU family the PCI interface supports")
  else
    match block_versions bus with
    | exception (Sys_error _ | Failure _) ->
        Error
          (Printf.sprintf
             "the amdgpu driver names %s's firmware while it holds the GPU; \
              fetch it before Nx_amd_device.detach %d, or after \
              Nx_amd_device.attach %d"
             bus i i)
    | ip_ver ->
        ignore (Amdev.load_firmware ~load:Amdev.fetch_image ip_ver);
        Ok ()

let of_device = amd_of
let queues a = Option.get a.queues
let compute a = match queues a with c, _, _ -> c
let aql a = match queues a with _, aql, _ -> aql
let sdma a = match queues a with _, _, sdma -> sdma
let props a = a.props

(* While [p] is reachable, so is its code object, and the entry at its handle is
   its own. *)
let kernel p =
  match amd_of (Nx_device.Program.device p) with
  | None -> None
  | Some a ->
      let descriptor = Nx_device.Program.handle p in
      Option.map
        (fun code ->
          let e = with_hw a (fun () -> Hashtbl.find a.kernels descriptor) in
          { code; descriptor; private_segment = e.scratch })
        (Nx_device.Program.code p)

module Thread_trace = Thread_trace
