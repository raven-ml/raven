(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Amd_defs
module Mmio = Nx_device_support.Mmio
module Pci = Nx_device_support.Pci
module Pci_memory = Nx_device_support.Pci_memory
module Page_table = Nx_device_support.Page_table
module Sysmem = Nx_device_support.Sysmem

external linux : unit -> bool = "caml_nx_amd_linux"

type interface = Kernel | Pci

type queue = {
  ring : nativeint;
  ring_bytes : int;
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

type handles = {
  compute : queue;
  aql : bool;
  sdma : queue list;
  signal : nativeint;
  props : props;
}

type kernel = {
  code : nativeint;
  descriptor : nativeint;
  entry : nativeint;
  rsrc1 : int;
  rsrc2 : int;
  rsrc3 : int;
  wave32 : bool;
  private_segment : int;
  group_segment : int;
  kernarg_segment : int;
  dispatch_ptr : bool;
  private_segment_buffer : bool;
}

type scratch = { address : nativeint; bytes : int; tmpring_size : int }

(* The GPU behind a device, through its interface. *)
type gpu =
  | Kfd_gpu of Kfd.t
  | Am_gpu of { am : Am.t; memory : Pci_memory.t; pci : Pci.t }

type mem = Kfd_mem of Kfd.mem | Am_mem of Pci_memory.memory

module Int_map = Map.Make (Int)

type amd = {
  index : int;
  gpu : gpu;
  hw : Mutex.t; (* the GPU's registers and page tables *)
  mutable allocs : alloc Int_map.t; (* the device's own memory, by address *)
  borrows : (int, mem) Hashtbl.t; (* host memory mapped for borrows *)
  reach : (int, bool) Hashtbl.t; (* whether it reaches a peer, by index *)
  kernels : (nativeint, kernel) Hashtbl.t; (* by descriptor address, under hw *)
  images : (string, mem) Hashtbl.t; (* uploaded code objects *)
  props : props;
  scratch_lock : Mutex.t;
  mutable scratch : (Nx_device.Buffer.t * int) option;
  mutable aql_desc : Mmio.t option; (* the AQL queue's descriptor *)
  mutable handles : handles option;
  mutable dev : Nx_device.t option;
}

(* Memory of a device and the peers it is mapped on. *)
and alloc = { mem : mem; mutable peers : amd list }

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

let unmap_from peer mem =
  with_hw peer (fun () ->
      match (peer.gpu, mem) with
      | Kfd_gpu k, Kfd_mem m -> Kfd.unmap_peer k m
      | Am_gpu g, Am_mem m -> Pci_memory.unmap g.memory { m with source = Peer }
      | _ -> invalid_arg "a peer of another interface")

let free_mem a mem =
  with_hw a (fun () ->
      match (a.gpu, mem) with
      | Kfd_gpu k, Kfd_mem m -> Kfd.free k m
      | Am_gpu g, Am_mem m -> Pci_memory.free g.memory m
      | _ -> invalid_arg "memory of another interface")

let register a mem =
  a.allocs <- Int_map.add (va mem) { mem; peers = [] } a.allocs

(* Frees [mem] after unmapping it from the peers that transfers mapped it on:
   their copies were waited for, so none still uses it. *)
let release a mem =
  Option.iter
    (fun r -> List.iter (fun peer -> unmap_from peer mem) r.peers)
    (Int_map.find_opt (va mem) a.allocs);
  a.allocs <- Int_map.remove (va mem) a.allocs;
  free_mem a mem

(* The allocation of [a] that holds the address [x]. *)
let find a x =
  match Int_map.find_last_opt (fun s -> s <= x) a.allocs with
  | Some (s, r) when x < s + size r.mem -> Some r
  | _ -> None

let memory_of mem : Nx_device.memory =
  let va = Nativeint.of_int (va mem) in
  { host = Option.map Mmio.address (host_view mem); device = va; handle = va }

let allocator a kind =
  let alloc n =
    Option.map
      (fun mem ->
        register a mem;
        memory_of mem)
      (alloc_mem a kind n)
  in
  let free (m : Nx_device.memory) =
    Option.iter
      (fun r -> release a r.mem)
      (Int_map.find_opt (Nativeint.to_int m.handle) a.allocs)
  in
  { Nx_device.alloc; free }

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
            { Nx_device.host = Some x; device = x; handle = x })
          r)
  in
  let unmap (m : Nx_device.memory) =
    match Hashtbl.find_opt a.borrows (Nativeint.to_int m.handle) with
    | Some mem ->
        Hashtbl.remove a.borrows (Nativeint.to_int m.handle);
        with_hw a (fun () ->
            match (a.gpu, mem) with
            | Kfd_gpu k, Kfd_mem m -> Kfd.unmap_host k m
            | Am_gpu g, Am_mem m -> Pci_memory.unmap g.memory m
            | _ -> ())
    | None -> ()
  in
  { Nx_device.map; unmap }

(* Queues *)

let mmio64 x = Mmio.v x 8

let sdma_queue (q : queue) =
  {
    Sdma.ring = Mmio.v q.ring q.ring_bytes;
    read_ptr = mmio64 q.read_ptr;
    write_ptr = mmio64 q.write_ptr;
    put = mmio64 q.put;
    doorbell = mmio64 q.doorbell;
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

let opened : (int * amd) list Atomic.t = Atomic.make []

let amd_of d =
  List.find_map
    (fun (_, a) ->
      match a.dev with Some d' when Nx_device.equal d d' -> Some a | _ -> None)
    (Atomic.get opened)

(* Whether [a]'s copy engine reaches the memory of [peer], which the topology
   fixes: over a link the driver reports, or through [peer]'s memory BAR when it
   is large, as mapping [peer]'s memory requires. *)
let reaches a peer =
  with_hw a (fun () ->
      match Hashtbl.find_opt a.reach peer.index with
      | Some r -> r
      | None ->
          let r =
            match (a.gpu, peer.gpu) with
            | Kfd_gpu k, Kfd_gpu k' -> Kfd.reaches k.node k'.node
            | Am_gpu _, Am_gpu g' -> not (Pci_memory.small_bar g'.memory)
            | _ -> false
          in
          Hashtbl.replace a.reach peer.index r;
          r)

let copy_queue a (timeline : Nx_device.memory) =
  let signal = Nativeint.to_int timeline.device in
  let props = a.props in
  let family = sdma_family props and max = max_copy props in
  let enqueue words =
    let q =
      match a.handles with
      | Some h -> sdma_queue (List.hd h.sdma)
      | None -> failwith "the SDMA queue is not set up"
    in
    let timeout_ms = Option.fold ~none:30_000 ~some:Nx_device.timeout a.dev in
    Sdma.submit q ~timeout_ms words
  in
  let submit ~dst ~src n v =
    enqueue
      (Sdma.packets ~family ~max ~signal ~dst:(Nativeint.to_int dst)
         ~src:(Nativeint.to_int src) n v)
  in
  let stamp ~slot v =
    enqueue (Sdma.stamp ~family ~signal ~slot:(Nativeint.to_int slot) v)
  in
  (* A transfer maps the destination's allocation on this GPU at its first use;
     freeing it unmaps it. *)
  let map_on_self peer x =
    match find peer (Nativeint.to_int x) with
    | None -> failwith "the destination is no memory of the other GPU"
    | Some r ->
        if not (List.memq a r.peers) then begin
          with_hw a (fun () ->
              match (a.gpu, peer.gpu, r.mem) with
              | Kfd_gpu k, _, Kfd_mem m -> Kfd.map_peer k m
              | Am_gpu g, Am_gpu g', Am_mem m -> (
                  match Pci_memory.map_peer g.memory g'.memory m with
                  | Ok _ -> ()
                  | Error why -> failwith why)
              | _ -> failwith "a peer of another interface");
          r.peers <- a :: r.peers
        end
  in
  let transfer d' =
    match amd_of d' with
    | Some peer when reaches a peer ->
        Some
          (fun ~dst ~src n v ->
            map_on_self peer dst;
            submit ~dst ~src n v)
    | _ -> None
  in
  { Nx_device.copy = submit; transfer; stamp }

(* Programs *)

let load a ~binary ~name =
  let major, _, _ = a.props.target in
  let obj, img = Code_object.image binary in
  let code =
    match Hashtbl.find_opt a.images binary with
    | Some mem -> mem
    | None -> (
        match alloc_mem a Visible (String.length img) with
        | None -> failwith "no GPU memory for the program"
        | Some mem ->
            register a mem;
            Mmio.write (Option.get (host_view mem)) 0 img;
            Mmio.barrier ();
            Hashtbl.replace a.images binary mem;
            mem)
  in
  let k =
    Code_object.kernel obj img ~name ~major ~lds_kib:(a.props.lds_bytes / 1024)
  in
  let base = va code in
  let at off = Nativeint.of_int (base + off) in
  let kernel =
    {
      code = Nativeint.of_int base;
      descriptor = at k.descriptor;
      entry = at k.entry;
      rsrc1 = k.rsrc1;
      rsrc2 = k.rsrc2;
      rsrc3 = k.rsrc3;
      wave32 = k.wave32;
      private_segment = k.private_segment;
      group_segment = k.group_segment;
      kernarg_segment = k.kernarg_segment;
      dispatch_ptr = k.dispatch_ptr;
      private_segment_buffer = k.private_segment_buffer;
    }
  in
  with_hw a (fun () -> Hashtbl.replace a.kernels kernel.descriptor kernel);
  kernel.descriptor

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

let scratch d n =
  let a =
    match amd_of d with
    | Some a -> a
    | None ->
        invalid_arg
          (Printf.sprintf "Nx_amd_device.scratch: %s is not an AMD device"
             (Nx_device.name d))
  in
  Mutex.protect a.scratch_lock (fun () ->
      let n = Int.max n 128 in
      let b, n =
        match a.scratch with
        | Some (b, have) when have >= n -> (b, have)
        | _ ->
            let b =
              Nx_device.Buffer.create d Nx_dtype.Scalar.UInt8
                (Scratch.bytes (scratch_gpu a.props) n)
            in
            Option.iter (fun desc -> aql_scratch a desc b n) a.aql_desc;
            a.scratch <- Some (b, n);
            (b, n)
      in
      {
        address = Nx_device.Buffer.address b;
        bytes = Nx_device.Buffer.nbytes b;
        tmpring_size = Scratch.tmpring_size (scratch_gpu a.props) n;
      })

(* Opening *)

let name i = if i = 0 then "AMD" else Printf.sprintf "AMD:%d" i

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
      ring = addr ring;
      ring_bytes = spec.ring_bytes;
      read_ptr = rptr;
      write_ptr = wptr;
      put = addr put;
      doorbell;
    }
  in
  (q, if aql then Some gart_view else None)

let make_device a ~budget ~sleep ?finalize () =
  let dev =
    Nx_device.make ~name:(name a.index) ~arch:(arch a.props.target) ~budget
      ~memory:(allocator a Vram) ~host_memory:(allocator a Host)
      ~mapping:(mapping a) ~copy_queue:(copy_queue a) ~load:(load a) ~sleep
      ~clock:(Nx_device.Device_clock { hz = 100_000_000 })
      ?finalize ()
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

let record ~index ~gpu ~props =
  {
    index;
    gpu;
    hw = Mutex.create ();
    allocs = Int_map.empty;
    borrows = Hashtbl.create 16;
    reach = Hashtbl.create 4;
    kernels = Hashtbl.create 16;
    images = Hashtbl.create 8;
    props;
    scratch_lock = Mutex.create ();
    scratch = None;
    aql_desc = None;
    handles = None;
    dev = None;
  }

let finish a dev (compute, aql, sdma) =
  a.handles <-
    Some
      {
        compute;
        aql;
        sdma;
        signal = Nx_device.Buffer.address (Nx_device.timeline dev);
        props = a.props;
      }

let open_kfd index =
  let k = Kfd.open_gpu index in
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
  let a = record ~index ~gpu:(Kfd_gpu k) ~props in
  let compute, aql, sdma = setup a ~saves:true ~sdma_queues:1 in
  let dev = make_device a ~budget:k.vram ~sleep:(fun ms -> Kfd.sleep k ms) () in
  finish a dev (compute, aql, sdma);
  a

(* Opens the device of [am], a GPU booted over [pci]. *)
let open_booted ~buses ~index pci (am : Am.t) =
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
  let a = record ~index ~gpu:(Am_gpu { am; memory; pci }) ~props in
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
let open_taken ?firmware ~buses ~index pci =
  Sysmem.reserve
    ~base:(Page_table.Space.base Am.space)
    (Page_table.Space.length Am.space);
  (try Pci.resize_bar pci 0 with Failure _ -> ());
  let am = Am.boot ?firmware pci in
  match open_booted ~buses ~index pci am with
  | a -> a
  | exception e ->
      (try Am.fini am ~failed:true with Failure _ -> ());
      raise e

(* A failed open gives the function back, so that a later one can take it. *)
let open_am ?firmware index =
  let buses = Am.buses () in
  let bus =
    match List.nth_opt buses index with
    | Some b -> b
    | None ->
        failwith
          (Printf.sprintf "no GPU %d; there are %d AMD GPUs" index
             (List.length buses))
  in
  let pci = Pci.take ~lock:"am" bus in
  match open_taken ?firmware ~buses:(List.length buses) ~index pci with
  | a -> a
  | exception e ->
      Pci.release pci;
      raise e

(* The interface of the process, fixed by its first successful open. *)
let chosen = ref None
let lock = Mutex.create ()

let default () =
  match !chosen with
  | Some i -> i
  | None -> if Kfd.available () then Kernel else Pci

let count ?interface () =
  if not (linux ()) then 0
  else
    Mutex.protect lock (fun () ->
        match Option.value interface ~default:(default ()) with
        | Kernel -> ( try Kfd.count () with Sys_error _ | Failure _ -> 0)
        | Pci -> List.length (Am.buses ()))

let interface_name = function Kernel -> "the kernel driver" | Pci -> "PCI"

let get ?interface ?firmware i =
  if i < 0 then invalid_arg (Printf.sprintf "Nx_amd_device.get: %d < 0" i);
  if not (linux ()) then Error "AMD: AMD GPUs need Linux"
  else
    Mutex.protect lock (fun () ->
        let iface = Option.value interface ~default:(default ()) in
        match !chosen with
        | Some c when c <> iface ->
            Error
              (Printf.sprintf
                 "AMD: this process reaches AMD GPUs through %s, not %s"
                 (interface_name c) (interface_name iface))
        | _ -> (
            match List.assoc_opt i (Atomic.get opened) with
            | Some a -> Ok (Option.get a.dev)
            | None -> (
                match
                  match iface with
                  | Kernel -> open_kfd i
                  | Pci -> open_am ?firmware i
                with
                | a ->
                    chosen := Some iface;
                    Atomic.set opened ((i, a) :: Atomic.get opened);
                    Ok (Option.get a.dev)
                | exception (Failure msg | Sys_error msg | Invalid_argument msg)
                  ->
                    Error ("AMD: " ^ msg)
                | exception Unix.Unix_error (e, fn, arg) ->
                    Error
                      (Printf.sprintf "AMD: %s %s: %s" fn arg
                         (Unix.error_message e))
                | exception Not_found -> Error "AMD: opening failed: Not_found")
            ))

let v ?interface ?firmware i =
  match get ?interface ?firmware i with
  | Ok d -> d
  | Error msg -> invalid_arg msg

let amd fn d =
  match amd_of d with
  | Some a -> a
  | None ->
      invalid_arg
        (Printf.sprintf "Nx_amd_device.%s: %s is not an AMD device" fn
           (Nx_device.name d))

let interface d =
  match (amd "interface" d).gpu with Kfd_gpu _ -> Kernel | Am_gpu _ -> Pci

let handles d = Option.get (amd "handles" d).handles

let kernel p =
  let d = Nx_device.Program.device p in
  let a = amd "kernel" d in
  match
    with_hw a (fun () ->
        Hashtbl.find_opt a.kernels (Nx_device.Program.handle p))
  with
  | Some k -> k
  | None -> invalid_arg "Nx_amd_device.kernel: the program is not an AMD kernel"
