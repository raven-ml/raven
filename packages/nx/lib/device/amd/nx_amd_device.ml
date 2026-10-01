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
  mutable allocs : alloc Int_map.t; (* the device's own memory, by address *)
  borrows : (int, mem) Hashtbl.t; (* host memory mapped for borrows *)
  reach : (int, bool) Hashtbl.t; (* whether it reaches a peer, by index *)
  kernels : (nativeint, kernel) Hashtbl.t; (* by descriptor address, under hw *)
  images : (string, Nx_device.Buffer.t) Hashtbl.t; (* uploaded code objects *)
  props : props;
  scratch_lock : Mutex.t;
  mutable scratch : (Nx_device.Buffer.t * int) option;
  mutable aql_desc : Mmio.t option; (* the AQL queue's descriptor *)
  mutable queues : (queue * bool * queue list) option;
      (* the compute queue, whether it takes AQL packets, the SDMA queues *)
  mutable dev : Nx_device.t option;
}

(* Memory of a device and the peers it is mapped on, which their borrows and
   transfers add to while the owner is not taken. *)
and alloc = { mem : mem; peers : t list Atomic.t }

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
  a.allocs <- Int_map.add (va mem) { mem; peers = Atomic.make [] } a.allocs

(* Frees [mem] after unmapping it from the peers that transfers mapped it on:
   their copies were waited for, so none still uses it. *)
let release a mem =
  Option.iter
    (fun r ->
      List.iter (fun peer -> unmap_from peer mem) (Atomic.exchange r.peers []))
    (Int_map.find_opt (va mem) a.allocs);
  a.allocs <- Int_map.remove (va mem) a.allocs;
  free_mem a mem

(* The allocation of [a] that holds the address [x]. *)
let find a x =
  match Int_map.find_last_opt (fun s -> s <= x) a.allocs with
  | Some (s, r) when x < s + size r.mem -> Some r
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
    Option.iter
      (fun r -> release a r.mem)
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

(* The opened GPUs, by the host of their machine and index there. *)
let opened : ((Nx_device.t * int) * t) list Atomic.t = Atomic.make []

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

(* Maps [peer]'s allocation that holds [x] on [a], at its first use, at the same
   address; freeing it unmaps it. *)
let map_peer a peer x =
  match find peer (Nativeint.to_int x) with
  | None -> Error "no memory of the other GPU"
  | Some r when List.memq a (Atomic.get r.peers) -> Ok ()
  | Some r ->
      Result.map
        (fun () ->
          let rec push () =
            let l = Atomic.get r.peers in
            if not (Atomic.compare_and_set r.peers l (a :: l)) then push ()
          in
          push ())
        (with_hw a (fun () ->
             match (a.gpu, peer.gpu, r.mem) with
             | Kfd_gpu k, _, Kfd_mem m -> (
                 match Kfd.map_peer k m with
                 | () -> Ok ()
                 | exception Failure why -> Error why)
             | Am_gpu g, Am_gpu g', Am_mem m ->
                 Result.map ignore (Pci_memory.map_peer g.memory g'.memory m)
             | _ -> Error "a peer of another interface"))

(* Borrows of another AMD GPU's memory that [a] reaches. *)
let peer a d' r =
  match amd_of d' with
  | Some peer when reaches a peer ->
      Result.map (fun () -> r) (map_peer a peer (Region.address r))
  | Some _ -> Error "the GPUs do not reach each other's memory"
  | None -> Error "memory of another vendor"

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
  (* A transfer maps the destination's allocation on this GPU. *)
  let transfer d' =
    match amd_of d' with
    | Some peer when reaches a peer ->
        Some
          (fun ~dst ~src n ~signal ->
            (match map_peer a peer dst with
            | Ok () -> ()
            | Error why -> failwith why);
            submit ~dst ~src n ~signal)
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
  | Am_gpu { memory; pci; _ }, Some { mem = Am_mem pm; _ } -> (
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
  | Am_gpu _, Some { mem = Kfd_mem _; _ } -> Error "memory of another interface"

(* Programs *)

(* The code object [binary], uploaded once, as a buffer of the device. *)
let upload a binary img =
  match Hashtbl.find_opt a.images binary with
  | Some code -> Some code
  | None ->
      Option.map
        (fun mem ->
          register a mem;
          Mmio.write (Option.get (host_view mem)) 0 img;
          Mmio.barrier ();
          let n = String.length img in
          let code =
            Driver.buffer (Option.get a.dev) (region_of mem n)
              Nx_dtype.Scalar.UInt8 n
          in
          Hashtbl.replace a.images binary code;
          code)
        (alloc_mem a Visible (String.length img))

(* A code object the device cannot run, or has no memory for, is refused, and
   the device stays usable. *)
let load a ~binary ~entry:name =
  let major, _, _ = a.props.target in
  match
    let obj, img = Code_object.image binary in
    ( img,
      Code_object.kernel obj img ~name ~major ~lds_kib:(a.props.lds_bytes / 1024)
    )
  with
  | exception Failure why -> Error why
  | img, k -> (
      match upload a binary img with
      | None ->
          Error
            "no GPU memory the host addresses for the program: the memory is \
             full, or the BAR is too small to map it (enable Resizable BAR in \
             the firmware settings)"
      | Some code ->
          let at off =
            Nativeint.add (Nx_device.Buffer.address code) (Nativeint.of_int off)
          in
          let kernel =
            {
              code;
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
          with_hw a (fun () ->
              Hashtbl.replace a.kernels kernel.descriptor kernel);
          Ok kernel.descriptor)

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

(* A GPU of another machine is named for it: ["AMD:1@HOST:PORT"]. *)
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

let make_device a ~budget ~sleep ?finalize () =
  let dev =
    Driver.device ~name:(name a.index) ~arch:(arch a.props.target)
      ~host:a.machine ~budget
      ~completion:(Sleep (fun ~timeline:_ -> sleep))
      ~load:(load a) ~peer:(peer a)
      ~reaches:(fun d' ->
        match amd_of d' with Some peer -> reaches a peer | None -> false)
      ~dma:(dma a) ~room:(room a) ?finalize
      (Device_local
         {
           memory = allocator a Vram;
           host_memory = allocator a Host;
           mapped =
             (match a.gpu with
             | Am_gpu g when Pci_memory.small_bar g.memory -> None
             | Kfd_gpu k when not (Kfd.flushes_hdp k) -> None
             | _ -> Some (allocator a Visible));
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

let record ~machine ~index ~gpu ~props =
  {
    index;
    machine;
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
  let a = record ~machine:Nx_device.host ~index ~gpu:(Kfd_gpu k) ~props in
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
  let a = record ~machine ~index ~gpu:(Am_gpu { am; memory; pci }) ~props in
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
  (try Pci.resize_bar pci 0 with Failure _ -> ());
  let am = Am.boot ?firmware pci in
  match open_booted ~machine ~buses ~index pci am with
  | a -> a
  | exception e ->
      (try Am.fini am ~failed:true with Failure _ -> ());
      raise e

(* A failed open gives the function back, so that a later one can take it. *)
let open_am ?firmware ~machine index =
  let remote = Nx_remote_device.remote machine in
  let buses = Am.buses ?remote () in
  let bus =
    match List.nth_opt buses index with
    | Some b -> b
    | None ->
        failwith
          (Printf.sprintf "no GPU %d; there are %d AMD GPUs" index
             (List.length buses))
  in
  let pci = Pci.take ?remote ~lock:"am" bus in
  match open_taken ?firmware ~machine ~buses:(List.length buses) ~index pci with
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

(* Raises [Lost] for [host] if its machine can no longer be reached: the
   synchronization of [host] meets the failed connection, which loses it. *)
let check_reach host =
  match Nx_remote_device.remote host with
  | Some r when Remote.failed r <> None -> Nx_device.synchronize host
  | Some _ | None -> ()

let count ?(host = Nx_device.host) ?interface () =
  match Nx_remote_device.remote host with
  | Some remote -> (
      try List.length (Am.buses ~remote ())
      with Failure _ as e ->
        check_reach host;
        raise e)
  | None when not (linux ()) -> 0
  | None ->
      Mutex.protect lock (fun () ->
          match Option.value interface ~default:(default ()) with
          | Kernel -> ( try Kfd.count () with Sys_error _ | Failure _ -> 0)
          | Pci -> List.length (Am.buses ()))

let interface_name = function Kernel -> "the kernel driver" | Pci -> "PCI"

(* Why GPU [i] of the machine of [machine] cannot be opened. *)
let refuse ~machine i why =
  check_reach machine;
  Error (Driver.name ~host:machine (name i) ^ ": " ^ why)

(* Opens [i] through [iface] on the machine of [machine], once. *)
let open_gpu ~machine ~iface ?firmware i =
  match List.assoc_opt (machine, i) (Atomic.get opened) with
  | Some a -> Ok (Option.get a.dev)
  | None -> (
      match
        match iface with
        | Kernel -> open_kfd i
        | Pci -> open_am ?firmware ~machine i
      with
      | a ->
          if machine == Nx_device.host then chosen := Some iface;
          Atomic.set opened (((machine, i), a) :: Atomic.get opened);
          Ok (Option.get a.dev)
      | exception (Failure why | Sys_error why | Invalid_argument why) ->
          refuse ~machine i why
      | exception Unix.Unix_error (e, fn, arg) ->
          refuse ~machine i
            (Printf.sprintf "%s %s: %s" fn arg (Unix.error_message e))
      | exception Not_found -> refuse ~machine i "opening failed: Not_found")

let get ?(host = Nx_device.host) ?interface ?firmware i =
  if i < 0 then invalid_arg (Printf.sprintf "Nx_amd_device.get: %d < 0" i);
  let remote = Nx_remote_device.remote host in
  let refuse = refuse ~machine:host i in
  Mutex.protect lock (fun () ->
      match (remote, interface) with
      | Some _, Some Kernel ->
          refuse "another machine's GPUs are reached over PCI"
      | Some _, _ -> open_gpu ~machine:host ~iface:Pci ?firmware i
      | None, _ when not (linux ()) -> refuse "AMD GPUs need Linux"
      | None, _ -> (
          let iface = Option.value interface ~default:(default ()) in
          match !chosen with
          | Some c when c <> iface ->
              refuse
                (Printf.sprintf
                   "this process reaches AMD GPUs through %s, not %s"
                   (interface_name c) (interface_name iface))
          | _ -> open_gpu ~machine:host ~iface ?firmware i))

let v ?host ?interface ?firmware i =
  match get ?host ?interface ?firmware i with
  | Ok d -> d
  | Error msg -> failwith msg

let of_device = amd_of
let interface a = match a.gpu with Kfd_gpu _ -> Kernel | Am_gpu _ -> Pci
let queues a = Option.get a.queues
let compute a = match queues a with c, _, _ -> c
let aql a = match queues a with _, aql, _ -> aql
let sdma a = match queues a with _, _, sdma -> sdma
let props a = a.props

let kernel p =
  Option.bind
    (amd_of (Nx_device.Program.device p))
    (fun a ->
      with_hw a (fun () ->
          Hashtbl.find_opt a.kernels (Nx_device.Program.handle p)))
