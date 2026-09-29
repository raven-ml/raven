(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Nv_defs
module P = Params
module Mmio = Nx_device_support.Mmio
module Pci = Nx_device_support.Pci
module Pci_memory = Nx_device_support.Pci_memory
module Page_table = Nx_device_support.Page_table
module Sysmem = Nx_device_support.Sysmem
module Firmware = Nx_device_support.Firmware

external linux : unit -> bool = "caml_nx_nv_linux"

type interface = Kernel | Pci

type channel = {
  ring : nativeint;
  entries : int;
  gp_get : nativeint;
  gp_put : nativeint;
  put : nativeint;
  doorbell : nativeint;
  token : int;
}

type props = {
  sm_version : int;
  sass_version : int;
  gpcs : int;
  tpcs_per_gpc : int;
  sms_per_tpc : int;
  warps_per_sm : int;
  gpfifo_class : int;
  compute_class : int;
  dma_class : int;
}

type handles = {
  compute : channel;
  copy : channel;
  signal : nativeint;
  shared_window : nativeint;
  local_window : nativeint;
  props : props;
}

type kernel = {
  image : nativeint;
  entry : nativeint;
  code_bytes : int;
  registers : int;
  shared_bytes : int;
  local_bytes : int;
  param_offset : int;
  banks : (int * nativeint * int) list;
  max_threads : int;
}

type local_memory = { address : nativeint; bytes : int; per_thread : int }

(* The GPU behind a device, through its interface. *)
type gpu =
  | Kernel_gpu of Nvk.gpu
  | Pci_gpu of {
      nvdev : Nvdev.t;
      gsp : Gsp.t;
      memory : Pci_memory.t;
      pci : Pci.t;
    }

type mem = Kernel_mem of Nvk.mem | Pci_mem of Pci_memory.memory

module Int_map = Map.Make (Int)

(* The objects the resource manager holds for a device. *)
type objects = {
  device : int;
  subdevice : int;
  group : int;
  ctxshare : int;
  mutable compute_channel : int;
  mutable debugger : int;
}

type nv = {
  index : int;
  gpu : gpu;
  rm : Rm.t;
  hw : Mutex.t; (* the GPU's page tables and memory objects *)
  mutable allocs : alloc Int_map.t; (* the device's own memory, by address *)
  borrows : (int, mem) Hashtbl.t; (* host memory mapped for borrows *)
  images : (string, mem) Hashtbl.t; (* uploaded cubins *)
  kernels : (nativeint, kernel) Hashtbl.t; (* by entry *)
  unreachable : int list; (* the GPUs peer access was refused with *)
  obj : objects;
  props : props;
  mutable ring : Pushbuf.ring option;
  mutable channels : (Pushbuf.channel * Pushbuf.channel) option;
  mutable handles : handles option;
  local_lock : Mutex.t;
  mutable local : (Nx_device.Buffer.t * int) option;
  mutable dev : Nx_device.t option;
}

(* Memory of a device and the peers it is mapped on. *)
and alloc = { mem : mem; mutable peers : nv list }

let va = function Kernel_mem m -> m.va | Pci_mem m -> m.mapping.va
let size = function Kernel_mem m -> m.size | Pci_mem m -> m.mapping.size

let host_view = function
  | Kernel_mem m ->
      if m.cpu then Some (Mmio.v (Nativeint.of_int m.va) m.size) else None
  | Pci_mem m -> m.host

(* The handle RM knows the memory by: its object, or on a GPU the process
   drives, its first physical address. *)
let rm_handle = function
  | Kernel_mem m -> m.handle
  | Pci_mem m -> fst (List.hd m.mapping.pages)

let with_hw n f = Mutex.protect n.hw f
let round_up n a = (n + a - 1) / a * a

(* Memory *)

type kind =
  | Vram
  | Visible (* the GPU's memory, which the process reaches *)
  | Host
  | Uncached (* the GPU's error notifiers *)
  | Channels (* the channels' rings, in the GPU's memory *)

let alloc_mem n kind bytes =
  with_hw n (fun () ->
      match n.gpu with
      | Kernel_gpu g ->
          let m =
            match kind with
            | Vram -> Nvk.alloc g bytes
            | Visible -> Nvk.alloc g ~cpu_access:true ~contiguous:true bytes
            | Host -> Nvk.alloc g ~host:true bytes
            | Uncached -> Nvk.alloc g ~uncached:true bytes
            | Channels ->
                Nvk.alloc g ~cpu_access:true ~contiguous:true
                  ~map_flags:
                    (P.bits D.nvos33_flags_caching_type
                       D.nvos33_flags_caching_type_writecombined)
                  bytes
          in
          Option.map (fun m -> Kernel_mem m) m
      | Pci_gpu p ->
          let m =
            match kind with
            | Vram -> Pci_memory.alloc p.memory bytes
            | Visible -> Pci_memory.alloc ~cpu_access:true p.memory bytes
            | Host -> Pci_memory.alloc ~host:true p.memory bytes
            | Uncached -> Pci_memory.alloc ~uncached:true p.memory bytes
            | Channels ->
                Pci_memory.alloc ~cpu_access:true ~devmem:true p.memory bytes
          in
          Option.map (fun m -> Pci_mem m) m)

let unmap_from peer mem =
  with_hw peer (fun () ->
      match (peer.gpu, mem) with
      | Pci_gpu p, Pci_mem m ->
          Pci_memory.unmap p.memory { m with source = Peer }
      | Kernel_gpu _, Kernel_mem _ -> () (* freeing the range unmaps it *)
      | _ -> invalid_arg "a peer of another interface")

let free_mem n mem =
  with_hw n (fun () ->
      match (n.gpu, mem) with
      | Kernel_gpu g, Kernel_mem m -> Nvk.free g m
      | Pci_gpu p, Pci_mem m -> Pci_memory.free p.memory m
      | _ -> invalid_arg "memory of another interface")

let register n mem =
  n.allocs <- Int_map.add (va mem) { mem; peers = [] } n.allocs

(* Frees [mem] after unmapping it from the peers that transfers mapped it on:
   their copies were waited for, so none still uses it. *)
let release n mem =
  Option.iter
    (fun r -> List.iter (fun peer -> unmap_from peer mem) r.peers)
    (Int_map.find_opt (va mem) n.allocs);
  n.allocs <- Int_map.remove (va mem) n.allocs;
  free_mem n mem

let find n x =
  match Int_map.find_last_opt (fun s -> s <= x) n.allocs with
  | Some (s, r) when x < s + size r.mem -> Some r
  | _ -> None

let memory_of mem : Nx_device.memory =
  let va = Nativeint.of_int (va mem) in
  { host = Option.map Mmio.address (host_view mem); device = va; handle = va }

let allocator n kind =
  let alloc bytes =
    Option.map
      (fun mem ->
        register n mem;
        memory_of mem)
      (alloc_mem n kind bytes)
  in
  let free (m : Nx_device.memory) =
    Option.iter
      (fun r -> release n r.mem)
      (Int_map.find_opt (Nativeint.to_int m.handle) n.allocs)
  in
  { Nx_device.alloc; free }

(* Memory the runtime keeps for the device's life. *)
let need what = function
  | Some m -> m
  | None -> failwith ("no GPU memory for the " ^ what)

let keep n kind what bytes =
  let m = need what (alloc_mem n kind bytes) in
  register n m;
  m

(* Borrows: host memory the GPU addresses at its own address. *)
let mapping n =
  let map x bytes =
    with_hw n (fun () ->
        match n.gpu with
        | Kernel_gpu g -> (
            match Nvk.map_host g x bytes with
            | Ok () -> Ok { Nx_device.host = Some x; device = x; handle = x }
            | Error why -> Error why)
        | Pci_gpu p ->
            Result.map
              (fun mem ->
                Hashtbl.replace n.borrows (Nativeint.to_int x) (Pci_mem mem);
                { Nx_device.host = Some x; device = x; handle = x })
              (Pci_memory.map_host p.memory x bytes))
  in
  let unmap (m : Nx_device.memory) =
    with_hw n (fun () ->
        match n.gpu with
        | Kernel_gpu g -> Nvk.unmap_host g m.handle
        | Pci_gpu p -> (
            let key = Nativeint.to_int m.handle in
            match Hashtbl.find_opt n.borrows key with
            | Some (Pci_mem mem) ->
                Hashtbl.remove n.borrows key;
                Pci_memory.unmap p.memory mem
            | _ -> ()))
  in
  { Nx_device.map; unmap }

(* Channels *)

let opened : (int * nv) list Atomic.t = Atomic.make []

let nv_of d =
  List.find_map
    (fun (_, n) ->
      match n.dev with Some d' when Nx_device.equal d d' -> Some n | _ -> None)
    (Atomic.get opened)

let timeout n = Option.fold ~none:30_000 ~some:Nx_device.timeout n.dev

let signal_word (timeline : Nx_device.memory) =
  Mmio.v (Option.get timeline.host) 8

(* Submits the work [body] for the timeline value [v] on [ch]: after [v - 1],
   ending by signaling [v] with [signal]. *)
let run n ch ~timeline ~signal v body =
  let addr = Nativeint.to_int timeline.Nx_device.device in
  let word = signal_word timeline in
  let words = Pushbuf.acquire addr (v - 1) @ body @ signal addr v in
  let seg =
    Pushbuf.segment (Option.get n.ring)
      ~signaled:(fun () -> Int64.to_int (Mmio.get64 word 0))
      ~timeout_ms:(timeout n) v words
  in
  Pushbuf.submit ch ~timeout_ms:(timeout n) seg (List.length words)

let channels n = Option.get n.channels

(* Whether [n]'s copy engine reaches the memory of [peer]. *)
let reaches n peer =
  match (n.gpu, peer.gpu) with
  | Kernel_gpu _, Kernel_gpu _ -> not (List.mem peer.index n.unreachable)
  | Pci_gpu _, Pci_gpu p -> not (Pci_memory.small_bar p.memory)
  | _ -> false

let copy_queue n (timeline : Nx_device.memory) =
  let submit ~dst ~src bytes v =
    let _, copy = channels n in
    run n copy ~timeline ~signal:Pushbuf.copy_release v
      (Pushbuf.copy ~dst:(Nativeint.to_int dst) ~src:(Nativeint.to_int src)
         bytes)
  in
  (* A transfer maps the destination's allocation on this GPU at its first use;
     freeing it unmaps it. *)
  let map_on_self peer x =
    match find peer (Nativeint.to_int x) with
    | None -> failwith "the destination is no memory of the other GPU"
    | Some r ->
        if not (List.memq n r.peers) then begin
          with_hw n (fun () ->
              match (n.gpu, peer.gpu, r.mem) with
              | Kernel_gpu g, _, Kernel_mem m -> Nvk.map_peer g m
              | Pci_gpu p, Pci_gpu p', Pci_mem m -> (
                  match Pci_memory.map_peer p.memory p'.memory m with
                  | Ok _ -> ()
                  | Error why -> failwith why)
              | _ -> failwith "a peer of another interface");
          r.peers <- n :: r.peers
        end
  in
  let transfer d' =
    match nv_of d' with
    | Some peer when reaches n peer ->
        Some
          (fun ~dst ~src bytes v ->
            map_on_self peer dst;
            submit ~dst ~src bytes v)
    | _ -> None
  in
  { Nx_device.copy = submit; transfer }

(* Programs *)

let load n ~binary ~name =
  let c = Cubin.load binary ~name in
  let mem =
    match Hashtbl.find_opt n.images binary with
    | Some mem -> mem
    | None ->
        let mem = keep n Visible "program" (String.length c.image) in
        let base = va mem in
        Mmio.write (Option.get (host_view mem)) 0 (Cubin.relocate c ~base);
        Mmio.barrier ();
        Hashtbl.replace n.images binary mem;
        mem
  in
  let base = va mem in
  let at off = Nativeint.of_int (base + off) in
  let k =
    {
      image = at 0;
      entry = at c.entry;
      code_bytes = c.code_bytes;
      registers = c.registers;
      shared_bytes = c.shared_bytes;
      local_bytes = c.local_bytes;
      param_offset = c.param_offset;
      banks = List.map (fun (i, off, bytes) -> (i, at off, bytes)) c.banks;
      max_threads = c.max_threads;
    }
  in
  Hashtbl.replace n.kernels k.entry k;
  k.entry

(* Faults *)

let fault_name table v =
  Option.value ~default:(Printf.sprintf "0x%x" v) (List.assoc_opt v table)

(* The faults the GPU's multiprocessors or its MMU report, one per line. *)
let fault_report n =
  let module S = D.Sm_error_states in
  let module E = D.Sm_error_state in
  let p = P.create S.sizeof in
  P.set p S.h_target_channel n.obj.compute_channel;
  P.set p S.num_s_ms_to_read 100;
  n.rm.control n.obj.debugger D.nv83de_ctrl_cmd_debug_read_all_sm_error_states
    (Some p);
  if P.get p S.mmu_fault_valid <> 0 then begin
    let module M = D.Mmu_fault_info in
    let module F = D.Mmu_fault_entry in
    let m = P.create M.sizeof in
    n.rm.control n.obj.debugger D.nv83de_ctrl_cmd_debug_read_mmu_fault_info
      (Some m);
    List.init (P.get m M.count) (fun i ->
        let f x = P.elt_field M.mmu_fault_info_list i x in
        Printf.sprintf "MMU fault: 0x%X | %s | %s"
          (P.get m (f F.fault_address))
          (fault_name D.fault_fault_types (P.get m (f F.fault_type)))
          (fault_name D.fault_access_types (P.get m (f F.access_type))))
  end
  else
    let _, _, count = S.sm_error_state_array in
    List.filter_map
      (fun i ->
        let f x = P.elt_field S.sm_error_state_array i x in
        let global = P.get p (f E.hww_global_esr)
        and warp = P.get p (f E.hww_warp_esr) in
        if global = 0 && warp = 0 then None
        else
          Some
            (Printf.sprintf "SM %d fault: esr=0x%x warp_esr=0x%x warp_pc=0x%x" i
               global warp
               (P.get p (f E.hww_warp_esr_pc64))))
      (List.init count Fun.id)

(* The check a wait runs every 200 ms the timeline stays still, without
   blocking: under [Pci], the GSP's messages are handled first, and an error it
   reported fails the device. *)
let check_faults n =
  (match n.gpu with Pci_gpu p -> Gsp.poll p.gsp | Kernel_gpu _ -> ());
  match fault_report n with
  | [] -> ()
  | report -> failwith (String.concat "\n" report)

(* Opening *)

let name i = if i = 0 then "NV" else Printf.sprintf "NV:%d" i

let arch v =
  if v = 0xa04 then "sm_120"
  else
    let minor = v land 0xff in
    Printf.sprintf "sm_%d%d"
      ((v lsr 8) land 0xff)
      (if minor > 0xf then minor lsr 4 else minor)

(* The memory windows at which kernels see their shared and local memory. *)
let shared_window = 0x729400000000
let local_window = 0x729300000000

let pick what classes available =
  match List.find_opt (fun c -> List.mem c available) classes with
  | Some c -> c
  | None -> failwith (Printf.sprintf "the GPU has no supported %s class" what)

(* The classes of the GPU's usermode, channels, compute and copy engines. *)
let kernel_classes (rm : Rm.t) device =
  let module C = D.Classlist in
  let p = P.create C.sizeof in
  rm.control device D.nv0080_ctrl_cmd_gpu_get_classlist (Some p);
  let count = P.get p C.num_classes in
  let list = P.create (4 * count) in
  P.set p C.class_list (Nativeint.to_int (P.address list));
  rm.control device D.nv0080_ctrl_cmd_gpu_get_classlist (Some p);
  let available = List.init count (fun i -> P.get list (4 * i, 4)) in
  ( pick "usermode" [ D.hopper_usermode_a; D.turing_usermode_a ] available,
    pick "channel"
      [ D.blackwell_channel_gpfifo_a; D.ampere_channel_gpfifo_a ]
      available,
    pick "compute"
      [ D.blackwell_compute_b; D.ada_compute_a; D.ampere_compute_b ]
      available,
    pick "copy" [ D.blackwell_dma_copy_b; D.ampere_dma_copy_b ] available )

let gr_indices =
  [
    D.nv2080_ctrl_gr_info_index_litter_num_gpcs;
    D.nv2080_ctrl_gr_info_index_litter_num_tpc_per_gpc;
    D.nv2080_ctrl_gr_info_index_litter_num_sm_per_tpc;
    D.nv2080_ctrl_gr_info_index_max_warps_per_sm;
    D.nv2080_ctrl_gr_info_index_sm_version;
  ]

let gr_info gpu (rm : Rm.t) subdevice =
  match gpu with
  | Pci_gpu _ ->
      let module S = D.Static_gr_info in
      let module L = D.Gr_info_list in
      let module I = D.Internal_gr_info in
      let p = P.create S.sizeof in
      rm.control subdevice D.nv2080_ctrl_cmd_internal_static_kgr_get_info
        (Some p);
      List.map
        (fun idx ->
          P.get p
            (P.elt_field S.engine_info 0 (P.elt_field L.info_list idx I.data)))
        gr_indices
  | Kernel_gpu _ ->
      let module G = D.Gr_get_info in
      let module I = D.Gr_info in
      let n = List.length gr_indices in
      let infos = P.create (n * I.sizeof) in
      List.iteri
        (fun i idx ->
          P.set infos (fst I.index + (i * I.sizeof), snd I.index) idx)
        gr_indices;
      let p = P.create G.sizeof in
      P.set p G.gr_info_list_size n;
      P.set p G.gr_info_list (Nativeint.to_int (P.address infos));
      rm.control subdevice D.nv2080_ctrl_cmd_gr_get_info (Some p);
      let r =
        List.init n (fun i ->
            P.get infos (fst I.data + (i * I.sizeof), snd I.data))
      in
      ignore (Sys.opaque_identity infos);
      r

(* A channel of [entries] entries at [offset] of the channels' memory [buf], its
   USERD after its ring, bound to [cls]. *)
let new_channel n ~buf ~put ~doorbell ~offset ~entries ~engine ~compute =
  let rm = n.rm in
  let notifier = keep n Uncached "channel's error notifier" (48 lsl 20) in
  let (module R : D.RELEASE) =
    match n.gpu with Kernel_gpu g -> g.c.release | Pci_gpu _ -> Gsp.release
  in
  let module G = R.Gpfifo_alloc in
  let p = P.create G.sizeof in
  P.set p G.gp_fifo_offset (va buf + offset);
  P.set p G.gp_fifo_entries entries;
  P.set p G.h_object_error (rm_handle notifier);
  P.set p G.h_object_buffer (rm_handle buf);
  P.set p (P.elt G.h_userd_memory 0) (rm_handle buf);
  P.set p (P.elt G.userd_offset 0) ((entries * 8) + offset);
  P.set p G.h_context_share n.obj.ctxshare;
  let ch = rm.alloc ~parent:n.obj.group n.props.gpfifo_class (Some p) in
  if compute then begin
    let obj = rm.alloc ~parent:ch engine None in
    let a = P.create D.Nv83de_alloc.sizeof in
    P.set a D.Nv83de_alloc.h_app_client rm.root;
    P.set a D.Nv83de_alloc.h_class3d_object obj;
    n.obj.debugger <- rm.alloc ~parent:n.obj.device D.gt200_debugger (Some a);
    n.obj.compute_channel <- ch
  end
  else ignore (rm.alloc ~parent:ch engine None : int);
  let t = P.create D.Work_submit_token.sizeof in
  P.set t D.Work_submit_token.work_submit_token 0xffff_ffff;
  rm.control ch D.nvc36f_ctrl_cmd_gpfifo_get_work_submit_token (Some t);
  (match n.gpu with
  | Kernel_gpu g -> Nvk.register_channel g ch
  | Pci_gpu _ -> ());
  let view = Option.get (host_view buf) in
  let userd = offset + (entries * 8) in
  {
    Pushbuf.ring = Mmio.sub view offset (entries * 8);
    entries;
    gp_get = Mmio.sub view (userd + fst D.Userd.gp_get) 4;
    gp_put = Mmio.sub view (userd + fst D.Userd.gp_put) 4;
    put;
    doorbell;
    token = P.get t D.Work_submit_token.work_submit_token;
  }

let public_channel (c : Pushbuf.channel) =
  {
    ring = Mmio.address c.ring;
    entries = c.entries;
    gp_get = Mmio.address c.gp_get;
    gp_put = Mmio.address c.gp_put;
    put = Mmio.address c.put;
    doorbell = Mmio.address c.doorbell;
    token = c.token;
  }

let budget n =
  match n.gpu with
  | Pci_gpu p -> Page_table.memory (Nvdev.mm p.nvdev)
  | Kernel_gpu g ->
      let module F = D.Fb_info in
      let (module R : D.RELEASE) = g.c.release in
      let module G = R.Fb_get_info in
      let p = P.create G.sizeof in
      P.set p G.fb_info_list_size 1;
      P.set p
        (P.elt_field G.fb_info_list 0 F.index)
        D.nv2080_ctrl_fb_info_index_heap_size;
      n.rm.control n.obj.subdevice D.nv2080_ctrl_cmd_fb_get_info_v2 (Some p);
      P.get p (P.elt_field G.fb_info_list 0 F.data) * 1024

(* Creates the device's RM objects, channels and runtime memory, and its
   {!Nx_device.t}. [doorbell] maps the usermode page, given the usermode
   class. *)
let setup ~index ~gpu ~(rm : Rm.t) ~instance ~doorbell ~classes =
  let alloc ~parent cls f size =
    let p = P.create size in
    f p;
    rm.alloc ~parent cls (Some p)
  in
  let device =
    alloc ~parent:rm.root D.nv01_device_0
      (fun p ->
        P.set p D.Nv0080_alloc.device_id instance;
        P.set p D.Nv0080_alloc.h_client_share rm.root;
        P.set p D.Nv0080_alloc.va_mode
          D.nv_device_allocation_vamode_optional_multiple_vaspaces)
      D.Nv0080_alloc.sizeof
  in
  let subdevice =
    alloc ~parent:device D.nv20_subdevice_0 ignore D.Nv2080_alloc.sizeof
  in
  let virtmem =
    alloc ~parent:device D.nv01_memory_virtual
      (fun p -> P.set p D.Memory_virtual_alloc.limit 0x1ffffffffffff)
      D.Memory_virtual_alloc.sizeof
  in
  (match gpu with
  | Kernel_gpu g ->
      g.device <- device;
      g.virtmem <- virtmem
  | Pci_gpu _ -> ());
  let usermode_class, gpfifo_class, compute_class, dma_class = classes device in
  let doorbell = doorbell ~subdevice usermode_class in
  let boost = P.create D.Perf_boost.sizeof in
  P.set boost D.Perf_boost.duration 0xffff_ffff;
  P.set boost D.Perf_boost.flags
    (P.bits D.nv2080_ctrl_perf_boost_flags_cuda
       D.nv2080_ctrl_perf_boost_flags_cuda_yes
    lor P.bits D.nv2080_ctrl_perf_boost_flags_cuda_priority
          D.nv2080_ctrl_perf_boost_flags_cuda_priority_high
    lor P.bits D.nv2080_ctrl_perf_boost_flags_cmd
          D.nv2080_ctrl_perf_boost_flags_cmd_boost_to_max);
  rm.control subdevice D.nv2080_ctrl_cmd_perf_boost (Some boost);
  let (module R : D.RELEASE) =
    match gpu with Kernel_gpu g -> g.Nvk.c.release | Pci_gpu _ -> Gsp.release
  in
  let vaspace =
    alloc ~parent:device D.fermi_vaspace_a
      (fun p ->
        P.set p R.Vaspace_alloc.va_base 0x1000;
        P.set p R.Vaspace_alloc.va_size 0x1fffffb000000;
        P.set p R.Vaspace_alloc.flags
          (D.nv_vaspace_allocation_flags_enable_page_faulting
         lor D.nv_vaspace_allocation_flags_is_externally_owned))
      R.Vaspace_alloc.sizeof
  in
  let unreachable =
    match gpu with
    | Kernel_gpu g ->
        let peers =
          List.filter_map
            (fun (i, n) ->
              match n.gpu with
              | Kernel_gpu g' -> Some (i, g')
              | Pci_gpu _ -> None)
            (Atomic.get opened)
        in
        let refused =
          Nvk.register g ~subdevice ~vaspace ~peers:(List.map snd peers)
        in
        List.filter_map
          (fun (i, g') -> if List.memq g' refused then Some i else None)
          peers
    | Pci_gpu _ -> []
  in
  let group =
    alloc ~parent:device D.kepler_channel_group_a
      (fun p ->
        P.set p D.Channel_group_alloc.engine_type D.nv2080_engine_type_graphics)
      D.Channel_group_alloc.sizeof
  in
  let ctxshare =
    alloc ~parent:group D.fermi_context_share_a
      (fun p ->
        P.set p D.Ctxshare_alloc.h_va_space vaspace;
        P.set p D.Ctxshare_alloc.flags
          D.nv_ctxshare_allocation_flags_subcontext_async)
      D.Ctxshare_alloc.sizeof
  in
  let gpcs, tpcs, sms, warps, sm =
    match gr_info gpu rm subdevice with
    | [ a; b; c; d; e ] -> (a, b, c, d, e)
    | _ -> assert false (* five indices asked *)
  in
  let props =
    {
      sm_version = sm;
      sass_version = ((sm land 0xf00) lsr 4) lor (sm land 0xf);
      gpcs;
      tpcs_per_gpc = tpcs;
      sms_per_tpc = sms;
      warps_per_sm = warps;
      gpfifo_class;
      compute_class;
      dma_class;
    }
  in
  let n =
    {
      index;
      gpu;
      rm;
      hw = Mutex.create ();
      allocs = Int_map.empty;
      borrows = Hashtbl.create 16;
      images = Hashtbl.create 8;
      kernels = Hashtbl.create 16;
      unreachable;
      obj =
        {
          device;
          subdevice;
          group;
          ctxshare;
          compute_channel = 0;
          debugger = 0;
        };
      props;
      ring = None;
      channels = None;
      handles = None;
      local_lock = Mutex.create ();
      local = None;
      dev = None;
    }
  in
  (* the channels, their put words, and the runtime's command segments *)
  let buf = keep n Channels "channels" (3 lsl 20) in
  let words = keep n Host "channels' positions" 0x1000 in
  let words_view = Option.get (host_view words) in
  Mmio.fill words_view 0 16 '\000';
  let segments = keep n Host "command segments" 0x10000 in
  n.ring <-
    Some (Pushbuf.ring (Option.get (host_view segments)) ~gpu:(va segments));
  let channel ~offset ~put ~compute engine =
    new_channel n ~buf
      ~put:(Mmio.sub words_view put 8)
      ~doorbell ~offset ~entries:0x10000 ~engine ~compute
  in
  let compute = channel ~offset:0 ~put:0 ~compute:true compute_class in
  let copy = channel ~offset:0x100000 ~put:8 ~compute:false dma_class in
  let s = P.create R.Group_schedule.sizeof in
  P.set s R.Group_schedule.b_enable 1;
  rm.control group D.nva06c_ctrl_cmd_gpfifo_schedule (Some s);
  n.channels <- Some (compute, copy);
  n

(* Binds the engines and the memory windows on the channels. *)
let bind_engines n =
  let d = Option.get n.dev in
  let timeline = Nx_device.timeline d in
  let tl =
    {
      Nx_device.host = Some (Nx_device.Buffer.host_address timeline);
      device = Nx_device.Buffer.address timeline;
      handle = 0n;
    }
  in
  let compute, copy = channels n in
  let hi v = (v lsr 32) land 0xffff_ffff and lo v = v land 0xffff_ffff in
  Nx_device.submit d ~touches:[] (fun v ->
      run n compute ~timeline:tl ~signal:Pushbuf.release v
        (Pushbuf.methods Pushbuf.compute D.nvc6c0_set_object
           [ n.props.compute_class ]
        @ Pushbuf.methods Pushbuf.compute
            D.nvc6c0_set_shader_local_memory_window_a
            [ hi local_window; lo local_window ]
        @ Pushbuf.methods Pushbuf.compute
            D.nvc6c0_set_shader_shared_memory_window_a
            [ hi shared_window; lo shared_window ]));
  Nx_device.submit d ~touches:[] (fun v ->
      run n copy ~timeline:tl ~signal:Pushbuf.release v
        (Pushbuf.methods Pushbuf.copy_engine D.nvc6c0_set_object
           [ n.props.dma_class ]))

let make_device n ?finalize () =
  let dev =
    Nx_device.make ~name:(name n.index) ~arch:(arch n.props.sm_version)
      ~budget:(budget n) ~memory:(allocator n Vram)
      ~host_memory:(allocator n Host) ~mapping:(mapping n)
      ~copy_queue:(copy_queue n) ~load:(load n)
      ~sleep:(fun _ -> check_faults n)
      ?finalize ()
  in
  n.dev <- Some dev;
  let compute, copy = channels n in
  n.handles <-
    Some
      {
        compute = public_channel compute;
        copy = public_channel copy;
        signal = Nx_device.Buffer.address (Nx_device.timeline dev);
        shared_window = Nativeint.of_int shared_window;
        local_window = Nativeint.of_int local_window;
        props = n.props;
      };
  bind_engines n

let open_kernel index =
  let g = Nvk.open_gpu index in
  let rm = Nvk.rm g.c in
  let doorbell ~subdevice cls =
    Mmio.v (Nativeint.add (Nvk.usermode g ~subdevice cls) 0x90n) 4
  in
  let n =
    setup ~index ~gpu:(Kernel_gpu g) ~rm ~instance:g.instance ~doorbell
      ~classes:(kernel_classes rm)
  in
  make_device n ();
  n

let pci_ids =
  [
    ( 0xff00,
      [
        0x2200;
        0x2400;
        0x2500;
        0x2600;
        0x2700;
        0x2800;
        0x2b00;
        0x2c00;
        0x2d00;
        0x2f00;
      ] );
  ]

let pci_buses () = Pci.scan ~vendor:0x10de ~class_:0x03 pci_ids

let firmware_url =
  "https://gitlab.com/kernel-firmware/linux-firmware/-/raw/" ^ D.firmware_commit
  ^ "/"

let fetch ?dir path =
  match List.assoc_opt path D.firmware_sha256 with
  | None -> failwith ("no pinned firmware " ^ path)
  | Some sha256 -> (
      match Firmware.get ?dir ~url:firmware_url ("nvidia/" ^ path) ~sha256 with
      | Ok s -> s
      | Error why -> failwith why)

(* Boots the GPU at [pci]: its firmware first, so that a missing image fails
   before the GPU is touched, then the memory below the GSP's region, the
   falcons and the GSP. *)
let boot ?firmware pci =
  let d = Nvdev.create pci in
  Falcon.wait_for_reset d;
  let dir = Nvdev.firmware_dir d in
  let gsp_fw = fetch ?dir:firmware "ga102/gsp/gsp-570.144.bin" in
  let bootloader_fw =
    fetch ?dir:firmware (dir ^ "/gsp/bootloader-570.144.bin")
  in
  let falcon_fw =
    fetch ?dir:firmware
      (if d.fmc_boot then dir ^ "/gsp/fmc-570.144.bin"
       else dir ^ "/gsp/booter_load-570.144.bin")
  in
  let layout = Gsp.images ~chip:(Nvdev.chip_name d) ~gsp_fw ~bootloader_fw in
  Nvdev.init_mm d
    ~top:
      (Gsp.managed_top ~vram:d.vram_size ~fmc:d.fmc_boot
         ~boot:(String.length layout.bootloader)
         ~image:(String.length layout.image));
  let flcn = Falcon.init_sw d ~firmware:falcon_fw in
  let g = Gsp.init_sw d flcn layout in
  Falcon.init_hw d flcn ~libos:g.libos ~wpr_meta:g.wpr_meta;
  Gsp.init_hw g;
  (d, g)

(* Leaves the GPU as its next open expects: a healthy GSP is told the driver
   unloads; a failed GPU, or one whose GSP cannot be told, stops reaching the
   memory the process releases, and its next open resets it. *)
let stop ~failed pci gsp =
  let off () = Nvdev.set_bus_master pci false in
  if failed then off ()
  else
    match Gsp.fini gsp with
    | () -> ()
    | exception e ->
        off ();
        raise e

(* Opens the GPU of [pci], which the process took. An open that fails after it
   touched the GPU stops the GPU's access to memory. *)
let open_taken ?firmware ~index pci =
  match
    Sysmem.reserve
      ~base:(Page_table.Space.base Nvdev.space)
      (Page_table.Space.length Nvdev.space);
    (try Pci.resize_bar pci 1 with Failure _ -> ());
    let nvdev, gsp = boot ?firmware pci in
    let memory = Pci_memory.create pci (Nvdev.mm nvdev) ~bar:1 in
    let rm = Gsp.rm gsp ~root:0xc1000000 in
    let root =
      rm.alloc ~parent:0 D.nv01_root (Some (P.create D.Nv0000_alloc.sizeof))
    in
    let rm = { rm with Rm.root } in
    let doorbell ~subdevice:_ _ =
      Mmio.sub (Pci.map_bar ~offset:0xbb0000 ~length:0x10000 pci 0) 0x90 4
    in
    let classes _ = (0, gsp.gpfifo_class, gsp.compute_class, gsp.dma_class) in
    let n =
      setup ~index
        ~gpu:(Pci_gpu { nvdev; gsp; memory; pci })
        ~rm ~instance:0 ~doorbell ~classes
    in
    make_device n
      ~finalize:(fun ~failed -> with_hw n (fun () -> stop ~failed pci gsp))
      ();
    n
  with
  | n -> n
  | exception e ->
      (try Nvdev.set_bus_master pci false with Failure _ -> ());
      raise e

(* A failed open gives the function back, so that a later one can take it. *)
let open_pci ?firmware index =
  let buses = pci_buses () in
  let bus =
    match List.nth_opt buses index with
    | Some b -> b
    | None ->
        failwith
          (Printf.sprintf "no GPU %d; there are %d NVIDIA GPUs" index
             (List.length buses))
  in
  let pci = Pci.take ~lock:"nv" bus in
  match open_taken ?firmware ~index pci with
  | n -> n
  | exception e ->
      Pci.release pci;
      raise e

(* The interface of the process, fixed by its first successful open. *)
let chosen = ref None
let lock = Mutex.create ()

let default () =
  match !chosen with
  | Some i -> i
  | None -> if Nvk.available () then Kernel else Pci

let count ?interface () =
  if not (linux ()) then 0
  else
    Mutex.protect lock (fun () ->
        match Option.value interface ~default:(default ()) with
        | Kernel -> ( try Nvk.count () with Sys_error _ | Failure _ -> 0)
        | Pci -> List.length (pci_buses ()))

let interface_name = function Kernel -> "the kernel driver" | Pci -> "PCI"

let get ?interface ?firmware i =
  if i < 0 then invalid_arg (Printf.sprintf "Nx_nv_device.get: %d < 0" i);
  if not (linux ()) then Error "NV: NVIDIA GPUs need Linux"
  else
    Mutex.protect lock (fun () ->
        let iface = Option.value interface ~default:(default ()) in
        match !chosen with
        | Some c when c <> iface ->
            Error
              (Printf.sprintf
                 "NV: this process reaches NVIDIA GPUs through %s, not %s"
                 (interface_name c) (interface_name iface))
        | _ -> (
            match List.assoc_opt i (Atomic.get opened) with
            | Some n -> Ok (Option.get n.dev)
            | None -> (
                match
                  match iface with
                  | Kernel -> open_kernel i
                  | Pci -> open_pci ?firmware i
                with
                | n ->
                    chosen := Some iface;
                    Atomic.set opened ((i, n) :: Atomic.get opened);
                    Ok (Option.get n.dev)
                | exception (Failure msg | Sys_error msg) -> Error ("NV: " ^ msg)
                )))

let v ?interface ?firmware i =
  match get ?interface ?firmware i with
  | Ok d -> d
  | Error msg -> invalid_arg msg

let nv fn d =
  match nv_of d with
  | Some n -> n
  | None ->
      invalid_arg
        (Printf.sprintf "Nx_nv_device.%s: %s is not an NV device" fn
           (Nx_device.name d))

let interface d =
  match (nv "interface" d).gpu with Kernel_gpu _ -> Kernel | Pci_gpu _ -> Pci

let handles d = Option.get (nv "handles" d).handles

let kernel p =
  let n = nv "kernel" (Nx_device.Program.device p) in
  match Hashtbl.find_opt n.kernels (Nx_device.Program.handle p) with
  | Some k -> k
  | None -> invalid_arg "Nx_nv_device.kernel: the program is not an NV kernel"

let local_memory d bytes =
  let n = nv "local_memory" d in
  Mutex.protect n.local_lock (fun () ->
      match n.local with
      | Some (b, per) when per >= bytes ->
          {
            address = Nx_device.Buffer.address b;
            bytes = Nx_device.Buffer.nbytes b;
            per_thread = per;
          }
      | _ ->
          let p = n.props in
          let per = round_up (Int.max bytes 0) 32 in
          let per_tpc =
            round_up
              (round_up (per * 32) 0x200 * p.warps_per_sm * p.sms_per_tpc)
              0x8000
          in
          let size = round_up (per_tpc * p.tpcs_per_gpc * p.gpcs) 0x20000 in
          let b = Nx_device.Buffer.create d Nx_dtype.Scalar.UInt8 size in
          let addr = Nativeint.to_int (Nx_device.Buffer.address b) in
          let timeline = Nx_device.timeline d in
          let tl =
            {
              Nx_device.host = Some (Nx_device.Buffer.host_address timeline);
              device = Nx_device.Buffer.address timeline;
              handle = 0n;
            }
          in
          let hi v = (v lsr 32) land 0xffff_ffff
          and lo v = v land 0xffff_ffff in
          let compute, _ = channels n in
          Nx_device.submit d ~touches:[] (fun v ->
              run n compute ~timeline:tl ~signal:Pushbuf.release v
                (Pushbuf.methods Pushbuf.compute
                   D.nvc6c0_set_shader_local_memory_a
                   [ hi addr; lo addr ]
                @ Pushbuf.methods Pushbuf.compute
                    D.nvc6c0_set_shader_local_memory_non_throttled_a
                    [ hi per_tpc; lo per_tpc; 0xff ]));
          n.local <- Some (b, per);
          { address = Nativeint.of_int addr; bytes = size; per_thread = per })

let invalidate_caches d =
  let n = nv "invalidate_caches" d in
  match n.gpu with
  | Pci_gpu _ ->
      n.rm.control n.obj.subdevice
        D.nv2080_ctrl_cmd_internal_bus_flush_with_sysmembar None
  | Kernel_gpu _ ->
      let module F = D.Flush_gpu_cache in
      let p = P.create F.sizeof in
      P.set p F.flags
        (P.bits D.nv2080_ctrl_fb_flush_gpu_cache_flags_write_back
           D.nv2080_ctrl_fb_flush_gpu_cache_flags_write_back_yes
        lor P.bits D.nv2080_ctrl_fb_flush_gpu_cache_flags_invalidate
              D.nv2080_ctrl_fb_flush_gpu_cache_flags_invalidate_yes
        lor P.bits D.nv2080_ctrl_fb_flush_gpu_cache_flags_flush_mode
              D.nv2080_ctrl_fb_flush_gpu_cache_flags_flush_mode_full_cache);
      n.rm.control n.obj.subdevice D.nv2080_ctrl_cmd_fb_flush_gpu_cache (Some p)
