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
module Remote = Nx_device_support.Remote
module Firmware = Nx_device_support.Firmware
module Driver = Nx_device.Driver
module Region = Driver.Region

external linux : unit -> bool = "caml_nx_nv_linux"

type interface = Kernel | Pci

type channel = {
  ring : Nx_device.Buffer.t;
  gp_get : Nx_device.Buffer.t;
  gp_put : Nx_device.Buffer.t;
  put : Nx_device.Buffer.t;
  doorbell : Nx_device.Buffer.t;
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

type kernel = {
  image : Nx_device.Buffer.t;
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

type t = {
  index : int;
  machine : Nx_device.t; (* the host of the GPU's machine *)
  gpu : gpu;
  rm : Rm.t;
  hw : Mutex.t; (* the GPU's page tables and memory objects *)
  mutable allocs : alloc Int_map.t; (* the device's own memory, by address *)
  borrows : (int, mem) Hashtbl.t;
      (* by address, host memory mapped for a copy or a borrow: under [Pci] the
         mapping, under [Kernel] another device's memory (the driver keeps
         borrows) *)
  images : (string, Nx_device.Buffer.t) Hashtbl.t; (* uploaded cubins *)
  kernels : (nativeint, kernel) Hashtbl.t; (* by entry, under [hw] *)
  unreachable : int list; (* the GPUs peer access was refused with at open *)
  obj : objects;
  props : props;
  mutable ring : Pushbuf.ring option;
  mutable channels : (Pushbuf.channel * Pushbuf.channel) option;
  mutable public : (channel * channel) option;
      (* the compute and copy channels, as their words' buffers *)
  local_lock : Mutex.t;
  mutable local : (Nx_device.Buffer.t * int) option;
  mutable dev : Nx_device.t option;
  mutable ready : bool; (* whether its open succeeded *)
}

(* Memory of a device and the peers it is mapped on, which their borrows and
   transfers add to while the owner is not taken. *)
and alloc = { mem : mem; peers : t list Atomic.t }

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
  n.allocs <- Int_map.add (va mem) { mem; peers = Atomic.make [] } n.allocs

(* Frees [mem] after unmapping it from the peers that transfers mapped it on:
   their copies were waited for, so none still uses it. *)
let release n mem =
  Option.iter
    (fun r ->
      List.iter (fun peer -> unmap_from peer mem) (Atomic.exchange r.peers []))
    (Int_map.find_opt (va mem) n.allocs);
  n.allocs <- Int_map.remove (va mem) n.allocs;
  free_mem n mem

let find n x =
  match Int_map.find_last_opt (fun s -> s <= x) n.allocs with
  | Some (s, r) when x < s + size r.mem -> Some r
  | _ -> None

(* The [bytes] bytes of [mem] from its start, at the same address for the host
   and the GPU when the host addresses them. *)
let region_of mem bytes =
  let va = Nativeint.of_int (va mem) in
  Region.v ?host:(Option.map Mmio.address (host_view mem)) ~handle:va va bytes

let allocator n kind =
  let alloc bytes =
    Option.map
      (fun mem ->
        register n mem;
        region_of mem bytes)
      (alloc_mem n kind bytes)
  in
  let free r =
    Option.iter
      (fun r -> release n r.mem)
      (Int_map.find_opt (Nativeint.to_int (Region.handle r)) n.allocs)
  in
  { Driver.alloc; free }

(* Memory the runtime keeps for the device's life. *)
let need what = function
  | Some m -> m
  | None -> failwith ("no GPU memory for the " ^ what)

let keep n kind what bytes =
  let m = need what (alloc_mem n kind bytes) in
  register n m;
  m

(* The opened GPUs, by the host of their machine and index there. *)
let opened : ((Nx_device.t * int) * t) list Atomic.t = Atomic.make []

let nv_of d =
  List.find_map
    (fun (_, n) ->
      match n.dev with Some d' when Nx_device.equal d d' -> Some n | _ -> None)
    (Atomic.get opened)

(* Host memory another device of [n]'s interface and machine allocated at [x],
   with that device. *)
let foreign n x =
  List.find_map
    (fun (_, n') ->
      match (n.gpu, n'.gpu) with
      | (Kernel_gpu _, Kernel_gpu _ | Pci_gpu _, Pci_gpu _)
        when n' != n && n'.machine == n.machine -> (
          match find n' x with
          | Some r when Option.is_some (host_view r.mem) -> Some (n', r.mem)
          | _ -> None)
      | _ -> None)
    (Atomic.get opened)

(* Host memory the GPU addresses at the host's address: another device's host
   memory, which that device keeps locked and described, mapped as its memory;
   and borrowed memory. *)
let mapping n =
  let map x bytes =
    let memory = Region.v ~host:x ~handle:x x bytes in
    let key = Nativeint.to_int x in
    with_hw n (fun () ->
        match (n.gpu, foreign n key) with
        | Kernel_gpu g, Some (_, (Kernel_mem m as mem)) -> (
            match Nvk.map_peer g m with
            | exception Failure why -> Error why
            | () ->
                Hashtbl.replace n.borrows key mem;
                Ok memory)
        | Pci_gpu p, Some ({ gpu = Pci_gpu owner; _ }, Pci_mem m) ->
            Result.map
              (fun mem ->
                Hashtbl.replace n.borrows key (Pci_mem mem);
                memory)
              (Pci_memory.map_peer p.memory owner.memory m)
        | Kernel_gpu g, _ ->
            Result.map (fun () -> memory) (Nvk.map_host g x bytes)
        | Pci_gpu p, _ ->
            Result.map
              (fun mem ->
                Hashtbl.replace n.borrows key (Pci_mem mem);
                memory)
              (Pci_memory.map_host p.memory x bytes))
  in
  let unmap r =
    let key = Nativeint.to_int (Region.handle r) in
    with_hw n (fun () ->
        let mapped = Hashtbl.find_opt n.borrows key in
        Hashtbl.remove n.borrows key;
        match (n.gpu, mapped) with
        | Kernel_gpu g, Some (Kernel_mem m) -> Nvk.unmap_peer g m
        | Kernel_gpu g, _ -> Nvk.unmap_host g (Region.handle r)
        | Pci_gpu p, Some (Pci_mem mem) -> Pci_memory.unmap p.memory mem
        | Pci_gpu _, _ -> ())
  in
  Driver.Pages { map; unmap }

(* Channels *)

let timeout n =
  Option.fold ~none:Driver.default_timeout ~some:Nx_device.timeout n.dev

(* The range at [x] of the GPU's machine. *)
let range n x bytes =
  match n.gpu with
  | Pci_gpu { pci; _ } -> (
      match Pci.remote pci with
      | Some r -> Mmio.remote (Remote.access r) x bytes
      | None -> Mmio.v x bytes)
  | Kernel_gpu _ -> Mmio.v x bytes

let signal_word n timeline =
  range n (Option.get (Region.host_address timeline)) 8

(* Submits the work [body] for the timeline value [v] on [ch]: after [v - 1],
   ending by signaling [v] with [signal]. *)
let run n ch ~timeline ~signal v body =
  let addr = Nativeint.to_int (Region.address timeline) in
  let word = signal_word n timeline in
  let words = Pushbuf.acquire addr (v - 1) @ body @ signal addr v in
  let seg =
    Pushbuf.segment (Option.get n.ring)
      ~signaled:(fun () -> Int64.to_int (Mmio.get64 word 0))
      ~timeout_ms:(timeout n) v words
  in
  Pushbuf.submit ch ~timeout_ms:(timeout n) seg (List.length words)

let channels n = Option.get n.channels

(* Whether each channel has room for half its ring, the most a submission writes
   into it: a ring whose entries are all written reads as empty, so at most
   [entries - 1] may be unfetched. *)
let room n () =
  match n.channels with
  | None -> true
  | Some (compute, copy) ->
      let half (c : Pushbuf.channel) =
        let put = Int64.to_int (Mmio.get64 c.put 0) in
        2 * ((put - Mmio.get32 c.gp_get 0 + c.entries) mod c.entries)
        < c.entries
      in
      half compute && half copy

(* The pairs of GPUs, (lower index, higher), whose peer access the driver
   refused: it is one per pair, so copies between them go through host memory in
   both directions. *)
let refused : (int * int) list Atomic.t = Atomic.make []
let pair i j = (Int.min i j, Int.max i j)

(* Whether [n]'s copy engine reaches the memory of [peer]. *)
let reaches n peer =
  match (n.gpu, peer.gpu) with
  | Kernel_gpu _, Kernel_gpu _ ->
      not (List.mem (pair n.index peer.index) (Atomic.get refused))
  | Pci_gpu _, Pci_gpu p ->
      n.machine == peer.machine && not (Pci_memory.small_bar p.memory)
  | _ -> false

(* Maps [peer]'s allocation that holds [x] on [n], at its first use, at the same
   address; freeing it unmaps it. *)
let map_peer n peer x =
  match find peer (Nativeint.to_int x) with
  | None -> Error "no memory of the other GPU"
  | Some r when List.memq n (Atomic.get r.peers) -> Ok ()
  | Some r ->
      Result.map
        (fun () ->
          let rec push () =
            let l = Atomic.get r.peers in
            if not (Atomic.compare_and_set r.peers l (n :: l)) then push ()
          in
          push ())
        (with_hw n (fun () ->
             match (n.gpu, peer.gpu, r.mem) with
             | Kernel_gpu g, _, Kernel_mem m -> (
                 match Nvk.map_peer g m with
                 | () -> Ok ()
                 | exception Failure why -> Error why)
             | Pci_gpu p, Pci_gpu p', Pci_mem m ->
                 Result.map ignore (Pci_memory.map_peer p.memory p'.memory m)
             | _ -> Error "a peer of another interface"))

(* Borrows of another NV GPU's memory that [n] reaches. *)
let peer n d' r =
  match nv_of d' with
  | Some peer when reaches n peer ->
      Result.map (fun () -> r) (map_peer n peer (Region.address r))
  | Some _ -> Error "the GPUs do not reach each other's memory"
  | None -> Error "memory of another vendor"

let queue n ~timeline =
  let submit ~dst ~src bytes ~signal:v =
    let _, copy = channels n in
    run n copy ~timeline ~signal:Pushbuf.copy_release v
      (Pushbuf.copy ~dst:(Nativeint.to_int dst) ~src:(Nativeint.to_int src)
         bytes)
  in
  (* A transfer maps the destination's allocation on this GPU. *)
  let transfer d' =
    match nv_of d' with
    | Some peer when reaches n peer ->
        Some
          (fun ~dst ~src bytes ~signal ->
            (match map_peer n peer dst with
            | Ok () -> ()
            | Error why -> failwith why);
            submit ~dst ~src bytes ~signal)
    | _ -> None
  in
  let stamp ~slot ~signal:v =
    let _, copy = channels n in
    run n copy ~timeline ~signal:Pushbuf.copy_release v
      (Pushbuf.copy_stamp (Nativeint.to_int slot))
  in
  {
    Driver.copy = submit;
    transfer;
    stamp;
    clock = Device_clock { hz = 1_000_000_000 };
  }

(* How the other functions of the GPU's machine reach its memory: its own
   through the memory BAR, and system memory at its pages. *)
let dma n r =
  match (n.gpu, find n (Nativeint.to_int (Region.address r))) with
  | Kernel_gpu _, _ -> Error "the kernel driver's memory is not described"
  | Pci_gpu _, None -> Error "no allocation of this GPU"
  | Pci_gpu { memory; pci; _ }, Some { mem = Pci_mem pm; _ } -> (
      let map = pm.mapping in
      match map.space with
      | Page_table.Sys -> Ok { Driver.bus = Pci.bus pci; pages = map.pages }
      | Phys when Pci_memory.small_bar memory ->
          Error "the memory BAR is too small for other functions to reach it"
      | Phys ->
          let start = fst (Pci.bar pci 1) in
          Ok
            {
              Driver.bus = Pci.bus pci;
              pages = List.map (fun (p, k) -> (p + start, k)) map.pages;
            }
      | Peer -> Error "memory of another GPU")
  | Pci_gpu _, Some { mem = Kernel_mem _; _ } ->
      Error "memory of another interface"

(* Programs *)

(* The cubin [binary], relocated and uploaded once, as a buffer of the
   device. *)
let upload n binary (c : Cubin.t) =
  match Hashtbl.find_opt n.images binary with
  | Some image -> Some image
  | None ->
      Option.map
        (fun mem ->
          register n mem;
          Mmio.write
            (Option.get (host_view mem))
            0
            (Cubin.relocate c ~base:(va mem));
          Mmio.barrier ();
          let bytes = String.length c.image in
          let image =
            Driver.buffer (Option.get n.dev) (region_of mem bytes)
              Nx_dtype.Scalar.UInt8 bytes
          in
          Hashtbl.replace n.images binary image;
          image)
        (alloc_mem n Visible (String.length c.image))

(* A cubin the device cannot run, or has no memory for, is refused, and the
   device stays usable. *)
let load n ~binary ~entry:name =
  match Cubin.load binary ~name with
  | exception Failure why -> Error why
  | c -> (
      match upload n binary c with
      | None -> Error "no GPU memory for the program"
      | Some image ->
          let at off =
            Nativeint.add
              (Nx_device.Buffer.address image)
              (Nativeint.of_int off)
          in
          let k =
            {
              image;
              entry = at c.entry;
              code_bytes = c.code_bytes;
              registers = c.registers;
              shared_bytes = c.shared_bytes;
              local_bytes = c.local_bytes;
              param_offset = c.param_offset;
              banks =
                List.map (fun (i, off, bytes) -> (i, at off, bytes)) c.banks;
              max_threads = c.max_threads;
            }
          in
          with_hw n (fun () -> Hashtbl.replace n.kernels k.entry k);
          Ok k.entry)

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

(* The faults a wait looks for: under [Pci], the GSP's messages are handled
   first, and an error it reported fails the device. *)
let check_faults n =
  (match n.gpu with Pci_gpu p -> Gsp.poll p.gsp | Kernel_gpu _ -> ());
  match fault_report n with
  | [] -> ()
  | report -> failwith (String.concat "\n" report)

(* The sleep of a wait whose timeline stayed still for 200 ms: it checks for
   faults, then polls the signal word every millisecond for at most [ms], so
   that the check runs once a sleep, at most every 200 ms. *)
let sleep n ~timeline ms =
  check_faults n;
  let word = signal_word n timeline in
  let seen = Mmio.get64 word 0 and until = Nvdev.now_ms () + ms in
  while Mmio.get64 word 0 = seen && Nvdev.now_ms () < until do
    Unix.sleepf 0.001
  done

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
   USERD after its ring, bound to [cls]. [keep] allocates its memory. *)
let new_channel n ~keep ~taken ~buf ~put ~doorbell ~offset ~entries ~engine
    ~compute =
  let rm = n.rm in
  let notifier = keep Uncached "channel's error notifier" (48 lsl 20) in
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
  | Kernel_gpu g ->
      let base = Nvk.register_channel g ch in
      taken (fun () -> Nvk.release_channel g base)
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

(* A channel's words, as buffers of [dev] at addresses the host and the GPU
   share. *)
let public_channel dev (c : Pushbuf.channel) =
  let buffer m s n =
    let x = Mmio.address m in
    Driver.buffer dev (Region.v ~host:x x (Mmio.length m)) s n
  in
  {
    ring = buffer c.ring Nx_dtype.Scalar.UInt64 c.entries;
    gp_get = buffer c.gp_get Nx_dtype.Scalar.UInt32 1;
    gp_put = buffer c.gp_put Nx_dtype.Scalar.UInt32 1;
    put = buffer c.put Nx_dtype.Scalar.UInt64 1;
    doorbell = buffer c.doorbell Nx_dtype.Scalar.UInt32 1;
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

(* Creates the device's RM objects, channels and runtime memory. [doorbell] maps
   the usermode page, given the usermode class. [taken undo] is given how to
   give back each thing the process holds after it took it, so that a failed
   open can give everything back, last first. *)
let setup ~taken ~machine ~index ~gpu ~(rm : Rm.t) ~instance ~doorbell ~classes
    =
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
  (* freeing the device frees every object under it *)
  taken (fun () -> rm.free ~parent:rm.root device);
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
            (fun ((_, i), n) ->
              match n.gpu with
              | Kernel_gpu g' -> Some (i, g')
              | Pci_gpu _ -> None)
            (Atomic.get opened)
        in
        let refused =
          Nvk.register g ~subdevice ~vaspace ~peers:(List.map snd peers)
        in
        taken (fun () -> Nvk.unregister g);
        List.filter_map
          (fun (i, g') -> if List.memq g' refused then Some i else None)
          peers
    | Pci_gpu _ -> []
  in
  let group =
    alloc ~parent:device D.kepler_channel_group_a
      (fun p ->
        P.set p R.Channel_group_alloc.engine_type D.nv2080_engine_type_graphics)
      R.Channel_group_alloc.sizeof
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
      machine;
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
      public = None;
      local_lock = Mutex.create ();
      local = None;
      dev = None;
      ready = false;
    }
  in
  let keep kind what bytes =
    let m = keep n kind what bytes in
    taken (fun () -> free_mem n m);
    m
  in
  (* the channels, their put words, and the runtime's command segments *)
  let buf = keep Channels "channels" (3 lsl 20) in
  let words = keep Host "channels' positions" 0x1000 in
  let words_view = Option.get (host_view words) in
  Mmio.fill words_view 0 16 '\000';
  let segments = keep Host "command segments" 0x10000 in
  n.ring <-
    Some (Pushbuf.ring (Option.get (host_view segments)) ~gpu:(va segments));
  let channel ~offset ~put ~compute engine =
    new_channel n ~keep ~taken ~buf
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
  let tl = Region.of_buffer (Nx_device.signal_word d) in
  let compute, copy = channels n in
  let hi v = (v lsr 32) land 0xffff_ffff and lo v = v land 0xffff_ffff in
  Nx_device.submit [ d ] ~touches:[] (fun s ->
      run n compute ~timeline:tl ~signal:Pushbuf.release
        (Nx_device.Submission.value s d)
        (Pushbuf.methods Pushbuf.compute D.nvc6c0_set_object
           [ n.props.compute_class ]
        @ Pushbuf.methods Pushbuf.compute
            D.nvc6c0_set_shader_local_memory_window_a
            [ hi local_window; lo local_window ]
        @ Pushbuf.methods Pushbuf.compute
            D.nvc6c0_set_shader_shared_memory_window_a
            [ hi shared_window; lo shared_window ]));
  Nx_device.submit [ d ] ~touches:[] (fun s ->
      run n copy ~timeline:tl ~signal:Pushbuf.release
        (Nx_device.Submission.value s d)
        (Pushbuf.methods Pushbuf.copy_engine D.nvc6c0_set_object
           [ n.props.dma_class ]))

(* The GPU's memory the host writes through BAR1, for mapped buffers. Under the
   driver-less interface a BAR of 256 MiB gives system memory for memory the
   host addresses: the GPU has no window, and mapped buffers are pinned
   memory. *)
let mapped_allocator n =
  match n.gpu with
  | Pci_gpu p when Pci_memory.small_bar p.memory -> None
  | _ -> Some (allocator n Visible)

let make_device n ?finalize () =
  let dev =
    Driver.device ~name:(name n.index) ~arch:(arch n.props.sm_version)
      ~host:n.machine ~budget:(budget n)
      ~completion:(Sleep (sleep n))
      ~load:(load n) ~peer:(peer n) ~dma:(dma n) ~room:(room n) ?finalize
      (Device_local
         {
           memory = allocator n Vram;
           host_memory = allocator n Host;
           mapped = mapped_allocator n;
           mapping = mapping n;
           queue = queue n;
         })
  in
  n.dev <- Some dev;
  let compute, copy = channels n in
  n.public <- Some (public_channel dev compute, public_channel dev copy);
  bind_engines n

(* A failed open gives back what it took, so that a later one can open the GPU:
   an error giving something back does not hide the open's. *)
let open_kernel index =
  let g = Nvk.open_gpu index in
  let rm = Nvk.rm g.c in
  let undo = ref [] in
  let taken f = undo := f :: !undo in
  taken (fun () -> Nvk.close_gpu g);
  let doorbell ~subdevice cls =
    let usermode = Nvk.usermode g ~subdevice cls in
    taken (fun () -> Nvk.release_usermode g usermode);
    Mmio.v (Nativeint.add usermode 0x90n) 4
  in
  match
    let n =
      setup ~taken ~machine:Nx_device.host ~index ~gpu:(Kernel_gpu g) ~rm
        ~instance:g.instance ~doorbell ~classes:(kernel_classes rm)
    in
    make_device n ();
    n
  with
  | n ->
      Atomic.set refused
        (List.map (pair index) n.unreachable @ Atomic.get refused);
      n
  | exception e ->
      List.iter (fun f -> try f () with Failure _ -> ()) !undo;
      raise e

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

let pci_buses ?remote () = Pci.scan ?remote ~vendor:0x10de ~class_:0x03 pci_ids

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
  let d = Nvdev.start d in
  Falcon.wait_for_reset d;
  let top =
    Gsp.managed_top ~vram:d.vram_size ~fmc:d.fmc_boot
      ~boot:(String.length layout.bootloader)
      ~image:(String.length layout.image)
  in
  Nvdev.init_mm d ~top;
  let flcn = Falcon.init_sw d ~firmware:falcon_fw in
  let g = Gsp.init_sw d flcn layout in
  Falcon.init_hw d flcn ~libos:g.libos ~wpr_meta:g.wpr_meta;
  Gsp.init_hw g;
  if d.fmc_boot then Gsp.check_region d ~top;
  (d, g)

(* Leaves the GPU as its next open expects, which resets it: a healthy GSP is
   told the driver unloads; then the GPU, failed or not, stops reaching the
   memory the process releases, which its GSP was given. *)
let stop ~failed pci gsp =
  Fun.protect
    ~finally:(fun () -> Nvdev.set_bus_master pci false)
    (fun () -> if not failed then Gsp.fini gsp)

(* Opens the GPU of [pci], which the process took. An open that fails after it
   touched the GPU stops the GPU's access to memory. *)
let open_taken ?firmware ~machine ~index pci =
  match
    Pci.reserve pci
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
    (* the next open resets the GPU: what this one took on it needs no giving
       back *)
    let n =
      setup ~taken:ignore ~machine ~index
        ~gpu:(Pci_gpu { nvdev; gsp; memory; pci })
        ~rm ~instance:0 ~doorbell ~classes
    in
    make_device n
      ~finalize:(fun ~failed ->
        if n.ready then with_hw n (fun () -> stop ~failed pci gsp))
      ();
    n
  with
  | n -> n
  | exception e ->
      (try Nvdev.set_bus_master pci false with Failure _ -> ());
      raise e

(* A failed open gives the function back, so that a later one can take it. *)
let open_pci ?firmware ~machine index =
  let remote = Nx_remote_device.remote machine in
  let buses = pci_buses ?remote () in
  let bus =
    match List.nth_opt buses index with
    | Some b -> b
    | None ->
        failwith
          (Printf.sprintf "no GPU %d; there are %d NVIDIA GPUs" index
             (List.length buses))
  in
  let pci = Pci.take ?remote ~lock:"nv" bus in
  match open_taken ?firmware ~machine ~index pci with
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

(* Raises [Lost] for [host] if its machine can no longer be reached: the
   synchronization of [host] meets the failed connection, which loses it. *)
let check_reach host =
  match Nx_remote_device.remote host with
  | Some r when Remote.failed r <> None -> Nx_device.synchronize host
  | Some _ | None -> ()

let count ?(host = Nx_device.host) ?interface () =
  match Nx_remote_device.remote host with
  | Some remote -> (
      try List.length (pci_buses ~remote ())
      with Failure _ as e ->
        check_reach host;
        raise e)
  | None when not (linux ()) -> 0
  | None ->
      Mutex.protect lock (fun () ->
          match Option.value interface ~default:(default ()) with
          | Kernel -> ( try Nvk.count () with Sys_error _ | Failure _ -> 0)
          | Pci -> List.length (pci_buses ()))

let interface_name = function Kernel -> "the kernel driver" | Pci -> "PCI"

(* Why GPU [i] of the machine of [machine] cannot be opened. *)
let refuse ~machine i why =
  check_reach machine;
  Error (Driver.name ~host:machine (name i) ^ ": " ^ why)

(* Opens [i] through [iface] on the machine of [machine], once. *)
let open_gpu ~machine ~iface ?firmware i =
  match List.assoc_opt (machine, i) (Atomic.get opened) with
  | Some n -> Ok (Option.get n.dev)
  | None -> (
      match
        match iface with
        | Kernel -> open_kernel i
        | Pci -> open_pci ?firmware ~machine i
      with
      | n ->
          n.ready <- true;
          if machine == Nx_device.host then chosen := Some iface;
          Atomic.set opened (((machine, i), n) :: Atomic.get opened);
          Ok (Option.get n.dev)
      | exception (Failure why | Sys_error why | Invalid_argument why) ->
          refuse ~machine i why
      | exception Unix.Unix_error (e, f, arg) ->
          refuse ~machine i
            (Printf.sprintf "%s %s: %s" f arg (Unix.error_message e)))

let get ?(host = Nx_device.host) ?interface ?firmware i =
  if i < 0 then invalid_arg (Printf.sprintf "Nx_nv_device.get: %d < 0" i);
  let remote = Nx_remote_device.remote host in
  let refuse = refuse ~machine:host i in
  Mutex.protect lock (fun () ->
      match (remote, interface) with
      | Some _, Some Kernel ->
          refuse "another machine's GPUs are reached over PCI"
      | Some _, _ -> open_gpu ~machine:host ~iface:Pci ?firmware i
      | None, _ when not (linux ()) -> refuse "NVIDIA GPUs need Linux"
      | None, _ -> (
          let iface = Option.value interface ~default:(default ()) in
          match !chosen with
          | Some c when c <> iface ->
              refuse
                (Printf.sprintf
                   "this process reaches NVIDIA GPUs through %s, not %s"
                   (interface_name c) (interface_name iface))
          | _ -> open_gpu ~machine:host ~iface ?firmware i))

let v ?host ?interface ?firmware i =
  match get ?host ?interface ?firmware i with
  | Ok d -> d
  | Error msg -> failwith msg

let of_device = nv_of
let compute n = fst (Option.get n.public)
let copy n = snd (Option.get n.public)
let shared_window _ = Nativeint.of_int shared_window
let local_window _ = Nativeint.of_int local_window
let props n = n.props

let kernel p =
  Option.bind
    (nv_of (Nx_device.Program.device p))
    (fun n ->
      with_hw n (fun () ->
          Hashtbl.find_opt n.kernels (Nx_device.Program.handle p)))

let local_memory n bytes =
  let d = Option.get n.dev in
  Mutex.protect n.local_lock (fun () ->
      match n.local with
      | Some (b, per) when per >= bytes ->
          {
            address = Nx_device.Buffer.address b;
            bytes = Nx_device.Buffer.nbytes b;
            per_thread = per;
          }
      | None when bytes <= 0 -> { address = 0n; bytes = 0; per_thread = 0 }
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
          let tl = Region.of_buffer (Nx_device.signal_word d) in
          let hi v = (v lsr 32) land 0xffff_ffff
          and lo v = v land 0xffff_ffff in
          let compute, _ = channels n in
          Nx_device.submit [ d ] ~touches:[] (fun s ->
              run n compute ~timeline:tl ~signal:Pushbuf.release
                (Nx_device.Submission.value s d)
                (Pushbuf.methods Pushbuf.compute
                   D.nvc6c0_set_shader_local_memory_a
                   [ hi addr; lo addr ]
                @ Pushbuf.methods Pushbuf.compute
                    D.nvc6c0_set_shader_local_memory_non_throttled_a
                    [ hi per_tpc; lo per_tpc; 0xff ]));
          n.local <- Some (b, per);
          { address = Nativeint.of_int addr; bytes = size; per_thread = per })

let invalidate_caches n =
  match n.gpu with
  | Pci_gpu _ ->
      n.rm.control n.obj.subdevice
        D.nv2080_ctrl_cmd_internal_bus_flush_with_sysmembar None
  | Kernel_gpu g ->
      let (module R : D.RELEASE) = g.c.release in
      let module F = D.Flush_gpu_cache in
      let p = P.create F.sizeof in
      P.set p F.flags
        (P.bits R.nv2080_ctrl_fb_flush_gpu_cache_flags_write_back
           D.nv2080_ctrl_fb_flush_gpu_cache_flags_write_back_yes
        lor P.bits R.nv2080_ctrl_fb_flush_gpu_cache_flags_invalidate
              D.nv2080_ctrl_fb_flush_gpu_cache_flags_invalidate_yes
        lor P.bits R.nv2080_ctrl_fb_flush_gpu_cache_flags_flush_mode
              D.nv2080_ctrl_fb_flush_gpu_cache_flags_flush_mode_full_cache);
      n.rm.control n.obj.subdevice D.nv2080_ctrl_cmd_fb_flush_gpu_cache (Some p)
