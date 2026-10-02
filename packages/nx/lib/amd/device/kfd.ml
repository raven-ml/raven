(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* AMD GPUs through the compute interface of Linux's amdgpu driver. *)

module D = Amd_defs
module Mmio = Nx_device_support.Mmio
module Pci = Nx_device_support.Pci

external open_file : string -> int = "caml_nx_amd_open"
external close_file : int -> unit = "caml_nx_amd_close"
external reserve : int -> nativeint = "caml_nx_amd_reserve"
external anon : int -> nativeint = "caml_nx_amd_anon"

external map_file : int -> nativeint -> int -> int64 -> nativeint
  = "caml_nx_amd_map"

external unmap_mem : nativeint -> int -> unit = "caml_nx_amd_unmap"
external version : int -> int * int = "caml_nx_kfd_version"
external acquire_vm : int -> int -> int -> unit = "caml_nx_kfd_acquire_vm"
external runtime_enable : int -> unit = "caml_nx_kfd_runtime_enable"

external kfd_alloc :
  int -> int -> nativeint -> int -> int -> int64 -> (int64 * int64, int) result
  = "caml_nx_kfd_alloc_byte" "caml_nx_kfd_alloc"

external kfd_free : int -> int64 -> unit = "caml_nx_kfd_free"
external kfd_map : int -> int64 -> int -> bool -> int = "caml_nx_kfd_map"
external create_event : int -> int -> int64 -> int = "caml_nx_kfd_create_event"

type queue_args = {
  ring : nativeint;
  ring_bytes : int;
  gpu : int;
  kind : int;
  eop : nativeint;
  eop_bytes : int;
  cwsr : nativeint;
  cwsr_bytes : int;
  ctl_stack_bytes : int;
  wptr : nativeint;
  rptr : nativeint;
}

external create_queue : int -> queue_args -> int * int64
  = "caml_nx_kfd_create_queue"

external wait_events : int -> int array -> int -> int -> string
  = "caml_nx_kfd_wait"

external drm_info : int -> int -> int -> int -> int -> string
  = "caml_nx_amd_drm_info"

let topology = "/sys/devices/virtual/kfd/kfd/topology/nodes"

let read file =
  In_channel.with_open_text file In_channel.input_all |> String.trim

let properties file =
  List.filter_map
    (fun l ->
      match String.split_on_char ' ' (String.trim l) with
      | [ k; v ] -> Option.map (fun v -> (k, v)) (int_of_string_opt v)
      | _ -> None)
    (String.split_on_char '\n' (read file))

let dir_entries d = try Array.to_list (Sys.readdir d) with Sys_error _ -> []

(* The topology nodes of GPUs, in node order. *)
let gpu_nodes () =
  dir_entries topology
  |> List.filter_map int_of_string_opt
  |> List.sort compare
  |> List.filter (fun n ->
      match
        int_of_string_opt (read (Printf.sprintf "%s/%d/gpu_id" topology n))
      with
      | Some id -> id <> 0
      | None | (exception Sys_error _) -> false)

let available () = Sys.file_exists "/dev/kfd"

(* The topology node of the GPU at the bus address [bus], if the driver holds
   it. Nodes follow the order in which the driver took its GPUs and leave out
   those it does not hold: GPUs are numbered by bus address instead. A node's
   [location_id] is its function's bus number and device-function byte. A GPU
   split into partitions has a node per partition, which share its address: its
   first is the GPU. *)
let node_of bus =
  List.find_opt
    (fun n ->
      let p = properties (Printf.sprintf "%s/%d/properties" topology n) in
      match (List.assoc_opt "domain" p, List.assoc_opt "location_id" p) with
      | Some domain, Some loc ->
          Pci.address ~domain ~bus:(loc lsr 8)
            ~device:((loc lsr 3) land 0x1f)
            ~fn:(loc land 7)
          = bus
      | _ -> false)
    (gpu_nodes ())

(* The process's KFD, opened once. *)
let kfd = lazy (open_file "/dev/kfd")

type t = {
  fd : int;
  drm : int;
  gpu_id : int;
  node : int;
  props : (string * int) list;
  ip_ver : (int * Amdev.version) list;
  vram : int;
  visible : int; (* of [vram], the bytes the host can map *)
  sysfs : string;
  mutable events : int array; (* signal, memory exception, hardware exception *)
  mutable doorbells : (nativeint * int64) option; (* the page, and its offset *)
  hdp : Mmio.t option;
      (* the driver's page of registers whose first word flushes the HDP, if it
         remaps one *)
}

let prop t k =
  match List.assoc_opt k t.props with
  | Some v -> v
  | None -> failwith (Printf.sprintf "KFD reports no %s" k)

(* The GPUs whose address space the process acquired, by topology node: the
   driver lets a process acquire it only once, so a failed open reuses them.
   Opens are serialized by the caller. *)
let acquired : (int, t) Hashtbl.t = Hashtbl.create 4

(* The page of registers the driver remaps for the process, whose first word
   flushes the host data path (HDP); [None] if the driver refuses it. *)
let remap_hdp fd gpu_id =
  let n = 0x1000 in
  let addr = reserve n in
  let flags =
    D.kfd_ioc_alloc_mem_flags_mmio_remap lor D.kfd_ioc_alloc_mem_flags_writable
    lor D.kfd_ioc_alloc_mem_flags_no_substitute
  in
  match kfd_alloc fd gpu_id addr n flags 0L with
  | Error _ ->
      unmap_mem addr n;
      None
  | Ok (_, offset) -> Some (Mmio.v (map_file fd addr n offset) n)

let open_new node =
  let dir = Printf.sprintf "%s/%d" topology node in
  let gpu_id = int_of_string (read (dir ^ "/gpu_id")) in
  let props = properties (dir ^ "/properties") in
  let render =
    match List.assoc_opt "drm_render_minor" props with
    | Some n -> n
    | None -> failwith "KFD reports no drm_render_minor"
  in
  let sysfs = Printf.sprintf "/sys/class/drm/renderD%d/device" render in
  let ip_base = sysfs ^ "/ip_discovery/die/0" in
  let ip_ver =
    List.filter_map
      (fun (hwip, hwid, name, required) ->
        let d = Printf.sprintf "%s/%d/0" ip_base hwid in
        match
          List.map
            (fun p -> int_of_string (read (Printf.sprintf "%s/%s" d p)))
            [ "major"; "minor"; "revision" ]
        with
        | [ a; b; c ] -> Some (hwip, (a, b, c))
        | _ | (exception (Sys_error _ | Failure _)) ->
            if required then
              failwith
                (Printf.sprintf
                   "the amdgpu driver reports no %s version (%s/%d/0)" name
                   ip_base hwid)
            else None)
      D.
        [
          (gc_hwip, gc_hwid, "GC", true);
          (sdma0_hwip, sdma0_hwid, "SDMA", true);
          (nbif_hwip, nbif_hwid, "NBIF", false);
        ]
  in
  (* The banks of the GPU's memory: heap type 1 is the part the host can map, 2
     the rest. *)
  let banks =
    List.filter_map
      (fun bank ->
        let p =
          properties (Printf.sprintf "%s/mem_banks/%s/properties" dir bank)
        in
        match
          (List.assoc_opt "heap_type" p, List.assoc_opt "size_in_bytes" p)
        with
        | Some ((1 | 2) as heap), Some n -> Some (heap, n)
        | _ -> None)
      (dir_entries (dir ^ "/mem_banks"))
  in
  let vram = List.fold_left (fun acc (_, n) -> acc + n) 0 banks in
  let visible =
    List.fold_left (fun acc (h, n) -> if h = 1 then acc + n else acc) 0 banks
  in
  let fd = Lazy.force kfd in
  let major, minor = version fd in
  let drm = open_file (Printf.sprintf "/dev/dri/renderD%d" render) in
  (try acquire_vm fd drm gpu_id
   with e ->
     close_file drm;
     raise e);
  if (major, minor) >= (1, 14) then runtime_enable fd;
  {
    fd;
    drm;
    gpu_id;
    node;
    props;
    ip_ver;
    vram;
    visible;
    sysfs;
    events = [||];
    doorbells = None;
    hdp = remap_hdp fd gpu_id;
  }

let open_gpu bus =
  match node_of bus with
  | None -> failwith (bus ^ " is not held by the amdgpu driver")
  | Some node -> (
      match Hashtbl.find_opt acquired node with
      | Some t -> t
      | None ->
          let t = open_new node in
          Hashtbl.replace acquired node t;
          t)

(* Whether the GPU at topology node [node] reaches the memory of the GPU at node
   [n]: over XGMI (an I/O link) or over PCIe through a large BAR (a P2P
   link). *)
let reaches node n =
  let dir = Printf.sprintf "%s/%d" topology node in
  List.exists
    (fun links ->
      List.exists
        (fun l ->
          List.assoc_opt "node_to"
            (properties (Printf.sprintf "%s/%s/%s/properties" dir links l))
          = Some n)
        (dir_entries (Printf.sprintf "%s/%s" dir links)))
    [ "io_links"; "p2p_links" ]

(* Memory *)

type kind = Vram | Visible | Host | Uncached

type mem = {
  va : int;
  size : int;
  handle : int64;
  host : Mmio.t option;
  mapped : nativeint option; (* the process mapping to release *)
}

let enomem = 12
let einval = 22

let map_handle t handle =
  match kfd_map t.fd handle t.gpu_id true with
  | 0 -> ()
  | e ->
      failwith
        (Printf.sprintf "mapping GPU memory on GPU %d failed (errno %d)"
           t.gpu_id e)

let unmap_handle t handle = ignore (kfd_map t.fd handle t.gpu_id false)

(* [n] bytes of [kind]: VRAM the host does not address, VRAM it does through the
   BAR, coherent host memory, or uncached GTT memory for rings. *)
let alloc t kind n =
  let open D in
  let base =
    kfd_ioc_alloc_mem_flags_writable lor kfd_ioc_alloc_mem_flags_executable
    lor kfd_ioc_alloc_mem_flags_no_substitute
  in
  let flags =
    match kind with
    | Vram -> base lor kfd_ioc_alloc_mem_flags_vram
    | Visible ->
        base lor kfd_ioc_alloc_mem_flags_vram lor kfd_ioc_alloc_mem_flags_public
    | Host ->
        base lor kfd_ioc_alloc_mem_flags_userptr
        lor kfd_ioc_alloc_mem_flags_coherent
        lor kfd_ioc_alloc_mem_flags_uncached lor kfd_ioc_alloc_mem_flags_public
    | Uncached ->
        base lor kfd_ioc_alloc_mem_flags_coherent
        lor kfd_ioc_alloc_mem_flags_uncached lor kfd_ioc_alloc_mem_flags_gtt
        lor kfd_ioc_alloc_mem_flags_public
  in
  let userptr = kind = Host in
  let addr = if userptr then anon n else reserve n in
  match
    kfd_alloc t.fd t.gpu_id addr n flags
      (if userptr then Int64.of_nativeint addr else 0L)
  with
  | Error e ->
      unmap_mem addr n;
      (* Without a large BAR, the GPU has no memory the host addresses. *)
      if e = enomem || (e = einval && kind = Visible) then None
      else
        failwith
          (Printf.sprintf "allocating %d bytes of GPU memory failed (errno %d)"
             n e)
  | Ok (handle, offset) -> (
      match
        if not userptr then ignore (map_file t.drm addr n offset);
        map_handle t handle
      with
      | () ->
          let visible = kind <> Vram in
          Some
            {
              va = Nativeint.to_int addr;
              size = n;
              handle;
              host = (if visible then Some (Mmio.v addr n) else None);
              mapped = Some addr;
            }
      | exception e ->
          kfd_free t.fd handle;
          unmap_mem addr n;
          raise e)

let free t m =
  unmap_handle t m.handle;
  Option.iter (fun a -> unmap_mem a m.size) m.mapped;
  kfd_free t.fd m.handle

(* Registers the process memory at [a] with the GPU, at [a]. *)
let map_host t a n =
  let open D in
  let flags =
    kfd_ioc_alloc_mem_flags_writable lor kfd_ioc_alloc_mem_flags_executable
    lor kfd_ioc_alloc_mem_flags_no_substitute
    lor kfd_ioc_alloc_mem_flags_userptr lor kfd_ioc_alloc_mem_flags_coherent
    lor kfd_ioc_alloc_mem_flags_uncached lor kfd_ioc_alloc_mem_flags_public
  in
  match kfd_alloc t.fd t.gpu_id a n flags (Int64.of_nativeint a) with
  | Error e ->
      Error (Printf.sprintf "the driver refuses to register it (errno %d)" e)
  | Ok (handle, _) -> (
      match map_handle t handle with
      | () ->
          Ok
            {
              va = Nativeint.to_int a;
              size = n;
              handle;
              host = None;
              mapped = None;
            }
      | exception Failure why ->
          kfd_free t.fd handle;
          Error why)

let unmap_host t m =
  unmap_handle t m.handle;
  kfd_free t.fd m.handle

let map_peer t m = map_handle t m.handle
let unmap_peer t m = unmap_handle t m.handle

(* Queues *)

(* The process's event page, which the first queue of any GPU creates, and which
   every GPU maps. *)
let event_page : (t * mem) option ref = ref None

let events t =
  if t.events = [||] then begin
    (match !event_page with
    | Some (owner, m) -> if owner.gpu_id <> t.gpu_id then map_peer t m
    | None -> (
        match alloc t Uncached 0x8000 with
        | Some m ->
            ignore (create_event t.fd D.kfd_ioc_event_signal m.handle);
            event_page := Some (t, m)
        | None -> failwith "no memory for the KFD event page"));
    t.events <-
      Array.map
        (fun kind -> create_event t.fd kind 0L)
        D.
          [|
            kfd_ioc_event_signal;
            kfd_ioc_event_memory;
            kfd_ioc_event_hw_exception;
          |]
  end

(* Creates a queue; the address of its doorbell. *)
let create_queue t args =
  events t;
  let _, doorbell = create_queue t.fd { args with gpu = t.gpu_id } in
  let page, base =
    match t.doorbells with
    | Some p -> p
    | None ->
        let base = Int64.logand doorbell (Int64.lognot 0x1fffL) in
        let page = map_file t.fd 0n 0x2000 base in
        t.doorbells <- Some (page, base);
        (page, base)
  in
  Nativeint.add page (Int64.to_nativeint (Int64.sub doorbell base))

(* Flushes the host data path (HDP), so that the host's writes to the GPU's
   memory through its BAR reach it, if the driver remaps its register. *)
let flush_hdp t =
  Option.iter
    (fun page ->
      Mmio.barrier ();
      Mmio.set32 page D.kfd_mmio_remap_hdp_mem_flush_cntl 0)
    t.hdp

let flushes_hdp t = Option.is_some t.hdp

(* Blocks at most [ms] on the GPU's events; raises the report of an exception of
   this GPU. *)
(* The compute units of each shader array that the amdgpu driver reports
   active: a bitmap per engine and array, as its device information lays them
   out. *)
let cu_bitmap t =
  let info =
    drm_info t.drm
      (D.drm_command_base + D.drm_amdgpu_info)
      D.Drm_amdgpu_info.sizeof D.amdgpu_info_dev_info
      D.Drm_amdgpu_info_device.sizeof
  in
  let at, row = D.Drm_amdgpu_info_device.cu_bitmap in
  Array.init 4 (fun i ->
      Array.init 4 (fun j ->
          Int32.to_int (String.get_int32_le info (at + (row * i) + (4 * j)))
          land 0xffff_ffff))

let sleep t ms =
  if t.events <> [||] then
    match wait_events t.fd t.events t.gpu_id ms with
    | "" -> ()
    | report -> failwith report
