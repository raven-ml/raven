(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* References:
   - NVIDIA's open-gpu-kernel-modules, release 570.144, the firmware's:
     src/nvidia/src/kernel/gpu/gsp/ (kernel_gsp.c, message_queue_cpu.c,
     kernel_gsp_fwsec.c, kernel_gsp_booter.c, and per architecture
     arch/turing/kernel_gsp_tu102.c, arch/ampere/kernel_gsp_ga102.c,
     arch/hopper/kernel_gsp_gh100.c, arch/blackwell/kernel_gsp_gb100.c),
     src/nvidia/inc/kernel/vgpu/rpc_global_enums.h,
     src/nvidia/arch/nvalloc/common/inc/gsp/rmgspseq.h,
     src/common/shared/msgq/inc/msgq/msgq_priv.h,
     src/common/uproc/os/common/include/libos_init_args.h, and the registers
     of src/common/inc/swref/published/ for tu102, ga102, gh100 and gb202.
   - Linux's nouveau driver: include/nvfw/fw.h and hs.h (the firmware
     containers), and nvkm/subdev/mmu/vmmtu102.c (the TLB invalidation and
     its wait).
   - The linux-firmware repository's nvidia/ directory and its
     LICENCE.nvidia. *)

open Rig_pci

let strf = Printf.sprintf
let ( let* ) = Result.bind
let gib = 1 lsl 30

(* GPUs *)

(* The memory BAR, BAR 1. *)
let memory_bar = 1

(* NVIDIA's driver holds no file of a GPU's devices past the process that opened
   it, and tells nothing of an unbound GPU. *)
let gpus =
  Gpus.make ~name:"NV-PCI" ~memory_bar ~nodes:Held.nodes
    ~unreleased:(fun ~root:_ _ -> None)
    ~teardown_ms:0 ~reset:Function.reset
    (fun (id : Machine.id) -> Rig_nv.is_gpu ~vendor:id.vendor ~class_:id.class_)

let buses ?(machine = Machine.this) () = Gpus.buses gpus machine
let count ?machine () = List.length (buses ?machine ())
let device_name i = Gpus.name gpus i

(* Boot reports *)

type image = { file : string; found : string option }
type report = { chip : string; images : image list }

(* [survey ~firmware boot42 rom] is the report on a GPU whose NV_PMC_BOOT_42
   holds [boot42] and whose VBIOS [rom ()] reads, read on Ampere and Ada only,
   and, once every image is found, what its boot loads: the firmware and the
   VBIOS's FWSEC. [open_] and [report] both answer from it. *)
let survey ~firmware boot42 rom =
  let* family, implementation = Chip.chip boot42 in
  let looked =
    List.map
      (fun file -> (file, Images.find firmware file))
      (Images.names family)
  in
  let images =
    List.map
      (fun (file, r) ->
        let found = Option.map (fun (i : Firmware.image) -> i.path) in
        { file; found = found (Result.to_option r) })
      looked
  in
  let* fwsec =
    match family with
    | Blackwell -> Ok None
    | Ampere | Ada -> Result.map Option.some (Vbios.fwsec (rom ()))
  in
  let report = { chip = Chip.name family implementation; images } in
  match List.map snd looked with
  | [ Ok gsp; Ok bootloader; Ok start ] ->
      let* fw =
        Images.parse family ~gsp:gsp.contents ~bootloader:bootloader.contents
          ~start:start.contents
      in
      Ok (report, Ok (fw, fwsec))
  | rs -> (
      match
        List.find_map (function Error why -> Some why | Ok _ -> None) rs
      with
      | Some why -> Ok (report, Error why)
      | None -> assert false (* three files, not all found *))

let report ~firmware ~chip ~vbios =
  Result.map fst (survey ~firmware chip (fun () -> vbios))

(* Memory *)

(* The GPU addresses of every NVIDIA GPU this path opens, [64 GiB, 1 TiB) of the
   process: below 2^40, as a device requires of every memory, and where an IOMMU
   maps device addresses. System memory lies at the same address for the process
   and the GPU, so the range is reserved on the GPU's machine. One range serves
   every GPU: a peer's memory maps at its owner's address. *)
let space = Space.create ~base:(64 * gib) ((1 lsl 40) - (64 * gib))

(* A GPU this path opened. [fault] is the first flush the GPU did not confirm.
   Memory the path gave holds this record until its free, also after stop, so
   stop drops [names] and releases [memory]. *)
type gpu = {
  index : int;
  fn : Function.t;
  memory : Memory.t;
  lock : Mutex.t; (* its memory, page tables and names *)
  fault : string option Atomic.t;
  mutable names : names option; (* [None] once stopped *)
}

(* The names the path gave the GPU's memory, which the RM's channel allocations
   take, with each name's region. *)
and names = { regions : (int, Memory.region) Hashtbl.t; mutable next : int }

let key : Memory.region Type.Id.t = Type.Id.make ()
let protect g f = Mutex.protect g.lock f

let host r =
  match Memory.host r with
  | Some w when Window.mapped w -> Some (Window.address w)
  | _ -> None

(* [memory g r] names [r] for [g]. *)
let memory g r =
  let handle =
    protect g @@ fun () ->
    match g.names with
    | None -> invalid_arg "Rig_nv_pci: the GPU was stopped"
    | Some n ->
        let h = n.next in
        n.next <- h + 1;
        Hashtbl.replace n.regions h r;
        h
  in
  { Rig_nv.address = Memory.address r; host = host r; handle; data = r }

(* The memory and address a channel's USERD is described at: the physical
   address of GPU memory, the bus address of system memory, of the byte [off] of
   the memory of handle [h]. *)
let locate g h off =
  protect g @@ fun () ->
  match Option.bind g.names (fun n -> Hashtbl.find_opt n.regions h) with
  | None -> None
  | Some r ->
      let target, pages = Memory.pages r in
      let rec find off = function
        | [] -> None
        | (a, n) :: _ when off < n -> Some (a + off)
        | (_, n) :: rest -> find (off - n) rest
      in
      let where =
        match target with Page_table.Gpu -> `Gpu | System | Peer _ -> `System
      in
      Option.map (fun a -> (where, a)) (find off pages)

let alloc g (kind : Rig_edge.memory) n =
  let kind =
    match kind with
    | Device -> Memory.Gpu
    | Mapped -> Memory.Bar
    | Pinned -> Memory.Host
  in
  match protect g (fun () -> Memory.alloc g.memory kind n) with
  | Ok (Some r) -> Some (memory g r)
  | Ok None -> None
  | Error why -> raise (Rig_nv.Fault why)

let map_host g a n =
  match protect g (fun () -> Memory.map_host g.memory a n) with
  | Ok (Some r) -> Some (memory g r)
  | Ok None | Error _ -> None

(* Each GPU's memory as its last open found it, by machine and number, for
   [reaches]. What [Memory.reaches] compares is the GPU's own, the same at
   every open, so a device's [reaches] answers from its GPU's number alone,
   also once it is lost, and keeps nothing of the device. *)
let learned : (Machine.t * int * Memory.t) list ref = ref []
let learned_lock = Mutex.create ()

let learn machine index memory =
  Mutex.protect learned_lock @@ fun () ->
  let other (m, i, _) = not (m == machine && i = index) in
  learned := (machine, index, memory) :: List.filter other !learned

let reaches machine ~index j =
  let known i =
    List.find_map
      (fun (m, i', o) -> if m == machine && i' = i then Some o else None)
      !learned
  in
  let reach () =
    match (known index, known j) with
    | Some m, Some o -> Memory.reaches m o
    | _ -> false
  in
  j = index || Mutex.protect learned_lock reach

let map_peer g (m : Memory.region Rig_nv.memory) =
  match protect g (fun () -> Memory.map_peer g.memory m.data) with
  | Ok (Some r) -> Some (memory g r)
  | Ok None | Error _ -> None

let free g (m : Memory.region Rig_nv.memory) =
  protect g @@ fun () ->
  Option.iter (fun n -> Hashtbl.remove n.regions m.handle) g.names;
  Memory.free g.memory m.data

(* Faults, hangs and stops *)

let check g gsp () =
  let why =
    match Gsp.check gsp with
    | Some _ as why -> why
    | None -> (
        match Atomic.get g.fault with
        | Some _ as why -> why
        | None -> Function.failed g.fn)
  in
  Option.iter (fun why -> raise (Rig_nv.Fault why)) why

(* Opening *)

(* The usermode doorbell, [NVC361_NOTIFY_CHANNEL_PENDING], in BAR 0. *)
let doorbell_at = 0xbb0090

(* The start of the region the FMC protects (WPR2), in the GPU's memory. *)
let wpr2 c =
  Chip.field Defs.nv_pfb_pri_mmu_wpr2_addr_lo_val
    (Chip.get c Defs.nv_pfb_pri_mmu_wpr2_addr_lo)
  lsl 12

(* A GSP that runs, booted by the GPU's kernel driver or a process, is reset
   before anything is written: the take proves no live process holds the GPU,
   and no boot continues from another's GSP. *)
let started hold (c : Chip.t) =
  if not (Chip.booted c) then Ok ()
  else
    let* () = Gpus.renew hold in
    if not (Chip.booted c) then Ok ()
    else
      Error
        (strf "%s still runs the GSP's firmware after its reset"
           (Function.bus c.fn))

let device g ~gsp ~hold ~tables (c : Chip.t) =
  let* () =
    match c.family with
    | Blackwell ->
        Rig_nv_abi.Fb_layout.check ~wpr2:(wpr2 c)
          ~top:(Page_table.memory tables)
    | Ampere | Ada -> Ok ()
  in
  let* rm = Gsp.rm gsp ~locate:(locate g) in
  let* device, subdevice, vaspace = Gsp.objects rm in
  let* gpu = Gsp.gpu gsp rm ~subdevice in
  Rig_nv.make
    {
      key;
      index = g.index;
      rm;
      device;
      subdevice;
      vaspace;
      gpu;
      budget = Page_table.main_pool tables;
      doorbell = Window.address c.regs + doorbell_at;
      alloc = alloc g;
      (* A GPU taken physically would keep writing the process's pages after its
         death: it maps no host memory. *)
      map_host =
        (match Function.addressing g.fn with
        | Machine.Physical -> None
        | Iommu -> Some (map_host g));
      reaches = reaches (Function.machine g.fn) ~index:g.index;
      map_peer = map_peer g;
      free = free g;
      register = (fun _ -> Ok ());
      unregister = (fun _ -> Ok ());
      check = check g gsp;
      hang_ms = Some Gpus.hang_ms;
      stop =
        (fun () ->
          Fun.protect
            (fun () -> Gpus.stop hold)
            ~finally:(fun () ->
              protect g (fun () ->
                  g.names <- None;
                  Memory.release g.memory)));
    }

(* [start] places the GSP's memory and boots it. Before the GSP's memory is
   taken, the start only reads the GPU's registers and writes its page tables in
   its memory, so a failure gives the GPU back as it found it. From then on the
   hold's stop is the GSP's: a failure stops the GPU through it, and loses
   it. *)
let start h (c : Chip.t) (fw : Images.t) fwsec ~failed =
  let* () = Falcon.run c (Falcon.wait_reset c.family) in
  let* memory = Chip.memory c in
  let* bar = Function.map ~combine:false c.fn memory_bar in
  let top =
    Rig_nv_abi.Fb_layout.top (Chip.layout c.family) ~memory
      ~boot:fw.bootloader.image.length ~image:fw.gsp.length
  in
  let tables =
    Page_table.create (Mmu.format c bar ~failed) space ~memory:top
      ~boot:(Gsp.boot_pool fw.start)
      ~tables:(if Window.length bar >= memory then Main else Pool)
      ~pages:(Rig_nv_abi.Page_entry.pages (Mmu.version c.family))
  in
  (* The boot pool holds only the falcons' images: the GSP's objects come from
     the main pool. *)
  Page_table.booted tables;
  let* gsp = Gsp.create { chip = c; memory; tables; bar; space } fw fwsec in
  Gpus.set_stop h (fun () -> Gsp.stop gsp);
  let* () = Gsp.boot gsp in
  Ok (gsp, tables)

(* The GPU's registers and doorbell are written from the process and from C: a
   machine whose windows the process does not map is refused before the first
   write. *)
let boot ~firmware ~index h fn =
  let machine = Function.machine fn in
  let* c = Chip.of_function fn in
  let* () =
    if Window.mapped c.regs then Ok ()
    else
      Error
        "the GPU's registers are not mapped into the process, as through a \
         transport"
  in
  let* () = started h c in
  let* _, loads =
    survey ~firmware (Chip.get c Defs.nv_pmc_boot_42) (fun () -> Vbios.read c)
  in
  let* fw, fwsec = loads in
  let* () =
    Machine.reserve machine ~base:(Space.base space) (Space.length space)
  in
  let fault = Atomic.make None in
  (* A flush the GPU did not confirm loses it: its bus mastering goes off at
     once, so that no stale translation reaches host memory. *)
  let failed why =
    Function.set_bus_master fn false;
    ignore (Atomic.compare_and_set fault None (Some why))
  in
  let* gsp, tables =
    match start h c fw fwsec ~failed with
    | r -> r
    | exception Rig_nv.Fault why -> Error why
  in
  let* () =
    match Atomic.get fault with Some why -> Error why | None -> Ok ()
  in
  let g =
    {
      index;
      fn;
      memory = Memory.create fn tables ~bar:memory_bar;
      lock = Mutex.create ();
      fault;
      names = Some { regions = Hashtbl.create 64; next = 1 };
    }
  in
  match device g ~gsp ~hold:h ~tables c with
  | Ok d ->
      learn machine index g.memory;
      Ok d
  | Error why | (exception Rig_nv.Fault why) -> Error why

let open_ ?(machine = Machine.this) ~firmware i =
  Gpus.open_ gpus machine i (boot ~firmware ~index:i)

(* Changes to the machine *)

let detach i = Gpus.detach gpus Machine.this i
let attach i = Gpus.attach gpus Machine.this i
let reset ?(machine = Machine.this) i = Gpus.reset gpus machine i

(* The system's firmware directory, searched after the caller's and rig's. *)
let system_firmware = "/lib/firmware"

let gpu ?machine ?(firmware = []) i : Rig_nv.t Rig_edge.opener =
  let rig = Option.to_list (Firmware.directory ()) in
  let firmware = firmware @ rig @ [ system_firmware ] in
  {
    name = device_name i;
    driver = (module Rig_nv);
    make = (fun () -> open_ ?machine ~firmware i);
  }
