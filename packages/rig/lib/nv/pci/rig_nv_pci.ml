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

(* Locks *)

(* A public function that takes a lock, the path's or rig.pci's, starts
   with [Rig_nv.Lock.check], before it changes anything. Locks nest in this
   order: rig.pci's locks of its GPUs, which an open, a detach, an attach, a
   reset and a stop hold; the path's own, one at a time ([machines_lock], a
   GPU's [lock], its GSP's); rig.pci's others, of memory and firmware. *)

(* GPUs *)

(* The memory BAR, BAR 1. *)
let memory_bar = 1

(* NVIDIA's driver refuses to boot a GPU whose protected memory (WPR2) is up
   (_kgspBootGspRm in kernel_gsp.c), and so does an open: a reset that leaves
   it up, as a GPU whose stop could not take it down may keep it, leaves a GPU
   neither can boot. *)
let ready fn =
  let* regs = Function.map ~combine:false fn 0 in
  let up = Window.get32 regs Defs.nv_pfb_pri_mmu_wpr2_addr_hi <> 0 in
  Function.unmap fn regs;
  if not up then Ok ()
  else
    Error
      (strf
         "%s keeps its protected memory (WPR2) up after its reset, so no \
          driver can boot it until a reset that takes WPR2 down, such as a \
          power cycle"
         (Function.bus fn))

(* NVIDIA's driver holds no file of a GPU's devices past the process that opened
   it, and tells nothing of an unbound GPU. *)
let gpus =
  Driver.make ~name:"NV-PCI" ~memory_bar:(Some memory_bar)
    ~siblings:Remove ~nodes:Held.nodes
    ~unreleased:(fun ~root:_ _ -> None)
    ~teardown_ms:0 ~reset:Function.reset ~ready
    (fun (id : Machine.id) -> Rig_nv.is_gpu ~vendor:id.vendor ~class_:id.class_)

let buses ?(machine = Machine.this) () = Driver.buses gpus machine
let count ?machine () = Result.map List.length (buses ?machine ())
let device_name i = Driver.name gpus i

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
  let parse ~gsp ~bootloader ~start unload =
    let contents (i : Firmware.image) = i.contents in
    Images.parse family ~gsp:(contents gsp) ~bootloader:(contents bootloader)
      ~start:(contents start) ~unload:(Option.map contents unload)
  in
  match List.map snd looked with
  | [ Ok gsp; Ok bootloader; Ok start ] ->
      let* fw = parse ~gsp ~bootloader ~start None in
      Ok (report, Ok (fw, fwsec))
  | [ Ok gsp; Ok bootloader; Ok start; Ok unload ] ->
      let* fw = parse ~gsp ~bootloader ~start (Some unload) in
      Ok (report, Ok (fw, fwsec))
  | rs -> (
      match
        List.find_map (function Error why -> Some why | Ok _ -> None) rs
      with
      | Some why -> Ok (report, Error why)
      | None -> assert false (* each family's files, not all found *))

let report ~firmware ~chip ~vbios =
  Rig_nv.Lock.check "Rig_nv_pci.report";
  Result.map fst (survey ~firmware chip (fun () -> Vbios.of_string vbios))

(* Memory *)

(* The GPU addresses of every NVIDIA GPU this path opens, from 64 GiB of the
   process to the windows ({!Rig_nv.windows}): below them, as a device requires
   of memory the host addresses, and where an IOMMU maps device addresses.
   System memory lies at the same address for the process and the GPU, so the
   range is reserved on the GPU's machine. One range serves every GPU: a peer's
   memory maps at its owner's address. It lies above the GSP's addresses
   ({!Rpc.reserved}), which the GPU's tables must leave unmapped. *)
let space = Space.create ~base:(64 * gib) (Rig_nv.windows - (64 * gib))

module Handles = Map.Make (Int)

(* A GPU this path opened. [fault] is the first flush the GPU did not confirm.
   Memory the path gave holds this record until its free, also after stop, so
   stop drops [names] and releases [memory]. *)
type gpu = {
  index : int;
  fn : Function.t;
  memory : Memory.t;
  lock : Rig_nv.Lock.t; (* its memory, page tables and names, the GSP's too *)
  fault : string option Atomic.t;
  mutable names : names option; (* [None] once stopped *)
}

(* The names the path gave the GPU's memory, which the RM's channel allocations
   take, with each name's region. The regions change by one store of a map
   made first, so an exception leaves them whole for the frees. *)
and names = { mutable regions : Memory.region Handles.t; mutable next : int }

(* What the path keeps of each machine, from its first open there: the key
   its devices share, so that a GPU's number names the same GPU for each of
   them, and each GPU's reach as its last open found it, by number. A GPU's
   reach is the same at every open ([Memory.reach]), so a device's [reaches]
   answers from its GPU's number alone, also once it is lost, and keeps nothing
   of the device. [reaches] takes no lock, as [Rig_nv.peer] must not block:
   the reaches change by swaps of lists made from the one read. *)
type known = {
  key : Memory.region Type.Id.t;
  reached : (int * Memory.reach) list Atomic.t;
}

let machines : (Machine.t * known) list ref = ref []
let machines_lock = Rig_nv.Lock.create ()

let known m =
  Rig_nv.Lock.protect machines_lock @@ fun () ->
  match List.assq_opt m !machines with
  | Some on -> on
  | None ->
      let on = { key = Type.Id.make (); reached = Atomic.make [] } in
      machines := (m, on) :: !machines;
      on

let rec learn on index reach =
  let l = Atomic.get on.reached in
  let l' = (index, reach) :: List.remove_assoc index l in
  if not (Atomic.compare_and_set on.reached l l') then learn on index reach

let reaches on ~index j =
  let l = Atomic.get on.reached in
  j = index
  ||
  match (List.assoc_opt index l, List.assoc_opt j l) with
  | Some r, Some o -> Memory.reaches r o
  | _ -> false

let protect g f = Rig_nv.Lock.protect g.lock f

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
        n.regions <- Handles.add h r n.regions;
        n.next <- h + 1;
        h
  in
  { Rig_nv.address = Memory.address r; host = host r; handle; data = r }

(* The memory and address a channel's USERD is described at: the physical
   address of GPU memory, the bus address of system memory, of the byte [off] of
   the memory of handle [h]. *)
let locate g h off =
  protect g @@ fun () ->
  match Option.bind g.names (fun n -> Handles.find_opt h n.regions) with
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

(* A mapping's [Error] is the GPU's refusal of the memory, such as pages the
   machine does not pin or another GPU's memory this one does not reach,
   unless the GPU's function can no longer be reached: that is the path's
   failure, a fault. *)
let mapped g = function
  | Ok (Some r) -> Some (memory g r)
  | Ok None -> None
  | Error _ -> (
      match Function.failed g.fn with
      | Some why -> raise (Rig_nv.Fault why)
      | None -> None)

let map_host g a n =
  mapped g (protect g (fun () -> Memory.map_host g.memory a n))

let map_peer g (m : Memory.region Rig_nv.memory) =
  mapped g (protect g (fun () -> Memory.map_peer g.memory m.data))

let free g (m : Memory.region Rig_nv.memory) =
  protect g @@ fun () ->
  Option.iter (fun n -> n.regions <- Handles.remove m.handle n.regions) g.names;
  Memory.free g.memory m.data

(* Another function reaches the memory as a peer GPU does, at its runs through
   the memory BAR, and rig.pci keeps them and the GPU's function for it until
   the process exits. *)
let dma g (m : Memory.region Rig_nv.memory) ~fn =
  Result.map
    (fun runs -> Rig_edge.Bus runs)
    (protect g (fun () -> Memory.bus g.memory m.data ~client:fn))

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
   and no boot continues from another's GSP. The reset refuses a GPU it
   leaves booted. *)
let started hold (c : Chip.t) =
  if Chip.booted c then Driver.renew hold else Ok ()

let device ~word ~on g ~gsp ~hold ~tables (c : Chip.t) =
  let* () =
    match c.family with
    | Blackwell ->
        Fb_layout.check ~wpr2:(wpr2 c)
          ~top:(Page_table.memory tables)
    | Ampere | Ada -> Ok ()
  in
  let* rm = Gsp.rm gsp ~locate:(locate g) in
  let* device, subdevice, vaspace = Gsp.objects rm in
  let* gpu = Gsp.gpu gsp rm ~subdevice in
  (* The flush word's page lives with the GPU's memory until the stop, which
     gives back its addresses, as a device that fails to make does at once. A
     GPU whose BAR has no room for it opens without Mapped memory. *)
  let* page = protect g (fun () -> Memory.alloc g.memory Memory.Bar 0x1000) in
  let free_page () =
    Option.iter (fun r -> protect g (fun () -> Memory.free g.memory r)) page
  in
  let flush = Option.bind page host in
  match
    Rig_nv.make ~word
      {
        key = on.key;
        index = g.index;
        rm;
        device;
        subdevice;
        vaspace;
        gpu;
        budget = Page_table.main_pool tables;
        doorbell = Window.address c.regs + doorbell_at;
        clock = Registers (Window.address c.regs);
        flush;
        alloc = alloc g;
        (* A GPU taken physically would keep writing the process's pages after
           its death: it maps no host memory. *)
        map_host =
          (match Function.addressing g.fn with
          | Machine.Physical -> None
          | Iommu -> Some (map_host g));
        reaches = reaches on ~index:g.index;
        map_peer = map_peer g;
        free = free g;
        bus = Some (Function.bus g.fn);
        dma = dma g;
        register = (fun _ -> Ok ());
        unregister = (fun _ -> Ok ());
        check = check g gsp;
        hang_ms = Some Driver.hang_ms;
        stop =
          (fun () ->
            Fun.protect
              (fun () -> Driver.stop hold)
              ~finally:(fun () ->
                protect g (fun () ->
                    g.names <- None;
                    Memory.release g.memory);
                free_page ()));
      }
  with
  | Ok _ as d -> d
  | Error _ as e ->
      free_page ();
      e
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      free_page ();
      Printexc.raise_with_backtrace e bt

(* [start] places the GSP's memory and boots it. Before the GSP's memory is
   taken, the start only reads the GPU's registers and writes its page tables in
   its memory, so a failure gives the GPU back as it found it. From then on the
   hold's stop is the GSP's: a failure stops the GPU through it, and loses
   it. *)
let start h (c : Chip.t) (fw : Images.t) fwsec ~failed ~lock =
  let* () = Falcon.run c (Falcon.wait_reset c.family) in
  let* memory = Chip.memory c in
  let* bar = Function.map ~combine:false c.fn memory_bar in
  let top =
    Fb_layout.top (Chip.layout c.family) ~memory
      ~boot:fw.bootloader.image.length ~image:fw.gsp.length
  in
  (* The GPU's MMU indexes each level with the address's own bits, from
     address 0 (mmuFmtVirtAddrToEntryIndex), whatever part of the addresses
     the space hands out. *)
  let tables =
    Page_table.create ~base:0 (Mmu.format c bar ~failed) space ~memory:top
      ~boot:(Gsp.boot_pool fw.start)
      ~tables:(if Window.length bar >= memory then Main else Pool)
      ~pages:(Page_entry.pages (Mmu.version c.family))
  in
  (* The boot pool holds only the falcons' images: the GSP's objects come from
     the main pool. *)
  Page_table.booted tables;
  let* gsp =
    Gsp.create { chip = c; memory; tables; tables_lock = lock; bar; space } fw
      fwsec
  in
  Driver.set_stop h (fun () -> Gsp.stop gsp);
  let* () = Gsp.boot gsp in
  Ok (gsp, tables)

(* The GPU's registers and doorbell are written from the process and from C: a
   machine whose windows the process does not map is refused before the first
   write. *)
let boot ~word ~on ~firmware ~index h fn =
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
  let lock = Rig_nv.Lock.create () in
  let* gsp, tables =
    match start h c fw fwsec ~failed ~lock with
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
      lock;
      fault;
      names = Some { regions = Handles.empty; next = 1 };
    }
  in
  match device ~word ~on g ~gsp ~hold:h ~tables c with
  | Ok d ->
      learn on index (Memory.reach g.memory);
      Ok d
  | Error why | (exception Rig_nv.Fault why) -> Error why

let open_ ?(machine = Machine.this) ~word ~firmware i =
  Rig_nv.Lock.check "Rig_nv_pci.gpu";
  let on = known machine in
  Driver.open_ gpus machine i (boot ~word ~on ~firmware ~index:i)

(* Changes to the machine *)

let detach i =
  Rig_nv.Lock.check "Rig_nv_pci.detach";
  Driver.detach gpus Machine.this i

let attach i =
  Rig_nv.Lock.check "Rig_nv_pci.attach";
  Driver.attach gpus Machine.this i

let reset ?(machine = Machine.this) i =
  Rig_nv.Lock.check "Rig_nv_pci.reset";
  Driver.reset gpus machine i

(* The system's firmware directory, searched after the caller's and rig's. *)
let system_firmware = "/lib/firmware"

let gpu ?machine ?(firmware = []) i : Rig_nv.t Rig_edge.opener =
  let rig = Option.to_list (Firmware.directory ()) in
  let firmware = firmware @ rig @ [ system_firmware ] in
  {
    name = device_name i;
    driver = (module Rig_nv);
    make = (fun ~word -> open_ ?machine ~word ~firmware i);
  }

let gsp_firmware = Images.gsp_image

module Rpc = Rpc
module Fb_layout = Fb_layout
module Page_entry = Page_entry
module Falcon = Falcon
