(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Sources: the Linux kernel's amdgpu driver (drivers/gpu/drm/amd) for the
   discovery table (discovery.h, amdgpu_discovery.c), firmware headers
   (amdgpu_ucode.h), the security processor (psp_gfx_if.h, psp_v13_0.c,
   psp_v14_0.c), the power manager (smu_v13_0.c, smu_v14_0.c), page tables and
   hubs (amdgpu_vm.h, gmc_v9_0.c, gmc_v11_0.c, gmc_v12_0.c, mmhub_v*.c,
   gfxhub_v*.c), compute queues (gfx_v9_4_3.c, gfx_v11_0.c, gfx_v12_0.c,
   v11_structs.h, v12_structs.h), copy engines (sdma_v4_4_2.c, sdma_v6_0.c,
   sdma_v7_0.c), interrupts (ih_v6_0.c, ih_v7_0.c, amdgpu_irq.h), and virtual
   functions (mxgpu_nv.c, amdgpu_virt.c). *)

module Machine = Rig_pci.Machine
module Memory = Rig_pci.Memory
module Window = Rig_pci.Window
module Driver = Rig_pci.Driver
module Amd = Rig_amd

let strf = Printf.sprintf
let ( let* ) = Result.bind

(* Letting go

   amdgpu releases a GPU's device, its teardown writing to the GPU, when the
   last file of its DRM nodes goes, KFD's references to them included, which
   Driver waits for. The release removes the GPU's node from KFD's topology at
   its start, and the device's [ip_discovery] directory at its end: of an unbound
   GPU, either one still there means amdgpu has not let go. A release that never
   comes, as after a failed resume, leaves the directory for good, and a load of
   amdgpu then fails on it. *)

let topology = "sys/class/kfd/kfd/topology/nodes"

(* The KFD topology's PCI location of the function at [bus], "DDDD:BB:DD.F": its
   domain and its bus, device and function as [bus << 8 | dev << 3 | fn], that
   of a GPU of one partition. *)
let location bus =
  Scanf.sscanf_opt bus "%x:%x:%x.%x%!" (fun d b dev fn ->
      (d, (b lsl 8) lor (dev lsl 3) lor fn))

let unreleased ~root bus =
  let file p = Filename.concat root p in
  let listed node =
    match
      In_channel.with_open_text
        (file (strf "%s/%s/properties" topology node))
        In_channel.input_lines
    with
    | exception Sys_error _ -> false
    | lines ->
        let field k =
          List.find_map
            (fun l ->
              match String.split_on_char ' ' l with
              | [ k'; v ] when k' = k -> int_of_string_opt v
              | _ -> None)
            lines
        in
        location bus
        = Option.bind (field "domain") (fun d ->
            Option.map (fun l -> (d, l)) (field "location_id"))
  in
  let nodes =
    match Sys.readdir (file topology) with
    | names -> Array.to_list names
    | exception Sys_error _ -> []
  in
  if List.exists listed nodes then
    Some "KFD's topology still lists it, which amdgpu's release removes"
  else if
    Sys.file_exists (file (strf "sys/bus/pci/devices/%s/ip_discovery" bus))
  then Some "its ip_discovery directory stays, which amdgpu's release removes"
  else None

(* Numbering *)

(* The kernel driver serves a GPU through DRM nodes, each with a [dev] file
   under the GPU's directory, which {!Driver.detach} finds itself. *)
let gpus =
  Driver.make ~name:"AMD-PCI" ~memory_bar:(Some 0) ~siblings:Remove
    ~nodes:(fun ~root:_ _ -> [])
    ~unreleased ~teardown_ms:30_000 ~reset:Boot.reset
    (fun (id : Machine.id) -> Amd.is_gpu ~vendor:id.vendor ~class_:id.class_)

let buses ?(machine = Machine.this) () = Driver.buses gpus machine
let count ?machine () = Result.map List.length (buses ?machine ())
let device_name i = Driver.name gpus i

(* Boot reports *)

type block = { name : string; version : int * int * int; instances : int list }
type image = { file : string; found : string option }
type report = { blocks : block list; images : image list }

(* [survey ~firmware d] is the report on a GPU whose discovery table is [d] and,
   once every image is found, what its boot loads: its registers' layout and its
   firmware. [open_] and [report] both answer from it. *)
let survey ~firmware (d : Discovery.t) =
  let* layout = Regs.layout d in
  let* named = Firmware.names d in
  (* Every image an open may read is one this library pins. *)
  let rec pin = function
    | [] -> Ok []
    | (b, file) :: rest -> (
        match List.assoc_opt file Defs.pinned with
        | Some digest -> Result.map (List.cons (file, digest)) (pin rest)
        | None ->
            let major, minor, rev = Option.get (Discovery.version d b) in
            Error
              (strf
                 "%s %d.%d.%d is a version this library does not boot: no \
                  image %s"
                 (Discovery.name b) major minor rev file))
  in
  let* pinned = pin named in
  let looked =
    List.map
      (fun (file, digest) ->
        (file, Rig_pci.Firmware.find firmware file ~digest))
      pinned
  in
  let found (i : Rig_pci.Firmware.image) = i.path in
  let images =
    List.map
      (fun (file, r) -> { file; found = Option.map found (Result.to_option r) })
      looked
  in
  (* [Regs.layout] found each block. *)
  let block hw =
    let version = Option.get (Discovery.version d hw) in
    let instances = List.map fst (Discovery.live d hw) in
    { name = Discovery.name hw; version; instances }
  in
  let report = { blocks = List.map block Regs.blocks; images } in
  let missing (_, r) = match r with Error why -> Some why | Ok _ -> None in
  match List.find_map missing looked with
  | Some why -> Ok (report, Error why)
  | None ->
      let contents file =
        Result.map
          (fun (i : Rig_pci.Firmware.image) -> i.contents)
          (List.assoc file looked)
      in
      let* firmware = Firmware.load contents d in
      Ok (report, Ok (layout, firmware))

let report ~firmware table =
  Amd.Lock.check "Rig_amd_pci.report";
  let* d = Discovery.of_string table in
  Result.map fst (survey ~firmware d)

(* Memory *)

let host r =
  match Memory.host r with
  | Some w when Window.mapped w -> Some (Window.address w)
  | _ -> None

let memory r = { Amd.address = Memory.address r; host = host r; data = r }

(* What the path's memory functions use: a GPU's memory and the lock its calls
   take. Memory the path gave holds them until its free, also after stop, so
   they hold nothing else of the GPU, and stop releases the memory. The lock
   nests inside the driver's lock and a hold's ([Driver]), and outside rig.pci's
   memory locks; [machines_lock] holds no other. *)
type gpu = { memory : Memory.t; lock : Amd.Lock.t }

let protect g f = Amd.Lock.protect g.lock f

(* What the path keeps of each machine, from its first open there: the key
   its devices share, so that a GPU's number names the same GPU for each of
   them, and each GPU's reach as its last open found it, by number. A GPU's
   reach is the same at every open ([Memory.reach]), so a device's [reaches]
   answers from its GPU's number alone, also once it is lost, and keeps nothing
   of the device. *)
type known = {
  key : Memory.region Type.Id.t;
  mutable reached : (int * Memory.reach) list;
}

let machines : (Machine.t * known) list ref = ref []
let machines_lock = Amd.Lock.create ()

let known m =
  Amd.Lock.protect machines_lock @@ fun () ->
  match List.assq_opt m !machines with
  | Some on -> on
  | None ->
      let on = { key = Type.Id.make (); reached = [] } in
      machines := (m, on) :: !machines;
      on

let learn on index reach =
  Amd.Lock.protect machines_lock @@ fun () ->
  on.reached <- (index, reach) :: List.remove_assoc index on.reached

let reaches on ~index j =
  let reach () =
    match (List.assoc_opt index on.reached, List.assoc_opt j on.reached) with
    | Some r, Some o -> Memory.reaches r o
    | _ -> false
  in
  j = index || Amd.Lock.protect machines_lock reach

(* A path function raises Rig_amd.Fault for a register sequence that does not
   complete. A flush the hubs do not confirm is raised by the next sleep. *)
let guard f = try f () with Regs.Stuck why -> raise (Amd.Fault why)

let alloc g (kind : Rig_edge.memory) n =
  let kind =
    match kind with
    | Device -> Memory.Gpu
    | Mapped -> Memory.Bar
    | Pinned -> Memory.Host
  in
  match protect g (fun () -> Memory.alloc g.memory kind n) with
  | Ok (Some r) -> Some (memory r)
  | Ok None -> None
  | Error why -> raise (Amd.Fault why)

(* A mapping's [Error] is the GPU's refusal of the memory, such as pages the
   machine does not pin or another GPU's memory this one does not reach,
   unless the GPU's function [fn] can no longer be reached: that is the
   path's failure, a fault. *)
let mapped fn = function
  | Ok (Some r) -> Some (memory r)
  | Ok None -> None
  | Error _ -> (
      match Rig_pci.Function.failed fn with
      | Some why -> raise (Amd.Fault why)
      | None -> None)

let map_host g fn a n =
  mapped fn (protect g (fun () -> Memory.map_host g.memory a n))

let map_peer g fn (m : Memory.region Amd.memory) =
  mapped fn (protect g (fun () -> Memory.map_peer g.memory m.data))

let free g (m : Memory.region Amd.memory) =
  protect g (fun () -> Memory.free g.memory m.data)

(* Another function reaches the memory as a peer GPU does, at its runs through
   the memory BAR, and rig.pci keeps them and the GPU's function for it until
   the process exits. *)
let dma g (m : Memory.region Amd.memory) ~client =
  Result.map
    (fun runs -> Rig_edge.Bus runs)
    (protect g (fun () -> Memory.bus g.memory m.data ~client))

(* Opening *)

(* The reference clock of SOC15 and SOC21 GPUs, which the timestamps of their
   packets count. *)
let clock_hz = 100_000_000

(* A compute unit's SIMDs: 4 on GC 9, 2 from GC 10 on (NUM_SIMD_PER_CU of the
   kernel's vega10_enum.h, navi10_enum.h, soc21_enum.h and soc24_enum.h). *)
let simds_per_cu (gpu : Rig_amd_abi.Gpu.t) =
  match gpu.gc with 9, _, _ -> 4 | _ -> 2

(* The interrupt context of a release. The path wakes on every entry of the
   interrupt ring, so any context but 0 serves. *)
let interrupt = 1

let path g h fn ~index : Memory.region Amd.path =
  let gc = Boot.gc g and gpu = Boot.gpu g in
  let mem = { memory = Boot.memory g; lock = Boot.lock g } in
  (* A fault rig lost the device for, a hang included, is the GPU's as much as
     one its interrupt ring reported: the stop leaves the GPU lost. *)
  let stop ~fault =
    Option.iter (Boot.faulted g) fault;
    Fun.protect
      (fun () -> Driver.stop h)
      ~finally:(fun () -> protect mem (fun () -> Memory.release mem.memory))
  in
  let on = known (Rig_pci.Function.machine fn) in
  {
    key = on.key;
    index;
    gpu;
    waves = gc.waves * simds_per_cu gpu;
    lds = gc.lds;
    clock_hz;
    clock = Round_trip;
    mec = Boot.mec g;
    wgps = Boot.wgps g;
    budget = Boot.budget g;
    alloc = (fun kind n -> guard (fun () -> alloc mem kind n));
    (* A GPU taken physically would keep writing the process's pages after its
       death: it maps no host memory. *)
    map_host =
      (match Rig_pci.Function.addressing fn with
      | Machine.Physical -> None
      | Iommu -> Some (fun a n -> guard (fun () -> map_host mem fn a n)));
    reaches = reaches on ~index;
    map_peer = (fun m -> guard (fun () -> map_peer mem fn m));
    free = (fun m -> guard (fun () -> free mem m));
    bus = Some (Rig_pci.Function.bus fn);
    dma = (fun m ~fn:client -> dma mem m ~client);
    queue =
      (fun kind ~ring ~bytes ~read ~write ->
        guard (fun () -> Boot.queue g kind ~ring ~bytes ~read ~write));
    retire = (fun bell -> guard (fun () -> Boot.retire g bell));
    most_ring = Boot.most_ring g;
    hdp = Boot.hdp g;
    interrupt;
    hang_ms = Some Driver.hang_ms;
    sleep = (fun ~ms -> guard (fun () -> Boot.sleep g ~ms));
    stable_power = (fun () -> guard (fun () -> Boot.stable_power g));
    stop;
  }

(* A boot that wrote to the GPU and failed stopped it, as lost, before it
   answered: the hold's stop then answers so. *)
let boot h fn ~firmware =
  let lost () = Driver.set_stop h (fun () -> `Lost) in
  let gpus =
    lazy
      (Result.map List.length (Driver.buses gpus (Rig_pci.Function.machine fn)))
  in
  let first d =
    let* _, loads = survey ~firmware d in
    loads
  in
  match Boot.start ~gpus fn first with
  | Ok g -> Ok g
  | Error (`Refused why) -> Error (`Refused why)
  | Error `Running -> Error `Running
  | Error (`Lost why) ->
      lost ();
      Error (`Refused why)
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      lost ();
      Printexc.raise_with_backtrace e bt

(* Firmware this library did not start, as a process that died leaves, is reset
   through the hold, and the GPU booted in full. *)
let booted h fn ~firmware =
  match boot h fn ~firmware with
  | Ok g -> Ok g
  | Error (`Refused why) -> Error why
  | Error `Running -> (
      let* () = Driver.renew h in
      match boot h fn ~firmware with
      | Ok g -> Ok g
      | Error (`Refused why) -> Error why
      | Error `Running ->
          Error
            "firmware this library did not start still runs on the GPU after \
             its reset")

let start ~word ~firmware ~index h fn =
  let* () =
    Machine.reserve
      (Rig_pci.Function.machine fn)
      ~base:(Rig_pci.Space.base Boot.space)
      (Rig_pci.Space.length Boot.space)
  in
  let* g = booted h fn ~firmware in
  Driver.set_stop h (fun () -> Boot.stop g);
  match Amd.make ~word (path g h fn ~index) with
  | Ok d ->
      (* A host resets a VF that holds its access long: every queue is made, so
         it goes back. *)
      Boot.give_back g;
      learn
        (known (Rig_pci.Function.machine fn))
        index
        (Memory.reach (Boot.memory g));
      Ok d
  | Error why | (exception (Amd.Fault why | Regs.Stuck why)) -> Error why

let open_ ?(machine = Machine.this) ~word ~firmware i =
  Amd.Lock.check "Rig_amd_pci.gpu";
  Driver.open_ gpus machine i (start ~word ~firmware ~index:i)

(* Changes to the machine *)

(* Each takes the driver's lock ([Driver]), which an open holds through its
   boot, so a call from a callback inside a lock is refused first. *)
let detach i =
  Amd.Lock.check "Rig_amd_pci.detach";
  Driver.detach gpus Machine.this i

let attach i =
  Amd.Lock.check "Rig_amd_pci.attach";
  Driver.attach gpus Machine.this i

let reset ?(machine = Machine.this) i =
  Amd.Lock.check "Rig_amd_pci.reset";
  Driver.reset gpus machine i

(* The system's firmware directory, searched after the caller's and rig's. *)
let system_firmware = "/lib/firmware"

let gpu ?machine ?(firmware = []) i : Rig_amd.t Rig_edge.opener =
  let rig = Option.to_list (Rig_pci.Firmware.directory ()) in
  let firmware = firmware @ rig @ [ system_firmware ] in
  {
    name = device_name i;
    driver = (module Rig_amd);
    make = (fun ~word -> open_ ?machine ~word ~firmware i);
  }

(* Formats *)

module Discovery = Discovery
module Firmware = Firmware
module Psp = Psp
module Smu = Smu
module Gmc = Gmc
module Gfx = Gfx
module Ih = Ih
