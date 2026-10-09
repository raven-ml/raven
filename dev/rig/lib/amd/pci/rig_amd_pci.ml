(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Machine = Rig_pci.Machine
module Memory = Rig_pci.Memory
module Window = Rig_pci.Window
module Gpus = Rig_pci.Gpus
module Amd = Rig_amd

let strf = Printf.sprintf
let ( let* ) = Result.bind

(* Letting go

   amdgpu releases a GPU's device, its teardown writing to the GPU, when the
   last file of its DRM nodes goes, KFD's references to them included, which
   Gpus waits for. The release removes the GPU's node from KFD's topology at its
   start, and the device's [ip_discovery] directory at its end: of an unbound
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
   under the GPU's directory, which {!Gpus.detach} finds itself. *)
let gpus =
  Gpus.make ~name:"AMD-PCI" ~memory_bar:0
    ~nodes:(fun ~root:_ _ -> [])
    ~unreleased ~teardown_ms:30_000 ~reset:Boot.reset
    (fun (id : Machine.id) -> Amd.is_gpu ~vendor:id.vendor ~class_:id.class_)

let buses ?(machine = Machine.this) () = Gpus.buses gpus machine
let count ?machine () = List.length (buses ?machine ())
let device_name i = Gpus.name gpus i

(* Boot reports *)

type block = { name : string; version : int * int * int; instances : int list }
type image = { file : string; found : string option }
type report = { blocks : block list; images : image list }

(* [survey ~firmware d] is the report on a GPU whose discovery table is [d] and,
   once every image is found, what its boot loads: its registers' layout and its
   firmware. [open_] and [report] both answer from it. *)
let survey ~firmware (d : Discovery.t) =
  let* layout = Regs.layout d in
  let* names = Images.names d in
  let looked =
    List.map
      (fun file ->
        let digest = List.assoc file Images.pinned in
        (file, Rig_pci.Firmware.find firmware file ~digest))
      names
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
      let contents file ~digest:_ =
        Result.map
          (fun (i : Rig_pci.Firmware.image) -> i.contents)
          (List.assoc file looked)
      in
      let* firmware = Images.load contents d in
      Ok (report, Ok (layout, firmware))

let report ~firmware table =
  let* d = Discovery.of_string table in
  Result.map fst (survey ~firmware d)

(* Memory *)

let host r =
  match Memory.host r with
  | Some w when Window.mapped w -> Some (Window.address w)
  | _ -> None

let memory r = { Amd.address = Memory.address r; host = host r; data = r }

(* The opened GPUs of this machine, by number, for [reaches]. *)
let opened : (int, Boot.t) Hashtbl.t = Hashtbl.create 4
let opened_lock = Mutex.create ()

(* A path function raises Rig_amd.Fault for a register sequence that does not
   complete. A flush the hubs do not confirm is raised by the next sleep. *)
let guard f = try f () with Regs.Stuck why -> raise (Amd.Fault why)

let alloc g kind n =
  let kind =
    match kind with
    | `Gpu -> Memory.Gpu
    | `Bar -> Memory.Bar
    | `System -> Memory.Host
  in
  match Boot.protect g (fun () -> Memory.alloc (Boot.memory g) kind n) with
  | Ok (Some r) -> Some (memory r)
  | Ok None -> None
  | Error why -> raise (Amd.Fault why)

let map_host g a n =
  match Boot.protect g (fun () -> Memory.map_host (Boot.memory g) a n) with
  | Ok (Some r) -> Some (memory r)
  | Ok None | Error _ -> None

let reaches g ~index j =
  j = index
  ||
  match Mutex.protect opened_lock (fun () -> Hashtbl.find_opt opened j) with
  | Some o -> Memory.reaches (Boot.memory g) (Boot.memory o)
  | None -> false

let map_peer g (m : Memory.region Amd.memory) =
  match Boot.protect g (fun () -> Memory.map_peer (Boot.memory g) m.data) with
  | Ok (Some r) -> Some (memory r)
  | Ok None | Error _ -> None

let free g (m : Memory.region Amd.memory) =
  Boot.protect g (fun () -> Memory.free (Boot.memory g) m.data)

(* Opening *)

let key : Memory.region Type.Id.t = Type.Id.make ()

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
  (* A fault rig lost the device for, a hang included, is the GPU's as much as
     one its interrupt ring reported: the stop leaves the GPU lost. *)
  let stop ~fault =
    Option.iter (Boot.faulted g) fault;
    Mutex.protect opened_lock (fun () -> Hashtbl.remove opened index);
    Gpus.stop h
  in
  {
    key;
    index;
    gpu;
    waves = gc.waves * simds_per_cu gpu;
    lds = gc.lds;
    clock_hz;
    mec = Boot.mec g;
    wgps = Boot.wgps g;
    budget = Boot.budget g;
    alloc = (fun kind n -> guard (fun () -> alloc g kind n));
    (* A GPU taken physically would keep writing the process's pages after its
       death: it maps no host memory. *)
    map_host =
      (match Rig_pci.Function.addressing fn with
      | Machine.Physical -> None
      | Iommu -> Some (fun a n -> guard (fun () -> map_host g a n)));
    reaches = reaches g ~index;
    map_peer = (fun m -> guard (fun () -> map_peer g m));
    free = (fun m -> guard (fun () -> free g m));
    queue =
      (fun kind ~ring ~bytes ~read ~write ->
        guard (fun () -> Boot.queue g kind ~ring ~bytes ~read ~write));
    hdp = Boot.hdp g;
    interrupt;
    hang_ms = Some Gpus.hang_ms;
    sleep = (fun ~ms -> guard (fun () -> Boot.sleep g ~ms));
    stable_power = (fun () -> guard (fun () -> Boot.stable_power g));
    stop;
  }

(* A boot that wrote to the GPU and failed stopped it, as lost, before it
   answered: the hold's stop then answers so. *)
let boot h fn ~firmware =
  let lost () = Gpus.set_stop h (fun () -> `Lost) in
  let gpus =
    lazy (List.length (Gpus.buses gpus (Rig_pci.Function.machine fn)))
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
      let* () = Gpus.renew h in
      match boot h fn ~firmware with
      | Ok g -> Ok g
      | Error (`Refused why) -> Error why
      | Error `Running ->
          Error
            "firmware this library did not start still runs on the GPU after \
             its reset")

let start ~firmware ~index h fn =
  let* () =
    Machine.reserve
      (Rig_pci.Function.machine fn)
      ~base:(Rig_pci.Space.base Boot.space)
      (Rig_pci.Space.length Boot.space)
  in
  let* g = booted h fn ~firmware in
  Gpus.set_stop h (fun () -> Boot.stop g);
  match Amd.make (path g h fn ~index) with
  | Ok d ->
      (* A host resets a VF that holds its access long: every queue is made, so
         it goes back. *)
      Boot.give_back g;
      Mutex.protect opened_lock (fun () -> Hashtbl.replace opened index g);
      Ok d
  | Error why | (exception (Amd.Fault why | Regs.Stuck why)) -> Error why

let open_ ?(machine = Machine.this) ~firmware i =
  Gpus.open_ gpus machine i (start ~firmware ~index:i)

(* Changes to the machine *)

let detach i = Gpus.detach gpus Machine.this i
let attach i = Gpus.attach gpus Machine.this i
let reset ?(machine = Machine.this) i = Gpus.reset gpus machine i
