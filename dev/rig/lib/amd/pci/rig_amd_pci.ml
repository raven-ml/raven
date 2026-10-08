(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Discovery = Discovery
module Regs = Regs
module Images = Images
module Gmc = Gmc
module Soc = Soc
module Sdma = Sdma
module Smu = Smu
module Psp = Psp
module Gfx = Gfx
module Boot = Boot
module Ih = Ih
module Machine = Rig_pci.Machine
module Memory = Rig_pci.Memory
module Window = Rig_pci.Window
module Gpus = Rig_pci.Gpus
module Amd = Rig_amd

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
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
  Gpus.make ~memory_bar:0
    ~nodes:(fun ~root:_ _ -> [])
    ~unreleased ~teardown_ms:30_000 ~reset:Boot.reset
    (fun (id : Machine.id) -> Amd.is_gpu ~vendor:id.vendor ~class_:id.class_)

let gpus_at root = Gpus.buses gpus (Machine.at root)
let count ?(machine = Machine.this) () = List.length (Gpus.buses gpus machine)

let index fn i =
  if i < 0 then invalid_argf "Rig_amd_pci.%s: GPU %d is negative" fn i

let device_name i =
  index "device_name" i;
  if i = 0 then "AMD-PCI" else strf "AMD-PCI:%d" i

(* Memory *)

(* The memory a path gives a device: its own, another GPU's mapped for it, or a
   view of its own GPU's, which maps nothing. A peer mapping keeps its owner's
   region, which a third GPU maps in turn. *)
type mem =
  | Own of Boot.t * Memory.region
  | Peer of { owner : Boot.t; region : Memory.region; mapped : Memory.region }
  | View of Boot.t * Memory.region

let origin_of = function
  | Own (o, r) | View (o, r) -> (o, r)
  | Peer { owner; region; _ } -> (owner, region)

(* Borrowed memory lies at its own address for the process, apart from where the
   GPU reaches it. *)
let host (r : Memory.region) =
  match (r.source, r.host) with
  | Borrowed a, _ -> Some a
  | _, Some w when Window.mapped w -> Some (Window.address w)
  | _ -> None

let memory (r : Memory.region) data =
  { Amd.address = r.mapping.va; host = host r; data }

(* The opened GPUs of this machine, by number, for [reaches]. *)
let opened : (int, Boot.t) Hashtbl.t = Hashtbl.create 4
let opened_lock = Mutex.create ()

(* A path function raises Rig_amd.Fault for any failure of the GPU but its
   refusal: a register sequence that does not complete is one. *)
let guard f = try f () with Regs.Stuck why -> raise (Amd.Fault why)

let alloc g kind n =
  let kind =
    match kind with
    | `Gpu -> Memory.Gpu
    | `Bar -> Memory.Bar
    | `System -> Memory.Host
  in
  match Boot.protect g (fun () -> Memory.alloc (Boot.memory g) kind n) with
  | Ok (Some r) -> Some (memory r (Own (g, r)))
  | Ok None -> None
  | Error why -> raise (Amd.Fault why)

let map_host g a n =
  match Boot.protect g (fun () -> Memory.map_host (Boot.memory g) a n) with
  | Ok r -> Some (memory r (Own (g, r)))
  | Error _ -> None

let reaches g ~index j =
  j = index
  ||
  match Mutex.protect opened_lock (fun () -> Hashtbl.find_opt opened j) with
  | Some o -> Boot.reaches g o
  | None -> false

let map_peer g (m : mem Amd.memory) =
  let owner, region = origin_of m.data in
  if owner == g then Some { m with data = View (owner, region) }
  else if not (Boot.reaches g owner) then None
  else
    let mapped () =
      Memory.map_peer (Boot.memory g) ~owner:(Boot.memory owner) region
    in
    match Boot.protect g mapped with
    | Ok mapped -> Some { m with data = Peer { owner; region; mapped } }
    | Error _ -> None

let free g (m : mem Amd.memory) =
  match m.data with
  | View _ -> ()
  | Peer { mapped; _ } ->
      Boot.protect g (fun () -> Memory.unmap (Boot.memory g) mapped)
  | Own (_, r) -> (
      let m = Boot.memory g in
      match r.source with
      | Memory.Allocated -> Boot.protect g (fun () -> Memory.free m r)
      | Borrowed _ | Peer -> Boot.protect g (fun () -> Memory.unmap m r))

(* Opening *)

let key : mem Type.Id.t = Type.Id.make ()

(* The reference clock of SOC15 and SOC21 GPUs, which the timestamps of their
   packets count. *)
let clock_hz = 100_000_000

(* Work whose timeline word has not moved for 30 s is a hang: no kernel bounds
   the GPU's work. *)
let hang_ms = 30_000

(* A compute unit's SIMDs: 4 on GC 9, 2 from GC 10 on (NUM_SIMD_PER_CU of the
   kernel's vega10_enum.h, navi10_enum.h, soc21_enum.h and soc24_enum.h). *)
let simds_per_cu (gpu : Rig_amd_abi.Gpu.t) =
  match gpu.gc with 9, _, _ -> 4 | _ -> 2

(* The interrupt context of a release. The path wakes on every entry of the
   interrupt ring, so any context but 0 serves. *)
let interrupt = 1

(* [finish ()] stops the GPU [g] once and gives its hold [h] back: released if
   it stopped clean after the device was made, lost otherwise, so that a GPU a
   failed open wrote to opens again only after a reset. *)
let finisher g h ~index ~made =
  let answer = ref None and lock = Mutex.create () in
  fun () ->
    Mutex.protect lock @@ fun () ->
    match !answer with
    | Some s -> s
    | None ->
        let s = Boot.stop g in
        answer := Some s;
        Mutex.protect opened_lock (fun () -> Hashtbl.remove opened index);
        (match s with
        | `Clean when Atomic.get made -> Gpus.release h
        | `Clean | `Lost | `Unknown -> Gpus.lose h);
        s

let path g fn ~index ~finish : mem Amd.path =
  let gc = Boot.gc g and gpu = Boot.gpu g in
  (* A fault the device raised itself, as a hang, is the GPU's as much as one
     its interrupt ring reported: the stop leaves the GPU lost. *)
  let stop ~fault =
    Option.iter (Boot.faulted g) fault;
    match finish () with `Clean | `Lost -> `Stopped | `Unknown -> `Unknown
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
    hang_ms = Some hang_ms;
    sleep = (fun ~ms -> guard (fun () -> Boot.sleep g ~ms));
    stable_power = (fun () -> guard (fun () -> Boot.stable_power g));
    stop;
  }

let start ~firmware ~index h fn =
  let* () =
    Machine.reserve
      (Rig_pci.Function.machine fn)
      ~base:(Rig_pci.Space.base Boot.space)
      (Rig_pci.Space.length Boot.space)
  in
  let boot () =
    match Boot.start fn (Rig_pci.Firmware.find firmware) with
    | r -> r
    | exception e ->
        let bt = Printexc.get_raw_backtrace () in
        Gpus.lose h;
        Printexc.raise_with_backtrace e bt
  in
  (* Firmware this library did not start, as a process that died leaves, is
     reset through the hold, and the GPU booted in full. *)
  let again = function
    | Ok g -> Ok g
    | Error (`Refused why) -> Error (`Refused why)
    | Error (`Lost why) -> Error (`Lost why)
    | Error `Running ->
        Error
          (`Refused
             "firmware this library did not start still runs on the GPU after \
              its reset")
  in
  let booted =
    match boot () with
    | Error `Running -> (
        match Gpus.renew h with
        | Error why -> Error (`Given_back why)
        | Ok () -> again (boot ()))
    | r -> again r
  in
  match booted with
  | Error (`Refused why | `Given_back why) -> Error why
  | Error (`Lost why) ->
      Gpus.lose h;
      Error why
  | Ok g -> (
      Mutex.protect opened_lock (fun () -> Hashtbl.replace opened index g);
      let made = Atomic.make false in
      let finish = finisher g h ~index ~made in
      match Amd.make (path g fn ~index ~finish) with
      | Ok d ->
          Atomic.set made true;
          (* A host resets a VF that holds its access long: every queue is made,
             so it goes back. *)
          Boot.give_back g;
          Ok (d, g)
      | Error why | (exception (Amd.Fault why | Regs.Stuck why)) ->
          ignore (finish ());
          Error why
      | exception e ->
          let bt = Printexc.get_raw_backtrace () in
          ignore (finish ());
          Printexc.raise_with_backtrace e bt)

(* The GPU's name in front of a refusal's message. *)
let named i r =
  Result.map_error (fun why -> strf "%s: %s" (device_name i) why) r

let open_ ?(machine = Machine.this) ~firmware i =
  index "open_" i;
  named i
  @@
  let* () =
    match Machine.files machine with
    | Some _ -> Ok ()
    | None -> Error "its machine is reached through a transport"
  in
  Gpus.open_ gpus machine i
    ~at_exit:(fun (_, g) -> ignore (Boot.stop g))
    (start ~firmware ~index:i)
  |> Result.map fst

(* Changes to the machine *)

let detach i =
  index "detach" i;
  named i (Gpus.detach gpus Machine.this i)

let attach i =
  index "attach" i;
  named i (Gpus.attach gpus Machine.this i)

let reset ?(machine = Machine.this) i =
  index "reset" i;
  named i (Gpus.reset gpus machine i)
