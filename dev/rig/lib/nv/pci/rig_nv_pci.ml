(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Rig_pci

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let ( let* ) = Result.bind
let mib = 1 lsl 20
let gib = 1 lsl 30

module Chip = Chip
module Falcon = Falcon
module Gsp = Gsp
module Held = Held
module Images = Images
module Layout = Layout
module Mmu = Mmu
module Msgq = Msgq
module Vbios = Vbios

(* GPUs *)

(* The memory BAR, BAR 1. *)
let memory_bar = 1

let gpus =
  Gpus.make ~memory_bar ~nodes:Held.nodes (fun (id : Machine.id) ->
      Rig_nv.is_gpu ~vendor:id.vendor ~class_:id.class_)

let count ?(machine = Machine.this) () = List.length (Gpus.buses gpus machine)

let device_name i =
  if i < 0 then invalid_argf "Rig_nv_pci.device_name: index %d < 0" i;
  if i = 0 then "NV-PCI" else strf "NV-PCI:%d" i

let pinned = Images.pinned
let origin = Images.origin
let named i r = Result.map_error (fun why -> device_name i ^ ": " ^ why) r

(* Memory *)

(* The GPU addresses of every NVIDIA GPU this path opens, [272 GiB, 384 GiB) of
   the process: system memory lies at the same address for the process and the
   GPU, so the range is reserved on the GPU's machine. *)
let space = Space.create ~base:(272 * gib) (112 * gib)

(* A GPU this path opened. [handles] holds the names the path gave its memory,
   which the RM's channel allocations take, with the region this GPU sees under
   each. *)
type gpu = {
  index : int;
  machine : Machine.t;
  fn : Function.t;
  memory : Memory.t;
  handles : (int, Memory.region) Hashtbl.t;
  mutable next : int;
  lock : Mutex.t; (* its memory, page tables and handles *)
}

(* The memory a path gives a device: its own, another GPU's mapped for it, or a
   view of its own GPU's, which maps nothing. A peer mapping keeps its owner's
   region, which a third GPU maps in turn. *)
type mem =
  | Own of gpu * Memory.region
  | Peer of { owner : gpu; region : Memory.region; mapped : Memory.region }
  | View of gpu * Memory.region

let key : mem Type.Id.t = Type.Id.make ()

let origin_of = function
  | Own (o, r) | View (o, r) -> (o, r)
  | Peer { owner; region; _ } -> (owner, region)

let seen = function
  | Own (_, r) | View (_, r) -> r
  | Peer { mapped; _ } -> mapped

let protect g f = Mutex.protect g.lock f

let host (r : Memory.region) =
  match r.host with
  | Some w when Window.mapped w -> Some (Window.address w)
  | _ -> None

let memory g data =
  let r = seen data in
  let handle =
    protect g @@ fun () ->
    let h = g.next in
    g.next <- h + 1;
    Hashtbl.replace g.handles h r;
    h
  in
  { Rig_nv.address = r.mapping.va; host = host r; handle; data }

(* The memory and address a channel's USERD is described at: the physical
   address of GPU memory, the bus address of system memory, of the byte [off] of
   the memory of handle [h]. *)
let locate g h off =
  protect g @@ fun () ->
  match Hashtbl.find_opt g.handles h with
  | None -> None
  | Some (r : Memory.region) ->
      let rec find off = function
        | [] -> None
        | (a, n) :: _ when off < n -> Some (a + off)
        | (_, n) :: rest -> find (off - n) rest
      in
      let where =
        match r.mapping.target with
        | Page_table.Gpu -> `Gpu
        | System | Peer _ -> `System
      in
      Option.map (fun a -> (where, a)) (find off r.mapping.pages)

let alloc g kind n =
  let kind =
    match kind with
    | `Gpu -> Memory.Gpu
    | `Bar -> Memory.Bar
    | `System -> Memory.Host
  in
  match protect g (fun () -> Memory.alloc g.memory kind n) with
  | Ok (Some r) -> Some (memory g (Own (g, r)))
  | Ok None -> None
  | Error why -> raise (Rig_nv.Fault why)

let map_host g a n =
  match protect g (fun () -> Memory.map_host g.memory a n) with
  | Ok r -> Some (memory g (Own (g, r)))
  | Error _ -> None

(* The GPUs open, for [reaches]. *)
let opened : gpu list ref = ref []
let opened_lock = Mutex.create ()

let reaches g j =
  let physical g = Function.addressing g.fn = Machine.Physical in
  let owner () =
    List.find_opt (fun o -> o.index = j && o.machine == g.machine) !opened
  in
  j = g.index
  ||
  match Mutex.protect opened_lock owner with
  | Some o -> physical g && physical o && not (Memory.small_bar o.memory)
  | None -> false

let map_peer g (m : mem Rig_nv.memory) =
  let owner, region = origin_of m.data in
  if owner == g then Some (memory g (View (owner, region)))
  else
    match
      protect g (fun () -> Memory.map_peer g.memory ~owner:owner.memory region)
    with
    | Ok mapped -> Some (memory g (Peer { owner; region; mapped }))
    | Error _ -> None

let free g (m : mem Rig_nv.memory) =
  protect g @@ fun () ->
  Hashtbl.remove g.handles m.handle;
  match m.data with
  | View _ -> ()
  | Peer { mapped; _ } -> Memory.unmap g.memory mapped
  | Own (_, r) -> (
      match r.source with
      | Memory.Allocated -> Memory.free g.memory r
      | Borrowed | Peer -> Memory.unmap g.memory r)

(* Faults, hangs and stops *)

(* Work whose timeline word has not moved for 30 s is a hang: no kernel bounds
   the GPU's work, and no other program shares a GPU this path boots. *)
let hang_ms = 30_000

let check g gsp () =
  let why =
    match Gsp.check gsp with
    | Some _ as why -> why
    | None -> Function.failed g.fn
  in
  Option.iter (fun why -> raise (Rig_nv.Fault why)) why

(* The GSP stops every channel, unless the GPU cannot be reached, which an
   unload would wait for in vain; then the hold is lost, which turns the GPU's
   bus mastering off, so it reaches system memory no more whatever its channels
   do. Whether that write reached the GPU is known only if the GPU answered
   before it. The GPU opens again after a reset: the GSP runs on. *)
let give_up hold fn ~unload =
  let failed = Function.failed fn in
  if failed = None then unload ();
  Gpus.lose hold;
  match failed with None -> `Stopped | Some _ -> `Unknown

let stop g gsp hold () =
  Mutex.protect opened_lock (fun () ->
      opened := List.filter (fun o -> o != g) !opened);
  give_up hold g.fn ~unload:(fun () -> ignore (Gsp.unload gsp))

(* Opening *)

(* The blocks the page tables map memory with: 512 MiB, 2 MiB and 4 KiB. *)
let pages = [ (512 * mib, 512 * mib); (2 * mib, 2 * mib); (0x1000, 0x1000) ]

(* The usermode doorbell, [NVC361_NOTIFY_CHANNEL_PENDING], in BAR 0. *)
let doorbell_at = 0xbb0090

(* The start of the region the FMC protects (WPR2), in the GPU's memory. *)
let wpr2 c =
  Chip.field Defs.nv_pfb_pri_mmu_wpr2_addr_lo_val
    (Chip.get c Defs.nv_pfb_pri_mmu_wpr2_addr_lo)
  lsl 12

let started c fn ~index =
  if Chip.booted c then
    Error
      (strf
         "%s was booted before, by its kernel driver or another process; \
          Rig_nv_pci.reset %d resets it"
         (Function.bus fn) index)
  else Ok ()

let device g ~gsp ~hold ~tables (c : Chip.t) =
  let* () =
    match c.family with
    | Blackwell -> Layout.check ~wpr2:(wpr2 c) ~top:(Page_table.memory tables)
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
      map_host = map_host g;
      reaches = reaches g;
      map_peer = map_peer g;
      free = free g;
      register = (fun _ -> Ok ());
      unregister = (fun _ -> Ok ());
      check = check g gsp;
      hang_ms = Some hang_ms;
      stop = stop g gsp hold;
    }

(* [start] writes to the GPU: from its first write, the GPU is in a state only a
   reset clears, so a failure loses it. A failure after the GSP started unloads
   it first. *)
let start ~index machine hold fn (c : Chip.t) (fw : Images.t) =
  Chip.bus_master c true;
  let* () = Falcon.run c (Falcon.wait_reset c.family) in
  let* bar = Function.map ~combine:false fn memory_bar in
  let top =
    Layout.top c.family ~memory:c.memory ~boot:fw.bootloader.image.length
      ~image:fw.gsp.length
  in
  let tables =
    Page_table.create (Mmu.format c bar) space ~memory:top
      ~boot:(Gsp.boot_pool fw.start)
      ~tables:(if Window.length bar >= c.memory then Main else Pool)
      ~pages
  in
  (* The boot pool holds only the falcons' images: the GSP's objects come from
     the main pool. *)
  Page_table.booted tables;
  let* gsp = Gsp.boot { chip = c; fn; tables; bar; space } fw in
  let g =
    {
      index;
      machine;
      fn;
      memory = Memory.create fn tables ~bar:memory_bar;
      handles = Hashtbl.create 64;
      next = 1;
      lock = Mutex.create ();
    }
  in
  let opened_device =
    match device g ~gsp ~hold ~tables c with
    | r -> r
    | exception Rig_nv.Fault why -> Error why
  in
  match opened_device with
  | Ok d ->
      Mutex.protect opened_lock (fun () -> opened := g :: !opened);
      Ok (d, gsp)
  | Error why ->
      ignore (Gsp.unload gsp);
      Error why

let boot ~firmware ~index machine hold fn =
  let* c = Chip.of_function fn in
  let* () = started c fn ~index in
  let* fw = Images.read c.family firmware in
  let* () =
    Machine.reserve machine ~base:(Space.base space) (Space.length space)
  in
  let lose why =
    Gpus.lose hold;
    Error why
  in
  match start ~index machine hold fn c fw with
  | Ok _ as r -> r
  | Error why -> lose why
  | exception Rig_nv.Fault why -> lose why

let open_ ?(machine = Machine.this) ~firmware i =
  if i < 0 then invalid_argf "Rig_nv_pci.open_: index %d < 0" i;
  named i
  @@
  match Machine.files machine with
  | None ->
      Error
        "the GPU's machine is reached through a transport, through which \
         Rig_nv makes no submissions"
  | Some _ ->
      Gpus.open_ gpus machine i
        ~at_exit:(fun (_, gsp) -> ignore (Gsp.unload gsp))
        (fun hold fn ->
          match boot ~firmware ~index:i machine hold fn with
          | r -> r
          | exception Rig_nv.Fault why -> Error why)
      |> Result.map fst

(* Changes to the machine *)

let detach i =
  if i < 0 then invalid_argf "Rig_nv_pci.detach: index %d < 0" i;
  named i (Gpus.detach gpus Machine.this i)

let attach i =
  if i < 0 then invalid_argf "Rig_nv_pci.attach: index %d < 0" i;
  named i (Gpus.attach gpus Machine.this i)

(* The command register and its bus master bit (PCI Express Base Specification,
   7.5.1.1.3). *)
let command = 0x04
let bus_master = 0x4

let reset ?(machine = Machine.this) i =
  if i < 0 then invalid_argf "Rig_nv_pci.reset: index %d < 0" i;
  named i
  @@ Gpus.reset gpus machine i (fun fn ->
      Function.set_config16 fn command
        (Function.config16 fn command land lnot bus_master);
      Function.reset fn)
