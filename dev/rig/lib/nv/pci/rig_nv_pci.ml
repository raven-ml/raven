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

(* The GPU addresses of every NVIDIA GPU this path opens, [64 GiB, 1 TiB) of the
   process: below 2^40, as a device requires of every memory, and where an IOMMU
   maps device addresses. System memory lies at the same address for the process
   and the GPU, so the range is reserved on the GPU's machine. One range serves
   every GPU: a peer's memory maps at its owner's address. *)
let space = Space.create ~base:(64 * gib) ((1 lsl 40) - (64 * gib))

(* Host memory [map_host] mapped for a GPU: whole pages from [at], mapped once
   for every memory given over them, which [users] counts. *)
type range = {
  at : int;
  bytes : int;
  region : Memory.region;
  mutable users : int;
}

(* A GPU this path opened. [handles] holds the names the path gave its memory,
   which the RM's channel allocations take, with the region this GPU sees under
   each and the memory's offset in it. *)
type gpu = {
  index : int;
  machine : Machine.t;
  fn : Function.t;
  memory : Memory.t;
  handles : (int, Memory.region * int) Hashtbl.t;
  mutable next : int;
  mutable ranges : range list;
  lock : Mutex.t; (* its memory, page tables, handles and ranges *)
}

(* The memory a path gives a device: its own, another GPU's mapped for it, a
   view of its own GPU's, which maps nothing, or host memory in a range. A peer
   mapping keeps its owner's region, which a third GPU maps in turn. *)
type mem =
  | Own of gpu * Memory.region
  | Peer of { owner : gpu; region : Memory.region; mapped : Memory.region }
  | View of gpu * Memory.region
  | Host of gpu * range

let key : mem Type.Id.t = Type.Id.make ()
let protect g f = Mutex.protect g.lock f

let window_address (r : Memory.region) =
  match r.host with
  | Some w when Window.mapped w -> Some (Window.address w)
  | _ -> None

(* [memory g data r ~off ~host] names [data] for [g]: its first byte lies [off]
   bytes into [r], the region [g] sees. *)
let memory g data (r : Memory.region) ~off ~host =
  let handle =
    protect g @@ fun () ->
    let h = g.next in
    g.next <- h + 1;
    Hashtbl.replace g.handles h (r, off);
    h
  in
  { Rig_nv.address = r.mapping.va + off; host; handle; data }

let whole g data r = memory g data r ~off:0 ~host:(window_address r)

(* The memory and address a channel's USERD is described at: the physical
   address of GPU memory, the bus address of system memory, of the byte [off] of
   the memory of handle [h]. *)
let locate g h off =
  protect g @@ fun () ->
  match Hashtbl.find_opt g.handles h with
  | None -> None
  | Some ((r : Memory.region), start) ->
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
      Option.map (fun a -> (where, a)) (find (start + off) r.mapping.pages)

let alloc g kind n =
  let kind =
    match kind with
    | `Gpu -> Memory.Gpu
    | `Bar -> Memory.Bar
    | `System -> Memory.Host
  in
  match protect g (fun () -> Memory.alloc g.memory kind n) with
  | Ok (Some r) -> Some (whole g (Own (g, r)) r)
  | Ok None | Error _ -> None

let round_up n a = (n + a - 1) / a * a

(* The end of the GPU addresses a device takes memory at. *)
let addresses_end = 1 lsl 40

(* This GPU's own system memory that holds the [n] bytes at [a]: there the GPU
   addresses it as the process does. *)
let own g a n =
  let holds _ ((r : Memory.region), _) found =
    let m = r.mapping in
    match found with
    | Some _ -> found
    | None when r.source = Allocated && m.target = System ->
        if m.va <= a && a + n <= m.va + m.size then Some r else None
    | None -> None
  in
  protect g (fun () -> Hashtbl.fold holds g.handles None)

(* Host memory maps by whole pages, once for the pages of one range: a range
   inside one mapped already shares it, and one that overlaps a mapped range
   without lying inside it is refused. The GPU addresses it where the process
   does, which must lie below [addresses_end]. *)
let map_host g a n =
  let page = Machine.page g.machine in
  let at = a land lnot (page - 1) in
  let bytes = round_up (a + n) page - at in
  let inside r = r.at <= at && at + bytes <= r.at + r.bytes in
  let overlaps r = at < r.at + r.bytes && r.at < at + bytes in
  let range () =
    match List.find_opt inside g.ranges with
    | Some r ->
        r.users <- r.users + 1;
        Some r
    | None when List.exists overlaps g.ranges -> None
    | None when at + bytes > addresses_end -> None
    | None -> (
        match Memory.map_host g.memory at bytes with
        | Error _ -> None
        | Ok region ->
            let r = { at; bytes; region; users = 1 } in
            g.ranges <- r :: g.ranges;
            Some r)
  in
  match own g a n with
  | Some r ->
      Some (memory g (View (g, r)) r ~off:(a - r.mapping.va) ~host:(Some a))
  | None ->
      Option.map
        (fun r ->
          memory g (Host (g, r)) r.region ~off:(a - r.at) ~host:(Some a))
        (protect g range)

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
  match m.data with
  | Host (o, r) when o == g ->
      protect g (fun () -> r.users <- r.users + 1);
      let off = m.address - r.region.mapping.va in
      Some (memory g m.data r.region ~off ~host:m.host)
  | Host _ -> None
  | Own (owner, region) | View (owner, region) | Peer { owner; region; _ } -> (
      if owner == g then Some (whole g (View (owner, region)) region)
      else
        match
          protect g (fun () ->
              Memory.map_peer g.memory ~owner:owner.memory region)
        with
        | Ok mapped -> Some (whole g (Peer { owner; region; mapped }) mapped)
        | Error _ -> None)

let free g (m : mem Rig_nv.memory) =
  protect g @@ fun () ->
  Hashtbl.remove g.handles m.handle;
  match m.data with
  | View _ -> ()
  | Peer { mapped; _ } -> Memory.unmap g.memory mapped
  | Host (_, r) ->
      r.users <- r.users - 1;
      if r.users = 0 then begin
        g.ranges <- List.filter (fun r' -> r' != r) g.ranges;
        Memory.unmap g.memory r.region
      end
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
  let s = give_up hold g.fn ~unload:(fun () -> ignore (Gsp.unload gsp)) in
  Gsp.free gsp;
  s

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

(* [start] boots the GPU's GSP, writing to the GPU: from its first write, the
   GPU is in a state only a reset clears. *)
let start fn (c : Chip.t) (fw : Images.t) =
  Chip.bus_master fn true;
  let* () = Falcon.run c (Falcon.wait_reset c.family) in
  let* memory = Chip.memory c in
  let* bar = Function.map ~combine:false fn memory_bar in
  let top =
    Layout.top c.family ~memory ~boot:fw.bootloader.image.length
      ~image:fw.gsp.length
  in
  let tables =
    Page_table.create (Mmu.format c bar) space ~memory:top
      ~boot:(Gsp.boot_pool fw.start)
      ~tables:(if Window.length bar >= memory then Main else Pool)
      ~pages
  in
  (* The boot pool holds only the falcons' images: the GSP's objects come from
     the main pool. *)
  Page_table.booted tables;
  let* gsp = Gsp.boot { chip = c; memory; fn; tables; bar; space } fw in
  Ok (gsp, tables)

(* A failure from the first write on loses the GPU; one after the GSP started
   unloads it first, and gives its memory back once the GPU masters the bus no
   more. *)
let boot ~firmware ~index machine hold fn =
  let* c = Chip.of_function fn in
  let* () = started c fn ~index in
  let* fw = Images.read c.family firmware in
  let* () =
    Machine.reserve machine ~base:(Space.base space) (Space.length space)
  in
  match start fn c fw with
  | Error why | (exception Rig_nv.Fault why) ->
      Gpus.lose hold;
      Error why
  | Ok (gsp, tables) -> (
      let g =
        {
          index;
          machine;
          fn;
          memory = Memory.create fn tables ~bar:memory_bar;
          handles = Hashtbl.create 64;
          next = 1;
          ranges = [];
          lock = Mutex.create ();
        }
      in
      match device g ~gsp ~hold ~tables c with
      | Ok d ->
          Mutex.protect opened_lock (fun () -> opened := g :: !opened);
          Ok (d, gsp)
      | Error why | (exception Rig_nv.Fault why) ->
          ignore (give_up hold fn ~unload:(fun () -> ignore (Gsp.unload gsp)));
          Gsp.free gsp;
          Error why)

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

(* Bus mastering goes off before the reset: Linux restores the function's
   configuration as it was before a reset, its command register included, and
   the GPU must not master the bus after it, whatever ran on it. *)
let reset ?(machine = Machine.this) i =
  if i < 0 then invalid_argf "Rig_nv_pci.reset: index %d < 0" i;
  named i
  @@ Gpus.reset gpus machine i (fun fn ->
      Chip.bus_master fn false;
      Function.reset fn)
