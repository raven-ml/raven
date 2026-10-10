(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* This machine's operations: functions taken through VFIO behind an IOMMU, or
   physically through /sys/bus/pci. *)

let strf = Printf.sprintf
let ( let* ) = Result.bind

external flock : Unix.file_descr -> unit = "caml_rig_pci_flock"
external file_map : Unix.file_descr -> int -> int -> int = "caml_rig_pci_map"
external file_unmap : int -> int -> unit = "caml_rig_pci_unmap"

let page = Sysmem.page
let round_page n = (n + page - 1) / page * page
let reserve = Sysmem.reserve

(* Locks *)

(* A function taken physically is locked by flock on its configuration space
   file: the lock is the file's, which every process that opens it shares.
   Behind VFIO the group's file admits one process at a time itself. *)
let lock files bus fd =
  let file = Sysfs.path files bus "config" in
  try flock fd with
  | Unix.Unix_error ((EAGAIN | EWOULDBLOCK), _, _) ->
      Fail.fail "%s is taken already; find who holds it: lsof %s" bus file
  | Unix.Unix_error (e, _, _) ->
      Fail.fail "locking %s with %s: %s" bus file (Unix.error_message e)

let rec update a f =
  let v = Atomic.get a in
  if not (Atomic.compare_and_set a v (f v)) then update a f

(* The configuration files this process locks for a change, each with the domain
   making it. A take inside a change, by that domain, shares the change's lock:
   the change holds the function for it. Two descriptors of one process exclude
   each other, so the take could not lock the file itself. *)
let changing = Atomic.make []
let in_change file = List.mem (file, Domain.self ()) (Atomic.get changing)

let locked files bus f =
  let file = Sysfs.path files bus "config" in
  match Unix.openfile file [ O_RDONLY; O_CLOEXEC ] 0 with
  | exception Unix.Unix_error (e, _, _) ->
      Error (strf "opening %s: %s" file (Unix.error_message e))
  | fd -> (
      Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
      match lock files bus fd with
      | exception Fail.Failed why -> Error why
      | () ->
          let key = (file, Domain.self ()) in
          update changing (List.cons key);
          Fun.protect ~finally:(fun () ->
              update changing (List.filter (fun k -> k <> key)))
          @@ f)

(* Taking *)

(* How a function is taken: through VFIO behind an IOMMU, or physically, its
   memory in files of its own. *)
type way = Iommu_take of Vfio.t | Physical_take of Sysmem.store

type taken = {
  files : Sysfs.t;
  bus : string;
  config : Unix.file_descr * int; (* where configuration space starts in it *)
  bars : (int * int) option array; (* read once: a held function keeps them *)
  seek : Mutex.t; (* a seek and its read or write, one at a time *)
  interrupts : Unix.file_descr option; (* the eventfd VFIO signals *)
  intx : int option; (* the INTx bit as found, if the take turned INTx off *)
  way : way;
  inherited : bool; (* a dead process left it reaching memory, at the take *)
  fds : Unix.file_descr list; (* every descriptor the take opened *)
}

(* Moves the [n] bytes of configuration space at [off] with [io]: the bytes
   moved, none if the system refused. *)
let config_io t io off b n =
  let fd, at = t.config in
  Mutex.protect t.seek @@ fun () ->
  match
    ignore (Unix.lseek fd (at + off) SEEK_SET);
    io fd b 0 n
  with
  | moved -> moved
  | exception Unix.Unix_error _ -> 0

(* Bytes the system does not give, as past the first 64 without CAP_SYS_ADMIN or
   of a function that left the bus, read as all ones, as the bus answers. *)
let config t off n =
  let b = Bytes.make n '\xff' in
  ignore (config_io t Unix.read off b n : int);
  let v = ref 0 in
  for i = n - 1 downto 0 do
    v := (!v lsl 8) lor Bytes.get_uint8 b i
  done;
  !v

(* A write the system refuses is dropped, as the bus drops one. *)
let set_config t off n x =
  let b = Bytes.init n (fun i -> Char.chr ((x lsr (8 * i)) land 0xff)) in
  ignore (config_io t Unix.single_write off b n : int);
  ignore (config t off n : int)

(* A function's command register, its bit that lets the function master the bus,
   reaching system memory by DMA, and its bit that keeps it from signalling
   legacy interrupts on its INTx line (PCI Express Base Specification,
   7.5.1.1.3). *)
let command = 0x04
let bus_master = 0x4
let intx_disable = 0x400
let stop_dma t = set_config t command 2 (config t command 2 land lnot bus_master)

(* Whether the function masters the bus no more: its bit reads 0, or the
   function left the bus and reads all ones. A write the system refused leaves
   it on, and its memory then stays, which leaks safely. *)
let stopped t =
  let c = config t command 2 in
  c land bus_master = 0 || c = 0xffff

let close_memory t s = if stopped t then Sysmem.close s

(* Turns INTx off: the bit as it was. *)
let intx_off t =
  let c = config t command 2 in
  set_config t command 2 (c lor intx_disable);
  c land intx_disable

(* Leaves a physical take's function as the take found it: its DMA stopped, its
   INTx as it was, and, once its bus mastering reads off, its memory files
   gone. *)
let give_back t s =
  stop_dma t;
  Option.iter
    (fun bit ->
      set_config t command 2 (config t command 2 land lnot intx_disable lor bit))
    t.intx;
  close_memory t s

(* The functions processes hold physically, with the process that took each.
   VFIO stops a function's DMA when its files close, at release or at exit;
   closing a physical take's files stops nothing, so this library does, before
   the memory the function reaches goes back to the system: then the memory
   files that list no function go. A child of fork inherits the list, and acts
   on none of it. The exit handler is registered when the library loads, before
   a driver above it registers its own, so it runs after theirs: a driver stops
   its GPU while it still reaches memory. It is a list in an atomic, so that a
   child of fork meets no held lock. *)
let physical = Atomic.make []
let hold t = update physical (List.cons (Unix.getpid (), t))
let unhold t = update physical (List.filter (fun (_, t') -> t' != t))

let () =
  at_exit (fun () ->
      let pid = Unix.getpid () in
      List.iter
        (fun (owner, t) ->
          if owner = pid then
            match t.way with
            | Physical_take s -> give_back t s
            | Iommu_take _ -> stop_dma t)
        (Atomic.get physical);
      Sysmem.exit ())

(* mmap maps whole pages from an offset on a page: the pages that hold the [n]
   bytes at [off]. *)
let pages off n =
  let first = off / page * page in
  (first, round_page (off + n) - first)

(* An empty window maps nothing: it is the BAR's bus address at [off]. Linux
   offers a prefetchable BAR's addresses write-combined through [resourceN_wc];
   VFIO maps BARs uncached. *)
let map t ~combine i off n =
  if n = 0 then Window.v (fst (Option.get t.bars.(i)) + off) 0
  else
    let first, len = pages off n in
    let window ?combines fd base =
      let a =
        Fail.step (strf "mapping BAR %d of %s" i t.bus) (fun () ->
            file_map fd (base + first) len)
      in
      Window.v ?combines (a + off - first) n
    in
    match t.way with
    | Iommu_take c ->
        window (Vfio.device c) (Vfio.bar_offset t.bus (Vfio.device c) i off n)
    | Physical_take _ ->
        let wc = Sysfs.path t.files t.bus (strf "resource%d_wc" i) in
        let combines = combine && Sys.file_exists wc in
        let file =
          if combines then wc
          else Sysfs.path t.files t.bus (strf "resource%d" i)
        in
        let fd =
          Fail.step ("opening " ^ file) (fun () ->
              Unix.openfile file [ O_RDWR; O_SYNC; O_CLOEXEC ] 0)
        in
        Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
        (* An access past a mapped file's end raises SIGBUS, which ends the
           process: Linux sizes a BAR's file as the BAR, so a shorter one is
           refused here. *)
        let bytes = (Unix.fstat fd).st_size in
        let size = snd (Option.get t.bars.(i)) in
        if bytes < size then
          Fail.fail "%s holds %d bytes, fewer than BAR %d's %d" file bytes i
            size;
        window ~combines fd 0

let unmap w =
  if Window.length w > 0 then
    let a, n = pages (Window.address w) (Window.length w) in
    Fail.bug (strf "unmapping %d bytes of a BAR at 0x%x" n a) (fun () ->
        file_unmap a n)

let interrupt t ms =
  match t.interrupts with Some fd -> Vfio.wait fd ms | None -> false

let reset t =
  match t.way with
  | Iommu_take c ->
      Fail.step ("resetting " ^ t.bus) (fun () -> Vfio.reset (Vfio.device c))
  | Physical_take _ -> Sysfs.reset t.files t.bus

let alloc_dma t ~contiguous ~va n =
  match t.way with
  | Physical_take s -> Sysmem.alloc ~contiguous ?va s n
  | Iommu_take c -> (
      let w = Sysmem.map ?va n in
      let n = Window.length w in
      match Vfio.map_dma "alloc_dma" t.bus c (Window.address w) n with
      | Some iova -> Some (w, [ (iova, n) ])
      | None ->
          Sysmem.unmap w;
          None
      | exception e ->
          Sysmem.unmap w;
          raise e)

let free_dma t w =
  match t.way with
  | Iommu_take c ->
      Vfio.unmap_dma t.bus c (Window.address w) (Window.length w);
      Sysmem.unmap w
  | Physical_take s -> Sysmem.free s w

(* Taken physically, a function reaches only memory that outlives the process:
   what alloc_dma gave on its machine. *)
let pin t a n =
  match t.way with
  | Physical_take s -> (
      match Sysmem.reach ~root:(Sysmem.root s) ~bus:t.bus a (round_page n) with
      | Some runs -> runs
      | None ->
          Fail.fail
            "%s reaches memory without an IOMMU and pins only memory alloc_dma \
             gave on its machine: the process's own pages go back to the \
             system when it dies; take it behind an IOMMU, bound to vfio-pci, \
             or give it memory from alloc_dma"
            t.bus)
  | Iommu_take c -> (
      let n = round_page n in
      match Vfio.map_dma "pin" t.bus c a n with
      | Some iova -> [ (iova, n) ]
      | None ->
          Fail.fail
            "%s has no device addresses left for %d bytes, or the process \
             reached its limit; %s"
            t.bus n Fail.memlock)

let unpin t a n =
  match t.way with
  | Physical_take _ -> Sysmem.unreach ~a ~n:(round_page n)
  | Iommu_take c -> Vfio.unmap_dma t.bus c a (round_page n)

(* A reset GPU reaches none of the memory processes that died left. Behind an
   IOMMU it never did: VFIO ended their access as their files closed. *)
let forget t =
  match t.way with
  | Physical_take s -> Sysmem.forget ~root:(Sysmem.root s) ~bus:t.bus
  | Iommu_take _ -> ()

(* A take's descriptors close whatever happens before. *)
let release t =
  Fun.protect ~finally:(fun () -> List.iter Unix.close t.fds) @@ fun () ->
  match t.way with
  | Iommu_take c -> Vfio.close c
  | Physical_take s ->
      unhold t;
      give_back t s

let fn t =
  {
    Ops.addressing =
      (match t.way with Physical_take _ -> Physical | Iommu_take _ -> Iommu);
    inherited = t.inherited;
    config = config t;
    set_config = set_config t;
    bar = (fun i -> if i < Array.length t.bars then t.bars.(i) else None);
    map =
      (fun ~combine i off n -> Fail.result (fun () -> map t ~combine i off n));
    unmap;
    interrupt = interrupt t;
    reset = (fun () -> Fail.result (fun () -> reset t));
    forget = (fun () -> Fail.result (fun () -> forget t));
    alloc_dma =
      (fun ~contiguous ~va n ->
        Fail.result (fun () -> alloc_dma t ~contiguous ~va n));
    free_dma = free_dma t;
    pin = (fun a n -> Fail.result (fun () -> pin t a n));
    unpin = unpin t;
    release = (fun () -> release t);
  }

let take_iommu files fds bus bars =
  let c, efd = Vfio.open_ files fds bus in
  {
    files;
    bus;
    config = (Vfio.device c, Vfio.config_offset bus (Vfio.device c));
    bars;
    seek = Mutex.create ();
    interrupts = Some efd;
    intx = None;
    way = Iommu_take c;
    inherited = false;
    fds = !fds;
  }

(* Bound to vfio-pci, a function taken physically has its interrupts through
   VFIO's no-IOMMU mode. Otherwise nothing handles them: the take turns its INTx
   off, which bus mastering does not gate, so that a GPU left running signals no
   line another device's handler shares. *)
let take_physical files fds bus bars =
  let file = Sysfs.path files bus "config" in
  let config =
    try Unix.openfile file [ O_RDWR; O_SYNC; O_CLOEXEC ] 0 with
    | Unix.Unix_error (((EACCES | EPERM) as e), _, _) ->
        Fail.fail
          "opening %s: %s; taking %s needs write access to its files under \
           /sys/bus/pci: run as root"
          file (Unix.error_message e) bus
    | Unix.Unix_error (e, _, _) ->
        Fail.fail "opening %s: %s" file (Unix.error_message e)
  in
  fds := config :: !fds;
  if not (in_change file) then lock files bus config;
  (* The first act of a process on the machine's memory: what processes that
     died left and no function reaches goes, without waiting for a reset. What
     the function still reaches stays, until its GPU is reset. *)
  let root = Sysfs.root files in
  Sysmem.collect_dead ~root;
  let inherited = Sysmem.left ~root ~bus in
  let interrupts =
    if Sysfs.driver files bus = Some "vfio-pci" then
      let _, _, efd = Vfio.open_function files fds bus Vfio.No_iommu in
      Some efd
    else None
  in
  let t =
    {
      files;
      bus;
      config = (config, 0);
      bars;
      seek = Mutex.create ();
      interrupts;
      intx = None;
      way = Physical_take (Sysmem.store ~root ~bus);
      inherited;
      fds = !fds;
    }
  in
  let t =
    if Option.is_some interrupts then t else { t with intx = Some (intx_off t) }
  in
  hold t;
  t

(* A failure gives back every descriptor taken. *)
(* CR: Hold the config-file lock before [access] and [bars]. Another
   process can attach after [access] chose Physical, so this take later
   disables INTx with the kernel driver bound. Have [locked] supply a
   private, single-use, nonescaping acquisition closure; pass it through
   one checked Function constructor, with Gpus' release check inside the
   same lock. Replace [in_change] with that explicit handoff and retain a
   duplicate lock fd until the physical take is released. Attach keeps its
   outer fd through reset/release and rebind. *)
let take files bus =
  if not (Sysfs.exists files bus) then
    Error (strf "%s is no PCI function of this machine" bus)
  else
    let fds = ref [] in
    let refused why =
      List.iter Unix.close !fds;
      Error why
    in
    match
      let* addressing = Sysfs.access files bus in
      let bars = Sysfs.bars files bus in
      let by =
        match addressing with
        | Ops.Iommu -> take_iommu
        | Physical -> take_physical
      in
      Ok (fn (by files fds bus bars))
    with
    | Ok _ as fn -> fn
    | Error why -> refused why
    | exception Fail.Failed why -> refused why

let ops files =
  {
    Ops.transport = Window.unsafe_transport 0;
    page;
    functions = (fun () -> Sysfs.functions files);
    take = take files;
    reserve = (fun ~base n -> Fail.result (fun () -> reserve ~base n));
  }
