(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* This machine's operations: functions taken through VFIO behind an IOMMU, or
   physically through /sys/bus/pci. *)

let strf = Printf.sprintf
let ( let* ) = Result.bind

external file_map : Unix.file_descr -> int -> int -> int = "caml_rig_pci_map"
external file_unmap : int -> int -> unit = "caml_rig_pci_unmap"

external pread : Unix.file_descr -> bytes -> int -> int -> int
  = "caml_rig_pci_pread"

external pwrite : Unix.file_descr -> string -> int -> int -> int
  = "caml_rig_pci_pwrite"

let page = Sysmem.page
let round_page n = (n + page - 1) / page * page
let reserve = Sysmem.reserve

let rec update a f =
  let v = Atomic.get a in
  if not (Atomic.compare_and_set a v (f v)) then update a f

(* Taking *)

(* How a function is taken: through VFIO behind an IOMMU, or physically, its
   memory in files of its own. *)
type way = Iommu_take of Vfio.t | Physical_take of Sysmem.store

type taken = {
  files : Sysfs.t;
  bus : string;
  config : Unix.file_descr * int; (* where configuration space starts in it *)
  bars : (int * int) option array; (* read once: a held function keeps them *)
  prefetchable : bool array; (* each BAR's, read with [bars] *)
  interrupts : Unix.file_descr option; (* the eventfd VFIO signals *)
  intx : int option; (* the INTx bit as found, if the take turned INTx off *)
  way : way;
  inherited : bool; (* a dead process left it reaching memory, at the take *)
  fds : Unix.file_descr list; (* every descriptor the take opened *)
}

(* The [n] bytes of configuration space at [off] are moved at their offset in
   the file, which a VFIO device's file requires: it has no position to seek
   to. A move the system refuses moves nothing. Linux 6.12 refuses one with
   EIO where VFIO cannot wake the function, EFAULT past a VFIO function's
   configuration space, ENODEV once the function has left the bus, and EPERM
   for a write the kernel's lockdown forbids. Behind VFIO a move also gives
   the host bridge's answer, as pcibios_err_to_errno turns it into an errno:
   ENOENT, ENOTTY, ENODEV, EFAULT, EIO, ENOSPC or ERANGE. Any other errno, such
   as EBADF or ESPIPE, comes from a mistake of this library's and raises. *)
let refusal : Unix.error -> bool = function
  | EIO | EFAULT | ENODEV | EPERM | ENOENT | ENOTTY | ENOSPC | ERANGE -> true
  | _ -> false

let config_read t off b n =
  let fd, at = t.config in
  Fail.bug ("reading configuration space of " ^ t.bus) @@ fun () ->
  try pread fd b n (at + off)
  with Unix.Unix_error (e, _, _) when refusal e -> 0

let config_write t off s n =
  let fd, at = t.config in
  Fail.bug ("writing configuration space of " ^ t.bus) @@ fun () ->
  try pwrite fd s n (at + off)
  with Unix.Unix_error (e, _, _) when refusal e -> 0

(* Bytes the system does not give, as past the first 64 without CAP_SYS_ADMIN or
   of a function that left the bus, read as all ones, as the bus answers. *)
let config t off n =
  let b = Bytes.make n '\xff' in
  ignore (config_read t off b n : int);
  let v = ref 0 in
  for i = n - 1 downto 0 do
    v := (!v lsl 8) lor Bytes.get_uint8 b i
  done;
  !v

(* A write the system refuses is dropped, as the bus drops one. *)
let set_config t off n x =
  let s = String.init n (fun i -> Char.chr ((x lsr (8 * i)) land 0xff)) in
  ignore (config_write t off s n : int);
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
   its GPU while it still reaches memory. It turns every function's bus
   mastering off before it gives any back, since one function may write
   memory another's take holds, such as a network adapter writing a GPU's
   memory through its BAR (Memory.bus). It is a list in an atomic, so that
   a child of fork meets no held lock. An exit from a callback inside a lock
   may hold a lock the handler takes: that take raises, the handler prints it,
   and goes on with the other functions. *)
let physical = Atomic.make []
let hold t = update physical (List.cons (Unix.getpid (), t))
let unhold t = update physical (List.filter (fun (_, t') -> t' != t))

let () =
  at_exit (fun () ->
      let pid = Unix.getpid () in
      let each f (owner, t) =
        if owner = pid then
          try f t
          with e ->
            prerr_endline
              (strf "releasing %s at exit: %s" t.bus (Printexc.to_string e))
      in
      let give_back t =
        match t.way with Physical_take s -> give_back t s | Iommu_take _ -> ()
      in
      let held = Atomic.get physical in
      List.iter (each stop_dma) held;
      List.iter (each give_back) held;
      try Sysmem.exit ()
      with e ->
        prerr_endline
          (strf "releasing system memory at exit: %s" (Printexc.to_string e)))

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
      Window.v ?combines ~prefetchable:t.prefetchable.(i) (a + off - first) n
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

(* Physical addresses are the kernel's to choose: Function refuses those past
   [below]. *)
let alloc_dma t ~contiguous ~va ~below n =
  match t.way with
  | Physical_take s -> Sysmem.alloc ~contiguous ?va s n
  | Iommu_take c -> (
      let w = Sysmem.map ?va n in
      let n = Window.length w in
      match Vfio.map_dma "alloc_dma" t.bus c ~below (Window.address w) n with
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
  | Physical_take _ -> Sysmem.free w

(* Taken physically, a function reaches only memory that outlives the process:
   what alloc_dma gave on its machine. *)
let pin t ~below a n =
  match t.way with
  | Physical_take s -> (
      match Sysmem.reach ~root:(Sysmem.root s) ~bus:t.bus a (round_page n) with
      | Some _ as runs -> runs
      | None ->
          Fail.fail
            "%s reaches memory without an IOMMU and pins only memory alloc_dma \
             gave on its machine: the process's own pages go back to the \
             system when it dies; take it behind an IOMMU, bound to vfio-pci, \
             or give it memory from alloc_dma"
            t.bus)
  | Iommu_take c ->
      let n = round_page n in
      Option.map
        (fun iova -> [ (iova, n) ])
        (Vfio.map_dma "pin" t.bus c ~below a n)

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

(* Behind an IOMMU a take's descriptors close whatever happens before: Linux
   then stops the function's DMA. A physical take's close only once it is
   given back and has left the exit's list, so that an exception before
   leaves the take, files open and the function locked, for the exit to give
   back, and the exit gives back no take through descriptors closed, whose
   numbers files opened since may hold. *)
let release t =
  match t.way with
  | Iommu_take c ->
      Fun.protect ~finally:(fun () -> List.iter Unix.close t.fds) @@ fun () ->
      Vfio.close c
  | Physical_take s ->
      give_back t s;
      unhold t;
      List.iter Unix.close t.fds

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
      (fun ~contiguous ~va ~below n ->
        Fail.result (fun () -> alloc_dma t ~contiguous ~va ~below n));
    free_dma = free_dma t;
    pin = (fun ~below a n -> Fail.result (fun () -> pin t ~below a n));
    unpin = unpin t;
    release = (fun () -> release t);
  }

(* A failure once the function is in its container gives the container its
   user back, so that a container left with none closes. *)
let take_iommu files fds bus bars prefetchable =
  let c, efd = Vfio.open_ files fds bus in
  let at =
    match Vfio.config_offset bus (Vfio.device c) with
    | at -> at
    | exception e ->
        let bt = Printexc.get_raw_backtrace () in
        Vfio.close c;
        Printexc.raise_with_backtrace e bt
  in
  {
    files;
    bus;
    config = (Vfio.device c, at);
    bars;
    prefetchable;
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
let take_physical l fds bars prefetchable =
  let files = Sysfs.files l and bus = Sysfs.bus l in
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
  let lock = Fail.step ("locking " ^ file) (fun () -> Sysfs.keep l) in
  fds := lock :: !fds;
  (* The first act of a process on the machine's memory: what processes that
     died left and no function reaches goes, without waiting for a reset. What
     the function still reaches stays, until its GPU is reset. *)
  let root = Sysfs.root files in
  Sysmem.collect_dead ~root;
  let inherited = Sysmem.left ~root ~bus in
  let interrupts =
    if Sysfs.driver l = Some Sysfs.vfio_pci then
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
      prefetchable;
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

(* A failure gives back every descriptor taken, as does a bug's [Failure].
   Behind VFIO the group's file admits one process at a time; taken
   physically, the function keeps its lock until its release. *)
let take l =
  let fds = ref [] in
  let refused why =
    List.iter Unix.close !fds;
    Error why
  in
  match
    let* addressing = Sysfs.access l in
    let bars = Sysfs.bars l and prefetchable = Sysfs.prefetchable l in
    match addressing with
    | Ops.Iommu ->
        Ok
          (fn
             (take_iommu (Sysfs.files l) fds (Sysfs.bus l) bars prefetchable))
    | Physical -> Ok (fn (take_physical l fds bars prefetchable))
  with
  | Ok _ as fn -> fn
  | Error why -> refused why
  | exception Fail.Failed why -> refused why
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      List.iter Unix.close !fds;
      Printexc.raise_with_backtrace e bt

let ops files =
  {
    Ops.transport = Window.unsafe_transport 0;
    page;
    functions = (fun () -> Sysfs.functions files);
    take =
      (fun bus ->
        if not (Sysfs.exists files bus) then
          Error (strf "%s is no PCI function of this machine" bus)
        else Sysfs.locked files bus take);
    reserve = (fun ~base n -> Fail.result (fun () -> reserve ~base n));
  }
