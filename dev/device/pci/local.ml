(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* This machine's operations: functions taken through VFIO behind an IOMMU, or
   physically through /sys/bus/pci. *)

external lock_file : string -> Unix.file_descr = "caml_device_pci_lock"
external file_map : Unix.file_descr -> int -> int -> int = "caml_device_pci_map"
external file_unmap : int -> int -> unit = "caml_device_pci_unmap"

let page = Sysmem.page
let round_page n = (n + page - 1) / page * page
let functions = Sysfs.functions
let reserve = Sysmem.reserve

(* Lock files *)

(* Locks [bus] under [name] in the temporary directory: its descriptor, or why
   not. *)
let lock bus name =
  let file =
    Filename.concat
      (Filename.get_temp_dir_name ())
      (Printf.sprintf "%s_%s.lock" name (String.lowercase_ascii bus))
  in
  match lock_file file with
  | fd -> Ok fd
  | exception Unix.Unix_error ((EAGAIN | EWOULDBLOCK), _, _) ->
      Error
        (Printf.sprintf "%s is held by another process (see: lsof %s)" bus file)
  | exception Unix.Unix_error (e, _, _) ->
      Error (Printf.sprintf "%s: %s" file (Unix.error_message e))
  | exception Failure why -> Error (Printf.sprintf "%s: %s" file why)

(* The lock every process of this library takes on a function. *)
let own_lock = "nx"

let locked bus f =
  match lock bus own_lock with
  | Error _ as e -> e
  | Ok fd -> Fun.protect ~finally:(fun () -> Unix.close fd) f

(* Taking *)

type taken = {
  bus : string;
  config : Unix.file_descr * int; (* where configuration space starts in it *)
  seek : Mutex.t; (* a seek and its read or write, one at a time *)
  interrupts : Unix.file_descr option; (* the eventfd VFIO signals *)
  container : Vfio.t option; (* behind an IOMMU *)
  files : Unix.file_descr list; (* every descriptor, the locks last *)
}

(* Moves the [n] bytes of configuration space at [off] with [io]. *)
let config_io t io off b n =
  let fd, at = t.config in
  Vfio.step ("configuration space of " ^ t.bus) @@ fun () ->
  Mutex.protect t.seek @@ fun () ->
  ignore (Unix.lseek fd (at + off) SEEK_SET);
  io fd b 0 n

let config t off n =
  let b = Bytes.create n in
  if config_io t Unix.read off b n < n then
    failwith
      (if off + n > Sysfs.header && Option.is_none t.container then
         Printf.sprintf
           "reading configuration space of %s past %d bytes needs CAP_SYS_ADMIN"
           t.bus Sysfs.header
       else Printf.sprintf "reading configuration space of %s at %d" t.bus off);
  let v = ref 0 in
  for i = n - 1 downto 0 do
    v := (!v lsl 8) lor Bytes.get_uint8 b i
  done;
  !v

let set_config t off n x =
  let b = Bytes.init n (fun i -> Char.chr ((x lsr (8 * i)) land 0xff)) in
  if config_io t Unix.single_write off b n < n then
    failwith
      (Printf.sprintf "writing configuration space of %s at %d" t.bus off);
  ignore (config t off n)

(* mmap maps whole pages from an offset on a page: the pages that hold the [n]
   bytes at [off], at least one. *)
let pages off n =
  let first = off / page * page in
  (first, Int.max page (round_page (off + n) - first))

(* An empty window maps the BAR's first page, which every BAR has, so that its
   address is its own. *)
let map t i off n =
  let off = if n = 0 then 0 else off in
  let first, len = pages off n in
  let window fd base =
    Window.v (file_map fd (base + first) len + off - first) n
  in
  match t.container with
  | Some c -> window c.device (Vfio.bar_offset t.bus c.device i off n)
  | None ->
      let file = Sysfs.path t.bus (Printf.sprintf "resource%d" i) in
      let fd =
        Vfio.step file (fun () ->
            Unix.openfile file [ O_RDWR; O_SYNC; O_CLOEXEC ] 0)
      in
      Fun.protect ~finally:(fun () -> Unix.close fd) (fun () -> window fd 0)

let unmap w =
  let a, n = pages (Window.address w) (Window.length w) in
  file_unmap a n

let interrupt t ms =
  match t.interrupts with Some fd -> Vfio.wait fd ms | None -> false

let reset t =
  match t.container with
  | Some c -> Vfio.step ("resetting " ^ t.bus) (fun () -> Vfio.reset c.device)
  | None -> Sysfs.reset t.bus

(* Physical addresses as runs: one per page, or one for contiguous memory. *)
let runs ~contiguous n = function
  | first :: _ when contiguous -> [ (first, n) ]
  | pages -> List.map (fun p -> (p, page)) pages

let alloc_dma t ~contiguous ~va n =
  match t.container with
  | None ->
      let w, pages = Sysmem.alloc ~contiguous ?va n in
      (w, runs ~contiguous (Window.length w) pages)
  | Some c -> (
      let w = Sysmem.map ?va n in
      let n = Window.length w in
      match Vfio.map_dma t.bus c (Window.address w) n with
      | iova -> (w, [ (iova, n) ])
      | exception e ->
          Sysmem.free w;
          raise e)

let free_dma t w =
  Option.iter
    (fun c -> Vfio.unmap_dma t.bus c (Window.address w) (Window.length w))
    t.container;
  Sysmem.free w

let pin t a n =
  match t.container with
  | None -> runs ~contiguous:false n (Sysmem.pin a n)
  | Some c ->
      let n = round_page n in
      [ (Vfio.map_dma t.bus c a n, n) ]

let unpin t a n =
  match t.container with
  | None -> Sysmem.unpin a n
  | Some c -> Vfio.unmap_dma t.bus c a (round_page n)

let release t =
  Option.iter Vfio.close t.container;
  List.iter Unix.close t.files

let fn t =
  {
    Ops.addressing =
      (match t.container with None -> Physical | Some _ -> Iommu);
    config = config t;
    set_config = set_config t;
    bar = Sysfs.bar t.bus;
    map = map t;
    unmap;
    interrupt = interrupt t;
    reset = (fun () -> reset t);
    alloc_dma = alloc_dma t;
    free_dma = free_dma t;
    pin = pin t;
    unpin = unpin t;
    release = (fun () -> release t);
  }

let take_iommu files bus =
  let c, efd = Vfio.open_ files bus in
  {
    bus;
    config = (c.device, Vfio.config_offset bus c.device);
    seek = Mutex.create ();
    interrupts = Some efd;
    container = Some c;
    files = !files;
  }

(* Bound to vfio-pci, a function taken physically has its interrupts through
   VFIO's no-IOMMU mode. *)
let take_physical files bus =
  let interrupts =
    if Sysfs.driver bus = Some "vfio-pci" then
      let _, _, efd = Vfio.open_function files bus Vfio.No_iommu in
      Some efd
    else None
  in
  let file = Sysfs.path bus "config" in
  let config =
    try Unix.openfile file [ O_RDWR; O_SYNC; O_CLOEXEC ] 0
    with Unix.Unix_error (e, _, _) ->
      failwith
        (Printf.sprintf
           "%s: %s; taking %s needs write access to its files under \
            /sys/bus/pci (run as root)"
           file (Unix.error_message e) bus)
  in
  files := config :: !files;
  {
    bus;
    config = (config, 0);
    seek = Mutex.create ();
    interrupts;
    container = None;
    files = !files;
  }

(* The function's own lock, which every process of this library takes, then
   [lock]'s, which other programs driving the same GPU take. A failure gives
   back every descriptor taken. *)
let take ~lock:name bus =
  if not (Sysfs.exists bus) then
    Error (Printf.sprintf "%s is no PCI function of this machine" bus)
  else
    let files = ref [] in
    let ( let* ) = Result.bind in
    let hold name =
      let* fd = lock bus name in
      files := fd :: !files;
      Ok ()
    in
    let refused why =
      List.iter Unix.close !files;
      Error why
    in
    match
      let* () = hold own_lock in
      let* () = hold name in
      let* addressing = Sysfs.access bus (Sysfs.state bus) in
      let by =
        match addressing with
        | Ops.Iommu -> take_iommu
        | Physical -> take_physical
      in
      Ok (fn (by files bus))
    with
    | Ok _ as fn -> fn
    | Error why -> refused why
    | exception (Failure why | Sys_error why) -> refused why
    | exception Unix.Unix_error (e, f, arg) ->
        refused (Printf.sprintf "%s %s: %s" f arg (Unix.error_message e))

let ops = { Ops.transport = Window.unsafe_transport 0; page; functions; take; reserve }
