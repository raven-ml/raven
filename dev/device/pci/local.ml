(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* This machine's operations: functions taken through VFIO behind an IOMMU, or
   physically through /sys/bus/pci. *)

external lock_file : string -> int = "caml_device_pci_lock"
external close : int -> unit = "caml_device_pci_close"
external open_file : string -> bool -> int = "caml_device_pci_open"
external pread : int -> int -> int -> string = "caml_device_pci_pread"
external pwrite : int -> int -> string -> unit = "caml_device_pci_pwrite"
external file_map : int -> int -> int -> int = "caml_device_pci_map"
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
  | -1 ->
      Error
        (Printf.sprintf "%s is held by another process (see: lsof %s)" bus file)
  | fd -> Ok fd
  | exception Failure why -> Error (Printf.sprintf "%s: %s" file why)

(* Messages *)

let user () =
  try (Unix.getpwuid (Unix.getuid ())).pw_name
  with Not_found -> string_of_int (Unix.getuid ())

(* [f ()], whose VFIO request failing is reported as [what]. *)
let step what f =
  try f ()
  with Unix.Unix_error (e, _, _) ->
    failwith (Printf.sprintf "%s: %s" what (Unix.error_message e))

(* Opens the VFIO file [file] of [bus], naming the step that grants it. *)
let open_vfio bus file =
  match Vfio.open_ file with
  | fd -> fd
  | exception Unix.Unix_error ((EACCES | EPERM), _, _) ->
      let u = user () in
      failwith
        (Printf.sprintf
           "%s: permission denied; grant it to %s with a udev rule: echo \
            'SUBSYSTEM==\"vfio\", KERNEL==\"%s\", OWNER=\"%s\"' | sudo tee -a \
            /etc/udev/rules.d/90-raven-vfio.rules && sudo udevadm control \
            --reload && sudo udevadm trigger --action=add \
            --subsystem-match=vfio (a no-IOMMU group also needs CAP_SYS_RAWIO)"
           file u (Filename.basename file) u)
  | exception Unix.Unix_error (ENOENT, _, _) ->
      failwith
        (Printf.sprintf "%s does not exist; bind %s to vfio-pci: %s" file bus
           (Sysfs.bind_vfio bus))
  | exception Unix.Unix_error (EBUSY, _, _) ->
      failwith (Printf.sprintf "%s is open in another process" file)
  | exception Unix.Unix_error (e, _, _) ->
      failwith (Printf.sprintf "%s: %s" file (Unix.error_message e))

(* The functions of IOMMU group [g] another driver than VFIO's holds, with that
   driver. Bridges are held by pcieport, which VFIO accepts. *)
let held g =
  let dir = Printf.sprintf "/sys/kernel/iommu_groups/%s/devices" g in
  match Sys.readdir dir with
  | exception Sys_error _ -> []
  | fns ->
      Array.to_list fns |> List.sort String.compare
      |> List.filter_map (fun f ->
          match Sysfs.driver f with
          | Some d when not (List.mem d [ "vfio-pci"; "pci-stub"; "pcieport" ])
            ->
              Some (f, d)
          | _ -> None)

let not_viable bus g =
  let held = held g in
  Printf.sprintf
    "IOMMU group %s of %s holds functions of other drivers%s; bind each to \
     vfio-pci: %s"
    g bus
    (String.concat ""
       (List.map (fun (f, d) -> Printf.sprintf ", %s (%s)" f d) held))
    (String.concat " && " (List.map (fun (f, _) -> Sysfs.bind_vfio f) held))

(* Mapping memory behind an IOMMU pins it, and the kernel counts the pinned
   bytes against the process's locked-memory limit. *)
let map_error bus n (e : Unix.error) =
  let u = user () in
  match e with
  | ENOMEM ->
      Printf.sprintf
        "mapping %d bytes for %s: %s; the locked-memory limit (ulimit -l) \
         bounds the memory an IOMMU maps for a process: raise it for %s with \
         echo '%s - memlock unlimited' | sudo tee \
         /etc/security/limits.d/90-raven.conf, then log in again"
        n bus (Unix.error_message e) u u
  | ENOSPC ->
      Printf.sprintf
        "mapping %d bytes for %s: the IOMMU holds as many mappings as Linux \
         allows (the dma_entry_limit parameter of vfio_iommu_type1)"
        n bus
  | e ->
      Printf.sprintf "mapping %d bytes for %s: %s" n bus (Unix.error_message e)

(* VFIO *)

(* Opens [bus]'s group [file] in a container of its own with the IOMMU model
   [m], then the function, whose first MSI vector goes to an eventfd. Each
   descriptor goes on [files] once open. Is the container, the function's
   descriptor and the eventfd. *)
let open_function files bus g m file =
  let opened fd =
    files := fd :: !files;
    fd
  in
  let container = opened (open_vfio bus "/dev/vfio/vfio") in
  let supported =
    step "VFIO" (fun () -> Vfio.supports container (Vfio.model m))
  in
  if not supported then
    failwith
      (match m with
      | Vfio.No_iommu ->
          "VFIO is not in its no-IOMMU mode (set \
           /sys/module/vfio/parameters/enable_unsafe_noiommu_mode to 1)"
      | Vfio.Type1v2 ->
          "VFIO has no type 1 IOMMU (load it: sudo modprobe vfio_iommu_type1)");
  let group = opened (open_vfio bus file) in
  let viable =
    step ("reading the status of " ^ file) (fun () -> Vfio.viable group)
  in
  if not viable then failwith (not_viable bus g);
  step
    ("attaching " ^ file ^ " to a VFIO container")
    (fun () -> Vfio.set_container group container);
  (* Linux attaches a group to an IOMMU only where the IOMMU also remaps the
     function's interrupts, so that it cannot raise another's. *)
  (match Vfio.set_iommu container (Vfio.model m) with
  | () -> ()
  | exception Unix.Unix_error (EPERM, _, _) when m = Vfio.Type1v2 ->
      failwith
        (Printf.sprintf
           "the IOMMU of %s does not remap interrupts, so Linux refuses it to \
            VFIO; enable interrupt remapping in the firmware settings, or \
            accept the risk: echo 1 | sudo tee \
            /sys/module/vfio_iommu_type1/parameters/allow_unsafe_interrupts"
           bus)
  | exception Unix.Unix_error (e, _, _) ->
      failwith
        (Printf.sprintf "attaching %s to a VFIO container: %s" file
           (Unix.error_message e)));
  let device =
    opened
      (step
         ("opening " ^ bus ^ " through VFIO")
         (fun () -> Vfio.device group bus))
  in
  let efd = opened (step "creating an eventfd" Vfio.eventfd) in
  step ("routing the interrupt of " ^ bus) (fun () -> Vfio.msi device efd);
  (container, device, efd)

(* Device addresses behind an IOMMU are taken from 4 GiB up to 1 TiB: every GPU
   reaches 40 bits, and an address a device truncates to 32 bits falls below 4
   GiB, where nothing is mapped, and faults. The window is the largest part of
   that range the IOMMU maps. *)
let iova_low = 1 lsl 32
let iova_high = 1 lsl 40

let iova bus container =
  let page_sizes, ranges, _ =
    step ("reading the IOMMU of " ^ bus) (fun () -> Vfio.iommu container)
  in
  let smallest = page_sizes land -page_sizes in
  if smallest > page then
    failwith
      (Printf.sprintf "the IOMMU of %s maps no page smaller than %d bytes" bus
         smallest);
  let ranges = if ranges = [] then [ (0, max_int) ] else ranges in
  let piece (first, last) =
    let a = round_page (Int.min iova_high (Int.max first iova_low)) in
    let b =
      if last >= iova_high - 1 then iova_high else (last + 1) / page * page
    in
    if a < b then Some (a, b - a) else None
  in
  let best acc (a, n) =
    match acc with Some (_, m) when m >= n -> acc | _ -> Some (a, n)
  in
  match List.fold_left best None (List.filter_map piece ranges) with
  | None ->
      failwith "the IOMMU maps no device addresses between 4 GiB and 1 TiB"
  | Some (base, n) -> Space.create ~base n

(* The container of a function behind an IOMMU, which maps system memory for it
   at device addresses the process allocates. Mappings are counted by their
   (address, bytes), as pins are. Once the function is released, closing the
   container has removed them all. *)
type container = {
  fd : int;
  iova : Space.t;
  maps : (int * int, int * int) Hashtbl.t; (* to (device address, count) *)
  mutex : Mutex.t;
  mutable closed : bool;
}

(* The device address at which [c] maps the [n] bytes at [a], mapping them for
   the first count. *)
let map_dma bus c a n =
  Mutex.protect c.mutex @@ fun () ->
  if c.closed then failwith (bus ^ " is released");
  match Hashtbl.find_opt c.maps (a, n) with
  | Some (iova, k) ->
      Hashtbl.replace c.maps (a, n) (iova, k + 1);
      iova
  | None ->
      let iova =
        match Space.alloc c.iova n with
        | Some x -> x
        | None ->
            failwith
              (Printf.sprintf "%s has no device addresses left for %d bytes" bus
                 n)
      in
      (match Vfio.map c.fd a iova n with
      | () -> ()
      | exception Unix.Unix_error (e, _, _) ->
          Space.free c.iova iova;
          failwith (map_error bus n e));
      Hashtbl.replace c.maps (a, n) (iova, 1);
      iova

(* Drops a count of the [n] bytes at [a], unmapping them with the last. A
   released function's container has unmapped them already. *)
let unmap_dma bus c a n =
  Mutex.protect c.mutex @@ fun () ->
  match Hashtbl.find_opt c.maps (a, n) with
  | None -> invalid_arg (Printf.sprintf "%s: 0x%x is not mapped" bus a)
  | Some (iova, k) when k > 1 -> Hashtbl.replace c.maps (a, n) (iova, k - 1)
  | Some (iova, _) ->
      if not c.closed then
        step (Printf.sprintf "unmapping %d bytes for %s" n bus) (fun () ->
            Vfio.unmap c.fd iova n);
      Hashtbl.remove c.maps (a, n);
      Space.free c.iova iova

(* Taking *)

type taken = {
  bus : string;
  config : int * int; (* the descriptor, and where configuration space starts *)
  device : int option; (* the function's VFIO descriptor *)
  interrupts : int option; (* the eventfd VFIO signals *)
  container : container option;
  files : int list; (* every descriptor, the locks last *)
}

(* Configuration space shows every reader its first [header] bytes; past them
   Linux shows only a reader with CAP_SYS_ADMIN. *)
let header = 64

let config t off n =
  let fd, at = t.config in
  let s = pread fd (at + off) n in
  if String.length s < n then
    failwith
      (if off + n > header && t.container = None then
         Printf.sprintf
           "reading configuration space of %s past %d bytes needs CAP_SYS_ADMIN"
           t.bus header
       else Printf.sprintf "reading configuration space of %s at %d" t.bus off);
  let v = ref 0 in
  for i = n - 1 downto 0 do
    v := (!v lsl 8) lor Char.code s.[i]
  done;
  !v

let set_config t off n x =
  let fd, at = t.config in
  pwrite fd (at + off)
    (String.init n (fun i -> Char.chr ((x lsr (8 * i)) land 0xff)));
  ignore (config t off n)

let map t i off n =
  match (t.device, t.container) with
  | Some device, Some _ ->
      let size, offset, mappable, areas =
        step (Printf.sprintf "reading region %d of %s" i t.bus) (fun () ->
            Vfio.region device i false)
      in
      let inside (o, k) = off >= o && off + n <= o + k in
      if (not mappable) || off + n > size then
        failwith (Printf.sprintf "%s: VFIO does not map BAR %d" t.bus i);
      (match areas with
      | Some areas when not (List.exists inside areas) ->
          failwith
            (Printf.sprintf
               "%s: VFIO maps only parts of BAR %d, none of which holds [0x%x, \
                0x%x)"
               t.bus i off (off + n))
      | _ -> ());
      Window.v (file_map device (offset + off) n) n
  | _ ->
      let fd =
        open_file (Sysfs.path t.bus (Printf.sprintf "resource%d" i)) true
      in
      Fun.protect
        ~finally:(fun () -> close fd)
        (fun () -> Window.v (file_map fd off n) n)

let unmap w = file_unmap (Window.address w) (Window.length w)

let interrupt t ms =
  match t.interrupts with Some fd -> Vfio.wait fd ms | None -> false

(* A function answers again once its vendor ID reads other than all ones. *)
let reset_tries = 100
let reset_step_s = 0.01

let reset t =
  (match t.device with
  | Some device -> step ("resetting " ^ t.bus) (fun () -> Vfio.reset device)
  | None -> Sysfs.reset t.bus);
  let rec wait k =
    if config t 0 2 = 0xffff then
      if k = 0 then
        failwith (Printf.sprintf "%s does not answer after its reset" t.bus)
      else begin
        Unix.sleepf reset_step_s;
        wait (k - 1)
      end
  in
  wait reset_tries

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
      match map_dma t.bus c (Window.address w) n with
      | iova -> (w, [ (iova, n) ])
      | exception e ->
          Sysmem.free w;
          raise e)

let free_dma t w =
  Option.iter
    (fun c -> unmap_dma t.bus c (Window.address w) (Window.length w))
    t.container;
  Sysmem.free w

let pin t a n =
  match t.container with
  | None -> runs ~contiguous:false n (Sysmem.pin a n)
  | Some c ->
      if a mod page <> 0 then
        invalid_arg (Printf.sprintf "Function.pin: 0x%x is not on a page" a);
      let n = round_page n in
      [ (map_dma t.bus c a n, n) ]

let unpin t a n =
  match t.container with
  | None -> Sysmem.unpin a n
  | Some c -> unmap_dma t.bus c a (round_page n)

let release t =
  Option.iter
    (fun c -> Mutex.protect c.mutex (fun () -> c.closed <- true))
    t.container;
  List.iter close t.files

let fn t addressing =
  {
    Ops.addressing;
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
  let g = Option.get (Sysfs.group bus) in
  let fd, device, efd =
    open_function files bus g Vfio.Type1v2 ("/dev/vfio/" ^ g)
  in
  let _, config, _, _ =
    step ("reading the configuration space of " ^ bus) (fun () ->
        Vfio.region device 0 true)
  in
  let container =
    {
      fd;
      iova = iova bus fd;
      maps = Hashtbl.create 64;
      mutex = Mutex.create ();
      closed = false;
    }
  in
  {
    bus;
    config = (device, config);
    device = Some device;
    interrupts = Some efd;
    container = Some container;
    files = !files;
  }

(* Bound to vfio-pci, a function taken physically has its interrupts through
   VFIO's no-IOMMU mode. *)
let take_physical files bus =
  let interrupts =
    if Sysfs.driver bus = Some "vfio-pci" then
      let g = Option.get (Sysfs.group bus) in
      let _, _, efd =
        open_function files bus g Vfio.No_iommu ("/dev/vfio/noiommu-" ^ g)
      in
      Some efd
    else None
  in
  let config =
    try open_file (Sysfs.path bus "config") true
    with Failure why ->
      failwith
        (Printf.sprintf
           "%s; taking %s needs write access to its files under /sys/bus/pci \
            (run as root)"
           why bus)
  in
  files := config :: !files;
  {
    bus;
    config = (config, 0);
    device = None;
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
    let locked name =
      let* fd = lock bus name in
      files := fd :: !files;
      Ok ()
    in
    let refused why =
      List.iter close !files;
      Error why
    in
    match
      let* () = locked "nx" in
      let* () = locked name in
      let* addressing = Sysfs.access bus (Sysfs.state bus) in
      match addressing with
      | Ops.Iommu -> Ok (fn (take_iommu files bus) Iommu)
      | Physical -> Ok (fn (take_physical files bus) Physical)
    with
    | Ok _ as fn -> fn
    | Error why -> refused why
    | exception (Failure why | Sys_error why) -> refused why
    | exception Unix.Unix_error (e, f, arg) ->
        refused (Printf.sprintf "%s %s: %s" f arg (Unix.error_message e))
