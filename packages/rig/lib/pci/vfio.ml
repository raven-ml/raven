(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A function opened through VFIO, and the container that maps system memory for
   it behind an IOMMU. VFIO's requests raise Unix.Unix_error with their errno
   when refused, ENOSYS without Linux. *)

let strf = Printf.sprintf

(* The IOMMU models. *)
type model = Type1v2 | No_iommu

(* The regions of a function. *)
type region = Bar of int | Config
type fd = Unix.file_descr

external api_version : unit -> int = "caml_rig_pci_vfio_api_version"
external version : fd -> int = "caml_rig_pci_vfio_version"
external supports : fd -> model -> bool = "caml_rig_pci_vfio_supports"
external set_iommu : fd -> model -> unit = "caml_rig_pci_vfio_set_iommu"
external viable : fd -> bool = "caml_rig_pci_vfio_viable"
external set_container : fd -> fd -> unit = "caml_rig_pci_vfio_set_container"
external device_fd : fd -> string -> fd = "caml_rig_pci_vfio_open_device"

(* [region fd r] is [r]'s size, its offset in the function's file, whether it
   maps, and [Some areas] if only those [(offset, bytes)] parts of it map. *)
external region : fd -> region -> int * int * bool * (int * int) list option
  = "caml_rig_pci_vfio_region"

(* [msi device efd] routes the function's first MSI vector to [efd]. *)
external msi : fd -> fd -> unit = "caml_rig_pci_vfio_msi"
external reset : fd -> unit = "caml_rig_pci_vfio_reset"

(* [iommu container] is the page sizes the IOMMU maps as a bitmap, 0 if it does
   not say, and the [(first, last)] device address ranges it maps, [[]] if it
   does not say. *)
external iommu : fd -> int * (int * int) list = "caml_rig_pci_vfio_iommu"

(* [map container va iova n] maps, readable and writable, the [n] bytes at [va]
   at the device address [iova]; the kernel pins them. *)
external map : fd -> int -> int -> int -> unit = "caml_rig_pci_vfio_map"
external unmap : fd -> int -> int -> unit = "caml_rig_pci_vfio_unmap"
external eventfd : unit -> fd = "caml_rig_pci_eventfd"
external wait : fd -> int -> bool = "caml_rig_pci_wait"

(* Opening the function may reset it, which takes up to seconds. A name with a
   NUL byte names no function: the kernel would read a shorter one. *)
let open_device group bus : fd =
  if String.contains bus '\000' then
    raise (Unix.Unix_error (ENODEV, "ioctl", bus));
  device_fd group bus

let page = Sysmem.page
let round_page n = (n + page - 1) / page * page

(* Messages *)

(* Opens the VFIO file [file] of [bus], naming the step that grants it. *)
let open_file bus file =
  match Unix.openfile file [ O_RDWR; O_CLOEXEC ] 0 with
  | fd -> fd
  | exception Unix.Unix_error (((EACCES | EPERM) as e), _, _) ->
      Fail.fail
        "opening %s: %s; give the user the file with a udev rule, or run as \
         root; a no-IOMMU group also needs CAP_SYS_RAWIO"
        file (Unix.error_message e)
  | exception Unix.Unix_error (ENOENT, _, _) ->
      Fail.fail "%s does not exist; bind %s to vfio-pci" file bus
  | exception Unix.Unix_error (EBUSY, _, _) ->
      Fail.fail "%s is open in another process; find it: lsof %s" file file
  | exception Unix.Unix_error (e, _, _) ->
      Fail.fail "opening %s: %s" file (Unix.error_message e)

(* VFIO takes an IOMMU group whole: every function of it bound to vfio-pci or to
   a driver VFIO accepts. *)
let not_viable files bus g =
  match Sysfs.group_holders files g with
  | [] -> strf "IOMMU group %s of %s is not viable" g bus
  | held ->
      strf
        "IOMMU group %s of %s also holds %s, bound to other drivers; VFIO \
         takes a group whole: bind each to vfio-pci"
        g bus
        (String.concat ", " (List.map (fun (f, d) -> strf "%s (%s)" f d) held))

(* Opening *)

(* [bus]'s group file for the model [m], and the group's descriptor once
   viable. *)
let open_group files bus m =
  let g =
    match Sysfs.group files bus with
    | Some g -> g
    | None -> Fail.fail "%s is bound to vfio-pci but in no IOMMU group" bus
  in
  let file =
    match m with
    | Type1v2 -> Sysfs.vfio_file files g
    | No_iommu -> Sysfs.noiommu_file files g
  in
  (* VFIO's own file first: without it no group of the machine opens. *)
  let vfio = Sysfs.vfio_file files "vfio" in
  if not (Sys.file_exists vfio) then
    Fail.fail "%s does not exist; bind %s to vfio-pci" vfio bus;
  let group = open_file bus file in
  let status = "reading the status of " ^ file in
  match Fail.step status (fun () -> viable group) with
  | true -> (file, group)
  | false ->
      Unix.close group;
      Fail.fail "%s" (not_viable files bus g)
  | exception e ->
      Unix.close group;
      raise e

(* A new container with the model [m] that holds [group], open at [file]. *)
let new_container files bus file group m =
  let container = open_file bus (Sysfs.vfio_file files "vfio") in
  match
    let v =
      Fail.step "checking VFIO's API version" (fun () -> version container)
    in
    if v <> api_version () then
      Fail.fail "VFIO speaks API version %d, expected %d" v (api_version ());
    if
      not
        (Fail.step "checking VFIO's IOMMU models" (fun () ->
             supports container m))
    then
      Fail.fail "%s"
        (match m with
        | No_iommu ->
            "VFIO is not in its no-IOMMU mode; turn it on: echo 1 | sudo tee \
             /sys/module/vfio/parameters/enable_unsafe_noiommu_mode"
        | Type1v2 ->
            "VFIO has no type 1 IOMMU; load it: sudo modprobe \
             vfio_iommu_type1");
    Fail.step
      ("attaching " ^ file ^ " to a VFIO container")
      (fun () -> set_container group container);
    (* Linux attaches a group to an IOMMU only where the IOMMU also remaps the
       function's interrupts, so that it cannot raise another's. *)
    match set_iommu container m with
    | () -> ()
    | exception Unix.Unix_error (EPERM, _, _) when m = Type1v2 ->
        Fail.fail
          "the IOMMU of %s does not remap interrupts, so Linux refuses it to \
           VFIO; enable interrupt remapping in the firmware settings, or \
           accept the risk: echo 1 | sudo tee \
           /sys/module/vfio_iommu_type1/parameters/allow_unsafe_interrupts"
          bus
    | exception Unix.Unix_error (e, _, _) ->
        Fail.fail "attaching %s to a VFIO container: %s" file
          (Unix.error_message e)
  with
  | () -> container
  | exception e ->
      Unix.close container;
      raise e

(* The function [bus] of [group], whose first MSI vector goes to a new eventfd:
   their descriptors, each on [fds] once open. *)
let open_device fds bus group =
  let opened fd =
    fds := fd :: !fds;
    fd
  in
  let device =
    opened
      (Fail.step
         ("opening " ^ bus ^ " through VFIO")
         (fun () -> open_device group bus))
  in
  let efd = opened (Fail.step "creating an eventfd" eventfd) in
  Fail.step ("routing the interrupt of " ^ bus) (fun () -> msi device efd);
  (device, efd)

(* Opens [bus]'s group in a container of its own with the IOMMU model [m], then
   the function, whose first MSI vector goes to an eventfd. Each descriptor goes
   on [fds] once open. Is the container, the function's descriptor and the
   eventfd. *)
let open_function files fds bus m =
  let file, group = open_group files bus m in
  fds := group :: !fds;
  let container = new_container files bus file group m in
  fds := container :: !fds;
  let device, efd = open_device fds bus group in
  (container, device, efd)

(* The offset of BAR [i] in [device]'s file, if VFIO maps its [n] bytes from
   [off]. The parts of a BAR it maps are whole pages, so the pages that hold
   those bytes map too. *)
let bar_offset bus device i off n =
  let size, offset, mappable, areas =
    Fail.step (strf "reading region %d of %s" i bus) (fun () ->
        region device (Bar i))
  in
  let inside (o, k) = off >= o && off + n <= o + k in
  if (not mappable) || off + n > size then
    Fail.fail "VFIO does not map BAR %d of %s" i bus;
  (match areas with
  | Some areas when not (List.exists inside areas) ->
      Fail.fail "VFIO maps parts of BAR %d of %s, and none holds [0x%x, 0x%x)" i
        bus off (off + n)
  | _ -> ());
  offset

let config_offset bus device =
  let _, offset, _, _ =
    Fail.step ("reading the configuration space of " ^ bus) (fun () ->
        region device Config)
  in
  offset

(* Containers *)

(* Device addresses behind an IOMMU are taken from 4 GiB up to 2^44. 44 bits is
   the narrowest DMA mask Linux gives a function this library drives: amdgpu's
   for GC 11 and GC 12.0 (gmc_v11_0.c, gmc_v12_0.c), where GC 9.4.2 and later
   take 48 (gmc_v9_0.c), GC 12.1 52, NVIDIA's GPUs 47 from Turing to Ada and 52
   from Hopper (open-gpu-kernel-modules' hwproject.h,
   NV_CHIP_EXTENDED_SYSTEM_PHYSICAL_ADDRESS_BITS), and mlx5 and bnxt 64. An
   address a device truncates to 32 bits falls below 4 GiB, where nothing is
   mapped, and faults. The window is the largest part of that range the IOMMU
   maps. *)
let iova_low = 1 lsl 32
let iova_high = 1 lsl 44

(* The window of an IOMMU that maps pages of [page_sizes], a bitmap, at the
   [(first, last)] device address ranges, all of them if [[]]: [(base, n)]. *)
let window bus page_sizes ranges =
  let smallest = page_sizes land -page_sizes in
  if smallest > page then
    Fail.fail
      "the IOMMU of %s maps no page smaller than %d bytes, and system pages \
       are %d bytes"
      bus smallest page;
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
      Fail.fail
        "the IOMMU of %s maps no device addresses between 4 GiB and 16 TiB"
        bus
  | Some w -> w

let iova bus fd =
  let page_sizes, ranges =
    Fail.step ("reading the IOMMU of " ^ bus) (fun () -> iommu fd)
  in
  window bus page_sizes ranges

(* VFIO's container is an IOMMU's address space, which several groups may
   share: one container per machine holds every function the process takes
   behind an IOMMU there, so that system memory several of them reach maps,
   pins and counts once. A group that a container refuses, such as one behind
   another IOMMU, or whose IOMMU leaves out part of the container's window,
   joins the next container that takes it, or one of its own. Mappings are
   counted by their (address, bytes), as Function counts pins, because DMA
   memory and a pin may be the same pair, for one function or several. *)
type container = {
  fd : fd;
  window : int * int; (* its device addresses: (base, n) *)
  iova : Tlsf.t; (* under [mutex] *)
  maps : (int * int) Tables.Range.t; (* to (device address, count) *)
  mutex : Lock.t;
  pid : int; (* the process that opened it *)
  mutable users : int; (* the functions it holds, under [containers_lock] *)
  mutable closed : bool; (* once its last function is released *)
}

(* A function behind an IOMMU, in its container. *)
type t = { c : container; device : fd; mutable released : bool }

(* The containers, with the root of their machine's files. A child of fork
   inherits the list and the parent's descriptors, which share their IOMMU
   address space with the parent's: it joins only containers it opened. *)
let containers : (string * container) list ref = ref []
let containers_lock = Lock.create ()

let container files bus fd =
  match iova bus fd with
  | (base, n) as window ->
      let c =
        {
          fd;
          window;
          iova = Tlsf.create ~base n;
          maps = Tables.Range.create 64;
          mutex = Lock.create ();
          pid = Unix.getpid ();
          users = 0;
          closed = false;
        }
      in
      containers := (Sysfs.root files, c) :: !containers;
      c
  | exception e ->
      Unix.close fd;
      raise e

(* Whether [c]'s IOMMU still maps the whole of [c]'s window: a group joining
   it leaves out the device addresses its own IOMMU does not map, and Linux
   refuses the join only where they hold a mapping already. *)
let keeps bus c =
  match iommu c.fd with
  | page_sizes, ranges ->
      let base, n = window bus page_sizes ranges in
      let b, m = c.window in
      base <= b && b + m <= base + n
  | exception (Unix.Unix_error _ | Fail.Failed _) -> false

(* The container of [bus]'s group, open at [file], with a count for the
   function: the first of the machine's that this process opened, takes the
   group and keeps its window, or a new one. [group] holds the group's
   descriptor while one is open: a group taken out again by closing its file
   opens anew. *)
let join files bus file group =
  let root = Sysfs.root files and pid = Unix.getpid () in
  let reopen () =
    Unix.close (Option.get !group);
    group := None;
    group := Some (open_file bus file)
  in
  let rec next = function
    | [] ->
        let g = Option.get !group in
        container files bus (new_container files bus file g Type1v2)
    | c :: rest -> (
        match set_container (Option.get !group) c.fd with
        | exception Unix.Unix_error _ -> next rest
        | () when keeps bus c -> c
        | () ->
            reopen ();
            next rest)
  in
  let mine (r, c) = if r = root && c.pid = pid then Some c else None in
  let c = next (List.filter_map mine !containers) in
  c.users <- c.users + 1;
  c

(* Drops a count of [c]'s functions, closing it with the last, which removes
   every mapping it holds. *)
let leave c =
  Lock.protect containers_lock @@ fun () ->
  c.users <- c.users - 1;
  if c.users = 0 then begin
    Lock.protect c.mutex (fun () -> c.closed <- true);
    containers := List.filter (fun (_, c') -> c' != c) !containers;
    Unix.close c.fd
  end

(* Opens [bus] behind its IOMMU, its descriptors but its container's on [fds]:
   the function and the eventfd its interrupts signal. *)
let open_ files fds bus =
  let file, group = open_group files bus Type1v2 in
  let group = ref (Some group) in
  let c =
    match
      Lock.protect containers_lock (fun () -> join files bus file group)
    with
    | c -> c
    | exception e ->
        Option.iter Unix.close !group;
        raise e
  in
  let group = Option.get !group in
  fds := group :: !fds;
  match open_device fds bus group with
  | device, efd -> ({ c; device; released = false }, efd)
  | exception e ->
      leave c;
      raise e

let give_back c iova =
  if not (Tlsf.free c.iova iova) then
    invalid_arg (strf "Vfio: no device addresses at 0x%x" iova)

(* A mapping is aligned to the largest power of two not above its size, from
   a page, so that the IOMMU maps large ranges with large pages. *)
let alignment n =
  let rec go b = if b > n / 2 then b else go (b * 2) in
  Int.max page (go 1)

(* The device address at which [t]'s container maps the [n] bytes at [a],
   ending at or below [below], mapping them for the first count. A mapping
   another function made past [below] serves none. *)
let map_dma fn bus t ~below a n =
  let c = t.c in
  Lock.protect c.mutex @@ fun () ->
  if t.released then Fail.err_released fn bus;
  match Tables.Range.find_opt c.maps (a, n) with
  | Some (iova, _) when iova > below - n -> None
  | Some (iova, k) ->
      Tables.Range.replace c.maps (a, n) (iova, k + 1);
      Some iova
  | None -> (
      match Tlsf.alloc ~align:(alignment n) ~below c.iova n with
      | None -> None
      | Some iova -> (
          match map c.fd a iova n with
          | () ->
              Tables.Range.replace c.maps (a, n) (iova, 1);
              Some iova
          (* Mapping memory pins it: the kernel counts the pinned bytes
             against the process's locked-memory limit (ENOMEM) and the
             mapping against the container's dma_entry_limit (ENOSPC). *)
          | exception Unix.Unix_error ((ENOMEM | ENOSPC), _, _) ->
              give_back c iova;
              None
          | exception Unix.Unix_error (e, _, _) ->
              give_back c iova;
              Fail.fail "mapping %d bytes for %s: %s" n bus
                (Unix.error_message e)))

(* Drops a count of the [n] bytes at [a], unmapping them with the last. A
   closed container has unmapped them already. *)
let unmap_dma bus t a n =
  let c = t.c in
  Lock.protect c.mutex @@ fun () ->
  match Tables.Range.find c.maps (a, n) with
  | iova, k when k > 1 -> Tables.Range.replace c.maps (a, n) (iova, k - 1)
  | iova, _ ->
      if not c.closed then
        Fail.bug (strf "unmapping %d bytes for %s" n bus) (fun () ->
            unmap c.fd iova n);
      Tables.Range.remove c.maps (a, n);
      give_back c iova

let device t = t.device

let close t =
  Lock.protect t.c.mutex (fun () -> t.released <- true);
  leave t.c
