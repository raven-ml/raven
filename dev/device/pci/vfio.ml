(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A function opened through VFIO, and the container that maps system memory for
   it behind an IOMMU. VFIO's requests (device_pci_vfio.c) raise Unix.Unix_error
   with their errno when refused, ENOSYS without Linux. *)

(* The IOMMU models. device_pci_vfio.c reads the constructors in this order as
   its enum model: keep the two in sync. *)
type model = Type1v2 | No_iommu

(* VFIO_API_VERSION, the version of VFIO's requests; device_pci_vfio.c asserts
   it against <linux/vfio.h>. *)
let api_version = 0

(* VFIO_PCI_CONFIG_REGION_INDEX: the region of configuration space. *)
let config_region = 7

type fd = Unix.file_descr

external version : fd -> int = "caml_device_pci_vfio_version"
external supports : fd -> model -> bool = "caml_device_pci_vfio_supports"
external viable : fd -> bool = "caml_device_pci_vfio_viable"
external set_container : fd -> fd -> unit = "caml_device_pci_vfio_set_container"
external set_iommu : fd -> model -> unit = "caml_device_pci_vfio_set_iommu"
external device : fd -> string -> fd = "caml_device_pci_vfio_device"

external region : fd -> int -> int * int * bool * (int * int) list option
  = "caml_device_pci_vfio_region"

external msi : fd -> fd -> unit = "caml_device_pci_vfio_msi"
external reset : fd -> unit = "caml_device_pci_vfio_reset"
external iommu : fd -> int * (int * int) list = "caml_device_pci_vfio_iommu"
external map : fd -> int -> int -> int -> unit = "caml_device_pci_vfio_map"
external unmap : fd -> int -> int -> unit = "caml_device_pci_vfio_unmap"
external eventfd : unit -> fd = "caml_device_pci_eventfd"
external wait : fd -> int -> bool = "caml_device_pci_wait"

let page = Sysmem.page
let round_page n = (n + page - 1) / page * page

(* Messages *)

let user () =
  try (Unix.getpwuid (Unix.getuid ())).pw_name
  with Not_found -> string_of_int (Unix.getuid ())

(* [f ()], whose system call failing is reported as [what]. *)
let step what f =
  try f ()
  with Unix.Unix_error (e, _, _) ->
    failwith (Printf.sprintf "%s: %s" what (Unix.error_message e))

(* Opens the VFIO file [file] of [bus], naming the step that grants it. *)
let open_file bus file =
  match Unix.openfile file [ O_RDWR; O_CLOEXEC ] 0 with
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
      failwith
        (Printf.sprintf "%s is open in another process; find it: lsof %s" file
           file)
  | exception Unix.Unix_error (e, _, _) ->
      failwith (Printf.sprintf "opening %s: %s" file (Unix.error_message e))

(* VFIO takes an IOMMU group whole: every function of it bound to vfio-pci or to
   a driver VFIO accepts. *)
let not_viable bus g =
  match Sysfs.group_holders g with
  | [] -> Printf.sprintf "IOMMU group %s of %s is not viable" g bus
  | held ->
      Printf.sprintf
        "IOMMU group %s of %s also holds %s, bound to other drivers; VFIO \
         takes a group whole: bind each to vfio-pci: %s"
        g bus
        (String.concat ", "
           (List.map (fun (f, d) -> Printf.sprintf "%s (%s)" f d) held))
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
         allows; raise the limit in \
         /sys/module/vfio_iommu_type1/parameters/dma_entry_limit"
        n bus
  | e ->
      Printf.sprintf "mapping %d bytes for %s: %s" n bus (Unix.error_message e)

(* Opening *)

(* Opens [bus]'s group in a container of its own with the IOMMU model [m], then
   the function, whose first MSI vector goes to an eventfd. Each descriptor goes
   on [files] once open. Is the container, the function's descriptor and the
   eventfd. *)
let open_function files bus m =
  let opened fd =
    files := fd :: !files;
    fd
  in
  let g = Option.get (Sysfs.group bus) in
  let file =
    match m with
    | Type1v2 -> "/dev/vfio/" ^ g
    | No_iommu -> Sysfs.noiommu_file g
  in
  let container = opened (open_file bus "/dev/vfio/vfio") in
  let v = step "checking VFIO's API version" (fun () -> version container) in
  if v <> api_version then
    failwith
      (Printf.sprintf "VFIO speaks API version %d, expected %d" v api_version);
  if not (step "checking VFIO's IOMMU models" (fun () -> supports container m))
  then
    failwith
      (match m with
      | No_iommu ->
          "VFIO is not in its no-IOMMU mode; turn it on: echo 1 | sudo tee \
           /sys/module/vfio/parameters/enable_unsafe_noiommu_mode"
      | Type1v2 ->
          "VFIO has no type 1 IOMMU; load it: sudo modprobe vfio_iommu_type1");
  let group = opened (open_file bus file) in
  if not (step ("reading the status of " ^ file) (fun () -> viable group)) then
    failwith (not_viable bus g);
  step
    ("attaching " ^ file ^ " to a VFIO container")
    (fun () -> set_container group container);
  (* Linux attaches a group to an IOMMU only where the IOMMU also remaps the
     function's interrupts, so that it cannot raise another's. *)
  (match set_iommu container m with
  | () -> ()
  | exception Unix.Unix_error (EPERM, _, _) when m = Type1v2 ->
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
      (step ("opening " ^ bus ^ " through VFIO") (fun () -> device group bus))
  in
  let efd = opened (step "creating an eventfd" eventfd) in
  step ("routing the interrupt of " ^ bus) (fun () -> msi device efd);
  (container, device, efd)

(* The offset of BAR [i] in [device]'s file, if VFIO maps its [n] bytes from
   [off]. The parts of a BAR it maps are whole pages, so the pages that hold
   those bytes map too. *)
let bar_offset bus device i off n =
  let size, offset, mappable, areas =
    step (Printf.sprintf "reading region %d of %s" i bus) (fun () ->
        region device i)
  in
  let inside (o, k) = off >= o && off + n <= o + k in
  if (not mappable) || off + n > size then
    failwith (Printf.sprintf "VFIO does not map BAR %d of %s" i bus);
  (match areas with
  | Some areas when not (List.exists inside areas) ->
      failwith
        (Printf.sprintf
           "VFIO maps parts of BAR %d of %s, and none holds [0x%x, 0x%x)" i bus
           off (off + n))
  | _ -> ());
  offset

let config_offset bus device =
  let _, offset, _, _ =
    step ("reading the configuration space of " ^ bus) (fun () ->
        region device config_region)
  in
  offset

(* Containers *)

(* Device addresses behind an IOMMU are taken from 4 GiB up to 1 TiB: every GPU
   reaches 40 bits, and an address a device truncates to 32 bits falls below 4
   GiB, where nothing is mapped, and faults. The window is the largest part of
   that range the IOMMU maps. *)
let iova_low = 1 lsl 32
let iova_high = 1 lsl 40

(* The window of an IOMMU that maps pages of [page_sizes], a bitmap, at the
   [(first, last)] device address ranges, all of them if [[]]: [(base, n)]. *)
let window bus page_sizes ranges =
  let smallest = page_sizes land -page_sizes in
  if smallest > page then
    failwith
      (Printf.sprintf
         "the IOMMU of %s maps no page smaller than %d bytes, and system pages \
          are %d bytes"
         bus smallest page);
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
      failwith
        (Printf.sprintf
           "the IOMMU of %s maps no device addresses between 4 GiB and 1 TiB"
           bus)
  | Some w -> w

let iova bus fd =
  let page_sizes, ranges =
    step ("reading the IOMMU of " ^ bus) (fun () -> iommu fd)
  in
  let base, n = window bus page_sizes ranges in
  Space.create ~base n

(* The container of a function behind an IOMMU, which maps system memory for it
   at device addresses the process allocates. Mappings are counted by their
   (address, bytes), as Function counts pins, because DMA memory and a pin may
   be the same pair. Once the function is released, closing the container has
   removed them all. *)
type t = {
  fd : fd;
  device : fd; (* the function's *)
  iova : Space.t;
  maps : (int * int, int * int) Hashtbl.t; (* to (device address, count) *)
  mutex : Mutex.t;
  mutable closed : bool;
}

(* Opens [bus] in a container, its descriptors on [files]: the container and the
   eventfd its interrupts signal. *)
let open_ files bus =
  let fd, device, efd = open_function files bus Type1v2 in
  let iova = iova bus fd in
  let maps = Hashtbl.create 64 in
  ({ fd; device; iova; maps; mutex = Mutex.create (); closed = false }, efd)

(* The device address at which [c] maps the [n] bytes at [a], mapping them for
   the first count. *)
let map_dma fn bus c a n =
  Mutex.protect c.mutex @@ fun () ->
  if c.closed then
    invalid_arg (Printf.sprintf "Function.%s: %s was released" fn bus);
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
              (Printf.sprintf "%s has no IOMMU addresses left for %d bytes" bus
                 n)
      in
      (match map c.fd a iova n with
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
  match Hashtbl.find c.maps (a, n) with
  | iova, k when k > 1 -> Hashtbl.replace c.maps (a, n) (iova, k - 1)
  | iova, _ ->
      if not c.closed then
        step (Printf.sprintf "unmapping %d bytes for %s" n bus) (fun () ->
            unmap c.fd iova n);
      Hashtbl.remove c.maps (a, n);
      Space.free c.iova iova

let close c = Mutex.protect c.mutex (fun () -> c.closed <- true)
