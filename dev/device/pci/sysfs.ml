(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* A host's files, under its root directory: "/" for this machine. *)
type t = {
  devices : string;
  groups : string;
  lockdown : string;
  bus_files : string; (* rescan and drivers_probe *)
  vfio : string;
}

let v root =
  let ( / ) = Filename.concat in
  {
    devices = root / "sys/bus/pci/devices";
    groups = root / "sys/kernel/iommu_groups";
    lockdown = root / "sys/kernel/security/lockdown";
    bus_files = root / "sys/bus/pci";
    vfio = root / "dev/vfio";
  }

let vfio_pci = "vfio-pci"
let path h bus file = strf "%s/%s/%s" h.devices bus file
let exists h bus = Sys.file_exists (Filename.concat h.devices bus)
let vfio_file h name = Filename.concat h.vfio name

(* A channel's failure names its file: "FILE: cause". *)
let read file =
  match In_channel.with_open_text file In_channel.input_all with
  | s -> String.trim s
  | exception Sys_error why -> Fail.fail "reading %s" why

(* Writes [s] to [file] in one write, which sysfs takes whole or refuses. A
   buffered channel would see the refusal only at its close, which drops it. *)
let put file s =
  let fd = Unix.openfile file [ O_WRONLY; O_CLOEXEC ] 0 in
  Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
  ignore (Unix.single_write_substring fd s 0 (String.length s))

let refused file (e : Unix.error) =
  let why = strf "writing %s: %s" file (Unix.error_message e) in
  match e with
  | EACCES | EPERM ->
      Fail.fail
        "%s; it needs CAP_SYS_ADMIN and write access to the file: run as root"
        why
  | _ -> Fail.fail "%s" why

let write file s =
  try put file s with Unix.Unix_error (e, _, _) -> refused file e

let readlink link =
  try Unix.readlink link
  with Unix.Unix_error (e, _, _) ->
    Fail.fail "reading the link %s: %s" link (Unix.error_message e)

(* [hex file s] is the number that [s], read from [file], spells in
   hexadecimal. *)
let hex file s =
  let digits = if String.starts_with ~prefix:"0x" s then s else "0x" ^ s in
  match int_of_string_opt digits with
  | Some n -> n
  | None -> Fail.fail "reading %s: %S is no hexadecimal number" file s

let read_hex file = hex file (read file)

(* Functions *)

let link h bus file =
  let l = path h bus file in
  if Sys.file_exists l then Some (Filename.basename (readlink l)) else None

let driver h bus = link h bus "driver"
let group h bus = link h bus "iommu_group"

(* A function the kernel lists but whose files cannot be read, as while it is
   being removed, is left out. *)
let functions h =
  if not (Sys.file_exists h.devices) then []
  else
    Sys.readdir h.devices |> Array.to_list
    |> List.filter_map (fun bus ->
        match
          {
            Ops.bus;
            vendor = read_hex (path h bus "vendor");
            device = read_hex (path h bus "device");
            class_ = read_hex (path h bus "class") lsr 16;
          }
        with
        | id -> Some id
        | exception Fail.Failed _ -> None)

(* The other functions of [bus]'s device, such as its audio function. *)
let siblings h bus =
  let prefix = String.sub bus 0 (String.length bus - 1) in
  List.filter
    (fun s -> s <> bus && exists h s)
    (List.init 8 (fun fn -> prefix ^ string_of_int fn))

let enabled h bus = read (path h bus "enable") <> "0"

(* BARs *)

(* Configuration space shows every reader its first [header] bytes; past them
   Linux shows only a reader with CAP_SYS_ADMIN. BAR registers start at
   [bar_base], and their low bits say what the BAR is: I/O or memory, and for
   memory whether it is 64 bits wide. *)
let header = 64
let bar_base = 0x10
let bars = 6
let io_bar = 1
let io_flags = 0b11
let memory_flags = 0xf
let wide_bar = 0b100
let type_mask = 0b110

let registers h bus =
  let file = path h bus "config" in
  let s =
    match
      In_channel.with_open_bin file (fun ic -> really_input_string ic header)
    with
    | s -> s
    | exception Sys_error why -> Fail.fail "reading %s" why
    | exception End_of_file ->
        Fail.fail "reading %s: shorter than %d bytes" file header
  in
  Array.init bars (fun i ->
      Int32.to_int (String.get_int32_le s (bar_base + (4 * i))) land 0xffff_ffff)

let resource h bus i =
  let file = path h bus "resource" in
  match List.nth_opt (String.split_on_char '\n' (read file)) i with
  | None -> None
  | Some line -> (
      match String.split_on_char ' ' line with
      | start :: stop :: _ ->
          let start = hex file start and stop = hex file stop in
          if stop <= start then None else Some (stop - start + 1)
      | _ -> None)

(* The bus address in the registers of BAR [i]; [None] for the upper half of a
   64-bit BAR. *)
let address regs i =
  let wide j = regs.(j) land io_bar = 0 && regs.(j) land type_mask = wide_bar in
  let rec upper j =
    j < i && if wide j then j + 1 = i || upper (j + 2) else upper (j + 1)
  in
  if upper 0 then None
  else if regs.(i) land io_bar <> 0 then Some (regs.(i) land lnot io_flags)
  else
    let low = regs.(i) land lnot memory_flags in
    if wide i && i + 1 < bars then Some (low lor (regs.(i + 1) lsl 32))
    else Some low

let bar h bus i =
  if i < 0 || i >= bars then None
  else
    match (address (registers h bus) i, resource h bus i) with
    | Some a, Some size -> Some (a, size)
    | _ -> None

(* How a function is taken *)

type iommu = No_iommu | Identity | Translating

type state = {
  driver : string option;
  iommu : iommu;
  siblings : string list;
  enabled : bool;
  locked_down : bool;
}

(* VFIO's no-IOMMU mode gives a function a group of its own, whose file is
   [noiommu-N]. A group's [type] is the domain its functions' DMA goes through
   while no VFIO container holds them. *)
let noiommu_file h g = vfio_file h ("noiommu-" ^ g)

let iommu_of h bus =
  match group h bus with
  | None -> No_iommu
  | Some g when Sys.file_exists (noiommu_file h g) -> No_iommu
  | Some g -> (
      match read (strf "%s/%s/type" h.groups g) with
      | "identity" -> Identity
      | _ -> Translating
      | exception Fail.Failed _ -> Translating)

(* The kernel's lockdown file lists the modes with the current one in
   brackets. *)
let locked_down h =
  match read h.lockdown with
  | s -> not (String.starts_with ~prefix:"[none]" s)
  | exception Fail.Failed _ -> false

let state h bus =
  {
    driver = driver h bus;
    iommu = iommu_of h bus;
    siblings = siblings h bus;
    enabled = enabled h bus;
    locked_down = locked_down h;
  }

let bind_vfio bus = strf "sudo driverctl set-override %s vfio-pci" bus

(* Bridges are held by pcieport, which VFIO accepts. *)
let group_holders h g =
  match Sys.readdir (strf "%s/%s/devices" h.groups g) with
  | exception Sys_error _ -> []
  | fns ->
      Array.to_list fns |> List.sort String.compare
      |> List.filter_map (fun f ->
          match driver h f with
          | Some d when not (List.mem d [ vfio_pci; "pci-stub"; "pcieport" ]) ->
              Some (f, d)
          | _ -> None)

let access h bus s =
  match s with
  | { driver = Some d; iommu = No_iommu; _ } when d <> vfio_pci ->
      Error (strf "%s is bound to the driver %s; detach the GPU first" bus d)
  | { driver = Some d; _ } when d <> vfio_pci ->
      Error
        (strf
           "%s is bound to the driver %s (to take it through the IOMMU, \
            without root, bind it to vfio-pci: %s)"
           bus d (bind_vfio bus))
  | { driver = Some _; iommu = Identity | Translating; _ } -> Ok Ops.Iommu
  | { driver = None; iommu = Translating; _ } ->
      Error
        (strf
           "the IOMMU translates the addresses %s reaches, so it cannot reach \
            physical ones; bind it to vfio-pci to take it through the IOMMU \
            (%s), or boot Linux with iommu=pt"
           bus (bind_vfio bus))
  | { siblings = s :: _; _ } ->
      Error (strf "%s shares its device with %s; detach the GPU first" bus s)
  | { driver = None; enabled = false; _ } ->
      Error (strf "%s is disabled; detach the GPU first" bus)
  | { locked_down = true; _ } ->
      Error
        (strf
           "the kernel is locked down (%s), which refuses mapping a BAR \
            outside VFIO; take %s behind an IOMMU: turn the IOMMU on and bind \
            it to vfio-pci (%s)"
           h.lockdown bus (bind_vfio bus))
  | _ -> Ok Ops.Physical

(* Changes *)

let detach h bus =
  match access h bus (state h bus) with
  | Ok _ -> ()
  | Error _ -> (
      (match driver h bus with
      | Some d when d <> vfio_pci ->
          write (path h bus "driver/unbind") bus;
          if driver h bus <> None then
            Fail.fail "the driver %s stays bound to %s" d bus
      | _ -> ());
      List.iter (fun s -> write (path h s "remove") "1") (siblings h bus);
      if driver h bus = None && not (enabled h bus) then
        write (path h bus "enable") "1";
      let s = state h bus in
      match (access h bus s, s) with
      | Ok _, _ -> ()
      | Error _, { siblings = sibling :: _; _ } ->
          Fail.fail "%s still shares its device with %s after removing it" bus
            sibling
      | Error _, { driver = None; enabled = false; _ } ->
          Fail.fail "%s is still disabled after enabling it" bus
      | Error why, _ -> Fail.fail "%s" why)

let reset h bus = write (path h bus "reset") "1"

(* A rescan brings back the functions [detach] removed. The function itself is
   on the bus already, so its driver is probed for it. *)
let attach h bus =
  match driver h bus with
  | Some d when d = vfio_pci ->
      Fail.fail
        "%s is bound to vfio-pci; unbind it and clear its driver_override \
         first: sudo driverctl unset-override %s"
        bus bus
  | Some _ -> ()
  | None ->
      if enabled h bus then write (path h bus "enable") "0";
      write (Filename.concat h.bus_files "rescan") "1";
      write (Filename.concat h.bus_files "drivers_probe") bus;
      if driver h bus = None then
        Fail.fail "no kernel driver took %s; load its module first" bus

(* [resourceN_resize] holds a bitmap of the sizes BAR [N] supports, bit [k] for
   [2^k] MiB, and takes the [k] to set. A bridge whose window cannot hold a size
   refuses it with ENOSPC, and a smaller one may fit. Any other refusal holds
   for every size, so the BAR keeps its own. The bitmap is an [int], whose
   highest bit is [largest]. *)
let largest = Sys.int_size - 2

let resize h bus i =
  let file = path h bus (strf "resource%d_resize" i) in
  if driver h bus = None && Sys.file_exists file then
    let sizes = read_hex file in
    let rec try_from k =
      if k >= 0 then
        if sizes land (1 lsl k) = 0 then try_from (k - 1)
        else
          match put file (string_of_int k) with
          | () -> ()
          | exception Unix.Unix_error (ENOSPC, _, _) -> try_from (k - 1)
          | exception Unix.Unix_error _ -> ()
    in
    try_from largest
