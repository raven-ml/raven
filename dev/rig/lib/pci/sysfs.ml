(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

external makedev : (int[@untagged]) -> (int[@untagged]) -> (int[@untagged])
  = "caml_rig_pci_makedev_byte" "caml_rig_pci_makedev"
[@@noalloc]

(* A machine's files, under its root directory: "/" for this machine. *)
type t = {
  root : string;
  devices : string;
  groups : string;
  lockdown : string;
  bus_files : string; (* rescan and drivers_probe *)
  vfio : string;
}

let v root =
  let ( / ) = Filename.concat in
  {
    root;
    devices = root / "sys/bus/pci/devices";
    groups = root / "sys/kernel/iommu_groups";
    lockdown = root / "sys/kernel/security/lockdown";
    bus_files = root / "sys/bus/pci";
    vfio = root / "dev/vfio";
  }

let vfio_pci = "vfio-pci"
let root m = m.root
let path m bus file = strf "%s/%s/%s" m.devices bus file
let exists m bus = Sys.file_exists (Filename.concat m.devices bus)
let vfio_file m name = Filename.concat m.vfio name

(* A channel's failure names its file: "FILE: cause". *)
let read file =
  match In_channel.with_open_text file In_channel.input_all with
  | s -> String.trim s
  | exception Sys_error why -> Fail.fail "reading %s" why

(* Writes [s] to [file] in one write, which sysfs takes whole or refuses. A
   buffered channel would see the refusal only at its close, which drops it.
   Sysfs ignores the truncation, as a shell's redirection relies on; a plain
   file holds [s] alone after it. *)
let put file s =
  let fd = Unix.openfile file [ O_WRONLY; O_TRUNC; O_CLOEXEC ] 0 in
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

let link m bus file =
  let l = path m bus file in
  if Sys.file_exists l then Some (Filename.basename (readlink l)) else None

let driver m bus = link m bus "driver"
let group m bus = link m bus "iommu_group"

(* A function the kernel lists but whose files cannot be read, as while it is
   being removed, is left out. *)
let functions m =
  if not (Sys.file_exists m.devices) then []
  else
    Sys.readdir m.devices |> Array.to_list
    |> List.filter_map (fun bus ->
        match
          {
            Ops.bus;
            vendor = read_hex (path m bus "vendor");
            device = read_hex (path m bus "device");
            class_ = read_hex (path m bus "class");
          }
        with
        | id -> Some id
        | exception Fail.Failed _ -> None)

(* The other functions of [bus]'s device, such as its audio function. *)
let siblings m bus =
  let prefix = String.sub bus 0 (String.length bus - 1) in
  List.filter
    (fun s -> s <> bus && exists m s)
    (List.init 8 (fun fn -> prefix ^ string_of_int fn))

let enabled m bus = read (path m bus "enable") <> "0"

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

let registers m bus =
  let file = path m bus "config" in
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

(* The size of the BAR on [line] of the [resource] file [file]: its start, end
   and flags, the end 0 for no BAR. *)
let size file line =
  match String.split_on_char ' ' line with
  | start :: stop :: _ ->
      let start = hex file start and stop = hex file stop in
      if stop <= start then None else Some (stop - start + 1)
  | _ -> None

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

let bars m bus =
  let regs = registers m bus and file = path m bus "resource" in
  let lines = Array.of_list (String.split_on_char '\n' (read file)) in
  Array.init bars (fun i ->
      match address regs i with
      | Some a when i < Array.length lines ->
          Option.map (fun n -> (a, n)) (size file lines.(i))
      | _ -> None)

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
let noiommu_file m g = vfio_file m ("noiommu-" ^ g)

let iommu_of m bus =
  match group m bus with
  | None -> No_iommu
  | Some g when Sys.file_exists (noiommu_file m g) -> No_iommu
  | Some g ->
      (* A group without a [type] file counts as translated. *)
      let file = strf "%s/%s/type" m.groups g in
      if Sys.file_exists file && read file = "identity" then Identity
      else Translating

(* The kernel's lockdown file lists the modes with the current one in
   brackets. *)
let locked_down m =
  match read m.lockdown with
  | s -> not (String.starts_with ~prefix:"[none]" s)
  | exception Fail.Failed _ -> false

let state m bus =
  {
    driver = driver m bus;
    iommu = iommu_of m bus;
    siblings = siblings m bus;
    enabled = enabled m bus;
    locked_down = locked_down m;
  }

(* Bridges are held by pcieport, which VFIO accepts. *)
let group_holders m g =
  match Sys.readdir (strf "%s/%s/devices" m.groups g) with
  | exception Sys_error _ -> []
  | fns ->
      Array.to_list fns |> List.sort String.compare
      |> List.filter_map (fun f ->
          match driver m f with
          | Some d when not (List.mem d [ vfio_pci; "pci-stub"; "pcieport" ]) ->
              Some (f, d)
          | _ -> None)

(* How a function in state [s] is taken, or why it cannot be. *)
let addressing m bus s =
  match s with
  | { driver = Some d; iommu = No_iommu; _ } when d <> vfio_pci ->
      Error (strf "%s is bound to the driver %s; detach the GPU first" bus d)
  | { driver = Some d; _ } when d <> vfio_pci ->
      Error
        (strf
           "%s is bound to the driver %s; to take it through the IOMMU without \
            root, bind it to vfio-pci"
           bus d)
  | { driver = Some _; iommu = Identity | Translating; _ } -> Ok Ops.Iommu
  | { driver = None; iommu = Translating; _ } ->
      Error
        (strf
           "the IOMMU translates the addresses %s reaches, so it cannot reach \
            physical memory; bind it to vfio-pci, or boot Linux with iommu=pt"
           bus)
  | { siblings = s :: _; _ } ->
      Error (strf "%s shares its device with %s; detach the GPU first" bus s)
  | { driver = None; enabled = false; _ } ->
      Error (strf "%s is disabled; detach the GPU first" bus)
  | { locked_down = true; _ } ->
      Error
        (strf
           "the kernel is locked down (%s) and refuses mapping a BAR outside \
            VFIO; turn the IOMMU on and bind %s to vfio-pci"
           m.lockdown bus)
  | _ -> Ok Ops.Physical

let access m bus = addressing m bus (state m bus)

(* Changes *)

(* [driver_override] names the one driver that may bind a function; none is
   named [none]. A function the process detached keeps every kernel driver off
   it: a probe, a rescan or a module load would otherwise bind its driver, which
   resets a GPU under whatever runs on it. *)
let no_driver = "none"
let override m bus = path m bus "driver_override"

(* Detach decides before it writes anything whether the function will be
   takeable once unbound, alone and enabled: one it could not take then could
   not be reset to go back to its driver. *)
let detach m bus =
  let s = state m bus in
  if s.driver <> Some vfio_pci then begin
    let detached = { s with driver = None; siblings = []; enabled = true } in
    Result.iter_error (Fail.fail "%s") (addressing m bus detached);
    write (override m bus) no_driver
  end;
  match addressing m bus s with
  | Ok _ -> ()
  | Error _ -> (
      (match driver m bus with
      | Some d when d <> vfio_pci ->
          write (path m bus "driver/unbind") bus;
          if driver m bus <> None then
            Fail.fail "the driver %s stays bound to %s" d bus
      | _ -> ());
      List.iter (fun s -> write (path m s "remove") "1") (siblings m bus);
      if driver m bus = None && not (enabled m bus) then
        write (path m bus "enable") "1";
      let s = state m bus in
      match (addressing m bus s, s) with
      | Ok _, _ -> ()
      | Error _, { siblings = sibling :: _; _ } ->
          Fail.fail "%s still shares its device with %s after removing it" bus
            sibling
      | Error _, { driver = None; enabled = false; _ } ->
          Fail.fail "%s is still disabled after enabling it" bus
      | Error why, _ -> Fail.fail "%s" why)

let reset m bus = write (path m bus "reset") "1"

(* A rescan brings back the functions [detach] removed. The function itself is
   on the bus already, so its driver is probed for it. *)
let attach m bus =
  match driver m bus with
  | Some d when d <> vfio_pci -> ()
  | bound ->
      if bound = Some vfio_pci then write (path m bus "driver/unbind") bus;
      if enabled m bus then write (path m bus "enable") "0";
      write (Filename.concat m.bus_files "rescan") "1";
      write (override m bus) "\n";
      write (Filename.concat m.bus_files "drivers_probe") bus;
      if driver m bus = None then
        Fail.fail "no kernel driver took %s; load its module first" bus

(* [resourceN_resize] holds a bitmap of the sizes BAR [N] supports, bit [k] for
   [2^k] MiB, and takes the [k] to set. A bridge whose window cannot hold a size
   refuses it with ENOSPC, and a smaller one may fit. Any other refusal holds
   for every size, so the BAR keeps its own. The bitmap is an [int], whose
   highest bit is [largest]. *)
let largest = Sys.int_size - 2

let resize m bus i =
  let file = path m bus (strf "resource%d_resize" i) in
  if driver m bus = None && Sys.file_exists file then
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

(* Open devices *)

let contents m file =
  match
    In_channel.with_open_bin (Filename.concat m.root file) In_channel.input_all
  with
  | s -> Some s
  | exception Sys_error _ -> None

(* A [dev] file holds a device's number as "MAJOR:MINOR". *)
let dev_number file =
  let s = read file in
  match List.map int_of_string_opt (String.split_on_char ':' s) with
  | [ Some major; Some minor ] -> makedev major minor
  | _ -> Fail.fail "reading %s: %S is no device number" file s

let entries dir =
  match Sys.readdir dir with
  | names -> Array.to_list names
  | exception Sys_error why -> Fail.fail "reading %s" why

(* The numbers of the devices under the directory [dir], from their [dev] files.
   Links are not followed: sysfs links reach the whole tree. *)
let rec numbers dir =
  List.concat_map
    (fun name ->
      let file = Filename.concat dir name in
      match Unix.lstat file with
      | { st_kind = S_DIR; _ } -> numbers file
      | { st_kind = S_REG; _ } when name = "dev" -> [ dev_number file ]
      | _ -> []
      | exception Unix.Unix_error (e, _, _) ->
          Fail.fail "reading %s: %s" file (Unix.error_message e))
    (entries dir)

(* The number of the character device at [file], if there is one. *)
let device file =
  match Unix.stat file with
  | { st_kind = S_CHR; st_rdev; _ } -> Some st_rdev
  | _ -> None
  | exception Unix.Unix_error (ENOENT, _, _) -> None
  | exception Unix.Unix_error (e, _, _) ->
      Fail.fail "reading %s: %s" file (Unix.error_message e)

(* The character devices the process holds open, by number, each with the file
   its descriptor names. A descriptor closed since the listing is left out. *)
let opened m =
  let fds = Filename.concat m.root "proc/self/fd" in
  List.filter_map
    (fun fd ->
      let link = Filename.concat fds fd in
      Option.map
        (fun n ->
          ( n,
            match Unix.readlink link with
            | f -> f
            | exception Unix.Unix_error _ -> link ))
        (device link))
    (entries fds)

let held m bus nodes =
  let gpu =
    numbers (Filename.concat m.devices bus)
    @ List.filter_map (fun f -> device (Filename.concat m.root f)) nodes
  in
  List.find_map
    (fun (n, file) -> if List.mem n gpu then Some file else None)
    (opened m)
