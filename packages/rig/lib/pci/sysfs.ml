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
  kernel : string -> string -> unit; (* a fixture's, acting on a write *)
}

let v ?(kernel = fun _ _ -> ()) root =
  let ( / ) = Filename.concat in
  {
    root;
    kernel;
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

(* Locks *)

external flock : Unix.file_descr -> unit = "caml_rig_pci_flock"

(* A function's configuration file is locked by flock: the lock is the open
   file's, which a duplicate descriptor shares and every other open of the file,
   in this process or another, is refused. *)
type 's lock = { files : t; bus : string; fd : Unix.file_descr }

let locked m bus (f : 's. 's lock -> ('a, string) result) =
  let file = path m bus "config" in
  match Unix.openfile file [ O_RDONLY; O_CLOEXEC ] 0 with
  | exception Unix.Unix_error (e, _, _) ->
      Error (strf "opening %s: %s" file (Unix.error_message e))
  | fd -> (
      Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
      match flock fd with
      | exception Unix.Unix_error ((EAGAIN | EWOULDBLOCK), _, _) ->
          Error (strf "%s is taken already; find who holds it: lsof %s" bus file)
      | exception Unix.Unix_error (e, _, _) ->
          Error
            (strf "locking %s with %s: %s" bus file (Unix.error_message e))
      | () -> f { files = m; bus; fd })

let files l = l.files
let bus l = l.bus
let keep l = Unix.dup ~cloexec:true l.fd

(* A channel's failure names its file: "FILE: cause". *)
let read file =
  match In_channel.with_open_text file In_channel.input_all with
  | s -> String.trim s
  | exception Sys_error why -> Fail.fail "reading %s" why

(* Writes [s] to [file] in one write, which sysfs takes whole or refuses. A
   buffered channel would see the refusal only at its close, which drops it.
   Sysfs ignores the truncation, as a shell's redirection relies on; a plain
   file holds [s] alone after it. [m]'s kernel acts on [s] before the write,
   as Linux does inside it. *)
let put m file s =
  let fd = Unix.openfile file [ O_WRONLY; O_TRUNC; O_CLOEXEC ] 0 in
  Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
  m.kernel file s;
  ignore (Unix.single_write_substring fd s 0 (String.length s))

let refused file (e : Unix.error) =
  let why = strf "writing %s: %s" file (Unix.error_message e) in
  match e with
  | EACCES | EPERM ->
      Fail.fail
        "%s; it needs CAP_SYS_ADMIN and write access to the file: run as root"
        why
  | _ -> Fail.fail "%s" why

let write m file s =
  try put m file s with Unix.Unix_error (e, _, _) -> refused file e

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

let driver_of m bus = link m bus "driver"
let driver l = driver_of l.files l.bus
let group m bus = link m bus "iommu_group"

(* A function the kernel lists but whose files cannot be read, as while it is
   being removed, is left out. A system without the directory has none. *)
let functions m =
  if not (Sys.file_exists m.devices) then Ok []
  else
    match Sys.readdir m.devices with
    | exception Sys_error why -> Error ("listing the PCI functions: " ^ why)
    | buses ->
        let id bus =
          match
            {
              Ops.bus;
              vendor = read_hex (path m bus "vendor");
              device = read_hex (path m bus "device");
              class_ = read_hex (path m bus "class");
            }
          with
          | id -> Some id
          | exception Fail.Failed _ -> None
        in
        Ok (List.filter_map id (Array.to_list buses))

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

(* The memory BARs whose register marks them prefetchable, by index. *)
let prefetchable_bar = 0b1000

let prefetchable { files = m; bus; _ } =
  let regs = registers m bus in
  Array.init bars (fun i ->
      address regs i <> None
      && regs.(i) land io_bar = 0
      && regs.(i) land prefetchable_bar <> 0)

let bars { files = m; bus; _ } =
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
  bus_reset : bool;
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

let translates m bus = iommu_of m bus = Translating

(* The kernel's lockdown file lists the modes with the current one in
   brackets. *)
let locked_down m =
  match read m.lockdown with
  | s -> not (String.starts_with ~prefix:"[none]" s)
  | exception Fail.Failed _ -> false

(* Linux resets a function with the first method of its [reset_method] that
   applies. Only [bus] and [cxl_bus] reset more than the function, and Linux
   refuses them while another function shares the bus, so a function whose
   reset is one of them must be alone on its device for {!reset} to work. A
   kernel older than the file (5.15) lists no methods: a function with a
   [reset] file then counts as reset through its bus. *)
let bus_methods = [ "bus"; "cxl_bus" ]

let bus_reset m bus =
  let methods = path m bus "reset_method" in
  if Sys.file_exists methods then
    List.mem (List.hd (String.split_on_char ' ' (read methods))) bus_methods
  else Sys.file_exists (path m bus "reset")

let state m bus =
  {
    driver = driver_of m bus;
    iommu = iommu_of m bus;
    siblings = siblings m bus;
    bus_reset = bus_reset m bus;
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
          match driver_of m f with
          | Some d when not (List.mem d [ vfio_pci; "pci-stub"; "pcieport" ]) ->
              Some (f, d)
          | _ -> None)

(* How a function in state [s] is taken, or why it cannot be. *)
let addressing m bus s =
  match s with
  | { driver = Some d; iommu = No_iommu; _ } when d <> vfio_pci ->
      Error (strf "%s is bound to the driver %s; detach it first" bus d)
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
  | { siblings = s :: _; bus_reset = true; _ } ->
      Error
        (strf
           "%s shares its device with %s, and Linux resets it through its bus, \
            which needs it alone; detach it first"
           bus s)
  | { driver = None; enabled = false; _ } ->
      Error (strf "%s is disabled; detach it first" bus)
  | { locked_down = true; _ } ->
      Error
        (strf
           "the kernel is locked down (%s) and refuses mapping a BAR outside \
            VFIO; turn the IOMMU on and bind %s to vfio-pci"
           m.lockdown bus)
  | _ -> Ok Ops.Physical

let access { files = m; bus; _ } = addressing m bus (state m bus)

(* Changes *)

(* [driver_override] names the one driver that may bind a function; none is
   named [none]. A function the process detached keeps every kernel driver off
   it: a probe, a rescan or a module load would otherwise bind its driver, which
   resets a GPU under whatever runs on it. *)
let no_driver = "none"
let override m bus = path m bus "driver_override"

(* The command register and its bit that lets the function master the bus (PCI
   Express Base Specification, 7.5.1.1.3). *)
let command = 0x04
let bus_master = 0x4

(* Turns [bus]'s bus mastering off. A kernel driver's unbind may leave it on:
   Linux clears it only once the function's enable count reaches zero, and an
   enable through sysfs counts too. *)
let stop_mastering m bus =
  let file = path m bus "config" in
  Fail.step ("writing " ^ file) @@ fun () ->
  let fd = Unix.openfile file [ O_RDWR; O_CLOEXEC ] 0 in
  Fun.protect ~finally:(fun () -> Unix.close fd) @@ fun () ->
  let b = Bytes.create 2 in
  ignore (Unix.lseek fd command SEEK_SET);
  if Unix.read fd b 0 2 = 2 then begin
    let v = Bytes.get_uint16_le b 0 in
    if v land bus_master <> 0 then begin
      Bytes.set_uint16_le b 0 (v land lnot bus_master);
      ignore (Unix.lseek fd command SEEK_SET);
      ignore (Unix.single_write fd b 0 2)
    end
  end

(* Detach decides before it writes anything whether the function will be
   takeable once unbound and enabled, alone if it removes the device's other
   functions: one it could not take then could not be reset to go back to its
   driver. A detach that keeps them refuses a function that has any and that
   Linux resets through its bus, naming each with its driver. *)
(* Behind an IOMMU that translates, a process takes a function through VFIO,
   which takes its IOMMU group whole. Detach binds the function to vfio-pci
   itself, and with [remove] the other functions of its device, which a GPU's
   group holds with it: the functions it moves are those not on vfio-pci. *)
let to_move m bus ~remove =
  List.filter
    (fun f -> driver_of m f <> Some vfio_pci)
    (bus :: (if remove then siblings m bus else []))

(* The functions of [bus]'s group that VFIO refuses and that detach leaves, with
   their drivers. *)
let left_in_group m bus ~remove =
  let moved = bus :: (if remove then siblings m bus else []) in
  match group m bus with
  | None -> []
  | Some g ->
      List.filter (fun (f, _) -> not (List.mem f moved)) (group_holders m g)

let vfio_refusal m bus ~remove =
  if not (Sys.file_exists (Filename.concat m.bus_files "drivers/vfio-pci")) then
    Some
      (strf
         "the IOMMU translates the addresses %s reaches, so it is taken \
          through vfio-pci, which is not loaded; load it: sudo modprobe \
          vfio-pci"
         bus)
  else
    match left_in_group m bus ~remove with
    | [] -> None
    | held ->
        Some
          (strf
             "IOMMU group %s of %s also holds %s, bound to other drivers; VFIO \
              takes a group whole: bind each to vfio-pci"
             (Option.get (group m bus))
             bus
             (String.concat ", "
                (List.map (fun (f, d) -> strf "%s (%s)" f d) held)))

let keep_off { files = m; bus; _ } ~remove =
  let s = state m bus in
  if s.iommu = Translating then begin
    Option.iter (Fail.fail "%s") (vfio_refusal m bus ~remove);
    if s.driver <> Some vfio_pci then write m (override m bus) vfio_pci
  end
  else if s.driver <> Some vfio_pci then begin
    if (not remove) && s.siblings <> [] && s.bus_reset then begin
      let named f =
        match driver_of m f with
        | Some d -> strf "%s (bound to %s)" f d
        | None -> strf "%s (unbound)" f
      in
      Fail.fail
        "%s shares its device with %s, which detach leaves, and Linux resets \
         it through its bus, which needs it alone; remove or detach them first"
        bus
        (String.concat ", " (List.map named s.siblings))
    end;
    let siblings = if remove then [] else s.siblings in
    let detached = { s with driver = None; siblings; enabled = true } in
    Result.iter_error (Fail.fail "%s") (addressing m bus detached);
    write m (override m bus) no_driver
  end

let unbind m f =
  match driver_of m f with
  | Some d when d <> vfio_pci ->
      write m (path m f "driver/unbind") f;
      if driver_of m f <> None then
        Fail.fail "the driver %s stays bound to %s" d f
  | _ -> ()

(* A GPU's kernel driver may read the driver data of its device's other
   functions as its own driver's: NVIDIA's nv_audio_dynamic_power
   (kernel-open/nvidia/nv.c) reads an audio function's as an ALSA card, and
   walks vfio-pci's as one. An unbound function's is NULL, which it skips. So
   a function's other functions leave vfio-pci before its kernel driver
   probes or unbinds. Each keeps an override that names vfio-pci, so that a
   change that fails after this leaves it for {!attach} to find. *)
let release m f =
  write m (override m f) vfio_pci;
  if driver_of m f = Some vfio_pci then begin
    write m (path m f "driver/unbind") f;
    if driver_of m f <> None then
      Fail.fail "the driver %s stays bound to %s" vfio_pci f
  end

(* A function detach leaves on no driver masters the bus no more: without an
   IOMMU it could write any host memory. Behind one that translates, the
   functions it moves stay unbound until {!bind}. The other functions go
   first, so that one whose driver stays bound leaves the GPU bound too,
   which {!attach} gives back as it is. With [remove], one that vfio-pci
   holds, as an attach that stopped part-way or an administrator leaves it,
   is released before the function's kernel driver unbinds. *)
let detach { files = m; bus; _ } ~remove =
  let s = state m bus in
  (match s.driver with
  | Some d when remove && d <> vfio_pci ->
      List.iter
        (fun f -> if driver_of m f = Some vfio_pci then release m f)
        s.siblings
  | _ -> ());
  (if s.iommu = Translating then begin
     let others = List.filter (( <> ) bus) (to_move m bus ~remove) in
     List.iter (fun f -> write m (override m f) vfio_pci) others;
     List.iter (unbind m) others;
     unbind m bus
   end
   else
     match addressing m bus s with
  | Ok _ -> ()
  | Error _ -> (
      unbind m bus;
      if remove then
        List.iter (fun s -> write m (path m s "remove") "1") (siblings m bus);
      if driver_of m bus = None && not (enabled m bus) then
        write m (path m bus "enable") "1";
      let s = state m bus in
      match (addressing m bus s, s) with
      | Ok _, _ -> ()
      | Error _, { siblings = sibling :: _; bus_reset = true; _ } ->
          Fail.fail "%s still shares its device with %s after removing it" bus
            sibling
      | Error _, { driver = None; enabled = false; _ } ->
          Fail.fail "%s is still disabled after enabling it" bus
      | Error why, _ -> Fail.fail "%s" why));
  if driver_of m bus = None then stop_mastering m bus

(* Linux binds each function whose [driver_override] names vfio-pci to it at a
   probe. *)
let bind { files = m; bus; _ } ~remove =
  let probe f =
    write m (override m f) vfio_pci;
    write m (Filename.concat m.bus_files "drivers_probe") f;
    if driver_of m f <> Some vfio_pci then
      Fail.fail
        "vfio-pci did not take %s at its probe, for the reason the kernel's \
         log (dmesg) gives: once that is fixed, call again; until then, \
         reboot to give the GPU back to its driver"
        f
  in
  if iommu_of m bus = Translating then List.iter probe (to_move m bus ~remove)

let reset m bus = write m (path m bus "reset") "1"

(* A rescan brings back the functions [detach] removed. The function itself is
   on the bus already, so its driver is probed for it. A detach that stopped
   part-way may leave the function bound with its override set, which would keep
   its driver off it at the next probe. With [siblings], each other function of
   the device that vfio-pci holds or that its override keeps for vfio-pci, as a
   detach behind a translating IOMMU leaves it, goes back to its driver the
   same way. Each is released before the function's probe, and has its
   override cleared and its driver probed after it. An attach that fails at
   the probe, as when the GPU's driver is not loaded, leaves them for the
   next attach. *)
let give_back m f =
  write m (override m f) "\n";
  write m (Filename.concat m.bus_files "drivers_probe") f;
  if driver_of m f = None then
    Fail.fail "no kernel driver took %s; load its module first" f

let attach { files = m; bus; _ } ~siblings:others =
  let held =
    if not others then []
    else
      List.filter
        (fun f ->
          driver_of m f = Some vfio_pci || read (override m f) = vfio_pci)
        (siblings m bus)
  in
  List.iter (release m) held;
  (match driver_of m bus with
  | Some d when d <> vfio_pci ->
      if not (List.mem (read (override m bus)) [ ""; "(null)" ]) then begin
        write m (Filename.concat m.bus_files "rescan") "1";
        write m (override m bus) "\n"
      end
  | bound ->
      if bound = Some vfio_pci then write m (path m bus "driver/unbind") bus;
      if enabled m bus then write m (path m bus "enable") "0";
      write m (Filename.concat m.bus_files "rescan") "1";
      write m (override m bus) "\n";
      write m (Filename.concat m.bus_files "drivers_probe") bus;
      if driver_of m bus = None then
        Fail.fail "no kernel driver took %s; load its module first" bus);
  List.iter (give_back m) held

(* [resourceN_resize] holds a bitmap of the sizes BAR [N] supports, bit [k] for
   [2^k] MiB, and takes the [k] to set. A bridge whose window cannot hold a size
   refuses it with ENOSPC, and a smaller one may fit. A file the process may not
   write is refused, naming it. Linux 6.12's other refusals hold for every size,
   so the BAR keeps its own: EBUSY for a BAR in use, ENOTSUPP where the
   firmware's assignment must stay or the function has no resizable BAR, ENOENT
   where no bridge window can be released, ENOMEM, ENODEV for a function that
   left the bus, and EINVAL where the boot console cannot leave a VGA function.
   Any other errno comes from a mistake of this library's and raises. The
   bitmap is an [int], whose highest bit is [largest]. *)
let largest = Sys.int_size - 2

(* Linux's own errno for an operation a device does not support, which [Unix]
   does not name. *)
let enotsupp = Unix.EUNKNOWNERR 524

let keeps_size : Unix.error -> bool = function
  | EBUSY | ENOENT | ENOMEM | ENODEV | EINVAL -> true
  | e -> e = enotsupp

let resize { files = m; bus; _ } i =
  let file = path m bus (strf "resource%d_resize" i) in
  if driver_of m bus = None && Sys.file_exists file then
    let sizes = read_hex file in
    let rec try_from k =
      if k >= 0 then
        if sizes land (1 lsl k) = 0 then try_from (k - 1)
        else
          Fail.bug ("writing " ^ file) @@ fun () ->
          match put m file (string_of_int k) with
          | () -> ()
          | exception Unix.Unix_error (ENOSPC, _, _) -> try_from (k - 1)
          | exception Unix.Unix_error (((EACCES | EPERM) as e), _, _) ->
              refused file e
          | exception Unix.Unix_error (e, _, _) when keeps_size e -> ()
    in
    try_from largest

(* Peer requests

   Linux decides whether a PCI bus carries one device's requests to another's
   BAR in drivers/pci/p2pdma.c (calc_map_type_and_dist). Walking up from the
   device that owns the BAR, the first bridge above both carries them, unless
   a device on the way up to it, that bridge included, or on the way up from
   the other device to below it, redirects peer requests to the root complex
   by ACS. Requests that go through the root complex, with no common bridge
   or with a redirect, are carried only where the CPU is an AMD one from Zen
   on (cpu_supports_p2pdma), or the root complexes are on Linux's list
   (pci_p2pdma_whitelist). *)

(* A host bridge on Linux's list, some only between devices under that one
   bridge. *)
type listed = Unlisted | Same_bridge | Any_bridge

let intel = 0x8086

(* Linux's list, by the vendor and device of the first device of the host
   bridge's root bus: Xeon E5 and Core i7; Xeon E7 v3, E5 v3 and Core i7;
   Skylake-E. *)
let whitelist =
  [
    (0x3c00, Same_bridge);
    (0x3c01, Same_bridge);
    (0x2f00, Same_bridge);
    (0x2f01, Same_bridge);
    (0x2030, Any_bridge);
    (0x2031, Any_bridge);
    (0x2032, Any_bridge);
    (0x2033, Any_bridge);
    (0x2020, Any_bridge);
    (0x09a2, Any_bridge);
  ]

(* AMD's CPUs from Zen, family 0x17, on carry requests between any of their
   root complexes (cpu_supports_p2pdma). *)
let zen = 0x17

type route = {
  nodes : (string * bool) list;
      (* the function, then each bridge above it up to its root bus, with
         whether it redirects peer requests *)
  host : string; (* its host bridge, as /sys/devices names it *)
  listed : listed;
  any_host : bool; (* the CPU carries requests between host bridges *)
}

(* The configuration space's extended capabilities start at [extended]; ACS's
   identifier, its control register and the controls that redirect peer
   requests: request redirect, completion redirect and egress control
   (include/uapi/linux/pci_regs.h). A reader without CAP_SYS_ADMIN sees 64
   bytes, so ACS cannot be read, and the device counts as redirecting. *)
let extended = 0x100
let config_size = 0x1000
let acs_id = 0x0d
let acs_control = 0x06
let acs_redirects = 0x04 lor 0x08 lor 0x20

let redirects dir =
  let c =
    In_channel.with_open_bin (Filename.concat dir "config") In_channel.input_all
  in
  let n = String.length c in
  let header at = Int32.to_int (String.get_int32_le c at) land 0xffff_ffff in
  let rec walk at hops =
    let h = header at in
    if h = 0 || h = 0xffff_ffff || hops = 0 then false
    else if h land 0xffff = acs_id then
      at + acs_control + 2 > n
      || String.get_uint16_le c (at + acs_control) land acs_redirects <> 0
    else
      let next = (h lsr 20) land 0xffc in
      next >= extended && walk next (hops - 1)
  in
  if n < extended then true
  else n >= config_size && walk extended (config_size / 4)

let is_bus s = Option.is_some (Bus_address.numbers s)

(* Whether the device at [dir] is a PCI Express root port: its capability
   list, from the pointer at [caps], holds the PCI Express capability, whose
   flags name the port's type (pci_pcie_type). A reader that sees 64 bytes
   cannot tell, and the device counts as none. *)
let caps = 0x34
let pcie_id = 0x10
let root_port = 0x4

let is_root_port dir =
  let c =
    In_channel.with_open_bin (Filename.concat dir "config") In_channel.input_all
  in
  let n = String.length c in
  let rec walk at hops =
    if at < 0x40 || at + 4 > n || hops = 0 then false
    else if String.get_uint8 c at = pcie_id then
      (String.get_uint16_le c (at + 2) lsr 4) land 0xf = root_port
    else walk (String.get_uint8 c (at + 1) land 0xfc) (hops - 1)
  in
  n > caps && walk (String.get_uint8 c caps land 0xfc) 48

(* The first device of a host bridge's root bus, its bus's first, which Linux
   takes if it is the bus's device 00.0 or a root port. *)
let listed host_dir =
  let first =
    List.filter is_bus (Array.to_list (Sys.readdir host_dir))
    |> List.sort Bus_address.compare
  in
  match first with
  | [] -> Unlisted
  | d :: _ -> (
      let dir = Filename.concat host_dir d in
      let file f = read_hex (Filename.concat dir f) in
      let _, _, device, fn = Option.get (Bus_address.numbers d) in
      if (device, fn) <> (0, 0) && not (is_root_port dir) then Unlisted
      else if file "vendor" <> intel then Unlisted
      else
        match List.assoc_opt (file "device") whitelist with
        | Some l -> l
        | None -> Unlisted)

(* /proc/cpuinfo's first [vendor_id] and [cpu family]. *)
let any_host m =
  match
    In_channel.with_open_text (Filename.concat m.root "proc/cpuinfo")
      In_channel.input_all
  with
  | exception Sys_error _ -> false
  | s ->
      let field name =
        String.split_on_char '\n' s
        |> List.find_map (fun line ->
            match String.index_opt line ':' with
            | Some i when String.trim (String.sub line 0 i) = name ->
                Some
                  (String.trim
                     (String.sub line (i + 1) (String.length line - i - 1)))
            | _ -> None)
      in
      field "vendor_id" = Some "AuthenticAMD"
      && Option.bind (field "cpu family") int_of_string_opt
         |> Option.fold ~none:false ~some:(fun f -> f >= zen)

(* A function's directory under /sys/devices names its host bridge, then each
   bridge down to it: /sys/devices/pci0000:00/0000:00:01.1/0000:01:00.0. *)
let route m bus =
  match Unix.realpath (Filename.concat m.devices bus) with
  | exception Unix.Unix_error _ -> None
  | dir -> (
      let rec split dir acc =
        let name = Filename.basename dir in
        if is_bus name then split (Filename.dirname dir) (dir :: acc)
        else (dir, acc)
      in
      let host_dir, down = split dir [] in
      let host = Filename.basename host_dir in
      if down = [] || not (String.starts_with ~prefix:"pci" host) then None
      else
        try
          let nodes =
            List.rev_map (fun d -> (Filename.basename d, redirects d)) down
          in
          Some { nodes; host; listed = listed host_dir; any_host = any_host m }
        with Sys_error _ | Fail.Failed _ -> None)

let through_host ~provider ~client =
  provider.any_host
  ||
  if provider.host = client.host then provider.listed <> Unlisted
  else provider.listed = Any_bridge && client.listed = Any_bridge

let forwards ~provider ~client =
  let rec up redirected = function
    | [] -> None
    | (b, r) :: rest -> (
        let redirected = redirected || r in
        let rec below acc = function
          | [] -> None
          | (b', _) :: _ when b' = b -> Some acc
          | (_, r') :: rest -> below (acc || r') rest
        in
        match below false client.nodes with
        | Some r' -> Some (redirected || r')
        | None -> up redirected rest)
  in
  match up false provider.nodes with
  | Some false -> true
  | Some true | None -> through_host ~provider ~client

(* Two functions meet below a bridge both routes pass, a root port included,
   else at their host bridge if they share one. *)
type distance = Switch | Bridge | System

let distance a b =
  let bridges r = List.map fst (List.tl r.nodes) in
  let theirs = bridges b in
  if List.exists (fun n -> List.mem n theirs) (bridges a) then Switch
  else if a.host = b.host then Bridge
  else System

(* Open devices *)

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

(* What reading a file needs where Linux refuses it. A process's [fd] and
   [map_files] directories are its owner's alone, which CAP_DAC_READ_SEARCH
   passes, and following their links needs ptrace's read access, which
   CAP_SYS_PTRACE gives over another user's process. Following a [map_files]
   link needs CAP_SYS_ADMIN or CAP_CHECKPOINT_RESTORE, for the process's own
   too. Other files need root. *)
type need = Root | Other_process | Mapped_file

let unreadable need file (e : Unix.error) =
  let why = strf "reading %s: %s" file (Unix.error_message e) in
  match (e, need) with
  | (EACCES | EPERM), Root -> Fail.fail "%s; run as root" why
  | (EACCES | EPERM), Other_process ->
      Fail.fail
        "%s; reading another process's files needs CAP_DAC_READ_SEARCH and \
         CAP_SYS_PTRACE: run as root"
        why
  | (EACCES | EPERM), Mapped_file ->
      Fail.fail
        "%s; following a mapped file's link needs CAP_SYS_ADMIN or \
         CAP_CHECKPOINT_RESTORE: run as root"
        why
  | _ -> Fail.fail "%s" why

(* The number of the character device at [file], if there is one. *)
let device need file =
  match Unix.stat file with
  | { st_kind = S_CHR; st_rdev; _ } -> Some st_rdev
  | _ -> None
  | exception Unix.Unix_error (ENOENT, _, _) -> None
  | exception Unix.Unix_error (e, _, _) -> unreadable need file e

(* The names in [dir], or [None] if it is gone, as the directory of a process
   gone since the listing is. *)
let listing need dir =
  match Unix.opendir dir with
  | exception Unix.Unix_error ((ENOENT | ESRCH), _, _) -> None
  | exception Unix.Unix_error (e, _, _) -> unreadable need dir e
  | d ->
      Fun.protect ~finally:(fun () -> Unix.closedir d) @@ fun () ->
      let rec go acc =
        match Unix.readdir d with
        | "." | ".." -> go acc
        | name -> go (name :: acc)
        | exception End_of_file -> Some acc
        | exception Unix.Unix_error ((ENOENT | ESRCH), _, _) -> Some acc
      in
      go []

(* The character devices open in the process whose [proc] directory is [proc],
   by number, each with the file its descriptor or mapping names: its open
   descriptors, under [fd], and its mapped files, under [map_files], which
   outlive a closed descriptor. A directory or an entry gone since the listing
   is left out. *)
let opened ~own proc =
  let others = if own then Root else Other_process in
  List.concat_map
    (fun (dir, link_need) ->
      let dir = Filename.concat proc dir in
      List.filter_map
        (fun name ->
          let link = Filename.concat dir name in
          Option.map
            (fun n ->
              ( n,
                match Unix.readlink link with
                | f -> f
                | exception Unix.Unix_error _ -> link ))
            (device link_need link))
        (Option.value (listing others dir) ~default:[]))
    [ ("fd", others); ("map_files", Mapped_file) ]

(* The numbers of the GPU at [bus]'s character devices. *)
let gpu_devices m bus nodes =
  numbers (Filename.concat m.devices bus)
  @ List.filter_map (fun f -> device Root (Filename.concat m.root f)) nodes

let find_held gpu opened =
  List.find_map
    (fun (n, file) -> if List.mem n gpu then Some file else None)
    opened

let held m bus nodes =
  let self = Filename.concat m.root "proc/self" in
  if listing Root (Filename.concat self "fd") = None then
    Fail.fail "reading %s: no such directory" (Filename.concat self "fd");
  find_held (gpu_devices m bus nodes) (opened ~own:true self)

let held_elsewhere m bus nodes =
  let gpu = gpu_devices m bus nodes and me = string_of_int (Unix.getpid ()) in
  let proc = Filename.concat m.root "proc" in
  List.find_map
    (fun pid ->
      if pid = me || int_of_string_opt pid = None then None
      else
        Option.map
          (fun file -> (int_of_string pid, file))
          (find_held gpu (opened ~own:false (Filename.concat proc pid))))
    (Option.value (listing Root proc) ~default:[])

(* A process's command, as [proc/PID/comm] gives it. *)
let command m pid =
  let file = Filename.concat m.root (strf "proc/%d/comm" pid) in
  match In_channel.with_open_text file In_channel.input_all with
  | s -> Some (String.trim s)
  | exception Sys_error _ -> None

let refusal { files = m; bus; _ } ~remove =
  let s = state m bus in
  if s.iommu = Translating then vfio_refusal m bus ~remove
  else
    match
      addressing m bus { s with driver = None; siblings = []; enabled = true }
    with
    | Ok _ -> None
    | Error why -> Some why

(* DRM's debugfs lists each file of a device, by the device's PCI name: one line
   a file after a header, its fields the opener's command, which may hold
   spaces, then its process id, the minor, master, authenticated, uid and magic.
   A file is listed from its open until its last reference goes, its descriptors
   closed, its mappings gone and any reference the kernel took to it, such as
   KFD's, dropped. *)
let debugfs = "sys/kernel/debug/dri"

let drm_clients m bus =
  if
    not
      (Sys.file_exists (Filename.concat (Filename.concat m.devices bus) "drm"))
  then []
  else
    let file =
      Filename.concat
        (Filename.concat (Filename.concat m.root debugfs) bus)
        "clients"
    in
    match In_channel.with_open_text file In_channel.input_lines with
    | exception Sys_error why ->
        Fail.fail "%s: debugfs lists the files of %s's DRM device there" why bus
    | [] -> []
    | _header :: lines ->
        List.filter_map
          (fun line ->
            let fields =
              List.filter (( <> ) "") (String.split_on_char ' ' line)
            in
            let n = List.length fields in
            if n < 7 then None
            else
              Option.map
                (fun pid ->
                  ( pid,
                    String.concat " "
                      (List.filteri (fun i _ -> i < n - 6) fields) ))
                (int_of_string_opt (List.nth fields (n - 6))))
          lines
