(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let root = "/sys/bus/pci/devices"
let groups = "/sys/kernel/iommu_groups"
let lockdown = "/sys/kernel/security/lockdown"
let vfio_pci = "vfio-pci"
let path bus file = Printf.sprintf "%s/%s/%s" root bus file
let exists bus = Sys.file_exists (Filename.concat root bus)

let read file =
  In_channel.with_open_text file In_channel.input_all |> String.trim

let write file s =
  try Out_channel.with_open_text file (fun oc -> output_string oc s)
  with Sys_error e ->
    let denied =
      List.exists
        (fun suffix -> String.ends_with ~suffix e)
        [ "Permission denied"; "Operation not permitted" ]
    in
    failwith
      (if denied then
         Printf.sprintf
           "%s; writing it needs CAP_SYS_ADMIN and write access to %s (run as \
            root)"
           e file
       else e)

let readlink link =
  try Unix.readlink link
  with Unix.Unix_error (e, _, _) ->
    failwith (Printf.sprintf "%s: %s" link (Unix.error_message e))

let hex s =
  int_of_string (if String.starts_with ~prefix:"0x" s then s else "0x" ^ s)

(* Functions *)

let link bus file =
  let l = path bus file in
  if Sys.file_exists l then Some (Filename.basename (readlink l)) else None

let driver bus = link bus "driver"
let group bus = link bus "iommu_group"

(* A function the kernel lists but whose files cannot be read, as while it is
   being removed, is left out. *)
let functions () =
  if not (Sys.file_exists root) then []
  else
    Sys.readdir root |> Array.to_list
    |> List.filter_map (fun bus ->
        match
          {
            Ops.bus;
            vendor = hex (read (path bus "vendor"));
            device = hex (read (path bus "device"));
            class_ = hex (read (path bus "class")) lsr 16;
          }
        with
        | id -> Some id
        | exception (Sys_error _ | Failure _) -> None)
    |> List.sort (fun (a : Ops.id) b -> Address.compare a.bus b.bus)

(* The other functions of [bus]'s device, such as its audio function. *)
let siblings bus =
  let prefix = String.sub bus 0 (String.length bus - 1) in
  List.filter
    (fun s -> s <> bus && exists s)
    (List.init 8 (fun fn -> prefix ^ string_of_int fn))

let enabled bus = read (path bus "enable") <> "0"

(* BARs *)

(* BAR registers start here in configuration space, every reader sees the first
   [header] bytes of it, and BAR register bits say what the BAR is. *)
let bar_base = 0x10
let bars = 6
let header = 64
let io_bar = 1
let wide_bar = 0b100
let type_mask = 0b110

let registers bus =
  let s =
    In_channel.with_open_bin (path bus "config") (fun ic ->
        really_input_string ic header)
  in
  Array.init bars (fun i ->
      Int32.to_int (String.get_int32_le s (bar_base + (4 * i))) land 0xffff_ffff)

let resource bus i =
  match
    List.nth_opt (String.split_on_char '\n' (read (path bus "resource"))) i
  with
  | None -> None
  | Some line -> (
      match String.split_on_char ' ' line with
      | start :: stop :: _ ->
          let start = hex start and stop = hex stop in
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
  else if regs.(i) land io_bar <> 0 then Some (regs.(i) land lnot 0b11)
  else
    let low = regs.(i) land lnot 0xf in
    if wide i && i + 1 < bars then Some (low lor (regs.(i + 1) lsl 32))
    else Some low

let bar bus i =
  if i < 0 || i >= bars then None
  else
    match (address (registers bus) i, resource bus i) with
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
let iommu_of bus =
  match group bus with
  | None -> No_iommu
  | Some g when Sys.file_exists ("/dev/vfio/noiommu-" ^ g) -> No_iommu
  | Some g -> (
      match read (Printf.sprintf "%s/%s/type" groups g) with
      | "identity" -> Identity
      | _ -> Translating
      | exception Sys_error _ -> Translating)

(* The kernel's lockdown file lists the modes with the current one in
   brackets. *)
let locked_down () =
  match read lockdown with
  | s -> not (String.starts_with ~prefix:"[none]" s)
  | exception Sys_error _ -> false

let state bus =
  {
    driver = driver bus;
    iommu = iommu_of bus;
    siblings = siblings bus;
    enabled = enabled bus;
    locked_down = locked_down ();
  }

let bind_vfio bus = Printf.sprintf "sudo driverctl set-override %s vfio-pci" bus

let access bus s =
  match s with
  | { driver = Some d; iommu = No_iommu; _ } when d <> vfio_pci ->
      Error (Printf.sprintf "%s is bound to the driver %s" bus d)
  | { driver = Some d; _ } when d <> vfio_pci ->
      Error
        (Printf.sprintf
           "%s is bound to the driver %s (to take it through the IOMMU, \
            without root, bind it to vfio-pci: %s)"
           bus d (bind_vfio bus))
  | { driver = Some _; iommu = Identity | Translating; _ } -> Ok Ops.Iommu
  | { driver = None; iommu = Translating; _ } ->
      Error
        (Printf.sprintf
           "the IOMMU translates the addresses %s reaches, so it cannot reach \
            physical ones; bind it to vfio-pci to take it through the IOMMU \
            (%s), or boot Linux with iommu=pt"
           bus (bind_vfio bus))
  | { siblings = s :: _; _ } ->
      Error (Printf.sprintf "%s shares its device with %s" bus s)
  | { driver = None; enabled = false; _ } ->
      Error (Printf.sprintf "%s is disabled" bus)
  | { locked_down = true; _ } ->
      Error
        (Printf.sprintf
           "the kernel is locked down (%s), which refuses mapping a BAR \
            outside VFIO; take %s behind an IOMMU: turn the IOMMU on and bind \
            it to vfio-pci (%s)"
           lockdown bus (bind_vfio bus))
  | _ -> Ok Ops.Physical

(* Changes *)

let detach bus =
  match access bus (state bus) with
  | Ok _ -> ()
  | Error _ -> (
      (match driver bus with
      | Some d when d <> vfio_pci ->
          write (path bus "driver/unbind") bus;
          if driver bus <> None then
            failwith (Printf.sprintf "the driver %s stays bound to %s" d bus)
      | _ -> ());
      List.iter (fun s -> write (path s "remove") "1") (siblings bus);
      if driver bus = None && not (enabled bus) then
        write (path bus "enable") "1";
      match access bus (state bus) with Ok _ -> () | Error why -> failwith why)

let reset bus = write (path bus "reset") "1"

(* A rescan brings back the functions [detach] removed. The function itself is
   on the bus already, so its driver is probed for it. *)
let attach bus =
  match driver bus with
  | Some d when d = vfio_pci ->
      failwith
        (Printf.sprintf
           "%s is bound to vfio-pci; unbind it and clear its driver_override \
            first: sudo driverctl unset-override %s"
           bus bus)
  | Some _ -> ()
  | None ->
      if enabled bus then write (path bus "enable") "0";
      write "/sys/bus/pci/rescan" "1";
      write "/sys/bus/pci/drivers_probe" bus;
      if driver bus = None then
        failwith
          (Printf.sprintf "no kernel driver took %s; load its module first" bus)

(* [resourceN_resize] holds a bitmap of the sizes BAR [N] supports, bit [k] for
   [2^k] MiB, and takes the [k] to set. A bridge whose window cannot hold a size
   refuses it with ENOSPC. *)
let resize bus i =
  let file = path bus (Printf.sprintf "resource%d_resize" i) in
  if driver bus = None && Sys.file_exists file then
    let sizes = hex (read file) in
    let rec try_from k =
      if k >= 0 then
        if sizes land (1 lsl k) = 0 then try_from (k - 1)
        else
          match write file (string_of_int k) with
          | () -> ()
          | exception Failure e
            when String.ends_with ~suffix:"No space left on device" e ->
              try_from (k - 1)
    in
    try_from (Sys.int_size - 2)
