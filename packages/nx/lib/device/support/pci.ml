(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external file_open : string -> bool -> int = "caml_nx_file_open"
external file_close : int -> unit = "caml_nx_file_close"
external pread : int -> int -> int -> string = "caml_nx_file_pread"
external pwrite : int -> int -> string -> unit = "caml_nx_file_pwrite"
external file_map : int -> int -> int -> nativeint = "caml_nx_file_map"
external file_lock : string -> int = "caml_nx_file_lock"
external file_unmap : nativeint -> int -> unit = "caml_nx_file_unmap"

external vfio_open : string -> string -> int * int * int * int
  = "caml_nx_vfio_open"

external vfio_wait : int -> int -> bool = "caml_nx_vfio_wait"

type local = {
  bus : string;
  config : int;
  interrupts : int option; (* the eventfd VFIO signals *)
  files : int list; (* every descriptor, the locks last *)
}

type t =
  | Local of local
  | Remote of {
      remote : Remote.t;
      id : int;
      bus : string;
      bars : (int, Mmio.t) Hashtbl.t; (* mapped whole, by BAR *)
    }

let root = "/sys/bus/pci/devices"
let path bus file = Printf.sprintf "%s/%s/%s" root bus file

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
           "%s; writing it needs root (run as root, or grant write access to \
            %s)"
           e file
       else e)

let readlink link =
  try Unix.readlink link
  with Unix.Unix_error (e, _, _) ->
    failwith (Printf.sprintf "%s: %s" link (Unix.error_message e))

let hex s =
  int_of_string (if String.starts_with ~prefix:"0x" s then s else "0x" ^ s)

(* Bus addresses *)

let address ~domain ~bus ~device ~fn =
  Printf.sprintf "%04x:%02x:%02x.%x" domain bus device fn

(* The numbers of the bus address [a], which Linux spells "DDDD:BB:DD.F". *)
let numbers a =
  let invalid () = invalid_arg (Printf.sprintf "%S is no PCI bus address" a) in
  let num s =
    if
      s <> ""
      && String.length s <= 8
      && String.for_all Char.Ascii.is_hex_digit s
    then int_of_string ("0x" ^ s)
    else invalid ()
  in
  match String.split_on_char ':' a with
  | [ domain; bus; df ] -> (
      match String.split_on_char '.' df with
      | [ device; fn ] -> (num domain, num bus, num device, num fn)
      | _ -> invalid ())
  | _ -> invalid ()

let compare_address a b = compare (numbers a) (numbers b)

(* Functions *)

let scan_local ~vendor ?class_ ids =
  if not (Sys.file_exists root) then []
  else
    Sys.readdir root |> Array.to_list
    |> List.filter (fun bus ->
        try
          let v = hex (read (path bus "vendor"))
          and d = hex (read (path bus "device")) in
          v = vendor
          && List.exists (fun (mask, l) -> List.mem (d land mask) l) ids
          &&
          match class_ with
          | None -> true
          | Some c -> hex (read (path bus "class")) lsr 16 = c
        with Sys_error _ | Failure _ -> false)
    |> List.sort compare_address

let driver bus =
  let link = path bus "driver" in
  if Sys.file_exists link then Some (Filename.basename (readlink link))
  else None

let scan ?remote ~vendor ?class_ ids =
  match remote with
  | None -> scan_local ~vendor ?class_ ids
  | Some r -> Remote.scan r ~vendor ?class_ ids

(* Locks [bus] for this process under [name], or raises naming the file. *)
let lock_file bus name =
  let file =
    Filename.concat
      (Filename.get_temp_dir_name ())
      (Printf.sprintf "%s_%s.lock" name (String.lowercase_ascii bus))
  in
  let fd = file_lock file in
  if fd < 0 then
    failwith
      (Printf.sprintf "%s is held by another process (see: lsof %s)" bus file);
  fd

let exists bus = Sys.file_exists (Filename.concat root bus)

(* The other functions of [bus]'s device, such as its audio function. *)
let siblings bus =
  let prefix = String.sub bus 0 (String.length bus - 1) in
  List.filter
    (fun s -> s <> bus && exists s)
    (List.init 8 (fun fn -> prefix ^ string_of_int fn))

let enabled bus = read (path bus "enable") <> "0"

let detached bus =
  if not (exists bus) then
    Error (Printf.sprintf "%s is no PCI function of this machine" bus)
  else
    match (driver bus, siblings bus) with
    | Some d, _ when d <> "vfio-pci" ->
        Error (Printf.sprintf "%s is bound to the driver %s" bus d)
    | _, s :: _ -> Error (Printf.sprintf "%s shares its device with %s" bus s)
    | Some _, [] -> Ok ()
    | None, [] when enabled bus -> Ok ()
    | None, [] -> Error (Printf.sprintf "%s is disabled" bus)

(* [f ()] while holding [bus]'s own lock, so that no process has it taken. *)
let with_lock bus f =
  if not (exists bus) then
    failwith (Printf.sprintf "%s is no PCI function of this machine" bus);
  let own = lock_file bus "nx" in
  Fun.protect ~finally:(fun () -> file_close own) f

let detach bus =
  with_lock bus @@ fun () ->
  (match driver bus with
  | Some d when d <> "vfio-pci" ->
      write (path bus "driver/unbind") bus;
      if driver bus <> None then
        failwith (Printf.sprintf "the driver %s stays bound to %s" d bus)
  | _ -> ());
  List.iter (fun s -> write (path s "remove") "1") (siblings bus);
  if driver bus = None && not (enabled bus) then write (path bus "enable") "1"

(* A rescan brings back the functions [detach] removed. The function itself is
   on the bus already, so its driver is probed for it. *)
let attach bus =
  with_lock bus @@ fun () ->
  match driver bus with
  | Some "vfio-pci" ->
      failwith
        (Printf.sprintf
           "%s is bound to vfio-pci; unbind it and clear its driver_override \
            first"
           bus)
  | Some _ -> ()
  | None ->
      if enabled bus then write (path bus "enable") "0";
      write "/sys/bus/pci/rescan" "1";
      write "/sys/bus/pci/drivers_probe" bus;
      if driver bus = None then
        failwith
          (Printf.sprintf "no kernel driver took %s; load its module first" bus)

(* The function's own lock, which every driver of this library takes whatever
   its name, then the driver's, which other drivers of the same GPU take. *)
let take_local ~lock bus =
  let own = lock_file bus "nx" in
  let files = ref [ own ] in
  match
    files := lock_file bus lock :: !files;
    (match detached bus with Ok () -> () | Error why -> failwith why);
    (try Out_channel.with_open_gen [ Open_wronly ] 0 (path bus "enable") ignore
     with Sys_error _ ->
       failwith
         (Printf.sprintf "cannot access the PCI function %s: run as root" bus));
    let interrupts =
      if driver bus = Some "vfio-pci" then begin
        let group = Filename.basename (readlink (path bus "iommu_group")) in
        let container, group, dev, efd =
          vfio_open ("/dev/vfio/noiommu-" ^ group) bus
        in
        files := efd :: dev :: group :: container :: !files;
        Some efd
      end
      else None
    in
    let config = file_open (path bus "config") true in
    files := config :: !files;
    Local { bus; config; interrupts; files = !files }
  with
  | p -> p
  | exception e ->
      List.iter file_close !files;
      raise e

let take ?remote ~lock bus =
  match remote with
  | None -> take_local ~lock bus
  | Some remote ->
      let id = Remote.take remote ~lock bus in
      Remote { remote; id; bus; bars = Hashtbl.create 4 }

let bus = function Local p -> p.bus | Remote p -> p.bus
let remote = function Local _ -> None | Remote p -> Some p.remote

let read_config t off n =
  match t with
  | Remote p -> Remote.read_config p.remote p.id off n
  | Local p ->
      let s = pread p.config off n in
      let v = ref 0 in
      for i = n - 1 downto 0 do
        v := (!v lsl 8) lor Char.code s.[i]
      done;
      !v

let write_config t off n v =
  match t with
  | Remote p -> Remote.write_config p.remote p.id off n v
  | Local p ->
      pwrite p.config off
        (String.init n (fun i -> Char.chr ((v lsr (8 * i)) land 0xff)));
      ignore (read_config t off n)

let bar_local p i =
  match
    List.nth_opt (String.split_on_char '\n' (read (path p.bus "resource"))) i
  with
  | Some line -> (
      match String.split_on_char ' ' line with
      | start :: stop :: _ ->
          let start = hex start and stop = hex stop in
          (start, stop - start + 1)
      | _ -> failwith (Printf.sprintf "%s: no BAR %d" p.bus i))
  | None -> failwith (Printf.sprintf "%s: no BAR %d" p.bus i)

let bar t i =
  match t with
  | Local p -> bar_local p i
  | Remote p -> Remote.bar p.remote p.id i

let map_bar ?(offset = 0) ?length t i =
  let length =
    match length with Some n -> n | None -> snd (bar t i) - offset
  in
  match t with
  | Local p ->
      let fd = file_open (path p.bus (Printf.sprintf "resource%d" i)) true in
      Fun.protect
        ~finally:(fun () -> file_close fd)
        (fun () -> Mmio.v (file_map fd offset length) length)
  | Remote p ->
      let whole =
        match Hashtbl.find_opt p.bars i with
        | Some m -> m
        | None ->
            let m = Remote.map_bar p.remote p.id i in
            Hashtbl.replace p.bars i m;
            m
      in
      Mmio.sub whole offset length

let resize_bar_local p i =
  let file = path p.bus (Printf.sprintf "resource%d_resize" i) in
  try
    let sizes = hex (read file) in
    let rec bits n k = if n = 0 then k else bits (n lsr 1) (k + 1) in
    write file (string_of_int (bits sizes 0 - 1))
  with Sys_error e | Failure e ->
    failwith
      (Printf.sprintf
         "cannot resize BAR %d of %s: %s; enable Resizable BAR in the firmware \
          settings"
         i p.bus e)

let resize_bar t i =
  match t with
  | Local p -> resize_bar_local p i
  | Remote p -> Remote.resize_bar p.remote p.id i

(* A function answers its configuration reads again once its vendor ID reads
   back as other than all ones. *)
let reset t =
  match t with
  | Remote p -> Remote.reset p.remote p.id
  | Local p ->
      write (path p.bus "reset") "1";
      let rec wait k =
        if read_config t 0 2 = 0xffff then
          if k = 0 then
            failwith (Printf.sprintf "%s does not answer after its reset" p.bus)
          else begin
            Unix.sleepf 0.01;
            wait (k - 1)
          end
      in
      wait 100

let wait_interrupt t ms =
  match t with
  | Local { interrupts = Some fd; _ } -> vfio_wait fd ms
  | Local { interrupts = None; _ } | Remote _ -> false

(* A remote BAR stays mapped on its machine until the function is released. *)
let unmap_bar m =
  if not (Mmio.is_remote m) then file_unmap (Mmio.address m) (Mmio.length m)

let release = function
  | Local p -> List.iter file_close p.files
  | Remote p -> Remote.release p.remote p.id

(* System memory of the function's machine *)

let page t = match remote t with None -> Sysmem.page | Some r -> Remote.page r

let reserve t ~base n =
  match remote t with
  | None -> Sysmem.reserve ~base n
  | Some r -> Remote.reserve r ~base n

let alloc_sysmem t ?contiguous ?va n =
  match remote t with
  | None -> Sysmem.alloc ?contiguous ?va n
  | Some r -> Remote.alloc_sysmem r ?contiguous ?va n

let free_sysmem t m =
  match remote t with None -> Sysmem.free m | Some r -> Remote.free_sysmem r m

let pin t a n =
  match remote t with None -> Sysmem.pin a n | Some r -> Remote.pin r a n

let unpin t a n =
  match remote t with None -> Sysmem.unpin a n | Some r -> Remote.unpin r a n
