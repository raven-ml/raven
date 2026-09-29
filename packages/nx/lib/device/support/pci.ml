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

type t = {
  bus : string;
  config : int;
  interrupts : int option; (* the eventfd VFIO signals *)
  files : int list; (* every descriptor, the lock last *)
}

let root = "/sys/bus/pci/devices"
let path bus file = Printf.sprintf "%s/%s/%s" root bus file

let read file =
  In_channel.with_open_text file In_channel.input_all |> String.trim

let write file s =
  try Out_channel.with_open_text file (fun oc -> output_string oc s)
  with Sys_error e ->
    failwith
      (Printf.sprintf
         "%s; writing it needs root (run as root, or grant write access to %s)"
         e file)

let hex s =
  int_of_string (if String.starts_with ~prefix:"0x" s then s else "0x" ^ s)

let scan ~vendor ?class_ ids =
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
    |> List.sort String.compare

let driver bus =
  let link = path bus "driver" in
  if Sys.file_exists link then Some (Filename.basename (Unix.readlink link))
  else None

let take ~lock bus =
  let file =
    Filename.concat
      (Filename.get_temp_dir_name ())
      (Printf.sprintf "%s_%s.lock" lock (String.lowercase_ascii bus))
  in
  let fd = file_lock file in
  if fd < 0 then
    failwith
      (Printf.sprintf "%s is held by another process (see: lsof %s)" bus file);
  let files = ref [ fd ] in
  match
    (try Out_channel.with_open_gen [ Open_wronly ] 0 (path bus "enable") ignore
     with Sys_error _ ->
       failwith
         (Printf.sprintf "cannot access the PCI function %s: run as root" bus));
    let vfio = driver bus = Some "vfio-pci" in
    (match driver bus with
    | Some d when d <> "vfio-pci" ->
        write (path bus "driver/unbind") bus;
        if driver bus <> None then
          failwith (Printf.sprintf "the driver %s stays bound to %s" d bus)
    | _ -> ());
    (* The other functions of the device, such as its audio function. *)
    let prefix = String.sub bus 0 (String.length bus - 1) in
    for fn = 1 to 7 do
      let sibling = Printf.sprintf "%s/%s%d" root prefix fn in
      if Sys.file_exists sibling then write (sibling ^ "/remove") "1"
    done;
    let interrupts =
      if vfio then begin
        let group =
          Filename.basename (Unix.readlink (path bus "iommu_group"))
        in
        let container, group, dev, efd =
          vfio_open ("/dev/vfio/noiommu-" ^ group) bus
        in
        files := efd :: dev :: group :: container :: !files;
        Some efd
      end
      else begin
        write (path bus "enable") "1";
        None
      end
    in
    let config = file_open (path bus "config") true in
    files := config :: !files;
    { bus; config; interrupts; files = !files }
  with
  | p -> p
  | exception e ->
      List.iter file_close !files;
      raise e

let bus p = p.bus

let read_config p off n =
  let s = pread p.config off n in
  let v = ref 0 in
  for i = n - 1 downto 0 do
    v := (!v lsl 8) lor Char.code s.[i]
  done;
  !v

let write_config p off n v =
  pwrite p.config off
    (String.init n (fun i -> Char.chr ((v lsr (8 * i)) land 0xff)));
  ignore (read_config p off n)

let bar p i =
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

let map_bar ?(offset = 0) ?length p i =
  let length =
    match length with Some n -> n | None -> snd (bar p i) - offset
  in
  let fd = file_open (path p.bus (Printf.sprintf "resource%d" i)) true in
  Fun.protect
    ~finally:(fun () -> file_close fd)
    (fun () -> Mmio.v (file_map fd offset length) length)

let resize_bar p i =
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

let wait_interrupt p ms =
  match p.interrupts with Some fd -> vfio_wait fd ms | None -> false

let unmap_bar m = file_unmap (Mmio.address m) (Mmio.length m)
let release p = List.iter file_close p.files
