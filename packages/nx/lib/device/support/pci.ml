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
external vfio_file : string -> int = "caml_nx_vfio_file"
external vfio_ioctl : int -> int -> int -> int = "caml_nx_vfio_ioctl"

external vfio_ioctl_bytes : int -> int -> bytes -> int
  = "caml_nx_vfio_ioctl_bytes"

external vfio_eventfd : unit -> int = "caml_nx_vfio_eventfd"
external vfio_wait : int -> int -> bool = "caml_nx_vfio_wait"

type addressing = Physical | Iommu
type iommu = No_iommu | Identity | Translating

type state = {
  driver : string option;
  iommu : iommu;
  siblings : string list;
  enabled : bool;
}

(* The VFIO container of a function behind an IOMMU, which maps system memory
   for it, at device addresses the process allocates. Mappings are counted by
   their (address, bytes), as pins are. Once the function is released, closing
   the container has removed them all. *)
type container = {
  fd : int;
  device : int; (* the function's VFIO file *)
  iova : Vfio.Iova.t;
  maps : (nativeint * int, int * int) Hashtbl.t; (* to (device address, pins) *)
  mutex : Mutex.t;
  mutable closed : bool;
}

type local = {
  bus : string;
  config : int * int; (* the file, and where configuration space starts in it *)
  interrupts : int option; (* the eventfd VFIO signals *)
  container : container option; (* behind an IOMMU *)
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

(* How a function is taken *)

let vfio_pci = "vfio-pci"
let groups = "/sys/kernel/iommu_groups"

let group bus =
  let link = path bus "iommu_group" in
  if Sys.file_exists link then Some (Filename.basename (readlink link))
  else None

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

let state bus =
  {
    driver = driver bus;
    iommu = iommu_of bus;
    siblings = siblings bus;
    enabled = enabled bus;
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
  | { driver = Some _; iommu = Identity | Translating; _ } -> Ok Iommu
  | { driver = None; iommu = Translating; _ } ->
      Error
        (Printf.sprintf
           "the IOMMU translates the addresses %s reaches, so it cannot reach \
            physical ones; bind it to vfio-pci to take it through the IOMMU \
            (%s), or boot Linux with iommu=pt"
           bus (bind_vfio bus))
  | { siblings = s :: _; _ } ->
      Error (Printf.sprintf "%s shares its device with %s" bus s)
  | { driver = Some _; _ } | { driver = None; enabled = true; _ } -> Ok Physical
  | { driver = None; enabled = false; _ } ->
      Error (Printf.sprintf "%s is disabled" bus)

let detached bus =
  if not (exists bus) then
    Error (Printf.sprintf "%s is no PCI function of this machine" bus)
  else access bus (state bus)

(* [f ()] while holding [bus]'s own lock, so that no process has it taken. *)
let with_lock bus f =
  if not (exists bus) then
    failwith (Printf.sprintf "%s is no PCI function of this machine" bus);
  let own = lock_file bus "nx" in
  Fun.protect ~finally:(fun () -> file_close own) f

(* A function {!detached} already is stays as it is: one bound to vfio-pci
   behind an IOMMU keeps its siblings, which the IOMMU keeps apart. *)
let detach bus =
  with_lock bus @@ fun () ->
  match detached bus with
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
      match detached bus with Ok _ -> () | Error why -> failwith why)

(* A rescan brings back the functions [detach] removed. The function itself is
   on the bus already, so its driver is probed for it. *)
let attach bus =
  with_lock bus @@ fun () ->
  match driver bus with
  | Some "vfio-pci" ->
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

(* VFIO *)

let user () =
  try (Unix.getpwuid (Unix.getuid ())).pw_name
  with Not_found -> string_of_int (Unix.getuid ())

let request = Vfio.request

(* [f ()], whose VFIO request failing is reported as [what]. *)
let step what f =
  try f ()
  with Unix.Unix_error (e, _, _) ->
    failwith (Printf.sprintf "%s: %s" what (Unix.error_message e))

(* Opens the VFIO file [file] of [bus], naming the step that grants it. *)
let open_vfio bus file =
  match vfio_file file with
  | fd -> fd
  | exception Unix.Unix_error ((EACCES | EPERM), _, _) ->
      let u = user () in
      failwith
        (Printf.sprintf
           "%s: permission denied; grant it to %s with a udev rule: echo \
            'SUBSYSTEM==\"vfio\", KERNEL==\"%s\", OWNER=\"%s\"' | sudo tee -a \
            /etc/udev/rules.d/90-raven-vfio.rules && sudo udevadm control \
            --reload && sudo udevadm trigger --action=add \
            --subsystem-match=vfio"
           file u (Filename.basename file) u)
  | exception Unix.Unix_error (ENOENT, _, _) ->
      failwith
        (Printf.sprintf "%s does not exist; bind %s to vfio-pci: %s" file bus
           (bind_vfio bus))
  | exception Unix.Unix_error (EBUSY, _, _) ->
      failwith (Printf.sprintf "%s is open in another process" file)
  | exception Unix.Unix_error (e, _, _) ->
      failwith (Printf.sprintf "%s: %s" file (Unix.error_message e))

(* The functions of IOMMU group [g] that a driver other than VFIO's holds, with
   that driver. Bridges are held by pcieport, which VFIO accepts. *)
let held g =
  let dir = Printf.sprintf "%s/%s/devices" groups g in
  match Sys.readdir dir with
  | exception Sys_error _ -> []
  | fns ->
      Array.to_list fns |> List.sort String.compare
      |> List.filter_map (fun f ->
          match driver f with
          | Some d when not (List.mem d [ vfio_pci; "pci-stub"; "pcieport" ]) ->
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
    (String.concat " && " (List.map (fun (f, _) -> bind_vfio f) held))

(* A structure the kernel fills in, asked for again at the size it needs when
   capabilities make it larger. *)
let ask fd r make =
  let b = make None in
  ignore (vfio_ioctl_bytes fd (request r) b);
  if Vfio.argsz b <= Bytes.length b then b
  else begin
    let b = make (Some (Vfio.argsz b)) in
    ignore (vfio_ioctl_bytes fd (request r) b);
    b
  end

let region bus device i =
  step (Printf.sprintf "reading region %d of %s" i bus) @@ fun () ->
  Vfio.region
    (ask device Device_get_region_info (fun argsz -> Vfio.region_info ?argsz i))

(* Opens [bus]'s group [file] in a container of its own with the IOMMU model
   [m], then the function, whose first MSI vector is routed to an eventfd. Each
   descriptor goes on [files] once open. Is the container, the function's file
   and the eventfd. *)
let open_function files bus g m file =
  let opened fd =
    files := fd :: !files;
    fd
  in
  let container = opened (open_vfio bus "/dev/vfio/vfio") in
  step "VFIO" (fun () ->
      if vfio_ioctl container (request Get_api_version) 0 <> Vfio.api_version
      then failwith "VFIO speaks another API version");
  let supported =
    step "VFIO" (fun () ->
        vfio_ioctl container (request Check_extension) (Vfio.iommu m) > 0)
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
  let status = Vfio.group_status () in
  step ("reading the status of " ^ file) (fun () ->
      ignore (vfio_ioctl_bytes group (request Group_get_status) status));
  if not (Vfio.viable status) then failwith (not_viable bus g);
  step
    ("attaching " ^ file ^ " to a VFIO container")
    (fun () ->
      let c = Bytes.create 4 in
      Bytes.set_int32_ne c 0 (Int32.of_int container);
      ignore (vfio_ioctl_bytes group (request Group_set_container) c));
  (* Linux attaches a group to an IOMMU only where the IOMMU also remaps the
     function's interrupts, so that it cannot raise another's. *)
  (match vfio_ioctl container (request Set_iommu) (Vfio.iommu m) with
  | _ -> ()
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
         (fun () ->
           vfio_ioctl_bytes group
             (request Group_get_device_fd)
             (Bytes.of_string (bus ^ "\000"))))
  in
  let efd = opened (step "creating an eventfd" vfio_eventfd) in
  step ("routing the interrupt of " ^ bus) (fun () ->
      ignore (vfio_ioctl_bytes device (request Device_set_irqs) (Vfio.msi efd)));
  (container, device, efd)

(* The device addresses a container maps at the system's pages. *)
let addresses bus fd =
  let info =
    step ("reading the IOMMU of " ^ bus) @@ fun () ->
    Vfio.iommu_of
      (ask fd Iommu_get_info (fun argsz -> Vfio.iommu_info ?argsz ()))
  in
  let smallest = info.page_sizes land -info.page_sizes in
  if smallest > Sysmem.page then
    failwith
      (Printf.sprintf "the IOMMU of %s maps no page smaller than %d bytes" bus
         smallest);
  Vfio.Iova.create ~page:Sysmem.page info.ranges

(* Taking *)

let take_iommu files bus =
  let g = Option.get (group bus) in
  let fd, device, efd =
    open_function files bus g Vfio.Type1v2 ("/dev/vfio/" ^ g)
  in
  let config = region bus device Vfio.config_region in
  let container =
    {
      fd;
      device;
      iova = addresses bus fd;
      maps = Hashtbl.create 64;
      mutex = Mutex.create ();
      closed = false;
    }
  in
  {
    bus;
    config = (device, config.offset);
    interrupts = Some efd;
    container = Some container;
    files = !files;
  }

let take_physical files bus =
  (try Out_channel.with_open_gen [ Open_wronly ] 0 (path bus "enable") ignore
   with Sys_error _ ->
     failwith
       (Printf.sprintf "cannot access the PCI function %s: run as root" bus));
  let interrupts =
    if driver bus = Some vfio_pci then
      let g = Option.get (group bus) in
      let _, _, efd =
        open_function files bus g Vfio.No_iommu ("/dev/vfio/noiommu-" ^ g)
      in
      Some efd
    else None
  in
  let config = file_open (path bus "config") true in
  files := config :: !files;
  { bus; config = (config, 0); interrupts; container = None; files = !files }

(* The function's own lock, which every driver of this library takes whatever
   its name, then the driver's, which other drivers of the same GPU take. *)
let take_local ~lock bus =
  let files = ref [ lock_file bus "nx" ] in
  match
    files := lock_file bus lock :: !files;
    match detached bus with
    | Error why -> failwith why
    | Ok Iommu -> take_iommu files bus
    | Ok Physical -> take_physical files bus
  with
  | p -> Local p
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

let addressing = function
  | Local { container = Some _; _ } -> Iommu
  | Local { container = None; _ } | Remote _ -> Physical

let read_config t off n =
  match t with
  | Remote p -> Remote.read_config p.remote p.id off n
  | Local { config = fd, at; _ } ->
      let s = pread fd (at + off) n in
      let v = ref 0 in
      for i = n - 1 downto 0 do
        v := (!v lsl 8) lor Char.code s.[i]
      done;
      !v

let write_config t off n v =
  match t with
  | Remote p -> Remote.write_config p.remote p.id off n v
  | Local { config = fd, at; _ } ->
      pwrite fd (at + off)
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
  | Local { bus; container = Some c; _ } ->
      let r = region bus c.device i in
      let inside (o, n) = offset >= o && offset + length <= o + n in
      if not r.mappable then
        failwith (Printf.sprintf "%s: VFIO does not map BAR %d" bus i);
      (match r.areas with
      | Some areas when not (List.exists inside areas) ->
          failwith
            (Printf.sprintf
               "%s: VFIO maps only parts of BAR %d, none of which holds [0x%x, \
                0x%x)"
               bus i offset (offset + length))
      | _ -> ());
      Mmio.v (file_map c.device (r.offset + offset) length) length
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
      (match p.container with
      | Some c ->
          step ("resetting " ^ p.bus) (fun () ->
              ignore (vfio_ioctl c.device (request Device_reset) 0))
      | None -> write (path p.bus "reset") "1");
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
  | Local p ->
      Option.iter
        (fun c -> Mutex.protect c.mutex (fun () -> c.closed <- true))
        p.container;
      List.iter file_close p.files
  | Remote p -> Remote.release p.remote p.id

(* System memory of the function's machine *)

let page t = match remote t with None -> Sysmem.page | Some r -> Remote.page r

let reserve t ~base n =
  match remote t with
  | None -> Sysmem.reserve ~base n
  | Some r -> Remote.reserve r ~base n

(* Mapping memory behind an IOMMU pins it, and the kernel counts the pinned
   bytes against the process's locked-memory limit. *)
let map_error bus n e =
  let u = user () in
  match (e : Unix.error) with
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

(* The device address at which [c] maps the [n] bytes at [a], mapping them for
   the first pin. *)
let map_dma bus c a n =
  Mutex.protect c.mutex @@ fun () ->
  if c.closed then failwith (bus ^ " is released");
  match Hashtbl.find_opt c.maps (a, n) with
  | Some (iova, k) ->
      Hashtbl.replace c.maps (a, n) (iova, k + 1);
      iova
  | None ->
      let iova =
        match Vfio.Iova.alloc c.iova n with
        | Some x -> x
        | None ->
            failwith
              (Printf.sprintf "%s has no device addresses left for %d bytes" bus
                 n)
      in
      (match
         vfio_ioctl_bytes c.fd (request Iommu_map_dma)
           (Vfio.map_dma ~va:a ~iova n)
       with
      | _ -> ()
      | exception Unix.Unix_error (e, _, _) ->
          Vfio.Iova.free c.iova iova;
          failwith (map_error bus n e));
      Hashtbl.replace c.maps (a, n) (iova, 1);
      iova

(* Releases a pin of the [n] bytes at [a], unmapping them with the last. A
   released function's container has unmapped them already. *)
let unmap_dma bus c a n =
  Mutex.protect c.mutex @@ fun () ->
  match Hashtbl.find_opt c.maps (a, n) with
  | None -> ()
  | Some (iova, k) when k > 1 -> Hashtbl.replace c.maps (a, n) (iova, k - 1)
  | Some (iova, _) ->
      if not c.closed then
        step (Printf.sprintf "unmapping %d bytes for %s" n bus) (fun () ->
            ignore
              (vfio_ioctl_bytes c.fd (request Iommu_unmap_dma)
                 (Vfio.unmap_dma ~iova n)));
      Hashtbl.remove c.maps (a, n);
      Vfio.Iova.free c.iova iova

let pages iova n =
  List.init (n / Sysmem.page) (fun i -> iova + (i * Sysmem.page))

let round_page n = (n + Sysmem.page - 1) / Sysmem.page * Sysmem.page

let alloc_sysmem t ?(contiguous = false) ?va n =
  match t with
  | Remote p -> Remote.alloc_sysmem p.remote ~contiguous ?va n
  | Local { container = None; _ } -> Sysmem.alloc ~contiguous ?va n
  | Local { bus; container = Some c; _ } -> (
      let m = Sysmem.map ?va n in
      let n = Mmio.length m in
      match map_dma bus c (Mmio.address m) n with
      | exception e ->
          Sysmem.free m;
          raise e
      | iova -> (m, if contiguous then [ iova ] else pages iova n))

let free_sysmem t m =
  match t with
  | Remote p -> Remote.free_sysmem p.remote m
  | Local { container = None; _ } -> Sysmem.free m
  | Local { bus; container = Some c; _ } ->
      unmap_dma bus c (Mmio.address m) (Mmio.length m);
      Sysmem.free m

let pin t a n =
  match t with
  | Remote p -> Remote.pin p.remote a n
  | Local { container = None; _ } -> Sysmem.pin a n
  | Local { bus; container = Some c; _ } ->
      if Nativeint.rem a (Nativeint.of_int Sysmem.page) <> 0n then
        invalid_arg (Printf.sprintf "Pci.pin: 0x%nx is not on a page" a);
      let n = round_page n in
      pages (map_dma bus c a n) n

let unpin t a n =
  match t with
  | Remote p -> Remote.unpin p.remote a n
  | Local { container = None; _ } -> Sysmem.unpin a n
  | Local { bus; container = Some c; _ } -> unmap_dma bus c a (round_page n)
