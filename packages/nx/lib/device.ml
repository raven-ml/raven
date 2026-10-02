(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A device is a memory, which nx.device opens, and the backend that computes
   eagerly on it, if any. Several devices may share one memory: their values
   share storage, and only who computes on them differs. Devices are plain
   values, equal when their memories and backends are. *)
type t = { memory : Nx_device.t; backend : Nx_backend.t option }

(* A memory's default backend is a fixed match over its kind: nx.cpu on the
   memories the host computes on (the host and test memories), none on the
   others. *)
let default_backend m =
  if Nx_device.runs_on_host m then Some Nx_cpu.backend else None

let check_runs_on what b m =
  let (module K) = Nx_backend.kernels b in
  if not (K.runs_on m) then
    invalid_arg
      (Printf.sprintf "Nx.Device.%s: %s does not compute on %s" what K.name
         (Nx_device.name m))

let make ?backend memory =
  match backend with
  | None -> { memory; backend = default_backend memory }
  | Some b ->
      check_runs_on "make" b memory;
      { memory; backend }

let with_backend b d =
  check_runs_on "with_backend" b d.memory;
  { d with backend = Some b }

let host = make Nx_device.host
let memory d = d.memory

let equal d d' =
  Nx_device.equal d.memory d'.memory
  && Option.equal Nx_backend.equal d.backend d'.backend

(* A device's name is its memory's, then its backend's when it is not the
   memory's default, as in ["CPU:1/nx-oxcaml"]. *)
let name d =
  let m = Nx_device.name d.memory in
  let default =
    Option.equal Nx_backend.equal d.backend (default_backend d.memory)
  in
  match d.backend with
  | Some b when not default -> m ^ "/" ^ Nx_backend.name b
  | Some _ | None -> m

let pp ppf d = Format.pp_print_string ppf (name d)

(* Test memories: ["CPU:k"] holds its values in the host's memory, which it maps
   as it is, and loads no programs, so the host computes on it. Each [k] is one
   memory, minted by its first use: nx.device mints a name once. *)

let tests = Hashtbl.create 4
let tests_lock = Mutex.create ()

let test_memory k =
  Mutex.protect tests_lock @@ fun () ->
  match Hashtbl.find_opt tests k with
  | Some m -> m
  | None ->
      let m =
        Nx_device.Driver.device
          ~name:(Printf.sprintf "CPU:%d" k)
          ~arch:"test" ~budget:max_int
          (Host_visible
             { memory = Nx_device.Driver.host_memory; mapping = Some Identity })
      in
      Hashtbl.add tests k m;
      m

let cpu k =
  if k < 1 then invalid_arg (Printf.sprintf "Nx.Device.cpu: %d < 1" k);
  make (test_memory k)

(* Wants *)

type want =
  | Host
  | Cpu of int
  | Gpu
  | Metal
  | Cuda of int
  | Nv of int
  | Amd of int
  | Nv_pci of int
  | Amd_pci of int

(* [Gpu] is the Mac's Metal GPU, and elsewhere the first GPU that opens through
   a kernel driver, the vendor's runtime before nx's own. Selection never takes
   a GPU over PCI, which detaches its kernel driver. *)
let gpu_memory () =
  let wants =
    if Device_metal.on_mac then [ Device_metal.get ]
    else
      [
        (fun () -> Nx_cuda_device.get 0);
        (fun () -> Nx_nv_device.get ~interface:Kernel 0);
        (fun () -> Nx_amd_device.get ~interface:Kernel 0);
      ]
  in
  let rec go reasons = function
    | [] -> Error ("no GPU opens: " ^ String.concat "; " (List.rev reasons))
    | open_ :: rest -> (
        match open_ () with Ok m -> Ok m | Error e -> go (e :: reasons) rest)
  in
  go [] wants

let index what i =
  if i < 0 then invalid_arg (Printf.sprintf "Nx.Device.get: %s %d < 0" what i)

let memory_of = function
  | Host -> Ok Nx_device.host
  | Cpu k ->
      if k < 1 then
        invalid_arg (Printf.sprintf "Nx.Device.get: CPU:%d needs k >= 1" k);
      Ok (test_memory k)
  | Gpu -> gpu_memory ()
  | Metal -> Device_metal.get ()
  | Cuda i ->
      index "CUDA" i;
      Nx_cuda_device.get i
  | Nv i ->
      index "NV" i;
      Nx_nv_device.get ~interface:Kernel i
  | Nv_pci i ->
      index "NV-PCI" i;
      Nx_nv_device.get ~interface:Pci i
  | Amd i ->
      index "AMD" i;
      Nx_amd_device.get ~interface:Kernel i
  | Amd_pci i ->
      index "AMD-PCI" i;
      Nx_amd_device.get ~interface:Pci i

let get w = Result.map (fun m -> make m) (memory_of w)
let v w = match get w with Ok d -> d | Error e -> failwith e

let gpu () =
  match get Gpu with Ok d -> d | Error e -> failwith ("Nx.Device.gpu: " ^ e)

let first ws =
  if ws = [] then invalid_arg "Nx.Device.first: no device";
  let rec go reasons = function
    | [] ->
        failwith
          ("Nx.Device.first: none opens: "
          ^ String.concat "; " (List.rev reasons))
    | w :: ws -> (
        match get w with Ok d -> d | Error e -> go (e :: reasons) ws)
  in
  go [] ws

(* Names *)

let pp_want ppf w =
  let s = Format.pp_print_string ppf and n = Format.fprintf ppf "%s:%d" in
  match w with
  | Host -> s "CPU"
  | Cpu k -> n "CPU" k
  | Gpu -> s "GPU"
  | Metal -> s "METAL"
  | Cuda i -> n "CUDA" i
  | Nv i -> n "NV" i
  | Amd i -> n "AMD" i
  | Nv_pci i -> n "NV-PCI" i
  | Amd_pci i -> n "AMD-PCI" i

let all ws =
  let rec repeated = function
    | [] -> ()
    | w :: rest ->
        if List.mem w rest then
          invalid_arg
            (Format.asprintf "Nx.Device.all: %a is wanted twice" pp_want w);
        repeated rest
  in
  repeated ws;
  let opened = List.map get ws in
  match
    List.filter_map (function Error e -> Some e | Ok _ -> None) opened
  with
  | [] -> List.map Result.get_ok opened
  | errors -> failwith ("Nx.Device.all: " ^ String.concat "; " errors)

(* A name and an index, "kind" or "kind:i", in any case, around blanks: an index
   left out is 0. *)
let want_of_name s =
  let index i =
    if i <> "" && String.for_all (fun c -> c >= '0' && c <= '9') i then
      int_of_string_opt i
    else None
  in
  let kind k i =
    match k with
    | "CPU" -> if i >= 1 then Some (Cpu i) else None
    | "CUDA" -> Some (Cuda i)
    | "NV" -> Some (Nv i)
    | "AMD" -> Some (Amd i)
    | "NV-PCI" -> Some (Nv_pci i)
    | "AMD-PCI" -> Some (Amd_pci i)
    | _ -> None
  in
  match String.split_on_char ':' (String.uppercase_ascii (String.trim s)) with
  | [ "CPU" ] -> Some Host
  | [ "GPU" ] -> Some Gpu
  | [ "METAL" ] -> Some Metal
  | [ (("CUDA" | "NV" | "AMD" | "NV-PCI" | "AMD-PCI") as k) ] -> kind k 0
  | [ k; i ] -> Option.bind (index i) (kind k)
  | _ -> None

let of_string s =
  let rec go acc = function
    | [] -> Ok (List.rev acc)
    | n :: ns -> (
        match want_of_name n with
        | Some w -> go (w :: acc) ns
        | None ->
            Error
              (Printf.sprintf
                 "%S is not a device; devices are GPU, CPU, CPU:k (k >= 1), \
                  METAL, CUDA:i, NV:i, AMD:i, NV-PCI:i and AMD-PCI:i"
                 (String.trim n)))
  in
  go [] (String.split_on_char ',' s)
