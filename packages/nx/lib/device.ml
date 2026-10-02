(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A device is a memory, which nx.device opens, and the backend that computes
   eagerly on it, if any. Several devices may share one memory: their values
   share storage, and only who computes on them differs. *)
type t = {
  memory : Nx_device.t;
  backend : (module Nx_backend.S) option;
  name : string; (* the memory's, then the backend's when not its default *)
}

(* Pairing

   There is one device per memory and backend, so devices compare physically:
   [of_memory] and [with_backend] find a pair in [paired] before they make it. A
   memory's default backend is a fixed match over its kind: nx.cpu on the
   memories the host computes on (the host and test memories), none on the
   others. A backend is known by its name, which a device's name carries: a
   module of a name already paired with a memory is that pair's backend. *)

module Memories = Map.Make (struct
  type t = Nx_device.t

  let compare = Nx_device.compare
end)

let default_backend m : (module Nx_backend.S) option =
  if Nx_device.runs_on_host m then Some (module Nx_cpu) else None

let backend_name (module K : Nx_backend.S) = K.name

(* The devices over each memory, the default one first. *)
let paired : t list Memories.t Atomic.t = Atomic.make Memories.empty
let paired_lock = Mutex.create ()

(* [over m] is the devices over [m], made with the default one if there are
   none. Called with [paired_lock] held. *)
let over m =
  match Memories.find_opt m (Atomic.get paired) with
  | Some ds -> ds
  | None ->
      let d =
        { memory = m; backend = default_backend m; name = Nx_device.name m }
      in
      Atomic.set paired (Memories.add m [ d ] (Atomic.get paired));
      [ d ]

let of_memory m =
  match Memories.find_opt m (Atomic.get paired) with
  | Some (d :: _) -> d
  | Some [] | None -> List.hd (Mutex.protect paired_lock (fun () -> over m))

let with_backend k d =
  let m = d.memory and name = backend_name k in
  let (module K) = k in
  if not (K.runs_on m) then
    invalid_arg
      (Printf.sprintf "Nx.Device.with_backend: %s does not compute on %s" name
         (Nx_device.name m));
  Mutex.protect paired_lock @@ fun () ->
  let ds = over m in
  let named d' =
    match d'.backend with Some k' -> backend_name k' = name | None -> false
  in
  match List.find_opt named ds with
  | Some d' -> d'
  | None ->
      let d' =
        { memory = m; backend = Some k; name = Nx_device.name m ^ "/" ^ name }
      in
      Atomic.set paired (Memories.add m (ds @ [ d' ]) (Atomic.get paired));
      d'

let host = of_memory Nx_device.host
let memory d = d.memory
let name d = d.name
let equal (d : t) d' = d == d'
let pp ppf d = Format.pp_print_string ppf d.name

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

(* Test memories: ["CPU:k"] holds its values in the host's memory, which it maps
   as it is, and loads no programs, so the host computes on it. Each [k] is one
   memory, made by its first use. *)

let tests = Hashtbl.create 4
let tests_lock = Mutex.create ()

let test k =
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
      Ok (test k)
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

let get w = Result.map of_memory (memory_of w)
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
