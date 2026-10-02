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
