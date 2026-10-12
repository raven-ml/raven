(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A device is a memory, which rig opens, and the backend that computes
   eagerly on it, if any. Several devices may share one memory: their values
   share storage, and only who computes on them differs. Devices are plain
   values, equal when their memories and backends are. *)
type t = { memory : Rig.t; backend : Nx_backend.t option }

(* A memory's default backend is a fixed match over its kind: nx.cpu on the
   memories the host computes on (the host and test memories), none on the
   others. *)
let default_backend m =
  if Rig.runs_on_host m then Some Nx_cpu.backend else None

let check_runs_on what b m =
  let (module K) = Nx_backend.kernels b in
  if not (K.runs_on m) then
    invalid_arg
      (Printf.sprintf "Nx.Device.%s: %s does not compute on %s" what K.name
         (Rig.name m))

let make ?backend memory =
  match backend with
  | None -> { memory; backend = default_backend memory }
  | Some b ->
      check_runs_on "make" b memory;
      { memory; backend }

let with_backend b d =
  check_runs_on "with_backend" b d.memory;
  { d with backend = Some b }

let host = make Rig.host
let memory d = d.memory

let equal d d' =
  Rig.equal d.memory d'.memory
  && Option.equal Nx_backend.equal d.backend d'.backend

(* A device's name is its memory's, then its backend's when the backend is not
   the memory's own, as in ["CPU:1/nx-oxcaml"]. *)
let name d =
  let m = Rig.name d.memory in
  match d.backend with
  | Some b ->
      let (module K) = Nx_backend.kernels b in
      if K.owns d.memory then m else m ^ "/" ^ K.name
  | None -> m

let pp ppf d = Format.pp_print_string ppf (name d)

(* Test memories: ["CPU:k"] is a memory device, whose memory the host
   addresses and computes on. Once it is lost, rig opens the name again as a
   fresh memory. *)

let cpu k =
  if k < 1 then invalid_arg (Printf.sprintf "Nx.Device.cpu: %d < 1" k);
  make (Rig.memory_device (Printf.sprintf "CPU:%d" k))
