(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The devices a command line names, as in ["cuda:0,metal,cpu"]: [cpu] (the
   host), [cpu:k] (a memory device that stands for another device on the host),
   and [metal], [cuda], [nv] or [amd] with an index, 0 when left out, in any
   case. Metal and CUDA open through the vendor's runtime, NV and AMD through
   the kernel driver. nx computes eagerly on the host and its memory devices; a
   set of GPUs computes under [Rune.jit] alone. *)

let gpu = function
  | "metal" -> Some (fun i -> Rig.open_ (Rig_metal.gpu i))
  | "cuda" -> Some (fun i -> Rig.open_ (Rig_cuda.gpu i))
  | "nv" -> Some (fun i -> Rig.open_ (Rig_nv_nvidia.gpu i))
  | "amd" -> Some (fun i -> Rig.open_ (Rig_amd_amdgpu.gpu i))
  | _ -> None

let index s =
  if s <> "" && String.for_all (fun c -> c >= '0' && c <= '9') s then
    int_of_string_opt s
  else None

(* [device name] is the device [name] names, opened now, or why it does not
   open. *)
let device name =
  let unknown = Error (Printf.sprintf "%S is not a device" name) in
  match
    String.split_on_char ':' (String.lowercase_ascii (String.trim name))
  with
  | [ "cpu" ] -> Ok Rig.host
  | [ "cpu"; k ] -> (
      match index k with
      | Some k when k >= 1 -> Rig.memory_device (Printf.sprintf "CPU:%d" k)
      | _ -> unknown)
  | [ vendor ] -> ( match gpu vendor with Some get -> get 0 | None -> unknown)
  | [ vendor; i ] -> (
      match (gpu vendor, index i) with
      | Some get, Some i -> get i
      | _ -> unknown)
  | _ -> unknown

let names s = String.split_on_char ',' s

(* [first s] is the first device [s] names that opens. *)
let first s =
  let rec go reasons = function
    | [] ->
        failwith ("no device opens: " ^ String.concat "; " (List.rev reasons))
    | name :: rest -> (
        match device name with Ok d -> d | Error e -> go (e :: reasons) rest)
  in
  go [] (names s)

(* [all s] is every device [s] names, opened. *)
let all s =
  List.map
    (fun name -> match device name with Ok d -> d | Error e -> failwith e)
    (names s)

(* [traffic d f] is [f ()] and the bytes that copies moved into and out of [d]'s
   memory while it ran, read from rig's profile. *)
let traffic d f =
  let y, events = Rig.Profile.take f in
  let count (i, o) = function
    | Rig.Profile.Copy { src; dst; bytes; _ } ->
        ( (if Rig.equal dst d then i + bytes else i),
          if Rig.equal src d then o + bytes else o )
    | _ -> (i, o)
  in
  (y, List.fold_left count (0, 0) events)
