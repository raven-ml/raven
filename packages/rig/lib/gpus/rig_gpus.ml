(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type gpu = {
  name : string;
  bus : string option;
  open_ : unit -> (Rig.t, string) result;
}

type path = { kind : string; gpus : unit -> (gpu list, string) result }

let gpu (o : _ Rig_edge.opener) bus =
  { name = o.name; bus; open_ = (fun () -> Rig.open_ o) }

(* A path's kind is its GPU 0's device's name, which making its opener reads
   without I/O. *)
let path (opener : int -> _ Rig_edge.opener) gpus =
  { kind = (opener 0).name; gpus }

let listed buses (opener : int -> _ Rig_edge.opener) =
  path opener (fun () ->
      Result.map (List.mapi (fun i b -> gpu (opener i) (Some b))) (buses ()))

let metal () =
  let gpus n = List.init n (fun i -> gpu (Rig_metal.gpu i) None) in
  Result.map gpus (Rig_metal.count ())

let paths ?firmware () =
  [
    path Rig_metal.gpu metal;
    listed Rig_cuda.buses Rig_cuda.gpu;
    listed Rig_nv_nvidia.buses Rig_nv_nvidia.gpu;
    listed Rig_amd_amdgpu.buses Rig_amd_amdgpu.gpu;
    listed Rig_nv_pci.buses (Rig_nv_pci.gpu ?firmware);
    listed Rig_amd_pci.buses (Rig_amd_pci.gpu ?firmware);
  ]

(* A refusal names what refused, unless its reason starts with that name, as
   the driver-less paths' do. *)
let named name why =
  if String.starts_with ~prefix:(name ^ ":") why then why
  else name ^ ": " ^ why

(* Each path is listed once the paths before it have opened no GPU. *)
let first ?firmware () =
  let rec gpus refused gs rest =
    match gs with
    | [] -> paths_from refused rest
    | g :: gs -> (
        match g.open_ () with
        | Ok d -> Ok d
        | Error why -> gpus (named g.name why :: refused) gs rest)
  and paths_from refused = function
    | [] when refused = [] -> Error "no path of rig sees a GPU here"
    | [] -> Error ("no GPU opens: " ^ String.concat "; " (List.rev refused))
    | p :: rest -> (
        match p.gpus () with
        | Ok gs -> gpus refused gs rest
        | Error why -> paths_from (named p.kind why :: refused) rest)
  in
  paths_from [] (paths ?firmware ())
