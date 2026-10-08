(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

type t = {
  name : string;
  block : string;
  event : int;
  register : int;
  instances : int;
  engines : int;
  arrays : int;
  wgps : int;
  offset : int;
}

type layout = { counters : t list; bytes : int }

(* The table rocprofiler defines for the GPU's processor: GFX11 and GFX12 GPUs
   share theirs. *)
let table (g : Gpu.t) =
  match g.target with
  | 9, 4, 2 -> Defs.counters "gfx942"
  | 9, 5, 0 -> Defs.counters "gfx950"
  | 11, _, _ -> Defs.counters "gfx11"
  | 12, _, _ -> Defs.counters "gfx12"
  | _ -> [||]

let names g = Array.to_list (Array.map (fun (n, _, _) -> n) (table g))

(* GFX10 on: two shader arrays per engine, two compute units per work-group
   processor. *)
let arrays_per_engine = 2
let units_per_wgp = 2

(* Where a block's counter runs: (instances, engines, arrays, work-group
   processors). The SQ counts per engine on GFX9, and per work-group processor
   of each array after. *)
let places (g : Gpu.t) = function
  | "GRBM" -> (1, 1, 1, 1)
  | "GL2C" -> (32, 1, 1, 1)
  | "TCC" -> (16, 1, 1, 1)
  | _ -> (
      match g.target with
      | 9, _, _ -> (1, g.shader_engines, 1, 1)
      | _ ->
          let per_array =
            g.compute_units / (g.shader_engines * arrays_per_engine)
          in
          (1, g.shader_engines, arrays_per_engine, per_array / units_per_wgp))

let layout g names =
  let table = table g in
  let registers = Hashtbl.create 4 in
  let rec go offset acc = function
    | [] -> Ok { counters = List.rev acc; bytes = offset }
    | name :: rest -> (
        match Array.find_opt (fun (n, _, _) -> n = name) table with
        | None -> Error (strf "%s counts no %s" (Gpu.processor g) name)
        | Some (_, block, event) ->
            let register =
              Option.value ~default:0 (Hashtbl.find_opt registers block)
            in
            Hashtbl.replace registers block (register + 1);
            let instances, engines, arrays, wgps = places g block in
            let c =
              {
                name;
                block;
                event;
                register;
                instances;
                engines;
                arrays;
                wgps;
                offset;
              }
            in
            let n = g.xccs * instances * engines * arrays * wgps in
            go (offset + (8 * n)) (c :: acc) rest)
  in
  go 0 [] names
