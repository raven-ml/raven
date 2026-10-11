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

(* Whether [block] has counter register [n]. A block's counter registers are
   numbered from 0, so the first it lacks is how many it has. *)
let has g block n =
  Option.is_some (Register.find g (strf "reg%s_PERFCOUNTER%d_LO" block n))

(* The first name [names] lists twice, if any. *)
let rec twice = function
  | [] -> None
  | n :: rest -> if List.mem n rest then Some n else twice rest

let layout g names =
  let table = table g in
  let registers = Hashtbl.create 4 in
  let taken block =
    Option.value ~default:0 (Hashtbl.find_opt registers block)
  in
  let rec go offset acc = function
    | [] -> Ok { counters = List.rev acc; bytes = offset }
    | name :: rest -> (
        match Array.find_opt (fun (n, _, _) -> n = name) table with
        | None -> Error (strf "%s counts no %s" (Gpu.processor g) name)
        | Some (_, block, _) when not (has g block (taken block)) ->
            Error
              (strf "%s counts at most %d %s counters at once" (Gpu.processor g)
                 (taken block) block)
        | Some (_, block, event) ->
            let register = taken block in
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
  match twice names with
  | Some n -> Error (strf "%s is listed twice" n)
  | None -> go 0 [] names
