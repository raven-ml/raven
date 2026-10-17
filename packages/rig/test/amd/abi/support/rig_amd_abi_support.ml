(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Rig_amd_abi

let strf = Printf.sprintf
let timeout = 10.

(* GPUs *)

let gpu ?target ?(sdma = (6, 0, 0)) ?(xccs = 1) ?(shader_engines = 4)
    ?(compute_units = 32) ?(array_units = 4) ?(scratch_slots = 32) gc =
  let target = Option.value ~default:gc target in
  {
    Gpu.target;
    gc;
    sdma;
    xccs;
    shader_engines;
    compute_units;
    array_units;
    scratch_slots;
  }

let harvest_none (g : Gpu.t) = Array.make (g.shader_engines * g.xccs) [| 1 |]

let families = [ (9, 4, 3); (11, 0, 0); (11, 0, 3); (11, 5, 0); (12, 0, 0) ]
let version (a, b, c) = strf "%d.%d.%d" a b c

(* Words *)

let words s =
  List.init
    (String.length s / 4)
    (fun i -> Int32.to_int (String.get_int32_le s (4 * i)) land 0xffff_ffff)

let encode p = words (Packet.encode Int64.of_int p)

(* PM4 packets *)

let set_sh_reg = 0x76
let set_uconfig_reg = 0x79
let pred_exec = 0x23
let sh_start = 0x2c00
let uconfig_start = 0xc000

(* A header: type 3 in bits 30-31, the body's words less one from bit 16, the
   opcode from bit 8. *)
let rec packets = function
  | [] -> []
  | h :: rest ->
      if h lsr 30 <> 3 then failwith (strf "0x%08x starts no type 3 packet" h);
      let n = ((h lsr 16) land 0x3fff) + 1 in
      if List.length rest < n then
        failwith (strf "a packet of %d words passes the end" n);
      let body = List.filteri (fun i _ -> i < n) rest in
      ((h lsr 8) land 0xff, body)
      :: packets (List.filteri (fun i _ -> i >= n) rest)

let set start = function
  | [] -> []
  | off :: vs -> List.mapi (fun i v -> (start + off + i, v)) vs

let sets (op, body) =
  if op = set_sh_reg then set sh_start body
  else if op = set_uconfig_reg then set uconfig_start body
  else []

let writes ws = List.concat_map sets (packets ws)
