(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Rig_amd_abi

let strf = Printf.sprintf
let timeout = 10.

(* GPUs *)

let gpu ?target ?(sdma = (6, 0, 0)) ?(xccs = 1) ?(shader_engines = 4)
    ?(compute_units = 32) ?(scratch_slots = 32) gc =
  let target = Option.value ~default:gc target in
  { Gpu.target; gc; sdma; xccs; shader_engines; compute_units; scratch_slots }

let families = [ (9, 4, 3); (11, 0, 0); (11, 0, 3); (11, 5, 0); (12, 0, 0) ]
let version (a, b, c) = strf "%d.%d.%d" a b c

(* Words *)

let words s =
  List.init
    (String.length s / 4)
    (fun i -> Int32.to_int (String.get_int32_le s (4 * i)) land 0xffff_ffff)

let encode p = words (Rig_packet.encode Int64.of_int p)

(* PM4 packets *)

(* The opcodes of soc15d.h and nvd.h the encoders write. *)
let names =
  [
    (0x10, "NOP");
    (0x15, "DISPATCH_DIRECT");
    (0x23, "PRED_EXEC");
    (0x37, "WRITE_DATA");
    (0x3c, "WAIT_REG_MEM");
    (0x3f, "INDIRECT_BUFFER");
    (0x40, "COPY_DATA");
    (0x46, "EVENT_WRITE");
    (0x49, "RELEASE_MEM");
    (0x58, "ACQUIRE_MEM");
    (0x76, "SET_SH_REG");
    (0x79, "SET_UCONFIG_REG");
    (0x93, "WAIT_REG_MEM64");
  ]

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

let name g a =
  let at (r : Register.t) =
    match Register.address g r with
    | a' -> a' = a
    | exception Invalid_argument _ -> false
  in
  match List.find_opt at (Register.registers g) with
  | Some r -> r.name
  | None -> strf "0x%x" a

let set start = function
  | [] -> []
  | off :: vs -> List.mapi (fun i v -> (start + off + i, v)) vs

let sets (op, body) =
  if op = set_sh_reg then set sh_start body
  else if op = set_uconfig_reg then set uconfig_start body
  else []

let writes ws = List.concat_map sets (packets ws)

let pm4 g ws =
  let line (op, body) =
    let op_name =
      Option.value ~default:(strf "0x%02x" op) (List.assoc_opt op names)
    in
    let body =
      match sets (op, body) with
      | [] -> List.map (strf "0x%x") body
      | fs -> List.map (fun (a, v) -> strf "%s=0x%x" (name g a) v) fs
    in
    String.concat " " (op_name :: body)
  in
  String.concat "\n" (List.map line (packets ws))
