(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Launch descriptors against NVIDIA's clc7c0qmd.h (version 3) and clcec0qmd.h
   (version 5), through their structures. *)

open Windtrap
open Device_nv_abi

let gpu compute_class =
  {
    Gpu.compute_class;
    sass_version = 0x89;
    gpcs = 12;
    tpcs_per_gpc = 6;
    sms_per_tpc = 2;
    warps_per_sm = 48;
    shared_window = 0x7294_0000_0000;
    local_window = 0x7293_0000_0000;
  }

let kernel =
  {
    Cubin.code = 0x80;
    code_bytes = 0x100;
    registers = 32;
    shared_bytes = 0;
    stack_bytes = 0x20;
    params_offset = 0;
    banks = [];
  }

let ada = 0xc9c0
let blackwell = 0xcec0

let qmd cls =
  Qmd.make
    (require_ok ~pp:Format.pp_print_string (Launch.make (gpu cls) kernel))

(* The field [(lo, bits)] of [s]. *)
let field s (lo, bits) =
  let n = ref 0 in
  for i = (lo + bits - 1) / 8 downto lo / 8 do
    n := (!n lsl 8) lor Char.code s.[i]
  done;
  (!n lsr (lo mod 8)) land ((1 lsl bits) - 1)

let encode q = Structure.encode Int64.of_int (Qmd.structure q)

(* An address with bit 48 set, 256-byte aligned. *)
let high = (1 lsl 48) lor 0x12_3456_7800

let tests =
  group "descriptors"
    [
      test "64 KiB of local memory a thread fills its field" (fun () ->
          (* SHADER_LOCAL_MEMORY_HIGH_SIZE, MW(1623:1600). *)
          equal int 0x10000
            (field (encode (Qmd.set_local_memory 0x10000 (qmd ada))) (1600, 24)));
      test "a program address keeps bit 48" (fun () ->
          (* PROGRAM_ADDRESS_UPPER MW(1584:1568); version 5's
             PROGRAM_ADDRESS_UPPER_SHIFTED4 MW(1076:1056). *)
          equal ~msg:"version 3" int
            ((high lsr 32) land 0x1ffff)
            (field (encode (Qmd.set_program high (qmd ada))) (1568, 17));
          equal ~msg:"version 5" int
            ((high lsr 36) land 0x1fffff)
            (field (encode (Qmd.set_program high (qmd blackwell))) (1056, 21)));
      test "a hole keeps the fields it shares bytes with" (fun () ->
          (* PROGRAM_PREFETCH_SIZE MW(1649:1641) shares its bytes with
             PROGRAM_PREFETCH_ADDR_UPPER_SHIFTED MW(1640:1632). *)
          equal int 1
            (field (encode (Qmd.set_program high (qmd ada))) (1641, 9)));
      test "holes are sorted and apart" (fun () ->
          let s =
            Qmd.structure
              (qmd ada |> Qmd.patch_dim (Grid X) 7 |> Qmd.set_program high)
          in
          let ats = List.map (fun (h : int Structure.hole) -> h.at) s.holes in
          equal (list int) (List.sort_uniq Int.compare ats) ats);
      test "a size past its limit is refused" (fun () ->
          ignore (Qmd.set_dim (Block Z) 64 (qmd blackwell));
          raises_match (Exn.invalid_arg ~substring:"Qmd.set_dim") (fun () ->
              Qmd.set_dim (Block Z) 65 (qmd blackwell)));
      test "two releases, then none" (fun () ->
          let q = qmd ada in
          let q = require_some (Qmd.release System 0x1000 1 q) in
          let q = require_some (Qmd.release_stamp Agent 0x2000 2 q) in
          is_none (Qmd.release System 0x3000 3 q));
    ]

let () = exit (run "device_nv_abi.qmd" [ tests ])
