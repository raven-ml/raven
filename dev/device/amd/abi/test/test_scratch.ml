(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Scratch as tinygrad sizes it: 48 compute units of 32 slots and 6 engines on
   GFX11, at least 128 bytes a lane in 256-byte wave units; 38 of 32, 4 engines
   and 8 dies on GFX9, in 1024-byte units. The descriptor's words follow ROCr's
   registers.h layouts. *)

open Windtrap
open Device_amd_abi

let gfx11 =
  {
    Gpu.target = (11, 0, 0);
    gc = (11, 0, 0);
    sdma = (6, 0, 0);
    xccs = 1;
    shader_engines = 6;
    compute_units = 48;
    scratch_slots = 32;
  }

let gfx9 =
  {
    Gpu.target = (9, 4, 2);
    gc = (9, 4, 3);
    sdma = (4, 4, 2);
    xccs = 8;
    shader_engines = 4;
    compute_units = 38;
    scratch_slots = 32;
  }

let words s =
  List.init
    (String.length s / 4)
    (fun i -> Int32.to_int (String.get_int32_le s (4 * i)) land 0xffff_ffff)

let sizes =
  group "sizes"
    [
      test "a GFX11 kernel of 0 bytes a lane takes 128" (fun () ->
          equal int (128 * 64 * 32 * 48) (Scratch.size gfx11 0));
      test "a GFX9 lane rounds up to 16 bytes, on every die" (fun () ->
          equal int (272 * 64 * 32 * 38 * 8) (Scratch.size gfx9 260));
      test "the scratch ring of GFX11 kernels of 0 bytes a lane" (fun () ->
          equal int (256 lor (32 lsl 12)) (Scratch.tmpring gfx11 0));
      test "the scratch ring of GFX9 kernels of 256 bytes a lane" (fun () ->
          equal int (1216 lor (16 lsl 12)) (Scratch.tmpring gfx9 256));
    ]

(* Words 3's selects of x, y, z and w, and its added thread id. *)
let selects = 4 lor (5 lsl 3) lor (6 lsl 6) lor (7 lsl 9) lor (1 lsl 23)

let descriptor =
  group "descriptor"
    [
      test "a GFX11 descriptor" (fun () ->
          equal (list int)
            [
              0x3456_7800;
              0x12 lor (1 lsl 30);
              0x10_0000;
              selects lor (0x14 lsl 12) lor (2 lsl 28);
            ]
            (words (Scratch.descriptor gfx11 ~base:0x12_3456_7800 0x10_0000)));
      test "a GFX9 descriptor splits the buffer among the dies" (fun () ->
          equal (list int)
            [
              0x3456_7800;
              0x12 lor (1 lsl 31);
              0x2_0000;
              selects lor (4 lsl 12) lor (4 lsl 15) lor (1 lsl 19) lor (3 lsl 21);
            ]
            (words (Scratch.descriptor gfx9 ~base:0x12_3456_7800 0x10_0000)));
      test "a GC with no layout is refused" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"Scratch.descriptor")
            (fun () ->
              Scratch.descriptor { gfx11 with gc = (10, 3, 0) } ~base:0 0));
    ]

let () = exit (run "device_amd_abi.scratch" [ sizes; descriptor ])
