(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The security processor's commands, against psp_gfx_if.h. *)

open Windtrap
open Rig_amd_pci

let u32 b off = Int32.to_int (String.get_int32_le b off) land 0xffff_ffff

(* [words b offs] is the 32-bit words of [b] at [offs]. *)
let words b offs = List.map (u32 b) offs

(* psp_gfx_if.h: a command buffer is 1024 bytes, its ID at 8, its command at 28,
   its response at 864 (status at +0, TMR size at +16); a frame is 64 bytes, the
   command's address at 0 and 4, the fence's at 12 and 16, its value at 20. *)
let security =
  group ~timeout:10. "security processor"
    [
      test "a firmware load names its address, size and type" (fun () ->
          let b = Psp.load_ip_fw ~at:0x12_3456_7000 ~bytes:0x800 ~fw_type:89 in
          equal int 1024 (String.length b);
          equal (list int)
            [ 6; 0x3456_7000; 0x12; 0x800; 89 ]
            (words b [ 8; 28; 32; 36; 40 ]));
      test "a table of contents names its address and size" (fun () ->
          let b = Psp.load_toc ~at:0x1000 ~bytes:0x40 in
          equal (list int) [ 0x20; 0x1000; 0; 0x40 ] (words b [ 8; 28; 32; 36 ]));
      test "a TMR names both its addresses, and says so" (fun () ->
          let b =
            Psp.setup_tmr ~at:0x2_0000_0000 ~fabric:0x3_0000_0000
              ~bytes:0x40_0000
          in
          equal (list int)
            [ 5; 0; 2; 0x40_0000; 2; 0; 3 ]
            (words b [ 8; 28; 32; 36; 40; 44; 48 ]));
      test "the bootloader loads every component the kernel loads, in order"
        (fun () ->
          (* amdgpu_psp.c psp_hw_start: KDB, SPL, SYS_DRV, SOC_DRV, INTF_DRV,
             DBG_DRV, RAS_DRV, IPKEYMGR_DRV, SPDM_DRV, then SOS; the types of
             amdgpu_ucode.h and the commands of amdgpu_psp.h. *)
          equal
            (list (pair int int))
            [
              (3, 0x80000);
              (5, 0x10000000);
              (2, 0x10000);
              (7, 0xb0000);
              (8, 0xd0000);
              (9, 0xc0000);
              (10, 0xe0000);
              (11, 0xf0000);
              (12, 0x20000000);
              (1, 0x20000);
            ]
            Psp.steps);
      test "the RLC's autoload and a partition carry their IDs" (fun () ->
          equal (list int) [ 0x21 ] (words Psp.autoload_rlc [ 8 ]);
          equal (list int) [ 0x27; 1 ] (words (Psp.partition ~mode:1) [ 8; 28 ]));
      test "a frame names its command, its fence and the fence's value"
        (fun () ->
          let f = Psp.frame ~command:0x1_0000_2000 ~fence:0x3000 ~value:7 in
          equal int 64 (String.length f);
          equal (list int)
            [ 0x2000; 1; 0x3000; 0; 7 ]
            (words f [ 0; 4; 12; 16; 20 ]));
      test "an answer's status and TMR size are read at the response" (fun () ->
          let b = Bytes.make 1024 '\000' in
          Bytes.set_int32_le b 864 0x1234l;
          Bytes.set_int32_le b 880 0x50_0000l;
          let b = Bytes.to_string b in
          equal (pair int int) (0x1234, 0x50_0000)
            (Psp.status b, Psp.tmr_bytes b));
    ]

let () = exit (run "rig_amd_pci.psp" [ security ])
