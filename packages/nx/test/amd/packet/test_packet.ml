(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Packets over integers, against the words the headers define: SDMA's packet
   headers (sdma_v6_0_0_pkt_open.h and its siblings), PM4's (soc15d.h, nvd.h),
   and the GC's register addresses in each generation's PM4 register space. The
   symbolic encoding's agreement with these is tolk's suite. *)

open Windtrap
module P = Nx_amd_packet

let words = list int

(* PACKET3(op, n): type 3, n + 1 words after the header. *)
let packet3 op n = (3 lsl 30) lor (n lsl 16) lor (op lsl 8)

(* A caller's value as a word's term, as it is. *)
let t value = P.Value value

let dwords =
  group "dwords"
    [
      test "a 64-bit value is its low word, then its high word" (fun () ->
          equal words [ 2; 1 ] (P.dwords [ W64 (t 0x1_0000_0002) ]));
      test "a 32-bit word is the low 32 bits" (fun () ->
          equal words [ 0xffff_ffff; 0x2 ]
            (P.dwords [ Dword 0x1_ffff_ffff; W32 (t 0x7_0000_0002) ]));
      test "a term adds, then shifts" (fun () ->
          equal words [ 0x3000_0000; 0 ]
            (P.dwords [ W64 (Shift (Add (Value 0x2ff_ffff_ff00, 0x100L), 12)) ]));
    ]

let sdma =
  group "Sdma"
    [
      test "a linear copy" (fun () ->
          equal words
            [ 1; 0xff; 0; 0; 0xa; 0x2345_6780; 1 ]
            (P.dwords
               (P.Sdma.copy ~sdma:(6, 0, 0) ~dst:0x1_2345_6780
                  ~src:0xa_0000_0000 0x100)));
      test "a copy of no bytes is no packet" (fun () ->
          equal words []
            (P.dwords (P.Sdma.copy ~sdma:(6, 0, 0) ~dst:0 ~src:0 0)));
      cases
        ~name:(fun ((a, b, c), _) -> Printf.sprintf "%d.%d.%d" a b c)
        "a copy past the largest is two, the second at its offset"
        [
          ((4, 0, 0), 1 lsl 22);
          ((4, 4, 2), 1 lsl 30);
          ((5, 0, 0), 1 lsl 22);
          ((5, 2, 0), 1 lsl 30);
          ((7, 0, 0), 1 lsl 30);
        ]
        (fun (sdma, max) ->
          equal words
            [
              1;
              max - 1;
              0;
              0x100;
              0;
              0x200;
              0;
              1;
              4;
              0;
              0x100 + max;
              0;
              0x200 + max;
              0;
            ]
            (P.dwords (P.Sdma.copy ~sdma ~dst:0x200 ~src:0x100 (max + 5))));
      test "a negative copy is refused" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"Sdma.copy") (fun () ->
              P.Sdma.copy ~sdma:(6, 0, 0) ~dst:0 ~src:0 (-1)));
      test "a poll for equality" (fun () ->
          equal words
            [
              8 lor (3 lsl 28) lor (1 lsl 31);
              0x10;
              0x2;
              5;
              0xffff_ffff;
              4 lor (0xfff lsl 16);
            ]
            (P.dwords (P.Sdma.poll 0x2_0000_0010 Equal 5 ~mask:0xffff_ffff)));
      test "a poll for at least" (fun () ->
          equal int
            (8 lor (5 lsl 28) lor (1 lsl 31))
            (List.hd (P.dwords (P.Sdma.poll 0 Greater_equal 0 ~mask:0))));
      test "a fence on version 4 takes no memory type" (fun () ->
          equal words [ 5; 0x8; 0; 7 ]
            (P.dwords (P.Sdma.fence ~sdma:(4, 4, 2) 8 7)));
      test "a fence from version 5 writes uncached" (fun () ->
          equal words
            [ 5 lor (3 lsl 16); 0x8; 0; 7 ]
            (P.dwords (P.Sdma.fence ~sdma:(5, 0, 0) 8 7)));
      test "a trap" (fun () -> equal words [ 6; 0 ] (P.dwords P.Sdma.trap));
      test "a global timestamp" (fun () ->
          equal words
            [ 0xd lor (2 lsl 8); 0x18; 0 ]
            (P.dwords (P.Sdma.timestamp 0x18)));
    ]

let pm4 =
  group "Pm4"
    [
      test "a write to a register, at one address" (fun () ->
          equal words
            [ packet3 0x37 3; 1 lsl 16; 0x1234; 0; 0xabcd ]
            (P.dwords (P.Pm4.write_data (Register 0x1234) 0xabcd)));
      test "a confirmed write to memory" (fun () ->
          equal words
            [ packet3 0x37 3; (1 lsl 20) lor (5 lsl 8); 0x40; 0x1; 9 ]
            (P.dwords (P.Pm4.write_data (Memory 0x1_0000_0040) 9)));
      test "a wait on a register" (fun () ->
          equal words
            [ packet3 0x3c 5; 3; 0x99; 0; 4; 4; 0x20 ]
            (P.dwords
               (P.Pm4.wait ~gc:(11, 0, 0) (Register 0x99) Equal 4 ~mask:4
                  ~interval:0x20)));
      test "a wait on memory" (fun () ->
          equal words
            [ packet3 0x3c 5; (1 lsl 4) lor 5; 0x8; 0; 1; 0xffff_ffff; 4 ]
            (P.dwords
               (P.Pm4.wait ~gc:(11, 0, 0) (Memory 8) Greater_equal 1
                  ~mask:0xffff_ffff ~interval:4)));
      cases
        ~name:(fun ((a, b, c), _) -> Printf.sprintf "%d.%d.%d" a b c)
        "a wait on a UCONFIG register, from UCONFIG's start on GFX9"
        [ ((9, 4, 3), 0x8e8); ((11, 0, 0), 0xc8e8); ((12, 0, 0), 0xc8e8) ]
        (fun (gc, reg) ->
          equal int reg
            (List.nth
               (P.dwords
                  (P.Pm4.wait ~gc (Register 0xc8e8) Equal 0 ~mask:1 ~interval:4))
               2));
      test "an SH register is set from SH's start" (fun () ->
          equal words
            [ packet3 0x76 1; 0x20c; 7 ]
            (P.dwords (P.Pm4.set_reg 0x2e0c [ W32 (t 7) ])));
      test "a UCONFIG register is set from UCONFIG's start" (fun () ->
          equal words
            [ packet3 0x79 1; 0x200; 7 ]
            (P.dwords (P.Pm4.set_reg 0xc200 [ W32 (t 7) ])));
      test "the program's address, from its bit 8" (fun () ->
          equal words
            [ packet3 0x76 2; 0x20c; 0x0123_4567; 0 ]
            (P.dwords (P.Pm4.set_program ~gc:(11, 0, 0) 0x1_2345_6700)));
      test "the scratch's address, from its bit 8" (fun () ->
          equal words
            [ packet3 0x76 2; 0x210; 0x0000_0100; 0x1 ]
            (P.dwords (P.Pm4.set_scratch ~gc:(9, 4, 3) 0x1_0000_0100_00)));
      cases ~name:(Printf.sprintf "0x%x") "a register no packet sets is refused"
        [ 0x2bff; 0x3000; 0xbfff ] (fun reg ->
          raises_match (Exn.invalid_arg ~substring:"set_reg") (fun () ->
              P.Pm4.set_reg reg [ Dword 0 ]));
      test "a predicated block on the dies of a mask" (fun () ->
          equal words
            [ packet3 0x23 0; (0x5 lsl 24) lor 12 ]
            (P.dwords (P.Pm4.pred_exec ~xcc_mask:0x5 ~dwords:12)));
      cases
        ~name:(fun ((a, b, c), w, _) ->
          Printf.sprintf "%d.%d.%d, %s" a b c
            (match w with P.Pm4.Wave32 -> "wave32" | Wave64 -> "wave64"))
        "a dispatch starts the compute shader at 0, in the wave it says"
        [
          ((11, 0, 0), P.Pm4.Wave32, 0x8005);
          ((12, 0, 0), Wave64, 0x5);
          ((9, 4, 3), Wave32, 0x5);
        ]
        (fun (gc, wave, init) ->
          equal words
            [ packet3 0x15 3; 2; 3; 4; init ]
            (P.dwords (P.Pm4.dispatch_direct ~gc wave (2, 3, 4))));
      cases
        ~name:(function P.Pm4.Posted -> "posted" | Confirmed -> "confirmed")
        "a counter copied to memory through the L2"
        [ P.Pm4.Posted; Confirmed ]
        (fun w ->
          let confirm =
            match w with P.Pm4.Posted -> 0 | Confirmed -> 1 lsl 20
          in
          equal words
            [ packet3 0x40 4; confirm lor (2 lsl 8) lor 4; 0x99; 0; 0x8; 0 ]
            (P.dwords (P.Pm4.copy_data w (Counter 0x99) 8)));
      test "the clock copied to memory, 64 bits, confirmed" (fun () ->
          equal words
            [
              packet3 0x40 4;
              9 lor (2 lsl 8) lor (1 lsl 16) lor (1 lsl 20);
              0;
              0;
              0x8;
              0;
            ]
            (P.dwords (P.Pm4.copy_data Confirmed Clock 8)));
      test "an indirect buffer" (fun () ->
          equal words
            [ packet3 0x3f 2; 0x100; 0x1; 16 lor (1 lsl 23) ]
            (P.dwords (P.Pm4.indirect_buffer 0x1_0000_0100 ~dwords:16)));
    ]

let gc =
  group "Gc"
    [
      cases
        ~name:(fun ((a, b, c), _) -> Printf.sprintf "%d.%d.%d" a b c)
        "a GC takes the registers of its family"
        [
          ((9, 4, 4), Some (9, 4, 3));
          ((11, 0, 2), Some (11, 0, 0));
          ((11, 0, 3), Some (11, 0, 3));
          ((12, 0, 1), Some (12, 0, 0));
          ((10, 3, 0), None);
          ((9, 4, 2), None);
        ]
        (fun (v, f) -> equal (option (triple int int int)) f (P.Gc.family v));
      (* The addresses tinygrad's register modules give these registers. *)
      cases
        ~name:(fun (((a, b, c), n), _) ->
          Printf.sprintf "%s of %d.%d.%d" n a b c)
        "a register's address"
        [
          (((11, 0, 0), "regCOMPUTE_PGM_LO"), 0x2e0c);
          (((12, 0, 0), "regCOMPUTE_PGM_LO"), 0x2e0c);
          (((9, 4, 3), "regCOMPUTE_PGM_LO"), 0x2e0c);
          (((11, 0, 0), "regCOMPUTE_PGM_RSRC3"), 0x2e28);
          (((9, 4, 3), "regCOMPUTE_PGM_RSRC3"), 0x2e2d);
          (((9, 4, 3), "regGRBM_GFX_INDEX"), 0xc200);
          (((11, 5, 0), "regGRBM_GFX_INDEX"), 0xc200);
        ]
        (fun ((v, name), addr) ->
          let r = require_some (P.Gc.find v name) in
          equal int addr (P.Gc.address v r));
      test "a GC of no family has no registers" (fun () ->
          equal int 0 (List.length (P.Gc.registers (10, 3, 0))));
      test "a field's value is cut to its width" (fun () ->
          let r = require_some (P.Gc.find (11, 0, 0) "regGRBM_GFX_INDEX") in
          equal int
            ((0xff lsl 16) lor (1 lsl 31))
            (P.Gc.encode r [ ("se_index", 0x1ff); ("se_broadcast_writes", 1) ]));
      (* As tinygrad sizes scratch: 48 compute units of 32 slots, 6 engines, at
         least 128 bytes a lane in 256-byte waves units; 38 of 32, 8 dies, 256
         bytes a lane in 1024-byte units on GFX9. *)
      test "the scratch ring of GFX11 kernels of 0 bytes a lane" (fun () ->
          equal int
            (256 lor (32 lsl 12))
            (P.Gc.tmpring_size ~gc:(11, 0, 0) ~compute_units:48 ~slots:32
               ~shader_engines:6 ~xccs:1 0));
      test "the scratch ring of GFX9 kernels of 256 bytes a lane" (fun () ->
          equal int
            (1216 lor (16 lsl 12))
            (P.Gc.tmpring_size ~gc:(9, 4, 3) ~compute_units:38 ~slots:32
               ~shader_engines:4 ~xccs:8 256));
      test "a field the register lacks is refused" (fun () ->
          let r = require_some (P.Gc.find (11, 0, 0) "regGRBM_GFX_INDEX") in
          raises_match (Exn.invalid_arg ~substring:"no field") (fun () ->
              P.Gc.encode r [ ("nope", 1) ]));
    ]

let () = exit (run "nx.amd.packet" [ dwords; sdma; pm4; gc ])
