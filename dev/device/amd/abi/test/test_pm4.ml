(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* PM4 packets over integers, against the words of soc15d.h (GFX9), nvd.h (GFX10
   on) and PAL's WAIT_REG_MEM64. *)

open Windtrap
open Device_amd_abi

let gpu gc =
  {
    Gpu.target = gc;
    gc;
    sdma = (6, 0, 0);
    xccs = 1;
    shader_engines = 6;
    compute_units = 48;
    scratch_slots = 32;
  }

let gfx11 = gpu (11, 0, 0)
let gfx9 = gpu (9, 4, 3)

let words p =
  let s = Packet.encode Int64.of_int p in
  List.init
    (String.length s / 4)
    (fun i -> Int32.to_int (String.get_int32_le s (4 * i)) land 0xffff_ffff)

(* PACKET3(op, n): type 3, n + 1 words after the header. *)
let packet3 op n = (3 lsl 30) lor (n lsl 16) lor (op lsl 8)

let memory =
  group "memory and registers"
    [
      test "a write to a register, at one address" (fun () ->
          equal (list int)
            [ packet3 0x37 3; 1 lsl 16; 0x1234; 0; 0xabcd ]
            (words (Pm4.write_data (Register 0x1234) 0xabcd)));
      test "a confirmed write to memory" (fun () ->
          equal (list int)
            [ packet3 0x37 3; (1 lsl 20) lor (5 lsl 8); 0x40; 0x1; 9 ]
            (words (Pm4.write_data (Memory 0x1_0000_0040) 9)));
      test "an SH register is set from SH's start" (fun () ->
          equal (list int)
            [ packet3 0x76 1; 0x20c; 7 ]
            (words (Pm4.set_reg 0x2e0c [ W32 (Value 7) ])));
      test "a UCONFIG register is set from UCONFIG's start" (fun () ->
          equal (list int)
            [ packet3 0x79 1; 0x200; 7 ]
            (words (Pm4.set_reg 0xc200 [ W32 (Value 7) ])));
      cases ~name:(Printf.sprintf "0x%x") "a register no packet sets is refused"
        [ 0x2bff; 0x3000; 0xbfff ] (fun reg ->
          raises_match (Exn.invalid_arg ~substring:"Pm4.set_reg") (fun () ->
              Pm4.set_reg reg [ Dword 0 ]));
      test "a counter copied to memory through the L2" (fun () ->
          equal (list int)
            [ packet3 0x40 4; (2 lsl 8) lor 4; 0x99; 0; 0x8; 0 ]
            (words (Pm4.copy_data Posted (Counter 0x99) 8)));
      test "the clock copied to memory, 64 bits, confirmed" (fun () ->
          equal (list int)
            [
              packet3 0x40 4;
              9 lor (2 lsl 8) lor (1 lsl 16) lor (1 lsl 20);
              0;
              0;
              0x8;
              0;
            ]
            (words (Pm4.copy_data Confirmed Clock 8)));
    ]

let waits =
  group "waits"
    [
      test "a wait on a register" (fun () ->
          equal (list int)
            [ packet3 0x3c 5; 3; 0x99; 0; 4; 4; 0x20 ]
            (words
               (Pm4.wait gfx11 (Register 0x99) Equal 4 ~mask:4 ~interval:0x20 ())));
      test "a wait on memory, every bit, every 4 clocks" (fun () ->
          equal (list int)
            [ packet3 0x3c 5; (1 lsl 4) lor 5; 0x8; 0; 1; 0xffff_ffff; 4 ]
            (words (Pm4.wait gfx11 (Memory 8) Greater_equal 1 ())));
      cases
        ~name:(fun (g, _) ->
          let a, b, c = g.Gpu.gc in
          Printf.sprintf "%d.%d.%d" a b c)
        "a wait on a UCONFIG register, from UCONFIG's start on GFX9"
        [ (gfx9, 0x8e8); (gfx11, 0xc8e8); (gpu (12, 0, 0), 0xc8e8) ]
        (fun (g, reg) ->
          equal int reg
            (List.nth
               (words (Pm4.wait g (Register 0xc8e8) Equal 0 ~mask:1 ()))
               2));
      test "a 64-bit wait compares every bit of both words" (fun () ->
          equal (list int)
            [
              packet3 0x93 7;
              (1 lsl 4) lor 5;
              0x10;
              0x2;
              7;
              1;
              0xffff_ffff;
              0xffff_ffff;
              4;
            ]
            (words
               (Pm4.wait_64 gfx11 0x2_0000_0010 Greater_equal 0x1_0000_0007 ())));
      test "GFX9 has no 64-bit wait" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"Pm4.wait_64") (fun () ->
              Pm4.wait_64 gfx9 0 Equal 0 ()));
    ]

(* nvd.h's RELEASE_MEM: CACHE_FLUSH_AND_INV_TS_EVENT at the end of the pipe, and
   the GCR bits of every cache. *)
let release_event = 0x14 lor (5 lsl 8)

let release_gcr =
  0x4000 lor 0x8000 lor 0x10_0000 lor 0x1000 lor 0x2000 lor 0x20_0000
  lor 0x40_0000

let caches =
  group "caches and signals"
    [
      test "a release with an interrupt" (fun () ->
          equal (list int)
            [
              packet3 0x49 6;
              release_event lor release_gcr;
              (2 lsl 29) lor (2 lsl 24);
              0x40;
              1;
              9;
              0;
              0x77;
            ]
            (words
               (Pm4.release_mem gfx11 System ~interrupt:0x77 0x1_0000_0040
                  (Data_64 9))));
      test "a release without an interrupt" (fun () ->
          equal (list int)
            [
              packet3 0x49 6;
              release_event lor release_gcr;
              1 lsl 29;
              0x40;
              0;
              9;
              0;
              0;
            ]
            (words (Pm4.release_mem gfx11 Agent 0x40 (Low_32 9))));
      test "a GFX9 release writes the L2 back" (fun () ->
          equal int
            (release_event lor 0x8000 lor 0x8_0000)
            (List.nth (words (Pm4.release_mem gfx9 System 0 (Low_32 0))) 1));
      cases
        ~name:(function Packet.Agent, _ -> "agent" | System, _ -> "system")
        "an acquire on GFX11 invalidates the L2 only for the system"
        [ (Packet.Agent, 0x3f0); (System, 0xc3f1) ]
        (fun (scope, cntl) ->
          equal (list int)
            [ packet3 0x58 6; 0; 0xffff_ffff; 0xffff_ffff; 0; 0; 0; cntl ]
            (words (Pm4.acquire_mem gfx11 scope)));
      test "a partial flush" (fun () ->
          equal (list int)
            [ packet3 0x46 0; 7 lor (4 lsl 8) ]
            (words (Pm4.event_write Cs_partial_flush)));
    ]

let control =
  group "control"
    [
      test "a predicated block counts its words" (fun () ->
          equal (list int)
            [ packet3 0x23 0; (0x5 lsl 24) lor 3; 1; 2; 3 ]
            (words (Pm4.pred_exec ~xcc_mask:0x5 [ Dword 1; Dword 2; Dword 3 ])));
      cases ~name:string_of_int "a die mask past 8 bits is refused"
        [ -1; 0x100 ] (fun xcc_mask ->
          raises_match (Exn.invalid_arg ~substring:"Pm4.pred_exec") (fun () ->
              Pm4.pred_exec ~xcc_mask []));
      test "a predicated block past 16383 words is refused" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"Pm4.pred_exec") (fun () ->
              Pm4.pred_exec ~xcc_mask:1
                (List.init 16384 (fun _ -> Packet.Dword 0))));
      test "an indirect buffer" (fun () ->
          equal (list int)
            [ packet3 0x3f 2; 0x100; 0x1; 16 lor (1 lsl 23) ]
            (words (Pm4.indirect_buffer 0x1_0000_0100 ~dwords:16)));
      cases ~name:string_of_int "an indirect buffer past IB_SIZE is refused"
        [ -1; 1 lsl 20 ]
        (fun dwords ->
          raises_match (Exn.invalid_arg ~substring:"Pm4.indirect_buffer")
            (fun () -> Pm4.indirect_buffer 0 ~dwords));
    ]

let kernel : Code_object.kernel =
  {
    descriptor = 0x1000;
    entry = 0x1100;
    group_segment = 1024;
    private_segment = 0;
    kernarg_size = 24;
    rsrc1 = 0x60af0000;
    rsrc2 = 0x1384;
    rsrc3 = 0;
    wave32 = true;
    dispatch_ptr = false;
    private_segment_buffer = false;
  }

let dispatch g =
  Pm4.dispatch g kernel ~program:0x1_0000_1100 ~scratch:0x2_0000_0000
    ~args:0x3_0000_0000 ~packet:0 ~threads:(64, 1, 1) ~groups:(2, 3, 4) ()

let runs =
  group "runs"
    [
      test "a dispatch ends in DISPATCH_DIRECT, wave32 at 0" (fun () ->
          let ws = words (dispatch gfx11) in
          equal (list int)
            [ packet3 0x15 3; 2; 3; 4; 0x8005 ]
            (List.filteri (fun i _ -> i >= List.length ws - 5) ws));
      test "a dispatch sets its program from bit 8" (fun () ->
          equal (list int)
            [ packet3 0x76 2; 0x20c; 0x0100_0011; 0 ]
            (List.filteri (fun i _ -> i < 4) (words (dispatch gfx11))));
      cases ~name:string_of_int "a wave limit outside 10 bits is refused"
        [ 0; 1024 ] (fun n ->
          raises_match (Exn.invalid_arg ~substring:"waves_per_array") (fun () ->
              Pm4.dispatch gfx11 kernel ~program:0 ~scratch:0 ~args:0 ~packet:0
                ~threads:(1, 1, 1) ~groups:(1, 1, 1) ~waves_per_array:n ()));
      test "a GC with no dispatch registers is refused" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"Pm4.dispatch") (fun () ->
              dispatch (gpu (10, 3, 0))));
      test "a run acquires, then flushes" (fun () ->
          equal (list int)
            (words (Pm4.acquire_mem gfx11 Agent)
            @ [ 9 ]
            @ words (Pm4.event_write Cs_partial_flush))
            (words (Pm4.run gfx11 [ Dword 9 ])));
    ]

let () =
  exit (run "device_amd_abi.pm4" [ memory; waits; caches; control; runs ])
