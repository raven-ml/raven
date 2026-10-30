(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Compute queues' descriptors, against v9_structs.h, v11_structs.h and
   v12_structs.h, and the shader array index, against the GC's register
   headers. *)

open Windtrap
open Rig_amd_pci
module S = Rig_amd_abi_support

let u32 b off = Int32.to_int (String.get_int32_le b off) land 0xffff_ffff

(* [words b offs] is the 32-bit words of [b] at [offs]. *)
let words b offs = List.map (u32 b) offs

(* The R9700, of GC 12.0.1, and an MI300 of GC 9.4.3 and eight dies. *)
let r9700 = S.gpu ~sdma:(7, 0, 1) ~compute_units:64 (12, 0, 1)
let mi300 = S.gpu ~target:(9, 4, 2) ~sdma:(4, 4, 2) ~xccs:8 (9, 4, 3)

let gfx12_queue =
  {
    Gfx.ring = 0x1_0000_0000;
    ring_bytes = 16 lsl 20;
    read = 0x3_0000;
    write = 0x3_0008;
    eop = 0x2000_0000;
    eop_bytes = 0x1000;
    doorbell = 3;
  }

(* v12_structs.h and v9_structs.h give each field's word; gc_12_0_0_sh_mask.h
   each register field's first bit. *)
let queues =
  group ~timeout:10. "compute queues"
    [
      cases
        "a shader array selection and a broadcast hold the bits GRBM_GFX_INDEX \
         states"
        ~id:fst
        (* The SE index at bit 16, the SH (GC 9) or SA index at bit 8, and the
           SH or SA, instance and SE broadcasts at bits 29, 30 and 31
           (gc_9_4_3_sh_mask.h, gc_11_0_0_sh_mask.h, gc_12_0_0_sh_mask.h). *)
        [
          ("GC 12.0.1", (12, 0, 1));
          ("GC 11.0.0", (11, 0, 0));
          ("GC 9.4.3", (9, 4, 3));
        ]
        (fun (_, gc) ->
          let g = S.gpu gc in
          equal int
            ((2 lsl 16) lor (1 lsl 8) lor (1 lsl 30))
            (Gfx.index g (`Array (2, 1)));
          equal int
            ((1 lsl 29) lor (1 lsl 30) lor (1 lsl 31))
            (Gfx.index g `All));
      test "a GC 12 queue's descriptor holds its ring, pointers and doorbell"
        (fun () ->
          let d =
            Gfx.mqd r9700 gfx12_queue ~base:0x8_0000_1000 `Pm4 ~xcc:0
          in
          equal int 2048 (String.length d);
          equal (list int)
            [
              0xc031_0800;
              0xffff_ffff;
              0x1000;
              8;
              0x5501;
              0x100_0000;
              0;
              0x3_0000;
              0x3_0008;
              0x4000_0018;
              0x515;
              0x30_0000;
              0x100;
              0x20_0000;
              9;
              0;
            ]
            (words d
               (List.map
                  (fun w -> 4 * w)
                  [
                    0;
                    23;
                    128;
                    129;
                    132;
                    136;
                    137;
                    139;
                    141;
                    143;
                    145;
                    149;
                    162;
                    165;
                    167;
                    181;
                  ])));
      test "a KIQ's descriptor is privileged and the kernel driver's" (fun () ->
          let d = Gfx.mqd r9700 gfx12_queue ~base:0 `Kiq ~xcc:0 in
          equal int 0x3 (u32 d (4 * 145) lsr 30));
      test
        "an AQL queue across dies holds its die and chunk over the thread masks"
        (fun () ->
          let d = Gfx.mqd mi300 gfx12_queue ~base:0 `Aql ~xcc:2 in
          equal (list int)
            [
              0xffff_ffff;
              0xffff_ffff;
              0xffff_ffff;
              0xffff_ffff;
              2;
              1;
              0x1000;
              1;
            ]
            (words d
               (List.map (fun w -> 4 * w) [ 23; 24; 26; 27; 39; 41; 226; 181 ])));
      cases "a GC of no queue descriptor layout is refused" ~id:S.version
        [ (10, 3, 0); (8, 0, 0); (13, 0, 0) ]
        (fun gc ->
          raises_match (Exn.invalid_arg ~sub:"Gfx.mqd") (fun () ->
              Gfx.mqd (S.gpu gc) gfx12_queue ~base:0 `Pm4 ~xcc:0));
    ]

let () = exit (run "rig_amd_pci.gfx" [ queues ])
