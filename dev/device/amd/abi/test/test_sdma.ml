(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* SDMA packets over integers, against the words of SDMA's packet headers
   (sdma_v6_0_0_pkt_open.h and its siblings). *)

open Windtrap
open Device_amd_abi

let gpu sdma =
  {
    Gpu.target = (11, 0, 0);
    gc = (11, 0, 0);
    sdma;
    xccs = 1;
    shader_engines = 4;
    compute_units = 32;
    scratch_slots = 32;
  }

let words p =
  let s = Packet.encode Int64.of_int p in
  List.init
    (String.length s / 4)
    (fun i -> Int32.to_int (String.get_int32_le s (4 * i)) land 0xffff_ffff)

let version (a, b, c) = Printf.sprintf "%d.%d.%d" a b c

let copy =
  group "copy"
    [
      test "a linear copy" (fun () ->
          equal (list int)
            [ 1; 0xff; 0; 0; 0xa; 0x2345_6780; 1 ]
            (words
               (Sdma.copy
                  (gpu (6, 0, 0))
                  ~dst:0x1_2345_6780 ~src:0xa_0000_0000 0x100)));
      test "a copy of no bytes is no packet" (fun () ->
          equal (list int) []
            (words (Sdma.copy (gpu (6, 0, 0)) ~dst:0 ~src:0 0)));
      cases
        ~name:(fun (v, _) -> version v)
        "a copy past the largest is two, the second at its offset"
        [
          ((4, 0, 0), 1 lsl 22);
          ((4, 4, 2), 1 lsl 30);
          ((5, 0, 0), 1 lsl 22);
          ((5, 2, 0), 1 lsl 30);
          ((7, 0, 0), 1 lsl 30);
        ]
        (fun (v, max) ->
          equal (list int)
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
            (words (Sdma.copy (gpu v) ~dst:0x200 ~src:0x100 (max + 5))));
      test "a negative copy is refused" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"Sdma.copy") (fun () ->
              Sdma.copy (gpu (6, 0, 0)) ~dst:0 ~src:0 (-1)));
    ]

let others =
  group "others"
    [
      test "a poll for equality" (fun () ->
          equal (list int)
            [
              8 lor (3 lsl 28) lor (1 lsl 31);
              0x10;
              0x2;
              5;
              0xff;
              4 lor (0xfff lsl 16);
            ]
            (words (Sdma.poll 0x2_0000_0010 Equal 5 ~mask:0xff ())));
      test "a poll for at least, on every bit" (fun () ->
          equal (list int)
            [
              8 lor (5 lsl 28) lor (1 lsl 31);
              0;
              0;
              0;
              0xffff_ffff;
              4 lor (0xfff lsl 16);
            ]
            (words (Sdma.poll 0 Greater_equal 0 ())));
      test "a fence on version 4 takes no memory type" (fun () ->
          equal (list int) [ 5; 0x8; 0; 7 ]
            (words (Sdma.fence (gpu (4, 4, 2)) 8 7)));
      test "a fence from version 5 writes uncached" (fun () ->
          equal (list int)
            [ 5 lor (3 lsl 16); 0x8; 0; 7 ]
            (words (Sdma.fence (gpu (5, 0, 0)) 8 7)));
      test "a trap" (fun () -> equal (list int) [ 6; 0 ] (words Sdma.trap));
      test "a global timestamp" (fun () ->
          equal (list int)
            [ 0xd lor (2 lsl 8); 0x18; 0 ]
            (words (Sdma.timestamp 0x18)));
    ]

let () = exit (run "device_amd_abi.sdma" [ copy; others ])
