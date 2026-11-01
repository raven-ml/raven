(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* SDMA packets over integers, against the words of SDMA's packet headers
   (sdma_v6_0_0_pkt_open.h and its siblings). *)

open Windtrap
open Rig_amd_abi
module S = Rig_amd_abi_support

let timeout = S.timeout
let gpu sdma = S.gpu ~sdma (11, 0, 0)
let words = S.encode
let version = S.version

(* The largest linear copy of each version, as the .mli states it. *)
let largest = function
  | 4, 4, s when s >= 2 -> 1 lsl 30
  | 5, m, _ when m >= 2 -> 1 lsl 30
  | a, _, _ when a >= 6 -> 1 lsl 30
  | _ -> 1 lsl 22

let versions =
  [
    (4, 0, 0);
    (4, 4, 0);
    (4, 4, 2);
    (4, 4, 5);
    (5, 0, 0);
    (5, 2, 0);
    (6, 0, 0);
    (7, 0, 0);
  ]

(* A version, an address pair and a length that [length max] draws, [max]
   the version's largest copy. *)
let copies_of length =
  let open Gen in
  let addr = int_range 0 ((1 lsl 47) - 1) in
  with_pp
    (fun ppf (v, dst, src, n) ->
      Format.fprintf ppf "SDMA %s, dst 0x%x, src 0x%x, %d bytes" (version v) dst
        src n)
    (let* v = of_list versions in
     let+ dst = addr and+ src = addr and+ n = length (largest v) in
     (v, dst, src, n))

(* Lengths of one linear copy, from 1 to the largest, its bounds included. *)
let linear_copies =
  copies_of (fun max ->
      Gen.frequency
        [
          (1, Gen.constant 1);
          (1, Gen.constant max);
          (1, Gen.constant (max - 1));
          (3, Gen.int_range 1 max);
        ])

(* A linear copy's 7 words: header, count less one, parameters, source, then
   destination, each address low word first. *)
let rec pieces = function
  | [] -> []
  | 1 :: count :: 0 :: s_lo :: s_hi :: d_lo :: d_hi :: rest ->
      (count + 1, s_lo lor (s_hi lsl 32), d_lo lor (d_hi lsl 32)) :: pieces rest
  | w :: _ -> failf "0x%x starts no linear copy" w

let copy =
  group ~timeout "copy"
    [
      prop "a linear copy is one piece of its bytes, up to the largest"
        linear_copies (fun (v, dst, src, n) ->
          let max = Sdma.max_copy (gpu v) in
          equal int ~msg:"the largest" (largest v) max;
          cover "the largest" (n = max);
          equal
            (list (triple int int int))
            [ (n, src, dst) ]
            (pieces (words (Sdma.copy_linear ~dst ~src ~bytes:n))));
    ]

(* The sub-window's layout of each GC, against ROCr's two layouts of
   COPY_LINEAR_SUB_WIND (sdma_registers.h): before GFX12, as Linux's SDMA 4
   to 6 headers lay it out, pitches of 19 bits from bit 13 of their words and
   offsets, widths and rows of 14 bits; from GFX12, pitches of 16 bits from
   bit 16 and offsets, widths and rows of 16 bits. Both: the element's size
   from bit 29 of the header, offsets and widths from bit 0, rows from bit
   16, and the operation 1 with the sub-operation 4 in bits 15:8. *)
let windows =
  let window (major, minor, step) =
    let sdma = if major >= 12 then (7, 0, 0) else (6, 0, 0) in
    Sdma.window (S.gpu ~sdma (major, minor, step))
  in
  let layout ~pitch ~most_width ~most_rows ~most_pitch =
    {
      Sdma.header = 1 lor (4 lsl 8);
      element = 29;
      src_x = 0;
      src_pitch = pitch;
      dst_x = 0;
      dst_pitch = pitch;
      width = 0;
      rows = 16;
      most_width;
      most_rows;
      most_pitch;
    }
  in
  let before =
    layout ~pitch:13 ~most_width:(1 lsl 14) ~most_rows:(1 lsl 14)
      ~most_pitch:(1 lsl 19)
  and from_12 =
    layout ~pitch:16 ~most_width:(1 lsl 16) ~most_rows:(1 lsl 16)
      ~most_pitch:(1 lsl 16)
  in
  let pp ppf (w : Sdma.window) =
    Format.fprintf ppf
      "header 0x%x, element %d, offsets %d %d, pitches %d %d, width %d, \
       rows %d, most %d %d %d"
      w.header w.element w.src_x w.dst_x w.src_pitch w.dst_pitch w.width
      w.rows w.most_width w.most_rows w.most_pitch
  in
  let window_t = Testable.make ~pp ~equal:( = ) in
  group ~timeout "windows"
    [
      cases "a sub-window's layout follows the GC" ~id:version
        [ (9, 4, 2); (9, 5, 0); (11, 0, 0); (11, 5, 0); (12, 0, 0); (12, 0, 1) ]
        (fun v ->
          let major, _, _ = v in
          equal window_t
            (if major >= 12 then from_12 else before)
            (window v));
    ]

(* A fill, word for word against CONSTANT_FILL's fields in the headers:
   fillsize at 31:30, and the count less one. *)
let fills =
  group ~timeout "fills"
    [
      cases "a fill sets as many bytes as a linear copy moves" ~id:version
        versions (fun v ->
          equal int (largest v) (Sdma.max_fill (gpu v)));
      test "a fill of dwords" (fun () ->
          equal (list int)
            [ 0xb lor (2 lsl 30); 0x40; 0x2; 0x6161_6161; 4095 ]
            (words
               (Sdma.const_fill ~dst:0x2_0000_0040 ~word:0x6161_6161
                  ~bytes:4096)));
    ]

let others =
  group ~timeout "others"
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

let () = exit (run "rig_amd_abi.sdma" [ copy; windows; fills; others ])
