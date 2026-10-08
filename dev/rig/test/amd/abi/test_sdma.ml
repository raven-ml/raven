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

(* A version, an address pair and a length around its largest copy. *)
let copies =
  let open Gen in
  let addr = int_range 0 ((1 lsl 47) - 1) in
  with_pp
    (fun ppf (v, dst, src, n) ->
      Format.fprintf ppf "SDMA %s, dst 0x%x, src 0x%x, %d bytes" (version v) dst
        src n)
    (let* v = of_list versions in
     let max = largest v in
     let+ dst = addr
     and+ src = addr
     and+ n =
       frequency
         [
           (2, int_range 0 4096);
           (1, constant max);
           ( 2,
             map
               (fun (k, d) -> (k * max) + d)
               (pair (int_range 0 3) (int_range (-2) 2))
             |> such_that (fun n -> n >= 0) );
           (1, int_range 0 (3 * max));
         ]
     in
     (v, dst, src, n))

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
      prop
        "a copy is linear copies of the largest size, in order, then the rest"
        copies (fun (v, dst, src, n) ->
          let max = largest v in
          cover "two pieces or more" (n > max);
          cover "exactly the largest" (n = max);
          let rec expected at =
            if at >= n then []
            else
              let k = Int.min max (n - at) in
              (k, src + at, dst + at) :: expected (at + k)
          in
          equal
            (list (triple int int int))
            (expected 0)
            (pieces (words (Sdma.copy (gpu v) ~dst ~src n))));
      prop "a linear copy is one piece of its bytes, up to the largest" copies
        (fun (v, dst, src, n) ->
          let max = Sdma.max_copy (gpu v) in
          equal int ~msg:"the largest" (largest v) max;
          let n = 1 + (n mod max) in
          cover "the largest" (n = max);
          equal
            (list (triple int int int))
            [ (n, src, dst) ]
            (pieces (words (Sdma.copy_linear ~dst ~src ~bytes:n)));
          equal (list int) ~msg:"as copy's"
            (words (Sdma.copy (gpu v) ~dst ~src n))
            (words (Sdma.copy_linear ~dst ~src ~bytes:n)));
      test "a copy of no bytes is no packet" (fun () ->
          equal (list int) []
            (words (Sdma.copy (gpu (6, 0, 0)) ~dst:0 ~src:0 0)));
      test "a negative copy is refused" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"Sdma.copy") (fun () ->
              Sdma.copy (gpu (6, 0, 0)) ~dst:0 ~src:0 (-1)));
      cases ~name:string_of_int "a copy past 2^48 bytes is refused"
        [ (1 lsl 48) + 1; max_int - (1 lsl 30) + 2; max_int ]
        (fun n ->
          raises_match (Exn.invalid_arg ~substring:"Sdma.copy") (fun () ->
              Sdma.copy (gpu (6, 0, 0)) ~dst:0 ~src:0 n));
      (* 2^18 copies of 1 GiB, of 7 words each. *)
      test "a copy of 2^48 bytes is taken" (fun () ->
          equal int
            ((1 lsl 18) * 7)
            (Rig_packet.size
               (Sdma.copy (gpu (6, 0, 0)) ~dst:0 ~src:0 (1 lsl 48))));
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

let () = exit (run "rig_amd_abi.sdma" [ copy; others ])
