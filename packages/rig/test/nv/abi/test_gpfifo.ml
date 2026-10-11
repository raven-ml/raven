(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Ring entries against NVIDIA's clc56f.h: GP_ENTRY0_GET 31:2, GP_ENTRY1_GET_HI
   7:0, LEVEL 9:9 (SUBROUTINE 1), LENGTH 30:10, SYNC 31:31. *)

open Windtrap
open Rig_nv_abi

let max_offset = (1 lsl 40) - 1

let entry a ~offset ~words =
  String.get_int64_le
    (Packet.encode Int64.of_int (Gpfifo.entry a ~offset ~words))
    0

(* An entry's fields: its segment's address, from GET and GET_HI, its length and
   its SYNC bit. *)
let fields e =
  let bits lo n =
    Int64.(
      to_int (logand (shift_right_logical e lo) (sub (shift_left 1L n) 1L)))
  in
  ((bits 2 30 lsl 2) lor (bits 32 8 lsl 32), bits 42 21, bits 63 1)

(* A segment at a 4-byte aligned address below 2^40, as an address and an
   offset, and its words, at the edges of their ranges. *)
let segment =
  let open Gen in
  let aligned hi = map (fun n -> n * 4) (int_range 0 (hi / 4)) in
  let edges = of_list ~pp:Format.pp_print_int [ 0; 1; Gpfifo.max_words ] in
  let* at =
    frequency [ (3, aligned max_offset); (1, constant (max_offset - 3)) ]
  in
  let+ offset = aligned at
  and+ words = frequency [ (3, int_range 0 Gpfifo.max_words); (1, edges) ] in
  (at - offset, offset, words)

let entries =
  group ~timeout:10. "entry"
    [
      prop
        "an entry names its segment's address and words, and waits for nothing"
        segment (fun (a, offset, words) ->
          cover "the most words" (words = Gpfifo.max_words);
          cover "the last address" (a + offset = max_offset - 3);
          equal (triple int int int)
            (a + offset, words, 0)
            (fields (entry a ~offset ~words)));
      prop "an entry adds to the address what offset and words alone decide"
        segment (fun (a, offset, words) ->
          match Gpfifo.entry a ~offset ~words with
          | [ W64 (Add (Value a', n)) ] ->
              equal int a a';
              equal int64 n (entry 0 ~offset ~words)
          | _ -> fail "an entry is [W64 (Add (Value addr, n))]");
      test "an entry of the most words keeps bit 63 clear" (fun () ->
          let a = 0x12_3456_7000 and words = Gpfifo.max_words in
          equal int64
            Int64.(
              add
                (of_int (a + 0x40))
                (logor (shift_left 1L 41) (shift_left (of_int words) 42)))
            (entry a ~offset:0x40 ~words));
      test "a segment holds at most 2^21 - 1 words" (fun () ->
          equal int ((1 lsl 21) - 1) Gpfifo.max_words);
      cases ~name:string_of_int
        "a count of words outside [0;max_words] is refused"
        [ min_int; -1; Gpfifo.max_words + 1; max_int ]
        (fun words ->
          raises_match (Exn.invalid_arg ~substring:"Gpfifo.entry") (fun () ->
              Gpfifo.entry 0 ~offset:0 ~words));
      cases ~name:string_of_int "an offset outside [0;2^40-1] is refused"
        [ min_int; -4; 1 lsl 40; max_int ]
        (fun offset ->
          raises_match (Exn.invalid_arg ~substring:"Gpfifo.entry") (fun () ->
              Gpfifo.entry 0 ~offset ~words:1));
    ]

let () = exit (run "rig_nv_abi.gpfifo" [ entries ])
