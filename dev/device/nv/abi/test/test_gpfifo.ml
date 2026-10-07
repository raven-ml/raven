(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Ring entries against NVIDIA's clc56f.h: GP_ENTRY0_GET 31:2, GP_ENTRY1_GET_HI
   7:0, LEVEL 9:9 (SUBROUTINE 1), LENGTH 30:10, SYNC 31:31. *)

open Windtrap
open Device_nv_abi

let entry a ~offset ~words =
  String.get_int64_le
    (Packet.encode Int64.of_int (Gpfifo.entry a ~offset ~words))
    0

let tests =
  group "entry"
    [
      test "an entry of the most words keeps bit 63 clear" (fun () ->
          let a = 0x12_3456_7000 and words = Gpfifo.max_words in
          equal int64
            Int64.(
              add
                (of_int (a + 0x40))
                (logor (shift_left 1L 41) (shift_left (of_int words) 42)))
            (entry a ~offset:0x40 ~words));
      test "more words than LENGTH holds is refused" (fun () ->
          equal int ((1 lsl 21) - 1) Gpfifo.max_words;
          raises_match (Exn.invalid_arg ~substring:"Gpfifo.entry") (fun () ->
              Gpfifo.entry 0 ~offset:0 ~words:(Gpfifo.max_words + 1)));
    ]

let () = exit (run "device_nv_abi.gpfifo" [ tests ])
