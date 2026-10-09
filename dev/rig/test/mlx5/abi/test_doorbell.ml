(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Doorbell records and access regions, against rdma-core's mlx5dv.h and mlx5.h
   (a queue pair's send count is word 1 of its record, a completion queue's
   consumer count word 0 and its arm word word 1; the arm word is sequence << 28
   | command | count, the command 0 for the next completion; registers sit at
   0x800 of an access region, the completion queue's at 0x20), mlx5.c's
   numbering of registers (4 per access region, register k of region j at 0x800
   + k * size), and the kernel's mapping offsets (main.c: the command, 3 for
   uncached pages, at bit 8 of the page index). *)

open Windtrap
open Rig_mlx5_abi

let records =
  group "records"
    [
      test "a record is two 32-bit words" (fun () -> equal int 8 Doorbell.size);
      test "a queue pair's send count is word 1" (fun () ->
          equal int 4 Doorbell.send);
      test "a completion queue's consumer count is word 0" (fun () ->
          equal int 0 Doorbell.consumed);
      test "its arm word is word 1" (fun () -> equal int 4 Doorbell.armed);
      prop "an arm stores the sequence's 2 bits, the count's 24 and the queue"
        Gen.(
          triple (int_range 0 1000)
            (int_range 0 (1 lsl 30))
            (int_range 0 0xff_ffff))
        (fun (sequence, count, cq) ->
          equal (pair int int)
            (((sequence land 3) lsl 28) lor (count land 0xff_ffff), cq)
            (Doorbell.arm ~sequence ~count ~cq));
      test "a queue past 24 bits is refused" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"Doorbell.arm") (fun () ->
              Doorbell.arm ~sequence:0 ~count:0 ~cq:0x100_0000));
    ]

let regions =
  group "access regions"
    [
      test "an access region has 4 registers" (fun () ->
          equal int 4 Uar.per_region);
      cases
        ~name:(fun (page, i, _) -> Printf.sprintf "page %d of %d bytes" i page)
        "a system page maps uncached at its index after the command"
        [
          (4096, 0, 0x30_0000);
          (4096, 1, 0x30_1000);
          (65536, 2, 0x302_0000);
          (4096, 255, 0x3f_f000);
        ]
        (fun (page, i, off) -> equal int off (Uar.mapping ~page i));
      prop "register k of region j of page i is at j * 4096 + 0x800 + k * size"
        Gen.(
          quad (int_range 0 15) (int_range 1 16) (int_range 0 1)
            (frequency
               [ (3, constant 512); (1, constant 0); (1, int_range 0 1024) ]))
        (fun (i, per_page, k, size) ->
          let j = i mod per_page in
          let r = (((i * per_page) + j) * 4) + k in
          equal (pair int int)
            (i, (j * 4096) + 0x800 + (k * size))
            (Uar.register ~per_page ~size r));
      cases ~name:(Printf.sprintf "register %d")
        "registers 2 and 3 are never rung" [ 2; 3; 6; 7 ] (fun r ->
          raises_match (Exn.invalid_arg ~substring:"never rung") (fun () ->
              Uar.register ~per_page:1 ~size:512 r));
      test "page 256 is refused" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"Uar.mapping") (fun () ->
              Uar.mapping ~page:4096 256));
      test "a completion queue's register is at 0x20" (fun () ->
          equal int 0x20 Uar.cq_doorbell);
    ]

let () = exit (run "rig_mlx5_abi.doorbell" [ records; regions ])
