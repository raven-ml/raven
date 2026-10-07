(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Local memory sizing on an RTX 5000 Ada's geometry. *)

open Windtrap
open Device_nv_abi

let ada =
  {
    Gpu.compute_class = 0xc9c0;
    sass_version = 0x89;
    gpcs = 11;
    tpcs_per_gpc = 6;
    sms_per_tpc = 2;
    warps_per_sm = 48;
    shared_window = 0x7294_0000_0000;
    local_window = 0x7293_0000_0000;
  }

let tests =
  group "make"
    [
      test "no local memory is none" (fun () ->
          let l = Local.make ada 0 in
          equal (list int) [ 0; 0; 0 ] [ l.per_thread; l.per_tpc; l.bytes ]);
      test "596 bytes a thread" (fun () ->
          (* 608 bytes a thread, 19456 a warp, 96 warps a TPC: 57 * 32 KiB; 66
             TPCs: 123273216 bytes, rounded up to 941 * 128 KiB. *)
          let l = Local.make ada 596 in
          equal (list int)
            [ 608; 57 * 0x8000; 941 * 0x20000 ]
            [ l.per_thread; l.per_tpc; l.bytes ]);
      test "a negative need is refused" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"Local.make") (fun () ->
              Local.make ada (-1)));
    ]

let () = exit (run "device_nv_abi.local" [ tests ])
