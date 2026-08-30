(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk

let () =
  run "Bump"
    [
      group "Alloc"
        [
          test "advances sequentially" (fun () ->
              let b = Bump.create ~size:100 () in
              equal int 0 (Bump.alloc b 10 ());
              equal int 10 (Bump.alloc b 5 ());
              equal int 15 (Bump.alloc b 1 ()));
          test "offsets results by base" (fun () ->
              let b = Bump.create ~size:100 ~base:0x1000 () in
              equal int 0x1000 (Bump.alloc b 16 ());
              equal int 0x1010 (Bump.alloc b 16 ()));
          test "aligns the allocated offset" (fun () ->
              let b = Bump.create ~size:100 () in
              equal int 0 (Bump.alloc b 3 ());
              equal int 8 (Bump.alloc b 4 ~align:8 ());
              equal int 12 (Bump.alloc b 1 ()));
          test "wraps to the start when exhausted" (fun () ->
              let b = Bump.create ~size:100 () in
              equal int 0 (Bump.alloc b 60 ());
              equal int 0 (Bump.alloc b 60 ());
              equal int 60 (Bump.alloc b 10 ()));
          test "raises when full and wrapping is disabled" (fun () ->
              let b = Bump.create ~size:100 ~wrap:false () in
              equal int 0 (Bump.alloc b 60 ());
              raises Out_of_memory (fun () -> Bump.alloc b 60 ()));
          test "alignment padding counts toward exhaustion" (fun () ->
              let b = Bump.create ~size:16 ~wrap:false () in
              equal int 0 (Bump.alloc b 1 ());
              raises Out_of_memory (fun () -> Bump.alloc b 9 ~align:8 ()));
        ];
    ]
