(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Fail-loud half of the [Device.Buffer.copy_from] contract.

   copy_from delegates to a runner the realize engine installs at
   initialization. Until then it must fail loudly rather than silently drop the
   copy. This executable deliberately never references the engine, so the
   installer never runs and copy_from stays unbacked, letting us observe the
   pre-install behaviour that test_device (which links the engine) cannot.

   The guarantee rests on this executable not linking [Realize]: if a future
   change makes it reference the engine — directly or through a helper — the
   installer runs, copy_from succeeds, and the assertion below fails. That is a
   signal to move this test, not to relax it. The delegation half is in
   test_device. *)

open Windtrap
open Tolk
open Tolk_uop

let device = Tolk_cpu.create "CPU:no-engine"

let allocated_i32 n =
  let buf = Device.create_buffer ~size:n ~dtype:Dtype.int32 device in
  Device.Buffer.ensure_allocated buf;
  buf

let fail_loud_tests =
  group "Buffer.copy_from without the engine"
    [
      test "raises before the copy runner is installed" (fun () ->
          (* Same size and dtype, so copy_from clears its own precondition
             checks and reaches the uninstalled runner rather than raising a
             size or dtype mismatch. *)
          let dst = allocated_i32 4 and src = allocated_i32 4 in
          raises_invalid_arg
            "Device.Buffer.copy_from: no copy runner installed; link the \
             realize engine to route buffer copies" (fun () ->
              Device.Buffer.copy_from ~dst ~src));
    ]

let () = run __FILE__ [ fail_loud_tests ]
