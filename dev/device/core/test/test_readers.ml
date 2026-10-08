(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The C readers of device_core.h, from native code and from bytecode, where
   another library's stubs resolve them at load. *)

open Windtrap
module C = Device_core
module B = Device_core.Buffer
module R = Device_core_support.Reader
module S = Device_dtype.Scalar

let timeout = 60.

let test_live () =
  let b = B.create C.host S.Float32 10 in
  equal int (B.address b) (R.host b);
  equal int 10 (R.length b);
  equal (option string) None (R.why b)

let test_view () =
  let b = B.create C.host S.UInt8 64 in
  let v = B.view b ~offset:8 S.Int4 6 in
  equal int (B.address b + 8) (R.host v);
  equal int 6 (R.length v)

let test_dead () =
  let b = B.create C.host S.UInt8 8 in
  C.Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
      ignore (C.Claim.consume c ~why:"donated" b));
  equal (option string) (Some "donated") (R.why b)

let tests =
  [
    group ~timeout "readers"
      [
        test "a live buffer reads as its address and length" test_live;
        test "a view reads as its own first byte and elements" test_view;
        test "a dead buffer reads as its reason" test_dead;
      ];
  ]

let () = exit (run "device_core.readers" tests)
