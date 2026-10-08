(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The C readers of rig.h, from native code and from bytecode, where another
   library's stubs resolve them at load. *)

open Windtrap
module C = Rig
module B = Rig.Buffer
module R = Rig_support.Reader

let timeout = 60.

let test_live () =
  let b = B.create C.host 40 in
  equal int (B.address b) (R.host b);
  equal int 40 (R.bytes b);
  equal (option string) None (R.why b)

let test_view () =
  let b = B.create C.host 64 in
  let v = B.view b ~first:8 ~length:6 in
  equal int (B.address b + 8) (R.host v);
  equal int 6 (R.bytes v)

let test_dead () =
  let b = B.create C.host 8 in
  C.Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
      ignore (C.Claim.consume c ~why:"donated" b));
  equal (option string) (Some "donated") (R.why b)

(* A claim from C resolves and claims, and its release ends it. *)
let test_claim () =
  let b = B.create C.host 8 in
  let answer = Testable.make ~pp:R.pp_answer ~equal:( = ) in
  equal ~msg:"claimed" answer R.Claimed (R.claim b B.Read);
  R.release b;
  C.Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
      equal ~msg:"released" bool true (C.Claim.exclusive c b))

let tests =
  [
    group ~timeout "readers"
      [
        test "a live buffer reads as its address and length" test_live;
        test "a view reads as its own first byte and elements" test_view;
        test "a dead buffer reads as its reason" test_dead;
        test "a claim from C claims and its release ends it" test_claim;
      ];
  ]

let () = exit (run "rig.readers" tests)
