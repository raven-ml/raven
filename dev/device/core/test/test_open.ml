(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module C = Device_core
module P = Device_core_support.Polled
module Support = Device_core_support

let timeout = 60.
let device = Testable.make ~pp:C.pp ~equal:C.equal
let memory name = require_ok ~pp:Format.pp_print_string (C.memory_device name)

let test_same_name () =
  let d = memory "open:same" in
  equal device d (memory "open:same");
  not_equal device d (memory "open:other")

let test_other_driver () =
  ignore (memory "open:taken");
  raises_match Exn.invalid_arg (fun () -> P.open_ "open:taken")

let test_facts () =
  let d = memory "open:facts" in
  let p, _ = P.open_ "open:facts-polled" in
  equal string "open:facts" (C.name d);
  equal bool true (C.computes d);
  equal bool true (C.runs_on_host d);
  equal bool false (C.runs_on_host p);
  equal bool true (C.shares_host_memory d);
  equal bool true (C.reaches d C.host);
  equal bool true (C.reaches C.host d);
  equal device C.host (C.host_of d);
  is_some (C.capability p P.capability_key)

let test_reopen () =
  let d, p = P.open_ "open:reopen" in
  let s = C.Submission.make ~reads:0 ~writes:0 ~waits:0 d [||] in
  P.fail p;
  raises_match (function C.Lost _ -> true | _ -> false) (fun () -> C.submit s);
  equal (list string) [ "stop" ] (P.log p);
  let d', _ = P.open_ "open:reopen" in
  not_equal device d d';
  equal (option string) None (C.lost d')

(* An open whose opener blocks holds back only opens of its own name. *)
let test_blocked_opener () =
  let lock = Mutex.create () and cond = Condition.create () in
  let inside = ref false and go = ref false in
  let make () =
    Mutex.protect lock (fun () ->
        inside := true;
        while not !go do
          Condition.wait cond lock
        done);
    Ok (P.make ())
  in
  let slow =
    Thread.create
      (fun () -> ignore (C.open_ (module P) ~name:"open:slow" make))
      ()
  in
  Support.await "a running opener" (fun () ->
      Mutex.protect lock (fun () -> !inside));
  let d, _ = P.open_ "open:fast" in
  equal string "open:fast" (C.name d);
  Mutex.protect lock (fun () ->
      go := true;
      Condition.signal cond);
  Thread.join slow

let tests =
  group ~timeout "opening"
    [
      test "one name opens one device, until it is lost" test_same_name;
      test "a name open as another driver's device raises" test_other_driver;
      test "a device states its facts" test_facts;
      test "a lost device's name opens anew once its stop answered" test_reopen;
      test "a blocked opener holds back no other name" test_blocked_opener;
    ]

let () = exit (run "device_core.open" [ tests ])
