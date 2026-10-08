(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module C = Device_core
module B = Device_core.Buffer
module Sub = Device_core.Submission
module P = Device_core_support.Polled
module Support = Device_core_support
module S = Device_dtype.Scalar

let timeout = 60.
let empty ?(waits = 0) d = Sub.make ~reads:0 ~writes:0 ~waits d [||]
let lost d = function C.Lost (d', _) -> C.equal d d' | _ -> false
let count call p = List.length (List.filter (( = ) call) (P.log p))

let test_failed_submit () =
  let d, p = P.open_ "loss:failed" in
  let s = empty d in
  ignore (C.submit s);
  P.fail p;
  raises_match (lost d) (fun () -> C.submit s);
  equal (option string) (Some "the submission failed") (C.lost d);
  raises_match (lost d) (fun () -> C.submit s);
  raises_match (lost d) (fun () -> B.create d S.UInt8 8);
  equal int 1 (count "stop" p)

let test_fault () =
  let d, p = P.open_ "loss:fault" in
  let v = C.Point.value (C.submit (empty d)) in
  P.fault p "the engine hung";
  raises_match (lost d) (fun () -> C.wait d v);
  equal (option string) (Some "the engine hung") (C.lost d);
  equal int 1 (count "stop" p)

(* After the stop answered, only free and unmap reach the driver. *)
let test_after_stop () =
  let d, p = P.open_ "loss:after-stop" in
  let b = B.create d S.UInt8 64 in
  P.fail p;
  (try ignore (C.submit (empty d)) with C.Lost _ -> ());
  ignore (Sys.opaque_identity b);
  Gc.full_major ();
  ignore (B.create C.host S.UInt8 8);
  let rec after = function
    | "stop" :: rest -> rest
    | _ :: rest -> after rest
    | [] -> []
  in
  List.iter (fun call -> mem string call [ "free"; "unmap" ]) (after (P.log p))

(* A device whose queue waits on a lost device's unreached value is lost with
   it. *)
let test_spread () =
  let producer, pp = P.open_ "loss:producer" in
  let consumer, _ = P.open_ ~waits_host:true "loss:consumer" in
  let a = C.submit (empty producer) in
  let s = empty ~waits:1 consumer in
  Sub.wait_for s 0 a;
  ignore (C.submit s);
  P.fail pp;
  raises_match (lost producer) (fun () -> C.submit (empty producer));
  equal (option string) (Some "loss:producer lost") (C.lost consumer)

(* A consumer whose waited value was reached stays. *)
let test_no_spread () =
  let producer, pp = P.open_ "loss:producer-2" in
  let consumer, _ = P.open_ ~waits_host:true "loss:consumer-2" in
  let a = C.submit (empty producer) in
  C.wait producer (C.Point.value a);
  let s = empty ~waits:1 consumer in
  Sub.wait_for s 0 a;
  ignore (C.submit s);
  P.fail pp;
  raises_match (lost producer) (fun () -> C.submit (empty producer));
  equal (option string) None (C.lost consumer)

(* An Unknown answer keeps the device's memory until its word reads its last
   value. *)
let test_unknown () =
  let d, p = P.open_ ~answer:`Unknown "loss:unknown" in
  let b = B.create d S.UInt8 64 in
  ignore (C.submit (empty d));
  P.fail p;
  (try ignore (C.submit (empty d)) with C.Lost _ -> ());
  ignore (Sys.opaque_identity b);
  Gc.full_major ();
  ignore (B.create C.host S.UInt8 8);
  equal int 0 (count "free" p);
  P.set_word p (C.submitted d);
  ignore (B.create C.host S.UInt8 8);
  equal int 1 (count "free" p)

(* A wait checks the loss after its value: a reached value of a lost device
   raises. *)
let test_reached () =
  let d, p = P.open_ "loss:reached" in
  let v = C.Point.value (C.submit (empty d)) in
  C.wait d v;
  P.fail p;
  raises_match (lost d) (fun () -> C.submit (empty d));
  raises_match (lost d) (fun () -> C.wait d v)

(* Two domains sleep on a device that faults: each raises its Lost, and the
   device is lost and stopped once. *)
let test_two_sleeps () =
  let d, p = P.open_ "loss:two-sleeps" in
  let v = C.Point.value (C.submit (empty d)) in
  P.gate p;
  let wait () =
    match C.wait d v with () -> "returned" | exception C.Lost (_, why) -> why
  in
  let waiters = List.init 2 (fun _ -> Domain.spawn wait) in
  Support.await "two sleeps at the gate" (fun () -> P.sleepers p = 2);
  P.fault p "the engine hung";
  P.open_gate p;
  equal (list string)
    [ "the engine hung"; "the engine hung" ]
    (List.map Domain.join waiters);
  equal (option string) (Some "the engine hung") (C.lost d);
  equal int 1 (count "stop" p)

let tests =
  [
    group ~timeout "loss"
      [
        test "a failed hand-over loses the device once" test_failed_submit;
        test "a fault a wait finds loses the device" test_fault;
        test "a stopped device is only freed and unmapped" test_after_stop;
        test "a queue waiting on a lost device's value is lost" test_spread;
        test "a queue whose wait was reached stays" test_no_spread;
        test "an Unknown answer keeps memory until the word drains" test_unknown;
        test "a wait on a lost device raises once its value is reached"
          test_reached;
        test "a fault two domains' sleeps find loses the device once"
          test_two_sleeps;
      ];
  ]

let () = exit (run "device_core.loss" tests)
