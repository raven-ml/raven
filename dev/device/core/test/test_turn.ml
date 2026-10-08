(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Admission: a device's turn, its room, and what interrupts a submit. Threads
   here are systhreads of the test's domain, so a turn or a wait that kept the
   domain lock would stop the test. *)

open Windtrap
module C = Device_core
module B = Device_core.Buffer
module Sub = Device_core.Submission
module P = Device_core_support.Polled
module Support = Device_core_support

let timeout = 60.
let lost d = function C.Lost (d', _) -> C.equal d d' | _ -> false
let count call p = List.length (List.filter (( = ) call) (P.log p))
let empty d = Sub.make ~reads:0 ~writes:0 ~waits:0 d [||]

(* A part that holds one unit of a Polled queue. *)
let bump () =
  let arg = B.create C.host 8 in
  {
    Sub.queue = "COMPUTE:0";
    after = [||];
    work =
      Sub.Fill { fill = Support.bump; arg; ring_units = 0; segment_bytes = 0 };
  }

let one_part d = Sub.make ~reads:0 ~writes:0 ~waits:0 d [| bump () |]
let value s = C.Point.value (C.submit s)

(* A thread running [f], whose outcome [join] gives. *)
let spawn f =
  let r = ref (Error Exit) in
  let t = Thread.create (fun () -> r := try Ok (f ()) with e -> Error e) () in
  fun () ->
    Thread.join t;
    !r

let pp_outcome ppf = function
  | Ok v -> Format.fprintf ppf "Ok %d" v
  | Error e -> Format.fprintf ppf "Error %s" (Printexc.to_string e)

let outcome = Testable.make ~pp:pp_outcome ~equal:( = )

(* Room *)

(* A submit that waits for room waits with the turn released: a submission with
   no parts, which fits, is handed over meanwhile. *)
let test_later_releases () =
  let d, p = P.open_ ~capacity:1 "turn:later" in
  equal int 1 (value (one_part d));
  P.gate p;
  let waiting = spawn (fun () -> value (one_part d)) in
  Support.await "a sleep at the gate" (fun () -> P.sleepers p = 1);
  equal int 2 (value (empty d));
  P.open_gate p;
  equal outcome (Ok 3) (waiting ())

(* A fault found while a submit waits for room loses the device, and the submit
   commits nothing. *)
let test_fault_waiting () =
  let d, p = P.open_ ~capacity:1 "turn:fault-room" in
  equal int 1 (value (one_part d));
  P.gate p;
  let waiting = spawn (fun () -> value (one_part d)) in
  Support.await "a sleep at the gate" (fun () -> P.sleepers p = 1);
  P.fault p "the engine hung";
  P.open_gate p;
  (match waiting () with
  | Error e -> equal bool true (lost d e)
  | Ok v -> failf "the waiting submit returned %d" v);
  equal int 1 (C.submitted d)

let test_never () =
  let d, _ = P.open_ ~capacity:1 "turn:never" in
  let never = Sub.make ~reads:0 ~writes:0 ~waits:0 d [| bump (); bump () |] in
  raises_match Exn.invalid_arg (fun () -> C.submit never);
  equal int 0 (C.submitted d);
  equal int 1 (value (one_part d))

(* Turns of a driver that blocks *)

(* A submit blocked in its driver holds its device's turn, and no other
   device's. *)
let test_turn_per_device () =
  let d, p = P.open_ ~capacity:1 ~may_block:true "turn:blocking" in
  let e, _ = P.open_ "turn:other" in
  equal int 1 (value (one_part d));
  let blocked = spawn (fun () -> value (one_part d)) in
  Support.await "a blocked submit" (fun () -> P.blocked p = 1);
  equal int 1 (value (empty e));
  ignore (P.run p);
  equal outcome (Ok 2) (blocked ())

(* Two threads of one domain submit to a device whose driver blocks; the domain
   runs the device's queue meanwhile. *)
let test_two_threads () =
  let d, p = P.open_ ~capacity:1 ~may_block:true "turn:two-threads" in
  equal int 1 (value (one_part d));
  let a = spawn (fun () -> value (one_part d)) in
  let b = spawn (fun () -> value (one_part d)) in
  Support.await "a blocked submit" (fun () -> P.blocked p = 1);
  ignore (P.run p);
  Support.await "the second blocked submit" (fun () ->
      P.blocked p = 1 && P.queued p = 1);
  ignore (P.run p);
  equal (slist outcome compare) [ Ok 2; Ok 3 ] [ a (); b () ];
  equal int 3 (C.submitted d)

(* A device lost while a submit blocks in its driver is stopped once that submit
   returned, and the submit raises Lost. *)
let test_loss_while_blocked () =
  let d, p = P.open_ ~capacity:1 ~may_block:true "turn:loss" in
  equal int 1 (value (one_part d));
  let blocked = spawn (fun () -> value (one_part d)) in
  Support.await "a blocked submit" (fun () -> P.blocked p = 1);
  P.fault p "the engine hung";
  raises_match (lost d) (fun () -> C.wait d 1);
  equal int 1 (P.blocked p);
  equal int 0 (count "stop" p);
  ignore (P.run p);
  (match blocked () with
  | Error e -> equal bool true (lost d e)
  | Ok v -> failf "the blocked submit returned %d" v);
  equal int 1 (count "stop" p)

(* Interruptions *)

(* Windtrap's [raises] lets [Sys.Break] through, as it ends a run. *)
let breaks f =
  Sys.catch_break true;
  Fun.protect
    ~finally:(fun () -> Sys.catch_break false)
    (fun () -> match f () with _ -> false | exception Sys.Break -> true)

(* After [Sys.Break] the device is used as before, and its stop still runs once
   at its loss: the interrupted call left nothing counted in flight. *)
let still_works d p ~submitted =
  equal (option string) None (C.lost d);
  equal int submitted (C.submitted d);
  C.wait d submitted;
  equal int (submitted + 1) (value (empty d));
  P.fail p;
  raises_match (lost d) (fun () -> C.submit (empty d));
  equal int 1 (count "stop" p)

let test_break_in_wait () =
  let d, p = P.open_ "turn:break-wait" in
  equal int 1 (value (empty d));
  P.interrupt p;
  equal bool true (breaks (fun () -> C.wait d 1));
  still_works d p ~submitted:1

(* A thread waiting for room is interrupted: nothing is assigned to it. *)
let test_break_waiting_for_room () =
  let d, p = P.open_ ~capacity:1 "turn:break-room" in
  equal int 1 (value (one_part d));
  P.gate p;
  let broke =
    breaks (fun () ->
        let waiting = spawn (fun () -> value (one_part d)) in
        Support.await "a sleep at the gate" (fun () -> P.sleepers p = 1);
        P.interrupt p;
        P.open_gate p;
        equal outcome (Error Sys.Break) (waiting ()))
  in
  equal bool false broke;
  still_works d p ~submitted:1

let tests =
  [
    group ~timeout "room"
      [
        test "a submit waiting for room lets a submission that fits through"
          test_later_releases;
        test "parts that never fit assign no value" test_never;
        test "a fault while a submit waits for room commits nothing"
          test_fault_waiting;
      ];
    group ~timeout "turns"
      [
        test "a submit blocked in its driver holds only its device's turn"
          test_turn_per_device;
        test "two threads of one domain submit to a device that blocks"
          test_two_threads;
        test "a loss during a blocked submit stops the device after it returns"
          test_loss_while_blocked;
      ];
    group ~timeout "interruptions"
      [
        test "Sys.Break in a wait loses nothing" test_break_in_wait;
        test "Sys.Break while waiting for room assigns no value"
          test_break_waiting_for_room;
      ];
  ]

let () = exit (run "device_core.turn" tests)
