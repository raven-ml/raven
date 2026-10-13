(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Admission: a device's turn, its room, and what interrupts a submit. Threads
   here are systhreads of the test's domain, so a turn or a wait that kept the
   domain lock would stop the test. *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support

let submit ?(buffers = [||]) ?waits s =
  Rig.submit ?waits s ~run:(Sub.Run.make ()) ~buffers

let timeout = 60.
let lost d = function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false
let count call p = List.length (List.filter (( = ) call) (P.log p))
let empty d = Sub.make d [||]

(* A part that holds one unit of a Polled queue. *)
let idle () =
  (Sub.fill Support.noop ~params:0)

let one_part d =
  let s = Sub.make d [| idle () |] and run = Sub.Run.make () in
  Sub.Run.groups run (Sub.block s 0) 0 0 0;
  (s, run)

let point (s, run) = Rig.submit s ~run ~buffers:[||]
let value x = Rig.Point.value (point x)

let empty_run d = (empty d, Sub.Run.make ())

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
  let base = Rig.submitted d in
  equal int (base + 1) (value (one_part d));
  P.gate p;
  let waiting = spawn (fun () -> value (one_part d)) in
  Support.await "a sleep at the gate" (fun () -> P.sleepers p = 1);
  equal int (base + 2) (value (empty_run d));
  P.open_gate p;
  equal outcome (Ok (base + 3)) (waiting ())

(* A fault found while a submit waits for room loses the device, and the submit
   commits nothing. *)
let test_fault_waiting () =
  let d, p = P.open_ ~capacity:1 "turn:fault-room" in
  let base = Rig.submitted d in
  equal int (base + 1) (value (one_part d));
  P.gate p;
  let waiting = spawn (fun () -> value (one_part d)) in
  Support.await "a sleep at the gate" (fun () -> P.sleepers p = 1);
  P.fault p "the engine hung";
  P.open_gate p;
  (match waiting () with
  | Error e -> equal bool true (lost d e)
  | Ok v -> failf "the waiting submit returned %d" v);
  equal int (base + 1) (Rig.submitted d)

let test_never () =
  let d, _ = P.open_ "turn:never" in
  let base = Rig.submitted d in
  let image =
    require_ok ~pp:Format.pp_print_string (Rig.Image.load d "functions")
  in
  let never = Sub.make d [| Sub.launch image "main" ~params:0 |] in
  let run = Sub.Run.make () in
  Sub.Run.groups run (Sub.block never 0) 0 0 0;
  raises_match Exn.invalid_arg (fun () ->
      Rig.submit never ~run ~buffers:[||]);
  equal int base (Rig.submitted d);
  equal int (base + 1) (value (one_part d))

(* Turns of a driver that blocks *)

(* A submit blocked in its driver holds its device's turn, and no other
   device's. *)
let test_turn_per_device () =
  let d, p = P.open_ ~capacity:1 ~may_block:true "turn:blocking" in
  let base = Rig.submitted d in
  let e, _ = P.open_ "turn:other" in
  let base_e = Rig.submitted e in
  equal int (base + 1) (value (one_part d));
  let blocked = spawn (fun () -> value (one_part d)) in
  Support.await "a blocked submit" (fun () -> P.blocked p = 1);
  equal int (base_e + 1) (value (empty_run e));
  ignore (P.run p);
  equal outcome (Ok (base + 2)) (blocked ())

(* Two threads of one domain submit to a device whose driver blocks; the domain
   runs the device's queue meanwhile. *)
let test_two_threads () =
  let d, p = P.open_ ~capacity:1 ~may_block:true "turn:two-threads" in
  let base = Rig.submitted d in
  equal int (base + 1) (value (one_part d));
  let a = spawn (fun () -> value (one_part d)) in
  let b = spawn (fun () -> value (one_part d)) in
  Support.await "a blocked submit" (fun () -> P.blocked p = 1);
  ignore (P.run p);
  Support.await "the second blocked submit" (fun () ->
      P.blocked p = 1 && P.queued p = 1);
  ignore (P.run p);
  equal (slist outcome compare)
    [ Ok (base + 2); Ok (base + 3) ]
    [ a (); b () ];
  equal int (base + 3) (Rig.submitted d)

(* A device lost while a submit blocks in its driver is stopped once that submit
   returned, and the submit raises Lost. *)
let test_loss_while_blocked () =
  let d, p = P.open_ ~capacity:1 ~may_block:true "turn:loss" in
  let first = point (one_part d) in
  equal int 1 (Rig.Point.value first);
  let blocked = spawn (fun () -> value (one_part d)) in
  Support.await "a blocked submit" (fun () -> P.blocked p = 1);
  P.fault p "the engine hung";
  raises_match (lost d) (fun () -> Rig.Point.wait first);
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
let still_works d p ~last =
  equal (option string) None (Rig.lost d);
  equal int (Rig.Point.value last) (Rig.submitted d);
  Rig.Point.wait last;
  equal int (Rig.Point.value last + 1) (value (empty_run d));
  P.fail p;
  raises_match (lost d) (fun () -> submit (empty d));
  equal int 1 (count "stop" p)

let test_break_in_wait () =
  let d, p = P.open_ "turn:break-wait" in
  let base = Rig.submitted d in
  let first = point (empty_run d) in
  equal int (base + 1) (Rig.Point.value first);
  P.interrupt p;
  equal bool true (breaks (fun () -> Rig.Point.wait first));
  still_works d p ~last:first

(* A signal whose handler returns ends a wait's sleep early, as a system call
   that a signal interrupts returns: the wait sleeps again, and returns once
   its value is reached, the device live and nothing assigned. *)
let test_signal_in_wait () =
  let d, p = P.open_ "turn:signal-wait" in
  let first = point (empty_run d) in
  let handled = ref 0 in
  P.interrupt p;
  let before =
    Sys.signal Sys.sigint (Sys.Signal_handle (fun _ -> incr handled))
  in
  Fun.protect
    ~finally:(fun () -> Sys.set_signal Sys.sigint before)
    (fun () -> Rig.Point.wait first);
  equal ~msg:"handled" int 1 !handled;
  greater ~msg:"sleeps" int ~than:1 (P.sleeps p);
  still_works d p ~last:first

(* A thread waiting for room is interrupted: nothing is assigned to it. *)
let test_break_waiting_for_room () =
  let d, p = P.open_ ~capacity:1 "turn:break-room" in
  let base = Rig.submitted d in
  let first = point (one_part d) in
  equal int (base + 1) (Rig.Point.value first);
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
  still_works d p ~last:first

(* A close that Sys.Break interrupted in its wait leaves the device live, and
   the next close finishes it: the device is lost as closed, stopped once. *)
let test_break_in_close () =
  let d, p = P.open_ "turn:break-close" in
  let base = Rig.submitted d in
  let first = point (empty_run d) in
  equal int (base + 1) (Rig.Point.value first);
  P.interrupt p;
  equal bool true (breaks (fun () -> Rig.close d));
  equal (option string) ~msg:"after the interrupted close" None (Rig.lost d);
  Rig.close d;
  equal (option string) ~msg:"after the second close" (Some "closed")
    (Rig.lost d);
  equal int 1 (count "stop" p)

(* A close waiting for another thread's stop returns to OCaml at least every
   200 ms. The losing thread's stop parks outside OCaml for up to 5 s, and a
   thread of no OCaml code raises SIGINT once the close blocks in its wait for
   the stop. The handler raises Sys.Break on the closing thread during the
   close, so the close takes Sys.Break with the stop still parked. A close
   whose wait never returns to OCaml could break only once the park ended,
   which the check that the stop is still parked catches. The next close
   finishes: the device keeps its loss's reason, stopped once. *)
let test_break_waiting_stop () =
  if not Support.sees_waits then
    skip ~reason:"the system shows no thread's waits" ();
  let d, p = P.open_ "turn:break-stop" in
  P.before p "stop" Support.park;
  P.fail p;
  let loser =
    spawn (fun () ->
        match submit (empty d) with _ -> 0 | exception Rig.Lost _ -> 1)
  in
  Support.await "a parked stop" Support.parked;
  let closer = Thread.id (Thread.self ()) and closing = Atomic.make true in
  let handle _ =
    if Thread.id (Thread.self ()) = closer && Atomic.get closing then
      raise Sys.Break
  in
  let close () =
    match Rig.close d with
    | () ->
        Atomic.set closing false;
        false
    | exception Sys.Break ->
        Atomic.set closing false;
        true
  in
  let before = Sys.signal Sys.sigint (Sys.Signal_handle handle) in
  let (broke, signalled), held =
    Fun.protect ~finally:(fun () -> Sys.set_signal Sys.sigint before)
    @@ fun () ->
    let r = Support.interrupt_in_wait close in
    (r, Support.parked ())
  in
  Support.unpark ();
  equal ~msg:"the signal" bool true signalled;
  equal ~msg:"the close" bool true broke;
  equal ~msg:"the stop parked as it broke" bool true held;
  equal ~msg:"the loser" outcome (Ok 1) (loser ());
  Rig.close d;
  equal (option string) (Some "the submission failed") (Rig.lost d);
  equal int 1 (count "stop" p)

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
        test "a signal whose handler returns ends no wait" test_signal_in_wait;
        test "Sys.Break while waiting for room assigns no value"
          test_break_waiting_for_room;
        test "a close Sys.Break interrupted finishes at the next close"
          test_break_in_close;
        test "a close waiting for another thread's stop takes Sys.Break"
          test_break_waiting_stop;
      ];
  ]

let () = exit (run "rig.turn" tests)
