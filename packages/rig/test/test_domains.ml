(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Domains: the locks a waiting function holds, what interrupts a wait, and
   calls made inside a driver call. Threads here are systhreads of the test's
   domain, so a lock or a wait that kept the domain lock would stop the test. *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support

let submit ?(buffers = [||]) ?(args = [||]) ?waits s =
  Rig.submit ?waits s ~buffers ~args

(* The values of a submission's one fill, which reads none. *)
let one = Support.zeros 1
let lost d = function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false
let count call p = List.length (List.filter (( = ) call) (P.log p))
let empty d = Sub.make d [||]
let names = Atomic.make 0

let fresh what =
  Printf.sprintf "domains:%s-%d" what (Atomic.fetch_and_add names 1)

(* Submissions of one part that holds one unit of a Polled queue, and of none,
   each with its values. *)
let one_part d = (Sub.make d [| Sub.fill Support.noop ~params:0 |], one)
let no_part d = (empty d, [||])
let point (s, args) = Rig.submit s ~buffers:[||] ~args
let value x = Rig.Point.value (point x)

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

(* What [Point.reached] answers. *)
let reach =
  Testable.make
    ~pp:(fun ppf r ->
      Format.pp_print_string ppf (if r then "reached" else "unreached"))
    ~equal:Bool.equal

(* Whether an outcome is [d]'s loss. *)
let lost_outcome d =
  satisfies ~claim:"Error (Lost of the device)" outcome (function
    | Error e -> lost d e
    | Ok _ -> false)

(* Locks *)

(* A submit blocked in its driver holds its device, and no other device. *)
let test_turn_per_device () =
  let d, p = P.open_ ~capacity:1 ~may_block:true (fresh "blocking") in
  let base = Rig.submitted d in
  let e, _ = P.open_ (fresh "other") in
  let base_e = Rig.submitted e in
  equal int (base + 1) (value (one_part d));
  let blocked = spawn (fun () -> value (one_part d)) in
  Support.await "a blocked submit" (fun () -> P.blocked p = 1);
  equal int (base_e + 1) (value (no_part e));
  ignore (P.run p);
  equal outcome (Ok (base + 2)) (blocked ())

(* Two threads of one domain submit to a device whose driver blocks; the domain
   runs the device's queue meanwhile. *)
let test_two_threads () =
  let d, p = P.open_ ~capacity:1 ~may_block:true (fresh "two-threads") in
  let base = Rig.submitted d in
  equal int (base + 1) (value (one_part d));
  let a = spawn (fun () -> value (one_part d)) in
  let b = spawn (fun () -> value (one_part d)) in
  Support.await "a blocked submit" (fun () -> P.blocked p = 1);
  ignore (P.run p);
  Support.await "the second blocked submit" (fun () ->
      P.blocked p = 1 && P.queued p = 1);
  ignore (P.run p);
  equal (slist outcome compare) [ Ok (base + 2); Ok (base + 3) ] [ a (); b () ];
  equal int (base + 3) (Rig.submitted d)

(* The submit raises Lost once it returned, and the device's stop runs at its
   close. *)
let test_loss_while_blocked () =
  let d, p = P.open_ ~capacity:1 ~may_block:true (fresh "loss") in
  let base = Rig.submitted d in
  let first = point (one_part d) in
  equal int (base + 1) (Rig.Point.value first);
  let blocked = spawn (fun () -> value (one_part d)) in
  Support.await "a blocked submit" (fun () -> P.blocked p = 1);
  P.fault p "the engine hung";
  raises_match (lost d) (fun () -> Rig.Point.wait first);
  equal ~msg:"submits blocked after the wait" int 1 (P.blocked p);
  equal ~msg:"stops after the wait" int 0 (count "stop" p);
  ignore (P.run p);
  lost_outcome d (blocked ());
  equal ~msg:"stops before the close" int 0 (count "stop" p);
  Rig.close d;
  equal ~msg:"stops after the close" int 1 (count "stop" p)

(* A submit that blocks in its driver holds the device: a wait for an earlier
   value goes on without its commit and returns once the work runs. *)
let test_wait_beside_blocked_submit () =
  let d, p = P.open_ ~lag:4 ~capacity:1 ~may_block:true (fresh "blocked") in
  let first = point (one_part d) in
  let second = spawn (fun () -> value (one_part d)) in
  Support.await "the second submit blocked" (fun () -> P.blocked p = 1);
  Rig.Point.wait first;
  equal outcome (Ok (Rig.Point.value first + 1)) (second ());
  Rig.Point.wait (Rig.Point.submitted d)

let software what = Result.get_ok (Rig.software (fresh what))

(* A buffer of two zeroed words on [m], and its host address. *)
let two_words m =
  let b = B.create m 16 in
  let a = Support.Reader.host b in
  Support.store a 0;
  Support.store (a + 8) 0;
  (b, a)

(* Work on [m] that runs the fill [f] over the buffer of slot 0. *)
let fill m f =
  Sub.make m [| Sub.fill ~refs:[| Sub.write 0 ~at:0 |] f ~params:8 |]

(* [f ()] while [Support.waiting] runs on [m] on a domain of its own: [f]'s
   result and the held work's point. *)
let while_held m f =
  let hold, h = two_words m in
  let holder =
    Domain.spawn (fun () ->
        submit ~args:one (fill m Support.waiting) ~buffers:[| hold |])
  in
  let release () =
    Support.store h 1;
    Domain.join holder
  in
  match
    Support.await "the held work running" (fun () -> Support.load (h + 8) = 1);
    f ()
  with
  | r -> (r, release ())
  | exception e ->
      ignore (release ());
      raise e

(* A domain that submits [f ()] and the thread it submits on, once it has
   published it. *)
let submitter f =
  let thread = Atomic.make None in
  let d =
    Domain.spawn (fun () ->
        Atomic.set thread (Some (Support.self ()));
        f ())
  in
  Support.await "the submitter's thread" (fun () -> Atomic.get thread <> None);
  (d, Option.get (Atomic.get thread))

(* Another domain's submit waits before it takes its value, and its work runs
   after the held work. *)
let test_memory_run_holds_submits () =
  if not Support.sees_waits then
    skip ~reason:"the system shows no thread's waits" ();
  let m = software "memory-run" in
  let base = Rig.submitted m in
  let count, c = two_words m in
  let (during, other), held =
    while_held m (fun () ->
        let other, t =
          submitter (fun () ->
              submit ~args:one (fill m Support.bump) ~buffers:[| count |])
        in
        Support.await "the other submit waiting" (fun () ->
            Support.waits_in_condition t);
        ((Rig.submitted m, Support.load c), other))
  in
  let p = Domain.join other in
  equal ~msg:"while held: the last value, the other's work" (pair int int)
    (base + 1, 0)
    during;
  equal ~msg:"the held value" int (base + 1) (Rig.Point.value held);
  equal ~msg:"the other's value" int (base + 2) (Rig.Point.value p);
  equal ~msg:"the other's work" int 1 (Support.load c)

(* Another domain's waits for the device's earlier work return, and so does a
   submit on another device that waits for it. *)
let test_memory_waits_beside_run () =
  let m = software "memory-waits" in
  let base = Rig.submitted m in
  let count, _ = two_words m in
  let first = submit ~args:one (fill m Support.bump) ~buffers:[| count |] in
  let e, _ = P.open_ (fresh "memory-waits-other") in
  let during, held =
    while_held m (fun () ->
        Rig.Point.wait first;
        B.wait count Read_write;
        let after = submit (empty e) ~waits:[| first |] in
        Rig.Point.wait after;
        (Rig.Point.reached first, Rig.signaled m))
  in
  equal ~msg:"reached and signaled while held" (pair reach int)
    (true, base + 1)
    during;
  equal ~msg:"the held value" int (base + 2) (Rig.Point.value held)

(* Waits *)

(* How [f ()] ended: ["Sys.Break"], or ["returned"]. Windtrap's [raises] lets
   [Sys.Break] through, as it ends a run. *)
let breaks f =
  Sys.catch_break true;
  Fun.protect
    ~finally:(fun () -> Sys.catch_break false)
    (fun () ->
      match f () with _ -> "returned" | exception Sys.Break -> "Sys.Break")

(* After [Sys.Break] the device is used as before, and once lost its stop runs
   once at its close: the interrupted call left nothing counted in flight. *)
let still_works d p ~last =
  equal ~msg:"the loss" (option string) None (Rig.lost d);
  equal ~msg:"the submitted value" int (Rig.Point.value last) (Rig.submitted d);
  Rig.Point.wait last;
  equal ~msg:"the next value" int (Rig.Point.value last + 1) (value (no_part d));
  P.fail p;
  raises_match (lost d) (fun () -> submit (empty d));
  Rig.close d;
  equal ~msg:"stops" int 1 (count "stop" p)

let test_break_in_wait () =
  let d, p = P.open_ (fresh "break-wait") in
  let first = point (no_part d) in
  P.interrupt p;
  equal ~msg:"the wait" string "Sys.Break"
    (breaks (fun () -> Rig.Point.wait first));
  still_works d p ~last:first

let test_break_in_wait_for () =
  let d, p = P.open_ (fresh "break-wait-for") in
  let first = point (no_part d) in
  P.interrupt p;
  equal ~msg:"the wait" string "Sys.Break"
    (breaks (fun () -> Rig.Point.wait_for ~ns:1_000_000_000 first));
  still_works d p ~last:first

(* A signal whose handler returns ends a wait's sleep early, as a system call
   that a signal interrupts returns: the wait sleeps again, and returns once its
   value is reached, the device live and nothing assigned. *)
let test_signal_in_wait () =
  let d, p = P.open_ (fresh "signal-wait") in
  let first = point (no_part d) in
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

(* A thread waiting for room, its sleep held at the gate, takes Sys.Break there,
   and the test's own thread none. *)
let test_break_waiting_for_room () =
  let d, p = P.open_ ~capacity:1 (fresh "break-room") in
  let first = point (one_part d) in
  P.gate p;
  let own =
    breaks (fun () ->
        let waiting = spawn (fun () -> value (one_part d)) in
        Support.await "a sleep at the gate" (fun () -> P.sleepers p = 1);
        P.interrupt p;
        P.open_gate p;
        equal ~msg:"the waiting submit" outcome (Error Sys.Break) (waiting ()))
  in
  equal ~msg:"the test's own thread" string "returned" own;
  still_works d p ~last:first

(* A submit waiting for another domain's work on a software device takes
   Sys.Break, and nothing is assigned to it. *)
let test_break_waiting_for_run () =
  if not Support.sees_waits then
    skip ~reason:"the system shows no thread's waits" ();
  let m = software "break-run" in
  let base = Rig.submitted m in
  let count, c = two_words m in
  let bump () = submit ~args:one (fill m Support.bump) ~buffers:[| count |] in
  let (broke, signalled), held =
    while_held m (fun () ->
        Support.interrupt_in_wait (fun () -> breaks (fun () -> bump ())))
  in
  satisfies ~claim:"SIGINT raised in the wait" bool Fun.id signalled;
  equal ~msg:"the submit" string "Sys.Break" broke;
  equal ~msg:"the held value" int (base + 1) (Rig.Point.value held);
  equal ~msg:"after the held work" int (base + 1) (Rig.submitted m);
  equal ~msg:"the next value" int (base + 2) (Rig.Point.value (bump ()));
  equal ~msg:"the work run" int 1 (Support.load c)

(* A close that Sys.Break interrupted in its wait leaves the device live, and
   the next close finishes it: the device is lost as closed, stopped once. *)
let test_break_in_close () =
  let d, p = P.open_ (fresh "break-close") in
  ignore (point (no_part d));
  P.interrupt p;
  equal ~msg:"the close" string "Sys.Break" (breaks (fun () -> Rig.close d));
  equal (option string) ~msg:"after the interrupted close" None (Rig.lost d);
  Rig.close d;
  equal (option string) ~msg:"after the second close" (Some "closed")
    (Rig.lost d);
  equal ~msg:"stops" int 1 (count "stop" p)

(* A close waiting for another thread's stop returns to OCaml at least every 200
   ms. The losing thread closes the device, whose stop parks outside OCaml until
   unparked, and a thread of no OCaml code raises SIGINT once the close blocks
   in its wait for the stop. The handler raises Sys.Break on the closing thread
   during the close, so the close takes Sys.Break with the stop still parked. A
   close whose wait never returns to OCaml could break only once the park ended,
   which the check that the stop is still parked catches. The next close
   finishes: the device keeps its loss's reason, stopped once. *)
let test_break_waiting_stop () =
  if not Support.sees_waits then
    skip ~reason:"the system shows no thread's waits" ();
  let d, p = P.open_ (fresh "break-stop") in
  P.before p "stop" Support.park;
  P.fail p;
  let loser =
    spawn (fun () ->
        match submit (empty d) with
        | _ -> 0
        | exception Rig.Lost _ ->
            Rig.close d;
            1)
  in
  let (broke, signalled), parked =
    Fun.protect ~finally:Support.unpark @@ fun () ->
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
          "returned"
      | exception Sys.Break ->
          Atomic.set closing false;
          "Sys.Break"
    in
    let before = Sys.signal Sys.sigint (Sys.Signal_handle handle) in
    Fun.protect ~finally:(fun () -> Sys.set_signal Sys.sigint before)
    @@ fun () ->
    let r = Support.interrupt_in_wait close in
    (r, Support.parked ())
  in
  satisfies ~claim:"SIGINT raised in the wait" bool Fun.id signalled;
  equal ~msg:"the close" string "Sys.Break" broke;
  satisfies ~claim:"the stop parked as the close broke" bool Fun.id parked;
  equal ~msg:"the loser" outcome (Ok 1) (loser ());
  Rig.close d;
  equal (option string) (Some "the submission failed") (Rig.lost d);
  equal int 1 (count "stop" p)

(* Inside a driver call *)

(* What a close of [d] answers: ["refused"] for the [Invalid_argument] of
   [Rig.close]. *)
let close_answer d =
  match Rig.close d with
  | () -> "returned"
  | exception Invalid_argument why
    when String.starts_with ~prefix:"Rig.close: " why ->
      "refused"

(* Waits for [pt], whose device's next sleep raises SIGINT inside its wait, with
   [handle] as SIGINT's handler. *)
let wait_handling handle pt =
  let before = Sys.signal Sys.sigint (Sys.Signal_handle handle) in
  Fun.protect
    ~finally:(fun () -> Sys.set_signal Sys.sigint before)
    (fun () -> Rig.Point.wait pt)

(* A close from a signal's handler that lands inside a wait on the closed
   device, on the waiting thread, as a test runner's release of its fixtures on
   SIGTERM does, raises Invalid_argument and leaves the device as it was: it
   would wait for the wait it interrupted. The wait then returns, and a close
   after it ends the device, stopped once. *)
let test_close_in_wait () =
  let d, p = P.open_ (fresh "close-in-wait") in
  let first = point (no_part d) in
  let closed = ref None in
  P.interrupt p;
  wait_handling (fun _ -> closed := Some (close_answer d)) first;
  equal ~msg:"the close in the handler" (option string) (Some "refused") !closed;
  equal ~msg:"the loss before the last close" (option string) None (Rig.lost d);
  Rig.close d;
  equal ~msg:"the loss" (option string) (Some "closed") (Rig.lost d);
  equal ~msg:"stops" int 1 (count "stop" p)

(* The same close while a profile is open, with the device's work recorded in
   it, refuses before it reads the profile: it waits for nothing, so the work
   its handler interrupted is still unreached once it refused. *)
let test_close_in_profiled_wait () =
  let d, p = P.open_ (fresh "close-in-profiled-wait") in
  let reached = ref [] and closed = ref None in
  let (), _ =
    Rig.Profile.take (fun () ->
        let pt = point (one_part d) in
        let handle _ =
          let before = Rig.Point.reached pt in
          closed := Some (close_answer d);
          reached := [ before; Rig.Point.reached pt ]
        in
        P.interrupt p;
        wait_handling handle pt)
  in
  equal ~msg:"the close in the handler" (option string) (Some "refused") !closed;
  equal ~msg:"the work before and after the close" (list reach) [ false; false ]
    !reached;
  equal ~msg:"the loss before the last close" (option string) None (Rig.lost d);
  Rig.close d;
  equal ~msg:"stops" int 1 (count "stop" p)

(* A close from a callback inside a device's stop, as a finaliser that lands
   there may make, raises Invalid_argument whatever the device: a close of the
   stopping device would wait for that stop. The close that runs the stop
   returns, the device stopped once, and the other device stays open. *)
let test_close_in_stop () =
  let d, p = P.open_ (fresh "close-in-stop") in
  let other = software "close-in-stop-other" in
  let answers = ref [] in
  P.before p "stop" (fun () ->
      answers := [ close_answer other; close_answer d ]);
  Rig.close d;
  equal ~msg:"closes in the stop" (list string) [ "refused"; "refused" ]
    !answers;
  equal ~msg:"the loss" (option string) (Some "closed") (Rig.lost d);
  equal ~msg:"stops" int 1 (count "stop" p);
  equal ~msg:"the other device's loss" (option string) None (Rig.lost other);
  Rig.close other

let tests =
  [
    group "locks"
      [
        test "a submit blocked in its driver holds only its device"
          test_turn_per_device;
        test
          "a submit blocked in its driver lets another thread of its domain \
           run the device's queue"
          test_two_threads;
        test "a loss during a blocked submit stops the device after it returns"
          test_loss_while_blocked;
        test "a wait beside a submit blocked in its driver returns"
          test_wait_beside_blocked_submit;
        test
          "a submit on a software device waits for another domain's running \
           work before its value"
          test_memory_run_holds_submits;
        test "waits on a software device go on while its work runs"
          test_memory_waits_beside_run;
      ];
    group "waits"
      [
        test "Sys.Break in a wait loses nothing" test_break_in_wait;
        test "Sys.Break in a bounded wait loses nothing" test_break_in_wait_for;
        test "a signal whose handler returns ends no wait" test_signal_in_wait;
        test "Sys.Break while waiting for room assigns no value"
          test_break_waiting_for_room;
        test
          "Sys.Break while waiting for a software device's run assigns no value"
          test_break_waiting_for_run;
        test "a close Sys.Break interrupted finishes at the next close"
          test_break_in_close;
        test "a close waiting for another thread's stop takes Sys.Break"
          test_break_waiting_stop;
      ];
    group "inside a driver call"
      [
        test "a close from a handler inside a wait on its device raises"
          test_close_in_wait;
        test
          "a close from a handler inside a profiled wait raises before it \
           reads the profile"
          test_close_in_profiled_wait;
        test "a close from a callback inside a stop raises" test_close_in_stop;
      ];
  ]

let () = exit (run "rig.domains" tests)
