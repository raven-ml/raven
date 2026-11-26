(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Points on timelines, on Polled devices, whose queues run only when a wait
   sleeps or the test runs them. *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support

let empty d = Sub.make d [||]
let submit ?waits s = Rig.submit ?waits s ~buffers:[||] ~args:[||]
let lost d = function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false
let device = Testable.make ~pp:Rig.pp ~equal:Rig.equal
let names = Atomic.make 0

let fresh what =
  Printf.sprintf "point:%s-%d" what (Atomic.fetch_and_add names 1)

(* What [reached] and [wait_for] answer. *)
let reach =
  Testable.make
    ~pp:(fun ppf r ->
      Format.pp_print_string ppf (if r then "reached" else "unreached"))
    ~equal:Bool.equal

(* rig's still interval, the most one sleep of a wait lasts (rig.mli's Domains:
   a wait returns to OCaml at least every 200 ms). *)
let still_ns = 200_000_000

(* Devices past slot 32,767 *)

(* The first device slot whose points set the top bit of an OCaml int. *)
let high = 32_768

(* Opens [high] devices and keeps them open: slots pass on only from closed
   devices, so every device opened after them lies at slot [high] or above. *)
let fillers =
  fixture (fun () ->
      for _ = 1 to high do
        ignore (P.open_ (fresh "filler"))
      done)

(* A device lost with a point it reached and a point its loss dropped, then
   closed, which frees its slot for the next device. *)
let lost_points name =
  let d, p = P.open_ name in
  let reached = submit (empty d) in
  Rig.Point.wait reached;
  P.fault p (name ^ " faulted");
  let dropped = submit (empty d) in
  (try Rig.Point.wait dropped with Rig.Lost _ -> ());
  Rig.close d;
  (d, reached, dropped)

(* [a]'s points, of a lost device whose slot a later device [b] took, with [b]'s
   reached point. The process's first loss keeps its slot, so one comes first;
   slots are taken last freed first. *)
type passed_on = {
  a : Rig.t;
  reached : Rig.Point.t;
  dropped : Rig.Point.t;
  b_reached : Rig.Point.t;
}

let passed_on () =
  ignore (lost_points (fresh "first"));
  let a, reached, dropped = lost_points (fresh "a") in
  let _, b_reached, _ = lost_points (fresh "b") in
  equal ~msg:"the slot passed on" (option device) None
    (Rig.Point.device reached);
  { a; reached; dropped; b_reached }

(* Point.device *)

let test_device_until_passed () =
  ignore (lost_points (fresh "first"));
  let a, reached, _ = lost_points (fresh "a") in
  Gc.full_major ();
  equal ~msg:"before its slot passes on" (option device) (Some a)
    (Rig.Point.device reached);
  let b, b_reached, _ = lost_points (fresh "b") in
  not_equal ~msg:"the device on its slot" device a b;
  equal ~msg:"once its slot passed on" (option device) None
    (Rig.Point.device reached);
  equal ~msg:"its device's loss once its slot passed on" (option string)
    (Some (Rig.name a ^ " faulted"))
    (Rig.lost a);
  equal ~msg:"the later device's point" (option device) (Some b)
    (Rig.Point.device b_reached)

(* Point *)

(* Each test opens its devices past slot 32,767, whose points set the top bit of
   an OCaml int. *)

(* A submit that waits for a point its lost device did not reach raises the
   loss. *)
let test_high_lost () =
  fillers ();
  let d, p = P.open_ ~answer:`Unknown (fresh "lost") in
  let c, _ = P.open_ (fresh "lost-consumer") in
  let a = submit (empty d) in
  P.fail p;
  raises_match ~msg:"the failed submit" (lost d) (fun () -> submit (empty d));
  raises_match ~msg:"a submit after the point" (lost d) (fun () ->
      submit (empty c) ~waits:[| a |])

(* Rule 5, host access: a host read follows every device's last write, so the
   copy has run once [Buffer.wait] returned. *)
let test_high_read () =
  fillers ();
  let d, p = P.open_ (fresh "write") in
  let src = B.create d 8 and dst = B.create d 8 in
  let copy = Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst)) in
  ignore (submit (Sub.make d [| copy |]));
  B.wait dst Read;
  equal int ~msg:"queued once the read returned" 0 (P.queued p)

(* A point of the submitting device in [waits] follows by device order: the
   submit waits on the host for nothing, so both submissions stay queued. *)
let test_high_own () =
  fillers ();
  let d, p = P.open_ (fresh "own") in
  let a = submit (empty d) in
  ignore (submit (empty d) ~waits:[| a |]);
  equal int ~msg:"queued" 2 (P.queued p)

(* Point.wait *)

let test_wait_passed_on () =
  let t = passed_on () in
  Rig.Point.wait t.reached;
  Rig.Point.wait t.b_reached

let test_wait_dropped () =
  let t = passed_on () in
  raises_match
    (function
      | Rig.Lost (d', why) ->
          Rig.equal t.a d' && why = Rig.name t.a ^ " faulted"
      | _ -> false)
    (fun () -> Rig.Point.wait t.dropped)

(* Point.reached *)

(* A driver that commits once [lag] values are uncommitted: values below [lag]
   run only once a wait commits them. *)
let lag = 4

let test_reached_commits () =
  let d, _ = P.open_ ~lag ~runs:`Itself (fresh "reached") in
  let points = List.init (lag - 1) (fun _ -> submit (empty d)) in
  let last = List.nth points (lag - 2) in
  Support.await "the last point reached" (fun () -> Rig.Point.reached last);
  equal ~msg:"signaled" int (Rig.Point.value last) (Rig.signaled d)

(* While a submit blocks in its driver for room, holding the device, reached of
   an earlier point waits for it at most a still interval, then answers: the
   submit is still blocked, and nothing ran the queue. *)
let test_reached_beside_blocked () =
  let d, p = P.open_ ~lag ~capacity:1 ~may_block:true (fresh "blocked") in
  let one = Sub.make d [| Sub.fill Support.noop ~params:0 |] in
  let submit_one () = Rig.submit one ~buffers:[||] ~args:(Support.zeros 1) in
  let first = submit_one () in
  let t = Thread.create (fun () -> ignore (submit_one ())) () in
  Support.await "the second submit blocked" (fun () -> P.blocked p = 1);
  equal ~msg:"the first point" reach false (Rig.Point.reached first);
  equal ~msg:"submits blocked once it answered" int 1 (P.blocked p);
  ignore (P.run p);
  Thread.join t

let test_reached_passed_on () =
  let t = passed_on () in
  equal (pair reach reach) (true, true)
    (Rig.Point.reached t.reached, Rig.Point.reached t.b_reached)

(* Point.wait_for *)

(* The times at which [p]'s sleeps start from now on, latest first. *)
let sleep_starts p =
  let starts = ref [] in
  let rec hook () =
    P.before p "sleep" (fun () ->
        starts := Rig.Profile.now () :: !starts;
        hook ())
  in
  hook ();
  starts

(* On work that runs long, a bounded wait answers false no sooner than [ns]
   after the call, and at most one round after it: a round sleeps once, so at
   most one sleep starts after [ns]. *)
let wait_for_late ns =
  cover "under a millisecond" (ns < 1_000_000);
  cover "a millisecond or more" (ns >= 1_000_000);
  let d, p = P.open_ (fresh "late") in
  let pt = submit (empty d) in
  P.stall p max_int;
  let starts = sleep_starts p in
  let t0 = Rig.Profile.now () in
  let answer = Rig.Point.wait_for ~ns pt in
  let waited = Rig.Profile.now () - t0 in
  let late = List.filter (fun t -> t >= t0 + ns) !starts in
  equal ~msg:"the answer" reach false answer;
  at_least ~msg:"the wait, in ns" int ~than:ns waited;
  at_most ~msg:"sleeps started after ns" int ~than:1 (List.length late);
  P.stall p 0;
  Rig.close d

(* 1 ns answers before any sleep; 50 ms spans a fraction of a still interval. *)
let late_examples = [ 1; 999_999; 1_000_000; 50_000_000 ]

let late_ns =
  Gen.one_of [ Gen.int_range 1 999_999; Gen.int_range 1_000_000 50_000_000 ]

(* Once its stall ends a wait reaches the work, and nothing was lost. At 1 ns
   the false comes before any sleep, the clock past [ns] at the first look. *)
let test_runs_on ns =
  let d, p = P.open_ (fresh "runs-on") in
  let pt = submit (empty d) in
  P.stall p max_int;
  equal ~msg:"the first answer" reach false (Rig.Point.wait_for ~ns pt);
  P.stall p 0;
  equal ~msg:"once the stall ends" reach true
    (Rig.Point.wait_for ~ns:1_000_000_000 pt);
  equal (option string) ~msg:"the loss" None (Rig.lost d);
  Rig.close d

(* An [ns] past the clock's range waits as [wait] does, through [stalls] sleeps
   that each run nothing, and no sooner than they took. A wait whose time left
   overflowed would answer before its last sleep. *)
let test_wait_for_unbounded (ns, stalls) =
  let d, p = P.open_ (fresh "unbounded") in
  let pt = submit (empty d) in
  P.stall p stalls;
  let t0 = Rig.Profile.now () in
  let answer = Rig.Point.wait_for ~ns pt in
  let waited = Rig.Profile.now () - t0 in
  equal ~msg:"the answer" reach true answer;
  at_least ~msg:"the wait, in ns" int ~than:(stalls * still_ns) waited;
  equal ~msg:"sleeps" int (stalls + 1) (P.sleeps p);
  Rig.close d

(* For a point its lost device never reached, and not for one reached before the
   loss. *)
let test_wait_for_lost ns =
  let d, _ = P.open_ (fresh "wait-for-lost") in
  let a = submit (empty d) in
  Rig.Point.wait a;
  let b = submit (empty d) in
  Rig.lose d "given up";
  raises_match ~msg:"wait, on the dropped point" (lost d) (fun () ->
      Rig.Point.wait b);
  raises_match ~msg:"wait_for, on the dropped point" (lost d) (fun () ->
      Rig.Point.wait_for ~ns b);
  Rig.Point.wait a;
  equal ~msg:"wait_for, on the point reached before the loss" reach true
    (Rig.Point.wait_for ~ns a);
  Rig.close d

let test_wait_for_lost_during () =
  let d, p = P.open_ (fresh "lost-during") in
  let pt = submit (empty d) in
  P.fault p "the engine hung";
  raises_match (lost d) (fun () -> Rig.Point.wait_for ~ns:1_000_000_000 pt);
  equal (option string) ~msg:"the loss" (Some "the engine hung") (Rig.lost d);
  Rig.close d

(* The point waits as before once outside the lock. *)
let test_wait_for_in_lock ns =
  let d, _ = P.open_ (fresh "in-lock") in
  let pt = submit (empty d) in
  raises_match Exn.invalid_arg (fun () ->
      Support.inside_lock (fun () -> Rig.Point.wait_for ~ns pt));
  Rig.Point.wait pt;
  Rig.close d

(* Twin devices in one state answer alike and leave the same committed work, one
   to [wait_for] with no time and one to [reached]. *)
type state = Reached | Committed | Uncommitted | Lost_before

let pp_state ppf s =
  Format.pp_print_string ppf
    (match s with
    | Reached -> "reached"
    | Committed -> "committed"
    | Uncommitted -> "uncommitted"
    | Lost_before -> "lost")

(* A device in [state], whose one point is its first submit's: its driver
   commits each value as it takes it, but for [Uncommitted], and runs only what
   a sleep or the test runs. *)
let in_state state =
  let lag = match state with Uncommitted -> 2 | _ -> 1 in
  let d, p = P.open_ ~lag (fresh "no-time") in
  let pt = submit (empty d) in
  (match state with
  | Reached -> ignore (P.run p)
  | Committed | Uncommitted -> ()
  | Lost_before -> Rig.lose d "given up");
  (d, p, pt)

let answered f pt =
  match f pt with r -> Ok r | exception Rig.Lost (_, why) -> Error why

let no_time (state, ns) =
  cover "reached" (state = Reached);
  cover "committed" (state = Committed);
  cover "uncommitted" (state = Uncommitted);
  cover "lost" (state = Lost_before);
  let d, p, pt = in_state state in
  let twin, twin_p, twin_pt = in_state state in
  let got = answered (Rig.Point.wait_for ~ns) pt in
  let want = answered Rig.Point.reached twin_pt in
  equal (result reach string) want got;
  equal ~msg:"the work that runs after" int (P.run twin_p) (P.run p);
  Rig.close d;
  Rig.close twin

let no_time_gen =
  Gen.pair
    (Gen.of_list ~pp:pp_state [ Reached; Committed; Uncommitted; Lost_before ])
    (Gen.one_of
       [ Gen.constant 0; Gen.int_range (-1_000_000) (-1); Gen.constant min_int ])

(* After one call, whatever it answered, the device runs the work with no other
   wait. A lag of 2 never commits a lone value on its own, and the driver's
   thread runs what is committed. *)
let bounded_commits ns =
  cover "one ns" (ns = 1);
  let d, _ = P.open_ ~lag:2 ~runs:`Itself (fresh "bounded-commits") in
  let pt = submit (empty d) in
  ignore (Rig.Point.wait_for ~ns pt : bool);
  Support.await "the work run" (fun () -> Rig.signaled d >= Rig.Point.value pt);
  Rig.close d

let test_short_waits_end () =
  let d, _ = P.open_ ~lag:2 ~runs:`Itself (fresh "short-waits") in
  let pt = submit (empty d) in
  Support.await "a short wait's true" (fun () -> Rig.Point.wait_for ~ns:1 pt);
  Rig.close d

(* Point.submitted *)

(* With nothing submitted it is reached at once, of no device. *)
let submitted_point n =
  cover "nothing submitted" (n = 0);
  let d, _ = P.open_ (fresh "submitted") in
  let points = List.init n (fun _ -> submit (empty d)) in
  let s = Rig.Point.submitted d in
  if n = 0 then begin
    equal ~msg:"its device" (option device) None (Rig.Point.device s);
    equal ~msg:"its answer" reach true (Rig.Point.reached s)
  end
  else begin
    equal ~msg:"its device" (option device) (Some d) (Rig.Point.device s);
    equal ~msg:"its value" int (Rig.submitted d) (Rig.Point.value s)
  end;
  Rig.Point.wait s;
  equal ~msg:"the earlier points" (list reach)
    (List.map (fun _ -> true) points)
    (List.map Rig.Point.reached points);
  Rig.close d

(* The host's work is done when its submit returns: before and after a
   submit. *)
let test_submitted_host () =
  let before = Rig.Point.submitted Rig.host in
  equal ~msg:"before a submit" reach true (Rig.Point.reached before);
  Rig.Point.wait before;
  ignore (submit (Sub.make Rig.host [||]));
  let after = Rig.Point.submitted Rig.host in
  equal ~msg:"its value" int (Rig.submitted Rig.host) (Rig.Point.value after);
  equal ~msg:"after a submit" reach true (Rig.Point.reached after);
  equal ~msg:"wait_for" reach true (Rig.Point.wait_for ~ns:1 after)

(* Point.pp *)

let printed p = Format.asprintf "%a" Rig.Point.pp p

let test_pp () =
  let d = Result.get_ok (Rig.software (fresh "pp")) in
  let p = submit (Sub.make d [||]) in
  equal string
    (Printf.sprintf "%s:%d" (Rig.name d) (Rig.Point.value p))
    (printed p)

let test_pp_ended () =
  let t = passed_on () in
  equal string
    (Printf.sprintf "ended:%d" (Rig.Point.value t.reached))
    (printed t.reached)

let tests =
  [
    group "Point"
      [
        test
          "a submit waiting for a point its device lost before reaching raises \
           Lost, past slot 32,767"
          test_high_lost;
        test "Buffer.wait for a read waits for the last write, past slot 32,767"
          test_high_read;
        test
          "a submit waiting for its own device's point waits for nothing on \
           the host, past slot 32,767"
          test_high_own;
      ];
    group "Point.device"
      [
        test "a point names its device until a later device takes its slot"
          test_device_until_passed;
      ];
    group "Point.wait"
      [
        test "a point reached before a loss returns once its slot passed on"
          test_wait_passed_on;
        test "a point a loss dropped raises that loss once its slot passed on"
          test_wait_dropped;
      ];
    group "Point.reached"
      [
        test "reached commits as wait does, so a loop on it ends"
          test_reached_commits;
        test
          "reached answers while a submit blocked in its driver holds the \
           device"
          test_reached_beside_blocked;
        test "a point reached before a loss is reached once its slot passed on"
          test_reached_passed_on;
      ];
    group "Point.wait_for"
      [
        prop ~count:20 ~examples:late_examples
          "false comes no sooner than ns, and at most one round after" late_ns
          wait_for_late;
        cases ~id:string_of_int "after false the work runs on" late_examples
          test_runs_on;
        cases
          ~id:(fun (ns, stalls) -> Printf.sprintf "%d, %d stalled" ns stalls)
          "an ns past the clock's range waits as wait"
          [ (max_int, 1); (max_int - 1, 0); (max_int / 2, 0) ]
          test_wait_for_unbounded;
        cases ~id:string_of_int "a bounded wait raises Lost where wait does"
          [ min_int; 0; 1; 1_000_000; max_int ]
          test_wait_for_lost;
        test "a loss during a bounded wait raises it" test_wait_for_lost_during;
        cases ~id:string_of_int
          "a bounded wait inside a lock is refused, whatever ns"
          [ min_int; 0; 1; max_int ] test_wait_for_in_lock;
        prop ~count:60
          ~examples:
            [
              (Reached, 0);
              (Committed, 0);
              (Uncommitted, min_int);
              (Lost_before, -1);
            ]
          "with no time a bounded wait answers as reached" no_time_gen no_time;
        prop ~count:30 ~examples:[ 1 ] "a bounded wait commits before its time"
          (Gen.int_range 1 1_000_000)
          bounded_commits;
        test "a loop of the shortest bounded waits ends" test_short_waits_end;
      ];
    group "Point.submitted"
      [
        prop ~count:20 ~examples:[ 0; 1 ]
          "a wait on the submitted point reaches every earlier point"
          (Gen.int_range 0 8) submitted_point;
        test "the host's submitted point is reached at once" test_submitted_host;
      ];
    group "Point.pp"
      [
        test "a point prints as its device's name and its value" test_pp;
        test "a point prints as ended:v once its slot passed on" test_pp_ended;
      ];
  ]

let () = exit (run "rig.point" tests)
