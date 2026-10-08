(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Commits: work a driver holds until it is committed. The devices here are
   Polled devices that commit on their own every [lag] values and run only
   committed work: a wait that sleeps on uncommitted work raises Polled's
   [Failure], so every call returning shows that it committed what it waited
   for. *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module H = Rig.Hold
module P = Rig_support.Polled
module Support = Rig_support

let timeout = 60.
let lag = 4
let lost d = function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false
let empty d = Sub.make ~reads:0 ~writes:0 d [||]
let names = Atomic.make 0

let fresh what =
  Printf.sprintf "commit:%s-%d" what (Atomic.fetch_and_add names 1)

let submit ?(reads = [||]) ?(waits = [||]) s =
  Rig.submit s ~reads ~writes:[||] ~waits

(* The last value [p]'s word holds, read without a call of rig. *)
let word p = Support.load (Nativeint.to_int (P.self p))

(* Two devices and their order *)

(* Two devices: [a] with a queue of two parts, so that a third uncommitted
   submission waits for room, and [b], whose queue waits on [a]'s word. Each
   submission bumps its device's counter and reads its device's buffer, whose
   stamps waits and copies follow. A scratch is device memory under a small
   budget, which an allocation reclaims. *)
type device = {
  d : Rig.t;
  p : P.t;
  run : Sub.t;
  scratched : Sub.t;
  buffer : B.t;
  last : Rig.Point.t option Atomic.t;
}

type world = device array

let capacity = 2
let budget = 64 * 1024
let scratch_bytes = 40 * 1024

let bump () =
  let arg = B.create Rig.host 8 in
  {
    Sub.queue = "COMPUTE:0";
    after = [||];
    work =
      Sub.Fill { fill = Support.bump; arg; ring_units = 0; segment_bytes = 0 };
  }

let device (d, p) =
  {
    d;
    p;
    run = Sub.make ~reads:1 ~writes:0 d [| bump () |];
    scratched = Sub.make ~reads:2 ~writes:0 d [| bump () |];
    buffer = B.create d 8;
    last = Atomic.make None;
  }

let open_world () =
  let a = P.open_ ~lag ~capacity ~budget (fresh "a") in
  let b = P.open_ ~lag ~waits_on:[ `Host ] ~budget (fresh "b") in
  [| device a; device b |]

let use ?(scratch = []) x ~waits =
  let pt =
    match scratch with
    | [] -> submit x.run ~reads:[| x.buffer |] ~waits
    | s -> submit x.scratched ~reads:(Array.of_list (x.buffer :: s)) ~waits
  in
  Atomic.set x.last (Some pt)

let submit_system ~note w i =
  let x = w.(i) in
  note "a room wait" (i = 0 && P.queued x.p >= capacity);
  use x ~waits:[||]

let follow_system ~note w =
  let a = w.(0) and b = w.(1) in
  let waits =
    match Atomic.get a.last with
    | None -> [||]
    | Some pt ->
        note "a foreign point" (word a.p < Rig.Point.value pt);
        [| pt |]
  in
  use b ~waits

let wait_system w i =
  let x = w.(i) in
  let v = Rig.submitted x.d in
  Rig.wait x.d v;
  word x.p >= v

let scratch_system w i =
  let x = w.(i) in
  cover "an allocation over the budget"
    (P.allocated x.p `Device + scratch_bytes > budget);
  use x ~scratch:[ B.create x.d scratch_bytes ] ~waits:[||]

(* A program's devices stay open, as every drain walks the closed ones, and
   finish their work, which holds the host memory of their fills. *)
let finish w = Array.iter (fun x -> Rig.wait x.d (Rig.submitted x.d)) w
let world : (unit, world) abstract = abstract "w" ~release:finish
let index = Gen.int_range 0 1

(* The model holds nothing: every call returns, and a wait finds its value
   reached. *)
let commands ~note =
  [
    command "open" (Gen.unit @-> makes world) Fun.id open_world;
    command "submit"
      (world ^-> index @-> returns unit)
      (fun _ _ -> ())
      (submit_system ~note);
    command "follow"
      (world ^-> returns unit)
      (fun _ -> ())
      (follow_system ~note);
    command "wait"
      (world ^-> index @-> returns bool)
      (fun _ _ -> true)
      wait_system;
    command "buffer wait"
      (world ^-> index @-> returns unit)
      (fun _ _ -> ())
      (fun w i -> B.wait w.(i).buffer B.Read_write);
    command "copy"
      (world ^-> index @-> returns unit)
      (fun _ _ -> ())
      (fun w i -> B.copy ~src:w.(i).buffer ~dst:(B.create Rig.host 8));
  ]

let scratch =
  command "scratch"
    (world ^-> index @-> returns unit)
    (fun _ _ -> ())
    scratch_system

(* Law 7: a driver commits within its bound. *)

(* Submits once on [d] a submission naming a hold of [m] whose release sets
   [released], and drops both: the hold is unreachable once this returns. *)
let[@inline never] submit_held d m released =
  let h = H.make ~release:(fun () -> Atomic.set released true) [ m ] in
  ignore (submit (Sub.make ~hold:h ~reads:0 ~writes:0 d [||]))

(* A hold's release runs once its stamp is reached, in a drain: without a wait,
   only the driver's own commit reaches it. *)
let test_lag_bounded () =
  let d, _ = P.open_ ~lag ~runs:`Itself (fresh "bounded") in
  let m = B.create d 64 and released = Atomic.make false in
  submit_held d m released;
  for _ = 1 to lag do
    ignore (submit (empty d))
  done;
  Support.await "the hold's release" (fun () ->
      Gc.full_major ();
      ignore (Sys.opaque_identity (B.create d 8));
      Atomic.get released)

(* Polling [signaled] reaches values its driver has not committed on its own. *)
let test_signaled_commits () =
  let d, _ = P.open_ ~lag ~runs:`Itself (fresh "signaled") in
  for _ = 1 to lag - 1 do
    ignore (submit (empty d))
  done;
  Support.await "signaled reaching the last value" (fun () ->
      Rig.signaled d = lag - 1)

(* Loss *)

(* The device is lost by an allocation, which waits for nothing, so its stop
   finds the values uncommitted. *)
let test_lost_uncommitted () =
  let d, p = P.open_ ~lag (fresh "lost") in
  ignore (submit (empty d));
  ignore (submit (empty d));
  P.fault p "the engine hung";
  raises_match (lost d) (fun () -> B.create d 4096);
  raises_match (lost d) (fun () -> Rig.wait d 2);
  raises_match (lost d) (fun () -> Rig.wait d 1);
  equal ~msg:"the word after the stop" int 2 (Rig.signaled d)

(* A submit that blocks in its driver holds the turn: a wait for an earlier
   value goes on without its commit and returns once the work runs. *)
let test_wait_beside_blocked_submit () =
  let d, p = P.open_ ~lag ~capacity:1 ~may_block:true (fresh "blocked") in
  let one = Sub.make ~reads:0 ~writes:0 d [| bump () |] in
  let v = Rig.Point.value (submit one) in
  let t = Thread.create (fun () -> ignore (submit one)) () in
  Support.await "the second submit blocked" (fun () -> P.blocked p = 1);
  Rig.wait d v;
  Thread.join t;
  Rig.wait d (Rig.submitted d)

let test_commit_fails () =
  let d, p = P.open_ ~lag (fresh "commit-fails") in
  ignore (submit (empty d));
  P.fail_commit p;
  raises_match
    (function
      | Rig.Lost (d', why) -> Rig.equal d d' && why = "the commit failed"
      | _ -> false)
    (fun () -> Rig.wait d 1);
  equal (option string) (Some "the commit failed") (Rig.lost d)

let tests =
  [
    group ~timeout "order"
      [
        stateful "every wait commits what it waits for"
          (scratch :: commands ~note:cover);
        (* A branch's calls run on another domain, where nothing covers, and a
           reclamation's full collections are left to one domain. *)
        stateful ~count:30 ~domains:2
          "every wait commits what it waits for, on two domains"
          (commands ~note:(fun _ _ -> ()));
      ];
    group ~timeout "lag"
      [
        test "a driver commits within its bound" test_lag_bounded;
        test "polling signaled reaches every value" test_signaled_commits;
      ];
    group ~timeout "loss"
      [
        test "a device lost with uncommitted values raises Lost"
          test_lost_uncommitted;
        test "a wait beside a submit blocked in its driver returns"
          test_wait_beside_blocked_submit;
        test "a commit that fails loses the device" test_commit_fails;
      ];
  ]

let () = exit (run "rig.commit" tests)
