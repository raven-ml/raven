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
module P = Rig_support.Polled
module Support = Rig_support

let timeout = 60.
let lag = 4
let lost d = function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false
let empty d = Sub.make d [||]
let names = Atomic.make 0

let fresh what =
  Printf.sprintf "commit:%s-%d" what (Atomic.fetch_and_add names 1)

let submit ?(buffers = [||]) ?waits s =
  Rig.submit ?waits (Sub.Run.make s) ~buffers

(* The last value [p]'s word holds, read without a call of rig. *)
let word p = Support.load (P.word_at p)

(* Two devices and their order *)

(* Two devices: [a] with a queue of two parts, so that a third uncommitted
   submission waits for room, and [b], whose queue waits on [a]'s word. Each
   submission occupies one queue part and reads its device's buffer, whose
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

let idle () =
  (Sub.fill Support.noop ~params:0)

let device (d, p) =
  {
    d;
    p;
    run = Sub.make d [| Support.touching [| Sub.read |] |];
    scratched = Sub.make d [| Support.touching [| Sub.read; Sub.read |] |];
    buffer = B.create d 8;
    last = Atomic.make None;
  }

let open_world () =
  let a = P.open_ ~lag ~capacity ~budget (fresh "a") in
  let b = P.open_ ~lag ~waits_on:[ `Host ] ~budget (fresh "b") in
  [| device a; device b |]

let use ?(scratch = []) ?waits x =
  let s = if scratch = [] then x.run else x.scratched in
  let run = Sub.Run.make s in
  Sub.Run.groups run (Sub.block s 0) 0 0 0;
  let pt =
    Rig.submit ?waits run ~buffers:(Array.of_list (x.buffer :: scratch))
  in
  Atomic.set x.last (Some pt)

let submit_system ~note w i =
  let x = w.(i) in
  note "a room wait" (i = 0 && P.queued x.p >= capacity);
  use x

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
  match Atomic.get x.last with
  | None -> true
  | Some pt ->
      Rig.Point.wait pt;
      word x.p >= Rig.Point.value pt

let scratch_system w i =
  let x = w.(i) in
  cover "an allocation over the budget"
    (P.allocated x.p B.Device + scratch_bytes > budget);
  use x ~scratch:[ B.create x.d scratch_bytes ]

(* A program's devices stay open, as every drain walks the closed ones, and
   finish their work, which holds the host memory of their fills. *)
let finish w = Array.iter (fun x -> Option.iter Rig.Point.wait (Atomic.get x.last)) w
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
    (* Submits on a until its queue is full, then once more, which waits for
       room: random runs of submits reach a full queue too rarely to count on. *)
    command "fill"
      (world ^-> returns unit)
      (fun _ -> ())
      (fun w ->
        for _ = 0 to capacity do
          submit_system ~note w 0
        done);
    command "wait"
      (world ^-> index @-> returns bool)
      (fun _ _ -> true)
      wait_system;
    command "buffer wait"
      (world ^-> index @-> returns unit)
      (fun _ _ -> ())
      (fun w i -> B.wait w.(i).buffer Rig.Read_write);
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

(* Submits once on [d] a submission that keeps a value whose collection sets
   [released], and drops both: the submission is unreachable once this
   returns. *)
let[@inline never] submit_kept d released =
  let v = ref () in
  Gc.finalise (fun _ -> Atomic.set released true) v;
  ignore (submit (Sub.make ~keep:v d [||]))

(* A collected submission lets go of its value, in a drain, once the work
   submitted before its collection is done: without a wait, only the driver's
   own commit reaches it. A drain first finds the submission collected, before
   the other submits. *)
let test_lag_bounded () =
  let d, _ = P.open_ ~lag ~runs:`Itself (fresh "bounded") in
  let released = Atomic.make false in
  submit_kept d released;
  Gc.full_major ();
  ignore (Sys.opaque_identity (B.create d 8));
  for _ = 1 to lag do
    ignore (submit (empty d))
  done;
  Support.await "the kept value's release" (fun () ->
      Gc.full_major ();
      ignore (Sys.opaque_identity (B.create d 8));
      Atomic.get released)

(* Reading [signaled] commits nothing: values the driver holds stay unreached
   while the device runs what it was given, until a wait commits them. *)
let test_signaled_reads () =
  let d, _ = P.open_ ~lag ~runs:`Itself (fresh "signaled") in
  let base = Rig.submitted d in
  let points = List.init (lag - 1) (fun _ -> submit (empty d)) in
  let last = List.nth points (lag - 2) in
  for _ = 1 to 10 do
    Unix.sleepf 0.001;
    equal ~msg:"signaled" int base (Rig.signaled d)
  done;
  Rig.Point.wait last;
  equal ~msg:"after the wait" int (base + lag - 1) (Rig.signaled d)

(* [reached] commits as a wait does, so a loop on it ends on values the
   driver holds. *)
let test_reached_commits () =
  let d, _ = P.open_ ~lag ~runs:`Itself (fresh "reached") in
  let base = Rig.submitted d in
  let points = List.init (lag - 1) (fun _ -> submit (empty d)) in
  let last = List.nth points (lag - 2) in
  Support.await "the last point reached" (fun () -> Rig.Point.reached last);
  equal ~msg:"signaled" int (base + lag - 1) (Rig.signaled d)

(* Loss *)

(* The device is lost by an allocation, which waits for nothing, so its stop
   finds the values uncommitted. *)
let test_lost_uncommitted () =
  let d, p = P.open_ ~lag (fresh "lost") in
  let base = Rig.submitted d in
  let first = submit (empty d) in
  let second = submit (empty d) in
  P.fault p "the engine hung";
  raises_match (lost d) (fun () -> B.create d 4096);
  raises_match (lost d) (fun () -> Rig.Point.wait second);
  raises_match (lost d) (fun () -> Rig.Point.wait first);
  raises_match ~msg:"reached" (lost d) (fun () -> Rig.Point.reached second);
  equal ~msg:"the word after the stop" int (base + 2) (Rig.signaled d)

(* A submit that blocks in its driver holds the turn: a wait for an earlier
   value goes on without its commit and returns once the work runs. *)
let test_wait_beside_blocked_submit () =
  let d, p = P.open_ ~lag ~capacity:1 ~may_block:true (fresh "blocked") in
  let one = Sub.make d [| idle () |] in
  let submit_one () =
    let run = Sub.Run.make one in
    Sub.Run.groups run (Sub.block one 0) 0 0 0;
    Rig.submit run ~buffers:[||]
  in
  let first = submit_one () in
  let second = ref None in
  let t = Thread.create (fun () -> second := Some (submit_one ())) () in
  Support.await "the second submit blocked" (fun () -> P.blocked p = 1);
  Rig.Point.wait first;
  Thread.join t;
  Option.iter Rig.Point.wait !second

let test_commit_fails () =
  let d, p = P.open_ ~lag (fresh "commit-fails") in
  let first = submit (empty d) in
  P.fail_commit p;
  raises_match
    (function
      | Rig.Lost (d', why) -> Rig.equal d d' && why = "the commit failed"
      | _ -> false)
    (fun () -> Rig.Point.wait first);
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
        test "signaled commits nothing" test_signaled_reads;
        test "a poll of reached commits" test_reached_commits;
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
