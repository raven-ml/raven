(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Timelines: the values work takes, committed work, and [signaled]. The devices
   are Polled devices that commit on their own every [lag] values and run only
   committed work: a wait that sleeps on uncommitted work raises Polled's
   [Failure], so every call returning shows that it committed what it waited
   for. *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support

let lag = 4
let lost d = function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false
let empty d = Sub.make d [||]
let names = Atomic.make 0

let fresh what =
  Printf.sprintf "commit:%s-%d" what (Atomic.fetch_and_add names 1)

let submit ?(buffers = [||]) ?(args = [||]) ?waits s =
  Rig.submit ?waits s ~buffers ~args

(* The last value [p]'s word holds, read without a call of rig. *)
let word p = Support.load (P.word_at p)

(* Committed work *)

(* Two devices: [a] with a queue of two parts, so that a third uncommitted
   submission waits for room, and [b], whose queue waits on [a]'s word. Each
   submission occupies one queue part and reads its device's buffer, whose
   stamps waits, reads and copies follow. A scratch is device memory under a
   small budget, which an allocation reclaims. *)
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
  let buffers = Array.of_list (x.buffer :: scratch) in
  let pt = Rig.submit ?waits s ~buffers ~args:(Support.zeros 1) in
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

(* A wait for the device's last point, and then its word and the point's value:
   0 and 0 before any submit. *)
let wait_system w i =
  let x = w.(i) in
  match Atomic.get x.last with
  | None -> (0, 0)
  | Some pt ->
      Rig.Point.wait pt;
      (word x.p, Rig.Point.value pt)

let wait_model _ _ = function
  | Error e -> raise e
  | Ok (word, value) ->
      at_least ~msg:"the word once the wait returned" int ~than:value word

let scratch_system w i =
  let x = w.(i) in
  cover "an allocation over the budget"
    (P.allocated x.p B.Device + scratch_bytes > budget);
  use x ~scratch:[ B.create x.d scratch_bytes ]

(* A program's devices stay open, as every drain walks the closed ones, and
   finish their work, which holds the host memory of their fills. *)
let finish w =
  Array.iter (fun x -> Option.iter Rig.Point.wait (Atomic.get x.last)) w

let world : (unit, world) abstract = abstract "w" ~release:finish
let index = Gen.int_range 0 1

(* The model holds nothing: every call returns, and a wait finds its value
   reached in its device's word. *)
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
       room: random runs of submits reach a full queue too rarely to count
       on. *)
    command "fill"
      (world ^-> returns unit)
      (fun _ -> ())
      (fun w ->
        for _ = 0 to capacity do
          submit_system ~note w 0
        done);
    command "wait"
      (world ^-> index @-> judges (pair int int))
      wait_model wait_system;
    command "buffer wait"
      (world ^-> index @-> returns unit)
      (fun _ _ -> ())
      (fun w i -> B.wait w.(i).buffer Rig.Read_write);
    command "read"
      (world ^-> index @-> returns unit)
      (fun _ _ -> ())
      (fun w i -> B.read w.(i).buffer Bigarray.char ignore);
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

(* Submits once on [d] a submission that keeps a value whose collection sets
   [released], and drops both: the submission is unreachable once this
   returns. *)
let[@inline never] [@local never] submit_kept d released =
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

(* The device is lost by an allocation, which waits for nothing, so its stop
   finds the values uncommitted: Polled's stop then writes the last value it
   received into its word. *)
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
  Rig.close d;
  equal ~msg:"the word after the stop" int (base + 2) (Rig.signaled d)

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

(* submitted *)

(* A device whose memory the host does not address, so that its own copy engine
   copies and fills it. *)
let opaque () = fst (P.open_ ~host_visible:false (fresh "opaque"))

(* Each row makes one copy or fill, on the device it names: the value it takes
   is the one after the device's submitted value, which it then is. *)
let next_value (_, make) =
  let d, work = make () in
  let before = Rig.submitted d in
  let pt = work () in
  equal ~msg:"its point's value" int (before + 1) (Rig.Point.value pt);
  equal ~msg:"the submitted value" int (before + 1) (Rig.submitted d)

let made_by =
  let copy d () =
    let src = B.create d 8 and dst = B.create d 8 in
    (d, fun () -> B.submit_copy ~src ~dst [| B.row 8 |])
  in
  let fill d () =
    let b = B.create d 8 in
    (d, fun () -> B.submit_fill b '\x2a')
  in
  [
    ("a device's copy", fun () -> copy (opaque ()) ());
    ("a device's fill", fun () -> fill (opaque ()) ());
    ("the host's copy", copy Rig.host);
    ("the host's fill", fill Rig.host);
  ]

(* signaled *)

(* Reading [signaled] commits nothing: values the driver holds stay unreached
   while the device runs what it was given, until a wait commits them. No event
   marks "nothing committed", so it samples ten reads over 10 ms. *)
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

(* Work that adds 1 to a word the host addresses, which the test runs on the
   device: once [signaled] reports its value, the host reads the word with no
   call of rig's. *)
let test_signaled_writes () =
  let d, p = P.open_ (fresh "writes") in
  let b = B.create d 8 in
  B.write b Bigarray.char (fun a -> Bigarray.Array1.fill a '\000');
  let bump = Sub.fill ~refs:[| Sub.write 0 ~at:0 |] Support.bump ~params:8 in
  let pt =
    submit (Sub.make d [| bump |]) ~buffers:[| b |] ~args:(Support.zeros 1)
  in
  less ~msg:"signaled before the run" int ~than:(Rig.Point.value pt)
    (Rig.signaled d);
  ignore (P.run p);
  equal ~msg:"signaled after the run" int (Rig.Point.value pt) (Rig.signaled d);
  equal ~msg:"the word" int 1 (Support.load (Support.Reader.host b));
  Rig.close d

(* A device behind a transport, whose word is read through its driver: a read
   that faults loses the device, and [signaled] raises the loss. *)
let test_signaled_loses () =
  let d, p = P.open_ ~transport:true (fresh "transport") in
  Rig.Point.wait (submit (empty d));
  P.fault_word p "the link went down";
  raises_match (lost d) (fun () -> Rig.signaled d);
  equal (option string) ~msg:"the loss" (Some "the link went down") (Rig.lost d);
  Rig.close d

let tests =
  [
    group "committed work"
      [
        stateful "every wait commits what it waits for"
          (scratch :: commands ~note:cover);
        (* A branch's calls run on another domain, where nothing covers, and a
           reclamation's full collections are left to one domain. *)
        stateful ~count:30 ~domains:2
          "every wait commits what it waits for, on two domains"
          (commands ~note:(fun _ _ -> ()));
        test
          "a collected submission's value goes, with no wait, once the driver \
           commits on its own"
          test_lag_bounded;
        test "a device lost with uncommitted values raises Lost"
          test_lost_uncommitted;
        test "a commit that fails loses the device" test_commit_fails;
      ];
    group "submitted"
      [
        cases ~id:fst
          "a copy or a fill takes the next value of the device that makes it"
          made_by next_value;
      ];
    group "signaled"
      [
        test "signaled commits nothing (sampled)" test_signaled_reads;
        test
          "while live, the writes of the work up to signaled are visible to \
           the host"
          test_signaled_writes;
        test "a read through the driver that loses the device raises Lost"
          test_signaled_loses;
      ];
  ]

let () = exit (run "rig.commit" tests)
