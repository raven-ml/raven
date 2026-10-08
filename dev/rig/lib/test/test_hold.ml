(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module C = Rig
module B = Rig.Buffer
module Sub = Rig.Submission
module H = Rig.Hold
module P = Rig_support.Polled
module Support = Rig_support

let timeout = 60.
let lost d = function C.Lost (d', _) -> C.equal d d' | _ -> false
let count call p = List.length (List.filter (( = ) call) (P.log p))
let empty d = Sub.make ~reads:0 ~writes:0 ~waits:0 d [||]

(* A drain on [d]: what {!Buffer.create} does first. *)
let drain d = ignore (Sys.opaque_identity (B.create d 8))

(* Submits once on [d] a submission naming a hold of [m] whose release counts
   its runs in [runs], and drops both: the hold is unreachable once this
   returns. *)
let[@inline never] submit_held ?(release = ignore) d m runs =
  let h =
    H.make
      ~release:(fun () ->
        Atomic.incr runs;
        release ())
      [ m ]
  in
  C.Point.value (C.submit (Sub.make ~hold:h ~reads:0 ~writes:0 ~waits:0 d [||]))

(* Releases *)

(* A hold named on two devices is released once both its stamps are reached. *)
let test_two_devices () =
  let d, pd = P.open_ "hold:first" and e, pe = P.open_ "hold:second" in
  let m = B.create d 64 and runs = Atomic.make 0 in
  (fun () ->
    let h = H.make ~release:(fun () -> Atomic.incr runs) [ m ] in
    let on x = Sub.make ~hold:h ~reads:0 ~writes:0 ~waits:0 x [||] in
    ignore (C.submit (on d));
    ignore (C.submit (on e)))
    ();
  Gc.full_major ();
  ignore (P.run pd);
  drain d;
  drain e;
  equal ~msg:"with one stamp reached" int 0 (Atomic.get runs);
  ignore (P.run pe);
  drain e;
  equal ~msg:"with both reached" int 1 (Atomic.get runs)

let test_release () =
  let d, p = P.open_ "hold:release" in
  let m = B.create d 64 and runs = Atomic.make 0 in
  ignore (submit_held d m runs);
  Gc.full_major ();
  equal ~msg:"after a collection" int 0 (Atomic.get runs);
  drain d;
  equal ~msg:"in a drain before its stamp is reached" int 0 (Atomic.get runs);
  ignore (P.run p);
  drain d;
  equal ~msg:"in a drain once its stamp is reached" int 1 (Atomic.get runs);
  drain d;
  equal ~msg:"in the next drain" int 1 (Atomic.get runs)

(* Memory in a hold returns by the hold: a read waits for the hold's work, which
   wrote nothing. *)
let test_wait_held () =
  let d, p = P.open_ "hold:wait" in
  let m = B.create d 64 in
  let h = H.make [ m ] in
  let v =
    C.Point.value
      (C.submit (Sub.make ~hold:h ~reads:0 ~writes:0 ~waits:0 d [||]))
  in
  B.wait m B.Read;
  equal int 0 (P.queued p);
  equal int v (C.signaled d)

(* On a lost device a hold's release waits for the stop's answer and for the
   word to reach the hold's stamp: an Unknown answer with a word short of it
   keeps the release. *)
let test_release_lost () =
  let d, p = P.open_ ~answer:`Unknown "hold:lost" in
  let m = B.create d 64 and runs = Atomic.make 0 in
  ignore (submit_held d m runs);
  P.fail p;
  raises_match (lost d) (fun () -> C.submit (empty d));
  equal int 1 (count "stop" p);
  Gc.full_major ();
  drain C.host;
  equal ~msg:"with the word short" int 0 (Atomic.get runs);
  P.set_word p (C.submitted d);
  drain C.host;
  equal ~msg:"once the word drained" int 1 (Atomic.get runs)

let test_release_raises () =
  let d, _ = P.open_ "hold:raises" in
  let m = B.create d 64 and runs = Atomic.make 0 in
  ignore (submit_held ~release:(fun () -> raise Exit) d m runs);
  C.wait d (C.submitted d);
  Gc.full_major ();
  raises Exit (fun () -> drain d);
  drain d;
  equal int 1 (Atomic.get runs)

(* A release counts as a call in flight on its device: a loss meanwhile stops
   the device once the release returned. *)
let test_release_counted () =
  let d, p = P.open_ "hold:counted" in
  let m = B.create d 64 and runs = Atomic.make 0 in
  let lock = Mutex.create () and cond = Condition.create () in
  let inside = ref false and go = ref false in
  let release () =
    Mutex.protect lock (fun () ->
        inside := true;
        while not !go do
          Condition.wait cond lock
        done)
  in
  ignore (submit_held ~release d m runs);
  C.wait d (C.submitted d);
  Gc.full_major ();
  let t = Thread.create (fun () -> try drain d with C.Lost _ -> ()) () in
  Support.await "a running release" (fun () ->
      Mutex.protect lock (fun () -> !inside));
  P.fail p;
  raises_match (lost d) (fun () -> C.submit (empty d));
  equal ~msg:"while the release runs" int 0 (count "stop" p);
  Mutex.protect lock (fun () ->
      go := true;
      Condition.signal cond);
  Thread.join t;
  equal ~msg:"once it returned" int 1 (count "stop" p)

(* Held memory returns once the hold is unreachable and its stamp reached. *)
let test_memory_returns () =
  let d, p = P.open_ "hold:memory" in
  let at =
    (fun () ->
      let m = B.create d 64 in
      ignore (submit_held d m (Atomic.make 0));
      B.address m)
      ()
  in
  let freed () =
    Gc.full_major ();
    drain d;
    C.free_cache d;
    drain d;
    List.exists (fun (a, _) -> a = at) (P.frees p)
  in
  equal ~msg:"while its stamp is unreached" bool false (freed ());
  ignore (P.run p);
  equal ~msg:"once it is reached" bool true (freed ())

(* Refusals *)

let test_refusals () =
  let d, _ = P.open_ "hold:refusals" in
  let m = B.create d 64 and m' = B.create d 64 in
  let h = H.make [ m ] in
  raises_match Exn.invalid_arg (fun () ->
      H.make [ B.view m ~first:8 ~length:8 ]);
  let s = Sub.make ~hold:h ~reads:1 ~writes:1 ~waits:0 d [||] in
  raises_match Exn.invalid_arg (fun () -> Sub.read s 0 m);
  raises_match Exn.invalid_arg (fun () -> Sub.write s 0 m);
  let h' = H.make [ m' ] in
  let copy = Sub.Copy { src = m'; dst = B.create d 64 } in
  let part = { Sub.queue = "COPY:0"; after = [||]; work = copy } in
  raises_match Exn.invalid_arg (fun () ->
      Sub.make ~hold:h ~reads:0 ~writes:0 ~waits:0 d [| part |]);
  ignore (Sub.make ~hold:h' ~reads:0 ~writes:0 ~waits:0 d [| part |])

let test_dead () =
  let b = B.create C.host 8 in
  C.Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
      ignore (C.Claim.consume c ~why:"donated" b));
  raises_match (Exn.invalid_arg ~substring:"donated") (fun () -> H.make [ b ])

(* A drain that finds a hold's device behind a transport faulted when it reads
   its word loses the device and returns: the stop the loss runs drains too, and
   no lock of the first drain is held across the read. *)
let test_release_transport_fault () =
  let d, p = P.open_ ~transport:true "hold:transport" in
  let runs = Atomic.make 0 in
  ignore (submit_held d (B.create d 64) runs);
  Gc.full_major ();
  Gc.full_major ();
  P.fault_word p "the link went down";
  drain C.host;
  equal (option string) (Some "the link went down") (C.lost d)

(* Memory put in a hold after a submission named it is refused at the next
   submit, by a part or by a slot: work on held memory raises the hold's stamps,
   which only a submission made with the hold does. *)
let test_held_after () =
  let d, _ = P.open_ "hold:after" in
  let src = B.create d 64 and dst = B.create d 64 in
  let copy =
    { Sub.queue = "COPY:0"; after = [||]; work = Sub.Copy { src; dst } }
  in
  let s = Sub.make ~reads:0 ~writes:0 ~waits:0 d [| copy |] in
  let h = H.make [ src ] in
  raises_match Exn.invalid_arg (fun () -> C.submit s);
  let m = B.create d 64 in
  let r = Sub.make ~reads:1 ~writes:0 ~waits:0 d [||] in
  Sub.read r 0 m;
  let h' = H.make [ m ] in
  raises_match Exn.invalid_arg (fun () -> C.submit r);
  ignore (Sys.opaque_identity (h, h'))

(* Two domains holding one memory: one hold takes it, the other raises. *)
type held = { mutable taken : bool }

let holds =
  abstract ~pp:(fun ppf r -> Format.fprintf ppf "taken %b" r.taken) "m"

let make_memory () =
  let d = require_ok ~pp:Format.pp_print_string (C.memory_device "hold:m") in
  (B.create d 64, ref [])

let take_hold (b, kept) = kept := H.make [ b ] :: !kept

let judge_take r = function
  | Ok () ->
      equal ~msg:"already held" bool false r.taken;
      r.taken <- true
  | Error (Invalid_argument _) -> equal ~msg:"held" bool true r.taken
  | Error e -> raise e

let hold_commands =
  [
    command "make"
      (Gen.unit @-> makes holds)
      (fun () -> { taken = false })
      make_memory;
    command "hold" (holds ^-> judges unit) judge_take take_hold;
  ]

let tests =
  [
    group ~timeout "releases"
      [
        test "a hold named on two devices is released once both are reached"
          test_two_devices;
        test "a release runs once, in a drain, after its stamp is reached"
          test_release;
        test "a release waits for a lost device's word and answer"
          test_release_lost;
        test "an exception a release raises reaches the draining call"
          test_release_raises;
        test "a loss during a release stops the device after it"
          test_release_counted;
        test ~timeout:10. "a drain that finds a transport's fault returns"
          test_release_transport_fault;
      ];
    group ~timeout "memory"
      [
        test "a read of held memory waits for the hold's work" test_wait_held;
        test "held memory returns once its hold is unreachable and reached"
          test_memory_returns;
        test "held memory is named only with its hold, in one hold"
          test_refusals;
        test "a dead buffer is not held" test_dead;
        test "memory held after a submission named it is refused at submit"
          test_held_after;
        stateful ~domains:2 "two domains holding one memory: one hold takes it"
          hold_commands;
      ];
  ]

let () = exit (run "rig.hold" tests)
