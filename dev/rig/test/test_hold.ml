(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module H = Rig.Hold
module P = Rig_support.Polled
module Support = Rig_support

let timeout = 60.
let lost d = function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false
let count call p = List.length (List.filter (( = ) call) (P.log p))
let empty d = Sub.make ~reads:0 ~writes:0 d [||]
let submit s = Rig.submit s ~reads:[||] ~writes:[||] ~waits:[||]

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
  Rig.Point.value (submit (Sub.make ~hold:h ~reads:0 ~writes:0 d [||]))

(* Releases *)

(* A hold named on two devices is released once both its stamps are reached. *)
let test_two_devices () =
  let d, pd = P.open_ "hold:first" and e, pe = P.open_ "hold:second" in
  let m = B.create d 64 and runs = Atomic.make 0 in
  (fun () ->
    let h = H.make ~release:(fun () -> Atomic.incr runs) [ m ] in
    let on x = Sub.make ~hold:h ~reads:0 ~writes:0 x [||] in
    ignore (submit (on d));
    ignore (submit (on e)))
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

(* A read of held memory waits for the hold's work, which may write it. *)
let test_wait_held () =
  let d, p = P.open_ "hold:wait" in
  let m = B.create d 64 in
  let h = H.make [ m ] in
  let v =
    Rig.Point.value (submit (Sub.make ~hold:h ~reads:0 ~writes:0 d [||]))
  in
  B.wait m B.Read;
  equal int 0 (P.queued p);
  equal int v (Rig.signaled d)

(* On a lost device a hold's release waits for the stop's answer and for the
   word to reach the hold's stamp: an Unknown answer with a word short of it
   keeps the release. *)
let test_release_lost () =
  let d, p = P.open_ ~answer:`Unknown "hold:lost" in
  let m = B.create d 64 and runs = Atomic.make 0 in
  ignore (submit_held d m runs);
  P.fail p;
  raises_match (lost d) (fun () -> submit (empty d));
  equal int 1 (count "stop" p);
  Gc.full_major ();
  drain Rig.host;
  equal ~msg:"with the word short" int 0 (Atomic.get runs);
  P.set_word p (Rig.submitted d);
  drain Rig.host;
  equal ~msg:"once the word drained" int 1 (Atomic.get runs)

let test_release_raises () =
  let d, _ = P.open_ "hold:raises" in
  let m = B.create d 64 and runs = Atomic.make 0 in
  ignore (submit_held ~release:(fun () -> raise Exit) d m runs);
  Rig.wait d (Rig.submitted d);
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
  Rig.wait d (Rig.submitted d);
  Gc.full_major ();
  let t = Thread.create (fun () -> try drain d with Rig.Lost _ -> ()) () in
  Support.await "a running release" (fun () ->
      Mutex.protect lock (fun () -> !inside));
  P.fail p;
  raises_match (lost d) (fun () -> submit (empty d));
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
    Rig.free_cache d;
    drain d;
    List.exists (fun (a, _) -> a = at) (P.frees p)
  in
  equal ~msg:"while its stamp is unreached" bool false (freed ());
  ignore (P.run p);
  equal ~msg:"once it is reached" bool true (freed ())

(* Held memory is named as any memory, by a run's buffers and by parts, with
   its hold, another or none. Its uses follow the hold's stamps: a read waits
   for the hold's work on another device, as a write does. *)
let test_named () =
  let d, _ = P.open_ "hold:named" and e, pe = P.open_ "hold:named-other" in
  let m = B.create d 64 and m' = B.create d 64 in
  let h = H.make [ m ] and h' = H.make [ m' ] in
  let v = Rig.Point.value (submit (Sub.make ~hold:h ~reads:0 ~writes:0 e [||])) in
  let answer = Testable.make ~pp:Support.Reader.pp_answer ~equal:( = ) in
  equal ~msg:"a host read before the hold's work" answer Support.Reader.Wait
    (Support.Reader.claim m B.Read);
  equal ~msg:"the hold's work, unrun" int 1 (P.queued pe);
  let s = Sub.make ~hold:h' ~reads:1 ~writes:1 d [||] in
  ignore (Rig.submit s ~reads:[| m |] ~writes:[| m' |] ~waits:[||]);
  at_least ~msg:"a read follows the hold's work" int ~than:v (Rig.signaled e);
  let copy = Sub.Copy { src = m'; dst = m } in
  let part = { Sub.queue = "COPY:0"; after = [||]; work = copy } in
  let filled c =
    let b = B.create Rig.host 64 in
    Bigarray.Array1.fill (B.bigarray Bigarray.char b) c;
    b
  in
  let contents b =
    let back = B.create Rig.host 64 in
    B.copy ~src:b ~dst:back;
    let ba = B.bigarray Bigarray.char back in
    String.init 64 (Bigarray.Array1.get ba)
  in
  B.copy ~src:(filled 'a') ~dst:m';
  ignore (submit (Sub.make ~reads:0 ~writes:0 d [| part |]));
  equal ~msg:"a copy with no hold" string (String.make 64 'a') (contents m);
  B.copy ~src:(filled 'b') ~dst:m';
  ignore (submit (Sub.make ~hold:h ~reads:0 ~writes:0 d [| part |]));
  equal ~msg:"a copy with another hold" string (String.make 64 'b') (contents m)

(* A submission made with a hold may use any of its memory, named or not: it
   follows another device's write of the hold's memory, still queued, though it
   names none of it. *)
let test_follows_hold () =
  let d, _ = P.open_ "hold:follows" and e, pe = P.open_ "hold:follows-other" in
  let m = B.create d 64 and m' = B.create d 64 in
  let h = H.make [ m; m' ] in
  let w = require_some (B.borrow e m') in
  let writes = Sub.make ~reads:0 ~writes:1 e [||] in
  let v =
    Rig.Point.value (Rig.submit writes ~reads:[||] ~writes:[| w |] ~waits:[||])
  in
  equal ~msg:"the write, unrun" int 1 (P.queued pe);
  ignore (submit (Sub.make ~hold:h ~reads:0 ~writes:0 d [||]));
  at_least ~msg:"the hold's submission follows it" int ~than:v (Rig.signaled e)

(* Refusals *)

let test_one_hold () =
  let d, _ = P.open_ "hold:one" in
  let m = B.create d 64 in
  let h = H.make [ m ] in
  raises_match Exn.invalid_arg (fun () ->
      H.make [ B.view m ~first:8 ~length:8 ]);
  ignore (Sys.opaque_identity h)

let test_dead () =
  let b = B.create Rig.host 8 in
  Rig.Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
      ignore (Rig.Claim.consume c ~why:"donated" b));
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
  drain Rig.host;
  equal (option string) (Some "the link went down") (Rig.lost d)

(* Held memory a device's queue copies copies in and out, and between memory of
   two holds. *)
let test_copy_held () =
  let d, _ = P.open_ ~host_visible:false "hold:copy" in
  let m = B.create d 64 in
  let h = H.make [ m ] in
  let into = B.create Rig.host 64 and back = B.create Rig.host 64 in
  Bigarray.Array1.fill (B.bigarray Bigarray.char into) 'h';
  B.copy ~src:into ~dst:m;
  B.copy ~src:m ~dst:back;
  let contents b =
    let ba = B.bigarray Bigarray.char b in
    String.init 64 (Bigarray.Array1.get ba)
  in
  equal ~msg:"in and out" string (String.make 64 'h') (contents back);
  let m' = B.create d 64 in
  let h' = H.make [ m' ] in
  B.copy ~src:m ~dst:m';
  B.copy ~src:m' ~dst:back;
  equal ~msg:"between two holds" string (String.make 64 'h') (contents back);
  ignore (Sys.opaque_identity (h, h'))

(* A submission made before its memory was put in a hold, and submitted after,
   raises the memory's own stamps: a host wait on the memory waits for it. *)
let test_held_after () =
  let d, p = P.open_ "hold:after" in
  let src = B.create d 64 and dst = B.create d 64 in
  let copy =
    { Sub.queue = "COPY:0"; after = [||]; work = Sub.Copy { src; dst } }
  in
  let s = Sub.make ~reads:0 ~writes:0 d [| copy |] in
  let h = H.make [ dst ] in
  let v = Rig.Point.value (submit s) in
  equal ~msg:"queued" int 1 (P.queued p);
  B.wait dst B.Read;
  equal ~msg:"after the wait" int 0 (P.queued p);
  at_least ~msg:"after the wait" int ~than:v (Rig.signaled d);
  ignore (Sys.opaque_identity h)

(* Two domains holding one memory: one hold takes it, the other raises. *)
type held = { mutable taken : bool }

let holds =
  abstract ~pp:(fun ppf r -> Format.fprintf ppf "taken %b" r.taken) "m"

let make_memory () =
  let d = require_ok ~pp:Format.pp_print_string (Rig.memory_device "hold:m") in
  (B.create d 64, Atomic.make None)

(* Keeps the hold reachable. Only one hold takes the memory, so one domain sets
   [kept], but both may reach it. *)
let take_hold (b, kept) = Atomic.set kept (Some (H.make [ b ]))

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
        test "held memory is named as any memory, its uses after the hold's"
          test_named;
        test "a submission with a hold follows foreign work on its memory"
          test_follows_hold;
        test "memory is in one hold at most" test_one_hold;
        test "a dead buffer is not held" test_dead;
        test "held memory a device's queue copies copies in and out"
          test_copy_held;
        test "a wait on memory held after a submission's make waits for it"
          test_held_after;
        stateful ~domains:2 "two domains holding one memory: one hold takes it"
          hold_commands;
      ];
  ]

let () = exit (run "rig.hold" tests)
