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

let submit s =
  Rig.submit s ~run:(Sub.Run.make ()) ~reads:[||] ~writes:[||] ~waits:[||]

(* A drain on [d]: what {!Buffer.create} does first. *)
let drain d = ignore (Sys.opaque_identity (B.create d 8))

(* Submits once on [d] a submission with a hold whose release counts its runs
   in [runs], and drops both: the hold is unreachable once this returns. *)
let[@inline never] submit_held ?(release = ignore) d runs =
  let h =
    H.make (fun () ->
        Atomic.incr runs;
        release ())
  in
  Rig.Point.value (submit (Sub.make ~hold:h ~reads:0 ~writes:0 d [||]))

(* Releases *)

(* A hold whose submissions run on two devices is released once both are
   done. *)
let test_two_devices () =
  let d, pd = P.open_ "hold:first" and e, pe = P.open_ "hold:second" in
  let runs = Atomic.make 0 in
  (fun () ->
    let h = H.make (fun () -> Atomic.incr runs) in
    let on x = Sub.make ~hold:h ~reads:0 ~writes:0 x [||] in
    ignore (submit (on d));
    ignore (submit (on e)))
    ();
  Gc.full_major ();
  ignore (P.run pd);
  drain d;
  drain e;
  equal ~msg:"with one device done" int 0 (Atomic.get runs);
  ignore (P.run pe);
  drain e;
  equal ~msg:"with both done" int 1 (Atomic.get runs)

let test_release () =
  let d, p = P.open_ "hold:release" in
  let runs = Atomic.make 0 in
  ignore (submit_held d runs);
  Gc.full_major ();
  equal ~msg:"after a collection" int 0 (Atomic.get runs);
  drain d;
  equal ~msg:"in a drain before its work is done" int 0 (Atomic.get runs);
  ignore (P.run p);
  drain d;
  equal ~msg:"in a drain once its work is done" int 1 (Atomic.get runs);
  drain d;
  equal ~msg:"in the next drain" int 1 (Atomic.get runs)

(* A hold with no submission is released once unreachable. *)
let test_release_unused () =
  let runs = Atomic.make 0 in
  (fun () -> ignore (Sys.opaque_identity (H.make (fun () -> Atomic.incr runs))))
    ();
  Gc.full_major ();
  drain Rig.host;
  equal int 1 (Atomic.get runs)

(* On a lost device a hold's release waits for the stop's answer and for the
   word to reach the last value: an Unknown answer with a word short of it
   keeps the release. *)
let test_release_lost () =
  let d, p = P.open_ ~answer:`Unknown "hold:lost" in
  let runs = Atomic.make 0 in
  ignore (submit_held d runs);
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
  let runs = Atomic.make 0 in
  ignore (submit_held ~release:(fun () -> raise Exit) d runs);
  Rig.wait d (Rig.submitted d);
  Gc.full_major ();
  raises Exit (fun () -> drain d);
  drain d;
  equal int 1 (Atomic.get runs)

(* A release counts as a call in flight on its device: a loss meanwhile stops
   the device once the release returned. *)
let test_release_counted () =
  let d, p = P.open_ "hold:counted" in
  let runs = Atomic.make 0 in
  let lock = Mutex.create () and cond = Condition.create () in
  let inside = ref false and go = ref false in
  let release () =
    Mutex.protect lock (fun () ->
        inside := true;
        while not !go do
          Condition.wait cond lock
        done)
  in
  ignore (submit_held ~release d runs);
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

(* A drain that finds a hold's device behind a transport faulted when it reads
   its word loses the device and returns: the stop the loss runs drains too, and
   no lock of the first drain is held across the read. *)
let test_release_transport_fault () =
  let d, p = P.open_ ~transport:true "hold:transport" in
  let runs = Atomic.make 0 in
  ignore (submit_held d runs);
  Gc.full_major ();
  Gc.full_major ();
  P.fault_word p "the link went down";
  drain Rig.host;
  equal (option string) (Some "the link went down") (Rig.lost d)

(* A hold orders no work: a submission made with it follows nothing another
   device's submission made with it did. *)
let test_hold_orders_nothing () =
  let d, _ = P.open_ "hold:orders" and e, pe = P.open_ "hold:orders-other" in
  let h = H.make ignore in
  ignore (submit (Sub.make ~hold:h ~reads:0 ~writes:0 e [||]));
  ignore (submit (Sub.make ~hold:h ~reads:0 ~writes:0 d [||]));
  Rig.wait d (Rig.submitted d);
  equal ~msg:"the other device's work, unrun" int 1 (P.queued pe)

(* Fixed memory *)

let answer = Testable.make ~pp:Support.Reader.pp_answer ~equal:( = )

(* A host read of [m] that claims, as a reader in C does: whether it must
   wait. *)
let host_read m =
  let a = Support.Reader.claim m B.Read in
  if a = Support.Reader.Claimed then Support.Reader.release m;
  a

(* Submits on [d] once a submission with [m], [d]'s memory, fixed for
   [access]: its point. *)
let fixing d m access =
  Rig.Point.value
    (submit (Sub.make ~fixed:[ (m, access) ] ~reads:0 ~writes:0 d [||]))

(* Fixed memory read is ordered as a read: a read of it, on the host or on
   another device, waits for no other reader, and a write waits for every
   reader. *)
let test_fixed_read () =
  let d, _ = P.open_ "hold:fixed-read" and e, pe = P.open_ "hold:fixed-read-e" in
  let m = B.create Rig.host (1 lsl 16) in
  let on_d = require_some (B.borrow d m) and on_e = require_some (B.borrow e m) in
  let v = fixing e on_e B.Read in
  equal ~msg:"a host read" answer Support.Reader.Claimed (host_read m);
  ignore (fixing d on_d B.Read);
  Rig.wait d (Rig.submitted d);
  equal ~msg:"another device's read, unrun" int 1 (P.queued pe);
  ignore (fixing d on_d B.Read_write);
  at_least ~msg:"a write follows the read" int ~than:v (Rig.signaled e)

(* Fixed memory written is ordered as a written run buffer: a read follows
   it. *)
let test_fixed_write () =
  let d, _ = P.open_ "hold:fixed-write" and e, pe = P.open_ "hold:fixed-write-e" in
  let m = B.create Rig.host (1 lsl 16) in
  let on_d = require_some (B.borrow d m) and on_e = require_some (B.borrow e m) in
  let v = fixing e on_e B.Read_write in
  equal ~msg:"a host read" answer Support.Reader.Wait (host_read m);
  Support.Reader.release m;
  equal ~msg:"the write, unrun" int 1 (P.queued pe);
  ignore (fixing d on_d B.Read);
  at_least ~msg:"a read follows the write" int ~than:v (Rig.signaled e)

(* One memory fixed in two submissions, of two programs, is ordered by its own
   stamps: each submission's fixed use is the memory's. *)
let test_fixed_shared () =
  let d, p = P.open_ "hold:fixed-shared" in
  let m = B.create d 64 in
  let reads = Sub.make ~fixed:[ (m, B.Read) ] ~reads:0 ~writes:0 d [||] in
  let writes = Sub.make ~fixed:[ (m, B.Read_write) ] ~reads:0 ~writes:0 d [||] in
  ignore (submit reads);
  equal ~msg:"after a read" answer Support.Reader.Claimed (host_read m);
  let v = Rig.Point.value (submit writes) in
  equal ~msg:"after a write" answer Support.Reader.Wait (host_read m);
  Support.Reader.release m;
  B.wait m B.Read;
  equal ~msg:"queued after the wait" int 0 (P.queued p);
  at_least int ~than:v (Rig.signaled d)

(* A copy into fixed memory a submission then reads is what it reads: the
   copy's write orders the submission. *)
let test_fixed_after_copy () =
  let d, p = P.open_ ~host_visible:false "hold:fixed-copy" in
  let m = B.create d 64 in
  let into = B.create Rig.host 64 in
  Bigarray.Array1.fill (B.bigarray Bigarray.char into) 'f';
  B.copy ~src:into ~dst:m;
  let v = fixing d m B.Read in
  ignore (P.run p);
  at_least int ~than:v (Rig.signaled d);
  let back = B.create Rig.host 64 in
  B.copy ~src:m ~dst:back;
  let ba = B.bigarray Bigarray.char back in
  equal string (String.make 64 'f') (String.init 64 (Bigarray.Array1.get ba))

(* A submission keeps its fixed memory: it returns once the submission is
   unreachable and its work done. *)
let test_fixed_returns () =
  let d, p = P.open_ "hold:fixed-returns" in
  let at =
    (fun () ->
      let m = B.create d 64 in
      ignore (fixing d m B.Read);
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
  equal ~msg:"while its work is undone" bool false (freed ());
  ignore (P.run p);
  equal ~msg:"once it is done" bool true (freed ())

let refused ~msg f = raises_match ~msg Exn.invalid_arg f

let test_fixed_refusals () =
  let d, _ = P.open_ "hold:fixed-refusals" and e, _ = P.open_ "hold:other" in
  let make fixed = Sub.make ~fixed ~reads:0 ~writes:0 d [||] in
  let dead = B.create d 8 in
  Rig.Claim.with_ ~read:[] ~donate:[ [ dead ] ] (fun c ->
      ignore (Rig.Claim.consume c ~why:"donated" dead));
  refused ~msg:"a dead buffer" (fun () -> make [ (dead, B.Read) ]);
  refused ~msg:"another device's" (fun () -> make [ (B.create e 8, B.Read) ]);
  let m = B.create d 8 in
  let s = make [ (m, B.Read) ] in
  ignore (Rig.Claim.with_ ~read:[] ~donate:[ [ m ] ] (fun c ->
      ignore (Rig.Claim.consume c ~why:"consumed" m)));
  refused ~msg:"fixed memory that died" (fun () -> submit s)

(* Two domains submitting submissions that fix one memory for reading, each on
   a device of its own: no submit waits for the other domain's, whose work
   stays queued, and the memory is read by both. *)
type readers = { mutable submits : int }

let readers_t =
  abstract ~pp:(fun ppf r -> Format.fprintf ppf "%d submits" r.submits) "r"

(* Each domain's device, opened once for every case, before any domain
   runs. *)
let reader_devices = (fst (P.open_ "hold:readers-0"), fst (P.open_ "hold:readers-1"))

(* The memory, and each domain's submission over it, on its device. *)
let make_readers () =
  let m = B.create Rig.host (1 lsl 16) in
  let d0, d1 = reader_devices in
  let device d =
    let on = require_some (B.borrow d m) in
    Sub.make ~fixed:[ (on, B.Read) ] ~reads:0 ~writes:0 d [||]
  in
  (m, device d0, device d1)

let submit_read (_, s, s') domain =
  ignore (submit (if domain = 0 then s else s'))

let reader_commands =
  [
    command "make" (Gen.unit @-> makes readers_t)
      (fun () -> { submits = 0 })
      make_readers;
    command "submit"
      (readers_t ^-> Gen.int_range 0 1 @-> judges unit)
      (fun r _ outcome ->
        (match outcome with Ok () -> () | Error e -> raise e);
        r.submits <- r.submits + 1)
      submit_read;
    command "host read"
      (readers_t ^-> judges answer)
      (fun _ outcome ->
        match outcome with
        | Ok a -> equal ~msg:"no reader waits for a read" answer Support.Reader.Claimed a
        | Error e -> raise e)
      (fun (m, _, _) -> host_read m);
  ]

let tests =
  [
    group ~timeout "releases"
      [
        test "a hold used on two devices is released once both are done"
          test_two_devices;
        test "a release runs once, in a drain, after its work is done"
          test_release;
        test "a hold no submission used is released once unreachable"
          test_release_unused;
        test "a release waits for a lost device's word and answer"
          test_release_lost;
        test "an exception a release raises reaches the draining call"
          test_release_raises;
        test "a loss during a release stops the device after it"
          test_release_counted;
        test ~timeout:10. "a drain that finds a transport's fault returns"
          test_release_transport_fault;
        test "a hold orders no work" test_hold_orders_nothing;
      ];
    group ~timeout "fixed memory"
      [
        test "fixed memory read waits for no other reader" test_fixed_read;
        test "fixed memory written orders as a written run buffer"
          test_fixed_write;
        test "one memory fixed in two submissions is ordered by its stamps"
          test_fixed_shared;
        test "a copy into fixed memory orders the submission that reads it"
          test_fixed_after_copy;
        test "fixed memory returns once its submission is unreachable and done"
          test_fixed_returns;
        test "fixed memory refuses what make and submit state"
          test_fixed_refusals;
        stateful ~domains:2
          "two domains' reads of one fixed memory wait for no other read"
          reader_commands;
      ];
  ]

let () = exit (run "rig.hold" tests)
