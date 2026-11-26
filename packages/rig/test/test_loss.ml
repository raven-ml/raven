(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A device's loss, what it does to memory, hangs, stops, and [lose]. A failure
   of the process is test_fail's, a fork's losses test_fork's. *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support

let empty d = Sub.make d [||]

let submit ?(buffers = [||]) ?(args = [||]) ?waits s =
  Rig.submit ?waits s ~buffers ~args

(* The values of a submission's one fill, which reads none. *)
let one = Support.zeros 1

(* A submission of one fill that uses a slot per element of [uses]. *)
let touching uses d = Sub.make d [| Support.touching uses |]
let borrow d b = require_some ~msg:"a borrow" (B.borrow d b)
let lost d = function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false
let count call p = List.length (List.filter (( = ) call) (P.log p))

(* Polled's reason for a failed hand-over ({!P.fail}). *)
let failed_why = "the submission failed"

(* Loses [d] through a failed hand-over, whose stop has answered once it
   returns. *)
let lose d p =
  P.fail p;
  try ignore (submit (empty d)) with Rig.Lost _ -> ()

(* Collects, then frees the host's cache and [d]'s, which first drains them: the
   memory of [d], lost, collected after its stop returns so, as do [d]'s
   mappings of host memory collected since. *)
let collect d =
  Gc.full_major ();
  Rig.free_cache Rig.host;
  Rig.free_cache d

let fresh =
  let n = Atomic.make 0 in
  fun what -> Printf.sprintf "loss:%s-%d" what (Atomic.fetch_and_add n 1)

(* What a claim from C answers where a use raises [d]'s loss. *)
exception Claim_lost

(* What a use of memory after [d]'s loss does: ["ran"], or ["Lost"] for [d]'s
   loss. *)
let outcome d use =
  match use () with
  | () -> "ran"
  | exception Rig.Lost (d', _) when Rig.equal d d' -> "Lost"
  | exception Claim_lost -> "Lost"
  | exception Rig.Lost (d', why) ->
      Printf.sprintf "Lost %s: %s" (Rig.name d') why

(* Loss *)

let test_failed_submit () =
  let d, p = P.open_ "loss:failed" in
  let s = empty d in
  ignore (submit s);
  P.fail p;
  raises_match (lost d) (fun () -> submit s);
  equal (option string) (Some failed_why) (Rig.lost d);
  raises_match (lost d) (fun () -> submit s);
  raises_match (lost d) (fun () -> B.create d 8);
  equal ~msg:"stops before the close" int 0 (count "stop" p);
  Rig.close d;
  equal ~msg:"stops after the close" int 1 (count "stop" p)

let test_fault () =
  let d, p = P.open_ "loss:fault" in
  let pt = submit (empty d) in
  P.fault p "the engine hung";
  raises_match (lost d) (fun () -> Rig.Point.wait pt);
  equal (option string) (Some "the engine hung") (Rig.lost d);
  raises_match (lost d) (fun () -> B.create d 8);
  equal ~msg:"stops before the close" int 0 (count "stop" p);
  Rig.close d;
  equal ~msg:"stops after the close" int 1 (count "stop" p)

(* Two domains sleep on a device that faults: each raises its Lost, and the
   device is lost once, and stopped once by its close. *)
let test_two_sleeps () =
  let d, p = P.open_ "loss:two-sleeps" in
  let pt = submit (empty d) in
  P.gate p;
  let wait () =
    match Rig.Point.wait pt with
    | () -> "returned"
    | exception Rig.Lost (_, why) -> why
  in
  let waiters = List.init 2 (fun _ -> Domain.spawn wait) in
  Support.await "two sleeps at the gate" (fun () -> P.sleepers p = 2);
  P.fault p "the engine hung";
  P.open_gate p;
  equal (list string)
    [ "the engine hung"; "the engine hung" ]
    (List.map Domain.join waiters);
  equal (option string) (Some "the engine hung") (Rig.lost d);
  Rig.close d;
  equal ~msg:"stops" int 1 (count "stop" p)

(* A lost device's name, architecture, budget, capability, last value handed
   over and last value its word reached keep their values. Its work may still
   run, so its stop leaves the word where it stood. *)
let test_facts transport () =
  let name = if transport then "loss:facts-transport" else "loss:facts" in
  let d, p = P.open_ ~transport ~answer:`Unknown name in
  Rig.set_budget d 4096;
  let base = Rig.submitted d in
  ignore (submit (empty d));
  ignore (submit (empty d));
  ignore (P.run p);
  let third = submit (empty d) in
  P.fault p "the engine hung";
  raises_match (lost d) (fun () -> Rig.Point.wait third);
  equal string name (Rig.name d);
  equal string "polled" (Rig.arch d);
  equal int 4096 (Rig.budget d);
  equal (option unit) (Some ()) (Rig.capability d P.capability_key);
  equal ~msg:"submitted" int (base + 3) (Rig.submitted d);
  equal ~msg:"signaled" int (base + 2) (Rig.signaled d);
  equal (option string) (Some "the engine hung") (Rig.lost d)

(* A device whose queue waits on a lost device's unreached value is lost with
   it. *)
let test_spread () =
  let producer, pp = P.open_ "loss:producer" in
  let consumer, _ = P.open_ ~waits_on:[ `Host ] "loss:consumer" in
  let a = submit (empty producer) in
  ignore (submit (empty consumer) ~waits:[| a |]);
  P.fail pp;
  raises_match (lost producer) (fun () -> submit (empty producer));
  equal (option string) (Some "loss:producer lost") (Rig.lost consumer)

let test_no_spread () =
  let producer, pp = P.open_ "loss:producer-2" in
  let consumer, _ = P.open_ ~waits_on:[ `Host ] "loss:consumer-2" in
  let a = submit (empty producer) in
  Rig.Point.wait a;
  ignore (submit (empty consumer) ~waits:[| a |]);
  P.fail pp;
  raises_match (lost producer) (fun () -> submit (empty producer));
  equal (option string) None (Rig.lost consumer)

(* A submit that waits for a point a lost device never reached raises that
   device's loss and runs nothing; one it reached is waited for as done. *)
let test_waits_unreached () =
  let producer, pp = P.open_ "loss:waits-producer" in
  let consumer, _ = P.open_ "loss:waits-consumer" in
  let reached = submit (empty producer) in
  Rig.Point.wait reached;
  let unreached = submit (empty producer) in
  lose producer pp;
  let before = Rig.submitted consumer in
  raises_match (lost producer) (fun () ->
      submit (empty consumer) ~waits:[| reached; unreached |]);
  equal ~msg:"nothing submitted" int before (Rig.submitted consumer);
  Rig.Point.wait (submit (empty consumer) ~waits:[| reached |]);
  equal (option string) None (Rig.lost consumer)

let test_reached () =
  let d, p = P.open_ "loss:reached" in
  let pt = submit (empty d) in
  Rig.Point.wait pt;
  let dropped = submit (empty d) in
  P.fail p;
  raises_match (lost d) (fun () -> submit (empty d));
  Rig.Point.wait pt;
  raises_match (lost d) (fun () -> Rig.Point.wait dropped)

(* A point that a device reached stays done when the device is lost by the read
   of its word that a later submit's wait makes: the submit and a wait on the
   point return. The device's word lies behind a transport, whose link goes down
   after the point was waited for. *)
let test_reached_at_loss () =
  let consumer, _ = P.open_ "loss:reached-consumer" in
  let d, p = P.open_ ~transport:true "loss:reached-producer" in
  let q = submit (empty d) in
  Rig.Point.wait q;
  P.fault_word p "the link went down";
  Rig.Point.wait (submit ~waits:[| q |] (empty consumer));
  Rig.Point.wait q;
  equal (option string) (Some "the link went down") (Rig.lost d);
  Rig.close d;
  Rig.close consumer

(* Memory after a loss *)

(* A host buffer that [d]'s work used as [uses] says, unreached when [d] was
   lost, and [e], live, which borrowed it before the loss: [(d, e, h, on_e)]. *)
let used_by_lost name uses =
  let d, p = P.open_ name and e, _ = P.open_ (name ^ "-other") in
  let h = B.create Rig.host B.paged_from in
  let on_e = borrow e h in
  ignore (submit ~args:one (touching uses d) ~buffers:[| borrow d h |]);
  lose d p;
  (d, e, h, on_e)

(* A submit on [e] of one fill that uses [on_e] as [use] says, waited for. *)
let run_on e use on_e =
  Rig.Point.wait (submit ~args:one (touching [| use |] e) ~buffers:[| on_e |])

(* The uses of a host buffer [h] that depend on the points of its last write,
   given [e] and [on_e], [h] borrowed on [e]. *)
let reads =
  [
    ("Buffer.wait Read", fun _ h _ -> B.wait h Rig.Read);
    ("Buffer.read", fun _ h _ -> B.read h Bigarray.char ignore);
    ( "Buffer.blit_to_bytes",
      fun _ h _ -> B.blit_to_bytes h 0 (Bytes.create 1) 0 1 );
    ( "a copy's source",
      fun _ h _ -> B.copy ~src:h ~dst:(B.create Rig.host (B.length h)) );
    ("a submit's read", fun e _ on_e -> run_on e Sub.read on_e);
    ( "a claim for reading",
      fun _ h _ ->
        match Support.Reader.claim h Rig.Read with
        | Claimed -> Support.Reader.release h
        | Lost -> raise Claim_lost
        | answer ->
            failf "the claim answered %a" Support.Reader.pp_answer answer );
  ]

(* The uses that depend on every point of its stamps. *)
let others =
  [
    ("Buffer.wait Read_write", fun _ h _ -> B.wait h Rig.Read_write);
    ("Buffer.write", fun _ h _ -> B.write h Bigarray.char ignore);
    ("Buffer.fill", fun _ h _ -> B.fill h 'x');
    ("Buffer.blit_from_string", fun _ h _ -> B.blit_from_string "x" 0 h 0 1);
    ( "a copy's destination",
      fun _ h _ -> B.copy ~src:(B.create Rig.host (B.length h)) ~dst:h );
    ("Buffer.borrow", fun e h _ -> ignore (B.borrow e h));
    ("Buffer.export", fun _ h _ -> ignore (B.export h Bigarray.char));
    ("Buffer.take", fun _ h _ -> ignore (B.take ~why:"taken" [ h ]));
    ("a submit's write", fun e _ on_e -> run_on e Sub.write on_e);
  ]

let use_name (what, _) = what
let outcome_of d (_, use) (_, e, h, on_e) = outcome d (fun () -> use e h on_e)

let read_after_read ((what, _) as use) =
  let lost_device = used_by_lost ("loss:read-" ^ what) [| Sub.read |] in
  let d, _, _, _ = lost_device in
  equal string "ran" (outcome_of d use lost_device)

(* Each use raises the loss before the device's stop, and runs after it: its
   unreached read no longer runs and changed no memory. *)
let other_after_read ((what, _) as use) =
  let lost_device = used_by_lost ("loss:other-" ^ what) [| Sub.read |] in
  let d, _, _, _ = lost_device in
  equal ~msg:"before the stop" string "Lost" (outcome_of d use lost_device);
  Rig.close d;
  equal ~msg:"after the stop" string "ran" (outcome_of d use lost_device)

(* Each use raises the loss before the device's stop and after it. *)
let use_after_write ((what, _) as use) =
  let lost_device = used_by_lost ("loss:written-" ^ what) [| Sub.write |] in
  let d, _, _, _ = lost_device in
  equal ~msg:"before the stop" string "Lost" (outcome_of d use lost_device);
  Rig.close d;
  collect d;
  equal ~msg:"after the stop" string "Lost" (outcome_of d use lost_device)

let test_unordered () =
  let d, e, _, on_e = used_by_lost "loss:unordered" [| Sub.write |] in
  equal string "ran" (outcome d (fun () -> run_on e Sub.unordered on_e))

(* A read a loss dropped counts as done once its device has stopped, and a write
   it dropped raises the loss for good, also once a later device took the lost
   device's slot: a write of the read memory follows the read then, on the host
   and on another device, and a read or a write of the written memory raises. *)
let test_read_after_stop () =
  let d, p = P.open_ "loss:read-after-stop" in
  let e, _ = P.open_ "loss:read-after-stop-other" in
  let read = B.create Rig.host B.paged_from
  and written = B.create Rig.host B.paged_from in
  let s = touching [| Sub.read; Sub.write |] d in
  let pt = submit ~args:one s ~buffers:[| borrow d read; borrow d written |] in
  lose d p;
  let write e b =
    Rig.Point.wait
      (submit ~args:one (touching [| Sub.write |] e) ~buffers:[| borrow e b |])
  in
  let check what e =
    B.fill read 'x';
    write e read;
    equal
      ~msg:(what ^ ", the bytes the host wrote")
      char 'x'
      (B.read read Bigarray.char (fun a -> a.{0}));
    raises_match ~msg:(what ^ ", a read of a dropped write") (lost d) (fun () ->
        B.wait written Rig.Read);
    raises_match ~msg:(what ^ ", a write after a dropped write") (lost d)
      (fun () -> write e written)
  in
  Rig.close d;
  check "stopped" e;
  collect d;
  (* A device that opens takes a free slot, which rig.mli does not name: the
     opens go on until one took the lost device's. *)
  let rec later others =
    let l, _ = P.open_ (fresh "read-after-stop-later") in
    if Rig.Point.device pt = None then begin
      List.iter Rig.close others;
      l
    end
    else later (l :: others)
  in
  check "its slot taken" (later [])

(* A refused submit, on a device lost before it, runs no work on its buffers'
   memory, which does not raise Lost for that device. *)
let test_unrun_names () =
  let d, p = P.open_ "loss:unrun" in
  let h = B.create Rig.host B.paged_from in
  let s = touching [| Sub.read |] d in
  let buffers = [| borrow d h |] in
  P.fail p;
  raises_match (lost d) (fun () -> submit (empty d));
  raises_match (lost d) (fun () -> submit ~args:one s ~buffers);
  equal int B.paged_from (B.read h Bigarray.char Bigarray.Array1.dim)

(* rig.mli's Order: a submit adds its point to the memory's stamps also when the
   driver fails to take the work, so a later write waits for that point, which
   the lost device never reached. *)
let test_failed_take_stamps () =
  let d, p = P.open_ "loss:failed-take" in
  let h = B.create Rig.host B.paged_from in
  let s = touching [| Sub.write |] d and buffers = [| borrow d h |] in
  P.fail p;
  raises_match ~msg:"the failed take" (lost d) (fun () ->
      submit ~args:one s ~buffers);
  raises_match ~msg:"a read" (lost d) (fun () -> B.wait h Rig.Read);
  raises_match ~msg:"a write" (lost d) (fun () ->
      B.write h Bigarray.char ignore)

(* A loss reaches the lost device; memory of other devices whose points it
   reached is ordinary memory again, and other devices go on. *)
let test_others_go_on () =
  let d, p = P.open_ "loss:lost" in
  let e, _ = P.open_ "loss:kept" in
  let named = B.create e 64 in
  let buffers = [| borrow d named |] in
  Rig.Point.wait (submit ~args:one (touching [| Sub.read |] d) ~buffers);
  P.fail p;
  raises_match (lost d) (fun () -> submit (empty d));
  B.wait named Rig.Read_write;
  let named = require_some ~msg:"a take" (Support.donated named) in
  Rig.Point.wait
    (submit ~args:one (touching [| Sub.write |] e) ~buffers:[| named |]);
  equal (option string) None (Rig.lost e)

(* A use a loss dropped stays in its memory until its device has stopped:
   devices that read the memory after it, through borrows taken before it, and
   end, take nothing of it. *)
let test_unreached_kept () =
  let d, p = P.open_ "loss:unreached-kept" in
  let h = B.create Rig.host B.paged_from in
  let readers =
    List.init 8 (fun i ->
        let e, _ = P.open_ (Printf.sprintf "loss:unreached-kept-%d" i) in
        (e, borrow e h))
  in
  ignore
    (submit ~args:one (touching [| Sub.read |] d) ~buffers:[| borrow d h |]);
  P.fail p;
  raises_match (lost d) (fun () -> submit (empty d));
  List.iter
    (fun (e, b) ->
      run_on e Sub.read b;
      Rig.close e)
    readers;
  raises_match (lost d) (fun () -> B.wait h Rig.Read_write)

(* A stop raises the word to the last value whatever ran: work the device did
   not reach before its loss stays unreached. *)
let test_stop_reaches_nothing () =
  let d, p = P.open_ "loss:stop-raises" in
  let e, _ = P.open_ "loss:stop-other" in
  let m = B.create e 64 in
  let b = borrow d m in
  let v =
    Rig.Point.value
      (submit ~args:one (touching [| Sub.write |] d) ~buffers:[| b |])
  in
  P.fail p;
  raises_match (lost d) (fun () -> submit (empty d));
  Rig.close d;
  at_least ~msg:"the word after the stop" int ~than:v (Rig.signaled d);
  raises_match (lost d) (fun () -> B.wait m Rig.Read)

(* A lost device's memory raises its loss whatever its points, also with no work
   ever on it, through any buffer: its own, a borrow on the host, and a borrow
   on another device, whether a submit names it in a slot, as fixed memory or in
   a copy. A take raises it for memory nothing keeps, since a kept memory
   refuses the take first. *)
let test_own_memory () =
  let d, p = P.open_ ~copies:false "loss:own" in
  let e, _ = P.open_ "loss:own-peer" in
  let m = B.create d 64 in
  let h = borrow Rig.host m in
  let unkept = borrow Rig.host (B.create d 64) in
  let b = borrow e m in
  let dst = B.create Rig.host 64 in
  let fixed = Sub.make ~fixed:[ (b, Rig.Read) ] e [||] in
  let copy =
    Sub.make e Sub.[| copy ~src:(Buffer b) ~dst:(Buffer (B.create e 64)) |]
  in
  P.fail p;
  raises_match (lost d) (fun () -> submit (empty d));
  List.iter
    (fun (what, f) -> raises_match ~msg:what (lost d) f)
    [
      ("a wait", fun () -> B.wait m Rig.Read);
      ("a wait on the host's borrow", fun () -> B.wait h Rig.Read);
      ("a read", fun () -> B.read h Bigarray.char ignore);
      ("a take", fun () -> ignore (B.take ~why:"taken" [ unkept ]));
      ("a copy", fun () -> B.copy ~src:h ~dst);
      ("a borrow", fun () -> ignore (B.borrow Rig.host m));
      ( "a submit on another device",
        fun () ->
          ignore (submit ~args:one (touching [| Sub.read |] e) ~buffers:[| b |])
      );
      ("fixed memory of a submit", fun () -> ignore (submit fixed));
      ("a copy a submit runs", fun () -> ignore (submit copy));
    ];
  equal (option string) None (Rig.lost e)

(* A lost device's borrow of other memory raises its loss; the memory itself,
   which the device's work never used, is ordinary. *)
let test_lost_borrow () =
  let d, p = P.open_ "loss:borrow" in
  let h = B.create Rig.host B.paged_from in
  let b = borrow d h in
  P.fail p;
  raises_match (lost d) (fun () -> submit (empty d));
  raises_match (lost d) (fun () -> B.wait b Rig.Read);
  B.wait h Rig.Read_write

(* An array exported before the loss reads and writes the memory with no check,
   while the buffer's uses raise the loss. *)
let test_export_before () =
  let d, p = P.open_ "loss:export" in
  let h = B.create Rig.host B.paged_from in
  let a = B.export h Bigarray.char in
  ignore
    (submit ~args:one (touching [| Sub.write |] d) ~buffers:[| borrow d h |]);
  lose d p;
  a.{0} <- 'e';
  equal char 'e' a.{0};
  raises_match (lost d) (fun () -> B.wait h Rig.Read)

(* Memory a device used before its loss, once collected and reused from its
   owner's cache, waits for nothing of the lost device. *)
let test_reused_after_loss () =
  let a, _ = P.open_ "loss:owner" in
  let u, pu = P.open_ "loss:user" in
  let n = 4096 in
  let at =
    (fun () ->
      let m = B.create a n in
      run_on u Sub.read (borrow u m);
      Support.at m)
      ()
  in
  P.fail pu;
  raises_match (lost u) (fun () -> submit (empty u));
  Gc.full_major ();
  Gc.full_major ();
  ignore (Sys.opaque_identity (B.create a 0));
  let b = B.create a n in
  equal ~msg:"the cached memory" int at (Support.at b);
  B.wait b Rig.Read_write

(* Hangs *)

let hang_ms = 100
let ms_since t0 = (Rig.Profile.now () - t0) / 1_000_000

let waited_for ~ns pt =
  if Rig.Point.wait_for ~ns pt then "reached" else "not reached"

(* Work that runs long is waited for where the driver states no hang bound:
   three stalled sleeps, then the one that runs it. *)
let test_still () =
  let d, p = P.open_ "loss:still" in
  let pt = submit (empty d) in
  P.stall p 3;
  Rig.Point.wait pt;
  equal (option string) None (Rig.lost d);
  equal ~msg:"sleeps" int 4 (P.sleeps p)

(* A consumer whose queue waits on a faulted producer, whose work only a look at
   the producer surfaces: no sleep of either runs work. *)
let faulted_pair ?(completion = `Store) what =
  let pd, pp = P.open_ (fresh (what ^ "-producer")) in
  let c, pc =
    P.open_ ~completion ~waits_on:[ `Host ] (fresh (what ^ "-consumer"))
  in
  let a = submit (empty pd) in
  let b = submit (empty c) ~waits:[| a |] in
  P.fault pp "the engine hung";
  P.stall pc max_int;
  (pd, c, b)

(* A loop of short bounded waits on the consumer surfaces the producer's fault:
   the consumer's clock outlives each wait, so its waits look. *)
let short_waits ns =
  let pd, c, b = faulted_pair "short-waits" in
  let rec loop () =
    match Rig.Point.wait_for ~ns b with
    | true -> "reached"
    | false -> loop ()
    | exception Rig.Lost (l, why) when Rig.equal l c -> why
  in
  equal string (Rig.name pd ^ " lost") (loop ());
  Rig.close c;
  Rig.close pd

(* A consumer whose word the host writes looks as one whose queue writes it. *)
let test_look_host () =
  let pd, c, b = faulted_pair ~completion:`Host "look-host" in
  let got =
    match Rig.Point.wait b with
    | () -> "reached"
    | exception Rig.Lost (l, why) when Rig.equal l c -> why
  in
  equal string (Rig.name pd ^ " lost") got;
  Rig.close c;
  Rig.close pd

(* A producer lost while a consumer's wait sleeps, its clock past the still
   interval, reaches the consumer once that sleep returns: the wait raises with
   no sleep after it. The producer's sleeps stall too, so a look runs none of
   its work. *)
let test_lost_producer () =
  let pd, pp = P.open_ (fresh "lost-producer") in
  let c, pc = P.open_ ~waits_on:[ `Host ] (fresh "lost-producer-consumer") in
  let a = submit (empty pd) in
  let b = submit (empty c) ~waits:[| a |] in
  P.stall pp max_int;
  P.stall pc max_int;
  P.gate pc;
  let waiter =
    Domain.spawn (fun () ->
        match Rig.Point.wait b with
        | () -> "returned"
        | exception Rig.Lost (l, why) when Rig.equal l c -> why)
  in
  Support.await "the consumer's sleep at the gate" (fun () -> P.sleepers pc = 1);
  let sleeps = P.sleeps pc in
  Rig.lose pd "given up";
  P.open_gate pc;
  equal string ~msg:"the reason" (Rig.name pd ^ " lost") (Domain.join waiter);
  equal ~msg:"sleeps after the loss" int sleeps (P.sleeps pc);
  Rig.close c;
  Rig.close pd

(* A wait looks at every producer its device's queue waits on, however many
   waits name one: a queue of 64 waits on a stalled producer and one on a
   faulted producer surfaces the fault, and the wait raises Lost. *)
let test_many_waits () =
  let a, pa = P.open_ "loss:many-a" in
  let b, pb = P.open_ "loss:many-b" in
  let c, _ =
    P.open_ ~completion:`Store ~waits_on:[ `Host ] "loss:many-consumer"
  in
  let waiting_on producer =
    submit (empty c) ~waits:[| submit (empty producer) |]
  in
  for _ = 1 to 64 do
    ignore (waiting_on a)
  done;
  let last = waiting_on b in
  P.stall pa max_int;
  P.fault pb "the engine hung";
  raises_match (lost c) (fun () -> Rig.Point.wait last);
  equal (option string) (Some "the engine hung") (Rig.lost b)

(* Work whose word stands still past the bound loses the device, not before the
   bound, and the driver's stop, which its close runs, is given the reason. *)
let test_hang () =
  let d, p = P.open_ ~hang_ms "loss:hang" in
  let pt = submit (empty d) in
  P.stall p max_int;
  let t0 = Rig.Profile.now () in
  raises_match (lost d) (fun () -> Rig.Point.wait pt);
  let waited = ms_since t0 in
  let why = Printf.sprintf "no progress for %d ms" hang_ms in
  equal (option string) ~msg:"the loss" (Some why) (Rig.lost d);
  Rig.close d;
  equal (option string) ~msg:"the stop's fault" (Some why) (P.stop_fault p);
  at_least ~msg:"the wait, in ms" int ~than:hang_ms waited

(* Work stalled three bounds with nothing waiting loses nothing, and the clock
   starts at the wait that finds it. *)
let test_hang_unwatched () =
  let d, p = P.open_ ~hang_ms "loss:hang-unwatched" in
  let pt = submit (empty d) in
  P.stall p max_int;
  Thread.delay (3. *. Float.of_int hang_ms /. 1000.);
  equal (option string) ~msg:"unwatched" None (Rig.lost d);
  let t0 = Rig.Profile.now () in
  raises_match (lost d) (fun () -> Rig.Point.wait pt);
  at_least ~msg:"the wait, in ms" int ~than:hang_ms (ms_since t0)

(* Each of the wait's sleeps, which the stall makes last up to the bound, moves
   the word to the next of five values, so each look finds a later value
   whatever the scheduler does, and the wait lasts over three bounds. *)
let test_hang_moving () =
  let d, p = P.open_ ~hang_ms "loss:hang-moving" in
  let s = empty d in
  let points = List.init 5 (fun _ -> submit s) in
  P.stall p max_int;
  let rec move = function
    | [] -> ()
    | pt :: rest ->
        P.before p "sleep" (fun () ->
            P.set_word p (Rig.Point.value pt);
            move rest)
  in
  move points;
  let t0 = Rig.Profile.now () in
  Rig.Point.wait (List.nth points 4);
  let waited = ms_since t0 in
  equal (option string) None (Rig.lost d);
  at_least ~msg:"the wait, in ms" int ~than:(3 * hang_ms) waited

(* The clock counts only while a committed value stays above the word. *)
let test_hang_idle () =
  let d, _ = P.open_ ~hang_ms "loss:hang-idle" in
  let s = empty d in
  Rig.Point.wait (submit s);
  Thread.delay (3. *. Float.of_int hang_ms /. 1000.);
  Rig.Point.wait (submit s);
  equal (option string) None (Rig.lost d)

let test_hang_spread () =
  let producer, pp = P.open_ ~hang_ms "loss:hang-producer" in
  let consumer, _ = P.open_ ~waits_on:[ `Host ] "loss:hang-consumer" in
  let a = submit (empty producer) in
  ignore (submit (empty consumer) ~waits:[| a |]);
  P.stall pp max_int;
  raises_match (lost producer) (fun () -> Rig.Point.wait a);
  equal (option string) (Some "loss:hang-producer lost") (Rig.lost consumer)

(* A consumer's clock starts once its producer's work is reached. The producer's
   work runs only when a sleep runs its queue, and the consumer's first two
   sleeps run nothing: the consumer waits two still intervals, past its bound,
   on a value its producer has not reached. *)
let test_hang_in_queue () =
  let producer, _ = P.open_ "loss:in-queue-producer" in
  let consumer, pc =
    P.open_ ~hang_ms ~waits_on:[ `Host ] "loss:in-queue-consumer"
  in
  let a = submit (empty producer) in
  let b = submit (empty consumer) ~waits:[| a |] in
  P.stall pc 2;
  let t0 = Rig.Profile.now () in
  Rig.Point.wait b;
  let waited = ms_since t0 in
  equal (option string) None (Rig.lost consumer);
  at_least ~msg:"the wait, in ms" int ~than:hang_ms waited

(* The check judges the word as it stands at the check. The wait is held at its
   producer's gate, since a wait whose word stood still for a still interval
   looks at the producers its queue waits on before it sleeps. The clock started
   at the wait's previous sleep, more than the bound before the check. *)
let test_hang_held_read () =
  let producer, pp = P.open_ "loss:held-read-producer" in
  let d, p =
    P.open_ ~hang_ms ~completion:`Store ~waits_on:[ `Host ] "loss:held-read"
  in
  let a = submit (empty producer) in
  let first = submit (empty d) ~waits:[| a |] in
  let b = submit (empty d) in
  Rig.Point.wait a;
  P.stall pp max_int;
  P.gate p;
  P.stall p 1;
  let waiting =
    Domain.spawn (fun () ->
        match Rig.Point.wait b with
        | () -> None
        | exception Rig.Lost (_, why) -> Some why)
  in
  Support.await "the clock's first sleep" (fun () -> P.sleepers p = 1);
  P.gate pp;
  P.open_gate p;
  Support.await "the wait held after its read" (fun () -> P.sleepers pp = 1);
  P.set_word p (Rig.Point.value first);
  Thread.delay (2. *. Float.of_int hang_ms /. 1000.);
  P.open_gate pp;
  equal (option string) ~msg:"the wait's loss" None (Domain.join waiting);
  equal (option string) ~msg:"the loss" None (Rig.lost d)

(* A later wait, past twice the bound, on a queue that waited on its producer
   all along loses nothing. *)
let test_short_wait_bound () =
  let pd, pp = P.open_ (fresh "short-bound-producer") in
  let c, _ =
    P.open_ ~hang_ms ~completion:`Store ~waits_on:[ `Host ]
      (fresh "short-bound-consumer")
  in
  let a = submit (empty pd) in
  let b = submit (empty c) ~waits:[| a |] in
  P.stall pp max_int;
  equal ~msg:"the short wait" string "not reached" (waited_for ~ns:10_000_000 b);
  Thread.delay (2. *. Float.of_int hang_ms /. 1000.);
  P.stall pp 0;
  Rig.Point.wait b;
  equal (option string) ~msg:"the loss" None (Rig.lost c);
  Rig.close c;
  Rig.close pd

let test_hang_two_waiters () =
  let d, p = P.open_ ~hang_ms (fresh "hang-two-waiters") in
  let pt = submit (empty d) in
  P.stall p max_int;
  let t0 = Rig.Profile.now () in
  let wait () =
    match Rig.Point.wait pt with
    | () -> Error "returned"
    | exception Rig.Lost (_, why) -> Ok (why, ms_since t0)
  in
  let other = Domain.spawn wait in
  let mine = wait () in
  let theirs = Domain.join other in
  let why = Printf.sprintf "no progress for %d ms" hang_ms in
  List.iter
    (fun (who, r) ->
      match r with
      | Error e -> failf "%s: the wait %s" who e
      | Ok (w, waited) ->
          equal string ~msg:(who ^ ": the reason") why w;
          at_least ~msg:(who ^ ": the wait, in ms") int ~than:hang_ms waited)
    [ ("this domain", mine); ("the other", theirs) ];
  Rig.close d

(* rig.mli's example, as written there. *)
let close_within ~ns d =
  (match Rig.Point.wait_for ~ns (Rig.Point.submitted d) with
  | true -> ()
  | false -> Rig.lose d "its work did not end in time"
  | exception Rig.Lost _ -> ());
  Rig.close d

let test_close_within () =
  let d, p = P.open_ "loss:close-within" in
  ignore (submit (empty d));
  P.stall p max_int;
  close_within ~ns:10_000_000 d;
  equal (option string) (Some "its work did not end in time") (Rig.lost d);
  equal ~msg:"stops" int 1 (P.stops p)

(* Stops *)

(* After the stop answered, only frees reach the driver: of memory and of
   mappings. *)
let test_after_stop () =
  let d, p = P.open_ "loss:after-stop" in
  let b = B.create d 64 in
  lose d p;
  Rig.close d;
  ignore (Sys.opaque_identity b);
  Gc.full_major ();
  ignore (B.create Rig.host 8);
  let rec after = function
    | "stop" :: rest -> rest
    | _ :: rest -> after rest
    | [] -> []
  in
  List.iter (fun call -> mem string call [ "free"; "unmap" ]) (after (P.log p))

(* Memory of a stopped device dropped once its stop answered is still freed
   through its driver. *)
let test_free_after_stop () =
  let d, p = P.open_ "loss:free-after-stop" in
  let b = ref (Some (B.create d 64)) in
  lose d p;
  Rig.close d;
  b := None;
  collect d;
  equal int 1 (count "free" p)

(* A faulted device's frees after its stop raise its fault, which rig drops: its
   memory and its mapping each go back once, and the drains that free them raise
   nothing. *)
let test_faulted_frees () =
  let d, p = P.open_ "loss:faulted-frees" in
  let h = B.create Rig.host B.paged_from in
  let kept = ref [ B.create d 64; B.create d 128 ] in
  let mapped = ref (Some (borrow d h)) in
  P.fault p "the device fell off the bus";
  raises_match (lost d) (fun () -> B.create d 8);
  Rig.close d;
  ignore (Sys.opaque_identity (!kept, !mapped));
  kept := [];
  mapped := None;
  collect d;
  collect d;
  equal ~msg:"held" (list int) [] (P.outstanding p);
  equal ~msg:"frees and unmaps" (list int) [ 2; 1 ]
    (List.map (fun call -> count call p) [ "free"; "unmap" ]);
  equal (option string) (Some "the device fell off the bus") (Rig.lost d)

(* A stopped device's mapping of host memory dropped once its stop answered is
   still unmapped through its driver. *)
let test_unmap_after_stop () =
  let d, p = P.open_ "loss:unmap-after-stop" in
  let h = ref (Some (B.create Rig.host B.paged_from)) in
  ignore (borrow d (Option.get !h));
  equal ~msg:"mapped" (list int) [ B.paged_from ] (P.host_maps p);
  lose d p;
  Rig.close d;
  h := None;
  collect d;
  equal int 1 (count "unmap" p)

(* A device whose stop answered Unknown keeps its memory until its word reads
   its last value, and frees memory dropped after that. *)
let test_unknown () =
  let d, p = P.open_ ~answer:`Unknown "loss:unknown" in
  let b = B.create d 64 in
  ignore (submit (empty d));
  lose d p;
  Rig.close d;
  ignore (Sys.opaque_identity b);
  collect d;
  equal ~msg:"frees before the word" int 0 (count "free" p);
  P.set_word p (Rig.submitted d);
  collect d;
  equal ~msg:"frees after the word" int 1 (count "free" p)

let test_free_after_unknown () =
  let d, p = P.open_ ~answer:`Unknown "loss:free-after-unknown" in
  let b = ref (Some (B.create d 64)) in
  ignore (submit (empty d));
  lose d p;
  Rig.close d;
  P.set_word p (Rig.submitted d);
  ignore (B.create Rig.host 8);
  b := None;
  collect d;
  equal int 1 (count "free" p)

(* A lost device whose word is its driver's own reports the value it read at the
   loss until its close runs its stop, then the last value its stop wrote
   there. *)
let test_own_word_stop () =
  let d, p = P.open_ ~transport:true "loss:own-word-stop" in
  let base = Rig.submitted d in
  let pt = submit (empty d) in
  P.fault p "the engine hung";
  raises_match (lost d) (fun () -> Rig.Point.wait pt);
  equal ~msg:"the word before the close" int base (Rig.signaled d);
  Rig.close d;
  equal ~msg:"the word" int (Rig.Point.value pt) (Rig.signaled d);
  equal ~msg:"stops" int 1 (count "stop" p)

(* A lost device whose word is its driver's own, and whose stop left its work
   running, reports the values that work writes there once its stop returned. *)
let test_own_word_work () =
  let d, p = P.open_ ~transport:true ~answer:`Unknown "loss:own-word-work" in
  let base = Rig.submitted d in
  let pt = submit (empty d) in
  P.fault p "the engine hung";
  raises_match (lost d) (fun () -> Rig.Point.wait pt);
  Rig.close d;
  equal ~msg:"the word at the loss" int base (Rig.signaled d);
  P.set_word p (Rig.Point.value pt);
  equal ~msg:"the word the work wrote" int (Rig.Point.value pt) (Rig.signaled d)

(* A closed device's memory and images that values reach after the close return
   to the driver, each once, at the first of [d]'s calls [returns] makes after
   they are collected: [free_cache], or a [Buffer.create] that raises the
   loss. *)
let close_frees name returns =
  let d, p = P.open_ name in
  let kept = ref [ B.create d 64; B.create d 128 ] in
  let image = ref (Some (require_ok (Rig.Image.load d "functions"))) in
  Rig.close d;
  equal ~msg:"held at the close" (slist int compare) [ 64; 128 ]
    (P.outstanding p);
  equal ~msg:"loaded at the close" int 0 (count "unload" p);
  ignore (Sys.opaque_identity (!kept, !image));
  kept := [];
  image := None;
  Gc.full_major ();
  returns d;
  equal ~msg:"once the buffers are collected" (list int) [] (P.outstanding p);
  equal ~msg:"frees" int 2 (count "free" p);
  equal ~msg:"unloads" int 1 (count "unload" p)

let test_close_frees () = close_frees "loss:close-frees" Rig.free_cache

let test_create_frees () =
  close_frees "loss:create-frees" (fun d ->
      raises_match (lost d) (fun () -> B.create d 8))

(* [d]'s mappings of memory that outlives it go back at its stop's return, once,
   since it has stopped then; another device's borrow of the memory stays, and
   the memory's death unmaps nothing more on [d]. [ends d] closes or loses
   [d]. *)
let outlived name ends =
  let d, p = P.open_ name in
  let e, pe = P.open_ (name ^ "-other") in
  let h = ref (Some (B.create Rig.host B.paged_from)) in
  let peer = ref (Some (B.create e 64)) in
  ignore (borrow d (Option.get !h));
  ignore (borrow d (Option.get !peer));
  let other = ref (Some (borrow e (Option.get !h))) in
  ends d p;
  equal ~msg:"unmapped at the stop" int 2 (count "unmap" p);
  collect d;
  equal ~msg:"held by the device" (list int) [] (P.outstanding p);
  equal ~msg:"the other device's mapping" (slist int compare)
    [ 64; B.paged_from ] (P.outstanding pe);
  raises_match (lost d) (fun () -> B.borrow d (Option.get !h));
  ignore (Sys.opaque_identity (!h, !peer, !other));
  h := None;
  peer := None;
  other := None;
  collect d;
  Rig.free_cache e;
  equal ~msg:"once the memory is freed" int 2 (count "unmap" p);
  equal ~msg:"the other device's, then" int 1 (count "unmap" pe)

let test_closed_unmaps () =
  outlived "loss:closed-unmaps" (fun d _ -> Rig.close d)

let test_lost_unmaps () =
  outlived "loss:lost-unmaps" (fun d p ->
      lose d p;
      Rig.close d)

(* A mapping of memory that dies before its device ends is released once, with
   the memory. *)
let test_memory_first () =
  let d, p = P.open_ "loss:memory-first" in
  let h = ref (Some (B.create Rig.host B.paged_from)) in
  ignore (borrow d (Option.get !h));
  ignore (Sys.opaque_identity !h);
  h := None;
  collect d;
  equal ~msg:"with the memory" int 1 (count "unmap" p);
  Rig.close d;
  collect d;
  collect d;
  equal ~msg:"after the close" int 1 (count "unmap" p)

(* The mapper's stop and the memory's death race over one mapping: the stopped
   device's unmap waits at Polled's free gate while the memory dies on another
   domain. The mapping is released once. *)
let test_unmap_race_device_first () =
  let d, p = P.open_ "loss:race-device-first" in
  let h = ref (Some (B.create Rig.host B.paged_from)) in
  ignore (borrow d (Option.get !h));
  P.gate_frees p;
  let closer = Domain.spawn (fun () -> Rig.close d) in
  Support.await "the stopped device's unmap at the gate" (fun () ->
      P.freers p = 1);
  ignore (Sys.opaque_identity !h);
  h := None;
  collect d;
  P.open_frees p;
  Domain.join closer;
  collect d;
  equal ~msg:"unmaps" int 1 (count "unmap" p)

(* The memory's death and its mapper's close race over one mapping: the memory's
   unmap waits at Polled's free gate while the close runs on another domain. The
   mapping is released once. *)
let test_unmap_race_memory_first () =
  let d, p = P.open_ "loss:race-memory-first" in
  P.gate_frees p;
  let dropper =
    Domain.spawn (fun () ->
        let h = ref (Some (B.create Rig.host B.paged_from)) in
        ignore (borrow d (Option.get !h));
        ignore (Sys.opaque_identity !h);
        h := None;
        collect d)
  in
  Support.await "the memory's unmap at the gate" (fun () -> P.freers p = 1);
  (* The unmap is a counted call of [d], which [d]'s stop waits for. *)
  let closer =
    Domain.spawn (fun () ->
        Rig.close d;
        collect d;
        collect d)
  in
  Support.await "the close's loss" (fun () -> Rig.lost d <> None);
  equal ~msg:"stopped while the unmap waits" int 0 (count "stop" p);
  P.open_frees p;
  Domain.join dropper;
  Domain.join closer;
  collect d;
  equal ~msg:"unmaps" int 1 (count "unmap" p)

(* A device that maps a host buffer and has stopped after its stop returned: an
   Unknown answer leaves its word unreached at the stop, and the word is set
   after it, with no call since that moves memory. *)
type stopped = { p : P.t; d : Rig.t; h : B.t option ref }

let stopped_made = Atomic.make 0

let[@inline never] make_stopped () =
  let n = Atomic.fetch_and_add stopped_made 1 in
  let d, p = P.open_ ~answer:`Unknown (Printf.sprintf "loss:stopped-%d" n) in
  let h = ref (Some (B.create Rig.host B.paged_from)) in
  ignore (borrow d (Option.get !h));
  ignore (submit (empty d));
  lose d p;
  Rig.close d;
  P.set_word p (Rig.submitted d);
  { p; d; h }

(* The buffer's free: it dies, and the host's drain gives it back. *)
let free_stopped t =
  t.h := None;
  Gc.full_major ();
  Rig.free_cache Rig.host

let drain_stopped t = Rig.free_cache t.d

(* The calls that give a late-stopped device's mappings back. *)
let returning =
  [
    ("free_cache", drain_stopped);
    ( "Buffer.create",
      fun t -> raises_match (lost t.d) (fun () -> B.create t.d 8) );
    ( "Buffer.borrow",
      fun t ->
        raises_match (lost t.d) (fun () -> B.borrow t.d (B.create Rig.host 8))
    );
  ]

let mapping_returns (_, call) =
  let t = make_stopped () in
  equal ~msg:"unmaps before the call" int 0 (count "unmap" t.p);
  call t;
  equal ~msg:"unmaps after it" int 1 (count "unmap" t.p);
  ignore (Sys.opaque_identity !(t.h))

(* The free takes the mapping, and its unmap waits at Polled's free gate on
   another domain while the device's drain runs on a third: the drain finds its
   note of the mapping and leaves the give to the free. *)
let test_drain_beside_free () =
  let t = make_stopped () in
  P.gate_frees t.p;
  let freer = Domain.spawn (fun () -> free_stopped t) in
  Support.await "the free's unmap at the gate" (fun () -> P.freers t.p = 1);
  Domain.join (Domain.spawn (fun () -> drain_stopped t));
  equal ~msg:"unmaps while the free's waits" int 0 (count "unmap" t.p);
  equal ~msg:"the free's unmap, still at the gate" int 1 (P.freers t.p);
  P.open_frees t.p;
  Domain.join freer;
  equal ~msg:"unmaps" int 1 (count "unmap" t.p)

(* The drain takes the mapping, and its unmap waits at Polled's free gate on
   another domain while the buffer's free runs: the free finds the mapping taken
   and leaves the give to the drain. *)
let test_free_beside_drain () =
  let t = make_stopped () in
  P.gate_frees t.p;
  let drainer = Domain.spawn (fun () -> drain_stopped t) in
  Support.await "the drain's unmap at the gate" (fun () -> P.freers t.p = 1);
  Domain.join (Domain.spawn (fun () -> free_stopped t));
  equal ~msg:"unmaps while the drain's waits" int 0 (count "unmap" t.p);
  equal ~msg:"the drain's unmap, still at the gate" int 1 (P.freers t.p);
  P.open_frees t.p;
  Domain.join drainer;
  equal ~msg:"unmaps" int 1 (count "unmap" t.p)

let stopped_order (_, first, second) =
  let t = make_stopped () in
  first t;
  equal ~msg:"unmaps after the first" int 1 (count "unmap" t.p);
  second t;
  equal ~msg:"unmaps after both" int 1 (count "unmap" t.p)

(* A close of a lost device keeps the loss's reason. *)
let test_close_lost () =
  let d, p = P.open_ "loss:close-lost" in
  P.fail p;
  raises_match (lost d) (fun () -> submit (empty d));
  Rig.close d;
  equal (option string) (Some failed_why) (Rig.lost d);
  equal ~msg:"stops" int 1 (count "stop" p)

(* Neither a use that raises the loss, nor a drain of its memory, nor a read of
   its word runs a lost device's stop. Its memory returns once its close has run
   the stop. *)
let test_stop_at_close () =
  let d, p = P.open_ "loss:stop-at-close" in
  let b = ref (Some (B.create d 64)) in
  lose d p;
  let uses =
    [
      (fun () -> ignore (B.create d 8));
      (fun () -> B.fill (Option.get !b) 'x');
      (fun () -> ignore (B.borrow d (B.create Rig.host 8)));
      (fun () -> ignore (Rig.Image.load d "functions"));
      (fun () -> ignore (submit (empty d)));
      (fun () -> ignore (Rig.signaled d));
      (fun () -> Rig.free_cache d);
    ]
  in
  List.iter (fun use -> try use () with Rig.Lost _ -> ()) uses;
  b := None;
  collect d;
  equal ~msg:"stops before the close" int 0 (count "stop" p);
  equal ~msg:"held before the close" (list int) [ 64 ] (P.outstanding p);
  Rig.close d;
  equal ~msg:"stops after the close" int 1 (count "stop" p);
  collect d;
  equal ~msg:"held once stopped" (list int) [] (P.outstanding p)

(* [x1] and [x2], lost together, [x2] by spread, owe their stops. A wait on
   another device [z] that sleeps returns while both stay owed, and the stops
   run at the closes, [x1]'s raising. A stop run after the first blocks until
   the test releases it, so one run inside the wait shows. *)
let test_wait_runs_no_stop () =
  let x1, p1 = P.open_ "loss:owed-1" in
  let x2, p2 = P.open_ ~waits_on:[ `Host ] "loss:owed-2" in
  let raised = Atomic.make false and held = Atomic.make false in
  let go = Atomic.make false in
  let stop () =
    if not (Atomic.exchange raised true) then failwith "the first stop"
    else begin
      Atomic.set held true;
      Support.await "the stop's release" (fun () -> Atomic.get go)
    end
  in
  P.before p1 "stop" stop;
  P.before p2 "stop" stop;
  let z, _ = P.open_ "loss:owed-waiter" in
  let pt = submit (empty z) and waited = Atomic.make None in
  ignore (submit (empty x2) ~waits:[| submit (empty x1) |]);
  P.fail p1;
  raises_match (lost x1) (fun () -> submit (empty x1));
  let waiter =
    Thread.create
      (fun () ->
        Atomic.set waited
          (Some
             (match Rig.Point.wait pt with
             | () -> "returned"
             | exception e -> "raised " ^ Printexc.to_string e)))
      ()
  in
  Support.await "the wait's end, or a stop held in it" (fun () ->
      Atomic.get waited <> None || Atomic.get held);
  let first =
    if Atomic.get held then "a stop held in the wait"
    else Option.get (Atomic.get waited)
  in
  Atomic.set go true;
  Thread.join waiter;
  equal ~msg:"the wait" string "returned" first;
  raises_match Exn.failure (fun () -> Rig.close x1);
  Rig.close x2;
  equal ~msg:"stops" (pair int int) (1, 1) (P.stops p1, P.stops p2)

(* Lost *)

exception Escaped of int * string * float

(* What escapes a driver's allocation, and the reason it loses the device with:
   rig.mli's [Lost], its constructor's name and its arguments, each immediate
   one as its integer, each string quoted and any other as [_]. *)
let escapes =
  [
    (Failure "no queue", "Failure(\"no queue\")");
    (Not_found, "Not_found");
    ( Escaped (3, "x", 1.5),
      Printexc.exn_slot_name (Escaped (0, "", 0.)) ^ "(3, \"x\", _)" );
  ]

let escape_reason (e, why) =
  let d, p = P.open_ (fresh "escape") in
  P.before p "alloc" (fun () -> raise e);
  raises ~msg:"the allocation" e (fun () -> B.create d 8);
  equal (option string) (Some why) (Rig.lost d)

let test_lost_prints () =
  let d, _ = P.open_ "loss:prints" in
  Rig.lose d "given up";
  match B.create d 8 with
  | _ -> fail "a lost device allocated"
  | exception e ->
      equal string "loss:prints lost: given up" (Printexc.to_string e)

(* lose *)

let test_lose () =
  let d, p = P.open_ "loss:lose" in
  let pt = submit (empty d) in
  Rig.lose d "given up";
  Rig.lose d "given up again";
  equal (option string) (Some "given up") (Rig.lost d);
  raises_match
    (function Rig.Lost (d', "given up") -> Rig.equal d d' | _ -> false)
    (fun () -> B.create d 8);
  raises_match (lost d) (fun () -> Rig.Point.wait pt);
  equal ~msg:"stops before the close" int 0 (count "stop" p);
  Rig.close d;
  equal ~msg:"stops after the close" int 1 (count "stop" p);
  equal (option string) ~msg:"the stop's fault" None (P.stop_fault p)

let test_lose_host () =
  raises_match Exn.invalid_arg (fun () -> Rig.lose Rig.host "given up")

(* A waiter on another domain, in a sleep over work that runs long, raises the
   loss once that sleep returns, with no sleep after it. The sleep is held at
   Polled's gate while the device is lost. *)
let test_lose_waiter () =
  let d, p = P.open_ "loss:lose-waiter" in
  let pt = submit (empty d) in
  P.stall p max_int;
  P.gate p;
  let waiter =
    Domain.spawn (fun () ->
        match Rig.Point.wait pt with
        | () -> "returned"
        | exception Rig.Lost (_, why) -> why)
  in
  Support.await "the waiter's sleep at the gate" (fun () -> P.sleepers p = 1);
  let sleeps = P.sleeps p in
  Rig.lose d "given up";
  P.open_gate p;
  equal string ~msg:"the reason" "given up" (Domain.join waiter);
  equal ~msg:"sleeps after the lose" int sleeps (P.sleeps p)

(* A close that another domain blocks in, over work that runs long, returns once
   the device is lost, and runs the stop once, with no fault. *)
let test_lose_close () =
  let d, p = P.open_ "loss:lose-close" in
  ignore (submit (empty d));
  P.stall p max_int;
  let closer = Domain.spawn (fun () -> Rig.close d) in
  Support.await "the close's sleep" (fun () -> P.sleeps p > 0);
  equal ~msg:"stops while the close waits" int 0 (P.stops p);
  Rig.lose d "given up";
  Domain.join closer;
  equal (option string) (Some "given up") (Rig.lost d);
  equal ~msg:"stops" int 1 (P.stops p);
  equal (option string) ~msg:"the stop's fault" None (P.stop_fault p)

let test_lose_spread () =
  let producer, _ = P.open_ "loss:lose-producer" in
  let consumer, _ = P.open_ ~waits_on:[ `Host ] "loss:lose-consumer" in
  let a = submit (empty producer) in
  ignore (submit (empty consumer) ~waits:[| a |]);
  Rig.lose producer "given up";
  equal (option string) (Some "loss:lose-producer lost") (Rig.lost consumer)

(* A lose races a fault the producer's driver recorded, which a wait on the
   producer raises: one of the two loses the producer, and its reason is the one
   every wait raises. The stop sees the fault only if the fault won. Its
   consumer, whose queue waits on the producer, is lost by either, as the
   producer's close shows, which returns once the loss has spread. *)
type race = { mutable why : string option }

let fault_why = "the engine hung"
let given_up = "given up"
let races = Atomic.make 0

let race_open () =
  let n = Atomic.fetch_and_add races 1 in
  let pd, pp = P.open_ (Printf.sprintf "loss:race-producer-%d" n) in
  let c, _ =
    P.open_ ~waits_on:[ `Host ] (Printf.sprintf "loss:race-consumer-%d" n)
  in
  let a = submit (empty pd) in
  ignore (submit (empty c) ~waits:[| a |]);
  P.fault pp fault_why;
  (pd, pp, c, a)

let race_lose r = if r.why = None then r.why <- Some given_up

let race_wait r =
  if r.why = None then r.why <- Some fault_why;
  Option.get r.why

let race_waited (_, _, _, a) =
  match Rig.Point.wait a with
  | () -> "returned"
  | exception Rig.Lost (_, why) -> why

let race_close r =
  if r.why = None then r.why <- Some fault_why;
  ((if r.why = Some fault_why then Some fault_why else None), true)

let race_closed (pd, pp, c, _) =
  Rig.close pd;
  (P.stop_fault pp, Rig.lost c = Some (Rig.name pd ^ " lost"))

let race_release (pd, _, c, _) =
  Rig.close pd;
  Rig.close c

let race_commands =
  let world = abstract "w" ~release:race_release in
  [
    command "open"
      (Gen.unit @-> makes world)
      (fun () -> { why = None })
      race_open;
    command "lose"
      (world ^-> returns unit)
      race_lose
      (fun (pd, _, _, _) -> Rig.lose pd given_up);
    command "wait" (world ^-> returns string) race_wait race_waited;
    command "close"
      (world ^-> returns (pair (option string) bool))
      race_close race_closed;
  ]

let tests =
  [
    group "loss"
      [
        test "a failed hand-over loses the device, its stop owed to its close"
          test_failed_submit;
        test "a fault a wait finds loses the device, its stop owed to its close"
          test_fault;
        test "a fault two domains' sleeps find loses the device once"
          test_two_sleeps;
        test "a lost device's properties keep their values" (test_facts false);
        test
          "a lost device behind a transport reports the value read at its loss"
          (test_facts true);
        test "a queue waiting on a lost device's value is lost" test_spread;
        test "a queue whose wait was reached stays" test_no_spread;
        test "a submit waiting for a lost device's unreached value raises"
          test_waits_unreached;
        test "a wait for a lost device's point raises iff the loss dropped it"
          test_reached;
        test "a reached point stays done when its device's word read loses it"
          test_reached_at_loss;
      ];
    group "memory after a loss"
      [
        cases ~id:use_name "a read of memory a lost device only read runs" reads
          read_after_read;
        cases ~id:use_name
          "every other use of memory a lost device only read raises its loss \
           until its stop"
          others other_after_read;
        cases ~id:use_name
          "memory a lost device last wrote raises its loss at every use, also \
           after its stop"
          (reads @ others) use_after_write;
        test "an unordered use of memory a lost device last wrote runs"
          test_unordered;
        test
          "a read a loss dropped counts as done once a later device took its \
           slot"
          test_read_after_stop;
        test "memory a refused submit named does not raise Lost"
          test_unrun_names;
        test "a submit whose driver fails to take it stamps the memory it names"
          test_failed_take_stamps;
        test "memory whose points a lost device reached is ordinary"
          test_others_go_on;
        test
          "memory a lost device read raises for writes until its stop, whoever \
           reads it meanwhile"
          test_unreached_kept;
        test "a stop's last value reaches no unreached work"
          test_stop_reaches_nothing;
        test "a lost device's memory raises its loss through any buffer"
          test_own_memory;
        test "a lost device's borrow raises its loss, the memory not"
          test_lost_borrow;
        test "an array exported before the loss reads and writes with no check"
          test_export_before;
        test "memory a lost device used, reused, waits for nothing lost"
          test_reused_after_loss;
      ];
    group "hangs"
      [
        test "a word still for three intervals loses nothing" test_still;
        cases ~id:string_of_int
          "a loop of short waits surfaces a producer's recorded fault"
          [ 1; 50_000_000 ] short_waits;
        test "a wait on a host-written word surfaces a producer's fault"
          test_look_host;
        test "a lost producer reaches a consumer's wait once its sleep returns"
          test_lost_producer;
        test
          "a wait looks at every producer its queue waits on, however many \
           waits name one"
          test_many_waits;
        test "work still past the hang bound loses the device" test_hang;
        test "work nothing waits for loses nothing to the bound"
          test_hang_unwatched;
        test "a word moving within the bound loses nothing" test_hang_moving;
        test "an idle device loses nothing to the bound" test_hang_idle;
        test "a queue waiting on a hung device is lost with it" test_hang_spread;
        test "a queue waiting on longer work elsewhere loses nothing"
          test_hang_in_queue;
        test "a word that moved while a wait was held loses nothing"
          test_hang_held_read;
        test "a short wait leaves the bound's clock alone" test_short_wait_bound;
        test "the bound comes no sooner with two waiters" test_hang_two_waiters;
        test "a close bounded in time loses and closes a stalled device"
          test_close_within;
      ];
    group "stops"
      [
        test "after its stop a device's driver is asked only to free and unmap"
          test_after_stop;
        test "a stopped device frees memory dropped after its stop"
          test_free_after_stop;
        test "a faulted device's frees after its stop raise nothing"
          test_faulted_frees;
        test "a stopped device unmaps host memory dropped after its stop"
          test_unmap_after_stop;
        test "an Unknown answer keeps memory until the word drains" test_unknown;
        test "an Unknown answer frees memory dropped after the word drained"
          test_free_after_unknown;
        test
          "a lost device behind a transport reports its stop's value before a \
           close"
          test_own_word_stop;
        test
          "a lost device behind a transport reports the values its running \
           work writes"
          test_own_word_work;
        test
          "a closed device's memory and images return once collected, at \
           free_cache"
          test_close_frees;
        test
          "a closed device's memory and images return once collected, at a \
           Buffer.create that raises"
          test_create_frees;
        test
          "a closed device's mappings of memory that outlives it go back at \
           its stop, once"
          test_closed_unmaps;
        test
          "a lost device's mappings of memory that outlives it go back at its \
           stop, once"
          test_lost_unmaps;
        test "a mapping of memory that dies first ends with the memory, once"
          test_memory_first;
        test
          "a stopped device's unmap held while the memory dies on another \
           domain: one unmap"
          test_unmap_race_device_first;
        test
          "a dead memory's unmap held while its mapper closes on another \
           domain: one unmap"
          test_unmap_race_memory_first;
        cases ~id:fst
          "a device that stopped after its stop returned gives its mappings \
           back at the first call on it that moves memory"
          returning mapping_returns;
        test
          "a Stopped device's drain while the free's unmap waits leaves the \
           mapping to the free"
          test_drain_beside_free;
        test
          "a Stopped device's free while the drain's unmap waits leaves the \
           mapping to the drain"
          test_free_beside_drain;
        cases
          ~id:(fun (order, _, _) -> order)
          "a Stopped device's free and drain, in either order, unmap once"
          [
            ("free, then drain", free_stopped, drain_stopped);
            ("drain, then free", drain_stopped, free_stopped);
          ]
          stopped_order;
        test "a close of a lost device keeps its loss's reason" test_close_lost;
        test "a lost device's stop runs at its close alone" test_stop_at_close;
        test "a wait runs no other device's stop" test_wait_runs_no_stop;
      ];
    group "Lost"
      [
        cases
          ~id:(fun (_, why) -> why)
          "an exception escaping a driver loses the device with its \
           constructor, its immediate arguments as integers, its strings \
           quoted and any other argument as _"
          escapes escape_reason;
        test "Lost prints as NAME lost: why" test_lost_prints;
      ];
    group "lose"
      [
        test "a lose gives the reason every later use raises" test_lose;
        test "the host is never lost" test_lose_host;
        test "a waiter on another domain raises the loss once its sleep returns"
          test_lose_waiter;
        test "a close blocked on long work returns once the device is lost"
          test_lose_close;
        test "a queue waiting on a device given up is lost" test_lose_spread;
        stateful ~domains:2 ~count:40 ~steps:6
          "a lose racing a recorded fault loses the producer once, for one \
           reason"
          race_commands;
      ];
  ]

let () = exit (run "rig.loss" tests)
