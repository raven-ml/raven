(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support

let timeout = 60.
let empty d = Sub.make d [||]

let submit ?(buffers = [||]) ?(waits = [||]) s =
  Rig.submit s ~run:(Sub.Run.make ()) ~buffers ~waits

let lost d = function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false
let count call p = List.length (List.filter (( = ) call) (P.log p))

let test_failed_submit () =
  let d, p = P.open_ "loss:failed" in
  let s = empty d in
  ignore (submit s);
  P.fail p;
  raises_match (lost d) (fun () -> submit s);
  equal (option string) (Some "the submission failed") (Rig.lost d);
  raises_match (lost d) (fun () -> submit s);
  raises_match (lost d) (fun () -> B.create d 8);
  equal int 1 (count "stop" p)

let test_fault () =
  let d, p = P.open_ "loss:fault" in
  let v = Rig.Point.value (submit (empty d)) in
  P.fault p "the engine hung";
  raises_match (lost d) (fun () -> Rig.wait d v);
  equal (option string) (Some "the engine hung") (Rig.lost d);
  equal int 1 (count "stop" p)

(* After the stop answered, only frees reach the driver: of memory, of
   mappings and of the word. *)
let test_after_stop () =
  let d, p = P.open_ "loss:after-stop" in
  let b = B.create d 64 in
  P.fail p;
  (try ignore (submit (empty d)) with Rig.Lost _ -> ());
  ignore (Sys.opaque_identity b);
  Gc.full_major ();
  ignore (B.create Rig.host 8);
  let rec after = function
    | "stop" :: rest -> rest
    | _ :: rest -> after rest
    | [] -> []
  in
  List.iter
    (fun call -> mem string call [ "free"; "unmap"; "word" ])
    (after (P.log p))

(* A submit refused because its device is lost runs no work on its buffers'
   memory, which does not raise Lost for that device. *)
let test_unrun_names () =
  let d, p = P.open_ "loss:unrun" in
  let h = B.create Rig.host (1 lsl 16) in
  let s = Sub.make ~access:[| B.Read |] d [||] in
  let buffers = [| require_some (B.borrow d h) |] in
  P.fail p;
  raises_match (lost d) (fun () -> submit (empty d));
  raises_match (lost d) (fun () -> submit s ~buffers);
  equal int (1 lsl 16) (Bigarray.Array1.dim (B.bigarray Bigarray.char h))

(* A wait looks at every producer its device's queue waits on, however many
   waits name one: a queue of 64 waits on a stalled producer and one on a
   faulted producer surfaces the fault, and the wait raises Lost. *)
let test_many_waits () =
  let a, pa = P.open_ "loss:many-a" in
  let b, pb = P.open_ "loss:many-b" in
  let c, _ =
    P.open_ ~completion:`Object ~waits_on:[ `Host ] "loss:many-consumer"
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

(* Loses [d] through a failed hand-over, then drains as an allocation does:
   [d]'s stop has answered and nothing of it is left to drain. *)
let lose d p =
  P.fail p;
  (try ignore (submit (empty d)) with Rig.Lost _ -> ());
  ignore (B.create Rig.host 8)

(* Collects, then drains as an allocation does. *)
let collect () =
  Gc.full_major ();
  Gc.full_major ();
  ignore (B.create Rig.host 8)

(* Memory of a stopped device dropped once its stop answered is still freed
   through its driver. *)
let test_free_after_stop () =
  let d, p = P.open_ "loss:free-after-stop" in
  let b = ref (Some (B.create d 64)) in
  lose d p;
  b := None;
  collect ();
  equal int 1 (count "free" p)

(* A stopped device's mapping of host memory dropped once its stop answered is
   still unmapped through its driver. *)
let test_unmap_after_stop () =
  let d, p = P.open_ "loss:unmap-after-stop" in
  let h = ref (Some (B.create Rig.host (1 lsl 16))) in
  ignore (require_some (B.borrow d (Option.get !h)));
  equal ~msg:"mapped" (list int) [ 1 lsl 16 ] (P.host_maps p);
  lose d p;
  h := None;
  collect ();
  equal int 1 (count "unmap" p)

(* A device whose stop answered Unknown frees memory dropped after its word read
   its last value. *)
let test_free_after_unknown () =
  let d, p = P.open_ ~answer:`Unknown "loss:free-after-unknown" in
  let b = ref (Some (B.create d 64)) in
  ignore (submit (empty d));
  lose d p;
  P.set_word p (Rig.submitted d);
  ignore (B.create Rig.host 8);
  b := None;
  collect ();
  equal int 1 (count "free" p)

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

(* A consumer whose waited value was reached stays. *)
let test_no_spread () =
  let producer, pp = P.open_ "loss:producer-2" in
  let consumer, _ = P.open_ ~waits_on:[ `Host ] "loss:consumer-2" in
  let a = submit (empty producer) in
  Rig.Point.wait a;
  ignore (submit (empty consumer) ~waits:[| a |]);
  P.fail pp;
  raises_match (lost producer) (fun () -> submit (empty producer));
  equal (option string) None (Rig.lost consumer)

(* An Unknown answer keeps the device's memory until its word reads its last
   value. *)
let test_unknown () =
  let d, p = P.open_ ~answer:`Unknown "loss:unknown" in
  let b = B.create d 64 in
  ignore (submit (empty d));
  P.fail p;
  (try ignore (submit (empty d)) with Rig.Lost _ -> ());
  ignore (Sys.opaque_identity b);
  Gc.full_major ();
  ignore (B.create Rig.host 8);
  equal int 0 (count "free" p);
  P.set_word p (Rig.submitted d);
  ignore (B.create Rig.host 8);
  equal int 1 (count "free" p)

(* A wait checks the loss after its value: a reached value of a lost device
   raises. *)
let test_reached () =
  let d, p = P.open_ "loss:reached" in
  let v = Rig.Point.value (submit (empty d)) in
  Rig.wait d v;
  P.fail p;
  raises_match (lost d) (fun () -> submit (empty d));
  raises_match (lost d) (fun () -> Rig.wait d v)

(* Two domains sleep on a device that faults: each raises its Lost, and the
   device is lost and stopped once. *)
let test_two_sleeps () =
  let d, p = P.open_ "loss:two-sleeps" in
  let v = Rig.Point.value (submit (empty d)) in
  P.gate p;
  let wait () =
    match Rig.wait d v with
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
  equal int 1 (count "stop" p)

(* A lost device still answers its facts: its name, architecture, budget and
   capability as before, the last value handed over and the last value its word
   reached. Its work may still run, so its stop leaves the word where it
   stood. *)
let test_facts transport () =
  let d, p =
    P.open_ ~transport ~answer:`Unknown
      (if transport then "loss:facts-transport" else "loss:facts")
  in
  Rig.set_budget d 4096;
  ignore (submit (empty d));
  ignore (submit (empty d));
  ignore (P.run p);
  ignore (submit (empty d));
  P.fault p "the engine hung";
  raises_match (lost d) (fun () -> Rig.wait d 3);
  equal bool true (String.starts_with ~prefix:"loss:facts" (Rig.name d));
  equal string "polled" (Rig.arch d);
  equal int 4096 (Rig.budget d);
  is_some (Rig.capability d P.capability_key);
  equal int 3 (Rig.submitted d);
  equal int 2 (Rig.signaled d);
  equal (option string) (Some "the engine hung") (Rig.lost d)

let test_printed () =
  let d, p = P.open_ "loss:printed" in
  P.fail p;
  match submit (empty d) with
  | _ -> failf "the failed submit returned"
  | exception e ->
      equal string "loss:printed lost: the submission failed"
        (Printexc.to_string e)

(* A loss reaches the lost device; memory of other devices whose points it
   reached is ordinary memory again, and other devices go on. *)
let test_others_go_on () =
  let d, p = P.open_ "loss:lost" in
  let e, _ = P.open_ "loss:kept" in
  let named = B.create e 64 in
  let s = Sub.make ~access:[| B.Read |] d [||] in
  Rig.Point.wait (submit s ~buffers:[| require_some (B.borrow d named) |]);
  P.fail p;
  raises_match (lost d) (fun () -> submit (empty d));
  B.wait named B.Read_write;
  Rig.Claim.read named;
  Rig.Claim.release named;
  let w = Sub.make ~access:[| B.Read_write |] e [||] in
  Rig.Point.wait (submit w ~buffers:[| named |]);
  equal (option string) None (Rig.lost e)

(* Other memory raises a lost device's loss for a use that waits for a point the
   device did not reach, and only for such a use: a read follows the last write
   alone. *)
let test_unreached () =
  let d, p = P.open_ "loss:unreached" in
  let h = B.create Rig.host (1 lsl 16) in
  let reads = Sub.make ~access:[| B.Read |] d [||] in
  ignore (submit reads ~buffers:[| require_some (B.borrow d h) |]);
  P.fail p;
  raises_match (lost d) (fun () -> submit (empty d));
  B.wait h B.Read;
  raises_match ~msg:"a write's wait" (lost d) (fun () ->
      B.wait h B.Read_write);
  raises_match ~msg:"a bigarray" (lost d) (fun () ->
      B.bigarray Bigarray.char h);
  raises_match ~msg:"a claim" (lost d) (fun () -> Rig.Claim.read h)

(* A stop raises the word to the last value whatever ran: work the device did
   not reach before its loss stays unreached. *)
let test_stop_reaches_nothing () =
  let d, p = P.open_ "loss:stop-raises" in
  let e, _ = P.open_ "loss:stop-other" in
  let m = B.create e 64 in
  let writes = Sub.make ~access:[| B.Read_write |] d [||] in
  let b = require_some (B.borrow d m) in
  let v = Rig.Point.value (submit writes ~buffers:[| b |]) in
  P.fail p;
  raises_match (lost d) (fun () -> submit (empty d));
  equal ~msg:"the word after the stop" bool true (Rig.signaled d >= v);
  raises_match (lost d) (fun () -> B.wait m B.Read)

(* A lost device's memory raises its loss whatever its points, also with no
   work ever on it, through any buffer: its own, a borrow on the host, and a
   borrow on another device. *)
let test_own_memory () =
  let d, p = P.open_ ~copies:false "loss:own" in
  let e, _ = P.open_ "loss:own-peer" in
  let m = B.create d 64 in
  let h = require_some (B.borrow Rig.host m) in
  let b = require_some (B.borrow e m) in
  let dst = B.create Rig.host 64 in
  P.fail p;
  raises_match (lost d) (fun () -> submit (empty d));
  List.iter
    (fun (what, f) -> raises_match ~msg:what (lost d) f)
    [
      ("a wait", fun () -> B.wait m B.Read);
      ("a wait on the host's borrow", fun () -> B.wait h B.Read);
      ("a bigarray", fun () -> ignore (B.bigarray Bigarray.char h));
      ("a claim", fun () -> Rig.Claim.read h);
      ("a copy", fun () -> B.copy ~src:h ~dst);
      ("a borrow", fun () -> ignore (B.borrow Rig.host m));
      ( "a submit on another device",
        fun () ->
          let s = Sub.make ~access:[| B.Read |] e [||] in
          ignore (submit s ~buffers:[| b |]) );
    ];
  equal (option string) None (Rig.lost e)

(* A lost device's borrow of other memory raises its loss; the memory itself,
   which the device's work never used, is ordinary. *)
let test_lost_borrow () =
  let d, p = P.open_ "loss:borrow" in
  let h = B.create Rig.host (1 lsl 16) in
  let b = require_some (B.borrow d h) in
  P.fail p;
  raises_match (lost d) (fun () -> submit (empty d));
  raises_match (lost d) (fun () -> B.wait b B.Read);
  B.wait h B.Read_write

(* Memory a device used before its loss, once collected and reused from its
   owner's cache, waits for nothing of the lost device: a loss leaves other
   devices' memory working. *)
let test_reused_after_loss () =
  let a, _ = P.open_ "loss:owner" in
  let u, pu = P.open_ "loss:user" in
  let n = 4096 in
  let at =
    (fun () ->
      let m = B.create a n in
      let s = Sub.make ~access:[| B.Read |] u [||] in
      Rig.Point.wait (submit s ~buffers:[| require_some (B.borrow u m) |]);
      B.address m)
      ()
  in
  P.fail pu;
  raises_match (lost u) (fun () -> submit (empty u));
  Gc.full_major ();
  Gc.full_major ();
  ignore (Sys.opaque_identity (B.create ~memory:Pinned a 8));
  let b = B.create a n in
  equal ~msg:"the cached memory" int at (B.address b);
  B.wait b B.Read_write

(* Closing *)

(* A close waits for the work submitted before it, then ends the device as a
   loss does, with the reason "closed": its stop runs once, before close
   returns. *)
let test_close () =
  let d, p = P.open_ "loss:close" in
  let v = Rig.Point.value (submit (empty d)) in
  Rig.close d;
  equal ~msg:"the work ran" int v (Rig.signaled d);
  equal (option string) (Some "closed") (Rig.lost d);
  equal int 1 (count "stop" p);
  match submit (empty d) with
  | _ -> failf "a closed device took a submit"
  | exception (Rig.Lost (d', why) as e) ->
      equal bool true (Rig.equal d d');
      equal string "closed" why;
      equal string "loss:close lost: closed" (Printexc.to_string e)

(* Two domains close one device: its stop runs once, and both closes return
   once it answered. The stop waits at the gate while the second close is
   sampled for 50 ms. *)
let test_two_closes () =
  let d, p = P.open_ "loss:two-closes" in
  P.gate p;
  let returned = Atomic.make 0 in
  let close () =
    Rig.close d;
    Atomic.incr returned;
    count "stop" p
  in
  let first = Domain.spawn close in
  Support.await "the stop at the gate" (fun () -> P.sleepers p = 1);
  let second = Domain.spawn close in
  Unix.sleepf 0.05;
  equal ~msg:"closes returned while the stop waits" int 0 (Atomic.get returned);
  equal ~msg:"stops at the gate" int 1 (P.sleepers p);
  P.open_gate p;
  equal ~msg:"the stops each close saw" (list int) [ 1; 1 ]
    [ Domain.join first; Domain.join second ];
  equal (option string) (Some "closed") (Rig.lost d)

(* A closed device's name opens a new device. *)
let test_close_reopen () =
  let d, _ = P.open_ "loss:reopen" in
  Rig.close d;
  let d', _ = P.open_ "loss:reopen" in
  equal bool false (Rig.equal d d');
  equal (option string) None (Rig.lost d');
  equal int 1 (Rig.Point.value (submit (empty d')))

(* A closed device's memory that buffers reach after the close returns to the
   driver as they are collected, each region once. *)
let test_close_frees () =
  let d, p = P.open_ "loss:close-frees" in
  let kept = ref [ B.create d 64; B.create ~memory:Pinned d 128 ] in
  Rig.close d;
  equal ~msg:"held at the close" (slist int compare) [ 64; 128 ]
    (P.outstanding p);
  ignore (Sys.opaque_identity !kept);
  kept := [];
  collect ();
  equal ~msg:"once the buffers are collected" (list int) [] (P.outstanding p);
  equal ~msg:"frees" int 2 (count "free" p)

(* A close of a lost device only waits for its stop: the device keeps its
   loss's reason. *)
(* Mappings end with the shorter-lived of their memory and their device. *)

(* [d]'s mapping of memory that outlives it is released once [d]'s word ends,
   once; another device's borrow of the memory stays, and the memory's death
   unmaps nothing more on [d]. [ends d] closes or loses [d]. *)
let outlived name ends =
  let d, p = P.open_ name in
  let e, pe = P.open_ (name ^ "-other") in
  let h = ref (Some (B.create Rig.host (1 lsl 16))) in
  let peer = ref (Some (B.create e 64)) in
  ignore (require_some (B.borrow d (Option.get !h)));
  ignore (require_some (B.borrow d (Option.get !peer)));
  let other = ref (Some (require_some (B.borrow e (Option.get !h)))) in
  ends d p;
  collect ();
  collect ();
  equal ~msg:"unmapped at the word's end" int 2 (count "unmap" p);
  equal ~msg:"held by the device" (list int) [] (P.outstanding p);
  equal ~msg:"the other device's mapping" (slist int compare)
    [ 64; 1 lsl 16 ] (P.outstanding pe);
  raises_match (lost d) (fun () -> B.borrow d (Option.get !h));
  ignore (Sys.opaque_identity (!h, !peer, !other));
  h := None;
  peer := None;
  other := None;
  collect ();
  equal ~msg:"once the memory is collected" int 2 (count "unmap" p);
  equal ~msg:"the other device's, then" int 1 (count "unmap" pe)

let test_closed_unmaps () = outlived "loss:closed-unmaps" (fun d _ -> Rig.close d)
let test_lost_unmaps () = outlived "loss:lost-unmaps" (fun d p -> lose d p)

(* A mapping of memory that dies before its device ends is released once, with
   the memory. *)
let test_memory_first () =
  let d, p = P.open_ "loss:memory-first" in
  let h = ref (Some (B.create Rig.host (1 lsl 16))) in
  ignore (require_some (B.borrow d (Option.get !h)));
  ignore (Sys.opaque_identity !h);
  h := None;
  collect ();
  equal ~msg:"with the memory" int 1 (count "unmap" p);
  Rig.close d;
  collect ();
  collect ();
  equal ~msg:"after the close" int 1 (count "unmap" p)

(* The device's word end and the memory's death race over one mapping: each
   order is held with the first one's unmap waiting at Polled's free gate,
   while the other runs on another domain. The mapping is released once. *)
let test_unmap_race_device_first () =
  let d, p = P.open_ "loss:race-device-first" in
  let h = ref (Some (B.create Rig.host (1 lsl 16))) in
  ignore (require_some (B.borrow d (Option.get !h)));
  Rig.close d;
  P.gate_frees p;
  let ender =
    Domain.spawn (fun () ->
        collect ();
        collect ())
  in
  Support.await "the word end's unmap at the gate" (fun () ->
      P.freers p = 1);
  ignore (Sys.opaque_identity !h);
  h := None;
  collect ();
  P.open_frees p;
  Domain.join ender;
  collect ();
  equal ~msg:"unmaps" int 1 (count "unmap" p)

let test_unmap_race_memory_first () =
  let d, p = P.open_ "loss:race-memory-first" in
  P.gate_frees p;
  let dropper =
    Domain.spawn (fun () ->
        let h = ref (Some (B.create Rig.host (1 lsl 16))) in
        ignore (require_some (B.borrow d (Option.get !h)));
        ignore (Sys.opaque_identity !h);
        h := None;
        collect ())
  in
  Support.await "the memory's unmap at the gate" (fun () -> P.freers p = 1);
  (* The unmap is a counted call of [d], which [d]'s stop waits for. *)
  let closer =
    Domain.spawn (fun () ->
        Rig.close d;
        collect ();
        collect ())
  in
  Support.await "the close's loss" (fun () -> Rig.lost d <> None);
  equal ~msg:"stopped while the unmap waits" int 0 (count "stop" p);
  P.open_frees p;
  Domain.join dropper;
  Domain.join closer;
  collect ();
  equal ~msg:"unmaps" int 1 (count "unmap" p)

let test_close_lost () =
  let d, p = P.open_ "loss:close-lost" in
  P.fail p;
  raises_match (lost d) (fun () -> submit (empty d));
  Rig.close d;
  equal (option string) (Some "the submission failed") (Rig.lost d);
  equal int 1 (count "stop" p)

(* A close whose device faults while its work runs ends the device lost with
   the fault's reason. *)
let test_close_fault () =
  let d, p = P.open_ "loss:close-fault" in
  ignore (submit (empty d));
  P.fault p "the engine hung";
  Rig.close d;
  equal (option string) (Some "the engine hung") (Rig.lost d);
  equal int 1 (count "stop" p)

(* A stopped device's timeline word goes back to its driver once nothing
   reads it, after the stop: the next drains move its readers off it, then
   give it back once every domain passed a minor collection. *)
let test_close_word () =
  let d, p = P.open_ "loss:close-word" in
  ignore (submit (empty d));
  Rig.close d;
  collect ();
  collect ();
  let rec after = function
    | "stop" :: rest -> rest
    | _ :: rest -> after rest
    | [] -> []
  in
  equal (list string) [ "word" ]
    (List.filter (( = ) "word") (after (P.log p)));
  equal ~msg:"the last value" int 1 (Rig.signaled d)

(* A word another device's queue waits on goes back only once that wait ran,
   after that device's mapping of it. *)
let test_word_waited () =
  let producer, pp = P.open_ "loss:word-producer" in
  let consumer, pc = P.open_ ~waits_on:[ `Host ] "loss:word-consumer" in
  let v = submit (empty producer) in
  ignore (submit (empty consumer) ~waits:[| v |]);
  equal ~msg:"waited in the queue" int 1 (List.length (P.last_waits pc));
  Rig.close producer;
  collect ();
  collect ();
  equal ~msg:"while the wait is queued" int 0 (count "word" pp);
  ignore (P.run pc);
  collect ();
  collect ();
  equal ~msg:"once it ran" int 1 (count "word" pp);
  equal ~msg:"the consumer's mapping" int 1 (count "unmap" pc)

(* A release on a device lost and still stopping, here a consumer's mapping of a
   closed producer's word, waits for the stop, and never raises the consumer's
   loss to whoever drains: the consumer's stop waits at the gate on another
   domain while the producer's word ends. *)
let test_release_while_stopping () =
  let producer, _ = P.open_ "loss:stopping-producer" in
  let consumer, pc = P.open_ ~waits_on:[ `Host ] "loss:stopping-consumer" in
  let v = submit (empty producer) in
  let w = submit (empty consumer) ~waits:[| v |] in
  Rig.Point.wait w;
  P.gate pc;
  let loser =
    Domain.spawn (fun () ->
        P.fail pc;
        try ignore (submit (empty consumer)) with Rig.Lost _ -> ())
  in
  Support.await "the consumer's stop at the gate" (fun () -> P.sleepers pc = 1);
  Rig.close producer;
  let drained =
    match
      collect ();
      collect ()
    with
    | () -> "drained"
    | exception e -> Printexc.to_string e
  in
  equal ~msg:"the drains" string "drained" drained;
  equal ~msg:"the mapping, while the stop runs" int 0 (count "unmap" pc);
  P.open_gate pc;
  Domain.join loser;
  collect ();
  equal ~msg:"the mapping, once stopped" int 1 (count "unmap" pc)

(* The word of a device whose work may still run is never given back. *)
let test_word_unknown () =
  let d, p = P.open_ ~answer:`Unknown "loss:word-unknown" in
  ignore (submit (empty d));
  lose d p;
  collect ();
  collect ();
  equal int 0 (count "word" p)

let test_close_host () =
  raises_match Exn.invalid_arg (fun () -> Rig.close Rig.host)

(* An io device whose stops are counted. *)
module Store = struct
  type t = int Atomic.t
  type region = unit

  exception Fault of string

  let region_key : region Type.Id.t = Type.Id.make ()
  let budget _ = max_int
  let alloc _ _ = Some ()
  let free _ () = ()
  let read _ () ~at:_ ~dst:_ ~len:_ = ()
  let write _ () ~at:_ ~src:_ ~len:_ = ()
  let pages _ () = None
  let prefetch _ () ~at:_ ~len:_ = ()
  let stop stops = Atomic.incr stops
end

(* An io device's close ends it at once: its work is the caller's. *)
let test_close_io () =
  let stops = Atomic.make 0 in
  let io =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_io (module Store) ~name:"loss:close-io" (fun () -> Ok stops))
  in
  let b = B.create io 8 in
  Rig.close io;
  equal (option string) (Some "closed") (Rig.lost io);
  equal int 1 (Atomic.get stops);
  raises_match (lost io) (fun () -> B.copy ~src:b ~dst:(B.create Rig.host 8))

(* A fault a counted call raises, such as an allocation's, loses the device. *)
let test_alloc_fault () =
  let d, p = P.open_ "loss:alloc" in
  P.fault p "the device fell off the bus";
  raises_match (lost d) (fun () -> B.create d 64);
  equal (option string) (Some "the device fell off the bus") (Rig.lost d);
  equal int 1 (count "stop" p)

let tests =
  [
    group ~timeout "loss"
      [
        test "a failed hand-over loses the device once" test_failed_submit;
        test "a fault a wait finds loses the device" test_fault;
        test "a stopped device is only freed" test_after_stop;
        test "memory a refused submit named does not raise Lost"
          test_unrun_names;
        test ~timeout:10. "a wait finds a fault past 64 waits on one producer"
          test_many_waits;
        test "a stopped device frees memory dropped after its stop"
          test_free_after_stop;
        test "a stopped device unmaps host memory dropped after its stop"
          test_unmap_after_stop;
        test "an Unknown answer frees memory dropped after the word drained"
          test_free_after_unknown;
        test "a queue waiting on a lost device's value is lost" test_spread;
        test "a queue whose wait was reached stays" test_no_spread;
        test "an Unknown answer keeps memory until the word drains" test_unknown;
        test "a wait on a lost device raises once its value is reached"
          test_reached;
        test "a fault two domains' sleeps find loses the device once"
          test_two_sleeps;
        test "a lost device answers its facts and values" (test_facts false);
        test "a lost device behind a transport answers its last reading"
          (test_facts true);
        test "Lost prints the device and the reason" test_printed;
        test "memory whose points a lost device reached is ordinary"
          test_others_go_on;
        test "memory a lost device's work did not reach raises its loss"
          test_unreached;
        test "a stop's last value reaches no unreached work"
          test_stop_reaches_nothing;
        test "a lost device's memory raises its loss through any buffer"
          test_own_memory;
        test "a lost device's borrow raises its loss, the memory not"
          test_lost_borrow;
        test "memory a lost device used, reused, waits for nothing lost"
          test_reused_after_loss;
        test "a fault an allocation raises loses the device" test_alloc_fault;
      ];
    group ~timeout "close"
      [
        test "a close waits for the work, then ends the device" test_close;
        test
          "two domains closing one device stop it once, and both return after \
           it (sampled for 50 ms)"
          test_two_closes;
        test "a closed device's name opens a new device" test_close_reopen;
        test "a closed device's memory returns as it is collected"
          test_close_frees;
        test
          "a closed device's mappings of memory that outlives it end with its \
           word, once"
          test_closed_unmaps;
        test
          "a lost device's mappings of memory that outlives it end with its \
           word, once"
          test_lost_unmaps;
        test "a mapping of memory that dies first ends with the memory, once"
          test_memory_first;
        test
          "a word end's unmap held while the memory dies on another domain: \
           one unmap"
          test_unmap_race_device_first;
        test
          "a dead memory's unmap held while its mapper closes on another \
           domain: one unmap"
          test_unmap_race_memory_first;
        test "a close of a lost device waits for its stop" test_close_lost;
        test "a fault during a close's wait is the device's loss"
          test_close_fault;
        test "a closed device's word goes back once nothing reads it"
          test_close_word;
        test "a word another queue waits on goes back once the wait ran"
          test_word_waited;
        test
          "a release on a device still stopping waits for its stop and raises \
           nothing"
          test_release_while_stopping;
        test "the word of a device whose work may run is kept"
          test_word_unknown;
        test "the host is never closed" test_close_host;
        test "an io device's close ends it at once" test_close_io;
      ];
  ]

let () = exit (run "rig.loss" tests)
