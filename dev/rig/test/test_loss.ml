(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module C = Rig
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support

let timeout = 60.
let empty d = Sub.make ~reads:0 ~writes:0 d [||]

let submit ?(reads = [||]) ?(writes = [||]) ?(waits = [||]) s =
  C.submit s ~reads ~writes ~waits

let lost d = function C.Lost (d', _) -> C.equal d d' | _ -> false
let count call p = List.length (List.filter (( = ) call) (P.log p))

let test_failed_submit () =
  let d, p = P.open_ "loss:failed" in
  let s = empty d in
  ignore (submit s);
  P.fail p;
  raises_match (lost d) (fun () -> submit s);
  equal (option string) (Some "the submission failed") (C.lost d);
  raises_match (lost d) (fun () -> submit s);
  raises_match (lost d) (fun () -> B.create d 8);
  equal int 1 (count "stop" p)

let test_fault () =
  let d, p = P.open_ "loss:fault" in
  let v = C.Point.value (submit (empty d)) in
  P.fault p "the engine hung";
  raises_match (lost d) (fun () -> C.wait d v);
  equal (option string) (Some "the engine hung") (C.lost d);
  equal int 1 (count "stop" p)

(* After the stop answered, only free and unmap reach the driver. *)
let test_after_stop () =
  let d, p = P.open_ "loss:after-stop" in
  let b = B.create d 64 in
  P.fail p;
  (try ignore (submit (empty d)) with C.Lost _ -> ());
  ignore (Sys.opaque_identity b);
  Gc.full_major ();
  ignore (B.create C.host 8);
  let rec after = function
    | "stop" :: rest -> rest
    | _ :: rest -> after rest
    | [] -> []
  in
  List.iter (fun call -> mem string call [ "free"; "unmap" ]) (after (P.log p))

(* A submit refused because its device is lost runs no work on its buffers'
   memory, which does not raise Lost for that device. *)
let test_unrun_names () =
  let d, p = P.open_ "loss:unrun" in
  let h = B.create C.host (1 lsl 16) in
  let s = Sub.make ~reads:1 ~writes:0 d [||] in
  let reads = [| require_some (B.borrow d h) |] in
  P.fail p;
  raises_match (lost d) (fun () -> submit (empty d));
  raises_match (lost d) (fun () -> submit s ~reads);
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
  raises_match (lost c) (fun () -> C.wait c (C.Point.value last));
  equal (option string) (Some "the engine hung") (C.lost b)

(* Loses [d] through a failed hand-over, then drains as an allocation does:
   [d]'s stop has answered and nothing of it is left to drain. *)
let lose d p =
  P.fail p;
  (try ignore (submit (empty d)) with C.Lost _ -> ());
  ignore (B.create C.host 8)

(* Collects, then drains as an allocation does. *)
let collect () =
  Gc.full_major ();
  Gc.full_major ();
  ignore (B.create C.host 8)

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
  let h = ref (Some (B.create C.host (1 lsl 16))) in
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
  P.set_word p (C.submitted d);
  ignore (B.create C.host 8);
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
  equal (option string) (Some "loss:producer lost") (C.lost consumer)

(* A consumer whose waited value was reached stays. *)
let test_no_spread () =
  let producer, pp = P.open_ "loss:producer-2" in
  let consumer, _ = P.open_ ~waits_on:[ `Host ] "loss:consumer-2" in
  let a = submit (empty producer) in
  C.wait producer (C.Point.value a);
  ignore (submit (empty consumer) ~waits:[| a |]);
  P.fail pp;
  raises_match (lost producer) (fun () -> submit (empty producer));
  equal (option string) None (C.lost consumer)

(* An Unknown answer keeps the device's memory until its word reads its last
   value. *)
let test_unknown () =
  let d, p = P.open_ ~answer:`Unknown "loss:unknown" in
  let b = B.create d 64 in
  ignore (submit (empty d));
  P.fail p;
  (try ignore (submit (empty d)) with C.Lost _ -> ());
  ignore (Sys.opaque_identity b);
  Gc.full_major ();
  ignore (B.create C.host 8);
  equal int 0 (count "free" p);
  P.set_word p (C.submitted d);
  ignore (B.create C.host 8);
  equal int 1 (count "free" p)

(* A wait checks the loss after its value: a reached value of a lost device
   raises. *)
let test_reached () =
  let d, p = P.open_ "loss:reached" in
  let v = C.Point.value (submit (empty d)) in
  C.wait d v;
  P.fail p;
  raises_match (lost d) (fun () -> submit (empty d));
  raises_match (lost d) (fun () -> C.wait d v)

(* Two domains sleep on a device that faults: each raises its Lost, and the
   device is lost and stopped once. *)
let test_two_sleeps () =
  let d, p = P.open_ "loss:two-sleeps" in
  let v = C.Point.value (submit (empty d)) in
  P.gate p;
  let wait () =
    match C.wait d v with () -> "returned" | exception C.Lost (_, why) -> why
  in
  let waiters = List.init 2 (fun _ -> Domain.spawn wait) in
  Support.await "two sleeps at the gate" (fun () -> P.sleepers p = 2);
  P.fault p "the engine hung";
  P.open_gate p;
  equal (list string)
    [ "the engine hung"; "the engine hung" ]
    (List.map Domain.join waiters);
  equal (option string) (Some "the engine hung") (C.lost d);
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
  C.set_budget d 4096;
  ignore (submit (empty d));
  ignore (submit (empty d));
  ignore (P.run p);
  ignore (submit (empty d));
  P.fault p "the engine hung";
  raises_match (lost d) (fun () -> C.wait d 3);
  equal bool true (String.starts_with ~prefix:"loss:facts" (C.name d));
  equal string "polled" (C.arch d);
  equal int 4096 (C.budget d);
  is_some (C.capability d P.capability_key);
  equal int 3 (C.submitted d);
  equal int 2 (C.signaled d);
  equal (option string) (Some "the engine hung") (C.lost d)

let test_printed () =
  let d, p = P.open_ "loss:printed" in
  P.fail p;
  match submit (empty d) with
  | _ -> failf "the failed submit returned"
  | exception e ->
      equal string "loss:printed lost: the submission failed"
        (Printexc.to_string e)

(* A loss reaches the lost device and the memory its stamps name, even work that
   was reached; other devices and their memory go on. *)
let test_others_go_on () =
  let d, p = P.open_ "loss:lost" in
  let e, _ = P.open_ "loss:kept" in
  let named = B.create e 64 and kept = B.create e 64 in
  let s = Sub.make ~reads:1 ~writes:0 d [||] in
  C.wait d
    (C.Point.value (submit s ~reads:[| require_some (B.borrow d named) |]));
  P.fail p;
  raises_match (lost d) (fun () -> submit (empty d));
  raises_match (lost d) (fun () -> B.wait named B.Read_write);
  B.wait kept B.Read_write;
  let w = Sub.make ~reads:0 ~writes:1 e [||] in
  C.wait e (C.Point.value (submit w ~writes:[| kept |]));
  equal (option string) None (C.lost e)

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
      let s = Sub.make ~reads:1 ~writes:0 u [||] in
      C.wait u
        (C.Point.value (submit s ~reads:[| require_some (B.borrow u m) |]));
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

(* A fault a counted call raises, such as an allocation's, loses the device. *)
let test_alloc_fault () =
  let d, p = P.open_ "loss:alloc" in
  P.fault p "the device fell off the bus";
  raises_match (lost d) (fun () -> B.create d 64);
  equal (option string) (Some "the device fell off the bus") (C.lost d);
  equal int 1 (count "stop" p)

let tests =
  [
    group ~timeout "loss"
      [
        test "a failed hand-over loses the device once" test_failed_submit;
        test "a fault a wait finds loses the device" test_fault;
        test "a stopped device is only freed and unmapped" test_after_stop;
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
        test "a loss leaves other devices and their memory working"
          test_others_go_on;
        test "memory a lost device used, reused, waits for nothing lost"
          test_reused_after_loss;
        test "a fault an allocation raises loses the device" test_alloc_fault;
      ];
  ]

let () = exit (run "rig.loss" tests)
