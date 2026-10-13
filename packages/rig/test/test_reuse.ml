(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Device slots taken again: a device that opens after another closed may
   continue that device's timeline, and an earlier device's points and stamps
   answer as they did. Polled devices run their queues only when a wait sleeps
   or the test runs them, so a device lost before any wait loses its queued
   work. *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support

let submit ?(buffers = [||]) ?waits s =
  Rig.submit ?waits s ~run:(Sub.Run.make ()) ~buffers

let timeout = 120.

(* A device lost, stopped and opened again under its name, beside another
   domain's uses *)

(* A world: a device under [name], lost and opened again by the cycles; a
   consumer that never closes, whose memory [b] the cycles' devices write
   through a borrow, and whose queue waits for their points; a waiter, whose
   queue waits for no device, so that the host waits for its submissions'
   points; and the points of every cycle, with whether a wait for each
   returned. *)
type world = {
  name : string;
  cycling : Mutex.t;
  current : (Rig.t * P.t) Atomic.t;
  consumer : Rig.t;
  waiter : Rig.t;
  b : B.t;
  points : (Rig.Point.t * bool) list Atomic.t;
}

let worlds = Atomic.make 0

let make_world () =
  let n = Atomic.fetch_and_add worlds 1 in
  let name = Printf.sprintf "reuse:cycled-%d" n in
  let consumer, _ =
    P.open_ ~waits_on:[ `Host ] (Printf.sprintf "reuse:consumer-%d" n)
  in
  {
    name;
    cycling = Mutex.create ();
    current = Atomic.make (P.open_ name);
    consumer;
    waiter = fst (P.open_ (Printf.sprintf "reuse:waiter-%d" n));
    b = B.create consumer 64;
    points = Atomic.make [];
  }

let close_world w =
  Rig.close (fst (Atomic.get w.current));
  Rig.close w.consumer;
  Rig.close w.waiter

let rec record w p =
  let l = Atomic.get w.points in
  if not (Atomic.compare_and_set w.points l (p :: l)) then record w p

let waited q =
  match Rig.Point.wait q with () -> true | exception Rig.Lost _ -> false

(* The model: whether [b]'s last write was lost, after which every ordered
   use of [b] raises [Lost] for good; whether a lost use or write of [b]
   makes every write of it raise for good, or a use whose device was lost
   during it may; whether a cycle lost a point, after which a submit that
   waits for every point raises it; and whether the consumer was lost, as its
   queue waited for a point a cycle lost, after which every use of [b], its
   memory, raises. *)
type model = {
  mutable poisoned : bool;
  mutable writes_raise : bool;
  mutable maybe_lost_use : bool;
  mutable any_lost : bool;
  mutable consumer_lost : bool;
}

let pp_model ppf r =
  Format.fprintf ppf
    "poisoned %b, writes raise %b, a use maybe lost %b, a point lost %b, \
     consumer lost %b"
    r.poisoned r.writes_raise r.maybe_lost_use r.any_lost r.consumer_lost

let world = abstract ~pp:pp_model ~release:close_world "w"

(* What a cycle did: its device wrote [b] at the point it then waited for,
   which returned or raised [Lost]; or its write raised [Lost], [b]'s last
   write being lost, and the point it waited for is its empty work's. *)
type cycle = Wrote of bool | Poisoned of bool

let pp_cycle ppf = function
  | Wrote ok -> Format.fprintf ppf "wrote, wait returned %b" ok
  | Poisoned ok -> Format.fprintf ppf "poisoned, wait returned %b" ok

let cycle_t = pair (Testable.make ~pp:pp_cycle ~equal:( = )) bool

(* Loses the current device after it wrote [b], or tried to, with its work
   queued unless [reached]: it faults, so its next counted call, or a sleep of
   another domain's wait for it, loses it, and its close waits for its stop.
   Then opens its name again, whose device takes the slot the lost one left,
   and waits for the point. Cycles take turns. *)
let cycle_locked ~reached w =
  let d, p = Atomic.get w.current in
  let write () =
    let bd = require_some (B.borrow d w.b) in
    submit (Sub.make d [| Support.touching [| Sub.write |] |]) ~buffers:[| bd |]
  in
  let q, poisoned =
    match write () with
    | q -> (q, false)
    | exception Rig.Lost _ -> (submit (Sub.make d [||]), true)
  in
  if reached then Rig.Point.wait q;
  P.fault p "cycled";
  (match B.create d 64 with
  | _ -> fail "an allocation of a faulted device returned"
  | exception Rig.Lost _ -> ());
  Rig.close d;
  let d', p' = P.open_ w.name in
  Atomic.set w.current (d', p');
  let ok = waited q in
  record w (q, ok);
  ((if poisoned then Poisoned ok else Wrote ok), Rig.submitted d' > 0)

let cycle_system ~reached w =
  Mutex.protect w.cycling (fun () -> cycle_locked ~reached w)

(* Judges a cycle's outcome, and whether the device opened again continued a
   timeline: the slot of the lost one, or another's. *)
let cycle_model ~reached r = function
  | Error e -> raise e
  | Ok (c, continued) -> (
      cover "a cycle's device continues a timeline" continued;
      match c with
      | Wrote ok ->
          is_false ~msg:"a write after a lost use"
            (r.writes_raise || r.consumer_lost);
          if reached then is_true ~msg:"the reached point's wait" ok;
          r.maybe_lost_use <- false;
          if not ok then begin
            r.poisoned <- true;
            r.writes_raise <- true;
            r.any_lost <- true
          end
      | Poisoned ok ->
          is_true ~msg:"a write raised Lost"
            (r.writes_raise || r.maybe_lost_use || r.consumer_lost);
          if reached then is_true ~msg:"the reached point's wait" ok;
          r.writes_raise <- true;
          r.maybe_lost_use <- false;
          if not ok then r.any_lost <- true)

(* The waiter's submit after every point a cycle recorded, which the host
   waits for: whether it raised [Lost] for another device, and whether one of
   those points was lost, by a device whose slot a later device took or by
   any. *)
let wait_all_system w =
  let points = Atomic.get w.points in
  let lost = List.exists (fun (_, ok) -> not ok) points in
  let reused =
    List.exists (fun (q, ok) -> (not ok) && Rig.Point.device q = None) points
  in
  let waits = Array.of_list (List.map fst points) in
  let raised =
    match Rig.Point.wait (submit ~waits (Sub.make w.waiter [||])) with
    | () -> false
    | exception Rig.Lost (d, _) when not (Rig.equal d w.waiter) -> true
  in
  (raised, lost, reused)

(* The submit raises iff a point it waited for was lost. *)
let wait_all_model _ = function
  | Error e -> raise e
  | Ok (raised, lost, reused) ->
      cover "a host wait for a point lost on a slot taken again" reused;
      equal ~msg:"the submit raised Lost for a lost point" bool lost raised

(* A host read of [b]: [true] if it returned, [false] if it raised [Lost]:
   for a lost write of it, or once the consumer, its memory's device, is
   lost. *)
let read_system w =
  match B.wait w.b Rig.Read with () -> true | exception Rig.Lost _ -> false

(* The consumer's read of [b] after every point a cycle recorded: [`Ok],
   [`Lost_self] if the consumer is lost, or [`Lost_other]. *)
let consume_system w =
  let waits = Array.of_list (List.map fst (Atomic.get w.points)) in
  let read () =
    let s = Sub.make w.consumer [| Support.touching [| Sub.read |] |] in
    Rig.Point.wait (submit s ~waits ~buffers:[| w.b |])
  in
  match read () with
  | () -> `Ok
  | exception Rig.Lost (d, _) when Rig.equal d w.consumer -> `Lost_self
  | exception Rig.Lost _ -> `Lost_other

(* A consumer whose queue waits for a cycle's point that the cycle's device
   loses is lost with it, which may come before the cycle records the
   point. *)
let consume_model r = function
  | Error e -> raise e
  | Ok `Ok ->
      is_false ~msg:"a read after a lost point"
        (r.poisoned || r.any_lost || r.consumer_lost)
  | Ok `Lost_other ->
      is_true ~msg:"another device's loss" (r.poisoned || r.any_lost)
  | Ok `Lost_self -> r.consumer_lost <- true

(* The current device's read of [b]: [`Ok], [`Lost_self] if that device was
   lost meanwhile, or [`Lost_other] for a loss of another device. *)
let use_system w =
  let d, _ = Atomic.get w.current in
  match
    match B.borrow d w.b with
    | None -> false
    | Some bd ->
        let s = Sub.make d [| Support.touching [| Sub.read |] |] in
        Rig.Point.wait (submit s ~buffers:[| bd |]);
        true
  with
  | true -> `Ok
  | false -> `Lost_self
  | exception Rig.Lost (d', _) when Rig.equal d d' -> `Lost_self
  | exception Rig.Lost _ -> `Lost_other

let pp_use ppf = function
  | `Ok -> Format.pp_print_string ppf "Ok"
  | `Lost_self -> Format.pp_print_string ppf "Lost_self"
  | `Lost_other -> Format.pp_print_string ppf "Lost_other"

let use_t = Testable.make ~pp:pp_use ~equal:( = )

(* A write or a use that the device lost raises at a borrow, which depends on
   every point, for good: a borrow by the device that took the lost one's slot
   raises it too. A device the cycles lose meanwhile may raise its own loss
   instead, and its read of [b] may be lost. *)
let use_model r = function
  | Error e -> raise e
  | Ok `Ok ->
      is_false ~msg:"a borrow of memory a lost use left"
        (r.writes_raise || r.consumer_lost);
      r.maybe_lost_use <- false
  | Ok `Lost_other ->
      is_true ~msg:"another device's loss"
        (r.writes_raise || r.maybe_lost_use || r.consumer_lost);
      r.writes_raise <- true;
      r.maybe_lost_use <- false
  | Ok `Lost_self -> r.maybe_lost_use <- true

(* Each recorded point still answers its first wait: a later device on its
   slot changes no answer. With whether a point a loss dropped names no device
   any more: its slot passed on. *)
let points_system w =
  let points = Atomic.get w.points in
  ( List.for_all (fun (q, ok) -> waited q = ok && Rig.Point.reached q) points,
    List.exists (fun (q, ok) -> (not ok) && Rig.Point.device q = None) points
  )

let points_model _ = function
  | Error e -> raise e
  | Ok (answers, reused) ->
      cover "a lost device's slot taken again" reused;
      is_true ~msg:"every point answers as before" answers

let cycle_commands =
  [
    command "world" (Gen.unit @-> makes world)
      (fun () ->
        {
          poisoned = false;
          writes_raise = false;
          maybe_lost_use = false;
          any_lost = false;
          consumer_lost = false;
        })
      (fun () -> make_world ());
    command "cycle lost"
      (world ^-> judges cycle_t)
      (cycle_model ~reached:false)
      (cycle_system ~reached:false);
    command "cycle reached"
      (world ^-> judges cycle_t)
      (cycle_model ~reached:true)
      (cycle_system ~reached:true);
    command "host read" (world ^-> returns bool)
      (fun r -> not (r.poisoned || r.consumer_lost))
      read_system;
    command "consume" (world ^-> judges use_t) consume_model consume_system;
    command "use" (world ^-> judges use_t) use_model use_system;
    command "wait all"
      (world ^-> judges (triple bool bool bool))
      wait_all_model wait_all_system;
    command "points" (world ^-> judges (pair bool bool)) points_model
      points_system;
  ]

(* Waits in a queue on a slot taken again *)

(* A consumer whose queue waits for a producer behind a transport reads that
   producer's own word. A later device on the producer's slot has a word of
   its own, which the consumer's waits for its points read. *)
let test_pair_per_occupant () =
  let consumer, _ = P.open_ ~waits_on:[ `Host ] "reuse:pair-consumer" in
  let wait_on () =
    let d, p = P.open_ ~transport:true "reuse:pair-producer" in
    let q = submit (Sub.make d [||]) in
    let c = submit ~waits:[| q |] (Sub.make consumer [||]) in
    ignore (P.run p);
    Rig.Point.wait c;
    (d, q)
  in
  let d, q = wait_on () in
  Rig.close d;
  Rig.free_cache d;
  let d', _ = wait_on () in
  equal ~msg:"the first producer's point once its slot passed on"
    (option bool) None
    (Option.map (fun _ -> true) (Rig.Point.device q));
  Rig.close d';
  Rig.close consumer

(* A closed device answers its own last value, also while a later device's
   make on its slot raises the slot's word. *)
let test_raised_by_make () =
  let d, _ = P.open_ "reuse:raised-first" in
  let q = submit (Sub.make d [||]) in
  Rig.Point.wait q;
  Rig.close d;
  Rig.free_cache d;
  let last = Rig.submitted d and during = ref None in
  let p = P.make () in
  let make ~word =
    let p = P.bind ~start:(last + 100) p ~word in
    during := Some (Rig.signaled d);
    Ok p
  in
  let later =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_ { (P.opener "reuse:raised-later" p) with make })
  in
  equal ~msg:"the later device took the slot" int (last + 100)
    (Rig.submitted later);
  equal ~msg:"the closed device's word while the later one opened"
    (option int) (Some last) !during;
  Rig.close later

(* The end of a timeline *)

(* The last value a point holds ends a device's timeline: a submit after it
   loses the device with the reason and assigns nothing, and the slot, whose
   word passed 2^46, goes to no later device. *)
let test_last_value () =
  let last = (1 lsl 47) - 1 in
  let p = P.make () in
  let d =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_ (P.opener ~start:(last - 1) "reuse:last-value" p))
  in
  let q = submit (Sub.make d [||]) in
  equal ~msg:"the last value" int last (Rig.Point.value q);
  Rig.Point.wait q;
  raises_match
    (function
      | Rig.Lost (d', why) ->
          Rig.equal d d'
          && why = "its timeline reached the last value a point holds, 2^47 - 1"
      | _ -> false)
    (fun () -> submit (Sub.make d [||]));
  equal ~msg:"no value assigned past it" int last (Rig.submitted d);
  Rig.close d;
  let later, _ = P.open_ "reuse:after-last-value" in
  less ~msg:"a later device's base" int ~than:(1 lsl 46) (Rig.submitted later)

let tests =
  [
    group ~timeout "reuse"
      [
        test "a closed device answers its value while a make raises its slot"
          test_raised_by_make;
        test "a submit past the last value loses the device, whose slot retires"
          test_last_value;
        test ~timeout:10.
          "a queue's wait for a producer's point reads the slot's current \
           device"
          test_pair_per_occupant;
        stateful ~count:40 ~domains:2
          "a slot taken again answers an earlier device's points and stamps"
          cycle_commands;
      ];
  ]

let () = exit (run "rig.reuse" tests)
