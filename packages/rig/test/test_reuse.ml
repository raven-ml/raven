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

(* A world: a device under [name], lost and opened again by the cycles,
   every other one with a word of its own, which the consumer's queue waits
   on through a mapping; a consumer that never closes, whose memory [b] the
   cycles' devices write through a borrow, and whose queue waits for their
   points; a waiter, whose queue waits for no device, so that the host waits
   for its submissions' points; and the points of every cycle, with whether a
   wait for each returned. *)
type world = {
  name : string;
  cycling : Mutex.t;
  current : (Rig.t * P.t) Atomic.t;
  mutable transport : bool;  (** The current device's, under [cycling]. *)
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
    transport = false;
    consumer;
    waiter = fst (P.open_ (Printf.sprintf "reuse:waiter-%d" n));
    b = B.create consumer 64;
    points = Atomic.make [];
  }

let close_world w =
  Rig.close (fst (Atomic.get w.current));
  Rig.close w.consumer;
  Rig.close w.waiter;
  equal ~msg:"Polled's violations" (list string) [] (P.violations ())

let rec record w p =
  let l = Atomic.get w.points in
  if not (Atomic.compare_and_set w.points l (p :: l)) then record w p

let waited q =
  match Rig.Point.wait q with () -> true | exception Rig.Lost _ -> false

let polled q =
  match Rig.Point.reached q with r -> r | exception Rig.Lost _ -> false

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
  w.transport <- not w.transport;
  let d', p' = P.open_ ~transport:w.transport w.name in
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

(* Each recorded point still answers its first wait, and [reached] answers
   as the wait does: a later device on its slot changes no answer. With
   whether a point a loss dropped names no device any more: its slot passed
   on. *)
let points_system w =
  let points = Atomic.get w.points in
  ( List.for_all (fun (q, ok) -> waited q = ok && polled q = ok) points,
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

(* Words of the driver's own *)

let no_violations () =
  equal ~msg:"Polled's violations" (list string) [] (P.violations ())

(* A thread running [f], whose outcome [join] gives. *)
let spawn f =
  let r = ref (Error Exit) in
  let t = Thread.create (fun () -> r := try Ok (f ()) with e -> Error e) () in
  fun () ->
    Thread.join t;
    match !r with Ok x -> x | Error e -> raise e

(* A device whose word is its driver's own gives it back once it closed:
   opens and closes leave no word behind. *)
let test_own_word_back () =
  for i = 1 to 64 do
    let d, p = P.open_ ~transport:true "reuse:own-word" in
    Rig.Point.wait (submit (Sub.make d [||]));
    Rig.close d;
    Rig.free_cache d;
    equal ~msg:(Printf.sprintf "the word's frees after close %d" i) int 1
      (P.word_frees p)
  done;
  no_violations ()

(* A consumer whose queue waits on a producer's word of its own completes
   after the producer closed: the word stays until the consumer let go of
   it. *)
let test_wait_on_closed () =
  let consumer, c = P.open_ ~waits_on:[ `Host ] "reuse:closed-consumer" in
  let d, p = P.open_ ~transport:true "reuse:closed-producer" in
  let q = submit (Sub.make d [||]) in
  let w = submit ~waits:[| q |] (Sub.make consumer [||]) in
  equal ~msg:"the consumer's waits in its queue" int 1
    (List.length (P.last_waits c));
  Rig.close d;
  Rig.free_cache d;
  equal ~msg:"the closed producer's word while a queue waits on it" int 0
    (P.word_frees p);
  ignore (P.run c);
  Rig.Point.wait w;
  Rig.close consumer;
  equal ~msg:"the word's frees once the consumer closed" int 1 (P.word_frees p);
  no_violations ()

(* A device given a word outside rig's array keeps it as its driver's own:
   rig frees it once, after the device stopped. *)
let test_given_outside () =
  let p = P.make () in
  let outside = Rig_gpu_support.Host.word () in
  let make ~word:_ = Ok (P.bind p ~word:outside) in
  let d =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_ { (P.opener "reuse:given-outside" p) with make })
  in
  Rig.Point.wait (submit (Sub.make d [||]));
  Rig.close d;
  Rig.free_cache d;
  equal ~msg:"the word's frees" int 1 (P.word_frees p);
  no_violations ()

(* Whether [c] gave back a mapping of [p]'s word, which lies at the word's
   address. *)
let unmapped c p = List.exists (fun (a, _) -> a = P.word_at p) (P.frees c)

(* A consumer that waits in its queue on a slot whose devices keep reopening
   keeps a mapping of the word of the device it waits on now. One of an
   earlier device goes back once its queue passed the waits that read it, at
   its next wait on a later device of any slot or at a call that moves its
   collected memory, also while its queue lags. *)
let test_mappings_released () =
  let consumer, c = P.open_ ~waits_on:[ `Host ] "reuse:released-consumer" in
  let name = "reuse:released-producer" in
  let last = ref None in
  (* Opens the slot's next device and submits the consumer's wait on it,
     after closing the one before. *)
  let wait_on () =
    Option.iter
      (fun d ->
        Rig.close d;
        Rig.free_cache d)
      !last;
    let d, p = P.open_ ~transport:true name in
    last := Some d;
    let q = submit (Sub.make d [||]) in
    let w = submit ~waits:[| q |] (Sub.make consumer [||]) in
    equal ~msg:"the consumer's waits in its queue" int 1
      (List.length (P.last_waits c));
    ignore (P.run p);
    (p, w)
  in
  let p1, w1 = wait_on () in
  Rig.Point.wait w1;
  let p2, w2 = wait_on () in
  Rig.Point.wait w2;
  Rig.free_cache consumer;
  equal ~msg:"the first device's mapping at a call that moves memory" bool
    true (unmapped c p1);
  let p3, w3 = wait_on () in
  Rig.Point.wait w3;
  (* The queue lags from here: it has not passed the wait on the fourth
     device. *)
  ignore (wait_on () : P.t * Rig.Point.t);
  equal ~msg:"the second device's mapping at a wait on the fourth" bool true
    (unmapped c p2);
  let _, w5 = wait_on () in
  equal ~msg:"the third device's mapping while the queue lags" bool false
    (unmapped c p3);
  ignore (P.run c);
  Rig.Point.wait w5;
  Rig.free_cache consumer;
  equal ~msg:"the third device's mapping once the queue passed its waits" bool
    true (unmapped c p3);
  Option.iter Rig.close !last;
  Rig.close consumer;
  no_violations ()

(* Collects a buffer of [d] that [d]'s work used, out of the caller's
   frame. *)
let cache_used d =
  let b = B.create d 64 in
  let s = Sub.make d [| Support.touching [| Sub.write |] |] in
  Rig.Point.wait (submit s ~buffers:[| b |])
[@@inline never]

(* A submit that waits on a later device of a slot gives back its device's
   mappings of words, and frees no memory, since a free may wait for the
   GPU's work: memory its device released, and a word whose last mapping
   went back, go back at the next call that moves memory. *)
let test_submit_frees () =
  let consumer, c = P.open_ ~waits_on:[ `Host ] "reuse:frees-consumer" in
  let name = "reuse:frees-producer" in
  let wait_on () =
    let d, p = P.open_ ~transport:true name in
    let q = submit (Sub.make d [||]) in
    let w = submit ~waits:[| q |] (Sub.make consumer [||]) in
    ignore (P.run p);
    Rig.Point.wait w;
    (d, p)
  in
  let d1, p1 = wait_on () in
  cache_used consumer;
  Gc.full_major ();
  ignore (B.create consumer 128 : B.t);
  (* The cached memory's free waits for a value the queue reaches after the
     free was asked. *)
  ignore (submit (Sub.make consumer [||]) : Rig.Point.t);
  Rig.free_cache consumer;
  ignore (P.run c);
  Rig.close d1;
  Rig.free_cache d1;
  let frees_since n = List.filter (( = ) "free") (List.drop n (P.log c)) in
  let calls = List.length (P.log c) in
  let d2, _ = wait_on () in
  Rig.close d2;
  Rig.free_cache d2;
  let d3, _ = wait_on () in
  equal ~msg:"the consumer's frees in the submits" (list string) []
    (frees_since calls);
  equal ~msg:"the first device's mapping at the wait on the third" bool true
    (unmapped c p1);
  equal ~msg:"the first device's word frees after that submit" int 0
    (P.word_frees p1);
  let calls = List.length (P.log c) in
  Rig.free_cache consumer;
  equal ~msg:"the first device's word frees at the next drain" int 1
    (P.word_frees p1);
  equal ~msg:"the consumer's frees at the next drain" (list string) [ "free" ]
    (frees_since calls);
  Rig.close d3;
  Rig.close consumer;
  no_violations ()

(* A decision for a producer that a later device of its slot replaced comes
   too late: the consumer keeps its pair of the later device, and the first
   producer's word goes back. The decision is held at the locate of its
   mapping while the slot passes on. *)
let test_stale_decision () =
  let consumer, c = P.open_ ~waits_on:[ `Host ] "reuse:stale-consumer" in
  (* The consumer's mapping of rig's array, made before the gate. *)
  let a, _ = P.open_ "reuse:stale-array" in
  let qa = submit (Sub.make a [||]) in
  Rig.Point.wait (submit ~waits:[| qa |] (Sub.make consumer [||]));
  let d, p = P.open_ ~transport:true "reuse:stale-producer" in
  let q = submit (Sub.make d [||]) in
  P.gate_locates c;
  let stale =
    spawn (fun () -> submit ~waits:[| q |] (Sub.make consumer [||]))
  in
  Support.await "the decision's locate at the gate" (fun () ->
      P.locaters c = 1);
  Rig.close d;
  Rig.free_cache d;
  let d', p' = P.open_ "reuse:stale-later" in
  let q' = submit (Sub.make d' [||]) in
  let w' = submit ~waits:[| q' |] (Sub.make consumer [||]) in
  P.open_locates c;
  let w = stale () in
  ignore (P.run p');
  Rig.Point.wait w';
  Rig.Point.wait w;
  let q'' = submit (Sub.make d' [||]) in
  let w'' = submit ~waits:[| q'' |] (Sub.make consumer [||]) in
  equal ~msg:"the consumer's waits on the later device, in its queue" int 1
    (List.length (P.last_waits c));
  ignore (P.run p');
  Rig.Point.wait w'';
  Rig.close d';
  Rig.close a;
  Rig.close consumer;
  equal ~msg:"the first producer's word frees" int 1 (P.word_frees p);
  no_violations ()

(* A consumer lost while its queue waits on another device's word keeps the
   word until it has stopped: its stop leaves its work, which may still read
   it. *)
let test_lost_consumer () =
  let consumer, c =
    P.open_ ~waits_on:[ `Host ] ~answer:`Unknown "reuse:lost-consumer"
  in
  let d, p = P.open_ ~transport:true "reuse:lost-producer" in
  let q = submit (Sub.make d [||]) in
  let w = submit ~waits:[| q |] (Sub.make consumer [||]) in
  P.fault c "lost";
  raises_match
    (function Rig.Lost (d', _) -> Rig.equal d' consumer | _ -> false)
    (fun () -> B.create consumer 64);
  Rig.close d;
  Rig.free_cache d;
  Rig.free_cache consumer;
  equal ~msg:"the word's frees while the consumer has not stopped" int 0
    (P.word_frees p);
  ignore (P.run c);
  equal ~msg:"the consumer's word" int (Rig.Point.value w)
    (Rig.signaled consumer);
  Rig.free_cache consumer;
  equal ~msg:"the word's frees once the consumer stopped" int 1 (P.word_frees p);
  Rig.close consumer;
  no_violations ()

(* An exception while a consumer maps a producer's word leaves the word to go
   back. *)
let test_map_raises () =
  let consumer, c = P.open_ ~waits_on:[ `Host ] "reuse:raise-consumer" in
  let d, p = P.open_ ~transport:true "reuse:raise-producer" in
  let q = submit (Sub.make d [||]) in
  P.before c "map_peer" (fun () -> raise Exit);
  raises Exit (fun () ->
      ignore (submit ~waits:[| q |] (Sub.make consumer [||])));
  Rig.close d;
  Rig.close consumer;
  equal ~msg:"the word's frees" int 1 (P.word_frees p);
  no_violations ()

(* An exception while a consumer maps a producer's word keeps its backtrace,
   also when the producer's slot was freed meanwhile, so that the word goes
   back, and its free faults. *)
let test_map_raises_trace () =
  Printexc.record_backtrace true;
  let consumer, c = P.open_ ~waits_on:[ `Host ] "reuse:trace-consumer" in
  let d, p = P.open_ ~transport:true "reuse:trace-producer" in
  let q = submit (Sub.make d [||]) in
  P.before c "map_peer" (fun () ->
      ignore (P.run p);
      Rig.close d;
      Rig.free_cache d;
      P.fault p "freed after its stop";
      raise Exit);
  (match submit ~waits:[| q |] (Sub.make consumer [||]) with
  | _ -> fail "the submit returned"
  | exception Exit ->
      let trace = Printexc.(raw_backtrace_to_string (get_raw_backtrace ())) in
      let first = List.hd (String.split_on_char '\n' trace) in
      contains ~msg:"where the exception was raised" ~sub:"test_reuse.ml"
        first);
  Rig.close consumer;
  (* The consumer, lost to the exception, stopped and drained before the
     word lost its count: a later call that moves memory gives it back. *)
  Rig.free_cache Rig.host;
  equal ~msg:"the word's frees" int 1 (P.word_frees p);
  no_violations ()

(* A word of the driver's own whose free raises after its device stopped
   was given back: the exception reaches the call whose drain freed it, and
   rig frees the word no more. *)
let test_word_free_raises () =
  let consumer, _ = P.open_ ~waits_on:[ `Host ] "reuse:raising-consumer" in
  let d, p = P.open_ ~transport:true "reuse:raising-producer" in
  let q = submit (Sub.make d [||]) in
  let w = submit ~waits:[| q |] (Sub.make consumer [||]) in
  ignore (P.run p);
  Rig.Point.wait w;
  Rig.close d;
  Rig.free_cache d;
  P.before p "free" (fun () -> raise Exit);
  raises Exit (fun () -> Rig.close consumer);
  equal ~msg:"the word's frees" int 1 (P.word_frees p);
  Rig.free_cache Rig.host;
  equal ~msg:"the word's frees after another drain" int 1 (P.word_frees p);
  no_violations ()

(* An exception while a stopping consumer gives back its mapping of a
   producer's word reaches the consumer's close, and the word still goes
   back. *)
let test_unmap_raises () =
  let consumer, c = P.open_ ~waits_on:[ `Host ] "reuse:unmap-consumer" in
  let d, p = P.open_ ~transport:true "reuse:unmap-producer" in
  let q = submit (Sub.make d [||]) in
  let w = submit ~waits:[| q |] (Sub.make consumer [||]) in
  ignore (P.run p);
  Rig.Point.wait w;
  Rig.close d;
  Rig.free_cache d;
  P.before c "unmap" (fun () -> raise Exit);
  raises Exit (fun () -> Rig.close consumer);
  Rig.free_cache Rig.host;
  equal ~msg:"the word's frees" int 1 (P.word_frees p);
  no_violations ()

(* An exception while a consumer gives back one of its retired mappings
   loses the consumer, and its other due mappings go back at its stop. Each
   producer's word goes back once. *)
let test_retired_unmap_raises () =
  let consumer, c = P.open_ ~waits_on:[ `Host ] "reuse:retired-consumer" in
  let produce name =
    let d, p = P.open_ ~transport:true name in
    (d, p, submit (Sub.make d [||]))
  in
  let wait_on (_, pa, qa) (_, pb, qb) =
    let w = submit ~waits:[| qa; qb |] (Sub.make consumer [||]) in
    List.iter (fun p -> ignore (P.run p)) [ pa; pb; c ];
    Rig.Point.wait w
  in
  let ((a, pa, _) as first_a) = produce "reuse:retired-a" in
  let ((b, pb, _) as first_b) = produce "reuse:retired-b" in
  wait_on first_a first_b;
  List.iter
    (fun d ->
      Rig.close d;
      Rig.free_cache d)
    [ a; b ];
  (* The waits on the slots' later devices replace both pairs, which retires
     both mappings at this submit's value. *)
  let ((a', _, _) as later_a) = produce "reuse:retired-a" in
  let ((b', _, _) as later_b) = produce "reuse:retired-b" in
  wait_on later_a later_b;
  P.before c "unmap" (fun () -> raise Exit);
  raises Exit (fun () -> Rig.free_cache consumer);
  Rig.close a';
  Rig.close b';
  Rig.close consumer;
  Rig.free_cache Rig.host;
  List.iter
    (fun p ->
      equal ~msg:"each first producer's word frees" int 1 (P.word_frees p))
    [ pa; pb ];
  no_violations ()

(* A submit that checked its waits and holds its device's turn takes the
   value after the last one given: a mapping its waits read, replaced
   before it took its value, goes back only once that value is reached. The
   submit is held in its driver's [room], and a drain meanwhile runs the
   releases that are due. *)
let test_replaced_in_turn () =
  let consumer, c =
    P.open_ ~waits_on:[ `Host ] ~may_block:true "reuse:turn-consumer"
  in
  let name = "reuse:turn-producer" in
  let d1, p1 = P.open_ ~transport:true name in
  let q1 = submit (Sub.make d1 [||]) in
  let w1 = submit ~waits:[| q1 |] (Sub.make consumer [||]) in
  ignore (P.run p1);
  ignore (P.run c);
  Rig.Point.wait w1;
  let q1' = submit (Sub.make d1 [||]) in
  P.gate_room c;
  let held =
    spawn (fun () -> submit ~waits:[| q1' |] (Sub.make consumer [||]))
  in
  Support.await "a submit in its room" (fun () -> P.in_room c = 1);
  Rig.close d1;
  Rig.free_cache d1;
  let d2, p2 = P.open_ ~transport:true name in
  let q2 = submit (Sub.make d2 [||]) in
  (* The replacement of [d1]'s pair by [d2]'s comes before the decision for
     a device of another slot, which the waits name after it: its mapping
     tells the replacement is done. *)
  let z, pz = P.open_ ~transport:true "reuse:turn-other" in
  let qz = submit (Sub.make z [||]) in
  let replaced = Atomic.make false in
  P.before c "map_peer" (fun () ->
      P.before c "map_peer" (fun () -> Atomic.set replaced true));
  let replacing =
    spawn (fun () -> submit ~waits:[| q2; qz |] (Sub.make consumer [||]))
  in
  Support.await "the replacement" (fun () -> Atomic.get replaced);
  P.open_room c;
  let w = held () in
  Rig.free_cache consumer;
  ignore (P.run c);
  Rig.Point.wait w;
  ignore (P.run p2);
  ignore (P.run pz);
  Rig.Point.wait (replacing ());
  Rig.close d2;
  Rig.close z;
  Rig.close consumer;
  List.iter
    (fun p -> equal ~msg:"every word's frees" int 1 (P.word_frees p))
    [ p1; p2; pz ];
  no_violations ()

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
        test "a device gives back its word of its own once closed"
          test_own_word_back;
        test "a queue waiting on a closed device's word completes"
          test_wait_on_closed;
        test "a device given a word outside rig's array gives it back once"
          test_given_outside;
        test "a consumer of a reopening slot gives back earlier mappings"
          test_mappings_released;
        test "a submit gives back mappings of words and frees no memory"
          test_submit_frees;
        test "a decision for a slot's earlier device comes too late"
          test_stale_decision;
        test "a lost consumer keeps a word it waits on until it stopped"
          test_lost_consumer;
        test "an exception mapping a word leaves the word to go back"
          test_map_raises;
        test "an exception mapping a word keeps its backtrace"
          test_map_raises_trace;
        test "an exception freeing a stopped device's word reaches the caller"
          test_word_free_raises;
        test "an exception unmapping a word leaves the word to go back"
          test_unmap_raises;
        test "an exception unmapping a retired mapping keeps the others"
          test_retired_unmap_raises;
        test "a replaced mapping outlives the submit holding the turn"
          test_replaced_in_turn;
        stateful ~count:40 ~domains:2
          "a slot taken again answers an earlier device's points and stamps"
          cycle_commands;
      ];
  ]

let () = exit (run "rig.reuse" tests)
