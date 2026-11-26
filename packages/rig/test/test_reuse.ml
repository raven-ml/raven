(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Device slots (rig.mli, "Opening and closing"): what holds a slot, how a later
   device continues a slot's timeline, what an open costs, and when a device
   gives back its mappings of other devices' words and its own word. Polled
   devices run their queues only when a wait sleeps or the test runs them, so a
   device lost before any wait loses its queued work.

   No call names a slot. A point names its device until a later device takes the
   device's slot (Point.device), which is how these tests see a slot pass on. A
   later open takes the slot freed last, which rig.mli does not promise: it
   promises only a free slot. *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support

let submit ?(buffers = [||]) ?(args = [||]) ?waits s =
  Rig.submit ?waits s ~buffers ~args

let empty d = Sub.make d [||]
let device = Testable.make ~pp:Rig.pp ~equal:Rig.equal
let lost d = function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false
let count call p = List.length (List.filter (String.equal call) (P.log p))

(* The values of a submission's one fill, which reads none. *)
let one = Support.zeros 1

let no_violations () =
  equal ~msg:"Polled's violations" (list string) [] (P.violations ())

(* The process's first failure holds its slot for good: tests that watch a lost
   device's slot pass on make sure one came first. *)
let first_failure () =
  if Rig.failure () = None then begin
    let first, _ = P.open_ "reuse:first-failure" in
    Rig.lose first "the process's first failure"
  end

(* A collection, a drain on [d], which {!Buffer.create} runs first, also on a
   lost [d], then a collection: memory and values collected return, and values
   the drain lets go of are collected. *)
let released d =
  Gc.full_major ();
  (match B.create d 8 with
  | b -> ignore (Sys.opaque_identity b)
  | exception Rig.Lost (d', _) when Rig.equal d d' -> ());
  Gc.full_major ()

(* A value whose collection counts in [runs]. *)
let counted runs =
  let v = ref () in
  Gc.finalise (fun _ -> Atomic.incr runs) v;
  v

(* A thread running [f], whose outcome [join] gives. *)
let spawn f =
  let r = ref (Error Exit) in
  let t = Thread.create (fun () -> r := try Ok (f ()) with e -> Error e) () in
  fun () ->
    Thread.join t;
    match !r with Ok x -> x | Error e -> raise e

(* What holds a slot *)

(* The opens [taker] makes before it closes them and gives up, so that a rig
   that never passes a slot on fails here and leaves later tests every slot but
   these. *)
let tries = 1024

(* What a point names, for a failure. *)
let naming p =
  Printf.sprintf "the point %s, which names %s"
    (Format.asprintf "%a" Rig.Point.pp p)
    (match Rig.Point.device p with Some d -> Rig.name d | None -> "none")

(* [taker p name] opens Polled devices named [name]-[i] until one takes the slot
   of [p]'s device, which [p] then shows by naming no device, and is that
   device: no call names a slot, and an open takes any free one. The others are
   closed. After [tries] opens it closes them all and fails. *)
let taker p name =
  let rec go i opened =
    if i > tries then begin
      List.iter Rig.close opened;
      failf "none of %d opens took the slot of %s" tries (naming p)
    end;
    let d, _ = P.open_ (Printf.sprintf "%s-%d" name i) in
    if Rig.Point.device p = None then begin
      List.iter Rig.close opened;
      d
    end
    else go (i + 1) (d :: opened)
  in
  go 1 []

(* The point of a Polled device named [name], closed and drained: its slot is
   free, freed before the slot a test then watches. *)
let freed name =
  let d, _ = P.open_ name in
  let p = submit (empty d) in
  Rig.Point.wait p;
  Rig.close d;
  Rig.free_cache d;
  p

(* [holds ~free d p name] opens later devices while [d]'s slot is held until one
   takes the slot of [free], a point {!freed} gave: an open took a free slot,
   and [p], a point of [d], still names [d]. The later device is closed. *)
let holds ~msg ~free d p name =
  let later = taker free name in
  equal ~msg (option device) (Some d) (Rig.Point.device p);
  Rig.close later

let[@inline never] [@local never] kept_submission d runs held =
  let s = Sub.make ~keep:(counted runs) d [||] in
  held := Some s;
  submit s

(* [slot_passes name ~end_] ends a device whose submission keeps a value: the
   device's points name it while the submission is reachable, and once it is not
   and a drain let the value go, a later device takes the slot. *)
let slot_passes name ~end_ =
  let d, p = P.open_ name in
  let runs = Atomic.make 0 and held = ref None in
  let pt = kept_submission d runs held in
  let free = freed (name ^ "-free") in
  end_ d p;
  released d;
  holds ~msg:"while the submission is reachable" ~free d pt (name ^ "-later");
  equal ~msg:"its value, while the submission is reachable" int 0
    (Atomic.get runs);
  held := None;
  released d;
  equal ~msg:"its value, once let go" int 1 (Atomic.get runs);
  Rig.close (taker pt (name ^ "-last"))

let test_slot_closed () =
  slot_passes "reuse:slot-closed" ~end_:(fun d _ -> Rig.close d)

let test_slot_lost () =
  first_failure ();
  slot_passes "reuse:slot-lost" ~end_:(fun d p ->
      P.fail p;
      raises_match (lost d) (fun () -> submit (empty d));
      Rig.close d)

let[@inline never] [@local never] hold_memory d held =
  held := Some (B.create d 64)

let test_slot_memory () =
  let d, _ = P.open_ "reuse:slot-memory" in
  let pt = submit (empty d) in
  Rig.Point.wait pt;
  let held = ref None in
  hold_memory d held;
  let free = freed "reuse:slot-memory-free" in
  Rig.close d;
  released d;
  holds ~msg:"while its memory is reachable" ~free d pt
    "reuse:slot-memory-later";
  held := None;
  released d;
  Rig.close (taker pt "reuse:slot-memory-last")

(* The failed hand-over loses the device, whose stop its close runs. *)
let test_slot_unstopped () =
  first_failure ();
  let d, p = P.open_ "reuse:slot-unstopped" in
  let pt = submit (empty d) in
  Rig.Point.wait pt;
  let free = freed "reuse:slot-unstopped-free" in
  P.fail p;
  raises_match (lost d) (fun () -> submit (empty d));
  released d;
  holds ~msg:"lost, before its stop" ~free d pt "reuse:slot-unstopped-later";
  equal ~msg:"its stops before its close" int 0 (P.stops p);
  Rig.close d;
  released d;
  Rig.close (taker pt "reuse:slot-unstopped-last")

(* The last value a point holds, 2^47 - 1, ends a device's timeline: a submit
   after it loses the device with the reason and assigns nothing. The device's
   timeline passed 2^46, so its slot goes to no later device: its points name
   it. *)
let test_last_value () =
  first_failure ();
  let last = (1 lsl 47) - 1 in
  let p = P.make () in
  let d =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_ (P.opener ~start:(last - 1) "reuse:last-value" p))
  in
  let q = submit (empty d) in
  equal ~msg:"the last value" int last (Rig.Point.value q);
  Rig.Point.wait q;
  raises_match
    (function
      | Rig.Lost (d', why) ->
          Rig.equal d d'
          && why = "its timeline reached the last value a point holds, 2^47 - 1"
      | _ -> false)
    (fun () -> submit (empty d));
  equal ~msg:"no value assigned past it" int last (Rig.submitted d);
  let free = freed "reuse:last-value-free" in
  Rig.close d;
  released d;
  let later = taker free "reuse:after-last-value" in
  equal ~msg:"its point once a later device took a free slot" (option device)
    (Some d) (Rig.Point.device q);
  less ~msg:"the later device's base" int ~than:(1 lsl 46) (Rig.submitted later);
  Rig.close later

(* A later device's [make] raises the slot's word to its own start, as a
   driver's word that starts elsewhere does. *)
let test_raised_by_make () =
  let d, _ = P.open_ "reuse:raised-first" in
  let q = submit (empty d) in
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
  equal ~msg:"the closed device's word while the later one opened" (option int)
    (Some last) !during;
  Rig.close later

(* A device lost, stopped and opened again under its name, beside another
   domain's uses *)

(* A world: a device under [name], lost and opened again by the cycles, every
   other one with a word of its own, which the consumer's queue waits on through
   a mapping; a consumer that never closes, whose memory [b] the cycles' devices
   write through a borrow, and whose queue waits for their points; a waiter,
   whose queue waits for no device, so that the host waits for its submissions'
   points; and the points of every cycle, with whether a wait for each
   returned. *)
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
  no_violations ()

let rec record w p =
  let l = Atomic.get w.points in
  if not (Atomic.compare_and_set w.points l (p :: l)) then record w p

let waited q =
  match Rig.Point.wait q with () -> true | exception Rig.Lost _ -> false

let polled q =
  match Rig.Point.reached q with r -> r | exception Rig.Lost _ -> false

(* The model: whether [b]'s last write was lost, after which every ordered use
   of [b] raises [Lost] for good; whether a lost use or write of [b] makes every
   write of it raise for good, or a use whose device was lost during it may;
   whether a cycle lost a point, after which a submit that waits for every point
   raises it; and whether the consumer was lost, as its queue waited for a point
   a cycle lost, after which every use of [b], its memory, raises. *)
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

let model_t = Testable.make ~pp:pp_model ~equal:( == )

(* [admits ~msg ~claim holds r] fails, printing the model, unless the outcome
   named [msg] is one [r] admits. *)
let admits ~msg ~claim holds r = satisfies ~msg ~claim model_t holds r
let world = abstract ~pp:pp_model ~release:close_world "w"

(* What a cycle did: its device wrote [b] at the point it then waited for, which
   returned or raised [Lost]; or its write raised [Lost], [b]'s last write being
   lost, and the point it waited for is its empty work's. *)
type cycle = Wrote of bool | Poisoned of bool

let pp_cycle ppf = function
  | Wrote ok -> Format.fprintf ppf "wrote, wait returned %b" ok
  | Poisoned ok -> Format.fprintf ppf "poisoned, wait returned %b" ok

let cycle = Testable.make ~pp:pp_cycle ~equal:( = )
let cycle_t = pair cycle bool

(* Loses the current device after it wrote [b], or tried to, with its work
   queued unless [reached]: it faults, so its next counted call, or a sleep of
   another domain's wait for it, loses it, and its close waits for its stop.
   Then opens its name again, whose device takes the slot the lost one left, and
   waits for the point. Cycles take turns. *)
let cycle_locked ~reached w =
  let d, p = Atomic.get w.current in
  let write () =
    let bd = require_some (B.borrow d w.b) in
    let s = Sub.make d [| Support.touching [| Sub.write |] |] in
    submit ~args:one s ~buffers:[| bd |]
  in
  let q, poisoned =
    match write () with
    | q -> (q, false)
    | exception Rig.Lost _ -> (submit (empty d), true)
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
      if reached then
        satisfies ~msg:"the reached point's wait" ~claim:"its wait returned"
          cycle
          (function Wrote ok | Poisoned ok -> ok)
          c;
      match c with
      | Wrote ok ->
          admits ~msg:"a write" ~claim:"no write raises, the consumer not lost"
            (fun r -> not (r.writes_raise || r.consumer_lost))
            r;
          r.maybe_lost_use <- false;
          if not ok then begin
            r.poisoned <- true;
            r.writes_raise <- true;
            r.any_lost <- true
          end
      | Poisoned ok ->
          admits ~msg:"a write that raised Lost"
            ~claim:"writes raise, a use maybe lost, or the consumer lost"
            (fun r -> r.writes_raise || r.maybe_lost_use || r.consumer_lost)
            r;
          r.writes_raise <- true;
          r.maybe_lost_use <- false;
          if not ok then r.any_lost <- true)

(* The waiter's submit after every point a cycle recorded, which the host waits
   for: whether it raised [Lost] for another device, and whether one of those
   points was lost, by a device whose slot a later device took or by any. *)
let wait_all_system w =
  let points = Atomic.get w.points in
  let lost = List.exists (fun (_, ok) -> not ok) points in
  let reused =
    List.exists (fun (q, ok) -> (not ok) && Rig.Point.device q = None) points
  in
  let waits = Array.of_list (List.map fst points) in
  let raised =
    match Rig.Point.wait (submit ~waits (empty w.waiter)) with
    | () -> false
    | exception Rig.Lost (d, _) when not (Rig.equal d w.waiter) -> true
  in
  (raised, lost, reused)

let outcome raised = if raised then "raised Lost" else "returned"

(* The submit raises iff a point it waited for was lost. *)
let wait_all_model _ = function
  | Error e -> raise e
  | Ok (raised, lost, reused) ->
      cover "a host wait for a point lost on a slot taken again" reused;
      equal ~msg:"the submit, as its points were lost or not" string
        (outcome lost) (outcome raised)

(* A host read of [b]: [true] if it returned, [false] if it raised [Lost]: for a
   lost write of it, or once the consumer, its memory's device, is lost. *)
let read_system w =
  match B.wait w.b Rig.Read with () -> true | exception Rig.Lost _ -> false

(* The consumer's read of [b] after every point a cycle recorded: [`Ok],
   [`Lost_self] if the consumer is lost, or [`Lost_other]. *)
let consume_system w =
  let waits = Array.of_list (List.map fst (Atomic.get w.points)) in
  let read () =
    let s = Sub.make w.consumer [| Support.touching [| Sub.read |] |] in
    Rig.Point.wait (submit ~args:one s ~waits ~buffers:[| w.b |])
  in
  match read () with
  | () -> `Ok
  | exception Rig.Lost (d, _) when Rig.equal d w.consumer -> `Lost_self
  | exception Rig.Lost _ -> `Lost_other

(* A consumer whose queue waits for a cycle's point that the cycle's device
   loses is lost with it, which may come before the cycle records the point. *)
let consume_model r = function
  | Error e -> raise e
  | Ok `Ok ->
      admits ~msg:"a read that returned" ~claim:"no point or write lost"
        (fun r -> not (r.poisoned || r.any_lost || r.consumer_lost))
        r
  | Ok `Lost_other ->
      admits ~msg:"a read that raised another device's loss"
        ~claim:"a point or write lost"
        (fun r -> r.poisoned || r.any_lost)
        r
  | Ok `Lost_self -> r.consumer_lost <- true

(* The current device's read of [b]: [`Ok], [`Lost_self] if that device was lost
   meanwhile, or [`Lost_other] for a loss of another device. *)
let use_system w =
  let d, _ = Atomic.get w.current in
  match
    match B.borrow d w.b with
    | None -> false
    | Some bd ->
        let s = Sub.make d [| Support.touching [| Sub.read |] |] in
        Rig.Point.wait (submit ~args:one s ~buffers:[| bd |]);
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
      admits ~msg:"a borrow that returned"
        ~claim:"no write raises, the consumer not lost"
        (fun r -> not (r.writes_raise || r.consumer_lost))
        r;
      r.maybe_lost_use <- false
  | Ok `Lost_other ->
      admits ~msg:"a borrow that raised another device's loss"
        ~claim:"writes raise, a use maybe lost, or the consumer lost"
        (fun r -> r.writes_raise || r.maybe_lost_use || r.consumer_lost)
        r;
      r.writes_raise <- true;
      r.maybe_lost_use <- false
  | Ok `Lost_self -> r.maybe_lost_use <- true

(* Each recorded point still answers its first wait, and [reached] answers as
   the wait does: a later device on its slot changes no answer. The points that
   answer otherwise, as their value, their first wait's answer, and the wait's
   and [reached]'s now; with whether a point a loss dropped names no device any
   more: its slot passed on. *)
let points_system w =
  let points = Atomic.get w.points in
  ( List.filter_map
      (fun (q, ok) ->
        let w = waited q and r = polled q in
        if w = ok && r = ok then None else Some (Rig.Point.value q, ok, w, r))
      points,
    List.exists (fun (q, ok) -> (not ok) && Rig.Point.device q = None) points )

let points_model _ = function
  | Error e -> raise e
  | Ok (wrong, reused) ->
      cover "a lost device's slot taken again" reused;
      equal ~msg:"points that answer otherwise than before"
        (list (quad int bool bool bool))
        [] wrong

let cycle_commands =
  [
    command "world"
      (Gen.unit @-> makes world)
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
    command "host read"
      (world ^-> returns bool)
      (fun r -> not (r.poisoned || r.consumer_lost))
      read_system;
    command "consume" (world ^-> judges use_t) consume_model consume_system;
    command "use" (world ^-> judges use_t) use_model use_system;
    command "wait all"
      (world ^-> judges (triple bool bool bool))
      wait_all_model wait_all_system;
    command "points"
      (world ^-> judges (pair (list (quad int bool bool bool)) bool))
      points_model points_system;
  ]

(* What an open costs *)

(* Opens and closes in a loop keep a fixed record per device, whatever work the
   device did: rig's records, name and reason, and the driver's state with its
   word ([kept_c], [kept_ml]). The driver's stop gives back the rest of its
   state, and rig its mappings of other devices' words and memory once the
   device is Stopped. Polled's handle also keeps a log of the driver's calls,
   which grows with the work and which the OCaml laws leave out. Each cycle
   opens a producer and a consumer whose queue waits for the producer's value;
   the consumer allocates, then both close. After a warm window of each kind,
   each window of [opens] cycles grows each heap by at most that a device.

   Windows differ by what both devices do with [kept] host buffers that the
   program keeps: nothing in a [Bare] window; in a [Mapping] one each borrows
   them, which maps them; in a [Reading] one each also reads them. A mapping
   kept per device and buffer would add at least a mapping record, its cell in
   the buffer's list and its region to the OCaml heap, 13 words, 416 KiB a
   window, and slots of stamps kept per device would add a 112-byte chunk to the
   C heap every four devices and buffer, 112 KiB a window. So a window of each
   kind grows each heap no more than a bare one, up to [step]: macOS's count of
   the C heap can step by 16 KiB or more over one window, and the laws check the
   least growth of [windows] of each kind. *)
let opens = 1024
let windows = 3
let kept = 2
let step = 16 * 1024

(* What a closed Polled device keeps, per device, baselines measured on the M1
   Max: about 1.1 KiB of C heap and 1.5 KiB of OCaml heap, Polled's log left
   out. *)
let kept_c = 1536
let kept_ml = 2048

type cycles = Bare | Mapping | Reading

let words v = Obj.reachable_words (Obj.repr v) * (Sys.word_size / 8)

(* The bytes of Polled's log of [p]'s calls. *)
let logged p = words (P.log p, P.frees p, P.allocs p, P.host_maps p)
let live () = (Gc.stat ()).live_words * (Sys.word_size / 8)

let test_open_close () =
  let hs = List.init kept (fun _ -> B.create Rig.host B.paged_from) in
  let reads = Support.touching (Array.make kept Sub.read) in
  let cycle kind i =
    let p, pp = P.open_ (Printf.sprintf "reuse:producer%d" (i mod 2)) in
    let d, dp =
      P.open_ ~waits_on:[ `Host ] (Printf.sprintf "reuse:cycle%d" (i mod 2))
    in
    ignore (B.create d 64 : B.t);
    let borrows d =
      Array.of_list (List.map (fun h -> require_some (B.borrow d h)) hs)
    in
    let work d =
      match kind with
      | Bare -> (empty d, [||], [||])
      | Mapping ->
          ignore (borrows d : B.t array);
          (empty d, [||], [||])
      | Reading -> (Sub.make d [| reads |], borrows d, Support.zeros 1)
    in
    let s, buffers, args = work p in
    let a = submit s ~buffers ~args in
    let s, buffers, args = work d in
    Rig.Point.wait (submit s ~buffers ~args ~waits:[| a |]);
    Rig.close d;
    Rig.close p;
    [ (p, pp); (d, dp) ]
  in
  let heap () =
    match Support.heap_bytes () with
    | Some n -> n
    | None -> skip ~reason:"the system counts no C heap" ()
  in
  let settle closed =
    Gc.full_major ();
    List.iter (fun (d, _) -> Rig.free_cache d) closed;
    List.iter (fun h -> B.wait h Rig.Read_write) hs
  in
  let window kind =
    let c = heap () and ml = live () in
    let closed = List.concat (List.init opens (fun i -> cycle kind (i + 1))) in
    settle closed;
    let logs = List.fold_left (fun n (_, p) -> n + logged p) 0 closed in
    (heap () - c, live () - ml - logs)
  in
  let kinds = [ ("bare", Bare); ("mappings", Mapping); ("reads", Reading) ] in
  List.iter (fun (_, kind) -> ignore (window kind)) kinds;
  let grown =
    List.init windows (fun _ -> List.map (fun (_, kind) -> window kind) kinds)
  in
  let least heap i =
    List.fold_left (fun n g -> Int.min n (heap (List.nth g i))) max_int grown
  in
  let shown heap i =
    String.concat ", "
      (List.map (fun g -> string_of_int (heap (List.nth g i))) grown)
  in
  let law i name (what, heap, record) =
    at_most int
      ~msg:
        (Printf.sprintf "the %s heap's least growth over a window of %s (%s)"
           what name (shown heap i))
      ~than:(2 * opens * record)
      (least heap i);
    at_most int
      ~msg:
        (Printf.sprintf
           "the %s heap's least growth over a window of %s (%s), past the \
            least of bare ones (%s)"
           what name (shown heap i) (shown heap 0))
      ~than:step
      (least heap i - least heap 0)
  in
  List.iteri
    (fun i (name, _) ->
      List.iter (law i name) [ ("C", fst, kept_c); ("OCaml", snd, kept_ml) ])
    kinds

(* A borrower that maps [peak] allocations at once keeps, once they are freed
   and it closed, no more than one that mapped a single allocation: no bucket of
   its record of mappings outlives the mappings. Each allocation is 64 bytes of
   another Polled device, which the borrower maps as a peer's. [order] frees the
   allocations before the close or after it. The law takes the least growth of
   [windows] runs, as the law above does. *)
let peak = 8192

type order = Freed_first | Closed_first

let test_mapped_peak order () =
  let run n =
    let ml = live () in
    let o, po = P.open_ "reuse:peak-owner" and d, pd = P.open_ "reuse:peak" in
    let[@inline never] [@local never] borrowed () =
      let bs = List.init n (fun _ -> B.create o 64) in
      List.iter (fun b -> ignore (require_some (B.borrow d b))) bs;
      ref bs
    in
    let bs = borrowed () in
    let free () =
      bs := [];
      Gc.full_major ();
      Rig.free_cache Rig.host;
      Rig.free_cache o
    in
    (match order with
    | Freed_first ->
        free ();
        Rig.close d
    | Closed_first ->
        Rig.close d;
        Rig.free_cache d;
        free ());
    Rig.close o;
    Gc.full_major ();
    List.iter Rig.free_cache [ o; d ];
    live () - ml - logged po - logged pd
  in
  let least n =
    List.fold_left Int.min max_int (List.init windows (fun _ -> run n))
  in
  ignore (run peak);
  let one = least 1 and many = least peak in
  at_most int
    ~msg:
      (Printf.sprintf "a borrower of %d's growth past one of 1 (%d)" peak one)
    ~than:step (many - one)

(* Mappings of other devices' words *)

(* A consumer whose queue waits for a producer behind a transport reads that
   producer's own word. A later device on the producer's slot has a word of its
   own, which the consumer's waits for its points read. *)
let test_pair_per_occupant () =
  let consumer, _ = P.open_ ~waits_on:[ `Host ] "reuse:pair-consumer" in
  let wait_on name =
    let d, p = P.open_ ~transport:true name in
    let q = submit (empty d) in
    let c = submit ~waits:[| q |] (empty consumer) in
    ignore (P.run p);
    Rig.Point.wait c;
    (d, q)
  in
  let d, q = wait_on "reuse:pair-producer" in
  Rig.close d;
  Rig.free_cache d;
  (* Later producers, each waited on, until one took [d]'s slot, which [q] then
     shows by naming no device. *)
  let rec later i opened =
    if i > tries then begin
      List.iter Rig.close opened;
      failf "none of %d producers took the slot of %s" tries (naming q)
    end;
    let d', _ = wait_on (Printf.sprintf "reuse:pair-producer-%d" i) in
    if Rig.Point.device q = None then d' :: opened
    else later (i + 1) (d' :: opened)
  in
  List.iter Rig.close (later 1 []);
  Rig.close consumer

(* A device whose driver declares [maps_host] reads the words of every slot in
   rig's array through one mapping of that array, closed producers' too, and
   gives it back once it is closed. *)
let test_close_unmaps_word () =
  let consumer, cp = P.open_ ~waits_on:[ `Host ] "reuse:array-consumer" in
  for i = 1 to 3 do
    let producer, _ = P.open_ (Printf.sprintf "reuse:array-producer-%d" i) in
    let a = submit (empty producer) in
    Rig.Point.wait (submit (empty consumer) ~waits:[| a |]);
    equal ~msg:"the consumer's wait in its queue" int 1
      (List.length (P.last_waits cp));
    Rig.close producer
  done;
  Rig.close consumer;
  Gc.full_major ();
  Rig.free_cache consumer;
  equal ~msg:"its mappings, and those given back" (pair int int) (1, 1)
    (count "map_host" cp, count "unmap" cp)

(* The first decision waits in its map, at Polled's [before] seam, until the
   second has decided and its submit returned; it then takes the second's way
   and gives its own mapping back. Both waits run. *)
let test_pair_race () =
  let producer, _ = P.open_ "reuse:race-producer" in
  let consumer, cp = P.open_ ~waits_on:[ `Host ] "reuse:race-consumer" in
  let a = submit (empty producer) in
  let mapping = Atomic.make false and second = Atomic.make false in
  P.before cp "map_host" (fun () ->
      Atomic.set mapping true;
      Support.await "the second submit" (fun () -> Atomic.get second));
  let wait () = submit (empty consumer) ~waits:[| a |] in
  let first = Domain.spawn wait in
  Support.await "the first decision's map" (fun () -> Atomic.get mapping);
  let q = wait () in
  Atomic.set second true;
  let p = Domain.join first in
  Rig.Point.wait p;
  Rig.Point.wait q;
  equal ~msg:"the maps of the words, and those given back" (pair int int) (2, 1)
    (count "map_host" cp, count "unmap" cp)

(* Whether [c] gave back a mapping of [p]'s word, which lies at the word's
   address. *)
let unmapped c p = List.exists (fun (a, _) -> a = P.word_at p) (P.frees c)

(* The names of the devices of [ps] whose words' mappings [c] gave back. *)
let gave_back c ps =
  List.filter_map (fun (name, p) -> if unmapped c p then Some name else None) ps

(* A consumer that waits in its queue on a slot whose devices keep reopening
   keeps a mapping of the word of the device it waits on now. One of an earlier
   device goes back once its queue passed the waits that read it, at its next
   wait on a later device of any slot or at a call that moves its collected
   memory, also while its queue lags. *)
let test_mappings_released () =
  let consumer, c = P.open_ ~waits_on:[ `Host ] "reuse:released-consumer" in
  let name = "reuse:released-producer" in
  let last = ref None in
  (* Opens the slot's next device and submits the consumer's wait on it, after
     closing the one before. *)
  let wait_on () =
    Option.iter
      (fun d ->
        Rig.close d;
        Rig.free_cache d)
      !last;
    let d, p = P.open_ ~transport:true name in
    last := Some d;
    let q = submit (empty d) in
    let w = submit ~waits:[| q |] (empty consumer) in
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
  equal ~msg:"at a call that moves memory" (list string) [ "first" ]
    (gave_back c [ ("first", p1); ("second", p2) ]);
  let p3, w3 = wait_on () in
  Rig.Point.wait w3;
  let given () = gave_back c [ ("first", p1); ("second", p2); ("third", p3) ] in
  (* The queue lags from here: it has not passed the wait on the fourth
     device. *)
  ignore (wait_on () : P.t * Rig.Point.t);
  equal ~msg:"at a wait on the fourth" (list string) [ "first"; "second" ]
    (given ());
  let _, w5 = wait_on () in
  equal ~msg:"at a wait on the fifth, while the queue lags" (list string)
    [ "first"; "second" ] (given ());
  ignore (P.run c);
  Rig.Point.wait w5;
  Rig.free_cache consumer;
  equal ~msg:"once the queue passed the third's waits" (list string)
    [ "first"; "second"; "third" ]
    (given ());
  Option.iter Rig.close !last;
  Rig.close consumer;
  no_violations ()

(* Collects a buffer of [d] that [d]'s work used, out of the caller's frame. *)
let[@inline never] [@local never] cache_used d =
  let b = B.create d 64 in
  let s = Sub.make d [| Support.touching [| Sub.write |] |] in
  Rig.Point.wait (submit ~args:one s ~buffers:[| b |])

(* A submit frees no memory, since a free may wait for the GPU's work: memory
   its device released, and a word whose last mapping went back, go back at the
   next call that moves memory. *)
let test_submit_frees () =
  let consumer, c = P.open_ ~waits_on:[ `Host ] "reuse:frees-consumer" in
  let name = "reuse:frees-producer" in
  let wait_on () =
    let d, p = P.open_ ~transport:true name in
    let q = submit (empty d) in
    let w = submit ~waits:[| q |] (empty consumer) in
    ignore (P.run p);
    Rig.Point.wait w;
    (d, p)
  in
  let d1, p1 = wait_on () in
  cache_used consumer;
  Gc.full_major ();
  ignore (B.create consumer 128 : B.t);
  (* The cached memory's free waits for a value the queue reaches after the free
     was asked. *)
  ignore (submit (empty consumer) : Rig.Point.t);
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
  equal ~msg:"mappings given back at the wait on the third" (list string)
    [ "first" ]
    (gave_back c [ ("first", p1) ]);
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

(* The consumer keeps its pair of the later device, and the first producer's
   word goes back. The decision is held at the locate of its mapping while the
   slot passes on. *)
let test_stale_decision () =
  let consumer, c = P.open_ ~waits_on:[ `Host ] "reuse:stale-consumer" in
  (* The consumer's mapping of rig's array, made before the gate. *)
  let a, _ = P.open_ "reuse:stale-array" in
  let qa = submit (empty a) in
  Rig.Point.wait (submit ~waits:[| qa |] (empty consumer));
  let d, p = P.open_ ~transport:true "reuse:stale-producer" in
  let q = submit (empty d) in
  P.gate_locates c;
  let stale = spawn (fun () -> submit ~waits:[| q |] (empty consumer)) in
  Support.await "the decision's locate at the gate" (fun () -> P.locaters c = 1);
  Rig.close d;
  Rig.free_cache d;
  let d', p' = P.open_ "reuse:stale-later" in
  let q' = submit (empty d') in
  let w' = submit ~waits:[| q' |] (empty consumer) in
  P.open_locates c;
  let w = stale () in
  ignore (P.run p');
  Rig.Point.wait w';
  Rig.Point.wait w;
  let q'' = submit (empty d') in
  let w'' = submit ~waits:[| q'' |] (empty consumer) in
  equal ~msg:"the consumer's waits on the later device, in its queue" int 1
    (List.length (P.last_waits c));
  ignore (P.run p');
  Rig.Point.wait w'';
  Rig.close d';
  Rig.close a;
  Rig.close consumer;
  equal ~msg:"the first producer's word frees" int 1 (P.word_frees p);
  no_violations ()

(* The submit is held in its driver's [room], and a drain meanwhile runs the
   releases that are due. *)
let test_replaced_in_turn () =
  let consumer, c =
    P.open_ ~waits_on:[ `Host ] ~may_block:true "reuse:turn-consumer"
  in
  let name = "reuse:turn-producer" in
  let d1, p1 = P.open_ ~transport:true name in
  let q1 = submit (empty d1) in
  let w1 = submit ~waits:[| q1 |] (empty consumer) in
  ignore (P.run p1);
  ignore (P.run c);
  Rig.Point.wait w1;
  let q1' = submit (empty d1) in
  P.gate_room c;
  let held = spawn (fun () -> submit ~waits:[| q1' |] (empty consumer)) in
  Support.await "a submit in its room" (fun () -> P.in_room c = 1);
  Rig.close d1;
  Rig.free_cache d1;
  let d2, p2 = P.open_ ~transport:true name in
  let q2 = submit (empty d2) in
  (* The replacement of [d1]'s pair by [d2]'s comes before the decision for a
     device of another slot, which the waits name after it: its mapping tells
     the replacement is done. *)
  let z, pz = P.open_ ~transport:true "reuse:turn-other" in
  let qz = submit (empty z) in
  let replaced = Atomic.make false in
  P.before c "map_peer" (fun () ->
      P.before c "map_peer" (fun () -> Atomic.set replaced true));
  let replacing =
    spawn (fun () -> submit ~waits:[| q2; qz |] (empty consumer))
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
  equal ~msg:"every word's frees" (list int) [ 1; 1; 1 ]
    (List.map P.word_frees [ p1; p2; pz ]);
  no_violations ()

(* The waits on the slots' later devices replace both pairs, which retires both
   mappings at that submit's value. Each producer's word goes back once. *)
let test_retired_unmap_raises () =
  let consumer, c = P.open_ ~waits_on:[ `Host ] "reuse:retired-consumer" in
  let produce name =
    let d, p = P.open_ ~transport:true name in
    (d, p, submit (empty d))
  in
  let wait_on (_, pa, qa) (_, pb, qb) =
    let w = submit ~waits:[| qa; qb |] (empty consumer) in
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
  let ((a', _, _) as later_a) = produce "reuse:retired-a" in
  let ((b', _, _) as later_b) = produce "reuse:retired-b" in
  wait_on later_a later_b;
  P.before c "unmap" (fun () -> raise Exit);
  raises Exit (fun () -> Rig.free_cache consumer);
  Rig.close a';
  Rig.close b';
  Rig.close consumer;
  Rig.free_cache Rig.host;
  equal ~msg:"each first producer's word frees" (list int) [ 1; 1 ]
    (List.map P.word_frees [ pa; pb ]);
  no_violations ()

(* A device's own word *)

let test_own_word_back () =
  let frees =
    List.init 64 (fun _ ->
        let d, p = P.open_ ~transport:true "reuse:own-word" in
        Rig.Point.wait (submit (empty d));
        Rig.close d;
        Rig.free_cache d;
        P.word_frees p)
  in
  equal ~msg:"each device's word frees after its close" (list int)
    (List.init 64 (fun _ -> 1))
    frees;
  no_violations ()

(* The word stays until the consumer let go of it. *)
let test_wait_on_closed () =
  let consumer, c = P.open_ ~waits_on:[ `Host ] "reuse:closed-consumer" in
  let d, p = P.open_ ~transport:true "reuse:closed-producer" in
  let q = submit (empty d) in
  let w = submit ~waits:[| q |] (empty consumer) in
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

(* A device given a word outside rig's array keeps it as its driver's own: rig
   frees it once, after the device stopped. *)
let test_given_outside () =
  let p = P.make () in
  let outside = Rig_gpu_support.Host.word () in
  let make ~word:_ = Ok (P.bind p ~word:outside) in
  let d =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_ { (P.opener "reuse:given-outside" p) with make })
  in
  Rig.Point.wait (submit (empty d));
  Rig.close d;
  Rig.free_cache d;
  equal ~msg:"the word's frees" int 1 (P.word_frees p);
  no_violations ()

(* Its stop, at its close, leaves its work, which may still read the word. *)
let test_lost_consumer () =
  let consumer, c =
    P.open_ ~waits_on:[ `Host ] ~answer:`Unknown "reuse:lost-consumer"
  in
  let d, p = P.open_ ~transport:true "reuse:lost-producer" in
  let q = submit (empty d) in
  let w = submit ~waits:[| q |] (empty consumer) in
  P.fault c "lost";
  raises_match (lost consumer) (fun () -> B.create consumer 64);
  Rig.close consumer;
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
  no_violations ()

let test_map_raises () =
  let consumer, c = P.open_ ~waits_on:[ `Host ] "reuse:raise-consumer" in
  let d, p = P.open_ ~transport:true "reuse:raise-producer" in
  let q = submit (empty d) in
  P.before c "map_peer" (fun () -> raise Exit);
  raises Exit (fun () -> ignore (submit ~waits:[| q |] (empty consumer)));
  Rig.close d;
  Rig.close consumer;
  equal ~msg:"the word's frees" int 1 (P.word_frees p);
  no_violations ()

(* Also when another thread closed the producer meanwhile, freeing its slot, so
   that the word goes back, and its free faults. The close runs on another
   thread, since one inside the consumer's driver call would raise. *)
let test_map_raises_trace () =
  Printexc.record_backtrace true;
  let consumer, c = P.open_ ~waits_on:[ `Host ] "reuse:trace-consumer" in
  let d, p = P.open_ ~transport:true "reuse:trace-producer" in
  let q = submit (empty d) in
  P.before c "map_peer" (fun () ->
      ignore (P.run p);
      spawn (fun () -> Rig.close d) ();
      Rig.free_cache d;
      P.fault p "freed after its stop";
      raise Exit);
  (match submit ~waits:[| q |] (empty consumer) with
  | _ -> fail "the submit returned"
  | exception Exit ->
      let trace = Printexc.(raw_backtrace_to_string (get_raw_backtrace ())) in
      let first = List.hd (String.split_on_char '\n' trace) in
      contains ~msg:"where the exception was raised" ~sub:"test_reuse.ml" first);
  Rig.close consumer;
  (* The consumer, lost to the exception, stopped and drained before the word
     lost its count: a later call that moves memory gives it back. *)
  Rig.free_cache Rig.host;
  equal ~msg:"the word's frees" int 1 (P.word_frees p);
  no_violations ()

(* The exception reaches the call whose drain freed the word, and rig frees it
   no more. *)
let test_word_free_raises () =
  let consumer, _ = P.open_ ~waits_on:[ `Host ] "reuse:raising-consumer" in
  let d, p = P.open_ ~transport:true "reuse:raising-producer" in
  let q = submit (empty d) in
  let w = submit ~waits:[| q |] (empty consumer) in
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

(* The exception reaches the consumer's close. *)
let test_unmap_raises () =
  let consumer, c = P.open_ ~waits_on:[ `Host ] "reuse:unmap-consumer" in
  let d, p = P.open_ ~transport:true "reuse:unmap-producer" in
  let q = submit (empty d) in
  let w = submit ~waits:[| q |] (empty consumer) in
  ignore (P.run p);
  Rig.Point.wait w;
  Rig.close d;
  Rig.free_cache d;
  P.before c "unmap" (fun () -> raise Exit);
  raises Exit (fun () -> Rig.close consumer);
  Rig.free_cache Rig.host;
  equal ~msg:"the word's frees" int 1 (P.word_frees p);
  no_violations ()

(* A device lost before it reached every value *)

(* With nothing submitted, its submitted point is reached at once, and no wait
   on it raises the earlier loss. *)
let test_submitted_after_reopen () =
  first_failure ();
  let name = "reuse:submitted-reopen" in
  let d, _ = P.open_ name in
  let dropped = submit (empty d) in
  Rig.lose d "given up";
  Rig.close d;
  Rig.free_cache d;
  let d' = taker dropped name in
  let s = Rig.Point.submitted d' in
  equal ~msg:"reached, and wait_for" (pair bool bool) (true, true)
    (Rig.Point.reached s, Rig.Point.wait_for ~ns:1_000_000 s);
  Rig.Point.wait s;
  raises_match ~msg:"the dropped point" (lost d) (fun () ->
      Rig.Point.wait dropped);
  Rig.close d'

let () =
  exit
    (run "rig.reuse"
       [
         group "slots"
           [
             test "a closed device holds its slot while a value it kept remains"
               test_slot_closed;
             test "a lost device holds its slot while a value it kept remains"
               test_slot_lost;
             test "a closed device holds its slot while its memory remains"
               test_slot_memory;
             test "a lost device holds its slot until its stop"
               test_slot_unstopped;
             test
               "a submit past the last value loses the device, whose slot no \
                later device takes"
               test_last_value;
             test
               "a closed device answers its last value while a later device \
                opens on its slot"
               test_raised_by_make;
             stateful ~count:70 ~domains:2
               "a slot taken again answers an earlier device's points and \
                stamps"
               cycle_commands;
             test "opens and closes keep a fixed record per device"
               test_open_close;
             test "a borrower closed after its peak's memory went keeps no more"
               (test_mapped_peak Freed_first);
             test
               "a borrower closed before its peak's memory went keeps no more"
               (test_mapped_peak Closed_first);
             test
               "a queue's wait for a producer's point reads the slot's current \
                device"
               test_pair_per_occupant;
             test
               "a device that maps the host reads every slot's word through \
                one mapping, given back at its close"
               test_close_unmaps_word;
             test
               "two domains that decide at once how a device waits for another \
                keep one mapping of its word"
               test_pair_race;
             test "a consumer of a reopening slot gives back earlier mappings"
               test_mappings_released;
             test "a submit gives back mappings of words and frees no memory"
               test_submit_frees;
             test "a decision for a slot's earlier device comes too late"
               test_stale_decision;
             test "a replaced mapping outlives the submit holding the turn"
               test_replaced_in_turn;
             test
               "an exception unmapping a retired mapping loses the consumer \
                and keeps the others due"
               test_retired_unmap_raises;
             test "a device gives back its word of its own once closed"
               test_own_word_back;
             test "a queue waiting on a closed device's word completes"
               test_wait_on_closed;
             test "a device given a word outside rig's array gives it back once"
               test_given_outside;
             test "a lost consumer keeps a word it waits on until it stopped"
               test_lost_consumer;
             test "an exception mapping a word leaves the word to go back"
               test_map_raises;
             test "an exception mapping a word keeps its backtrace"
               test_map_raises_trace;
             test
               "an exception freeing a stopped device's word reaches the caller"
               test_word_free_raises;
             test "an exception unmapping a word leaves the word to go back"
               test_unmap_raises;
             test
               "a device on a lost device's slot has its submitted point \
                reached"
               test_submitted_after_reopen;
           ];
       ])
