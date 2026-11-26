(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Devices: opening and closing, their properties, and machines (rig.mli,
   "Devices"). A device's slot, and what it keeps once closed, are test_reuse's
   and test_limit's. *)

open Windtrap
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support

let submit ?(buffers = [||]) ?(args = [||]) ?waits s =
  Rig.submit ?waits s ~buffers ~args

let empty d = Sub.make d [||]
let device = Testable.make ~pp:Rig.pp ~equal:Rig.equal
let opened r = require_ok ~pp:Format.pp_print_string r
let software name = opened (Rig.software name)
let io name = Support.io name
let lost d = function Rig.Lost (d', _) -> Rig.equal d d' | _ -> false
let count call p = List.length (List.filter (( = ) call) (P.log p))

(* The opener of a fresh Polled device named [name]. *)
let polled name = P.opener name (P.make ())

(* Opens [machine]'s host [name], a driver's device. *)
let open_host machine name = Rig.open_host ~machine (polled name)

(* A Polled device named [name] on the machine named [machine], whose host
   {!Support.machine} opens. *)
let remote ?bus ?granules ?alignment ?cache_bytes machine name =
  let p = P.make ?bus ?granules ?alignment ?cache_bytes () in
  opened (Rig.open_ ~machine:(Support.machine machine) (P.opener name p))

(* [answers f rows] is [(what, f d)] for each device [(what, d)] of [rows]. *)
let answers f rows = List.map (fun (what, d) -> (what, f d)) rows

(* The operating system's name, as [uname -s] prints it. *)
let os () =
  let ic = Unix.open_process_args_in "uname" [| "uname"; "-s" |] in
  let name = input_line ic in
  ignore (Unix.close_process_in ic);
  name

(* [refuses ~sub driver p name] opens [p] as [name] with [driver]: the open
   raises [Invalid_argument] naming [sub], [p], handed to rig and held by no
   device, is stopped once, and the name stays free. *)
let refuses ?(driver = (module P : Rig_edge.Driver with type t = P.t)) ~sub p
    name =
  raises_match (Exn.invalid_arg ~sub) (fun () ->
      Rig.open_ { name; driver; make = (fun ~word -> Ok (P.bind p ~word)) });
  equal ~msg:"the handle's stops" int 1 (count "stop" p);
  ignore (opened (Rig.open_ (polled name)))

(* The driver [B] whose copy engine is [engine]. *)
let engine (module B : Rig_edge.Driver with type t = P.t) engine =
  let module D = struct
    include B

    let facts d = { (B.facts d) with copy_engine = engine }
  end in
  (module D : Rig_edge.Driver with type t = P.t)

(* Polled whose queue named "COPY:0" runs fills alone. *)
module Copyless = struct
  include P

  let facts d =
    let f = P.facts d in
    let fills name = { Rig_edge.name; runs = [ Fill ] } in
    {
      f with
      queues = [ fills "COMPUTE:0"; fills "COPY:0" ];
      copy_engine = None;
    }
end

(* Polled whose facts name no C state. *)
module Edgeless = struct
  include P

  let facts d = { (P.facts d) with edge = 0n }
end

(* Zeroed bytes, whose first member is NULL as a C state's. *)
let zeroed =
  lazy
    (let b = B.create Rig.host 64 in
     B.fill b '\000';
     b)

(* Polled whose facts name a C state whose driver is NULL. *)
module Driverless = struct
  include P

  let facts d =
    let at = Support.Reader.host (Lazy.force zeroed) in
    { (P.facts d) with edge = Nativeint.of_int at }
end

(* Polled that states the host addresses its memory, which it does not. *)
module Overstated = struct
  include P

  let facts d = { (P.facts d) with host_addresses = true }
end

(* [open_] *)

let test_same_name () =
  let d = software "open:same" in
  equal device d (software "open:same");
  not_equal device d (software "open:other")

(* A Polled open of a name that another driver's device holds: a software
   device's or an io device's. *)
let other_driver =
  cases ~id:fst "an open of a name another driver's device holds raises"
    [ ("a software device's", `Software); ("an io device's", `Io) ]
  @@ fun (what, held) ->
  let name = "open:taken-" ^ what in
  (match held with
  | `Software -> ignore (software name)
  | `Io -> ignore (io name));
  raises_match Exn.invalid_arg (fun () -> Rig.open_ (polled name))

let test_failed_open () =
  let fails ~word:_ = Error "no hardware" in
  equal (result device string) (Error "no hardware")
    (Rig.open_ { name = "open:failed"; driver = (module P); make = fails });
  equal string "open:failed"
    (Rig.name (opened (Rig.open_ (polled "open:failed"))))

(* Rig_edge.facts: a copy engine is a queue that runs copies. *)
let copy_engine_refused =
  cases ~id:fst "facts whose copy engine runs no copies raise"
    [
      ("past the queues", engine (module P) (Some 2));
      ("negative", engine (module P) (Some (-1)));
      ("a queue of fills", engine (module Copyless) (Some 1));
    ]
  @@ fun (what, driver) ->
  refuses ~driver ~sub:"copy engine" (P.make ()) ("open:engine " ^ what)

(* Rig_edge.facts: a hang bound is at least 1 ms. *)
let hang_refused =
  cases ~id:string_of_int "facts whose hang bound is below 1 ms raise"
    [ 0; -1; min_int ]
  @@ fun n ->
  refuses ~sub:"hang bound" (P.make ~hang_ms:n ())
    (Printf.sprintf "open:hang %d" n)

(* Rig_edge.facts: each counter has a value, and counters come with the driver's
   measures, which Polled has none of. *)
let counters_refused =
  cases ~id:snd "facts whose counters have no value or no measures raise"
    [
      ([ ("A", 0) ], "has 0 values");
      ([ ("A", -1) ], "has -1 values");
      ([ ("A", 1) ], "gives no measures");
    ]
  @@ fun (counters, sub) ->
  refuses ~sub (P.make ~counters ()) ("open:counters " ^ sub)

(* Rig_edge.facts: [edge] is a C state whose first member is its driver. *)
let edge_refused =
  cases ~id:fst "facts whose edge names no C state or driver raise"
    [
      ("no state", (module Edgeless : Rig_edge.Driver with type t = P.t));
      ("no driver", (module Driverless));
    ]
  @@ fun (what, driver) ->
  refuses ~driver ~sub:"rig_driver" (P.make ()) ("open:edge " ^ what)

(* An exception raised at any allocation of an open, as [Out_of_memory] or a
   signal handler's is, leaves the name free, or open where the open published
   its device, and stops once a handle that [make] handed to rig and that no
   device holds: its caller no longer owns it. A memory profile raises at the
   [n]th allocation inside the call, for each [n] until an open returns. *)
exception Injected

let raising_at n armed =
  let left = ref n in
  let alloc _ =
    if !armed then begin
      decr left;
      if !left = 0 then raise Injected
    end;
    None
  in
  { Gc.Memprof.null_tracker with alloc_minor = alloc; alloc_major = alloc }

(* [opens_raising n open_] is [Some r] where [open_ ()] returned [r] under a
   memory profile raising at its [n]th allocation, or [None] where the profile
   raised. *)
let opens_raising n open_ =
  let armed = ref false in
  let profile = Gc.Memprof.start ~sampling_rate:1. (raising_at n armed) in
  let stop () =
    armed := false;
    Gc.Memprof.stop ();
    Gc.Memprof.discard profile
  in
  Fun.protect ~finally:stop (fun () ->
      armed := true;
      match open_ () with
      | r ->
          armed := false;
          Some r
      | exception Injected -> None)

let test_raise_anywhere () =
  let rec from n =
    let name = Printf.sprintf "open:raise-%d" n in
    let p = P.make () and handed = ref false in
    let make ~word =
      let r = Ok (P.bind p ~word) in
      handed := true;
      r
    in
    let msg = Printf.sprintf "raised at allocation %d" n in
    match
      opens_raising n (fun () -> Rig.open_ { name; driver = (module P); make })
    with
    | Some r ->
        Rig.close (opened r);
        greater ~msg:"allocations that raised" int ~than:1 n
    | None ->
        (* An open that published its device left it open: the name answers it
           without calling [make]. *)
        let calls = ref 0 in
        let again ~word =
          incr calls;
          Ok (P.bind (P.make ()) ~word)
        in
        let d =
          opened (Rig.open_ { name; driver = (module P); make = again })
        in
        equal ~msg (option string) None (Rig.lost d);
        let left_open = !calls = 0 in
        if left_open && not !handed then
          failf "%s: the name is open, and make handed no handle" msg;
        let stopped = !handed && not left_open in
        equal ~msg int (if stopped then 1 else 0) (count "stop" p);
        Rig.close d;
        from (n + 1)
  in
  from 1

(* A lost device's name opens again only once its stop returned: before, the
   open is [Error]. The stop runs in a close, held at Polled's gate. *)
let test_reopen_early () =
  let d, p = P.open_ "open:early" in
  P.gate p;
  P.fail p;
  let loser =
    Thread.create
      (fun () ->
        (try ignore (submit (empty d)) with Rig.Lost _ -> ());
        Rig.close d)
      ()
  in
  Support.await "a stop at the gate" (fun () -> P.sleepers p = 1);
  is_error ~pp:Rig.pp (Rig.open_ (polled "open:early"));
  P.open_gate p;
  Thread.join loser;
  not_equal device d (fst (P.open_ "open:early"))

let test_reopen () =
  let d, p = P.open_ "open:reopen" in
  let s = empty d in
  P.fail p;
  raises_match (lost d) (fun () -> submit s);
  equal ~msg:"the calls before the open" (list string) [] (P.log p);
  let d', _ = P.open_ "open:reopen" in
  equal ~msg:"the calls after the open" (list string) [ "stop" ] (P.log p);
  not_equal device d d';
  equal (option string) None (Rig.lost d')

(* [x1] and [x2], lost together, [x2] by spread, owe their stops; an open whose
   stop raises raises it, and the next open of the name opens it anew, the stop
   having returned. *)
let test_reopen_owed () =
  let x1, p1 = P.open_ "open:owed-1" in
  let x2, p2 = P.open_ ~waits_on:[ `Host ] "open:owed-2" in
  let raised = Atomic.make false in
  let stop () =
    if not (Atomic.exchange raised true) then failwith "the first stop"
  in
  P.before p1 "stop" stop;
  P.before p2 "stop" stop;
  let first = submit (empty x1) in
  ignore (submit (empty x2) ~waits:[| first |]);
  P.fail p1;
  raises_match (lost x1) (fun () -> submit (empty x1));
  equal ~msg:"the spread" (option string) (Some "open:owed-1 lost")
    (Rig.lost x2);
  equal ~msg:"stops before the opens" (pair int int) (0, 0)
    (P.stops p1, P.stops p2);
  let reopen name = opened (Rig.open_ (polled name)) in
  raises_match
    (function Failure _ -> true | _ -> false)
    (fun () -> reopen "open:owed-1");
  let y = reopen "open:owed-1" in
  let y' = reopen "open:owed-2" in
  equal ~msg:"stops" (pair int int) (1, 1) (P.stops p1, P.stops p2);
  List.iter
    (fun (x, y) -> not_equal device x y)
    [ (x1, y); (x1, y'); (x2, y); (x2, y') ]

(* Once a machine's host is closed or lost, the open answers the host's loss as
   its [Lost] prints. *)
let test_open_on_ended_host () =
  let gpu machine = Rig.open_ ~machine (polled "open:ended-gpu") in
  let closed = opened (open_host "ended-closed" "CPU") in
  Rig.close closed;
  let lost = P.make () in
  let host =
    opened (Rig.open_host ~machine:"ended-lost" (P.opener "CPU" lost))
  in
  P.fault lost "unplugged";
  (try ignore (B.create host 8) with Rig.Lost _ -> ());
  equal
    (list (result device string))
    [
      Error "CPU@ended-closed lost: closed";
      Error "CPU@ended-lost lost: unplugged";
    ]
    [ gpu closed; gpu host ]

(* An open whose opener blocks holds back only opens of its own name, which
   answer the device that opener makes. *)
let test_blocked_opener () =
  let lock = Mutex.create () and cond = Condition.create () in
  let inside = ref false and go = ref false and calls = ref 0 in
  let make ~word =
    Mutex.protect lock (fun () ->
        incr calls;
        inside := true;
        while not !go do
          Condition.wait cond lock
        done);
    Ok (P.bind (P.make ()) ~word)
  in
  let opener : P.t Rig_edge.opener =
    { name = "open:slow"; driver = (module P); make }
  in
  let first = ref None and second = ref None in
  let slow =
    Thread.create (fun () -> first := Some (opened (Rig.open_ opener))) ()
  in
  Support.await "a running opener" (fun () ->
      Mutex.protect lock (fun () -> !inside));
  let started = Atomic.make false in
  let same =
    Thread.create
      (fun () ->
        Atomic.set started true;
        let another = { opener with make = (fun ~word -> make ~word) } in
        second := Some (opened (Rig.open_ another)))
      ()
  in
  Support.await "a concurrent open of the same name" (fun () ->
      Atomic.get started);
  equal string "open:fast" (Rig.name (fst (P.open_ "open:fast")));
  Mutex.protect lock (fun () ->
      go := true;
      Condition.broadcast cond);
  Thread.join slow;
  Thread.join same;
  let d = Option.get !first in
  equal device d (Option.get !second);
  equal ~msg:"the opener's calls" int 1 !calls;
  Rig.close d;
  let reopened = opened (Rig.open_ opener) in
  not_equal device d reopened;
  equal ~msg:"the opener's calls after a close" int 2 !calls;
  Rig.close reopened

(* The inner open raises, which ends the outer one, and the name then opens. *)
let test_opener_opens_itself () =
  let rec opener : P.t Rig_edge.opener =
    { name = "open:itself"; driver = (module P); make }
  and make ~word =
    ignore (Rig.open_ opener);
    Ok (P.bind (P.make ()) ~word)
  in
  raises_match ~msg:"the open" Exn.invalid_arg (fun () -> Rig.open_ opener);
  let d, _ = P.open_ "open:itself" in
  equal string "open:itself" (Rig.name d);
  Rig.close d

(* An open that waits for another open of its name returns to OCaml at least
   every 200 ms, where a pending signal's handler runs. The other opener parks
   outside OCaml until unparked, and a thread of no OCaml code raises SIGINT
   once the open blocks in its condition wait, so only the waiting open can take
   the signal before the other open ends. The handler raises Sys.Break on the
   waiting thread alone. *)
let test_break_waiting_open () =
  if not Support.sees_waits then
    skip ~reason:"the system shows no thread's blocking call" ();
  let name = "open:break-waiting" in
  let make ~word =
    Support.park ();
    Ok (P.bind (P.make ()) ~word)
  in
  let first = ref None in
  let opener =
    Thread.create
      (fun () -> first := Some (Rig.open_ { name; driver = (module P); make }))
      ()
  in
  let calls = ref 0 in
  let got =
    Fun.protect ~finally:Support.unpark @@ fun () ->
    Support.await "the first opener" Support.parked;
    let waiting = Thread.id (Thread.self ()) and taken = ref "nowhere" in
    let handle _ =
      if Thread.id (Thread.self ()) = waiting then raise Sys.Break
      else taken := "on another thread"
    in
    let second ~word =
      incr calls;
      Ok (P.bind (P.make ()) ~word)
    in
    let before = Sys.signal Sys.sigint (Sys.Signal_handle handle) in
    Fun.protect
      ~finally:(fun () -> Sys.set_signal Sys.sigint before)
      (fun () ->
        Support.interrupt_in_wait (fun () ->
            match Rig.open_ { name; driver = (module P); make = second } with
            | Ok _ -> "opened, the signal taken " ^ !taken
            | Error e -> "Error " ^ e
            | exception Sys.Break -> "Sys.Break"))
  in
  Thread.join opener;
  equal ~msg:"the waiting open, and whether it waited" (pair string bool)
    ("Sys.Break", true) got;
  equal ~msg:"its opener's calls" int 0 !calls;
  is_ok ~pp:Format.pp_print_string ~msg:"the other open" (Option.get !first)

(* A signal handler that runs in the wait, where it returns to OCaml, calls rig
   as any code does: the wait holds no lock. The opener is parked outside OCaml,
   so the waiting thread is the one that runs the handler. *)
let test_handled_in_open_wait () =
  if not Support.sees_waits then
    skip ~reason:"the system shows no thread's blocking call" ();
  let name = "open:handled-in-wait" in
  let parked ~word =
    Support.park ();
    Ok (P.bind (P.make ()) ~word)
  in
  let first = ref None in
  let opener =
    Thread.create
      (fun () ->
        first := Some (Rig.open_ { name; driver = (module P); make = parked }))
      ()
  in
  let call = ref "no handler ran" in
  let r, raised =
    Fun.protect ~finally:Support.unpark @@ fun () ->
    Support.await "the opener" Support.parked;
    let handle _ =
      (call :=
         match B.create Rig.host 8 with
         | _ -> "ran"
         | exception Invalid_argument why -> why);
      Support.unpark ()
    in
    let before = Sys.signal Sys.sigint (Sys.Signal_handle handle) in
    Fun.protect
      ~finally:(fun () -> Sys.set_signal Sys.sigint before)
      (fun () -> Support.interrupt_in_wait (fun () -> Rig.open_ (polled name)))
  in
  Thread.join opener;
  equal ~msg:"the handler's call of rig, and whether the open waited"
    (pair string bool) ("ran", true) (!call, raised);
  let d = opened r in
  equal ~msg:"the device the opener made" device d (opened (Option.get !first));
  Rig.close d

(* [unpark_on_signal f] is [f ()], run while a SIGINT, which
   {!Support.interrupt_in_wait} raises once the calling thread blocks in a
   condition wait, unparks the parked thread instead of raising, at the time
   [unparked_at] records: the parked thread then goes on while the caller
   waits. *)
let unparked_at = ref 0.

let unpark_on_signal f =
  let unpark _ =
    unparked_at := Unix.gettimeofday ();
    Support.unpark ()
  in
  let before = Sys.signal Sys.sigint (Sys.Signal_handle unpark) in
  Fun.protect
    ~finally:(fun () -> Sys.set_signal Sys.sigint before)
    (fun () -> Support.interrupt_in_wait f)

(* An opener that parks outside OCaml, then raises [Exit]. *)
let parks_then_raises ~word:_ =
  Support.park ();
  raise Exit

(* An open that an exception ends wakes the opens of its name waiting for it,
   and the next one opens. The first opener raises only once the second open
   waits. The time runs from the unpark: a wait runs no signal handler, so the
   handler unparks when the open's wait returns to OCaml, and the open starts
   its next wait right after. Unwoken, that wait returns 200 ms after its start,
   rig's period; the end wakes it well within 150 ms. rig shows no count of the
   waits its period ended, so the time is what tells a missed wake-up. *)
let test_ended_wakes_waiter () =
  if not Support.sees_waits then
    skip ~reason:"the system shows no thread's blocking call" ();
  let name = "open:ended-wakes" in
  let first = ref "" in
  let opener =
    Thread.create
      (fun () ->
        first :=
          match
            Rig.open_ { name; driver = (module P); make = parks_then_raises }
          with
          | _ -> "returned"
          | exception Exit -> "raised Exit")
      ()
  in
  Support.await "the first opener" Support.parked;
  let second, waited = unpark_on_signal (fun () -> Rig.open_ (polled name)) in
  let took = Unix.gettimeofday () -. !unparked_at in
  Thread.join opener;
  equal ~msg:"the first open, and whether the second waited" (pair string bool)
    ("raised Exit", true) (!first, waited);
  less ~msg:"seconds from the end to the second open's return" float_exact
    ~than:0.15 took;
  Rig.close (opened second)

(* [open_io] *)

let test_io_name () =
  let d = io "open:io-name" in
  equal device d (opened (Rig.open_io "open:io-name"))

let test_io_driver_name () =
  ignore (P.open_ "open:io-driver");
  raises_match Exn.invalid_arg (fun () -> Rig.open_io "open:io-driver")

(* The opener's [make] runs on the thread that opens the name. *)
let test_io_inside_open () =
  let name = "open:io-inside" in
  let inner = ref (Ok ()) in
  let make ~word =
    (inner :=
       match Rig.open_io name with
       | _ -> Ok ()
       | exception Invalid_argument why -> Error why);
    Ok (P.bind (P.make ()) ~word)
  in
  ignore (opened (Rig.open_ { name; driver = (module P); make }));
  satisfies ~claim:"Invalid_argument" (result unit string) Result.is_error
    !inner

(* rig.mli: "An io device runs no work and makes no memory". *)
let test_io_no_memory () =
  let d = io "open:io-memory" in
  equal ~msg:"an empty buffer's length" int 0 (B.length (B.create d 0));
  raises_match ~msg:"a buffer of 8 bytes" Exn.invalid_arg (fun () ->
      B.create d 8)

(* rig.mli, budget: "It starts at [max_int] for host and an io device";
   capacity: [None] for an io device, "whose memory is the machine's". *)
let test_io_budget () =
  let d = io "open:io-budget" in
  equal ~msg:"its budget" int max_int (Rig.budget d);
  is_none ~msg:"its capacity" (Rig.capacity d)

let test_io_raise_anywhere () =
  let rec from n =
    let name = Printf.sprintf "open:raise-io-%d" n in
    match opens_raising n (fun () -> Rig.open_io name) with
    | Some r ->
        Rig.close (opened r);
        greater ~msg:"allocations that raised" int ~than:1 n
    | None ->
        let d = opened (Rig.open_io name) in
        equal
          ~msg:(Printf.sprintf "raised at allocation %d" n)
          (option string) None (Rig.lost d);
        Rig.close d;
        from (n + 1)
  in
  from 1

(* [software] *)

let test_software_driver_name () =
  ignore (P.open_ "open:software-driver");
  raises_match Exn.invalid_arg (fun () -> Rig.software "open:software-driver")

(* [close] *)

(* A close ends the device as a loss does, with the reason "closed": its stop
   runs once, before close returns. *)
let test_close () =
  let d, p = P.open_ "open:close" in
  let v = Rig.Point.value (submit (empty d)) in
  Rig.close d;
  equal ~msg:"the work ran" int v (Rig.signaled d);
  equal (option string) (Some "closed") (Rig.lost d);
  equal ~msg:"stops" int 1 (count "stop" p);
  match submit (empty d) with
  | _ -> failf "a closed device took a submit"
  | exception (Rig.Lost (d', why) as e) ->
      equal device d d';
      equal string "closed" why;
      equal string "open:close lost: closed" (Printexc.to_string e)

(* The stop waits at the gate while the second close is sampled for 50 ms: it
   waits in rig, where no driver seam sees it. *)
let test_two_closes () =
  let d, p = P.open_ "open:two-closes" in
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
  equal ~msg:"closes returned, and stops at the gate, while the stop waits"
    (pair int int) (0, 1)
    (Atomic.get returned, P.sleepers p);
  P.open_gate p;
  equal ~msg:"the stops each close saw" (list int) [ 1; 1 ]
    [ Domain.join first; Domain.join second ];
  equal (option string) (Some "closed") (Rig.lost d)

let test_close_fault () =
  let d, p = P.open_ "open:close-fault" in
  ignore (submit (empty d));
  P.fault p "the engine hung";
  Rig.close d;
  equal (option string) (Some "the engine hung") (Rig.lost d);
  equal ~msg:"stops" int 1 (count "stop" p)

let test_close_host () =
  raises_match Exn.invalid_arg (fun () -> Rig.close Rig.host);
  equal ~msg:"the host's loss" (option string) None (Rig.lost Rig.host)

(* An io device's work is the caller's. *)
let test_close_io () =
  let d = io "open:close-io" in
  let b = Support.io_buffer d 8 in
  Rig.close d;
  equal (option string) (Some "closed") (Rig.lost d);
  raises_match (lost d) (fun () -> B.copy ~src:b ~dst:(B.create Rig.host 8))

let test_close_machine () =
  let far = opened (open_host "closing" "CPU") in
  let g = opened (Rig.open_ ~machine:far (polled "open:closing-gpu")) in
  Rig.close far;
  equal
    (list (option string))
    [ Some "closed"; Some "closed" ]
    [ Rig.lost g; Rig.lost far ]

let test_close_lost_host () =
  let lost = P.make () in
  let far =
    opened (Rig.open_host ~machine:"closing-lost" (P.opener "CPU" lost))
  in
  let g = opened (Rig.open_ ~machine:far (polled "open:closing-lost-gpu")) in
  P.fault lost "unplugged";
  (try ignore (B.create far 8) with Rig.Lost _ -> ());
  Rig.close far;
  equal (option string) (Some "closed") (Rig.lost g)

(* The opener parks outside OCaml until the close waits for it, then makes its
   device, which the close then closes. *)
let test_close_opening () =
  if not Support.sees_waits then
    skip ~reason:"the system shows no thread's blocking call" ();
  let far = opened (open_host "close-opening" "CPU") in
  let make ~word =
    Support.park ();
    Ok (P.bind (P.make ()) ~word)
  in
  let first = ref (Error "not returned") in
  let opener =
    Thread.create
      (fun () ->
        first :=
          Rig.open_ ~machine:far
            { name = "open:close-opening"; driver = (module P); make })
      ()
  in
  Support.await "the opener" Support.parked;
  let (), waited = unpark_on_signal (fun () -> Rig.close far) in
  Thread.join opener;
  let g = opened !first in
  equal ~msg:"the device that was opening, and whether the close waited"
    (pair (option string) bool)
    (Some "closed", true)
    (Rig.lost g, waited);
  is_error ~pp:Rig.pp ~msg:"a later open"
    (Rig.open_ ~machine:far (polled "open:close-opening-later"))

(* A close of a machine's host waits for the opens in flight on the machine. A
   Ctrl-C during that wait ends the close and leaves the machine as it was: its
   host is not lost, and its devices open. The opener parks outside OCaml until
   the close has taken the signal. *)
let test_break_waiting_close () =
  if not Support.sees_waits then
    skip ~reason:"the system shows no thread's blocking call" ();
  let far = opened (open_host "break-close" "CPU") in
  let gpu name make =
    Rig.open_ ~machine:far { name; driver = (module P); make }
  in
  let make ~word =
    Support.park ();
    Ok (P.bind (P.make ()) ~word)
  in
  let first = ref None in
  let opener =
    Thread.create (fun () -> first := Some (gpu "open:break-close-1" make)) ()
  in
  let got =
    Fun.protect ~finally:Support.unpark @@ fun () ->
    Support.await "the opener" Support.parked;
    Sys.catch_break true;
    Fun.protect
      ~finally:(fun () -> Sys.catch_break false)
      (fun () ->
        Support.interrupt_in_wait (fun () ->
            match Rig.close far with
            | () -> "closed"
            | exception Sys.Break -> "Sys.Break"))
  in
  Thread.join opener;
  equal ~msg:"the close, and whether it waited" (pair string bool)
    ("Sys.Break", true) got;
  equal ~msg:"the host" (option string) None (Rig.lost far);
  is_ok ~pp:Format.pp_print_string ~msg:"the open in flight" (Option.get !first);
  is_ok ~pp:Format.pp_print_string ~msg:"a later open"
    (gpu "open:break-close-2" (fun ~word -> Ok (P.bind (P.make ()) ~word)));
  Rig.close far

(* Two closes of one host wait for an open on its machine. A Ctrl-C ends the
   second close's wait, and the first close still holds the machine: an open on
   it is an [Error] until the first close ends it. *)
let test_break_second_close () =
  if not Support.sees_waits then
    skip ~reason:"the system shows no thread's blocking call" ();
  let far = opened (open_host "second-close" "CPU") in
  let gpu name make =
    Rig.open_ ~machine:far { name; driver = (module P); make }
  in
  let make ~word =
    Support.park ();
    Ok (P.bind (P.make ()) ~word)
  in
  let first_open = ref None in
  let opener =
    Thread.create
      (fun () -> first_open := Some (gpu "open:second-close-1" make))
      ()
  in
  let closed = Printf.sprintf "%s lost: closed" (Rig.name far) in
  let first_close, got, during =
    Fun.protect ~finally:Support.unpark @@ fun () ->
    Support.await "the opener" Support.parked;
    let first_close = Thread.create (fun () -> Rig.close far) () in
    let probe ~word:_ = Error "probe" in
    Support.await "the first close" (fun () ->
        gpu "open:second-close-probe" probe = Error closed);
    let main = Thread.id (Thread.self ()) in
    (* The first close's thread returns to OCaml too: a signal it takes is
       raised again, until the second close's thread takes it. *)
    let handle _ =
      if Thread.id (Thread.self ()) = main then raise Sys.Break
      else Unix.kill (Unix.getpid ()) Sys.sigint
    in
    let before = Sys.signal Sys.sigint (Sys.Signal_handle handle) in
    let interrupted =
      Fun.protect
        ~finally:(fun () -> Sys.set_signal Sys.sigint before)
        (fun () ->
          Support.interrupt_in_wait (fun () ->
              match Rig.close far with
              | () -> "closed"
              | exception Sys.Break -> "Sys.Break"))
    in
    let during =
      gpu "open:second-close-2" (fun ~word -> Ok (P.bind (P.make ()) ~word))
    in
    (first_close, interrupted, during)
  in
  Thread.join opener;
  Thread.join first_close;
  equal ~msg:"the second close, and whether it waited" (pair string bool)
    ("Sys.Break", true) got;
  equal ~msg:"an open while the first close runs" (result unit string)
    (Error closed) (Result.map ignore during);
  equal ~msg:"the host" (option string) (Some "closed") (Rig.lost far);
  ignore (Option.get !first_open)

(* The opener raises only once the close waits. *)
let test_close_ends_interrupted_open () =
  if not Support.sees_waits then
    skip ~reason:"the system shows no thread's blocking call" ();
  let far = opened (open_host "close-ended" "CPU") in
  let first = ref "" in
  let opener =
    Thread.create
      (fun () ->
        first :=
          match
            Rig.open_ ~machine:far
              {
                name = "open:close-ended";
                driver = (module P);
                make = parks_then_raises;
              }
          with
          | _ -> "returned"
          | exception Exit -> "raised Exit")
      ()
  in
  Support.await "the opener" Support.parked;
  let (), waited = unpark_on_signal (fun () -> Rig.close far) in
  Thread.join opener;
  equal ~msg:"the open, and whether the close waited" (pair string bool)
    ("raised Exit", true) (!first, waited);
  equal ~msg:"the host" (option string) (Some "closed") (Rig.lost far)

(* An open that its opener's exception ended, with no open after it, is no open
   in flight. *)
let test_close_after_raised_open () =
  let far = opened (open_host "close-raised" "CPU") in
  raises Exit (fun () ->
      Rig.open_ ~machine:far
        {
          name = "open:close-raised";
          driver = (module P);
          make = (fun ~word:_ -> raise Exit);
        });
  Rig.close far;
  equal ~msg:"the host" (option string) (Some "closed") (Rig.lost far)

let test_close_inside_open () =
  let far = opened (open_host "close-inside" "CPU") in
  let inner = ref (Ok ()) in
  let make ~word =
    (inner :=
       match Rig.close far with
       | () -> Ok ()
       | exception Invalid_argument why -> Error why);
    Ok (P.bind (P.make ()) ~word)
  in
  ignore
    (opened
       (Rig.open_ ~machine:far
          { name = "open:close-inside"; driver = (module P); make }));
  satisfies ~claim:"Invalid_argument" (result unit string) Result.is_error
    !inner;
  equal ~msg:"the host" (option string) None (Rig.lost far)

(* Properties *)

(* Collects, then frees [d]'s cache, which first drains [d]: a closed device's
   memory collected after its stop returns so. *)
let settle d =
  Gc.full_major ();
  Rig.free_cache d

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

(* Its points name it until a later device took its slot, then name no device,
   and their waits return. *)
let test_ended () =
  let key : string list Type.Id.t = Type.Id.make () in
  let cap = [ "add" ] in
  let make ~word =
    Ok (P.bind (P.make ~capability:(Rig_edge.Capability (key, cap)) ()) ~word)
  in
  let d =
    opened (Rig.open_ { name = "open:ended"; driver = (module P); make })
  in
  let k = Rig.Local.new_key (fun _ -> ()) in
  Rig.Local.get k d;
  ignore (B.create d 64 : B.t);
  let base = Rig.submitted d in
  let p = submit (empty d) in
  Rig.Point.wait p;
  let facts d =
    ( Rig.arch d,
      [
        Rig.computes d;
        Rig.runs_on_host d;
        Rig.reaches Rig.host d;
        Rig.reaches d Rig.host;
      ] )
  in
  let before = facts d in
  Rig.set_budget d 4096;
  Rig.close d;
  settle d;
  equal string "open:ended" (Rig.name d);
  equal ~msg:"arch and facts" (pair string (list bool)) before (facts d);
  equal device Rig.host (Rig.host_of d);
  satisfies ~claim:"the record declared"
    (option (list string))
    (function Some c -> c == cap | None -> false)
    (Rig.capability d key);
  equal ~msg:"budget and allocated" (pair int int) (4096, 0)
    (Rig.budget d, Rig.allocated d);
  equal (option string) (Some "closed") (Rig.lost d);
  let answers () =
    equal ~msg:"submitted, signaled, reached" (triple int int bool)
      (base + 1, base + 1, true)
      (Rig.submitted d, Rig.signaled d, Rig.Point.reached p);
    Rig.Point.wait p;
    raises_match (lost d) (fun () -> B.create d 1);
    raises_match (lost d) (fun () -> Rig.Local.get k d);
    raises_match (lost d) (fun () -> submit (empty d))
  in
  answers ();
  equal (option device) (Some d) (Rig.Point.device p);
  let later = taker p "open:ended-later" in
  (* "its values go on past every value the slot's earlier devices were
     given" *)
  at_least ~msg:"the later device's first value, less 1" int ~than:(base + 1)
    (Rig.submitted later);
  answers ();
  Rig.close later

(* [name] *)

(* rig.mli: host "is this process's device, named "CPU"". *)
let test_host_name () = equal string "CPU" (Rig.name Rig.host)

let test_name () =
  equal
    (list (pair string string))
    [
      ("a software device", "open:name-software");
      ("a driver's device", "open:name-polled");
      ("an io device", "open:name-io");
      ("another machine's device", "open:name-far@name");
      ("another machine's host", "CPU@name");
    ]
    (answers Rig.name
       [
         ("a software device", software "open:name-software");
         ("a driver's device", fst (P.open_ "open:name-polled"));
         ("an io device", io "open:name-io");
         ("another machine's device", remote "name" "open:name-far");
         ("another machine's host", Support.machine "name");
       ])

(* [equal] *)

let test_reopened () =
  let d, _ = P.open_ "open:reopened" in
  equal device d d;
  Rig.close d;
  let d' = fst (P.open_ "open:reopened") in
  not_equal device d d';
  equal (option string) None (Rig.lost d')

(* [pp] *)

let test_pp () =
  let devices =
    [
      ("the host", Rig.host);
      ("a software device", software "open:pp-software");
      ("a driver's device", fst (P.open_ "open:pp-polled"));
      ("an io device", io "open:pp-io");
      ("another machine's device", remote "pp" "open:pp-far");
    ]
  in
  equal
    (list (pair string string))
    (answers Rig.name devices)
    (answers (Format.asprintf "%a" Rig.pp) devices)

(* [arch] *)

(* Polled names its processor ["polled"], or the [arch] it is made with. *)
let test_driver_arch () =
  equal
    (list (pair string string))
    [ ("Polled's own", "polled"); ("one made gfx1201", "gfx1201") ]
    (answers Rig.arch
       [
         ("Polled's own", fst (P.open_ "open:arch"));
         ("one made gfx1201", fst (P.open_ ~arch:"gfx1201" "open:arch-gfx"));
       ])

(* [instruction_set] *)

(* rig.mli: "["x86_64"] for a level of x86-64, ["arm64"] for an arm64 arch, and
   the arch itself for any other". *)
let instruction_sets =
  cases ~id:fst "a driver's device's instruction set is its arch's"
    [
      ("x86-64", "x86_64");
      ("x86-64-v3", "x86_64");
      ("x86-64-v4", "x86_64");
      ("x86-64-v5", "x86-64-v5");
      ("arm64", "arm64");
      ("arm64+bf16+fp16", "arm64");
      ("gfx1201", "gfx1201");
      ("gfx90a:xnack+", "gfx90a:xnack+");
      ("", "");
    ]
  @@ fun (arch, isa) ->
  let d, _ = P.open_ ~arch ("open:isa " ^ arch) in
  equal string isa (Rig.instruction_set d)

(* rig.mli's arch names the instruction sets rig knows, and [""] for another. *)
let test_host_isa () =
  mem string
    (Rig.instruction_set Rig.host)
    [
      "arm64";
      "x86_64";
      "ppc64le";
      "ppc64";
      "s390x";
      "riscv64";
      "loongarch64";
      "";
    ]

(* [bus] *)

let test_bus () =
  let bus = "0000:03:00.0" in
  equal
    (list (pair string (option string)))
    [
      ("a driver's stated", Some bus);
      ("a driver's, none stated", None);
      ("the host", None);
      ("a software device", None);
      ("an io device", None);
    ]
    (answers Rig.bus
       [
         ("a driver's stated", fst (P.open_ ~bus "open:bus"));
         ("a driver's, none stated", fst (P.open_ "open:no-bus"));
         ("the host", Rig.host);
         ("a software device", software "open:bus-software");
         ("an io device", io "open:bus-io");
       ])

(* rig.mli: [None] for "a device of another machine", whatever its driver
   states. *)
let test_remote_bus () =
  let d = remote ~bus:"0000:03:00.0" "bus" "open:bus-far" in
  equal (option string) None (Rig.bus d)

(* [computes] *)

let test_computes () =
  equal
    (list (pair string bool))
    [
      ("the host", true);
      ("a software device", true);
      ("a driver's device", true);
      ("another machine's device", true);
      ("an io device", false);
    ]
    (answers Rig.computes
       [
         ("the host", Rig.host);
         ("a software device", software "open:computes-software");
         ("a driver's device", fst (P.open_ "open:computes-polled"));
         ("another machine's device", remote "computes" "open:computes-far");
         ("an io device", io "open:computes-io");
       ])

(* [runs_on_host] *)

let test_runs_on_host () =
  equal
    (list (pair string bool))
    [
      ("the host", true);
      ("a software device", true);
      ("a driver's device", false);
      ("another machine's device", false);
      ("an io device", false);
    ]
    (answers Rig.runs_on_host
       [
         ("the host", Rig.host);
         ("a software device", software "open:on-host-software");
         ("a driver's device", fst (P.open_ "open:on-host-polled"));
         ("another machine's device", remote "on-host" "open:on-host-far");
         ("an io device", io "open:on-host-io");
       ])

(* [offers] *)

let kinds = B.[ ("Device", Device); ("Pinned", Pinned); ("Mapped", Mapped) ]

(* What [offers] answers of each kind on [d], in [kinds]' order. *)
let offered d = List.map (fun (_, k) -> Rig.offers d k) kinds

(* The devices of the offers table: the host, a software device, an io device,
   and Polled devices whose memory the host addresses or not, with Mapped memory
   or without. *)
let offering prefix =
  let polled ?transport ~host_visible ~mapped name =
    let d, p = P.open_ ?transport ~host_visible ~mapped (prefix ^ name) in
    (d, Some p)
  in
  [
    ("the host", (Rig.host, None));
    ("a software device", (software (prefix ^ "software"), None));
    ("an io device", (io (prefix ^ "io"), None));
    ("addressed", polled ~host_visible:true ~mapped:true "addressed");
    ("addressed, no Mapped", polled ~host_visible:true ~mapped:false "own");
    ("hidden", polled ~host_visible:false ~mapped:true "hidden");
    ("hidden, no Mapped", polled ~host_visible:false ~mapped:false "none");
    ( "behind a transport",
      polled ~transport:true ~host_visible:true ~mapped:true "transport" );
  ]

(* rig.mli, offers: Device memory on every device but an io device; Pinned where
   the host does not address the device's memory ([host_addresses], which Polled
   states with [host_visible] and without [transport]); Mapped where the host
   addresses the memory, or the driver makes it. *)
let offers_table =
  [
    ("the host", [ true; false; true ]);
    ("a software device", [ true; false; true ]);
    ("an io device", [ false; false; false ]);
    ("addressed", [ true; false; true ]);
    ("addressed, no Mapped", [ true; false; true ]);
    ("hidden", [ true; true; true ]);
    ("hidden, no Mapped", [ true; true; false ]);
    ("behind a transport", [ true; true; true ]);
  ]

let test_offers () =
  let devices = offering "open:offers-" in
  equal
    (list (pair string (list bool)))
    offers_table
    (answers (fun (d, _) -> offered d) devices)

(* Where [offers] is [false], {!Buffer.create} of a word raises. *)
let test_offers_create () =
  let made d memory =
    match B.create ~memory d 8 with
    | b ->
        ignore (Sys.opaque_identity b);
        true
    | exception Invalid_argument _ -> false
  in
  let devices = offering "open:offers-create-" in
  equal
    (list (pair string (list bool)))
    (answers (fun (d, _) -> offered d) devices)
    (answers (fun (d, _) -> List.map (fun (_, k) -> made d k) kinds) devices)

(* Unlike an io device's refusal, which holds for [n > 0], a kind the device
   does not offer raises for every [n], 0 included (rig.mli, Buffer.create):
   Mapped memory where the driver makes none and the host does not address the
   device's memory, Pinned memory where the host addresses it. *)
let no_kind_empty =
  cases ~id:fst "create of no bytes of a kind not offered raises"
    [ ("Mapped", `Mapped); ("Pinned", `Pinned) ]
  @@ fun (kind, which) ->
  let name = "open:offers-empty-" ^ kind in
  let d, memory, sub =
    match which with
    | `Mapped ->
        ( fst (P.open_ ~host_visible:false ~mapped:false name),
          B.Mapped,
          "no Mapped memory" )
    | `Pinned -> (fst (P.open_ name), B.Pinned, "no Pinned memory")
  in
  raises_match (Exn.invalid_arg ~sub) (fun () -> B.create ~memory d 0)

let test_offers_lost () =
  let devices = offering "open:offers-lost-" in
  let before = answers (fun (d, _) -> offered d) devices in
  List.iter
    (fun (_, (d, p)) ->
      match p with
      | Some p -> (
          P.fault p "gone";
          try ignore (B.create d 8) with Rig.Lost _ -> ())
      | None -> if d != Rig.host then Rig.close d)
    devices;
  equal ~msg:"every device but the host lost"
    (list (pair string bool))
    (List.map (fun (what, _) -> (what, what <> "the host")) devices)
    (answers (fun (d, _) -> Rig.lost d <> None) devices);
  equal
    (list (pair string (list bool)))
    before
    (answers (fun (d, _) -> offered d) devices)

(* [alignment] *)

(* rig.mli, alignment: "[16] for host and [64] for a software device", a
   driver's as it states. An io device makes no buffer: [1]. *)
let test_alignment () =
  equal
    (list (pair string int))
    [
      ("the host", 16);
      ("a software device", 64);
      ("a driver's device", 256);
      ("a driver's device the host does not address", 256);
      ("an io device", 1);
      ("another machine's device", 512);
    ]
    (answers Rig.alignment
       [
         ("the host", Rig.host);
         ("a software device", software "open:alignment-software");
         ("a driver's device", fst (P.open_ ~alignment:256 "open:alignment"));
         ( "a driver's device the host does not address",
           fst
             (P.open_ ~alignment:256 ~host_visible:false "open:alignment-hidden")
         );
         ("an io device", io "open:alignment-io");
         ( "another machine's device",
           remote ~alignment:512 "alignment" "open:alignment-far" );
       ])

(* Every non-empty buffer starts on its device's alignment, of each kind it has
   and on either side of the sizes the host aligns further. *)
let test_aligned () =
  let d = software "open:aligned-software" in
  let p, _ = P.open_ ~alignment:256 "open:aligned-polled" in
  let hidden, _ =
    P.open_ ~alignment:256 ~host_visible:false "open:aligned-hidden"
  in
  let sizes = [ 1; 15; 16; 17; 127; 128; 129; 4096; 65535; 1 lsl 18 ] in
  let off =
    List.concat_map
      (fun (d, kinds) ->
        List.concat_map
          (fun memory ->
            List.filter_map
              (fun n ->
                let b = B.create ~memory d n in
                if B.aligned b ~at:0 (Rig.alignment d) then None
                else Some (Printf.sprintf "%s, %d bytes" (Rig.name d) n))
              sizes)
          kinds)
      B.
        [
          (Rig.host, [ Device; Mapped ]);
          (d, [ Device; Mapped ]);
          (p, [ Device; Mapped ]);
          (hidden, [ Device; Pinned; Mapped ]);
        ]
  in
  equal ~msg:"buffers off their device's alignment" (list string) [] off

let alignment_refused =
  cases ~id:string_of_int "an alignment that is not a power of two raises"
    [ 0; -16; 24; 48 ]
  @@ fun n ->
  refuses ~sub:"power of two" (P.make ~alignment:n ())
    (Printf.sprintf "open:alignment %d" n)

(* A region off the device's alignment goes back to the driver and faults the
   device: Polled's start on a page, never on 2^60. *)
let test_misaligned () =
  let d, p = P.open_ ~alignment:(1 lsl 60) "open:misaligned" in
  raises_match ~msg:"the allocation" (lost d) (fun () -> B.create d 64);
  satisfies ~claim:"lost" (option string) Option.is_some (Rig.lost d);
  equal ~msg:"the region freed" (list string) [ "alloc"; "free" ]
    (List.filter (fun c -> c = "alloc" || c = "free") (P.log p))

(* [granules] *)

(* rig.mli, granules: "[[(0, 4)]] ... host, ..., an io device and a software
   device", a driver's as it states them, here NVIDIA RM's. *)
let test_granules () =
  let kib = 1024 and mib = 1 lsl 20 in
  let rm = [ (0, 64 * kib); (2 * mib, 2 * mib) ] in
  equal
    (list (pair string (list (pair int int))))
    [
      ("the host", [ (0, 4) ]);
      ("a software device", [ (0, 4) ]);
      ("an io device", [ (0, 4) ]);
      ("a driver's device", rm);
      ("another machine's device", rm);
    ]
    (answers Rig.granules
       [
         ("the host", Rig.host);
         ("a software device", software "open:granules-software");
         ("an io device", io "open:granules-io");
         ("a driver's device", fst (P.open_ ~granules:rm "open:granules"));
         ( "another machine's device",
           remote ~granules:rm "granules" "open:granules-far" );
       ])

(* Rig_edge.facts: units from 0, each a power of two of at least a word and none
   below the one before. *)
let units g =
  String.concat "; " (List.map (fun (a, b) -> Printf.sprintf "(%d, %d)" a b) g)

let granules_refused =
  cases ~id:units "units that do not start from 0 and increase raise"
    [
      [];
      [ (4, 4) ];
      [ (0, 2) ];
      [ (0, 12) ];
      [ (0, 4); (0, 8) ];
      [ (0, 64); (1024, 16) ];
    ]
  @@ fun granules ->
  refuses ~sub:"units" (P.make ~granules ()) ("open:granules " ^ units granules)

(* [cache_bytes] *)

let test_cache_bytes () =
  let mib48 = 48 lsl 20 in
  equal
    (list (pair string (option int)))
    [
      ("a software device, the host's", Rig.cache_bytes Rig.host);
      ("a driver's device", Some mib48);
      ("a driver's device, none stated", None);
      ("an io device", None);
      ("another machine's device", Some mib48);
    ]
    (answers Rig.cache_bytes
       [
         ("a software device, the host's", software "open:cache-software");
         ("a driver's device", fst (P.open_ ~cache_bytes:mib48 "open:cache"));
         ("a driver's device, none stated", fst (P.open_ "open:cache-none"));
         ("an io device", io "open:cache-io");
         ( "another machine's device",
           remote ~cache_bytes:mib48 "cache" "open:cache-far" );
       ])

(* macOS states the caches, and Linux in sysfs where the firmware describes
   them: a cache's size file that holds a positive size. A sandbox may show no
   sysfs, and other systems state none. *)
let test_host_cache () =
  let sysfs = "/sys/devices/system/cpu/cpu0/cache" in
  let sized i =
    let file = Printf.sprintf "%s/index%d/size" sysfs i in
    match In_channel.with_open_text file In_channel.input_all with
    | s -> Scanf.sscanf_opt s "%d" Fun.id
    | exception Sys_error _ -> None
  in
  let rec stated i =
    Sys.file_exists (Printf.sprintf "%s/index%d" sysfs i)
    && (Option.fold ~none:false ~some:(fun n -> n > 0) (sized i)
       || stated (i + 1))
  in
  let states =
    Sys.os_type = "Unix"
    && match os () with "Darwin" -> true | "Linux" -> stated 0 | _ -> false
  in
  match Rig.cache_bytes Rig.host with
  | Some n -> greater int ~msg:"the host's cache" ~than:0 n
  | None -> if states then fail "the host states no cache"

let cache_refused =
  cases ~id:string_of_int "a last-level cache below 1 byte raises"
    [ 0; -1; min_int ]
  @@ fun n ->
  refuses ~sub:"last-level cache" (P.make ~cache_bytes:n ())
    (Printf.sprintf "open:cache %d" n)

(* [capability] *)

let test_capability () =
  let key : string list Type.Id.t = Type.Id.make () in
  let other : int Type.Id.t = Type.Id.make () in
  let cap = [ "add" ] in
  let p = P.make ~capability:(Rig_edge.Capability (key, cap)) () in
  let d = opened (Rig.open_ (P.opener "open:capability" p)) in
  satisfies ~claim:"the record declared"
    (option (list string))
    (function Some c -> c == cap | None -> false)
    (Rig.capability d key);
  equal ~msg:"another key" (option int) None (Rig.capability d other)

(* [open_host] *)

let test_one_host () =
  let h = opened (open_host "one" "A") in
  raises_match ~msg:"another name" Exn.invalid_arg (fun () ->
      open_host "one" "B");
  equal ~msg:"the same name" device h (opened (open_host "one" "A"))

(* The machine's host [A] parks in its [make] until the open of [B] has
   answered. *)
let test_host_opening () =
  let make ~word =
    Support.park ();
    Ok (P.bind (P.make ()) ~word)
  in
  let first = ref (Error "not returned") in
  let opener =
    Thread.create
      (fun () ->
        first :=
          Rig.open_host ~machine:"opening"
            { name = "A"; driver = (module P); make })
      ()
  in
  Fun.protect ~finally:Support.unpark (fun () ->
      Support.await "the first host's opener" Support.parked;
      raises_match (Exn.invalid_arg ~sub:"opening") (fun () ->
          open_host "opening" "B"));
  Thread.join opener;
  equal string "A@opening" (Rig.name (opened !first))

(* A machine has one host for its life: once it is closed or lost, [open_host]
   of the machine answers [Error], under its name or another. *)
let test_host_for_life () =
  let closed = opened (open_host "life-closed" "CPU") in
  Rig.close closed;
  let lost = P.make () in
  let host =
    opened (Rig.open_host ~machine:"life-lost" (P.opener "CPU" lost))
  in
  P.fault lost "unplugged";
  (try ignore (B.create host 8) with Rig.Lost _ -> ());
  Rig.close host;
  List.iter
    (fun (machine, name) ->
      is_error ~pp:Rig.pp ~msg:(machine ^ " " ^ name) (open_host machine name))
    [
      ("life-closed", "CPU");
      ("life-closed", "OTHER");
      ("life-lost", "CPU");
      ("life-lost", "OTHER");
    ]

let test_host_runs () =
  let far = opened (open_host "runs" "HOST") in
  Rig.Point.wait (submit (empty far));
  equal ~msg:"its signaled value" int (Rig.submitted far) (Rig.signaled far)

(* [host_of] *)

let test_host_of () =
  let far = opened (open_host "host-of" "HOST") in
  let beside machine name = opened (Rig.open_ ~machine (polled name)) in
  let g = beside far "open:host-of-gpu" in
  equal
    (list (pair string device))
    [
      ("the host", Rig.host);
      ("a software device", Rig.host);
      ("a driver's device", Rig.host);
      ("an io device", Rig.host);
      ("a device beside the host", Rig.host);
      ("another machine's host", far);
      ("a device beside it", far);
      ("a device beside that device", far);
    ]
    (answers Rig.host_of
       [
         ("the host", Rig.host);
         ("a software device", software "open:host-of-software");
         ("a driver's device", fst (P.open_ "open:host-of-polled"));
         ("an io device", io "open:host-of-io");
         ("a device beside the host", beside Rig.host "open:host-of-here");
         ("another machine's host", far);
         ("a device beside it", g);
         ("a device beside that device", beside g "open:host-of-beside");
       ])

(* The machine's devices run work as before. *)
let test_host_loss_spares_devices () =
  let lost = P.make () in
  let far = opened (Rig.open_host ~machine:"host-loss" (P.opener "CPU" lost)) in
  let g = opened (Rig.open_ ~machine:far (polled "open:host-loss-gpu")) in
  P.fault lost "unplugged";
  (try ignore (B.create far 8) with Rig.Lost _ -> ());
  equal (option string) ~msg:"the host" (Some "unplugged") (Rig.lost far);
  equal (option string) ~msg:"its device" None (Rig.lost g);
  Rig.Point.wait (submit (empty g));
  equal (option string) ~msg:"after its work" None (Rig.lost g)

(* [reaches] *)

(* [reach pairs] is [(what, reaches d d')] for each [(what, d, d')]. *)
let reach pairs = List.map (fun (what, d, d') -> (what, Rig.reaches d d')) pairs
let reachable = list (pair string bool)

let test_reaches_own () =
  let far = opened (open_host "reaches-own" "HOST") in
  let g = opened (Rig.open_ ~machine:far (polled "open:reaches-own-far")) in
  let sw = software "open:reaches-own-software" in
  let d, _ = P.open_ "open:reaches-own" in
  let hidden, _ = P.open_ ~host_visible:false "open:reaches-own-hidden" in
  let io = io "open:reaches-own-io" in
  equal reachable
    (List.map
       (fun w -> (w, true))
       [
         "the host";
         "a software device";
         "a driver's device";
         "a driver's device the host does not address";
         "an io device";
         "another machine's host";
         "another machine's device";
       ])
    (reach
       [
         ("the host", Rig.host, Rig.host);
         ("a software device", sw, sw);
         ("a driver's device", d, d);
         ("a driver's device the host does not address", hidden, hidden);
         ("an io device", io, io);
         ("another machine's host", far, far);
         ("another machine's device", g, g);
       ])

(* Polled states [host_addresses] with [host_visible] and without [transport],
   whatever its queues. *)
let test_reaches_from_host () =
  let sw = software "open:from-host-software" in
  let d, _ = P.open_ "open:from-host" in
  let h, _ = P.open_ ~host_visible:false "open:from-host-hidden" in
  let g, _ =
    P.open_ ~copies:false ~host_visible:false "open:from-host-no-copy"
  in
  let t, _ = P.open_ ~transport:true "open:from-host-transport" in
  equal reachable
    [
      ("a software device", true);
      ("addressed", true);
      ("not addressed", false);
      ("not addressed, no copy queue", false);
      ("behind a transport", false);
    ]
    (reach
       [
         ("a software device", Rig.host, sw);
         ("addressed", Rig.host, d);
         ("not addressed", Rig.host, h);
         ("not addressed, no copy queue", Rig.host, g);
         ("behind a transport", Rig.host, t);
       ])

let test_reaches_maps_host () =
  let sw = software "open:maps-host-software" in
  let sw' = software "open:maps-host-software-2" in
  let d, _ = P.open_ "open:maps-host" in
  let h, _ = P.open_ ~host_visible:false "open:maps-host-hidden" in
  let n, _ = P.open_ ~maps_host:false "open:maps-host-none" in
  equal reachable
    [
      ("a software device, the host", true);
      ("a software device, another", true);
      ("a mapping driver's device, the host", true);
      ("a mapping driver's device, a software device", true);
      ("one the host does not address, the host", true);
      ("one the host does not address, a software device", true);
      ("no maps_host, the host", false);
      ("no maps_host, a software device", false);
    ]
    (reach
       [
         ("a software device, the host", sw, Rig.host);
         ("a software device, another", sw, sw');
         ("a mapping driver's device, the host", d, Rig.host);
         ("a mapping driver's device, a software device", d, sw);
         ("one the host does not address, the host", h, Rig.host);
         ("one the host does not address, a software device", h, sw);
         ("no maps_host, the host", n, Rig.host);
         ("no maps_host, a software device", n, sw);
       ])

let test_reaches_peers () =
  let d, _ = P.open_ "open:peers" in
  let e, _ = P.open_ "open:peers-other" in
  let f, _ = P.open_ ~peers:false "open:peers-alone" in
  equal reachable
    [ ("a peer it maps", true); ("one that maps no peer", false) ]
    (reach [ ("a peer it maps", d, e); ("one that maps no peer", f, d) ])

let test_reaches_machines () =
  let far = opened (open_host "reaches-far" "HOST") in
  let g = opened (Rig.open_ ~machine:far (polled "open:reaches-far-gpu")) in
  let sw = software "open:reaches-far-software" in
  (* A device of [g]'s driver here, which maps its peers: the peer rule alone
     would let them reach each other. *)
  let peer, _ = P.open_ "open:reaches-far-peer" in
  equal reachable
    (List.map
       (fun w -> (w, false))
       [
         "its device, a peer of its driver here";
         "a peer of its driver here, its device";
         "its device, the host";
         "the host, its device";
         "its host, the host";
         "the host, its host";
         "its device, a software device";
         "a software device, its device";
       ])
    (reach
       [
         ("its device, a peer of its driver here", g, peer);
         ("a peer of its driver here, its device", peer, g);
         ("its device, the host", g, Rig.host);
         ("the host, its device", Rig.host, g);
         ("its host, the host", far, Rig.host);
         ("the host, its host", Rig.host, far);
         ("its device, a software device", g, sw);
         ("a software device, its device", sw, g);
       ])

let test_reaches_io () =
  let io = io "open:reaches-io" in
  let sw = software "open:reaches-io-software" in
  let d, _ = P.open_ "open:reaches-io-polled" in
  equal reachable
    (List.map
       (fun w -> (w, false))
       [
         "io, the host";
         "the host, io";
         "io, a software device";
         "a software device, io";
         "io, a driver's device";
         "a driver's device, io";
       ])
    (reach
       [
         ("io, the host", io, Rig.host);
         ("the host, io", Rig.host, io);
         ("io, a software device", io, sw);
         ("a software device, io", sw, io);
         ("io, a driver's device", io, d);
         ("a driver's device, io", d, io);
       ])

(* Where the host reaches a device's memory, every buffer of it has a host
   address, of each memory it has and in each of Polled's modes, and on a
   software device. It has no Pinned memory. *)
let test_reach_addresses () =
  let n = ref 0 in
  let check ~msg d =
    if Rig.reaches Rig.host d then begin
      List.iter
        (fun memory ->
          let b = B.create ~memory d 64 in
          not_equal ~msg int 0 (Support.Reader.host b))
        B.[ Device; Mapped ];
      raises_match ~msg Exn.invalid_arg (fun () -> B.create ~memory:Pinned d 64)
    end
  in
  let bools = [ false; true ] in
  List.iter
    (fun copies ->
      List.iter
        (fun host_visible ->
          List.iter
            (fun transport ->
              incr n;
              let name = Printf.sprintf "open:reach-addresses-%d" !n in
              let d, _ = P.open_ ~copies ~host_visible ~transport name in
              let msg =
                Printf.sprintf "copies %b, host_visible %b, transport %b" copies
                  host_visible transport
              in
              check ~msg d;
              Rig.close d)
            bools)
        bools)
    bools;
  check ~msg:"a software device" (software "open:reach-addresses-software")

(* A region without the host address the facts promise goes back to the driver
   and faults the device. *)
let test_overstated () =
  let p = P.make ~host_visible:false () in
  let d =
    opened
      (Rig.open_
         {
           name = "open:overstated";
           driver = (module Overstated);
           make = (fun ~word -> Ok (P.bind p ~word));
         })
  in
  raises_match ~msg:"the allocation" (lost d) (fun () -> B.create d 64);
  satisfies ~claim:"lost" (option string) Option.is_some (Rig.lost d);
  equal ~msg:"the region freed" (list string) [ "alloc"; "free" ]
    (List.filter (fun c -> c = "alloc" || c = "free") (P.log p))

let () =
  exit
    (run "rig.open"
       [
         group "open_"
           [
             test "opens of one name answer one device" test_same_name;
             other_driver;
             test "an opener's error is the open's, and leaves the name free"
               test_failed_open;
             copy_engine_refused;
             hang_refused;
             counters_refused;
             edge_refused;
             test
               "an exception at any allocation of an open stops the handle it \
                was handed, the name free or open"
               test_raise_anywhere;
             test "an open while a lost device's stop runs is Error"
               test_reopen_early;
             test
               "an open of a lost device's name runs its stop and makes a new \
                device"
               test_reopen;
             test
               "an open whose owed stop raises raises it, and the next open \
                makes a new device"
               test_reopen_owed;
             test
               "an open on a machine whose host ended is Error, as the host's \
                Lost prints"
               test_open_on_ended_host;
             test "a blocked opener holds back opens of its name alone"
               test_blocked_opener;
             test "an open of its own name from its make raises"
               test_opener_opens_itself;
             test "an open waiting for its name's open takes Sys.Break"
               test_break_waiting_open;
             test "a signal handler in an open's wait for its name calls rig"
               test_handled_in_open_wait;
             test
               "an open an exception ended wakes the opens waiting for its name"
               test_ended_wakes_waiter;
           ];
         group "open_io"
           [
             test "two opens of one name answer one device" test_io_name;
             test "an open of a name a driver's device holds raises"
               test_io_driver_name;
             test "an open from a make of its name's open raises"
               test_io_inside_open;
             test "an io device makes no memory" test_io_no_memory;
             test "an io device's budget is max_int, its capacity None"
               test_io_budget;
             test
               "an exception at any allocation of an io open leaves the name \
                free or open"
               test_io_raise_anywhere;
           ];
         group "software"
           [
             test "an open of a name a driver's device holds raises"
               test_software_driver_name;
           ];
         group "close"
           [
             test
               "a close waits for the work submitted before it, then loses the \
                device \"closed\""
               test_close;
             test
               "two domains closing one device stop it once, and both return \
                after it (sampled for 50 ms)"
               test_two_closes;
             test "a fault during a close's wait is the device's loss"
               test_close_fault;
             test "a close of the host raises" test_close_host;
             test "an io device's close ends it at once" test_close_io;
             test "a close of a machine's host closes its devices"
               test_close_machine;
             test "a close of a lost machine's host closes its devices"
               test_close_lost_host;
             test
               "a close of a machine's host closes a device that was still \
                opening"
               test_close_opening;
             test
               "a Ctrl-C in a close's wait for an open leaves the machine open"
               test_break_waiting_close;
             test "a Ctrl-C in a second close leaves the first close's mark"
               test_break_second_close;
             test
               "a close of a machine's host ends an open in flight that an \
                exception ended"
               test_close_ends_interrupted_open;
             test
               "a close of a machine's host returns after an open its opener's \
                exception ended"
               test_close_after_raised_open;
             test
               "a close of a machine's host from a make of an open on it raises"
               test_close_inside_open;
           ];
         group "properties"
           [
             test
               "a closed device answers its properties and timeline as it did"
               test_ended;
           ];
         group "name"
           [
             test "the host is named CPU" test_host_name;
             test
               "a device is named by its opener, NAME@MACHINE on another \
                machine"
               test_name;
           ];
         group "equal"
           [ test "a reopened name is another device" test_reopened ];
         group "pp" [ test "pp formats a device's name" test_pp ];
         group "arch"
           [
             test "a driver's device's arch is the processor its driver names"
               test_driver_arch;
           ];
         group "instruction_set"
           [
             instruction_sets;
             test "the host's instruction set is one rig.mli names, or \"\""
               test_host_isa;
           ];
         group "bus"
           [
             test
               "bus is the driver's address, None for the host, software and \
                io devices"
               test_bus;
             test
               "a device of another machine has no bus, whatever its driver \
                states"
               test_remote_bus;
           ];
         group "computes"
           [ test "every device but an io device computes" test_computes ];
         group "runs_on_host"
           [
             test "the host and a software device alone run on the host"
               test_runs_on_host;
           ];
         group "offers"
           [
             test "each device offers the kinds rig.mli lists" test_offers;
             test "create makes a word of a kind iff the device offers it"
               test_offers_create;
             test "a lost device offers what it offered at its open"
               test_offers_lost;
             no_kind_empty;
           ];
         group "alignment"
           [
             test "a device's alignment is its stated one" test_alignment;
             test "a non-empty buffer starts on its device's alignment"
               test_aligned;
             alignment_refused;
             test "a region off its device's alignment faults the device"
               test_misaligned;
           ];
         group "granules"
           [
             test "a device's units are its driver's, a word's elsewhere"
               test_granules;
             granules_refused;
           ];
         group "cache_bytes"
           [
             test "a device states its driver's last-level cache"
               test_cache_bytes;
             test "the host states a cache where its system states one"
               test_host_cache;
             cache_refused;
           ];
         group "capability"
           [
             test
               "a driver's capability is the record it declared, None under \
                another key"
               test_capability;
           ];
         group "open_host"
           [
             test "a machine whose host is open under one name refuses another"
               test_one_host;
             test "a machine whose host is still opening refuses another name"
               test_host_opening;
             test "a machine has one host for its life" test_host_for_life;
             test "another machine's host runs submissions" test_host_runs;
           ];
         group "host_of"
           [
             test "a device's host is its machine's host" test_host_of;
             test "the loss of a machine's host loses none of its devices"
               test_host_loss_spares_devices;
           ];
         group "reaches"
           [
             test "a device reaches its own memory" test_reaches_own;
             test
               "the host reaches a software device and a device whose memory \
                it addresses"
               test_reaches_from_host;
             test
               "a device that maps the host reaches its memory and every \
                software device's"
               test_reaches_maps_host;
             test "a driver's device reaches a peer it maps" test_reaches_peers;
             test "devices of different machines never reach each other"
               test_reaches_machines;
             test "an io device and any other never reach each other"
               test_reaches_io;
             test "a device the host reaches has buffers with host addresses"
               test_reach_addresses;
             test "a region against the host_addresses fact faults the device"
               test_overstated;
           ];
       ])
