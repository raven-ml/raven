(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module P = Rig_support.Polled
module Support = Rig_support

let submit ?(buffers = [||]) ?waits s =
  Rig.submit ?waits s ~run:(Rig.Submission.Run.make ()) ~buffers

let timeout = 60.
let device = Testable.make ~pp:Rig.pp ~equal:Rig.equal
let memory name = Rig.memory_device name

let test_same_name () =
  let d = memory "open:same" in
  equal device d (memory "open:same");
  not_equal device d (memory "open:other")

let test_other_driver () =
  ignore (memory "open:taken");
  raises_match Exn.invalid_arg (fun () -> P.open_ "open:taken")

let test_facts () =
  let d = memory "open:facts" in
  let p, _ = P.open_ "open:facts-polled" in
  equal string "open:facts" (Rig.name d);
  equal bool true (Rig.computes d);
  equal bool true (Rig.runs_on_host d);
  equal bool false (Rig.runs_on_host p);
  equal bool true (Rig.shares_host_memory d);
  equal bool true (Rig.reaches d Rig.host);
  equal bool true (Rig.reaches Rig.host d);
  equal device Rig.host (Rig.host_of d);
  is_some (Rig.capability p P.capability_key)

let test_host () =
  equal string "CPU" (Rig.name Rig.host);
  mem string (Rig.arch Rig.host) [ "arm64"; "x86_64" ];
  equal bool true (Rig.computes Rig.host);
  equal bool true (Rig.runs_on_host Rig.host);
  equal bool true (Rig.shares_host_memory Rig.host);
  equal device Rig.host (Rig.host_of Rig.host)

(* What a driver's device reaches follows its facts: the host reaches its memory
   where the host addresses its regions, whatever its queues, and it reaches the
   host's and its peers' where it maps them. *)
let test_reach () =
  let d, _ = P.open_ "open:reach" in
  let e, _ = P.open_ "open:reach-peer" in
  let f, _ = P.open_ ~peers:false "open:reach-alone" in
  let h, _ = P.open_ ~host_visible:false "open:reach-hidden" in
  let g, _ = P.open_ ~copies:false ~host_visible:false "open:reach-no-copy" in
  let reach d =
    [ Rig.reaches d Rig.host; Rig.reaches Rig.host d; Rig.shares_host_memory d ]
  in
  equal string "polled" (Rig.arch d);
  equal ~msg:"a copy queue, memory the host addresses" (list bool)
    [ true; true; true ] (reach d);
  equal ~msg:"a copy queue, memory the host does not address" (list bool)
    [ true; false; false ] (reach h);
  equal ~msg:"no copy queue, memory the host does not address" (list bool)
    [ true; false; false ] (reach g);
  equal (list bool) [ true; false ] [ Rig.reaches d e; Rig.reaches f d ]

(* Where the host reaches a device's memory, every buffer of it has a host
   address, of each memory and in each of Polled's modes, and on a memory
   device. *)
let test_reach_addresses () =
  let n = ref 0 in
  let check ~msg d =
    if Rig.reaches Rig.host d then
      List.iter
        (fun memory ->
          let b = Rig.Buffer.create ~memory d 64 in
          not_equal ~msg int 0 (Support.Reader.host b))
        Rig.Buffer.[ Device; Pinned; Mapped ]
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
  check ~msg:"a memory device" (memory "open:reach-addresses-memory")

(* An io device whose memory is bytes that nothing reads, as host pages. *)
module Store = struct
  type t = unit

  type region =
    (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

  exception Fault of string

  let region_key : region Type.Id.t = Type.Id.make ()
  let budget () = max_int

  let alloc () n =
    Some (Bigarray.Array1.create Bigarray.char Bigarray.c_layout n)

  let free () _ = ()
  let read () _ ~at:_ ~dst:_ ~len:_ = ()
  let write () _ ~at:_ ~src:_ ~len:_ = ()
  let pages () r = Some r
  let prefetch () _ ~at:_ ~len:_ = ()
  let stop () = ()
end

(* Another io library, of the same shape. *)
module Other = struct
  include Store

  let region_key : region Type.Id.t = Type.Id.make ()
end

(* Polled whose queues are named ["Q0"], ["Q1"]. *)
module Renamed = struct
  include P

  let facts d =
    let f = P.facts d in
    let name i (q : Rig_edge.queue) = { q with name = Printf.sprintf "Q%d" i } in
    { f with queues = List.mapi name f.queues }
end

(* Polled whose queue named "COPY:0" runs no copy, and which has no copy
   engine. *)
module Copyless = struct
  include P

  let facts d =
    let f = P.facts d in
    let fills name = { Rig_edge.name; runs = [ Fill ] } in
    let queues = [ fills "COMPUTE:0"; fills "COPY:0" ] in
    { f with queues; copy_engine = None }
end

(* Polled with its first queue alone, which runs fills, copies and launches,
   and is its copy engine. *)
module One_queue = struct
  include P

  let facts d =
    let f = P.facts d in
    { f with queues = [ List.hd f.queues ]; copy_engine = Some 0 }
end

(* The driver [B] whose copy engine is [engine]. *)
let engine (module B : Rig_edge.Driver with type t = P.t) engine =
  let module D = struct
    include B

    let facts d = { (B.facts d) with copy_engine = engine }
  end in
  (module D : Rig_edge.Driver with type t = P.t)

(* Polled that states the host addresses its memory, which it does not. *)
module Overstated = struct
  include P

  let facts d = { (P.facts d) with host_addresses = true }
end

let open_as (module D : Rig_edge.Driver with type t = P.t) name p =
  require_ok ~pp:Format.pp_print_string
    (Rig.open_ { name; driver = (module D); make = (fun () -> Ok p) })

(* Bytes into memory of [d] the host does not address and back, which [d]'s
   copy queue runs: the copies [p] ran and the bytes back. *)
let round_trip d p =
  let n = 1 lsl 16 in
  let src = Rig.Buffer.create Rig.host n in
  Rig.Buffer.write src Bigarray.char (fun a -> Bigarray.Array1.fill a 'q');
  let dst = Rig.Buffer.create d n in
  let back = Rig.Buffer.create Rig.host n in
  Rig.Buffer.copy ~src ~dst;
  Rig.Buffer.copy ~src:dst ~dst:back;
  let got =
    Rig.Buffer.read back Bigarray.char (fun a ->
        String.init n (Bigarray.Array1.get a))
  in
  (P.submits p, got = String.make n 'q')

(* rig's copies go to the copy engine the facts name, whatever the queues'
   names and order: a queue named "COPY:0" may run none, and a device without
   a copy engine takes no copy part. *)
let test_copy_engine () =
  let p = P.make ~host_visible:false () in
  let d = open_as (module Renamed) "open:renamed" p in
  equal ~msg:"renamed: copies run, bytes back" (pair int bool) (2, true)
    (round_trip d p);
  let p = P.make ~host_visible:false () in
  let d = open_as (module One_queue) "open:one-queue" p in
  equal ~msg:"one queue: copies run, bytes back" (pair int bool) (2, true)
    (round_trip d p);
  let p = P.make ~host_visible:false () in
  let d = open_as (engine (module P) (Some 0)) "open:first-engine" p in
  let (), events = Rig.Profile.take (fun () -> ignore (round_trip d p)) in
  let lanes =
    List.filter_map
      (function
        | Rig.Profile.Span s when Rig.equal s.device d -> Some s.lane
        | _ -> None)
      events
  in
  equal ~msg:"the first queue runs the copies" (list string)
    [ "COMPUTE:0"; "COMPUTE:0" ] lanes;
  let c = open_as (module Copyless) "open:copyless" (P.make ()) in
  let src = Rig.Buffer.create c 8 and dst = Rig.Buffer.create c 8 in
  raises_match (Exn.invalid_arg ~substring:"runs no copies") (fun () ->
      Rig.Submission.(make c [| copy ~src:(Buffer src) ~dst:(Buffer dst) |]))

(* A copy engine that names no queue, or a queue that runs no copy, refuses
   the open, and the driver's device is stopped. *)
let test_copy_engine_refused () =
  let refuses name driver =
    let p = P.make () in
    raises_match ~msg:name (Exn.invalid_arg ~substring:"copy engine") (fun () ->
        Rig.open_ { name; driver; make = (fun () -> Ok p) });
    equal ~msg:(name ^ ": stopped") bool true (List.mem "stop" (P.log p))
  in
  refuses "open:engine-past" (engine (module P) (Some 2));
  refuses "open:engine-negative" (engine (module P) (Some (-1)));
  refuses "open:engine-fills" (engine (module Copyless) (Some 1))

(* A region without the host address the facts promise refuses the
   allocation, and goes back to the driver. *)
let test_overstated () =
  let p = P.make ~host_visible:false () in
  let d = open_as (module Overstated) "open:overstated" p in
  raises_match (Exn.invalid_arg ~substring:"host address") (fun () ->
      Rig.Buffer.create d 64);
  equal ~msg:"the region freed" (list string) [ "alloc"; "free" ]
    (List.filter (fun c -> c = "alloc" || c = "free") (P.log p))

let test_hang_refused () =
  List.iteri
    (fun i n ->
      let p = P.make ~hang_ms:n () in
      let name = Printf.sprintf "open:hang-%d" i in
      raises_match ~msg:(Printf.sprintf "hang_ms %d" n)
        (Exn.invalid_arg ~substring:"hang bound") (fun () ->
          Rig.open_ { name; driver = (module P); make = (fun () -> Ok p) });
      equal ~msg:"stopped" bool true (List.mem "stop" (P.log p)))
    [ 0; -1; min_int ]

(* Polled whose facts name no C state. *)
module Edgeless = struct
  include P

  let facts d = { (P.facts d) with edge = 0n }
end

(* Zeroed bytes, whose first member is NULL as a C state's. *)
let zeroed =
  lazy
    (let b = Rig.Buffer.create Rig.host 64 in
     Rig.Buffer.fill b '\000';
     b)

module Driverless = struct
  include P

  let facts d =
    let at = Support.Reader.host (Lazy.force zeroed) in
    { (P.facts d) with edge = Nativeint.of_int at }
end

(* An edge that names no C state, or a state whose first member is NULL,
   refuses the open, the driver's device is stopped, and the name stays
   free. *)
let test_edge_refused () =
  List.iter
    (fun (name, driver) ->
      let p = P.make () in
      raises_match ~msg:name (Exn.invalid_arg ~substring:"rig_driver")
        (fun () -> Rig.open_ { name; driver; make = (fun () -> Ok p) });
      equal ~msg:("stopped, " ^ name) bool true (List.mem "stop" (P.log p));
      ignore (open_as (module P) name (P.make ())))
    [
      ("open:edgeless", (module Edgeless : Rig_edge.Driver with type t = P.t));
      ("open:driverless", (module Driverless));
    ]

let open_store name =
  require_ok ~pp:Format.pp_print_string
    (Rig.open_io { name; io = (module Store); make = (fun () -> Ok ()) })

(* Opens [machine]'s host [name], a driver's device. *)
let open_host machine name =
  Rig.open_host ~machine
    { name; driver = (module P); make = (fun () -> Ok (P.make ())) }

let test_io () =
  let io = open_store "open:io" in
  equal string "" (Rig.arch io);
  equal bool false (Rig.computes io);
  equal (list bool) [ false; false ]
    [ Rig.reaches io Rig.host; Rig.reaches Rig.host io ];
  equal int 8 (Rig.Buffer.length (Rig.Buffer.create io 8))

(* A name an io library opened stays that library's: another's open of it
   raises. *)
let test_io_key () =
  ignore (open_store "open:io-key");
  raises_match Exn.invalid_arg (fun () ->
      Rig.open_io
        { name = "open:io-key"; io = (module Other); make = (fun () -> Ok ()) })

(* A region an io library gave is a buffer of its device, which gives it back by
   the library's key alone, and borrows on the host through its pages. *)
let test_of_io () =
  let io = open_store "open:of-io" in
  let r = Bigarray.Array1.create Bigarray.char Bigarray.c_layout 16 in
  Bigarray.Array1.fill r 'p';
  let b = Rig.Buffer.of_io io Store.region_key r ~access:Read_write 16 in
  equal ~msg:"the region given" bool true
    (match Rig.Buffer.io b Store.region_key with
    | Some r' -> r' == r
    | None -> false);
  is_none (Rig.Buffer.io b Other.region_key);
  raises_match Exn.invalid_arg (fun () ->
      Rig.Buffer.of_io io Other.region_key r ~access:Read_write 16);
  let dead = Rig.Buffer.of_io io Store.region_key r ~access:Read_write 16 in
  Rig.Claim.with_ ~read:[ dead ] ~donate:[] (fun c ->
      ignore (Rig.Claim.consume c ~why:"consumed" dead));
  raises_match (Exn.invalid_arg ~substring:"consumed") (fun () ->
      Rig.Buffer.io dead Store.region_key);
  let h = require_some (Rig.Buffer.borrow Rig.host b) in
  equal string (String.make 16 'p')
    (Rig.Buffer.read h Bigarray.char (fun a ->
         String.init 16 (Bigarray.Array1.get a)))

(* A device of another machine is named after it, and its host is the device
   opened as that machine's, which runs work and loads code. A device opened
   beside any device of the machine is of that machine. *)
let test_machine () =
  let far = require_ok ~pp:Format.pp_print_string (open_host "far" "HOST") in
  let beside machine name =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_ ~machine
         { name; driver = (module P); make = (fun () -> Ok (P.make ())) })
  in
  let g = beside far "open:gpu" in
  let g' = beside g "open:gpu-beside" in
  equal string "open:gpu@far" (Rig.name g);
  equal device far (Rig.host_of g);
  equal ~msg:"beside a device" device far (Rig.host_of g');
  equal ~msg:"beside this machine's host" device Rig.host
    (Rig.host_of (beside Rig.host "open:here"));
  equal device far (Rig.host_of far);
  equal ~msg:"shares this process's host memory" (list bool) [ false; false ]
    [ Rig.shares_host_memory g; Rig.shares_host_memory far ];
  equal (list bool) [ false; false ]
    [ Rig.reaches g Rig.host; Rig.reaches Rig.host g ];
  equal ~msg:"computes" bool true (Rig.computes far);
  ignore (submit (Rig.Submission.make far [||]))

(* A machine has one host: [~host] names another machine's, and a second host of
   a machine whose host is open under another name is refused. *)
let test_one_host () =
  let h = require_ok ~pp:Format.pp_print_string (open_host "one" "A") in
  raises_match ~msg:"a second" Exn.invalid_arg (fun () -> open_host "one" "B");
  equal ~msg:"the same name" device h
    (require_ok ~pp:Format.pp_print_string (open_host "one" "A"))

(* A machine has one host for its life: once it is closed or lost, [open_host]
   of the machine answers [Error], under its name or another. *)
let test_host_for_life () =
  let closed =
    require_ok ~pp:Format.pp_print_string (open_host "life-closed" "CPU")
  in
  Rig.close closed;
  let lost = P.make () in
  let host =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_host ~machine:"life-lost"
         { name = "CPU"; driver = (module P); make = (fun () -> Ok lost) })
  in
  P.fault lost "unplugged";
  (try ignore (Rig.Buffer.create host 8) with Rig.Lost _ -> ());
  Rig.close host;
  List.iter
    (fun (machine, name) ->
      is_error
        ~pp:(fun ppf d -> Format.pp_print_string ppf (Rig.name d))
        ~msg:(machine ^ " " ^ name)
        (open_host machine name))
    [
      ("life-closed", "CPU");
      ("life-closed", "OTHER");
      ("life-lost", "CPU");
      ("life-lost", "OTHER");
    ]

(* Closing another machine's host closes every device of its machine too. *)
let test_close_machine () =
  let far = require_ok ~pp:Format.pp_print_string (open_host "closing" "CPU") in
  let g =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_ ~machine:far
         {
           name = "open:closing-gpu";
           driver = (module P);
           make = (fun () -> Ok (P.make ()));
         })
  in
  Rig.close far;
  equal
    (list (option string))
    [ Some "closed"; Some "closed" ]
    [ Rig.lost g; Rig.lost far ]

(* Once a machine's host is closed or lost, its devices open no more: the open
   answers the host's loss. *)
let test_open_on_ended_host () =
  let gpu machine =
    Rig.open_ ~machine
      {
        name = "open:ended-gpu";
        driver = (module P);
        make = (fun () -> Ok (P.make ()));
      }
  in
  let closed =
    require_ok ~pp:Format.pp_print_string (open_host "ended-closed" "CPU")
  in
  Rig.close closed;
  let lost = P.make () in
  let host =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_host ~machine:"ended-lost"
         { name = "CPU"; driver = (module P); make = (fun () -> Ok lost) })
  in
  P.fault lost "unplugged";
  (try ignore (Rig.Buffer.create host 8) with Rig.Lost _ -> ());
  equal
    (list (result device string))
    [
      Error "CPU@ended-closed lost: closed";
      Error "CPU@ended-lost lost: unplugged";
    ]
    [ gpu closed; gpu host ]

(* The loss of another machine's host loses none of its devices: they run work
   as before. *)
let test_host_loss_spares_devices () =
  let lost = P.make () in
  let far =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_host ~machine:"host-loss"
         { name = "CPU"; driver = (module P); make = (fun () -> Ok lost) })
  in
  let g =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_ ~machine:far
         {
           name = "open:host-loss-gpu";
           driver = (module P);
           make = (fun () -> Ok (P.make ()));
         })
  in
  P.fault lost "unplugged";
  (try ignore (Rig.Buffer.create far 8) with Rig.Lost _ -> ());
  equal (option string) ~msg:"the host" (Some "unplugged") (Rig.lost far);
  equal (option string) ~msg:"its device" None (Rig.lost g);
  Rig.Point.wait (submit (Rig.Submission.make g [||]));
  equal (option string) ~msg:"after its work" None (Rig.lost g)

(* Closing a lost host closes its machine's devices too. *)
let test_close_lost_host () =
  let lost = P.make () in
  let far =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_host ~machine:"closing-lost"
         { name = "CPU"; driver = (module P); make = (fun () -> Ok lost) })
  in
  let g =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_ ~machine:far
         {
           name = "open:closing-lost-gpu";
           driver = (module P);
           make = (fun () -> Ok (P.make ()));
         })
  in
  P.fault lost "unplugged";
  (try ignore (Rig.Buffer.create far 8) with Rig.Lost _ -> ());
  Rig.close far;
  equal (option string) (Some "closed") (Rig.lost g)

(* A fault while the device's facts are read is the open's error, and the
   driver's handle is stopped: nothing it opened stays. *)
let test_fault_at_open () =
  let p = P.make () in
  P.fault p "no device";
  equal (result device string) (Error "open:faulted: no device")
    (Rig.open_
       { name = "open:faulted"; driver = (module P); make = (fun () -> Ok p) });
  equal (list string) [ "stop" ] (P.log p)

let test_point () =
  let d = memory "open:point" in
  let p = submit (Rig.Submission.make d [||]) in
  equal string "open:point:1" (Format.asprintf "%a" Rig.Point.pp p)

(* An opener's error is the open's, and leaves the name free. *)
let test_failed_open () =
  let fails () = Error "no hardware" in
  equal (result device string) (Error "no hardware")
    (Rig.open_ { name = "open:failed"; driver = (module P); make = fails });
  let d, _ = P.open_ "open:failed" in
  equal string "open:failed" (Rig.name d)

(* An exception the opener raises is raised again, the name left unopened. *)
let test_opener_raises () =
  raises Exit (fun () ->
      Rig.open_
        {
          name = "open:raises";
          driver = (module P);
          make = (fun () -> raise Exit);
        });
  let d, _ = P.open_ "open:raises" in
  equal string "open:raises" (Rig.name d)

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

(* An io library whose handle counts its stops. *)
module Stops = struct
  type t = int ref
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
  let stop stops = incr stops
end

(* [raise_anywhere what handle stops opens] opens names with [opens name make]
   over handles of [handle ()], whose stops [stops] counts. *)
let raise_anywhere what handle stops opens () =
  let rec from n =
    let name = Printf.sprintf "open:raise-%s-%d" what n in
    let h = handle () and handed = ref false in
    let make () =
      let r = Ok h in
      handed := true;
      r
    in
    let armed = ref false in
    ignore (Gc.Memprof.start ~sampling_rate:1. (raising_at n armed));
    armed := true;
    let opened =
      match opens name make with
      | r ->
          armed := false;
          Some r
      | exception Injected -> None
    in
    armed := false;
    Gc.Memprof.stop ();
    let msg = Printf.sprintf "raised at allocation %d" n in
    match opened with
    | Some r ->
        ignore (require_ok ~pp:Format.pp_print_string r);
        greater ~msg:"allocations that raised" int ~than:1 n
    | None ->
        (* An open that published its device left it open: the name answers
           it without calling [make]. *)
        let calls = ref 0 in
        let d =
          require_ok ~pp:Format.pp_print_string
            (opens name (fun () ->
                 incr calls;
                 Ok (handle ())))
        in
        equal ~msg (option string) None (Rig.lost d);
        let left_open = !calls = 0 in
        if left_open then equal ~msg:("handed, " ^ msg) bool true !handed;
        let stopped = !handed && not left_open in
        equal ~msg int (if stopped then 1 else 0) (stops h);
        from (n + 1)
  in
  from 1

let test_driver_raise_anywhere =
  raise_anywhere "driver"
    (fun () -> P.make ())
    (fun p -> List.length (List.filter (( = ) "stop") (P.log p)))
    (fun name make -> Rig.open_ { name; driver = (module P); make })

let test_io_raise_anywhere =
  raise_anywhere "io"
    (fun () -> ref 0)
    ( ! )
    (fun name make -> Rig.open_io { name; io = (module Stops); make })

(* An open that waits for another open of its name returns to OCaml at least
   every 200 ms, where a pending signal's handler runs. The other opener parks
   outside OCaml for up to 5 s, and a thread of no OCaml code raises SIGINT
   once the open blocks in its condition wait, so only the waiting open can
   take the signal before the other open ends. The handler raises Sys.Break on
   the waiting thread alone. *)
let test_break_waiting_open () =
  if not Support.sees_waits then
    skip ~reason:"the system shows no thread's blocking call" ();
  let name = "open:break-waiting" in
  let make () =
    Support.park ();
    Ok (P.make ())
  in
  let first = ref None in
  let opener =
    Thread.create
      (fun () -> first := Some (Rig.open_ { name; driver = (module P); make }))
      ()
  in
  Support.await "the first opener" Support.parked;
  let waiting = Thread.id (Thread.self ()) and taken = ref "nowhere" in
  let handle _ =
    if Thread.id (Thread.self ()) = waiting then raise Sys.Break
    else taken := "on another thread"
  in
  let calls = ref 0 in
  let second () =
    incr calls;
    Ok (P.make ())
  in
  let before = Sys.signal Sys.sigint (Sys.Signal_handle handle) in
  let got, raised =
    Fun.protect
      ~finally:(fun () -> Sys.set_signal Sys.sigint before)
      (fun () ->
        Support.interrupt_in_wait (fun () ->
            match Rig.open_ { name; driver = (module P); make = second } with
            | Ok _ -> "opened, the signal taken " ^ !taken
            | Error e -> "Error " ^ e
            | exception Sys.Break -> "Sys.Break"))
  in
  Support.unpark ();
  Thread.join opener;
  equal ~msg:"the open waited" bool true raised;
  equal ~msg:"the waiting open" string "Sys.Break" got;
  equal ~msg:"its opener's calls" int 0 !calls;
  is_ok ~msg:"the other open" (Option.get !first)

(* A lost device's name opens again only once its stop returned: before, the
   open is an [Error]. *)
let test_reopen_early () =
  let d, p = P.open_ "open:early" in
  P.gate p;
  P.fail p;
  let loser =
    Thread.create
      (fun () ->
        try ignore (submit (Rig.Submission.make d [||])) with Rig.Lost _ -> ())
      ()
  in
  Support.await "a stop at the gate" (fun () -> P.sleepers p = 1);
  is_error
    (Rig.open_
       {
         name = "open:early";
         driver = (module P);
         make = (fun () -> Ok (P.make ()));
       });
  P.open_gate p;
  Thread.join loser;
  let d', _ = P.open_ "open:early" in
  not_equal device d d'

let test_reopen () =
  let d, p = P.open_ "open:reopen" in
  let s = Rig.Submission.make d [||] in
  P.fail p;
  raises_match (function Rig.Lost _ -> true | _ -> false) (fun () -> submit s);
  equal (list string) [ "stop" ] (P.log p);
  let d', _ = P.open_ "open:reopen" in
  not_equal device d d';
  equal (option string) None (Rig.lost d')

(* Moves the memory of closed devices back and ends their words: each round
   collects, then drains every device through an allocation. *)
let settle () =
  for _ = 1 to 4 do
    Gc.full_major ();
    ignore (Rig.Buffer.create Rig.host 8 : Rig.Buffer.t)
  done

let lost = function Rig.Lost _ -> true | _ -> false

(* A closed device's record outlives its resources: once its memory returned
   and its word ended, it answers its properties and timeline as it did. *)
let test_ended () =
  let key : string list Type.Id.t = Type.Id.make () in
  let cap = [ "add" ] in
  let make () = Ok (P.make ~capability:(Rig_edge.Capability (key, cap)) ()) in
  let d =
    Result.get_ok
      (Rig.open_ { name = "open:ended"; driver = (module P); make })
  in
  let k = Rig.Local.new_key (fun _ -> ()) in
  Rig.Local.get k d;
  ignore (Rig.Buffer.create d 64 : Rig.Buffer.t);
  let p = submit (Rig.Submission.make d [||]) in
  Rig.Point.wait p;
  let facts d =
    [
      Rig.computes d;
      Rig.runs_on_host d;
      Rig.shares_host_memory d;
      Rig.reaches Rig.host d;
      Rig.reaches d Rig.host;
    ]
  in
  let before = facts d and arch = Rig.arch d in
  Rig.set_budget d 4096;
  Rig.close d;
  settle ();
  equal string "open:ended" (Rig.name d);
  equal string arch (Rig.arch d);
  equal (list bool) before (facts d);
  equal device Rig.host (Rig.host_of d);
  equal bool true
    (match Rig.capability d key with Some c -> c == cap | None -> false);
  equal int 4096 (Rig.budget d);
  equal int 0 (Rig.allocated d);
  equal (option string) (Some "closed") (Rig.lost d);
  equal int 1 (Rig.submitted d);
  equal int 1 (Rig.signaled d);
  equal device d (Rig.Point.device p);
  equal bool true (Rig.Point.reached p);
  equal string "open:ended:1" (Format.asprintf "%a" Rig.Point.pp p);
  raises_match lost (fun () -> Rig.Point.wait p);
  raises_match lost (fun () -> Rig.Buffer.create d 1);
  raises_match lost (fun () -> Rig.Local.get k d);
  raises_match lost (fun () -> submit (Rig.Submission.make d [||]))

(* A point outlives the values of its device: once the program holds none,
   the point names the ended stand-in, lost, which has reached every value. *)
let test_forgotten () =
  let point () =
    let d, _ = P.open_ "open:forgotten" in
    let p = submit (Rig.Submission.make d [||]) in
    Rig.Point.wait p;
    Rig.close d;
    p
  in
  let p = point () in
  settle ();
  let e = Rig.Point.device p in
  equal string "ended" (Rig.name e);
  equal (option string) (Some "ended") (Rig.lost e);
  equal bool false (Rig.computes e);
  equal bool false (Rig.reaches Rig.host e);
  equal int 1 (Rig.Point.value p);
  equal bool true (Rig.Point.reached p);
  equal string "ended:1" (Format.asprintf "%a" Rig.Point.pp p);
  raises_match lost (fun () -> Rig.Point.wait p);
  raises_match lost (fun () -> Rig.Buffer.create e 1)

(* Opens and closes keep nothing once the program holds no value of the
   devices. A window opens and closes [opens] devices, then the rounds of
   [settle] return their memory and end their words. After a warm window,
   the OCaml heap ends each window where it started it. The C heap moves
   either way by what the allocator's per-thread caches hold (glibc's),
   while a leak grows it in every window: the law checks the least growth
   of [windows]. *)
let opens = 1024
let windows = 3

let test_ended_kept () =
  let cycle i =
    let d, _ = P.open_ (Printf.sprintf "open:kept%d" (i mod 2)) in
    ignore (Rig.Buffer.create d 64 : Rig.Buffer.t);
    Rig.Point.wait (submit (Rig.Submission.make d [||]));
    Rig.close d
  in
  let heap () =
    match Support.heap_bytes () with
    | Some n -> n
    | None -> skip ~reason:"the system counts no C heap" ()
  in
  let live () = (Gc.stat ()).live_words * (Sys.word_size / 8) in
  let window () =
    let c = heap () and ml = live () in
    for i = 1 to opens do
      cycle i
    done;
    settle ();
    (heap () - c, live () - ml)
  in
  ignore (window ());
  let grown = List.init windows (fun _ -> window ()) in
  let cs = List.map fst grown in
  at_most int
    ~msg:
      (Printf.sprintf "the C heap's least growth over a window (%s)"
         (String.concat ", " (List.map string_of_int cs)))
    ~than:0
    (List.fold_left Int.min max_int cs);
  List.iter
    (fun (_, ml) ->
      at_most int ~msg:"the OCaml heap's growth over a window" ~than:0 ml)
    grown

(* An open whose opener blocks holds back only opens of its own name. *)
let test_blocked_opener () =
  let lock = Mutex.create () and cond = Condition.create () in
  let inside = ref false and go = ref false and calls = ref 0 in
  let make () =
    Mutex.protect lock (fun () ->
        incr calls;
        inside := true;
        while not !go do
          Condition.wait cond lock
        done);
    Ok (P.make ())
  in
  let opener : P.t Rig_edge.opener =
    { name = "open:slow"; driver = (module P); make }
  in
  let first = ref None and second = ref None in
  let slow =
    Thread.create (fun () -> first := Some (require_ok (Rig.open_ opener))) ()
  in
  Support.await "a running opener" (fun () ->
      Mutex.protect lock (fun () -> !inside));
  let started = Atomic.make false in
  let same =
    Thread.create
      (fun () ->
        Atomic.set started true;
        let another = { opener with make = (fun () -> make ()) } in
        second := Some (require_ok (Rig.open_ another)))
      ()
  in
  Support.await "a concurrent open of the same name" (fun () ->
      Atomic.get started);
  let d, _ = P.open_ "open:fast" in
  equal string "open:fast" (Rig.name d);
  Mutex.protect lock (fun () ->
      go := true;
      Condition.broadcast cond);
  Thread.join slow;
  Thread.join same;
  let d = Option.get !first in
  equal device d (Option.get !second);
  equal int 1 !calls;
  Rig.close d;
  let reopened = require_ok (Rig.open_ opener) in
  not_equal device d reopened;
  equal int 2 !calls;
  Rig.close reopened

let tests =
  group ~timeout "opening"
    [
      test "one name opens one device, until it is lost" test_same_name;
      test "a name open as another driver's device raises" test_other_driver;
      test "a device states its facts" test_facts;
      test "rig's copies go to the copy engine the facts name" test_copy_engine;
      test "a copy engine that names no copying queue refuses the open"
        test_copy_engine_refused;
      test "a region against the host_addresses fact refuses the allocation"
        test_overstated;
      test "a hang bound below 1 ms refuses the open" test_hang_refused;
      test "an edge that names no C state or driver refuses the open"
        test_edge_refused;
      test "the host states its facts" test_host;
      test "a driver's device reaches by its facts and its peers" test_reach;
      test "where the host reaches a device, its buffers have host addresses"
        test_reach_addresses;
      test "an io device computes nothing and reaches nothing" test_io;
      test "a name stays with the io library that opened it" test_io_key;
      test "a region an io library gave is a buffer of its device" test_of_io;
      test "a device of another machine is named after it, its host the io's"
        test_machine;
      test "a machine has one host" test_one_host;
      test "a machine has one host for its life" test_host_for_life;
      test "closing a machine's host closes its devices" test_close_machine;
      test "a machine whose host ended opens no more devices"
        test_open_on_ended_host;
      test "the loss of a machine's host loses none of its devices"
        test_host_loss_spares_devices;
      test "closing a lost host closes its machine's devices"
        test_close_lost_host;
      test "a point prints as its device's name and its value" test_point;
      test "an opener's error leaves the name free" test_failed_open;
      test "an opener's exception is raised, the name left free"
        test_opener_raises;
      test "an exception at any allocation of an open stops what it was handed"
        test_driver_raise_anywhere;
      test "an exception at any allocation of an io open stops its handle"
        test_io_raise_anywhere;
      test "a lost device's name opens again only once its stop returned"
        test_reopen_early;
      test "a fault reading a device's facts stops its handle"
        test_fault_at_open;
      test "a lost device's name opens anew once its stop answered" test_reopen;
      test "a blocked opener holds back no other name" test_blocked_opener;
      test "an open waiting for its name's open takes Sys.Break"
        test_break_waiting_open;
      test "an ended device answers as it did before its end" test_ended;
      test "a point whose device the program dropped names the ended device"
        test_forgotten;
      test "an ended device no value references keeps nothing" test_ended_kept;
    ]

let () = exit (run "rig.open" [ tests ])
