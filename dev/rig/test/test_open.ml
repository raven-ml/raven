(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module P = Rig_support.Polled
module Support = Rig_support

let submit ?(buffers = [||]) ?(waits = [||]) s =
  Rig.submit s ~run:(Rig.Submission.Run.make ()) ~buffers ~waits

let timeout = 60.
let device = Testable.make ~pp:Rig.pp ~equal:Rig.equal
let memory name = require_ok ~pp:Format.pp_print_string (Rig.memory_device name)

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
    let name i (q : Rig.queue) = { q with name = Printf.sprintf "Q%d" i } in
    { f with queues = List.mapi name f.queues }
end

(* Polled whose queue named "COPY:0" runs no copy. *)
module Copyless = struct
  include P

  let facts d =
    let f = P.facts d in
    let fills name = { Rig.name; runs = [ Fill ] } in
    { f with queues = [ fills "COMPUTE:0"; fills "COPY:0" ] }
end

(* Polled with its first queue alone, which runs fills, copies and launches. *)
module One_queue = struct
  include P

  let facts d =
    let f = P.facts d in
    { f with queues = [ List.hd f.queues ] }
end

(* Polled that states the host addresses its memory, which it does not. *)
module Overstated = struct
  include P

  let facts d = { (P.facts d) with host_addresses = true }
end

let open_as (module D : Rig.Driver with type t = P.t) name p =
  require_ok ~pp:Format.pp_print_string
    (Rig.open_ (module D) ~name (fun () -> Ok p))

(* Bytes into memory of [d] the host does not address and back, which [d]'s
   copy queue runs: the copies [p] ran and the bytes back. *)
let round_trip d p =
  let n = 1 lsl 16 in
  let src = Rig.Buffer.create Rig.host n in
  Bigarray.Array1.fill (Rig.Buffer.bigarray Bigarray.char src) 'q';
  let dst = Rig.Buffer.create d n in
  let back = Rig.Buffer.create Rig.host n in
  Rig.Buffer.copy ~src ~dst;
  Rig.Buffer.copy ~src:dst ~dst:back;
  let ba = Rig.Buffer.bigarray Bigarray.char back in
  (P.submits p, String.init n (Bigarray.Array1.get ba) = String.make n 'q')

(* rig's copies go to the first queue after the first that runs copies, or to
   the first where it alone does, whatever the queues' names: a queue named
   "COPY:0" may run none. *)
let test_queue_names () =
  let p = P.make ~host_visible:false () in
  let d = open_as (module Renamed) "open:renamed" p in
  equal ~msg:"renamed: copies run, bytes back" (pair int bool) (2, true)
    (round_trip d p);
  let p = P.make ~host_visible:false () in
  let d = open_as (module One_queue) "open:one-queue" p in
  equal ~msg:"one queue: copies run, bytes back" (pair int bool) (2, true)
    (round_trip d p);
  let c = open_as (module Copyless) "open:copyless" (P.make ()) in
  equal (list string) [ "COMPUTE:0"; "COPY:0" ]
    (List.map (fun (q : Rig.queue) -> q.name) (Rig.queues c))

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
        (Exn.invalid_arg ~substring:"hang bound")
        (fun () -> Rig.open_ (module P) ~name (fun () -> Ok p));
      equal ~msg:"stopped" bool true (List.mem "stop" (P.log p)))
    [ 0; -1; min_int ]

let open_store name =
  require_ok ~pp:Format.pp_print_string
    (Rig.open_io (module Store) ~name (fun () -> Ok ()))

(* Opens [machine]'s host [name], a driver's device. *)
let open_host machine name =
  Rig.open_host (module P) ~machine ~name (fun () -> Ok (P.make ()))

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
      Rig.open_io (module Other) ~name:"open:io-key" (fun () -> Ok ()))

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
    (let ba = Rig.Buffer.bigarray Bigarray.char h in
     String.init 16 (Bigarray.Array1.get ba))

(* A device of another machine is named after it, and its host is the device
   opened as that machine's, which runs work and loads code. *)
let test_machine () =
  let far = require_ok ~pp:Format.pp_print_string (open_host "far" "HOST") in
  let g =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_
         (module P)
         ~machine:"far" ~name:"open:gpu"
         (fun () -> Ok (P.make ())))
  in
  equal string "open:gpu@far" (Rig.name g);
  equal device far (Rig.host_of g);
  equal device far (Rig.host_of far);
  equal ~msg:"shares this process's host memory" (list bool) [ false; false ]
    [ Rig.shares_host_memory g; Rig.shares_host_memory far ];
  equal (list bool) [ false; false ]
    [ Rig.reaches g Rig.host; Rig.reaches Rig.host g ];
  equal ~msg:"computes" bool true (Rig.computes far);
  ignore (submit (Rig.Submission.make far [||]))

(* A device of another machine opens once that machine's host is open, so every
   device has a host: before, the open raises and leaves the name unopened. *)
let test_machine_without_host () =
  let open_gpu () =
    Rig.open_
      (module P)
      ~machine:"hostless" ~name:"open:lone-gpu"
      (fun () -> Ok (P.make ()))
  in
  raises_match Exn.invalid_arg (fun () -> open_gpu ());
  let far =
    require_ok ~pp:Format.pp_print_string (open_host "hostless" "HOST")
  in
  let g = require_ok ~pp:Format.pp_print_string (open_gpu ()) in
  equal device far (Rig.host_of g)

(* A machine has one host: [~host] names another machine's, and a second host of
   a machine whose host is open under another name is refused. *)
let test_one_host () =
  let h = require_ok ~pp:Format.pp_print_string (open_host "one" "A") in
  raises_match ~msg:"a second" Exn.invalid_arg (fun () -> open_host "one" "B");
  equal ~msg:"the same name" device h
    (require_ok ~pp:Format.pp_print_string (open_host "one" "A"))

(* Closing another machine's host closes every device of its machine too. *)
let test_close_machine () =
  let far = require_ok ~pp:Format.pp_print_string (open_host "closing" "CPU") in
  let g =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_
         (module P)
         ~machine:"closing" ~name:"open:closing-gpu"
         (fun () -> Ok (P.make ())))
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
    Rig.open_
      (module P)
      ~machine ~name:"open:ended-gpu"
      (fun () -> Ok (P.make ()))
  in
  let closed =
    require_ok ~pp:Format.pp_print_string (open_host "ended-closed" "CPU")
  in
  Rig.close closed;
  let lost = P.make () in
  let host =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_host
         (module P)
         ~machine:"ended-lost" ~name:"CPU"
         (fun () -> Ok lost))
  in
  P.fault lost "unplugged";
  (try ignore (Rig.Buffer.create host 8) with Rig.Lost _ -> ());
  equal
    (list (result device string))
    [
      Error "CPU@ended-closed lost: closed";
      Error "CPU@ended-lost lost: unplugged";
    ]
    [ gpu "ended-closed"; gpu "ended-lost" ]

(* Closing a lost host closes its machine's devices too. *)
let test_close_lost_host () =
  let lost = P.make () in
  let far =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_host
         (module P)
         ~machine:"closing-lost" ~name:"CPU"
         (fun () -> Ok lost))
  in
  let g =
    require_ok ~pp:Format.pp_print_string
      (Rig.open_
         (module P)
         ~machine:"closing-lost" ~name:"open:closing-lost-gpu"
         (fun () -> Ok (P.make ())))
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
    (Rig.open_ (module P) ~name:"open:faulted" (fun () -> Ok p));
  equal (list string) [ "stop" ] (P.log p)

let test_point () =
  let d = memory "open:point" in
  let p = submit (Rig.Submission.make d [||]) in
  equal string "open:point:1" (Format.asprintf "%a" Rig.Point.pp p)

(* An opener's error is the open's, and leaves the name free. *)
let test_failed_open () =
  let fails () = Error "no hardware" in
  equal (result device string) (Error "no hardware")
    (Rig.open_ (module P) ~name:"open:failed" fails);
  let d, _ = P.open_ "open:failed" in
  equal string "open:failed" (Rig.name d)

(* An exception the opener raises is raised again, the name left unopened. *)
let test_opener_raises () =
  raises Exit (fun () ->
      Rig.open_ (module P) ~name:"open:raises" (fun () -> raise Exit));
  let d, _ = P.open_ "open:raises" in
  equal string "open:raises" (Rig.name d)

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
  is_error (Rig.open_ (module P) ~name:"open:early" (fun () -> Ok (P.make ())));
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

(* An open whose opener blocks holds back only opens of its own name. *)
let test_blocked_opener () =
  let lock = Mutex.create () and cond = Condition.create () in
  let inside = ref false and go = ref false in
  let make () =
    Mutex.protect lock (fun () ->
        inside := true;
        while not !go do
          Condition.wait cond lock
        done);
    Ok (P.make ())
  in
  let slow =
    Thread.create
      (fun () -> ignore (Rig.open_ (module P) ~name:"open:slow" make))
      ()
  in
  Support.await "a running opener" (fun () ->
      Mutex.protect lock (fun () -> !inside));
  let d, _ = P.open_ "open:fast" in
  equal string "open:fast" (Rig.name d);
  Mutex.protect lock (fun () ->
      go := true;
      Condition.signal cond);
  Thread.join slow

let tests =
  group ~timeout "opening"
    [
      test "one name opens one device, until it is lost" test_same_name;
      test "a name open as another driver's device raises" test_other_driver;
      test "a device states its facts" test_facts;
      test "a queue's name gives it no meaning to rig" test_queue_names;
      test "a region against the host_addresses fact refuses the allocation"
        test_overstated;
      test "a hang bound below 1 ms refuses the open" test_hang_refused;
      test "the host states its facts" test_host;
      test "a driver's device reaches by its facts and its peers" test_reach;
      test "where the host reaches a device, its buffers have host addresses"
        test_reach_addresses;
      test "an io device computes nothing and reaches nothing" test_io;
      test "a name stays with the io library that opened it" test_io_key;
      test "a region an io library gave is a buffer of its device" test_of_io;
      test "a device of another machine is named after it, its host the io's"
        test_machine;
      test "a device of another machine opens once that machine's host is"
        test_machine_without_host;
      test "a machine has one host" test_one_host;
      test "closing a machine's host closes its devices" test_close_machine;
      test "a machine whose host ended opens no more devices"
        test_open_on_ended_host;
      test "closing a lost host closes its machine's devices"
        test_close_lost_host;
      test "a point prints as its device's name and its value" test_point;
      test "an opener's error leaves the name free" test_failed_open;
      test "an opener's exception is raised, the name left free"
        test_opener_raises;
      test "a lost device's name opens again only once its stop returned"
        test_reopen_early;
      test "a fault reading a device's facts stops its handle"
        test_fault_at_open;
      test "a lost device's name opens anew once its stop answered" test_reopen;
      test "a blocked opener holds back no other name" test_blocked_opener;
    ]

let () = exit (run "rig.open" [ tests ])
