(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module C = Rig
module P = Rig_support.Polled
module Support = Rig_support

let timeout = 60.
let device = Testable.make ~pp:C.pp ~equal:C.equal
let memory name = require_ok ~pp:Format.pp_print_string (C.memory_device name)

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
  equal string "open:facts" (C.name d);
  equal bool true (C.computes d);
  equal bool true (C.runs_on_host d);
  equal bool false (C.runs_on_host p);
  equal bool true (C.shares_host_memory d);
  equal bool true (C.reaches d C.host);
  equal bool true (C.reaches C.host d);
  equal device C.host (C.host_of d);
  is_some (C.capability p P.capability_key)

let test_host () =
  equal string "CPU" (C.name C.host);
  mem string (C.arch C.host) [ "arm64"; "x86_64" ];
  equal bool true (C.computes C.host);
  equal bool true (C.runs_on_host C.host);
  equal bool true (C.shares_host_memory C.host);
  equal device C.host (C.host_of C.host)

(* What a driver's device reaches follows its copy queue and its peers. *)
let test_reach () =
  let d, _ = P.open_ "open:reach" in
  let e, _ = P.open_ "open:reach-peer" in
  let f, _ = P.open_ ~peers:false "open:reach-alone" in
  let g, _ = P.open_ ~copies:false "open:reach-no-copy" in
  equal string "polled" (C.arch d);
  equal (list bool) [ true; false; false ]
    [ C.reaches d C.host; C.reaches C.host d; C.shares_host_memory d ];
  equal (list bool) [ true; true; true ]
    [ C.reaches g C.host; C.reaches C.host g; C.shares_host_memory g ];
  equal (list bool) [ true; false ] [ C.reaches d e; C.reaches f d ]

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

let open_store ?machine ?host name =
  require_ok ~pp:Format.pp_print_string
    (C.open_io (module Store) ?machine ?host ~name (fun () -> Ok ()))

let test_io () =
  let io = open_store "open:io" in
  equal string "" (C.arch io);
  equal bool false (C.computes io);
  equal (list bool) [ false; false ]
    [ C.reaches io C.host; C.reaches C.host io ];
  equal int 8 (C.Buffer.length (C.Buffer.create io 8))

(* A name an io library opened stays that library's: another's open of it
   raises. *)
let test_io_key () =
  ignore (open_store "open:io-key");
  raises_match Exn.invalid_arg (fun () ->
      C.open_io (module Other) ~name:"open:io-key" (fun () -> Ok ()))

(* A region an io library gave is a buffer of its device, which gives it back by
   the library's key alone, and borrows on the host through its pages. *)
let test_of_io () =
  let io = open_store "open:of-io" in
  let r = Bigarray.Array1.create Bigarray.char Bigarray.c_layout 16 in
  Bigarray.Array1.fill r 'p';
  let b = C.Buffer.of_io io Store.region_key r 16 in
  equal ~msg:"the region given" bool true
    (match C.Buffer.io b Store.region_key with
    | Some r' -> r' == r
    | None -> false);
  is_none (C.Buffer.io b Other.region_key);
  raises_match Exn.invalid_arg (fun () ->
      C.Buffer.of_io io Other.region_key r 16);
  let dead = C.Buffer.of_io io Store.region_key r 16 in
  C.Claim.with_ ~read:[ dead ] ~donate:[] (fun c ->
      ignore (C.Claim.consume c ~why:"consumed" dead));
  raises_match (Exn.invalid_arg ~substring:"consumed") (fun () ->
      C.Buffer.io dead Store.region_key);
  let h = require_some (C.Buffer.borrow C.host b) in
  equal string (String.make 16 'p')
    (let ba = C.Buffer.bigarray Bigarray.char h in
     String.init 16 (Bigarray.Array1.get ba))

(* A device of another machine is named after it, and its host is the io device
   opened as that machine's. *)
let test_machine () =
  let far = open_store ~machine:"far" ~host:true "HOST" in
  let g =
    require_ok ~pp:Format.pp_print_string
      (C.open_
         (module P)
         ~machine:"far" ~name:"open:gpu"
         (fun () -> Ok (P.make ())))
  in
  equal string "open:gpu@far" (C.name g);
  equal device far (C.host_of g);
  equal device far (C.host_of far);
  equal (list bool) [ false; false ] [ C.reaches g C.host; C.reaches C.host g ]

(* A device of a machine whose host is not open has no host here: it shares no
   memory with this process's host. *)
let test_machine_without_host () =
  let g =
    require_ok ~pp:Format.pp_print_string
      (C.open_
         (module P)
         ~machine:"hostless" ~name:"open:lone-gpu"
         (fun () -> Ok (P.make ())))
  in
  raises_match Exn.invalid_arg (fun () -> C.host_of g);
  equal bool false (C.shares_host_memory g)

let test_point () =
  let d = memory "open:point" in
  let p = C.submit (C.Submission.make ~reads:0 ~writes:0 ~waits:0 d [||]) in
  equal string "open:point:1" (Format.asprintf "%a" C.Point.pp p)

(* An opener's error is the open's, and leaves the name free. *)
let test_failed_open () =
  let fails () = Error "no hardware" in
  equal (result device string) (Error "no hardware")
    (C.open_ (module P) ~name:"open:failed" fails);
  let d, _ = P.open_ "open:failed" in
  equal string "open:failed" (C.name d)

let test_reopen () =
  let d, p = P.open_ "open:reopen" in
  let s = C.Submission.make ~reads:0 ~writes:0 ~waits:0 d [||] in
  P.fail p;
  raises_match (function C.Lost _ -> true | _ -> false) (fun () -> C.submit s);
  equal (list string) [ "stop" ] (P.log p);
  let d', _ = P.open_ "open:reopen" in
  not_equal device d d';
  equal (option string) None (C.lost d')

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
      (fun () -> ignore (C.open_ (module P) ~name:"open:slow" make))
      ()
  in
  Support.await "a running opener" (fun () ->
      Mutex.protect lock (fun () -> !inside));
  let d, _ = P.open_ "open:fast" in
  equal string "open:fast" (C.name d);
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
      test "the host states its facts" test_host;
      test "a driver's device reaches by its copies and its peers" test_reach;
      test "an io device computes nothing and reaches nothing" test_io;
      test "a name stays with the io library that opened it" test_io_key;
      test "a region an io library gave is a buffer of its device" test_of_io;
      test "a device of another machine is named after it, its host the io's"
        test_machine;
      test "a device of a machine whose host is not open has no host"
        test_machine_without_host;
      test "a point prints as its device's name and its value" test_point;
      test "an opener's error leaves the name free" test_failed_open;
      test "a lost device's name opens anew once its stop answered" test_reopen;
      test "a blocked opener holds back no other name" test_blocked_opener;
    ]

let () = exit (run "rig.open" [ tests ])
