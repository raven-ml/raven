(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module C = Device_core
module B = Device_core.Buffer
module Sub = Device_core.Submission
module P = Device_core_support.Polled
module Support = Device_core_support
module S = Device_dtype.Scalar

let timeout = 60.
let memory name = require_ok ~pp:Format.pp_print_string (C.memory_device name)

let empty ?(reads = 0) ?(writes = 0) ?(waits = 0) d =
  Sub.make ~reads ~writes ~waits d [||]

let page_bytes = 1 lsl 16

(* A part that adds 1 to the 64-bit word at the start of [arg]. *)
let bump arg =
  {
    Sub.queue = "COMPUTE:0";
    after = [||];
    work =
      Sub.Fill { fill = Support.bump; arg; ring_units = 0; segment_bytes = 0 };
  }

let word b = Support.load (B.address b)

let test_values () =
  let d = memory "submit:values" in
  let s = empty d in
  let values = List.init 5 (fun _ -> C.Point.value (C.submit s)) in
  equal (list int) [ 1; 2; 3; 4; 5 ] values;
  equal int 5 (C.submitted d);
  equal int 5 (C.signaled d);
  equal bool true (C.equal d (C.Point.device (C.submit s)))

let test_fill () =
  let d = memory "submit:fill" in
  let arg = B.create C.host S.UInt64 1 in
  Support.store (B.address arg) 0;
  let s = Sub.make ~reads:0 ~writes:0 ~waits:0 d [| bump arg; bump arg |] in
  ignore (C.submit s);
  ignore (C.submit s);
  equal int 4 (word arg)

(* Polled runs nothing until a wait reaches its sleep. *)
let test_polled () =
  let d, p = P.open_ "submit:polled" in
  let s = empty d in
  let a = C.submit s in
  let b = C.submit s in
  equal int 2 (P.queued p);
  equal int 0 (C.signaled d);
  C.wait d (C.Point.value b);
  equal int 0 (P.queued p);
  equal bool true (C.Point.value a < C.Point.value b)

let test_refusals () =
  let d = memory "submit:refusals" in
  let arg = B.create C.host S.UInt64 1 in
  raises_match Exn.invalid_arg (fun () ->
      Sub.make ~reads:0 ~writes:0 ~waits:0 d
        [| { (bump arg) with after = [| 0 |] } |]);
  raises_match Exn.invalid_arg (fun () ->
      Sub.make ~reads:0 ~writes:0 ~waits:0 d
        [| { (bump arg) with queue = "COPY:9" } |]);
  raises_match Exn.invalid_arg (fun () ->
      Sub.make ~reads:(-1) ~writes:0 ~waits:0 d [||]);
  raises_match Exn.invalid_arg (fun () -> C.submit (empty ~reads:1 d))

let test_unset_wait () =
  let d = memory "submit:unset-wait" in
  equal int 1 (C.Point.value (C.submit (empty ~waits:2 d)))

let test_wait_beyond () =
  let d = memory "submit:beyond" and e = memory "submit:beyond-2" in
  let p = C.submit (empty e) in
  raises_match Exn.invalid_arg (fun () -> C.wait e (C.Point.value p + 1));
  let s = empty ~waits:1 d in
  Sub.wait_for s 0 p;
  ignore (C.submit s)

(* A read of memory another device wrote waits for that write: on the host,
   since a memory device waits on no other device in its queue. *)
let test_read_waits () =
  let producer, pp = P.open_ "submit:producer" in
  let consumer = memory "submit:consumer" in
  let on = B.create producer S.UInt8 page_bytes in
  let w = Sub.make ~reads:0 ~writes:1 ~waits:0 producer [||] in
  Sub.write w 0 on;
  ignore (C.submit w);
  equal int 1 (P.queued pp);
  let r = Sub.make ~reads:1 ~writes:0 ~waits:0 consumer [||] in
  Sub.read r 0 on;
  ignore (C.submit r);
  equal int 0 (P.queued pp)

(* A Polled device that waits on host-written words waits for a producer in its
   queue: the submit hands it over without waiting. *)
let test_in_queue () =
  let producer, pp = P.open_ "submit:iq-producer" in
  let consumer, cp = P.open_ ~waits_host:true "submit:iq-consumer" in
  let a = C.submit (empty producer) in
  let s = empty ~waits:1 consumer in
  Sub.wait_for s 0 a;
  let b = C.submit s in
  equal int 1 (P.queued pp);
  equal int 1 (P.queued cp);
  equal int 0 (P.run cp);
  C.wait producer (C.Point.value a);
  equal int 1 (P.run cp);
  equal int (C.Point.value b) (C.signaled consumer)

(* A full queue answers Later: the submit waits for one more value. *)
let test_room () =
  let d, p = P.open_ ~capacity:1 "submit:room" in
  let arg = B.create C.host S.UInt64 1 in
  let s = Sub.make ~reads:0 ~writes:0 ~waits:0 d [| bump arg |] in
  ignore (C.submit s);
  ignore (C.submit s);
  equal int 1 (P.queued p);
  equal int 1 (C.signaled d)

(* A copy on a device that runs no copies is refused where the caller can act:
   when the submission is made. *)
let test_copy_refused () =
  let d, _ = P.open_ ~copies:false "submit:no-copies" in
  let src = B.create d S.UInt8 8 and dst = B.create d S.UInt8 8 in
  let copy =
    { Sub.queue = "COMPUTE:0"; after = [||]; work = Sub.Copy { src; dst } }
  in
  raises_match Exn.invalid_arg (fun () ->
      Sub.make ~reads:0 ~writes:0 ~waits:0 d [| copy |])

let test_never () =
  let d, _ = P.open_ ~capacity:1 "submit:never" in
  let arg = B.create C.host S.UInt64 1 in
  let s = Sub.make ~reads:0 ~writes:0 ~waits:0 d [| bump arg; bump arg |] in
  raises_match Exn.invalid_arg (fun () -> C.submit s)

(* A submission keeps its parts' memory while it is reachable. *)
let test_parts_held () =
  let d, p = P.open_ "submit:held" in
  let s =
    let src = B.create d S.UInt8 64 and dst = B.create d S.UInt8 64 in
    Sub.make ~reads:0 ~writes:0 ~waits:0 d
      [| { Sub.queue = "COPY:0"; after = [||]; work = Sub.Copy { src; dst } } |]
  in
  Gc.full_major ();
  ignore (B.create d S.UInt8 64);
  equal int 0 (List.length (List.filter (( = ) "free") (P.log p)));
  C.wait d (C.Point.value (C.submit s))

let test_allocation () =
  let d = memory "submit:words" in
  let s = empty ~reads:1 d and b = B.create C.host S.UInt8 8 in
  Sub.read s 0 b;
  ignore (C.submit s);
  let before = Gc.minor_words () in
  for _ = 1 to 100 do
    Sub.read s 0 b;
    ignore (Sys.opaque_identity (C.submit s))
  done;
  let words = int_of_float (Gc.minor_words () -. before) / 100 in
  equal int 0 words

let tests =
  [
    group ~timeout "values"
      [
        test "values follow one another from 1" test_values;
        test "a submission's fills run with its value" test_fill;
        test "a device's work completes once a wait reaches it" test_polled;
        test "a wait slot left unset waits for nothing" test_unset_wait;
        test "a wait names a submitted value" test_wait_beyond;
      ];
    group ~timeout "refusals"
      [
        test "a submission refuses what it cannot run" test_refusals;
        test "parts that never fit are refused" test_never;
        test "a copy on a device that runs no copies is refused"
          test_copy_refused;
      ];
    group ~timeout "order"
      [
        test "a read waits for another device's write" test_read_waits;
        test "a queue that waits on host words waits in the queue" test_in_queue;
        test "a full queue's submit waits for room" test_room;
        test "a submission keeps its parts' memory" test_parts_held;
      ];
    group ~timeout "cost"
      [
        test "a submit that does not wait allocates at most 32 words"
          test_allocation;
      ];
  ]

let () = exit (run "device_core.submit" tests)
