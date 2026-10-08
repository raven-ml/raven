(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The GSP's message queues: records against message_queue_cpu.c and the RPC
   headers (open-gpu-kernel-modules 570.144), and the rings on host memory, the
   suite playing the GSP's side. *)

open Windtrap
module Msgq = Device_nv_pci.Msgq
module Window = Device_pci.Window
module Support = Device_nv_pci_support

(* rpc_global_enums.h's numbers. *)
let continuation = 0x47
let gsp_rm_control = 0x4c
let run_cpu_sequencer = 0x1002
let rc_triggered = 0x1004
let mmu_fault_queued = 0x1005
let os_error_log = 0x1006
let libos_print = 0x100c

(* An element is 4 KiB; its RPC header ends at byte 80, so a record of 16
   elements carries 16 * 4096 - 80 bytes of body. *)
let element = 4096
let record_max = (16 * element) - 80

let bodies =
  Gen.(
    one_of
      [
        string_of ~size:(int_range 0 64) char;
        map
          (fun n -> String.init n (fun i -> Char.chr (i land 0xff)))
          (int_range 0 ((2 * record_max) + 10));
      ])

let test_checksum =
  prop "an element's checksum makes its words fold to zero"
    Gen.(triple (int_range 0 0xffff) (int_range 0 0x2000) bodies)
    (fun (seq, fn, body) ->
      let e = Msgq.element ~seq fn body in
      equal int 0 (String.length e mod element);
      equal int 0 (Msgq.checksum e))

let test_records =
  prop "records carry the body, the first with its function" bodies (fun body ->
      let rs = Msgq.records gsp_rm_control body in
      cover "one record" (List.length rs = 1);
      cover "continued" (List.length rs > 1);
      equal string body (String.concat "" (List.map snd rs));
      equal int gsp_rm_control (fst (List.hd rs));
      List.iter (fun (fn, _) -> equal int continuation fn) (List.tl rs);
      List.iter
        (fun (_, b) -> at_most int ~than:record_max (String.length b))
        rs;
      equal int
        (Int.max 1 ((String.length body + record_max - 1) / record_max))
        (List.length rs))

(* A message reads back as the element that wrote it, its result the pending
   value the sender writes (NV_VGPU_MSG_RESULT_RPC_PENDING). *)
let test_message =
  prop "a message reads back its element"
    Gen.(pair (int_range 0 0x2000) (string_of ~size:(int_range 0 9000) char))
    (fun (fn, body) ->
      let e = Msgq.element ~seq:7 fn body in
      let m, count = require_ok (Msgq.message e) in
      equal int fn m.fn;
      equal int 0xffff_ffff m.result;
      equal string body m.body;
      equal int (String.length e / element) count)

let test_not_a_message () =
  let e = Bytes.of_string (Msgq.element ~seq:0 gsp_rm_control "x") in
  Bytes.set e 52 '\000';
  contains ~sub:"signature" (require_error (Msgq.message (Bytes.to_string e)));
  contains ~sub:"shorter" (require_error (Msgq.message "short"))

(* Faults *)

(* rpc_rc_triggered_v17_02: the channel at byte 4, the exception type at 16;
   nverror.h's ROBUST_CHANNEL_FIFO_ERROR_MMU_ERR_FLT is 31. *)
let rc ~chid ~xid =
  let b = Bytes.make 48 '\000' in
  Bytes.set_int32_le b 4 (Int32.of_int chid);
  Bytes.set_int32_le b 16 (Int32.of_int xid);
  Bytes.to_string b

let event fn body =
  fst (require_ok (Msgq.message (Msgq.element ~seq:0 fn body)))

let test_rc () =
  let why =
    require_some (Msgq.fault (event rc_triggered (rc ~chid:3 ~xid:31)))
  in
  contains ~sub:"channel 3" why;
  contains ~sub:"FIFO_ERROR_MMU_ERR_FLT" why

let test_no_fault =
  cases "other messages carry no fault"
    ~name:(fun (n, _) -> n)
    [
      ("an error log", os_error_log);
      ("a CPU sequence", run_cpu_sequencer);
      ("a log print", libos_print);
      ("an answer", gsp_rm_control);
    ]
    (fun (_, fn) -> is_none (Msgq.fault (event fn (String.make 300 'x'))))

let test_mmu () = is_some (Msgq.fault (event mmu_fault_queued ""))

(* Rings *)

(* Two queues of 8 elements after their header page; the suite writes the GSP's
   header of the status queue as the GSP does once it runs. *)
let count = 8
let size = 0x1000 + (count * element)

let queues () =
  let w = Support.window (2 * size) in
  let doorbell = Support.window 4 in
  Window.set32 doorbell 0 0xffff_ffff;
  let q = Msgq.create w ~doorbell in
  let stat = Window.sub w size size in
  (q, w, stat, doorbell)

(* msgqTxHeader: msgSize at 8, msgCount at 12, writePtr at 16, rxHdrOff at 24,
   entryOff at 28; each reader's position at rxHdrOff of the other queue's
   page. *)
let start stat =
  Window.set32 stat 8 element;
  Window.set32 stat 12 count;
  Window.set32 stat 24 32;
  Window.set32 stat 28 0x1000

let test_ready () =
  let q, _, stat, _ = queues () in
  equal bool false (Msgq.ready q);
  start stat;
  equal bool true (Msgq.ready q)

let test_send () =
  let q, w, stat, doorbell = queues () in
  start stat;
  equal bool true (Msgq.ready q);
  equal bool true (Msgq.send q gsp_rm_control "hello");
  equal int 1 (Window.get32 w 16);
  equal int 0 (Window.get32 doorbell 0);
  let m, _ = require_ok (Msgq.message (Window.read w 0x1000 element)) in
  equal string "hello" m.body;
  (* A body of two records takes 16 elements then 1: more than the queue holds,
     so nothing is written. *)
  equal bool false
    (Msgq.send q gsp_rm_control (String.make (record_max + 1) 'x'));
  equal int 1 (Window.get32 w 16)

(* The suite consumes what the process sends, as the GSP would, so the writes
   wrap around the ring's end. *)
let test_wrap () =
  let q, w, stat, _ = queues () in
  start stat;
  ignore (Msgq.ready q);
  for i = 0 to (3 * count) - 1 do
    let body = String.make (element + i) (Char.chr (65 + (i mod 26))) in
    equal bool true (Msgq.send q gsp_rm_control body);
    let wp = Window.get32 w 16 in
    let rp = Window.get32 stat 32 in
    let s =
      String.concat ""
        (List.init 2 (fun k ->
             Window.read w (0x1000 + ((rp + k) mod count * element)) element))
    in
    let m, n = require_ok (Msgq.message s) in
    equal string body m.body;
    equal int wp ((rp + n) mod count);
    Window.set32 stat 32 wp
  done

let test_full () =
  let q, w, stat, _ = queues () in
  start stat;
  ignore (Msgq.ready q);
  for _ = 1 to count - 1 do
    equal bool true (Msgq.send q gsp_rm_control "x")
  done;
  equal bool false (Msgq.send q gsp_rm_control "x");
  equal int (count - 1) (Window.get32 w 16)

let test_receive () =
  let q, w, stat, _ = queues () in
  start stat;
  ignore (Msgq.ready q);
  is_none (Msgq.receive q);
  (* The GSP writes an event of two elements at element 7, wrapping. *)
  Window.set32 w 32 7;
  Window.set32 stat 16 7;
  ignore (Msgq.receive q);
  let body = rc ~chid:5 ~xid:31 ^ String.make element 'z' in
  let e = Msgq.element ~seq:0 rc_triggered body in
  Window.write stat (0x1000 + (7 * element)) (String.sub e 0 element);
  Window.write stat 0x1000 (String.sub e element element);
  Window.set32 stat 16 1;
  let m = require_ok (require_some (Msgq.receive q)) in
  equal string body m.body;
  equal int 1 (Window.get32 w 32);
  is_none (Msgq.receive q)

let test_receive_bad () =
  let q, w, stat, _ = queues () in
  start stat;
  ignore (Msgq.ready q);
  Window.set32 stat 16 1;
  ignore (require_error (require_some (Msgq.receive q)));
  equal int 1 (Window.get32 w 32)

let () =
  exit
  @@ run "device_nv_pci.msgq"
       [
         group ~timeout:20. "records"
           [
             test_checksum;
             test_records;
             test_message;
             test "a GSP message is checked" test_not_a_message;
           ];
         group ~timeout:10. "faults"
           [
             test "a stopped channel is a fault naming it" test_rc;
             test "a queued MMU fault is a fault" test_mmu;
             test_no_fault;
           ];
         group ~timeout:10. "rings"
           [
             test "the status queue is ready once the GSP wrote it" test_ready;
             test "a send writes its record and the doorbell" test_send;
             test "sends wrap around the ring" test_wrap;
             test "a full queue takes nothing" test_full;
             test "a received message is consumed" test_receive;
             test "what is no message is received as an error" test_receive_bad;
           ];
       ]
