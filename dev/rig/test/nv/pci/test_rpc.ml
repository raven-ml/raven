(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The GSP's records, calls and CPU sequences, against open-gpu-kernel-modules
   570.144: message_queue_cpu.c and the RPC headers, g_rpc-structures.h
   (rpc_gsp_rm_alloc_v03_00, rpc_gsp_rm_control_v03_00,
   rpc_set_page_directory_v1E_05, rpc_unloading_guest_driver_v1F_07,
   rpc_run_cpu_sequencer_v17_00, rpc_rc_triggered_v17_02), g_os_nvoc.h
   (PACKED_REGISTRY_TABLE) and rmgspseq.h (the opcodes). *)

open Windtrap

let u8 s o = Char.code s.[o]
let u32 s o = Int32.to_int (String.get_int32_le s o) land 0xffff_ffff
let u64 s o = Int64.to_int (String.get_int64_le s o)

(* rpc_global_enums.h's numbers. *)
let continuation = 0x47
let gsp_rm_control = 0x4c
let run_cpu_sequencer = 0x1002
let rc_triggered = 0x1004
let mmu_fault_queued = 0x1005
let os_error_log = 0x1006
let libos_print = 0x100c

(* Records *)

(* An element is 4 KiB; its RPC header ends at byte 80, so a record of 16
   elements carries 16 * 4096 - 80 bytes of body. GSP_MSG_QUEUE_ELEMENT holds
   two buffers of 16 bytes, then its checksum at 0x20, its sequence number at
   0x24 and its element count at 0x28; its RPC header starts at 0x30, the
   signature at its byte 4. *)
let element = 4096
let record_max = (16 * element) - 80

(* The XOR of a string's 64-bit words, folded to 32 bits. *)
let fold s =
  let x = ref 0L in
  for i = 0 to (String.length s / 8) - 1 do
    x := Int64.logxor !x (String.get_int64_le s (8 * i))
  done;
  Int64.(to_int (logand (logxor !x (shift_right_logical !x 32)) 0xffff_ffffL))

let bodies =
  Gen.(
    one_of
      [
        string_of ~size:(int_range 0 64) char;
        map
          (fun n -> String.init n (fun i -> Char.chr (i land 0xff)))
          (int_range 0 ((2 * record_max) + 10));
      ])

let test_records =
  prop "records carry the body, the first with its function, each checksummed"
    Gen.(pair (int_range 0 0xffff) bodies)
    (fun (seq, body) ->
      let rs = Rpc.records ~seq gsp_rm_control body in
      cover "one record" (List.length rs = 1);
      cover "continued" (List.length rs > 1);
      equal int
        (Int.max 1 ((String.length body + record_max - 1) / record_max))
        (List.length rs);
      let ms =
        List.mapi
          (fun i r ->
            equal ~msg:"whole elements" int 0 (String.length r mod element);
            equal ~msg:"the checksum" int 0 (fold r);
            equal ~msg:"the sequence number" int (seq + i) (u32 r 0x24);
            let m, count = require_ok (Rpc.message r) in
            equal ~msg:"the element count" int (String.length r / element) count;
            equal int count (Rpc.elements r);
            at_most int ~than:record_max (String.length m.body);
            m)
          rs
      in
      equal string body (String.concat "" (List.map (fun m -> m.Rpc.body) ms));
      equal int gsp_rm_control (List.hd ms).fn;
      List.iter (fun m -> equal int continuation m.Rpc.fn) (List.tl ms))

(* A message reads back as the record that wrote it, its result the pending
   value the sender writes (NV_VGPU_MSG_RESULT_RPC_PENDING). *)
let test_message =
  prop "a message reads back its record"
    Gen.(pair (int_range 0 0x2000) (string_of ~size:(int_range 0 9000) char))
    (fun (fn, body) ->
      let r = List.hd (Rpc.records ~seq:7 fn body) in
      let m, _ = require_ok (Rpc.message r) in
      equal int fn m.fn;
      equal int 0xffff_ffff m.result;
      equal string body m.body)

(* The RPC header's signature is at byte 0x34 of the element. *)
let test_not_a_message () =
  let e = Bytes.of_string (List.hd (Rpc.records ~seq:0 gsp_rm_control "x")) in
  Bytes.set e 52 '\000';
  contains ~sub:"signature" (require_error (Rpc.message (Bytes.to_string e)));
  contains ~sub:"shorter" (require_error (Rpc.message "short"));
  let e = List.hd (Rpc.records ~seq:0 gsp_rm_control (String.make 5000 'x')) in
  contains ~sub:"longer" (require_error (Rpc.message (String.sub e 0 element)))

(* Faults *)

(* rpc_rc_triggered_v17_02: the channel at byte 4, the exception type at 16;
   nverror.h's ROBUST_CHANNEL_FIFO_ERROR_MMU_ERR_FLT is 31. *)
let rc ~chid ~xid =
  let b = Bytes.make 48 '\000' in
  Bytes.set_int32_le b 4 (Int32.of_int chid);
  Bytes.set_int32_le b 16 (Int32.of_int xid);
  Bytes.to_string b

let event fn body =
  fst (require_ok (Rpc.message (List.hd (Rpc.records ~seq:0 fn body))))

let test_rc () =
  let why =
    require_some (Rpc.fault (event rc_triggered (rc ~chid:3 ~xid:31)))
  in
  contains ~sub:"channel 3" why;
  contains ~sub:"FIFO_ERROR_MMU_ERR_FLT" why;
  contains ~sub:"Xid 31" why

let test_rc_unknown () =
  let why =
    require_some (Rpc.fault (event rc_triggered (rc ~chid:1 ~xid:0xffff)))
  in
  contains ~sub:"an unknown error" why;
  is_some (Rpc.fault (event rc_triggered "short"))

let test_no_fault =
  cases "other messages carry no fault"
    ~name:(fun (n, _) -> n)
    [
      ("an error log", os_error_log);
      ("a CPU sequence", run_cpu_sequencer);
      ("a log print", libos_print);
      ("an answer", gsp_rm_control);
    ]
    (fun (_, fn) -> is_none (Rpc.fault (event fn (String.make 300 'x'))))

let test_mmu () = is_some (Rpc.fault (event mmu_fault_queued ""))

(* Calls *)

(* rpc_gsp_rm_alloc_v03_00: hClient, hParent, hObject, hClass, status,
   paramsSize and flags as words, the parameters at 0x20. *)
let test_alloc () =
  let b = Rpc.rm_alloc ~client:0xc1000000 ~parent:7 ~obj:9 ~cls:0xc7c0 "abc" in
  equal int (0x20 + 3) (String.length b);
  equal (list int)
    [ 0xc1000000; 7; 9; 0xc7c0; 0; 3; 0 ]
    (List.init 7 (fun i -> u32 b (4 * i)));
  equal string "abc" (String.sub b 0x20 3)

(* rpc_gsp_rm_control_v03_00: hClient, hObject, cmd, status, paramsSize, flags,
   the parameters at 0x18. *)
let test_control () =
  let b = Rpc.rm_control ~client:1 ~obj:2 ~cmd:0x20800a0a "xy" in
  equal int (0x18 + 2) (String.length b);
  equal (list int)
    [ 1; 2; 0x20800a0a; 0; 2; 0 ]
    (List.init 6 (fun i -> u32 b (4 * i)));
  equal string "xy" (String.sub b 0x18 2)

let params = Gen.string_of ~size:(Gen.int_range 0 600) Gen.char

let test_answer =
  prop "an answer gives back the status and the parameters" params (fun p ->
      equal
        (result (pair int string) string)
        (Ok (0, p))
        (Rpc.rm_answer `Control (Rpc.rm_control ~client:1 ~obj:2 ~cmd:3 p));
      let a =
        Bytes.of_string (Rpc.rm_alloc ~client:1 ~parent:2 ~obj:3 ~cls:4 p)
      in
      Bytes.set_int32_le a 0x10 0x56l;
      equal
        (result (pair int string) string)
        (Ok (0x56, p))
        (Rpc.rm_answer `Alloc (Bytes.to_string a)))

let test_short_answer () =
  contains ~sub:"shorter" (require_error (Rpc.rm_answer `Alloc "short"));
  let b = Bytes.of_string (Rpc.rm_control ~client:1 ~obj:2 ~cmd:3 "") in
  Bytes.set_int32_le b 0x10 64l;
  contains ~sub:"64 bytes"
    (require_error (Rpc.rm_answer `Control (Bytes.to_string b)))

(* rpc_set_page_directory_v1E_05: hClient, hDevice, pasid, then at 0x10
   NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS_v1E_05: physAddress, numEntries at
   8, flags at 12 (APERTURE_VIDMEM, 0x8), hVASpace at 16, chId at 20,
   subDeviceId at 24, pasid at 28; no PASID is 0xffffffff. *)
let test_page_directory () =
  let b =
    Rpc.page_directory ~client:1 ~device:2 ~vaspace:3 ~root:0x20_0000 ~entries:4
  in
  equal int 48 (String.length b);
  equal (list int) [ 1; 2; 0xffffffff ] [ u32 b 0; u32 b 4; u32 b 8 ];
  equal int 0x20_0000 (u64 b 0x10);
  equal (list int)
    [ 4; 8; 3; 0; 1; 0xffffffff ]
    (List.map (fun o -> u32 b (0x10 + o)) [ 8; 12; 16; 20; 24; 28 ])

(* rpc_unloading_guest_driver_v1F_07: bInPMTransition, bGc6Entering, then
   newLevel at 4. *)
let test_unloading () =
  equal int 8 (String.length Rpc.unloading);
  equal (list int) [ 0; 0; 0x40 ]
    [ u8 Rpc.unloading 0; u8 Rpc.unloading 1; u32 Rpc.unloading 4 ]

(* PACKED_REGISTRY_TABLE: size and numEntries, then PACKED_REGISTRY_ENTRY of 16
   bytes each (nameOffset, type at 4, data at 8, length at 12), then the names,
   each ending in a zero byte; a DWORD entry's type is 1. *)
let test_registry =
  let key = Gen.string_of ~size:(Gen.int_range 1 20) (Gen.char_range 'A' 'z') in
  prop "the registry holds each key, its value and its name"
    Gen.(list ~size:(int_range 0 6) (pair key (int_range 0 0xffff_ffff)))
    (fun keys ->
      let t = Rpc.registry keys in
      equal int (String.length t) (u32 t 0);
      equal int (List.length keys) (u32 t 4);
      List.iteri
        (fun i (k, v) ->
          let e = 8 + (16 * i) in
          let name = u32 t e in
          equal string (k ^ "\000") (String.sub t name (String.length k + 1));
          equal (list int) [ 1; v; 4 ]
            [ u8 t (e + 4); u32 t (e + 8); u32 t (e + 12) ])
        keys)

(* CPU sequences *)

(* rpc_run_cpu_sequencer_v17_00: bufferSizeDWord, cmdIndex (the words used),
   regSaveArea of 8 words, then the commands from 0x28, each its opcode and its
   payload, a GSP_SEQ_BUF_PAYLOAD_ structure. Opcodes: REG_WRITE 0 (addr, val),
   REG_MODIFY 1 (addr, mask, val), REG_POLL 2 (addr, mask, val, timeout, error),
   DELAY_US 3 (val), REG_STORE 4 (addr, index), CORE_RESET 5, CORE_START 6,
   CORE_WAIT_FOR_HALT 7, CORE_RESUME 8. *)
let sequencer words =
  let n = List.length words in
  let b = Bytes.make (0x28 + (4 * n)) '\000' in
  Bytes.set_int32_le b 0 (Int32.of_int n);
  Bytes.set_int32_le b 4 (Int32.of_int n);
  List.iteri
    (fun i w -> Bytes.set_int32_le b (0x28 + (4 * i)) (Int32.of_int w))
    words;
  Bytes.to_string b

let pp_step ppf (s : Rpc.step) =
  match s with
  | Write (r, x) -> Format.fprintf ppf "write 0x%x 0x%x" r x
  | Modify (r, m, x) -> Format.fprintf ppf "modify 0x%x 0x%x 0x%x" r m x
  | Poll (r, m, x) -> Format.fprintf ppf "poll 0x%x 0x%x 0x%x" r m x
  | Delay_us n -> Format.fprintf ppf "delay %d" n
  | Store (r, i) -> Format.fprintf ppf "store 0x%x %d" r i
  | Core_reset -> Format.fprintf ppf "core reset"
  | Core_start -> Format.fprintf ppf "core start"
  | Core_wait_for_halt -> Format.fprintf ppf "core wait for halt"
  | Core_resume -> Format.fprintf ppf "core resume"

let step = Testable.make ~pp:pp_step ~equal:( = )

let test_sequence =
  cases "each opcode is its step, its payload read in order"
    ~name:(fun (n, _, _) -> n)
    [
      ("a write", [ 0; 0x1000; 0x5 ], [ Rpc.Write (0x1000, 0x5) ]);
      ("a modify", [ 1; 0x2000; 0xff; 0x10 ], [ Modify (0x2000, 0xff, 0x10) ]);
      ( "a poll, its timeout and error left",
        [ 2; 0x3000; 0xff; 0x10; 1000; 0 ],
        [ Poll (0x3000, 0xff, 0x10) ] );
      ("a delay", [ 3; 20 ], [ Delay_us 20 ]);
      ("a store", [ 4; 0x4000; 2 ], [ Store (0x4000, 2) ]);
      ( "the core's commands",
        [ 5; 6; 7; 8 ],
        [ Core_reset; Core_start; Core_wait_for_halt; Core_resume ] );
      ( "one after another",
        [ 0; 0x1000; 0x5; 3; 20; 6 ],
        [ Write (0x1000, 0x5); Delay_us 20; Core_start ] );
      ("nothing", [], []);
    ]
    (fun (_, words, steps) ->
      equal (list step) steps (require_ok (Rpc.sequence (sequencer words))))

let test_sequence_refused =
  cases "a sequence it cannot run is refused"
    ~name:(fun (n, _, _) -> n)
    [
      ("an unknown opcode", sequencer [ 9 ], "unknown opcode 9");
      ("a command cut short", sequencer [ 0; 0x1000 ], "ending inside");
      ("a header cut short", String.make 0x20 '\000', "shorter");
      ( "more words than the message holds",
        String.sub (sequencer [ 3; 20 ]) 0 0x2c,
        "longer" );
    ]
    (fun (_, body, why) ->
      contains ~sub:why (require_error (Rpc.sequence body)))

let () =
  exit
  @@ run "rig_nv_pci.rpc"
       [
         group ~timeout:20. "records"
           [
             test_records;
             test_message;
             test "a GSP message is checked" test_not_a_message;
           ];
         group ~timeout:10. "faults"
           [
             test "a stopped channel is a fault naming it" test_rc;
             test "a stopped channel of an unknown error is a fault"
               test_rc_unknown;
             test "a queued MMU fault is a fault" test_mmu;
             test_no_fault;
           ];
         group ~timeout:10. "calls"
           [
             test "an allocation" test_alloc;
             test "a control" test_control;
             test_answer;
             test "a short answer is refused" test_short_answer;
             test "a page directory" test_page_directory;
             test "unloading" test_unloading;
             test_registry;
           ];
         group ~timeout:10. "sequences" [ test_sequence; test_sequence_refused ];
       ]
