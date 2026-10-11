(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Channel methods against NVIDIA's class headers (open-gpu-kernel-modules
   570.144): clc56f.h for the host's methods and pushbuffer headers, clc7c0.h
   for the compute engine's, clc7b5.h for the copy engine's. A packet is read
   back as the method writes it makes; each operation is checked against the
   fields its header defines. *)

open Windtrap
open Rig_nv_abi
module S = Rig_nv_abi_support

let strf = Printf.sprintf

(* Methods, by their byte addresses *)

let set_object = 0x000
let sem_addr_lo = 0x05c
let sem_addr_hi = 0x060
let sem_payload_lo = 0x064
let sem_payload_hi = 0x068
let sem_execute = 0x06c
let non_stall_interrupt = 0x020
let wait_for_idle = 0x110
let shared_window_a = 0x2a0
let shared_window_b = 0x2a4
let send_pcas_a = 0x2b4
let send_signaling_pcas2_b = 0x2c0
let local_non_throttled_a = 0x2e4
let local_non_throttled_b = 0x2e8
let local_non_throttled_c = 0x2ec
let local_memory_a = 0x790
let local_memory_b = 0x794
let local_window_a = 0x7b0
let local_window_b = 0x7b4
let invalidate_no_wfi = 0x1698
let set_semaphore_a = 0x240
let set_semaphore_b = 0x244
let set_semaphore_payload = 0x248
let set_semaphore_payload_upper = 0x24c
let launch_dma = 0x300
let offset_in_upper = 0x400
let offset_in_lower = 0x404
let offset_out_upper = 0x408
let offset_out_lower = 0x40c
let line_length_in = 0x418

let names =
  [
    (set_object, "SET_OBJECT");
    (sem_addr_lo, "SEM_ADDR_LO");
    (sem_addr_hi, "SEM_ADDR_HI");
    (sem_payload_lo, "SEM_PAYLOAD_LO");
    (sem_payload_hi, "SEM_PAYLOAD_HI");
    (sem_execute, "SEM_EXECUTE");
    (non_stall_interrupt, "NON_STALL_INTERRUPT");
    (wait_for_idle, "WAIT_FOR_IDLE");
    (shared_window_a, "SET_SHADER_SHARED_MEMORY_WINDOW_A");
    (shared_window_b, "SET_SHADER_SHARED_MEMORY_WINDOW_B");
    (send_pcas_a, "SEND_PCAS_A");
    (send_signaling_pcas2_b, "SEND_SIGNALING_PCAS2_B");
    (local_non_throttled_a, "SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A");
    (local_non_throttled_b, "SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_B");
    (local_non_throttled_c, "SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_C");
    (local_memory_a, "SET_SHADER_LOCAL_MEMORY_A");
    (local_memory_b, "SET_SHADER_LOCAL_MEMORY_B");
    (local_window_a, "SET_SHADER_LOCAL_MEMORY_WINDOW_A");
    (local_window_b, "SET_SHADER_LOCAL_MEMORY_WINDOW_B");
    (invalidate_no_wfi, "INVALIDATE_SHADER_CACHES_NO_WFI");
    (set_semaphore_a, "SET_SEMAPHORE_A");
    (set_semaphore_b, "SET_SEMAPHORE_B");
    (set_semaphore_payload, "SET_SEMAPHORE_PAYLOAD");
    (set_semaphore_payload_upper, "SET_SEMAPHORE_PAYLOAD_UPPER");
    (launch_dma, "LAUNCH_DMA");
    (offset_in_upper, "OFFSET_IN_UPPER");
    (offset_in_lower, "OFFSET_IN_LOWER");
    (offset_out_upper, "OFFSET_OUT_UPPER");
    (offset_out_lower, "OFFSET_OUT_LOWER");
    (line_length_in, "LINE_LENGTH_IN");
  ]

(* The subchannels the .mli binds: compute on 1, copy on 4. A host method runs
   on any subchannel, so its subchannel is read as [any]; SET_OBJECT binds the
   one it is sent on. *)
let compute = 1
let copy = 4
let any = -1
let host m = m > set_object && m < 0x100

(* Reading words back *)

(* The method writes of [ws]: each header (DMA_METHOD_ADDRESS 11:0 in words,
   SUBCHANNEL 15:13, COUNT 28:16, SEC_OP 31:29) is an incrementing method
   (SEC_OP_INC_METHOD 1) whose [count] arguments go to consecutive methods. *)
let rec writes = function
  | [] -> []
  | h :: rest ->
      if h lsr 29 <> 1 then failf "0x%08x is no incrementing method header" h;
      let n = (h lsr 16) land 0x1fff and sub = (h lsr 13) land 7 in
      if n = 0 || List.length rest < n then
        failf "a header of %d words before %d words" n (List.length rest);
      let m = (h land 0xfff) lsl 2 in
      List.mapi
        (fun i v ->
          let m = m + (4 * i) in
          ((if host m then any else sub), m, v))
        (List.filteri (fun i _ -> i < n) rest)
      @ writes (List.filteri (fun i _ -> i >= n) rest)

let read (p : Template.t) = writes (S.words p.bytes)
let lo v = Int64.to_int v land 0xffff_ffff
let hi v = Int64.to_int (Int64.shift_right_logical v 32) land 0xffff_ffff
let bits v lo n = (v lsr lo) land ((1 lsl n) - 1)

let name m =
  match List.assoc_opt m names with Some n -> n | None -> strf "0x%x" m

let pp_write ppf (sub, m, v) =
  Format.fprintf ppf "%s %s 0x%08x"
    (if sub = any then "host" else strf "subch %d" sub)
    (name m) v

let write = Testable.make ~pp:pp_write ~equal:( = )

(* Operations *)

type call =
  | Set_object of Method.engine * int
  | Acquire of int64 * int64
  | Release of int64 * int64
  | Release_stamp of int64 * int64
  | Interrupt
  | Shared_window of int64
  | Local_window of int64
  | Local_memory of int64 * int64
  | Invalidate
  | Wait_for_idle
  | Schedule of int64
  | Copy of int64 * int64 * int64
  | Copy_release of int64 * int64
  | Copy_release_stamp of int64 * int64

let engine_name : Method.engine -> string = function
  | Compute -> "Compute"
  | Copy -> "Copy"

let pp_call ppf = function
  | Set_object (e, c) ->
      Format.fprintf ppf "set_object %s 0x%x" (engine_name e) c
  | Acquire (a, v) -> Format.fprintf ppf "acquire 0x%Lx 0x%Lx" a v
  | Release (a, v) -> Format.fprintf ppf "release 0x%Lx 0x%Lx" a v
  | Release_stamp (a, v) -> Format.fprintf ppf "release_stamp 0x%Lx 0x%Lx" a v
  | Interrupt -> Format.fprintf ppf "interrupt"
  | Shared_window a -> Format.fprintf ppf "shared_memory_window 0x%Lx" a
  | Local_window a -> Format.fprintf ppf "local_memory_window 0x%Lx" a
  | Local_memory (a, p) ->
      Format.fprintf ppf "local_memory 0x%Lx ~per_tpc:0x%Lx" a p
  | Invalidate -> Format.fprintf ppf "invalidate_caches"
  | Wait_for_idle -> Format.fprintf ppf "wait_for_idle"
  | Schedule a -> Format.fprintf ppf "schedule 0x%Lx" a
  | Copy (d, s, n) ->
      Format.fprintf ppf "copy ~dst:0x%Lx ~src:0x%Lx 0x%Lx" d s n
  | Copy_release (a, v) -> Format.fprintf ppf "copy_release 0x%Lx 0x%Lx" a v
  | Copy_release_stamp (a, v) ->
      Format.fprintf ppf "copy_release_stamp 0x%Lx 0x%Lx" a v

(* The operands of [c], in order. *)
let args = function
  | Set_object _ | Interrupt | Invalidate | Wait_for_idle -> []
  | Shared_window a | Local_window a | Schedule a -> [ a ]
  | Acquire (a, v)
  | Release (a, v)
  | Release_stamp (a, v)
  | Local_memory (a, v)
  | Copy_release (a, v)
  | Copy_release_stamp (a, v) ->
      [ a; v ]
  | Copy (d, s, n) -> [ d; s; n ]

(* The words of [c], each operand known, or left to the slot of its position if
   [slots]. *)
let encode ~slots c =
  let o i v = if slots then Template.Slot i else Int (Int64.to_int v) in
  let a i = o i (List.nth (args c) i) in
  match c with
  | Set_object (e, c) -> Method.set_object e c
  | Acquire _ -> Method.acquire (a 0) (a 1)
  | Release _ -> Method.release (a 0) (a 1)
  | Release_stamp _ -> Method.release_stamp (a 0) (a 1)
  | Interrupt -> Method.interrupt
  | Shared_window _ -> Method.shared_memory_window (a 0)
  | Local_window _ -> Method.local_memory_window (a 0)
  | Local_memory _ -> Method.local_memory (a 0) ~per_tpc:(a 1)
  | Invalidate -> Method.invalidate_caches
  | Wait_for_idle -> Method.wait_for_idle
  | Schedule _ -> Method.schedule (a 0)
  | Copy _ -> Method.copy ~dst:(a 0) ~src:(a 1) (a 2)
  | Copy_release _ -> Method.copy_release (a 0) (a 1)
  | Copy_release_stamp _ -> Method.copy_release_stamp (a 0) (a 1)

let packet = encode ~slots:false

(* What an operation must write: its operands' methods, in any order, then the
   method that acts on them, whose named fields have the given values. *)
type expected = {
  operands : (int * int * int) list;
  trigger : (int * int) option;
  fields : (string * int * int * int) list;
      (* a trigger's fields: name, low bit, width, value *)
}

let operands operands = { operands; trigger = None; fields = [] }

let semaphore a v =
  [
    (any, sem_addr_lo, lo a);
    (any, sem_addr_hi, hi a);
    (any, sem_payload_lo, lo v);
    (any, sem_payload_hi, hi v);
  ]

(* SEM_EXECUTE: OPERATION 2:0 (RELEASE 1, ACQ_CIRC_GEQ 3), RELEASE_WFI 20:20,
   PAYLOAD_SIZE 24:24 (64BIT 1), RELEASE_TIMESTAMP 25:25. *)
let execute ~op ~wfi ~stamp =
  [
    ("OPERATION", 0, 3, op);
    ("RELEASE_WFI", 20, 1, wfi);
    ("PAYLOAD_SIZE", 24, 1, 1);
    ("RELEASE_TIMESTAMP", 25, 1, stamp);
  ]

let copy_semaphore a v =
  [
    (copy, set_semaphore_a, hi a);
    (copy, set_semaphore_b, lo a);
    (copy, set_semaphore_payload, lo v);
    (copy, set_semaphore_payload_upper, hi v);
  ]

(* LAUNCH_DMA of a release: SEMAPHORE_TYPE 4:3 (ONE_WORD 1, FOUR_WORD 2),
   SEMAPHORE_PAYLOAD_SIZE 27:27 (TWO_WORD 1), FLUSH_ENABLE 2:2, FLUSH_TYPE 25:25
   SYS (0). *)
let copy_release ~kind =
  [
    ("SEMAPHORE_TYPE", 3, 2, kind);
    ("SEMAPHORE_PAYLOAD_SIZE", 27, 1, 1);
    ("FLUSH_ENABLE", 2, 1, 1);
    ("FLUSH_TYPE", 25, 1, 0);
  ]

(* INVALIDATE_SHADER_CACHES_NO_WFI: INSTRUCTION 0:0, GLOBAL_DATA 4:4, CONSTANT
   12:12. *)
let invalidated =
  [ ("INSTRUCTION", 0, 1, 1); ("GLOBAL_DATA", 4, 1, 1); ("CONSTANT", 12, 1, 1) ]

let expected = function
  | Set_object (e, c) ->
      let sub = match e with Compute -> compute | Copy -> copy in
      operands [ (sub, set_object, c) ]
  | Acquire (a, v) ->
      {
        operands = semaphore a v;
        trigger = Some (any, sem_execute);
        fields = execute ~op:3 ~wfi:0 ~stamp:0;
      }
  | Release (a, v) ->
      {
        operands = semaphore a v;
        trigger = Some (any, sem_execute);
        fields = execute ~op:1 ~wfi:1 ~stamp:0;
      }
  | Release_stamp (a, v) ->
      {
        operands = semaphore a v;
        trigger = Some (any, sem_execute);
        fields = execute ~op:1 ~wfi:1 ~stamp:1;
      }
  | Interrupt ->
      { operands = []; trigger = Some (any, non_stall_interrupt); fields = [] }
  | Shared_window a ->
      operands
        [ (compute, shared_window_a, hi a); (compute, shared_window_b, lo a) ]
  | Local_window a ->
      operands
        [ (compute, local_window_a, hi a); (compute, local_window_b, lo a) ]
  | Local_memory (a, p) ->
      operands
        [
          (compute, local_memory_a, hi a);
          (compute, local_memory_b, lo a);
          (compute, local_non_throttled_a, hi p);
          (compute, local_non_throttled_b, lo p);
        ]
  | Invalidate ->
      {
        operands = [];
        trigger = Some (compute, invalidate_no_wfi);
        fields = invalidated;
      }
  | Wait_for_idle ->
      { operands = []; trigger = Some (compute, wait_for_idle); fields = [] }
  | Schedule a ->
      {
        operands = [ (compute, send_pcas_a, Int64.to_int a lsr 8) ];
        trigger = Some (compute, send_signaling_pcas2_b);
        fields = [];
      }
  | Copy (d, s, n) ->
      {
        operands =
          [
            (copy, offset_in_upper, hi s);
            (copy, offset_in_lower, lo s);
            (copy, offset_out_upper, hi d);
            (copy, offset_out_lower, lo d);
            (copy, line_length_in, lo n);
          ];
        trigger = Some (copy, launch_dma);
        (* SEMAPHORE_TYPE NONE, SRC and DST_MEMORY_LAYOUT PITCH 7:7 and 8:8,
           MULTI_LINE_ENABLE 9:9 FALSE: one line of [n] bytes. *)
        fields =
          [
            ("SEMAPHORE_TYPE", 3, 2, 0);
            ("SRC_MEMORY_LAYOUT", 7, 1, 1);
            ("DST_MEMORY_LAYOUT", 8, 1, 1);
            ("MULTI_LINE_ENABLE", 9, 1, 0);
          ];
      }
  | Copy_release (a, v) ->
      {
        operands = copy_semaphore a v;
        trigger = Some (copy, launch_dma);
        fields = copy_release ~kind:1;
      }
  | Copy_release_stamp (a, v) ->
      {
        operands = copy_semaphore a v;
        trigger = Some (copy, launch_dma);
        fields = copy_release ~kind:2;
      }

(* Drawing operations with operands in the ranges the .mli states *)

let window =
  Gen.map
    (fun n -> Int64.of_int ((1 lsl 40) + n))
    (Gen.int_range 0 ((1 lsl 49) - (1 lsl 40) - 1))

(* A 64-bit payload an unsigned int holds. *)
let payload =
  Gen.map (fun v -> Int64.of_int (Int64.to_int v land max_int)) S.u64

let call =
  let open Gen in
  let semaphore = S.address ~bits:40 ~align:3
  and stamped = S.address ~bits:40 ~align:4
  and wide = S.address ~bits:49 ~align:0 in
  let engine =
    of_list
      ~pp:(fun ppf e -> Format.pp_print_string ppf (engine_name e))
      [ Method.Compute; Copy ]
  in
  with_pp pp_call
    (one_of
       [
         (let+ e = engine and+ c = int_range 0 0xffff in
          Set_object (e, c));
         (let+ a = semaphore and+ v = payload in
          Acquire (a, v));
         (let+ a = semaphore and+ v = payload in
          Release (a, v));
         (let+ a = stamped and+ v = payload in
          Release_stamp (a, v));
         constant Interrupt;
         constant Wait_for_idle;
         map (fun a -> Shared_window a) window;
         map (fun a -> Local_window a) window;
         (let+ a = S.address ~bits:40 ~align:0
          and+ p = S.address ~bits:40 ~align:15 in
          Local_memory (a, p));
         constant Invalidate;
         map (fun a -> Schedule a) (S.address ~bits:40 ~align:8);
         (let+ d = wide
          and+ s = wide
          and+ n =
            frequency
              [
                (3, int_range 0 Method.max_copy);
                (1, of_list ~pp:Format.pp_print_int [ 0; 1; Method.max_copy ]);
              ]
          in
          Copy (d, s, Int64.of_int n));
         (let+ a = semaphore and+ v = payload in
          Copy_release (a, v));
         (let+ a = stamped and+ v = payload in
          Copy_release_stamp (a, v));
       ])

(* The law: an operation's words write its operands, then act on them *)

let check c =
  let e = expected c in
  let ws = read (packet c) in
  let ws, last =
    match e.trigger with
    | None -> (ws, None)
    | Some _ -> (
        match List.rev ws with
        | [] -> fail "an operation that acts writes at least its trigger"
        | t :: rest -> (List.rev rest, Some t))
  in
  (* NON_THROTTLED_C, the most SMs that take the memory, is checked apart. *)
  let c_writes, ws =
    List.partition (fun (_, m, _) -> m = local_non_throttled_c) ws
  in
  (match (c, c_writes) with
  | Local_memory _, [ (sub, _, n) ] ->
      equal ~msg:"NON_THROTTLED_C's subchannel" int compute sub;
      at_least ~msg:"MAX_SM_COUNT, every SM up to 255" int ~than:0xff n
  | Local_memory _, _ -> fail "local_memory writes NON_THROTTLED_C once"
  | _, [] -> ()
  | _, _ -> fail "only local_memory writes NON_THROTTLED_C");
  equal ~msg:"operands" (slist write compare) e.operands ws;
  match (e.trigger, last) with
  | None, _ -> ()
  | Some (sub, m), Some ((sub', m', v) as w) ->
      equal ~msg:"the last method" (pair int string)
        (sub, name m)
        (sub', name m');
      List.iter
        (fun (f, lo, n, x) ->
          equal
            ~msg:(Format.asprintf "%s of %a" f pp_write w)
            int x (bits v lo n))
        e.fields
  | Some _, None -> fail "an operation that acts writes its trigger last"

let constructor = function
  | Set_object _ -> "set_object"
  | Acquire _ -> "acquire"
  | Release _ -> "release"
  | Release_stamp _ -> "release_stamp"
  | Interrupt -> "interrupt"
  | Shared_window _ -> "shared_memory_window"
  | Local_window _ -> "local_memory_window"
  | Local_memory _ -> "local_memory"
  | Invalidate -> "invalidate_caches"
  | Wait_for_idle -> "wait_for_idle"
  | Schedule _ -> "schedule"
  | Copy _ -> "copy"
  | Copy_release _ -> "copy_release"
  | Copy_release_stamp _ -> "copy_release_stamp"

let all =
  [
    "set_object";
    "acquire";
    "release";
    "release_stamp";
    "interrupt";
    "shared_memory_window";
    "local_memory_window";
    "local_memory";
    "invalidate_caches";
    "wait_for_idle";
    "schedule";
    "copy";
    "copy_release";
    "copy_release_stamp";
  ]

let operations =
  group ~timeout:10. "operations"
    [
      prop ~count:500
        "an operation's words write its operands, then the method that acts"
        call (fun c ->
          List.iter (fun n -> cover n (constructor c = n)) all;
          check c);
      prop "an operation's slots, filled, are its words with its operands" call
        (fun c ->
          cover "a slot" (args c <> []);
          equal string (packet c).bytes
            (Template.fill (List.nth (args c)) (encode ~slots:true c)));
      test "a release waits for idle" (fun () ->
          check (Release (0x12_3456_7890L, 0x2_0000_0005L)));
      test "a schedule fetches the descriptor and schedules it" (fun () ->
          (* SEND_SIGNALING_PCAS2_B PCAS_ACTION 3:0: PREFETCH_SCHEDULE 9. *)
          match List.rev (read (Method.schedule (Int 0xff_ffff_ff00))) with
          | (_, m, v) :: _ when m = send_signaling_pcas2_b ->
              equal int 9 (bits v 0 4)
          | _ -> fail "a schedule ends with SEND_SIGNALING_PCAS2_B");
      test "a copy moves at most 2^31 bytes" (fun () ->
          equal int (1 lsl 31) Method.max_copy);
    ]

(* The words of each operation, by method name *)

let listing () =
  let show c =
    String.concat "\n"
      (Format.asprintf "%a" pp_call c
      :: List.map (Format.asprintf "  %a" pp_write) (read (packet c)))
  in
  String.concat "\n"
    (List.map show
       [
         Set_object (Compute, 0xc9c0);
         Set_object (Copy, 0xc9b5);
         Acquire (0x12_3456_7808L, 0x3000_0000_0000_0001L);
         Release (0x12_3456_7808L, 7L);
         Release_stamp (0x12_3456_7810L, 7L);
         Interrupt;
         Shared_window 0x7294_0000_0000L;
         Local_window 0x7293_0000_0000L;
         Local_memory (0x40_0000_0000L, 0x18_8000L);
         Invalidate;
         Schedule 0x12_3456_7800L;
         Copy (0x1_0000_0000L, 0x2_0000_0000L, 0x8000_0000L);
         Copy_release (0x12_3456_7808L, 7L);
         Copy_release_stamp (0x12_3456_7810L, 7L);
       ])

let words =
  group ~timeout:10. "words"
    [
      test "each operation's words" (fun () ->
          expect (listing ())
          @@ __POS_OF__
               {|
            set_object Compute 0xc9c0
              subch 1 SET_OBJECT 0x0000c9c0
            set_object Copy 0xc9b5
              subch 4 SET_OBJECT 0x0000c9b5
            acquire 0x1234567808 0x3000000000000001
              host SEM_ADDR_LO 0x34567808
              host SEM_ADDR_HI 0x00000012
              host SEM_PAYLOAD_LO 0x00000001
              host SEM_PAYLOAD_HI 0x30000000
              host SEM_EXECUTE 0x01000003
            release 0x1234567808 0x7
              host SEM_ADDR_LO 0x34567808
              host SEM_ADDR_HI 0x00000012
              host SEM_PAYLOAD_LO 0x00000007
              host SEM_PAYLOAD_HI 0x00000000
              host SEM_EXECUTE 0x01100001
            release_stamp 0x1234567810 0x7
              host SEM_ADDR_LO 0x34567810
              host SEM_ADDR_HI 0x00000012
              host SEM_PAYLOAD_LO 0x00000007
              host SEM_PAYLOAD_HI 0x00000000
              host SEM_EXECUTE 0x03100001
            interrupt
              host NON_STALL_INTERRUPT 0x00000000
            shared_memory_window 0x729400000000
              subch 1 SET_SHADER_SHARED_MEMORY_WINDOW_A 0x00007294
              subch 1 SET_SHADER_SHARED_MEMORY_WINDOW_B 0x00000000
            local_memory_window 0x729300000000
              subch 1 SET_SHADER_LOCAL_MEMORY_WINDOW_A 0x00007293
              subch 1 SET_SHADER_LOCAL_MEMORY_WINDOW_B 0x00000000
            local_memory 0x4000000000 ~per_tpc:0x188000
              subch 1 SET_SHADER_LOCAL_MEMORY_A 0x00000040
              subch 1 SET_SHADER_LOCAL_MEMORY_B 0x00000000
              subch 1 SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A 0x00000000
              subch 1 SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_B 0x00188000
              subch 1 SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_C 0x000000ff
            invalidate_caches
              subch 1 INVALIDATE_SHADER_CACHES_NO_WFI 0x00001011
            schedule 0x1234567800
              subch 1 SEND_PCAS_A 0x12345678
              subch 1 SEND_SIGNALING_PCAS2_B 0x00000009
            copy ~dst:0x100000000 ~src:0x200000000 0x80000000
              subch 4 OFFSET_IN_UPPER 0x00000002
              subch 4 OFFSET_IN_LOWER 0x00000000
              subch 4 OFFSET_OUT_UPPER 0x00000001
              subch 4 OFFSET_OUT_LOWER 0x00000000
              subch 4 LINE_LENGTH_IN 0x80000000
              subch 4 LAUNCH_DMA 0x00000182
            copy_release 0x1234567808 0x7
              subch 4 SET_SEMAPHORE_A 0x00000012
              subch 4 SET_SEMAPHORE_B 0x34567808
              subch 4 SET_SEMAPHORE_PAYLOAD 0x00000007
              subch 4 SET_SEMAPHORE_PAYLOAD_UPPER 0x00000000
              subch 4 LAUNCH_DMA 0x0800000c
            copy_release_stamp 0x1234567810 0x7
              subch 4 SET_SEMAPHORE_A 0x00000012
              subch 4 SET_SEMAPHORE_B 0x34567810
              subch 4 SET_SEMAPHORE_PAYLOAD 0x00000007
              subch 4 SET_SEMAPHORE_PAYLOAD_UPPER 0x00000000
              subch 4 LAUNCH_DMA 0x08000014
            |});
    ]

let () = exit (run "rig_nv_abi.method" [ operations; words ])
