(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Channel methods against NVIDIA's class headers (open-gpu-kernel-modules
   570.144): clc56f.h for the host's methods and pushbuffer headers, clc7c0.h
   for the compute engine's, clc7b5.h for the copy engine's. A packet is read
   back as the method writes it makes, each named by its header. *)

open Windtrap
open Rig_nv_abi
module S = Rig_nv_abi_support

let strf = Printf.sprintf

(* Methods, by their byte addresses *)

let names =
  [
    (0x000, "SET_OBJECT");
    (0x05c, "SEM_ADDR_LO");
    (0x060, "SEM_ADDR_HI");
    (0x064, "SEM_PAYLOAD_LO");
    (0x068, "SEM_PAYLOAD_HI");
    (0x06c, "SEM_EXECUTE");
    (0x020, "NON_STALL_INTERRUPT");
    (0x110, "WAIT_FOR_IDLE");
    (0x2a0, "SET_SHADER_SHARED_MEMORY_WINDOW_A");
    (0x2a4, "SET_SHADER_SHARED_MEMORY_WINDOW_B");
    (0x2b4, "SEND_PCAS_A");
    (0x2c0, "SEND_SIGNALING_PCAS2_B");
    (0x2e4, "SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A");
    (0x2e8, "SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_B");
    (0x2ec, "SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_C");
    (0x790, "SET_SHADER_LOCAL_MEMORY_A");
    (0x794, "SET_SHADER_LOCAL_MEMORY_B");
    (0x7b0, "SET_SHADER_LOCAL_MEMORY_WINDOW_A");
    (0x7b4, "SET_SHADER_LOCAL_MEMORY_WINDOW_B");
    (0x1698, "INVALIDATE_SHADER_CACHES_NO_WFI");
    (0x240, "SET_SEMAPHORE_A");
    (0x244, "SET_SEMAPHORE_B");
    (0x248, "SET_SEMAPHORE_PAYLOAD");
    (0x24c, "SET_SEMAPHORE_PAYLOAD_UPPER");
    (0x300, "LAUNCH_DMA");
    (0x400, "OFFSET_IN_UPPER");
    (0x404, "OFFSET_IN_LOWER");
    (0x408, "OFFSET_OUT_UPPER");
    (0x40c, "OFFSET_OUT_LOWER");
    (0x410, "PITCH_IN");
    (0x414, "PITCH_OUT");
    (0x418, "LINE_LENGTH_IN");
    (0x41c, "LINE_COUNT");
    (0x704, "SET_REMAP_CONST_B");
    (0x708, "SET_REMAP_COMPONENTS");
  ]

(* A host method, between SET_OBJECT and 0x100, runs on any subchannel, so its
   subchannel is read as [any]. *)
let any = -1
let host m = m > 0 && m < 0x100

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

let name m =
  match List.assoc_opt m names with Some n -> n | None -> strf "0x%x" m

let pp_write ppf (sub, m, v) =
  Format.fprintf ppf "%s %s 0x%08x"
    (if sub = any then "host" else strf "subch %d" sub)
    (name m) v

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
  | Box of int64 * int64 * int64 * int64 * int64 * int64
  | Fill of Method.component * int64 * int64 * int64
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
  | Box (d, s, dp, sp, rows, n) ->
      Format.fprintf ppf
        "box ~dst:0x%Lx ~src:0x%Lx ~dst_pitch:0x%Lx ~src_pitch:0x%Lx \
         ~rows:0x%Lx 0x%Lx"
        d s dp sp rows n
  | Fill (c, d, v, n) ->
      Format.fprintf ppf "fill %s ~dst:0x%Lx 0x%Lx 0x%Lx"
        (match c with One -> "One" | Four -> "Four")
        d v n
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
  | Box (d, s, dp, sp, rows, n) -> [ d; s; dp; sp; rows; n ]
  | Fill (_, d, v, n) -> [ d; v; n ]

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
  | Box _ ->
      Method.box ~dst:(a 0) ~src:(a 1) ~dst_pitch:(a 2) ~src_pitch:(a 3)
        ~rows:(a 4) (a 5)
  | Fill (c, _, _, _) -> Method.fill c ~dst:(a 0) (a 1) (a 2)
  | Copy_release _ -> Method.copy_release (a 0) (a 1)
  | Copy_release_stamp _ -> Method.copy_release_stamp (a 0) (a 1)

let packet = encode ~slots:false

let calls =
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
    Wait_for_idle;
    Schedule 0x12_3456_7800L;
    Copy (0x1_0000_0000L, 0x2_0000_0000L, 0x8000_0000L);
    Box (0x1_0000_0000L, 0x2_0000_0000L, 0x300L, 0x500L, 0xffff_ffffL, 0x100L);
    Fill (Four, 0x1_0000_0004L, 0x6161_6161L, 0x2000_0000L);
    Fill (One, 0x1_0000_0001L, 0x61L, 3L);
    Copy_release (0x12_3456_7808L, 7L);
    Copy_release_stamp (0x12_3456_7810L, 7L);
  ]

let operations =
  group ~timeout:10. "operations"
    [
      cases ~name:(Format.asprintf "%a" pp_call)
        "an operation's slots, filled, are its words with its operands" calls
        (fun c ->
          equal string (packet c).bytes
            (Template.fill (List.nth (args c)) (encode ~slots:true c)));
      test "a copy's row and a fill take at most 2^31 bytes" (fun () ->
          equal int (1 lsl 31) Method.max_copy);
    ]

(* The words of each operation, by method name *)

let listing () =
  let show c =
    String.concat "\n"
      (Format.asprintf "%a" pp_call c
      :: List.map (Format.asprintf "  %a" pp_write) (read (packet c)))
  in
  String.concat "\n" (List.map show calls)

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
            wait_for_idle
              subch 1 WAIT_FOR_IDLE 0x00000000
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
            box ~dst:0x100000000 ~src:0x200000000 ~dst_pitch:0x300 ~src_pitch:0x500 ~rows:0xffffffff 0x100
              subch 4 OFFSET_IN_UPPER 0x00000002
              subch 4 OFFSET_IN_LOWER 0x00000000
              subch 4 OFFSET_OUT_UPPER 0x00000001
              subch 4 OFFSET_OUT_LOWER 0x00000000
              subch 4 PITCH_IN 0x00000500
              subch 4 PITCH_OUT 0x00000300
              subch 4 LINE_LENGTH_IN 0x00000100
              subch 4 LINE_COUNT 0xffffffff
              subch 4 LAUNCH_DMA 0x00000382
            fill Four ~dst:0x100000004 0x61616161 0x20000000
              subch 4 SET_REMAP_CONST_B 0x61616161
              subch 4 SET_REMAP_COMPONENTS 0x00030005
              subch 4 OFFSET_OUT_UPPER 0x00000001
              subch 4 OFFSET_OUT_LOWER 0x00000004
              subch 4 LINE_LENGTH_IN 0x20000000
              subch 4 LAUNCH_DMA 0x00000582
            fill One ~dst:0x100000001 0x61 0x3
              subch 4 SET_REMAP_CONST_B 0x00000061
              subch 4 SET_REMAP_COMPONENTS 0x00000005
              subch 4 OFFSET_OUT_UPPER 0x00000001
              subch 4 OFFSET_OUT_LOWER 0x00000001
              subch 4 LINE_LENGTH_IN 0x00000003
              subch 4 LAUNCH_DMA 0x00000582
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
