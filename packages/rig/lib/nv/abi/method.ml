(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The methods are those of NVIDIA's class headers in open-gpu-kernel-modules
   570.144: clc56f.h (the host's methods, pushbuffer headers), clc7c0.h
   (compute) and clc7b5.h (copy). *)

module D = Defs

type engine = Compute | Copy

(* The subchannels: the host's methods run on any, and the engines are bound
   where NVK binds them (Mesa 25.2, nv_push.h:65-78). *)
let host = 0
let subchannel = function Compute -> 1 | Copy -> 4

(* [v] in the field [f] of a word. *)
let set (f : D.field) v = v lsl f.lo

(* A word the layout knows, and an operand's words. *)
let dword n = Template.u32 0 (Int n)
let u32 = Template.u32
let u64 = Template.u64

(* [ws] as the arguments of incrementing methods from [m] on subchannel [s]: a
   header naming [m] and counting the words, then the words. *)
let methods s m ws =
  let ws = Template.concat ws in
  let header =
    set D.nvc56f_dma_sec_op D.nvc56f_dma_sec_op_inc_method
    lor set D.nvc56f_dma_method_count (String.length ws.bytes / 4)
    lor set D.nvc56f_dma_method_subchannel s
    lor set D.nvc56f_dma_method_address (m lsr 2)
  in
  Template.concat [ dword header; ws ]

(* An address as two words, its high word first. *)
let hi_lo addr = [ u32 32 addr; u32 0 addr ]
let set_object e cls = methods (subchannel e) D.nvc56f_set_object [ dword cls ]

(* Semaphores *)

(* SEM_ADDR_LO to SEM_EXECUTE: the address, the 64-bit payload, the
   operation. *)
let semaphore addr v op =
  let size =
    set D.nvc56f_sem_execute_payload_size
      D.nvc56f_sem_execute_payload_size_64bit
  in
  methods host D.nvc56f_sem_addr_lo
    [ u64 0L addr; u64 0L v; dword (size lor op) ]

let acquire addr v =
  semaphore addr v
    (set D.nvc56f_sem_execute_operation
       D.nvc56f_sem_execute_operation_acq_circ_geq)

(* A release that waits for idle makes the channel's writes visible to the
   system first. *)
let release_op =
  set D.nvc56f_sem_execute_operation D.nvc56f_sem_execute_operation_release
  lor set D.nvc56f_sem_execute_release_wfi D.nvc56f_sem_execute_release_wfi_en

let release addr v = semaphore addr v release_op

let release_stamp addr v =
  semaphore addr v
    (release_op
    lor set D.nvc56f_sem_execute_release_timestamp
          D.nvc56f_sem_execute_release_timestamp_en)

let interrupt = methods host D.nvc56f_non_stall_interrupt [ dword 0 ]

(* Compute *)

let shared_memory_window addr =
  methods (subchannel Compute) D.nvc7c0_set_shader_shared_memory_window_a
    (hi_lo addr)

let local_memory_window addr =
  methods (subchannel Compute) D.nvc7c0_set_shader_local_memory_window_a
    (hi_lo addr)

(* The most multiprocessors the memory serves: all of them, as NVK writes it
   (Mesa 25.2, nvk_queue.c:147). *)
let all_sms = 0xff

let local_memory addr ~per_tpc =
  Template.concat
    [
      methods (subchannel Compute) D.nvc7c0_set_shader_local_memory_a
        (hi_lo addr);
      methods (subchannel Compute)
        D.nvc7c0_set_shader_local_memory_non_throttled_a
        (hi_lo per_tpc @ [ dword all_sms ]);
    ]

let invalidate_caches =
  methods (subchannel Compute) D.nvc7c0_invalidate_shader_caches_no_wfi
    [
      dword
        (set D.nvc7c0_invalidate_shader_caches_no_wfi_instruction
           D.nvc7c0_invalidate_shader_caches_no_wfi_instruction_true
        lor set D.nvc7c0_invalidate_shader_caches_no_wfi_global_data
              D.nvc7c0_invalidate_shader_caches_no_wfi_global_data_true
        lor set D.nvc7c0_invalidate_shader_caches_no_wfi_constant
              D.nvc7c0_invalidate_shader_caches_no_wfi_constant_true);
    ]

(* WAIT_FOR_IDLE (0x110), one word on the compute subchannel. *)
let wait_for_idle =
  methods (subchannel Compute) D.nvc7c0_wait_for_idle [ dword 0 ]

(* SEND_PCAS_A takes the descriptor's address shifted right by 8, as its field
   QMD_ADDRESS_SHIFTED8 says. *)
let qmd_address_shift = 8

let schedule addr =
  let action =
    set D.nvc7c0_send_signaling_pcas2_b_pcas_action
      D.nvc7c0_send_signaling_pcas2_b_pcas_action_prefetch_schedule
  in
  Template.concat
    [
      methods (subchannel Compute) D.nvc7c0_send_pcas_a
        [ u32 qmd_address_shift addr ];
      methods (subchannel Compute) D.nvc7c0_send_signaling_pcas2_b
        [ dword action ];
    ]

(* Copies *)

let max_copy = 1 lsl 31

let copy ~dst ~src n =
  let launch =
    set D.nvc7b5_launch_dma_data_transfer_type
      D.nvc7b5_launch_dma_data_transfer_type_non_pipelined
    lor set D.nvc7b5_launch_dma_src_memory_layout
          D.nvc7b5_launch_dma_src_memory_layout_pitch
    lor set D.nvc7b5_launch_dma_dst_memory_layout
          D.nvc7b5_launch_dma_dst_memory_layout_pitch
  in
  Template.concat
    [
      methods (subchannel Copy) D.nvc7b5_offset_in_upper (hi_lo src @ hi_lo dst);
      methods (subchannel Copy) D.nvc7b5_line_length_in [ u32 0 n ];
      methods (subchannel Copy) D.nvc7b5_launch_dma [ dword launch ];
    ]

(* SET_SEMAPHORE_A to SET_SEMAPHORE_PAYLOAD_UPPER, then a launch that writes the
   64-bit payload once the earlier copies are complete and flushed to the
   system. *)
let copy_semaphore addr v kind =
  let flush =
    set D.nvc7b5_launch_dma_flush_enable D.nvc7b5_launch_dma_flush_enable_true
    lor set D.nvc7b5_launch_dma_flush_type D.nvc7b5_launch_dma_flush_type_sys
  in
  Template.concat
    [
      methods (subchannel Copy) D.nvc7b5_set_semaphore_a
        (hi_lo addr @ [ u64 0L v ]);
      methods (subchannel Copy) D.nvc7b5_launch_dma
        [
          dword
            (flush
            lor set D.nvc7b5_launch_dma_semaphore_type kind
            lor set D.nvc7b5_launch_dma_semaphore_payload_size
                  D.nvc7b5_launch_dma_semaphore_payload_size_two_word);
        ];
    ]

let copy_release addr v =
  copy_semaphore addr v
    D.nvc7b5_launch_dma_semaphore_type_release_one_word_semaphore

let copy_release_stamp addr v =
  copy_semaphore addr v
    D.nvc7b5_launch_dma_semaphore_type_release_four_word_semaphore
