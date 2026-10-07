(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Packet
module D = Defs

type engine = Compute | Copy

(* The subchannels: the host's methods run on any, and the engines are bound
   where NVK binds them (Mesa 25.2, nv_push.h:65-78). *)
let host = 0
let subchannel = function Compute -> 1 | Copy -> 4

(* [v] in the field [f] of a word. *)
let set (f : D.field) v = v lsl f.lo

(* [ws] as the arguments of incrementing methods from [m] on subchannel [s]: a
   header naming [m] and counting the words, then the words. *)
let methods s m ws =
  let header =
    set D.nvc56f_dma_sec_op D.nvc56f_dma_sec_op_inc_method
    lor set D.nvc56f_dma_method_count (size ws)
    lor set D.nvc56f_dma_method_subchannel s
    lor set D.nvc56f_dma_method_address (m lsr 2)
  in
  Dword header :: ws

(* An address as two words, its high word first. *)
let hi_lo addr = [ W32 (Shift (Value addr, 32)); W32 (Value addr) ]
let set_object e cls = methods (subchannel e) D.nvc56f_set_object [ Dword cls ]

(* Semaphores *)

(* SEM_ADDR_LO to SEM_EXECUTE: the address, the 64-bit payload, the
   operation. *)
let semaphore addr v op =
  let size =
    set D.nvc56f_sem_execute_payload_size
      D.nvc56f_sem_execute_payload_size_64bit
  in
  methods host D.nvc56f_sem_addr_lo
    [ W64 (Value addr); W64 (Value v); Dword (size lor op) ]

let acquire addr v =
  semaphore addr v
    (set D.nvc56f_sem_execute_operation
       D.nvc56f_sem_execute_operation_acq_circ_geq)

(* A release that waits for idle makes the channel's writes visible to the
   system first, so it serves both scopes. *)
let release_op = function
  | Agent | System ->
      set D.nvc56f_sem_execute_operation D.nvc56f_sem_execute_operation_release
      lor set D.nvc56f_sem_execute_release_wfi
            D.nvc56f_sem_execute_release_wfi_en

let release s addr v = semaphore addr v (release_op s)

let release_stamp s addr v =
  semaphore addr v
    (release_op s
    lor set D.nvc56f_sem_execute_release_timestamp
          D.nvc56f_sem_execute_release_timestamp_en)

(* NON_STALL_INTERRUPT (0x20), one word on the host's subchannel. Its header is
   written out, 0x20010008 by [methods]'s rule, so that the value is static data
   the library builds nothing for at initialisation. *)
let interrupt = [ Dword 0x2001_0008; Dword 0 ]

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
  methods (subchannel Compute) D.nvc7c0_set_shader_local_memory_a (hi_lo addr)
  @ methods (subchannel Compute)
      D.nvc7c0_set_shader_local_memory_non_throttled_a
      (hi_lo per_tpc @ [ Dword all_sms ])

(* Both scopes invalidate every cache until the lighter [Agent] form is
   measured. *)
let invalidate_caches = function
  | Agent | System ->
      methods (subchannel Compute) D.nvc7c0_invalidate_shader_caches_no_wfi
        [
          Dword
            (set D.nvc7c0_invalidate_shader_caches_no_wfi_instruction
               D.nvc7c0_invalidate_shader_caches_no_wfi_instruction_true
            lor set D.nvc7c0_invalidate_shader_caches_no_wfi_global_data
                  D.nvc7c0_invalidate_shader_caches_no_wfi_global_data_true
            lor set D.nvc7c0_invalidate_shader_caches_no_wfi_constant
                  D.nvc7c0_invalidate_shader_caches_no_wfi_constant_true);
        ]

(* SEND_PCAS_A takes the descriptor's address shifted right by 8, as its field
   QMD_ADDRESS_SHIFTED8 says. *)
let qmd_address_shift = 8

let schedule addr =
  methods (subchannel Compute) D.nvc7c0_send_pcas_a
    [ W32 (Shift (Value addr, qmd_address_shift)) ]
  @ methods (subchannel Compute) D.nvc7c0_send_signaling_pcas2_b
      [
        Dword
          (set D.nvc7c0_send_signaling_pcas2_b_pcas_action
             D.nvc7c0_send_signaling_pcas2_b_pcas_action_prefetch_schedule);
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
  methods (subchannel Copy) D.nvc7b5_offset_in_upper (hi_lo src @ hi_lo dst)
  @ methods (subchannel Copy) D.nvc7b5_line_length_in [ W32 (Value n) ]
  @ methods (subchannel Copy) D.nvc7b5_launch_dma [ Dword launch ]

(* SET_SEMAPHORE_A to SET_SEMAPHORE_PAYLOAD_UPPER, then a launch that writes the
   64-bit payload once the earlier copies are complete and flushed. A flush to
   the system serves both scopes. *)
let copy_semaphore s addr v kind =
  let flush =
    match s with
    | Agent | System ->
        set D.nvc7b5_launch_dma_flush_enable
          D.nvc7b5_launch_dma_flush_enable_true
        lor set D.nvc7b5_launch_dma_flush_type
              D.nvc7b5_launch_dma_flush_type_sys
  in
  methods (subchannel Copy) D.nvc7b5_set_semaphore_a
    (hi_lo addr @ [ W64 (Value v) ])
  @ methods (subchannel Copy) D.nvc7b5_launch_dma
      [
        Dword
          (flush
          lor set D.nvc7b5_launch_dma_semaphore_type kind
          lor set D.nvc7b5_launch_dma_semaphore_payload_size
                D.nvc7b5_launch_dma_semaphore_payload_size_two_word);
      ]

let copy_release s addr v =
  copy_semaphore s addr v
    D.nvc7b5_launch_dma_semaphore_type_release_one_word_semaphore

let copy_release_stamp s addr v =
  copy_semaphore s addr v
    D.nvc7b5_launch_dma_semaphore_type_release_four_word_semaphore
