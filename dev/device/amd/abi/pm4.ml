(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Packet

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let major (g : Gpu.t) =
  let m, _, _ = g.gc in
  m

let gc_name (g : Gpu.t) =
  let a, b, c = g.gc in
  strf "GC %d.%d.%d" a b c

(* PACKET3: type 3, the opcode, and the words of the body less one. *)
let packet op body =
  let n = size body - 1 in
  Dword
    ((Defs.packet_type3 lsl 30)
    lor ((op land 0xff) lsl 8)
    lor ((n land 0x3fff) lsl 16))
  :: body

(* Memory and registers *)

type 'v location = Register of int | Memory of 'v

(* A register's words are at most a 16-bit offset past UCONFIG's start. *)
let uconfig_extent = 0xffff

let set_reg reg ws =
  let op, start, stop =
    if Defs.packet3_set_sh_reg_start <= reg && reg < Defs.packet3_set_sh_reg_end
    then
      ( Defs.packet3_set_sh_reg,
        Defs.packet3_set_sh_reg_start,
        Defs.packet3_set_sh_reg_end )
    else if
      Defs.packet3_set_uconfig_reg_start <= reg
      && reg < Defs.packet3_set_uconfig_reg_start + uconfig_extent
    then
      ( Defs.packet3_set_uconfig_reg,
        Defs.packet3_set_uconfig_reg_start,
        Defs.packet3_set_uconfig_reg_start + uconfig_extent )
    else
      invalid_argf
        "Pm4.set_reg: register 0x%x is in neither the SH nor the UCONFIG range"
        reg
  in
  let n = size ws in
  if reg + n > stop then
    invalid_argf
      "Pm4.set_reg: %d words from register 0x%x pass its range's end 0x%x" n reg
      stop;
  packet op (Dword (reg - start) :: ws)

let at = function
  | Memory a -> [ W64 (Value a) ]
  | Register r -> [ Dword r; Dword 0 ]

let write_data loc v =
  let control =
    match loc with
    | Register _ ->
        Defs.wr_one_addr
        lor Defs.packet3_write_data__dst_sel__mem_mapped_register
            lsl Defs.write_data_dst_sel
    | Memory _ ->
        Defs.wr_confirm
        lor Defs.packet3_write_data__dst_sel__memory lsl Defs.write_data_dst_sel
  in
  packet Defs.packet3_write_data ((Dword control :: at loc) @ [ W32 (Value v) ])

type write = Posted | Confirmed
type source = Counter of int | Clock

let copy_data write source addr =
  let confirm =
    match write with
    | Posted -> 0
    | Confirmed -> Defs.packet3_copy_data__wr_confirm__wait_for_confirmation
  in
  let sel, count, reg =
    match source with
    | Counter r -> (Defs.packet3_copy_data__src_sel__perfcounters, 0, r)
    | Clock ->
        ( Defs.packet3_copy_data__src_sel__gpu_clock_count,
          Defs.packet3_copy_data__count_sel__64_bits_of_data,
          0 )
  in
  let control =
    (sel lsl Defs.packet3_copy_data__src_sel)
    lor Defs.packet3_copy_data__dst_sel__tc_l2
        lsl Defs.packet3_copy_data__dst_sel
    lor (count lsl Defs.packet3_copy_data__count_sel)
    lor (confirm lsl Defs.packet3_copy_data__wr_confirm)
  in
  packet Defs.packet3_copy_data
    [ Dword control; Dword reg; Dword 0; W64 (Value addr) ]

(* Waits, caches and signals *)

(* The poll interval a wait takes by default, in clocks of its poll timer. *)
let default_interval = 4

(* The spaces a wait reads: registers, or memory. *)
let register_space = 0
let memory_space = 1

let function_of = function
  | Equal -> Defs.packet3_wait_reg_mem__function__equal_to_the_reference_value
  | Greater_equal ->
      Defs.packet3_wait_reg_mem__function__greater_than_or_equal_reference_value

let wait_control space cmp =
  (space lsl Defs.wait_reg_mem_mem_space)
  lor (function_of cmp lsl Defs.wait_reg_mem_function)

(* GFX9's wait addresses a UCONFIG register from UCONFIG's start. *)
let wait g loc cmp v ?(mask = 0xffff_ffff) ?(interval = default_interval) () =
  let loc =
    match loc with
    | Register r when major g = 9 && r >= Defs.packet3_set_uconfig_reg_start ->
        Register (r - Defs.packet3_set_uconfig_reg_start)
    | loc -> loc
  in
  let space =
    match loc with Memory _ -> memory_space | Register _ -> register_space
  in
  packet Defs.packet3_wait_reg_mem
    ((Dword (wait_control space cmp) :: at loc)
    @ [ W32 (Value v); Dword mask; Dword interval ])

let wait_64 g addr cmp v ?(interval = default_interval) () =
  if major g = 9 then
    invalid_argf "Pm4.wait_64: %s has no 64-bit wait" (gc_name g);
  packet Defs.packet3_wait_reg_mem64
    [
      Dword (wait_control memory_space cmp);
      W64 (Value addr);
      W64 (Value v);
      Dword 0xffff_ffff;
      Dword 0xffff_ffff;
      Dword interval;
    ]

(* An acquire's address range, all of memory: its size and base words. *)
let everything = [ Dword 0xffff_ffff; Dword 0xffff_ffff; Dword 0; Dword 0 ]

(* The poll interval of GFX9's coherence wait, as the kernel driver sets it. *)
let coherence_poll = 0xA

(* At [Agent], the scalar, vector and L1 caches; at [System], the instruction
   cache and the L2 too, written back first. *)
let acquire_mem g scope =
  let all = match scope with Agent -> 0 | System -> 1 in
  if major g <> 9 then
    let cntl =
      (all lsl Defs.packet3_acquire_mem_gcr_cntl_gli_inv)
      lor (1 lsl Defs.packet3_acquire_mem_gcr_cntl_glm_inv)
      lor (1 lsl Defs.packet3_acquire_mem_gcr_cntl_glm_wb)
      lor (1 lsl Defs.packet3_acquire_mem_gcr_cntl_glk_inv)
      lor (1 lsl Defs.packet3_acquire_mem_gcr_cntl_glk_wb)
      lor (1 lsl Defs.packet3_acquire_mem_gcr_cntl_glv_inv)
      lor (1 lsl Defs.packet3_acquire_mem_gcr_cntl_gl1_inv)
      lor (all lsl Defs.packet3_acquire_mem_gcr_cntl_gl2_inv)
      lor (all lsl Defs.packet3_acquire_mem_gcr_cntl_gl2_wb)
    in
    packet Defs.packet3_acquire_mem
      ((Dword 0 :: everything) @ [ Dword 0; Dword cntl ])
  else
    let coher =
      (all lsl Defs.packet3_acquire_mem_cp_coher_cntl_sh_icache_action_ena)
      lor (1 lsl Defs.packet3_acquire_mem_cp_coher_cntl_sh_kcache_action_ena)
      lor (all lsl Defs.packet3_acquire_mem_cp_coher_cntl_tc_action_ena)
      lor (1 lsl Defs.packet3_acquire_mem_cp_coher_cntl_tcl1_action_ena)
      lor (all lsl Defs.packet3_acquire_mem_cp_coher_cntl_tc_wb_action_ena)
    in
    packet Defs.packet3_acquire_mem
      ((Dword coher :: everything) @ [ Dword coherence_poll ])

type 'v data = Low_32 of 'v | Data_64 of 'v

(* Both scopes write the L2 back and invalidate every cache: [System]'s promise,
   and more than [Agent] needs, which wants no L2 operation. *)
let release_caches g = function
  | Agent | System ->
      if major g = 9 then Defs.eop_tc_wb_action_en lor Defs.eop_tc_nc_action_en
      else
        Defs.packet3_release_mem_gcr_glv_inv
        lor Defs.packet3_release_mem_gcr_gl1_inv
        lor Defs.packet3_release_mem_gcr_gl2_inv
        lor Defs.packet3_release_mem_gcr_glm_wb
        lor Defs.packet3_release_mem_gcr_glm_inv
        lor Defs.packet3_release_mem_gcr_gl2_wb
        lor Defs.packet3_release_mem_gcr_seq

(* The release's destination, DST_SEL, is 0: memory. *)
let release_mem g scope ?interrupt addr d =
  let event =
    (Defs.cache_flush_and_inv_ts_event lsl Defs.event_type)
    lor (Defs.event_index__mec_release_mem__end_of_pipe lsl Defs.event_index)
  in
  let sel, v =
    match d with
    | Low_32 v -> (Defs.data_sel__mec_release_mem__send_32_bit_low, v)
    | Data_64 v -> (Defs.data_sel__mec_release_mem__send_64_bit_data, v)
  in
  let int_sel, ctxid =
    match interrupt with
    | None -> (Defs.int_sel__mec_release_mem__none, 0)
    | Some id ->
        (Defs.int_sel__mec_release_mem__send_interrupt_after_write_confirm, id)
  in
  packet Defs.packet3_release_mem
    [
      Dword (event lor release_caches g scope);
      Dword ((sel lsl Defs.data_sel) lor (int_sel lsl Defs.int_sel));
      W64 (Value addr);
      W64 (Value v);
      Dword ctxid;
    ]

type event = Cs_partial_flush | Thread_trace_marker | Thread_trace_finish

(* The event index a partial flush takes. *)
let partial_flush_index = 4

let event_write e =
  let event, index =
    match e with
    | Cs_partial_flush -> (Defs.cs_partial_flush, partial_flush_index)
    | Thread_trace_marker -> (Defs.thread_trace_marker, 0)
    | Thread_trace_finish -> (Defs.thread_trace_finish, 0)
  in
  packet Defs.packet3_event_write
    [ Dword ((event lsl Defs.event_type) lor (index lsl Defs.event_index)) ]

(* Control *)

(* PRED_EXEC's die mask, the virtual XCC select from bit 24, and its 14-bit
   count of words. *)
let xcc_select = 24
let max_xcc_mask = 0xff
let max_predicated = 0x3fff

let pred_exec ~xcc_mask p =
  if xcc_mask < 0 || xcc_mask > max_xcc_mask then
    invalid_argf "Pm4.pred_exec: xcc_mask 0x%x, expected 0 to 0xff" xcc_mask;
  let n = size p in
  if n > max_predicated then
    invalid_argf "Pm4.pred_exec: %d words, expected at most 16383" n;
  packet Defs.packet3_pred_exec [ Dword ((xcc_mask lsl xcc_select) lor n) ] @ p

(* IB_SIZE is 20 bits; bit 20 is CHAIN. *)
let max_indirect = 0xf_ffff

let indirect_buffer addr ~dwords =
  if dwords < 0 || dwords > max_indirect then
    invalid_argf "Pm4.indirect_buffer: %d words, expected 0 to 1048575" dwords;
  packet Defs.packet3_indirect_buffer
    [ W64 (Value addr); Dword (dwords lor Defs.indirect_buffer_valid) ]

(* Runs *)

(* COMPUTE_PGM_LO and DISPATCH_SCRATCH_BASE_LO take 256-byte aligned addresses,
   from their bit 8, with their _HI words after them. *)
let address_shift = 8

(* COMPUTE_RESOURCE_LIMITS.WAVES_PER_SH: a 10-bit field, whose 0 sets no
   limit. *)
let no_wave_limit = 0
let max_waves_per_array = 0x3ff

(* GFX11 runs kernels privileged, for their context save and restore:
   COMPUTE_PGM_RSRC1.PRIV. *)
let priv = 1 lsl 20

(* COMPUTE_PGM_RSRC2.LDS_SIZE: its first bit and its width; its granule, 1280
   bytes on GFX950 and 512 on GFX9 to GFX12 (AMDGPUUsage, LDS_SIZE; LLVM's
   AMDGPU.td at 52c11435, FeatureISAVersion9_5_Common and the generations'
   FeatureLDSEncodingGranularity). *)
let lds_shift = 15
let lds_mask = 0x1ff
let lds_granule_gfx950 = 1280
let lds_granule = 512

(* The scratch's buffer descriptor in a kernel's first user SGPRs: the base
   address with SWIZZLE_ENABLE, bit 63, the most records, and a word of its
   format and lane stride. No primary source gives these words: ROCR-Runtime
   dispatches through AQL, whose queue descriptor the CP hands kernels
   ({!Scratch.descriptor}). These are the words tinygrad's PM4 dispatch writes
   (tinygrad/runtime/ops_amd.py), unverified on hardware; only GFX9 and GFX10
   kernels read them. *)
let swizzle_enable = Int64.min_int
let num_records = 0xffff_ffff
let scratch_format = 0x20c14000

let dispatch (g : Gpu.t) (k : Code_object.kernel) ~program ~scratch ~args
    ~packet:dispatch_packet ~threads:(tx, ty, tz) ~groups:(gx, gy, gz)
    ?waves_per_array () =
  let fn = "Pm4.dispatch" in
  let d : Defs.dispatch =
    match Defs.dispatch (Defs.gc g.gc) with
    | Some d -> d
    | None -> invalid_argf "%s: %s has no register of a dispatch" fn (gc_name g)
  in
  let limits =
    match waves_per_array with
    | None -> no_wave_limit
    | Some n when n < 1 || n > max_waves_per_array ->
        invalid_argf "%s: waves_per_array %d, expected 1 to 1023" fn n
    | Some n -> n lsl fst d.waves_per_sh
  in
  let initiator = if k.wave32 then d.initiator_wave32 else d.initiator_wave64 in
  let rsrc1 = if major g = 11 then k.rsrc1 lor priv else k.rsrc1 in
  let granule =
    match g.gc with 9, 5, _ -> lds_granule_gfx950 | _ -> lds_granule
  in
  let lds = (k.group_segment + granule - 1) / granule land lds_mask in
  (* The user SGPRs the descriptor enables, in their fixed order. *)
  let user =
    (if k.private_segment_buffer then
       [
         W64 (Or (Value scratch, swizzle_enable));
         Dword num_records;
         Dword scratch_format;
       ]
     else [])
    @ (if k.dispatch_ptr then [ W64 (Value dispatch_packet) ] else [])
    @ [ W64 (Value args) ]
  in
  let zeros n = List.init n (fun _ -> Dword 0) in
  set_reg d.pgm_lo [ W64 (Shift (Value program, address_shift)) ]
  @ set_reg d.pgm_rsrc1 [ Dword rsrc1; Dword (k.rsrc2 lor (lds lsl lds_shift)) ]
  @ set_reg d.pgm_rsrc3 [ Dword k.rsrc3 ]
  @ set_reg d.tmpring_size [ Dword (Scratch.tmpring g k.private_segment) ]
  @ set_reg d.scratch_base_lo [ W64 (Shift (Value scratch, address_shift)) ]
  @ set_reg d.restart_x (zeros 3)
  @ set_reg d.user_data_0 user
  @ set_reg d.resource_limits [ Dword limits ]
  @ set_reg d.start_x
      (zeros 3 @ [ W32 (Value tx); W32 (Value ty); W32 (Value tz) ] @ zeros 2)
  @ packet Defs.packet3_dispatch_direct
      [ W32 (Value gx); W32 (Value gy); W32 (Value gz); Dword initiator ]

let run g p = acquire_mem g Agent @ p @ event_write Cs_partial_flush
