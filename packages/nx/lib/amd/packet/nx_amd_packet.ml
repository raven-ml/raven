(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Packet_defs

(* Words *)

type 'v term =
  | Value of 'v
  | Add of 'v term * int64
  | Shift of 'v term * int
  | Or of 'v term * int64

let rec eval = function
  | Value v -> Int64.of_int v
  | Add (t, n) -> Int64.add (eval t) n
  | Shift (t, n) -> Int64.shift_right_logical (eval t) n
  | Or (t, n) -> Int64.logor (eval t) n

type 'v word = Dword of int | W32 of 'v term | W64 of 'v term

let w32 v = W32 (Value v)
let w64 v = W64 (Value v)
let mask32 = 0xffff_ffff

(* The 32-bit word of [n] from bit [at]. *)
let word32 n at = Int64.to_int (Int64.shift_right_logical n at) land mask32

let dwords ws =
  List.concat_map
    (function
      | Dword n -> [ n land mask32 ]
      | W32 t -> [ word32 (eval t) 0 ]
      | W64 t ->
          let n = eval t in
          [ word32 n 0; word32 n 32 ])
    ws

let size ws =
  List.fold_left
    (fun n w -> n + match w with Dword _ | W32 _ -> 1 | W64 _ -> 2)
    0 ws

type version = int * int * int
type comparison = Equal | Greater_equal
type 'v location = Register of int | Memory of 'v

let major (m, _, _) = m

(* The encodings of a comparison, which SDMA's polls share. *)
let function_of = function
  | Equal -> D.packet3_wait_reg_mem__function__equal_to_the_reference_value
  | Greater_equal ->
      D.packet3_wait_reg_mem__function__greater_than_or_equal_reference_value

(* GC registers *)

module Gc = struct
  type register = D.register = {
    name : string;
    offset : int;
    segment : int;
    fields : (string * (int * int)) list;
  }

  let tables = D.gc_registers

  let family gc =
    List.fold_left
      (fun best (v, _) ->
        if major v = major gc && compare v gc <= 0 then
          match best with Some b when compare b v >= 0 -> best | _ -> Some v
        else best)
      None tables

  let registers gc =
    match family gc with Some v -> List.assoc v tables | None -> []

  let find gc name = List.find_opt (fun r -> r.name = name) (registers gc)

  let address gc r =
    let bases =
      List.fold_left
        (fun acc (m, b) -> if m <= major gc then b else acc)
        [||] D.gc_bases
    in
    if r.segment >= Array.length bases then
      invalid_arg
        (Printf.sprintf "Nx_amd_packet.Gc.address: no base for GC %d" (major gc));
    bases.(r.segment) + r.offset

  let encode r fs =
    List.fold_left
      (fun w (f, v) ->
        match List.assoc_opt f r.fields with
        | Some (lo, hi) -> w lor ((v land ((1 lsl (hi - lo + 1)) - 1)) lsl lo)
        | None ->
            invalid_arg
              (Printf.sprintf "Nx_amd_packet.Gc.encode: %s has no field %s"
                 r.name f))
      0 fs

  module Thread_trace = struct
    let rt_freq_4096_clk = D.sq_tt_rt_freq_4096_clk
    let wtype_include_cs_bit = D.sq_tt_wtype_include_cs_bit
    let token_mask_sqdec_bit = D.sq_tt_token_mask_sqdec_bit
    let token_mask_shdec_bit = D.sq_tt_token_mask_shdec_bit
    let token_mask_gfxudec_bit = D.sq_tt_token_mask_gfxudec_bit
    let token_mask_comp_bit = D.sq_tt_token_mask_comp_bit
    let token_mask_context_bit = D.sq_tt_token_mask_context_bit
    let token_exclude_vmemexec_shift = D.sq_tt_token_exclude_vmemexec_shift
    let token_exclude_aluexec_shift = D.sq_tt_token_exclude_aluexec_shift
    let token_exclude_valuinst_shift = D.sq_tt_token_exclude_valuinst_shift
    let token_exclude_immediate_shift = D.sq_tt_token_exclude_immediate_shift
    let token_exclude_inst_shift = D.sq_tt_token_exclude_inst_shift
  end

  let lanes = 64
  let min_per_lane = 128

  let tmpring_size ~gc ~compute_units ~slots ~shader_engines ~xccs n =
    let r =
      match find gc "regCOMPUTE_TMPRING_SIZE" with
      | Some r -> r
      | None ->
          invalid_arg
            (Printf.sprintf
               "Nx_amd_packet.Gc.tmpring_size: GC %d has no \
                COMPUTE_TMPRING_SIZE"
               (major gc))
    in
    let gfx9 = major gc = 9 in
    let align = if gfx9 then 1024 else 256 in
    let unit = align / lanes in
    let per_lane = (Int.max n min_per_lane + unit - 1) / unit * unit in
    let per_xcc = per_lane * lanes * slots * compute_units in
    let wave = ((lanes * per_lane) + align - 1) / align in
    let waves = per_xcc / (wave * align) / if gfx9 then 1 else shader_engines in
    encode r
      [
        ("waves", Int.min waves (compute_units * slots * xccs));
        ("wavesize", wave);
      ]
end

(* PM4 *)

module Pm4 = struct
  let header op n =
    (D.packet_type3 lsl 30)
    lor ((op land 0xff) lsl 8)
    lor ((n land 0x3fff) lsl 16)

  let packet op body = Dword (header op (size body - 1)) :: body

  (* A register's words are at most a 16-bit offset past UCONFIG's start. *)
  let uconfig_extent = 0xffff

  let set_reg reg ws =
    let op, start =
      if D.packet3_set_sh_reg_start <= reg && reg < D.packet3_set_sh_reg_end
      then (D.packet3_set_sh_reg, D.packet3_set_sh_reg_start)
      else if
        D.packet3_set_uconfig_reg_start <= reg
        && reg < D.packet3_set_uconfig_reg_start + uconfig_extent
      then (D.packet3_set_uconfig_reg, D.packet3_set_uconfig_reg_start)
      else
        invalid_arg
          (Printf.sprintf
             "Nx_amd_packet.Pm4.set_reg: no PM4 packet sets the register 0x%x"
             reg)
    in
    packet op (Dword (reg - start) :: ws)

  let at = function Memory a -> [ w64 a ] | Register r -> [ Dword r; Dword 0 ]

  (* GFX9's wait addresses a UCONFIG register from UCONFIG's start. *)
  let wait ~gc loc cmp v ~mask ~interval =
    let loc =
      match loc with
      | Register r when major gc = 9 && r >= D.packet3_set_uconfig_reg_start ->
          Register (r - D.packet3_set_uconfig_reg_start)
      | loc -> loc
    in
    let space = match loc with Memory _ -> 1 | Register _ -> 0 in
    let info =
      (space lsl D.wait_reg_mem_mem_space)
      lor (function_of cmp lsl D.wait_reg_mem_function)
    in
    packet D.packet3_wait_reg_mem
      ((Dword info :: at loc) @ [ w32 v; Dword mask; Dword interval ])

  let all_bits = Dword 0xffff_ffff

  (* GFX9's coherence control and GFX10's cache control, in their words. *)
  type caches = Data_caches | All_caches

  let acquire_mem ~gc caches =
    let all = match caches with Data_caches -> 0 | All_caches -> 1 in
    let i = all and l = all in
    let everything = [ all_bits; all_bits; Dword 0; Dword 0 ] in
    if major gc <> 9 then
      let flags =
        (i lsl D.packet3_acquire_mem_gcr_cntl_gli_inv)
        lor (1 lsl D.packet3_acquire_mem_gcr_cntl_glm_inv)
        lor (1 lsl D.packet3_acquire_mem_gcr_cntl_glm_wb)
        lor (1 lsl D.packet3_acquire_mem_gcr_cntl_glk_inv)
        lor (1 lsl D.packet3_acquire_mem_gcr_cntl_glk_wb)
        lor (1 lsl D.packet3_acquire_mem_gcr_cntl_glv_inv)
        lor (1 lsl D.packet3_acquire_mem_gcr_cntl_gl1_inv)
        lor (l lsl D.packet3_acquire_mem_gcr_cntl_gl2_inv)
        lor (l lsl D.packet3_acquire_mem_gcr_cntl_gl2_wb)
      in
      packet D.packet3_acquire_mem
        ((Dword 0 :: everything) @ [ Dword 0; Dword flags ])
    else
      let coher =
        (i lsl D.packet3_acquire_mem_cp_coher_cntl_sh_icache_action_ena)
        lor (1 lsl D.packet3_acquire_mem_cp_coher_cntl_sh_kcache_action_ena)
        lor (l lsl D.packet3_acquire_mem_cp_coher_cntl_tc_action_ena)
        lor (1 lsl D.packet3_acquire_mem_cp_coher_cntl_tcl1_action_ena)
        lor (l lsl D.packet3_acquire_mem_cp_coher_cntl_tc_wb_action_ena)
      in
      (* The poll interval of the coherence wait, as the kernel driver sets
         it. *)
      let poll = 0xA in
      packet D.packet3_acquire_mem ((Dword coher :: everything) @ [ Dword poll ])

  type 'v data = Low_32 of 'v | Data_64 of 'v

  let release_mem ~gc addr d =
    let flags =
      if major gc <> 9 then
        D.packet3_release_mem_gcr_glv_inv lor D.packet3_release_mem_gcr_gl1_inv
        lor D.packet3_release_mem_gcr_gl2_inv
        lor D.packet3_release_mem_gcr_glm_wb
        lor D.packet3_release_mem_gcr_glm_inv
        lor D.packet3_release_mem_gcr_gl2_wb lor D.packet3_release_mem_gcr_seq
      else D.eop_tc_wb_action_en lor D.eop_tc_nc_action_en
    in
    let event =
      (D.cache_flush_and_inv_ts_event lsl D.event_type)
      lor (D.event_index__mec_release_mem__end_of_pipe lsl D.event_index)
    in
    let sel, data =
      match d with
      | Low_32 v -> (D.data_sel__mec_release_mem__send_32_bit_low, [ w64 v ])
      | Data_64 v -> (D.data_sel__mec_release_mem__send_64_bit_data, [ w64 v ])
    in
    let int_sel =
      D.int_sel__mec_release_mem__send_interrupt_after_write_confirm
    in
    (* Its destination, DST_SEL, is 0: memory. *)
    let memsel = (sel lsl D.data_sel) lor (int_sel lsl D.int_sel) in
    packet D.packet3_release_mem
      ((Dword (event lor flags) :: Dword memsel :: w64 addr :: data)
      @ [ Dword 0 ])

  (* The die mask is the virtual XCC select, from bit 24. *)
  let xcc_select = 24

  let pred_exec ~xcc_mask ~dwords =
    packet D.packet3_pred_exec [ Dword ((xcc_mask lsl xcc_select) lor dwords) ]

  type event = Cs_partial_flush | Thread_trace_marker | Thread_trace_finish

  (* The event index a partial flush takes. *)
  let event_index_partial_flush = 4

  let event_write e =
    let event, index =
      match e with
      | Cs_partial_flush -> (D.cs_partial_flush, event_index_partial_flush)
      | Thread_trace_marker -> (D.thread_trace_marker, 0)
      | Thread_trace_finish -> (D.thread_trace_finish, 0)
    in
    packet D.packet3_event_write
      [ Dword ((event lsl D.event_type) lor (index lsl D.event_index)) ]

  type write = Posted | Confirmed
  type source = Counter of int | Clock

  let copy_data write source addr =
    let confirm =
      match write with
      | Posted -> 0
      | Confirmed -> D.packet3_copy_data__wr_confirm__wait_for_confirmation
    in
    let sel, count, reg =
      match source with
      | Counter r -> (D.packet3_copy_data__src_sel__perfcounters, 0, r)
      | Clock ->
          ( D.packet3_copy_data__src_sel__gpu_clock_count,
            D.packet3_copy_data__count_sel__64_bits_of_data,
            0 )
    in
    let control =
      (sel lsl D.packet3_copy_data__src_sel)
      lor (D.packet3_copy_data__dst_sel__tc_l2 lsl D.packet3_copy_data__dst_sel)
      lor (count lsl D.packet3_copy_data__count_sel)
      lor (confirm lsl D.packet3_copy_data__wr_confirm)
    in
    packet D.packet3_copy_data [ Dword control; Dword reg; Dword 0; w64 addr ]

  let write_data loc v =
    let control =
      match loc with
      | Register _ ->
          D.wr_one_addr
          lor D.packet3_write_data__dst_sel__mem_mapped_register
              lsl D.write_data_dst_sel
      | Memory _ ->
          D.wr_confirm
          lor (D.packet3_write_data__dst_sel__memory lsl D.write_data_dst_sel)
    in
    packet D.packet3_write_data ((Dword control :: at loc) @ [ w32 v ])

  let indirect_buffer addr ~dwords =
    packet D.packet3_indirect_buffer
      [ w64 addr; Dword (dwords lor D.indirect_buffer_valid) ]

  let register gc name =
    match Gc.find gc name with
    | Some r -> Gc.address gc r
    | None ->
        invalid_arg
          (Printf.sprintf "Nx_amd_packet.Pm4: GC %d has no %s" (major gc) name)

  (* Guards the tables of what encoders find once per GC version. *)
  let lock = Mutex.create ()

  (* The registers take 256-byte aligned addresses, from their bit 8. *)
  let address_shift = 8

  (* The registers a dispatch sets, found once per GC version: encoders read
     them for each dispatch. *)
  type compute = {
    pgm_lo : int;
    pgm_rsrc1 : int;
    pgm_rsrc3 : int;
    tmpring_size : int;
    scratch_base_lo : int;
    restart_x : int;
    user_data_0 : int;
    resource_limits : int;
    start_x : int;
  }

  let computes : (version, compute) Hashtbl.t = Hashtbl.create 4

  let compute gc =
    Mutex.lock lock;
    let known = Hashtbl.find_opt computes gc in
    Mutex.unlock lock;
    match known with
    | Some c -> c
    | None ->
        let r name = register gc ("regCOMPUTE_" ^ name) in
        let c =
          {
            pgm_lo = r "PGM_LO";
            pgm_rsrc1 = r "PGM_RSRC1";
            pgm_rsrc3 = r "PGM_RSRC3";
            tmpring_size = r "TMPRING_SIZE";
            scratch_base_lo = r "DISPATCH_SCRATCH_BASE_LO";
            restart_x = r "RESTART_X";
            user_data_0 = r "USER_DATA_0";
            resource_limits = r "RESOURCE_LIMITS";
            start_x = r "START_X";
          }
        in
        Mutex.protect lock (fun () -> Hashtbl.replace computes gc c);
        c

  let set_program ~gc addr =
    set_reg (compute gc).pgm_lo [ W64 (Shift (Value addr, address_shift)) ]

  let set_scratch ~gc addr =
    set_reg (compute gc).scratch_base_lo
      [ W64 (Shift (Value addr, address_shift)) ]

  let initiator = "regCOMPUTE_DISPATCH_INITIATOR"

  type wave = Wave32 | Wave64

  (* The initiator word of a GC version's dispatches of a wave size, encoded
     once per pair. *)
  let initiators : (version * wave, int) Hashtbl.t = Hashtbl.create 4

  let initiator_word ~gc wave =
    Mutex.lock lock;
    let known = Hashtbl.find_opt initiators (gc, wave) in
    Mutex.unlock lock;
    match known with
    | Some w -> w
    | None ->
        let r =
          match Gc.find gc initiator with
          | Some r -> r
          | None ->
              invalid_arg
                (Printf.sprintf
                   "Nx_amd_packet.Pm4.dispatch_direct: GC %d has no %s"
                   (major gc) initiator)
        in
        let wave32 = match wave with Wave32 -> 1 | Wave64 -> 0 in
        let lanes = if major gc = 9 then [] else [ ("cs_w32_en", wave32) ] in
        let w =
          Gc.encode r
            (lanes @ [ ("force_start_at_000", 1); ("compute_shader_en", 1) ])
        in
        Mutex.protect lock (fun () -> Hashtbl.replace initiators (gc, wave) w);
        w

  let dispatch_direct ~gc wave (x, y, z) =
    packet D.packet3_dispatch_direct
      [ w32 x; w32 y; w32 z; Dword (initiator_word ~gc wave) ]

  (* GFX11 runs kernels privileged, for their context save and restore:
     COMPUTE_PGM_RSRC1.PRIV. *)
  let priv = 1 lsl 20

  (* COMPUTE_PGM_RSRC2.LDS_SIZE: its first bit, its width, and its granule. *)
  let lds_shift = 15
  let lds_mask = 0x1ff
  let lds_granule = 512

  (* The buffer descriptor of a kernel's scratch, as HSA's runtime makes it: the
     base address with SWIZZLE_ENABLE, bit 63, the most records, and the word of
     its format and lane stride. *)
  let swizzle_enable = Int64.min_int
  let num_records = 0xffff_ffff
  let scratch_format = 0x20c14000

  let dispatch ~gc (k : Nx_amd_code_object.kernel) ~program ~scratch ~packet
      ~args ~tmpring ~limits ~threads:(tx, ty, tz) ~groups =
    let c = compute gc in
    let rsrc1 = if major gc = 11 then k.rsrc1 lor priv else k.rsrc1 in
    let lds = (k.group_segment + lds_granule - 1) / lds_granule land lds_mask in
    let zeros n = List.init n (fun _ -> Dword 0) in
    (* The user SGPRs the descriptor enables, in their fixed order. *)
    let user =
      (if k.private_segment_buffer then
         [
           W64 (Or (Value scratch, swizzle_enable));
           Dword num_records;
           Dword scratch_format;
         ]
       else [])
      @ (if k.dispatch_ptr then [ w64 packet ] else [])
      @ [ w64 args ]
    in
    set_program ~gc program
    @ set_reg c.pgm_rsrc1
        [ Dword rsrc1; Dword (k.rsrc2 lor (lds lsl lds_shift)) ]
    @ set_reg c.pgm_rsrc3 [ Dword k.rsrc3 ]
    @ set_reg c.tmpring_size [ Dword tmpring ]
    @ set_scratch ~gc scratch
    @ set_reg c.restart_x (zeros 3)
    @ set_reg c.user_data_0 user
    @ set_reg c.resource_limits [ Dword limits ]
    @ set_reg c.start_x (zeros 3 @ [ w32 tx; w32 ty; w32 tz ] @ zeros 2)
    @ dispatch_direct ~gc (if k.wave32 then Wave32 else Wave64) groups
end

(* AQL *)

module Aql = struct
  module P = D.Dispatch

  let header =
    (1 lsl D.hsa_packet_header_barrier)
    lor (D.hsa_fence_scope_system lsl D.hsa_packet_header_scacquire_fence_scope)
    lor (D.hsa_fence_scope_system lsl D.hsa_packet_header_screlease_fence_scope)

  (* The words of the [P.sizeof] bytes [b], each [(offset, word)] of [holes] in
     place of the bytes it covers. *)
  let words b holes =
    let rec go off =
      if off >= P.sizeof then []
      else
        match List.assoc_opt off holes with
        | Some (W64 _ as w) -> w :: go (off + 8)
        | Some w -> w :: go (off + 4)
        | None ->
            Dword (Int32.to_int (Bytes.get_int32_le b off) land 0xffff_ffff)
            :: go (off + 4)
    in
    go 0

  let set b (off, width) v =
    match width with
    | 2 -> Bytes.set_uint16_le b off v
    | _ -> Bytes.set_int32_le b off (Int32.of_int v)

  (* The packet's three dimensions, which a dispatch always gives. *)
  let dimensions = 3

  let dispatch ~threads:(tx, ty, tz) ~grid:(gx, gy, gz) ~private_segment
      ~group_segment ~descriptor ~args =
    let b = Bytes.make P.sizeof '\000' in
    set b P.header
      (header
      lor (D.hsa_packet_type_kernel_dispatch lsl D.hsa_packet_header_type));
    set b P.setup (dimensions lsl D.hsa_kernel_dispatch_packet_setup_dimensions);
    set b P.workgroup_size_x tx;
    set b P.workgroup_size_y ty;
    set b P.workgroup_size_z tz;
    set b P.private_segment_size private_segment;
    set b P.group_segment_size group_segment;
    words b
      [
        (fst P.grid_size_x, w32 gx);
        (fst P.grid_size_y, w32 gy);
        (fst P.grid_size_z, w32 gz);
        (fst P.kernel_object, w64 descriptor);
        (fst P.kernarg_address, w64 args);
      ]

  (* ROCr's vendor packet of PM4 commands (amd_aql_pm4_ib_packet_t): its format,
     and the words left after its four of PM4. *)
  let format_pm4_ib = 1
  let dw_count_remain = 10

  let indirect_buffer addr ~dwords =
    let hdr =
      header
      lor (D.hsa_packet_type_vendor_specific lsl D.hsa_packet_header_type)
      lor (format_pm4_ib lsl 16)
    in
    (Dword hdr :: Pm4.indirect_buffer addr ~dwords)
    @ (Dword dw_count_remain :: List.init dw_count_remain (fun _ -> Dword 0))
end

(* SDMA *)

module Sdma = struct
  let field (mask, shift) v = (v land mask) lsl shift

  (* The copy count's width: 30 bits from SDMA 4.4.2 below 5 and from 5.2, 22
     bits otherwise. *)
  let max_copy v =
    if
      (compare (4, 4, 2) v <= 0 && compare v (5, 0, 0) < 0)
      || compare v (5, 2, 0) >= 0
    then 1 lsl 30
    else 1 lsl 22

  (* One linear copy per [max_copy] bytes, each at its offset into both
     buffers. *)
  let copy ~sdma ~dst ~src n =
    if n < 0 then
      invalid_arg (Printf.sprintf "Nx_amd_packet.Sdma.copy: %d bytes" n);
    let max = max_copy sdma in
    List.concat
      (List.init
         ((n + max - 1) / max)
         (fun i ->
           let off = i * max in
           let at v =
             if off = 0 then Value v else Add (Value v, Int64.of_int off)
           in
           [
             Dword
               (D.sdma_op_copy
               lor field D.sdma_pkt_copy_linear_header_sub_op
                     D.sdma_subop_copy_linear);
             Dword (Int.min max (n - off) - 1);
             Dword 0;
             W64 (at src);
             W64 (at dst);
           ]))

  (* The interval and retries of a poll, as the kernel driver sets them. *)
  let interval = 0x04
  let retries = 0xfff

  let poll addr cmp v ~mask =
    [
      Dword
        (D.sdma_op_poll_regmem
        lor field D.sdma_pkt_poll_regmem_header_func (function_of cmp)
        lor field D.sdma_pkt_poll_regmem_header_mem_poll 1);
      w64 addr;
      w32 v;
      Dword mask;
      Dword
        (field D.sdma_pkt_poll_regmem_dw5_interval interval
        lor field D.sdma_pkt_poll_regmem_dw5_retry_count retries);
    ]

  (* The uncached memory type, MTYPE_UC. *)
  let mtype_uc = 3

  let fence ~sdma addr v =
    let mtype =
      if compare sdma D.sdma_fence_mtype_from >= 0 then
        field D.sdma_pkt_fence_header_mtype mtype_uc
      else 0
    in
    [ Dword (D.sdma_op_fence lor mtype); w64 addr; w32 v ]

  let trap = [ Dword D.sdma_op_trap; Dword 0 ]

  let timestamp addr =
    [
      Dword
        (D.sdma_op_timestamp
        lor field D.sdma_pkt_timestamp_get_global_header_sub_op
              D.sdma_subop_timestamp_get_global);
      w64 addr;
    ]
end
