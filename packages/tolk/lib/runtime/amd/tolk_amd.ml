(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

module Hcq = Hcq
module Amd_tables = Amd_tables
module Compiler_amd = Compiler_amd
module Reg = Amd_tables.Reg
module Ip = Amd_tables.Ip
module Q = Hcq.Q

let major (m, _, _) = m
let lo32 v = Int64.to_int (Int64.logand v 0xFFFFFFFFL)
let hi32 v = Int64.to_int (Int64.shift_right_logical v 32)
let va64 = Int64.of_nativeint
let event_index_partial_flush = 4
let wait_reg_mem_function_eq = 3
let wait_reg_mem_function_geq = 5

(* Devices *)

type queue_event = { event_id : int }

type 'meta device = {
  target : int * int * int;
  xccs : int;
  soc : (module Amd_tables.Soc);
  pm4 : (module Amd_tables.Pm4);
  sdma : (module Amd_tables.Sdma);
  gc : Ip.t;
  nbio : Ip.t;
  max_copy_size : int;
  sqtt_enabled : bool;
  mutable tmpring_size : int;
  mutable scratch : 'meta Hcq.Buffer.t;
  is_am : bool;
  queue_event_mailbox_ptr : nativeint;
  queue_event : queue_event;
}

let device ~target ~xccs ~gc_version ~nbio_version ~sdma_version
    ?(sqtt_enabled = false) ~tmpring_size ~scratch ~is_am
    ~queue_event_mailbox_ptr ~queue_event () =
  let gfx9 = major target = 9 in
  let gc_bases, nbio_bases =
    if gfx9 then
      (Amd_regs_defs.Vega.gc_bases.(0), Amd_regs_defs.Vega.nbio_bases.(0))
    else (Amd_regs_defs.Navi.gc_bases.(0), Amd_regs_defs.Navi.nbio_bases.(0))
  in
  {
    target;
    xccs;
    soc = Amd_tables.soc ~target_major:(major target);
    pm4 = Amd_tables.pm4 ~gfx9;
    sdma = Amd_tables.sdma ~version:sdma_version;
    gc = Ip.create ~name:"gc" ~version:gc_version ~bases:gc_bases;
    nbio =
      Ip.create
        ~name:(if major target < 12 then "nbio" else "nbif")
        ~version:nbio_version ~bases:nbio_bases;
    max_copy_size = (if major sdma_version >= 5 then 0x40000000 else 0x400000);
    sqtt_enabled;
    tmpring_size;
    scratch;
    is_am;
    queue_event_mailbox_ptr;
    queue_event;
  }

(* Programs *)

type 'meta program = {
  dev : 'meta device;
  prog_addr : nativeint;
  rsrc1 : int;
  rsrc2 : int;
  rsrc3 : int;
  wave32 : bool;
  enable_private_segment_sgpr : bool;
  enable_dispatch_ptr : bool;
}

(* Compute queue *)

module Compute_queue = struct
  type 'meta t = { dev : 'meta device; q : Q.t }

  let wait_reg_mem_function_eq = wait_reg_mem_function_eq
  let wait_reg_mem_function_geq = wait_reg_mem_function_geq
  let create dev = { dev; q = Q.create () }
  let q t = t.q

  let pkt3 t op payload =
    let module P = (val t.dev.pm4) in
    Q.push t.q (P.packet3 op (Array.length payload - 1));
    for i = 0 to Array.length payload - 1 do
      Q.push t.q (Array.unsafe_get payload i)
    done

  let wreg t (reg : Reg.t) vals =
    let module P = (val t.dev.pm4) in
    let set_packet, set_packet_start =
      if
        P.packet3_set_sh_reg_start <= reg.addr
        && reg.addr < P.packet3_set_sh_reg_end
      then (P.packet3_set_sh_reg, P.packet3_set_sh_reg_start)
      else if
        P.packet3_set_uconfig_reg_start <= reg.addr
        && reg.addr < P.packet3_set_uconfig_reg_start + 0xffff
      then (P.packet3_set_uconfig_reg, P.packet3_set_uconfig_reg_start)
      else
        invalid_arg
          (Printf.sprintf "cannot set %s (0x%x) via a pm4 packet" reg.name
             reg.addr)
    in
    Q.push t.q (P.packet3 set_packet (Array.length vals));
    Q.push t.q (reg.addr - set_packet_start);
    for i = 0 to Array.length vals - 1 do
      Q.push t.q (Array.unsafe_get vals i)
    done

  let wreg_fields t reg fields = wreg t reg [| Reg.encode reg fields |]

  (* Predication brackets a run of commands: [pred_open] emits the packet and
     returns the stream position its body starts at (or -1 on single-die
     devices, where no packet is emitted), and [pred_close] back-patches the
     packet with the number of dwords it predicates, known only once the body
     has been emitted. *)
  let pred_open t ~xcc_mask =
    if t.dev.xccs > 1 then begin
      let module P = (val t.dev.pm4) in
      pkt3 t P.packet3_pred_exec [| xcc_mask lsl 24 |];
      Q.length t.q
    end
    else -1

  let pred_close t start =
    if start >= 0 then
      Q.set t.q (start - 1) (Q.get t.q (start - 1) lor (Q.length t.q - start))

  let pred_exec t ~xcc_mask f =
    let start = pred_open t ~xcc_mask in
    f ();
    pred_close t start

  let wait_reg_mem t ?(mask = 0xffffffff) ?mem ?(reg = 0) ?(reg_done = 0)
      ?(op = wait_reg_mem_function_geq) value =
    let module P = (val t.dev.pm4) in
    let wrm_info =
      P.wait_reg_mem_mem_space (match mem with Some _ -> 1 | None -> 0)
      lor P.wait_reg_mem_operation
            (match mem with None when reg_done > 0 -> 1 | _ -> 0)
      lor P.wait_reg_mem_function op
      lor P.wait_reg_mem_engine 0
    in
    match mem with
    | Some addr ->
        let a = va64 addr in
        pkt3 t P.packet3_wait_reg_mem
          [| wrm_info; lo32 a; hi32 a; value; mask; 4 |]
    | None ->
        pkt3 t P.packet3_wait_reg_mem
          [| wrm_info; reg; reg_done; value; mask; 4 |]

  let acquire_mem t ?(addr = 0n) ?(sz = Int64.minus_one) ?(gli = 1) ?(glm = 1)
      ?(glk = 1) ?(glv = 1) ?(gl1 = 1) ?(gl2 = 1) () =
    let module P = (val t.dev.pm4) in
    let a = va64 addr in
    if major t.dev.target <> 9 then begin
      let module N = Amd_pm4_defs.Nv in
      let cache_flags_dw =
        N.packet3_acquire_mem_gcr_cntl_gli_inv gli
        lor N.packet3_acquire_mem_gcr_cntl_glm_inv glm
        lor N.packet3_acquire_mem_gcr_cntl_glm_wb glm
        lor N.packet3_acquire_mem_gcr_cntl_glk_inv glk
        lor N.packet3_acquire_mem_gcr_cntl_glk_wb glk
        lor N.packet3_acquire_mem_gcr_cntl_glv_inv glv
        lor N.packet3_acquire_mem_gcr_cntl_gl1_inv gl1
        lor N.packet3_acquire_mem_gcr_cntl_gl2_inv gl2
        lor N.packet3_acquire_mem_gcr_cntl_gl2_wb gl2
      in
      pkt3 t P.packet3_acquire_mem
        [| 0; lo32 sz; hi32 sz; lo32 a; hi32 a; 0; cache_flags_dw |]
    end
    else begin
      let module S = Amd_pm4_defs.Soc15 in
      let cp_coher_cntl =
        S.packet3_acquire_mem_cp_coher_cntl_sh_icache_action_ena gli
        lor S.packet3_acquire_mem_cp_coher_cntl_sh_kcache_action_ena glk
        lor S.packet3_acquire_mem_cp_coher_cntl_tc_action_ena gl2
        lor S.packet3_acquire_mem_cp_coher_cntl_tcl1_action_ena gl1
        lor S.packet3_acquire_mem_cp_coher_cntl_tc_wb_action_ena gl2
      in
      pkt3 t P.packet3_acquire_mem
        [| cp_coher_cntl; lo32 sz; hi32 sz; lo32 a; hi32 a; 0x0000000A |]
    end

  let release_mem t ?(address = 0n) ?(value = 0L) ?(data_sel = 0) ?(int_sel = 2)
      ?(ctxid = 0) ?(cache_flush = false) () =
    let module P = (val t.dev.pm4) in
    let event_dw, memsel_dw, ctxid =
      if major t.dev.target <> 9 then
        let module N = Amd_pm4_defs.Nv in
        let cache_flags_dw =
          if not cache_flush then 0
          else
            N.packet3_release_mem_gcr_glv_inv
            lor N.packet3_release_mem_gcr_gl1_inv
            lor N.packet3_release_mem_gcr_gl2_inv
            lor N.packet3_release_mem_gcr_glm_wb
            lor N.packet3_release_mem_gcr_glm_inv
            lor N.packet3_release_mem_gcr_gl2_wb
            lor N.packet3_release_mem_gcr_seq
        in
        ( N.packet3_release_mem_event_type P.cache_flush_and_inv_ts_event
          lor N.packet3_release_mem_event_index
                P.event_index__mec_release_mem__end_of_pipe
          lor cache_flags_dw,
          N.packet3_release_mem_data_sel data_sel
          lor N.packet3_release_mem_int_sel int_sel
          lor N.packet3_release_mem_dst_sel 0,
          ctxid )
      else
        let module S = Amd_pm4_defs.Soc15 in
        let cache_flags_dw =
          if not cache_flush then 0
          else S.eop_tc_wb_action_en lor S.eop_tc_nc_action_en
        in
        ( S.event_type P.cache_flush_and_inv_ts_event
          lor S.event_index P.event_index__mec_release_mem__end_of_pipe
          lor cache_flags_dw,
          S.data_sel data_sel lor S.int_sel int_sel,
          0 )
    in
    let a = va64 address in
    pkt3 t P.packet3_release_mem
      [| event_dw; memsel_dw; lo32 a; hi32 a; lo32 value; hi32 value; ctxid |]

  let memory_barrier t =
    let pf =
      let ma, mi, _ = Ip.version t.dev.nbio in
      if ma = 7 && mi = 11 then "1" else "0"
    in
    let req : Reg.t = Ip.reg t.dev.nbio ("regBIF_BX_PF" ^ pf ^ "_GPU_HDP_FLUSH_REQ") in
    let done_ : Reg.t =
      Ip.reg t.dev.nbio ("regBIF_BX_PF" ^ pf ^ "_GPU_HDP_FLUSH_DONE")
    in
    wait_reg_mem t ~reg:req.addr ~reg_done:done_.addr 0xffffffff;
    acquire_mem t ()

  let exec t (prg : 'meta program) ~kernargs ~global_size:(gx, gy, gz)
      ~local_size:(lx, ly, lz) =
    if prg.enable_dispatch_ptr then
      invalid_arg "Compute_queue.exec: dispatch-pointer programs are not supported";
    if prg.dev.sqtt_enabled then
      invalid_arg "Compute_queue.exec: thread-trace capture is not supported";
    if prg.enable_private_segment_sgpr && t.dev.xccs <> 1 then
      invalid_arg
        "Compute_queue.exec: architected flat scratch requires a single xcc";

    acquire_mem t ~gli:0 ~gl2:0 ();

    let kernarg = va64 (Hcq.Buffer.va kernargs) in
    let user_regs =
      if prg.enable_private_segment_sgpr then begin
        let scratch = va64 (Hcq.Buffer.va prg.dev.scratch) in
        (* flat-scratch descriptor: word1 bit 31 enables swizzling; word3 is
           0x14 << 12 | 2 << 28 | 2 << 21 | 1 << 23 *)
        [|
          lo32 scratch;
          hi32 scratch lor (1 lsl 31);
          0xffffffff;
          0x20c14000;
          lo32 kernarg;
          hi32 kernarg;
        |]
      end
      else [| lo32 kernarg; hi32 kernarg |]
    in

    let gc = t.dev.gc in
    let prog_addr = Int64.shift_right_logical (va64 prg.prog_addr) 8 in
    wreg t (Ip.reg gc "regCOMPUTE_PGM_LO") [| lo32 prog_addr; hi32 prog_addr |];
    wreg t (Ip.reg gc "regCOMPUTE_PGM_RSRC1") [| prg.rsrc1; prg.rsrc2 |];
    wreg t (Ip.reg gc "regCOMPUTE_PGM_RSRC3") [| prg.rsrc3 |];
    wreg t (Ip.reg gc "regCOMPUTE_TMPRING_SIZE") [| prg.dev.tmpring_size |];

    (* architected flat scratch: each die takes its own slice of the scratch
       buffer *)
    for xcc_id = 0 to t.dev.xccs - 1 do
      let start = pred_open t ~xcc_mask:(1 lsl xcc_id) in
      let scratch_base =
        Int64.shift_right_logical
          (Int64.add
             (va64 (Hcq.Buffer.va prg.dev.scratch))
             (Int64.of_int
                (Hcq.Buffer.size prg.dev.scratch / t.dev.xccs * xcc_id)))
          8
      in
      wreg t
        (Ip.reg gc "regCOMPUTE_DISPATCH_SCRATCH_BASE_LO")
        [| lo32 scratch_base; hi32 scratch_base |];
      pred_close t start
    done;

    wreg t (Ip.reg gc "regCOMPUTE_RESTART_X") [| 0; 0; 0 |];
    wreg t (Ip.reg gc "regCOMPUTE_USER_DATA_0") user_regs;
    wreg_fields t
      (Ip.reg gc "regCOMPUTE_RESOURCE_LIMITS")
      [ ("waves_per_sh", Tolk.Helpers.getenv "WAVES_PER_SH" 0) ];
    wreg t (Ip.reg gc "regCOMPUTE_START_X") [| 0; 0; 0; lx; ly; lz; 0; 0 |];

    let module P = (val t.dev.pm4) in
    let initiator =
      Reg.encode
        (Ip.reg gc "regCOMPUTE_DISPATCH_INITIATOR")
        (("force_start_at_000", 1) :: ("compute_shader_en", 1)
        ::
        (if major prg.dev.target <> 9 then
           [ ("cs_w32_en", if prg.wave32 then 1 else 0) ]
         else []))
    in
    pkt3 t P.packet3_dispatch_direct [| gx; gy; gz; initiator |];

    let module Soc = (val t.dev.soc) in
    pkt3 t P.packet3_event_write
      [|
        P.event_type Soc.cs_partial_flush
        lor P.event_index event_index_partial_flush;
      |]

  let wait t ?(value = 0) sg =
    wait_reg_mem t ~mem:(Hcq.Signal.value_addr sg) ~mask:0xffffffff value

  let timestamp t sg =
    let module P = (val t.dev.pm4) in
    pred_exec t ~xcc_mask:0b1 (fun () ->
        (* all prior writes must retire before the clock is sampled *)
        release_mem t ();
        release_mem t
          ~address:(Hcq.Signal.timestamp_addr sg)
          ~data_sel:P.data_sel__mec_release_mem__send_gpu_clock_counter
          ~int_sel:P.int_sel__mec_release_mem__none ();
        (* the timestamp write must land before any later read observes it *)
        acquire_mem t ())

  let write t ?(b64 = false) buf value =
    let module P = (val t.dev.pm4) in
    let data_sel =
      if b64 then P.data_sel__mec_release_mem__send_64_bit_data
      else P.data_sel__mec_release_mem__send_32_bit_low
    in
    release_mem t ~address:(Hcq.Buffer.va buf) ~value ~data_sel
      ~int_sel:P.int_sel__mec_release_mem__none ()

  let poll_bit t buf ~value ~mask =
    wait_reg_mem t ~mem:(Hcq.Buffer.va buf) ~mask ~op:wait_reg_mem_function_eq
      value

  let signal t ?(value = 0) sg =
    let module P = (val t.dev.pm4) in
    pred_exec t ~xcc_mask:0b1 (fun () ->
        (* the end-of-pipe event goes through the queue's EOP buffer; queues
           must be created with one *)
        release_mem t
          ~address:(Hcq.Signal.value_addr sg)
          ~value:(Int64.of_int value)
          ~data_sel:P.data_sel__mec_release_mem__send_32_bit_low
          ~int_sel:P.int_sel__mec_release_mem__none ~cache_flush:true ();
        match Hcq.Signal.owner sg with
        | Some dev when Hcq.Signal.is_timeline sg && not dev.is_am ->
            release_mem t ~address:dev.queue_event_mailbox_ptr
              ~value:(Int64.of_int dev.queue_event.event_id)
              ~data_sel:P.data_sel__mec_release_mem__send_32_bit_low
              ~int_sel:
                P.int_sel__mec_release_mem__send_interrupt_after_write_confirm
              ~ctxid:dev.queue_event.event_id ()
        | _ -> ())
end

(* Copy queue *)

module Copy_queue = struct
  type 'meta t = {
    dev : 'meta device;
    q : Q.t;
    max_copy_size : int;
    mutable cmd_sizes_rev : int list;
  }

  let create ?max_copy_size (dev : 'meta device) =
    let max_copy_size =
      match max_copy_size with Some s -> s | None -> dev.max_copy_size
    in
    { dev; q = Q.create (); max_copy_size; cmd_sizes_rev = [] }

  let q t = t.q
  let cmd_sizes t = List.rev t.cmd_sizes_rev

  (* every packet records its dword count: submission needs the command
     boundaries to split a stream across the ring's wrap point *)
  let cmd t payload =
    for i = 0 to Array.length payload - 1 do
      Q.push t.q (Array.unsafe_get payload i)
    done;
    t.cmd_sizes_rev <- Array.length payload :: t.cmd_sizes_rev

  let copy t ~dest ~src size =
    let module S = (val t.dev.sdma) in
    let copy_commands = (size + t.max_copy_size - 1) / t.max_copy_size in
    let copied = ref 0 in
    for _ = 1 to copy_commands do
      let step = min (size - !copied) t.max_copy_size in
      let s = Int64.add (va64 (Hcq.Buffer.va src)) (Int64.of_int !copied) in
      let d = Int64.add (va64 (Hcq.Buffer.va dest)) (Int64.of_int !copied) in
      cmd t
        [|
          S.sdma_op_copy
          lor S.sdma_pkt_copy_linear_header_sub_op S.sdma_subop_copy_linear;
          S.sdma_pkt_copy_linear_count_count (step - 1);
          0;
          lo32 s;
          hi32 s;
          lo32 d;
          hi32 d;
        |];
      copied := !copied + step
    done

  let fence_flags t =
    let module S = (val t.dev.sdma) in
    if major t.dev.target <> 9 then
      S.sdma_op_fence lor Amd_sdma_defs.V6_0_0.sdma_pkt_fence_header_mtype 3
    else S.sdma_op_fence

  let signal t ?(value = 0) sg =
    let module S = (val t.dev.sdma) in
    let va = va64 (Hcq.Signal.value_addr sg) in
    cmd t [| fence_flags t; lo32 va; hi32 va; value |];
    match Hcq.Signal.owner sg with
    | Some dev when Hcq.Signal.is_timeline sg && not dev.is_am ->
        let mb = va64 dev.queue_event_mailbox_ptr in
        cmd t [| fence_flags t; lo32 mb; hi32 mb; dev.queue_event.event_id |];
        cmd t
          [|
            S.sdma_op_trap;
            S.sdma_pkt_trap_int_context_int_context dev.queue_event.event_id;
          |]
    | _ -> ()

  let wait t ?(value = 0) sg =
    let module S = (val t.dev.sdma) in
    let va = va64 (Hcq.Signal.value_addr sg) in
    cmd t
      [|
        S.sdma_op_poll_regmem
        lor S.sdma_pkt_poll_regmem_header_func wait_reg_mem_function_geq
        lor S.sdma_pkt_poll_regmem_header_mem_poll 1;
        lo32 va;
        hi32 va;
        value;
        0xffffffff;
        S.sdma_pkt_poll_regmem_dw5_interval 0x04
        lor S.sdma_pkt_poll_regmem_dw5_retry_count 0xfff;
      |]

  let timestamp t sg =
    let module S = (val t.dev.sdma) in
    let ta = va64 (Hcq.Signal.timestamp_addr sg) in
    cmd t
      [|
        S.sdma_op_timestamp
        lor S.sdma_pkt_timestamp_get_header_sub_op
              S.sdma_subop_timestamp_get_global;
        lo32 ta;
        hi32 ta;
      |]

  let write t ?(b64 = false) buf value =
    let module S = (val t.dev.sdma) in
    let va = va64 (Hcq.Buffer.va buf) in
    if b64 then
      cmd t [| S.sdma_op_write; lo32 va; hi32 va; 1; lo32 value; hi32 value |]
    else cmd t [| S.sdma_op_write; lo32 va; hi32 va; 0; lo32 value |]
end
