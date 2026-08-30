(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

module Hcq = Hcq
module Amd_tables = Amd_tables
module Compiler_amd = Compiler_amd
module System = System
module Reg = Amd_tables.Reg
module Ip = Amd_tables.Ip
module Q = Hcq.Q

let major (m, _, _) = m
let lo32 v = Int64.to_int (Int64.logand v 0xFFFFFFFFL)
let hi32 v = Int64.to_int (Int64.shift_right_logical v 32)
let va64 = Int64.of_nativeint
let round_up n align = (n + align - 1) / align * align
let ceildiv a b = (a + b - 1) / b

let prop props name =
  match List.assoc_opt name props with
  | Some v -> v
  | None -> failwith ("missing device property " ^ name)
let event_index_partial_flush = 4
let wait_reg_mem_function_eq = 3
let wait_reg_mem_function_geq = 5

(* Kernel-driver ioctls. The numeric ABI constants are pinned against the
   vendored kfd_ioctl.h by _Static_asserts in tolk_amd_stubs.c. *)
module Kfd = struct
  type alloc_error = Einval | Enomem

  type mem_fault = {
    mmu_va : int64;
    not_present : int;
    read_only : int;
    no_execute : int;
    imprecise : int;
  }

  type hw_fault = {
    reset_type : int;
    reset_cause : int;
    memory_lost : int;
    hw_gpu_id : int;
  }

  external get_version : int -> int * int = "caml_tolk_kfd_get_version"

  external acquire_vm : int -> drm_fd:int -> gpu_id:int -> unit
    = "caml_tolk_kfd_acquire_vm"

  external runtime_enable : int -> mode_mask:int -> unit
    = "caml_tolk_kfd_runtime_enable"

  external alloc_memory_of_gpu :
    int ->
    va:nativeint ->
    size:int ->
    gpu_id:int ->
    flags:int ->
    mmap_offset:int64 ->
    (int64 * int64, alloc_error) result
    = "caml_tolk_kfd_alloc_memory_of_gpu_bc" "caml_tolk_kfd_alloc_memory_of_gpu"

  external free_memory_of_gpu : int -> handle:int64 -> unit
    = "caml_tolk_kfd_free_memory_of_gpu"

  external map_memory_to_gpu : int -> handle:int64 -> gpu_ids:int array -> unit
    = "caml_tolk_kfd_map_memory_to_gpu"

  external unmap_memory_from_gpu :
    int -> handle:int64 -> gpu_ids:int array -> unit
    = "caml_tolk_kfd_unmap_memory_from_gpu"

  external create_event :
    int -> event_page_offset:int64 -> event_type:int -> auto_reset:int ->
    int * int
    = "caml_tolk_kfd_create_event"

  external wait_events :
    int ->
    queue_event_id:int ->
    mem_fault_event_id:int ->
    hw_fault_event_id:int ->
    timeout_ms:int ->
    mem_fault option * hw_fault option
    = "caml_tolk_kfd_wait_events"

  external create_queue :
    int ->
    ring_base:nativeint ->
    ring_size:int ->
    gpu_id:int ->
    queue_type:int ->
    queue_percentage:int ->
    queue_priority:int ->
    eop_buffer_address:nativeint ->
    eop_buffer_size:int ->
    ctx_save_restore_address:nativeint ->
    ctx_save_restore_size:int ->
    ctl_stack_size:int ->
    write_pointer_address:nativeint ->
    read_pointer_address:nativeint ->
    int64 * nativeint * nativeint
    = "caml_tolk_kfd_create_queue_bc" "caml_tolk_kfd_create_queue"

  let alloc_mem_flags_vram = 1 lsl 0
  let alloc_mem_flags_gtt = 1 lsl 1
  let alloc_mem_flags_userptr = 1 lsl 2
  let alloc_mem_flags_uncached = 1 lsl 25
  let alloc_mem_flags_coherent = 1 lsl 26
  let alloc_mem_flags_no_substitute = 1 lsl 28
  let alloc_mem_flags_public = 1 lsl 29
  let alloc_mem_flags_executable = 1 lsl 30
  let alloc_mem_flags_writable = 1 lsl 31
  let queue_type_compute = 0x0
  let queue_type_sdma = 0x1
  let event_type_signal = 0
  let event_type_hw_exception = 3
  let event_type_memory = 8
  let max_queue_percentage = 100
end

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
  mutable max_private_segment_size : int;
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
    max_private_segment_size = 0;
    is_am;
    queue_event_mailbox_ptr;
    queue_event;
  }

let ensure_has_local_memory (dev : 'meta device) ~props ~alloc ~free
    private_segment_size =
  if dev.max_private_segment_size < private_segment_size then begin
    let lanes_per_wave = 64 in
    let mem_alignment_size = if major dev.target <> 9 then 256 else 1024 in
    let size_per_thread =
      round_up private_segment_size (mem_alignment_size / lanes_per_wave)
    in
    let max_slots_scratch_cu = prop props "max_slots_scratch_cu" in
    let cu_cnt =
      prop props "simd_count" / prop props "simd_per_cu" / dev.xccs
    in
    let size_per_xcc =
      size_per_thread * lanes_per_wave * max_slots_scratch_cu * cu_cnt
    in
    let old_size = Hcq.Buffer.size dev.scratch in
    if old_size > 0 then free dev.scratch;
    let scratch, grown =
      match alloc (size_per_xcc * dev.xccs) with
      | buf -> (buf, true)
      (* out of memory: fall back to the old size so the device stays
         usable, and leave the sizing state untouched *)
      | exception Failure _ when old_size > 0 -> (alloc old_size, false)
    in
    dev.scratch <- scratch;
    if grown then begin
      let se_cnt =
        prop props "array_count"
        / prop props "simd_arrays_per_engine"
        / dev.xccs
      in
      (* the per-die slicing below is only validated on generation-9
         multi-die parts; every other supported chip is single-die *)
      let max_scratch_waves = cu_cnt * max_slots_scratch_cu * dev.xccs in
      let wave_scratch =
        ceildiv (lanes_per_wave * size_per_thread) mem_alignment_size
      in
      let num_waves =
        size_per_xcc
        / (wave_scratch * mem_alignment_size)
        / (if major dev.target <> 9 then se_cnt else 1)
      in
      dev.tmpring_size <-
        Amd_tables.tmpring_size ~target_major:(major dev.target)
          ~waves:(min num_waves max_scratch_waves)
          ~wavesize:wave_scratch;
      dev.max_private_segment_size <- private_segment_size
    end
  end

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

(* Queue descriptors *)

module Queue_desc = struct
  type t = {
    ring : Hcq.Mmio.t;
    read_ptr : Hcq.Mmio.t;
    write_ptr : Hcq.Mmio.t;
    doorbell : Hcq.Mmio.t;
    mutable put_value : int;
  }

  let signal_doorbell t =
    Hcq.Mmio.write64 t.write_ptr 0 (Int64.of_int t.put_value);
    (* the doorbell read triggers a device fetch: every ring and pointer
       store must be globally visible before it lands *)
    Hcq.Mmio.fence ();
    Hcq.Mmio.write64 t.doorbell 0 (Int64.of_int t.put_value)
end

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

  let submit t (qd : Queue_desc.t) =
    let cmds = Q.dwords t.q in
    let ring_len = Hcq.Mmio.size qd.ring / 4 in
    let cmds =
      if t.dev.xccs = 1 then cmds
      else begin
        (* predication only takes effect inside indirect buffers, not in the
           ring itself: wrap the stream in an in-ring indirect buffer, padded
           so its body never straddles the wrap point *)
        let module P = (val t.dev.pm4) in
        let n = Array.length cmds in
        let ib_start = (qd.put_value + 5) mod ring_len in
        let ib_pad = if ib_start + n > ring_len then ring_len - ib_start else 0 in
        let ib_ptr =
          Int64.add
            (va64 (Hcq.Mmio.addr qd.ring))
            (Int64.of_int ((qd.put_value + 5 + ib_pad) mod ring_len * 4))
        in
        Array.concat
          [
            [|
              P.packet3 P.packet3_indirect_buffer 2;
              lo32 ib_ptr;
              hi32 ib_ptr;
              n lor P.indirect_buffer_valid;
              P.packet3 P.packet3_nop (ib_pad + n - 1);
            |];
            Array.make ib_pad 0;
            cmds;
          ]
      end
    in
    for i = 0 to Array.length cmds - 1 do
      Hcq.Mmio.write32 qd.ring
        ((qd.put_value + i) mod ring_len * 4)
        (Int32.of_int (Array.unsafe_get cmds i))
    done;
    qd.put_value <- qd.put_value + Array.length cmds;
    Queue_desc.signal_doorbell qd
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

  let submit t (qd : Queue_desc.t) =
    let cmds = Q.dwords t.q in
    let n = Array.length cmds in
    let nbytes = Hcq.Mmio.size qd.ring in
    (* the engine fetches packets as units, so a packet must never straddle
       the ring end: blit whole packets up to the end, and restart at the
       ring start with the rest, zero-filling the gap *)
    let tail_blit_dword =
      let rec fit acc = function
        | sz :: rest when (acc + sz) * 4 < nbytes - (qd.put_value mod nbytes) ->
            fit (acc + sz) rest
        | _ -> acc
      in
      fit 0 (cmd_sizes t)
    in
    let rem_packet_cnt = n - tail_blit_dword in
    let total_bytes =
      (if rem_packet_cnt = 0 then tail_blit_dword * 4
       else (nbytes - (qd.put_value mod nbytes)) mod nbytes)
      + (rem_packet_cnt * 4)
    in
    if total_bytes >= nbytes then
      invalid_arg "Copy_queue.submit: stream does not fit in the ring";
    while
      qd.put_value + total_bytes - Int64.to_int (Hcq.Mmio.read64 qd.read_ptr 0)
      > nbytes
    do
      ()
    done;
    let start = qd.put_value mod nbytes / 4 in
    for i = 0 to tail_blit_dword - 1 do
      Hcq.Mmio.write32 qd.ring
        ((start + i) * 4)
        (Int32.of_int (Array.unsafe_get cmds i))
    done;
    qd.put_value <- qd.put_value + (tail_blit_dword * 4);
    if rem_packet_cnt > 0 then begin
      let zero_fill = nbytes - (qd.put_value mod nbytes) in
      for i = 0 to (zero_fill / 4) - 1 do
        Hcq.Mmio.write32 qd.ring ((qd.put_value mod nbytes) + (i * 4)) 0l
      done;
      qd.put_value <- qd.put_value + zero_fill;
      for i = 0 to rem_packet_cnt - 1 do
        Hcq.Mmio.write32 qd.ring (i * 4)
          (Int32.of_int (Array.unsafe_get cmds (tail_blit_dword + i)))
      done;
      qd.put_value <- qd.put_value + (rem_packet_cnt * 4)
    end;
    Queue_desc.signal_doorbell qd
end

(* Programs *)

module Program = struct
  type 'meta t = {
    params : 'meta program;
    name : string;
    lib_gpu : 'meta Hcq.Buffer.t;
    group_segment_size : int;
    private_segment_size : int;
    kernargs_segment_size : int;
    kernargs_alloc_size : int;
  }

  let r_amdgpu_rel64 = 5

  let load (dev : 'meta device) ~alloc ~props ~name lib =
    let elf = Tolk.Elf.load lib in
    let image = Tolk.Elf.image elf in
    let sections = Tolk.Elf.sections elf in
    let rodata =
      match Tolk.Elf.find_section elf ".rodata" with
      | Some s -> s.Tolk.Elf.addr
      | None -> failwith ".rodata section not found"
    in
    List.iter
      (fun (r : Tolk.Elf.reloc) ->
        if r.symbol.shndx = 0 then
          failwith
            ("Attempting to relocate against an undefined symbol "
           ^ r.symbol.name);
        if r.r_type <> r_amdgpu_rel64 then
          failwith (Printf.sprintf "unknown AMD reloc %d" r.r_type);
        (* the patched slot holds the target's displacement from the slot *)
        Bytes.set_int64_le image r.offset
          (Int64.of_int
             (sections.(r.symbol.shndx).addr + r.symbol.value - r.offset
            + r.addend)))
      (Tolk.Elf.relocs elf);
    let lib_gpu = alloc (round_up (Bytes.length image) 0x1000) in
    Hcq.Mmio.blit_bytes (Hcq.Buffer.cpu_view lib_gpu) ~off:0 image;
    (* the kernel descriptor sits at the start of [.rodata] *)
    let u32 off =
      Int32.to_int (Bytes.get_int32_le image (rodata + off)) land 0xFFFFFFFF
    in
    let group_segment_size = u32 Amd_kd_defs.group_segment_fixed_size in
    let private_segment_size = u32 Amd_kd_defs.private_segment_fixed_size in
    let kernargs_segment_size = u32 Amd_kd_defs.kernarg_size in
    let entry_off =
      Int64.to_int
        (Bytes.get_int64_le image
           (rodata + Amd_kd_defs.kernel_code_entry_byte_offset))
    in
    let code_props =
      Bytes.get_uint16_le image (rodata + Amd_kd_defs.kernel_code_properties)
    in
    let lds_size = (group_segment_size + 511) / 512 land 0x1FF in
    if lds_size > prop props "lds_size_in_kb" * 1024 / 512 then
      failwith "Too many resources requested: group_segment_size";
    let enable_dispatch_ptr =
      code_props
      land Amd_hsa_defs.amd_kernel_code_properties_enable_sgpr_dispatch_ptr
      <> 0
    in
    {
      params =
        {
          dev;
          prog_addr =
            Nativeint.add (Hcq.Buffer.va lib_gpu)
              (Nativeint.of_int (rodata + entry_off));
          (* generation 11 must run waves privileged: unprivileged waves
             are corrupted by compute-wave save/restore *)
          rsrc1 =
            (u32 Amd_kd_defs.compute_pgm_rsrc1
            lor if major dev.target = 11 then 1 lsl 20 else 0);
          rsrc2 = u32 Amd_kd_defs.compute_pgm_rsrc2 lor (lds_size lsl 15);
          rsrc3 = u32 Amd_kd_defs.compute_pgm_rsrc3;
          wave32 = code_props land 0x400 <> 0;
          enable_private_segment_sgpr =
            code_props
            land
            Amd_hsa_defs
            .amd_kernel_code_properties_enable_sgpr_private_segment_buffer
            <> 0;
          enable_dispatch_ptr;
        };
      name;
      lib_gpu;
      group_segment_size;
      private_segment_size;
      kernargs_segment_size;
      kernargs_alloc_size =
        (kernargs_segment_size
        +
        if enable_dispatch_ptr then Amd_hsa_defs.Kernel_dispatch_packet.size
        else 0);
    }

  let free ~free:release t = release t.lib_gpu

  let call t ~kernargs ~queue ~timeline ~timeline_value ?wait ?timeout_ms
      ~bufs ~vals ~global_size ~local_size () =
    if t.params.enable_dispatch_ptr then
      invalid_arg "Program.call: dispatch-pointer programs are not supported";
    let slot = Hcq.Kernargs.alloc kernargs t.kernargs_alloc_size in
    Hcq.Kernargs.write_args slot ~bufs ~vals;
    let cq = Compute_queue.create t.params.dev in
    Compute_queue.wait cq ~value:(timeline_value - 1) timeline;
    Compute_queue.memory_barrier cq;
    (match wait with
    | Some (st, _) -> Compute_queue.timestamp cq st
    | None -> ());
    Compute_queue.exec cq t.params ~kernargs:slot ~global_size ~local_size;
    (match wait with
    | Some (_, en) -> Compute_queue.timestamp cq en
    | None -> ());
    Compute_queue.signal cq ~value:timeline_value timeline;
    Compute_queue.submit cq queue;
    match wait with
    | None -> None
    | Some (st, en) ->
        Hcq.Signal.wait timeline ?timeout_ms timeline_value;
        Some ((Hcq.Signal.timestamp en -. Hcq.Signal.timestamp st) /. 1e6)
end

(* Kernel-driver interface *)

module Kfd_iface = struct
  type mem = { handle : int64; owner : int }

  type ip_versions = {
    gc : int * int * int;
    sdma : int * int * int;
    nbif : int * int * int;
  }

  type queue_type = Compute | Sdma

  type t = {
    gpu_id : int;
    props : (string * int) list;
    ip_versions : ip_versions;
    drm_fd : int;
    queue_event : queue_event;
    queue_event_mailbox_ptr : nativeint;
    mem_fault_event_id : int;
    hw_fault_event_id : int;
    mutable doorbells : (int64 * nativeint) option;
    mutable mem_fault : Kfd.mem_fault option;
    mutable hw_fault : Kfd.hw_fault option;
  }

  let topology = "/sys/devices/virtual/kfd/kfd/topology/nodes"

  (* Driver-wide state, shared by every device: the driver file descriptor,
     the usable GPU nodes, and the one interrupt-mailbox page. *)
  let state : (int * string array) option ref = ref None
  let event_page : mem Hcq.Buffer.t option ref = ref None

  let read_file path = In_channel.with_open_bin path In_channel.input_all
  let int_of_file path = int_of_string (String.trim (read_file path))

  let usable_gpu node =
    match int_of_file (topology ^ "/" ^ node ^ "/gpu_id") with
    | id -> id <> 0
    | exception _ -> false

  let scan () =
    match !state with
    | Some s -> s
    | None ->
        let fd = Hcq.File_io.openfile "/dev/kfd" ~flags:Hcq.File_io.o_rdwr in
        let gpus =
          Array.of_list (List.filter usable_gpu (Array.to_list (Sys.readdir topology)))
        in
        Array.sort
          (fun a b -> Int.compare (int_of_string a) (int_of_string b))
          gpus;
        state := Some (fd, gpus);
        (fd, gpus)

  let count () = Array.length (snd (scan ()))

  let parse_props text =
    let tokens line =
      List.filter (( <> ) "") (String.split_on_char ' ' (String.trim line))
    in
    List.filter_map
      (fun line ->
        match tokens line with
        | key :: v :: _ -> Some (key, int_of_string v)
        | _ -> None)
      (String.split_on_char '\n' text)

  let discover_ips sysfs_path =
    let base = sysfs_path ^ "/ip_discovery/die/0" in
    let version name hwid =
      let part p = Printf.sprintf "%s/%d/0/%s" base hwid p in
      match (int_of_file (part "major"), int_of_file (part "minor"),
             int_of_file (part "revision"))
      with
      | v -> v
      | exception Sys_error _ ->
          failwith
            (Printf.sprintf "Kfd_iface: no %s ip version under %s" name base)
    in
    {
      gc = version "gc" Amd_regs_defs.gc_hwid;
      sdma = version "sdma" Amd_regs_defs.sdma0_hwid;
      nbif = version "nbif" Amd_regs_defs.nbif_hwid;
    }

  let map_to_gpu ~kfd ~gpu_id b =
    Kfd.map_memory_to_gpu kfd ~handle:(Hcq.Buffer.meta b).handle
      ~gpu_ids:[| gpu_id |]

  let alloc_raw ~kfd ~drm_fd ~gpu_id ?(host = false) ?(uncached = false)
      ?(cpu_access = false) ?cpu_addr size =
    let flags =
      Kfd.alloc_mem_flags_writable lor Kfd.alloc_mem_flags_executable
      lor Kfd.alloc_mem_flags_no_substitute
      lor (if uncached then
             Kfd.alloc_mem_flags_coherent lor Kfd.alloc_mem_flags_uncached
             lor Kfd.alloc_mem_flags_gtt
           else if host then Kfd.alloc_mem_flags_userptr
           else Kfd.alloc_mem_flags_vram)
      (* an externally provided mapping must stay uncachable for the CPU *)
      lor (match cpu_addr with
          | Some _ ->
              Kfd.alloc_mem_flags_coherent lor Kfd.alloc_mem_flags_uncached
          | None -> 0)
      lor (if cpu_access || host then Kfd.alloc_mem_flags_public else 0)
    in
    let userptr = flags land Kfd.alloc_mem_flags_userptr <> 0 in
    let module F = Hcq.File_io in
    (* reserve the virtual range now so the CPU mapping can later land at the
       exact address the device was given *)
    let addr =
      if userptr then
        match cpu_addr with
        | Some a -> a
        | None ->
            F.mmap ~addr:0n ~size
              ~prot:(F.prot_read lor F.prot_write)
              ~flags:(F.map_shared lor F.map_anonymous)
              ~fd:(-1) ~offset:0L
      else
        F.mmap ~addr:0n ~size ~prot:F.prot_none
          ~flags:(F.map_private lor F.map_anonymous lor F.map_noreserve)
          ~fd:(-1) ~offset:0L
    in
    let mmap_offset = if userptr then Int64.of_nativeint addr else 0L in
    match Kfd.alloc_memory_of_gpu kfd ~va:addr ~size ~gpu_id ~flags ~mmap_offset with
    | Error e ->
        if cpu_addr = None then F.munmap addr ~size;
        failwith
          (match e with
          | Kfd.Einval
            when flags land Kfd.alloc_mem_flags_vram <> 0 && cpu_access ->
              "Cannot allocate host-visible VRAM. Ensure the resizable BAR \
               option is enabled on your system."
          | Kfd.Einval -> "AMDKFD_IOC_ALLOC_MEMORY_OF_GPU: Invalid argument"
          | Kfd.Enomem ->
              Printf.sprintf "Cannot allocate %d bytes: no memory is available."
                size)
    | Ok (handle, mmap_offset) ->
        if not userptr then begin
          let mapped =
            F.mmap ~addr ~size
              ~prot:(F.prot_read lor F.prot_write)
              ~flags:(F.map_shared lor F.map_fixed)
              ~fd:drm_fd ~offset:mmap_offset
          in
          assert (mapped = addr)
        end;
        let view =
          if cpu_access || host then Some (Hcq.Mmio.make ~addr ~size) else None
        in
        let b =
          Hcq.Buffer.make ~va:addr ~size ?view ~meta:{ handle; owner = gpu_id }
            ()
        in
        map_to_gpu ~kfd ~gpu_id b;
        b

  let create ~device_id =
    let kfd, gpus = scan () in
    if device_id >= Array.length gpus then
      failwith
        (Printf.sprintf
           "No device found for %d. Requesting more devices than the system \
            has?"
           device_id);
    let node = topology ^ "/" ^ gpus.(device_id) in
    let gpu_id = int_of_file (node ^ "/gpu_id") in
    let props = parse_props (read_file (node ^ "/properties")) in
    let drm_minor = List.assoc "drm_render_minor" props in
    let ip_versions =
      discover_ips (Printf.sprintf "/sys/class/drm/renderD%d/device" drm_minor)
    in
    let drm_fd =
      Hcq.File_io.openfile
        (Printf.sprintf "/dev/dri/renderD%d" drm_minor)
        ~flags:Hcq.File_io.o_rdwr
    in
    let kfd_ver = Kfd.get_version kfd in
    Kfd.acquire_vm kfd ~drm_fd ~gpu_id;
    if kfd_ver >= (1, 14) then Kfd.runtime_enable kfd ~mode_mask:0;
    let page =
      match !event_page with
      | Some page ->
          map_to_gpu ~kfd ~gpu_id page;
          page
      | None ->
          let page = alloc_raw ~kfd ~drm_fd ~gpu_id ~uncached:true 0x8000 in
          (* register the page so signal-event slots live in it *)
          ignore
            (Kfd.create_event kfd
               ~event_page_offset:(Hcq.Buffer.meta page).handle
               ~event_type:Kfd.event_type_signal ~auto_reset:0
              : int * int);
          event_page := Some page;
          page
    in
    let queue_event_id, queue_event_slot =
      Kfd.create_event kfd ~event_page_offset:0L
        ~event_type:Kfd.event_type_signal ~auto_reset:1
    in
    let mem_fault_event_id, _ =
      Kfd.create_event kfd ~event_page_offset:0L
        ~event_type:Kfd.event_type_memory ~auto_reset:0
    in
    let hw_fault_event_id, _ =
      Kfd.create_event kfd ~event_page_offset:0L
        ~event_type:Kfd.event_type_hw_exception ~auto_reset:0
    in
    {
      gpu_id;
      props;
      ip_versions;
      drm_fd;
      queue_event = { event_id = queue_event_id };
      queue_event_mailbox_ptr =
        Nativeint.add (Hcq.Buffer.va page)
          (Nativeint.of_int (queue_event_slot * 8));
      mem_fault_event_id;
      hw_fault_event_id;
      doorbells = None;
      mem_fault = None;
      hw_fault = None;
    }

  let props t = t.props
  let ip_versions t = t.ip_versions
  let queue_event t = t.queue_event
  let queue_event_mailbox_ptr t = t.queue_event_mailbox_ptr

  let alloc t ?host ?uncached ?cpu_access ?cpu_addr size =
    let kfd, _ = scan () in
    alloc_raw ~kfd ~drm_fd:t.drm_fd ~gpu_id:t.gpu_id ?host ?uncached
      ?cpu_access ?cpu_addr size

  let free t b =
    let kfd, _ = scan () in
    let b = Hcq.Buffer.base b in
    let meta = Hcq.Buffer.meta b in
    Kfd.unmap_memory_from_gpu kfd ~handle:meta.handle ~gpu_ids:[| t.gpu_id |];
    if meta.owner = t.gpu_id then begin
      if Hcq.Buffer.va b <> 0n then
        Hcq.File_io.munmap (Hcq.Buffer.va b) ~size:(Hcq.Buffer.size b);
      Kfd.free_memory_of_gpu kfd ~handle:meta.handle
    end

  let map t b =
    let kfd, _ = scan () in
    map_to_gpu ~kfd ~gpu_id:t.gpu_id b;
    Hcq.Buffer.make ~va:(Hcq.Buffer.va b) ~size:(Hcq.Buffer.size b)
      ~meta:(Hcq.Buffer.meta b) ()

  let create_queue t queue_type ~ring ~gart ~rptr ~wptr ?eop_buffer
      ?cwsr_buffer ?(ctl_stack_size = 0) ?(ctx_save_restore_size = 0)
      ?(xcc_id = 0) () =
    let kfd, _ = scan () in
    let buf_va = function Some b -> Hcq.Buffer.va b | None -> 0n in
    let buf_size = function Some b -> Hcq.Buffer.size b | None -> 0 in
    let doorbell_offset, rptr_addr, wptr_addr =
      Kfd.create_queue kfd
        ~ring_base:(Hcq.Buffer.va ring)
        ~ring_size:(Hcq.Buffer.size ring)
        ~gpu_id:t.gpu_id
        ~queue_type:
          (match queue_type with
          | Compute -> Kfd.queue_type_compute
          | Sdma -> Kfd.queue_type_sdma)
        ~queue_percentage:(Kfd.max_queue_percentage lor (xcc_id lsl 8))
        ~queue_priority:(Tolk.Helpers.getenv "AMD_KFD_QUEUE_PRIORITY" 7)
        ~eop_buffer_address:(buf_va eop_buffer)
        ~eop_buffer_size:(buf_size eop_buffer)
        ~ctx_save_restore_address:(buf_va cwsr_buffer)
        ~ctx_save_restore_size ~ctl_stack_size
        ~write_pointer_address:
          (Nativeint.add (Hcq.Buffer.va gart) (Nativeint.of_int wptr))
        ~read_pointer_address:
          (Nativeint.add (Hcq.Buffer.va gart)
             (Nativeint.of_int (rptr + (8 * xcc_id))))
    in
    let doorbells_base, doorbells_addr =
      match t.doorbells with
      | Some d -> d
      | None ->
          (* the doorbell region is two pages *)
          let base = Int64.logand doorbell_offset (Int64.lognot 0x1fffL) in
          let addr =
            Hcq.File_io.mmap ~addr:0n ~size:0x2000
              ~prot:(Hcq.File_io.prot_read lor Hcq.File_io.prot_write)
              ~flags:Hcq.File_io.map_shared ~fd:kfd ~offset:base
          in
          t.doorbells <- Some (base, addr);
          (base, addr)
    in
    {
      Queue_desc.ring =
        Hcq.Mmio.make ~addr:(Hcq.Buffer.va ring) ~size:(Hcq.Buffer.size ring);
      read_ptr = Hcq.Mmio.make ~addr:rptr_addr ~size:8;
      write_ptr = Hcq.Mmio.make ~addr:wptr_addr ~size:8;
      doorbell =
        Hcq.Mmio.make
          ~addr:
            (Nativeint.add doorbells_addr
               (Nativeint.of_int
                  (Int64.to_int (Int64.sub doorbell_offset doorbells_base))))
          ~size:8;
      put_value = 0;
    }

  let poll_events t ~timeout_ms =
    let kfd, _ = scan () in
    let memf, hwf =
      Kfd.wait_events kfd ~queue_event_id:t.queue_event.event_id
        ~mem_fault_event_id:t.mem_fault_event_id
        ~hw_fault_event_id:t.hw_fault_event_id ~timeout_ms
    in
    (* fault data is latched: once seen, every later poll keeps raising *)
    (match memf with Some _ -> t.mem_fault <- memf | None -> ());
    match hwf with Some _ -> t.hw_fault <- hwf | None -> ()

  let on_device_hang t =
    if t.mem_fault = None && t.hw_fault = None then (
      try poll_events t ~timeout_ms:1 with Failure _ -> ());
    let report =
      (match t.mem_fault with
      | Some f ->
          [
            Printf.sprintf
              "MMU fault: 0x%LX | NotPresent=%d ReadOnly=%d NoExecute=%d \
               imprecise=%d"
              f.Kfd.mmu_va f.Kfd.not_present f.Kfd.read_only f.Kfd.no_execute
              f.Kfd.imprecise;
          ]
      | None -> [])
      @
      match t.hw_fault with
      | Some f ->
          [
            Printf.sprintf
              "HW fault: reset_type=%d reset_cause=%d memory_lost=%d gpu_id=%d"
              f.Kfd.reset_type f.Kfd.reset_cause f.Kfd.memory_lost
              f.Kfd.hw_gpu_id;
          ]
      | None -> []
    in
    failwith (String.concat "\n" report)

  let sleep t ~timeout_ms =
    poll_events t ~timeout_ms;
    if t.mem_fault <> None || t.hw_fault <> None then on_device_hang t
end
