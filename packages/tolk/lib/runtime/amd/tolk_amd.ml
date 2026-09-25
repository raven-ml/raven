(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

module Hcq = Tolk_hcq.Hcq
module Amd_tables = Amd_tables
module Compiler_amd = Compiler_amd
module System = Tolk_hcq.System
module Amdev = Amdev
module Am_ip = Am_ip
module Am_boot = Am_boot
module Reg = Amd_tables.Reg
module Ip = Amd_tables.Ip
module Q = Hcq.Q
module Timeline = Hcq.Timeline

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
let aql_header =
  let open Amd_hsa_defs in
  (1 lsl hsa_packet_header_barrier)
  lor (hsa_fence_scope_system lsl hsa_packet_header_scacquire_fence_scope)
  lor (hsa_fence_scope_system lsl hsa_packet_header_screlease_fence_scope)

let dispatch_header =
  let open Amd_hsa_defs in
  aql_header lor (hsa_packet_type_kernel_dispatch lsl hsa_packet_header_type)
  lor (3 lsl (16 + hsa_kernel_dispatch_packet_setup_dimensions))

let indirect_header =
  aql_header lor (Amd_hsa_defs.hsa_packet_type_vendor_specific
    lsl Amd_hsa_defs.hsa_packet_header_type) lor (1 lsl 16)

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
  let queue_type_compute_aql = 0x2
  let queue_type_sdma = 0x1
  let event_type_signal = 0
  let event_type_hw_exception = 3
  let event_type_memory = 8
  let max_queue_percentage = 100
end

(* Devices *)

type queue_event = { event_id : int }
type queue_type = Compute | Compute_aql | Sdma

type ip_versions = {
  gc : int * int * int;
  sdma : int * int * int;
  nbif : int * int * int;
}

type 'meta device = {
  target : int * int * int;
  xccs : int;
  is_aql : bool;
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
    ?(sqtt_enabled = false) ?(is_aql = false) ~tmpring_size ~scratch ~is_am
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
    is_aql;
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

let scratch_layout (dev : 'meta device) ~props private_segment_size =
  let alignment = if major dev.target = 9 then 1024 else 256 in
  let per_thread = round_up (max private_segment_size 128) (alignment / 64) in
  let cu = prop props "simd_count" / prop props "simd_per_cu" / dev.xccs in
  let slots = prop props "max_slots_scratch_cu" in
  let per_xcc = per_thread * 64 * slots * cu in
  let se = prop props "array_count" / prop props "simd_arrays_per_engine" / dev.xccs in
  let wavesize = ceildiv (64 * per_thread) alignment in
  let waves = min (cu * slots * dev.xccs)
      (per_xcc / (wavesize * alignment) / (if major dev.target = 9 then 1 else se)) in
  per_xcc * dev.xccs,
  Amd_tables.tmpring_size ~target_major:(major dev.target) ~waves ~wavesize

let ensure_has_local_memory (dev : 'meta device) ~props ~alloc ~free private_segment_size =
  let private_segment_size = max private_segment_size 128 in
  if dev.max_private_segment_size < private_segment_size then begin
    let size, tmpring_size = scratch_layout dev ~props private_segment_size in
    (* Allocation failure must preserve the old backing and fail the larger
       launch. Returning success with undersized scratch corrupts memory. *)
    let scratch = alloc size in
    let previous = dev.scratch in
    dev.scratch <- scratch;
    dev.tmpring_size <- tmpring_size;
    dev.max_private_segment_size <- private_segment_size;
    if Hcq.Buffer.size previous > 0 then free previous
  end

(* Programs *)

type 'meta program = {
  dev : 'meta device;
  prog_addr : nativeint;
  kernel_object : nativeint;
  group_segment_size : int;
  private_segment_size : int;
  rsrc1 : int;
  rsrc2 : int;
  rsrc3 : int;
  wave32 : bool;
  enable_private_segment_sgpr : bool;
  enable_dispatch_ptr : bool;
}

(* Queue descriptors *)

module Queue_desc = struct
  type aql = {
    descriptor : Hcq.Mmio.t;
    commands : Hcq.Mmio.t;
    address : nativeint;
    allocator : Tolk.Bump.t;
  }
  type t = {
    ring : Hcq.Mmio.t;
    aql : aql option;
    read_ptr : Hcq.Mmio.t;
    write_ptr : Hcq.Mmio.t;
    doorbell : Hcq.Mmio.t;
    hdp_flush : Hcq.Mmio.t option;
    resetup : (unit -> unit) option;
  }

  let initialize_aql ~descriptor ~commands ~cu_count ~waves_per_cu =
    let module A = Amd_hsa_defs.Amd_queue in
    Hcq.Mmio.blit_bytes descriptor ~off:0 (Bytes.make A.size '\000');
    let put off value = Hcq.Mmio.write32 descriptor off (Int32.of_int value) in
    put A.queue_properties (Amd_hsa_defs.amd_queue_properties_is_ptr64
      lor Amd_hsa_defs.amd_queue_properties_enable_profiling);
    put A.read_dispatch_id_field_base_byte_offset A.read_dispatch_id;
    put A.max_cu_id (cu_count - 1);
    put A.max_wave_id (waves_per_cu - 1);
    {descriptor; commands = Hcq.Buffer.cpu_view commands; address = Hcq.Buffer.va commands;
      allocator = Tolk.Bump.create ~size:(Hcq.Buffer.size commands) ~wrap:true ()}

  let update_scratch (dev : 'meta device) t =
    let module A = Amd_hsa_defs.Amd_queue in
    let module R = Amd_hsa_defs.Scratch_resource in
    let swizzle, format = match major dev.target with
      | 9 -> R.gfx9_swizzle, R.gfx9_format
      | 11 -> R.gfx11_swizzle, R.gfx11_format
      | 12 -> R.gfx12_swizzle, R.gfx12_format
      | _ -> invalid_arg "AQL scratch: unsupported target" in
    let base = va64 (Hcq.Buffer.va dev.scratch) in
    let put off value = Hcq.Mmio.write32 t.descriptor off (Int32.of_int value) in
    Hcq.Mmio.write64 t.descriptor A.scratch_backing_memory_location base;
    put A.scratch_wave64_lane_byte_size dev.max_private_segment_size;
    Array.iteri (fun i value -> put (A.scratch_resource_descriptor + i * 4) value)
      [|lo32 base; hi32 base lor swizzle; Hcq.Buffer.size dev.scratch / dev.xccs; format|];
    put A.compute_tmpring_size dev.tmpring_size;
    Hcq.Mmio.fence ()

  let signal_doorbell t value =
    Hcq.Mmio.fence ();
    Hcq.Mmio.write64 t.write_ptr 0 (Int64.of_int value);
    (* the doorbell read triggers a device fetch: every ring and pointer
       store must be globally visible before it lands *)
    Hcq.Mmio.fence ();
    (* ops_amd.py:680-683: driver-less queues also flush the host data
       path, so host writes to device memory reach the engines *)
    Option.iter (fun view -> Hcq.Mmio.write32 view 0 0l) t.hdp_flush;
    Hcq.Mmio.fence ();
    Hcq.Mmio.write64 t.doorbell 0 (Int64.of_int (value - if Option.is_some t.aql then 1 else 0))
end

(* The interface seam: the device runtime drives the GPU through one of
   two interchangeable interfaces — the kernel driver ({!Kfd_iface}) or
   the driver-less PCI path ({!Pci_iface}) — and this record is every
   call the shared runtime makes on the selected one. *)
module Iface = struct
  type 'mem t = {
    props : (string * int) list;
    ip_versions : ip_versions;
    is_am : bool;
    queue_event : queue_event;
    queue_event_mailbox_ptr : nativeint;
    alloc :
      ?host:bool -> ?uncached:bool -> ?cpu_access:bool -> int ->
      'mem Hcq.Buffer.t;
    free : 'mem Hcq.Buffer.t -> unit;
    kind : 'mem Hcq.Buffer.t Type.Id.t;
    map : Tolk.Device.Buffer.t -> 'mem Hcq.Buffer.t;
    unmap : 'mem Hcq.Buffer.t -> unit;
    empty_scratch : 'mem Hcq.Buffer.t;
    create_queue :
      queue_type ->
      ring:'mem Hcq.Buffer.t ->
      gart:'mem Hcq.Buffer.t ->
      rptr:int ->
      wptr:int ->
      ?eop_buffer:'mem Hcq.Buffer.t ->
      ?cwsr_buffer:'mem Hcq.Buffer.t ->
      ?ctl_stack_size:int ->
      ?ctx_save_restore_size:int ->
      unit ->
      Queue_desc.t;
    sleep : int -> unit;
    on_device_hang : unit -> unit;
    register :
      (compute_queue:Queue_desc.t ->
      tl:('mem, 'mem device) Timeline.t ->
      unit)
      option;
    after_sync : (unit -> unit) option;
    device_fini : (unit -> unit) option;
  }
end

(* Compute queue *)

module Compute_queue = struct
  type packet = Indirect of int * int | Dispatch of int array
  type 'meta t = { dev : 'meta device; q : Q.t;
    mutable packets : packet list; mutable run_start : int }

  let wait_reg_mem_function_eq = wait_reg_mem_function_eq
  let wait_reg_mem_function_geq = wait_reg_mem_function_geq
  let create dev = { dev; q = Q.create (); packets = []; run_start = 0 }
  let close_run t =
    let stop = Q.length t.q in
    if stop > t.run_start then t.packets <- Indirect (t.run_start, stop - t.run_start) :: t.packets;
    t.run_start <- stop
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
    if t.dev.is_aql then begin
      close_run t;
      List.iter (fun (global, local) ->
          if global < 1 || local < 1 || local > 0xffff || global > 0xffffffff / local then
            invalid_arg "Compute_queue.exec: launch dimensions exceed AQL fields") [gx,lx; gy,ly; gz,lz];
      let object_ = va64 prg.kernel_object and args = va64 (Hcq.Buffer.va kernargs) in
      t.packets <- Dispatch [|dispatch_header; lx lor (ly lsl 16); lz;
        gx * lx; gy * ly; gz * lz; prg.private_segment_size; prg.group_segment_size;
        lo32 object_; hi32 object_; lo32 args; hi32 args; 0; 0; 0; 0|] :: t.packets
    end else begin
    if prg.enable_dispatch_ptr && Hcq.Buffer.size kernargs < Amd_hsa_defs.Kernel_dispatch_packet.size then
      invalid_arg "Compute_queue.exec: dispatch packet missing from kernargs";
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
        |]
      end
      else [||]
    in
    let dispatch = if prg.enable_dispatch_ptr then
        let address = Int64.add kernarg (Int64.of_int
            (Hcq.Buffer.size kernargs - Amd_hsa_defs.Kernel_dispatch_packet.size)) in
        [|lo32 address; hi32 address|] else [||] in
    let user_regs = Array.concat [user_regs; dispatch; [|lo32 kernarg; hi32 kernarg|]] in

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

    end

  let wait t ?(value = 0) sg =
    let value = if Hcq.Signal.is_timeline sg then value land 0xffffffff else value in
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
    if t.dev.is_aql then close_run t;
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
    if Q.length t.q = 0 && t.packets = [] then ()
    else if t.dev.is_aql then begin
      close_run t;
      let staging = match qd.aql with Some aql -> aql
        | None -> invalid_arg "Compute_queue.submit: missing AQL staging" in
      let bytes = Q.length t.q * 4 in
      let offset = Tolk.Bump.alloc staging.allocator bytes ~align:256 () in
      for i = 0 to Q.length t.q - 1 do
        Hcq.Mmio.write32 staging.commands (offset + i * 4) (Int32.of_int (Q.get t.q i))
      done;
      let module P = (val t.dev.pm4) in
      let packet = function
        | Dispatch words -> words
        | Indirect (start, count) ->
            let address = Int64.add (va64 staging.address) (Int64.of_int (offset + start * 4)) in
            [|indirect_header; P.packet3 P.packet3_indirect_buffer 2; lo32 address; hi32 address;
              count lor P.indirect_buffer_valid; 10; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0|] in
      let put = Int64.to_int (Hcq.Mmio.read64 qd.write_ptr 0) in
      let entries = Hcq.Mmio.size qd.ring / 64 in
      let packets = List.rev t.packets in
      if List.length packets >= entries then invalid_arg "AQL submission exceeds ring capacity";
      List.iteri (fun i p -> Array.iteri (fun j value ->
          Hcq.Mmio.write32 qd.ring (((put + i) mod entries) * 64 + j * 4) (Int32.of_int value))
          (packet p)) packets;
      Queue_desc.signal_doorbell qd (put + List.length packets)
    end else begin
    let put = Int64.to_int (Hcq.Mmio.read64 qd.write_ptr 0) in
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
        let ib_start = (put + 5) mod ring_len in
        let ib_pad = if ib_start + n > ring_len then ring_len - ib_start else 0 in
        let ib_ptr =
          Int64.add
            (va64 (Hcq.Mmio.addr qd.ring))
            (Int64.of_int ((put + 5 + ib_pad) mod ring_len * 4))
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
        ((put + i) mod ring_len * 4)
        (Int32.of_int (Array.unsafe_get cmds i))
    done;
    Queue_desc.signal_doorbell qd (put + Array.length cmds)
    end
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
    let value = if Hcq.Signal.is_timeline sg then value land 0xffffffff else value in
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
    let value = if Hcq.Signal.is_timeline sg then value land 0xffffffff else value in
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
    let put = ref (Int64.to_int (Hcq.Mmio.read64 qd.write_ptr 0)) in
    let cmds = Q.dwords t.q in
    let n = Array.length cmds in
    let nbytes = Hcq.Mmio.size qd.ring in
    (* the engine fetches packets as units, so a packet must never straddle
       the ring end: blit whole packets up to the end, and restart at the
       ring start with the rest, zero-filling the gap *)
    let tail_blit_dword =
      let rec fit acc = function
        | sz :: rest when (acc + sz) * 4 < nbytes - (!put mod nbytes) ->
            fit (acc + sz) rest
        | _ -> acc
      in
      fit 0 (cmd_sizes t)
    in
    let rem_packet_cnt = n - tail_blit_dword in
    let total_bytes =
      (if rem_packet_cnt = 0 then tail_blit_dword * 4
       else (nbytes - (!put mod nbytes)) mod nbytes)
      + (rem_packet_cnt * 4)
    in
    if total_bytes >= nbytes then
      invalid_arg "Copy_queue.submit: stream does not fit in the ring";
    while
      !put + total_bytes - Int64.to_int (Hcq.Mmio.read64 qd.read_ptr 0)
      > nbytes
    do
      ()
    done;
    let start = !put mod nbytes / 4 in
    for i = 0 to tail_blit_dword - 1 do
      Hcq.Mmio.write32 qd.ring
        ((start + i) * 4)
        (Int32.of_int (Array.unsafe_get cmds i))
    done;
    put := !put + (tail_blit_dword * 4);
    if rem_packet_cnt > 0 then begin
      let zero_fill = nbytes - (!put mod nbytes) in
      for i = 0 to (zero_fill / 4) - 1 do
        Hcq.Mmio.write32 qd.ring ((!put mod nbytes) + (i * 4)) 0l
      done;
      put := !put + zero_fill;
      for i = 0 to rem_packet_cnt - 1 do
        Hcq.Mmio.write32 qd.ring (i * 4)
          (Int32.of_int (Array.unsafe_get cmds (tail_blit_dword + i)))
      done;
      put := !put + (rem_packet_cnt * 4)
    end;
    Queue_desc.signal_doorbell qd !put
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

  type data = {
    desc_offset : int;
    entry_offset : int;
    rsrc1 : int; rsrc2 : int; rsrc3 : int;
    wave32 : bool;
    enable_private_segment_sgpr : bool;
    enable_dispatch_ptr : bool;
    group_segment_size : int;
    private_segment_size : int;
    kernargs_segment_size : int;
  }

  let r_amdgpu_rel64 = 5

  let image ~target ~props lib =
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
    ({
      desc_offset = rodata;
      entry_offset = rodata + entry_off;
      rsrc1 = u32 Amd_kd_defs.compute_pgm_rsrc1
        lor (if major target = 11 then 1 lsl 20 else 0);
      rsrc2 = u32 Amd_kd_defs.compute_pgm_rsrc2 lor (lds_size lsl 15);
      rsrc3 = u32 Amd_kd_defs.compute_pgm_rsrc3;
      wave32 = code_props land 0x400 <> 0;
      enable_private_segment_sgpr = code_props land
        Amd_hsa_defs.amd_kernel_code_properties_enable_sgpr_private_segment_buffer <> 0;
      enable_dispatch_ptr;
      group_segment_size;
      private_segment_size;
      kernargs_segment_size;
    }, image)

  let load (dev : 'meta device) ~alloc ~props ~name lib =
    let data, image = image ~target:dev.target ~props lib in
    let lib_gpu = alloc (round_up (Bytes.length image) 0x1000) in
    Hcq.Mmio.blit_bytes (Hcq.Buffer.cpu_view lib_gpu) ~off:0 image;
    {
      params = {
        dev;
        prog_addr = Nativeint.add (Hcq.Buffer.va lib_gpu) (Nativeint.of_int data.entry_offset);
        kernel_object = Nativeint.add (Hcq.Buffer.va lib_gpu) (Nativeint.of_int data.desc_offset);
        group_segment_size = data.group_segment_size;
        private_segment_size = data.private_segment_size;
        rsrc1 = data.rsrc1; rsrc2 = data.rsrc2; rsrc3 = data.rsrc3;
        wave32 = data.wave32;
        enable_private_segment_sgpr = data.enable_private_segment_sgpr;
        enable_dispatch_ptr = data.enable_dispatch_ptr;
      };
      name; lib_gpu;
      group_segment_size = data.group_segment_size;
      private_segment_size = data.private_segment_size;
      kernargs_segment_size = data.kernargs_segment_size;
      kernargs_alloc_size = data.kernargs_segment_size +
        (if data.enable_dispatch_ptr then Amd_hsa_defs.Kernel_dispatch_packet.size else 0);
    }

  let free ~free:release t = release t.lib_gpu

  let call t ~layout ~kernargs ~queue ~timeline ~timeline_value ?wait ?timeout_ms
      ~bufs ~vals ~global_size ~local_size () =
    let packet = if t.params.enable_dispatch_ptr then begin
        let module P = Amd_hsa_defs.Kernel_dispatch_packet in
        let packet = Bytes.make P.size '\000' in
        Bytes.set_int32_le packet P.header (Int32.of_int dispatch_header);
        let gx, gy, gz = global_size and lx, ly, lz = local_size in
        List.iter (fun (group, local, group_offset, local_offset) ->
            if group < 1 || local < 1 || local > 0xffff || group > 0xffffffff / local then
              invalid_arg "Program.call: launch dimensions exceed dispatch packet fields";
            Bytes.set_uint16_le packet local_offset local;
            Bytes.set_int32_le packet group_offset (Int32.of_int (group * local)))
          [gx, lx, P.grid_size_x, P.workgroup_size_x;
           gy, ly, P.grid_size_y, P.workgroup_size_y;
           gz, lz, P.grid_size_z, P.workgroup_size_z];
        Bytes.set_int32_le packet P.private_segment_size (Int32.of_int t.private_segment_size);
        Bytes.set_int32_le packet P.group_segment_size (Int32.of_int t.group_segment_size);
        Some packet
      end else None in
    let slot = Hcq.Kernargs.alloc kernargs t.kernargs_alloc_size
        ~wait:(fun () -> Hcq.Signal.wait timeline ?timeout_ms (timeline_value - 1)) in
    Hcq.Kernargs.write_args layout slot ~bufs ~vals;
    Option.iter (Hcq.Mmio.blit_bytes (Hcq.Buffer.cpu_view slot) ~off:t.kernargs_segment_size) packet;
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

(* Static packet templates, patched and submitted by compiled host code. *)
module Encoded_queue = struct
  open Tolk_uop
  module U = Uop
  module D = Dtype

  let u32 n = U.const (Const.int D.uint32 n)
  let u64 n = U.const (Const.int D.uint64 n)
  let cast dtype src = U.cast ~src ~dtype
  let op op lhs rhs = U.alu_binary ~op ~lhs ~rhs
  let add = op Ops.Add
  let mul = op Ops.Mul
  let shr = op Ops.Shr
  let bor = op Ops.Or
  let sub a b = add a (U.alu_unary ~op:Ops.Neg ~src:b)
  let index ptr i = U.index ~ptr ~idxs:[i] ()
  let load ptr i = U.load ~src:(index ptr i) ()
  let store ptr i value = U.store ~dst:(index ptr i) ~value ()
  let addr name src = U.getaddr ~device:name ~src ()
  let i32 n = U.const (Const.int D.int32 n)
  let zero = i32 0
  let placeholder ?allocation ?(volatile = false) name tag dtype size =
    U.placeholder ~shape:[size] ~dtype ~slot:0 ~device:(U.Single name)
      ?allocation ~volatile () |> U.with_tag tag
  let words value =
    if D.itemsize (U.dtype value) = 8 then
      [cast D.uint32 value; cast D.uint32 (shr value (u64 32))]
    else [cast D.uint32 value]
  let buffer ~name ~tag ~after values =
    let words = List.concat_map words values in
    let size = List.length words * 4 in
    let buf = U.placeholder ~shape:[size] ~dtype:D.uint8
        ~slot:(U.fresh_buffer_slot ()) ~device:(U.Single name) () |> U.with_tag tag in
    Tolk.Hcq2.patch ~blob:(String.make size '\000') ~after buf
      (List.mapi (fun i word -> i * 4, word) words)
  let context name = U.placeholder ~shape:[2] ~dtype:D.uint64 ~slot:0
      ~device:(U.Single name) ~volatile:true ~allocation:("hcq_submission", "") ()
  let publish ~name ~after ~wptr ~doorbell ~next ~is_am ~lag =
    let flush = if is_am then addr "CPU"
        (placeholder name "hdp_flush" D.uint32 1 ~volatile:true) else u64 0 in
    Tolk.Hcq2.ccall ~host:name ~after ~name:"tolk_hcq_publish" ~dtype:D.void
      [index (context name) zero; index wptr zero; index doorbell zero;
       next; flush; u64 lag]

  let reserve ~name ~kind ~after ~put ~needed ~capacity =
    let rptr = placeholder name ("read_ptr_" ^ kind) D.uint64 1 ~volatile:true in
    let ready = Tolk.Hcq2.ccall ~host:name ~after ~name:"tolk_hcq_reserve" ~dtype:D.uint32
        [index (context name) zero; index rptr zero; put; needed; u64 capacity] in
    op Ops.Cmpeq ready (u32 1)
  let guarded_count ready count = U.alu_ternary ~op:Ops.Where ~a:ready ~b:count ~c:zero

  let lower = Hcq.Submission.lower

  let push ~name ~kind ~ring_size ~is_am ~dependency source ~unit ~lag =
    let ring = placeholder name ("ring_" ^ kind) D.uint32 (ring_size / 4) ~volatile:true in
    let wptr = placeholder name ("write_ptr_" ^ kind) D.uint64 1 ~volatile:true in
    let doorbell = placeholder name ("doorbell_" ^ kind) D.uint64 1 ~volatile:true in
    let p = load (U.after ~src:wptr ~deps:[dependency]) zero in
    let rs = i32 (ring_size / 4) and n = i32 (U.max_numel source / 4) in
    let tail = cast D.int32 (op Ops.Cmod (mul p (u64 (unit / 4))) (u64 (ring_size / 4))) in
    let remaining = sub rs tail in
    let first = U.alu_ternary ~op:Ops.Where ~a:(op Ops.Cmplt remaining n) ~b:remaining ~c:n in
    let ready = reserve ~name ~kind ~after:[dependency] ~put:p
        ~needed:(u64 (U.max_numel source / unit)) ~capacity:(ring_size / unit) in
    let source = U.bitcast ~src:source ~dtype:D.uint32 in
    let copy after dst src count axis =
      let i = U.range ~size:(guarded_count ready count) ~axis ~kind:Axis_type.Loop ~dtype:D.int32 ~parents:[after] () in
      U.end_ ~value:(store (U.after ~src:ring ~deps:[after]) (add dst i)
        (load source (add src i))) ~ranges:[i] in
    let copied = copy dependency tail zero first 10 in
    let copied = copy copied zero first (sub n first) 11 in
    let next = add p (u64 (U.max_numel source * 4 / unit)) in
    publish ~name ~after:[copied] ~wptr ~doorbell ~next ~is_am ~lag


  let encode (dev : 'meta device) ~props ~name ~compute_ring_size ~copy_ring_size u =
    match U.op u, U.arg u, U.children u with
    | Ops.Custom_function, U.Arg.String ("submit_amd_compute" | "submit_amd_copy" as kind),
        [linear; dependency] ->
        let compute = kind = "submit_amd_compute" in
        let module P = (val dev.pm4) in
        let module S = (val dev.sdma) in
        let commands = ref [] and aql_packets = ref [] and run_start = ref 0 in
        let command_address = U.variable ~name:"amd_command_address" ~min_val:0
            ~max_val:((1 lsl 48) - 1) ~dtype:D.uint64 () in
        let aql xs = aql_packets := List.rev_append (List.concat_map words xs) !aql_packets in
        let close_run () =
          let stop = List.length !commands * 4 in
          if stop > !run_start then
            aql ([u32 indirect_header; u32 (P.packet3 P.packet3_indirect_buffer 2);
              add command_address (u64 !run_start);
              u32 ((stop - !run_start) / 4 lor P.indirect_buffer_valid); u32 10]
              @ List.init 10 (fun _ -> u32 0));
          run_start := stop in
        let q xs = commands := List.rev_append (List.concat_map words xs) !commands in
        let pkt cmd xs =
          let xs = List.concat_map words xs in
          q (u32 (P.packet3 cmd (List.length xs - 1)) :: xs) in
        let wreg reg xs =
          let reg : Reg.t = Ip.reg dev.gc reg in
          let cmd, base =
            if reg.addr >= P.packet3_set_sh_reg_start && reg.addr < P.packet3_set_sh_reg_end then
              P.packet3_set_sh_reg, P.packet3_set_sh_reg_start
            else P.packet3_set_uconfig_reg, P.packet3_set_uconfig_reg_start in
          pkt cmd (u32 (reg.addr - base) :: xs) in
        let append_static f =
          let cq = Compute_queue.create dev in
          f cq; q (Array.to_list (Q.dwords (Compute_queue.q cq)) |> List.map u32) in
        let release ~timestamp signal value =
          let cq = Compute_queue.create dev in
          Compute_queue.release_mem cq ~data_sel:(if timestamp then
              P.data_sel__mec_release_mem__send_gpu_clock_counter
            else P.data_sel__mec_release_mem__send_32_bit_low)
            ~int_sel:(if timestamp then P.int_sel__mec_release_mem__none else
              P.int_sel__mec_release_mem__send_interrupt_after_write_confirm)
            ~cache_flush:(not timestamp) ();
          let packet = Q.dwords (Compute_queue.q cq) in
          let address = add (addr name signal) (u64 (if timestamp then 8 else 0)) in
          let args = [u32 packet.(1); u32 packet.(2); address; cast D.uint64 value; u32 packet.(7)] in
          if dev.xccs > 1 then pkt P.packet3_pred_exec [u32 ((1 lsl 24) lor 8)];
          pkt P.packet3_release_mem args in
        let dims xs = List.init 3 (fun i -> if i >= List.length xs then u32 1 else
            match List.nth xs i with
            | U.Launch_int n -> u32 n | U.Launch_float f -> u32 (int_of_float f)
            | U.Launch_sym v -> cast D.uint32 v) in
        let dispatch_packet ?(kernel_object = u64 0) ?(kernargs = u64 0) (data : Program.data) info =
          let local = dims info.U.local_size and global = dims info.U.global_size in
          [u32 dispatch_header; bor (List.nth local 0) (op Ops.Shl (List.nth local 1) (u32 16));
           List.nth local 2] @ List.map2 mul global local @
          [u32 data.private_segment_size; u32 data.group_segment_size;
           kernel_object; kernargs; u64 0; u64 0] in
        let kernargs body args =
          let info = Option.get (U.as_program_info body) in
          let object_ = U.to_elf body in
          let data, image = Program.image ~target:dev.target ~props object_.lib in
          let program = placeholder ~allocation:("amd_image", Marshal.to_string (data.private_segment_size, Bytes.to_string image) [])
              name "program" D.uint8 (Bytes.length image) in
          let buffers = List.filter (fun a -> not (U.is_bound_var a)) args in
          let bound = List.filter_map (fun a -> match U.as_bind a with
              | Some {var; value} -> Option.map (fun n -> n, value) (U.program_var_name var)
              | None -> None) args in
          let vars = List.map (fun v -> match U.program_var_name v with
              | Some n -> Option.value (List.assoc_opt n bound) ~default:v | None -> v) info.vars in
          let actuals = List.map (fun i -> addr name (List.nth buffers i)) info.globals @ vars in
          let rows = Tiny_elf.layout object_.signature |> List.map (fun (f : Tiny_elf.field) ->
              let dtype = if f.argument.addrspace = D.Alu then f.argument.dtype else D.uint64 in
              f.offset, cast dtype (List.nth actuals f.argument.slot)) in
          let size = data.Program.kernargs_segment_size +
            (if data.enable_dispatch_ptr then Amd_hsa_defs.Kernel_dispatch_packet.size else 0) in
          let arena = U.placeholder ~shape:[max 8 (round_up size 8)] ~dtype:D.uint8
              ~slot:(U.fresh_buffer_slot ()) ~device:(U.Single name) () |> U.with_tag "kernargs" in
          let rows = if data.enable_dispatch_ptr then rows @
              (List.concat_map words (dispatch_packet data info) |> List.mapi
                (fun i value -> data.kernargs_segment_size + i * 4, value)) else rows in
          let arena = Tolk.Hcq2.patch ~after:[dependency] arena rows in
          data, program, arena, info in
        List.iter (fun node -> match U.as_call node, U.arg node with
          | Some {body; args}, _ when U.op body = Ops.Program && compute ->
              let data, program, arena, info = kernargs body args in
              if dev.is_aql then begin
                close_run ();
                aql (dispatch_packet data info ~kernel_object:(add (addr name program) (u64 data.desc_offset))
                  ~kernargs:(addr name arena))
              end else begin
              let local = dims info.local_size and global = dims info.global_size in
              let size, tmpring_size = scratch_layout dev ~props data.private_segment_size in
              let scratch = placeholder ~allocation:("amd_scratch", string_of_int data.private_segment_size)
                  name "scratch" D.uint8 size |> addr name in
              append_static (fun cq -> Compute_queue.acquire_mem cq ~gli:0 ~gl2:0 ());
              wreg "regCOMPUTE_PGM_LO" [shr (add (addr name program) (u64 data.entry_offset)) (u64 8)];
              wreg "regCOMPUTE_PGM_RSRC1" [u32 data.rsrc1; u32 data.rsrc2];
              wreg "regCOMPUTE_PGM_RSRC3" [u32 data.rsrc3];
              wreg "regCOMPUTE_TMPRING_SIZE" [u32 tmpring_size];
              wreg "regCOMPUTE_DISPATCH_SCRATCH_BASE_LO" [shr scratch (u64 8)];
              wreg "regCOMPUTE_RESTART_X" [u32 0; u32 0; u32 0];
              let user = if data.enable_private_segment_sgpr then
                  [bor scratch (U.const (Const.int64 D.uint64 Int64.min_int)); u32 0xffffffff; u32 0x20c14000]
                else [] in
              let user = if data.enable_dispatch_ptr then
                  user @ [add (addr name arena) (u64 data.kernargs_segment_size)] else user in
              wreg "regCOMPUTE_USER_DATA_0" (user @ [addr name arena]);
              wreg "regCOMPUTE_RESOURCE_LIMITS" [u32 (Reg.encode (Ip.reg dev.gc "regCOMPUTE_RESOURCE_LIMITS")
                  ["waves_per_sh", Tolk.Helpers.getenv "WAVES_PER_SH" 0])];
              wreg "regCOMPUTE_START_X" ([u32 0; u32 0; u32 0] @ local @ [u32 0; u32 0]);
              let initiator = Reg.encode (Ip.reg dev.gc "regCOMPUTE_DISPATCH_INITIATOR")
                  (["force_start_at_000", 1; "compute_shader_en", 1] @
                   if major dev.target = 9 then [] else ["cs_w32_en", Bool.to_int data.wave32]) in
              pkt P.packet3_dispatch_direct (global @ [u32 initiator]);
              let module Soc = (val dev.soc) in
              pkt P.packet3_event_write [u32 (P.event_type Soc.cs_partial_flush lor P.event_index event_index_partial_flush)]
              end
          | Some {body; args = [dst; src]}, _ when U.op body = Ops.Store && not compute ->
              let bytes = U.max_numel dst * D.itemsize (U.dtype dst) in
              let offset = ref 0 in
              while !offset < bytes do
                let size = min dev.max_copy_size (bytes - !offset) in
                q [u32 (S.sdma_op_copy lor S.sdma_pkt_copy_linear_header_sub_op S.sdma_subop_copy_linear);
                   u32 (size - 1); u32 0; add (addr name src) (u64 !offset); add (addr name dst) (u64 !offset)];
                offset := !offset + size
              done
          | _, U.Arg.Typed ("barrier", _) -> if compute then append_static Compute_queue.memory_barrier
          | _, U.Arg.Typed ("wait", _) ->
              let args = U.src node in
              if compute then pkt P.packet3_wait_reg_mem [u32 (P.wait_reg_mem_mem_space 1 lor
                  P.wait_reg_mem_operation 0 lor P.wait_reg_mem_function wait_reg_mem_function_geq lor P.wait_reg_mem_engine 0);
                addr name args.(0); cast D.uint32 args.(1); u32 0xffffffff; u32 4]
              else q [u32 (S.sdma_op_poll_regmem lor S.sdma_pkt_poll_regmem_header_func wait_reg_mem_function_geq
                  lor S.sdma_pkt_poll_regmem_header_mem_poll 1); addr name args.(0); cast D.uint32 args.(1);
                u32 0xffffffff; u32 (S.sdma_pkt_poll_regmem_dw5_interval 4 lor S.sdma_pkt_poll_regmem_dw5_retry_count 0xfff)]
          | _, U.Arg.Typed ("store", _) ->
              let args = U.src node in
              if compute then begin
                if dev.is_aql then close_run ();
                release ~timestamp:false args.(0) args.(1)
              end
              else q [u32 (S.sdma_op_fence lor (if major dev.target = 9 then 0 else
                    Amd_sdma_defs.V6_0_0.sdma_pkt_fence_header_mtype 3));
                  addr name args.(0); cast D.uint32 args.(1); u32 S.sdma_op_trap; u32 0]
          | _, U.Arg.Typed ("timestamp", _) ->
              let signal = (U.src node).(0) in
              if compute then release ~timestamp:true signal (u64 0)
              else q [u32 (S.sdma_op_timestamp lor S.sdma_pkt_timestamp_get_header_sub_op
                  S.sdma_subop_timestamp_get_global); add (addr name signal) (u64 8)]
          | _ -> invalid_arg "AMD queue: unsupported instruction") (U.children linear);
        if compute && dev.is_aql then close_run ();
        let stream = buffer ~name:(if compute then name else "CPU")
            ~tag:(if compute then "cmdbuf_compute" else "cmdbuf_copy")
            ~after:[dependency] (List.rev !commands) in
        if compute && dev.is_aql then begin
          let packets = List.rev !aql_packets |> List.map (U.substitute ~walk:true
              [command_address, addr name stream]) in
          let aql = buffer ~name:"CPU" ~tag:"aql_compute" ~after:[stream] packets in
          Some (push ~name ~kind:"compute" ~ring_size:compute_ring_size ~is_am:dev.is_am
            ~dependency:stream aql ~unit:64 ~lag:1)
        end else if compute then begin
          let ib = buffer ~name:"CPU" ~tag:"ib_compute" ~after:[stream]
              [u32 (P.packet3 P.packet3_indirect_buffer 2); addr name stream;
               u32 (U.max_numel stream / 4 lor P.indirect_buffer_valid)] in
          Some (push ~name ~kind:"compute" ~ring_size:compute_ring_size ~is_am:dev.is_am
            ~dependency:stream ib ~unit:4 ~lag:0)
        end else begin
          let ring_size = match copy_ring_size with Some size -> size
            | None -> invalid_arg "AMD device has no SDMA queue" in
          let size = U.max_numel stream / 4 and rs = ring_size / 4 in
          if size >= rs then invalid_arg "AMD SDMA command stream exceeds its ring";
          let ring = placeholder name "ring_copy" D.uint32 rs ~volatile:true in
          let wptr = placeholder name "write_ptr_copy" D.uint64 1 ~volatile:true in
          let bell = placeholder name "doorbell_copy" D.uint64 1 ~volatile:true in
          let p = load (U.after ~src:wptr ~deps:[dependency]) zero in
          let tail = cast D.int32 (op Ops.Cdiv (op Ops.Cmod p (u64 ring_size)) (u64 4)) in
          let fits = cast D.int32 (op Ops.Cmplt (i32 (size - 1)) (sub (i32 rs) tail)) in
          let start = mul fits tail in
          let padding = mul (sub (i32 1) fits) (sub (i32 rs) tail) in
          let room = reserve ~name ~kind:"copy" ~after:[stream] ~put:p
              ~needed:(cast D.uint64 (mul padding (i32 4))) ~capacity:ring_size in
          let z = U.range ~size:(guarded_count room padding) ~axis:10 ~kind:Axis_type.Loop
              ~dtype:D.int32 ~parents:[stream] () in
          let cleared = U.end_ ~value:(store ring (add tail z) (u32 0)) ~ranges:[z] in
          let padded = add p (cast D.uint64 (mul padding (i32 4))) in
          let published = publish ~name ~after:[cleared] ~wptr ~doorbell:bell ~next:padded
              ~is_am:dev.is_am ~lag:0 in
          let room = reserve ~name ~kind:"copy" ~after:[published] ~put:padded
              ~needed:(u64 (size * 4)) ~capacity:ring_size in
          let i = U.range ~size:(guarded_count room (i32 size)) ~axis:11 ~kind:Axis_type.Loop
              ~dtype:D.int32 ~parents:[published] () in
          let copied = U.end_ ~value:(store (U.after ~src:ring ~deps:[published]) (add start i)
              (load (U.bitcast ~src:stream ~dtype:D.uint32) i)) ~ranges:[i] in
          let next = add p (cast D.uint64 (mul (add padding (i32 size)) (i32 4))) in
          Some (publish ~name ~after:[copied] ~wptr ~doorbell:bell ~next
            ~is_am:dev.is_am ~lag:0)
        end
    | _ -> None
end


(* Kernel-driver interface *)

module Kfd_iface = struct
  type ownership = Owned | Registered | Imported
  type mem = { handle : int64; owner : int; ownership : ownership }
  let kind : mem Hcq.Buffer.t Type.Id.t = Type.Id.make ()

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
        let gpus = Array.of_list
            (Tolk_hcq.System.filter_visible_devices "AMD" (Array.to_list gpus)) in
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

  let discover_ips sysfs_path : ip_versions =
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
    if Option.is_some cpu_addr && (not host || uncached) then
      invalid_arg "KFD cpu_addr requires host registration";
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
    let complete = ref false in
    Fun.protect
      ~finally:(fun () -> if not !complete && cpu_addr = None then F.munmap addr ~size)
      (fun () ->
        let mmap_offset = if userptr then Int64.of_nativeint addr else 0L in
        match Kfd.alloc_memory_of_gpu kfd ~va:addr ~size ~gpu_id ~flags ~mmap_offset with
        | Error e ->
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
            Fun.protect
              ~finally:(fun () -> if not !complete then Kfd.free_memory_of_gpu kfd ~handle)
              (fun () ->
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
                  Hcq.Buffer.make ~va:addr ~size ?view ~meta:{ handle; owner = gpu_id;
                    ownership = (if cpu_addr = None then Owned else Registered) }
                    ()
                in
                map_to_gpu ~kfd ~gpu_id b;
                complete := true;
                b))

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
    if meta.ownership <> Imported && meta.owner <> t.gpu_id then
      invalid_arg "KFD free requires the owning interface";
    Kfd.unmap_memory_from_gpu kfd ~handle:meta.handle ~gpu_ids:[| t.gpu_id |];
    if meta.ownership <> Imported then begin
      if meta.ownership = Owned && Hcq.Buffer.va b <> 0n then
        Hcq.File_io.munmap (Hcq.Buffer.va b) ~size:(Hcq.Buffer.size b);
      Kfd.free_memory_of_gpu kfd ~handle:meta.handle
    end

  let map t b =
    let kfd, _ = scan () in
    map_to_gpu ~kfd ~gpu_id:t.gpu_id b;
    Hcq.Buffer.make ~va:(Hcq.Buffer.va b) ~size:(Hcq.Buffer.size b)
      ~meta:{ (Hcq.Buffer.meta b) with ownership = Imported } ()

  let map_storage t source =
    let module B = Tolk.Device.Buffer in
    let Tolk.Device.Allocator.Pack allocator = B.allocator source in
    match Type.Id.provably_equal kind allocator.kind with
    | Some Type.Equal -> map t (Option.get (B.get kind source))
    | None ->
        let address = match B.host_addr source with
          | Some address -> address
          | None -> raise (Tolk_uop.Storage.Mapping_unavailable "KFD map requires KFD or host-accessible storage") in
        if Nativeint.logand address 0xfffn <> 0n then
          raise (Tolk_uop.Storage.Mapping_unavailable "KFD host mapping requires page alignment");
        alloc t ~host:true ~cpu_access:true ~cpu_addr:address
          (round_up (B.nbytes source) 0x1000)

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
          | Compute_aql -> Kfd.queue_type_compute_aql
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
      Queue_desc.aql = None;
      ring =
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
      hdp_flush = None;
      resetup = None;
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
    let report =
      match report with
      | [] -> [ "no memory or hardware fault reported by the driver" ]
      | r -> r
    in
    failwith (String.concat "\n" report)

  let sleep t ~timeout_ms =
    poll_events t ~timeout_ms;
    if t.mem_fault <> None || t.hw_fault <> None then on_device_hang t

  let iface t =
    {
      Iface.props = t.props;
      ip_versions = t.ip_versions;
      is_am = false;
      queue_event = t.queue_event;
      queue_event_mailbox_ptr = t.queue_event_mailbox_ptr;
      alloc =
        (fun ?host ?uncached ?cpu_access size ->
          alloc t ?host ?uncached ?cpu_access size);
      free = free t;
      kind;
      map = map_storage t;
      unmap = free t;
      empty_scratch =
        Hcq.Buffer.make ~va:0n ~size:0
          ~meta:{ handle = 0L; owner = 0; ownership = Imported } ();
      create_queue =
        (fun queue_type ~ring ~gart ~rptr ~wptr ?eop_buffer ?cwsr_buffer
             ?ctl_stack_size ?ctx_save_restore_size () ->
          create_queue t queue_type ~ring ~gart ~rptr ~wptr ?eop_buffer
            ?cwsr_buffer ?ctl_stack_size ?ctx_save_restore_size ());
      (* long waits back off to driver-event sleeps, which also surface
         faults *)
      sleep =
        (fun spent_ms -> if spent_ms > 200 then sleep t ~timeout_ms:200);
      on_device_hang = (fun () -> on_device_hang t);
      register = None;
      after_sync = None;
      device_fini = None;
    }
end

(* Driver-less PCI interface: ops_amd.py:843-908 PCIIface *)

module Pci_iface = struct
  module Base = System.Pci_iface_base
  module Am_defs = Amd_tables.Am_defs

  type mem = Base.mem

  type t = {
    base : (Am_boot.t, Amdev.Am_page_table.t) Base.t;
    props : (string * int) list;
    ip_versions : ip_versions;
  }

  (* ops_amd.py:845: the supported consumer PCI ids (RDNA3/RDNA4) *)
  let vendor = 0x1002

  let pci_ids =
    [ (0xffff, [ 0x74a1; 0x744c; 0x7480; 0x7550; 0x7551; 0x7590; 0x75a0 ]) ]

  let am t = Base.dev_impl t.base

  (* ops_amd.py:852 _compute_props: synthesize the topology properties
     the kernel driver would publish, from the discovery table *)
  let compute_props ~gc_info ~gc_ver:(ma, mi, rv) ~xccs =
    let gfxver = (ma * 10000) + (mi * 100) + rv in
    let cu_per_sa, max_sh_per_se, num_se, max_slots, max_waves, lds =
      match gc_info with
      | Amdev.Gc_info_v2 g ->
          ( g.num_cu_per_sh,
            g.num_sh_per_se,
            g.num_se,
            g.max_scratch_slots_per_cu,
            g.max_waves_per_simd,
            g.lds_size )
      | Amdev.Gc_info_v1 g ->
          ( 2 * (g.num_wgp0_per_sa + g.num_wgp1_per_sa),
            g.num_sa_per_se,
            g.num_se,
            g.max_scratch_slots_per_cu,
            g.max_waves_per_simd,
            g.lds_size )
    in
    let array_count = max_sh_per_se * num_se * xccs in
    [
      ("cu_per_simd_array", cu_per_sa);
      ("simd_count", 2 * cu_per_sa * array_count);
      ("simd_per_cu", 2);
      ("array_count", array_count);
      ("max_slots_scratch_cu", max_slots);
      ("max_waves_per_simd", max_waves);
      ("simd_arrays_per_engine", max_sh_per_se);
      ("lds_size_in_kb", lds);
      ("num_xcc", xccs);
      ("gfx_target_version", if gfxver = 90403 then 90402 else gfxver);
    ]

  (* ops_amd.py:844 PCIIface.__init__ *)
  let create ~device_id =
    let base =
      Base.create ~name:"AMD" ~devpref:"AM" ~dev_id:device_id ~vendor
        ~devices:pci_ids ~vram_bar:0
        ~va_start:(Nativeint.of_int Amdev.va_base)
        ~va_size:Amdev.va_size
        ~dev_impl:(fun pci_dev ->
          let boot = Am_boot.create (Amdev.create pci_dev) in
          Am_boot.init boot;
          boot)
        ~mm:(fun boot -> Amdev.mm boot.Am_boot.adev)
        ()
    in
    let boot = Base.dev_impl base in
    let ip_ver hwip = Amdev.ip_ver boot.Am_boot.adev hwip in
    {
      base;
      props =
        compute_props
          ~gc_info:(Amdev.gc_info boot.Am_boot.adev)
          ~gc_ver:(ip_ver Am_defs.gc_hwip)
          ~xccs:(Am_ip.Gfx.xccs boot.Am_boot.gfx);
      ip_versions =
        {
          gc = ip_ver Am_defs.gc_hwip;
          sdma = ip_ver Am_defs.sdma0_hwip;
          nbif = ip_ver Am_defs.nbif_hwip;
        };
    }

  let alloc t ?host ?uncached ?cpu_access size =
    Base.alloc t.base ?host ?uncached ?cpu_access size

  let free t b = Base.free t.base b

  (* ops_amd.py:877 PCIIface.create_queue *)
  let create_queue t queue_type ~ring ~gart ~rptr ~wptr ?eop_buffer
      ?cwsr_buffer ?(ctl_stack_size = 0) ?(ctx_save_restore_size = 0) () =
    (* the driver-less path has no compute-wave save/restore *)
    if cwsr_buffer <> None || ctl_stack_size <> 0 || ctx_save_restore_size <> 0
    then invalid_arg "Pci_iface.create_queue: no cwsr state for am";
    let boot = am t in
    let ring_addr = Nativeint.to_int (Hcq.Buffer.va ring) in
    let ring_size = Hcq.Buffer.size ring in
    let rptr_addr = Nativeint.to_int (Hcq.Buffer.va gart) + rptr in
    let wptr_addr = Nativeint.to_int (Hcq.Buffer.va gart) + wptr in
    let setup =
      match queue_type with
      | Sdma ->
          fun () ->
            Am_ip.Sdma.setup_ring boot.Am_boot.sdma ~ring_addr ~ring_size
              ~rptr_addr ~wptr_addr ~idx:0
      | Compute | Compute_aql ->
          let eop =
            match eop_buffer with
            | Some eop -> eop
            | None ->
                invalid_arg
                  "Pci_iface.create_queue: compute queues need an eop buffer"
          in
          fun () ->
            Am_ip.Gfx.setup_ring boot.Am_boot.gfx ~ring_addr ~ring_size
              ~rptr_addr ~wptr_addr
              ~eop_addr:(Nativeint.to_int (Hcq.Buffer.va eop))
              ~eop_size:(Hcq.Buffer.size eop) ~idx:0 ~aql:(queue_type = Compute_aql)
    in
    let doorbell_index = setup () in
    {
      Queue_desc.aql = None;
      ring = Hcq.Buffer.cpu_view ring;
      read_ptr = Hcq.Mmio.view (Hcq.Buffer.cpu_view gart) ~off:rptr ~size:8 ();
      write_ptr = Hcq.Mmio.view (Hcq.Buffer.cpu_view gart) ~off:wptr ~size:8 ();
      doorbell =
        Hcq.Mmio.view
          (Amdev.doorbell64 boot.Am_boot.adev)
          ~off:(doorbell_index * 8) ~size:8 ();
      hdp_flush = Some (Hcq.Mmio.view (Amdev.mmio boot.Am_boot.adev)
        ~off:(Amdev.Am_register.read (Amdev.reg boot.Am_boot.adev
          "regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL")) ~size:4 ());
      resetup = Some (fun () -> ignore (setup () : int));
    }

  (* The open driver-less devices, so any wait can collect interrupts
     for all of them (the reference walks its global device table; the
     runtime here has none, so the interface keeps its own). The entry
     hides the timeline's buffer metadata, so scripted devices register
     like real ones. *)
  type registered =
    | Registered : {
        r_am : Am_boot.t;
        r_compute : Queue_desc.t;
        r_tl : ('mem, 'mem device) Timeline.t;
      }
        -> registered

  let registry : registered list ref = ref []

  let register ~am ~compute_queue ~tl =
    registry :=
      Registered { r_am = am; r_compute = compute_queue; r_tl = tl }
      :: !registry

  let unregister am =
    registry := List.filter (fun (Registered r) -> r.r_am != am) !registry

  (* ops_amd.py:891 _collect_interrupts *)
  let collect_interrupts ?(reset = false) ?(drain_only = false) () =
    List.iter
      (fun entry ->
        match entry with
        | Registered r ->
            let boot = r.r_am in
            if drain_only then Am_ip.Ih.drain boot.Am_boot.ih
            else
              Am_ip.Ih.interrupt_handler boot.Am_boot.ih ~soc:boot.Am_boot.soc
                ~gmc:boot.Am_boot.gmc ~smu:boot.Am_boot.smu;
            if
              reset
              && Am_boot.recover boot
                   ~force:(r.r_tl.Timeline.error_state <> None)
            then begin
              (* the processors lost their queues: rebuild the compute
                 queue and rewind the timeline to the last completed
                 value *)
              Hcq.Mmio.write64 r.r_compute.Queue_desc.read_ptr 0 0L;
              Hcq.Mmio.write64 r.r_compute.Queue_desc.write_ptr 0 0L;
              (match r.r_compute.Queue_desc.resetup with
              | Some resetup -> resetup ()
              | None -> ());
              Hcq.Signal.set_value r.r_tl.Timeline.timeline
                (Timeline.submitted r.r_tl);
              r.r_tl.Timeline.error_state <- None
            end)
      !registry

  (* Protocol timeouts during interrupt collection surface as the
     device's fault report, so the timeline latches them like any other
     fault. *)
  let as_fault f = try f () with Am_ip.Timeout_error msg -> failwith msg

  (* ops_amd.py:898 PCIIface.sleep *)
  let sleep boot ~timeout_ms =
    as_fault (fun () ->
        (match Amdev.pci_dev boot.Am_boot.adev with
        | Some pci_dev -> System.Pci_device.wait_irq pci_dev ~timeout_ms
        | None -> ());
        collect_interrupts ();
        if Amdev.is_err_state boot.Am_boot.adev then
          failwith "Device is in error state")

  (* ops_amd.py:903 on_device_hang *)
  let on_device_hang () =
    as_fault (fun () -> collect_interrupts ~reset:true ());
    failwith "Device hang detected"

  let iface t =
    {
      Iface.props = t.props;
      ip_versions = t.ip_versions;
      is_am = true;
      (* driver-less devices have no driver events; signal packets skip
         the mailbox when the owner is_am *)
      queue_event = { event_id = 0 };
      queue_event_mailbox_ptr = 0n;
      alloc =
        (fun ?host ?uncached ?cpu_access size ->
          alloc t ?host ?uncached ?cpu_access size);
      free = free t;
      kind = Base.kind;
      map = Base.map t.base;
      unmap = Base.unmap t.base;
      empty_scratch = Base.empty;
      create_queue =
        (fun queue_type ~ring ~gart ~rptr ~wptr ?eop_buffer ?cwsr_buffer
             ?ctl_stack_size ?ctx_save_restore_size () ->
          create_queue t queue_type ~ring ~gart ~rptr ~wptr ?eop_buffer
            ?cwsr_buffer ?ctl_stack_size ?ctx_save_restore_size ());
      sleep =
        (fun spent_ms ->
          if spent_ms > 200 then sleep (am t) ~timeout_ms:200);
      on_device_hang = (fun () -> on_device_hang ());
      register =
        Some (fun ~compute_queue ~tl -> register ~am:(am t) ~compute_queue ~tl);
      after_sync = Some (fun () -> collect_interrupts ~drain_only:true ());
      device_fini = Some (fun () -> Am_boot.fini (am t));
    }
end

(* Device runtime *)

module State = struct
  type 'mem t = {
    name : string;
    buffer_kind : 'mem Hcq.Buffer.t Type.Id.t;
    iface : 'mem Iface.t;
    hw : 'mem device;
    compute_queue : Queue_desc.t;
    sdma_queue : Queue_desc.t option;
    kernargs : 'mem Hcq.Kernargs.t;
    pool : 'mem Hcq.Signal.Pool.t;
    tl : ('mem, 'mem device) Timeline.t;
    submission : Hcq.Submission.t;
    mutable scratch : Tolk.Device.Buffer.t option;
    (* The device's LRU-wrapped allocator; set right after creation and used
       for scratch sizing. *)
    mutable allocator : 'mem Hcq.Buffer.t Tolk.Device.Allocator.t option;
  }

  let check_submission t =
    Timeline.guarded_wait t.tl (fun () -> Hcq.Submission.check t.submission)

  let prepare t =
    check_submission t;
    Timeline.prepare t.tl;
    Hcq.Submission.prepare ~timeout_ms:(Tolk.Helpers.getenv "HCQ_TIMEOUT_MS" 30000)
      t.submission

  let invalidate_caches t =
    prepare t;
    let cq = Compute_queue.create t.hw in
    Compute_queue.memory_barrier cq;
    Compute_queue.signal cq
      ~value:(Timeline.next_timeline t.tl)
      t.tl.Timeline.timeline;
    Compute_queue.submit cq t.compute_queue;
    Timeline.synchronize t.tl

  let synchronize t =
    check_submission t;
    Timeline.synchronize t.tl;
    match t.iface.Iface.after_sync with
    | Some after_sync -> after_sync ()
    | None -> ()
end

module Allocator = struct
  (* One DMA stream ordered against the device timeline: wait for the last
     submitted work, append the packets of [build], advance the timeline. *)
  let submit_copy state qd build =
    let tl = state.State.tl in
    State.prepare state;
    let cp = Copy_queue.create state.State.hw in
    Copy_queue.wait cp
      ~value:(Timeline.submitted tl)
      tl.Timeline.timeline;
    build cp;
    Copy_queue.signal cp ~value:(Timeline.next_timeline tl) tl.Timeline.timeline;
    Copy_queue.submit cp qd

  let submit_chunk state qd ~dest ~src len =
    submit_copy state qd (fun cp -> Copy_queue.copy cp ~dest ~src len)

  let copyin state buf bytes =
    match state.State.sdma_queue with
    | None ->
        (* Without a DMA engine every buffer is host-visible: write the
           mapping directly once the device is idle. *)
        Timeline.synchronize state.State.tl;
        Hcq.Mmio.blit_bytes (Hcq.Buffer.cpu_view buf) ~off:0 bytes
    | Some qd ->
        Timeline.copyin state.State.tl ~submit_chunk:(submit_chunk state qd)
          buf bytes

  let copyout state bytes buf =
    Timeline.synchronize state.State.tl;
    match state.State.sdma_queue with
    | None ->
        let len = Bytes.length bytes in
        Bytes.blit
          (Hcq.Mmio.read_bytes (Hcq.Buffer.cpu_view buf) ~off:0 ~len)
          0 bytes 0 len
    | Some qd ->
        Timeline.copyout state.State.tl ~submit_chunk:(submit_chunk state qd)
          bytes buf

  let transfer state ~dest ~src ~dest_device ~src_device nbytes =
    if Tolk.Device.canonicalize dest_device <> Tolk.Device.canonicalize src_device then false
    else begin
      let qd = Option.get state.State.sdma_queue in
      submit_copy state qd (fun cp -> Copy_queue.copy cp ~dest ~src nbytes);
      true
    end

  let raw state =
    let alloc size (spec : Tolk.Device.Buffer_spec.t) =
      match spec.external_ptr with
      | Some _ ->
          invalid_arg "AMD buffers cannot adopt an external pointer"
      | None ->
          (* Without a DMA engine host transfers write the CPU mapping, so
             every allocation must be host-visible. *)
          state.State.iface.Iface.alloc ~host:spec.host
            ~uncached:spec.uncached
            ~cpu_access:
              (spec.cpu_access || Option.is_none state.State.sdma_queue)
            size
    in
    (* A queued kernel may still use the memory. *)
    let free buf _size (_ : Tolk.Device.Buffer_spec.t) =
      State.synchronize state;
      state.State.iface.Iface.free buf
    in
    let offset buf size byte_offset =
      Hcq.Buffer.offset buf ~off:byte_offset ~size ()
    in
    let has_sdma = Option.is_some state.State.sdma_queue in
    {
      Tolk.Device.Allocator.kind = state.State.buffer_kind;
      host = (fun buf -> Option.map Hcq.Mmio.addr (Hcq.Buffer.view buf));
      mapping = Some {
        map = state.State.iface.Iface.map;
        unmap = (fun b -> State.synchronize state; state.State.iface.Iface.unmap b);
      };
      synchronize = (fun () -> State.synchronize state);
      alloc;
      free;
      copyin = copyin state;
      copyout = copyout state;
      addr = Some Hcq.Buffer.va;
      offset = Some offset;
      transfer = (if has_sdma then Some (transfer state) else None);
      supports_transfer = has_sdma;
      copy_from_disk = None;
      supports_copy_from_disk = false;
    }

  let create state =
    let allocator = Tolk.Device.Lru_allocator.wrap (raw state) in
    state.State.allocator <- Some allocator;
    Tolk.Device.Allocator.Pack allocator
end

module Runtime = struct
  (* Retained links own the backing they captured. Growing the device's
     current scratch drops only its reference, so older programs keep valid
     addresses until their links are released. *)
  let ensure_scratch state size =
    if state.State.hw.is_aql && state.State.hw.max_private_segment_size < max size 128 then
      State.synchronize state;
    let owner = ref None in
    ensure_has_local_memory state.State.hw ~props:state.State.iface.Iface.props
      ~alloc:(fun size ->
        let spec = {Tolk.Device.Buffer_spec.default with nolru = true} in
        let buffer = Tolk.Device.Buffer.create ~device:state.State.name ~size
            ~dtype:Tolk_uop.Dtype.uint8 ~spec
            (Tolk.Device.Allocator.Pack (Option.get state.State.allocator)) in
        Tolk.Device.Buffer.ensure_allocated buffer;
        let raw = Option.get (Tolk.Device.Buffer.get state.State.buffer_kind buffer) in
        owner := Some buffer;
        raw)
      ~free:ignore size;
    Option.iter (fun buffer ->
        state.State.scratch <- Some buffer;
        Option.iter (Queue_desc.update_scratch state.State.hw) state.State.compute_queue.Queue_desc.aql) !owner

  let default_local = [| 1; 1; 1 |]

  let runtime state (obj : Tolk_uop.Tiny_elf.t) =
    let name = obj.name and lib = obj.lib in
    let layout = Tolk_uop.Tiny_elf.layout obj.signature in
    let prg =
      Program.load state.State.hw
        ~alloc:(fun size ->
          state.State.iface.Iface.alloc ~cpu_access:true size)
        ~props:state.State.iface.Iface.props ~name lib
    in
    (match ensure_scratch state prg.Program.private_segment_size with
     | () -> ()
     | exception exn ->
         let backtrace = Printexc.get_raw_backtrace () in
         Program.free ~free:state.State.iface.Iface.free prg;
         Printexc.raise_with_backtrace exn backtrace);
    let call bufs ~global ~local ~vals ~wait ~timeout:_ =
      State.prepare state;
      let bufs = Array.map (fun buf ->
          match Tolk.Device.Buffer.get ~device:state.State.name state.State.buffer_kind buf with
          | Some raw -> Hcq.Buffer.va raw | None -> 0n) bufs in
      let local = Option.value local ~default:default_local in
      let tl = state.State.tl in
      let timeline_value = Timeline.next_timeline tl in
      let launch ?timing () =
        Program.call prg ~layout ~kernargs:state.State.kernargs
          ~queue:state.State.compute_queue ~timeline:tl.Timeline.timeline
          ~timeline_value ?wait:timing ~bufs ~vals
          ~global_size:(global.(0), global.(1), global.(2))
          ~local_size:(local.(0), local.(1), local.(2))
          ()
      in
      if not wait then
        (try launch () with Hcq.Signal.Timeout _ as exn ->
          Timeline.guarded_wait tl (fun () -> raise exn))
      else begin
        (match tl.Timeline.error_state with Some e -> raise e | None -> ());
        let st_slot = Hcq.Signal.Pool.get state.State.pool in
        let en_slot = Hcq.Signal.Pool.get state.State.pool in
        Fun.protect
          ~finally:(fun () ->
            Hcq.Signal.Pool.put state.State.pool en_slot;
            Hcq.Signal.Pool.put state.State.pool st_slot)
          (fun () ->
            let st = Hcq.Signal.make ~timestamp_divider:100. st_slot in
            let en = Hcq.Signal.make ~timestamp_divider:100. en_slot in
            Timeline.guarded_wait tl (fun () -> launch ~timing:(st, en) ()))
      end
    in
    let free () =
      Program.free ~free:state.State.iface.Iface.free prg
    in
    { Tolk.Device.call; free; handle = 0n }
end

module Queue = struct
  open Tolk
  open Tolk_uop
  module U = Uop
  module B = Device.Buffer

  let bufferize state u =
    let name = state.State.name in
    let size = U.max_numel u and dtype = U.dtype u in
    let borrow_view view =
      let allocator = Storage.Host_allocator.make ~synchronize:(fun () -> State.synchronize state) in
      let spec = {Device.Buffer_spec.default with external_ptr = Some (Hcq.Mmio.addr view); nolru = true} in
      B.create ~device:"CPU" ~size ~dtype ~spec (Device.Allocator.Pack allocator) in
    let borrowed raw =
      let allocator = { (Allocator.raw state) with
        alloc = (fun _ _ -> raw); free = (fun _ _ _ -> State.synchronize state) } in
      B.create ~device:name ~size ~dtype
        ~spec:{Device.Buffer_spec.default with nolru = true} (Device.Allocator.Pack allocator) in
    let allocate ?(host = false) ?(cpu_access = true) () =
      let spec = {Device.Buffer_spec.default with host; cpu_access; nolru = true} in
      B.create ~device:name ~size ~dtype ~spec (Device.Allocator.Pack (Allocator.raw state)) in
    match U.as_param u with
    | Some {param = {allocation = Some ("hcq_submission", _); _}; _} ->
        Some (Hcq.Submission.buffer state.State.submission)
    | Some {param = {allocation = Some ("cfunc", data); _}; _} ->
        let libs, symbol = (Marshal.from_string data 0 : string list * string) in
        if libs <> [] then invalid_arg "AMD host helpers do not load libraries";
        let allocator = Storage.Host_allocator.make ~synchronize:(fun () -> ()) in
        let b = B.create ~device:"CPU" ~size:1 ~dtype:Dtype.uint64 (Device.Allocator.Pack allocator) in
        let bytes = Bytes.create 8 in
        Bytes.set_int64_le bytes 0 (Int64.of_nativeint (Hcq.Submission.symbol symbol));
        B.ensure_allocated b; B.copyin b bytes; Some b
    | Some {param = {allocation = Some ("amd_image", data); _}; _} ->
        let requested, image = (Marshal.from_string data 0 : int * string) in
        Runtime.ensure_scratch state requested;
        let b = allocate () in
        B.ensure_allocated b; B.copyin b (Bytes.of_string image); Some b
    | Some {param = {allocation = Some ("amd_scratch", requested); _}; _} ->
        Runtime.ensure_scratch state (int_of_string requested);
        state.State.scratch
    | Some _ ->
        (match U.node_tag u with
         | Some "timeline" -> Some (borrowed (Hcq.Signal.buf state.State.tl.Timeline.timeline))
         | Some "slots" -> Some (allocate ~host:true ())
         | Some ("kernargs" | "cmdbuf_compute") -> Some (allocate ())
         | Some "hdp_flush" -> Some (borrow_view (Option.get state.State.compute_queue.Queue_desc.hdp_flush))
         | Some tag ->
             let descriptor, suffix = if Filename.check_suffix tag "_compute" then
                 Some state.State.compute_queue, "_compute"
               else if Filename.check_suffix tag "_copy" then state.State.sdma_queue, "_copy"
               else None, "" in
             Option.bind descriptor (fun q ->
                 let field = String.sub tag 0 (String.length tag - String.length suffix) in
                 Option.map borrow_view (match field with
                   | "ring" -> Some q.Queue_desc.ring
                   | "read_ptr" -> Some q.Queue_desc.read_ptr
                   | "write_ptr" -> Some q.Queue_desc.write_ptr
                   | "doorbell" -> Some q.Queue_desc.doorbell
                   | _ -> None))
         | None -> None)
    | None -> None

  let create state =
    let host = try Device.get "CPU" with Failure _ -> Tolk_cpu.create "CPU" in
    let copy call = Option.is_some state.State.sdma_queue && match U.as_call call with
      | Some {args; _} -> List.for_all (fun arg ->
          match U.device_of arg with
          | Some (U.Single name) ->
              List.hd (String.split_on_char ':' name) = "CPU"
              || Device.peer_group (Device.get name) = Device.peer_group (Device.get state.State.name)
          | _ -> false) args
      | None -> false in
    let completion () =
      let timeline = state.State.tl in
      let value = Timeline.submitted timeline in
      fun () ->
        State.check_submission state;
        (match timeline.Timeline.error_state with Some exn -> raise exn | None -> ());
        Timeline.guarded_wait timeline (fun () ->
            Hcq.Signal.wait timeline.Timeline.timeline value) in
    Device.{timestamp_divider = 100.; completion; prepare = (fun () -> State.prepare state); host = Device.name host; copy;
      encode = Encoded_queue.encode state.State.hw ~props:state.State.iface.Iface.props
        ~name:state.State.name ~compute_ring_size:(Hcq.Mmio.size state.State.compute_queue.Queue_desc.ring)
        ~copy_ring_size:(Option.map (fun q -> Hcq.Mmio.size q.Queue_desc.ring) state.State.sdma_queue);
      lower = Encoded_queue.lower state.State.name;
      compile = Codegen.to_program ~optimize:false host (Device.renderer host)}
end

(* The shared device open path over the selected interface: everything
   from topology sizing to renderer wiring is interface-independent. *)
let open_device ~name iface =
  let props = iface.Iface.props in
  let ip = iface.Iface.ip_versions in
  let trgt = prop props "gfx_target_version" in
  let target = (trgt / 10000, trgt / 100 mod 100, trgt mod 100) in
  let tmaj, tmin, tstp = target in
  let arch = Printf.sprintf "gfx%d%x%x" tmaj tmin tstp in
  if not (List.mem target [ (9, 4, 2); (9, 5, 0) ] || tmaj = 11 || tmaj = 12)
  then failwith ("Unsupported arch: " ^ arch);
  let xccs =
    match List.assoc_opt "num_xcc" props with Some n -> n | None -> 1
  in
  let is_aql = Tolk.Helpers.getenv "AMD_AQL" (Bool.to_int (xccs > 1)) <> 0 in
  let se_cnt =
    prop props "array_count" / prop props "simd_arrays_per_engine" / xccs
  in
  let cu_cnt = prop props "simd_count" / prop props "simd_per_cu" / xccs in
  let waves_per_cu =
    prop props "max_waves_per_simd" * prop props "simd_per_cu"
  in
  let wave_cnt =
    if tmaj <> 9 then cu_cnt * waves_per_cu
    else min (cu_cnt * 40) (se_cnt * xccs * 512)
  in
  (* Compute-wave save/restore sizing: per-CU register, LDS and hardware
     state, plus the preemption control stack, each rounded to whole pages. *)
  let sgrp_size_per_cu = 0x4000 and hwreg_size_per_cu = 0x1000 in
  let lds_size_per_cu =
    if (tmaj, tmin) = (9, 5) then prop props "lds_size_in_kb" lsl 10
    else 0x10000
  in
  let vgpr_size_per_cu =
    if
      List.mem target
        [ (11, 0, 0); (11, 0, 1); (11, 5, 1); (12, 0, 0); (12, 0, 1) ]
    then 0x60000
    else if tmaj = 9 then 0x80000
    else 0x40000
  in
  let wg_data_size =
    round_up
      ((vgpr_size_per_cu + sgrp_size_per_cu + lds_size_per_cu
       + hwreg_size_per_cu)
      * cu_cnt)
      System.page_size
  in
  let ctl_stack_size =
    round_up
      (((if tmaj <> 9 then 12 else 8) * wave_cnt) + 8 + 40)
      System.page_size
  in
  let debug_memory_size = round_up (wave_cnt * 32) 64 in
  let create_queue queue_type ~ring_size ?(eop_buffer_size = 0)
      ?(ctx_save_restore_size = 0) ?(ctl_stack_size = 0) () =
    let ring =
      iface.Iface.alloc ~host:true ~uncached:true ~cpu_access:true ring_size
    in
    let gart = iface.Iface.alloc ~host:true ~uncached:true ~cpu_access:true 0x100 in
    let eop_buffer =
      if eop_buffer_size = 0 then None
      else Some (iface.Iface.alloc eop_buffer_size)
    in
    let cwsr_buffer =
      if ctx_save_restore_size = 0 then None
      else
        Some
          (iface.Iface.alloc
             (round_up
                ((ctx_save_restore_size + debug_memory_size) * xccs)
                System.page_size))
    in
    (* The queue's pointers live at the dispatch-id slots of an HSA queue
       descriptor laid out in the gart buffer. *)
    let aql = if queue_type = Compute_aql then
        let commands = iface.Iface.alloc ~cpu_access:true (16 lsl 20) in
        Some (Queue_desc.initialize_aql ~descriptor:(Hcq.Buffer.cpu_view gart) ~commands
          ~cu_count:(cu_cnt * xccs) ~waves_per_cu)
      else None in
    let queue = iface.Iface.create_queue queue_type ~ring ~gart
      ~rptr:Amd_hsa_defs.Amd_queue.read_dispatch_id
      ~wptr:Amd_hsa_defs.Amd_queue.write_dispatch_id ?eop_buffer ?cwsr_buffer
      ~ctx_save_restore_size ~ctl_stack_size () in
    {queue with Queue_desc.aql}
  in
  let compute_queue =
    (* driver-less devices carry no compute-wave save/restore state *)
    create_queue (if is_aql then Compute_aql else Compute) ~ring_size:(16 lsl 20) ~eop_buffer_size:0x1000
      ~ctx_save_restore_size:
        (if iface.Iface.is_am then 0 else wg_data_size + ctl_stack_size)
      ~ctl_stack_size:(if iface.Iface.is_am then 0 else ctl_stack_size) ()
  in
  let sdma_queue =
    if Tolk.Helpers.getenv "AMD_DISABLE_SDMA" 0 <> 0 then None
    else
      match create_queue Sdma ~ring_size:(16 lsl 20) () with
      | qd -> Some qd
      | exception Failure _ -> None
  in
  let hw =
    device ~target ~xccs ~is_aql ~gc_version:ip.gc ~nbio_version:ip.nbif
      ~sdma_version:ip.sdma ~tmpring_size:0
      ~scratch:iface.Iface.empty_scratch ~is_am:iface.Iface.is_am
      ~queue_event_mailbox_ptr:iface.Iface.queue_event_mailbox_ptr
      ~queue_event:iface.Iface.queue_event ()
  in
  let pool =
    Hcq.Signal.Pool.create ~alloc_page:(fun () ->
        iface.Iface.alloc ~host:true ~uncached:true ~cpu_access:true 0x1000)
  in
  let timeline_signal () =
    Hcq.Signal.make ~is_timeline:true ~timestamp_divider:100.
      ~sleep:iface.Iface.sleep ~owner:hw
      (Hcq.Signal.Pool.get pool)
  in
  let bounce_count = 32 and bounce_size = 2 lsl 20 in
  let state =
    {
      State.name = name;
      buffer_kind = iface.Iface.kind;
      iface;
      hw;
      compute_queue;
      sdma_queue;
      kernargs =
        Hcq.Kernargs.create (iface.Iface.alloc ~cpu_access:true (16 lsl 20));
      pool;
      tl =
        {
          Timeline.timeline = timeline_signal ();

          error_state = None;
          bounce =
            Array.init bounce_count (fun _ ->
                iface.Iface.alloc ~host:true bounce_size);
          bounce_timeline = Array.make bounce_count 0;
          bounce_next = 0;
          on_hang = iface.Iface.on_device_hang;
        };
      allocator = None;
      submission = Hcq.Submission.create ();
      scratch = None;
    }
  in
  (match iface.Iface.register with
  | Some register -> register ~compute_queue ~tl:state.State.tl
  | None -> ());
  (match iface.Iface.device_fini with
  | Some fini ->
      at_exit (fun () ->
          (* finalize even when the device faulted: the shutdown records
             the session's error flag for the next boot *)
          (try Timeline.synchronize state.State.tl
           with e ->
             Printf.eprintf
               "%s synchronization failed before finalizing: %s\n%!" name
               (Printexc.to_string e));
          fini ())
  | None -> ());
  let allocator = Allocator.create state in
  Runtime.ensure_scratch state 128;
  let renderer_set = Tolk.Device.Renderer_set.make ~device:name ~arch
      [ "HIP", (fun target ->
          let arch = match Tolk.Gpu_target.parse_amd_arch target.Tolk_uop.Target.arch with
            | Some arch -> arch
            | None -> invalid_arg ("unsupported AMD architecture: " ^ target.arch) in
          Tolk.Renderer.with_compiler (Compiler_amd.create ~arch:target.arch)
            (Tolk.Cstyle.amd arch)) ] in
  Tolk.Device.make ~name ~allocator ~renderer_set
    ~peer_group:(if iface.Iface.is_am then "PCIDevice" else "AMD")
    ~runtime:(Runtime.runtime state)
    ~synchronize:(fun () -> State.synchronize state)
    ~invalidate_caches:(fun () -> State.invalidate_caches state)
    ~queue:(Queue.create state) ~bufferize:(Queue.bufferize state) ()

let create name =
  let device_id =
    match String.index_opt name ':' with
    | Some i -> (
        let suffix = String.sub name (i + 1) (String.length name - i - 1) in
        match int_of_string_opt suffix with
        | Some id -> id
        | None -> invalid_arg (Printf.sprintf "invalid AMD device %S" name))
    | None -> 0
  in
  let kfd () =
    let iface = Kfd_iface.iface (Kfd_iface.create ~device_id) in
    fun () -> open_device ~name iface
  in
  let pci () =
    let iface = Pci_iface.iface (Pci_iface.create ~device_id) in
    fun () -> open_device ~name iface
  in
  (* Select the interface before opening the runtime: a later compiler or
     queue error must not retry a working kernel driver through PCI. *)
  let open_runtime = Tolk.Helpers.select_interface ~device:name
      [ "KFD", kfd; "PCI", pci ] in
  open_runtime ()
