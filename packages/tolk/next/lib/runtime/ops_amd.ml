(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops

type gpu = {
  target : int * int * int;
  sdma : int * int * int;
  xccs : int;
  shader_engines : int;
  compute_units : int;
  scratch_slots_per_cu : int;
  aql : bool;
  compute_ring : int;
  copy_rings : int list;
}

let major gpu =
  let m, _, _ = gpu.target in
  m

module G = Amd_gpu

(* A field of a word at its bits [(hi, lo)]. *)
let bits (hi, lo) v =
  (v
  land ((1 lsl ((hi - lo) [@mutate off "every value fits its field"] + 1)) - 1)
  )
  lsl lo

let packet3 op n =
  (G.packet_type3 lsl 30) lor ((op land 0xff) lsl 8) lor ((n land 0x3fff) lsl 16)

let u32 n = int ~dtype:Dtype.Uint32 (n land 0xffff_ffff)
let u64 n = int ~dtype:Dtype.Uint64 n
let binary s = v Op.Binary ~arg:(Bytes s)
let q_of = Hcq2.Queue.q

(* PM4 *)

let event_index_partial_flush = 4
let wait_reg_mem_function_eq = 3
let wait_reg_mem_function_geq = 5

let aql_hdr =
  (1 lsl G.hsa_packet_header_barrier)
  lor (G.hsa_fence_scope_system lsl G.hsa_packet_header_scacquire_fence_scope)
  lor (G.hsa_fence_scope_system lsl G.hsa_packet_header_screlease_fence_scope)

(* The ring and its pointers, tagged [name_queue] like the placeholders the
   engine binds to the device's queue. *)
let queue_args devs queue ring =
  let arg name shape dt =
    placeholder ~slot:0 ~device:(Multi devs) ~volatile:true
      ~tag:(Tag.String (Hcq2.to_name [ name; queue ]))
      shape dt
  in
  ( arg "ring" [ ring / 4 ] Dtype.Uint32,
    arg "write_ptr" [ 1 ] Dtype.Uint64,
    arg "doorbell" [ 1 ] Dtype.Uint64,
    arg "put_value" [ 1 ] Dtype.Uint64 )

let dw vals =
  List.fold_left
    (fun n w -> n + if Dtype.itemsize (dtype w) = 8 then 2 else 1)
    0 vals

(* A device's signal word, which only the batch's own work moves past the value
   its queues wait for (DIVERGENCES D37). *)
let is_signal_word s =
  match tag (fst (Hcq2.unwrap_view s)) with
  | Some (Tag.String "timeline") -> true
  | _ -> false

(* Program data *)

type program = {
  desc_offset : int;
  entry_point_offset : int;
  rsrc1 : int;
  rsrc2 : int;
  rsrc3 : int;
  wave32 : bool;
  private_segment_size : int;
  group_segment_size : int;
  kernargs_segment_size : int;
  enable_dispatch_ptr : bool;
  enable_private_segment_sgpr : bool;
  image_size : int;
}

let r_amdgpu_rel64 = 5

(* The kernel descriptor at the start of the code object's .rodata, with its
   relocations applied, as the device's loader applies them. *)
let program_data gpu lib =
  let o = Nx_device_elf.load lib in
  let image = Bytes.of_string o.image in
  List.iter
    (fun (r : Nx_device_elf.relocation) ->
      match (r.kind, r.target) with
      | k, Offset sym when k = r_amdgpu_rel64 ->
          Bytes.set_int64_le image r.at (Int64.of_int (sym - r.at + r.addend))
      | k, _ -> invalid_arg (Printf.sprintf "an unknown AMD relocation %d" k))
    o.relocations;
  let rodata =
    match
      List.find_opt
        (fun (s : Nx_device_elf.section) -> s.name = ".rodata")
        o.sections
    with
    | Some s -> s.offset
    | None -> invalid_arg "an AMD code object without .rodata"
  in
  let u32 off =
    Int32.to_int (Bytes.get_int32_le image (rodata + off)) land 0xffff_ffff
  in
  let props =
    Bytes.get_uint16_le image (rodata + G.kd_kernel_code_properties)
  in
  let group = u32 G.kd_group_segment_fixed_size in
  let lds = (group + 511) / 512 land 0x1ff in
  {
    desc_offset = rodata;
    entry_point_offset =
      rodata
      + Int64.to_int
          (Bytes.get_int64_le image
             (rodata + G.kd_kernel_code_entry_byte_offset));
    (* gfx11 runs kernels privileged, for their context save and restore. *)
    rsrc1 =
      (u32 G.kd_compute_pgm_rsrc1 lor if major gpu = 11 then 1 lsl 20 else 0);
    rsrc2 = u32 G.kd_compute_pgm_rsrc2 lor (lds lsl 15);
    rsrc3 = u32 G.kd_compute_pgm_rsrc3;
    wave32 = props land 0x400 <> 0;
    private_segment_size = u32 G.kd_private_segment_fixed_size;
    group_segment_size = group;
    kernargs_segment_size = u32 G.kd_kernarg_size;
    enable_dispatch_ptr =
      props land G.amd_kernel_code_properties_enable_sgpr_dispatch_ptr <> 0;
    enable_private_segment_sgpr =
      props land G.amd_kernel_code_properties_enable_sgpr_private_segment_buffer
      <> 0;
    image_size = Helpers.round_up (Bytes.length image) 4;
  }

(* The program's code object, which the engine loads on the devices (DIVERGENCES
   D38). *)
let amd_build_program gpu prg devs =
  let obj = Device.Tiny_elf.of_program prg in
  let data = program_data gpu obj.lib in
  ( data,
    placeholder ~slot:0 ~device:(Multi devs)
      ~tag:(Tag.Tuple [ String "program"; Bytes obj.lib; String obj.name ])
      [ data.image_size ] Dtype.Uint8 )

(* As words: the grid may be symbolic. *)
let dispatch_packet data (info : program_info) ?(kernel_object = u64 0)
    ?(kernarg_address = u64 0) () =
  let local =
    List.map
      (function Int l -> l | Sym _ -> invalid_arg "a symbolic local size")
      info.local_size
  in
  let pkt = Bytes.make G.dispatch_size '\000' in
  Bytes.set_uint16_le pkt G.dispatch_header
    (aql_hdr
    lor (G.hsa_packet_type_kernel_dispatch lsl G.hsa_packet_header_type));
  Bytes.set_uint16_le pkt G.dispatch_setup
    (3 lsl G.hsa_kernel_dispatch_packet_setup_dimensions);
  List.iteri
    (fun i l ->
      Bytes.set_uint16_le pkt (G.dispatch_workgroup_size_x + (2 * i)) l)
    local;
  Bytes.set_int32_le pkt G.dispatch_private_segment_size
    (Int32.of_int data.private_segment_size);
  Bytes.set_int32_le pkt G.dispatch_group_segment_size
    (Int32.of_int data.group_segment_size);
  let part a b = binary (Bytes.sub_string pkt a (b - a)) in
  let grid =
    List.map2
      (fun g l ->
        match g with
        | Int g -> u32 (g * l)
        | Sym g -> cast O.(g * int l) Dtype.Uint32)
      info.global_size local
  in
  (part 0 G.dispatch_grid_size_x :: grid)
  @ [
      part G.dispatch_private_segment_size G.dispatch_kernel_object;
      kernel_object;
      kernarg_address;
      part (G.dispatch_kernel_object + 16) G.dispatch_size;
    ]

(* The scratch memory's COMPUTE_TMPRING_SIZE for kernels of [n] bytes per lane:
   the waves it serves and the size of one. *)
let tmpring_size gpu n =
  let n = max n 128 and lanes_per_wave = 64 in
  let mem_alignment_size = if major gpu <> 9 then 256 else 1024 in
  let size_per_thread =
    Helpers.round_up n (mem_alignment_size / lanes_per_wave)
  in
  let size_per_xcc =
    size_per_thread * lanes_per_wave * gpu.scratch_slots_per_cu
    * gpu.compute_units
  in
  let max_scratch_waves =
    gpu.compute_units * gpu.scratch_slots_per_cu * gpu.xccs
  in
  let wave_scratch =
    Helpers.ceildiv (lanes_per_wave * size_per_thread) mem_alignment_size
  in
  let num_waves =
    size_per_xcc
    / (wave_scratch * mem_alignment_size)
    / if major gpu <> 9 then gpu.shader_engines else 1
  in
  let waves, wavesize =
    match major gpu with
    | 9 ->
        G.(compute_tmpring_size_waves_gfx9, compute_tmpring_size_wavesize_gfx9)
    | 11 ->
        G.(compute_tmpring_size_waves_gfx11, compute_tmpring_size_wavesize_gfx11)
    | _ ->
        G.(compute_tmpring_size_waves_gfx12, compute_tmpring_size_wavesize_gfx12)
  in
  bits waves (min num_waves max_scratch_waves) lor bits wavesize wave_scratch

(* An AQL queue's packets: dispatches and runs of PM4 packets, and loops around
   them, each with its range and the bytes of its trip in the command buffer. *)
type aql = Packets of Ops.t list | Trips of Ops.t * int * aql list

(* The compute queue, of PM4 packets or of AQL packets around them. *)
let compute_queue ~host gpu q : Hcq2.commands =
  let devs = Hcq2.Queue.devices q in
  let dev = List.hd devs in
  let queue = Hcq2.Queue.name q in
  let gfx9 = major gpu = 9 in
  let getaddr = getaddr ~device:dev in
  let pkt3 cmd vals =
    ignore (q_of q (u32 (packet3 cmd (dw vals - 1)) :: vals))
  in
  let wreg reg vals =
    let set, start =
      if
        (G.packet3_set_sh_reg_start <= reg
        && reg < G.packet3_set_sh_reg_end)
        [@mutate
          off
            "every register written is an SH register, far from the range's \
             ends"]
      then (G.packet3_set_sh_reg, G.packet3_set_sh_reg_start)
      else if
        G.packet3_set_uconfig_reg_start <= reg
        && reg < G.packet3_set_uconfig_reg_start + 0xffff
      then (G.packet3_set_uconfig_reg, G.packet3_set_uconfig_reg_start)
      else
        invalid_arg (Printf.sprintf "no PM4 packet sets the register 0x%x" reg)
    in
    pkt3 set (u32 (reg - start) :: vals)
  in
  (* The count fills in when the block closes. *)
  let pred_exec xcc_mask f =
    if gpu.xccs > 1 then pkt3 G.packet3_pred_exec [ u32 (xcc_mask lsl 24) ];
    let start = Hcq2.Queue.size q in
    f ();
    if gpu.xccs > 1 then
      Hcq2.Queue.set_dword q (start - 4)
        (Hcq2.Queue.get_dword q (start - 4)
        lor ((Hcq2.Queue.size q - start) / 4))
  in
  let wait_reg_mem ?(mask = 0xffff_ffff) ?mem ?(reg = 0) ?(reg_done = 0)
      ?(op = wait_reg_mem_function_geq) value =
    let info =
      (Bool.to_int (Option.is_some mem) lsl G.wait_reg_mem_mem_space)
      lor Bool.to_int
            ((Option.is_none mem && reg_done > 0)
             [@mutate
               off
                 "a wait on memory has no done register, and one on a register \
                  has one"])
          lsl G.wait_reg_mem_operation
      lor (op lsl G.wait_reg_mem_function)
      lor (0 lsl G.wait_reg_mem_engine)
    in
    let at =
      match mem with Some m -> [ m ] | None -> [ u32 reg; u32 reg_done ]
    in
    pkt3 G.packet3_wait_reg_mem ((u32 info :: at) @ [ value; u32 mask; u32 4 ])
  in
  let acquire_mem ?(gli = 1) ?(gl2 = 1) () =
    let everything = [ u32 0xffff_ffff; u32 0xffff_ffff; u32 0; u32 0 ] in
    if not gfx9 then
      let cache_flags =
        (gli lsl G.packet3_acquire_mem_gcr_cntl_gli_inv)
        lor (1 lsl G.packet3_acquire_mem_gcr_cntl_glm_inv)
        lor (1 lsl G.packet3_acquire_mem_gcr_cntl_glm_wb)
        lor (1 lsl G.packet3_acquire_mem_gcr_cntl_glk_inv)
        lor (1 lsl G.packet3_acquire_mem_gcr_cntl_glk_wb)
        lor (1 lsl G.packet3_acquire_mem_gcr_cntl_glv_inv)
        lor (1 lsl G.packet3_acquire_mem_gcr_cntl_gl1_inv)
        lor (gl2 lsl G.packet3_acquire_mem_gcr_cntl_gl2_inv)
        lor (gl2 lsl G.packet3_acquire_mem_gcr_cntl_gl2_wb)
      in
      pkt3 G.packet3_acquire_mem
        ((u32 0 :: everything) @ [ u32 0; u32 cache_flags ])
    else
      let cp_coher_cntl =
        (gli lsl G.packet3_acquire_mem_cp_coher_cntl_sh_icache_action_ena)
        lor (1 lsl G.packet3_acquire_mem_cp_coher_cntl_sh_kcache_action_ena)
        lor (gl2 lsl G.packet3_acquire_mem_cp_coher_cntl_tc_action_ena)
        lor (1 lsl G.packet3_acquire_mem_cp_coher_cntl_tcl1_action_ena)
        lor (gl2 lsl G.packet3_acquire_mem_cp_coher_cntl_tc_wb_action_ena)
      in
      pkt3 G.packet3_acquire_mem
        ((u32 cp_coher_cntl :: everything) @ [ u32 0x0000000A ])
  in
  let release_mem ~address ~value ~data_sel ~int_sel ?(cache_flush = false) () =
    let cache_flags =
      if not cache_flush then 0
      else if not gfx9 then
        G.packet3_release_mem_gcr_glv_inv lor G.packet3_release_mem_gcr_gl1_inv
        lor G.packet3_release_mem_gcr_gl2_inv
        lor G.packet3_release_mem_gcr_glm_wb
        lor G.packet3_release_mem_gcr_glm_inv
        lor G.packet3_release_mem_gcr_gl2_wb lor G.packet3_release_mem_gcr_seq
      else G.eop_tc_wb_action_en lor G.eop_tc_nc_action_en
    in
    let event_dw =
      (G.cache_flush_and_inv_ts_event lsl G.event_type)
      lor (G.event_index__mec_release_mem__end_of_pipe lsl G.event_index)
    in
    let memsel_dw =
      (* Its destination (DST_SEL) is 0, memory. *)
      (data_sel lsl G.data_sel) lor (int_sel lsl G.int_sel)
    in
    pkt3 G.packet3_release_mem
      [
        u32 (event_dw lor cache_flags);
        u32 memsel_dw;
        address;
        cast value Dtype.Uint64;
        u32 0;
      ]
  in
  let memory_barrier () =
    wait_reg_mem ~reg:G.bif_bx_pf_gpu_hdp_flush_req
      ~reg_done:G.bif_bx_pf_gpu_hdp_flush_done (u32 0xffff_ffff);
    acquire_mem ()
  in
  let kernargs call prg data =
    let info =
      match arg prg with
      | Program p -> p
      | _ -> invalid_arg "an AMD command runs a compiled program"
    in
    let bufs = Realize.get_call_arg_uops call in
    (* A bound value is a bare constant: the variable has the width. *)
    let args =
      List.map (fun g -> getaddr (List.nth bufs g)) info.globals
      @ List.map2
          (fun v b -> ccast b (dtype v))
          info.vars
          (Realize.get_call_var_uops call prg)
    in
    let words =
      Hcq2.pack_args (Hcq2.layout_args args) data.kernargs_segment_size
    in
    let packet =
      if data.enable_dispatch_ptr then dispatch_packet data info () else []
    in
    (info, v Op.Linear ~src:(words @ packet) ~arg:(String "kernargs"))
  in
  let wait signal value =
    let op =
      if is_signal_word signal then wait_reg_mem_function_eq
      else wait_reg_mem_function_geq
    in
    wait_reg_mem ~op ~mem:(getaddr signal) (cast value Dtype.Uint32)
  in
  let timestamp signal =
    pred_exec 1 (fun () ->
        release_mem
          ~address:O.(getaddr signal + u64 8)
          ~value:(u64 0)
          ~data_sel:G.data_sel__mec_release_mem__send_gpu_clock_counter
          ~int_sel:G.int_sel__mec_release_mem__none ())
  in
  (* A device's value is written whole, in one 64-bit write (D37). *)
  let signal_mem signal value =
    let data_sel =
      if is_signal_word signal then
        G.data_sel__mec_release_mem__send_64_bit_data
      else G.data_sel__mec_release_mem__send_32_bit_low
    in
    pred_exec 1 (fun () ->
        release_mem ~address:(getaddr signal) ~value ~data_sel
          ~int_sel:
            G.int_sel__mec_release_mem__send_interrupt_after_write_confirm
          ~cache_flush:true ())
  in
  let ring_queue = (queue, gpu.compute_ring) in
  let push cmdbuf words ?(unit = 4) ?(doorbell_lag = 0) () =
    let ring, wptr, doorbell, put =
      queue_args devs (fst ring_queue) (snd ring_queue)
    in
    (* put counts units, the ring dwords. *)
    let rs = snd ring_queue / 4 and n = max_numel words / 4 in
    let p = load (index put [ int 0 ]) [] in
    let per_unit = unit / 4 and units = max_numel words / unit in
    let tail = cast O.(p * int per_unit % int rs) Dtype.Int32 in
    let first = minimum O.(int rs - tail) (int n) in
    let copy cmdbuf rid dst src count =
      let i = range ~dtype:Dtype.Int32 ~src:[ cmdbuf ] (Sym count) [ rid ] in
      end_
        (store
           (index (after ring [ cmdbuf ]) [ O.(dst + i) ])
           (load (index (bitcast words Dtype.Uint32) [ O.(src + i) ]) []))
        [ i ]
    in
    let cmdbuf = copy cmdbuf 10 tail (int 0) first in
    let cmdbuf = copy cmdbuf 11 (int 0) first O.(int n - first) in
    let nxt = O.(p + int units) in
    let w = store (index (after wptr [ cmdbuf ]) [ int 0 ]) nxt in
    store
      (index
         (after doorbell [ store (index (after put [ w ]) [ int 0 ]) nxt ])
         [ int 0 ])
      O.(nxt - int doorbell_lag)
  in
  let program call prg =
    let data, lib = amd_build_program gpu prg devs in
    let info, ka = kernargs call prg data in
    (data, lib, info, ka)
  in
  let ib_blob cmdbuf =
    let b = Bytes.create 16 in
    List.iteri
      (fun i w -> Bytes.set_int32_le b (4 * i) (Int32.of_int w))
      [
        packet3 G.packet3_indirect_buffer 2;
        0;
        0;
        max_numel cmdbuf / 4 lor G.indirect_buffer_valid;
      ];
    Bytes.to_string b
  in
  if not gpu.aql then
    let exec call prg =
      let data, lib, info, ka = program call prg in
      let prog_addr = O.(getaddr lib + int data.entry_point_offset) in
      let scratch_addr =
        getaddr
          (rtag ~tag:(Tag.String "scratch")
             (placeholder ~slot:0 ~device:(Multi devs)
                [ data.private_segment_size ]
                Dtype.Uint8))
      in
      let args_addr = getaddr ka in
      let user_regs =
        (if data.enable_private_segment_sgpr then
           [
             O.(scratch_addr lor const (`Int (Z.shift_left Z.one 63)));
             u32 0xffff_ffff;
             u32 0x20c14000;
           ]
         else [])
        @ (if data.enable_dispatch_ptr then
             [ O.(args_addr + int data.kernargs_segment_size) ]
           else [])
        @ [ args_addr ]
      in
      let dispatch_init =
        (if gfx9 then 0
         else
           bits G.compute_dispatch_initiator_cs_w32_en (Bool.to_int data.wave32))
        lor bits G.compute_dispatch_initiator_force_start_at_000 1
        lor bits G.compute_dispatch_initiator_compute_shader_en 1
      in
      acquire_mem ~gli:0 ~gl2:0 ();
      wreg G.compute_pgm_lo [ O.(prog_addr lsr int 8) ];
      wreg G.compute_pgm_rsrc1 [ u32 data.rsrc1; u32 data.rsrc2 ];
      wreg
        (if gfx9 then G.compute_pgm_rsrc3_gfx9 else G.compute_pgm_rsrc3)
        [ u32 data.rsrc3 ];
      wreg G.compute_tmpring_size
        [ u32 (tmpring_size gpu data.private_segment_size) ];
      (* Architected flat scratch: each die gets its part. *)
      for xcc = 0 to gpu.xccs - 1 do
        let part = data.private_segment_size / gpu.xccs * xcc in
        pred_exec (1 lsl xcc) (fun () ->
            wreg G.compute_dispatch_scratch_base_lo
              [ O.((scratch_addr + int part) lsr int 8) ])
      done;
      wreg G.compute_restart_x [ u32 0; u32 0; u32 0 ];
      wreg G.compute_user_data_0 user_regs;
      wreg G.compute_resource_limits [ u32 (Helpers.getenv "WAVES_PER_SH" 0) ];
      wreg G.compute_start_x
        ([ u32 0; u32 0; u32 0 ]
        @ List.map (function Int l -> u32 l | Sym l -> l) info.local_size
        @ [ u32 0; u32 0 ]);
      pkt3 G.packet3_dispatch_direct
        (List.map (function Int g -> u32 g | Sym g -> g) info.global_size
        @ [ u32 dispatch_init ]);
      pkt3 G.packet3_event_write
        [
          u32
            ((G.cs_partial_flush lsl G.event_type)
            lor (event_index_partial_flush lsl G.event_index));
        ]
    in
    (* The ring gets an indirect buffer packet: 4 dwords, and put stays aligned
       so that it never wraps mid packet. *)
    let submit cmdbuf =
      let base, off = Hcq2.unwrap_view cmdbuf in
      let ib =
        placeholder ~device:(Single host)
          ~tag:(Tag.String (Hcq2.to_name [ "ib"; queue ]))
          [ 16 ] Dtype.Uint8
      in
      push cmdbuf
        (Hcq2.patch ib
           [ (int 4, O.(getaddr base + int off)) ]
           ~blob:(ib_blob cmdbuf))
        ()
    in
    {
      Hcq2.exec;
      copy = (fun _ _ _ -> invalid_arg "an AMD compute queue does not copy");
      wait;
      signal = signal_mem;
      timestamp;
      memory_barrier;
      loop = Hcq2.Queue.loop q;
      submit;
    }
  else
    (* The ring holds 64-byte AQL packets: a dispatch for each kernel, and the
       PM4 packets between them wrapped as indirect buffers. The packets point
       into the command buffer, whose address binds at submit. *)
    let cmd_addr =
      variable ~dtype:Dtype.Uint64 "cmdbuf" (`Int Z.zero)
        (`Int (Z.shift_left Z.one 48))
    in
    let items = ref [] and run_start = ref 0 in
    let add ws = items := !items @ [ Packets ws ] in
    let close_run end_ =
      if end_ > !run_start then begin
        let hdr =
          aql_hdr
          lor (G.hsa_packet_type_vendor_specific lsl G.hsa_packet_header_type)
          lor (1 lsl 16)
        in
        let ib =
          [
            u32 (packet3 G.packet3_indirect_buffer 2);
            O.(cmd_addr + int !run_start);
            u32 ((end_ - !run_start) / 4 lor G.indirect_buffer_valid);
          ]
        in
        add ((u32 hdr :: ib) @ (u32 10 :: List.init 10 (fun _ -> u32 0)))
      end;
      run_start := end_
    in
    (* Close the PM4 run, so that the signal waits for the dispatches before it:
       the dies could otherwise race. *)
    let signal s value =
      close_run (Hcq2.Queue.size q);
      signal_mem s value
    in
    let exec call prg =
      let data, lib, info, ka = program call prg in
      close_run (Hcq2.Queue.size q);
      add
        (dispatch_packet data info
           ~kernel_object:O.(getaddr lib + int data.desc_offset)
           ~kernarg_address:(getaddr ka) ());
      run_start := Hcq2.Queue.size q
    in
    (* A loop's packets are once per trip, each pointing into its trip's bytes
       of the command buffer, which the queue repeats. *)
    let loop r body =
      close_run (Hcq2.Queue.size q);
      let outer = !items and start = Hcq2.Queue.size q and trip = ref 0 in
      items := [];
      Hcq2.Queue.loop q r (fun () ->
          body ();
          close_run (Hcq2.Queue.size q);
          trip := Hcq2.Queue.size q - start);
      items := outer @ [ Trips (r, !trip, !items) ];
      run_start := Hcq2.Queue.size q
    in
    (* The doorbell is the last packet's index. *)
    let submit cmdbuf =
      close_run (max_numel cmdbuf);
      let base, off = Hcq2.unwrap_view cmdbuf in
      Hcq2.Queue.reset q;
      let rec emit addr = function
        | Packets ws ->
            ignore (q_of q (src (substitute (sink ws) [ (cmd_addr, addr) ])))
        | Trips (r, trip, items) ->
            Hcq2.Queue.loop q r (fun () ->
                List.iter
                  (emit O.(addr + (cast r Dtype.Uint64 * int trip)))
                  items)
      in
      List.iter (emit O.(getaddr base + int off)) !items;
      let size = Hcq2.Queue.size q in
      (* A submission writes at most half the ring, which the device leaves room
         for (DIVERGENCES D39). *)
      if 2 * size > gpu.compute_ring then
        invalid_arg
          (Printf.sprintf
             "AQL packets of %d bytes exceed half their ring of %d bytes" size
             gpu.compute_ring);
      push cmdbuf
        (Hcq2.bufferize_cmdbuf q "aql" host)
        ~unit:64 ~doorbell_lag:1 ()
    in
    {
      Hcq2.exec;
      copy = (fun _ _ _ -> invalid_arg "an AMD compute queue does not copy");
      wait;
      signal;
      timestamp;
      memory_barrier;
      loop;
      submit;
    }

(* SDMA *)

let copy_queue ~host gpu q : Hcq2.commands =
  let devs = Hcq2.Queue.devices q in
  let getaddr = getaddr ~device:(List.hd devs) in
  let queue = Hcq2.Queue.name q in
  let ring =
    match String.split_on_char ':' queue with
    | [ _; i ] -> (
        match List.nth_opt gpu.copy_rings (int_of_string i) with
        | Some ring -> ring
        | None ->
            invalid_arg
              (Printf.sprintf "the AMD GPU has no copy queue %s" queue))
    | _ -> invalid_arg (Printf.sprintf "%s is no AMD copy queue" queue)
  in
  let loop = Hcq2.Queue.loop q in
  let q words = ignore (q_of q words) in
  let sdma_major, _, _ = gpu.sdma in
  let max_copy_size =
    if (gpu.sdma >= (4, 4, 2) && sdma_major < 5) || gpu.sdma >= (5, 2, 0) then
      0x40000000
    else 0x400000
  in
  let copy dst src sz =
    let addr a off = if off = 0 then a else O.(a + u64 off) in
    for k = 0 to Helpers.ceildiv sz max_copy_size - 1 do
      let off = k * max_copy_size in
      q
        [
          u32
            (G.sdma_op_copy
            lor bits G.sdma_pkt_copy_linear_header_sub_op
                  G.sdma_subop_copy_linear);
          u32 (min (sz - off) max_copy_size - 1);
          u32 0;
          addr (getaddr src) off;
          addr (getaddr dst) off;
        ]
    done
  in
  let wait signal value =
    let func =
      if is_signal_word signal then wait_reg_mem_function_eq
      else wait_reg_mem_function_geq
    in
    q
      [
        u32
          (G.sdma_op_poll_regmem
          lor bits G.sdma_pkt_poll_regmem_header_func func
          lor bits G.sdma_pkt_poll_regmem_header_mem_poll 1);
        getaddr signal;
        cast value Dtype.Uint32;
        u32 ((1 lsl (8 * min (Dtype.itemsize (dtype value)) 4)) - 1);
        u32
          (bits G.sdma_pkt_poll_regmem_dw5_interval 0x04
          lor bits G.sdma_pkt_poll_regmem_dw5_retry_count 0xfff);
      ]
  in
  let timestamp signal =
    q
      [
        u32
          (G.sdma_op_timestamp
          lor bits G.sdma_pkt_timestamp_get_header_sub_op
                G.sdma_subop_timestamp_get_global);
        O.(getaddr signal + u64 8);
      ]
  in
  (* A device's value is written 32 bits at a time: its high half only when its
     low half is 0, the high half every later value shares; four NOPs otherwise
     (D37). *)
  let signal signal value =
    let fence =
      G.sdma_op_fence
      lor if major gpu <> 9 then bits G.sdma_pkt_fence_header_mtype 3 else 0
    in
    let high =
      if not (is_signal_word signal) then []
      else
        let carry = eq (cast value Dtype.Uint32) (u32 0) in
        List.map
          (fun w -> where carry w (const_like w (`Int Z.zero)))
          [
            u32 fence;
            O.(getaddr signal + u64 4);
            cast O.(value lsr u64 32) Dtype.Uint32;
          ]
    in
    q
      ([ u32 fence; getaddr signal; cast value Dtype.Uint32 ]
      @ high
      @ [ u32 G.sdma_op_trap; u32 0 ])
  in
  (* SDMA needs the command buffer whole in the ring: if it does not fit before
     the ring's end, it restarts at 0 and zeroes the tail. *)
  let submit cmdbuf =
    let ring_p, wptr, doorbell, put = queue_args devs queue ring in
    (* In host memory: streamed into the ring, the device never reads it. *)
    let base = fst (Hcq2.unwrap_view cmdbuf) in
    let on_host =
      match arg base with
      | Param p ->
          replace ~arg:(Param { p with device = Some (Single host) }) base
      | _ -> invalid_arg "an AMD command buffer is a placeholder"
    in
    let cmdbuf = substitute cmdbuf [ (base, on_host) ] in
    let rs = ring / 4 and size_dw = max_numel cmdbuf / 4 in
    (* Zeroing the tail can double what a submission writes, which is at most
       half the ring (DIVERGENCES D39). *)
    if 4 * (size_dw * 4) > ring then
      invalid_arg
        (Printf.sprintf
           "an SDMA command buffer of %d bytes exceeds a quarter of its ring \
            of %d bytes"
           (size_dw * 4) ring);
    let put_b = load (index put [ int 0 ]) [] in
    let ring_bytes = rs * 4 in
    let tail = cast O.(put_b % int ring_bytes // int 4) Dtype.Int32 in
    let fits = cast O.(int rs - tail >= int size_dw) Dtype.Int32 in
    let start_dw = O.(fits * tail)
    and zero_amt = O.((int 1 - fits) * (int rs - tail)) in
    let zi = range ~dtype:Dtype.Int32 ~src:[ cmdbuf ] (Sym zero_amt) [ 10 ] in
    let zero_tail =
      end_ (store (index ring_p [ O.(tail + zi) ]) (u32 0)) [ zi ]
    in
    let i = range ~dtype:Dtype.Int32 ~src:[ cmdbuf ] (Int size_dw) [ 11 ] in
    let copy =
      end_
        (store
           (index (after ring_p [ zero_tail ]) [ O.(start_dw + i) ])
           (load (index (bitcast cmdbuf Dtype.Uint32) [ i ]) []))
        [ i ]
    in
    let next_put =
      O.(put_b + cast ((zero_amt + int size_dw) * int 4) (dtype put_b))
    in
    let w = store (index (after wptr [ copy ]) [ int 0 ]) next_put in
    store
      (index
         (after doorbell [ store (index (after put [ w ]) [ int 0 ]) next_put ])
         [ int 0 ])
      next_put
  in
  {
    Hcq2.exec = (fun _ _ -> invalid_arg "an AMD copy queue runs no program");
    copy;
    wait;
    signal;
    timestamp;
    (* A copy queue has nothing to flush. *)
    memory_barrier = (fun () -> ());
    loop;
    submit;
  }

let queues ~host ~reaches gpu =
  (match gpu.target with
  | 9, 4, 2 | 9, 5, 0 -> ()
  | m, _, _ when m = 11 || m = 12 -> ()
  | m, n, s ->
      invalid_arg (Printf.sprintf "gfx%d%x%x is not a supported AMD GPU" m n s));
  if gpu.xccs > 1 && not gpu.aql then
    invalid_arg "an AMD GPU of several dies takes AQL packets";
  let commands q =
    if String.starts_with ~prefix:"COMPUTE" (Hcq2.Queue.name q) then
      compute_queue ~host gpu q
    else copy_queue ~host gpu q
  in
  { Hcq2.commands; copy_queue = gpu.copy_rings <> []; host; reaches }

(* What the engine links *)

type storage =
  | Ring of string
  | Write_ptr of string
  | Put of string
  | Doorbell of string
  | Program of { binary : string; name : string }
  | Scratch of int

let storage u =
  match tag u with
  | Some (Tag.Tuple [ String "program"; Bytes binary; String name ]) ->
      Some (Program { binary; name })
  | Some (Tag.String "scratch") -> Some (Scratch (max_numel u))
  | Some (Tag.String t) -> (
      (* [name_queue_index], as queue_args tags them. *)
      match List.rev (String.split_on_char '_' t) with
      | i :: queue :: name -> (
          let queue = String.uppercase_ascii queue ^ ":" ^ i in
          match String.concat "_" (List.rev name) with
          | "ring" -> Some (Ring queue)
          | "write_ptr" -> Some (Write_ptr queue)
          | "put_value" -> Some (Put queue)
          | "doorbell" -> Some (Doorbell queue)
          | _ -> None)
      | _ -> None)
  | _ -> None
