(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Defs
module Window = Rig_pci.Window
module Register = Rig_amd_abi.Register
module Pm4 = Rig_amd_abi.Pm4
module Packet = Rig_amd_abi.Packet

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let lo32 v = v land 0xffff_ffff
let hi32 v = (v lsr 32) land 0xffff_ffff
let rec log2 n = if n <= 1 then 0 else 1 + log2 (n lsr 1)

(* Queue descriptors *)

type queue = {
  ring : int;
  ring_bytes : int;
  read : int;
  write : int;
  eop : int;
  eop_bytes : int;
  doorbell : int;
}

let layout_of_gc = function
  | 9, _, _ -> D.mqd_v9
  | 11, _, _ -> D.mqd_v11
  | 12, _, _ -> D.mqd_v12
  | a, b, c ->
      invalid_argf "Rig_amd_pci.open_: GC %d.%d.%d has no queue descriptor" a b
        c

(* The values the kernel's MQD initialisation gives a compute queue
   (gfx_v11_0_compute_mqd_init and its siblings). *)
let mqd_header = 0xC031_0800
let pipe_priority = 2
let queue_priority = 0xf
let quantum = 0x111
let hq_status0 = 0x2000_4000
let all_units = 0xffff_ffff

(* The stride between the descriptors of a queue's dies. *)
let die_stride = 0x1000

let mqd l q ~base ~kiq ~aql ~xcc ~xccs =
  let gc = (Regs.gpu l).gc in
  let (f : D.mqd) = layout_of_gc gc in
  let b = Bytes.make f.sizeof '\000' in
  let set (off, _) v = Bytes.set_int32_le b off (Int32.of_int (lo32 v)) in
  let enc name kvs = Register.encode (Regs.register l name) kvs in
  set f.header mqd_header;
  set f.cp_mqd_base_addr_lo (lo32 base);
  set f.cp_mqd_base_addr_hi (hi32 base);
  set f.cp_hqd_pipe_priority pipe_priority;
  set f.cp_hqd_queue_priority queue_priority;
  set f.cp_hqd_quantum quantum;
  set f.cp_hqd_persistent_state
    (enc "regCP_HQD_PERSISTENT_STATE"
       [ ("preload_size", 0x55); ("preload_req", 1) ]);
  set f.cp_hqd_pq_base_lo (lo32 (q.ring lsr 8));
  set f.cp_hqd_pq_base_hi (hi32 (q.ring lsr 8));
  set f.cp_hqd_pq_rptr_report_addr_lo (lo32 q.read);
  set f.cp_hqd_pq_rptr_report_addr_hi (hi32 q.read);
  set f.cp_hqd_pq_wptr_poll_addr_lo (lo32 q.write);
  set f.cp_hqd_pq_wptr_poll_addr_hi (hi32 q.write);
  set f.cp_hqd_pq_doorbell_control
    (enc "regCP_HQD_PQ_DOORBELL_CONTROL"
       [ ("doorbell_offset", q.doorbell * 2); ("doorbell_en", 1) ]);
  set f.cp_hqd_pq_control
    (enc "regCP_HQD_PQ_CONTROL"
       ([
          ("rptr_block_size", 5);
          ("unord_dispatch", 0);
          ("queue_size", log2 (q.ring_bytes / 4) - 1);
        ]
       @ (if kiq then [ ("priv_state", 1); ("kmd_queue", 1) ] else [])
       @
       if aql then
         [
           ("queue_full_en", 1);
           ("slot_based_wptr", 2);
           ("no_update_rptr", Bool.to_int (xcc <> 0 || xccs = 1));
         ]
       else []));
  set f.cp_hqd_ib_control
    (enc "regCP_HQD_IB_CONTROL" [ ("min_ib_avail_size", 3) ]);
  set f.cp_hqd_hq_status0 hq_status0;
  set f.cp_mqd_control (enc "regCP_MQD_CONTROL" [ ("priv_state", 1) ]);
  set f.cp_hqd_vmid 0;
  set f.cp_hqd_aql_control (Bool.to_int aql);
  set f.cp_hqd_eop_base_addr_lo (lo32 (q.eop lsr 8));
  set f.cp_hqd_eop_base_addr_hi (hi32 (q.eop lsr 8));
  set f.cp_hqd_eop_control
    (enc "regCP_HQD_EOP_CONTROL" [ ("eop_size", log2 (q.eop_bytes / 4) - 1) ]);
  (* The thread management words first: on GFX9 four of them share their place
     with the fields of a queue across dies, set after them, as the kernel's
     kfd_mqd_manager_v9 does. *)
  List.iter (fun w -> set w all_units) f.compute_static_thread_mgmt;
  if aql && xccs > 1 then begin
    Option.iter (fun w -> set w 1) f.compute_tg_chunk_size;
    Option.iter (fun w -> set w xcc) f.compute_current_logic_xcc_id;
    Option.iter (fun w -> set w die_stride) f.cp_mqd_stride_size
  end;
  Bytes.to_string b

(* The GC *)

type t = {
  r : Regs.t;
  gmc : Gmc.t;
  vram : Window.t;
  doorbells : Window.t;
  mutable starts : (string * int) list; (* the RS64 engines' start addresses *)
  mqds : int array; (* the compute queue's descriptors, the KIQ's *)
  gc : Discovery.version;
  xccs : int list;
  mutable kiq : (Window.t * int) option; (* a VF's KIQ memory and its address *)
}

let make r gmc vram doorbells ~mqds =
  let l = Regs.layout_of r in
  {
    r;
    gmc;
    vram;
    doorbells;
    starts = [];
    mqds;
    gc = Regs.version l D.gc_hwid;
    xccs = List.init (Regs.gpu l).xccs Fun.id;
    kiq = None;
  }

let grbm_select ?(me = 0) ?(pipe = 0) ?(queue = 0) ?(vmid = 0) g ~inst =
  Regs.write ~inst g.r "regGRBM_GFX_CNTL"
    [ ("meid", me); ("pipeid", pipe); ("vmid", vmid); ("queueid", queue) ]

let each_xcc g f = List.iter (fun inst -> f inst) g.xccs

(* [f ()] with the RLC in safe mode on die [inst]: the GC's clocks held ungated
   while registers that gating would stall are written. A virtual function's
   host owns the RLC: [f ()] alone. *)
let safe_mode g ~inst f =
  let r = g.r in
  let mode message =
    Regs.write ~inst r "regRLC_SAFE_MODE" [ ("message", message); ("cmd", 1) ]
  in
  if Regs.vf r then f ()
  else begin
    mode 1;
    Regs.wait r "the RLC's safe mode" (fun () ->
        Regs.read ~inst r "regRLC_SAFE_MODE" land 1 = 0);
    match f () with
    | v ->
        mode 0;
        v
    | exception e ->
        let bt = Printexc.get_raw_backtrace () in
        mode 0;
        Printexc.raise_with_backtrace e bt
  end

(* Micro-engines *)

(* The MEC runs after its enable for 50 ms with no state to poll. *)
let mec_ms = 50

let enable_mec g =
  each_xcc g (fun inst ->
      if g.gc >= (10, 0, 0) then
        Regs.update ~inst g.r "regCP_MEC_RS64_CNTL"
          [ ("mec_pipe0_reset", 0); ("mec_pipe0_active", 1); ("mec_halt", 0) ]
      else Regs.write ~inst ~value:0 g.r "regCP_MEC_CNTL" []);
  Regs.pause g.r mec_ms

let halt g =
  each_xcc g (fun inst ->
      if g.gc >= (10, 0, 0) then
        Regs.update ~inst g.r "regCP_MEC_RS64_CNTL" [ ("mec_halt", 1) ]
      else
        Regs.update ~inst g.r "regCP_MEC_CNTL"
          [ ("mec_me1_halt", 1); ("mec_me2_halt", 1) ])

(* RS64 engines start at the address their image gives, in 32-bit words. *)
let start_engine g ~engine ~cntl ~me ~inst =
  match List.assoc_opt engine g.starts with
  | None ->
      raise (Regs.Stuck (strf "the %s's image has no start address" engine))
  | Some start ->
      grbm_select g ~me ~inst;
      Regs.write64 ~inst g.r
        (strf "regCP_%s_PRGRM_CNTR_START" cntl)
        ~lo:"" ~hi:"_HI" (start lsr 2);
      grbm_select g ~inst;
      let reg =
        if engine = "MEC" then "regCP_MEC_RS64_CNTL" else "regCP_ME_CNTL"
      in
      let field = String.lowercase_ascii engine ^ "_pipe0_reset" in
      Regs.update ~inst g.r reg [ (field, 1) ];
      Regs.update ~inst g.r reg [ (field, 0) ]

let config_mec g =
  each_xcc g (fun inst ->
      if g.gc < (10, 0, 0) then
        Regs.update ~inst g.r "regCP_MEC_CNTL"
          [
            ("mec_invalidate_icache", 1);
            ("mec_me1_pipe0_reset", 1);
            ("mec_me2_pipe0_reset", 1);
            ("mec_me1_halt", 1);
            ("mec_me2_halt", 1);
          ];
      if g.gc >= (12, 0, 0) then begin
        start_engine g ~engine:"PFP" ~cntl:"PFP" ~me:0 ~inst;
        start_engine g ~engine:"ME" ~cntl:"ME" ~me:0 ~inst
      end;
      if g.gc >= (10, 0, 0) then
        start_engine g ~engine:"MEC" ~cntl:"MEC_RS64" ~me:1 ~inst)

(* The queues this library programs: compute queue 0 of MEC 1, a second the last
   boot may have left, and a virtual function's KIQ. *)
let kiq_queue = (2, 1, 0)

let dequeue g ~wait =
  let queues =
    [ (1, 0, 0); (1, 0, 1) ] @ if Regs.vf g.r then [ kiq_queue ] else []
  in
  let left = ref true in
  (* Under the RLC's safe mode, which holds the GC's clocks ungated, as the
     kernel's gfx_v12_0_reset_kcq resets a queue. *)
  each_xcc g (fun inst ->
      safe_mode g ~inst @@ fun () ->
      List.iter
        (fun (me, pipe, queue) ->
          grbm_select g ~me ~pipe ~queue ~inst;
          if Regs.read ~inst g.r "regCP_HQD_ACTIVE" land 1 = 1 then begin
            Regs.write ~inst ~value:2 g.r "regCP_HQD_DEQUEUE_REQUEST" [];
            Regs.write ~inst ~value:1 g.r "regSPI_COMPUTE_QUEUE_RESET" [];
            if wait then
              (* A wave the reset cannot stop holds the queue. *)
              try
                Regs.wait g.r "a compute queue's dequeue" (fun () ->
                    Regs.read ~inst g.r "regCP_HQD_ACTIVE" land 1 = 0)
              with Regs.Stuck _ -> left := false
          end)
        queues;
      grbm_select g ~inst);
  !left

let reset_mec g =
  ignore (dequeue g ~wait:true);
  if g.gc < (12, 0, 0) then begin
    each_xcc g (fun inst ->
        Regs.write ~inst g.r "regGRBM_SOFT_RESET"
          [ ("soft_reset_cp", 1); ("soft_reset_cpc", 1) ]);
    Regs.pause g.r mec_ms;
    each_xcc g (fun inst ->
        Regs.write ~inst ~value:0 g.r "regGRBM_SOFT_RESET" [])
  end;
  config_mec g;
  enable_mec g

(* Queues *)

let eop_bytes = 0x1000

(* Programs queue [q] of MEC [me], pipe [pipe], queue [queue] on each of
   [insts], from the descriptor at [mqd] (one page per die): dequeues the
   hardware queue if it is active, as the kernel's kiq_init_register does, so
   that no running queue has its registers rewritten under it, writes the
   descriptor, copies its registers into the hardware queue and activates it. *)
let program g ~me ~pipe ~queue ~insts ~mqd:at ~kiq ~aql q =
  let l = Regs.layout_of g.r in
  let xccs = List.length g.xccs in
  List.iter
    (fun xcc ->
      grbm_select g ~me ~pipe ~queue ~inst:xcc;
      let active () =
        Regs.read ~inst:xcc g.r "regCP_HQD_ACTIVE" land 1 = 1
      in
      if active () then begin
        Regs.write ~inst:xcc ~value:1 g.r "regCP_HQD_DEQUEUE_REQUEST" [];
        Regs.wait g.r "an active queue's dequeue before its descriptor"
          (fun () -> not (active ()));
        Regs.write ~inst:xcc ~value:0 g.r "regCP_HQD_DEQUEUE_REQUEST" []
      end;
      let pa = at + (die_stride * xcc) in
      let d = mqd l q ~base:(Gmc.mc g.gmc pa) ~kiq ~aql ~xcc ~xccs in
      Window.write g.vram pa d;
      Window.flush g.vram;
      (* The hardware queue's registers, CP_MQD_BASE_ADDR to CP_HQD_PQ_WPTR_HI,
         mirror the descriptor from its word 0x80. *)
      let first = Regs.address ~inst:xcc l "regCP_MQD_BASE_ADDR" in
      let last = Regs.address ~inst:xcc l "regCP_HQD_PQ_WPTR_HI" in
      for i = 0 to last - first do
        Regs.set g.r (first + i)
          (Int32.to_int (String.get_int32_le d (4 * (0x80 + i)))
          land 0xffff_ffff)
      done;
      Gmc.flush_hdp g.gmc;
      Regs.write ~inst:xcc ~value:1 g.r "regCP_HQD_ACTIVE" [];
      (* The CP takes the queue's doorbell once told to, as the kernel tells
         it after activating a queue it programs itself
         (gfx_v12_0_kiq_init_register and its siblings). *)
      Regs.update ~inst:xcc g.r "regCP_PQ_STATUS" [ ("doorbell_enable", 1) ];
      grbm_select g ~inst:xcc)
    insts

let queue g kind ~ring ~bytes ~read ~write ~eop =
  let aql = kind = `Aql in
  let doorbell = D.amdgpu_navi10_doorbell_mec_ring0 in
  let insts = if aql then g.xccs else [ 0 ] in
  program g ~me:1 ~pipe:0 ~queue:0 ~insts ~mqd:g.mqds.(0) ~kiq:false ~aql
    { ring; ring_bytes = bytes; read; write; eop; eop_bytes; doorbell };
  doorbell

(* A virtual function's KIQ, per die: a ring of 4 KiB, its pointers, its fence,
   and an end-of-pipe buffer. *)
let kiq_bytes = 0x3000
let kiq_ring = 0x1000
let kiq_doorbell xcc = D.amdgpu_navi10_doorbell_kiq + (xcc * 0x20)

let start_kiq g m =
  match
    Rig_pci.Memory.alloc m Rig_pci.Memory.Host (kiq_bytes * List.length g.xccs)
  with
  | Error why -> raise (Regs.Stuck (strf "the KIQ's memory: %s" why))
  | Ok None -> raise (Regs.Stuck "no GPU addresses for the KIQ")
  | Ok (Some region) ->
      let va = region.mapping.va in
      let w = Option.get region.host in
      each_xcc g (fun xcc ->
          let b = va + (kiq_bytes * xcc) in
          program g ~me:2 ~pipe:1 ~queue:0 ~insts:[ xcc ] ~mqd:g.mqds.(1)
            ~kiq:true ~aql:false
            {
              ring = b;
              ring_bytes = kiq_ring;
              read = b + 0x1000;
              write = b + 0x1008;
              eop = b + 0x2000;
              eop_bytes;
              doorbell = kiq_doorbell xcc;
            });
      each_xcc g (fun inst ->
          Regs.update ~inst g.r "regRLC_CP_SCHEDULERS"
            [ ("scheduler0", (2 lsl 5) lor (1 lsl 3) lor 0x80) ]);
      g.kiq <- Some (w, va)

let wait_autoload g =
  let r = g.r in
  if Regs.has (Regs.layout_of r) "regRLC_RLCS_BOOTLOAD_STATUS" then
    Regs.wait r "the RLC's autoload" (fun () ->
        Regs.read r "regCP_STAT" = 0
        && Regs.field r "regRLC_RLCS_BOOTLOAD_STATUS" "bootload_complete" = 1)

let start g m images ~partial =
  g.starts <- images.Images.starts;
  let r = g.r in
  if partial then reset_mec g
  else begin
    config_mec g;
    each_xcc g (fun inst ->
        Regs.write ~inst
          ~value:(Regs.read ~inst r "regTCP_CNTL" lor 0x2000_0000)
          r "regTCP_CNTL" [];
        Regs.write ~inst ~value:1 r "regRLC_CNTL" [];
        Regs.update ~inst r "regRLC_SRM_CNTL"
          [ ("srm_enable", 1); ("auto_incr_addr", 1) ];
        Regs.write ~inst ~value:0xf r "regRLC_SPM_MC_CNTL" []);
    let major, minor, _ = g.gc in
    let address_mode, alignment_mode =
      match major with
      | 9 ->
          ( D.soc15_sh_mem_address_mode_64,
            D.soc15_sh_mem_alignment_mode_unaligned )
      | 11 ->
          ( D.soc21_sh_mem_address_mode_64,
            D.soc21_sh_mem_alignment_mode_unaligned )
      | _ ->
          ( D.soc24_sh_mem_address_mode_64,
            D.soc24_sh_mem_alignment_mode_unaligned )
    in
    each_xcc g (fun inst ->
        if List.mem g.gc [ (9, 4, 3); (9, 5, 0) ] then begin
          Regs.write ~inst ~value:0x2a11_4042 r "regGB_ADDR_CONFIG" [];
          Regs.update ~inst r "regTCP_UTCL1_CNTL2" [ ("spare", 1) ]
        end;
        Regs.update ~inst r "regGRBM_CNTL" [ ("read_timeout", 0xff) ];
        for vmid = 0 to 15 do
          grbm_select g ~vmid ~inst;
          Regs.write ~inst r "regSH_MEM_CONFIG"
            ((if major >= 10 then [ ("initial_inst_prefetch", 3) ]
              else [ ("retry_disable", 1) ])
            @ (if (major, minor) = (9, 4) then [ ("f8_mode", 1) ] else [])
            @ [
                ("address_mode", address_mode);
                ("alignment_mode", alignment_mode);
              ]);
          Regs.write ~inst r "regSH_MEM_BASES"
            [ ("shared_base", 1); ("private_base", 2) ]
        done;
        grbm_select g ~inst;
        Regs.write ~inst ~value:(0x100 * inst) r
          "regCP_MEC_DOORBELL_RANGE_LOWER" [];
        Regs.write ~inst
          ~value:((0x100 * inst) + 0xf8)
          r "regCP_MEC_DOORBELL_RANGE_UPPER" []);
    (* The doorbells of the MEC's queues, routed to the GC; NBIO 7.9 routes them
       itself. *)
    if not (Soc.nbio79 r) then begin
      Soc.route r ~port:0 ~awid:0x3 ~awaddr:0x3;
      Soc.route r ~port:3 ~awid:0x6 ~awaddr:0x3
    end;
    enable_mec g;
    if Regs.vf r then start_kiq g m
  end

(* Clock gating *)

let gate g =
  let r = g.r in
  if Regs.has (Regs.layout_of r) "regMM_ATC_L2_MISC_CG" then
    Regs.write r "regMM_ATC_L2_MISC_CG" [ ("enable", 1); ("mem_ls_enable", 1) ];
  let major, _, _ = g.gc in
  each_xcc g (fun inst ->
      safe_mode g ~inst @@ fun () ->
      Regs.update ~inst r "regRLC_CGCG_CGLS_CTRL"
        [
          ("cgcg_gfx_idle_threshold", 0x36);
          ("cgcg_en", 1);
          ("cgls_rep_compansat_delay", 0xf);
          ("cgls_en", 1);
        ];
      Regs.update ~inst r "regCP_RB_WPTR_POLL_CNTL"
        [ ("poll_frequency", 0x100); ("idle_poll_count", 0x90) ];
      Regs.update ~inst r "regCP_INT_CNTL"
        [
          ("cntx_busy_int_enable", 1);
          ("cntx_empty_int_enable", 1);
          ("cmp_busy_int_enable", 1);
        ];
      if major >= 10 then begin
        Regs.update ~inst r "regSDMA0_RLC_CGCG_CTRL" [ ("cgcg_int_enable", 1) ];
        Regs.update ~inst r "regSDMA1_RLC_CGCG_CTRL" [ ("cgcg_int_enable", 1) ]
      end;
      Regs.update ~inst r "regRLC_CGTT_MGCG_OVERRIDE"
        ((if major = 9 then
            [ ("gfxip_mgls_override", 0); ("gfxip_rep_fgcg_override", 0) ]
          else [])
        @ (if major >= 11 then
             [ ("perfmon_clock_state", 1); ("gfxip_repeater_fgcg_override", 0) ]
           else [])
        @ [
            ("gfxip_fgcg_override", 0);
            ("grbm_cgtt_sclk_override", 0);
            ("rlc_cgtt_sclk_override", 0);
            ("gfxip_mgcg_override", 0);
            ("gfxip_cgls_override", 0);
            ("gfxip_cgcg_override", 0);
          ]))

(* Processors *)

(* GC 10 renamed the shader array fields, SH to SA. *)
let index l sel =
  let reg = Regs.register l "regGRBM_GFX_INDEX" in
  let sa = if (Regs.gpu l).gc < (10, 0, 0) then "sh" else "sa" in
  Rig_amd_abi.Register.encode reg
    (match sel with
    | `Array (se, a) ->
        [
          ("se_index", se); (sa ^ "_index", a); ("instance_broadcast_writes", 1);
        ]
    | `All ->
        [
          ("se_broadcast_writes", 1);
          (sa ^ "_broadcast_writes", 1);
          ("instance_broadcast_writes", 1);
        ])

let wgps g =
  let r = g.r in
  let l = Regs.layout_of r in
  let d = Regs.discovery l in
  let field = if g.gc < (10, 0, 0) then "inactive_cus" else "inactive_wgps" in
  (* A shader array has [units] compute units, two to a work-group processor
     from GC 10. *)
  let per_array = if g.gc < (10, 0, 0) then d.gc.units else d.gc.units / 2 in
  let all = (1 lsl per_array) - 1 in
  let select ~inst sel =
    Regs.write ~inst ~value:(index l sel) r "regGRBM_GFX_INDEX" []
  in
  let rows =
    List.concat_map
      (fun inst ->
        List.init d.gc.engines (fun se ->
            Array.init d.gc.arrays (fun sh ->
                select ~inst (`Array (se, sh));
                let off =
                  Regs.field ~inst r "regCC_GC_SHADER_ARRAY_CONFIG" field
                  lor Regs.field ~inst r "regGC_USER_SHADER_ARRAY_CONFIG" field
                in
                all land lnot off)))
      g.xccs
  in
  each_xcc g (fun inst -> select ~inst `All);
  Array.of_list rows

(* A virtual function's TLBs *)

(* The KIQ polls a register every 0x20 clocks, as the kernel's does. *)
let kiq_poll_interval = 0x20

let invalidate g =
  match g.kiq with
  | None -> invalid_arg "Gfx.invalidate: the GPU has no KIQ"
  | Some (w, va) ->
      let l = Regs.layout_of g.r in
      let gpu = Regs.gpu l in
      let vmid = 0 in
      let hub ip insts =
        let name s = strf "reg%sVM_INVALIDATE_ENG17_%s" ip s in
        let req =
          Register.encode
            (Regs.register l (name "REQ"))
            [
              ("flush_type", 0);
              ("per_vmid_invalidate_req", 1 lsl vmid);
              ("invalidate_l2_ptes", 1);
              ("invalidate_l2_pde0", 1);
              ("invalidate_l2_pde1", 1);
              ("invalidate_l2_pde2", 1);
              ("invalidate_l1_ptes", 1);
              ("clear_protection_fault_status_addr", 0);
            ]
        in
        List.iter
          (fun inst ->
            (* GC's instances are the dies; MM's are reached from die 0's
               KIQ. *)
            let xcc, ri = if ip = "GC" then (inst, 0) else (0, inst) in
            let req_at = Regs.address ~inst:ri l (name "REQ") in
            let ack_at = Regs.address ~inst:ri l (name "ACK") in
            let base = kiq_bytes * xcc in
            let ptrs = Window.sub w (base + 0x1000) 0x18 in
            let wptr = Int64.to_int (Window.get64 ptrs 8) in
            let fence = va + base + 0x1010 in
            let words =
              Packet.encode Int64.of_int
                (Pm4.write_data (Register req_at) req
                @ Pm4.wait gpu (Register ack_at) Equal (1 lsl vmid)
                    ~mask:(1 lsl vmid) ~interval:kiq_poll_interval ()
                @ Pm4.write_data (Memory fence) (wptr + 1))
            in
            let n = String.length words / 4 in
            for i = 0 to n - 1 do
              Window.set32 w
                (base + (4 * ((wptr + i) mod (kiq_ring / 4))))
                (Int32.to_int (String.get_int32_le words (4 * i))
                land 0xffff_ffff)
            done;
            Window.flush w;
            Window.set64 ptrs 8 (Int64.of_int (wptr + n));
            Window.set64 g.doorbells
              (8 * kiq_doorbell xcc)
              (Int64.of_int (wptr + n));
            Regs.wait g.r (strf "the KIQ's TLB invalidation on die %d" xcc)
              (fun () -> Int64.to_int (Window.get64 ptrs 16) = wptr + 1))
          insts
      in
      hub "GC" g.xccs;
      hub "MM" (Gmc.instances g.gmc `Mm)
