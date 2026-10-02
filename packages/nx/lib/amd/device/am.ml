(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The runtime's own driver: boots an AMD GPU's IP blocks over PCI, keeps its
   interrupt ring, sets up its queues, and leaves it for the next process. *)

module D = Amd_defs
module Mmio = Nx_device_support.Mmio
module Pci = Nx_device_support.Pci
module Page_table = Nx_device_support.Page_table
open Amdev

(* The marker a session leaves in [SCRATCH_REG7]: a GPU that carries it boots
   partially, since its boot memory holds this runtime's layout. *)
let session = 0x5241_0001

(* The virtual address space every AM GPU of the process shares. *)
let space = Page_table.Space.create ~base:0x2000_0000_0000 (1 lsl 44)

type ih_ring = { ring : int; wptr : int; suffix : string; id : int }

type t = {
  d : Amdev.t;
  mm : Page_table.t;
  fw : Amdev.firmware;
  partial : bool;
  memscratch : int; (* xgmi address *)
  dummy_page : int; (* xgmi address *)
  ih : ih_ring list;
  ih_view : Mmio.t;
  (* PSP *)
  psp : string; (* the prefix of its message registers *)
  msg1 : Mmio.t;
  msg1_addr : int;
  cmd : int;
  fence : int;
  psp_ring : int;
  mutable tmr_size : int;
  mutable tmr : int;
  boot_time_tmr : bool;
  autoload_tmr : bool;
  (* SMU *)
  smu : Am_reg.version; (* the family of its messages *)
  driver_table : int;
  clocks : (int, int list) Hashtbl.t;
  (* GFX and SDMA *)
  mqd : int array;
  mutable sdma_regs : (string * int) list; (* rings set up, to stop *)
  sdma_name : string;
}

let getbits v lo hi = (v lsr lo) land ((1 lsl (hi - lo + 1)) - 1)
let rec bit_length n = if n <= 0 then 0 else 1 + bit_length (n lsr 1)
let ge a b = compare a b >= 0
let lt a b = compare a b < 0
let gcv t = gc t.d
let ver t hwip = version t.d hwip
let is_nbio_79 d = List.mem (version d D.nbio_hwip) [ (7, 9, 0); (7, 9, 1) ]

let palloc ?align ?(zero = true) ?boot t n =
  match Page_table.palloc ?align ~zero ?boot t n with
  | Some pa -> pa
  | None -> failwith "out of GPU boot memory"

(* SOC *)

let doorbell_enable d ?(aid = 0) ?(offset = 0) ?(size = 0) ~port ~awid ~awaddr
    () =
  let name =
    Printf.sprintf "%s_DOORBELL_ENTRY_%d_CTRL"
      (if ge (gc d) (12, 0, 0) then "regGDC_S2A0_S2A" else "regS2A")
      port
  in
  let f s = Printf.sprintf "s2a_doorbell_port%d_%s" port s in
  let r = reg d name in
  let v =
    Am_reg.encode r
      [
        (f "enable", 1);
        (f "awid", awid);
        (f "range_size", size);
        (f "awaddr_31_28_value", awaddr);
        (f "range_offset", offset);
      ]
  in
  if is_nbio_79 d then indirect_wreg_pcie d ~aid (Am_reg.addr r) v
  else write d ~value:v name []

let soc_init d =
  if is_nbio_79 d then begin
    let live =
      List.fold_left
        (fun acc (i, _) ->
          let harvested =
            Option.value ~default:[] (List.assoc_opt D.gc_hwip d.harvested)
          in
          if List.mem i harvested || i >= 8 then acc else acc lor (1 lsl i))
        0
        (Option.value ~default:[] (List.assoc_opt D.gc_hwip d.bases))
    in
    write d ~value:(0xff land lnot live) "regXCC_DOORBELL_FENCE" [];
    let r = reg d "regXCC_DOORBELL_FENCE" in
    List.iter
      (fun aid ->
        indirect_wreg_pcie d ~aid (Am_reg.addr r)
          (Am_reg.encode r [ ("shub_slv_mode", 1) ]))
      (List.tl d.aids);
    write d ~value:0x7ff "regBIFC_GFX_INT_MONITOR_MASK" [];
    write d ~value:0xfffff "regBIFC_DOORBELL_ACCESS_EN_PF" []
  end
  else update d "regRCC_DEV0_EPF2_STRAP2" [ ("strap_no_soft_reset_dev0_f2", 0) ];
  write d ~value:1 "regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN" []

let soc_clockgating d =
  if ge (version d D.hdp_hwip) (5, 2, 1) then
    update d "regHDP_MEM_POWER_CTRL"
      [ ("atomic_mem_power_ctrl_en", 1); ("atomic_mem_power_ds_en", 1) ]

(* GMC *)

let pf_status_reg d ip =
  Printf.sprintf "reg%sVM_L2_PROTECTION_FAULT_STATUS%s" ip
    (if ge (gc d) (12, 0, 0) then "_LO32" else "")

let flush_hdp d =
  if d.is_vf then
    write d ~value:0 "regBIF_BX_DEV0_EPF0_VF0_HDP_MEM_COHERENCY_FLUSH_CNTL" []
  else wreg d (read d "regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL" / 4) 0

let packet3 op n =
  (3 lsl 30) lor ((n land 0x3FFF) lsl 16) lor ((op land 0xFF) lsl 8)

(* Invalidates the TLBs of hub [ip] ("GC" or "MM") for [vmid]. A VF asks its KIQ
   to, since the invalidation engines are the host's. *)
let flush_tlb d ip ~vmid =
  flush_hdp d;
  if ip = "MM" || d.gc_hub then begin
    let r n = Printf.sprintf "reg%sVM_INVALIDATE_ENG17_%s" ip n in
    let req =
      Am_reg.encode
        (reg d (r "REQ"))
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
    let insts = if ip = "MM" then d.mm_insts else List.init d.xccs Fun.id in
    match d.kiq with
    | Some (kiq, kiq_va) when d.is_vf ->
        List.iter
          (fun inst ->
            let xcc, ri = if ip = "GC" then (inst, 0) else (0, inst) in
            let req_addr = Am_reg.addr ~inst:ri (reg d (r "REQ"))
            and ack_addr = Am_reg.addr ~inst:ri (reg d (r "ACK")) in
            let base = 0x3000 * xcc in
            let ring = Mmio.sub kiq base 0x1000
            and ptrs = Mmio.sub kiq (base + 0x1000) 0x18 in
            let wptr = Int64.to_int (Mmio.get64 ptrs 8) in
            let fence = kiq_va + base + 0x1010 in
            let pkt =
              [
                packet3 D.packet3_write_data 3;
                1 lsl 16;
                req_addr;
                0;
                req;
                packet3 D.packet3_wait_reg_mem 5;
                3;
                ack_addr;
                0;
                1 lsl vmid;
                1 lsl vmid;
                0x20;
                packet3 D.packet3_write_data 3;
                D.wr_confirm lor (5 lsl 8);
                lo32 fence;
                hi32 fence;
                wptr + 1;
              ]
            in
            List.iteri
              (fun i w -> Mmio.set32 ring (4 * ((wptr + i) mod 0x400)) w)
              pkt;
            let next = wptr + List.length pkt in
            Mmio.set64 ptrs 8 (Int64.of_int next);
            Mmio.set64 d.doorbells
              (8 * (D.amdgpu_doorbell_kiq + (xcc * 0x20)))
              (Int64.of_int next);
            wait_cond
              ~msg:(Printf.sprintf "KIQ TLB flush on XCC %d" xcc)
              (fun () -> Int64.to_int (Mmio.get64 ptrs 16))
              (wptr + 1))
          insts
    | _ ->
        List.iter
          (fun inst ->
            if ip = "MM" then
              wait_cond ~msg:"MM TLB flush semaphore"
                (fun () -> read d ~inst "regMMVM_INVALIDATE_ENG17_SEM" land 1)
                1;
            write d ~inst ~value:req (r "REQ") [];
            wait_cond ~msg:"TLB flush"
              (fun () -> read d ~inst (r "ACK") land (1 lsl vmid))
              (1 lsl vmid);
            if ip = "MM" then begin
              write d ~inst ~value:0 "regMMVM_INVALIDATE_ENG17_SEM" [];
              if ge (gc d) (11, 0, 0) then begin
                update d ~inst "regMMVM_L2_BANK_SELECT_RESERVED_CID2"
                  [ ("reserved_cache_private_invalidation", 1) ];
                ignore (read d ~inst "regMMVM_L2_BANK_SELECT_RESERVED_CID2")
              end
            end)
          insts
  end

let trans_further d = lt (gc d) (10, 0, 0)

let enable_vm_addressing t ip ~vmid ~inst =
  let d = t.d in
  let ctx s = Printf.sprintf "reg%sVM_CONTEXT%d_%s" ip vmid s in
  let vm_base = Page_table.Space.base space in
  let vm_end = Int.min (vm_base + (1 lsl 48) - 1) 0x7fffffffffff in
  write_pair d ~inst
    (ctx "PAGE_TABLE_START_ADDR")
    ~lo:"_LO32" ~hi:"_HI32" (vm_base lsr 12);
  write_pair d ~inst
    (ctx "PAGE_TABLE_END_ADDR")
    ~lo:"_LO32" ~hi:"_HI32" (vm_end lsr 12);
  write_pair d ~inst
    (ctx "PAGE_TABLE_BASE_ADDR")
    ~lo:"_LO32" ~hi:"_HI32"
    (paddr2xgmi d (Page_table.root t.mm) lor 1);
  let faults =
    [ "pde0"; "dummy_page"; "range"; "valid"; "read"; "write"; "execute" ]
  in
  write d ~inst ~value:0x1800000 (ctx "CNTL")
    (List.map (fun x -> (x ^ "_protection_fault_enable_interrupt", 1)) faults
    @ List.map (fun x -> (x ^ "_protection_fault_enable_default", 1)) faults
    @ [
        ("enable_context", 1);
        ( "page_table_depth",
          (if trans_further d then 2 else 3) - D.amdgpu_vm_pdb2 );
        ("page_table_block_size", if trans_further d then 9 else 0);
      ])

let init_hub t ip insts =
  let d = t.d in
  let r s = Printf.sprintf "reg%s%s" ip s in
  List.iter
    (fun inst ->
      write d ~inst ~value:0 (r "MC_VM_AGP_BASE") [];
      write d ~inst ~value:(0xffffffffffff lsr 24) (r "MC_VM_AGP_BOT") [];
      write d ~inst ~value:0 (r "MC_VM_AGP_TOP") [];
      write d ~inst ~value:(d.fb_base lsr 18)
        (r "MC_VM_SYSTEM_APERTURE_LOW_ADDR")
        [];
      write d ~inst ~value:(d.fb_end lsr 18)
        (r "MC_VM_SYSTEM_APERTURE_HIGH_ADDR")
        [];
      write_pair d ~inst
        (r "MC_VM_SYSTEM_APERTURE_DEFAULT_ADDR")
        ~lo:"_LSB" ~hi:"_MSB" (t.memscratch lsr 12);
      write_pair d ~inst
        (r "VM_L2_PROTECTION_FAULT_DEFAULT_ADDR")
        ~lo:"_LO32" ~hi:"_HI32" (t.dummy_page lsr 12);
      update d ~inst
        (r "VM_L2_PROTECTION_FAULT_CNTL2")
        [ ("active_page_migration_pte_read_retry", 1) ];
      update d ~inst (r "MC_VM_MX_L1_TLB_CNTL")
        [
          ("enable_l1_tlb", 1);
          ("system_access_mode", 3);
          ("enable_advanced_driver_model", 1);
          ("system_aperture_unmapped_access", 0);
          ("mtype", mtype_uc (gc d));
        ];
      update d ~inst (r "VM_L2_CNTL")
        [
          ("enable_l2_cache", 1);
          ("enable_default_page_out_to_system_memory", 1);
          ("l2_pde0_cache_tag_generation_mode", 0);
          ("pde_fault_classification", 0);
          ("context1_identity_access_mode", 1);
          ("identity_mode_fragment_size", 0);
          ( "enable_l2_fragment_processing",
            if lt (gc d) (10, 0, 0) then 1 else 0 );
        ];
      update d ~inst (r "VM_L2_CNTL2")
        [ ("invalidate_all_l1_tlbs", 1); ("invalidate_l2_cache", 1) ];
      write d ~inst (r "VM_L2_CNTL3")
        [
          ("l2_cache_4k_associativity", 1);
          ("l2_cache_bigk_associativity", 1);
          ("bank_select", if trans_further d then 12 else 9);
          ("l2_cache_bigk_fragment_size", if trans_further d then 9 else 6);
        ];
      write d ~inst (r "VM_L2_CNTL4") [ ("l2_cache_4k_partition_count", 1) ];
      if ge (gc d) (10, 0, 0) then
        write d ~inst (r "VM_L2_CNTL5") [ ("walker_priority_client_id", 0x1ff) ];
      enable_vm_addressing t ip ~vmid:0 ~inst;
      write_pair d ~inst
        (r "VM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR")
        ~lo:"_LO32" ~hi:"_HI32" 0xfffffffff;
      write_pair d ~inst
        (r "VM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR")
        ~lo:"_LO32" ~hi:"_HI32" 0;
      write_pair d ~inst
        (r "VM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET")
        ~lo:"_LO32" ~hi:"_HI32" 0;
      for eng = 0 to 17 do
        write_pair d ~inst
          (r (Printf.sprintf "VM_INVALIDATE_ENG%d_ADDR_RANGE" eng))
          ~lo:"_LO32" ~hi:"_HI32" 0x1fffffffff
      done)
    insts;
  if ip = "GC" then d.gc_hub <- true

(* SMU *)

let smu_msg t name =
  match List.assoc_opt name (D.smu_messages t.smu) with
  | Some m -> m
  | None ->
      failwith
        (Printf.sprintf "no SMU message %s for SMU %s" name
           (Am_reg.pp_version t.smu))

let send_msg ?(read_back = false) ?(timeout_ms = 10_000) ?(debug = false) t msg
    param =
  let d = t.d in
  let resp, arg, cmd =
    if debug then
      ("mmMP1_SMN_C2PMSG_54", "mmMP1_SMN_C2PMSG_53", "mmMP1_SMN_C2PMSG_75")
    else ("mmMP1_SMN_C2PMSG_90", "mmMP1_SMN_C2PMSG_82", "mmMP1_SMN_C2PMSG_66")
  in
  write d ~value:0 resp [];
  write d ~value:param arg [];
  write d ~value:msg cmd [];
  wait_cond ~timeout_ms
    ~msg:(Printf.sprintf "SMU message 0x%x" msg)
    (fun () -> read d resp)
    1;
  if read_back then read d arg else 0

let smu_init t =
  let mc = paddr2mc t.d t.driver_table in
  send_msg t (smu_msg t "PPSMC_MSG_SetDriverDramAddrHigh") (hi32 mc) |> ignore;
  send_msg t (smu_msg t "PPSMC_MSG_SetDriverDramAddrLow") (lo32 mc) |> ignore;
  send_msg t (smu_msg t "PPSMC_MSG_EnableAllSmuFeatures") 0 |> ignore

let smu_alive t =
  (try
     ignore (send_msg ~timeout_ms:100 t (smu_msg t "PPSMC_MSG_GetSmuVersion") 0)
   with Failure _ -> ());
  read t.d "mmMP1_SMN_C2PMSG_90" <> 0

let mode1_reset t =
  let mp0 = ver t D.mp0_hwip in
  if ge mp0 (14, 0, 0) || List.mem mp0 [ (13, 0, 0); (13, 0, 7); (13, 0, 10) ]
  then ignore (send_msg ~debug:true t 2 0)
  else if List.mem mp0 [ (13, 0, 6); (13, 0, 12); (13, 0, 15) ] then
    ignore (send_msg t (smu_msg t "PPSMC_MSG_GfxDriverReset") 1)
  else ignore (send_msg t (smu_msg t "PPSMC_MSG_Mode1Reset") 0);
  if not (is_hive t.d) then begin
    sleep_ms 500;
    (* Configuration reads fail fast on a wedged GPU, where register reads would
       hang, so the reset is watched through them. *)
    wait_cond ~timeout_ms:2000
      ~msg:
        (Printf.sprintf "%s did not return from its reset; reboot the machine"
           t.d.bus)
      (fun () -> Pci.read_config t.d.pci 0 2)
      0x1002
  end

let clocks t clk =
  match Hashtbl.find_opt t.clocks clk with
  | Some l -> l
  | None ->
      let by_index = smu_msg t "PPSMC_MSG_GetDpmFreqByIndex" in
      let q i =
        send_msg ~read_back:true t by_index ((clk lsl 16) lor i) land 0x7fffffff
      in
      let l = List.init (q 0xff) q in
      Hashtbl.replace t.clocks clk l;
      l

(* Sets every clock to its [level]th state, the last one if negative. *)
let set_clocks t level =
  let clks =
    List.map (smu_msg t) [ "PPCLK_UCLK"; "PPCLK_FCLK"; "PPCLK_SOCCLK" ]
    @
    if List.mem (ver t D.mp0_hwip) [ (13, 0, 6); (13, 0, 12); (13, 0, 15) ] then
      []
    else [ smu_msg t "PPCLK_GFXCLK" ]
  in
  List.iter
    (fun clk ->
      match clocks t clk with
      | [] -> ()
      | vals ->
          let n = List.length vals in
          let v = List.nth vals (if level < 0 then n + level else level) in
          (try
             ignore
               (send_msg ~timeout_ms:20 t
                  (smu_msg t "PPSMC_MSG_SetSoftMinByFreq")
                  ((clk lsl 16) lor v))
           with Failure _ -> ());
          if ge (gcv t) (10, 0, 0) then
            ignore
              (send_msg t
                 (smu_msg t "PPSMC_MSG_SetSoftMaxByFreq")
                 ((clk lsl 16) lor v)))
    clks

(* The machine-check banks of the SMU, for a fatal hardware error. *)
let aca_banks t ~uncorrectable =
  let count, dump =
    if uncorrectable then
      ("PPSMC_MSG_QueryValidMcaCount", "PPSMC_MSG_McaBankDumpDW")
    else ("PPSMC_MSG_QueryValidMcaCeCount", "PPSMC_MSG_McaBankCeDumpDW")
  in
  match List.assoc_opt count (D.smu_messages t.smu) with
  | None -> []
  | Some count ->
      let dump = smu_msg t dump in
      let rd bank i =
        send_msg ~read_back:true t dump ((bank lsl 16) lor ((i * 8) + 4))
        lsl 32
        lor send_msg ~read_back:true t dump ((bank lsl 16) lor (i * 8))
      in
      List.init (send_msg ~read_back:true t count 0) (fun bank ->
          List.init 16 (rd bank))

(* PSP *)

let psp_reg t n = Printf.sprintf "%s_%d" t.psp n
let sos_alive t = read t.d (psp_reg t 81) <> 0

let wait_bootloader t =
  wait_cond ~msg:"the PSP bootloader is not ready"
    (fun () -> read t.d (psp_reg t 35) land 0x80000000)
    0x80000000

(* Places [data] in the PSP's message buffer, padded to 16 bytes. *)
let prep_msg1 t data =
  let n = String.length data + 4 in
  let n = (n + 15) / 16 * 16 in
  if n > Mmio.length t.msg1 then failwith "PSP message buffer too small";
  Mmio.write t.msg1 0 (data ^ String.make (n - String.length data) '\000');
  flush_hdp t.d

let bootloader_load t fw_type comp =
  match List.assoc_opt fw_type t.fw.sos with
  | None -> ()
  | Some data ->
      wait_bootloader t;
      prep_msg1 t data;
      write t.d ~value:(t.msg1_addr lsr 20) (psp_reg t 36) [];
      write t.d ~value:comp (psp_reg t 35) [];
      if comp <> D.psp_bl__load_sosdrv then wait_bootloader t

let ring_submit t cmd =
  let d = t.d in
  let prev = read d (psp_reg t 67) in
  let frame = Bytes.make D.Psp_gfx_rb_frame.sizeof '\000' in
  let cmd_mc = paddr2mc d t.cmd and fence_mc = paddr2mc d t.fence in
  let open D.Psp_gfx_rb_frame in
  set frame fence_value (prev + 1);
  set frame cmd_buf_addr_lo (lo32 cmd_mc);
  set frame cmd_buf_addr_hi (hi32 cmd_mc);
  set frame fence_addr_lo (lo32 fence_mc);
  set frame fence_addr_hi (hi32 fence_mc);
  Mmio.write d.vram t.cmd (Bytes.to_string cmd);
  Mmio.write d.vram (t.psp_ring + (prev * 4)) (Bytes.to_string frame);
  write d ~value:(prev + (D.Psp_gfx_rb_frame.sizeof / 4)) (psp_reg t 67) [];
  wait_cond ~msg:"the PSP ring is not responding"
    (fun () -> Mmio.get32 d.vram t.fence)
    (prev + 1);
  let resp = Mmio.read d.vram t.cmd (Bytes.length cmd) in
  let status = get resp 0 D.Psp_gfx_cmd_resp.resp__status in
  if status <> 0 then
    failwith
      (Printf.sprintf "PSP command %d failed with status 0x%x"
         (get resp 0 D.Psp_gfx_cmd_resp.cmd_id)
         status);
  resp

let psp_cmd id =
  let b = Bytes.make D.Psp_gfx_cmd_resp.sizeof '\000' in
  set b D.Psp_gfx_cmd_resp.cmd_id id;
  b

let load_ip_fw t (types, data) =
  prep_msg1 t data;
  List.iter
    (fun ty ->
      let open D.Psp_gfx_cmd_resp in
      let c = psp_cmd D.gfx_cmd_id_load_ip_fw in
      set c cmd__cmd_load_ip_fw__fw_phy_addr_hi (hi32 t.msg1_addr);
      set c cmd__cmd_load_ip_fw__fw_phy_addr_lo (lo32 t.msg1_addr);
      set c cmd__cmd_load_ip_fw__fw_size (String.length data);
      set c cmd__cmd_load_ip_fw__fw_type ty;
      ignore (ring_submit t c))
    types

let load_toc t n =
  let open D.Psp_gfx_cmd_resp in
  let c = psp_cmd D.gfx_cmd_id_load_toc in
  set c cmd__cmd_load_toc__toc_phy_addr_hi (hi32 t.msg1_addr);
  set c cmd__cmd_load_toc__toc_phy_addr_lo (lo32 t.msg1_addr);
  set c cmd__cmd_load_toc__toc_size n;
  get (ring_submit t c) 0 resp__tmr_size

let tmr_init t =
  if t.partial then t.tmr_size <- read t.d "regSCRATCH_REG5"
  else begin
    let toc = List.assoc D.psp_fw_type_psp_toc t.fw.sos in
    prep_msg1 t toc;
    t.tmr_size <- load_toc t (String.length toc)
  end;
  (* The first runtime allocation on full and partial boots alike, so that the
     resident TMR keeps its address. *)
  if not t.boot_time_tmr then
    t.tmr <-
      palloc ~boot:false ~align:D.psp_tmr_alignment ~zero:false t.mm t.tmr_size

let tmr_load t =
  let open D.Psp_gfx_cmd_resp in
  let c = psp_cmd D.gfx_cmd_id_setup_tmr in
  let mc = if t.tmr <> 0 then paddr2mc t.d t.tmr else 0 in
  let sys = if t.tmr <> 0 then paddr2xgmi t.d t.tmr else 0 in
  set c cmd__cmd_setup_tmr__buf_phy_addr_hi (hi32 mc);
  set c cmd__cmd_setup_tmr__buf_phy_addr_lo (lo32 mc);
  set c cmd__cmd_setup_tmr__system_phy_addr_hi (hi32 sys);
  set c cmd__cmd_setup_tmr__system_phy_addr_lo (lo32 sys);
  let bit, _ = cmd__cmd_setup_tmr__bitfield__virt_phy_addr_bits in
  Bytes.set_uint8 c (bit / 8)
    (Bytes.get_uint8 c (bit / 8) lor (1 lsl (bit mod 8)));
  set c cmd__cmd_setup_tmr__buf_size (if t.tmr <> 0 then t.tmr_size else 0);
  ignore (ring_submit t c)

let ring_create t =
  let d = t.d in
  if read d (psp_reg t 71) <> 0 then begin
    write d ~value:D.gfx_ctrl_cmd_id_destroy_rings (psp_reg t 64) [];
    sleep_ms 20
  end;
  wait_cond ~msg:"the PSP's OS is not ready"
    (fun () -> read d (psp_reg t 64) land 0x80000000)
    0x80000000;
  let mc = paddr2mc d t.psp_ring in
  write d ~value:(lo32 mc) (psp_reg t 69) [];
  write d ~value:(hi32 mc) (psp_reg t 70) [];
  write d ~value:0x10000 (psp_reg t 71) [];
  write d ~value:(D.psp_ring_type__km lsl 16) (psp_reg t 64) [];
  sleep_ms 20;
  wait_cond ~msg:"the PSP ring was not created"
    (fun () -> read d (psp_reg t 64) land 0x8000FFFF)
    0x80000000

let spatial_partition t mode =
  let c = psp_cmd D.gfx_cmd_id_sriov_spatial_part in
  set c D.Psp_gfx_cmd_resp.cmd__cmd_spatial_part__mode mode;
  ignore (ring_submit t c)

let psp_init t =
  let spl =
    if ge (ver t D.mp0_hwip) (14, 0, 0) then D.psp_fw_type_psp_spl
    else D.psp_fw_type_psp_kdb
  in
  let components =
    D.
      [
        (psp_fw_type_psp_kdb, psp_bl__load_key_database);
        (spl, psp_bl__load_tos_spl_table);
        (psp_fw_type_psp_sys_drv, psp_bl__load_sysdrv);
        (psp_fw_type_psp_soc_drv, psp_bl__load_socdrv);
        (psp_fw_type_psp_intf_drv, psp_bl__load_intfdrv);
        (psp_fw_type_psp_dbg_drv, psp_bl__load_dbgdrv);
        (psp_fw_type_psp_ras_drv, psp_bl__load_rasdrv);
        (psp_fw_type_psp_sos, psp_bl__load_sosdrv);
      ]
  in
  if not (sos_alive t) then begin
    List.iter (fun (fw, comp) -> bootloader_load t fw comp) components;
    wait_cond ~msg:"the PSP's OS failed to start"
      (fun () -> Bool.to_int (sos_alive t))
      1
  end;
  ring_create t;
  if List.mem_assoc D.psp_fw_type_psp_toc t.fw.sos then tmr_init t;
  Option.iter (load_ip_fw t) t.fw.smu_psp;
  if (not t.boot_time_tmr) || not t.autoload_tmr then tmr_load t;
  List.iter (load_ip_fw t) t.fw.descs;
  if ge (gcv t) (11, 0, 0) then
    ignore (ring_submit t (psp_cmd D.gfx_cmd_id_autoload_rlc))
  else
    load_ip_fw t
      ([ D.gfx_fw_type_reg_list ], List.assoc D.psp_fw_type_psp_rl t.fw.sos)

(* GFX *)

let grbm_select ?(me = 0) ?(pipe = 0) ?(queue = 0) ?(vmid = 0) ?(inst = 0) d =
  write d ~inst "regGRBM_GFX_CNTL"
    [ ("meid", me); ("pipeid", pipe); ("vmid", vmid); ("queueid", queue) ]

let xccs_list d = List.init d.xccs Fun.id

let enable_mec d =
  List.iter
    (fun inst ->
      if ge (gc d) (10, 0, 0) then
        update d ~inst "regCP_MEC_RS64_CNTL"
          [ ("mec_pipe0_reset", 0); ("mec_pipe0_active", 1); ("mec_halt", 0) ]
      else write d ~inst ~value:0 "regCP_MEC_CNTL" [])
    (xccs_list d);
  sleep_ms 50

let halt_mec d =
  List.iter
    (fun inst ->
      if ge (gc d) (10, 0, 0) then
        update d ~inst "regCP_MEC_RS64_CNTL" [ ("mec_halt", 1) ]
      else
        update d ~inst "regCP_MEC_CNTL"
          [ ("mec_me1_halt", 1); ("mec_me2_halt", 1) ])
    (xccs_list d)

let config_mec t =
  let d = t.d in
  let helper ~eng ~cntl ~me xcc =
    grbm_select d ~me ~pipe:0 ~inst:xcc;
    write_pair d ~inst:xcc
      (Printf.sprintf "regCP_%s_PRGRM_CNTR_START" cntl)
      ~lo:"" ~hi:"_HI"
      (List.assoc eng t.fw.ucode_start lsr 2);
    grbm_select d ~inst:xcc;
    let r =
      Printf.sprintf "regCP_%s_CNTL" (if eng = "MEC" then "MEC_RS64" else "ME")
    in
    let f = String.lowercase_ascii eng ^ "_pipe0_reset" in
    update d ~inst:xcc r [ (f, 1) ];
    update d ~inst:xcc r [ (f, 0) ]
  in
  List.iter
    (fun xcc ->
      if lt (gc d) (10, 0, 0) then
        update d ~inst:xcc "regCP_MEC_CNTL"
          [
            ("mec_invalidate_icache", 1);
            ("mec_me1_pipe0_reset", 1);
            ("mec_me2_pipe0_reset", 1);
            ("mec_me1_halt", 1);
            ("mec_me2_halt", 1);
          ];
      if ge (gc d) (12, 0, 0) then begin
        helper ~eng:"PFP" ~cntl:"PFP" ~me:0 xcc;
        helper ~eng:"ME" ~cntl:"ME" ~me:0 xcc
      end;
      if ge (gc d) (10, 0, 0) then helper ~eng:"MEC" ~cntl:"MEC_RS64" ~me:1 xcc)
    (xccs_list d)

let dequeue_hqds ?(wait = true) d =
  List.iter
    (fun (me, pipe, queue) ->
      List.iter
        (fun inst ->
          grbm_select d ~me ~pipe ~queue ~inst;
          if read d ~inst "regCP_HQD_ACTIVE" land 1 = 1 then begin
            write d ~inst ~value:2 "regCP_HQD_DEQUEUE_REQUEST" [];
            write d ~inst ~value:1 "regSPI_COMPUTE_QUEUE_RESET" [];
            if wait && d.errors = [] then
              (* A wedged wave can survive the reset; the kernel driver
                 tolerates it too. *)
              try
                wait_cond ~msg:"HQD dequeue"
                  (fun () -> read d ~inst "regCP_HQD_ACTIVE" land 1)
                  0
              with Failure _ -> ()
          end)
        (xccs_list d))
    ([ (1, 0, 0); (1, 0, 1) ] @ if d.is_vf then [ (2, 1, 0) ] else []);
  grbm_select d

let reset_mec t =
  let d = t.d in
  dequeue_hqds d;
  if lt (gc d) (12, 0, 0) then begin
    List.iter
      (fun inst ->
        write d ~inst "regGRBM_SOFT_RESET"
          [ ("soft_reset_cp", 1); ("soft_reset_cpc", 1) ])
      (xccs_list d);
    sleep_ms 50;
    List.iter
      (fun inst -> write d ~inst ~value:0 "regGRBM_SOFT_RESET" [])
      (xccs_list d)
  end;
  config_mec t;
  enable_mec d

(* Programs a compute queue's MQD and hardware queue descriptor; returns its
   doorbell index. *)
let setup_compute_ring ?kiq t ~ring ~ring_bytes ~rptr ~wptr ~eop ~eop_bytes ~idx
    ~aql =
  let d = t.d in
  let is_kiq = Option.is_some kiq in
  let me, pipe, queue = if is_kiq then (2, 1, 0) else (1, idx / 4, idx mod 4) in
  let doorbell =
    match kiq with
    | Some xcc -> D.amdgpu_doorbell_kiq + (xcc * 0x20)
    | None -> D.amdgpu_navi10_doorbell_mec_ring0
  in
  let slot = queue + if is_kiq then 2 else 0 in
  let major, _, _ = gc d in
  let module M = (val D.mqd major : D.MQD) in
  let enc name kvs = Am_reg.encode (reg d name) kvs in
  let insts =
    match kiq with
    | Some x -> [ x ]
    | None -> List.init (if aql then d.xccs else 1) Fun.id
  in
  List.iter
    (fun xcc ->
      grbm_select d ~me ~pipe ~queue ~inst:xcc;
      let b = Bytes.make M.sizeof '\000' in
      let mqd_pa = t.mqd.(slot) + (0x1000 * xcc) in
      let mqd_mc = paddr2mc d mqd_pa in
      set b M.header 0xC0310800;
      set b M.cp_mqd_base_addr_lo (lo32 mqd_mc);
      set b M.cp_mqd_base_addr_hi (hi32 mqd_mc);
      set b M.cp_hqd_pipe_priority 2;
      set b M.cp_hqd_queue_priority 0xf;
      set b M.cp_hqd_quantum 0x111;
      set b M.cp_hqd_persistent_state
        (enc "regCP_HQD_PERSISTENT_STATE"
           [ ("preload_size", 0x55); ("preload_req", 1) ]);
      set b M.cp_hqd_pq_base_lo (lo32 (ring lsr 8));
      set b M.cp_hqd_pq_base_hi (hi32 (ring lsr 8));
      set b M.cp_hqd_pq_rptr_report_addr_lo (lo32 rptr);
      set b M.cp_hqd_pq_rptr_report_addr_hi (hi32 rptr);
      set b M.cp_hqd_pq_wptr_poll_addr_lo (lo32 wptr);
      set b M.cp_hqd_pq_wptr_poll_addr_hi (hi32 wptr);
      set b M.cp_hqd_pq_doorbell_control
        (enc "regCP_HQD_PQ_DOORBELL_CONTROL"
           [ ("doorbell_offset", doorbell * 2); ("doorbell_en", 1) ]);
      set b M.cp_hqd_pq_control
        (enc "regCP_HQD_PQ_CONTROL"
           ([
              ("rptr_block_size", 5);
              ("unord_dispatch", 0);
              ("queue_size", bit_length (ring_bytes / 4) - 2);
            ]
           @ (if is_kiq then [ ("priv_state", 1); ("kmd_queue", 1) ] else [])
           @
           if aql then
             [
               ("queue_full_en", 1);
               ("slot_based_wptr", 2);
               ("no_update_rptr", Bool.to_int (xcc <> 0 || d.xccs = 1));
             ]
           else []));
      set b M.cp_hqd_ib_control
        (enc "regCP_HQD_IB_CONTROL" [ ("min_ib_avail_size", 3) ]);
      set b M.cp_hqd_hq_status0 0x20004000;
      set b M.cp_mqd_control (enc "regCP_MQD_CONTROL" [ ("priv_state", 1) ]);
      set b M.cp_hqd_vmid 0;
      set b M.cp_hqd_aql_control (Bool.to_int aql);
      set b M.cp_hqd_eop_base_addr_lo (lo32 (eop lsr 8));
      set b M.cp_hqd_eop_base_addr_hi (hi32 (eop lsr 8));
      set b M.cp_hqd_eop_control
        (enc "regCP_HQD_EOP_CONTROL"
           [ ("eop_size", bit_length (eop_bytes / 4) - 2) ]);
      if aql && d.xccs > 1 then begin
        set b (Option.get M.compute_tg_chunk_size) 1;
        set b (Option.get M.compute_current_logic_xcc_id) xcc;
        set b (Option.get M.cp_mqd_stride_size) 0x1000
      end;
      List.iter (fun f -> set b f 0xffffffff) M.compute_static_thread_mgmt;
      let mqd = Bytes.to_string b in
      Mmio.write d.vram mqd_pa mqd;
      let first = Am_reg.addr ~inst:xcc (reg d "regCP_MQD_BASE_ADDR")
      and last = Am_reg.addr ~inst:xcc (reg d "regCP_HQD_PQ_WPTR_HI") in
      for i = 0 to last - first do
        wreg d ~inst:xcc (first + i) (get mqd (4 * (0x80 + i)) (0, 4))
      done;
      write d ~inst:xcc ~value:1 "regCP_HQD_ACTIVE" [];
      flush_hdp d;
      grbm_select d ~inst:xcc)
    insts;
  doorbell

let gfx_init t =
  let d = t.d in
  (* GC 9.4.3 has no RLC autoload to wait for. *)
  if has_reg d "regRLC_RLCS_BOOTLOAD_STATUS" then
    wait_cond ~msg:"the RLC autoload"
      (fun () ->
        Bool.to_int
          (read d "regCP_STAT" = 0
          || field d "regRLC_RLCS_BOOTLOAD_STATUS" "bootload_complete" = 0))
      1;
  init_hub t "GC" (xccs_list d);
  if t.partial then reset_mec t
  else begin
    config_mec t;
    List.iter
      (fun inst ->
        write d ~inst
          ~value:(read d ~inst "regTCP_CNTL" lor 0x20000000)
          "regTCP_CNTL" [])
      (xccs_list d);
    List.iter
      (fun inst -> write d ~inst ~value:1 "regRLC_CNTL" [])
      (xccs_list d);
    List.iter
      (fun inst ->
        update d ~inst "regRLC_SRM_CNTL"
          [ ("srm_enable", 1); ("auto_incr_addr", 1) ])
      (xccs_list d);
    List.iter
      (fun inst -> write d ~inst ~value:0xf "regRLC_SPM_MC_CNTL" [])
      (xccs_list d);
    (match version d D.nbio_hwip with
    | 7, 9, _ -> ()
    | _ ->
        doorbell_enable d ~port:0 ~awid:0x3 ~awaddr:0x3 ();
        doorbell_enable d ~port:3 ~awid:0x6 ~awaddr:0x3 ());
    let major, minor, _ = gc d in
    List.iter
      (fun inst ->
        if List.mem (gc d) [ (9, 4, 3); (9, 5, 0) ] then begin
          write d ~inst ~value:0x2a114042 "regGB_ADDR_CONFIG" [];
          update d ~inst "regTCP_UTCL1_CNTL2" [ ("spare", 1) ]
        end;
        update d ~inst "regGRBM_CNTL" [ ("read_timeout", 0xff) ];
        for vmid = 0 to 15 do
          grbm_select d ~vmid ~inst;
          write d ~inst "regSH_MEM_CONFIG"
            ((if major >= 10 then [ ("initial_inst_prefetch", 3) ]
              else [ ("retry_disable", 1) ])
            @ (if (major, minor) = (9, 4) then [ ("f8_mode", 1) ] else [])
            @ [
                ("address_mode", D.sh_mem_address_mode_64 major);
                ("alignment_mode", D.sh_mem_alignment_mode_unaligned major);
              ]);
          write d ~inst "regSH_MEM_BASES"
            [ ("shared_base", 1); ("private_base", 2) ]
        done;
        grbm_select d ~inst;
        write d ~inst ~value:(0x100 * inst) "regCP_MEC_DOORBELL_RANGE_LOWER" [];
        write d ~inst
          ~value:((0x100 * inst) + 0xf8)
          "regCP_MEC_DOORBELL_RANGE_UPPER" [])
      (xccs_list d);
    enable_mec d;
    if d.is_vf then begin
      (* A VF's KIQ: per XCC a ring, its pointers and fence, and an EOP. *)
      let size = 0x3000 * d.xccs in
      let va =
        match Page_table.Space.alloc space size with
        | Some va -> va
        | None -> failwith "no GPU addresses for the KIQ"
      in
      let kiq, pages = Pci.alloc_sysmem t.d.pci ~va size in
      let page = Pci.page t.d.pci in
      ignore
        (Page_table.map ~snooped:true ~uncached:true t.mm ~va Page_table.Sys
           (List.map (fun p -> (p, page)) pages));
      List.iter
        (fun xcc ->
          let b = va + (0x3000 * xcc) in
          ignore
            (setup_compute_ring ~kiq:xcc t ~ring:b ~ring_bytes:0x1000
               ~rptr:(b + 0x1000) ~wptr:(b + 0x1008) ~eop:(b + 0x2000)
               ~eop_bytes:0x1000 ~idx:0 ~aql:false))
        (xccs_list d);
      List.iter
        (fun inst ->
          update d ~inst "regRLC_CP_SCHEDULERS"
            [ ("scheduler0", (2 lsl 5) lor (1 lsl 3) lor 0x80) ])
        (xccs_list d);
      d.kiq <- Some (kiq, va)
    end;
    if d.xccs > 1 && not d.is_vf then spatial_partition t 1
  end

let gfx_clockgating d =
  if has_reg d "regMM_ATC_L2_MISC_CG" then
    write d "regMM_ATC_L2_MISC_CG" [ ("enable", 1); ("mem_ls_enable", 1) ];
  let major, _, _ = gc d in
  List.iter
    (fun inst ->
      write d ~inst "regRLC_SAFE_MODE" [ ("message", 1); ("cmd", 1) ];
      wait_cond ~msg:"RLC safe mode"
        (fun () -> read d ~inst "regRLC_SAFE_MODE" land 1)
        0;
      update d ~inst "regRLC_CGCG_CGLS_CTRL"
        [
          ("cgcg_gfx_idle_threshold", 0x36);
          ("cgcg_en", 1);
          ("cgls_rep_compansat_delay", 0xf);
          ("cgls_en", 1);
        ];
      update d ~inst "regCP_RB_WPTR_POLL_CNTL"
        [ ("poll_frequency", 0x100); ("idle_poll_count", 0x90) ];
      update d ~inst "regCP_INT_CNTL"
        [
          ("cntx_busy_int_enable", 1);
          ("cntx_empty_int_enable", 1);
          ("cmp_busy_int_enable", 1);
        ];
      if major >= 10 then begin
        update d ~inst "regSDMA0_RLC_CGCG_CTRL" [ ("cgcg_int_enable", 1) ];
        update d ~inst "regSDMA1_RLC_CGCG_CTRL" [ ("cgcg_int_enable", 1) ]
      end;
      update d ~inst "regRLC_CGTT_MGCG_OVERRIDE"
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
          ]);
      write d ~inst "regRLC_SAFE_MODE" [ ("message", 0); ("cmd", 1) ])
    (xccs_list d)

(* IH *)

let ih_bytes = 256 lsl 10

let ih_init t =
  let d = t.d in
  List.iter
    (fun r ->
      write_pair d "regIH_RB_BASE" ~lo:r.suffix ~hi:("_HI" ^ r.suffix)
        (paddr2mc d r.ring lsr 8);
      write d
        ("regIH_RB_CNTL" ^ r.suffix)
        ([
           ("mc_space", 4);
           ("wptr_overflow_clear", 1);
           ("rb_size", bit_length ((ih_bytes / 4) - 1));
           ("mc_snoop", 1);
           ("mc_ro", 0);
           ("mc_vmid", 0);
         ]
        @
        if r.id = 0 then [ ("wptr_overflow_enable", 1); ("rptr_rearm", 1) ]
        else [ ("rb_full_drain_enable", 1) ]);
      if r.id = 0 then
        write_pair d "regIH_RB_WPTR_ADDR" ~lo:"_LO" ~hi:"_HI"
          (paddr2mc d r.wptr);
      write d ~value:0 ("regIH_RB_WPTR" ^ r.suffix) [];
      write d ~value:0 ("regIH_RB_RPTR" ^ r.suffix) [];
      write d ("regIH_DOORBELL_RPTR" ^ r.suffix) [ ("enable", 0) ])
    t.ih;
  if version d D.osssys_hwip <> (4, 4, 2) then begin
    update d "regIH_STORM_CLIENT_LIST_CNTL" [ ("client18_is_storm_client", 1) ];
    update d "regIH_INT_FLOOD_CNTL" [ ("flood_cntl_enable", 1) ];
    update d "regIH_MSI_STORM_CTRL" [ ("delay", 3) ]
  end;
  List.iter
    (fun r ->
      update d
        ("regIH_RB_CNTL" ^ r.suffix)
        (("rb_enable", 1) :: (if r.id = 0 then [ ("enable_intr", 1) ] else [])))
    t.ih

let ih_drain t =
  let d = t.d in
  let wptr = fields d "regIH_RB_WPTR" in
  write d
    ~value:(List.assoc "offset" wptr mod (ih_bytes / 4))
    "regIH_RB_RPTR" [];
  if List.assoc "rb_overflow" wptr <> 0 then begin
    update d "regIH_RB_WPTR" [ ("rb_overflow", 0) ];
    update d "regIH_RB_CNTL" [ ("wptr_overflow_clear", 1) ];
    update d "regIH_RB_CNTL" [ ("wptr_overflow_clear", 0) ]
  end

(* The name of the interrupt source [src] of [client] on this GPU. *)
let source_name t client src =
  let major, _, _ = gcv t in
  let soc21 = major >= 11 in
  let gfx_clients =
    if soc21 then D.[ soc21_ih_clientid_grbm_cp; soc21_ih_clientid_gfx ]
    else D.soc15_ih_clientid_grbm_cp :: D.soc15_ih_clientid_se_sh
  in
  let sdma_clients = if soc21 then [] else D.soc15_ih_clientid_sdma in
  let prefix =
    if List.mem client gfx_clients then Some (Printf.sprintf "GFX_%d" major)
    else if List.mem client sdma_clients then
      let m, _, _ = ver t D.sdma0_hwip in
      Some (Printf.sprintf "SDMA0_%d" m)
    else None
  in
  match prefix with
  | None -> ""
  | Some p ->
      List.fold_left
        (fun acc (block, v, name) ->
          if v = src && String.starts_with ~prefix:p block then name else acc)
        "" D.ih_sources

let client_name t client =
  let major, _, _ = gcv t in
  let names = if major >= 11 then D.soc21_ih_clients else D.soc15_ih_clients in
  Option.value ~default:(string_of_int client) (List.assoc_opt client names)

(* Reads the interrupt ring up to the hardware's write pointer. An error
   interrupt, a page fault, or a fatal hardware error records its report in
   [t.d.errors]: the device is then in an unknown state. *)
let interrupts t =
  let d = t.d in
  let words = ih_bytes / 4 in
  let wptr = field d "regIH_RB_WPTR" "offset" in
  let rptr = ref (read d "regIH_RB_RPTR") in
  let report s = d.errors <- d.errors @ [ Printf.sprintf "%s: %s" d.bus s ] in
  while !rptr <> wptr do
    let e =
      Array.init 8 (fun i -> Mmio.get32 t.ih_view (4 * ((!rptr + i) mod words)))
    in
    rptr := (!rptr + 8) mod words;
    let client = e.(0) land 0xff and src = (e.(0) lsr 8) land 0xff in
    let ring = (e.(0) lsr 16) land 0xff and vmid = (e.(0) lsr 24) land 0xf in
    let vmid_type = (e.(0) lsr 31) land 1 in
    let pasid = e.(3) land 0xffff and node = (e.(3) lsr 16) land 0xff in
    let ctx = Array.sub e 4 4 in
    let name = source_name t client src in
    if name <> "SDMA_TRAP" && name <> "CP_EOP_INTR" then begin
      let line =
        Printf.sprintf
          "interrupt client=%s src=%s(%d) ring=%d vmid=%d(%d) pasid=%d node=%d \
           ctx=[0x%x, 0x%x, 0x%x, 0x%x]"
          (client_name t client) name src ring vmid vmid_type pasid node ctx.(0)
          ctx.(1) ctx.(2) ctx.(3)
      in
      let major, _, _ = gcv t in
      if name = "SQ_INTERRUPT_ID" then begin
        let soc21 = major >= 11 in
        let enc =
          if soc21 then getbits ctx.(1) 6 7 else getbits ctx.(0) 26 27
        in
        let err =
          if soc21 then getbits ctx.(0) 21 24
          else
            getbits
              (ctx.(0) land 0xfff
              lor ((ctx.(0) lsr 16) land 0xf000)
              lor ((ctx.(1) lsl 16) land 0xff0000))
              20 23
        in
        if enc = 2 then
          report
            (Printf.sprintf "%s: shader error %s" line
               (match
                  List.nth_opt
                    [ "EDC_FUE"; "ILLEGAL_INST"; "MEMVIOL"; "EDC_FED" ]
                    err
                with
               | Some e -> e
               | None -> string_of_int err))
      end
      else if
        name = "UTCL2_FAULT" || (major = 9 && client = D.soc15_ih_clientid_utcl2)
      then begin
        let status = fields d (pf_status_reg d "GC") in
        let va =
          (read d "regGCVM_L2_PROTECTION_FAULT_ADDR_HI32" lsl 32)
          lor read d "regGCVM_L2_PROTECTION_FAULT_ADDR_LO32"
        in
        update d "regGCVM_L2_PROTECTION_FAULT_CNTL"
          [ ("clear_protection_fault_status_addr", 1) ];
        report
          (Printf.sprintf "%s: page fault at 0x%x (%s)" line (va lsl 12)
             (String.concat ", "
                (List.filter_map
                   (fun (f, v) ->
                     if v <> 0 then Some (Printf.sprintf "%s=%d" f v) else None)
                   status)))
      end
      else report line
    end
  done;
  ih_drain t;
  if not d.is_vf then begin
    let athub =
      field d "regBIF_BX0_BIF_DOORBELL_INT_CNTL"
        "ras_athub_err_event_interrupt_status"
    and cntlr =
      field d "regBIF_BX0_BIF_DOORBELL_INT_CNTL" "ras_cntlr_interrupt_status"
    in
    if athub <> 0 || cntlr <> 0 then begin
      let banks =
        aca_banks t ~uncorrectable:true @ aca_banks t ~uncorrectable:false
      in
      report
        (Printf.sprintf "fatal hardware error%s%s; machine-check banks: %s"
           (if athub <> 0 then " RAS_ATHUB_ERR_EVENT" else "")
           (if cntlr <> 0 then " RAS_CNTLR" else "")
           (String.concat "; "
              (List.map
                 (fun regs ->
                   String.concat " " (List.map (Printf.sprintf "0x%x") regs))
                 banks)));
      write d "regBIF_BX0_BIF_DOORBELL_INT_CNTL"
        [
          ("ras_cntlr_interrupt_clear", cntlr);
          ("ras_athub_err_event_interrupt_clear", athub);
        ]
    end
  end

(* SDMA *)

let sdma_init t =
  let d = t.d in
  let v = ver t D.sdma0_hwip in
  let pipes = if lt v (5, 0, 0) then 16 else 1 in
  for p = 0 to pipes - 1 do
    let pipe, inst = if lt v (5, 0, 0) then ("", p) else (string_of_int p, 0) in
    let r s = Printf.sprintf "regSDMA%s_%s" pipe s in
    if ge v (6, 0, 0) then begin
      update d ~inst (r "WATCHDOG_CNTL") [ ("queue_hang_count", 100) ];
      update d ~inst (r "UTCL1_CNTL") [ ("resp_mode", 3); ("redo_delay", 9) ];
      update d ~inst (r "UTCL1_PAGE")
        ([ ("rd_l2_policy", 2); ("wr_l2_policy", 3) ]
        @ if t.sdma_name = "F32" then [ ("llc_noalloc", 1) ] else []);
      update d ~inst
        (r (t.sdma_name ^ "_CNTL"))
        [
          ("halt", 0);
          ((if t.sdma_name = "F32" then "th1_reset" else "reset"), 0);
        ]
    end;
    update d ~inst (r "CNTL")
      (("trap_enable", 1)
      :: (if compare v (5, 2, 0) <= 0 then [ ("utc_l1_enable", 1) ] else []))
  done;
  if is_nbio_79 d then
    List.iter
      (fun aid ->
        List.iteri
          (fun dev (port, awid, offset, awaddr) ->
            let entry = dev + 1 + (4 * aid) in
            write d
              (Printf.sprintf "regDOORBELL0_CTRL_ENTRY_%d" entry)
              [
                (Printf.sprintf "bif_doorbell%d_range_size_entry" entry, 20);
                ( Printf.sprintf "bif_doorbell%d_range_offset_entry" entry,
                  (D.amdgpu_navi10_doorbell_sdma_engine0 + ((entry - 1) * 0xA))
                  * 2 );
              ];
            doorbell_enable d ~port ~awid ~awaddr ~offset ~size:4 ~aid ())
          [
            (1, 0xe, 0xe, 0x1);
            (2, 0x8, 0x8, 0x2);
            (5, 0x9, 0x9, 0x8);
            (6, 0xa, 0xa, 0x9);
          ])
      d.aids
  else
    doorbell_enable d ~port:2 ~awid:0xe ~awaddr:0x3
      ~offset:(D.amdgpu_navi10_doorbell_sdma_engine0 * 2)
      ~size:4 ()

let sdma_halt t =
  if ge (ver t D.sdma0_hwip) (6, 0, 0) then
    update t.d (Printf.sprintf "regSDMA0_%s_CNTL" t.sdma_name) [ ("halt", 1) ]

let sdma_fini t =
  let d = t.d in
  List.iter
    (fun (r, inst) ->
      update d ~inst (r ^ "_RB_CNTL") [ ("rb_enable", 0) ];
      update d ~inst (r ^ "_IB_CNTL") [ ("ib_enable", 0) ];
      update d ~inst (r ^ "_DOORBELL") [ ("enable", 0) ];
      update d ~inst (r ^ "_DOORBELL_OFFSET") [ ("offset", 0) ])
    t.sdma_regs;
  if ge (ver t D.sdma0_hwip) (6, 0, 0) then begin
    write d "regGRBM_SOFT_RESET" [ ("soft_reset_sdma0", 1) ];
    sleep_ms 10;
    write d ~value:0 "regGRBM_SOFT_RESET" []
  end

(* Sets up SDMA queue [idx] on [ring]; returns its doorbell index. *)
let setup_sdma_ring t ~ring ~ring_bytes ~rptr ~wptr ~idx =
  let d = t.d in
  let v = ver t D.sdma0_hwip in
  if ge v (5, 0, 0) && idx > 0 then
    failwith (Printf.sprintf "%s: SDMA queue %d is not available" d.bus idx);
  let pipe = idx / 4 and queue = idx mod 4 in
  let r, inst =
    match v with
    | 4, 4, _ -> ("regSDMA_GFX", pipe + (queue * 4))
    | _ -> (Printf.sprintf "regSDMA%d_QUEUE%d" pipe queue, 0)
  in
  let doorbell =
    D.amdgpu_navi10_doorbell_sdma_engine0 + ((pipe + (queue * 4)) * 0xA)
  in
  t.sdma_regs <- t.sdma_regs @ [ (r, inst) ];
  write d ~inst ~value:1 (r ^ "_MINOR_PTR_UPDATE") [];
  write_pair d ~inst (r ^ "_RB_RPTR") ~lo:"" ~hi:"_HI" 0;
  write_pair d ~inst (r ^ "_RB_WPTR") ~lo:"" ~hi:"_HI" 0;
  write_pair d ~inst (r ^ "_RB_BASE") ~lo:"" ~hi:"_HI" (ring lsr 8);
  write_pair d ~inst (r ^ "_RB_RPTR_ADDR") ~lo:"_LO" ~hi:"_HI" rptr;
  write_pair d ~inst (r ^ "_RB_WPTR_POLL_ADDR") ~lo:"_LO" ~hi:"_HI" wptr;
  update d ~inst (r ^ "_DOORBELL_OFFSET") [ ("offset", doorbell * 2) ];
  update d ~inst (r ^ "_DOORBELL") [ ("enable", 1) ];
  write d ~inst ~value:0 (r ^ "_MINOR_PTR_UPDATE") [];
  write d ~inst (r ^ "_RB_CNTL")
    ((match v with
       | 4, 4, _ -> []
       | _ -> [ (String.lowercase_ascii t.sdma_name ^ "_wptr_poll_enable", 1) ])
    @ [
        ("rb_vmid", 0);
        ("rptr_writeback_enable", 1);
        ("rptr_writeback_timer", 4);
        ("rb_enable", 1);
        ("rb_priv", 1);
        ("rb_size", bit_length (ring_bytes / 4) - 1);
      ]);
  update d ~inst (r ^ "_IB_CNTL") [ ("ib_enable", 1) ];
  doorbell

(* Boot *)

let disable_aspm pci =
  (* L1 across retimers makes reads oscillate to 0xffffffff; clearing the GPU's
     end is enough, since L1 needs both ends. The walk is bounded: a dead link
     can return 0xff pointers forever. *)
  let rec walk cap seen =
    if cap = 0 || List.mem cap seen then None
    else if Pci.read_config pci cap 1 = 0x10 then Some cap
    else walk (Pci.read_config pci (cap + 1) 1 land 0xfc) (cap :: seen)
  in
  match walk (Pci.read_config pci 0x34 1 land 0xfc) [] with
  | Some cap ->
      Pci.write_config pci (cap + 0x10) 2
        (Pci.read_config pci (cap + 0x10) 2 land lnot 3)
  | None -> ()

let pci_command = 0x04
let pci_command_master = 0x4

let set_bus_master pci on =
  let c = Pci.read_config pci pci_command 2 in
  Pci.write_config pci pci_command 2
    (if on then c lor pci_command_master else c land lnot pci_command_master)

(* The GPU's registers and discovered layout, before any block is touched. *)
let open_hw pci =
  let bus = Pci.bus pci in
  disable_aspm pci;
  let vram = Pci.map_bar pci 0 and doorbells = Pci.map_bar pci 2 in
  let mmio = Pci.map_bar pci 5 in
  let is_vf = Mmio.get32 mmio (D.mmrcc_iov_func_identifier * 4) land 1 = 1 in
  let vf_access =
    if is_vf then vf_request mmio D.idh_req_gpu_init_access else 0
  in
  let vram_size = Mmio.get32 mmio (0xde3 * 4) lsl 20 in
  let large_bar = Mmio.length vram >= vram_size in
  (* The discovery table sits 64 KiB below the end of VRAM. *)
  let at = vram_size - (64 lsl 10) and n = 10 lsl 10 in
  let tbl = if large_bar then Mmio.read vram at n else read_vram mmio at n in
  let disc = parse_discovery tbl in
  let regs, rlcg = build_regs ~ip_ver:disc.ip_ver ~bases:disc.bases ~is_vf in
  let insts hwip = Option.value ~default:[] (List.assoc_opt hwip disc.bases) in
  let harvested hwip =
    Option.value ~default:[] (List.assoc_opt hwip disc.harvested)
  in
  let gc_ver = ip_version disc.ip_ver D.gc_hwip in
  (* Live AIDs, as the kernel counts them: four SDMAs each, alive iff the
     group's mask is 0xf, 0x3 or 0xc. Dead AIDs must never be touched through
     the indirect window: a write poisons the whole fabric. *)
  let sdma = insts D.sdma0_hwip in
  let live =
    List.filter (fun (k, _) -> not (List.mem k (harvested D.sdma0_hwip))) sdma
  in
  let max_aid = List.fold_left (fun m (k, _) -> Int.max m (k lsr 2)) 0 sdma in
  let aids =
    0
    :: List.filter
         (fun aid ->
           let mask =
             List.fold_left
               (fun m (i, _) ->
                 if i lsr 2 = aid then m + (1 lsl (i land 3)) else m)
               0 live
           in
           List.mem mask [ 0xf; 0x3; 0xc ])
         (List.init max_aid (fun i -> i + 1))
  in
  let xccs =
    List.length
      (List.filter
         (fun (i, _) -> not (List.mem i (harvested D.gc_hwip)))
         (insts D.gc_hwip))
  in
  let nbio79 =
    List.mem (ip_version disc.ip_ver D.nbio_hwip) [ (7, 9, 0); (7, 9, 1) ]
  in
  let d =
    {
      pci;
      bus;
      vram;
      doorbells;
      mmio;
      vram_size;
      large_bar;
      is_vf;
      vf_access;
      rlcg;
      ip_ver = disc.ip_ver;
      bases = disc.bases;
      harvested = disc.harvested;
      gc_info = disc.gc_info;
      reserved_vram =
        (match gc_ver with 9, (4 | 5), _ -> 384 lsl 20 | _ -> 64 lsl 20);
      regs;
      aids;
      xccs;
      xgmi_phys_id = 0;
      xgmi_max_region = 0;
      xgmi_seg_sz = 0;
      paddr_base = 0;
      fb_base = 0;
      fb_end = 0;
      mc_base = 0;
      address_mask =
        (1 lsl match gc_ver with 9, (4 | 5), _ -> 48 | _ -> 44) - 1;
      mm_insts = (if nbio79 then aids else List.map fst (insts D.mmhub_hwip));
      gc_hub = false;
      kiq = None;
      errors = [];
    }
  in
  (* The memory controller's apertures. *)
  let lfb = has_reg d "regGCMC_VM_XGMI_LFB_CNTL" in
  let xgmi_phys_id =
    if lfb then field d "regGCMC_VM_XGMI_LFB_CNTL" "pf_lfb_region" else 0
  in
  let xgmi_max_region =
    if lfb then field d "regGCMC_VM_XGMI_LFB_CNTL" "pf_max_region" else 0
  in
  let xgmi_seg_sz =
    if lfb then field d "regGCMC_VM_XGMI_LFB_SIZE" "pf_lfb_size" lsl 24 else 0
  in
  let paddr_base = xgmi_phys_id * xgmi_seg_sz in
  let fb_base = (read d "regMMMC_VM_FB_LOCATION_BASE" land 0xFFFFFF) lsl 24 in
  let fb_end = (read d "regMMMC_VM_FB_LOCATION_TOP" land 0xFFFFFF) lsl 24 in
  {
    d with
    xgmi_phys_id;
    xgmi_max_region;
    xgmi_seg_sz;
    paddr_base;
    fb_base;
    fb_end;
    mc_base = fb_base + paddr_base;
  }

(* The PCI functions of the GPUs this driver boots, by device id. *)
let buses ?remote () =
  Pci.scan ?remote ~vendor:0x1002
    [
      ( 0xffff,
        [
          0x74a1;
          0x74b5;
          0x744c;
          0x7480;
          0x7550;
          0x7551;
          0x7590;
          0x75a0;
          0x75a8;
          0x75b0;
          0x75b3;
        ] );
    ]

(* Whether [d] boots partially: this runtime left it cleanly. Full boots over
   live state can kill the fabric of GC 9.5.0; a partial boot with a MEC reset
   is the deepest safe reset there. *)
let partial_boot d =
  let marked = read d "regSCRATCH_REG7" = session in
  (marked && gc d = (9, 5, 0))
  || marked
     && read d "regSCRATCH_REG6" = 0
     && read d (pf_status_reg d "GC") = 0

(* The physical blocks the main pool hands out, and their alignments, largest
   first. *)
let main_blocks =
  List.init 28 (fun k ->
      let i = 27 - k in
      (1 lsl (i + 12), if i >= 9 then 2 lsl 20 else 0x1000))

(* Boots the GPU of [pci]. A GPU this runtime left cleanly boots partially: only
   its GFX and SDMA blocks start again, from the boot memory the last session
   left. Any other GPU is reset, then fully booted. *)
let boot ?firmware pci =
  let d = open_hw pci in
  let flush () =
    flush_tlb d "GC" ~vmid:0;
    flush_tlb d "MM" ~vmid:0
  in
  let mm =
    Page_table.create (entry d ~flush) space
      ~memory:(d.vram_size - d.reserved_vram)
      ~boot:(3 lsl 20) ~tables:(not d.large_bar) ~pages:main_blocks
  in
  let fw = load_firmware ?dir:firmware d in
  let boot_alloc ?align ?(zero = false) n =
    palloc ?align ~zero ~boot:true mm n
  in
  let memscratch = paddr2xgmi d (boot_alloc 0x1000) in
  let dummy_page = paddr2xgmi d (boot_alloc 0x1000) in
  let ih =
    List.map
      (fun (suffix, id) ->
        let ring = boot_alloc ih_bytes in
        let wptr = boot_alloc 0x1000 in
        { ring; wptr; suffix; id })
      [ ("", 0); ("_RING1", 1) ]
  in
  let mp0 = version d D.mp0_hwip in
  let msg1_pa = boot_alloc ~align:D.psp_1_meg D.psp_1_meg in
  let cmd = boot_alloc D.psp_cmd_buffer_size in
  let fence = boot_alloc ~zero:true D.psp_fence_buffer_size in
  let psp_ring = boot_alloc 0x10000 in
  let driver_table = boot_alloc 0x4000 in
  let mqd =
    Array.init (2 + Bool.to_int d.is_vf) (fun _ -> boot_alloc (0x1000 * d.xccs))
  in
  let partial = partial_boot d in
  (* A partial boot that fails midway must not be trusted by the next open. *)
  if partial && not d.is_vf then write d ~value:1 "regSCRATCH_REG6" [];
  let t =
    {
      d;
      mm;
      fw;
      partial;
      memscratch;
      dummy_page;
      ih;
      ih_view = Mmio.sub d.vram (List.hd ih).ring ih_bytes;
      psp =
        (if lt mp0 (14, 0, 0) then "regMP0_SMN_C2PMSG"
         else "regMPASP_SMN_C2PMSG");
      msg1 = Mmio.sub d.vram msg1_pa D.psp_1_meg;
      msg1_addr = paddr2mc d msg1_pa;
      cmd;
      fence;
      psp_ring;
      tmr_size = 0;
      tmr = 0;
      boot_time_tmr =
        List.mem mp0 [ (13, 0, 6); (13, 0, 14); (14, 0, 2); (14, 0, 3) ];
      autoload_tmr = not (List.mem mp0 [ (13, 0, 6); (13, 0, 14) ]);
      smu = Am_reg.family "smu" (version d D.mp1_hwip);
      driver_table;
      clocks = Hashtbl.create 4;
      mqd;
      sdma_regs = [];
      sdma_name =
        (if lt (version d D.sdma0_hwip) (7, 0, 0) then "F32" else "MCU");
    }
  in
  if not partial then begin
    if (not d.is_vf) && sos_alive t && smu_alive t then begin
      set_bus_master pci false;
      if is_hive d then
        failwith
          (d.bus
         ^ " is in a hive left in an unknown state; reset the hive before \
            opening it");
      (* Quiesce before the reset: a mode1 reset over live engines at full
         clocks can wedge the GPU until it is power cycled. *)
      dequeue_hqds d;
      set_clocks t 0;
      halt_mec d;
      sdma_halt t;
      sleep_ms 100;
      mode1_reset t
    end;
    set_bus_master pci true;
    soc_init d;
    init_hub t "MM" d.mm_insts;
    ih_init t;
    if not d.is_vf then begin
      psp_init t;
      smu_init t
    end
  end
  else begin
    (* The GPU keeps its state across a partial boot, but a server that stopped
       its DMA when its last client left turned bus mastering off. *)
    set_bus_master pci true;
    if not d.is_vf then tmr_init t
  end;
  Page_table.booted mm;
  gfx_init t;
  sdma_init t;
  if not d.is_vf then begin
    set_clocks t (-1);
    soc_clockgating d;
    gfx_clockgating d;
    write d ~value:t.tmr_size "regSCRATCH_REG5" [];
    write d ~value:session "regSCRATCH_REG7" [];
    write d ~value:1 "regSCRATCH_REG6" []
  end;
  t

(* Leaves the GPU stopped and marked for the next boot: clean for a partial one,
   or dirty for a full one if [failed] or a fault was reported. A GPU that
   failed, or whose engines could not be stopped, also loses bus mastering: the
   process's memory it may still reach is about to be released. The waits are
   bounded, and a failed GPU's queues are not waited for. *)
let fini t ~failed =
  let d = t.d in
  let stop () =
    (if d.is_vf && d.vf_access = 0 then
       try d.vf_access <- vf_request d.mmio D.idh_req_gpu_fini_access
       with Failure _ -> ());
    sdma_fini t;
    dequeue_hqds ~wait:(not failed) d;
    if not d.is_vf then set_clocks t 0;
    interrupts t;
    if not d.is_vf then
      write d
        ~value:(if failed || d.errors <> [] then 1 else 0)
        "regSCRATCH_REG6" [];
    if d.vf_access <> 0 then release_vf_access d
  in
  match stop () with
  | () -> if failed then set_bus_master d.pci false
  | exception e ->
      set_bus_master d.pci false;
      raise e

(* Waits at most [ms] for an interrupt, then reads the interrupt ring; raises
   the reports of a device that faulted. *)
let sleep t ms =
  ignore (Pci.wait_interrupt t.d.pci ms);
  interrupts t;
  if t.d.errors <> [] then failwith (String.concat "; " t.d.errors)
