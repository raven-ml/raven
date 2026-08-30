(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

module Am = Amd_tables.Am_defs
module Am_register = Amdev.Am_register
module Firmware = Amdev.Firmware
module Helpers = Tolk.Helpers
module Memory = Tolk.Memory
module Mmio = Tolk_hcq.Hcq.Mmio

let debug = Helpers.getenv "DEBUG" 0

exception Timeout_error of string

(* helpers.py:538 wait_cond, over the device clock so tests can script
   the passage of time. *)
let wait_cond adev ?(timeout_ms = 10000) ~value ~msg cb =
  let start = Amdev.now_ms adev in
  let rec go last =
    if Amdev.now_ms adev - start < timeout_ms then begin
      let v = cb () in
      if v = value then () else go v
    end
    else
      raise
        (Timeout_error
           (Printf.sprintf
              "%s. Timed out after %d ms, condition not met: %d != %d" msg
              timeout_ms last value))
  in
  go 0

(* time.sleep equivalents in the boot protocols only need elapsed wall
   time, so waiting on the device clock keeps them scriptable too. *)
let sleep_ms adev ms =
  let start = Amdev.now_ms adev in
  while Amdev.now_ms adev - start < ms do
    ()
  done

let lo32 v = v land 0xffffffff
let hi32 v = v lsr 32

(* int.bit_length: the number of bits needed to represent [n]. *)
let bit_length n =
  let rec go acc n = if n = 0 then acc else go (acc + 1) (n lsr 1) in
  go 0 n

(* helpers.py:82 getbits *)
let getbits v lo hi = (v lsr lo) land ((1 lsl (hi - lo + 1)) - 1)

(* Soc: ip.py AM_SOC *)

module Soc = struct
  type t = {
    adev : Amdev.t;
    mtype_uc : int;
    sh_mem_address_mode_64 : int;
    sh_mem_alignment_mode_unaligned : int;
    ih_clients : (int * string) list;
    ih_srcs_names : (int * (int * string) list) list;
  }

  (* ip.py:16 AM_SOC.init_sw *)
  let create adev =
    let gc_major, _, _ = Amdev.ip_ver adev Am.gc_hwip in
    let sdma_major, _, _ = Amdev.ip_ver adev Am.sdma0_hwip in
    let mtype_uc, address_mode_64, alignment_mode_unaligned =
      match gc_major with
      | 9 ->
          Amd_soc_defs.Soc_9.
            (mtype_uc, sh_mem_address_mode_64, sh_mem_alignment_mode_unaligned)
      | 11 ->
          Amd_soc_defs.Soc_11.
            (mtype_uc, sh_mem_address_mode_64, sh_mem_alignment_mode_unaligned)
      | 12 ->
          Amd_soc_defs.Soc_12.
            (mtype_uc, sh_mem_address_mode_64, sh_mem_alignment_mode_unaligned)
      | maj ->
          invalid_arg (Printf.sprintf "no soc constants for gfx%d" maj)
    in
    let ih_soc21 = gc_major >= 11 in
    let gfx_ih_clients =
      if ih_soc21 then
        [ Am.soc21_ih_clientid_grbm_cp; Am.soc21_ih_clientid_gfx ]
      else
        [
          Am.soc15_ih_clientid_grbm_cp; Am.soc15_ih_clientid_se0sh;
          Am.soc15_ih_clientid_se1sh; Am.soc15_ih_clientid_se2sh;
          Am.soc15_ih_clientid_se3sh;
        ]
    in
    let sdma_ih_clients =
      if ih_soc21 then []
      else
        [
          Am.soc15_ih_clientid_sdma0; Am.soc15_ih_clientid_sdma1;
          Am.soc15_ih_clientid_sdma2; Am.soc15_ih_clientid_sdma3;
          Am.soc15_ih_clientid_sdma4; Am.soc15_ih_clientid_sdma5;
          Am.soc15_ih_clientid_sdma6; Am.soc15_ih_clientid_sdma7;
        ]
    in
    let gfx_srcs =
      match gc_major with
      | 9 -> Am.gfx_9_srcids
      | 11 -> Am.gfx_11_srcids
      | 12 -> Am.gfx_12_srcids
      | _ -> []
    in
    let sdma_srcs =
      match sdma_major with
      | 4 -> Am.sdma0_4_srcids
      | 5 -> Am.sdma0_5_srcids
      | _ -> []
    in
    {
      adev;
      mtype_uc;
      sh_mem_address_mode_64 = address_mode_64;
      sh_mem_alignment_mode_unaligned = alignment_mode_unaligned;
      ih_clients =
        (if ih_soc21 then Am.soc21_ih_clientid_names
         else Am.soc15_ih_clientid_names);
      ih_srcs_names =
        List.map (fun c -> (c, gfx_srcs)) gfx_ih_clients
        @ List.map (fun c -> (c, sdma_srcs)) sdma_ih_clients;
    }

  let mtype_uc t = t.mtype_uc
  let sh_mem_address_mode_64 t = t.sh_mem_address_mode_64
  let sh_mem_alignment_mode_unaligned t = t.sh_mem_alignment_mode_unaligned
  let ih_client_name t client = List.assoc_opt client t.ih_clients

  let ih_src_name t ~client ~src =
    match List.assoc_opt client t.ih_srcs_names with
    | None -> ""
    | Some srcs -> (
        match List.assoc_opt src srcs with Some name -> name | None -> "")

  (* ip.py:30 AM_SOC.init_hw *)
  let init_hw t ~vmhubs =
    let adev = t.adev in
    (if List.mem (Amdev.ip_ver adev Am.nbio_hwip) [ (7, 9, 0); (7, 9, 1) ]
     then begin
       let fence = Amdev.reg adev "regXCC_DOORBELL_FENCE" in
       Am_register.write fence ~value:0x0 [];
       for aid = 1 to vmhubs - 1 do
         Amdev.indirect_wreg_pcie adev ~aid
           (Am_register.reg fence).Amd_tables.Reg.addr
           (Amd_tables.Reg.encode (Am_register.reg fence)
              [ ("shub_slv_mode", 1) ])
       done;
       Am_register.write
         (Amdev.reg adev "regBIFC_GFX_INT_MONITOR_MASK")
         ~value:0x7ff [];
       Am_register.write
         (Amdev.reg adev "regBIFC_DOORBELL_ACCESS_EN_PF")
         ~value:0xfffff []
     end
     else
       Am_register.update
         (Amdev.reg adev "regRCC_DEV0_EPF2_STRAP2")
         [ ("strap_no_soft_reset_dev0_f2", 0x0) ]);
    Am_register.write
      (Amdev.reg adev "regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN")
      ~value:0x1 []

  (* ip.py:39 AM_SOC.set_clockgating_state *)
  let set_clockgating_state t =
    if Amdev.ip_ver t.adev Am.hdp_hwip >= (5, 2, 1) then
      Am_register.update
        (Amdev.reg t.adev "regHDP_MEM_POWER_CTRL")
        [ ("atomic_mem_power_ctrl_en", 1); ("atomic_mem_power_ds_en", 1) ]

  (* ip.py:42 AM_SOC.doorbell_enable *)
  let doorbell_enable t ~port ?(awid = 0) ?(awaddr_31_28_value = 0)
      ?(offset = 0) ?(size = 0) ?(aid = 0) () =
    let adev = t.adev in
    let pref =
      if Amdev.ip_ver adev Am.gc_hwip >= (12, 0, 0) then "regGDC_S2A0_S2A"
      else "regS2A"
    in
    let reg =
      Amdev.reg adev (Printf.sprintf "%s_DOORBELL_ENTRY_%d_CTRL" pref port)
    in
    let field name = Printf.sprintf "s2a_doorbell_port%d_%s" port name in
    let value =
      Amd_tables.Reg.encode (Am_register.reg reg)
        [
          (field "enable", 1); (field "awid", awid);
          (field "range_size", size);
          (field "awaddr_31_28_value", awaddr_31_28_value);
          (field "range_offset", offset);
        ]
    in
    if List.mem (Amdev.ip_ver adev Am.nbio_hwip) [ (7, 9, 0); (7, 9, 1) ]
    then
      Amdev.indirect_wreg_pcie adev ~aid
        (Am_register.reg reg).Amd_tables.Reg.addr value
    else Am_register.write reg ~value []
end

(* Gmc: ip.py AM_GMC *)

module Gmc = struct
  type hub = Mm | Gc

  type t = {
    adev : Amdev.t;
    vmhubs : int;
    fb_base : int;
    fb_end : int;
    vm_base : int;
    vm_end : int;
    trans_futher : bool;
    memscratch_xgmi_paddr : int;
    dummy_page_xgmi_paddr : int;
    mutable mm_hub_initted : bool;
    mutable gc_hub_initted : bool;
  }

  let hub_pref = function Mm -> "MM" | Gc -> "GC"

  (* ip.py:51 AM_GMC.init_sw; the fabric position and entry encoding
     live in Amdev (address topology, Am_page_table). *)
  let create adev =
    let mm = Amdev.mm adev in
    let vmhubs =
      List.length
        (List.assoc Am.mmhub_hwip (Amdev.discovery adev).Amdev.regs_offset)
    in
    let fb_base =
      (Am_register.read (Amdev.reg adev "regMMMC_VM_FB_LOCATION_BASE")
      land 0xFFFFFF)
      lsl 24
    in
    let fb_end =
      (Am_register.read (Amdev.reg adev "regMMMC_VM_FB_LOCATION_TOP")
      land 0xFFFFFF)
      lsl 24
    in
    let vm_base = Memory.va_base mm in
    let vm_end =
      min (vm_base + (1 lsl Memory.va_bits mm) - 1) 0x7fffffffffff
    in
    let palloc_xgmi () =
      Amdev.paddr2xgmi adev (Memory.palloc mm 0x1000 ~zero:false ~boot:true ())
    in
    let memscratch_xgmi_paddr = palloc_xgmi () in
    let dummy_page_xgmi_paddr = palloc_xgmi () in
    {
      adev;
      vmhubs;
      fb_base;
      fb_end;
      vm_base;
      vm_end;
      trans_futher = Amdev.ip_ver adev Am.gc_hwip < (10, 0, 0);
      memscratch_xgmi_paddr;
      dummy_page_xgmi_paddr;
      (* The memory hub is programmed before any TLB flush and keeps
         its state across a partial boot, so it starts marked ready. *)
      mm_hub_initted = true;
      gc_hub_initted = false;
    }

  let vmhubs t = t.vmhubs
  let hub_initted t = function Mm -> t.mm_hub_initted | Gc -> t.gc_hub_initted

  (* ip.py:81 pf_status_reg *)
  let pf_status_reg t hub =
    Printf.sprintf "reg%sVM_L2_PROTECTION_FAULT_STATUS%s" (hub_pref hub)
      (if Amdev.ip_ver t.adev Am.gc_hwip >= (12, 0, 0) then "_LO32" else "")

  (* ip.py:85 flush_hdp *)
  let flush_hdp adev =
    Amdev.wreg adev
      (Am_register.read (Amdev.reg adev "regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL")
      / 4)
      0x0

  (* ip.py:86 flush_tlb *)
  let flush_tlb t ?(flush_type = 0) ~xccs hub ~vmid =
    let adev = t.adev in
    flush_hdp adev;
    (* Can't issue TLB invalidation if the hub isn't initialized. *)
    if hub_initted t hub then begin
      let pref = hub_pref hub in
      for _ = 1 to (match hub with Mm -> t.vmhubs | Gc -> xccs) do
        (if hub = Mm then
           wait_cond adev ~value:1 ~msg:"mm flush_tlb timeout" (fun () ->
               Am_register.read (Amdev.reg adev "regMMVM_INVALIDATE_ENG17_SEM")
               land 0x1));
        Am_register.write
          (Amdev.reg adev
             (Printf.sprintf "reg%sVM_INVALIDATE_ENG17_REQ" pref))
          [
            ("flush_type", flush_type);
            ("per_vmid_invalidate_req", 1 lsl vmid);
            ("invalidate_l2_ptes", 1); ("invalidate_l2_pde0", 1);
            ("invalidate_l2_pde1", 1); ("invalidate_l2_pde2", 1);
            ("invalidate_l1_ptes", 1);
            ("clear_protection_fault_status_addr", 0);
          ];
        wait_cond adev ~value:(1 lsl vmid) ~msg:"flush_tlb timeout"
          (fun () ->
            Am_register.read
              (Amdev.reg adev
                 (Printf.sprintf "reg%sVM_INVALIDATE_ENG17_ACK" pref))
            land (1 lsl vmid));
        if hub = Mm then
          Am_register.write
            (Amdev.reg adev "regMMVM_INVALIDATE_ENG17_SEM")
            ~value:0x0 [];
        if Amdev.ip_ver adev Am.gc_hwip >= (11, 0, 0) && hub = Mm then begin
          Am_register.update
            (Amdev.reg adev "regMMVM_L2_BANK_SELECT_RESERVED_CID2")
            [ ("reserved_cache_private_invalidation", 1) ];
          (* Read back the register to ensure the invalidation is
             complete *)
          ignore
            (Am_register.read
               (Amdev.reg adev "regMMVM_L2_BANK_SELECT_RESERVED_CID2"))
        end
      done
    end

  (* ip.py:107 enable_vm_addressing *)
  let enable_vm_addressing t page_table hub ~vmid =
    let adev = t.adev in
    let pref = hub_pref hub in
    let name fmt = Printf.sprintf fmt pref vmid in
    Amdev.wreg_pair adev
      (name "reg%sVM_CONTEXT%d_PAGE_TABLE_START_ADDR")
      ~lo:"_LO32" ~hi:"_HI32" (t.vm_base lsr 12);
    Amdev.wreg_pair adev
      (name "reg%sVM_CONTEXT%d_PAGE_TABLE_END_ADDR")
      ~lo:"_LO32" ~hi:"_HI32" (t.vm_end lsr 12);
    Amdev.wreg_pair adev
      (name "reg%sVM_CONTEXT%d_PAGE_TABLE_BASE_ADDR")
      ~lo:"_LO32" ~hi:"_HI32"
      (Amdev.paddr2xgmi adev (Amdev.Am_page_table.paddr page_table) lor 1);
    let fault_kinds =
      [ "pde0"; "dummy_page"; "range"; "valid"; "read"; "write"; "execute" ]
    in
    let fault_flags suffix = List.map (fun x -> (x ^ suffix, 1)) fault_kinds in
    Am_register.write
      (Amdev.reg adev (name "reg%sVM_CONTEXT%d_CNTL"))
      ~value:0x1800000
      (fault_flags "_protection_fault_enable_interrupt"
      @ fault_flags "_protection_fault_enable_default"
      @ [
          ("enable_context", 1);
          ( "page_table_depth",
            (if t.trans_futher then 2 else 3)
            - Amdev.Am_page_table.lv page_table );
          ("page_table_block_size", if t.trans_futher then 9 else 0);
        ])

  (* ip.py:117 init_hub. The register layer resolves die instance 0
     only, so the per-instance loop programs every instance through
     the same registers; exact on single-die parts. *)
  let init_hub t ~soc hub ~inst_cnt =
    let adev = t.adev in
    let pref = hub_pref hub in
    let r name = Amdev.reg adev (Printf.sprintf "reg%s%s" pref name) in
    let pair name = Printf.sprintf "reg%s%s" pref name in
    for _ = 1 to inst_cnt do
      (* Init system apertures *)
      Am_register.write (r "MC_VM_AGP_BASE") ~value:0 [];
      (* disable AGP *)
      Am_register.write (r "MC_VM_AGP_BOT") ~value:(0xffffffffffff lsr 24) [];
      Am_register.write (r "MC_VM_AGP_TOP") ~value:0 [];
      Am_register.write
        (r "MC_VM_SYSTEM_APERTURE_LOW_ADDR")
        ~value:(t.fb_base lsr 18) [];
      Am_register.write
        (r "MC_VM_SYSTEM_APERTURE_HIGH_ADDR")
        ~value:(t.fb_end lsr 18) [];
      Amdev.wreg_pair adev
        (pair "MC_VM_SYSTEM_APERTURE_DEFAULT_ADDR")
        ~lo:"_LSB" ~hi:"_MSB"
        (t.memscratch_xgmi_paddr lsr 12);
      Amdev.wreg_pair adev
        (pair "VM_L2_PROTECTION_FAULT_DEFAULT_ADDR")
        ~lo:"_LO32" ~hi:"_HI32"
        (t.dummy_page_xgmi_paddr lsr 12);
      Am_register.update
        (r "VM_L2_PROTECTION_FAULT_CNTL2")
        [ ("active_page_migration_pte_read_retry", 1) ];
      (* Init TLB and cache *)
      Am_register.update
        (r "MC_VM_MX_L1_TLB_CNTL")
        [
          ("enable_l1_tlb", 1); ("system_access_mode", 3);
          ("enable_advanced_driver_model", 1);
          ("system_aperture_unmapped_access", 0);
          ("mtype", Soc.mtype_uc soc);
        ];
      Am_register.update (r "VM_L2_CNTL")
        [
          ("enable_l2_cache", 1);
          ("enable_default_page_out_to_system_memory", 1);
          ("l2_pde0_cache_tag_generation_mode", 0);
          ("pde_fault_classification", 0);
          ("context1_identity_access_mode", 1);
          ("identity_mode_fragment_size", 0);
          ("enable_l2_fragment_processing", Bool.to_int t.trans_futher);
        ];
      Am_register.update (r "VM_L2_CNTL2")
        [ ("invalidate_all_l1_tlbs", 1); ("invalidate_l2_cache", 1) ];
      Am_register.write (r "VM_L2_CNTL3")
        [
          ("l2_cache_4k_associativity", 1); ("l2_cache_bigk_associativity", 1);
          ("bank_select", if t.trans_futher then 12 else 9);
          ("l2_cache_bigk_fragment_size", if t.trans_futher then 9 else 6);
        ];
      Am_register.write (r "VM_L2_CNTL4")
        [ ("l2_cache_4k_partition_count", 1) ];
      if Amdev.ip_ver adev Am.gc_hwip >= (10, 0, 0) then
        Am_register.write (r "VM_L2_CNTL5")
          [ ("walker_priority_client_id", 0x1ff) ];
      enable_vm_addressing t
        (Memory.root_page_table (Amdev.mm adev))
        hub ~vmid:0;
      (* Disable identity aperture *)
      Amdev.wreg_pair adev
        (pair "VM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR")
        ~lo:"_LO32" ~hi:"_HI32" 0xfffffffff;
      Amdev.wreg_pair adev
        (pair "VM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR")
        ~lo:"_LO32" ~hi:"_HI32" 0x0;
      Amdev.wreg_pair adev
        (pair "VM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET")
        ~lo:"_LO32" ~hi:"_HI32" 0x0;
      for eng_i = 0 to 17 do
        Amdev.wreg_pair adev
          (Printf.sprintf "reg%sVM_INVALIDATE_ENG%d_ADDR_RANGE" pref eng_i)
          ~lo:"_LO32" ~hi:"_HI32" 0x1fffffffff
      done
    done;
    match hub with
    | Mm -> t.mm_hub_initted <- true
    | Gc -> t.gc_hub_initted <- true

  (* ip.py:83 AM_GMC.init_hw *)
  let init_hw t ~soc = init_hub t ~soc Mm ~inst_cnt:t.vmhubs
end

(* Smu: ip.py AM_SMU *)

module Smu = struct
  type t = {
    adev : Amdev.t;
    smu_mod : (module Amd_tables.Smu);
    driver_table_paddr : int;
    mutable clocks_cache : (int list * (int * int list) list) list;
  }

  (* ip.py:175 AM_SMU.init_sw *)
  let create adev =
    {
      adev;
      smu_mod = Amd_tables.smu ~version:(Amdev.ip_ver adev Am.mp1_hwip);
      driver_table_paddr =
        Memory.palloc (Amdev.mm adev) 0x4000 ~zero:false ~boot:true ();
      clocks_cache = [];
    }

  let driver_table_paddr t = t.driver_table_paddr

  let require name = function
    | Some v -> v
    | None ->
        invalid_arg
          (Printf.sprintf "smu interface does not provide %s" name)

  let mp1_reg t n =
    Amdev.reg t.adev (Printf.sprintf "mmMP1_SMN_C2PMSG_%d" n)

  (* ip.py:236 _smu_cmn_send_msg *)
  let smu_cmn_send_msg t ~debug:dbg msg param =
    Am_register.write (mp1_reg t (if dbg then 54 else 90)) ~value:0 [];
    Am_register.write (mp1_reg t (if dbg then 53 else 82)) ~value:param [];
    Am_register.write (mp1_reg t (if dbg then 75 else 66)) ~value:msg []

  (* ip.py:241 _send_msg; default timeout is 10 seconds *)
  let send_msg ?(timeout_ms = 10000) ?(debug = false) t msg param =
    smu_cmn_send_msg t ~debug msg param;
    wait_cond t.adev ~timeout_ms ~value:1
      ~msg:(Printf.sprintf "SMU msg 0x%x timeout" msg)
      (fun () -> Am_register.read (mp1_reg t (if debug then 54 else 90)))

  let send_msg_read ?timeout_ms ?(debug = false) t msg param =
    send_msg ?timeout_ms ~debug t msg param;
    Am_register.read (mp1_reg t (if debug then 53 else 82))

  (* ip.py:179 AM_SMU.init_hw *)
  let init_hw t =
    let module M = (val t.smu_mod) in
    let dt_mc = Amdev.paddr2mc t.adev t.driver_table_paddr in
    send_msg t M.ppsmc_msg_setdriverdramaddrhigh (hi32 dt_mc);
    send_msg t M.ppsmc_msg_setdriverdramaddrlow (lo32 dt_mc);
    send_msg t M.ppsmc_msg_enableallsmufeatures 0

  (* ip.py:184 is_smu_alive *)
  let is_smu_alive t =
    let module M = (val t.smu_mod) in
    (try send_msg ~timeout_ms:100 t M.ppsmc_msg_getsmuversion 0
     with Timeout_error _ -> ());
    Am_register.read (mp1_reg t 90) <> 0

  (* ip.py:188 mode1_reset *)
  let mode1_reset t =
    if debug >= 2 then
      Printf.printf "am %s: mode1 reset\n%!" (Amdev.devfmt t.adev);
    let module M = (val t.smu_mod) in
    let mp0 = Amdev.ip_ver t.adev Am.mp0_hwip in
    (if mp0 >= (14, 0, 0) || List.mem mp0 [ (13, 0, 0); (13, 0, 7); (13, 0, 10) ]
     then
       let debugsmc_msg_mode1reset = 2 in
       send_msg ~debug:true t debugsmc_msg_mode1reset 0
     else if List.mem mp0 [ (13, 0, 6); (13, 0, 12) ] then
       send_msg t
         (require "PPSMC_MSG_GfxDriverReset" M.ppsmc_msg_gfxdriverreset)
         1
     else
       send_msg t (require "PPSMC_MSG_Mode1Reset" M.ppsmc_msg_mode1reset) 0);
    if not (Amdev.is_hive t.adev) then sleep_ms t.adev 500 (* 500ms *)

  (* ip.py:197 read_table *)
  let read_table t ~size arg =
    let module M = (val t.smu_mod) in
    (if
       List.mem (Amdev.ip_ver t.adev Am.mp0_hwip) [ (13, 0, 6); (13, 0, 12) ]
     then
       send_msg t
         (require "PPSMC_MSG_GetMetricsTable" M.ppsmc_msg_getmetricstable)
         arg
     else
       send_msg t
         (require "PPSMC_MSG_TransferTableSmu2Dram"
            M.ppsmc_msg_transfertablesmu2dram)
         arg);
    Mmio.read_bytes (Amdev.vram t.adev) ~off:t.driver_table_paddr ~len:size

  (* ip.py:202 read_clocks *)
  let read_clocks t clk_list =
    match List.assoc_opt clk_list t.clocks_cache with
    | Some clocks -> clocks
    | None ->
        let module M = (val t.smu_mod) in
        let freq arg =
          send_msg_read t M.ppsmc_msg_getdpmfreqbyindex arg land 0x7fffffff
        in
        let clocks =
          List.filter_map
            (fun clck ->
              match freq ((clck lsl 16) lor 0xff) with
              | 0 -> None
              | cnt ->
                  let rec levels i =
                    if i = cnt then []
                    else
                      let v = freq ((clck lsl 16) lor i) in
                      v :: levels (i + 1)
                  in
                  Some (clck, levels 0))
            clk_list
        in
        t.clocks_cache <- (clk_list, clocks) :: t.clocks_cache;
        clocks

  (* ip.py:207 set_clocks; [level] indexes each clock's level list,
     from the end when negative, and [None] lifts the limits. *)
  let set_clocks t ~level =
    let module M = (val t.smu_mod) in
    let clks =
      [ M.ppclk_uclk; M.ppclk_fclk; M.ppclk_socclk ]
      @
      if
        not
          (List.mem
             (Amdev.ip_ver t.adev Am.mp0_hwip)
             [ (13, 0, 6); (13, 0, 12) ])
      then [ require "PPCLK_GFXCLK" M.ppclk_gfxclk ]
      else []
    in
    let gc10 = Amdev.ip_ver t.adev Am.gc_hwip >= (10, 0, 0) in
    match level with
    | None ->
        List.iter
          (fun clck ->
            (try
               send_msg ~timeout_ms:20 t M.ppsmc_msg_setsoftminbyfreq
                 (clck lsl 16)
             with Timeout_error _ -> ());
            if gc10 then
              send_msg t M.ppsmc_msg_setsoftmaxbyfreq
                ((clck lsl 16) lor 0xffff))
          clks
    | Some level ->
        List.iter
          (fun (clck, vals) ->
            let v =
              List.nth vals
                (if level < 0 then List.length vals + level else level)
            in
            (try
               send_msg ~timeout_ms:20 t M.ppsmc_msg_setsoftminbyfreq
                 ((clck lsl 16) lor v)
             with Timeout_error _ -> ());
            if gc10 then
              send_msg t M.ppsmc_msg_setsoftmaxbyfreq ((clck lsl 16) lor v))
          (read_clocks t clks)

  (* ip.py:221 set_power_limit *)
  let set_power_limit t watts =
    let ppt_limit = max (int_of_float (Float.round watts)) 1 in
    let module M = (val t.smu_mod) in
    send_msg t M.ppsmc_msg_setpptlimit ppt_limit;
    if debug >= 2 then
      Printf.printf "am %s: GPU power limit set to %dW\n%!"
        (Amdev.devfmt t.adev) ppt_limit

  (* ip.py:226 _aca_read_reg *)
  let aca_read_reg t ~ue bank_idx reg_idx =
    let module M = (val t.smu_mod) in
    let msg =
      if ue then require "PPSMC_MSG_McaBankDumpDW" M.ppsmc_msg_mcabankdumpdw
      else require "PPSMC_MSG_McaBankCeDumpDW" M.ppsmc_msg_mcabankcedumpdw
    in
    let hi = send_msg_read t msg ((bank_idx lsl 16) lor ((reg_idx * 8) + 4)) in
    let lo = send_msg_read t msg ((bank_idx lsl 16) lor (reg_idx * 8)) in
    Int64.logor (Int64.shift_left (Int64.of_int hi) 32) (Int64.of_int lo)

  (* ip.py:231 _aca_read_banks *)
  let aca_read_banks t ~ue =
    let module M = (val t.smu_mod) in
    match M.ppsmc_msg_queryvalidmcacount with
    | None -> []
    | Some count_msg ->
        let count_msg =
          if ue then count_msg
          else
            require "PPSMC_MSG_QueryValidMcaCeCount"
              M.ppsmc_msg_queryvalidmcacecount
        in
        let banks = send_msg_read t count_msg 0 in
        let rec bank idx =
          if idx = banks then []
          else
            let rec regs reg_idx =
              if reg_idx = 16 then []
              else aca_read_reg t ~ue idx reg_idx :: regs (reg_idx + 1)
            in
            regs 0 :: bank (idx + 1)
        in
        bank 0
end

(* Psp: ip.py AM_PSP *)

module Psp = struct
  type t = {
    adev : Amdev.t;
    fw : Firmware.t;
    reg_pref : string;
    msg1_paddr : int;
    msg1_addr : int;
    msg1_view : Mmio.t;
    cmd_paddr : int;
    fence_paddr : int;
    ring_size : int;
    ring_paddr : int;
    max_tmr_size : int;
    mutable tmr_size : int;
    boot_time_tmr : bool;
    autoload_tmr : bool;
    tmr_paddr : int;
  }

  (* ip.py:560 AM_PSP.init_sw *)
  let create adev ~fw =
    let mp0 = Amdev.ip_ver adev Am.mp0_hwip in
    let reg_pref =
      if mp0 < (14, 0, 0) then "regMP0_SMN_C2PMSG" else "regMPASP_SMN_C2PMSG"
    in
    let mm = Amdev.mm adev in
    let msg1_paddr =
      Memory.palloc mm Am.psp_1_meg ~align:Am.psp_1_meg ~zero:false ~boot:true
        ()
    in
    let msg1_addr = Amdev.paddr2mc adev msg1_paddr in
    let msg1_view =
      Mmio.view (Amdev.vram adev) ~off:msg1_paddr ~size:Am.psp_1_meg ()
    in
    let cmd_paddr =
      Memory.palloc mm Am.psp_cmd_buffer_size ~zero:false ~boot:true ()
    in
    let fence_paddr =
      Memory.palloc mm Am.psp_fence_buffer_size ~zero:true ~boot:true ()
    in
    let ring_size = 0x10000 in
    let ring_paddr = Memory.palloc mm ring_size ~zero:false ~boot:true () in
    let max_tmr_size = 0x1300000 in
    let boot_time_tmr =
      List.mem mp0 [ (13, 0, 6); (13, 0, 14); (14, 0, 2); (14, 0, 3) ]
    in
    let autoload_tmr = not (List.mem mp0 [ (13, 0, 6); (13, 0, 14) ]) in
    let tmr_paddr =
      if not boot_time_tmr then
        Memory.palloc mm max_tmr_size ~align:Am.psp_tmr_alignment ~zero:false
          ~boot:true ()
      else 0
    in
    {
      adev;
      fw;
      reg_pref;
      msg1_paddr;
      msg1_addr;
      msg1_view;
      cmd_paddr;
      fence_paddr;
      ring_size;
      ring_paddr;
      max_tmr_size;
      tmr_size = 0;
      boot_time_tmr;
      autoload_tmr;
      tmr_paddr;
    }

  let msg1_paddr t = t.msg1_paddr
  let cmd_paddr t = t.cmd_paddr
  let fence_paddr t = t.fence_paddr
  let ring_paddr t = t.ring_paddr
  let tmr_paddr t = t.tmr_paddr
  let creg t n = Amdev.reg t.adev (Printf.sprintf "%s_%d" t.reg_pref n)

  (* ip.py:605 is_sos_alive *)
  let is_sos_alive t = Am_register.read (creg t 81) <> 0x0

  (* ip.py:607 _wait_for_bootloader *)
  let wait_for_bootloader t =
    wait_cond t.adev ~value:0x80000000 ~msg:"BL not ready" (fun () ->
        Am_register.read (creg t 35) land 0x80000000)

  (* ip.py:609 _prep_msg1; padded to 16 bytes past a 4-byte tail for
     platforms whose copy path requires it. *)
  let prep_msg1 t data =
    if Bytes.length data > Mmio.size t.msg1_view then
      failwith
        (Printf.sprintf "msg1 buffer is too small 0x%x > 0x%x"
           (Bytes.length data) (Mmio.size t.msg1_view));
    let padded = Bytes.make ((Bytes.length data + 4 + 15) / 16 * 16) '\x00' in
    Bytes.blit data 0 padded 0 (Bytes.length data);
    Mmio.blit_bytes t.msg1_view ~off:0 padded;
    Gmc.flush_hdp t.adev

  (* ip.py:615 _bootloader_load_component *)
  let bootloader_load_component t fw_type compid =
    match List.assoc_opt fw_type t.fw.Firmware.sos_fw with
    | None -> ()
    | Some data ->
        wait_for_bootloader t;
        if debug >= 2 then
          Printf.printf "am %s: loading sos component: %s\n%!"
            (Amdev.devfmt t.adev)
            (match List.assoc_opt fw_type Am.psp_fw_type_names with
            | Some name -> name
            | None -> string_of_int fw_type);
        prep_msg1 t data;
        Am_register.write (creg t 36) ~value:(t.msg1_addr lsr 20) [];
        Am_register.write (creg t 35) ~value:compid [];
        if compid <> Am.psp_bl__load_sosdrv then wait_for_bootloader t

  (* ip.py:634 _ring_create *)
  let ring_create t =
    (* If the ring is already created, destroy it *)
    if Am_register.read (creg t 71) <> 0 then begin
      Am_register.write (creg t 64) ~value:Am.gfx_ctrl_cmd_id_destroy_rings [];
      (* There might be handshake issue with hardware which needs delay *)
      sleep_ms t.adev 20
    end;
    (* Wait until the sOS is ready *)
    wait_cond t.adev ~value:0x80000000 ~msg:"sOS not ready" (fun () ->
        Am_register.read (creg t 64) land 0x80000000);
    Amdev.wreg_pair t.adev t.reg_pref ~lo:"_69" ~hi:"_70"
      (Amdev.paddr2mc t.adev t.ring_paddr);
    Am_register.write (creg t 71) ~value:t.ring_size [];
    Am_register.write (creg t 64) ~value:(Am.psp_ring_type__km lsl 16) [];
    (* There might be handshake issue with hardware which needs delay *)
    sleep_ms t.adev 20;
    wait_cond t.adev ~value:0x80000000 ~msg:"sOS ring not created" (fun () ->
        Am_register.read (creg t 64) land 0x8000FFFF)

  (* ip.py:654 _ring_submit *)
  let ring_submit t cmd =
    let prev_wptr = Am_register.read (creg t 67) in
    let fence_value = prev_wptr + 1 in
    let vram = Amdev.vram t.adev in
    let msg = Bytes.make Am.Psp_gfx_rb_frame.sizeof '\x00' in
    Am.Psp_gfx_rb_frame.set_fence_value msg fence_value;
    Am.Psp_gfx_rb_frame.set_cmd_buf_addr_lo msg
      (lo32 (Amdev.paddr2mc t.adev t.cmd_paddr));
    Am.Psp_gfx_rb_frame.set_cmd_buf_addr_hi msg
      (hi32 (Amdev.paddr2mc t.adev t.cmd_paddr));
    Am.Psp_gfx_rb_frame.set_fence_addr_lo msg
      (lo32 (Amdev.paddr2mc t.adev t.fence_paddr));
    Am.Psp_gfx_rb_frame.set_fence_addr_hi msg
      (hi32 (Amdev.paddr2mc t.adev t.fence_paddr));
    Mmio.blit_bytes vram ~off:t.cmd_paddr cmd;
    Mmio.blit_bytes vram ~off:(t.ring_paddr + (prev_wptr * 4)) msg;
    (* Move the wptr *)
    Am_register.write (creg t 67)
      ~value:(prev_wptr + (Am.Psp_gfx_rb_frame.sizeof / 4))
      [];
    wait_cond t.adev ~value:fence_value ~msg:"sOS ring not responding"
      (fun () -> Int32.to_int (Mmio.read32 vram t.fence_paddr) land 0xffffffff);
    let resp =
      Mmio.read_bytes vram ~off:t.cmd_paddr ~len:Am.Psp_gfx_cmd_resp.sizeof
    in
    if Am.Psp_gfx_cmd_resp.resp_status resp 0 <> 0 then
      failwith
        (Printf.sprintf "PSP command failed %d %d"
           (Am.Psp_gfx_cmd_resp.cmd_id resp 0)
           (Am.Psp_gfx_cmd_resp.resp_status resp 0));
    resp

  let cmd_make cmd_id =
    let cmd = Bytes.make Am.Psp_gfx_cmd_resp.sizeof '\x00' in
    Am.Psp_gfx_cmd_resp.set_cmd_id cmd cmd_id;
    cmd

  (* ip.py:672 _load_ip_fw_cmd *)
  let load_ip_fw_cmd t fw_types fw_bytes =
    prep_msg1 t fw_bytes;
    List.iter
      (fun fw_type ->
        if debug >= 2 then
          Printf.printf "am %s: loading fw: %s\n%!" (Amdev.devfmt t.adev)
            (match List.assoc_opt fw_type Am.psp_gfx_fw_type_names with
            | Some name -> name
            | None -> string_of_int fw_type);
        let cmd = cmd_make Am.gfx_cmd_id_load_ip_fw in
        Am.Psp_gfx_cmd_resp.Cmd_load_ip_fw.set_fw_phy_addr_hi cmd
          (hi32 t.msg1_addr);
        Am.Psp_gfx_cmd_resp.Cmd_load_ip_fw.set_fw_phy_addr_lo cmd
          (lo32 t.msg1_addr);
        Am.Psp_gfx_cmd_resp.Cmd_load_ip_fw.set_fw_size cmd
          (Bytes.length fw_bytes);
        Am.Psp_gfx_cmd_resp.Cmd_load_ip_fw.set_fw_type cmd fw_type;
        ignore (ring_submit t cmd))
      fw_types

  (* ip.py:692 _load_toc_cmd *)
  let load_toc_cmd t toc_size =
    let cmd = cmd_make Am.gfx_cmd_id_load_toc in
    Am.Psp_gfx_cmd_resp.Cmd_load_toc.set_toc_phy_addr_hi cmd
      (hi32 t.msg1_addr);
    Am.Psp_gfx_cmd_resp.Cmd_load_toc.set_toc_phy_addr_lo cmd
      (lo32 t.msg1_addr);
    Am.Psp_gfx_cmd_resp.Cmd_load_toc.set_toc_size cmd toc_size;
    ring_submit t cmd

  (* ip.py:628 _tmr_init: load TOC and calculate TMR size *)
  let tmr_init t =
    let fwm = List.assoc Am.psp_fw_type_psp_toc t.fw.Firmware.sos_fw in
    prep_msg1 t fwm;
    let resp = load_toc_cmd t (Bytes.length fwm) in
    t.tmr_size <- Am.Psp_gfx_cmd_resp.resp_tmr_size resp 0;
    if t.tmr_size > t.max_tmr_size then
      failwith
        (Printf.sprintf "tmr size 0x%x exceeds the maximum 0x%x" t.tmr_size
           t.max_tmr_size)

  (* ip.py:682 _tmr_load_cmd *)
  let tmr_load_cmd t =
    let tmr_xgmi_paddr =
      if t.tmr_paddr <> 0 then Amdev.paddr2xgmi t.adev t.tmr_paddr else 0
    in
    let buf_mc = if t.tmr_paddr <> 0 then Amdev.paddr2mc t.adev t.tmr_paddr else 0 in
    let cmd = cmd_make Am.gfx_cmd_id_setup_tmr in
    Am.Psp_gfx_cmd_resp.Cmd_setup_tmr.set_buf_phy_addr_hi cmd (hi32 buf_mc);
    Am.Psp_gfx_cmd_resp.Cmd_setup_tmr.set_buf_phy_addr_lo cmd (lo32 buf_mc);
    Am.Psp_gfx_cmd_resp.Cmd_setup_tmr.set_system_phy_addr_hi cmd
      (hi32 tmr_xgmi_paddr);
    Am.Psp_gfx_cmd_resp.Cmd_setup_tmr.set_system_phy_addr_lo cmd
      (lo32 tmr_xgmi_paddr);
    Am.Psp_gfx_cmd_resp.Cmd_setup_tmr.set_virt_phy_addr cmd 1;
    Am.Psp_gfx_cmd_resp.Cmd_setup_tmr.set_buf_size cmd
      (if t.tmr_paddr <> 0 then t.tmr_size else 0);
    ignore (ring_submit t cmd)

  (* ip.py:698 _spatial_partition_cmd *)
  let spatial_partition_cmd t mode =
    let cmd = cmd_make Am.gfx_cmd_id_sriov_spatial_part in
    Am.Psp_gfx_cmd_resp.Cmd_spatial_part.set_mode cmd mode;
    ignore (ring_submit t cmd)

  (* ip.py:703 _rlc_autoload_cmd *)
  let rlc_autoload_cmd t =
    ignore (ring_submit t (cmd_make Am.gfx_cmd_id_autoload_rlc))

  (* ip.py:582 AM_PSP.init_hw *)
  let init_hw t =
    let spl_key =
      if Amdev.ip_ver t.adev Am.mp0_hwip >= (14, 0, 0) then
        Am.psp_fw_type_psp_spl
      else Am.psp_fw_type_psp_kdb
    in
    let sos_components =
      [
        (Am.psp_fw_type_psp_kdb, Am.psp_bl__load_key_database);
        (spl_key, Am.psp_bl__load_tos_spl_table);
        (Am.psp_fw_type_psp_sys_drv, Am.psp_bl__load_sysdrv);
        (Am.psp_fw_type_psp_soc_drv, Am.psp_bl__load_socdrv);
        (Am.psp_fw_type_psp_intf_drv, Am.psp_bl__load_intfdrv);
        (Am.psp_fw_type_psp_dbg_drv, Am.psp_bl__load_dbgdrv);
        (Am.psp_fw_type_psp_ras_drv, Am.psp_bl__load_rasdrv);
        (Am.psp_fw_type_psp_sos, Am.psp_bl__load_sosdrv);
      ]
    in
    if not (is_sos_alive t) then begin
      List.iter
        (fun (fw_type, compid) -> bootloader_load_component t fw_type compid)
        sos_components;
      wait_cond t.adev ~value:1 ~msg:"sOS failed to start" (fun () ->
          Bool.to_int (is_sos_alive t))
    end;
    ring_create t;
    if List.mem_assoc Am.psp_fw_type_psp_toc t.fw.Firmware.sos_fw then
      tmr_init t;
    (* SMU fw should be loaded before TMR. *)
    (match t.fw.Firmware.smu_psp_desc with
    | Some (fw_types, fw_bytes) -> load_ip_fw_cmd t fw_types fw_bytes
    | None -> ());
    if (not t.boot_time_tmr) || not t.autoload_tmr then tmr_load_cmd t;
    List.iter
      (fun (fw_types, fw_bytes) -> load_ip_fw_cmd t fw_types fw_bytes)
      t.fw.Firmware.descs;
    if Amdev.ip_ver t.adev Am.gc_hwip >= (11, 0, 0) then rlc_autoload_cmd t
    else
      load_ip_fw_cmd t
        [ Am.gfx_fw_type_reg_list ]
        (List.assoc Am.psp_fw_type_psp_rl t.fw.Firmware.sos_fw)
end

(* Gfx: ip.py AM_GFX *)

module Gfx = struct
  type t = {
    adev : Amdev.t;
    xccs : int;
    mqd_paddr : int array;
    mqd_mc : int array;
  }

  (* The queue-descriptor fields shared by every generation's layout. *)
  module type Compute_mqd = sig
    val sizeof : int
    val set_header : bytes -> int -> unit
    val set_cp_mqd_base_addr_lo : bytes -> int -> unit
    val set_cp_mqd_base_addr_hi : bytes -> int -> unit
    val set_cp_hqd_vmid : bytes -> int -> unit
    val set_cp_hqd_persistent_state : bytes -> int -> unit
    val set_cp_hqd_pipe_priority : bytes -> int -> unit
    val set_cp_hqd_queue_priority : bytes -> int -> unit
    val set_cp_hqd_quantum : bytes -> int -> unit
    val set_cp_hqd_pq_base_lo : bytes -> int -> unit
    val set_cp_hqd_pq_base_hi : bytes -> int -> unit
    val set_cp_hqd_pq_rptr_report_addr_lo : bytes -> int -> unit
    val set_cp_hqd_pq_rptr_report_addr_hi : bytes -> int -> unit
    val set_cp_hqd_pq_wptr_poll_addr_lo : bytes -> int -> unit
    val set_cp_hqd_pq_wptr_poll_addr_hi : bytes -> int -> unit
    val set_cp_hqd_pq_doorbell_control : bytes -> int -> unit
    val set_cp_hqd_pq_control : bytes -> int -> unit
    val set_cp_hqd_ib_control : bytes -> int -> unit
    val set_cp_hqd_hq_status0 : bytes -> int -> unit
    val set_cp_mqd_control : bytes -> int -> unit
    val set_cp_hqd_aql_control : bytes -> int -> unit
    val set_cp_hqd_eop_base_addr_lo : bytes -> int -> unit
    val set_cp_hqd_eop_base_addr_hi : bytes -> int -> unit
    val set_cp_hqd_eop_control : bytes -> int -> unit
    val set_compute_static_thread_mgmt_se0 : bytes -> int -> unit
    val set_compute_static_thread_mgmt_se1 : bytes -> int -> unit
    val set_compute_static_thread_mgmt_se2 : bytes -> int -> unit
    val set_compute_static_thread_mgmt_se3 : bytes -> int -> unit
  end

  let mqd_mod adev : (module Compute_mqd) =
    match Amdev.ip_ver adev Am.gc_hwip with
    | 9, _, _ -> (module Am.V9_mqd)
    | 11, _, _ -> (module Am.V11_compute_mqd)
    | 12, _, _ -> (module Am.V12_compute_mqd)
    | ma, _, _ ->
        invalid_arg (Printf.sprintf "no compute queue descriptor for gfx%d" ma)

  (* ip.py:248 AM_GFX.init_sw *)
  let create adev =
    let xccs =
      List.length
        (List.assoc Am.gc_hwip (Amdev.discovery adev).Amdev.regs_offset)
    in
    let mqd_paddr =
      Array.init 2 (fun _ ->
          Memory.palloc (Amdev.mm adev) (0x1000 * xccs) ~zero:false ~boot:true
            ())
    in
    { adev; xccs; mqd_paddr; mqd_mc = Array.map (Amdev.paddr2mc adev) mqd_paddr }

  let xccs t = t.xccs

  (* ip.py:372 _grbm_select *)
  let grbm_select ?(me = 0) ?(pipe = 0) ?(queue = 0) ?(vmid = 0) t =
    Am_register.write
      (Amdev.reg t.adev "regGRBM_GFX_CNTL")
      [ ("meid", me); ("pipeid", pipe); ("vmid", vmid); ("queueid", queue) ]

  (* ip.py:375 _enable_mec *)
  let enable_mec t =
    for _ = 1 to t.xccs do
      if Amdev.ip_ver t.adev Am.gc_hwip >= (10, 0, 0) then
        Am_register.update
          (Amdev.reg t.adev "regCP_MEC_RS64_CNTL")
          [ ("mec_pipe0_reset", 0); ("mec_pipe0_active", 1); ("mec_halt", 0) ]
      else Am_register.write (Amdev.reg t.adev "regCP_MEC_CNTL") ~value:0x0 []
    done;
    (* Wait for MEC to be ready *)
    sleep_ms t.adev 50

  (* ip.py:381 _config_mec *)
  let config_mec t ~fw =
    let adev = t.adev in
    let config_helper ~eng_name ~cntl_reg ~eng_reg ~pipe_cnt ?(me = 0) () =
      for pipe = 0 to pipe_cnt - 1 do
        grbm_select ~me ~pipe t;
        Amdev.wreg_pair adev
          (Printf.sprintf "regCP_%s_PRGRM_CNTR_START" eng_reg)
          ~lo:"" ~hi:"_HI"
          (List.assoc eng_name fw.Firmware.ucode_start lsr 2)
      done;
      grbm_select t;
      let cntl = Amdev.reg adev (Printf.sprintf "regCP_%s_CNTL" cntl_reg) in
      let resets v =
        List.init pipe_cnt (fun pipe ->
            ( Printf.sprintf "%s_pipe%d_reset"
                (String.lowercase_ascii eng_name)
                pipe,
              v ))
      in
      Am_register.update cntl (resets 1);
      Am_register.update cntl (resets 0)
    in
    for _ = 1 to t.xccs do
      if Amdev.ip_ver adev Am.gc_hwip < (10, 0, 0) then
        Am_register.update
          (Amdev.reg adev "regCP_MEC_CNTL")
          [
            ("mec_invalidate_icache", 1); ("mec_me1_pipe0_reset", 1);
            ("mec_me2_pipe0_reset", 1); ("mec_me1_halt", 1);
            ("mec_me2_halt", 1);
          ];
      if Amdev.ip_ver adev Am.gc_hwip >= (12, 0, 0) then begin
        config_helper ~eng_name:"PFP" ~cntl_reg:"ME" ~eng_reg:"PFP"
          ~pipe_cnt:1 ();
        config_helper ~eng_name:"ME" ~cntl_reg:"ME" ~eng_reg:"ME" ~pipe_cnt:1
          ()
      end;
      if Amdev.ip_ver adev Am.gc_hwip >= (10, 0, 0) then
        config_helper ~eng_name:"MEC" ~cntl_reg:"MEC_RS64" ~eng_reg:"MEC_RS64"
          ~pipe_cnt:1 ~me:1 ()
    done

  (* ip.py:399 _dequeue_hqds *)
  let dequeue_hqds t =
    let adev = t.adev in
    for q = 0 to 1 do
      for _ = 1 to t.xccs do
        grbm_select ~me:1 ~pipe:0 ~queue:q t;
        if Am_register.read (Amdev.reg adev "regCP_HQD_ACTIVE") land 1 <> 0
        then begin
          (* 1 - DRAIN_PIPE; 2 - RESET_WAVES *)
          Am_register.write
            (Amdev.reg adev "regCP_HQD_DEQUEUE_REQUEST")
            ~value:0x2 [];
          Am_register.write
            (Amdev.reg adev "regSPI_COMPUTE_QUEUE_RESET")
            ~value:0x1 [];
          if not (Amdev.is_err_state adev) then
            wait_cond adev ~value:0 ~msg:"HQD dequeue timeout" (fun () ->
                Am_register.read (Amdev.reg adev "regCP_HQD_ACTIVE") land 1)
        end
      done
    done;
    grbm_select t

  let fini_hw t = dequeue_hqds t

  (* ip.py:305 reset_mec *)
  let reset_mec t ~fw =
    dequeue_hqds t;
    (* gfx12+ resets through the per-pipe engine controls instead *)
    if Amdev.ip_ver t.adev Am.gc_hwip < (12, 0, 0) then begin
      for _ = 1 to t.xccs do
        Am_register.write
          (Amdev.reg t.adev "regGRBM_SOFT_RESET")
          [ ("soft_reset_cp", 1); ("soft_reset_cpc", 1) ]
      done;
      sleep_ms t.adev 50;
      for _ = 1 to t.xccs do
        Am_register.write (Amdev.reg t.adev "regGRBM_SOFT_RESET") ~value:0x0 []
      done
    end;
    config_mec t ~fw;
    enable_mec t

  (* ip.py:253 AM_GFX.init_hw *)
  let init_hw t ~soc ~gmc ~psp ~fw ~partial_boot =
    let adev = t.adev in
    let gc_ver = Amdev.ip_ver adev Am.gc_hwip in
    let gc_major, gc_minor, _ = gc_ver in
    (* Wait for RLC autoload to complete *)
    wait_cond adev ~value:1 ~msg:"RLC autoload timeout" (fun () ->
        Bool.to_int
          (Am_register.read (Amdev.reg adev "regCP_STAT") = 0
          || List.assoc "bootload_complete"
               (Am_register.read_bitfields
                  (Amdev.reg adev "regRLC_RLCS_BOOTLOAD_STATUS"))
             = 0));
    Gmc.init_hub gmc ~soc Gmc.Gc ~inst_cnt:t.xccs;
    if partial_boot then reset_mec t ~fw
    else begin
      config_mec t ~fw;
      (* NOTE: Golden reg for gfx11. No values for this reg provided.
         The kernel just ors 0x20000000 to this reg. *)
      for _ = 1 to t.xccs do
        let tcp = Amdev.reg adev "regTCP_CNTL" in
        Am_register.write tcp ~value:(Am_register.read tcp lor 0x20000000) []
      done;
      for _ = 1 to t.xccs do
        Am_register.write (Amdev.reg adev "regRLC_CNTL") ~value:0x1 []
      done;
      for _ = 1 to t.xccs do
        Am_register.update
          (Amdev.reg adev "regRLC_SRM_CNTL")
          [ ("srm_enable", 1); ("auto_incr_addr", 1) ]
      done;
      for _ = 1 to t.xccs do
        Am_register.write (Amdev.reg adev "regRLC_SPM_MC_CNTL") ~value:0xf []
      done;
      (let nbio_ma, nbio_mi, _ = Amdev.ip_ver adev Am.nbio_hwip in
       if (nbio_ma, nbio_mi) <> (7, 9) then begin
         Soc.doorbell_enable soc ~port:0 ~awid:0x3 ~awaddr_31_28_value:0x3 ();
         Soc.doorbell_enable soc ~port:3 ~awid:0x6 ~awaddr_31_28_value:0x3 ()
       end);
      for xcc = 0 to t.xccs - 1 do
        if List.mem gc_ver [ (9, 4, 3); (9, 5, 0) ] then begin
          (* Golden value for mi300/mi350 *)
          Am_register.write
            (Amdev.reg adev "regGB_ADDR_CONFIG")
            ~value:0x2a114042 [];
          Am_register.update
            (Amdev.reg adev "regTCP_UTCL1_CNTL2")
            [ ("spare", 1) ]
        end;
        Am_register.update
          (Amdev.reg adev "regGRBM_CNTL")
          [ ("read_timeout", 0xff) ];
        for i = 0 to 15 do
          grbm_select ~vmid:i t;
          Am_register.write
            (Amdev.reg adev "regSH_MEM_CONFIG")
            ((if gc_major >= 10 then [ ("initial_inst_prefetch", 3) ]
              else [ ("retry_disable", 1) ])
            @ (if (gc_major, gc_minor) = (9, 4) then [ ("f8_mode", 1) ]
               else [])
            @ [
                ("address_mode", Soc.sh_mem_address_mode_64 soc);
                ("alignment_mode", Soc.sh_mem_alignment_mode_unaligned soc);
              ]);
          (* Configure apertures:
             LDS:     0x10000000'00000000 - 0x10000001'00000000 (4GB)
             Scratch: 0x20000000'00000000 - 0x20000001'00000000 (4GB) *)
          Am_register.write
            (Amdev.reg adev "regSH_MEM_BASES")
            [ ("shared_base", 0x1); ("private_base", 0x2) ]
        done;
        grbm_select t;
        (* Configure MEC doorbell range *)
        Am_register.write
          (Amdev.reg adev "regCP_MEC_DOORBELL_RANGE_LOWER")
          ~value:(0x100 * xcc) [];
        Am_register.write
          (Amdev.reg adev "regCP_MEC_DOORBELL_RANGE_UPPER")
          ~value:((0x100 * xcc) + 0xf8) []
      done;
      enable_mec t;
      (* Set 1 partition *)
      if t.xccs > 1 then Psp.spatial_partition_cmd psp 1
    end

  (* ip.py:316 setup_ring *)
  let setup_ring t ~ring_addr ~ring_size ~rptr_addr ~wptr_addr ~eop_addr
      ~eop_size ~idx ~aql =
    let adev = t.adev in
    let pipe, queue = (idx / 4, idx mod 4) in
    let doorbell = Am.amdgpu_navi10_doorbell_mec_ring0 in
    let gc_major, _, _ = Amdev.ip_ver adev Am.gc_hwip in
    let encode name fields =
      Amd_tables.Reg.encode (Am_register.reg (Amdev.reg adev name)) fields
    in
    let module M = (val mqd_mod adev) in
    for xcc = 0 to (if aql then t.xccs else 1) - 1 do
      grbm_select ~me:1 ~pipe ~queue t;
      let mqd = Bytes.make M.sizeof '\x00' in
      M.set_header mqd 0xC0310800;
      M.set_cp_mqd_base_addr_lo mqd (lo32 (t.mqd_mc.(queue) + (0x1000 * xcc)));
      M.set_cp_mqd_base_addr_hi mqd (hi32 (t.mqd_mc.(queue) + (0x1000 * xcc)));
      M.set_cp_hqd_pipe_priority mqd 0x2;
      M.set_cp_hqd_queue_priority mqd 0xf;
      M.set_cp_hqd_quantum mqd 0x111;
      M.set_cp_hqd_persistent_state mqd
        (encode "regCP_HQD_PERSISTENT_STATE"
           [ ("preload_size", 0x55); ("preload_req", 1) ]);
      M.set_cp_hqd_pq_base_lo mqd (lo32 (ring_addr lsr 8));
      M.set_cp_hqd_pq_base_hi mqd (hi32 (ring_addr lsr 8));
      M.set_cp_hqd_pq_rptr_report_addr_lo mqd (lo32 rptr_addr);
      M.set_cp_hqd_pq_rptr_report_addr_hi mqd (hi32 rptr_addr);
      M.set_cp_hqd_pq_wptr_poll_addr_lo mqd (lo32 wptr_addr);
      M.set_cp_hqd_pq_wptr_poll_addr_hi mqd (hi32 wptr_addr);
      M.set_cp_hqd_pq_doorbell_control mqd
        (encode "regCP_HQD_PQ_DOORBELL_CONTROL"
           [ ("doorbell_offset", doorbell * 2); ("doorbell_en", 1) ]);
      M.set_cp_hqd_pq_control mqd
        (encode "regCP_HQD_PQ_CONTROL"
           ([
              ("rptr_block_size", 5); ("unord_dispatch", 0);
              ("queue_size", bit_length (ring_size / 4) - 2);
            ]
           @
           if aql then
             [
               ("queue_full_en", 1); ("slot_based_wptr", 2);
               ("no_update_rptr", Bool.to_int (xcc <> 0 || t.xccs = 1));
             ]
           else []));
      M.set_cp_hqd_ib_control mqd
        (encode "regCP_HQD_IB_CONTROL" [ ("min_ib_avail_size", 0x3) ]);
      M.set_cp_hqd_hq_status0 mqd 0x20004000;
      M.set_cp_mqd_control mqd (encode "regCP_MQD_CONTROL" [ ("priv_state", 1) ]);
      M.set_cp_hqd_vmid mqd 0;
      M.set_cp_hqd_aql_control mqd (Bool.to_int aql);
      M.set_cp_hqd_eop_base_addr_lo mqd (lo32 (eop_addr lsr 8));
      M.set_cp_hqd_eop_base_addr_hi mqd (hi32 (eop_addr lsr 8));
      M.set_cp_hqd_eop_control mqd
        (encode "regCP_HQD_EOP_CONTROL"
           [ ("eop_size", bit_length (eop_size / 4) - 2) ]);
      (if aql && t.xccs > 1 then
         match gc_major with
         | 9 ->
             Am.V9_mqd.set_compute_tg_chunk_size mqd 1;
             Am.V9_mqd.set_compute_current_logic_xcc_id mqd xcc;
             Am.V9_mqd.set_cp_mqd_stride_size mqd 0x1000
         | ma ->
             invalid_arg
               (Printf.sprintf
                  "multi-die queues need the gfx9 descriptor, not gfx%d" ma));
      M.set_compute_static_thread_mgmt_se0 mqd 0xffffffff;
      M.set_compute_static_thread_mgmt_se1 mqd 0xffffffff;
      M.set_compute_static_thread_mgmt_se2 mqd 0xffffffff;
      M.set_compute_static_thread_mgmt_se3 mqd 0xffffffff;
      (* ip.py:337: 8 shader engines on gfx10+, 4 below *)
      (match gc_major with
      | 11 ->
          Am.V11_compute_mqd.set_compute_static_thread_mgmt_se4 mqd 0xffffffff;
          Am.V11_compute_mqd.set_compute_static_thread_mgmt_se5 mqd 0xffffffff;
          Am.V11_compute_mqd.set_compute_static_thread_mgmt_se6 mqd 0xffffffff;
          Am.V11_compute_mqd.set_compute_static_thread_mgmt_se7 mqd 0xffffffff
      | 12 ->
          Am.V12_compute_mqd.set_compute_static_thread_mgmt_se4 mqd 0xffffffff;
          Am.V12_compute_mqd.set_compute_static_thread_mgmt_se5 mqd 0xffffffff;
          Am.V12_compute_mqd.set_compute_static_thread_mgmt_se6 mqd 0xffffffff;
          Am.V12_compute_mqd.set_compute_static_thread_mgmt_se7 mqd 0xffffffff
      | _ -> ());
      Mmio.blit_bytes (Amdev.vram adev)
        ~off:(t.mqd_paddr.(queue) + (0x1000 * xcc))
        mqd;
      (* The queue-bringup registers mirror the descriptor's register
         block, dword for dword from its 0x80th dword. *)
      let base =
        (Am_register.reg (Amdev.reg adev "regCP_MQD_BASE_ADDR"))
          .Amd_tables.Reg.addr
      in
      let last =
        (Am_register.reg (Amdev.reg adev "regCP_HQD_PQ_WPTR_HI"))
          .Amd_tables.Reg.addr
      in
      for i = 0 to last - base do
        Amdev.wreg adev (base + i)
          (Int32.to_int (Bytes.get_int32_le mqd ((0x80 + i) * 4))
          land 0xffffffff)
      done;
      Am_register.write (Amdev.reg adev "regCP_HQD_ACTIVE") ~value:0x1 [];
      Gmc.flush_hdp adev;
      grbm_select t
    done;
    doorbell

  (* ip.py:350 AM_GFX.set_clockgating_state *)
  let set_clockgating_state t =
    let adev = t.adev in
    let gc_major, _, _ = Amdev.ip_ver adev Am.gc_hwip in
    (match Amdev.reg adev "regMM_ATC_L2_MISC_CG" with
    | reg -> Am_register.write reg [ ("enable", 1); ("mem_ls_enable", 1) ]
    | exception Invalid_argument _ -> ());
    for _ = 1 to t.xccs do
      Am_register.write
        (Amdev.reg adev "regRLC_SAFE_MODE")
        [ ("message", 1); ("cmd", 1) ];
      wait_cond adev ~value:0 ~msg:"RLC safe mode timeout" (fun () ->
          Am_register.read (Amdev.reg adev "regRLC_SAFE_MODE") land 0x1);
      Am_register.update
        (Amdev.reg adev "regRLC_CGCG_CGLS_CTRL")
        [
          ("cgcg_gfx_idle_threshold", 0x36); ("cgcg_en", 1);
          ("cgls_rep_compansat_delay", 0xf); ("cgls_en", 1);
        ];
      Am_register.update
        (Amdev.reg adev "regCP_RB_WPTR_POLL_CNTL")
        [ ("poll_frequency", 0x100); ("idle_poll_count", 0x90) ];
      Am_register.update
        (Amdev.reg adev "regCP_INT_CNTL")
        [
          ("cntx_busy_int_enable", 1); ("cntx_empty_int_enable", 1);
          ("cmp_busy_int_enable", 1);
        ];
      if gc_major >= 10 then begin
        Am_register.update
          (Amdev.reg adev "regSDMA0_RLC_CGCG_CTRL")
          [ ("cgcg_int_enable", 1) ];
        Am_register.update
          (Amdev.reg adev "regSDMA1_RLC_CGCG_CTRL")
          [ ("cgcg_int_enable", 1) ]
      end;
      let feats_gfx9 =
        if gc_major = 9 then
          [ ("gfxip_mgls_override", 0); ("gfxip_rep_fgcg_override", 0) ]
        else []
      in
      let feats_gfx11 =
        if gc_major >= 11 then
          [ ("perfmon_clock_state", 1); ("gfxip_repeater_fgcg_override", 0) ]
        else []
      in
      Am_register.update
        (Amdev.reg adev "regRLC_CGTT_MGCG_OVERRIDE")
        (feats_gfx9 @ feats_gfx11
        @ [
            ("gfxip_fgcg_override", 0); ("grbm_cgtt_sclk_override", 0);
            ("rlc_cgtt_sclk_override", 0); ("gfxip_mgcg_override", 0);
            ("gfxip_cgls_override", 0); ("gfxip_cgcg_override", 0);
          ]);
      Am_register.write
        (Amdev.reg adev "regRLC_SAFE_MODE")
        [ ("message", 0); ("cmd", 1) ]
    done
end

(* Ih: ip.py AM_IH *)

module Ih = struct
  type t = {
    adev : Amdev.t;
    ring_size : int;
    (* ring paddr, read/write-pointer buffer paddr, register suffix,
       ring id *)
    rings : (int * int * string * int) array;
    ring_view : Mmio.t;
  }

  (* ip.py:410 AM_IH.init_sw *)
  let create adev =
    let mm = Amdev.mm adev in
    let ring_size = 256 lsl 10 in
    let alloc_ring () =
      let ring = Memory.palloc mm ring_size ~zero:false ~boot:true () in
      let rwptr = Memory.palloc mm 0x1000 ~zero:false ~boot:true () in
      (ring, rwptr)
    in
    let ring0, rwptr0 = alloc_ring () in
    let ring1, rwptr1 = alloc_ring () in
    {
      adev;
      ring_size;
      rings = [| (ring0, rwptr0, "", 0); (ring1, rwptr1, "_RING1", 1) |];
      ring_view = Mmio.view (Amdev.vram adev) ~off:ring0 ~size:ring_size ();
    }

  (* ip.py:416 AM_IH.init_hw *)
  let init_hw t =
    let adev = t.adev in
    Array.iter
      (fun (ring_vm, rwptr_vm, suf, ring_id) ->
        Amdev.wreg_pair adev "regIH_RB_BASE" ~lo:suf ~hi:("_HI" ^ suf)
          (Amdev.paddr2mc adev ring_vm lsr 8);
        Am_register.write
          (Amdev.reg adev ("regIH_RB_CNTL" ^ suf))
          ([
             ("mc_space", 4); ("wptr_overflow_clear", 1);
             ("rb_size", bit_length ((t.ring_size / 4) - 1));
             ("mc_snoop", 1); ("mc_ro", 0); ("mc_vmid", 0);
           ]
          @
          if ring_id = 0 then
            [ ("wptr_overflow_enable", 1); ("rptr_rearm", 1) ]
          else [ ("rb_full_drain_enable", 1) ]);
        if ring_id = 0 then
          Amdev.wreg_pair adev "regIH_RB_WPTR_ADDR" ~lo:"_LO" ~hi:"_HI"
            (Amdev.paddr2mc adev rwptr_vm);
        Am_register.write (Amdev.reg adev ("regIH_RB_WPTR" ^ suf)) ~value:0 [];
        Am_register.write (Amdev.reg adev ("regIH_RB_RPTR" ^ suf)) ~value:0 [];
        Am_register.write
          (Amdev.reg adev ("regIH_DOORBELL_RPTR" ^ suf))
          [ ("enable", 0) ])
      t.rings;
    if Amdev.ip_ver adev Am.osssys_hwip <> (4, 4, 2) then begin
      Am_register.update
        (Amdev.reg adev "regIH_STORM_CLIENT_LIST_CNTL")
        [ ("client18_is_storm_client", 1) ];
      Am_register.update
        (Amdev.reg adev "regIH_INT_FLOOD_CNTL")
        [ ("flood_cntl_enable", 1) ];
      Am_register.update
        (Amdev.reg adev "regIH_MSI_STORM_CTRL")
        [ ("delay", 3) ]
    end;
    (* toggle interrupts *)
    Array.iter
      (fun (_, _, suf, ring_id) ->
        Am_register.update
          (Amdev.reg adev ("regIH_RB_CNTL" ^ suf))
          (("rb_enable", 1)
          :: (if ring_id = 0 then [ ("enable_intr", 1) ] else [])))
      t.rings

  (* ip.py:439 drain *)
  let drain t =
    let adev = t.adev in
    let _, _, suf, _ = t.rings.(0) in
    let wptr =
      Am_register.read_bitfields (Amdev.reg adev ("regIH_RB_WPTR" ^ suf))
    in
    Am_register.write
      (Amdev.reg adev "regIH_RB_RPTR")
      ~value:(List.assoc "offset" wptr mod (t.ring_size / 4))
      [];
    if List.assoc "rb_overflow" wptr <> 0 then begin
      Am_register.update
        (Amdev.reg adev ("regIH_RB_WPTR" ^ suf))
        [ ("rb_overflow", 0) ];
      Am_register.update
        (Amdev.reg adev ("regIH_RB_CNTL" ^ suf))
        [ ("wptr_overflow_clear", 1) ];
      Am_register.update
        (Amdev.reg adev ("regIH_RB_CNTL" ^ suf))
        [ ("wptr_overflow_clear", 0) ]
    end

  (* ip.py:449 interrupt_handler *)
  let interrupt_handler t ~soc ~gmc ~smu =
    let adev = t.adev in
    let devfmt = Amdev.devfmt adev in
    let gc_major, _, _ = Amdev.ip_ver adev Am.gc_hwip in
    let _, _, suf, _ = t.rings.(0) in
    let ring_dwords = t.ring_size / 4 in
    let wptr =
      List.assoc "offset"
        (Am_register.read_bitfields (Amdev.reg adev ("regIH_RB_WPTR" ^ suf)))
    in
    let rptr = ref (Am_register.read (Amdev.reg adev "regIH_RB_RPTR")) in
    while !rptr <> wptr do
      let entry =
        Array.init 8 (fun i ->
            Int32.to_int
              (Mmio.read32 t.ring_view (((!rptr + i) mod ring_dwords) * 4))
            land 0xffffffff)
      in
      rptr := (!rptr + 8) mod ring_dwords;
      let client = Am.soc15_client_id_from_ih_entry entry in
      let src = Am.soc15_source_id_from_ih_entry entry in
      let ring_id = Am.soc15_ring_id_from_ih_entry entry in
      let vmid = Am.soc15_vmid_from_ih_entry entry in
      let vmid_type = Am.soc15_vmid_type_from_ih_entry entry in
      let pasid = Am.soc15_pasid_from_ih_entry entry in
      let node = Am.soc15_nodeid_from_ih_entry entry in
      let ctx0 = Am.soc15_context_id0_from_ih_entry entry in
      let ctx1 = Am.soc15_context_id1_from_ih_entry entry in
      let ctx2 = Am.soc15_context_id2_from_ih_entry entry in
      let ctx3 = Am.soc15_context_id3_from_ih_entry entry in
      let src_name = Soc.ih_src_name soc ~client ~src in
      if not (List.mem src_name [ "SDMA_TRAP"; "CP_EOP_INTR" ]) then begin
        Printf.printf
          "am %s: IH (%#x/%#x) client=%s src=%s(%d) ring=%d vmid=%d(%d) \
           pasid=%d node=%d ctx=[%#x, %#x, %#x, %#x]\n\
           %!"
          devfmt !rptr wptr
          (match Soc.ih_client_name soc client with
          | Some name -> name
          | None -> string_of_int client)
          src_name src ring_id vmid vmid_type pasid node ctx0 ctx1 ctx2 ctx3;
        if src_name = "SQ_INTERRUPT_ID" then begin
          let is_soc21 = gc_major >= 11 in
          let enc_type =
            if is_soc21 then getbits ctx1 6 7 else getbits ctx0 26 27
          in
          let err_type =
            if is_soc21 then getbits ctx0 21 24
            else
              getbits
                ((ctx0 land 0xfff)
                lor ((ctx0 lsr 16) land 0xf000)
                lor ((ctx1 lsl 16) land 0xff0000))
                20 23
          in
          let err_info =
            if enc_type = 2 then
              Printf.sprintf " (%s)"
                (List.nth
                   [ "EDC_FUE"; "ILLEGAL_INST"; "MEMVIOL"; "EDC_FED" ]
                   err_type)
            else ""
          in
          Printf.printf "am %s: sq_intr: %s%s\n%!" devfmt
            (List.nth [ "auto"; "wave"; "error" ] enc_type)
            err_info;
          if enc_type = 2 then Amdev.set_err_state adev true
        end
        else if
          src_name = "UTCL2_FAULT"
          || (gc_major = 9 && client = Am.soc15_ih_clientid_utcl2)
        then begin
          let bf =
            Am_register.read_bitfields
              (Amdev.reg adev (Gmc.pf_status_reg gmc Gmc.Gc))
          in
          let va =
            Am_register.read
              (Amdev.reg adev "regGCVM_L2_PROTECTION_FAULT_ADDR_HI32")
            lsl 32
            lor Am_register.read
                  (Amdev.reg adev "regGCVM_L2_PROTECTION_FAULT_ADDR_LO32")
          in
          Printf.printf "am %s: GCVM_L2_PROTECTION_FAULT_STATUS: %s %#x\n%!"
            devfmt
            (String.concat ", "
               (List.map (fun (k, v) -> Printf.sprintf "%s=%d" k v) bf))
            (va lsl 12);
          Am_register.update
            (Amdev.reg adev "regGCVM_L2_PROTECTION_FAULT_CNTL")
            [ ("clear_protection_fault_status_addr", 1) ];
          Amdev.set_err_state adev true
        end
        else Amdev.set_err_state adev true
      end
    done;
    drain t;
    let bif_intr =
      Am_register.read_bitfields
        (Amdev.reg adev "regBIF_BX0_BIF_DOORBELL_INT_CNTL")
    in
    let athub_err = List.assoc "ras_athub_err_event_interrupt_status" bif_intr in
    let cntlr_err = List.assoc "ras_cntlr_interrupt_status" bif_intr in
    if athub_err <> 0 || cntlr_err <> 0 then begin
      Printf.printf "am %s: fatal hardware error detected: %s%s\n%!" devfmt
        (if athub_err <> 0 then "RAS_ATHUB_ERR_EVENT " else "")
        (if cntlr_err <> 0 then "RAS_CNTLR" else "");
      let acas =
        Smu.aca_read_banks smu ~ue:true @ Smu.aca_read_banks smu ~ue:false
      in
      List.iter
        (fun regs ->
          let reg1 = List.nth regs 1 and reg5 = List.nth regs 5 in
          let bit w n =
            Int64.logand (Int64.shift_right_logical w n) 1L <> 0L
          in
          let acatyp =
            if bit reg1 61 && bit reg1 57 then "Uncorrectable"
            else "Correctable"
          in
          let hwid =
            Int64.to_int (Int64.logand (Int64.shift_right_logical reg5 32) 0xFFFL)
          in
          Printf.printf "am %s: %s ACA: %s (%#03x) mcatype=%#06x regs=[%s]\n%!"
            devfmt acatyp
            (match List.assoc_opt hwid Am.hwid_names with
            | Some name -> name
            | None -> "")
            hwid
            (Int64.to_int
               (Int64.logand (Int64.shift_right_logical reg5 48) 0xFFFFL))
            (String.concat ", " (List.map (Printf.sprintf "%#Lx") regs)))
        acas;
      Am_register.write
        (Amdev.reg adev "regBIF_BX0_BIF_DOORBELL_INT_CNTL")
        [
          ("ras_cntlr_interrupt_clear", cntlr_err);
          ("ras_athub_err_event_interrupt_clear", athub_err);
        ];
      Amdev.set_err_state adev true
    end
end

(* Sdma: ip.py AM_SDMA *)

module Sdma = struct
  type t = {
    adev : Amdev.t;
    (* "F32" or "MCU": the engine's control-thread name, which prefixes
       its halt and poll-enable fields. *)
    sdma_name : string;
    (* Register-name prefixes of the queues brought up so far, in setup
       order. Engine selection by register instance (the pre-5.0
       generations) is not represented until the register layer
       addresses instances. *)
    mutable sdma_reginst : string list;
  }

  (* ip.py:499 AM_SDMA.init_sw *)
  let create adev =
    {
      adev;
      sdma_name =
        (if Amdev.ip_ver adev Am.sdma0_hwip < (7, 0, 0) then "F32" else "MCU");
      sdma_reginst = [];
    }

  (* ip.py:500 AM_SDMA.init_hw *)
  let init_hw t ~soc =
    let adev = t.adev in
    let sdma_ver = Amdev.ip_ver adev Am.sdma0_hwip in
    (* The old generations number the engines as register instances,
       the new ones as register-name suffixes. *)
    let pipe_cnt = if sdma_ver < (5, 0, 0) then 16 else 1 in
    for pipe_id = 0 to pipe_cnt - 1 do
      let pipe = if sdma_ver < (5, 0, 0) then "" else string_of_int pipe_id in
      let r name = Amdev.reg adev (Printf.sprintf "regSDMA%s_%s" pipe name) in
      if sdma_ver >= (6, 0, 0) then begin
        (* 10s, 100ms per unit *)
        Am_register.update (r "WATCHDOG_CNTL") [ ("queue_hang_count", 100) ];
        Am_register.update (r "UTCL1_CNTL")
          [ ("resp_mode", 3); ("redo_delay", 9) ];
        (* rd=noa, wr=bypass *)
        Am_register.update (r "UTCL1_PAGE")
          ([ ("rd_l2_policy", 2); ("wr_l2_policy", 3) ]
          @ if t.sdma_name = "F32" then [ ("llc_noalloc", 1) ] else []);
        Am_register.update
          (r (t.sdma_name ^ "_CNTL"))
          [
            ("halt", 0);
            ((if t.sdma_name = "F32" then "th1_reset" else "reset"), 0);
          ]
      end;
      Am_register.update (r "CNTL")
        (("trap_enable", 1)
        :: (if sdma_ver <= (5, 2, 0) then [ ("utc_l1_enable", 1) ] else []))
    done;
    if List.mem (Amdev.ip_ver adev Am.nbio_hwip) [ (7, 9, 0); (7, 9, 1) ]
    then
      for aid_id = 0 to 3 do
        List.iteri
          (fun dev_inst (port, awid, offset, awaddr) ->
            let entry = dev_inst + 1 + (4 * aid_id) in
            Am_register.write
              (Amdev.reg adev
                 (Printf.sprintf "regDOORBELL0_CTRL_ENTRY_%d" entry))
              [
                (Printf.sprintf "bif_doorbell%d_range_size_entry" entry, 20);
                ( Printf.sprintf "bif_doorbell%d_range_offset_entry" entry,
                  (Am.amdgpu_navi10_doorbell_sdma_engine0 + ((entry - 1) * 0xA))
                  * 2 );
              ];
            Soc.doorbell_enable soc ~port ~awid ~awaddr_31_28_value:awaddr
              ~offset ~size:4 ~aid:aid_id ())
          [ (1, 0xe, 0xe, 0x1); (2, 0x8, 0x8, 0x2); (5, 0x9, 0x9, 0x8);
            (6, 0xa, 0xa, 0x9) ]
      done
    else
      Soc.doorbell_enable soc ~port:2 ~awid:0xe ~awaddr_31_28_value:0x3
        ~offset:(Am.amdgpu_navi10_doorbell_sdma_engine0 * 2) ~size:4 ()

  (* ip.py:537 setup_ring *)
  let setup_ring t ~ring_addr ~ring_size ~rptr_addr ~wptr_addr ~idx =
    let adev = t.adev in
    let sdma_ver = Amdev.ip_ver adev Am.sdma0_hwip in
    let sdma_ma, sdma_mi, _ = sdma_ver in
    if sdma_ver >= (5, 0, 0) && idx > 0 then
      failwith
        (Printf.sprintf "am %s: sdma queue %d is not available"
           (Amdev.devfmt adev) idx);
    let pipe, queue = (idx / 4, idx mod 4) in
    let reg =
      if (sdma_ma, sdma_mi) = (4, 4) then "regSDMA_GFX"
      else Printf.sprintf "regSDMA%d_QUEUE%d" pipe queue
    in
    let doorbell =
      Am.amdgpu_navi10_doorbell_sdma_engine0 + ((pipe + (queue * 4)) * 0xA)
    in
    t.sdma_reginst <- t.sdma_reginst @ [ reg ];
    let r name = Amdev.reg adev (reg ^ name) in
    Am_register.write (r "_MINOR_PTR_UPDATE") ~value:0x1 [];
    Amdev.wreg_pair adev (reg ^ "_RB_RPTR") ~lo:"" ~hi:"_HI" 0;
    Amdev.wreg_pair adev (reg ^ "_RB_WPTR") ~lo:"" ~hi:"_HI" 0;
    Amdev.wreg_pair adev (reg ^ "_RB_BASE") ~lo:"" ~hi:"_HI" (ring_addr lsr 8);
    Amdev.wreg_pair adev (reg ^ "_RB_RPTR_ADDR") ~lo:"_LO" ~hi:"_HI" rptr_addr;
    Amdev.wreg_pair adev
      (reg ^ "_RB_WPTR_POLL_ADDR")
      ~lo:"_LO" ~hi:"_HI" wptr_addr;
    Am_register.update (r "_DOORBELL_OFFSET") [ ("offset", doorbell * 2) ];
    Am_register.update (r "_DOORBELL") [ ("enable", 1) ];
    Am_register.write (r "_MINOR_PTR_UPDATE") ~value:0x0 [];
    Am_register.write (r "_RB_CNTL")
      ((if (sdma_ma, sdma_mi) <> (4, 4) then
          [ (String.lowercase_ascii t.sdma_name ^ "_wptr_poll_enable", 1) ]
        else [])
      @ [
          ("rb_vmid", 0); ("rptr_writeback_enable", 1);
          ("rptr_writeback_timer", 4); ("rb_enable", 1); ("rb_priv", 1);
          ("rb_size", bit_length (ring_size / 4) - 1);
        ]);
    Am_register.update (r "_IB_CNTL") [ ("ib_enable", 1) ];
    doorbell

  (* ip.py:525 AM_SDMA.fini_hw *)
  let fini_hw t =
    let adev = t.adev in
    List.iter
      (fun reg ->
        Am_register.update
          (Amdev.reg adev (reg ^ "_RB_CNTL"))
          [ ("rb_enable", 0) ];
        Am_register.update
          (Amdev.reg adev (reg ^ "_IB_CNTL"))
          [ ("ib_enable", 0) ];
        Am_register.update (Amdev.reg adev (reg ^ "_DOORBELL")) [ ("enable", 0) ];
        Am_register.update
          (Amdev.reg adev (reg ^ "_DOORBELL_OFFSET"))
          [ ("offset", 0) ])
      t.sdma_reginst;
    if Amdev.ip_ver adev Am.sdma0_hwip >= (6, 0, 0) then begin
      Am_register.write
        (Amdev.reg adev "regGRBM_SOFT_RESET")
        [ ("soft_reset_sdma0", 1) ];
      sleep_ms adev 10;
      Am_register.write (Amdev.reg adev "regGRBM_SOFT_RESET") ~value:0x0 []
    end
end
