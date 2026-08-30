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

(* ip.py:85 AM_GMC.flush_hdp *)
let flush_hdp adev =
  Amdev.wreg adev
    (Am_register.read (Amdev.reg adev "regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL")
    / 4)
    0x0

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
    flush_hdp t.adev

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
