(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Defs
module Window = Rig_pci.Window
module Page_table = Rig_pci.Page_table

let strf = Printf.sprintf
let lo32 v = v land 0xffff_ffff
let hi32 v = (v lsr 32) land 0xffff_ffff

(* Commands *)

(* [command id fields] is a command buffer of command [id] with each [(field,
   value)] of its command struct set. *)
let command id fields =
  let b = Bytes.make D.psp_command_bytes '\000' in
  let set (off, n) v =
    match n with
    | 4 -> Bytes.set_int32_le b off (Int32.of_int (lo32 v))
    | n -> invalid_arg (strf "Psp: a field of %d bytes" n)
  in
  set D.psp_command_id id;
  List.iter (fun ((off, n), v) -> set (D.psp_command_at + off, n) v) fields;
  b

let load_ip_fw ~at ~bytes ~fw_type:ty =
  let open D.Psp_gfx_cmd_load_ip_fw in
  Bytes.to_string
    (command D.gfx_cmd_id_load_ip_fw
       [
         (fw_phy_addr_lo, lo32 at);
         (fw_phy_addr_hi, hi32 at);
         (fw_size, bytes);
         (fw_type, ty);
       ])

let load_toc ~at ~bytes =
  let open D.Psp_gfx_cmd_load_toc in
  Bytes.to_string
    (command D.gfx_cmd_id_load_toc
       [
         (toc_phy_addr_lo, lo32 at);
         (toc_phy_addr_hi, hi32 at);
         (toc_size, bytes);
       ])

let setup_tmr ~at ~fabric ~bytes =
  let open D.Psp_gfx_cmd_setup_tmr in
  let b =
    command D.gfx_cmd_id_setup_tmr
      [
        (buf_phy_addr_lo, lo32 at);
        (buf_phy_addr_hi, hi32 at);
        (buf_size, bytes);
        (system_phy_addr_lo, lo32 fabric);
        (system_phy_addr_hi, hi32 fabric);
      ]
  in
  (* The driver passes both addresses. *)
  let bit, _ = virt_phy_addr in
  let i = D.psp_command_at + (bit / 8) in
  Bytes.set_uint8 b i (Bytes.get_uint8 b i lor (1 lsl (bit mod 8)));
  Bytes.to_string b

let autoload_rlc = Bytes.to_string (command D.gfx_cmd_id_autoload_rlc [])

let partition ~mode =
  Bytes.to_string
    (command D.gfx_cmd_id_sriov_spatial_part
       [ (D.Psp_gfx_cmd_sriov_spatial_part.mode, mode) ])

let frame ~command ~fence ~value =
  let b = Bytes.make D.Psp_gfx_rb_frame.sizeof '\000' in
  let set (off, _) v = Bytes.set_int32_le b off (Int32.of_int (lo32 v)) in
  let open D.Psp_gfx_rb_frame in
  set cmd_buf_addr_lo (lo32 command);
  set cmd_buf_addr_hi (hi32 command);
  set fence_addr_lo (lo32 fence);
  set fence_addr_hi (hi32 fence);
  set fence_value value;
  Bytes.to_string b

let response b (off, _) =
  Int32.to_int (String.get_int32_le b (D.psp_response_at + off))
  land 0xffff_ffff

let status b = response b D.Psp_gfx_resp.status
let tmr_bytes b = response b D.Psp_gfx_resp.tmr_size

(* The processor *)

type memory = { message : int; command : int; fence : int; ring : int }

type t = {
  r : Regs.t;
  gmc : Gmc.t;
  vram : Window.t;
  tables : Page_table.t;
  m : memory;
  msg : Window.t; (* the message buffer *)
  prefix : string; (* of the message registers, which MP0 14 renamed *)
  mp0 : Discovery.version;
  mutable fence_value : int;
  mutable tmr : int; (* its physical address, 0 for the one set up at boot *)
  mutable tmr_size : int;
}

let ring_bytes = 0x1_0000

(* MP0 14 renamed the message registers. *)
let prefix_of mp0 =
  if mp0 < (14, 0, 0) then "regMP0_SMN_C2PMSG" else "regMPASP_SMN_C2PMSG"

let make r gmc vram tables m =
  let mp0 = Regs.version (Regs.layout_of r) D.mp0_hwid in
  {
    r;
    gmc;
    vram;
    tables;
    m;
    msg = Window.sub vram m.message D.psp_1_meg;
    prefix = prefix_of mp0;
    mp0;
    fence_value = 0;
    tmr = 0;
    tmr_size = 0;
  }

let tmr p = p.tmr_size

(* The message registers: the bootloader's status and command (35), its message
   address (36), the ring's control (64), write pointer (67), address (69, 70)
   and size (71), and the OS's sign of life (81). *)
let reg p n = strf "%s_%d" p.prefix n
let read p n = Regs.read p.r (reg p n)
let write p n v = Regs.write ~value:v p.r (reg p n) []
let ready = 0x8000_0000

let bootloader r =
  let mp0 = Regs.version (Regs.layout_of r) D.mp0_hwid in
  let v = Regs.read r (strf "%s_35" (prefix_of mp0)) in
  v <> 0xffff_ffff && v land ready <> 0

let running r =
  let mp0 = Regs.version (Regs.layout_of r) D.mp0_hwid in
  Regs.read r (strf "%s_81" (prefix_of mp0)) <> 0

let alive p = running p.r

(* The PSPs of these versions set up their TMR at boot, and those of the first
   two load the GC's firmware into it themselves. *)
let boot_time_tmr p =
  List.mem p.mp0 [ (13, 0, 6); (13, 0, 14); (14, 0, 2); (14, 0, 3) ]

let autoload_tmr p = not (List.mem p.mp0 [ (13, 0, 6); (13, 0, 14) ])

(* Writes [data] into the message buffer, the rest of the buffer zeroed as
   amdgpu does before each component, and has it reach the GPU's memory. *)
let message p data =
  let n = String.length data in
  if n > Window.length p.msg then
    raise (Regs.Stuck (strf "a message of %d bytes exceeds the PSP's buffer" n));
  Window.write p.msg 0 data;
  Window.fill p.msg n (Window.length p.msg - n) '\000';
  Window.flush p.msg;
  Gmc.flush_hdp p.gmc

let wait_bootloader p =
  Regs.wait p.r "the PSP's bootloader" (fun () -> read p 35 land ready <> 0)

let bootloader_load p images (fw_type, step) =
  match List.assoc_opt fw_type images.Images.sos with
  | None -> ()
  | Some data ->
      wait_bootloader p;
      message p data;
      write p 36 (Gmc.mc p.gmc p.m.message lsr 20);
      write p 35 step;
      if step <> D.psp_bl__load_sosdrv then wait_bootloader p

(* Places command [cmd] in the command buffer and its frame on the ring, and
   waits for the frame's fence. The write pointer counts 32-bit words, and wraps
   at the ring's end. *)
let submit p what cmd =
  let wptr = read p 67 in
  let at = wptr * 4 mod ring_bytes in
  p.fence_value <- p.fence_value + 1;
  let v = p.fence_value in
  Window.write p.vram p.m.command cmd;
  let f =
    frame ~command:(Gmc.mc p.gmc p.m.command) ~fence:(Gmc.mc p.gmc p.m.fence)
      ~value:v
  in
  Window.write p.vram (p.m.ring + at) f;
  Window.flush p.vram;
  Gmc.flush_hdp p.gmc;
  write p 67 ((wptr + (String.length f / 4)) mod (ring_bytes / 4));
  Regs.wait p.r (strf "the PSP's command %s" what) (fun () ->
      Window.get32 p.vram p.m.fence = v);
  let resp = Window.read p.vram p.m.command (String.length cmd) in
  let s = status resp in
  if s <> 0 then
    raise
      (Regs.Stuck (strf "the PSP's command %s failed with status 0x%x" what s));
  resp

let load p (types, data) =
  message p data;
  let at = Gmc.mc p.gmc p.m.message in
  List.iter
    (fun ty ->
      ignore
        (submit p
           (strf "load firmware %d" ty)
           (load_ip_fw ~at ~bytes:(String.length data) ~fw_type:ty)))
    types

(* The TMR: its size from the table of contents, or from the scratch register
   the last boot left it in, and its memory the first runtime allocation of both
   boots, so that it keeps its address. *)
let tmr_init p images ~partial =
  (if partial then p.tmr_size <- Regs.read p.r "regSCRATCH_REG5"
   else
     match List.assoc_opt D.psp_fw_type_psp_toc images.Images.sos with
     | None -> ()
     | Some toc ->
         message p toc;
         let resp =
           submit p "load the table of contents"
             (load_toc ~at:(Gmc.mc p.gmc p.m.message) ~bytes:(String.length toc))
         in
         p.tmr_size <- tmr_bytes resp);
  if (not (boot_time_tmr p)) && p.tmr_size > 0 then
    match
      Page_table.palloc ~boot:false ~align:D.psp_tmr_alignment ~zero:false
        p.tables p.tmr_size
    with
    | Some pa -> p.tmr <- pa
    | None -> raise (Regs.Stuck "no GPU memory for the PSP's TMR")

let tmr_load p =
  let at, fabric, bytes =
    if p.tmr = 0 then (0, 0, 0)
    else (Gmc.mc p.gmc p.tmr, Gmc.fabric p.gmc p.tmr, p.tmr_size)
  in
  ignore (submit p "set up the TMR" (setup_tmr ~at ~fabric ~bytes))

(* A ring left by the last boot is destroyed first; the processor answers each
   control command in bit 31 of the control register. *)
let settle_ms = 20

let ring_create p =
  if read p 71 <> 0 then begin
    write p 64 D.gfx_ctrl_cmd_id_destroy_rings;
    Regs.pause p.r settle_ms
  end;
  Regs.wait p.r "the PSP's OS" (fun () -> read p 64 land ready <> 0);
  let mc = Gmc.mc p.gmc p.m.ring in
  write p 69 (lo32 mc);
  write p 70 (hi32 mc);
  write p 71 ring_bytes;
  write p 64 (D.psp_ring_type__km lsl 16);
  Regs.pause p.r settle_ms;
  Regs.wait p.r "the PSP's ring" (fun () -> read p 64 land 0x8000_ffff = ready);
  p.fence_value <- Window.get32 p.vram p.m.fence

(* The bootloader's steps, in amdgpu_psp.c's order: each loads one of the SOS's
   components, when the image has it. *)
let steps =
  D.
    [
      (psp_fw_type_psp_kdb, psp_bl__load_key_database);
      (psp_fw_type_psp_spl, psp_bl__load_tos_spl_table);
      (psp_fw_type_psp_sys_drv, psp_bl__load_sysdrv);
      (psp_fw_type_psp_soc_drv, psp_bl__load_socdrv);
      (psp_fw_type_psp_intf_drv, psp_bl__load_intfdrv);
      (psp_fw_type_psp_dbg_drv, psp_bl__load_dbgdrv);
      (psp_fw_type_psp_ras_drv, psp_bl__load_rasdrv);
      (psp_fw_type_psp_ipkeymgr_drv, psp_bl__load_ipkeymgrdrv);
      (psp_fw_type_psp_spdm_drv, psp_bl__load_spdmdrv);
      (psp_fw_type_psp_sos, psp_bl__load_sosdrv);
    ]

let os p images =
  if not (alive p) then begin
    List.iter (bootloader_load p images) steps;
    Regs.wait p.r "the PSP's OS to start" (fun () -> alive p)
  end;
  ring_create p;
  tmr_init p images ~partial:false

let power_firmware p images = Option.iter (load p) images.Images.smu

let firmware p images =
  power_firmware p images;
  if (not (boot_time_tmr p)) || not (autoload_tmr p) then tmr_load p;
  List.iter (load p) images.pieces;
  let gc = Regs.version (Regs.layout_of p.r) D.gc_hwid in
  if gc >= (11, 0, 0) then ignore (submit p "autoload the RLC" autoload_rlc)
  else
    match List.assoc_opt D.psp_fw_type_psp_rl images.sos with
    | Some rl -> load p ([ D.gfx_fw_type_reg_list ], rl)
    | None -> raise (Regs.Stuck "the PSP's image has no register list")

let start p images ~partial =
  if partial then tmr_init p images ~partial
  else begin
    os p images;
    firmware p images
  end

let set_partition p mode =
  ignore (submit p "set the partition mode" (partition ~mode))
