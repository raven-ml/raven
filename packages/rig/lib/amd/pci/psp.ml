(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Defs
module Window = Rig_pci.Window
module Page_table = Rig_pci.Page_table
module Psp = Rig_amd_abi.Psp

let strf = Printf.sprintf
let lo32 v = v land 0xffff_ffff
let hi32 v = (v lsr 32) land 0xffff_ffff

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
  mp0 : Rig_amd_abi.Discovery.version;
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
let message p (data : Rig_amd_abi.Firmware.piece) =
  let n = data.len in
  if n > Window.length p.msg then
    raise (Regs.Stuck (strf "a message of %d bytes exceeds the PSP's buffer" n));
  Window.blit_string data.source data.off p.msg 0 n;
  Window.fill p.msg n (Window.length p.msg - n) '\000';
  Window.flush p.msg;
  Gmc.flush_hdp p.gmc

let wait_bootloader p =
  Regs.wait p.r "the PSP's bootloader" (fun () -> read p 35 land ready <> 0)

let bootloader_load p images (fw_type, step) =
  match List.assoc_opt fw_type images.Rig_amd_abi.Firmware.sos with
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
    Psp.frame ~command:(Gmc.mc p.gmc p.m.command)
      ~fence:(Gmc.mc p.gmc p.m.fence) ~value:v
  in
  Window.write p.vram (p.m.ring + at) f;
  Window.flush p.vram;
  Gmc.flush_hdp p.gmc;
  write p 67 ((wptr + (String.length f / 4)) mod (ring_bytes / 4));
  Regs.wait p.r (strf "the PSP's command %s" what) (fun () ->
      Window.get32 p.vram p.m.fence = v);
  let resp = Window.read p.vram p.m.command (String.length cmd) in
  let s = Psp.status resp in
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
           (Psp.load_ip_fw ~at ~bytes:data.len ~fw_type:ty)))
    types

(* The TMR: its size from the table of contents, or from the scratch register
   the last boot left it in, and its memory the first runtime allocation of both
   boots, so that it keeps its address. *)
let tmr_init p images ~partial =
  (if partial then p.tmr_size <- Regs.read p.r "regSCRATCH_REG5"
   else
     match
       List.assoc_opt D.psp_fw_type_psp_toc images.Rig_amd_abi.Firmware.sos
     with
     | None -> ()
     | Some toc ->
         message p toc;
         let resp =
           submit p "load the table of contents"
             (Psp.load_toc ~at:(Gmc.mc p.gmc p.m.message) ~bytes:toc.len)
         in
         p.tmr_size <- Psp.tmr_bytes resp);
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
  ignore (submit p "set up the TMR" (Psp.setup_tmr ~at ~fabric ~bytes))

(* A ring left by the last boot is destroyed first; the processor answers each
   control command in bit 31 of the control register, after 20 ms, as the kernel
   waits before it polls (psp_v14_0_ring_create, psp_v14_0_ring_stop). *)
let settle_us = 20_000

let ring_create p =
  if read p 71 <> 0 then begin
    write p 64 D.gfx_ctrl_cmd_id_destroy_rings;
    Regs.pause settle_us
  end;
  Regs.wait p.r "the PSP's OS" (fun () -> read p 64 land ready <> 0);
  let mc = Gmc.mc p.gmc p.m.ring in
  write p 69 (lo32 mc);
  write p 70 (hi32 mc);
  write p 71 ring_bytes;
  write p 64 (D.psp_ring_type__km lsl 16);
  Regs.pause settle_us;
  Regs.wait p.r "the PSP's ring" (fun () -> read p 64 land 0x8000_ffff = ready);
  p.fence_value <- Window.get32 p.vram p.m.fence

let os p images =
  if not (alive p) then begin
    List.iter (bootloader_load p images) Psp.steps;
    Regs.wait p.r "the PSP's OS to start" (fun () -> alive p)
  end;
  ring_create p;
  tmr_init p images ~partial:false

let power_firmware p images =
  Option.iter (load p) images.Rig_amd_abi.Firmware.smu

let firmware p images =
  power_firmware p images;
  if (not (boot_time_tmr p)) || not (autoload_tmr p) then tmr_load p;
  List.iter (load p) images.pieces;
  let gc = Regs.version (Regs.layout_of p.r) D.gc_hwid in
  if gc >= (11, 0, 0) then ignore (submit p "autoload the RLC" Psp.autoload_rlc)
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
  ignore (submit p "set the partition mode" (Psp.partition ~mode))
