(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Commands, frames and statuses are laid out as the kernel's psp_gfx_if.h
   lays them out. *)

module D = Boot_defs

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

(* The bootloader *)

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
