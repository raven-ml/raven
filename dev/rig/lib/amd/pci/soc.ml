(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module D = Defs

let strf = Printf.sprintf

let nbio79 r =
  match Regs.version (Regs.layout_of r) D.nbif_hwid with
  | 7, 9, (0 | 1) -> true
  | _ -> false

let gc r = Regs.version (Regs.layout_of r) D.gc_hwid

(* Dies past the eighth have no doorbell fence bit. *)
let fence_dies = 8

(* The register BAR's hole the HDP flush registers are remapped to:
   MMIO_REG_HOLE_OFFSET, 0x1a000 on NBIO 7.9 and the page below 512 KiB on the
   others (nbio_v7_9.c, nbio_v4_3.c, nbif_v6_3_1.c, for 4 KiB pages). The memory
   flush is its first word, the register flush its second
   (KFD_MMIO_REMAP_HDP_MEM_FLUSH_CNTL, _REG_FLUSH_CNTL of kfd_ioctl.h). *)
let hole r = if nbio79 r then 0x1_a000 else 0x8_0000 - 0x1000
let hdp_flush r = hole r
let reg_flush = 4

let remap_hdp r =
  Regs.write ~value:(hole r) r "regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL" [];
  Regs.write
    ~value:(hole r + reg_flush)
    r "regBIF_BX0_REMAP_HDP_REG_FLUSH_CNTL" []

(* The dummy read the handler makes before an interrupt it does not send by MSI;
   snooped, as the kernel sets it for every ring. *)
let interrupts r ~dummy =
  Regs.write ~value:(dummy lsr 8) r "regBIF_BX0_INTERRUPT_CNTL2" [];
  Regs.update r "regBIF_BX0_INTERRUPT_CNTL"
    [ ("ih_dummy_rd_override", 0); ("ih_req_nonsnoop_en", 0) ]

let start r =
  if nbio79 r then begin
    let d = Regs.discovery (Regs.layout_of r) in
    let fused =
      Option.value ~default:[] (List.assoc_opt D.gc_hwid d.harvested)
    in
    let live =
      List.fold_left
        (fun acc (i, _) ->
          if List.mem i fused || i >= fence_dies then acc else acc lor (1 lsl i))
        0
        (Option.value ~default:[] (List.assoc_opt D.gc_hwid d.bases))
    in
    Regs.write ~value:(0xff land lnot live) r "regXCC_DOORBELL_FENCE" [];
    let fence = Regs.register (Regs.layout_of r) "regXCC_DOORBELL_FENCE" in
    List.iter
      (fun aid ->
        Regs.set_pcie ~aid r
          (Regs.address (Regs.layout_of r) "regXCC_DOORBELL_FENCE")
          (Rig_amd_abi.Register.encode fence [ ("shub_slv_mode", 1) ]))
      (List.tl (Discovery.aids d));
    Regs.write ~value:0x7ff r "regBIFC_GFX_INT_MONITOR_MASK" [];
    Regs.write ~value:0xf_ffff r "regBIFC_DOORBELL_ACCESS_EN_PF" []
  end
  else
    Regs.update r "regRCC_DEV0_EPF2_STRAP2"
      [ ("strap_no_soft_reset_dev0_f2", 0) ];
  Regs.write ~value:1 r "regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN" [];
  if not (Regs.vf r) then remap_hdp r

let route ?(aid = 0) ?(offset = 0) ?(size = 0) r ~port ~awid ~awaddr =
  let name =
    strf "%s_DOORBELL_ENTRY_%d_CTRL"
      (if gc r >= (12, 0, 0) then "regGDC_S2A0_S2A" else "regS2A")
      port
  in
  let f s = strf "s2a_doorbell_port%d_%s" port s in
  let l = Regs.layout_of r in
  let v =
    Rig_amd_abi.Register.encode (Regs.register l name)
      [
        (f "enable", 1);
        (f "awid", awid);
        (f "range_size", size);
        (f "awaddr_31_28_value", awaddr);
        (f "range_offset", offset);
      ]
  in
  if nbio79 r then Regs.set_pcie ~aid r (Regs.address l name) v
  else Regs.write ~value:v r name []

let gate r =
  if Regs.version (Regs.layout_of r) D.hdp_hwid >= (5, 2, 1) then
    Regs.update r "regHDP_MEM_POWER_CTRL"
      [ ("atomic_mem_power_ctrl_en", 1); ("atomic_mem_power_ds_en", 1) ]
