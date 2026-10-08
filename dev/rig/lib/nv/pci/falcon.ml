(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Field

let strf = Printf.sprintf

module L = Defs.Legacy
module B = Defs.Blackwell

(* Register sequences *)

type cond = Is of int | Not of int | Differs of int

type op =
  | Write of int * int
  | Modify of int * int * int
  | Copy of int * int
  | Poll of string * int * int * cond
  | Delay of int
  | When of int * int * cond * op list
  | Expect of string * int * int * cond

(* How long a poll waits: the GPU's boot steps take milliseconds; a step that
   takes 30 s has failed. *)
let poll_ms = 30_000

let meets c r mask = function
  | Is x -> Chip.get c r land mask = x
  | Not x -> Chip.get c r land mask <> x
  | Differs r' -> Chip.get c r <> Chip.get c r'

let rec run c = function
  | [] -> Ok ()
  | op :: ops -> (
      let continue () = run c ops in
      match op with
      | Write (r, x) ->
          Chip.set c r x;
          continue ()
      | Modify (r, mask, x) ->
          Chip.set c r (Chip.get c r land lnot mask lor (x land mask));
          continue ()
      | Copy (r, r') ->
          Chip.set c r' (Chip.get c r);
          continue ()
      | Delay us ->
          Chip.delay c ((us + 999) / 1000);
          continue ()
      | Poll (what, r, mask, cond) -> (
          match
            Chip.wait c what ~ms:poll_ms (fun () -> meets c r mask cond)
          with
          | Ok () -> continue ()
          | Error _ as e -> e)
      | When (r, mask, cond, then_) -> (
          let taken = if meets c r mask cond then run c then_ else Ok () in
          match taken with Ok () -> continue () | Error _ as e -> e)
      | Expect (what, r, mask, cond) ->
          if meets c r mask cond then continue ()
          else Error (strf "%s: register 0x%x reads 0x%x" what r (Chip.get c r))
      )

(* Fields *)

let lo32 x = x land 0xffff_ffff
let hi32 x = (x lsr 32) land 0xffff_ffff

(* Falcons *)

let gsp = Defs.nv_pgsp
let sec2 = Defs.nv_psec

let wait_reset : Chip.family -> op list = function
  | Blackwell ->
      [
        Poll
          ( "the FSP's boot",
            B.nv_therm_i2cs_scratch,
            mask B.nv_therm_i2cs_scratch_data,
            Is Defs.nv_therm_i2cs_scratch_fsp_boot_complete_status_success );
      ]
  | Ampere | Ada ->
      [
        Poll
          ( "the GPU's boot to unlock its progress",
            L.nv_pgc6_aon_secure_scratch_group_05_priv_level_mask,
            mask
              L
              .nv_pgc6_aon_secure_scratch_group_05_priv_level_mask_read_protection_level0,
            Is
              (put
                 L
                 .nv_pgc6_aon_secure_scratch_group_05_priv_level_mask_read_protection_level0
                 Defs
                 .nv_pgc6_aon_secure_scratch_group_05_priv_level_mask_read_protection_level0_enable)
          );
        Poll
          ( "the GPU's boot",
            L.nv_pgc6_aon_secure_scratch_group_05 0,
            mask L.nv_pgc6_aon_secure_scratch_group_05_0_gfw_boot_progress,
            Is
              Defs
              .nv_pgc6_aon_secure_scratch_group_05_0_gfw_boot_progress_completed
          );
      ]

(* How long a falcon's reset is held. *)
let reset_us = 100_000

let reset base core =
  let engine =
    if base = gsp then L.nv_pgsp_falcon_engine else L.nv_psec_falcon_engine
  in
  let hwcfg2 = base + L.nv_pfalcon_falcon_hwcfg2 in
  let bcr = base + L.nv_priscv_riscv_bcr_ctrl in
  let reset x = Write (engine, put L.nv_pgsp_falcon_engine_reset x) in
  [
    reset 1;
    Delay reset_us;
    reset 0;
    Poll
      ( "the falcon's memories to be scrubbed",
        hwcfg2,
        mask L.nv_pfalcon_falcon_hwcfg2_mem_scrubbing,
        Is Defs.nv_pfalcon_falcon_hwcfg2_mem_scrubbing_done );
  ]
  @
  match core with
  | `Riscv ->
      [
        Write
          ( bcr,
            put L.nv_priscv_riscv_bcr_ctrl_core_select
              Defs.nv_priscv_riscv_bcr_ctrl_core_select_riscv
            lor put L.nv_priscv_riscv_bcr_ctrl_brfetch 1 );
      ]
  | `Falcon ->
      (* A falcon with a RISC-V core starts on it: select the falcon, and give
         it the chip's identity. *)
      [
        When
          ( hwcfg2,
            mask L.nv_pfalcon_falcon_hwcfg2_riscv,
            Not 0,
            [
              Write (bcr, 0);
              Poll
                ( "the falcon core's selection",
                  bcr,
                  mask L.nv_priscv_riscv_bcr_ctrl_valid,
                  Is
                    (put L.nv_priscv_riscv_bcr_ctrl_valid
                       Defs.nv_priscv_riscv_bcr_ctrl_valid_true) );
              Copy (Defs.nv_pmc_boot_0, base + L.nv_pfalcon_falcon_rm);
            ] );
      ]

let start base =
  let cpuctl = base + L.nv_pfalcon_falcon_cpuctl in
  let alias = mask L.nv_pfalcon_falcon_cpuctl_alias_en in
  [
    When
      ( cpuctl,
        alias,
        Not 0,
        [
          Write
            ( base + L.nv_pfalcon_falcon_cpuctl_alias,
              put L.nv_pfalcon_falcon_cpuctl_alias_startcpu 1 );
        ] );
    When
      ( cpuctl,
        alias,
        Is 0,
        [ Write (cpuctl, put L.nv_pfalcon_falcon_cpuctl_startcpu 1) ] );
  ]

let wait_halt base =
  [
    Poll
      ( "the falcon to halt",
        base + L.nv_pfalcon_falcon_cpuctl,
        mask L.nv_pfalcon_falcon_cpuctl_halted,
        Not 0 );
  ]

(* DMA *)

type section = { off : int; pa : int; va : int; size : int }

(* A DMA command moves 256 bytes (DMATRFCMD_SIZE_256B). *)
let chunk = 256

let dma base ~imem ~image s =
  let r x = base + x in
  let cmd = r L.nv_pfalcon_falcon_dmatrfcmd in
  let ready =
    Poll
      ( "the falcon's DMA queue",
        cmd,
        mask L.nv_pfalcon_falcon_dmatrfcmd_full,
        Is 0 )
  in
  let command =
    put L.nv_pfalcon_falcon_dmatrfcmd_size
      Defs.nv_pfalcon_falcon_dmatrfcmd_size_256b
    lor put L.nv_pfalcon_falcon_dmatrfcmd_imem (Bool.to_int imem)
    lor put L.nv_pfalcon_falcon_dmatrfcmd_sec (Bool.to_int imem)
  in
  (* The base is in 256-byte units; the falcon adds each command's offset from
     the virtual address, which tags the instruction memory. *)
  let src = image + s.off - s.va in
  let chunks =
    List.init
      ((s.size + chunk - 1) / chunk)
      (fun i ->
        let x = i * chunk in
        [
          ready;
          Write (r L.nv_pfalcon_falcon_dmatrfmoffs, s.pa + x);
          Write (r L.nv_pfalcon_falcon_dmatrffboffs, s.va + x);
          Write (cmd, command);
        ])
  in
  [
    ready;
    Write (r L.nv_pfalcon_falcon_dmatrfbase, lo32 (src lsr 8));
    Write (r L.nv_pfalcon_falcon_dmatrfbase1, hi32 (src lsr 8) land 0x1ff);
  ]
  @ List.concat chunks
  @ [
      Poll
        ("the falcon's DMA", cmd, mask L.nv_pfalcon_falcon_dmatrfcmd_idle, Not 0);
    ]

(* Heavy-secure ucodes *)

type hs = {
  image : int;
  code : section;
  data : section;
  pkc : int;
  engines : int;
  ucode : int;
}

let hs base ?mailbox u =
  let r x = base + x in
  let mailboxes =
    match mailbox with
    | None -> []
    | Some m ->
        [
          Write (r L.nv_pfalcon_falcon_mailbox0, lo32 m);
          Write (r L.nv_pfalcon_falcon_mailbox1, hi32 m);
        ]
  in
  [
    (* The falcon reaches the GPU's memory at physical addresses, without a
       context. *)
    Modify
      ( r L.nv_pfalcon_fbif_ctl,
        mask L.nv_pfalcon_fbif_ctl_allow_phys_no_ctx,
        put L.nv_pfalcon_fbif_ctl_allow_phys_no_ctx 1 );
    Write (r L.nv_pfalcon_falcon_dmactl, 0);
    Modify
      ( r (L.nv_pfalcon_fbif_transcfg 0),
        mask L.nv_pfalcon_fbif_transcfg_target
        lor mask L.nv_pfalcon_fbif_transcfg_mem_type,
        put L.nv_pfalcon_fbif_transcfg_mem_type
          Defs.nv_pfalcon_fbif_transcfg_mem_type_physical );
  ]
  @ dma base ~imem:true ~image:u.image u.code
  @ dma base ~imem:false ~image:u.image u.data
  @ [
      Write (r (L.nv_pfalcon2_falcon_brom_paraaddr 0), u.pkc);
      Write (r L.nv_pfalcon2_falcon_brom_engidmask, u.engines);
      Write
        ( r L.nv_pfalcon2_falcon_brom_curr_ucode_id,
          put L.nv_pfalcon2_falcon_brom_curr_ucode_id_val u.ucode );
      Write
        ( r L.nv_pfalcon2_falcon_mod_sel,
          put L.nv_pfalcon2_falcon_mod_sel_algo
            Defs.nv_pfalcon2_falcon_mod_sel_algo_rsa3k );
      Write (r L.nv_pfalcon_falcon_bootvec, u.code.va);
    ]
  @ mailboxes @ start base @ wait_halt base

(* Boots *)

let all = 0xffff_ffff

let legacy ~fwsec ~booter ~libos ~wpr_meta =
  reset gsp `Falcon @ hs gsp fwsec
  @ [
      Expect
        ( "FWSEC did not set up the GPU's protected memory (WPR2)",
          Defs.nv_pfb_pri_mmu_wpr2_addr_hi,
          all,
          Not 0 );
    ]
  @ reset gsp `Riscv
  @ [
      Write (L.nv_pgsp_falcon_mailbox0, lo32 libos);
      Write (L.nv_pgsp_falcon_mailbox1, hi32 libos);
    ]
  @ reset sec2 `Falcon
  @ hs sec2 ~mailbox:wpr_meta booter
  @ [
      Expect
        ("the booter failed", sec2 + L.nv_pfalcon_falcon_mailbox0, all, Is 0);
      Write (gsp + L.nv_pfalcon_falcon_os, 0);
      Expect
        ( "the GSP's core does not run",
          gsp + L.nv_priscv_riscv_cpuctl,
          mask L.nv_priscv_riscv_cpuctl_active_stat,
          Is
            (put L.nv_priscv_riscv_cpuctl_active_stat
               Defs.nv_priscv_riscv_cpuctl_active_stat_active) );
    ]

(* Encodings *)

let cot_args ~libos ~wpr_meta =
  let module F = Defs.Fmc_boot_params in
  let module A = Defs.Acr_boot_params in
  let module R = Defs.Rm_params in
  let b = Bytes.make F.sizeof '\000' in
  let acr = at F.boot_gsp_rm_params and rm = at F.gsp_rm_params in
  set b (acr A.gsp_rm_desc_offset) wpr_meta;
  set b (acr A.gsp_rm_desc_size) Defs.Wpr_meta.sizeof;
  set b (acr A.target) Defs.gsp_dma_target_coherent_system;
  set b (acr A.b_is_gsp_rm_boot) 1;
  set b (rm R.boot_args_offset) libos;
  set b (rm R.target) Defs.gsp_dma_target_coherent_system;
  Bytes.unsafe_to_string b

(* The COT payload's version, NVDM_PAYLOAD_COT's 2 in kern_fsp_cot_payload.h's
   FSP COT interface. *)
let cot_version = 2

let cot_payload ~args ~fmc (m : Images.fmc) =
  let module C = Defs.Cot_payload in
  let b = Bytes.make C.sizeof '\000' in
  set b C.version cot_version;
  set b C.size C.sizeof;
  set b C.frts_vidmem_offset (fst Layout.cot_frts);
  set b C.frts_vidmem_size (snd Layout.cot_frts);
  set b C.gsp_boot_args_sysmem_offset args;
  set b C.gsp_fmc_sysmem_offset fmc;
  let words (off, _, n) (r : Images.range) =
    (* An array of 32-bit words: a part not a whole number of words is padded
       with zeros, as the public key's 97 bytes are. *)
    let len = Int.min r.length (4 * n) in
    Bytes.blit_string r.contents r.at b off len
  in
  words C.hash384 m.hash;
  words C.public_key m.public_key;
  words C.signature m.signature;
  Bytes.unsafe_to_string b

(* The FSP's queue *)

(* An NVDM message's MCTP headers (fsp_mctp_format.h): the transport header of a
   message in one packet, its start and end, and the message header of NVIDIA's
   vendor-defined type [kind]. *)
let header kind =
  let b = Bytes.create 8 in
  let word off x = Bytes.set_int32_le b off (Int32.of_int x) in
  word 0 (put Defs.mctp_header_som 1 lor put Defs.mctp_header_eom 1);
  word 4
    (put Defs.mctp_msg_header_type Defs.mctp_msg_header_type_vendor_pci
    lor put Defs.mctp_msg_header_vendor_id Defs.mctp_msg_header_vendor_id_nv
    lor put Defs.mctp_msg_header_nvdm_type kind);
  Bytes.unsafe_to_string b

(* The FSP's message queue holds less than 1 KiB. *)
let fsp_max = 0x400

let fsp kind payload =
  let pad = 4 - (String.length payload mod 4) in
  let msg = header kind ^ payload ^ String.make pad '\000' in
  if String.length msg >= fsp_max then
    invalid_arg
      (strf "Falcon.fsp: a message of %d bytes, the queue holds %d"
         (String.length msg) fsp_max);
  let emem = B.nv_pfsp_ememc 0 and data = B.nv_pfsp_ememd 0 in
  let config ~write ~read =
    put B.nv_pfsp_ememc_aincw (Bool.to_int write)
    lor put B.nv_pfsp_ememc_aincr (Bool.to_int read)
  in
  let word i = Int32.to_int (String.get_int32_le msg (4 * i)) land all in
  [ Write (emem, config ~write:true ~read:false) ]
  @ List.init (String.length msg / 4) (fun i -> Write (data, word i))
  @ [
      Write (B.nv_pfsp_queue_tail 0, String.length msg - 4);
      Write (B.nv_pfsp_queue_head 0, 0);
      Poll
        ( "the FSP's answer",
          B.nv_pfsp_msgq_head 0,
          all,
          Differs (B.nv_pfsp_msgq_tail 0) );
      Write (emem, config ~write:false ~read:true);
      Copy (B.nv_pfsp_msgq_head 0, B.nv_pfsp_msgq_tail 0);
    ]

let cot payload =
  fsp Defs.nvdm_type_cot payload
  @ [
      Poll
        ( "the GSP's boot",
          gsp + B.nv_pfalcon_falcon_hwcfg2,
          mask B.nv_pfalcon_falcon_hwcfg2_riscv_br_priv_lockdown,
          Is 0 );
    ]
