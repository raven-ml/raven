(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The falcons' register sequences, against NVIDIA's register headers
   (open-gpu-kernel-modules 570.144, src/common/inc/swref/published): the GSP's
   falcon at NV_PGSP's 0x110000, SEC2 at NV_PSEC's 0x840000 (dev_gsp.h,
   dev_sec_pri.h); NV_PGSP_FALCON_ENGINE 0x1103c0 (dev_gsp.h);
   NV_PFALCON_FALCON_* (dev_falcon_v4.h), NV_PFALCON2_FALCON_*
   (dev_falcon_second_pri.h) and NV_PRISCV_RISCV_* (dev_riscv_pri.h) from the
   falcon's base, as ga102 places them. *)

open Windtrap
module Falcon = Rig_nv_pci.Falcon

(* Sequences as text, what a poll or check waits for left out. *)
let rec show (op : Falcon.op) =
  let cond : Falcon.cond -> string = function
    | Is x -> Printf.sprintf "= 0x%x" x
    | Not x -> Printf.sprintf "<> 0x%x" x
    | Differs r -> Printf.sprintf "<> [0x%x]" r
  in
  match op with
  | Write (r, x) -> Printf.sprintf "[0x%x] := 0x%x" r x
  | Modify (r, m, x) -> Printf.sprintf "[0x%x] & 0x%x := 0x%x" r m x
  | Copy (r, r') -> Printf.sprintf "[0x%x] := [0x%x]" r' r
  | Poll (_, r, m, c) -> Printf.sprintf "wait [0x%x] & 0x%x %s" r m (cond c)
  | Delay us -> Printf.sprintf "delay %d us" us
  | When (r, m, c, ops) ->
      Printf.sprintf "when [0x%x] & 0x%x %s { %s }" r m (cond c)
        (String.concat "; " (List.map show ops))
  | Expect (_, r, m, c) -> Printf.sprintf "check [0x%x] & 0x%x %s" r m (cond c)

let same expected ops = equal (list string) expected (List.map show ops)

(* HWCFG2 at 0xf4: MEM_SCRUBBING bit 12, RISCV bit 10; BCR_CTRL at 0x1668: VALID
   bit 0, CORE_SELECT bit 4, BRFETCH bit 8; NV_PFALCON_FALCON_RM at 0x84 takes
   NV_PMC_BOOT_0 (0). *)
let test_reset () =
  let common =
    [
      "[0x1103c0] := 0x1";
      "delay 100000 us";
      "[0x1103c0] := 0x0";
      "wait [0x1100f4] & 0x1000 = 0x0";
    ]
  in
  same
    (common
    @ [
        "when [0x1100f4] & 0x400 <> 0x0 { [0x111668] := 0x0; wait [0x111668] & \
         0x1 = 0x1; [0x110084] := [0x0] }";
      ])
    (Falcon.reset Falcon.gsp `Falcon);
  same (common @ [ "[0x111668] := 0x110" ]) (Falcon.reset Falcon.gsp `Riscv);
  (* SEC2's engine register, NV_PSEC_FALCON_ENGINE (dev_sec_pri.h). *)
  same [ "[0x8403c0] := 0x1" ] [ List.hd (Falcon.reset Falcon.sec2 `Falcon) ]

(* CPUCTL at 0x100: STARTCPU bit 1, HALTED bit 4, ALIAS_EN bit 6; CPUCTL_ALIAS
   at 0x130. *)
let test_start () =
  same
    [
      "when [0x840100] & 0x40 <> 0x0 { [0x840130] := 0x2 }";
      "when [0x840100] & 0x40 = 0x0 { [0x840100] := 0x2 }";
    ]
    (Falcon.start Falcon.sec2);
  same [ "wait [0x840100] & 0x10 <> 0x0" ] (Falcon.wait_halt Falcon.sec2)

(* DMATRFBASE 0x110 and DMATRFBASE1 0x128 take the source in 256-byte units,
   DMATRFMOFFS 0x114 the falcon's address, DMATRFFBOFFS 0x11c the offset from
   the base; DMATRFCMD 0x118: FULL bit 0, IDLE bit 1, SEC 3:2, IMEM bit 4, SIZE
   10:8 (256B is 6). *)
let test_dma () =
  let s = { Falcon.off = 0x100; pa = 0x40; va = 0x100; size = 0x101 } in
  let chunk x =
    [
      "wait [0x110118] & 0x1 = 0x0";
      Printf.sprintf "[0x110114] := 0x%x" (0x40 + x);
      Printf.sprintf "[0x11011c] := 0x%x" (0x100 + x);
      "[0x110118] := 0x614";
    ]
  in
  same
    ([
       "wait [0x110118] & 0x1 = 0x0";
       "[0x110110] := 0x123400";
       "[0x110128] := 0x0";
     ]
    @ chunk 0 @ chunk 0x100
    @ [ "wait [0x110118] & 0x2 <> 0x0" ])
    (Falcon.dma Falcon.gsp ~imem:true ~image:0x1234_0000 s);
  let data = Falcon.dma Falcon.gsp ~imem:false ~image:0x1234_0000 s in
  mem string "[0x110118] := 0x600" (List.map show data)

(* BROM_PARAADDR(0) 0x1210, BROM_ENGIDMASK 0x119c, BROM_CURR_UCODE_ID 0x1198,
   MOD_SEL 0x1180 (RSA3K is 1), BOOTVEC 0x104, MAILBOX0 0x40, MAILBOX1 0x44;
   FBIF_CTL 0x624 (ALLOW_PHYS_NO_CTX bit 7), DMACTL 0x10c, FBIF_TRANSCFG(0)
   0x600 (TARGET 1:0, MEM_TYPE bit 2, PHYSICAL 1). *)
let test_hs () =
  let u =
    {
      Falcon.image = 0x100_0000;
      code = { off = 0; pa = 0; va = 0x200; size = 0x100 };
      data = { off = 0x100; pa = 0; va = 0; size = 0x100 };
      pkc = 0x10;
      engines = 1;
      ucode = 3;
    }
  in
  let ops = List.map show (Falcon.hs Falcon.sec2 ~mailbox:0x1_2345_6789 u) in
  let starts =
    [
      "[0x840624] & 0x80 := 0x80";
      "[0x84010c] := 0x0";
      "[0x840600] & 0x7 := 0x4";
    ]
  in
  equal (list string) starts (List.filteri (fun i _ -> i < 3) ops);
  List.iter
    (fun w -> mem string w ops)
    [
      "[0x841210] := 0x10";
      "[0x84119c] := 0x1";
      "[0x841198] := 0x3";
      "[0x841180] := 0x1";
      "[0x840104] := 0x200";
      "[0x840040] := 0x23456789";
      "[0x840044] := 0x1";
    ];
  equal string "wait [0x840100] & 0x10 <> 0x0"
    (List.nth ops (List.length ops - 1))

(* The legacy boot: FWSEC on the GSP's falcon, a check of WPR2's upper address
   (NV_PFB_PRI_MMU_WPR2_ADDR_HI 0x1fa828, dev_fb.h), the GSP's RISC-V core given
   its libos arguments in its mailboxes (0x110040, 0x110044), the booter on SEC2
   given the WPR metadata, a check of SEC2's first mailbox, and of the GSP's
   core running (NV_PRISCV_RISCV_CPUCTL 0x1388, ACTIVE_STAT bit 7). *)
let test_legacy () =
  let u image =
    {
      Falcon.image;
      code = { off = 0; pa = 0; va = 0; size = 0x100 };
      data = { off = 0x100; pa = 0; va = 0; size = 0x100 };
      pkc = 0;
      engines = 1;
      ucode = 1;
    }
  in
  let ops =
    List.map show
      (Falcon.legacy ~fwsec:(u 0x1000) ~booter:(u 0x2000) ~libos:0x1_0000_2000
         ~wpr_meta:0x3000)
  in
  let index s =
    let rec go i = function
      | [] -> failf "no %s" s
      | x :: _ when x = s -> i
      | _ :: l -> go (i + 1) l
    in
    go 0 ops
  in
  let order =
    List.map index
      [
        "check [0x1fa828] & 0xffffffff <> 0x0";
        "[0x110040] := 0x2000";
        "[0x110044] := 0x1";
        "[0x840040] := 0x3000";
        "check [0x840040] & 0xffffffff = 0x0";
        "[0x110080] := 0x0";
        "check [0x111388] & 0x80 = 0x80";
      ]
  in
  equal (list int) (List.sort compare order) order

(* The COT boot *)

(* GSP_FMC_BOOT_PARAMS (gspifpub.h): GSP_ACR_BOOT_GSP_RM_PARAMS at 8 (target,
   descriptor size, descriptor offset at 8, bIsGspRmBoot at 28), GSP_RM_PARAMS
   at 40 (target, boot arguments at 8); GSP_DMA_TARGET_COHERENT_SYSTEM is 1; the
   descriptor is the 256-byte GspFwWprMeta. *)
let test_cot_args () =
  let a = Falcon.cot_args ~libos:0x1_0000_1000 ~wpr_meta:0x2_0000_3000 in
  let u32 o = Int32.to_int (String.get_int32_le a o) land 0xffff_ffff in
  let u64 o = Int64.to_int (String.get_int64_le a o) in
  equal int 80 (String.length a);
  equal (list int)
    [ 1; 256; 0x2_0000_3000; 1; 1; 0x1_0000_1000 ]
    [ u32 8; u32 12; u64 16; Char.code a.[36]; u32 40; u64 48 ]

(* NVDM_PAYLOAD_COT (kern_fsp_cot_payload.h, packed): version 2 and its size at
   0 and 2, the FMC at 4, FRTS's offset from the end of memory at 0x18 (28 MiB)
   and size at 0x20 (1 MiB), the hash at 0x24, the public key at 0x54, the
   signature at 0x1d4, the boot arguments at 0x354. *)
let test_cot_payload () =
  let range s : Rig_nv_pci.Images.range =
    { contents = "xx" ^ s; at = 2; length = String.length s }
  in
  let fmc : Rig_nv_pci.Images.fmc =
    {
      fmc = range "image";
      hash = range (String.make 48 'H');
      signature = range (String.make 96 'S');
      public_key = range (String.make 97 'P');
    }
  in
  let p = Falcon.cot_payload ~args:0x4000 ~fmc:0x5000 fmc in
  let u16 o = String.get_uint16_le p o in
  let u32 o = Int32.to_int (String.get_int32_le p o) land 0xffff_ffff in
  let u64 o = Int64.to_int (String.get_int64_le p o) in
  equal int 860 (String.length p);
  equal (list int)
    [ 2; 860; 0x5000; 0x1c0_0000; 0x10_0000; 0x4000 ]
    [ u16 0; u16 2; u64 4; u64 0x18; u32 0x20; u64 0x354 ];
  equal string (String.make 48 'H') (String.sub p 0x24 48);
  equal string (String.make 97 'P' ^ "\000\000\000") (String.sub p 0x54 100);
  equal string (String.make 96 'S') (String.sub p 0x1d4 96)

(* The FSP's queue: NV_PFSP_EMEMC(0) 0x8f2ac0 (AINCW bit 24, AINCR bit 25),
   EMEMD(0) 0x8f2ac4, QUEUE_HEAD(0) 0x8f2c00, QUEUE_TAIL(0) 0x8f2c04,
   MSGQ_HEAD(0) 0x8f2c80, MSGQ_TAIL(0) 0x8f2c84 (dev_fsp_pri.h, gh100). The
   message: an MCTP transport header with SOM and EOM (bits 31, 30), a message
   header of NVIDIA's vendor type 0x7e, vendor 0x10de and the NVDM type in its
   top byte (fsp_mctp_format.h), then the payload, padded with a word of zeros
   at least. *)
let test_fsp () =
  same
    [
      "[0x8f2ac0] := 0x1000000";
      "[0x8f2ac4] := 0xc0000000";
      "[0x8f2ac4] := 0x1410de7e";
      "[0x8f2ac4] := 0x64636261";
      "[0x8f2ac4] := 0x0";
      "[0x8f2c04] := 0xc";
      "[0x8f2c00] := 0x0";
      "wait [0x8f2c80] & 0xffffffff <> [0x8f2c84]";
      "[0x8f2ac0] := 0x2000000";
      "[0x8f2c84] := [0x8f2c80]";
    ]
    (Falcon.fsp 0x14 "abcd");
  raises_match (Exn.invalid_arg ~substring:"Falcon.fsp") (fun () ->
      Falcon.fsp 0x14 (String.make 1012 'x'))

let () =
  exit
  @@ run "rig_nv_pci.falcon"
       [
         group ~timeout:10. "falcons"
           [
             test "a reset selects the falcon's core" test_reset;
             test "a start uses the alias the falcon enables" test_start;
             test "a DMA copies 256 bytes a command" test_dma;
             test "a heavy-secure ucode is loaded and started" test_hs;
           ];
         group ~timeout:10. "boots"
           [
             test "the legacy boot checks each step" test_legacy;
             test "the FMC's arguments" test_cot_args;
             test "the COT payload" test_cot_payload;
             test "an FSP message" test_fsp;
           ];
       ]
