(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The GPU's security processors, which start its GSP. On Ampere and Ada the
   GSP's falcon runs FWSEC from the VBIOS, to set up the protected region of
   memory (WPR2), then SEC2 runs the booter, which loads the GSP's firmware. On
   Blackwell the FSP boots the GSP from the FMC image. *)

module D = Nv_defs
module P = Params
module Mmio = Nx_device_support.Mmio
module Elf = Nx_device_elf

let gsp_falcon = 0x110000
let sec2 = 0x840000
let lo32 v = v land 0xffff_ffff
let hi32 v = (v lsr 32) land 0xffff_ffff
let round_up n a = (n + a - 1) / a * a

(* Falcon operations *)

let wait = Nvdev.wait_until

let reset (d : Nvdev.t) ?(riscv = false) base =
  let engine =
    if base = gsp_falcon then "NV_PGSP_FALCON_ENGINE"
    else "NV_PSEC_FALCON_ENGINE"
  in
  Nvdev.write d engine [ ("reset", 1) ];
  Unix.sleepf 0.1;
  Nvdev.write d engine [ ("reset", 0) ];
  wait "falcon memory scrubbing" (fun () ->
      Nvdev.read_field d ~base "NV_PFALCON_FALCON_HWCFG2" "mem_scrubbing" = 0);
  if riscv then
    Nvdev.write d ~base "NV_PRISCV_RISCV_BCR_CTRL"
      [ ("core_select", 1); ("valid", 0); ("brfetch", 1) ]
  else if Nvdev.read_field d ~base "NV_PFALCON_FALCON_HWCFG2" "riscv" = 1 then begin
    Nvdev.write d ~base "NV_PRISCV_RISCV_BCR_CTRL" [ ("core_select", 0) ];
    wait "the falcon's RISC-V core" (fun () ->
        Nvdev.read_field d ~base "NV_PRISCV_RISCV_BCR_CTRL" "valid" = 1);
    Nvdev.write d ~base ~value:(Nvdev.chip_id d) "NV_PFALCON_FALCON_RM" []
  end

let disable_ctx_req d base =
  Nvdev.update d ~base "NV_PFALCON_FBIF_CTL" [ ("allow_phys_no_ctx", 1) ];
  Nvdev.write d ~base "NV_PFALCON_FALCON_DMACTL" []

let start_cpu d base =
  if Nvdev.read_field d ~base "NV_PFALCON_FALCON_CPUCTL" "alias_en" = 1 then
    Nvdev.wreg d (base + D.nv_pfalcon_falcon_cpuctl_alias) 0x2
  else Nvdev.write d ~base "NV_PFALCON_FALCON_CPUCTL" [ ("startcpu", 1) ]

let wait_cpu_halted d base =
  wait "the falcon halting" (fun () ->
      Nvdev.read_field d ~base "NV_PFALCON_FALCON_CPUCTL" "halted" = 1)

(* Copies [size] bytes from [src] in the GPU's memory into the falcon's memory
   at [dest], 256 bytes per command. *)
let execute_dma d base cmd ~dest ~mem_off ~src size =
  let ready () =
    wait "the falcon's DMA" (fun () ->
        Nvdev.read_field d ~base "NV_PFALCON_FALCON_DMATRFCMD" "full" = 0)
  in
  ready ();
  Nvdev.write d ~base
    ~value:(lo32 (src lsr 8))
    "NV_PFALCON_FALCON_DMATRFBASE" [];
  Nvdev.write d ~base
    ~value:(hi32 (src lsr 8) land 0x1ff)
    "NV_PFALCON_FALCON_DMATRFBASE1" [];
  let rec go xfered =
    if xfered < size then begin
      ready ();
      Nvdev.write d ~base ~value:(dest + xfered) "NV_PFALCON_FALCON_DMATRFMOFFS"
        [];
      Nvdev.write d ~base ~value:(mem_off + xfered)
        "NV_PFALCON_FALCON_DMATRFFBOFFS" [];
      Nvdev.write d ~base ~value:cmd "NV_PFALCON_FALCON_DMATRFCMD" [];
      go (xfered + 256)
    end
  in
  go 0;
  wait "the falcon's DMA to complete" (fun () ->
      Nvdev.read_field d ~base "NV_PFALCON_FALCON_DMATRFCMD" "idle" = 1)

type hs = {
  image : int; (* the image's address in the GPU's memory *)
  code_off : int;
  data_off : int;
  imem_pa : int;
  imem_va : int;
  imem_size : int;
  dmem_pa : int;
  dmem_va : int;
  dmem_size : int;
  pkc_off : int;
  engine_mask : int;
  ucode_id : int;
}

(* Runs a heavy-secure ucode on the falcon at [base], and is its mailboxes. *)
let execute_hs d base ?mailbox h =
  disable_ctx_req d base;
  Nvdev.update d ~base ~i:0 "NV_PFALCON_FBIF_TRANSCFG"
    [
      ("target", 0); ("mem_type", D.nv_pfalcon_fbif_transcfg_mem_type_physical);
    ];
  let cmd ~imem =
    Nvdev.encode d "NV_PFALCON_FALCON_DMATRFCMD"
      [
        ("write", 0);
        ("size", D.nv_pfalcon_falcon_dmatrfcmd_size_256b);
        ("ctxdma", 0);
        ("imem", Bool.to_int imem);
        ("sec", Bool.to_int imem);
      ]
  in
  execute_dma d base (cmd ~imem:true) ~dest:h.imem_pa ~mem_off:h.imem_va
    ~src:(h.image + h.code_off - h.imem_va)
    h.imem_size;
  execute_dma d base (cmd ~imem:false) ~dest:h.dmem_pa ~mem_off:h.dmem_va
    ~src:(h.image + h.data_off - h.dmem_va)
    h.dmem_size;
  Nvdev.write d ~base ~i:0 ~value:h.pkc_off "NV_PFALCON2_FALCON_BROM_PARAADDR"
    [];
  Nvdev.write d ~base ~value:h.engine_mask "NV_PFALCON2_FALCON_BROM_ENGIDMASK"
    [];
  Nvdev.write d ~base "NV_PFALCON2_FALCON_BROM_CURR_UCODE_ID"
    [ ("val", h.ucode_id) ];
  Nvdev.write d ~base "NV_PFALCON2_FALCON_MOD_SEL"
    [ ("algo", D.nv_pfalcon2_falcon_mod_sel_algo_rsa3k) ];
  Nvdev.write d ~base ~value:h.imem_va "NV_PFALCON_FALCON_BOOTVEC" [];
  Option.iter
    (fun m ->
      Nvdev.write d ~base ~value:(lo32 m) "NV_PFALCON_FALCON_MAILBOX0" [];
      Nvdev.write d ~base ~value:(hi32 m) "NV_PFALCON_FALCON_MAILBOX1" [])
    mailbox;
  start_cpu d base;
  wait_cpu_halted d base;
  ( Nvdev.read d ~base "NV_PFALCON_FALCON_MAILBOX0",
    Nvdev.read d ~base "NV_PFALCON_FALCON_MAILBOX1" )

(* FWSEC, from the VBIOS *)

let u16 s off = String.get_uint16_le s off

(* The VBIOS in the PROM window of the registers. *)
let read_vbios d = Mmio.read d.Nvdev.mmio 0x300000 0x100000

(* The FWSEC descriptor, signature and image of the VBIOS [rom]: the expansion
   ROM follows the base image, and its BIT table points to the falcon ucode
   table, whose FWSEC entry points to the descriptor. *)
let find_fwsec rom =
  let len = String.length rom in
  let rec images off base_size =
    if off + D.offsetof_pci_exp_rom_pci_data_struct_ptr + 2 > len then
      failwith "the VBIOS has no expansion ROM"
    else
      let pci = u16 rom (off + D.offsetof_pci_exp_rom_pci_data_struct_ptr) in
      let img_len =
        u16 rom (off + pci + D.offsetof_pci_data_struct_image_len)
        * D.pci_rom_image_block_size
      in
      let code_type =
        Char.code rom.[off + pci + D.offsetof_pci_data_struct_code_type]
      in
      if code_type = D.nv_bcrt_hash_info_base_code_type_vbios_ext then
        off - base_size
      else if img_len = 0 then failwith "the VBIOS has no expansion ROM"
      else
        images (off + img_len)
          (if code_type = D.nv_bcrt_hash_info_base_code_type_vbios_base then
             img_len
           else base_size)
  in
  let ext = images 0 0 in
  let bit = 0x1b0 in
  let module H = D.Bit_header in
  if P.read rom bit H.signature <> D.bit_header_signature then
    failwith "the VBIOS has no BIT table";
  let token_size = P.read rom bit H.token_size in
  let token i f =
    let at = bit + P.read rom bit H.header_size + (i * token_size) in
    if token_size = D.bit_token_v1_00_size_6 then
      P.read rom at
        (match f with
        | `Id -> D.Bit_token_6.token_id
        | `Version -> D.Bit_token_6.data_version
        | `Size -> D.Bit_token_6.data_size
        | `Ptr -> D.Bit_token_6.data_ptr)
    else
      P.read rom at
        (match f with
        | `Id -> D.Bit_token_8.token_id
        | `Version -> D.Bit_token_8.data_version
        | `Size -> D.Bit_token_8.data_size
        | `Ptr -> D.Bit_token_8.data_ptr)
  in
  let desc = ref None in
  for i = 0 to P.read rom bit H.token_entries - 1 do
    if
      token i `Id = D.bit_token_falcon_data
      && token i `Version = 2
      && token i `Size >= D.bit_data_falcon_data_v2_size_4
    then begin
      let data = token i `Ptr land 0xffff in
      let table = ext + P.read rom data D.Falcon_data.falcon_ucode_table_ptr in
      let module T = D.Ucode_table_header in
      let module E = D.Ucode_table_entry in
      for j = 0 to P.read rom table T.entry_count - 1 do
        let e =
          table
          + P.read rom table T.header_size
          + (j * P.read rom table T.entry_size)
        in
        if P.read rom e E.application_id = D.falcon_ucode_entry_appid_fwsec_prod
        then desc := Some (ext + P.read rom e E.desc_ptr)
      done
    end
  done;
  match !desc with
  | None -> failwith "the VBIOS has no FWSEC ucode"
  | Some off ->
      let v_desc = P.read rom off D.Ucode_desc_header.v_desc in
      if
        P.field D.nv_bit_falcon_ucode_desc_header_vdesc_version v_desc
        <> D.nv_bit_falcon_ucode_desc_header_vdesc_version_v3
      then failwith "the VBIOS's FWSEC descriptor is not of version 3";
      let size = P.field D.nv_bit_falcon_ucode_desc_header_vdesc_size v_desc in
      let stored = P.read rom off D.Ucode_desc_v3.stored_size in
      let signature =
        String.sub rom
          (off + D.falcon_ucode_desc_v3_size_44)
          (size - D.falcon_ucode_desc_v3_size_44)
      in
      let image = String.sub rom (off + size) (round_up stored 256) in
      (off, signature, image)

(* The FWSEC image, patched to run the command [cmd_id] with [cmd]: the DMEM
   mapper of its application interface names the command, whose arguments are
   copied to its input buffer, and the image carries the production
   signature. *)
let patch_fwsec ~desc ~signature image ~cmd_id cmd =
  let img = Bytes.of_string image in
  let imem = desc D.Ucode_desc_v3.imem_load_size in
  let hdr = imem + desc D.Ucode_desc_v3.interface_offset in
  let module H = D.App_interface_header in
  let module E = D.App_interface_entry in
  let dmem = ref None in
  for i = 0 to P.read image hdr H.entry_count - 1 do
    let e =
      hdr + P.read image hdr H.header_size + (i * P.read image hdr H.entry_size)
    in
    if P.read image e E.id = D.falcon_application_interface_entry_id_dmemmapper
    then dmem := Some (P.read image e E.dmem_offset)
  done;
  let mapper =
    match !dmem with
    | Some off -> imem + off
    | None -> failwith "FWSEC has no DMEM mapper"
  in
  P.write img mapper D.Dmem_mapper.init_cmd cmd_id;
  Bytes.blit_string cmd 0 img
    (imem + P.read image mapper D.Dmem_mapper.cmd_in_buffer_offset)
    (String.length cmd);
  let n = D.bcrt30_rsa3k_sig_size in
  Bytes.blit_string signature
    (String.length signature - n)
    img
    (imem + desc D.Ucode_desc_v3.pkc_data_offset)
    n;
  Bytes.to_string img

(* The FRTS command: FWSEC reads the VBIOS and sets up the 1 MiB FRTS region at
   [frts] in the GPU's memory. *)
let frts_cmd frts =
  let module C = D.Frts_cmd in
  let module V = D.Read_vbios_desc in
  let module R = D.Frts_region_desc in
  let c = Bytes.make C.sizeof '\000' in
  let v = fst C.read_vbios_desc and r = fst C.frts_region_desc in
  P.write c v V.version 1;
  P.write c v V.size V.sizeof;
  P.write c v V.flags D.fwseclic_read_vbios_struct_flags;
  P.write c r R.version 1;
  P.write c r R.size R.sizeof;
  P.write c r R.frts_region_offset4_k (frts lsr 12);
  P.write c r R.frts_region_size D.fwseclic_frts_region_size_1mb_in_4k;
  P.write c r R.frts_region_media_type D.fwseclic_frts_region_media_fb;
  Bytes.to_string c

(* Booter, from its firmware container *)

type booter = {
  image : string;
  data_off : int;
  data_size : int;
  code_off : int;
  code_size : int;
}

(* The booter of the container [b], patched with its production signature. *)
let booter b =
  let module B = D.Bin_header in
  let module H = D.Hs_header in
  let module L = D.Hs_load_header in
  let hs = P.read b 0 B.header_offset in
  let lh = P.read b hs H.header_offset in
  let u32 off = P.read b off (0, 4) in
  let patch_loc = u32 (P.read b hs H.patch_loc) in
  let patch_sig = u32 (P.read b hs H.patch_sig) in
  let sig_len = P.read b hs H.sig_prod_size / u32 (P.read b hs H.num_sig) in
  let signature =
    String.sub b (P.read b hs H.sig_prod_offset + patch_sig) sig_len
  in
  let image =
    Bytes.of_string
      (String.sub b (P.read b 0 B.data_offset) (P.read b 0 B.data_size))
  in
  Bytes.blit_string signature 0 image patch_loc sig_len;
  let app, _, _ = L.app in
  let app = lh + app in
  {
    image = Bytes.to_string image;
    data_off = P.read b lh L.os_data_offset;
    data_size = P.read b lh L.os_data_size;
    code_off = u32 app;
    code_size = u32 (app + 4);
  }

(* The two boots *)

type legacy = {
  fwsec : int * int -> int; (* a field of FWSEC's descriptor *)
  frts_image : int; (* the patched FWSEC image in the GPU's memory *)
  booter_image : int;
  boot : booter;
}

type cot = {
  args : Mmio.t; (* the FMC's boot parameters *)
  args_sysmem : int;
  fmc_image : int; (* the FMC image, as the FSP reads it *)
  hash : string;
  signature : string;
  public_key : string;
}

type t = Legacy of legacy | Cot of cot

(* The FRTS region: 1 MiB, 1 MiB below the top of the GPU's memory. *)
let frts_offset (d : Nvdev.t) = d.vram_size - 0x100000 - 0x100000

(* The FRTS region the FSP places for a COT boot: 1 MiB whose top lies 28 MiB
   below the end of the GPU's memory, which leaves room above it for the VGA
   workspace and the PMU's reservation. *)
let cot_frts_offset = 0x1c00000
let cot_frts_size = 0x100000

let wait_for_reset (d : Nvdev.t) =
  if d.fmc_boot then begin
    Nvdev.include_regs d "dev_therm" "gb202";
    wait "the GPU coming out of reset" (fun () ->
        Nvdev.read d "NV_THERM_I2CS_SCRATCH" = 0xff)
  end
  else
    wait "the GPU coming out of reset" (fun () ->
        Nvdev.read_field d "NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK"
          "read_protection_level0"
        = 1
        && Nvdev.read d ~i:0 "NV_PGC6_AON_SECURE_SCRATCH_GROUP_05" land 0xff
           = 0xff)

let section (o : Elf.t) name =
  match List.find_opt (fun (s : Elf.section) -> s.name = name) o.sections with
  | Some s -> s.contents
  | None -> failwith ("the firmware has no section " ^ name)

(* Prepares the boot: [booter_load] (Ampere, Ada) or [fmc] (Blackwell) is the
   firmware image. *)
let init_sw (d : Nvdev.t) ~firmware =
  let include_all l = List.iter (fun (n, a) -> Nvdev.include_regs d n a) l in
  if d.fmc_boot then begin
    include_all
      [
        ("dev_gsp", "ga102");
        ("dev_falcon_v4", "gh100");
        ("dev_vm", "gh100");
        ("dev_fsp_pri", "gh100");
        ("dev_bus", "tu102");
      ];
    let args, _, pages =
      Nvdev.boot_mem d
        ~data:(String.make D.Fmc_boot_params.sizeof '\000')
        D.Fmc_boot_params.sizeof
    in
    let o = Elf.load firmware in
    let image = section o "image" in
    let _, _, fmc =
      Nvdev.boot_mem d ~contiguous:true ~data:image (String.length image)
    in
    Cot
      {
        args;
        args_sysmem = List.hd pages;
        fmc_image = List.hd fmc;
        hash = section o "hash";
        signature = section o "signature";
        public_key = section o "publickey" ^ "\000\000\000";
      }
  end
  else begin
    include_all
      [
        ("dev_gsp", "ga102");
        ("dev_falcon_v4", "ga102");
        ("dev_riscv_pri", "ga102");
        ("dev_fbif_v4", "ga102");
        ("dev_falcon_second_pri", "ga102");
        ("dev_sec_pri", "ga102");
        ("dev_bus", "tu102");
      ];
    let rom = read_vbios d in
    (* No digest covers the VBIOS: an offset past its end means a ROM laid out
       otherwise than this walk reads it. *)
    let off, patched =
      match
        let off, signature, image = find_fwsec rom in
        ( off,
          patch_fwsec
            ~desc:(fun f -> P.read rom off f)
            ~signature image
            ~cmd_id:D.falcon_application_interface_dmem_mapper_v3_cmd_frts
            (frts_cmd (frts_offset d)) )
      with
      | r -> r
      | exception Invalid_argument _ ->
          failwith "the VBIOS is not laid out as expected"
    in
    let desc f = P.read rom off f in
    let vram data =
      match Nvdev.boot_mem d ~sysmem:false ~data (String.length data) with
      | _, Some pa, _ -> pa
      | _ -> assert false (* the GPU's memory has an address *)
    in
    let boot = booter firmware in
    Legacy
      {
        fwsec = desc;
        frts_image = vram patched;
        booter_image = vram boot.image;
        boot;
      }
  end

(* Sends [buf], an NVDM message of type [nvdm], to the FSP and waits for its
   answer. *)
let fsp_send d nvdm buf =
  let header =
    let h = Bytes.create 8 in
    Bytes.set_int32_le h 0 (Int32.of_int ((1 lsl 31) lor (1 lsl 30)));
    Bytes.set_int32_le h 4
      (Int32.of_int (0x7e lor (0x10de lsl 8) lor (nvdm lsl 24)));
    Bytes.to_string h
  in
  let pad = 4 - (String.length buf mod 4) in
  let msg = header ^ buf ^ String.make pad '\000' in
  if String.length msg >= 0x400 then failwith "an FSP message longer than 1 KiB";
  Nvdev.write d ~i:0 "NV_PFSP_EMEMC"
    [ ("offs", 0); ("blk", 0); ("aincw", 1); ("aincr", 0) ];
  for i = 0 to (String.length msg / 4) - 1 do
    Nvdev.write d ~i:0 ~value:(P.read msg (4 * i) (0, 4)) "NV_PFSP_EMEMD" []
  done;
  Nvdev.write d ~i:0 ~value:(String.length msg - 4) "NV_PFSP_QUEUE_TAIL" [];
  Nvdev.write d ~i:0 "NV_PFSP_QUEUE_HEAD" [];
  wait "the FSP's answer" (fun () ->
      Nvdev.read d ~i:0 "NV_PFSP_MSGQ_HEAD"
      <> Nvdev.read d ~i:0 "NV_PFSP_MSGQ_TAIL");
  Nvdev.write d ~i:0 "NV_PFSP_EMEMC"
    [ ("offs", 0); ("blk", 0); ("aincw", 0); ("aincr", 1) ];
  Nvdev.write d ~i:0
    ~value:(Nvdev.read d ~i:0 "NV_PFSP_MSGQ_HEAD")
    "NV_PFSP_MSGQ_TAIL" []

(* Boots the GSP with its libos arguments at [libos] and its WPR metadata at
   [wpr_meta], both as the GSP reads them. *)
let init_hw (d : Nvdev.t) t ~libos ~wpr_meta =
  match t with
  | Legacy l ->
      let desc = l.fwsec in
      let module V = D.Ucode_desc_v3 in
      reset d gsp_falcon;
      ignore
        (execute_hs d gsp_falcon
           {
             image = l.frts_image;
             code_off = 0;
             data_off = desc V.imem_load_size;
             imem_pa = desc V.imem_phys_base;
             imem_va = desc V.imem_virt_base;
             imem_size = desc V.imem_load_size;
             dmem_pa = desc V.dmem_phys_base;
             dmem_va = 0;
             dmem_size = desc V.dmem_load_size;
             pkc_off = desc V.pkc_data_offset;
             engine_mask = desc V.engine_id_mask;
             ucode_id = desc V.ucode_id;
           });
      if Nvdev.read d "NV_PFB_PRI_MMU_WPR2_ADDR_HI" = 0 then
        failwith "FWSEC did not set up the GPU's protected memory (WPR2)";
      reset d ~riscv:true gsp_falcon;
      Nvdev.write d ~value:(lo32 libos) "NV_PGSP_FALCON_MAILBOX0" [];
      Nvdev.write d ~value:(hi32 libos) "NV_PGSP_FALCON_MAILBOX1" [];
      reset d sec2;
      let b = l.boot in
      let m0, m1 =
        execute_hs d sec2 ~mailbox:wpr_meta
          {
            image = l.booter_image;
            code_off = b.code_off;
            data_off = b.data_off;
            imem_pa = 0;
            imem_va = b.code_off;
            imem_size = b.code_size;
            dmem_pa = 0;
            dmem_va = 0;
            dmem_size = b.data_size;
            pkc_off = 0x10;
            engine_mask = 1;
            ucode_id = 3;
          }
      in
      if m0 <> 0 then
        failwith
          (Printf.sprintf "the booter failed: mailbox 0x%08x 0x%08x" m0 m1);
      Nvdev.write d ~base:gsp_falcon "NV_PFALCON_FALCON_OS" [];
      if
        Nvdev.read_field d ~base:gsp_falcon "NV_PRISCV_RISCV_CPUCTL"
          "active_stat"
        <> 1
      then failwith "the GSP's core is not running"
  | Cot c ->
      let module F = D.Fmc_boot_params in
      let module A = D.Acr_boot_params in
      let module R = D.Rm_params in
      let args = Bytes.make F.sizeof '\000' in
      let a = fst F.boot_gsp_rm_params and r = fst F.gsp_rm_params in
      P.write args a A.gsp_rm_desc_offset wpr_meta;
      P.write args a A.gsp_rm_desc_size D.Wpr_meta.sizeof;
      P.write args a A.target D.gsp_dma_target_coherent_system;
      P.write args a A.b_is_gsp_rm_boot 1;
      P.write args r R.boot_args_offset libos;
      P.write args r R.target D.gsp_dma_target_coherent_system;
      Mmio.write c.args 0 (Bytes.to_string args);
      let module C = D.Cot_payload in
      let p = Bytes.make C.sizeof '\000' in
      P.write p 0 C.version 2;
      P.write p 0 C.size C.sizeof;
      P.write p 0 C.frts_vidmem_offset cot_frts_offset;
      P.write p 0 C.frts_vidmem_size cot_frts_size;
      P.write p 0 C.gsp_boot_args_sysmem_offset c.args_sysmem;
      P.write p 0 C.gsp_fmc_sysmem_offset c.fmc_image;
      let words field s =
        let off, _, _ = field in
        Bytes.blit_string s 0 p off (String.length s / 4 * 4)
      in
      words C.hash384 c.hash;
      words C.signature c.signature;
      words C.public_key c.public_key;
      fsp_send d D.nvdm_type_cot (Bytes.to_string p);
      wait "the GSP's boot" (fun () ->
          Nvdev.read_field d ~base:gsp_falcon "NV_PFALCON_FALCON_HWCFG2"
            "riscv_br_priv_lockdown"
          = 0)
