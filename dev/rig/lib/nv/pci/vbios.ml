(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let window = 0x100000

let read (c : Chip.t) =
  Rig_pci.Window.read c.regs (Defs.Legacy.nv_prom_data 0) window

type fwsec = {
  image : string;
  imem_pa : int;
  imem_va : int;
  imem_size : int;
  dmem_pa : int;
  dmem_size : int;
  pkc : int;
  engines : int;
  ucode : int;
}

(* What the ROM lacks, raised by the walk and answered as [Error]. *)
exception Bad of string

let badf fmt = Printf.ksprintf (fun s -> raise (Bad s)) fmt

let check rom off n =
  if off < 0 || off + n > String.length rom then
    badf "the VBIOS ends before a structure it points to, at 0x%x" off

(* A field of a packed structure at [base], (offset, bytes). *)
let get rom base (off, n) =
  check rom (base + off) n;
  Field.get rom (base + off, n)

let u8 rom off = get rom off (0, 1)
let u16 rom off = get rom off (0, 2)
let u32 rom off = get rom off (0, 4)

(* Expansion ROM images *)

(* An image's PCI data structure flags its last image in bit 7
   (pci_exp_table.h's PCI_LAST_IMAGE). *)
let last_image = 0x80

let pci_data_signatures =
  Defs.
    [
      pci_data_struct_signature;
      pci_data_struct_signature_nv;
      pci_data_struct_signature_nv2;
    ]

(* The offset the falcon data's pointers are relative to: the extension image's
   offset less the base image's size, or 0 without both (s_locateExpansionRoms).
   An image's length is that of NVIDIA's extension of its PCI data structure
   where it has one. *)
let expansion rom =
  let rec walk block ext base =
    let pci =
      block + u16 rom (block + Defs.offsetof_pci_exp_rom_pci_data_struct_ptr)
    in
    if not (List.mem (u32 rom pci) pci_data_signatures) then
      badf "the VBIOS's image at 0x%x has no PCI data structure" block;
    let len = u16 rom (pci + Defs.offsetof_pci_data_struct_image_len) in
    let last =
      u8 rom (pci + Defs.offsetof_pci_data_struct_last_image) land last_image
      <> 0
    in
    let x =
      (pci + u16 rom (pci + Defs.offsetof_pci_data_struct_len) + 0xf)
      land lnot 0xf
    in
    let len, last =
      let rev = u16 rom (x + Defs.offsetof_pci_data_ext_struct_rev) in
      if
        u32 rom (x + Defs.offsetof_pci_data_ext_struct_sig)
        <> Defs.nv_pci_data_ext_sig
        || not
             (rev = Defs.nv_pci_data_ext_rev_10
             || rev = Defs.nv_pci_data_ext_rev_11)
      then (len, last)
      else
        let sub =
          u16 rom (x + Defs.offsetof_pci_data_ext_struct_subimage_len)
        in
        let ext_len = u16 rom (x + Defs.offsetof_pci_data_ext_struct_len) in
        if Defs.offsetof_pci_data_ext_struct_last_image + 1 <= ext_len then
          ( sub,
            u8 rom (x + Defs.offsetof_pci_data_ext_struct_last_image)
            land last_image
            <> 0 )
        else (sub, last && sub >= len)
    in
    let size = len * Defs.pci_rom_image_block_size in
    let kind = u8 rom (pci + Defs.offsetof_pci_data_struct_code_type) in
    let ext =
      if ext = None && kind = Defs.nv_bcrt_hash_info_base_code_type_vbios_ext
      then Some block
      else ext
    in
    let base =
      if base = None && kind = Defs.nv_bcrt_hash_info_base_code_type_vbios_base
      then Some size
      else base
    in
    if last then match (ext, base) with Some e, Some b -> e - b | _ -> 0
    else if size = 0 then
      badf "the VBIOS's image at 0x%x is empty and not the last" block
    else walk (block + size) ext base
  in
  walk 0 None None

(* The BIT table *)

(* The BIT header: its identifier and signature, and a checksum that makes the
   sum of its bytes 0 modulo 256 (s_vbiosFindBitHeader). *)
let bit rom =
  let header at =
    u16 rom at = Defs.bit_header_id
    && u32 rom (at + 2) = Defs.bit_header_signature
    &&
    let size = u8 rom (at + Defs.bit_header_size_offset) in
    let sum = ref 0 in
    for i = 0 to size - 1 do
      sum := !sum + u8 rom (at + i)
    done;
    !sum land 0xff = 0
  in
  let rec find at =
    if at + 6 > String.length rom then badf "the VBIOS has no BIT table"
    else if header at then at
    else find (at + 1)
  in
  find 0

(* The offset and size of the first production FWSEC descriptor of version 3
   that a falcon data token's ucode table names
   (s_vbiosParseFwsecUcodeDescFromBit). *)
let fwsec_desc rom =
  let ext = expansion rom in
  let at = bit rom in
  let module H = Defs.Bit_header in
  let token_size = get rom at H.token_size in
  let token =
    if token_size >= Defs.bit_token_v1_00_size_8 then
      Defs.Bit_token_8.(token_id, data_version, data_size, data_ptr)
    else if token_size >= Defs.bit_token_v1_00_size_6 then
      Defs.Bit_token_6.(token_id, data_version, data_size, data_ptr)
    else badf "the VBIOS's BIT tokens are %d bytes" token_size
  in
  let id, version, size, ptr = token in
  let descs table =
    let module T = Defs.Ucode_table_header in
    let module E = Defs.Ucode_table_entry in
    let entry_size = get rom table T.entry_size in
    let header_size = get rom table T.header_size in
    if
      get rom table T.version <> Defs.falcon_ucode_table_hdr_v1_version
      || header_size < Defs.falcon_ucode_table_hdr_v1_size_6
      || entry_size < Defs.falcon_ucode_table_entry_v1_size_6
    then []
    else
      List.init (get rom table T.entry_count) (fun i ->
          table + header_size + (i * entry_size))
      |> List.filter_map (fun e ->
          let app = get rom e E.application_id in
          if
            app <> Defs.falcon_ucode_entry_appid_fwsec_prod
            && app <> Defs.falcon_ucode_entry_appid_firmware_sec_lic
          then None
          else
            let desc = ext + get rom e E.desc_ptr in
            let v = get rom desc Defs.Ucode_desc_header.v_desc in
            let field f = Chip.field f v in
            (* Bit 0 set: the descriptor states its version. *)
            let has_version = v land 1 = 1 in
            let size = field Defs.nv_bit_falcon_ucode_desc_header_vdesc_size in
            if
              has_version
              && field Defs.nv_bit_falcon_ucode_desc_header_vdesc_version
                 = Defs.nv_bit_falcon_ucode_desc_header_vdesc_version_v3
              && size >= Defs.falcon_ucode_desc_v3_size_44
            then Some (desc, size)
            else None)
  in
  let tokens =
    List.init (get rom at H.token_entries) (fun i ->
        at + get rom at H.header_size + (i * token_size))
  in
  let found =
    List.find_map
      (fun t ->
        if
          get rom t id = Defs.bit_token_falcon_data
          && get rom t version = 2
          && get rom t size >= Defs.bit_data_falcon_data_v2_size_4
        then
          let data = get rom t ptr in
          let table =
            ext + get rom data Defs.Falcon_data.falcon_ucode_table_ptr
          in
          match descs table with d :: _ -> Some d | [] -> None
        else None)
      tokens
  in
  match found with
  | Some d -> d
  | None -> badf "the VBIOS has no production FWSEC with a version 3 descriptor"

(* Patching *)

(* The FRTS command's arguments (FWSECLIC_FRTS_CMD): the VBIOS read from the
   GPU's own ROM, and the 1 MiB region at [frts] of the GPU's memory. *)
let frts_command frts =
  let module C = Defs.Frts_cmd in
  let b = Bytes.make C.sizeof '\000' in
  let set (off, n) x =
    match n with
    | 4 -> Bytes.set_int32_le b off (Int32.of_int x)
    | _ -> Bytes.set_int64_le b off (Int64.of_int x)
  in
  set C.read_vbios_desc_version 1;
  set C.read_vbios_desc_size Defs.Read_vbios_desc.sizeof;
  set C.read_vbios_desc_flags Defs.fwseclic_read_vbios_struct_flags;
  set C.frts_region_desc_version 1;
  set C.frts_region_desc_size Defs.Frts_region_desc.sizeof;
  set C.frts_region_desc_frts_region_offset4_k (frts lsr 12);
  set C.frts_region_desc_frts_region_size
    Defs.fwseclic_frts_region_size_1mb_in_4k;
  set C.frts_region_desc_frts_region_media_type
    Defs.fwseclic_frts_region_media_fb;
  Bytes.unsafe_to_string b

let round_up n a = (n + a - 1) / a * a

let fwsec rom ~frts =
  match
    let desc, size = fwsec_desc rom in
    let module D = Defs.Ucode_desc_v3 in
    let f = get rom desc in
    let imem = f D.imem_load_size in
    let stored = f D.stored_size in
    let signatures = Defs.falcon_ucode_desc_v3_size_44 in
    check rom (desc + signatures) (size - signatures);
    check rom (desc + size) (round_up stored 256);
    let image =
      Bytes.of_string (String.sub rom (desc + size) (round_up stored 256))
    in
    let put at s =
      if at < 0 || at + String.length s > Bytes.length image then
        badf "FWSEC's image ends before the place of its patch, 0x%x" at;
      Bytes.blit_string s 0 image at (String.length s)
    in
    let image_u8 at = u8 (Bytes.unsafe_to_string image) at in
    let image_u32 at = u32 (Bytes.unsafe_to_string image) at in
    (* The application interface in the data names the DMEM mapper, whose
       command the patch sets and whose input buffer takes its arguments. *)
    let module A = Defs.App_interface_header in
    let module E = Defs.App_interface_entry in
    let module M = Defs.Dmem_mapper in
    let hdr = imem + f D.interface_offset in
    let entries =
      List.init
        (image_u8 (hdr + fst A.entry_count))
        (fun i ->
          hdr
          + image_u8 (hdr + fst A.header_size)
          + (i * image_u8 (hdr + fst A.entry_size)))
    in
    let mapper =
      match
        List.find_opt
          (fun e ->
            image_u32 (e + fst E.id)
            = Defs.falcon_application_interface_entry_id_dmemmapper)
          entries
      with
      | Some e -> imem + image_u32 (e + fst E.dmem_offset)
      | None -> badf "FWSEC has no DMEM mapper"
    in
    let word x =
      let b = Bytes.create 4 in
      Bytes.set_int32_le b 0 (Int32.of_int x);
      Bytes.unsafe_to_string b
    in
    put
      (mapper + fst M.init_cmd)
      (word Defs.falcon_application_interface_dmem_mapper_v3_cmd_frts);
    put
      (imem + image_u32 (mapper + fst M.cmd_in_buffer_offset))
      (frts_command frts);
    let n = Defs.bcrt30_rsa3k_sig_size in
    let blob = String.sub rom (desc + signatures) (size - signatures) in
    if String.length blob < n then badf "FWSEC's descriptor holds no signature";
    put
      (imem + f D.pkc_data_offset)
      (String.sub blob (String.length blob - n) n);
    {
      image = Bytes.unsafe_to_string image;
      imem_pa = f D.imem_phys_base;
      imem_va = f D.imem_virt_base;
      imem_size = imem;
      dmem_pa = f D.dmem_phys_base;
      dmem_size = f D.dmem_load_size;
      pkc = f D.pkc_data_offset;
      engines = f D.engine_id_mask;
      ucode = f D.ucode_id;
    }
  with
  | v -> Ok v
  | exception Bad why -> Error why
