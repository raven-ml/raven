(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

module G = Nv_gsp_defs

let round_up n align = (n + align - 1) / align * align

(* Little-endian field access over a plain byte image, keyed on the
   (byte offset, byte size) layout pairs from the generated tables. A
   [base] shifts the whole layout, so a structure embedded at a nonzero
   offset or an array element reuses the same field pairs. *)

let read_field ?(base = 0) b (off, size) =
  let p = base + off in
  match size with
  | 1 -> Char.code (Bytes.get b p)
  | 2 -> Bytes.get_uint16_le b p
  | 4 -> Int32.to_int (Bytes.get_int32_le b p) land 0xffffffff
  | 8 -> Int64.to_int (Bytes.get_int64_le b p)
  | _ -> invalid_arg "Ip: unsupported field size"

let set_field ?(base = 0) b (off, size) v =
  let p = base + off in
  match size with
  | 1 -> Bytes.set_uint8 b p (v land 0xff)
  | 2 -> Bytes.set_uint16_le b p (v land 0xffff)
  | 4 -> Bytes.set_int32_le b p (Int32.of_int (v land 0xffffffff))
  | 8 -> Bytes.set_int64_le b p (Int64.of_int v)
  | _ -> invalid_arg "Ip: unsupported field size"

let u32 b off = read_field b (off, 4)

(* The four-byte words of a byte image as unsigned integers, dropping a
   trailing partial word. *)
let u32_array b =
  Array.init (Bytes.length b / 4) (fun i -> u32 b (i * 4))

(* Firmware images *)

module Firmware = struct
  let hex digest =
    let n = Bytes.length digest in
    let out = Bytes.create (n * 2) in
    for i = 0 to n - 1 do
      let v = Bytes.get_uint8 digest i in
      Bytes.set out (2 * i) "0123456789abcdef".[v lsr 4];
      Bytes.set out ((2 * i) + 1) "0123456789abcdef".[v land 0xf]
    done;
    Bytes.unsafe_to_string out

  (* A compressed image is decoded through the system [zstd] into a
     temporary file, then read back. *)
  let zstd_dc path =
    let tmp = Filename.temp_file "tolk_nv_fw" ".bin" in
    Fun.protect
      ~finally:(fun () -> try Sys.remove tmp with Sys_error _ -> ())
      (fun () ->
        let cmd =
          Printf.sprintf "zstd -q -d -c %s > %s" (Filename.quote path)
            (Filename.quote tmp)
        in
        let status = Sys.command cmd in
        if status <> 0 then
          failwith
            (Printf.sprintf "zstd -d -c %s failed with exit code %d" path status);
        Bytes.of_string (In_channel.with_open_bin tmp In_channel.input_all))

  let url ~chip_dir name =
    Printf.sprintf "%s/nvidia/%s/gsp/%s" G.upstream chip_dir name

  let digest ~chip_dir name =
    match List.assoc_opt name G.firmware with
    | None ->
        failwith (Printf.sprintf "firmware %s has no pinned digest table" name)
    | Some by_chip -> (
        match List.assoc_opt chip_dir by_chip with
        | Some sha -> sha
        | None ->
            failwith
              (Printf.sprintf "firmware %s has no pinned digest for chip dir %s"
                 name chip_dir))

  let fetch ?dir ~chip_dir name =
    let dir =
      match dir with
      | Some d -> d
      | None -> Tolk.Helpers.getenv_str "NV_FW_PATH" "/lib/firmware/nvidia"
    in
    let sha = digest ~chip_dir name in
    let plain = Filename.concat (Filename.concat dir chip_dir) (Filename.concat "gsp" name) in
    let zst = plain ^ ".zst" in
    let blob =
      if Sys.file_exists plain then
        Bytes.of_string (In_channel.with_open_bin plain In_channel.input_all)
      else if Sys.file_exists zst then zstd_dc zst
      else
        failwith
          (Printf.sprintf
             "firmware %s not found: searched %s and %s; fetch it from %s \
              (expected sha256 %s)"
             name plain zst (url ~chip_dir name) sha)
    in
    let actual = hex (Tolk.Helpers.sha256 blob) in
    if not (String.equal actual sha) then
      failwith
        (Printf.sprintf
           "firmware %s sha mismatch: expected %s but got %s for %s (pinned \
            source %s)"
           name sha actual plain (url ~chip_dir name));
    blob
end

(* Falcon microcontroller boot images *)

module Flcn = struct
  type desc = {
    imem_load_size : int;
    imem_phys_base : int;
    imem_virt_base : int;
    dmem_phys_base : int;
    dmem_load_size : int;
    pkc_data_offset : int;
    engine_id_mask : int;
    ucode_id : int;
    stored_size : int;
    interface_offset : int;
  }

  type ucode = { desc : desc; frts_offset : int; frts_image : bytes }

  (* The VBIOS image lives behind a fixed 1 MiB register window; read it
     word by word into a flat byte image. [read32] answers with the
     unsigned 32-bit value at a byte address in the register aperture. *)
  let read_vbios ~read32 =
    let base = 0x00300000 and len = 0x00100000 in
    let b = Bytes.create len in
    for i = 0 to (len / 4) - 1 do
      let w = read32 (base + (i * 4)) in
      Bytes.set_int32_le b (i * 4) (Int32.of_int (w land 0xffffffff))
    done;
    b

  (* Walk the PCI expansion-ROM images to the extended VBIOS block; its
     base is where the falcon ucode tables are addressed from. Each image
     header carries a pointer to its PCI data structure, whose block count
     gives the image length and whose code type distinguishes the base
     image from the extended one. *)
  let expansion_rom_off rom =
    let rec walk off block_size =
      let pci_blck = read_field rom (off + G.offsetof_pci_exp_rom_pci_data_struct_ptr, 2) in
      let imglen =
        read_field rom (off + pci_blck + G.offsetof_pci_data_struct_image_len, 2)
        * G.pci_rom_image_block_size
      in
      let code_type =
        read_field rom (off + pci_blck + G.offsetof_pci_data_struct_code_type, 1)
      in
      if code_type = G.nv_bcrt_hash_info_base_code_type_vbios_ext then
        off - block_size
      else
        let block_size =
          if code_type = G.nv_bcrt_hash_info_base_code_type_vbios_base then imglen
          else block_size
        in
        walk (off + imglen) block_size
    in
    walk 0 0

  (* Locate the production FWSEC ucode descriptor by walking the BIT
     tokens to the falcon data token, then its ucode table. Returns the
     descriptor's byte offset in the ROM and its size. *)
  let fwsec_desc rom expansion =
    let bit_addr = 0x1b0 in
    let signature = read_field rom ~base:bit_addr G.Bit_header_v1_00.signature in
    if signature <> 0x00544942 then
      failwith
        (Printf.sprintf "Invalid BIT header signature 0x%x" signature);
    let token_entries = read_field rom ~base:bit_addr G.Bit_header_v1_00.tokenentries in
    let header_size = read_field rom ~base:bit_addr G.Bit_header_v1_00.headersize in
    let token_size = read_field rom ~base:bit_addr G.Bit_header_v1_00.tokensize in
    let found = ref None in
    for i = 0 to token_entries - 1 do
      let tbase = bit_addr + header_size + (i * token_size) in
      let token_id = read_field rom ~base:tbase G.Bit_token_v1_00.tokenid in
      let data_version = read_field rom ~base:tbase G.Bit_token_v1_00.dataversion in
      let data_size = read_field rom ~base:tbase G.Bit_token_v1_00.datasize in
      if
        token_id = G.bit_token_falcon_data && data_version = 2
        && data_size >= G.bit_data_falcon_data_v2_size_4
      then begin
        let data_ptr = read_field rom ~base:tbase G.Bit_token_v1_00.dataptr in
        let table_ptr =
          expansion
          + read_field rom
              ~base:(data_ptr land 0xffff)
              G.Bit_data_falcon_data_v2.falconucodetableptr
        in
        let entry_count = read_field rom ~base:table_ptr G.Falcon_ucode_table_hdr_v1.entrycount in
        let hdr_size = read_field rom ~base:table_ptr G.Falcon_ucode_table_hdr_v1.headersize in
        let entry_size = read_field rom ~base:table_ptr G.Falcon_ucode_table_hdr_v1.entrysize in
        for j = 0 to entry_count - 1 do
          let ebase = table_ptr + hdr_size + (j * entry_size) in
          let app_id = read_field rom ~base:ebase G.Falcon_ucode_table_entry_v1.applicationid in
          if app_id = G.falcon_ucode_entry_appid_fwsec_prod then begin
            let desc_off =
              expansion + read_field rom ~base:ebase G.Falcon_ucode_table_entry_v1.descptr
            in
            let vdesc = read_field rom ~base:desc_off G.Falcon_ucode_desc_header.vdesc in
            found := Some (desc_off, vdesc lsr 16)
          end
        done
      end
    done;
    match !found with
    | Some d -> d
    | None -> failwith "FWSEC production ucode descriptor not found in VBIOS"

  let read_desc rom off =
    {
      imem_load_size = read_field rom ~base:off G.Falcon_ucode_desc_v3.imemloadsize;
      imem_phys_base = read_field rom ~base:off G.Falcon_ucode_desc_v3.imemphysbase;
      imem_virt_base = read_field rom ~base:off G.Falcon_ucode_desc_v3.imemvirtbase;
      dmem_phys_base = read_field rom ~base:off G.Falcon_ucode_desc_v3.dmemphysbase;
      dmem_load_size = read_field rom ~base:off G.Falcon_ucode_desc_v3.dmemloadsize;
      pkc_data_offset = read_field rom ~base:off G.Falcon_ucode_desc_v3.pkcdataoffset;
      engine_id_mask = read_field rom ~base:off G.Falcon_ucode_desc_v3.engineidmask;
      ucode_id = read_field rom ~base:off G.Falcon_ucode_desc_v3.ucodeid;
      stored_size = read_field rom ~base:off G.Falcon_ucode_desc_v3.storedsize;
      interface_offset = read_field rom ~base:off G.Falcon_ucode_desc_v3.interfaceoffset;
    }

  (* The FWSEC command that reserves the frame-buffer resident tables. Its
     region descriptor points at the top 1 MiB of the 2 MiB reservation
     below the VGA workspace. *)
  let frts_cmd ~frts_offset =
    let b = Bytes.make G.Fwseclic_frts_cmd.sizeof '\000' in
    set_field b G.Fwseclic_read_vbios_desc.version 0x1;
    set_field b G.Fwseclic_read_vbios_desc.size G.Fwseclic_read_vbios_desc.sizeof;
    set_field b G.Fwseclic_read_vbios_desc.flags 2;
    let rbase = fst G.Fwseclic_frts_cmd.frtsregiondesc in
    set_field b ~base:rbase G.Fwseclic_frts_region_desc.version 0x1;
    set_field b ~base:rbase G.Fwseclic_frts_region_desc.size G.Fwseclic_frts_region_desc.sizeof;
    set_field b ~base:rbase G.Fwseclic_frts_region_desc.frtsregionoffset4k (frts_offset lsr 12);
    set_field b ~base:rbase G.Fwseclic_frts_region_desc.frtsregionsize 0x100;
    set_field b ~base:rbase G.Fwseclic_frts_region_desc.frtsregionmediatype 2;
    b

  (* Patch the FWSEC image for boot: select the FRTS command in the DMEM
     mapper, splice the command payload into its input buffer, and splice
     the trailing 0x180-byte signature over the descriptor's PKC data
     region. Offsets in the image are relative to the loaded IMEM. *)
  let patch_fwsec ~desc ~image ~signature ~cmd_id ~cmd =
    let patched = Bytes.copy image in
    let app_hdr_off = desc.imem_load_size + desc.interface_offset in
    let entry_count = read_field image ~base:app_hdr_off G.Falcon_application_interface_header_v1.entrycount in
    let dmem_offset = ref 0 in
    for i = 0 to entry_count - 1 do
      let ebase =
        app_hdr_off + G.Falcon_application_interface_header_v1.sizeof
        + (i * G.Falcon_application_interface_entry_v1.sizeof)
      in
      if read_field image ~base:ebase G.Falcon_application_interface_entry_v1.id
         = G.falcon_application_interface_entry_id_dmemmapper
      then
        dmem_offset := read_field image ~base:ebase G.Falcon_application_interface_entry_v1.dmemoffset
    done;
    let dmem_mapper_offset = desc.imem_load_size + !dmem_offset in
    set_field patched ~base:dmem_mapper_offset G.Falcon_application_interface_dmem_mapper_v3.init_cmd cmd_id;
    let cmd_in_buffer_offset =
      read_field image ~base:dmem_mapper_offset G.Falcon_application_interface_dmem_mapper_v3.cmd_in_buffer_offset
    in
    let cmd_off = desc.imem_load_size + cmd_in_buffer_offset in
    Bytes.blit cmd 0 patched cmd_off (Bytes.length cmd);
    let sig_off = desc.imem_load_size + desc.pkc_data_offset in
    Bytes.blit signature (Bytes.length signature - 0x180) patched sig_off 0x180;
    patched

  let prep_ucode ~rom ~vram_size =
    let expansion = expansion_rom_off rom in
    let desc_off, desc_size = fwsec_desc rom expansion in
    let desc = read_desc rom desc_off in
    let sig_total_size = desc_size - G.falcon_ucode_desc_v3_size_44 in
    let signature = Bytes.sub rom (desc_off + G.falcon_ucode_desc_v3_size_44) sig_total_size in
    let image = Bytes.sub rom (desc_off + desc_size) (round_up desc.stored_size 256) in
    (* The FRTS reservation is the top 1 MiB of the 2 MiB below the VGA
       workspace at the end of VRAM. *)
    let frts_offset = vram_size - 0x100000 - 0x100000 in
    let frts_image =
      patch_fwsec ~desc ~image ~signature ~cmd_id:0x15 ~cmd:(frts_cmd ~frts_offset)
    in
    { desc; frts_offset; frts_image }

  type booter = {
    image : bytes;
    data_off : int;
    data_sz : int;
    code_off : int;
    code_sz : int;
  }

  (* Parse a heavy-secured bootloader image: the nvfw container header
     points at a heavy-secure header, whose load header and app entry give
     the code and data spans, and whose patch points name where the
     production signature is spliced into the image. *)
  let prep_booter ~blob =
    let header_offset = read_field blob G.Nvfw_bin_hdr.header_offset in
    let data_offset = read_field blob G.Nvfw_bin_hdr.data_offset in
    let data_size = read_field blob G.Nvfw_bin_hdr.data_size in
    let hs_hdr_off = read_field blob ~base:header_offset G.Nvfw_hs_header_v2.header_offset in
    let patch_loc = u32 blob (read_field blob ~base:header_offset G.Nvfw_hs_header_v2.patch_loc) in
    let patch_sig = u32 blob (read_field blob ~base:header_offset G.Nvfw_hs_header_v2.patch_sig) in
    let num_sig = u32 blob (read_field blob ~base:header_offset G.Nvfw_hs_header_v2.num_sig) in
    let sig_prod_offset = read_field blob ~base:header_offset G.Nvfw_hs_header_v2.sig_prod_offset in
    let sig_prod_size = read_field blob ~base:header_offset G.Nvfw_hs_header_v2.sig_prod_size in
    let os_data_offset = read_field blob ~base:hs_hdr_off G.Nvfw_hs_load_header_v2.os_data_offset in
    let os_data_size = read_field blob ~base:hs_hdr_off G.Nvfw_hs_load_header_v2.os_data_size in
    let app_base = hs_hdr_off + G.Nvfw_hs_load_header_v2.sizeof in
    let app_offset = read_field blob ~base:app_base G.Nvfw_hs_load_header_v2_app.offset in
    let app_size = read_field blob ~base:app_base G.Nvfw_hs_load_header_v2_app.size in
    let sig_len = sig_prod_size / num_sig in
    let sig_off = sig_prod_offset + patch_sig in
    let image = Bytes.sub blob data_offset data_size in
    Bytes.blit blob sig_off image patch_loc sig_len;
    {
      image;
      data_off = os_data_offset;
      data_sz = os_data_size;
      code_off = app_offset;
      code_sz = app_size;
    }
end

(* Chain-of-trust falcon boot image (Blackwell) *)

module Flcn_cot = struct
  type fmc = {
    image : bytes;
    hash : int array;
    signature : int array;
    public_key : int array;
  }

  (* The chain-of-trust firmware is an ELF whose named sections carry the
     bootable image and the three verification blobs. The hash, signature
     and public key are consumed as 32-bit words; the public key is padded
     so its final word is whole. *)
  let init_fmc_image ~blob =
    let elf = Tolk.Elf.load blob in
    let section name =
      match Tolk.Elf.find_section elf name with
      | Some (s : Tolk.Elf.section) -> s.content
      | None -> failwith (Printf.sprintf "FMC image missing %S section" name)
    in
    {
      image = section "image";
      hash = u32_array (section "hash");
      signature = u32_array (section "signature");
      public_key = u32_array (Bytes.cat (section "publickey") (Bytes.make 3 '\000'));
    }
end

(* GSP firmware image and its page hierarchy *)

module Gsp = struct
  type radix3 = { npages : int array; offsets : int array; image_off : int }

  (* A three-level page hierarchy covering the GSP image: the deepest
     level maps the image's 4 KiB pages, and each level above holds one
     8-byte pointer per page of the level below, 512 (a 4 KiB page's
     worth) per directory page. [offsets] are the byte positions of each
     level within the hierarchy region, and [image_off] is where the image
     itself begins, right after the last directory level. *)
  let radix3 ~image_len =
    let npages = Array.make 4 0 in
    npages.(3) <- round_up image_len 0x1000 / 0x1000;
    for i = 3 downto 1 do
      npages.(i - 1) <-
        ((npages.(i) - 1) lsr (G.libos_memory_region_radix_page_log2 - 3)) + 1
    done;
    let offsets = Array.make 4 0 in
    for i = 1 to 3 do
      offsets.(i) <- offsets.(i - 1) + (npages.(i - 1) * 0x1000)
    done;
    { npages; offsets; image_off = offsets.(3) }

  type split = { image : bytes; signature : bytes }

  (* Split the GSP ELF into the firmware image and the per-chip signature
     section. The signature section is named for the chip family, e.g.
     [.fwsignature_ga10x]. *)
  let split_gsp_image ~blob ~chip_name =
    let elf = Tolk.Elf.load blob in
    let section name =
      match Tolk.Elf.find_section elf name with
      | Some (s : Tolk.Elf.section) -> s.content
      | None -> failwith (Printf.sprintf "GSP image missing %S section" name)
    in
    let family = String.lowercase_ascii (String.sub chip_name 0 4) in
    {
      image = section ".fwimage";
      signature = section (Printf.sprintf ".fwsignature_%sx" family);
    }

  type bootloader = {
    image : bytes;
    monitor_code_offset : int;
    monitor_data_offset : int;
    manifest_offset : int;
  }

  (* Parse the RISC-V bootloader container: the nvfw header points at the
     ucode descriptor giving the monitor code, data and manifest offsets,
     and the data span carries the bootloader image itself. *)
  let init_boot_binary_image ~blob =
    let header_offset = read_field blob G.Nvfw_bin_hdr.header_offset in
    let data_offset = read_field blob G.Nvfw_bin_hdr.data_offset in
    let data_size = read_field blob G.Nvfw_bin_hdr.data_size in
    {
      image = Bytes.sub blob data_offset data_size;
      monitor_code_offset = read_field blob ~base:header_offset G.Rm_riscv_ucode_desc.monitorcodeoffset;
      monitor_data_offset = read_field blob ~base:header_offset G.Rm_riscv_ucode_desc.monitordataoffset;
      manifest_offset = read_field blob ~base:header_offset G.Rm_riscv_ucode_desc.manifestoffset;
    }
end
