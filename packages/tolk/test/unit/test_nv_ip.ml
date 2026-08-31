(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The firmware and boot-image parsing layer of the driver-less NVIDIA boot
   chain, on hand-built synthetic images: the VBIOS FWSEC walk and its FRTS
   patch and signature splice, the heavy-secured booter and RISC-V
   bootloader header parses, the chain-of-trust ELF split, the pure page
   hierarchy, and the sha-verified firmware loader over a temporary
   directory standing in for the firmware root. *)

open Windtrap
module Ip = Tolk_nv.Ip
module Gsp_defs = Tolk_nv.Nv_tables.Gsp_defs

(* Little-endian byte pokes for building fixtures. *)
let set8 b off v = Bytes.set_uint8 b off (v land 0xff)
let set16 b off v = Bytes.set_uint16_le b off (v land 0xffff)
let set32 b off v = Bytes.set_int32_le b off (Int32.of_int (v land 0xffffffff))
let set64 b off v = Bytes.set_int64_le b off (Int64.of_int v)
let get32 b off = Int32.to_int (Bytes.get_int32_le b off) land 0xffffffff

let hex digest =
  let n = Bytes.length digest in
  let out = Bytes.create (n * 2) in
  for i = 0 to n - 1 do
    let v = Bytes.get_uint8 digest i in
    Bytes.set out (2 * i) "0123456789abcdef".[v lsr 4];
    Bytes.set out ((2 * i) + 1) "0123456789abcdef".[v land 0xf]
  done;
  Bytes.unsafe_to_string out

let contains ~needle haystack =
  let nl = String.length needle and hl = String.length haystack in
  if nl = 0 then true
  else
    let rec loop i =
      if i + nl > hl then false
      else if String.sub haystack i nl = needle then true
      else loop (i + 1)
    in
    loop 0

let raises_failure_with subs f =
  raises_match
    (function
      | Failure m -> List.for_all (fun n -> contains ~needle:n m) subs
      | _ -> false)
    f

(* A synthetic VBIOS ROM shaped like the on-board image the FWSEC walk
   expects: two PCI expansion-ROM images (a base then the extended one, so
   the extended base resolves to 0), a BIT header at 0x1b0 with one falcon
   data token, a one-entry ucode table pointing at a production FWSEC
   descriptor, then the descriptor, its 0x180-byte signature, and the
   FWSEC image carrying an application interface with a DMEM mapper. *)
let build_vbios () =
  let rom = Bytes.make 0x800 '\000' in
  (* PCI image 0 (base): pci data at 0x100, 2 blocks of 0x200 = 0x400. *)
  set16 rom 0x18 0x100;
  set16 rom 0x110 2;
  set8 rom 0x114 0 (* VBIOS_BASE *);
  (* PCI image 1 (extended) at 0x400: pci data at 0x100, extended code. *)
  set16 rom 0x418 0x100;
  set8 rom 0x514 0xe0 (* VBIOS_EXT -> expansion base 0x400 - 0x400 = 0 *);
  (* BIT header at 0x1b0. *)
  set32 rom 0x1b2 0x00544942 (* "BIT" *);
  set8 rom 0x1b8 0xc (* header size *);
  set8 rom 0x1b9 8 (* token size *);
  set8 rom 0x1ba 1 (* token entries *);
  (* Falcon data token at 0x1b0 + 0xc = 0x1bc. *)
  set8 rom 0x1bc 0x70 (* BIT_TOKEN_FALCON_DATA *);
  set8 rom 0x1bd 2 (* data version *);
  set16 rom 0x1be 4 (* data size *);
  set32 rom 0x1c0 0x1d0 (* data ptr *);
  (* Falcon data at 0x1d0: ucode table ptr. *)
  set32 rom 0x1d0 0x1e0;
  (* Ucode table header at 0x1e0. *)
  set8 rom 0x1e1 6 (* header size *);
  set8 rom 0x1e2 6 (* entry size *);
  set8 rom 0x1e3 1 (* entry count *);
  (* Ucode entry 0 at 0x1e0 + 6 = 0x1e6. *)
  set8 rom 0x1e6 0x85 (* FWSEC_PROD *);
  set32 rom 0x1e8 0x200 (* desc ptr *);
  (* Ucode descriptor at 0x200. vDesc high half is the descriptor size. *)
  let desc_size = 0x1ac in
  set32 rom 0x200 (desc_size lsl 16);
  set32 rom 0x204 0x300 (* stored size *);
  set32 rom 0x208 0x80 (* pkc data offset *);
  set32 rom 0x20c 0x00 (* interface offset *);
  set32 rom 0x210 0x1000 (* imem phys base *);
  set32 rom 0x214 0x100 (* imem load size *);
  set32 rom 0x218 0x2000 (* imem virt base *);
  set32 rom 0x21c 0x3000 (* dmem phys base *);
  set32 rom 0x220 0x200 (* dmem load size *);
  set16 rom 0x224 0x5 (* engine id mask *);
  set8 rom 0x226 0x3 (* ucode id *);
  (* Signature: 0x180 bytes at desc + 0x2c = 0x22c. *)
  for i = 0 to 0x17f do
    set8 rom (0x22c + i) ((i + 0x11) land 0xff)
  done;
  (* FWSEC image at desc + desc_size = 0x3ac, length round_up(0x300,256). *)
  let image_start = 0x3ac in
  (* Application interface header at imem_load_size + interface_offset. *)
  set8 rom (image_start + 0x103) 1 (* entry count *);
  set32 rom (image_start + 0x104) 4 (* entry id = DMEMMAPPER *);
  set32 rom (image_start + 0x108) 0x10 (* dmem offset *);
  (* DMEM mapper at imem_load_size + dmem_offset = 0x110. *)
  set32 rom (image_start + 0x118) 0x50 (* cmd in buffer offset *);
  rom

(* A heavy-secured booter container: an nvfw binary header pointing at a
   heavy-secure header, whose load header and app entry carry the data and
   code spans, and whose patch points name where the production signature
   is spliced into the extracted image. *)
let build_booter () =
  let blob = Bytes.make 0x200 '\000' in
  set32 blob 0xc 0x40 (* header_offset -> hs header *);
  set32 blob 0x10 0x100 (* data_offset *);
  set32 blob 0x14 0x80 (* data_size *);
  (* u32 cells the hs header points at for the patch values. *)
  set32 blob 0x20 0x10 (* patch_loc value *);
  set32 blob 0x24 0x8 (* patch_sig value *);
  set32 blob 0x28 1 (* num_sig value *);
  (* hs header at 0x40. *)
  set32 blob 0x40 0xc0 (* sig_prod_offset *);
  set32 blob 0x44 0x10 (* sig_prod_size *);
  set32 blob 0x48 0x20 (* patch_loc cell *);
  set32 blob 0x4c 0x24 (* patch_sig cell *);
  set32 blob 0x58 0x28 (* num_sig cell *);
  set32 blob 0x5c 0x80 (* header_offset -> load header *);
  (* load header at 0x80. *)
  set32 blob 0x88 0x10 (* os_data_offset *);
  set32 blob 0x8c 0x20 (* os_data_size *);
  (* app at 0x80 + 0x14 = 0x94. *)
  set32 blob 0x94 0x30 (* code offset *);
  set32 blob 0x98 0x40 (* code size *);
  (* signature bytes: sig_prod_offset + patch_sig = 0xc8, length 0x10. *)
  for j = 0 to 0xf do
    set8 blob (0xc8 + j) ((0x40 + j) land 0xff)
  done;
  blob

(* A RISC-V bootloader container: nvfw header pointing at a ucode
   descriptor with the monitor code, data and manifest offsets. *)
let build_bootloader () =
  let blob = Bytes.make 0x100 '\000' in
  set32 blob 0xc 0x40 (* header_offset -> ucode desc *);
  set32 blob 0x10 0x80 (* data_offset *);
  set32 blob 0x14 0x40 (* data_size *);
  (* RM_RISCV_UCODE_DESC at 0x40: manifest 0x20, data 0x28, code 0x30. *)
  set32 blob (0x40 + 0x20) 0x111 (* manifest offset *);
  set32 blob (0x40 + 0x28) 0x222 (* monitor data offset *);
  set32 blob (0x40 + 0x30) 0x333 (* monitor code offset *);
  blob

(* A minimal 64-bit little-endian ELF (ET_DYN) carrying the named program
   sections, so the chain-of-trust split has real sections to find. *)
let build_elf sections =
  let buf = Buffer.create 1024 in
  Buffer.add_bytes buf (Bytes.make 64 '\000');
  let pad_to align =
    while Buffer.length buf mod align <> 0 do
      Buffer.add_char buf '\000'
    done
  in
  let add content =
    pad_to 8;
    let off = Buffer.length buf in
    Buffer.add_bytes buf content;
    off
  in
  let offs = List.map (fun (_, c) -> add c) sections in
  let shstr = Buffer.create 64 in
  Buffer.add_char shstr '\000';
  let name s =
    let o = Buffer.length shstr in
    Buffer.add_string shstr s;
    Buffer.add_char shstr '\000';
    o
  in
  let names = List.map (fun (n, _) -> name n) sections in
  let n_shstr = name ".shstrtab" in
  let off_shstr = add (Buffer.to_bytes shstr) in
  pad_to 8;
  let e_shoff = Buffer.length buf in
  let shdr ~nm ~ty ~off ~size =
    let b = Bytes.make 64 '\000' in
    set32 b 0 nm;
    set32 b 4 ty;
    set64 b 24 off;
    set64 b 32 size;
    set64 b 48 1;
    Buffer.add_bytes buf b
  in
  shdr ~nm:0 ~ty:0 ~off:0 ~size:0;
  List.iter2
    (fun (_, c) (nm, off) -> shdr ~nm ~ty:1 ~off ~size:(Bytes.length c))
    sections
    (List.combine names offs);
  shdr ~nm:n_shstr ~ty:3 ~off:off_shstr ~size:(Buffer.length shstr);
  let obj = Buffer.to_bytes buf in
  Bytes.blit_string "\x7fELF\x02\x01\x01" 0 obj 0 7;
  set16 obj 16 3 (* ET_DYN *);
  set16 obj 18 0xB7 (* AArch64 *);
  set32 obj 20 1;
  set64 obj 40 e_shoff;
  set16 obj 52 64;
  set16 obj 58 64;
  set16 obj 60 (List.length sections + 2);
  set16 obj 62 (List.length sections + 1);
  obj

let mkdir_p path =
  let rec go p =
    if not (Sys.file_exists p) then begin
      go (Filename.dirname p);
      (try Sys.mkdir p 0o755 with Sys_error _ -> ())
    end
  in
  go path

let write_file path content =
  Out_channel.with_open_bin path (fun oc -> Out_channel.output_string oc content)

let () =
  run "Nv_ip"
    [
      group "vbios fwsec"
        [
          test "descriptor fields parse out of the ucode table" (fun () ->
              let u = Ip.Flcn.prep_ucode ~rom:(build_vbios ()) ~vram_size:0x10000000 in
              equal int 0x100 u.desc.imem_load_size;
              equal int 0x1000 u.desc.imem_phys_base;
              equal int 0x2000 u.desc.imem_virt_base;
              equal int 0x3000 u.desc.dmem_phys_base;
              equal int 0x200 u.desc.dmem_load_size;
              equal int 0x80 u.desc.pkc_data_offset;
              equal int 0x5 u.desc.engine_id_mask;
              equal int 0x3 u.desc.ucode_id;
              equal int 0x300 u.desc.stored_size;
              equal int 0 u.desc.interface_offset);
          test "frts offset is the top 1 MiB of the 2 MiB reservation"
            (fun () ->
              let u = Ip.Flcn.prep_ucode ~rom:(build_vbios ()) ~vram_size:0x10000000 in
              (* vram_size - 0x100000 - 0x100000 *)
              equal int 0xfe00000 u.frts_offset;
              equal int 0x300 (Bytes.length u.frts_image));
          test "the dmem mapper is patched to select the frts command"
            (fun () ->
              let u = Ip.Flcn.prep_ucode ~rom:(build_vbios ()) ~vram_size:0x10000000 in
              (* init_cmd at imem_load_size + dmem_offset + 0x2c = 0x13c *)
              equal int 0x15 (get32 u.frts_image 0x13c));
          test "the frts command is spliced into the input buffer" (fun () ->
              let u = Ip.Flcn.prep_ucode ~rom:(build_vbios ()) ~vram_size:0x10000000 in
              (* cmd buffer at imem_load_size + cmd_in_buffer_offset = 0x150 *)
              equal int 1 (get32 u.frts_image 0x150) (* read desc version *);
              equal int 0x18 (get32 u.frts_image 0x154) (* read desc size *);
              equal int 2 (get32 u.frts_image 0x164) (* read desc flags *);
              equal int 1 (get32 u.frts_image 0x168) (* region version *);
              equal int 0x14 (get32 u.frts_image 0x16c) (* region size field *);
              equal int 0xfe00 (get32 u.frts_image 0x170) (* offset >> 12 *);
              equal int 0x100 (get32 u.frts_image 0x174) (* region byte size *);
              equal int 2 (get32 u.frts_image 0x178) (* media type *));
          test "the signature is spliced over the pkc data region" (fun () ->
              let u = Ip.Flcn.prep_ucode ~rom:(build_vbios ()) ~vram_size:0x10000000 in
              (* sig at imem_load_size + pkc_data_offset = 0x180 *)
              let expected =
                String.init 0x180 (fun i -> Char.chr ((i + 0x11) land 0xff))
              in
              equal string expected
                (Bytes.sub_string u.frts_image 0x180 0x180));
        ];
      group "read_vbios"
        [
          test "assembles the register window little-endian" (fun () ->
              let b = Ip.Flcn.read_vbios ~read32:(fun a -> a lsr 2) in
              equal int 0x100000 (Bytes.length b);
              (* word i is read32 (0x300000 + i*4) = 0xc0000 + i *)
              equal int 0xc0000 (get32 b 0);
              equal int 0xc0010 (get32 b 0x40));
        ];
      group "booter"
        [
          test "header parse and signature splice" (fun () ->
              let bt = Ip.Flcn.prep_booter ~blob:(build_booter ()) in
              equal int 0x10 bt.data_off;
              equal int 0x20 bt.data_sz;
              equal int 0x30 bt.code_off;
              equal int 0x40 bt.code_sz;
              equal int 0x80 (Bytes.length bt.image);
              (* signature (0x10 bytes) spliced at patch_loc = 0x10 *)
              let expected =
                String.init 0x10 (fun j -> Char.chr ((0x40 + j) land 0xff))
              in
              equal string expected (Bytes.sub_string bt.image 0x10 0x10));
        ];
      group "bootloader"
        [
          test "riscv ucode descriptor offsets" (fun () ->
              let bl = Ip.Gsp.init_boot_binary_image ~blob:(build_bootloader ()) in
              equal int 0x40 (Bytes.length bl.image);
              equal int 0x333 bl.monitor_code_offset;
              equal int 0x222 bl.monitor_data_offset;
              equal int 0x111 bl.manifest_offset);
        ];
      group "fmc split"
        [
          test "elf sections split into image and word blobs" (fun () ->
              let image = Bytes.of_string "FMC-IMAGE-BYTES!" in
              let hash =
                Bytes.of_string "\x01\x02\x03\x04\x05\x06\x07\x08"
              in
              let signature =
                Bytes.init 16 (fun i -> Char.chr (0x10 + i))
              in
              let publickey =
                Bytes.of_string "\xaa\xbb\xcc\xdd\xee"
              in
              let elf =
                build_elf
                  [
                    ("image", image);
                    ("hash", hash);
                    ("signature", signature);
                    ("publickey", publickey);
                  ]
              in
              let f = Ip.Flcn_cot.init_fmc_image ~blob:elf in
              equal string (Bytes.to_string image) (Bytes.to_string f.image);
              equal (array int) [| 0x04030201; 0x08070605 |] f.hash;
              equal (array int)
                [| 0x13121110; 0x17161514; 0x1b1a1918; 0x1f1e1d1c |]
                f.signature;
              (* publickey padded by 3 zero bytes before the final word *)
              equal (array int) [| 0xddccbbaa; 0xee |] f.public_key);
        ];
      group "radix3"
        [
          (* Derived from the pinned source page math (ip.py:406-409): the
             deepest level maps round_up(len,0x1000)/0x1000 pages, and each
             level above holds one entry per 512 pages of the level below;
             offsets are the running byte position of each level. *)
          test "single-page image" (fun () ->
              let r = Ip.Gsp.radix3 ~image_len:0x1000 in
              equal (array int) [| 1; 1; 1; 1 |] r.npages;
              equal (array int) [| 0; 0x1000; 0x2000; 0x3000 |] r.offsets;
              equal int 0x3000 r.image_off);
          test "nine-megabyte image spans a middle directory page" (fun () ->
              let r = Ip.Gsp.radix3 ~image_len:0x900000 in
              equal (array int) [| 1; 1; 5; 2304 |] r.npages;
              equal (array int) [| 0; 0x1000; 0x2000; 0x7000 |] r.offsets;
              equal int 0x7000 r.image_off);
          test "one page past a directory boundary adds a directory page"
            (fun () ->
              let r = Ip.Gsp.radix3 ~image_len:0x201000 in
              equal (array int) [| 1; 1; 2; 513 |] r.npages;
              equal (array int) [| 0; 0x1000; 0x2000; 0x4000 |] r.offsets;
              equal int 0x4000 r.image_off);
        ];
      group "firmware loader"
        [
          test "unknown file has no pinned digest table" (fun () ->
              raises_failure_with [ "no pinned digest table" ] (fun () ->
                  Ip.Firmware.fetch ~dir:"/nonexistent" ~chip_dir:"ga102"
                    "not-a-real-file.bin"));
          test "unknown chip dir has no pinned digest" (fun () ->
              raises_failure_with [ "no pinned digest for chip dir" ] (fun () ->
                  Ip.Firmware.fetch ~dir:"/nonexistent" ~chip_dir:"zz999"
                    "bootloader-570.144.bin"));
          test "missing file names both paths, the digest and the source"
            (fun () ->
              let dir =
                Filename.concat (Filename.get_temp_dir_name ())
                  (Printf.sprintf "tolk_nv_ip_missing_%d" (Random.int 1_000_000))
              in
              mkdir_p (Filename.concat (Filename.concat dir "ga102") "gsp");
              let sha =
                List.assoc "ga102"
                  (List.assoc "bootloader-570.144.bin" Gsp_defs.firmware)
              in
              raises_failure_with
                [
                  Filename.concat
                    (Filename.concat (Filename.concat dir "ga102") "gsp")
                    "bootloader-570.144.bin";
                  ".zst";
                  Gsp_defs.upstream;
                  sha;
                ]
                (fun () ->
                  Ip.Firmware.fetch ~dir ~chip_dir:"ga102"
                    "bootloader-570.144.bin"));
          test "a present plain file is read and its digest is checked"
            (fun () ->
              let dir =
                Filename.concat (Filename.get_temp_dir_name ())
                  (Printf.sprintf "tolk_nv_ip_plain_%d" (Random.int 1_000_000))
              in
              let gsp = Filename.concat (Filename.concat dir "ga102") "gsp" in
              mkdir_p gsp;
              let content = "not the real firmware" in
              write_file (Filename.concat gsp "bootloader-570.144.bin") content;
              let sha =
                List.assoc "ga102"
                  (List.assoc "bootloader-570.144.bin" Gsp_defs.firmware)
              in
              let actual = hex (Tolk.Helpers.sha256 (Bytes.of_string content)) in
              raises_failure_with [ "sha mismatch"; sha; actual ] (fun () ->
                  Ip.Firmware.fetch ~dir ~chip_dir:"ga102"
                    "bootloader-570.144.bin"));
          test "a zstd file is decoded before the digest is checked" (fun () ->
              if Sys.command "command -v zstd >/dev/null 2>&1" <> 0 then ()
              else begin
                let dir =
                  Filename.concat (Filename.get_temp_dir_name ())
                    (Printf.sprintf "tolk_nv_ip_zst_%d" (Random.int 1_000_000))
                in
                let gsp = Filename.concat (Filename.concat dir "ga102") "gsp" in
                mkdir_p gsp;
                let content = "compressed placeholder firmware" in
                let scratch = Filename.temp_file "tolk_nv_ip_src" ".bin" in
                write_file scratch content;
                let zst =
                  Filename.concat gsp "bootloader-570.144.bin.zst"
                in
                let status =
                  Sys.command
                    (Printf.sprintf "zstd -q -f %s -o %s" (Filename.quote scratch)
                       (Filename.quote zst))
                in
                (try Sys.remove scratch with Sys_error _ -> ());
                is_true (status = 0);
                let actual =
                  hex (Tolk.Helpers.sha256 (Bytes.of_string content))
                in
                raises_failure_with [ "sha mismatch"; actual ] (fun () ->
                    Ip.Firmware.fetch ~dir ~chip_dir:"ga102"
                      "bootloader-570.144.bin")
              end);
        ];
    ]
