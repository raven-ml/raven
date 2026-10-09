(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Where the GSP's firmware goes: the radix-3 table's levels, the WPR metadata,
   and the region the GSP reserves at the top of the GPU's memory, against the
   rules of kgspCalculateFbLayout_TU102 and gsp_fw_wpr_meta.h
   (open-gpu-kernel-modules 570.144). *)

open Windtrap

let mib = 1 lsl 20
let u32 s o = Int32.to_int (String.get_int32_le s o) land 0xffff_ffff
let u64 s o = Int64.to_int (String.get_int64_le s o)

(* Radix-3: three levels of 64-bit addresses, 512 to a 4 KiB page. *)
let test_radix3 =
  cases "the radix-3 table has a page per 512 of the level below"
    ~name:(fun (name, _, _) -> name)
    [
      ("one byte", 1, [| 1; 1; 1; 1 |]);
      ("one page", 4096, [| 1; 1; 1; 1 |]);
      ("512 pages", 512 * 4096, [| 1; 1; 1; 512 |]);
      ("513 pages", (512 * 4096) + 1, [| 1; 1; 2; 513 |]);
      ("15,517 pages", 63_557_632, [| 1; 1; 31; 15_517 |]);
      ( "512 x 512 pages and one",
        (512 * 512 * 4096) + 1,
        [| 1; 2; 513; 262_145 |] );
    ]
    (fun (_, n, pages) -> equal (array int) pages (Fb_layout.radix3 n))

(* The fields of GspFwWprMeta, 256 bytes: magic, revision, then
   sysmemAddrOfRadix3Elf, sizeOfRadix3Elf, sysmemAddrOfBootloader,
   sizeOfBootloader, bootloaderCodeOffset, bootloaderDataOffset,
   bootloaderManifestOffset, sysmemAddrOfSignature, sizeOfSignature, then the
   region from gspFwRsvdStart, all of 8 bytes, and pmuReservedSize, of 4, at
   0xf4. *)
let magic = 0
let revision = 8
let radix3_elf = 0x10
let bootloader = 0x20
let code = 0x30
let signature = 0x48
let gsp_fw_rsvd_start = 0x58
let non_wpr_heap_offset = 0x60
let non_wpr_heap_size = 0x68
let gsp_fw_wpr_start = 0x70
let gsp_fw_heap_offset = 0x78
let gsp_fw_heap_size = 0x80
let gsp_fw_offset = 0x88
let boot_bin_offset = 0x90
let frts_offset = 0x98
let frts_size = 0xa0
let gsp_fw_wpr_end = 0xa8
let fb_size = 0xb0
let vga_workspace_offset = 0xb8
let vga_workspace_size = 0xc0
let pmu_reserved_size = 0xf4

(* GSP_FW_WPR_META_MAGIC and _REVISION. *)
let meta_magic = 0xdc3aae21371a60b3L
let meta_revision = 1

let meta layout ~memory ~boot ~image =
  Fb_layout.wpr_meta layout ~memory
    ~gsp:{ address = 0x1000_0000; size = image }
    ~signature:{ address = 0x2000_0000; size = 0x1001 }
    ~bootloader:{ address = 0x3000_0000; size = boot }
    ~code:0x10 ~data:0x20 ~manifest:0x30

(* GPUs of 4 to 96 GiB of memory, firmware images of up to 128 MiB. *)
let gen =
  Gen.(
    triple
      (map (fun g -> g * 1024 * mib) (int_range 4 96))
      (int_range 1 (1 * mib))
      (int_range 1 (128 * mib)))

(* The images: each address and size where the header puts it, the signature's
   size on 4 KiB. *)
let test_images () =
  let m = meta Process ~memory:(8 * 1024 * mib) ~boot:0x9000 ~image:0x80_0000 in
  equal int 256 (String.length m);
  equal int64 meta_magic (String.get_int64_le m magic);
  equal int meta_revision (u64 m revision);
  equal (list int)
    [
      0x1000_0000;
      0x80_0000;
      0x3000_0000;
      0x9000;
      0x10;
      0x20;
      0x30;
      0x2000_0000;
      0x2000;
    ]
    (List.map (u64 m)
       [
         radix3_elf;
         radix3_elf + 8;
         bootloader;
         bootloader + 8;
         code;
         code + 8;
         code + 16;
         signature;
         signature + 8;
       ])

(* From the top down: the VGA workspace ends memory, FRTS ends where the
   workspace starts and ends the protected region, the bootloader then the image
   lie below it on 4 KiB and 64 KiB, the GSP's heap below them on 1 MiB, the
   metadata's 1 MiB below the heap, then the heap outside the protected region,
   where the reserved region starts. *)
let test_wpr =
  prop "each part lies below the one above it, aligned" gen
    (fun (memory, boot, image) ->
      let f = u64 (meta Process ~memory ~boot ~image) in
      equal int memory (f fb_size);
      equal int memory (f vga_workspace_offset + f vga_workspace_size);
      equal int (f vga_workspace_offset) (f gsp_fw_wpr_end);
      equal int (f gsp_fw_wpr_end) (f frts_offset + f frts_size);
      at_most int ~than:(f frts_offset) (f boot_bin_offset + boot);
      at_most int ~than:(f boot_bin_offset) (f gsp_fw_offset + image);
      at_most int ~than:(f gsp_fw_offset)
        (f gsp_fw_heap_offset + f gsp_fw_heap_size);
      equal int (f gsp_fw_heap_offset - mib) (f gsp_fw_wpr_start);
      at_most int ~than:(f gsp_fw_wpr_start)
        (f non_wpr_heap_offset + f non_wpr_heap_size);
      equal int (f non_wpr_heap_offset) (f gsp_fw_rsvd_start);
      equal int 0 (f boot_bin_offset mod 4096);
      equal int 0 (f gsp_fw_offset mod 0x10000);
      List.iter
        (fun k -> equal int 0 (f k mod mib))
        [ gsp_fw_heap_offset; gsp_fw_wpr_start; non_wpr_heap_offset ];
      equal int (Fb_layout.frts ~memory) (f frts_offset);
      equal int (memory - (2 * mib)) (Fb_layout.frts ~memory))

(* The FMC lays the region out itself from five sizes: the VGA workspace
   (0x20000), the PMU's reservation, the heap outside the protected region, the
   GSP's heap and FRTS (1 MiB); the offsets stay zero. *)
let test_fmc () =
  let m = meta Fmc ~memory:(8 * 1024 * mib) ~boot:0x9000 ~image:0x80_0000 in
  equal int 0x20000 (u64 m vga_workspace_size);
  equal int mib (u64 m frts_size);
  not_equal int 0 (u32 m pmu_reserved_size);
  not_equal int 0 (u64 m non_wpr_heap_size);
  not_equal int 0 (u64 m gsp_fw_heap_size);
  equal (list int) [ 0; 0; 0; 0; 0 ]
    (List.map (u64 m)
       [
         frts_offset;
         gsp_fw_offset;
         vga_workspace_offset;
         fb_size;
         gsp_fw_rsvd_start;
       ])

(* The memory the process manages ends on 2 MiB, 64 MiB below the end of memory
   at least, and below the GSP's region. *)
let test_top =
  prop "the managed memory ends below the GSP's region" gen
    (fun (memory, boot, image) ->
      let rsvd = u64 (meta Process ~memory ~boot ~image) gsp_fw_rsvd_start in
      List.iter
        (fun layout ->
          let top = Fb_layout.top layout ~memory ~boot ~image in
          equal int 0 (top mod (2 * mib));
          at_most int ~than:(memory - (64 * mib)) top)
        Fb_layout.[ Process; Fmc ];
      at_most int ~than:rsvd (Fb_layout.top Process ~memory ~boot ~image);
      (* The FMC places FRTS's end 28 MiB below the end of memory, and every
         other part below it. *)
      let frts_end, frts = Fb_layout.cot_frts in
      equal (pair int int) (28 * mib, mib) Fb_layout.cot_frts;
      at_most int
        ~than:(memory - frts_end - frts - boot - image)
        (Fb_layout.top Fmc ~memory ~boot ~image))

(* The GSP puts the heap outside its protected region below WPR2, on 1 MiB:
   2.125 MiB on Blackwell (the FMC's non-WPR heap). *)
let test_check =
  cases "the managed memory must end below the GSP's heap"
    ~name:(fun (name, _, _, _) -> name)
    [
      ("below", 0x7_0000_0000, 0x6_ffc0_0000, true);
      ("at the heap's start", 0x7_0000_0000, 0x6_ffd0_0000, true);
      ("in the heap", 0x7_0000_0000, 0x6_ffe0_0000, false);
      ("above WPR2", 0x7_0000_0000, 0x7_0020_0000, false);
    ]
    (fun (_, wpr2, top, ok) ->
      equal bool ok (Result.is_ok (Fb_layout.check ~wpr2 ~top)))

let () =
  exit
  @@ run "rig_nv_pci.fb_layout"
       [
         group ~timeout:10. "radix-3" [ test_radix3 ];
         group ~timeout:10. "metadata"
           [
             test "the images" test_images;
             test_wpr;
             test "the FMC's sizes" test_fmc;
           ];
         group ~timeout:10. "region" [ test_top; test_check ];
       ]
