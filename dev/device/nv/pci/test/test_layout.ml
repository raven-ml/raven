(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Where the GSP's firmware goes: the radix-3 table's levels, and the region the
   GSP reserves at the top of the GPU's memory, against the rules of
   kgspCalculateFbLayout_TU102 (open-gpu-kernel-modules 570.144). *)

open Windtrap
module Layout = Device_nv_pci.Layout

let mib = 1 lsl 20

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
    (fun (_, n, pages) -> equal (array int) pages (Layout.radix3 n))

(* The fields of GspFwWprMeta (gsp_fw_wpr_meta.h), as (offset, bytes). *)
let gsp_fw_rsvd_start = (0x58, 8)
let non_wpr_heap_offset = (0x60, 8)
let non_wpr_heap_size = (0x68, 8)
let gsp_fw_wpr_start = (0x70, 8)
let gsp_fw_heap_offset = (0x78, 8)
let gsp_fw_heap_size = (0x80, 8)
let gsp_fw_offset = (0x88, 8)
let boot_bin_offset = (0x90, 8)
let frts_offset = (0x98, 8)
let frts_size = (0xa0, 8)
let gsp_fw_wpr_end = (0xa8, 8)
let fb_size = (0xb0, 8)
let vga_workspace_offset = (0xb8, 8)
let vga_workspace_size = (0xc0, 8)

(* GPUs of 4 to 96 GiB of memory, firmware images of up to 128 MiB. *)
let gen =
  Gen.(
    triple
      (map (fun g -> g * 1024 * mib) (int_range 4 96))
      (int_range 1 (1 * mib))
      (int_range 1 (128 * mib)))

(* From the top down: the VGA workspace ends memory, FRTS ends where the
   workspace starts and ends the protected region, the bootloader then the image
   lie below it on 4 KiB and 64 KiB, the GSP's heap below them on 1 MiB, the
   metadata's 1 MiB below the heap, then the heap outside the protected region,
   where the reserved region starts. *)
let test_wpr =
  prop "each part lies below the one above it, aligned" gen
    (fun (memory, boot, image) ->
      let w = Layout.wpr ~memory ~boot ~image in
      let f k = List.assoc k w in
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
      equal int (Layout.frts ~memory) (f frts_offset))

(* The memory the process manages ends on 2 MiB, 64 MiB below the end of memory
   at least, and below the GSP's region. *)
let test_top =
  prop "the managed memory ends below the GSP's region" gen
    (fun (memory, boot, image) ->
      let rsvd =
        List.assoc gsp_fw_rsvd_start (Layout.wpr ~memory ~boot ~image)
      in
      List.iter
        (fun family ->
          let top = Layout.top family ~memory ~boot ~image in
          equal int 0 (top mod (2 * mib));
          at_most int ~than:(memory - (64 * mib)) top)
        Device_nv_pci.Chip.[ Ampere; Ada; Blackwell ];
      at_most int ~than:rsvd (Layout.top Ada ~memory ~boot ~image);
      (* The FMC places FRTS's end 28 MiB below the end of memory, and every
         other part below it. *)
      let frts_end, frts = Layout.cot_frts in
      at_most int
        ~than:(memory - frts_end - frts - boot - image)
        (Layout.top Blackwell ~memory ~boot ~image))

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
      equal bool ok (Result.is_ok (Layout.check ~wpr2 ~top)))

let () =
  exit
  @@ run "device_nv_pci.layout"
       [
         group ~timeout:10. "radix-3" [ test_radix3 ];
         group ~timeout:10. "region" [ test_wpr; test_top; test_check ];
       ]
