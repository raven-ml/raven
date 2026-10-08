(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let mib = 1 lsl 20
let page = 0x1000
let round_down n a = n / a * a

(* Radix-3 *)

(* A page of the table holds 4 KiB of 64-bit addresses, 2^9 of them
   (LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2 = 12). *)
let per_page = 9

let radix3 n =
  let up n = ((n - 1) lsr per_page) + 1 in
  let n3 = (n + page - 1) / page in
  let n2 = up n3 in
  let n1 = up n2 in
  [| up n1; n1; n2; n3 |]

(* The region on Ampere and Ada *)

(* The parts of the region the process sizes itself: the VGA workspace
   (NV_PRAMIN's 1 MiB, kgspCalculateFbLayout_TU102 on a GPU without display),
   the FRTS region (kgspGetFrtsSize_TU102), the GSP's heap, and the heap outside
   the protected region. *)
let vga = mib
let frts_size = mib
let heap = 0x8100000
let non_wpr = mib

(* The metadata's part: sizeof(GspFwWprMeta) aligned up to 1 MiB
   (kgspCalculateFbLayout_TU102's wprMetaSize). *)
let meta = mib

let wpr ~memory ~boot ~image =
  let module W = Defs.Wpr_meta in
  let vga_off = memory - vga in
  let frts_off = vga_off - frts_size in
  let boot_off = round_down (frts_off - boot) page in
  let gsp_off = round_down (boot_off - image) 0x10000 in
  let heap_off = round_down (gsp_off - heap) mib in
  let wpr_start = heap_off - meta in
  let non_wpr_off = round_down (wpr_start - non_wpr) mib in
  [
    (W.vga_workspace_size, vga);
    (W.vga_workspace_offset, vga_off);
    (W.gsp_fw_wpr_end, vga_off);
    (W.frts_size, frts_size);
    (W.frts_offset, frts_off);
    (W.boot_bin_offset, boot_off);
    (W.gsp_fw_offset, gsp_off);
    (W.gsp_fw_heap_size, heap);
    (W.fb_size, memory);
    (W.gsp_fw_heap_offset, heap_off);
    (W.gsp_fw_wpr_start, wpr_start);
    (W.non_wpr_heap_size, non_wpr);
    (W.non_wpr_heap_offset, non_wpr_off);
    (W.gsp_fw_rsvd_start, non_wpr_off);
  ]

let frts ~memory = memory - vga - frts_size

(* The region on Blackwell *)

let cot_frts = (0x1c00000, mib)

(* The sizes the FMC lays the region out with: the VGA workspace (0x20000,
   kgspCalculateFbLayout_TU102's VBIOS_WORKSPACE_SIZE), the PMU's reservation,
   the heap outside the protected region, the GSP's heap, and FRTS. *)
let fmc_vga = 0x20000
let fmc_pmu = 0x1820000
let fmc_non_wpr = 0x220000
let fmc_heap = 0x8700000

let fmc_sizes =
  let module W = Defs.Wpr_meta in
  [
    (W.vga_workspace_size, fmc_vga);
    (W.pmu_reserved_size, fmc_pmu);
    (W.non_wpr_heap_size, fmc_non_wpr);
    (W.gsp_fw_heap_size, fmc_heap);
    (W.frts_size, snd cot_frts);
  ]

(* The FMC aligns each of the six parts below FRTS, by at most 1 MiB: 6 MiB
   bounds what alignment takes. *)
let fmc_alignment = 6 * mib

(* The memory the process keeps clear of the GSP below the end of memory. *)
let margin = 64 * mib

let top (family : Chip.family) ~memory ~boot ~image =
  let bound =
    match family with
    | Ampere | Ada ->
        List.assoc Defs.Wpr_meta.gsp_fw_rsvd_start (wpr ~memory ~boot ~image)
    | Blackwell ->
        memory - fst cot_frts - snd cot_frts - boot - image - fmc_heap - meta
        - fmc_non_wpr - fmc_alignment
  in
  Int.min (memory - margin) (round_down bound (2 * mib))

let check ~wpr2 ~top =
  let start = round_down (wpr2 - fmc_non_wpr) mib in
  if top <= start then Ok ()
  else
    Error
      (strf
         "the GSP reserved the GPU's memory from 0x%x, below 0x%x, the top of \
          the memory the process manages"
         start top)
