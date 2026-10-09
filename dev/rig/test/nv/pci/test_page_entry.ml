(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* NVIDIA's page-table entries, against the bits of dev_mmu.h (tu102 for version
   2, gh100 for version 3, open-gpu-kernel-modules 570.144). *)

open Windtrap

let pa = 0x1234_5000

(* Expected entries, field by field from the headers: - V2 PTE: VALID 0:0,
   APERTURE 2:1 (0 video, 1 peer, 2 coherent system, 3 non-coherent system), VOL
   3:3, ADDRESS_SYS 53:8 and ADDRESS_VID 32:8 (the address shifted by 12),
   ADDRESS_VID_PEER 35:33, KIND 63:56, GENERIC_MEMORY 0x06. - V3 PTE: VALID 0:0,
   APERTURE 2:1, PCF 7:3 (0 cached, 1 uncached), KIND 11:8, ADDRESS 51:12,
   PEER_ID 63:61. *)
let ptes =
  Page_entry.
    [
      ("V2 GPU", V2, Gpu, false, 0x0600_0000_0123_4501L);
      ("V2 GPU uncached", V2, Gpu, true, 0x0600_0000_0123_4509L);
      ("V2 peer 3", V2, Peer 3, false, 0x0600_0006_0123_4503L);
      ( "V2 system snooped",
        V2,
        System { snooped = true },
        false,
        0x0600_0000_0123_4505L );
      ( "V2 system not snooped",
        V2,
        System { snooped = false },
        false,
        0x0600_0000_0123_4507L );
      ("V3 GPU", V3, Gpu, false, 0x0000_0000_1234_5601L);
      ("V3 GPU uncached", V3, Gpu, true, 0x0000_0000_1234_5609L);
      ("V3 peer 3", V3, Peer 3, false, 0x6000_0000_1234_5603L);
      ( "V3 system snooped",
        V3,
        System { snooped = true },
        false,
        0x0000_0000_1234_5605L );
      ( "V3 system not snooped",
        V3,
        System { snooped = false },
        false,
        0x0000_0000_1234_5607L );
    ]

let test_pte =
  cases "a page's entry has the header's bits"
    ~name:(fun (name, _, _, _, _) -> name)
    ptes
    (fun (_, v, target, uncached, e) ->
      equal int64 e (Page_entry.pte v ~pa target ~uncached))

(* V2 PDE: APERTURE 2:1 (1 video), NO_ATS 5:5, ADDRESS_VID 32:8. V3 PDE:
   APERTURE 2:1, PCF 5:3 (2 valid, cached, no ATS), ADDRESS 51:12. *)
let test_pde =
  cases "a directory entry points to its table in the GPU's memory"
    ~name:(fun (name, _, _) -> name)
    [ ("V2", Page_entry.V2, 0x500_0522L); ("V3", V3, 0x5000_5012L) ]
    (fun (_, v, e) -> equal int64 e (Page_entry.pde v ~child:0x5000_5000))

(* V2 dual: NO_ATS 5:5 in the low half; APERTURE_SMALL 66:65 and
   ADDRESS_SMALL_VID 96:72 in the high half. V3 dual: APERTURE_SMALL 66:65,
   PCF_SMALL 69:67, ADDRESS_SMALL 115:76. *)
let test_dual =
  cases "a dual entry holds a table in its high half or a page in its low half"
    ~name:(fun (name, _, _, _) -> name)
    [
      ("V2 table", Page_entry.V2, `Table 0x5000_5000, (0x20L, 0x5000_502L));
      ("V3 table", V3, `Table 0x5000_5000, (0L, 0x5000_5012L));
      ("V2 page", V2, `Page 0x1234L, (0x1234L, 0L));
      ("V3 nothing", V3, `None, (0L, 0L));
    ]
    (fun (_, v, e, halves) ->
      equal (pair int64 int64) halves (Page_entry.dual v e))

(* A version's root is the table the highest level indexes: 4 entries of a
   49-bit address on version 2, 2 of a 57-bit one on version 3. A system address
   is 46 frame bits on version 2 and 40 on version 3, above 12 of page. Pages
   are 512 MiB, 2 MiB and 4 KiB on both. *)
let test_levels =
  cases "a version's levels, addresses and pages"
    ~name:(fun (name, _, _, _) -> name)
    [ ("V2", Page_entry.V2, 4, 58); ("V3", V3, 2, 52) ]
    (fun (_, v, entries, pa_bits) ->
      let levels = Page_entry.levels v in
      let top = List.nth levels (List.length levels - 1) in
      equal int entries (1 lsl (Page_entry.bits v - top));
      equal int pa_bits (Page_entry.pa_bits v);
      equal
        (list (pair int int))
        [ (1 lsl 29, 1 lsl 29); (1 lsl 21, 1 lsl 21); (1 lsl 12, 1 lsl 12) ]
        (Page_entry.pages v))

let () =
  exit
  @@ run "rig_nv_pci.page_entry"
       [
         group ~timeout:10. "entries" [ test_pte; test_pde; test_dual ];
         group ~timeout:10. "levels" [ test_levels ];
       ]
