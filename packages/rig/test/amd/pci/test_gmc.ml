(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The memory controller's window and page-table entries, against amdgpu_vm.h's
   bits. *)

open Windtrap
open Rig_amd_pci

let strf = Printf.sprintf

(* Page-table entries *)

let gfx9 = (9, 4, 3)
let gfx11 = (11, 0, 0)
let gfx12 = (12, 0, 1)

(* The bits of amdgpu_vm.h: valid 0, system 1, snooped 2, executable 4, readable
   5, writeable 6, fragment from 7, PDE_PTE 54, TF 56, PDE_BFS from 59, IS_PTE
   and GFX12's PDE_PTE 63; MTYPE_UC (3) at 57, 48 and 54. *)
let entries =
  [
    ( "GFX12 leaf",
      gfx12,
      3,
      0x1000,
      `Page `Gpu,
      false,
      false,
      0,
      0x8000_0000_0000_1071L );
    (* GFX12 maps system memory MTYPE_UC however cached it is asked to be, as
       the kernel's GART does for the queues and words of VMID 0
       (gmc_v12_0_gart_init). *)
    ( "GFX12 2 MiB uncached system page",
      gfx12,
      2,
      0x20_0000,
      `Page `System,
      true,
      true,
      9,
      0x80c0_0000_0020_04f7L );
    ( "GFX12 cached system page",
      gfx12,
      3,
      0x1000,
      `Page `System,
      false,
      true,
      0,
      0x80c0_0000_0000_1077L );
    ("GFX12 root table", gfx12, 0, 0x5000, `Table, false, false, 0, 0x5001L);
    ( "GFX11 uncached leaf",
      gfx11,
      3,
      0x3000,
      `Page `Gpu,
      true,
      false,
      0,
      0x0003_0000_0000_3071L );
    ( "GFX11 1 GiB page",
      gfx11,
      1,
      0x4000_0000,
      `Page `Gpu,
      false,
      false,
      0,
      0x0040_0000_4000_0071L );
    ( "GFX9 PDB1 table",
      gfx9,
      1,
      0x7000,
      `Table,
      false,
      false,
      0,
      0x4800_0000_0000_7001L );
    ( "GFX9 PDB0 table",
      gfx9,
      2,
      0x8000,
      `Table,
      false,
      false,
      0,
      0x0100_0000_0000_8001L );
    ( "GFX9 2 MiB page",
      gfx9,
      2,
      0x20_0000,
      `Page `Gpu,
      false,
      false,
      0,
      0x0000_0000_0020_0071L );
    ( "GFX9 1 GiB page",
      gfx9,
      1,
      0x4000_0000,
      `Page `Gpu,
      false,
      false,
      0,
      0x0040_0000_4000_0071L );
    ( "GFX9 uncached leaf",
      gfx9,
      3,
      0x1000,
      `Page `Gpu,
      true,
      false,
      0,
      0x0600_0000_0000_1071L );
  ]

(* The memory controller's window on the GPU's memory, in 16 MiB units: a GPU of
   2039 units, the R9700's 32624 MiB. *)
let unit_bytes = 16 lsl 20
let units = 2039

let apertures =
  group ~timeout:10. "apertures"
    [
      cases "a window holds the GPU's memory, or is refused" ~id:fst
        [
          ("exactly", ((0x8000, 0x8000 + units - 1, 0), true));
          ("one unit short", ((0x8000, 0x8000 + units - 2, 0), false));
          ("registers a reset cleared", ((0, 0, 0), false));
          ("a top below its base", ((0x8000, 0x7fff, 0), false));
          ( "a die's memory one segment in",
            ((0x8000, 0x8000 + (2 * units) - 1, units * unit_bytes), true) );
          ( "a die's memory past the window",
            ((0x8000, 0x8000 + (2 * units) - 1, 2 * units * unit_bytes), false)
          );
        ]
        (fun (_, ((base, top, fabric), holds)) ->
          equal bool holds
            (Result.is_ok
               (Gmc.window ~base ~top ~fabric ~memory:(units * unit_bytes))));
    ]

let page_tables =
  group ~timeout:10. "page tables"
    [
      cases "an entry holds the bits amdgpu_vm.h states"
        ~id:(fun (n, _, _, _, _, _, _, _, _) -> n)
        entries
        (fun (_, gc, level, pa, target, uncached, snooped, fragment, e) ->
          equal int64 e
            (Gmc.entry ~gc ~level ~pa target ~uncached ~snooped ~fragment));
      cases "an address that is no page below 2^48 raises" ~id:(strf "0x%x")
        [ 0x1234; 1 lsl 48 ]
        (fun pa ->
          raises_match Exn.invalid_arg (fun () ->
              Gmc.entry ~gc:gfx12 ~level:3 ~pa (`Page `Gpu) ~uncached:false
                ~snooped:false ~fragment:0));
      (let gen =
         let open Gen in
         let+ gc = of_list [ gfx9; gfx11; gfx12 ]
         and+ level = int_range 0 3
         and+ target = of_list [ `Table; `Page `Gpu; `Page `System ]
         and+ uncached = bool
         and+ snooped = bool
         and+ fragment = int_range 0 31
         and+ page = int_range 0 ((1 lsl 36) - 1) in
         (gc, level, target, uncached, snooped, fragment, page lsl 12)
       in
       prop "an entry's address round-trips and its flags ignore it" gen
         (fun (gc, level, target, uncached, snooped, fragment, pa) ->
           let e pa =
             Gmc.entry ~gc ~level ~pa target ~uncached ~snooped ~fragment
           in
           let address = 0x0000_ffff_ffff_f000L in
           cover "an address above 4 GiB" (pa >= 1 lsl 32);
           cover "a table" (target = `Table);
           equal int64 (Int64.of_int pa) (Int64.logand (e pa) address);
           equal int64 (e 0) (Int64.logand (e pa) (Int64.lognot address))));
    ]

let () = exit (run "rig_amd_pci.gmc" [ apertures; page_tables ])
