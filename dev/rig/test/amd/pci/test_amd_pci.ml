(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Discovery = Rig_amd_pci.Discovery

let strf = Printf.sprintf

let fixture name =
  In_channel.with_open_bin ("fixtures/" ^ name) In_channel.input_all

(* The R9700's blocks as the amdgpu driver lists them (fixtures/r9700.txt):
   hardware ID, instance, version and segment bases. *)
let listing =
  fixture "r9700.txt" |> String.split_on_char '\n'
  |> List.filter (( <> ) "")
  |> List.map (fun line ->
      match String.split_on_char ' ' line with
      | hw :: inst :: version :: _harvest :: bases ->
          let version =
            match List.map int_of_string (String.split_on_char '.' version) with
            | [ a; b; c ] -> (a, b, c)
            | _ -> failwith ("a version in r9700.txt: " ^ line)
          in
          ( int_of_string hw,
            int_of_string inst,
            version,
            Array.of_list (List.map int_of_string bases) )
      | _ -> failwith ("a line of r9700.txt: " ^ line))

let table name =
  match Discovery.of_string (fixture name) with
  | Ok d -> d
  | Error why -> failf "%s: %s" name why

let version = triple int int int
let bases = array int

(* Discovery *)

let gc = 11
let sdma0 = 42
let mp0 = 255
let mp1 = 1
let umc = 150

let listed name =
  test
    (strf "%s states every block's version and bases as amdgpu lists them" name)
  @@ fun () ->
  let d = table name in
  List.iter
    (fun (hw, inst, v, b) ->
      equal
        ~msg:(strf "the bases of block %d instance %d" hw inst)
        (list (pair int bases))
        [ (inst, b) ]
        (List.filter (fun (i, _) -> i = inst) (Discovery.live d hw));
      if inst = 0 then
        equal
          ~msg:(strf "the version of block %d" hw)
          (option version) (Some v) (Discovery.version d hw))
    listing

let discovery =
  group ~timeout:10. "discovery"
    [
      cases ~name:Fun.id "place" [ "offset"; "bytes" ] (function
        | "offset" -> equal int (64 * 1024) Discovery.offset
        | _ -> equal int (10 * 1024) Discovery.bytes);
      listed "r9700.bin";
      listed "r9700_wide.bin";
      test "the R9700's blocks of a boot have their versions" (fun () ->
          let d = table "r9700.bin" in
          List.iter
            (fun (b, v) ->
              equal ~msg:(Discovery.name b) (option version) (Some v)
                (Discovery.version d b))
            [
              (gc, (12, 0, 1));
              (sdma0, (7, 0, 1));
              (mp0, (14, 0, 3));
              (mp1, (14, 0, 3));
            ]);
      test "the R9700's GC table states its shape" (fun () ->
          let g = (table "r9700.bin").gc in
          equal (list int) [ 4; 2; 8; 32; 16; 65536 ]
            Discovery.
              [ g.engines; g.arrays; g.units; g.scratch_slots; g.waves; g.lds ]);
      test "a fused instance is harvested and not live" (fun () ->
          let d = table "r9700_fused.bin" in
          equal (list (pair int (list int))) [ (umc, [ 7 ]) ] d.harvested;
          equal (list int) [ 0; 1; 2; 3; 4; 5; 6 ]
            (List.map fst (Discovery.live d umc)));
      test "a table without fused instances has none" (fun () ->
          equal (list (pair int (list int))) [] (table "r9700.bin").harvested);
      cases
        ~name:(fun (b, _) -> strf "block %d" b)
        "names"
        [
          (gc, "GC"); (sdma0, "SDMA0"); (mp0, "MP0"); (mp1, "MP1"); (0x7fff, "");
        ]
        (fun (b, n) -> equal string n (Discovery.name b));
    ]

(* A table damaged anywhere is refused or read, and reading never raises. *)

let contains ~sub s =
  let n = String.length sub in
  let rec at i =
    i + n <= String.length s && (String.sub s i n = sub || at (i + 1))
  in
  at 0

let damaged =
  let tables = [ "r9700.bin"; "r9700_wide.bin"; "r9700_fused.bin" ] in
  let length = Discovery.bytes in
  let gen =
    let open Gen in
    let+ name = of_list tables
    and+ cut = int_range 0 length
    and+ flips =
      list ~size:(int_range 0 3)
        (pair (int_range 0 (length - 1)) (int_range 1 255))
    in
    (name, cut, flips)
  in
  prop ~timeout:30. "a damaged table is refused or read, never raised" gen
    (fun (name, cut, flips) ->
      let b = Bytes.of_string (fixture name) in
      List.iter
        (fun (i, x) -> Bytes.set_uint8 b i (Bytes.get_uint8 b i lxor x))
        flips;
      let r = Discovery.of_string (Bytes.sub_string b 0 cut) in
      let refused sub =
        match r with Error e -> contains ~sub e | Ok _ -> false
      in
      cover "read" (Result.is_ok r);
      cover "refused by a checksum" (refused "checksum");
      cover "refused by a field outside" (refused "outside"))

(* Registers *)

module Regs = Rig_amd_pci.Regs

let layout d =
  match Regs.layout d with Ok l -> l | Error why -> failf "layout: %s" why

(* [with_version d b v] is [d] with block [b] of version [v]. *)
let with_version (d : Discovery.t) b v =
  { d with versions = (b, v) :: List.remove_assoc b d.versions }

let registers =
  group ~timeout:10. "registers"
    [
      test "the R9700 is a GPU of 64 compute units that runs gfx1201" (fun () ->
          let g = Regs.gpu (layout (table "r9700.bin")) in
          equal (list int) [ 12; 0; 1; 1; 4; 64; 32 ]
            (let a, b, c = g.target in
             [
               a;
               b;
               c;
               g.xccs;
               g.shader_engines;
               g.compute_units;
               g.scratch_slots;
             ]);
          equal version (12, 0, 1) g.gc;
          equal version (7, 0, 1) g.sdma);
      cases ~name:fst
        "a register lies at its block's segment base plus its offset"
        [
          (* (name, segment base + offset), from the R9700's bases and the
             kernel's register headers of its blocks' versions. *)
          ("regGRBM_GFX_CNTL", 0xa000 + 0x900);
          ("regMMVM_L2_CNTL", 0x1a000 + 0x4e4);
          ("regIH_RB_CNTL", 0x10a0 + 0x80);
          ("mmMP1_SMN_C2PMSG_90", 0x16000 + 0x29a);
          ("regBIF_BX_PF0_RSMU_INDEX", 0x14);
          ("regHDP_MEM_POWER_CTRL", 0xf20 + 0xd4);
          ("regCC_GC_SHADER_ARRAY_CONFIG", 0x1260 + 0x100f);
          ("regGC_USER_SHADER_ARRAY_CONFIG", 0xa000 + 0x5b90);
        ]
        (fun (name, a) ->
          equal int a (Regs.address (layout (table "r9700.bin")) name));
      test "a block takes its major's latest table at or before its version"
        (fun () ->
          let d = table "r9700.bin" in
          let later = layout (with_version d 34 (4, 1, 7)) in
          equal int
            (Regs.address (layout d) "regMMVM_L2_CNTL")
            (Regs.address later "regMMVM_L2_CNTL"));
      cases
        ~name:(fun (b, v, _) ->
          strf "block %d at %s" b
            (let a, b, c = v in
             strf "%d.%d.%d" a b c))
        "a block of a version with no table is refused, naming it"
        [
          (34, (1, 7, 0), "MMHUB 1.7.0 is a version this library does not boot");
          (34, (2, 0, 0), "MMHUB 2.0.0 is a version this library does not boot");
          (11, (10, 3, 0), "GC 10.3.0 is a version this library does not boot");
        ]
        (fun (b, v, msg) ->
          equal (result pass string) (Error msg)
            (Result.map ignore
               (Regs.layout (with_version (table "r9700.bin") b v))));
      test "a GPU without a block a boot programs is refused" (fun () ->
          let d = table "r9700.bin" in
          equal (result pass string) (Error "the GPU has no OSSSYS block")
            (Result.map ignore
               (Regs.layout
                  { d with versions = List.remove_assoc 40 d.versions })));
      test "every GC register lies in a range a virtual function guards"
        (fun () ->
          let l = layout (table "r9700.bin") in
          List.iter
            (fun (r : Rig_amd_abi.Register.t) ->
              let a = Regs.address l r.name in
              if
                not
                  (List.exists
                     (fun (lo, hi) -> lo <= a && a <= hi)
                     (Regs.guarded l))
              then failf "%s at 0x%x is in no guarded range" r.name a)
            (Rig_amd_abi.Register.registers (Regs.gpu l)));
    ]

(* Page-table entries *)

module Gmc = Rig_amd_pci.Gmc

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
    (* GFX12 maps system memory MTYPE_NC however uncached it is asked to be: the
       kernel's gmc_v12_0_get_vm_pte works around a hardware bug so. *)
    ( "GFX12 2 MiB uncached system page",
      gfx12,
      2,
      0x20_0000,
      `Page `System,
      true,
      true,
      9,
      0x8000_0000_0020_04f7L );
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
      cases ~name:fst "a window holds the GPU's memory, or is refused"
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
      cases
        ~name:(fun (n, _, _, _, _, _, _, _, _) -> n)
        "an entry holds the bits amdgpu_vm.h states" entries
        (fun (_, gc, level, pa, target, uncached, snooped, fragment, e) ->
          equal int64 e
            (Gmc.entry ~gc ~level ~pa target ~uncached ~snooped ~fragment));
      cases ~name:(strf "0x%x") "an address that is no page below 2^48 raises"
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

let dies =
  group ~timeout:10. "dies"
    [
      test "a GPU of one die has die 0" (fun () ->
          equal (list int) [ 0 ] (Discovery.aids (table "r9700.bin")));
      test "a die is live with all its SDMA instances or a pair" (fun () ->
          let d = table "r9700.bin" in
          let sdma = List.init 16 (fun i -> (i, [| 0 |])) in
          let d =
            {
              d with
              bases = (42, sdma) :: List.remove_assoc 42 d.bases;
              harvested = [ (42, [ 8; 9; 12 ]) ];
            }
          in
          equal (list int) [ 0; 1; 2 ] (Discovery.aids d));
    ]

(* Power manager *)

module Smu = Rig_amd_pci.Smu

(* The IDs of amdgpu's message headers (smu_v13_0_0_ppsmc.h,
   smu_v13_0_6_ppsmc.h, smu_v13_0_12_ppsmc.h, smu_v14_0_2_ppsmc.h) and clock
   enumerations, by the MP1 versions amdgpu drives with each. *)
let messages =
  [
    ((14, 0, 3), "PPSMC_MSG_SetSoftMinByFreq", Some 0x19);
    ((14, 0, 3), "PPSMC_MSG_GetDpmFreqByIndex", Some 0x1f);
    ((14, 0, 3), "PPCLK_GFXCLK", Some 0);
    ((13, 0, 10), "PPSMC_MSG_Mode1Reset", Some 0x2f);
    ((13, 0, 14), "PPSMC_MSG_GfxDriverReset", Some 3);
    ((13, 0, 14), "PPCLK_UCLK", Some 3);
    ((13, 0, 14), "PPCLK_GFXCLK", None);
    ((13, 0, 12), "PPSMC_MSG_McaBankDumpDW", Some 0x37);
    ((13, 0, 4), "PPSMC_MSG_GetSmuVersion", None);
  ]

let power =
  group ~timeout:10. "power manager"
    [
      cases
        ~name:(fun ((a, b, c), n, _) -> strf "%s of MP1 %d.%d.%d" n a b c)
        "a message has its header's ID" messages
        (fun (mp1, name, id) -> equal (option int) id (Smu.message mp1 name));
      cases
        ~name:(fun ((a, b, c), clock, features, _) ->
          strf "%s of MP1 %d.%d.%d, features 0x%x" clock a b c features)
        "a clock's DPM runs iff its feature bit is enabled"
        (* The DPM feature bits of smu14_driver_if_v14_0.h (UCLK 3, FCLK 4,
           SOCCLK 5, GFXCLK 1) and smu_v13_0_6_pmfw.h (UCLK 6, GFXCLK 3). *)
        [
          ((14, 0, 3), "PPCLK_UCLK", (1 lsl 3) lor (1 lsl 5), true);
          ((14, 0, 3), "PPCLK_SOCCLK", (1 lsl 3) lor (1 lsl 5), true);
          ((14, 0, 3), "PPCLK_FCLK", (1 lsl 3) lor (1 lsl 5), false);
          ((14, 0, 3), "PPCLK_GFXCLK", (1 lsl 3) lor (1 lsl 5), false);
          ((14, 0, 3), "PPCLK_UCLK", 0, false);
          ((13, 0, 6), "PPCLK_UCLK", 1 lsl 6, true);
          ((13, 0, 6), "PPCLK_GFXCLK", 1 lsl 6, false);
        ]
        (fun (mp1, clock, features, on) ->
          equal bool on (Smu.dpm mp1 ~features clock));
      test "a clock's request holds the clock above the value" (fun () ->
          equal int 0x2_0010 (Smu.clock_request ~clock:2 0x10));
      test "a GPU whose MP1 has no messages is refused, naming it" (fun () ->
          equal (result pass string)
            (Error "MP1 13.0.4 is a version this library does not boot")
            (Result.map ignore
               (Regs.layout (with_version (table "r9700.bin") 1 (13, 0, 4)))));
    ]

(* Security processor *)

module Psp = Rig_amd_pci.Psp

let u32 b off = Int32.to_int (String.get_int32_le b off) land 0xffff_ffff

(* [words b offs] is the 32-bit words of [b] at [offs]. *)
let words b offs = List.map (u32 b) offs

(* psp_gfx_if.h: a command buffer is 1024 bytes, its ID at 8, its command at 28,
   its response at 864 (status at +0, TMR size at +16); a frame is 64 bytes, the
   command's address at 0 and 4, the fence's at 12 and 16, its value at 20. *)
let security =
  group ~timeout:10. "security processor"
    [
      test "a firmware load names its address, size and type" (fun () ->
          let b = Psp.load_ip_fw ~at:0x12_3456_7000 ~bytes:0x800 ~fw_type:89 in
          equal int 1024 (String.length b);
          equal (list int)
            [ 6; 0x3456_7000; 0x12; 0x800; 89 ]
            (words b [ 8; 28; 32; 36; 40 ]));
      test "a table of contents names its address and size" (fun () ->
          let b = Psp.load_toc ~at:0x1000 ~bytes:0x40 in
          equal (list int) [ 0x20; 0x1000; 0; 0x40 ] (words b [ 8; 28; 32; 36 ]));
      test "a TMR names both its addresses, and says so" (fun () ->
          let b =
            Psp.setup_tmr ~at:0x2_0000_0000 ~fabric:0x3_0000_0000
              ~bytes:0x40_0000
          in
          equal (list int)
            [ 5; 0; 2; 0x40_0000; 2; 0; 3 ]
            (words b [ 8; 28; 32; 36; 40; 44; 48 ]));
      test "the bootloader loads every component the kernel loads, in order"
        (fun () ->
          (* amdgpu_psp.c psp_hw_start: KDB, SPL, SYS_DRV, SOC_DRV, INTF_DRV,
             DBG_DRV, RAS_DRV, IPKEYMGR_DRV, SPDM_DRV, then SOS; the types of
             amdgpu_ucode.h and the commands of amdgpu_psp.h. *)
          equal
            (list (pair int int))
            [
              (3, 0x80000);
              (5, 0x10000000);
              (2, 0x10000);
              (7, 0xb0000);
              (8, 0xd0000);
              (9, 0xc0000);
              (10, 0xe0000);
              (11, 0xf0000);
              (12, 0x20000000);
              (1, 0x20000);
            ]
            Psp.steps);
      test "the RLC's autoload and a partition carry their IDs" (fun () ->
          equal (list int) [ 0x21 ] (words Psp.autoload_rlc [ 8 ]);
          equal (list int) [ 0x27; 1 ] (words (Psp.partition ~mode:1) [ 8; 28 ]));
      test "a frame names its command, its fence and the fence's value"
        (fun () ->
          let f = Psp.frame ~command:0x1_0000_2000 ~fence:0x3000 ~value:7 in
          equal int 64 (String.length f);
          equal (list int)
            [ 0x2000; 1; 0x3000; 0; 7 ]
            (words f [ 0; 4; 12; 16; 20 ]));
      test "an answer's status and TMR size are read at the response" (fun () ->
          let b = Bytes.make 1024 '\000' in
          Bytes.set_int32_le b 864 0x1234l;
          Bytes.set_int32_le b 880 0x50_0000l;
          let b = Bytes.to_string b in
          equal (pair int int) (0x1234, 0x50_0000)
            (Psp.status b, Psp.tmr_bytes b));
    ]

(* Compute queues *)

module Gfx = Rig_amd_pci.Gfx

let gfx12_queue =
  {
    Gfx.ring = 0x1_0000_0000;
    ring_bytes = 16 lsl 20;
    read = 0x3_0000;
    write = 0x3_0008;
    eop = 0x2000_0000;
    eop_bytes = 0x1000;
    doorbell = 3;
  }

(* A synthetic GPU of GC 9.4.3, for the fields of its descriptor. *)
let gfx9_layout () =
  let d = table "r9700.bin" in
  let d = with_version (with_version d 11 (9, 4, 3)) 42 (4, 4, 2) in
  let d = with_version (with_version d 255 (13, 0, 6)) 1 (13, 0, 6) in
  let d = with_version (with_version d 34 (1, 8, 0)) 40 (4, 4, 2) in
  let d = with_version (with_version d 41 (4, 4, 2)) 108 (7, 9, 0) in
  layout d

(* v12_structs.h and v9_structs.h give each field's word; gc_12_0_0_sh_mask.h
   each register field's first bit. *)
(* The block versions of other GPUs, over the R9700's table: GC, NBIO, MMHUB,
   OSSSYS, HDP, SDMA, MP0 and MP1. *)
let blocks gc nbio mmhub osssys hdp sdma mp =
  [
    (11, gc);
    (108, nbio);
    (34, mmhub);
    (40, osssys);
    (41, hdp);
    (42, sdma);
    (255, mp);
    (1, mp);
  ]

let navi31 =
  blocks (11, 0, 0) (4, 3, 0) (3, 0, 0) (6, 0, 0) (6, 0, 0) (6, 0, 0) (13, 0, 0)

let mi300_blocks =
  blocks (9, 4, 3) (7, 9, 0) (1, 8, 0) (4, 4, 2) (4, 4, 2) (4, 4, 2) (13, 0, 6)

let queues =
  group ~timeout:10. "compute queues"
    [
      cases ~name:fst
        "the format's GC registers and the boot's agree where both name one"
        [
          ("GC 12.0.1", []);
          ("GC 11.0.0", navi31);
          ("GC 11.0.3", (11, (11, 0, 3)) :: navi31);
          ("GC 11.5.0", (11, (11, 5, 0)) :: navi31);
          ("GC 9.4.3", mi300_blocks);
          ("GC 9.5.0", (11, (9, 5, 0)) :: mi300_blocks);
        ]
        (fun (_, blocks) ->
          let d =
            List.fold_left
              (fun d (b, v) -> with_version d b v)
              (table "r9700.bin") (List.rev blocks)
          in
          match Regs.layout d with Ok _ -> () | Error why -> fail why);
      cases ~name:fst
        "a shader array selection and a broadcast hold the bits GRBM_GFX_INDEX \
         states"
        (* The SE index at bit 16, the SH (GC 9) or SA index at bit 8, and the
           SH or SA, instance and SE broadcasts at bits 29, 30 and 31
           (gc_9_4_3_sh_mask.h, gc_11_0_0_sh_mask.h, gc_12_0_0_sh_mask.h). *)
        [ ("GC 12.0.1", []); ("GC 11.0.0", navi31); ("GC 9.4.3", mi300_blocks) ]
        (fun (_, blocks) ->
          let l =
            layout
              (List.fold_left
                 (fun d (b, v) -> with_version d b v)
                 (table "r9700.bin") blocks)
          in
          equal int
            ((2 lsl 16) lor (1 lsl 8) lor (1 lsl 30))
            (Gfx.index l (`Array (2, 1)));
          equal int
            ((1 lsl 29) lor (1 lsl 30) lor (1 lsl 31))
            (Gfx.index l `All));
      test "a GC 12 queue's descriptor holds its ring, pointers and doorbell"
        (fun () ->
          let d =
            Gfx.mqd
              (layout (table "r9700.bin"))
              gfx12_queue ~base:0x8_0000_1000 ~kiq:false ~aql:false ~xcc:0
              ~xccs:1
          in
          equal int 2048 (String.length d);
          equal (list int)
            [
              0xc031_0800;
              0xffff_ffff;
              0x1000;
              8;
              0x5501;
              0x100_0000;
              0;
              0x3_0000;
              0x3_0008;
              0x4000_0018;
              0x515;
              0x30_0000;
              0x100;
              0x20_0000;
              9;
              0;
            ]
            (words d
               (List.map
                  (fun w -> 4 * w)
                  [
                    0;
                    23;
                    128;
                    129;
                    132;
                    136;
                    137;
                    139;
                    141;
                    143;
                    145;
                    149;
                    162;
                    165;
                    167;
                    181;
                  ])));
      test "a KIQ's descriptor is privileged and the kernel driver's" (fun () ->
          let d =
            Gfx.mqd
              (layout (table "r9700.bin"))
              gfx12_queue ~base:0 ~kiq:true ~aql:false ~xcc:0 ~xccs:1
          in
          equal int 0x3 (u32 d (4 * 145) lsr 30));
      test
        "an AQL queue across dies holds its die and chunk over the thread masks"
        (fun () ->
          let d =
            Gfx.mqd (gfx9_layout ()) gfx12_queue ~base:0 ~kiq:false ~aql:true
              ~xcc:2 ~xccs:8
          in
          equal (list int)
            [
              0xffff_ffff;
              0xffff_ffff;
              0xffff_ffff;
              0xffff_ffff;
              2;
              1;
              0x1000;
              1;
            ]
            (words d
               (List.map (fun w -> 4 * w) [ 23; 24; 26; 27; 39; 41; 226; 181 ])));
    ]

(* Sessions *)

module Boot = Rig_amd_pci.Boot

let sessions =
  let plan ?(mark = Boot.session) ?(dirty = 0) ?(fault = 0) ?(gc = (12, 0, 1))
      alive =
    Boot.plan ~mark ~dirty ~fault ~gc ~alive
  in
  let how = function
    | `Partial -> "partial"
    | `Full -> "full"
    | `Booted -> "booted"
  in
  cases ~name:fst "a GPU boots as its marks and firmware say"
    [
      ("clean mark", (plan true, `Partial));
      ("dirty mark, firmware running", (plan ~dirty:1 true, `Booted));
      ("dirty mark, no firmware", (plan ~dirty:1 false, `Full));
      ("clean mark and a fault", (plan ~fault:0x10 true, `Booted));
      ("dirty mark on GC 9.5.0", (plan ~dirty:1 ~gc:(9, 5, 0) true, `Partial));
      ("no mark, firmware running", (plan ~mark:0 true, `Booted));
      ("no mark, no firmware", (plan ~mark:0 false, `Full));
    ]
    (fun (_, (got, want)) -> equal string (how want) (how got))

(* The order of pci_restore_state: the PCI Express capability (DevCtl, LnkCtl,
   DevCtl2, LnkCtl2), the resizable BARs, the header from its end with the BARs,
   the command register last. *)
let restoration =
  test "a reset restores the resizable BARs before the BARs, the command last"
    (fun () ->
      equal
        (list (pair int int))
        [
          (0x88, 2);
          (0x90, 2);
          (0xa8, 2);
          (0xb0, 2);
          (0x208, 4);
          (0x210, 4);
          (0x3c, 4);
          (0x30, 4);
          (0x10, 4);
          (0x14, 4);
          (0x18, 4);
          (0x1c, 4);
          (0x20, 4);
          (0x24, 4);
          (0x0c, 4);
          (0x04, 2);
        ]
        (Boot.writes ~pcie:(Some 0x80) ~rebars:[ 0x208; 0x210 ]))

(* Interrupts *)

module Ih = Rig_amd_pci.Ih

(* An entry as the IH v6 lays it out: client in bits 0-7 of word 0, source in
   8-15, ring 16-23, VMID 24-27; PASID in bits 0-15 of word 3, node 16-23; four
   context words from word 4. Client and source IDs are soc15_ih_clientid.h's
   and the ivsrcid headers'. *)
let entry ?(ctx = [| 0; 0; 0; 0 |]) ?(vmid = 0) ?(pasid = 0) client source =
  [|
    client lor (source lsl 8) lor (vmid lsl 24);
    0;
    0;
    pasid;
    ctx.(0);
    ctx.(1);
    ctx.(2);
    ctx.(3);
  |]

let grbm_cp = 0x14
let soc21_gfx = 0xa
let soc15_sdma0 = 0x8
let soc15_utcl2 = 0x1b
let sdma4 = (4, 4, 2)
let sdma7 = (7, 0, 1)

let report =
  Testable.structural ~pp:(fun ppf -> function
    | Ih.Page_fault -> Format.pp_print_string ppf "Page_fault"
    | Ih.Fault s -> Format.fprintf ppf "Fault %S" s)

let decoded = option report

let interrupts =
  group ~timeout:10. "interrupts"
    [
      cases ~name:fst "a release reports nothing"
        [
          ("GFX12 end of pipe", (gfx12, sdma7, entry grbm_cp 0xb5));
          ("GFX12 copy trap", (gfx12, sdma7, entry soc21_gfx 0x31));
          ("GFX9 end of pipe", (gfx9, sdma4, entry grbm_cp 0xb5));
          ("GFX9 copy trap", (gfx9, sdma4, entry soc15_sdma0 0xe0));
        ]
        (fun (_, (gc, sdma, e)) -> equal decoded None (Ih.decode ~gc ~sdma e));
      cases ~name:fst "a page walker's fault is a page fault"
        [
          ("GFX12", (gfx12, sdma7, entry soc21_gfx 0));
          ("GFX9", (gfx9, sdma4, entry soc15_utcl2 0));
        ]
        (fun (_, (gc, sdma, e)) ->
          equal decoded (Some Ih.Page_fault) (Ih.decode ~gc ~sdma e));
      test "an error names its client, source and words" (fun () ->
          equal decoded
            (Some
               (Ih.Fault
                  "interrupt client=GRBM_CP src=CP_PRIV_REG_FAULT(184) ring=0 \
                   vmid=3(0) pasid=7 node=0 ctx=[0x1, 0x2, 0x3, 0x4]"))
            (Ih.decode ~gc:gfx11 ~sdma:(6, 0, 0)
               (entry ~vmid:3 ~pasid:7 ~ctx:[| 1; 2; 3; 4 |] grbm_cp 0xb8)));
      cases ~name:fst "a shader error names its kind"
        [
          ( "GFX12 illegal instruction",
            (gfx12, [| 1 lsl 21; 2 lsl 6; 0; 0 |], "ILLEGAL_INST") );
          ( "GFX9 memory violation",
            (gfx9, [| 2 lsl 26; 2 lsl 4; 0; 0 |], "MEMVIOL") );
        ]
        (fun (_, (gc, ctx, kind)) ->
          match Ih.decode ~gc ~sdma:sdma7 (entry ~ctx grbm_cp 0xef) with
          | Some (Ih.Fault s) -> ends_with ~affix:("shader error " ^ kind) s
          | r -> failf "%a" (Testable.pp decoded) r);
      cases ~name:fst "an interrupt that reports no error is none"
        [
          ( "a shader's other interrupt",
            (gfx12, entry ~ctx:[| 0; 1 lsl 6; 0; 0 |] grbm_cp 0xef) );
          ("an idle GC", (gfx9, entry grbm_cp 0xe9));
          ("a client with no sources", (gfx12, entry 0x1e 0));
        ]
        (fun (_, (gc, e)) -> equal decoded None (Ih.decode ~gc ~sdma:sdma7 e));
      test "an entry of another length raises" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Ih.decode ~gc:gfx12 ~sdma:sdma7 [| 0 |]));
    ]

(* Firmware *)

module Images = Rig_amd_pci.Images

(* Images built as amdgpu_ucode.h lays them out: the common header (size at 0,
   header size at 4, version at 8 and 10, ucode version at 16, ucode size at 20,
   ucode offset at 24), a version's 32-bit fields at their offsets, and pieces
   placed at theirs. Every piece is a distinct string, so that a test sees which
   bytes went where. *)
let image ~major ~minor ?(ucode_version = 0) ?(ucode = (0x100, 0)) fields pieces
    =
  let size =
    List.fold_left (fun m (at, p) -> max m (at + String.length p)) 0x100 pieces
  in
  let b = Bytes.make size '\000' in
  let u16 at v = Bytes.set_uint16_le b at v in
  let u32 at v = Bytes.set_int32_le b at (Int32.of_int v) in
  u32 0 size;
  u32 4 0x100;
  u16 8 major;
  u16 10 minor;
  u32 16 ucode_version;
  u32 20 (snd ucode);
  u32 24 (fst ucode);
  List.iter (fun (at, v) -> u32 at v) fields;
  List.iter (fun (at, p) -> Bytes.blit_string p 0 b at (String.length p)) pieces;
  Bytes.to_string b

(* A payload at 0x100 of the given pieces, back to back: the image's ucode
   (offset, size) and each piece's offset. *)
let payload pieces =
  let at, offs =
    List.fold_left
      (fun (at, offs) p -> (at + String.length p, offs @ [ at ]))
      (0x100, []) pieces
  in
  ((0x100, at - 0x100), offs, List.combine offs pieces)

(* The PSP's image, version 2.1: three components, the third auxiliary. *)
let sos =
  let ucode, _, placed = payload [ "SOS!"; "KDB"; "AUX" ] in
  image ~major:2 ~minor:1 ~ucode
    [
      (32, 3);
      (36, 2);
      (40, 1);
      (48, 0);
      (52, 4);
      (56, 3);
      (64, 4);
      (68, 3);
      (72, 9);
      (80, 7);
      (84, 3);
    ]
    placed

let smu = image ~major:2 ~minor:0 ~ucode:(0x100, 3) [] [ (0x100, "SMU") ]

let sdma7 =
  image ~major:3 ~minor:0 ~ucode:(0x100, 4) [ (40, 4) ] [ (0x100, "TH0!") ]

(* An RS64 engine's image: code at the ucode offset, its stack, its start. *)
let rs64 ~ucode_version code stack start =
  image ~major:2 ~minor:0 ~ucode_version
    ~ucode:(0x100, String.length code)
    [
      (36, String.length code);
      (44, String.length stack);
      (48, 0x180);
      (52, start land 0xffff_ffff);
      (56, start lsr 32);
    ]
    [ (0x100, code); (0x180, stack) ]

let imu =
  image ~major:1 ~minor:0 ~ucode:(0x100, 7)
    [ (32, 3); (40, 4) ]
    [ (0x100, "IRMDRAM") ]

let rlc23 =
  image ~major:2 ~minor:3 ~ucode:(0x100, 5)
    [
      (156, 4);
      (160, 0x140);
      (164, 4);
      (168, 0x150);
      (180, 2);
      (184, 0x160);
      (196, 2);
      (200, 0x170);
    ]
    [
      (0x100, "RLC-G");
      (0x140, "IRAM");
      (0x150, "DRAM");
      (0x160, "RP");
      (0x170, "RV");
    ]

let r9700_images =
  [
    ("amdgpu/psp_14_0_3_sos.bin", sos);
    ("amdgpu/smu_14_0_3.bin", smu);
    ("amdgpu/sdma_7_0_1.bin", sdma7);
    ( "amdgpu/gc_12_0_1_pfp.bin",
      rs64 ~ucode_version:1 "PFPCODE" "PS" 0x1_0000_1000 );
    ("amdgpu/gc_12_0_1_me.bin", rs64 ~ucode_version:2 "MECODE" "MS" 0x2000);
    ("amdgpu/gc_12_0_1_mec.bin", rs64 ~ucode_version:0x2a "MECCODE" "CS" 0x3000);
    ("amdgpu/gc_12_0_1_imu.bin", imu);
    ("amdgpu/gc_12_0_1_rlc.bin", rlc23);
  ]

let finder images path ~digest =
  if String.length digest <> 64 then failf "%s: digest %S" path digest;
  match List.assoc_opt path images with
  | Some img -> Ok img
  | None -> Error (strf "no %s" path)

let pieces = list (pair (list int) string)

let load images d =
  match Images.load (finder images) d with
  | Ok fw -> fw
  | Error why -> failf "load: %s" why

(* A GPU of the given block versions. *)
let gpu ~mp0 ~mp1 ~sdma ~gc =
  {
    Discovery.versions = [ (11, gc); (42, sdma); (255, mp0); (1, mp1) ];
    bases = [];
    harvested = [];
    gc =
      {
        engines = 1;
        arrays = 1;
        units = 1;
        scratch_slots = 1;
        waves = 1;
        lds = 1;
      };
  }

let mi300 = gpu ~mp0:(13, 0, 6) ~mp1:(13, 0, 6) ~sdma:(4, 4, 2) ~gc:(9, 4, 3)

let mi300_images =
  let ucode, _, placed = payload [ "MECC"; "JUMP" ] in
  [
    ("amdgpu/psp_13_0_6_sos.bin", sos);
    ( "amdgpu/smu_13_0_6.bin",
      image ~major:2 ~minor:1 ~ucode:(0x100, 0)
        [
          (36, 2);
          (40, 0x80);
          (0x80, 0x1234);
          (0x84, 0x140);
          (0x88, 3);
          (0x8c, 0x50325358);
          (0x90, 0x150);
          (0x94, 4);
        ]
        [ (0x140, "OLD"); (0x150, "P2S!") ] );
    ( "amdgpu/sdma_4_4_2.bin",
      image ~major:1 ~minor:0 ~ucode:(0x100, 4) [] [ (0x100, "SDMA") ] );
    ( "amdgpu/gc_9_4_3_mec.bin",
      image ~major:1 ~minor:0 ~ucode_version:7 ~ucode
        [ (36, 1); (40, 1) ]
        placed );
    ( "amdgpu/gc_9_4_3_rlc.bin",
      image ~major:2 ~minor:1 ~ucode:(0x100, 3)
        [
          (116, 2); (120, 0x140); (132, 2); (136, 0x150); (148, 2); (152, 0x160);
        ]
        [ (0x100, "RLC"); (0x140, "CN"); (0x150, "GP"); (0x160, "SR") ] );
  ]

let firmware =
  group ~timeout:10. "firmware"
    [
      test "the R9700's table names its images in load order" (fun () ->
          equal
            (result (list string) string)
            (Ok
               [
                 "amdgpu/psp_14_0_3_sos.bin";
                 "amdgpu/smu_14_0_3.bin";
                 "amdgpu/sdma_7_0_1.bin";
                 "amdgpu/gc_12_0_1_pfp.bin";
                 "amdgpu/gc_12_0_1_me.bin";
                 "amdgpu/gc_12_0_1_mec.bin";
                 "amdgpu/gc_12_0_1_imu.bin";
                 "amdgpu/gc_12_0_1_rlc.bin";
               ])
            (Images.names (table "r9700.bin")));
      test "a GC 12 GPU's images cut into the PSP's pieces" (fun () ->
          let fw = load r9700_images (table "r9700.bin") in
          equal (list (pair int string)) [ (1, "SOS!"); (3, "KDB") ] fw.sos;
          equal (option (pair (list int) string)) (Some ([ 18 ], "SMU")) fw.smu;
          equal pieces
            [
              ([ 71 ], "TH0!");
              ([ 87 ], "PFPCODE");
              ([ 90 ], "PS");
              ([ 88 ], "MECODE");
              ([ 92 ], "MS");
              ([ 89 ], "MECCODE");
              ([ 94 ], "CS");
              ([ 68 ], "IRM");
              ([ 69 ], "DRAM");
              ([ 26 ], "IRAM");
              ([ 48 ], "DRAM");
              ([ 25 ], "RP");
              ([ 7 ], "RV");
              ([ 8 ], "RLC-G");
            ]
            fw.pieces;
          equal
            (list (pair string int))
            [ ("PFP", 0x1_0000_1000); ("ME", 0x2000); ("MEC", 0x3000) ]
            fw.starts;
          equal int 0x2a fw.mec);
      test "a GC 9 GPU's images cut into the PSP's pieces" (fun () ->
          let fw = load mi300_images mi300 in
          equal (option (pair (list int) string)) None fw.smu;
          equal pieces
            [
              ([ 129 ], "P2S!");
              ([ 9; 10; 52; 53 ], "SDMA");
              ([ 4 ], "MECC");
              ([ 5 ], "JUMP");
              ([ 22 ], "CN");
              ([ 20 ], "GP");
              ([ 21 ], "SR");
              ([ 8 ], "RLC");
            ]
            fw.pieces;
          equal (list (pair string int)) [] fw.starts;
          equal int 7 fw.mec);
      test "an MP1 of 13.0.12 has no image" (fun () ->
          let d =
            gpu ~mp0:(13, 0, 12) ~mp1:(13, 0, 12) ~sdma:(4, 4, 5) ~gc:(9, 5, 0)
          in
          equal
            (result (list string) string)
            (Ok
               [
                 "amdgpu/psp_13_0_12_sos.bin";
                 "amdgpu/sdma_4_4_5.bin";
                 "amdgpu/gc_9_5_0_mec.bin";
                 "amdgpu/gc_9_5_0_rlc.bin";
               ])
            (Images.names d));
      test "a version with no pinned image is refused, naming the block"
        (fun () ->
          let d =
            gpu ~mp0:(13, 0, 5) ~mp1:(13, 0, 6) ~sdma:(4, 4, 2) ~gc:(9, 4, 3)
          in
          let r = Images.load (finder mi300_images) d in
          equal (result pass string)
            (Error
               "MP0 13.0.5 is a version this library does not boot: no image \
                amdgpu/psp_13_0_5_sos.bin")
            (Result.map ignore r));
      test "a GPU without a block a boot needs is refused" (fun () ->
          let d =
            { mi300 with versions = List.remove_assoc 42 mi300.versions }
          in
          equal
            (result (list string) string)
            (Error "the GPU has no SDMA0 block") (Images.names d));
      test "find's refusal is the load's" (fun () ->
          equal (result pass string) (Error "no amdgpu/smu_13_0_6.bin")
            (Result.map ignore
               (Images.load
                  (finder
                     (List.remove_assoc "amdgpu/smu_13_0_6.bin" mi300_images))
                  mi300)));
      cases ~name:fst "a damaged image is refused, naming it"
        [
          ( "a header of another version",
            ("amdgpu/sdma_4_4_2.bin", image ~major:4 ~minor:0 [] []) );
          ( "a piece past the end",
            ( "amdgpu/gc_9_4_3_rlc.bin",
              image ~major:2 ~minor:1 ~ucode:(0x100, 0x1000) [] [] ) );
          ( "a header cut short",
            ("amdgpu/psp_13_0_6_sos.bin", String.sub sos 0 20) );
        ]
        (fun (_, (path, img)) ->
          match
            Images.load
              (finder ((path, img) :: List.remove_assoc path mi300_images))
              mi300
          with
          | Ok _ -> failf "%s was read" path
          | Error why -> starts_with ~affix:(path ^ ": ") why);
      test "every pinned image has a BLAKE2b-256 digest, under amdgpu/"
        (fun () ->
          List.iter
            (fun (path, digest) ->
              equal ~msg:path int 64 (String.length digest);
              starts_with ~msg:path ~affix:"amdgpu/" path)
            Images.pinned;
          let paths = List.map fst Images.pinned in
          equal int (List.length paths)
            (List.length (List.sort_uniq compare paths)));
    ]

(* Numbering *)

module Tree = Rig_pci_support.Tree

(* A machine whose AMD GPUs are a display controller, a processing accelerator
   and a display controller of another subclass, beside an AMD audio function
   and another vendor's GPU, listed out of bus order. *)
let machine () =
  let fn bus vendor class_ = { (Tree.gpu bus) with vendor; class_ } in
  Tree.make
    [
      fn "0000:83:00.0" 0x1002 0x120000;
      fn "0000:03:00.1" 0x1002 0x040300;
      fn "0000:03:00.0" 0x1002 0x030000;
      fn "0000:01:00.0" 0x10de 0x030000;
      fn "0000:02:00.0" 0x1002 0x038000;
    ]

let numbering =
  group "numbering"
    [
      test "GPUs are AMD's display controllers and accelerators in bus order"
        (fun () ->
          let root = machine () in
          equal (list string)
            [ "0000:02:00.0"; "0000:03:00.0"; "0000:83:00.0" ]
            (Rig_amd_pci.gpus_at root);
          equal int 3 (Rig_amd_pci.count ~machine:(Rig_pci.Machine.at root) ()));
      test "a machine with no PCI functions has no GPU" (fun () ->
          equal int 0
            (Rig_amd_pci.count ~machine:(Rig_pci.Machine.at (Tree.make [])) ()));
      cases ~name:fst "GPUs are named by number"
        [
          ("0", (0, "AMD-PCI"));
          ("1", (1, "AMD-PCI:1"));
          ("12", (12, "AMD-PCI:12"));
        ]
        (fun (_, (i, name)) -> equal string name (Rig_amd_pci.device_name i));
      test "an open past the last GPU names the GPU and the count" (fun () ->
          let machine = Rig_pci.Machine.at (machine ()) in
          match Rig_amd_pci.open_ ~machine ~firmware:[] 3 with
          | Ok _ -> fail "GPU 3 opened"
          | Error why ->
              equal string "AMD-PCI:3: no such GPU; the machine has 3" why);
      cases ~name:fst "a negative GPU number raises"
        [
          ("device_name", fun () -> ignore (Rig_amd_pci.device_name (-1)));
          ("open_", fun () -> ignore (Rig_amd_pci.open_ ~firmware:[] (-1)));
          ("detach", fun () -> ignore (Rig_amd_pci.detach (-1)));
          ("attach", fun () -> ignore (Rig_amd_pci.attach (-1)));
          ("reset", fun () -> ignore (Rig_amd_pci.reset (-1)));
        ]
        (fun (_, f) -> raises_match Exn.invalid_arg f);
    ]

(* Letting go: a fixture tree's unbound AMD GPU at 0000:05:00.0, and what of
   amdgpu's device the kernel still lists for it. amdgpu's release removes its
   node from KFD's topology at its start and its [ip_discovery] directory at its
   end, so either one listed means amdgpu has not let go. *)

let r9700 = "0000:05:00.0"

let unbound () =
  Tree.make [ { (Tree.gpu r9700) with vendor = 0x1002; class_ = 0x030000 } ]

let refusal root =
  match Rig_amd_pci.open_ ~machine:(Rig_pci.Machine.at root) ~firmware:[] 0 with
  | Ok _ -> fail "a GPU opened"
  | Error why -> why

let letting_go =
  group "letting go"
    [
      test "an unbound GPU whose ip_discovery stays is refused an open"
        (fun () ->
          let root = unbound () in
          Tree.add root
            (strf "sys/bus/pci/devices/%s/ip_discovery/die/0/GC/0/major" r9700)
            "12\n";
          Windtrap.contains ~sub:"ip_discovery" (refusal root));
      test "an unbound GPU KFD's topology lists is refused an open" (fun () ->
          let root = unbound () in
          Tree.add root "sys/class/kfd/kfd/topology/nodes/1/properties"
            "domain 0\nlocation_id 1280\n";
          Windtrap.contains ~sub:"topology" (refusal root));
      test "an unbound GPU amdgpu let go of is not refused for it" (fun () ->
          let root = unbound () in
          Tree.add root "sys/class/kfd/kfd/topology/nodes/1/properties"
            "domain 0\nlocation_id 17152\n";
          let why = refusal root in
          if contains ~sub:"not let go" why then
            failf "refused as held by amdgpu: %s" why);
    ]

let () =
  exit
    (run "rig_amd_pci"
       [
         discovery;
         damaged;
         dies;
         registers;
         apertures;
         page_tables;
         power;
         security;
         queues;
         sessions;
         restoration;
         interrupts;
         firmware;
         numbering;
         letting_go;
       ])
