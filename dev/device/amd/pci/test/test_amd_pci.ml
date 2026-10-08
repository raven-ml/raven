(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Discovery = Device_amd_pci.Discovery

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

module Regs = Device_amd_pci.Regs

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
            (fun (r : Device_amd_abi.Register.t) ->
              let a = Regs.address l r.name in
              if
                not
                  (List.exists
                     (fun (lo, hi) -> lo <= a && a <= hi)
                     (Regs.guarded l))
              then failf "%s at 0x%x is in no guarded range" r.name a)
            (Device_amd_abi.Register.registers (Regs.gpu l)));
    ]

(* Firmware *)

module Images = Device_amd_pci.Images

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
      test "every pinned image has a BLAKE2b-256 digest and its URL" (fun () ->
          List.iter
            (fun (path, digest, url) ->
              equal ~msg:path int 64 (String.length digest);
              ends_with ~msg:path ~affix:("/" ^ path) url)
            Images.pinned;
          let paths = List.map (fun (p, _, _) -> p) Images.pinned in
          equal int (List.length paths)
            (List.length (List.sort_uniq compare paths)));
    ]

let () = exit (run "device_amd_pci" [ discovery; damaged; registers; firmware ])
