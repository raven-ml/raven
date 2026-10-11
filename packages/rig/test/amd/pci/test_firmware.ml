(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Firmware images, built as their headers lay them out, cut into the pieces the
   security processor loads. *)

open Windtrap
open Rig_amd_pci

let strf = Printf.sprintf

let table name =
  let path = "../fixtures/" ^ name in
  match
    Discovery.of_string (In_channel.with_open_bin path In_channel.input_all)
  with
  | Ok d -> d
  | Error why -> failf "%s: %s" name why

(* The blocks that name images: GC, SDMA0, MP0 and MP1. *)
let gc = 11
let sdma0 = 42
let mp0 = 255
let mp1 = 1

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

let finder images path =
  match List.assoc_opt path images with
  | Some img -> Ok img
  | None -> Error (strf "no %s" path)

let piece_bytes (piece : Firmware.piece) =
  String.sub piece.source piece.off piece.len

let piece_contents (types, piece) = (types, piece_bytes piece)
let pieces = list (pair (list int) string)

let load images d =
  match Firmware.load (finder images) d with
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

let firmware_round_trip =
  let gen =
    let open Gen in
    let+ version = of_list [ 1; 2; 3 ]
    and+ gap = int_range 0 32
    and+ code = string_of ~size:(int_range 0 64) char
    and+ control = string_of ~size:(int_range 0 64) char in
    (version, gap, code, control)
  in
  let gen =
    Gen.with_pp
      (fun ppf (v, gap, code, control) ->
        Format.fprintf ppf "(%d, %d, %S, %S)" v gap code control)
      gen
  in
  prop "SDMA images preserve their payloads"
    ~examples:[ (1, 0, "", ""); (2, 1, "x", "\000"); (3, 0, "\000", "") ]
    gen
    (fun (version, gap, code, control) ->
      cover "empty payload" (code = "");
      cover "one byte" (String.length code = 1);
      cover "SDMA 1" (version = 1);
      cover "SDMA 2" (version = 2);
      cover "SDMA 3" (version = 3);
      let off = 0x100 + gap in
      let encode (code, control) =
        let control_off = off + String.length code + gap in
        let fields =
          match version with
          | 1 -> []
          | 2 ->
              [
                (36, String.length code);
                (48, control_off);
                (52, String.length control);
              ]
          | _ -> [ (40, String.length code) ]
        in
        image ~major:version ~minor:0
          ~ucode:(off, String.length code)
          fields
          [ (off, code); (control_off, control) ]
      in
      let decode img =
        let path = "amdgpu/sdma_7_0_1.bin" in
        let fw =
          load
            ((path, img) :: List.remove_assoc path r9700_images)
            (table "r9700.bin")
        in
        let code_types = if version = 1 then [ 9; 10; 52; 53 ] else [ 71 ] in
        let code = piece_bytes (List.assoc code_types fw.pieces) in
        let control =
          if version = 2 then piece_bytes (List.assoc [ 72 ] fw.pieces) else ""
        in
        (code, control)
      in
      Law.round_trip (pair string string) string encode decode
        (code, if version = 2 then control else ""))

let firmware =
  group ~timeout:10. "firmware"
    [
      firmware_round_trip;
      test "pieces keep their source after the firmware cache replaces it"
        (fun () ->
          let path = Filename.temp_file "rig_firmware" ".bin" in
          let replacement = path ^ ".new" in
          Fun.protect
            ~finally:(fun () ->
              List.iter
                (fun file -> if Sys.file_exists file then Sys.remove file)
                [ path; replacement ])
            (fun () ->
              let write file =
                Out_channel.with_open_bin file (fun oc ->
                    output_string oc sdma7)
              in
              let collect () =
                Gc.full_major ();
                Gc.full_major ();
                Gc.full_major ()
              in
              write path;
              let digest = Rig_pci.Firmware.digest sdma7 in
              let find file =
                if file = "amdgpu/sdma_7_0_1.bin" then
                  let img =
                    require_ok
                      (Rig_pci.Firmware.find
                         [ Filename.dirname path ]
                         (Filename.basename path) ~digest)
                  in
                  Ok img.contents
                else finder r9700_images file
              in
              let d = table "r9700.bin" in
              let old, source =
                let fw = require_ok (Firmware.load find d) in
                let piece = List.assoc [ 71 ] fw.pieces in
                let source = Weak.create 1 in
                Weak.set source 0 (Some piece.source);
                (ref (Some fw), source)
              in
              write replacement;
              Unix.rename replacement path;
              let current = require_ok (Firmware.load find d) in
              collect ();
              equal ~msg:"the old set keeps its image" bool true
                (Weak.check source 0);
              let retained () =
                let fw = require_some !old in
                let piece = List.assoc [ 71 ] fw.pieces in
                let current = List.assoc [ 71 ] current.pieces in
                equal string "TH0!" (piece_bytes piece);
                equal ~msg:"the cache read the replacement" bool false
                  (piece.source == current.source)
              in
              retained ();
              old := None;
              collect ();
              equal ~msg:"dropping the old set releases its image" bool false
                (Weak.check source 0)));
      test "the R9700's blocks name its images in load order" (fun () ->
          equal
            (result (list (pair int string)) string)
            (Ok
               [
                 (mp0, "amdgpu/psp_14_0_3_sos.bin");
                 (mp1, "amdgpu/smu_14_0_3.bin");
                 (sdma0, "amdgpu/sdma_7_0_1.bin");
                 (gc, "amdgpu/gc_12_0_1_pfp.bin");
                 (gc, "amdgpu/gc_12_0_1_me.bin");
                 (gc, "amdgpu/gc_12_0_1_mec.bin");
                 (gc, "amdgpu/gc_12_0_1_imu.bin");
                 (gc, "amdgpu/gc_12_0_1_rlc.bin");
               ])
            (Firmware.names (table "r9700.bin")));
      test "a GC 12 GPU's images cut into the PSP's pieces" (fun () ->
          let fw = load r9700_images (table "r9700.bin") in
          equal
            (list (pair int string))
            [ (1, "SOS!"); (3, "KDB") ]
            (List.map piece_contents fw.sos);
          equal
            (option (pair (list int) string))
            (Some ([ 18 ], "SMU"))
            (Option.map piece_contents fw.smu);
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
            (List.map piece_contents fw.pieces);
          equal
            (list (pair string int))
            [ ("PFP", 0x1_0000_1000); ("ME", 0x2000); ("MEC", 0x3000) ]
            fw.starts;
          equal int 0x2a fw.mec);
      test "a GC 9 GPU's images cut into the PSP's pieces" (fun () ->
          let fw = load mi300_images mi300 in
          equal
            (option (pair (list int) string))
            None
            (Option.map piece_contents fw.smu);
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
            (List.map piece_contents fw.pieces);
          equal (list (pair string int)) [] fw.starts;
          equal int 7 fw.mec);
      test "an MP1 of 13.0.12 has no image" (fun () ->
          let d =
            gpu ~mp0:(13, 0, 12) ~mp1:(13, 0, 12) ~sdma:(4, 4, 5) ~gc:(9, 5, 0)
          in
          equal
            (result (list (pair int string)) string)
            (Ok
               [
                 (mp0, "amdgpu/psp_13_0_12_sos.bin");
                 (sdma0, "amdgpu/sdma_4_4_5.bin");
                 (gc, "amdgpu/gc_9_5_0_mec.bin");
                 (gc, "amdgpu/gc_9_5_0_rlc.bin");
               ])
            (Firmware.names d));
      test "a GPU without a block a boot needs is refused" (fun () ->
          let d =
            { mi300 with versions = List.remove_assoc 42 mi300.versions }
          in
          equal
            (result (list (pair int string)) string)
            (Error "the GPU has no SDMA0 block") (Firmware.names d));
      test "find's refusal is the load's" (fun () ->
          equal (result pass string) (Error "no amdgpu/smu_13_0_6.bin")
            (Result.map ignore
               (Firmware.load
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
            Firmware.load
              (finder ((path, img) :: List.remove_assoc path mi300_images))
              mi300
          with
          | Ok _ -> failf "%s was read" path
          | Error why -> starts_with ~affix:(path ^ ": ") why);
    ]

let () = exit (run "rig_amd_pci.firmware" [ firmware ])
