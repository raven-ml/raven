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

(* The blocks that name images: GC, SDMA0 and MP0. *)
let gc = 11
let sdma0 = 42
let mp0 = 255

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

let navi31 = gpu ~mp0:(13, 0, 0) ~mp1:(13, 0, 0) ~sdma:(6, 0, 0) ~gc:(11, 0, 0)

(* A GC 11 GPU's images, of the header versions the other GPUs' lack: the PSP's
   2.0, which lists all its components; SDMA 2.0, its control thread's code
   apart; the RLC's 2.2. *)
let navi31_images =
  let ucode, _, placed = payload [ "SYS"; "SOS!" ] in
  [
    ( "amdgpu/psp_13_0_0_sos.bin",
      image ~major:2 ~minor:0 ~ucode
        [ (32, 2); (36, 2); (44, 0); (48, 3); (52, 1); (60, 3); (64, 4) ]
        placed );
    ( "amdgpu/smu_13_0_0.bin",
      image ~major:2 ~minor:1 ~ucode:(0x100, 3) [] [ (0x100, "PMF") ] );
    ( "amdgpu/sdma_6_0_0.bin",
      image ~major:2 ~minor:0 ~ucode:(0x100, 4)
        [ (36, 4); (48, 0x110); (52, 3) ]
        [ (0x100, "CTX!"); (0x110, "CTL") ] );
    ("amdgpu/gc_11_0_0_mec.bin", rs64 ~ucode_version:0x2b "MECCODE" "CS" 0x3000);
    ("amdgpu/gc_11_0_0_imu.bin", imu);
    ( "amdgpu/gc_11_0_0_rlc.bin",
      image ~major:2 ~minor:2 ~ucode:(0x100, 5)
        [ (156, 4); (160, 0x140); (164, 4); (168, 0x150) ]
        [ (0x100, "RLC-G"); (0x140, "IRAM"); (0x150, "DRAM") ] );
  ]

(* A damage to an image: cut to its first [n] bytes, or little-endian words of 2
   or 4 bytes written at offsets, most of them in the headers' first 256
   bytes. *)
type damage = Cut of int | Words of (int * int * int) list

let pp_damage ppf = function
  | Cut n -> Format.fprintf ppf "cut to %d bytes" n
  | Words ws ->
      Format.fprintf ppf "%s"
        (String.concat ", "
           (List.map (fun (at, w, v) -> strf "%d bytes 0x%x at %d" w v at) ws))

let damage img d =
  match d with
  | Cut n -> String.sub img 0 n
  | Words ws ->
      let b = Bytes.of_string img in
      List.iter
        (fun (at, w, v) ->
          if w = 2 then Bytes.set_uint16_le b at (v land 0xffff)
          else Bytes.set_int32_le b at (Int32.of_int v))
        ws;
      Bytes.to_string b

let gpus =
  [
    ("R9700", (r9700_images, lazy (table "r9700.bin")));
    ("MI300", (mi300_images, lazy mi300));
    ("Navi 31", (navi31_images, lazy navi31));
  ]

(* One image of one GPU, damaged. Lengths and values sit at the edges: no byte,
   a header's 256, one byte short, the image's length and its neighbours, and
   the largest 16- and 32-bit values. *)
let damaged =
  let open Gen in
  let* gpu, (images, _) = of_list gpus in
  let* path, img = of_list images in
  let len = String.length img in
  let cut =
    frequency
      [
        (1, of_list [ 0; 0x100; len - 1 ]);
        (1, int_range 0 0x100);
        (1, int_range 0 len);
      ]
  in
  let word =
    let* w = of_list [ 2; 4 ] in
    let+ at =
      frequency
        [
          (3, map (fun i -> 2 * i) (int_range 0 ((0x100 - w) / 2)));
          (1, int_range 0 (len - w));
        ]
    and+ v =
      frequency
        [
          ( 2,
            of_list
              [ 0; 1; 0xffff; 0x7fff_ffff; 0xffff_ffff; len - 1; len; len + 1 ]
          );
          (1, int_range 0 0xffff_ffff);
        ]
    in
    (at, w, v)
  in
  let+ d =
    frequency
      [
        (1, map (fun n -> Cut n) cut);
        (1, map (fun ws -> Words ws) (list ~size:(int_range 1 3) word));
      ]
  in
  (gpu, path, d)

let damaged =
  Gen.with_pp
    (fun ppf (gpu, path, d) ->
      Format.fprintf ppf "%s's %s, %a" gpu path pp_damage d)
    damaged

(* An image damaged anywhere is read or refused, naming it, and never raises; a
   piece it reads lies in its image. *)
let damage_law (gpu, path, d) =
  let images, table = List.assoc gpu gpus in
  let img = damage (List.assoc path images) d in
  let images = (path, img) :: List.remove_assoc path images in
  let r = Firmware.load (finder images) (Lazy.force table) in
  cover "a cut image" (match d with Cut _ -> true | Words _ -> false);
  cover "an image with words written"
    (match d with Words _ -> true | Cut _ -> false);
  cover "refused" (Result.is_error r);
  cover "read" (Result.is_ok r);
  match r with
  | Error why -> starts_with ~affix:(path ^ ": ") why
  | Ok fw ->
      let inside (piece : Firmware.piece) =
        at_least int ~msg:"a piece's offset" ~than:0 piece.off;
        at_least int ~msg:"a piece's length" ~than:0 piece.len;
        at_most int ~msg:"a piece's end"
          ~than:(String.length piece.source)
          (piece.off + piece.len)
      in
      List.iter (fun (_, p) -> inside p) fw.sos;
      Option.iter (fun (_, p) -> inside p) fw.smu;
      List.iter (fun (_, p) -> inside p) fw.pieces

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
      test "a GC 11 GPU's images cut into the PSP's pieces" (fun () ->
          let fw = load navi31_images navi31 in
          equal
            (list (pair int string))
            [ (2, "SYS"); (1, "SOS!") ]
            (List.map piece_contents fw.sos);
          equal
            (option (pair (list int) string))
            (Some ([ 18 ], "PMF"))
            (Option.map piece_contents fw.smu);
          equal pieces
            [
              ([ 72 ], "CTL");
              ([ 71 ], "CTX!");
              ([ 89 ], "MECCODE");
              ([ 94 ], "CS");
              ([ 68 ], "IRM");
              ([ 69 ], "DRAM");
              ([ 26 ], "IRAM");
              ([ 48 ], "DRAM");
              ([ 8 ], "RLC-G");
            ]
            (List.map piece_contents fw.pieces);
          equal (list (pair string int)) [ ("MEC", 0x3000) ] fw.starts;
          equal int 0x2b fw.mec);
      prop "a damaged image is read or refused, naming it, and never raises"
        damaged damage_law;
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
