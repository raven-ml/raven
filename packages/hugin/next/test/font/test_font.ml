(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_next_gg
open Hugin_next_font
open Build

let read path = In_channel.with_open_bin path In_channel.input_all
let regular_ttf = read "../../font/fonts/Inter-Regular.ttf"
let bold_ttf = read "../../font/fonts/Inter-Bold.ttf"
let em units = Float.of_int units /. 2048.
let font_t = Testable.make ~pp:Font.pp ~equal:Font.equal

let decoded data =
  match Font.of_string data with
  | Ok f -> f
  | Error e -> failf "%a" Font.pp_error e

(* The bundled faces decoded inside the tests: [Font.regular] and [Font.bold]
   are decoded when the library initialises, out of the tests' sight. *)
let regular () = decoded regular_ttf
let bold () = decoded bold_ttf

(* Errors *)

let is_error kind ?(sub = "") result =
  let e = require_error ~pp:Font.pp result in
  match (kind, e) with
  | `Malformed, Font.Malformed msg
  | `Unsupported, Font.Unsupported msg
  | `Io, Font.Io msg ->
      contains ~sub msg
  | _ -> failf "unexpected error: %a" Font.pp_error e

let rejected =
  let glyf = snd (table regular_ttf "glyf")
  and loca = snd (table regular_ttf "loca") in
  (* Glyph 174, é, is a composite whose first component is glyph 171. *)
  let eacute = glyf + (2 * get_u16 regular_ttf (loca + (2 * 174))) in
  [
    ("an empty string", `Malformed, "", "the file ends before byte 4");
    ("a truncated file", `Malformed, String.sub regular_ttf 0 100, "");
    ("text", `Malformed, "not a font at all, only text", "not an OpenType file");
    ("CFF outlines", `Unsupported, set_u32 regular_ttf 0 0x4F54544F, "CFF");
    ( "a collection",
      `Unsupported,
      set_u32 regular_ttf 0 0x74746366,
      "collection" );
    ("WOFF", `Unsupported, set_u32 regular_ttf 0 0x774F4646, "WOFF");
    ( "a missing 'hmtx' table",
      `Malformed,
      rename regular_ttf "hmtx" "hmtX",
      "no 'hmtx'" );
    ( "no 'glyf' outlines",
      `Unsupported,
      rename regular_ttf "glyf" "glyX",
      "glyf" );
    ( "a wrong 'head' magic",
      `Malformed,
      set_u32 regular_ttf (head_pos regular_ttf + 12) 0,
      "magic" );
    ( "zero units per em",
      `Malformed,
      set_u16 regular_ttf (head_pos regular_ttf + 18) 0,
      "units per em" );
    ( "a table past the end of the file",
      `Malformed,
      set_u32 regular_ttf (fst (table regular_ttf "name") + 12) 1_000_000,
      "'name' table extends past" );
    ( "glyph offsets out of order",
      `Malformed,
      set_u16 regular_ttf (loca + 2) (get_u16 regular_ttf (loca + 4) + 1),
      "'loca'" );
    ( "a composite containing itself",
      `Malformed,
      set_u16 regular_ttf (eacute + 12) 174,
      "contains itself" );
    ( "a missing component",
      `Malformed,
      set_u16 regular_ttf (eacute + 12) 9999,
      "does not exist" );
    ( "a component placed by point matching",
      `Unsupported,
      set_u16 regular_ttf (eacute + 10)
        (get_u16 regular_ttf (eacute + 10) land lnot 2),
      "point matching" );
    ( "a truncated 'hmtx' table",
      `Malformed,
      set_u32 regular_ttf (fst (table regular_ttf "hmtx") + 12) 100,
      Printf.sprintf "the 'hmtx' table ends before byte %d"
        (snd (table regular_ttf "hmtx") + 102) );
    ( "a truncated glyph",
      `Malformed,
      set_u16 regular_ttf (loca + 94) (get_u16 regular_ttf (loca + 92) + 6),
      (* H has one contour: its instruction length at byte 12 is cut. *)
      Printf.sprintf "the data of glyph 46 ends before byte %d"
        (glyf + (2 * get_u16 regular_ttf (loca + 92)) + 14) );
    ( "a truncated 'cmap' table",
      `Malformed,
      set_u32 regular_ttf (fst (table regular_ttf "cmap") + 12) 1888,
      "the 'cmap' table ends before byte" );
    ( "a character map subtable past its table",
      `Malformed,
      set_u32 regular_ttf (fst (table regular_ttf "cmap") + 12) 10,
      "the 'cmap' table ends before byte" );
    ( "format 4 segments out of order",
      `Malformed,
      (let sub = snd (table regular_ttf "cmap") + 20 in
       set_u16 regular_ttf (sub + 16) (get_u16 regular_ttf (sub + 14))),
      "segments not in increasing order" );
    ( "a component past the last glyph",
      `Malformed,
      set_u16 regular_ttf (eacute + 12) 720,
      "component 720 does not exist" );
    ( "15 units per em",
      `Malformed,
      set_u16 regular_ttf (head_pos regular_ttf + 18) 15,
      "units per em" );
    ( "16385 units per em",
      `Malformed,
      set_u16 regular_ttf (head_pos regular_ttf + 18) 16385,
      "units per em" );
    ( "glyph data past the 'glyf' table",
      `Malformed,
      set_u16 regular_ttf (loca + 1440) 30000,
      "glyph data past the 'glyf' table" );
    ( "no glyphs",
      `Malformed,
      set_u16 regular_ttf (snd (table regular_ttf "maxp") + 4) 0,
      "no glyphs" );
    ( "no horizontal metrics",
      `Malformed,
      set_u16 regular_ttf (snd (table regular_ttf "hhea") + 34) 0,
      "no horizontal metrics" );
    ( "a glyph offset format 2",
      `Malformed,
      set_u16 regular_ttf (head_pos regular_ttf + 50) 2,
      "loca format 2" );
    ( "no Unicode character map",
      (* Both subtables become Macintosh ones. *)
      `Unsupported,
      (let cmap = snd (table regular_ttf "cmap") in
       set_u16 (set_u16 regular_ttf (cmap + 4) 1) (cmap + 12) 1),
      "no Unicode character map" );
  ]

let of_file_reads () =
  let path = Filename.concat (temp_dir ()) "inter.ttf" in
  Out_channel.with_open_bin path (fun oc ->
      Out_channel.output_string oc regular_ttf);
  equal font_t Font.regular (require_ok ~pp:Font.pp_error (Font.of_file path));
  is_error `Io ~sub:"missing.ttf"
    (Font.of_file (Filename.concat (temp_dir ()) "missing.ttf"))

let loading =
  group "loading"
    [
      cases
        ~name:(fun (n, _, _, _) -> n)
        "of_string rejects" rejected
        (fun (_, kind, data, sub) -> is_error kind ~sub (Font.of_string data));
      test "of_file decodes a file and reports what it cannot read"
        of_file_reads;
      test "pp_error formats the kind and the message" (fun () ->
          equal string "malformed font: x"
            (Format.asprintf "%a" Font.pp_error (Font.Malformed "x"));
          equal string "unsupported font: y"
            (Format.asprintf "%a" Font.pp_error (Font.Unsupported "y"));
          equal string "z" (Format.asprintf "%a" Font.pp_error (Font.Io "z")));
    ]

(* Identity and names, from the bundled files' tables *)

let identity =
  group "identity"
    [
      test "the bundled faces are the files they embed" (fun () ->
          equal string regular_ttf (Font.bytes Font.regular);
          equal string bold_ttf (Font.bytes Font.bold));
      test "equal compares bytes" (fun () ->
          let reloaded = decoded regular_ttf in
          is_false (reloaded == Font.regular);
          equal font_t Font.regular reloaded;
          not_equal font_t Font.regular Font.bold);
      test "compare is a total order compatible with equal" (fun () ->
          Law.order
            (Testable.with_compare Font.compare font_t)
            (Font.regular, Font.bold, decoded regular_ttf));
      cases
        ~name:(fun (n, _, _, _, _) -> n)
        "names and style of"
        [
          ("regular", regular (), "Inter-Regular", 400, `Normal);
          ("bold", bold (), "Inter-Bold", 700, `Normal);
        ]
        (fun (_, f, ps, weight, slant) ->
          equal string "Inter" (Font.family f);
          equal string ps (Font.postscript_name f);
          equal int weight (Font.weight f);
          is_true (Font.slant f = slant));
      test "a font without names names nothing" (fun () ->
          let f = decoded (rename regular_ttf "name" "namX") in
          equal string "" (Font.family f);
          equal string "" (Font.postscript_name f));
      test "fsSelection gives the slant" (fun () ->
          let sel = os2_pos regular_ttf + 62 in
          is_true (Font.slant (decoded (set_u16 regular_ttf sel 0x81)) = `Italic);
          is_true
            (Font.slant (decoded (set_u16 regular_ttf sel 0x280)) = `Oblique));
      test "without OS/2, weight is 400 and head gives the slant" (fun () ->
          let data = rename bold_ttf "OS/2" "OS/X" in
          let f = decoded data in
          equal int 400 (Font.weight f);
          is_true (Font.slant f = `Normal);
          let italic = set_u16 data (head_pos data + 44) 0x3 in
          is_true (Font.slant (decoded italic) = `Italic));
      test "the weight class is clamped to [1, 1000]" (fun () ->
          let w = os2_pos regular_ttf + 4 in
          equal int 1000 (Font.weight (decoded (set_u16 regular_ttf w 5000)));
          equal int 1 (Font.weight (decoded (set_u16 regular_ttf w 0))));
      test "pp formats family, weight and slant" (fun () ->
          equal string "(font \"Inter\" 700 normal)"
            (Format.asprintf "%a" Font.pp (bold ())));
    ]

(* Metrics: Inter has 2048 units per em; values read with fontTools. *)

let pp_box ppf b =
  Format.fprintf ppf "[%.17g, %.17g; %.17g, %.17g]" (Box2.minx b) (Box2.miny b)
    (Box2.maxx b) (Box2.maxy b)

let box_t = Testable.make ~pp:pp_box ~equal:Box2.equal

let metric_cases =
  [
    ("ascent", Font.ascent, em 1984, em 1984);
    ("descent", Font.descent, em 494, em 494);
    ("line_gap", Font.line_gap, 0., 0.);
    ("cap_height", Font.cap_height, em 1490, em 1490);
    ("x_height", Font.x_height, em 1118, em 1118);
    ("italic_angle", Font.italic_angle, 0., 0.);
  ]

let typo_metrics () =
  let ascender = os2_pos regular_ttf + 68 in
  let data = set_u16 regular_ttf ascender 2000 in
  equal ~msg:"USE_TYPO_METRICS set" float_exact (em 2000)
    (Font.ascent (decoded data));
  let data = set_u16 data (os2_pos data + 62) 0x40 in
  equal ~msg:"USE_TYPO_METRICS clear" float_exact (em 1984)
    (Font.ascent (decoded data))

let heights_from_ink () =
  let f = decoded (rename regular_ttf "OS/2" "OS/X") in
  equal ~msg:"the top of H" float_exact (em 1490) (Font.cap_height f);
  equal ~msg:"the top of x" float_exact (em 1118) (Font.x_height f)

let italic_angle () =
  let post = snd (table regular_ttf "post") in
  let f = decoded (set_u32 regular_ttf (post + 4) (-12 * 65536)) in
  equal (float 1e-15) (-12. *. Float.pi /. 180.) (Font.italic_angle f)

let metrics =
  group "metrics"
    [
      cases
        ~name:(fun (n, _, _, _) -> n)
        "metric" metric_cases
        (fun (_, m, r, b) ->
          equal ~msg:"regular" float_exact r (m (regular ()));
          equal ~msg:"bold" float_exact b (m (bold ())));
      test "bounds is the head table's box" (fun () ->
          equal box_t
            (Box2.of_pts
               (P2.v (em (-1513)) (em (-2269)))
               (P2.v (em 5290) (em 660)))
            (Font.bounds (regular ()));
          equal box_t
            (Box2.of_pts
               (P2.v (em (-1613)) (em (-2279)))
               (P2.v (em 5290) (em 685)))
            (Font.bounds (bold ())));
      test "vertical metrics follow USE_TYPO_METRICS" typo_metrics;
      test "cap and x heights fall back to the ink of H and x" heights_from_ink;
      test "italic_angle is post's angle in radians" italic_angle;
    ]

(* Glyphs *)

let mapped =
  [ ('H', 46); ('x', 267); ('A', 2); ('V', 130); (' ', 560) ]
  |> List.map (fun (c, g) -> (Uchar.of_char c, g))
  |> List.append [ (Uchar.of_int 0x2192, 584); (Uchar.of_int 0xE9, 174) ]

let unmapped = [ 0x1F600; 0xFFFF; 0x10FFFF; 0x0400 ]

(* The byte of the regular face holding the glyph id of 'A', in segment 3 of its
   character map, which reads its glyph ids from the array. *)
let glyph_of_a data =
  let cmap = snd (table data "cmap") in
  let sub = cmap + get_u32 data (cmap + 16) in
  let seg2 = get_u16 data (sub + 6) in
  let ro = sub + 16 + (3 * seg2) + 6 in
  ro + get_u16 data ro + (2 * (65 - get_u16 data (sub + 16 + seg2 + 6)))

let missing_glyph () =
  let pos = glyph_of_a regular_ttf in
  equal ~msg:"the patched entry is A's" int 2 (get_u16 regular_ttf pos);
  equal int 0
    (Font.glyph (decoded (set_u16 regular_ttf pos 900)) (Uchar.of_char 'A'))

(* (first, second, regular, bold) in font units, as HarfBuzz applies them. *)
let kerned =
  [
    ("AV", -140, -162);
    ("To", -160, -160);
    ("LT", -197, -186);
    ("Yo", -157, -224);
    ("AT", -174, -182);
    ("HH", 0, 0);
    ("Av", -139, -139);
  ]

let kern_of f s =
  Font.kerning f
    (Font.glyph f (Uchar.of_char s.[0]))
    (Font.glyph f (Uchar.of_char s.[1]))

let legacy_kerning () =
  (* A = 2, V = 130, T = 115, o = 222 *)
  let table =
    kern_table
      [
        (0x3, [ (2, 130, -999) ]);
        (0x5, [ (2, 130, -999) ]);
        (0x1, [ (2, 100, -30); (2, 130, -100); (115, 222, -50) ]);
        (0x1, [ (2, 130, -20) ]);
      ]
  in
  let f = decoded (replace regular_ttf "GPOS" "kern" table) in
  equal ~msg:"summed over horizontal subtables" float_exact (em (-120))
    (kern_of f "AV");
  equal float_exact (em (-50)) (kern_of f "To");
  equal float_exact 0. (kern_of f "VA");
  equal ~msg:"between two pairs" float_exact 0. (Font.kerning f 2 131);
  equal ~msg:"a pair of the same first glyph" float_exact (em (-30))
    (Font.kerning f 2 100);
  equal ~msg:"after every pair" float_exact 0. (Font.kerning f 222 222)

let malformed_kern =
  [
    ( "pairs past the table",
      (let t = kern_table [ (0x1, [ (2, 130, -100) ]) ] in
       String.sub t 0 (String.length t - 2)),
      "the 'kern' table ends before byte" );
    ( "pairs out of order by their second glyph",
      kern_table [ (0x1, [ (2, 130, -1); (2, 100, -1) ]) ],
      "'kern': pairs not in increasing order" );
    ( "a repeated pair",
      kern_table [ (0x1, [ (2, 130, -1); (2, 130, -1) ]) ],
      "'kern': pairs not in increasing order" );
  ]

let glyph_range =
  [
    ("advance", "Font.advance", fun f g -> ignore (Font.advance f g));
    ("kerning", "Font.kerning", fun f g -> ignore (Font.kerning f g 0));
    ( "kerning's second glyph",
      "Font.kerning",
      fun f g -> ignore (Font.kerning f 0 g) );
    ("outline", "Font.outline", fun f g -> ignore (Font.outline f g));
    ("ink", "Font.ink", fun f g -> ignore (Font.ink f g));
  ]

(* Every mapping of the character map and every kerned pair of printable ASCII,
   against listings made with fontTools and HarfBuzz. *)

let cmap_listing f =
  let b = Buffer.create 8192 in
  for c = 0 to 0x10FFFF do
    if Uchar.is_valid c then
      let g = Font.glyph f (Uchar.of_int c) in
      if g <> 0 then Printf.bprintf b "U+%04X %d\n" c g
  done;
  Buffer.contents b

let kerning_listing f =
  let b = Buffer.create 8192 in
  for c = 0x21 to 0x7E do
    for c' = 0x21 to 0x7E do
      let k = kern_of f (Printf.sprintf "%c%c" (Char.chr c) (Char.chr c')) in
      let units = Float.to_int (Float.round (k *. 2048.)) in
      if units <> 0 then
        Printf.bprintf b "%c%c %d\n" (Char.chr c) (Char.chr c') units
    done
  done;
  Buffer.contents b

let glyphs =
  group "glyphs"
    [
      test "glyph_count is maxp's" (fun () ->
          equal int 720 (Font.glyph_count (regular ()));
          equal int 720 (Font.glyph_count (bold ())));
      cases
        ~name:(fun (u, _) -> Printf.sprintf "U+%04X" (Uchar.to_int u))
        "glyph maps" mapped
        (fun (u, g) -> equal int g (Font.glyph (regular ()) u));
      cases ~name:(Printf.sprintf "U+%04X") "glyph is 0 for the unmapped"
        unmapped (fun u ->
          equal int 0 (Font.glyph (regular ()) (Uchar.of_int u)));
      test "glyph is 0 for a glyph the font does not have" missing_glyph;
      test "the character map holds fontTools' mappings" (fun () ->
          expect_file
            (cmap_listing (regular ()))
            "packages/hugin/next/test/font/cmap.expected");
      test "kerning of printable ASCII is HarfBuzz's" (fun () ->
          expect_file
            (kerning_listing (regular ()))
            "packages/hugin/next/test/font/kerning.expected");
      test "advance is hmtx's" (fun () ->
          equal float_exact (em 1522) (Font.advance (regular ()) 46);
          equal float_exact (em 1530) (Font.advance (bold ()) 46);
          equal float_exact (em 576) (Font.advance (regular ()) 560));
      cases
        ~name:(fun (s, _, _) -> s)
        "kerning of" kerned
        (fun (s, r, b) ->
          equal ~msg:"regular" float_exact (em r) (kern_of (regular ()) s);
          equal ~msg:"bold" float_exact (em b) (kern_of (bold ()) s));
      test "without GPOS, kerning reads the horizontal pairs of kern"
        legacy_kerning;
      cases
        ~name:(fun (n, _, _) -> n)
        "kern is malformed with" malformed_kern
        (fun (_, table, sub) ->
          is_error `Malformed ~sub
            (Font.of_string (replace regular_ttf "GPOS" "kern" table)));
      test "a kern table of another version has no kerning" (fun () ->
          let table = kern_table [ (0x1, [ (2, 130, -100) ]) ] in
          let f =
            decoded
              (replace regular_ttf "GPOS" "kern"
                 ("\000\001" ^ String.sub table 2 (String.length table - 2)))
          in
          equal float_exact 0. (kern_of f "AV"));
      test "a font without GPOS or kern has no kerning" (fun () ->
          equal float_exact 0.
            (kern_of (decoded (rename regular_ttf "GPOS" "GPOX")) "AV"));
      cases
        ~name:(fun (n, _, _) -> n)
        "raises on a glyph out of range in" glyph_range
        (fun (_, fn, f) ->
          let msg g = Printf.sprintf "%s: glyph %d not in [0, 719]" fn g in
          raises_match
            (Exn.invalid_arg ~substring:(msg (-1)))
            (fun () -> f (regular ()) (-1));
          raises_match
            (Exn.invalid_arg ~substring:(msg 720))
            (fun () -> f (regular ()) 720));
    ]

let tiny ?tables ?cmap ?upem glyphs =
  decoded (tiny_file ?tables ?cmap ?upem glyphs)

let unit v = Float.of_int v /. 1000.
let pt x y = P2.v (unit x) (0. -. unit y)

let path_near =
  let points p =
    Path.fold
      ~move:(fun acc x y -> y :: x :: acc)
      ~line:(fun acc x y -> y :: x :: acc)
      ~cubic:(fun acc a b c d e f -> f :: e :: d :: c :: b :: a :: acc)
      ~close:(fun acc -> Float.infinity :: acc)
      [] p
  in
  Testable.make ~pp:Path.pp ~equal:(fun p q ->
      let a = points p and b = points q in
      List.length a = List.length b
      && List.for_all2 (fun a b -> a = b || Float.abs (a -. b) <= 1e-12) a b)

let all_off_curve () =
  let f =
    tiny [ ""; simple [ [ off 0 50; off 50 100; off 100 50; off 50 0 ] ] ]
  in
  let expected =
    Path.empty
    |> Path.move_to (pt 25 75)
    |> Path.quad_to (pt 50 100) (pt 75 75)
    |> Path.quad_to (pt 100 50) (pt 75 25)
    |> Path.quad_to (pt 50 0) (pt 25 25)
    |> Path.quad_to (pt 0 50) (pt 25 75)
    |> Path.close
  in
  equal path_near expected (Font.outline f 1)

let starts_on_curve () =
  let f = tiny [ ""; simple [ [ off 0 0; on 100 0; on 100 100 ] ] ] in
  let expected =
    Path.empty
    |> Path.move_to (pt 100 0)
    |> Path.line_to (pt 100 100)
    |> Path.quad_to (pt 0 0) (pt 100 0)
    |> Path.close
  in
  equal path_near expected (Font.outline f 1)

let component_cases =
  [
    ( "an offset in words",
      (words lor xy, be16 100 ^ be16 (-50), []),
      Affine.translate 100. (-50.) );
    ( "an offset in signed bytes",
      (xy, "\xFB\x03", []),
      Affine.translate (-5.) 3. );
    ( "an offset of the extreme bytes",
      (xy, "\x80\x7F", []),
      Affine.translate (-128.) 127. );
    ( "a scaled offset through a two by two",
      ( words lor xy lor two_by_two lor scaled_offset,
        be16 100 ^ be16 50,
        [ half; half; -half; half ] ),
      { Affine.xx = 0.5; yx = 0.5; xy = -0.5; yy = 0.5; x0 = 25.; y0 = 75. } );
    ( "a scale",
      (words lor xy lor one_scale, be16 100 ^ be16 0, [ half ]),
      Affine.(translate 100. 0. * scale 0.5 0.5) );
    ( "x and y scales",
      (words lor xy lor xy_scales, be16 0 ^ be16 0, [ half; one ]),
      Affine.scale 0.5 1. );
    ( "a two by two",
      (words lor xy lor two_by_two, be16 0 ^ be16 0, [ 0; one; -one; 0 ]),
      { Affine.xx = 0.; yx = 1.; xy = -1.; yy = 0.; x0 = 0.; y0 = 0. } );
    ( "a scaled offset",
      (words lor xy lor one_scale lor scaled_offset, be16 100 ^ be16 0, [ half ]),
      Affine.(scale 0.5 0.5 * translate 100. 0.) );
  ]

(* The square of glyph 1 placed by [placement] in font units, then in em. *)
let component_placement (_, (flags, args, transform), placement) =
  let f =
    tiny [ ""; simple [ square ]; composite [ (flags, 1, args, transform) ] ]
  in
  let to_em = Affine.scale (1. /. 1000.) (-1. /. 1000.) in
  let square = Path.polygon [| 0.; 100.; 100.; 0. |] [| 0.; 0.; 100.; 100. |] in
  equal path_near
    (Path.transform Affine.(to_em * placement) square)
    (Font.outline f 2)

let family_of records =
  Font.family (tiny ~tables:[ ("name", name_table records) ] [ "" ])

let names () =
  let mac = (1, 0, 0, 1, "Caf\x8E") in
  let fr = (3, 1, 0x40C, 1, utf16 "Famille") in
  let us = (3, 1, 0x409, 1, utf16 "Family") in
  equal ~msg:"Macintosh Roman" string "Caf\u{E9}" (family_of [ mac ]);
  equal ~msg:"Windows over Macintosh" string "Famille" (family_of [ mac; fr ]);
  equal ~msg:"US English first" string "Family" (family_of [ mac; fr; us ]);
  equal ~msg:"typographic family" string "Typo"
    (family_of [ us; (3, 10, 0x409, 16, utf16 "Typo") ])

let fallbacks () =
  let f = tiny [ ""; simple [ square ] ] in
  equal ~msg:"ascent from hhea" float_exact 0.8 (Font.ascent f);
  equal ~msg:"descent from hhea" float_exact 0.2 (Font.descent f);
  equal ~msg:"line gap from hhea" float_exact 0.1 (Font.line_gap f);
  equal ~msg:"weight" int 400 (Font.weight f);
  equal ~msg:"cap height is the ascent without H" float_exact 0.8
    (Font.cap_height f);
  equal ~msg:"x height is 0.5 without x" float_exact 0.5 (Font.x_height f);
  equal ~msg:"italic angle" float_exact 0. (Font.italic_angle f);
  equal ~msg:"no kerning" float_exact 0. (Font.kerning f 1 1);
  equal ~msg:"the last advance repeats" float_exact 0.5 (Font.advance f 1);
  let f = tiny ~cmap:(cmap_12 [ (0x48, 0x48, 1) ]) [ ""; simple [ square ] ] in
  equal ~msg:"cap height is the top of H" float_exact 0.1 (Font.cap_height f)

let format_12_lookup () =
  let f =
    tiny
      ~cmap:(cmap_12 [ (0x41, 0x41, 1); (0x1F600, 0x1F601, 2) ])
      [ ""; simple [ square ]; "" ]
  in
  equal int 1 (Font.glyph f (Uchar.of_char 'A'));
  equal int 2 (Font.glyph f (Uchar.of_int 0x1F600));
  equal ~msg:"beyond the glyphs" int 0 (Font.glyph f (Uchar.of_int 0x1F601));
  equal int 0 (Font.glyph f (Uchar.of_char 'B'))

let kerning_font gpos =
  tiny ~tables:[ ("GPOS", gpos) ] (List.init 8 (fun _ -> ""))

let units f g g' = Float.to_int (Float.round (Font.kerning f g g' *. 1000.))

let kern_cases =
  let x_advance = 0x4 and placements_then_advance = 0x7 and x_placement = 0x1 in
  let simple = [ ("kern", [ 0 ]) ] in
  [
    ( "format 1 with glyph coverage",
      gpos_table simple
        [
          lookup 2
            [
              pairs_1 ~format:x_advance
                [ (1, [ (2, [ -10 ]); (4, [ -11 ]) ]); (3, [ (4, [ -20 ]) ]) ];
            ];
        ],
      [ (1, 2, -10); (1, 4, -11); (3, 4, -20); (1, 3, 0); (2, 1, 0); (0, 2, 0) ]
    );
    ( "format 1 with range coverage",
      gpos_table simple
        [
          lookup 2
            [
              pairs_1
                ~coverage:(coverage_2 [ (5, 6, 0) ])
                ~format:x_advance
                [ (5, [ (1, [ -5 ]) ]); (6, [ (1, [ -6 ]) ]) ];
            ];
        ],
      [ (5, 1, -5); (6, 1, -6); (4, 1, 0); (7, 1, 0) ] );
    ( "placements before the advance",
      gpos_table simple
        [
          lookup 2
            [
              pairs_1 ~format:placements_then_advance
                [ (1, [ (2, [ 5; 6; -30 ]) ]) ];
            ];
        ],
      [ (1, 2, -30) ] );
    ( "the first covering subtable of a lookup",
      gpos_table simple
        [
          lookup 2
            [
              pairs_1 ~format:x_placement [ (1, [ (2, [ 5 ]) ]) ];
              pairs_1 ~format:x_advance [ (1, [ (2, [ -50 ]); (3, [ -60 ]) ]) ];
            ];
        ],
      [ (1, 2, 0); (1, 3, -60) ] );
    ( "format 2 with class definitions",
      gpos_table simple
        [
          lookup 2
            [
              pairs_2
                ~coverage:(coverage_1 [ 1; 2; 5 ])
                ~classes1:(classes_1 1 [ 1; 1 ])
                ~classes2:(classes_2 [ (3, 4, 1); (6, 6, 2) ])
                ~format:x_advance
                [ [ [ 0 ]; [ -1 ]; [ -2 ] ]; [ [ -3 ]; [ -40 ]; [ -41 ] ] ];
            ];
        ],
      [
        (1, 3, -40); (2, 4, -40); (1, 6, -41); (1, 5, -3); (5, 3, -1); (3, 3, 0);
      ] );
    ( "every kern lookup once, other features ignored",
      gpos_table
        [ ("kern", [ 0; 1; 0 ]); ("kern", [ 1 ]); ("liga", [ 2 ]) ]
        [
          lookup 2 [ pairs_1 ~format:x_advance [ (1, [ (2, [ -1 ]) ]) ] ];
          lookup 2 [ pairs_1 ~format:x_advance [ (1, [ (2, [ -10 ]) ]) ] ];
          lookup 2 [ pairs_1 ~format:x_advance [ (1, [ (2, [ -100 ]) ]) ] ];
        ],
      [ (1, 2, -11) ] );
    ( "extension lookups of pair adjustments only",
      gpos_table
        [ ("kern", [ 0; 1; 2 ]) ]
        [
          lookup 9
            [ extension 2 (pairs_1 ~format:x_advance [ (1, [ (2, [ -7 ]) ]) ]) ];
          lookup 9
            [
              extension 4 (pairs_1 ~format:x_advance [ (1, [ (2, [ -100 ]) ]) ]);
            ];
          lookup 1 [ pairs_1 ~format:x_advance [ (1, [ (2, [ -100 ]) ]) ] ];
        ],
      [ (1, 2, -7) ] );
    ( "a first glyph after every covered glyph",
      gpos_table simple
        [
          lookup 2
            [
              pairs_1 ~format:x_advance
                [
                  (1, [ (2, [ -1 ]); (3, [ -1 ]); (4, [ -1 ]); (5, [ -1 ]) ]);
                  (3, [ (4, [ -20 ]) ]);
                ];
            ];
        ],
      [ (4, 2, 0); (3, 4, -20) ] );
    ( "a second glyph after every pair",
      gpos_table simple
        [
          lookup 2
            [
              pairs_1 ~format:x_advance
                [
                  (1, [ (2, [ -10 ]) ]);
                  (3, [ (4, [ -20 ]); (5, [ -21 ]); (6, [ -22 ]) ]);
                ];
            ];
        ],
      [ (1, 3, 0); (1, 2, -10); (3, 6, -22) ] );
    ( "second value records",
      gpos_table simple
        [
          lookup 2
            [
              pairs_1 ~format:x_advance ~format2:x_advance
                [ (1, [ (2, [ -10; -99 ]); (4, [ -11; -98 ]) ]) ];
            ];
        ],
      [ (1, 2, -10); (1, 4, -11) ] );
    ( "format 2 with second value records",
      gpos_table simple
        [
          lookup 2
            [
              pairs_2 ~format2:x_advance
                ~coverage:(coverage_1 [ 1; 2; 3; 5 ])
                ~classes1:(classes_1 1 [ 1; 1 ])
                ~classes2:(classes_2 [ (3, 4, 1) ])
                ~format:x_advance
                [ [ [ 0; -9 ]; [ -1; -9 ] ]; [ [ -3; -9 ]; [ -40; -9 ] ] ];
            ];
        ],
      [ (1, 3, -40); (3, 3, -1); (1, 7, -3); (5, 4, -1) ] );
    ( "a kern feature after another",
      gpos_table
        [ ("liga", [ 1 ]); ("kern", [ 0 ]) ]
        [
          lookup 2 [ pairs_1 ~format:x_advance [ (1, [ (2, [ -5 ]) ]) ] ];
          lookup 2 [ pairs_1 ~format:x_advance [ (1, [ (2, [ -100 ]) ]) ] ];
        ],
      [ (1, 2, -5) ] );
  ]

(* The kerning of glyphs 1 and 2 in a font whose GPOS table adjusts them by
   [-10] in a lookup that [features] may refer to, and whose kern table adjusts
   them by [-30]. *)
let gpos_beside_kern features =
  let gpos =
    gpos_table features
      [ lookup 2 [ pairs_1 ~format:0x4 [ (1, [ (2, [ -10 ]) ]) ] ] ]
  in
  let kern = kern_table [ (0x1, [ (1, 2, -30) ]) ] in
  units
    (tiny
       ~tables:[ ("GPOS", gpos); ("kern", kern) ]
       (List.init 8 (fun _ -> "")))
    1 2

let kern_fallback =
  [
    ("only a liga feature uses kern", [ ("liga", [ 0 ]) ], -30);
    ("no feature uses kern", [], -30);
    ("a kern feature uses GPOS", [ ("kern", [ 0 ]) ], -10);
    ("a kern feature without lookups uses GPOS", [ ("kern", []) ], 0);
  ]

let malformed_gpos =
  let x_advance = 0x4 in
  let simple = [ ("kern", [ 0 ]) ] in
  let one st = gpos_table simple [ lookup 2 [ st ] ] in
  [
    ( "glyph coverage out of order",
      one
        (pairs_1
           ~coverage:(coverage_1 [ 3; 1 ])
           ~format:x_advance
           [ (3, []); (1, []) ]),
      "coverage not in increasing order" );
    ( "range coverage out of order",
      one
        (pairs_1
           ~coverage:(coverage_2 [ (4, 5, 0); (5, 6, 2) ])
           ~format:x_advance
           [ (4, []); (5, []); (6, []); (7, []) ]),
      "coverage ranges not in increasing order" );
    ( "coverage of format 3",
      one (pairs_1 ~coverage:(be16 3 ^ be16 0) ~format:x_advance [ (1, []) ]),
      "coverage format 3" );
    ( "coverage beyond the pair sets",
      one
        (pairs_1 ~coverage:(coverage_1 [ 1; 2 ]) ~format:x_advance [ (1, []) ]),
      "coverage beyond the pair sets" );
    ( "pairs out of order",
      one (pairs_1 ~format:x_advance [ (1, [ (4, [ -1 ]); (2, [ -1 ]) ]) ]),
      "pairs not in increasing order" );
    ( "a class beyond the class count",
      one
        (pairs_2 ~coverage:(coverage_1 [ 1 ]) ~classes1:(classes_1 1 [ 2 ])
           ~classes2:(classes_1 1 [ 0 ]) ~format:x_advance
           [ [ [ 0 ] ]; [ [ 0 ] ] ]),
      "class beyond the class count" );
    ( "a class range beyond the class count",
      one
        (pairs_2 ~coverage:(coverage_1 [ 1 ]) ~classes1:(classes_1 1 [ 0 ])
           ~classes2:(classes_2 [ (1, 2, 1) ])
           ~format:x_advance [ [ [ 0 ] ] ]),
      "class beyond the class count" );
    ( "class ranges out of order",
      one
        (pairs_2 ~coverage:(coverage_1 [ 1 ])
           ~classes1:(classes_2 [ (3, 4, 0); (4, 5, 0) ])
           ~classes2:(classes_1 1 [ 0 ]) ~format:x_advance [ [ [ 0 ] ] ]),
      "class ranges not in increasing order" );
    ( "an empty class matrix",
      one
        (pairs_2 ~coverage:(coverage_1 [ 1 ]) ~classes1:(classes_1 1 [ 0 ])
           ~classes2:(classes_1 1 [ 0 ]) ~format:x_advance []),
      "empty class matrix" );
    ( "a pair adjustment of format 3",
      one (be16 3 ^ be16 6 ^ be16 0 ^ coverage_1 [ 1 ]),
      "pair adjustment format 3" );
    ( "a missing lookup",
      gpos_table [ ("kern", [ 1 ]) ] [ lookup 2 [] ],
      "lookup 1 does not exist" );
    ( "a pair set past the table",
      (let st = pairs_1 ~format:x_advance [ (1, [ (2, [ -1 ]) ]) ] in
       one (String.sub st 0 (String.length st - 2))),
      "'GPOS' table ends before" );
    ( "glyph coverage with a repeated glyph",
      one
        (pairs_1
           ~coverage:(coverage_1 [ 1; 1 ])
           ~format:x_advance
           [ (1, []); (2, []) ]),
      "coverage not in increasing order" );
    ( "range coverage beyond the pair sets",
      one
        (pairs_1
           ~coverage:(coverage_2 [ (5, 6, 0) ])
           ~format:x_advance
           [ (5, []) ]),
      "coverage beyond the pair sets" );
    ( "pairs with a repeated second glyph",
      one (pairs_1 ~format:x_advance [ (1, [ (2, [ -1 ]); (2, [ -1 ]) ]) ]),
      "pairs not in increasing order" );
    ( "an empty class row",
      one
        (pairs_2 ~coverage:(coverage_1 [ 1 ]) ~classes1:(classes_1 1 [ 0 ])
           ~classes2:(classes_1 1 [ 0 ]) ~format:x_advance [ [] ]),
      "empty class matrix" );
    ( "pairs out of order with second value records",
      one
        (pairs_1 ~format:x_advance ~format2:x_advance
           [ (1, [ (4, [ -1; -1 ]); (2, [ -1; -1 ]) ]) ]),
      "pairs not in increasing order" );
    ( "a class matrix past the table",
      (let st =
         Bytes.of_string
           (pairs_2 ~format2:x_advance ~coverage:(coverage_1 [ 1 ])
              ~classes1:(classes_1 1 [ 0 ]) ~classes2:(classes_1 1 [ 0 ])
              ~format:x_advance
              [ [ [ 0; 0 ] ] ])
       in
       Bytes.set_uint16_be st 12 1000;
       one (Bytes.to_string st)),
      "the 'GPOS' table ends before byte" );
    ( "class definitions of format 3",
      one
        (pairs_2 ~coverage:(coverage_1 [ 1 ])
           ~classes1:(be16 3 ^ be16 0)
           ~classes2:(classes_1 1 [ 0 ]) ~format:x_advance [ [ [ 0 ] ] ]),
      "class definition format 3" );
  ]

let decoding =
  group "decoding"
    [
      test "a contour without on-curve points starts between its first two"
        all_off_curve;
      test "a contour starts at its first on-curve point" starts_on_curve;
      test "contours of one point have no ink" (fun () ->
          is_true
            (Path.is_empty
               (Font.outline (tiny [ ""; simple [ [ on 5 5 ]; [ on 7 7 ] ] ]) 1)));
      cases
        ~name:(fun (n, _, _) -> n)
        "a component placed by" component_cases component_placement;
      test "names come from US English, Windows, then Macintosh records" names;
      test "missing tables fall back as documented" fallbacks;
      test "format 12 maps characters beyond the BMP" format_12_lookup;
      cases
        ~name:(fun (n, _, _) -> n)
        "kerning reads" kern_cases
        (fun (_, gpos, pairs) ->
          let f = kerning_font gpos in
          List.iter
            (fun (g, g', k) ->
              equal ~msg:(Printf.sprintf "%d %d" g g') int k (units f g g'))
            pairs);
      cases
        ~name:(fun (n, _, _) -> n)
        "a GPOS table beside a kern table with" kern_fallback
        (fun (_, features, k) -> equal int k (gpos_beside_kern features));
      cases
        ~name:(fun (n, _, _) -> n)
        "GPOS is malformed with" malformed_gpos
        (fun (_, gpos, sub) ->
          is_error `Malformed ~sub
            (Font.of_string
               (tiny_file
                  ~tables:[ ("GPOS", gpos) ]
                  (List.init 8 (fun _ -> "")))));
      test "format 12 groups out of order are malformed" (fun () ->
          is_error `Malformed ~sub:"format 12"
            (Font.of_string
               (tiny_file
                  ~cmap:(cmap_12 [ (0x41, 0x45, 1); (0x43, 0x43, 1) ])
                  [ ""; simple [ square ] ])));
    ]

(* Character maps built subtable by subtable. *)

let glyph_of_a_in cmap =
  Font.glyph (tiny ~cmap (List.init 8 (fun _ -> ""))) (Uchar.of_char 'A')

let a_to g = format_4 [ (0x41, 0x41, g - 0x41, None) ]

let selection =
  [
    ("Windows over Unicode platform", [ (0, 3, a_to 2); (3, 1, a_to 3) ], 3);
    ("Unicode platform over Macintosh", [ (1, 0, a_to 1); (0, 3, a_to 2) ], 2);
    ( "format 12 over format 4",
      [ (3, 1, a_to 3); (3, 10, format_12 [ (0x41, 0x41, 4) ]) ],
      4 );
    ( "format 12 of any platform",
      [ (0, 4, format_12 [ (0x41, 0x41, 5) ]); (3, 1, a_to 3) ],
      5 );
    ( "Windows format 12 first",
      [
        (0, 4, format_12 [ (0x41, 0x41, 5) ]);
        (3, 10, format_12 [ (0x41, 0x41, 6) ]);
      ],
      6 );
    ("the first of equals", [ (3, 1, a_to 1); (3, 1, a_to 2) ], 1);
    ("Windows symbols ignored", [ (3, 0, a_to 1); (3, 1, a_to 2) ], 2);
    ("Windows encoding 10 for format 4", [ (3, 10, a_to 7) ], 7);
  ]

let format_4_lookup () =
  let cmap =
    cmap_table
      [
        ( 3,
          1,
          format_4
            [ (0x41, 0x43, 1 - 0x41, None); (0x61, 0x63, 2, Some [ 3; 0; 5 ]) ]
        );
      ]
  in
  let f = tiny ~cmap (List.init 8 (fun _ -> "")) in
  let g c = Font.glyph f (Uchar.of_char c) in
  equal (list int) [ 0; 1; 2; 3; 0 ] [ g '@'; g 'A'; g 'B'; g 'C'; g 'D' ];
  equal ~msg:"glyph ids then the delta, 0 staying 0" (list int)
    [ 0; 5; 0; 7; 0 ]
    [ g '`'; g 'a'; g 'b'; g 'c'; g 'd' ]

let unsupported_maps =
  [
    ("only Macintosh", [ (1, 0, a_to 1) ]);
    ("only Windows symbols", [ (3, 0, a_to 1) ]);
    ( "format 6 only",
      [ (3, 1, cat [ be16 6; be16 10; be16 0; be16 0x41; be16 0 ]) ] );
  ]

let format_12_overflow () =
  let sub =
    cat
      [ be16 12; be16 0; be32 40; be32 0; be32 5; be32 0x41; be32 0x41; be32 1 ]
  in
  let data = tiny_file ~cmap:(cmap_table [ (3, 10, sub) ]) [ ""; "" ] in
  (* The subtable starts at byte 12 of the table; five groups end at 76. *)
  is_error `Malformed
    ~sub:
      (Printf.sprintf "the 'cmap' table ends before byte %d"
         (snd (table data "cmap") + 12 + 76))
    (Font.of_string data)

(* Components *)

let two_components () =
  let half = 8192 in
  let f =
    tiny
      [
        "";
        simple [ square ];
        composite
          [
            (words lor xy lor one_scale, 1, be16 100 ^ be16 0, [ half ]);
            (words lor xy, 1, be16 0 ^ be16 200, []);
          ];
      ]
  in
  let to_em = Affine.scale (1. /. 1000.) (-1. /. 1000.) in
  let square = Path.polygon [| 0.; 100.; 100.; 0. |] [| 0.; 0.; 100.; 100. |] in
  let expected =
    Path.append
      (Path.transform Affine.(to_em * translate 0. 200.) square)
      (Path.transform Affine.(to_em * translate 100. 0. * scale 0.5 0.5) square)
  in
  equal path_near expected (Font.outline f 2)

let composite_points () =
  let line n = simple [ List.init n (fun i -> on i 0) ] in
  let comp gs =
    composite (List.map (fun g -> (words lor xy, g, be16 0 ^ be16 0, [])) gs)
  in
  let fits = tiny_file [ ""; line 65534; line 1; comp [ 1; 2 ] ] in
  is_ok ~pp:Font.pp_error (Font.of_string fits);
  let over = tiny_file [ ""; line 65534; line 1; comp [ 1; 2; 2 ] ] in
  is_error `Malformed ~sub:"glyph 3: more than 65535 points"
    (Font.of_string over)

(* Glyph 1 is 256 empty components, so 255 copies of it expand to 255 * 257 =
   65535 components; glyph [k + 1] of [nested] is 100 copies of glyph [k], so
   glyph 3 expands to 1,010,100. *)
let composite_components () =
  let comp gs = composite (List.map (fun g -> (xy, g, "\000\000", [])) gs) in
  let wide = comp (List.init 256 (fun _ -> 0)) in
  let fits = tiny_file [ ""; wide; comp (List.init 255 (fun _ -> 1)) ] in
  is_ok ~pp:Font.pp_error (Font.of_string fits);
  let over = tiny_file [ ""; wide; comp (0 :: List.init 255 (fun _ -> 1)) ] in
  is_error `Malformed ~sub:"glyph 2: more than 65535 components"
    (Font.of_string over);
  let nested = "" :: List.init 3 (fun k -> comp (List.init 100 (fun _ -> k))) in
  is_error `Malformed ~sub:"glyph 3: more than 65535 components"
    (Font.of_string (tiny_file nested))

let os2_heights () =
  let os2 = os2_pos regular_ttf in
  let cap = set_u16 regular_ttf (os2 + 88) 1400 in
  equal ~msg:"OS/2 version 4" float_exact (em 1400)
    (Font.cap_height (decoded cap));
  equal ~msg:"OS/2 version 2" float_exact (em 1400)
    (Font.cap_height (decoded (set_u16 cap os2 2)));
  equal ~msg:"OS/2 version 1" float_exact (em 1490)
    (Font.cap_height (decoded (set_u16 cap os2 1)));
  equal ~msg:"zero in OS/2" float_exact (em 1490)
    (Font.cap_height (decoded (set_u16 regular_ttf (os2 + 88) 0)))

let ignored_names () =
  equal ~msg:"Macintosh Japanese" string "" (family_of [ (1, 1, 0, 1, "X") ]);
  equal ~msg:"Unicode platform" string ""
    (family_of [ (0, 3, 0, 1, utf16 "X") ]);
  equal ~msg:"Macintosh Roman 0x80" string "\u{C4}"
    (family_of [ (1, 0, 0, 1, "\x80") ])

let units_per_em () =
  equal float_exact (500. /. 16.) (Font.advance (tiny ~upem:16 [ ""; "" ]) 1);
  equal float_exact (500. /. 16384.)
    (Font.advance (tiny ~upem:16384 [ ""; "" ]) 1)

let square_glyph = simple [ square ]

(* [square_glyph] cut to [n] bytes, read past its end at byte [at] of its
   data. *)
let truncated n at () =
  let data = tiny_file [ ""; String.sub square_glyph 0 n ] in
  is_error `Malformed
    ~sub:
      (Printf.sprintf "the data of glyph 1 ends before byte %d"
         (snd (table data "glyf") + at))
    (Font.of_string data)

(* A glyph of one contour of four on-curve points, flags in one repeat. *)
let repeated repeat =
  cat
    [
      be16 1;
      be16 0;
      be16 0;
      be16 0;
      be16 0;
      be16 3;
      be16 0;
      "\x09";
      String.make 1 (Char.chr repeat);
      cat (List.map be16 [ 0; 100; 0; -100 ]);
      cat (List.map be16 [ 0; 0; 100; 0 ]);
    ]

let repeated_flags () =
  equal path_near
    (Path.transform
       (Affine.scale (1. /. 1000.) (-1. /. 1000.))
       (Path.polygon [| 0.; 100.; 100.; 0. |] [| 0.; 0.; 100.; 100. |]))
    (Font.outline (tiny [ ""; repeated 3 ]) 1);
  is_error `Malformed ~sub:"glyph 1: flags repeat past its points"
    (Font.of_string (tiny_file [ ""; repeated 4 ]))

let repeated_end () =
  let glyph =
    cat
      [
        be16 2;
        be16 0;
        be16 0;
        be16 0;
        be16 0;
        be16 1;
        be16 1;
        be16 0;
        "\x01\x01";
        be16 0;
        be16 1;
        be16 0;
        be16 1;
      ]
  in
  is_error `Malformed ~sub:"glyph 1: contour ends not in increasing order"
    (Font.of_string (tiny_file [ ""; glyph ]))

let instructions () =
  let f = tiny [ ""; simple ~instructions:"\x01\x02\x03" [ square ] ] in
  equal path_near
    (Font.outline (tiny [ ""; square_glyph ]) 1)
    (Font.outline f 1)

(* Glyph 1's square placed by [first], then by an offset of (0, 200). *)
let after_first (first, placement) =
  let f =
    tiny
      [
        "";
        square_glyph;
        composite [ first; (words lor xy, 1, be16 0 ^ be16 200, []) ];
      ]
  in
  let to_em = Affine.scale (1. /. 1000.) (-1. /. 1000.) in
  let square = Path.polygon [| 0.; 100.; 100.; 0. |] [| 0.; 0.; 100.; 100. |] in
  equal path_near
    (Path.append
       (Path.transform Affine.(to_em * translate 0. 200.) square)
       (Path.transform Affine.(to_em * placement) square))
    (Font.outline f 2)

let name_past () =
  let name =
    cat
      [
        be16 0;
        be16 1;
        be16 18;
        be16 3;
        be16 1;
        be16 0x409;
        be16 1;
        be16 10;
        be16 0;
        "ab";
      ]
  in
  let data = tiny_file ~tables:[ ("name", name) ] [ "" ] in
  is_error `Malformed
    ~sub:
      (Printf.sprintf "the 'name' table ends before byte %d"
         (snd (table data "name") + 28))
    (Font.of_string data)

let single_segment_cut () =
  let sub = format_4 [ (0x61, 0x61, 0, Some [ 1 ]) ] in
  let cut = String.sub sub 0 (String.length sub - 2) in
  is_error `Malformed ~sub:"the 'cmap' table ends before byte"
    (Font.of_string (tiny_file ~cmap:(cmap_table [ (3, 1, cut) ]) [ ""; "" ]))

(* The final U+FFFF segment of many fonts has a bogus glyph id offset. *)
let bogus_last_segment () =
  let sub = Bytes.of_string (format_4 [ (0x41, 0x41, 1 - 0x41, None) ]) in
  (* Two segments: the range offset of the second is at 16 + 6 * 2 + 2. *)
  Bytes.set_uint16_be sub 30 0x7FFF;
  let f = tiny ~cmap:(cmap_table [ (3, 1, Bytes.to_string sub) ]) [ ""; "" ] in
  equal int 1 (Font.glyph f (Uchar.of_char 'A'));
  equal int 0 (Font.glyph f (Uchar.of_int 0xFFFF))

let edges =
  group "edges"
    [
      cases
        ~name:(fun (n, _, _) -> n)
        "character map selection" selection
        (fun (_, subtables, g) ->
          equal int g (glyph_of_a_in (cmap_table subtables)));
      test "format 4 adds deltas to characters or to glyph ids" format_4_lookup;
      cases ~name:fst "a character map is unsupported with" unsupported_maps
        (fun (_, subtables) ->
          is_error `Unsupported ~sub:"no Unicode character map"
            (Font.of_string (tiny_file ~cmap:(cmap_table subtables) [ ""; "" ])));
      test "format 12 groups past the table are malformed" format_12_overflow;
      test "components after a scaled one are read in place" two_components;
      test "a composite has at most 65535 points" composite_points;
      test "a composite has at most 65535 components" composite_components;
      test "a contour of two points has ink" (fun () ->
          let f = tiny [ ""; simple [ [ on 0 0; on 100 0 ] ] ] in
          equal (option box_t)
            (Some (Box2.v 0. 0. 0.1 0.))
            (Path.bounds (Font.outline f 1)));
      test "a glyph of no contours with a header is empty" (fun () ->
          is_true (Path.is_empty (Font.outline (tiny [ ""; simple [] ]) 1)));
      test "cap height reads OS/2 version 2 and later" os2_heights;
      test "names of other platforms and encodings are ignored" ignored_names;
      test "units per em may be 16 or 16384" units_per_em;
      test "a glyph cut in its flags is malformed" (truncated 16 17);
      test "a glyph cut in its y coordinates is malformed" (truncated 33 34);
      test "flags repeat up to the last point" repeated_flags;
      test "repeated contour ends are malformed" repeated_end;
      test "instructions are skipped" instructions;
      cases ~name:fst "a component follows one with"
        [
          ( "x and y scales",
            ( (words lor xy lor xy_scales, 1, be16 0 ^ be16 0, [ half; one ]),
              Affine.scale 0.5 1. ) );
          ( "a two by two",
            ( ( words lor xy lor two_by_two,
                1,
                be16 0 ^ be16 0,
                [ one; 0; 0; one ] ),
              Affine.id ) );
        ]
        (fun (_, c) -> after_first c);
      test "a name past its table is malformed" name_past;
      test "a single-character segment's glyph ids are checked"
        single_segment_cut;
      test "the final segment's glyph id offset is not read" bogus_last_segment;
      test "an odd segment count is malformed" (fun () ->
          let sub = Bytes.of_string (format_4 [ (0x41, 0x41, 0, None) ]) in
          Bytes.set_uint16_be sub 6 3;
          is_error `Malformed ~sub:"odd segment count"
            (Font.of_string
               (tiny_file
                  ~cmap:(cmap_table [ (3, 1, Bytes.to_string sub) ])
                  [ ""; "" ])));
      test "format 12 groups sharing a character are malformed" (fun () ->
          is_error `Malformed ~sub:"format 12"
            (Font.of_string
               (tiny_file
                  ~cmap:(cmap_12 [ (0x41, 0x45, 1); (0x45, 0x46, 1) ])
                  [ ""; "" ])));
    ]

(* Outlines *)

let outline f g = Format.asprintf "%a" Path.pp (Font.outline f g)

let contains_box outer inner =
  let slack = 1e-12 in
  Box2.minx outer -. slack <= Box2.minx inner
  && Box2.miny outer -. slack <= Box2.miny inner
  && Box2.maxx inner <= Box2.maxx outer +. slack
  && Box2.maxy inner <= Box2.maxy outer +. slack

let all_glyphs_within f =
  for g = 0 to Font.glyph_count f - 1 do
    match Path.bounds (Font.outline f g) with
    | None -> ()
    | Some b ->
        satisfies
          ~msg:(Printf.sprintf "glyph %d" g)
          box_t ~claim:"inside the font's bounds"
          (contains_box (Font.bounds f))
          b
  done

(* [ink_is_bounds f] checks {!Font.ink} against the outline of every glyph of
   [f]. *)
let ink_is_bounds f =
  for g = 0 to Font.glyph_count f - 1 do
    equal
      ~msg:(Printf.sprintf "glyph %d" g)
      (option box_t)
      (Path.bounds (Font.outline f g))
      (Font.ink f g)
  done

let box_near =
  let near a b = Float.abs (a -. b) <= 1e-15 in
  Testable.make ~pp:pp_box ~equal:(fun a b ->
      near (Box2.minx a) (Box2.minx b)
      && near (Box2.miny a) (Box2.miny b)
      && near (Box2.maxx a) (Box2.maxx b)
      && near (Box2.maxy a) (Box2.maxy b))

(* Glyph 1 is a ring of four quadratics whose ink spans 12.5 to 87.5 on both
   axes and whose control points reach 0 and 100. Glyph 2 is glyph 1 turned an
   eighth of a turn about the origin and scaled by 1/√2: its ink spans -25 to 25
   in x and 25 to 75 in y, where the image of the box of glyph 1 spans -37.5 to
   37.5 and 12.5 to 87.5. Glyphs 0, 3 and 4 have no ink: no data, no contours,
   and a contour of one point. *)
let ink_of_curves_and_components () =
  let f =
    tiny
      [
        "";
        simple [ [ off 0 50; off 50 100; off 100 50; off 50 0 ] ];
        composite
          [
            ( words lor xy lor two_by_two,
              1,
              be16 0 ^ be16 0,
              [ half; half; -half; half ] );
          ];
        simple [];
        simple [ [ on 10 10 ] ];
      ]
  in
  ink_is_bounds f;
  let box x0 y0 x1 y1 = Some (Box2.of_pts (P2.v x0 y0) (P2.v x1 y1)) in
  equal ~msg:"glyph 1" (option box_near)
    (box 0.0125 (-0.0875) 0.0875 (-0.0125))
    (Font.ink f 1);
  equal ~msg:"glyph 2" (option box_near)
    (box (-0.025) (-0.075) 0.025 (-0.025))
    (Font.ink f 2);
  List.iter
    (fun g ->
      is_none ~msg:(Printf.sprintf "glyph %d" g) ~pp:pp_box (Font.ink f g))
    [ 0; 3; 4 ]

(* Ink boxes read with fontTools, in font units, y up. *)
let ink =
  [
    ("H", 46, (180, 0, 1342, 1490));
    ("o", 222, (104, -24, 1124, 1132));
    ("g", 184, (104, -442, 1098, 1132));
    ("eacute", 174, (104, -24, 1094, 1558));
    ("arrowright", 584, (256, 0, 1752, 1304));
  ]

let outlines =
  group "outlines"
    [
      test "H is its contour of lines" (fun () ->
          expect (outline (regular ()) 46)
          @@ __POS_OF__
               {|
            M 0.0878906 0 L 0.0878906 -0.727539 L 0.180664 -0.727539 L 0.180664 -0.413086
            L 0.5625 -0.413086 L 0.5625 -0.727539 L 0.655273 -0.727539 L 0.655273 0
            L 0.5625 0 L 0.5625 -0.331055 L 0.180664 -0.331055 L 0.180664 0 Z
            |});
      test "o is two contours of curves" (fun () ->
          expect (outline (regular ()) 222)
          @@ __POS_OF__
               {|
            M 0.299316 0.0117188 C 0.249837 0.0117188 0.206462 0 0.169189 -0.0234375
            C 0.131917 -0.046875 0.102865 -0.0797526 0.0820312 -0.12207
            C 0.0611979 -0.164388 0.0507812 -0.213542 0.0507812 -0.269531
            C 0.0507812 -0.326497 0.0611979 -0.376221 0.0820312 -0.418701
            C 0.102865 -0.461182 0.131917 -0.494141 0.169189 -0.517578
            C 0.206462 -0.541016 0.249837 -0.552734 0.299316 -0.552734
            C 0.349121 -0.552734 0.392741 -0.541016 0.430176 -0.517578
            C 0.467611 -0.494141 0.496745 -0.461182 0.517578 -0.418701
            C 0.538411 -0.376221 0.548828 -0.326497 0.548828 -0.269531
            C 0.548828 -0.213542 0.538411 -0.164388 0.517578 -0.12207
            C 0.496745 -0.0797526 0.467611 -0.046875 0.430176 -0.0234375
            C 0.392741 0 0.349121 0.0117188 0.299316 0.0117188 Z M 0.299316 -0.0668945
            C 0.3361 -0.0668945 0.366374 -0.0763346 0.390137 -0.0952148
            C 0.4139 -0.114095 0.431478 -0.138916 0.442871 -0.169678
            C 0.454264 -0.200439 0.459961 -0.233724 0.459961 -0.269531
            C 0.459961 -0.305664 0.454264 -0.339274 0.442871 -0.370361
            C 0.431478 -0.401449 0.4139 -0.426514 0.390137 -0.445557
            C 0.366374 -0.4646 0.3361 -0.474121 0.299316 -0.474121
            C 0.262858 -0.474121 0.23291 -0.4646 0.209473 -0.445557
            C 0.186035 -0.426514 0.16862 -0.40153 0.157227 -0.370605
            C 0.145833 -0.339681 0.140137 -0.30599 0.140137 -0.269531
            C 0.140137 -0.233724 0.145833 -0.200439 0.157227 -0.169678
            C 0.16862 -0.138916 0.186035 -0.114095 0.209473 -0.0952148
            C 0.23291 -0.0763346 0.262858 -0.0668945 0.299316 -0.0668945 Z
            |});
      test "é places its accent by its component offset" (fun () ->
          expect (outline (regular ()) 174)
          @@ __POS_OF__
               {|
            M 0.306641 0.0117188 C 0.253906 0.0117188 0.208415 0 0.170166 -0.0234375
            C 0.131917 -0.046875 0.102458 -0.0795898 0.0817871 -0.121582
            C 0.0611165 -0.163574 0.0507812 -0.212565 0.0507812 -0.268555
            C 0.0507812 -0.324544 0.0608724 -0.373861 0.0810547 -0.416504
            C 0.101237 -0.459147 0.129801 -0.492513 0.166748 -0.516602
            C 0.203695 -0.54069 0.246908 -0.552734 0.296387 -0.552734
            C 0.325358 -0.552734 0.353923 -0.547933 0.38208 -0.53833
            C 0.410238 -0.528727 0.435791 -0.513265 0.45874 -0.491943
            C 0.481689 -0.470622 0.5 -0.442546 0.513672 -0.407715
            C 0.527344 -0.372884 0.53418 -0.330241 0.53418 -0.279785 L 0.53418 -0.243164
            L 0.139648 -0.243164
            C 0.141927 -0.186198 0.158285 -0.142497 0.188721 -0.112061
            C 0.219157 -0.0816243 0.258626 -0.0664062 0.307129 -0.0664062
            C 0.339355 -0.0664062 0.367106 -0.0734863 0.390381 -0.0876465
            C 0.413656 -0.101807 0.430339 -0.122721 0.44043 -0.150391
            L 0.525391 -0.126953
            C 0.512695 -0.085612 0.487061 -0.0521647 0.448486 -0.0266113
            C 0.409912 -0.00105794 0.36263 0.0117188 0.306641 0.0117188 Z
            M 0.140137 -0.317383 L 0.444824 -0.317383
            C 0.440592 -0.363932 0.426025 -0.401774 0.401123 -0.430908
            C 0.376221 -0.460042 0.341309 -0.474609 0.296387 -0.474609
            C 0.265137 -0.474609 0.2382 -0.467285 0.215576 -0.452637
            C 0.192952 -0.437988 0.17513 -0.418783 0.162109 -0.39502
            C 0.149089 -0.371257 0.141764 -0.345378 0.140137 -0.317383 Z
            M 0.255859 -0.617188 L 0.324219 -0.760742 L 0.422363 -0.760742
            L 0.327637 -0.617188 Z
            |});
      test "a glyph without ink is empty" (fun () ->
          is_true (Path.is_empty (Font.outline (regular ()) 560));
          is_true (Path.is_empty (Font.outline (regular ()) 0));
          is_none ~pp:pp_box (Font.ink (regular ()) 560);
          is_none ~pp:pp_box (Font.ink (regular ()) 0));
      cases
        ~name:(fun (n, _, _) -> n)
        "ink and the outline's bounds are fontTools' box of" ink
        (fun (_, g, (x0, y0, x1, y1)) ->
          let box =
            Some
              (Box2.of_pts
                 (P2.v (em x0) (0. -. em y1))
                 (P2.v (em x1) (0. -. em y0)))
          in
          equal ~msg:"outline" (option box_t) box
            (Path.bounds (Font.outline (regular ()) g));
          equal ~msg:"ink" (option box_t) box (Font.ink (regular ()) g));
      test "ink is the outline's bounds for every glyph of regular" (fun () ->
          ink_is_bounds (regular ()));
      test "ink is the outline's bounds for every glyph of bold" (fun () ->
          ink_is_bounds (bold ()));
      test "ink bounds curves and transformed components tightly"
        ink_of_curves_and_components;
      test "every glyph of regular lies within its bounds" (fun () ->
          all_glyphs_within (regular ()));
      test "every glyph of bold lies within its bounds" (fun () ->
          all_glyphs_within (bold ()));
    ]

let () =
  exit
    (run "hugin.next.font"
       [ loading; identity; metrics; glyphs; decoding; edges; outlines ])
