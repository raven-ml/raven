(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_gg
open Hugin_font
open Build

let read path = In_channel.with_open_bin path In_channel.input_all
let regular_ttf = read "../lib/font/fonts/Inter-Regular.ttf"
let bold_ttf = read "../lib/font/fonts/Inter-Bold.ttf"
let em units = Float.of_int units /. 2048.
let font_t = Testable.make ~pp:Font.pp ~equal:Font.equal
let path_t = Testable.make ~pp:Path.pp ~equal:Path.equal

let pp_box ppf b =
  Format.fprintf ppf "[%.17g, %.17g; %.17g, %.17g]" (Box2.minx b) (Box2.miny b)
    (Box2.maxx b) (Box2.maxy b)

let box_t = Testable.make ~pp:pp_box ~equal:Box2.equal

let decoded data =
  match Font.of_string data with
  | Ok f -> f
  | Error e -> failf "%a" Font.pp_error e

(* [subset f gs] is [Font.subset f gs] checked as an OpenType file and
   decoded. *)
let subset f gs =
  let s = Font.subset f gs in
  check_sfnt s;
  decoded s

(* The bundled faces decoded inside the tests: [Font.regular] and [Font.bold]
   are decoded when the library initialises, out of the tests' sight. *)
let regular () = decoded regular_ttf
let bold () = decoded bold_ttf
let glyph f c = Font.glyph f (Uchar.of_int c)

let is_error kind ?(sub = "") result =
  let e = require_error ~pp:Font.pp result in
  match (kind, e) with
  | `Malformed, Font.Malformed msg
  | `Unsupported, Font.Unsupported msg
  | `Io, Font.Io msg ->
      contains ~sub msg
  | _ -> failf "unexpected error: %a" Font.pp_error e

(* Fonts built table by table, [upem] 1000 units per em. *)
let tiny ?tables ?cmap ?upem glyphs =
  decoded (tiny_file ?tables ?cmap ?upem glyphs)

let units f g g' = Float.to_int (Float.round (Font.kerning f g g' *. 1000.))
let pt x y = P2.v (Float.of_int x /. 1000.) (Float.of_int (-y) /. 1000.)

(* Glyph 1 of the tiny fonts, a square of 100 units, in font units and in em
   with y down. *)
let square_units =
  Path.polygon [| 0.; 100.; 100.; 0. |] [| 0.; 0.; 100.; 100. |]

let to_em = Affine.scale (1. /. 1000.) (-1. /. 1000.)
let square_path = Path.transform to_em square_units

(* Loading *)

let rejected =
  let glyf = snd (table regular_ttf "glyf")
  and loca = snd (table regular_ttf "loca") in
  let count = Font.glyph_count Font.regular in
  (* é is a composite of a glyph and an accent. *)
  let eacute =
    glyf + (2 * get_u16 regular_ttf (loca + (2 * glyph Font.regular 0xE9)))
  in
  let hmtx = table regular_ttf "hmtx" in
  let cmap subtables = tiny_file ~cmap:(cmap_table subtables) [ ""; "" ] in
  let a_to_1 = format_4 [ (0x41, 0x41, 1 - 0x41, None) ] in
  [
    ("an empty string", `Malformed, "", "the file ends before byte 4");
    ("text", `Malformed, "not a font at all, only text", "not an OpenType file");
    ("CFF outlines", `Unsupported, set_u32 regular_ttf 0 0x4F54544F, "CFF");
    ( "a collection",
      `Unsupported,
      set_u32 regular_ttf 0 0x74746366,
      "collection" );
    ("WOFF", `Unsupported, set_u32 regular_ttf 0 0x774F4646, "WOFF");
    ( "no 'glyf' outlines",
      `Unsupported,
      rename regular_ttf "glyf" "glyX",
      "glyf" );
    ( "no 'hmtx' table",
      `Malformed,
      rename regular_ttf "hmtx" "hmtX",
      "no 'hmtx'" );
    ( "a 'hmtx' table cut short, saying where",
      `Malformed,
      set_u32 regular_ttf (fst hmtx + 12) 100,
      Printf.sprintf "the 'hmtx' table ends before byte %d" (snd hmtx + 102) );
    ( "a table past the end of the file",
      `Malformed,
      set_u32 regular_ttf (fst (table regular_ttf "name") + 12) 1_000_000,
      "'name' table extends past" );
    ( "a wrong 'head' magic",
      `Malformed,
      set_u32 regular_ttf (head_pos regular_ttf + 12) 0,
      "magic" );
    ( "15 units per em",
      `Malformed,
      set_u16 regular_ttf (head_pos regular_ttf + 18) 15,
      "units per em" );
    ( "16385 units per em",
      `Malformed,
      set_u16 regular_ttf (head_pos regular_ttf + 18) 16385,
      "units per em" );
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
    ( "glyph offsets out of order",
      `Malformed,
      set_u16 regular_ttf (loca + 2) (get_u16 regular_ttf (loca + 4) + 1),
      "'loca'" );
    ( "a composite containing itself",
      `Malformed,
      set_u16 regular_ttf (eacute + 12) (glyph Font.regular 0xE9),
      "contains itself" );
    ( "a component past the last glyph",
      `Malformed,
      set_u16 regular_ttf (eacute + 12) count,
      Printf.sprintf "component %d does not exist" count );
    ( "a component placed by point matching",
      `Unsupported,
      set_u16 regular_ttf (eacute + 10)
        (get_u16 regular_ttf (eacute + 10) land lnot 2),
      "point matching" );
    ( "only Macintosh character maps",
      `Unsupported,
      cmap [ (1, 0, a_to_1) ],
      "no Unicode character map" );
    ( "only Windows symbol character maps",
      `Unsupported,
      cmap [ (3, 0, a_to_1) ],
      "no Unicode character map" );
    ( "only a format 6 character map",
      `Unsupported,
      cmap [ (3, 1, cat [ be16 6; be16 10; be16 0; be16 0x41; be16 0 ]) ],
      "no Unicode character map" );
  ]

let of_file_reads () =
  let path = Filename.concat (temp_dir ()) "inter.ttf" in
  Out_channel.with_open_bin path (fun oc ->
      Out_channel.output_string oc regular_ttf);
  equal font_t Font.regular (require_ok ~pp:Font.pp_error (Font.of_file path));
  is_error `Io ~sub:"missing.ttf"
    (Font.of_file (Filename.concat (temp_dir ()) "missing.ttf"))

let simple_line n = simple [ List.init n (fun i -> on i 0) ]
let composite_of gs = composite (List.map (fun g -> (xy, g, "\000\000", [])) gs)

let composite_points () =
  let fits =
    tiny_file [ ""; simple_line 65534; simple_line 1; composite_of [ 1; 2 ] ]
  in
  is_ok ~pp:Font.pp_error (Font.of_string fits);
  let over =
    tiny_file [ ""; simple_line 65534; simple_line 1; composite_of [ 1; 2; 2 ] ]
  in
  is_error `Malformed ~sub:"glyph 3: more than 65535 points"
    (Font.of_string over)

(* Glyph 1 is 256 empty components, so 255 copies of it expand to 255 * 257 =
   65535 components; glyph [k + 1] of [nested] is 100 copies of glyph [k], so
   glyph 3 expands to 1,010,100. *)
let composite_components () =
  let wide = composite_of (List.init 256 (fun _ -> 0)) in
  let fits =
    tiny_file [ ""; wide; composite_of (List.init 255 (fun _ -> 1)) ]
  in
  is_ok ~pp:Font.pp_error (Font.of_string fits);
  let over =
    tiny_file [ ""; wide; composite_of (0 :: List.init 255 (fun _ -> 1)) ]
  in
  is_error `Malformed ~sub:"glyph 2: more than 65535 components"
    (Font.of_string over);
  let nested =
    "" :: List.init 3 (fun k -> composite_of (List.init 100 (fun _ -> k)))
  in
  is_error `Malformed ~sub:"glyph 3: more than 65535 components"
    (Font.of_string (tiny_file nested))

let loading =
  group "loading"
    [
      cases
        ~name:(fun (n, _, _, _) -> n)
        "of_string rejects" rejected
        (fun (_, kind, data, sub) -> is_error kind ~sub (Font.of_string data));
      test "a composite expands to at most 65535 points" composite_points;
      test "a composite expands to at most 65535 components"
        composite_components;
      test "of_file decodes a file and reports what it cannot read"
        of_file_reads;
      test "pp_error formats the kind and the message" (fun () ->
          equal string "malformed font: x"
            (Format.asprintf "%a" Font.pp_error (Font.Malformed "x"));
          equal string "unsupported font: y"
            (Format.asprintf "%a" Font.pp_error (Font.Unsupported "y"));
          equal string "z" (Format.asprintf "%a" Font.pp_error (Font.Io "z")));
    ]

(* Robustness: [of_string] checks all it reads, so a font it returns answers
   every query. The law runs over fonts with bytes overwritten, from the regular
   face and from two small fonts in which every byte is structure: a GPOS table
   of each pair adjustment format, a kern table, character maps of formats 4 and
   12, composite and off-curve glyphs, names and glyph names. *)

let rich_glyphs =
  [
    "";
    simple [ square ];
    simple [ [ off 0 50; off 50 100; off 100 50; off 50 0 ] ];
    composite
      [
        ( words lor xy lor two_by_two,
          1,
          be16 10 ^ be16 20,
          [ half; half; -half; half ] );
        (xy lor one_scale lor scaled_offset, 2, "\001\002", [ half ]);
      ];
    "";
    "";
  ]

let rich_names =
  name_table [ (1, 0, 0, 1, "Caf\x8E"); (3, 1, 0x409, 16, utf16 "Typo") ]

(* A post table of format 2 naming each glyph by a standard name. *)
let rich_post =
  cat
    [
      be32 0x00020000;
      String.make 28 '\000';
      be16 (List.length rich_glyphs);
      cat (List.mapi (fun i _ -> be16 i) rich_glyphs);
    ]

let rich_gpos =
  gpos_table
    [ ("kern", [ 0; 1; 2 ]); ("liga", [ 2 ]) ]
    [
      lookup 2
        [
          pairs_1
            ~coverage:(coverage_2 [ (1, 1, 0); (3, 3, 1) ])
            ~format:0x7 ~format2:0x4
            [
              (1, [ (2, [ 5; 6; -30; 9 ]); (4, [ 1; 2; -40; 9 ]) ]);
              (3, [ (4, [ 0; 0; -20; 9 ]) ]);
            ];
          pairs_2
            ~coverage:(coverage_2 [ (1, 3, 0) ])
            ~classes1:(classes_1 1 [ 1; 1 ])
            ~classes2:(classes_2 [ (3, 4, 1); (5, 5, 2) ])
            ~format:0x4
            [ [ [ 0 ]; [ -1 ]; [ -2 ] ]; [ [ -3 ]; [ -4 ]; [ -5 ] ] ];
        ];
      lookup 9 [ extension 2 (pairs_1 ~format:0x4 [ (2, [ (1, [ -7 ]) ]) ]) ];
      lookup 2 [ pairs_1 ~format:0x4 [ (1, [ (1, [ -100 ]) ]) ] ];
    ]

let seeds =
  [
    ("regular", regular_ttf);
    ( "GPOS",
      tiny_file
        ~tables:
          [ ("GPOS", rich_gpos); ("name", rich_names); ("post", rich_post) ]
        ~cmap:
          (cmap_table
             [
               ( 3,
                 1,
                 format_4
                   [
                     (0x41, 0x43, 1 - 0x41, None);
                     (0x61, 0x63, 2, Some [ 3; 0; 2 ]);
                   ] );
             ])
        rich_glyphs );
    ( "kern",
      tiny_file
        ~tables:
          [
            ( "kern",
              kern_table
                [ (0x1, [ (1, 2, -30); (3, 4, -5) ]); (0x5, [ (1, 2, -9) ]) ] );
            ("name", rich_names);
          ]
        ~cmap:(cmap_12 [ (0x41, 0x43, 1); (0x1F600, 0x1F601, 4) ])
        rich_glyphs );
  ]

(* Some bytes of a seed overwritten, each in a table drawn first so that small
   tables are hit as often as large ones. *)
let gen_damaged =
  let open Gen in
  let* name, data = of_list seeds in
  let tables = Array.of_list (extents data) in
  let byte =
    frequency [ (3, int_range 0 255); (1, of_list [ 0; 0x7F; 0x80; 0xFF ]) ]
  in
  let at = triple (int_range 0 (Array.length tables - 1)) nat byte in
  let+ patches = list ~size:(int_range 1 6) at in
  let damaged =
    List.fold_left
      (fun data (t, i, v) ->
        let off, len = tables.(t) in
        if len = 0 then data
        else
          patch data (off + (i mod len)) (fun b pos -> Bytes.set_uint8 b pos v))
      data patches
  in
  (name, patches, damaged)

let gen_damaged =
  Gen.with_pp
    (fun ppf (name, patches, _) ->
      Format.fprintf ppf "%s with %a" name
        (Format.pp_print_list ~pp_sep:Format.pp_print_space
           (fun ppf (t, i, v) ->
             Format.fprintf ppf "table %d byte %d := %d" t i v))
        patches)
    gen_damaged

(* The characters [answers] looks up: Latin, punctuation and arrows, the end of
   the BMP and characters past it. *)
let probed =
  List.concat_map
    (fun (a, b) -> List.init (b - a + 1) (fun i -> a + i))
    [ (0, 0x100); (0x2000, 0x2200); (0xFFF0, 0xFFFF); (0x1F5F0, 0x1F610) ]
  |> List.filter Uchar.is_valid

(* Every query of [f], each raising nothing, and the laws that tie [ink] to
   [outline] and a subset to [f]. [at] names [f] in failures. *)
let answers ~at f =
  let n = Font.glyph_count f in
  let queries () =
    (* Every pair of a small font, and two partners of each glyph of a large
       one. *)
    let partners g =
      if n <= 16 then List.init n Fun.id else [ ((g * 37) + 11) mod n ]
    in
    for g = 0 to n - 1 do
      ignore (Font.advance f g, Font.outline f g, Font.ink f g);
      List.iter
        (fun g' -> ignore (Font.kerning f g g', Font.kerning f g' g))
        (partners g)
    done;
    List.iter (fun c -> ignore (glyph f c)) probed;
    ignore
      ( Font.family f,
        Font.postscript_name f,
        (Font.weight f, Font.slant f),
        (Font.ascent f, Font.descent f, Font.line_gap f),
        (Font.cap_height f, Font.x_height f, Font.italic_angle f),
        Font.bounds f )
  in
  (match queries () with
  | () -> ()
  | exception e -> failf "%s: a query raised %s" at (Printexc.to_string e));
  for g = 0 to n - 1 do
    equal
      ~msg:(Printf.sprintf "%s: ink of glyph %d" at g)
      (option box_t)
      (Path.bounds (Font.outline f g))
      (Font.ink f g)
  done;
  let kept = List.filter (fun g -> g mod 3 = 1) (List.init n Fun.id) in
  match Font.subset f kept with
  | exception e -> failf "%s: subsetting raised %s" at (Printexc.to_string e)
  | s ->
      check_sfnt s;
      let f' = decoded s in
      List.iter
        (fun g ->
          equal
            ~msg:(Printf.sprintf "%s: subset outline of glyph %d" at g)
            path_t (Font.outline f g) (Font.outline f' g))
        kept

(* [damaged ~at data] checks that [of_string] rejects [data] or returns a font
   that [answers]. *)
let damaged ~at data =
  match Font.of_string data with
  | exception e -> failf "%s: of_string raised %s" at (Printexc.to_string e)
  | Error _ -> `Rejected
  | Ok f ->
      answers ~at f;
      `Accepted

(* Each byte of the small seeds set to the extremes and to its neighbours. *)
let one_byte_away () =
  List.iter
    (fun (name, data) ->
      if name <> "regular" then
        String.iteri
          (fun i c ->
            let c = Char.code c in
            List.iter
              (fun v ->
                let at = Printf.sprintf "%s with byte %d := %d" name i v in
                ignore
                  (damaged ~at
                     (patch data i (fun b pos ->
                          Bytes.set_uint8 b pos (v land 0xFF)))))
              [ 0; 0x7F; 0x80; 0xFF; c + 1; c - 1 ])
          data)
    seeds

let robustness =
  group "robustness"
    [
      test "the seeds answer every query" (fun () ->
          List.iter (fun (at, data) -> answers ~at (decoded data)) seeds);
      test "a font one byte away from a small seed is rejected or answers"
        one_byte_away;
      prop ~count:300 "a font of bytes overwritten is rejected or answers"
        gen_damaged (fun (name, _, data) ->
          let verdict = damaged ~at:name data in
          cover "rejected" (verdict = `Rejected);
          cover "accepted" (verdict = `Accepted));
    ]

(* Identity and names *)

let family_of records =
  Font.family (tiny ~tables:[ ("name", name_table records) ] [ "" ])

let names () =
  let mac = (1, 0, 0, 1, "Caf\x8E") in
  let fr = (3, 1, 0x40C, 1, utf16 "Famille") in
  let us = (3, 1, 0x409, 1, utf16 "Family") in
  equal ~msg:"Macintosh Roman" string "Caf\u{E9}" (family_of [ mac ]);
  equal ~msg:"Macintosh Roman 0x80" string "\u{C4}"
    (family_of [ (1, 0, 0, 1, "\x80") ]);
  equal ~msg:"Windows over Macintosh" string "Famille" (family_of [ mac; fr ]);
  equal ~msg:"US English first" string "Family" (family_of [ mac; fr; us ]);
  equal ~msg:"typographic family" string "Typo"
    (family_of [ us; (3, 10, 0x409, 16, utf16 "Typo") ]);
  equal ~msg:"Macintosh Japanese ignored" string ""
    (family_of [ (1, 1, 0, 1, "X") ]);
  equal ~msg:"Unicode platform ignored" string ""
    (family_of [ (0, 3, 0, 1, utf16 "X") ])

let slant_t =
  Testable.make
    ~pp:(fun ppf s ->
      Format.pp_print_string ppf
        (match s with
        | `Normal -> "normal"
        | `Italic -> "italic"
        | `Oblique -> "oblique"))
    ~equal:( = )

let slants () =
  let sel = os2_pos regular_ttf + 62 in
  equal slant_t `Italic (Font.slant (decoded (set_u16 regular_ttf sel 0x81)));
  equal slant_t `Oblique (Font.slant (decoded (set_u16 regular_ttf sel 0x280)));
  let data = rename bold_ttf "OS/2" "OS/X" in
  equal ~msg:"without OS/2" int 400 (Font.weight (decoded data));
  equal ~msg:"without OS/2" slant_t `Normal (Font.slant (decoded data));
  equal ~msg:"head's italic bit" slant_t `Italic
    (Font.slant (decoded (set_u16 data (head_pos data + 44) 0x3)))

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
        ~name:(fun (n, _, _, _) -> n)
        "names and style of"
        [
          ("regular", regular (), "Inter-Regular", 400);
          ("bold", bold (), "Inter-Bold", 700);
        ]
        (fun (_, f, ps, weight) ->
          equal string "Inter" (Font.family f);
          equal string ps (Font.postscript_name f);
          equal int weight (Font.weight f);
          equal slant_t `Normal (Font.slant f));
      test "names come from US English, Windows, then Macintosh records" names;
      test "a font without names names nothing" (fun () ->
          let f = decoded (rename regular_ttf "name" "namX") in
          equal string "" (Font.family f);
          equal string "" (Font.postscript_name f));
      test "OS/2, else head, gives the slant" slants;
      test "the weight class is clamped to [1, 1000]" (fun () ->
          let w = os2_pos regular_ttf + 4 in
          equal int 1000 (Font.weight (decoded (set_u16 regular_ttf w 5000)));
          equal int 1 (Font.weight (decoded (set_u16 regular_ttf w 0))));
      test "pp escapes only quotes, backslashes and controls in the family"
        (fun () ->
          equal string "(font \"Inter\" 700 normal)"
            (Format.asprintf "%a" Font.pp (bold ()));
          let name = name_table [ (1, 0, 0, 1, "Caf\x8E\"\n") ] in
          equal string {|(font "Café\"\n" 400 normal)|}
            (Format.asprintf "%a" Font.pp
               (tiny ~tables:[ ("name", name) ] [ "" ])));
    ]

(* Metrics: Inter has 2048 units per em; values read with fontTools. *)

let metric_cases =
  [
    ("ascent", Font.ascent, em 1984);
    ("descent", Font.descent, em 494);
    ("line_gap", Font.line_gap, 0.);
    ("cap_height", Font.cap_height, em 1490);
    ("x_height", Font.x_height, em 1118);
    ("italic_angle", Font.italic_angle, 0.);
  ]

let typo_metrics () =
  let data = set_u16 regular_ttf (os2_pos regular_ttf + 68) 2000 in
  equal ~msg:"USE_TYPO_METRICS set" float_exact (em 2000)
    (Font.ascent (decoded data));
  let data = set_u16 data (os2_pos data + 62) 0x40 in
  equal ~msg:"USE_TYPO_METRICS clear" float_exact (em 1984)
    (Font.ascent (decoded data))

let heights () =
  let os2 = os2_pos regular_ttf in
  let cap = set_u16 regular_ttf (os2 + 88) 1400 in
  equal ~msg:"OS/2 version 4" float_exact (em 1400)
    (Font.cap_height (decoded cap));
  equal ~msg:"OS/2 version 2" float_exact (em 1400)
    (Font.cap_height (decoded (set_u16 cap os2 2)));
  equal ~msg:"OS/2 version 1: the top of H" float_exact (em 1490)
    (Font.cap_height (decoded (set_u16 cap os2 1)));
  equal ~msg:"zero in OS/2: the top of H" float_exact (em 1490)
    (Font.cap_height (decoded (set_u16 regular_ttf (os2 + 88) 0)));
  let f = decoded (rename regular_ttf "OS/2" "OS/X") in
  equal ~msg:"without OS/2: the top of H" float_exact (em 1490)
    (Font.cap_height f);
  equal ~msg:"without OS/2: the top of x" float_exact (em 1118)
    (Font.x_height f)

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

let metrics =
  group "metrics"
    [
      cases
        ~name:(fun (n, _, _) -> n)
        "metric" metric_cases
        (fun (_, m, v) ->
          equal ~msg:"regular" float_exact v (m (regular ()));
          equal ~msg:"bold" float_exact v (m (bold ())));
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
      test "cap and x heights read OS/2 version 2 on, else the ink of H and x"
        heights;
      test "italic_angle is post's angle in radians" (fun () ->
          let post = snd (table regular_ttf "post") in
          let f = decoded (set_u32 regular_ttf (post + 4) (-12 * 65536)) in
          equal (float 1e-15) (-12. *. Float.pi /. 180.) (Font.italic_angle f));
      test "missing tables fall back as documented" fallbacks;
      test "units per em may be 16 or 16384" (fun () ->
          equal float_exact (500. /. 16.)
            (Font.advance (tiny ~upem:16 [ ""; "" ]) 1);
          equal float_exact (500. /. 16384.)
            (Font.advance (tiny ~upem:16384 [ ""; "" ]) 1));
    ]

(* Character maps *)

let glyph_count = 40

(* A format 4 segment: first and last character, delta, and glyph ids. *)
type segment = int * int * int * int list option

let pp_segment ppf ((first, last, delta, ids) : segment) =
  Format.fprintf ppf "U+%04X-U+%04X %+d%s" first last delta
    (match ids with
    | None -> ""
    | Some ids -> " [" ^ String.concat " " (List.map string_of_int ids) ^ "]")

(* Segments in increasing order below U+FFFF, starting at U+0020 or near the end
   of the plane, adding deltas to characters or reading glyph ids. *)
let gen_segments =
  let open Gen in
  let piece =
    let* gap = frequency [ (4, int_range 0 3); (1, int_range 4 3000) ]
    and+ len = int_range 1 4 in
    let+ target = int_range 0 (glyph_count + 3)
    and+ ids =
      option (list ~size:(constant len) (int_range 0 (glyph_count + 3)))
    in
    (gap, len, target, ids)
  in
  let+ start = of_list [ 0x20; 0xFFF0 ]
  and+ pieces = list ~size:(int_range 0 8) piece in
  let _, segments =
    List.fold_left
      (fun (next, acc) (gap, len, target, ids) ->
        let first = next + gap in
        let last = first + len - 1 in
        if last >= 0xFFFF then (next, acc)
        else
          (* With ids the delta is added to them; without, to characters. *)
          let delta =
            match ids with
            | None -> (target - first) land 0xFFFF
            | Some _ -> (target - 5) land 0xFFFF
          in
          (last + 1, (first, last, delta, ids) :: acc))
      (start, []) pieces
  in
  List.rev segments

let gen_segments =
  Gen.with_pp
    (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_segment)
    gen_segments

(* The glyph of [c] under the format 4 [segments], as OpenType defines it, or
   [0] for a glyph the font does not have. *)
let format_4_glyph segments c =
  let g =
    match
      List.find_opt
        (fun (first, last, _, _) -> first <= c && c <= last)
        segments
    with
    | None -> 0
    | Some (_, _, delta, None) -> (c + delta) land 0xFFFF
    | Some (first, _, delta, Some ids) -> (
        match List.nth ids (c - first) with
        | 0 -> 0
        | id -> (id + delta) land 0xFFFF)
  in
  if g < glyph_count then g else 0

(* Groups of format 12 in increasing order, some past the BMP. *)
let gen_groups =
  let open Gen in
  let piece =
    triple
      (frequency [ (3, int_range 0 3); (1, int_range 4 300_000) ])
      (frequency [ (3, int_range 1 4); (1, int_range 5 2000) ])
      (int_range 0 (glyph_count + 3))
  in
  let+ pieces = list ~size:(int_range 0 8) piece in
  let _, groups =
    List.fold_left
      (fun (next, acc) (gap, len, g) ->
        let first = next + gap in
        let last = first + len - 1 in
        if last > 0x10FFFF then (next, acc)
        else (last + 1, (first, last, g) :: acc))
      (0x20, []) pieces
  in
  List.rev groups

let gen_groups =
  Gen.with_pp
    (Format.pp_print_list ~pp_sep:Format.pp_print_space (fun ppf (a, b, g) ->
         Format.fprintf ppf "U+%04X-U+%04X:%d" a b g))
    gen_groups

let format_12_glyph groups c =
  match
    List.find_opt (fun (first, last, _) -> first <= c && c <= last) groups
  with
  | Some (first, _, g) when g + (c - first) < glyph_count -> g + (c - first)
  | _ -> 0

(* [maps f ranges expected] checks the glyph of every character around each of
   [ranges] against [expected]. *)
let maps f ranges expected =
  List.iter
    (fun (first, last) ->
      for c = first - 1 to last + 1 do
        if Uchar.is_valid c then
          equal ~msg:(Printf.sprintf "U+%04X" c) int (expected c) (glyph f c)
      done)
    ((0xFFFF, 0xFFFF) :: ranges)

let empty_glyphs = List.init glyph_count (fun _ -> "")
let all_glyphs = List.init glyph_count Fun.id
let glyph_of_a_in cmap = glyph (tiny ~cmap empty_glyphs) 0x41
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

(* Every mapping of the character map, against a listing made with fontTools. *)
let cmap_listing f =
  let b = Buffer.create 8192 in
  for c = 0 to 0x10FFFF do
    if Uchar.is_valid c then
      let g = glyph f c in
      if g <> 0 then Printf.bprintf b "U+%04X %d\n" c g
  done;
  Buffer.contents b

let character_maps =
  group "character maps"
    [
      prop
        "glyph reads format 4 segments as OpenType defines them, and so does a \
         subset"
        gen_segments (fun segments ->
          let f =
            tiny ~cmap:(cmap_table [ (3, 1, format_4 segments) ]) empty_glyphs
          in
          cover "a segment of glyph ids"
            (List.exists (fun (_, _, _, ids) -> ids <> None) segments);
          cover "a segment near U+FFFF"
            (List.exists (fun (_, last, _, _) -> last > 0xFF00) segments);
          let ranges =
            List.map (fun (first, last, _, _) -> (first, last)) segments
          in
          maps f ranges (format_4_glyph segments);
          maps (subset f all_glyphs) ranges (format_4_glyph segments));
      prop
        "glyph reads format 12 groups as OpenType defines them, and so does a \
         subset"
        gen_groups (fun groups ->
          let f = tiny ~cmap:(cmap_12 groups) empty_glyphs in
          cover "a group past the BMP"
            (List.exists (fun (_, last, _) -> last > 0xFFFF) groups);
          let ranges =
            List.concat_map
              (fun (first, last, _) ->
                [
                  (first, Int.min last (first + 8));
                  (Int.max first (last - 8), last);
                ])
              groups
          in
          maps f ranges (format_12_glyph groups);
          maps (subset f all_glyphs) ranges (format_12_glyph groups));
      cases
        ~name:(fun (n, _, _) -> n)
        "character map selection" selection
        (fun (_, subtables, g) ->
          equal int g (glyph_of_a_in (cmap_table subtables)));
      test "glyph is 0 for a glyph the font does not have" (fun () ->
          equal int 0
            (glyph (tiny ~cmap:(cmap_12 [ (0x41, 0x41, 2) ]) [ ""; "" ]) 0x41));
      test "the bundled character map holds fontTools' mappings" (fun () ->
          expect_file
            (cmap_listing (regular ()))
            "packages/hugin/test/golden/cmap.expected");
    ]

(* Kerning *)

let x_advance = 0x4

(* Pairs of glyphs below 8 with their adjustments, at most one per pair. *)
let gen_pairs =
  let open Gen in
  let+ pairs =
    list ~size:(int_range 0 12)
      (triple (int_range 0 7) (int_range 0 7) (int_range (-300) 300))
  in
  List.sort_uniq (fun (a, b, _) (c, d, _) -> compare (a, b) (c, d)) pairs

let pp_pairs =
  Format.pp_print_list ~pp_sep:Format.pp_print_space (fun ppf (g, g', v) ->
      Format.fprintf ppf "%d,%d:%d" g g' v)

let pair_value pairs g g' =
  match List.find_opt (fun (a, b, _) -> a = g && b = g') pairs with
  | Some (_, _, v) -> v
  | None -> 0

let kerns_as f expected =
  for g = 0 to 7 do
    for g' = 0 to 7 do
      equal
        ~msg:(Printf.sprintf "%d %d" g g')
        int (expected g g') (units f g g')
    done
  done

let kerning_font tables = tiny ~tables (List.init 8 (fun _ -> ""))
let one_lookup st = gpos_table [ ("kern", [ 0 ]) ] [ lookup 2 [ st ] ]

(* Format 1 of [pairs], with its coverage in glyphs or in ranges, an advance
   after placements or alone, and second value records or none. *)
let gen_format_1 =
  let open Gen in
  let+ pairs = gen_pairs
  and+ ranges = bool
  and+ placements = bool
  and+ second = bool in
  let firsts = List.sort_uniq compare (List.map (fun (g, _, _) -> g) pairs) in
  let values v =
    (if placements then [ 7; -7; v ] else [ v ]) @ if second then [ 99 ] else []
  in
  let sets =
    List.map
      (fun g ->
        ( g,
          List.filter_map
            (fun (a, b, v) -> if a = g then Some (b, values v) else None)
            pairs ))
      firsts
  in
  let coverage =
    if ranges then Some (coverage_2 (List.mapi (fun i g -> (g, g, i)) firsts))
    else None
  in
  let st =
    pairs_1 ?coverage
      ~format:(if placements then 0x7 else x_advance)
      ~format2:(if second then x_advance else 0)
      sets
  in
  (pairs, (ranges, placements, second), st)

let gen_format_1 =
  Gen.with_pp
    (fun ppf (pairs, (r, p, s), _) ->
      Format.fprintf ppf "%a (ranges %b, placements %b, second %b)" pp_pairs
        pairs r p s)
    gen_format_1

(* Format 2 of the classes of glyphs below 8 and a matrix of adjustments, its
   class definitions as one array or as ranges. *)
let gen_format_2 =
  let open Gen in
  let* count1 = int_range 1 3 and+ count2 = int_range 1 3 in
  let classes count = array ~size:(constant 8) (int_range 0 (count - 1)) in
  let+ covered = subsequence [ 0; 1; 2; 3; 4; 5; 6; 7 ]
  and+ classes1 = classes count1
  and+ classes2 = classes count2
  and+ matrix =
    array ~size:(constant count1)
      (array ~size:(constant count2) (int_range (-300) 300))
  and+ ranges = bool in
  let def classes =
    if ranges then
      classes_2
        (List.filter_map
           (fun g -> if classes.(g) = 0 then None else Some (g, g, classes.(g)))
           (List.init 8 Fun.id))
    else classes_1 0 (Array.to_list classes)
  in
  let st =
    pairs_2
      ~coverage:(coverage_1 (if covered = [] then [ 7 ] else covered))
      ~classes1:(def classes1) ~classes2:(def classes2) ~format:x_advance
      (Array.to_list
         (Array.map
            (fun row -> List.map (fun v -> [ v ]) (Array.to_list row))
            matrix))
  in
  let covered = if covered = [] then [ 7 ] else covered in
  let expected g g' =
    if List.mem g covered then matrix.(classes1.(g)).(classes2.(g')) else 0
  in
  (covered, classes1, classes2, matrix, ranges, expected, st)

let gen_format_2 =
  let ints a = String.concat " " (Array.to_list (Array.map string_of_int a)) in
  Gen.with_pp
    (fun ppf (covered, c1, c2, m, ranges, _, _) ->
      Format.fprintf ppf
        "covered [%s] classes1 [%s] classes2 [%s] matrix [%s] ranges %b"
        (String.concat " " (List.map string_of_int covered))
        (ints c1) (ints c2)
        (String.concat "; " (Array.to_list (Array.map ints m)))
        ranges)
    gen_format_2

(* Subtables of a kern table: coverage, which marks which count, and pairs. *)
let gen_kern =
  let open Gen in
  list ~size:(int_range 1 3)
    (pair (of_list [ 0x1; 0x9; 0x0; 0x3; 0x5; 0x101 ]) gen_pairs)

let gen_kern =
  Gen.with_pp
    (Format.pp_print_list ~pp_sep:Format.pp_print_space (fun ppf (c, pairs) ->
         Format.fprintf ppf "0x%X: %a" c pp_pairs pairs))
    gen_kern

(* (name, features, lookups, pairs of glyphs and their kerning). *)
let lookup_rules =
  let pairs adjustments = pairs_1 ~format:x_advance adjustments in
  [
    ( "the first subtable of a lookup covering the pair",
      [ ("kern", [ 0 ]) ],
      [
        lookup 2
          [
            pairs_1 ~format:0x1 [ (1, [ (2, [ 5 ]) ]) ];
            pairs [ (1, [ (2, [ -50 ]); (3, [ -60 ]) ]) ];
          ];
      ],
      [ (1, 2, 0); (1, 3, -60) ] );
    ( "every kern lookup once, summed, other features ignored",
      [ ("kern", [ 0; 1; 0 ]); ("kern", [ 1 ]); ("liga", [ 2 ]) ],
      [
        lookup 2 [ pairs [ (1, [ (2, [ -1 ]) ]) ] ];
        lookup 2 [ pairs [ (1, [ (2, [ -10 ]) ]) ] ];
        lookup 2 [ pairs [ (1, [ (2, [ -100 ]) ]) ] ];
      ],
      [ (1, 2, -11) ] );
    ( "extension lookups of pair adjustments only",
      [ ("kern", [ 0; 1; 2 ]) ],
      [
        lookup 9 [ extension 2 (pairs [ (1, [ (2, [ -7 ]) ]) ]) ];
        lookup 9 [ extension 4 (pairs [ (1, [ (2, [ -100 ]) ]) ]) ];
        lookup 1 [ pairs [ (1, [ (2, [ -100 ]) ]) ] ];
      ],
      [ (1, 2, -7) ] );
    ( "a kern feature after another feature",
      [ ("liga", [ 1 ]); ("kern", [ 0 ]) ],
      [
        lookup 2 [ pairs [ (1, [ (2, [ -5 ]) ]) ] ];
        lookup 2 [ pairs [ (1, [ (2, [ -100 ]) ]) ] ];
      ],
      [ (1, 2, -5) ] );
  ]

(* A GPOS table whose [features] may refer to a lookup adjusting glyphs 1 and 2
   by [-10], beside a kern table adjusting them by [-30]. *)
let kern_fallback =
  [
    ("only a liga feature uses kern", [ ("liga", [ 0 ]) ], -30);
    ("no feature uses kern", [], -30);
    ("a kern feature uses GPOS", [ ("kern", [ 0 ]) ], -10);
    ("a kern feature without lookups uses GPOS", [ ("kern", []) ], 0);
  ]

let kern_of f s =
  Font.kerning f (glyph f (Char.code s.[0])) (glyph f (Char.code s.[1]))

(* (pair, regular, bold) in font units, as HarfBuzz applies them. *)
let kerned =
  [
    ("AV", -140, -162);
    ("To", -160, -160);
    ("LT", -197, -186);
    ("Yo", -157, -224);
    ("HH", 0, 0);
  ]

(* Every kerned pair of printable ASCII, against a listing made with
   HarfBuzz. *)
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

let kerning =
  group "kerning"
    [
      prop "kerning reads GPOS pair adjustments of format 1" gen_format_1
        (fun (pairs, _, st) ->
          kerns_as (kerning_font [ ("GPOS", one_lookup st) ]) (pair_value pairs));
      prop "kerning reads GPOS pair adjustments of format 2" gen_format_2
        (fun (_, _, _, _, _, expected, st) ->
          kerns_as (kerning_font [ ("GPOS", one_lookup st) ]) expected);
      prop "without GPOS, kerning sums the horizontal format 0 kern subtables"
        gen_kern (fun subtables ->
          (* Horizontal (bit 0), not minimum (bit 1), not cross-stream (bit 2),
             of format 0 (bits 8 to 15). *)
          let counted =
            List.filter
              (fun (c, _) -> c land 1 = 1 && c land 6 = 0 && c lsr 8 = 0)
              subtables
          in
          cover "a subtable left out"
            (List.compare_lengths counted subtables < 0);
          kerns_as
            (kerning_font [ ("kern", kern_table subtables) ])
            (fun g g' ->
              List.fold_left
                (fun acc (_, pairs) -> acc + pair_value pairs g g')
                0 counted));
      cases
        ~name:(fun (n, _, _, _) -> n)
        "kerning takes" lookup_rules
        (fun (_, features, lookups, expected) ->
          let f = kerning_font [ ("GPOS", gpos_table features lookups) ] in
          List.iter
            (fun (g, g', k) ->
              equal ~msg:(Printf.sprintf "%d %d" g g') int k (units f g g'))
            expected);
      cases
        ~name:(fun (n, _, _) -> n)
        "a GPOS table beside a kern table when" kern_fallback
        (fun (_, features, k) ->
          let gpos =
            gpos_table features
              [
                lookup 2 [ pairs_1 ~format:x_advance [ (1, [ (2, [ -10 ]) ]) ] ];
              ]
          in
          let kern = kern_table [ (0x1, [ (1, 2, -30) ]) ] in
          equal int k
            (units (kerning_font [ ("GPOS", gpos); ("kern", kern) ]) 1 2));
      test "a kern table of another version has no kerning" (fun () ->
          let table = kern_table [ (0x1, [ (1, 2, -100) ]) ] in
          let table =
            "\000\001" ^ String.sub table 2 (String.length table - 2)
          in
          equal int 0 (units (kerning_font [ ("kern", table) ]) 1 2));
      cases
        ~name:(fun (s, _, _) -> s)
        "kerning of the bundled faces is HarfBuzz's for" kerned
        (fun (s, r, b) ->
          equal ~msg:"regular" float_exact (em r) (kern_of (regular ()) s);
          equal ~msg:"bold" float_exact (em b) (kern_of (bold ()) s));
      test "kerning of printable ASCII is HarfBuzz's" (fun () ->
          expect_file
            (kerning_listing (regular ()))
            "packages/hugin/test/golden/kerning.expected");
    ]

(* Outlines *)

(* A component's offset in words or in bytes, its transform, and whether the
   transform applies to the offset. *)
type scale =
  | Unit
  | One of int
  | Xy of int * int
  | Two of int * int * int * int

type component = {
  in_words : bool;
  dx : int;
  dy : int;
  scale : scale;
  scaled : bool;
}

let pp_component ppf c =
  Format.fprintf ppf "{%s (%d, %d) %s%s}"
    (if c.in_words then "words" else "bytes")
    c.dx c.dy
    (match c.scale with
    | Unit -> "unit"
    | One s -> Printf.sprintf "scale %d" s
    | Xy (x, y) -> Printf.sprintf "scales %d %d" x y
    | Two (a, b, c, d) -> Printf.sprintf "2x2 %d %d %d %d" a b c d)
    (if c.scaled then " scaled offset" else "")

let gen_component =
  let open Gen in
  let f2dot14 = int_range (-32768) 32767 in
  let* in_words = bool in
  let offset =
    if in_words then int_range (-32768) 32767 else int_range (-128) 127
  in
  let+ dx = offset
  and+ dy = offset
  and+ scale =
    one_of
      [
        constant Unit;
        map (fun s -> One s) f2dot14;
        map (fun (x, y) -> Xy (x, y)) (pair f2dot14 f2dot14);
        map
          (fun (a, b, c, d) -> Two (a, b, c, d))
          (quad f2dot14 f2dot14 f2dot14 f2dot14);
      ]
  and+ scaled = bool in
  { in_words; dx; dy; scale; scaled }

let encode c =
  let flags, transform =
    match c.scale with
    | Unit -> (0, [])
    | One s -> (one_scale, [ s ])
    | Xy (x, y) -> (xy_scales, [ x; y ])
    | Two (a, b, c, d) -> (two_by_two, [ a; b; c; d ])
  in
  let flags =
    flags lor xy
    lor (if c.in_words then words else 0)
    lor if c.scaled then scaled_offset else 0
  in
  let args =
    if c.in_words then be16 c.dx ^ be16 c.dy
    else
      String.init 2 (fun i ->
          Char.chr ((if i = 0 then c.dx else c.dy) land 0xFF))
  in
  (flags, 1, args, transform)

(* The map OpenType gives a component: x' = a x + c y + e, y' = b x + d y + f,
   with the offset (e, f) through the matrix when scaled. *)
let placement c =
  let v x = Float.of_int x /. 16384. in
  let a, b, cc, d =
    match c.scale with
    | Unit -> (1., 0., 0., 1.)
    | One s -> (v s, 0., 0., v s)
    | Xy (x, y) -> (v x, 0., 0., v y)
    | Two (a, b, c, d) -> (v a, v b, v c, v d)
  in
  let e = Float.of_int c.dx and f = Float.of_int c.dy in
  let e, f =
    if c.scaled then ((a *. e) +. (cc *. f), (b *. e) +. (d *. f)) else (e, f)
  in
  { Affine.xx = a; yx = b; xy = cc; yy = d; x0 = e; y0 = f }

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
      && List.for_all2
           (fun a b ->
             a = b || Float.abs (a -. b) <= 1e-9 *. Float.max 1. (Float.abs a))
           a b)

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

(* A glyph of one contour of four on-curve points, its flags one repeated
   [repeat] times. *)
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

(* Ink boxes read with fontTools, in font units, y up. *)
let inks =
  [
    ("H", 0x48, (180, 0, 1342, 1490));
    ("o", 0x6F, (104, -24, 1124, 1132));
    ("g", 0x67, (104, -442, 1098, 1132));
    ("eacute", 0xE9, (104, -24, 1094, 1558));
    ("arrowright", 0x2192, (256, 0, 1752, 1304));
  ]

let contains_box outer inner =
  let slack = 1e-12 in
  Box2.minx outer -. slack <= Box2.minx inner
  && Box2.miny outer -. slack <= Box2.miny inner
  && Box2.maxx inner <= Box2.maxx outer +. slack
  && Box2.maxy inner <= Box2.maxy outer +. slack

let glyph_range =
  [
    ("advance", "Font.advance", fun f g -> ignore (Font.advance f g));
    ("kerning", "Font.kerning", fun f g -> ignore (Font.kerning f g 0));
    ( "kerning's second glyph",
      "Font.kerning",
      fun f g -> ignore (Font.kerning f 0 g) );
    ("outline", "Font.outline", fun f g -> ignore (Font.outline f g));
    ("ink", "Font.ink", fun f g -> ignore (Font.ink f g));
    ("subset", "Font.subset", fun f g -> ignore (Font.subset f [ 0; g ]));
  ]

let outlines =
  group "outlines"
    [
      prop "a composite is its components placed as OpenType defines"
        (Gen.with_pp
           (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_component)
           (Gen.list ~size:(Gen.int_range 1 3) gen_component))
        (fun components ->
          let f =
            tiny
              [ ""; simple [ square ]; composite (List.map encode components) ]
          in
          let expected =
            List.fold_left
              (fun acc c ->
                Path.append
                  (Path.transform Affine.(to_em * placement c) square_units)
                  acc)
              Path.empty components
          in
          equal path_near expected (Font.outline f 2));
      test "a contour without on-curve points starts between its first two"
        all_off_curve;
      test "a contour starts at its first on-curve point" starts_on_curve;
      test "flags repeat up to the last point" (fun () ->
          equal path_near square_path (Font.outline (tiny [ ""; repeated 3 ]) 1);
          is_error `Malformed ~sub:"glyph 1: flags repeat past its points"
            (Font.of_string (tiny_file [ ""; repeated 4 ])));
      test "instructions are skipped" (fun () ->
          let f = tiny [ ""; simple ~instructions:"\x01\x02\x03" [ square ] ] in
          equal path_near square_path (Font.outline f 1));
      test "contours of one point have no ink, of two have a line" (fun () ->
          let f =
            tiny
              [
                "";
                simple [ [ on 5 5 ]; [ on 7 7 ] ];
                simple [ [ on 0 0; on 100 0 ] ];
                simple [];
              ]
          in
          is_none ~pp:pp_box (Font.ink f 1);
          equal (option box_t) (Some (Box2.v 0. 0. 0.1 0.)) (Font.ink f 2);
          is_none ~pp:pp_box (Font.ink f 3));
      test "H is its contour of lines" (fun () ->
          expect
            (Format.asprintf "%a" Path.pp
               (Font.outline (regular ()) (glyph Font.regular 0x48)))
          @@ __POS_OF__
               {|
            M 0.0878906 0 L 0.0878906 -0.727539 L 0.180664 -0.727539 L 0.180664 -0.413086
            L 0.5625 -0.413086 L 0.5625 -0.727539 L 0.655273 -0.727539 L 0.655273 0
            L 0.5625 0 L 0.5625 -0.331055 L 0.180664 -0.331055 L 0.180664 0 Z
            |});
      cases
        ~name:(fun (n, _, _) -> n)
        "ink and the outline's bounds are fontTools' box of" inks
        (fun (_, c, (x0, y0, x1, y1)) ->
          let f = regular () in
          let box =
            Some
              (Box2.of_pts
                 (P2.v (em x0) (0. -. em y1))
                 (P2.v (em x1) (0. -. em y0)))
          in
          equal ~msg:"outline" (option box_t) box
            (Path.bounds (Font.outline f (glyph f c)));
          equal ~msg:"ink" (option box_t) box (Font.ink f (glyph f c)));
      test "a glyph without ink is empty" (fun () ->
          let f = regular () in
          List.iter
            (fun g ->
              is_true (Path.is_empty (Font.outline f g));
              is_none ~pp:pp_box (Font.ink f g))
            [ 0; glyph f 0x20 ]);
      cases ~name:fst "every glyph lies within the bounds of"
        [ ("regular", regular_ttf); ("bold", bold_ttf) ]
        (fun (_, data) ->
          let f = decoded data in
          for g = 0 to Font.glyph_count f - 1 do
            Option.iter
              (satisfies
                 ~msg:(Printf.sprintf "glyph %d" g)
                 box_t ~claim:"inside the font's bounds"
                 (contains_box (Font.bounds f)))
              (Font.ink f g)
          done);
      test "the bundled digits share one advance" (fun () ->
          List.iter
            (fun f ->
              let advance c = Font.advance f (glyph f (Char.code c)) in
              String.iter
                (fun c ->
                  equal ~msg:(String.make 1 c) float_exact (advance '0')
                    (advance c))
                "123456789")
            [ regular (); bold () ]);
      test "glyph_count is maxp's and advance hmtx's" (fun () ->
          equal int 730 (Font.glyph_count (regular ()));
          equal int 730 (Font.glyph_count (bold ()));
          equal float_exact (em 1522)
            (Font.advance (regular ()) (glyph Font.regular 0x48));
          equal float_exact (em 1530)
            (Font.advance (bold ()) (glyph Font.bold 0x48)));
      cases
        ~name:(fun (n, _, _) -> n)
        "raises on a glyph out of range in" glyph_range
        (fun (_, fn, f) ->
          let last = Font.glyph_count Font.regular - 1 in
          let msg g = Printf.sprintf "%s: glyph %d not in [0, %d]" fn g last in
          raises_match
            (Exn.invalid_arg ~substring:(msg (-1)))
            (fun () -> f Font.regular (-1));
          raises_match
            (Exn.invalid_arg ~substring:(msg (last + 1)))
            (fun () -> f Font.regular (last + 1)));
    ]

(* Subsetting *)

let gen_subset =
  let open Gen in
  let* bold = bool in
  let f = if bold then Font.bold else Font.regular in
  let+ gs =
    list ~size:(int_range 0 30) (int_range 0 (Font.glyph_count f - 1))
  in
  (f, gs)

let gen_subset =
  Gen.with_pp
    (fun ppf (f, gs) ->
      Format.fprintf ppf "%s [%s]" (Font.postscript_name f)
        (String.concat "; " (List.map string_of_int gs)))
    gen_subset

let eacute = glyph Font.regular 0xE9

(* What the subset of [f] to [gs] keeps of [f], as [Font.subset] states it. *)
let subset_keeps (f, gs) =
  let s = Font.subset f gs in
  check_sfnt s;
  let f' = decoded s in
  let n = Font.glyph_count f' in
  cover "a composite" (List.mem eacute gs);
  at_most ~msg:"glyph count" int ~than:(Font.glyph_count f) n;
  at_least ~msg:"glyph count" int ~than:(List.fold_left Int.max 0 gs + 1) n;
  for g = 0 to n - 1 do
    let msg = Printf.sprintf "glyph %d" g in
    let kept = g = 0 || List.mem g gs in
    let dropped = Path.is_empty (Font.outline f' g) && Font.advance f' g = 0. in
    if kept || not dropped then begin
      equal ~msg path_t (Font.outline f g) (Font.outline f' g);
      equal ~msg float_exact (Font.advance f g) (Font.advance f' g)
    end
  done;
  for c = 0 to 0x3000 do
    if Uchar.is_valid c then begin
      let g = glyph f c and g' = glyph f' c in
      let msg = Printf.sprintf "U+%04X" c in
      if List.mem g gs then equal ~msg int g g'
      else if g' <> 0 then equal ~msg int g g'
    end
  done;
  List.iter
    (fun (name, m) -> equal ~msg:name float_exact (m f) (m f'))
    [
      ("ascent", Font.ascent);
      ("descent", Font.descent);
      ("line gap", Font.line_gap);
      ("cap height", Font.cap_height);
      ("x height", Font.x_height);
      ("italic angle", Font.italic_angle);
    ];
  equal box_t (Font.bounds f) (Font.bounds f');
  equal string (Font.family f) (Font.family f');
  equal string (Font.postscript_name f) (Font.postscript_name f');
  equal int (Font.weight f) (Font.weight f');
  equal slant_t (Font.slant f) (Font.slant f');
  List.iter
    (fun g -> equal ~msg:"kerning" float_exact 0. (Font.kerning f' g g))
    gs;
  equal ~msg:"a subset of the subset" string s (Font.subset f' gs);
  equal ~msg:"the glyphs reversed and repeated" string s
    (Font.subset f (List.rev_append gs gs))

let subsetting =
  group "subsetting"
    [
      prop "a subset keeps its glyphs, their characters and the face's metrics"
        ~examples:
          [
            (Font.regular, [ eacute ]);
            (Font.bold, List.init (Font.glyph_count Font.bold) Fun.id);
          ]
        gen_subset subset_keeps;
      test "a kept composite keeps its components and their characters"
        (fun () ->
          let f = subset Font.regular [ eacute ] in
          equal path_t
            (Font.outline Font.regular eacute)
            (Font.outline f eacute);
          equal int (glyph Font.regular 0x65) (glyph f 0x65));
      test "a subset maps characters past the BMP" (fun () ->
          let f =
            tiny
              ~cmap:(cmap_12 [ (0x41, 0x41, 1); (0x1F600, 0x1F601, 2) ])
              [ ""; simple [ square ]; simple [ square ]; simple [ square ] ]
          in
          let f' = subset f [ 1; 3 ] in
          equal int 1 (glyph f' 0x41);
          equal int 0 (glyph f' 0x1F600);
          equal int 3 (glyph f' 0x1F601));
      test "a subset of 128 to 256 kB of outlines has long offsets" (fun () ->
          let f =
            tiny [ ""; simple_line 15000; simple_line 15000; simple [ square ] ]
          in
          let f' = subset f [ 1; 2; 3 ] in
          List.iter
            (fun g -> equal path_t (Font.outline f g) (Font.outline f' g))
            [ 1; 2; 3 ]);
      test "a subset maps more characters than format 4 holds" (fun () ->
          (* Characters share a glyph, so each is a segment of its own. *)
          let groups = List.init 9000 (fun i -> (0x100 + i, 0x100 + i, 1)) in
          let f = tiny ~cmap:(cmap_12 groups) [ ""; simple [ square ] ] in
          let f' = subset f [ 1 ] in
          maps f' [ (0x100, 0x100 + 8999) ] (glyph f));
      test "a character two segments hold maps as the first maps it" (fun () ->
          let segments =
            [ (0x41, 0x43, 1 - 0x41, None); (0x42, 0x44, 3 - 0x42, None) ]
          in
          let f =
            tiny ~cmap:(cmap_table [ (3, 1, format_4 segments) ]) empty_glyphs
          in
          let f' = subset f (List.init glyph_count Fun.id) in
          maps f' [ (0x41, 0x44) ] (glyph f));
    ]

let () =
  exit
    (run "hugin.font"
       [
         loading;
         robustness;
         identity;
         metrics;
         character_maps;
         kerning;
         outlines;
         subsetting;
       ])
