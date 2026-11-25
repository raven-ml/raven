(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_gg
open Hugin_font
open Build

let read path = In_channel.with_open_bin path In_channel.input_all
let regular_ttf = read "../../lib/font/fonts/Inter-Regular.ttf"
let bold_ttf = read "../../lib/font/fonts/Inter-Bold.ttf"

(* Inter has 2048 units per em. *)
let em units = Float.of_int units /. 2048.
let font_t = Testable.make ~pp:Font.pp ~equal:Font.equal
let path_t = Testable.make ~pp:Path.pp ~equal:Path.equal

let pp_box ppf b =
  Format.fprintf ppf "[%.17g, %.17g; %.17g, %.17g]" (Box2.minx b) (Box2.miny b)
    (Box2.maxx b) (Box2.maxy b)

let box_t = Testable.make ~pp:pp_box ~equal:Box2.equal

let slant_t =
  Testable.make
    ~pp:(fun ppf s ->
      Format.pp_print_string ppf
        (match s with
        | `Normal -> "normal"
        | `Italic -> "italic"
        | `Oblique -> "oblique"))
    ~equal:( = )

let outlines_t =
  Testable.make
    ~pp:(fun ppf o ->
      Format.pp_print_string ppf
        (match o with `Truetype -> "`Truetype" | `Cff -> "`Cff"))
    ~equal:( = )

let decoded data =
  match Font.of_string data with
  | Ok f -> f
  | Error e -> failf "%a" Font.pp_error e

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

(* [subset f gs] is [Font.subset f gs] checked as an OpenType file and
   decoded. *)
let subset f gs =
  let s = Font.subset f gs in
  check_sfnt s;
  decoded s

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

(* CFF and CFF2 fixtures, written with their outlines by gen/gen.py. *)
let cff_otf = read "golden/cff.otf"
let cff2_otf = read "golden/cff2.otf"
let cff_2048_otf = read "golden/cff-2048.otf"

(* Bundled faces *)

let bundled =
  group "bundled faces"
    [
      cases "are Inter"
        ~id:(fun (n, _, _, _) -> n)
        [
          ("regular", Font.regular, "Inter-Regular", 400);
          ("bold", Font.bold, "Inter-Bold", 700);
        ]
        (fun (_, f, ps, weight) ->
          equal string "Inter" (Font.family f);
          equal string ps (Font.postscript_name f);
          equal int weight (Font.weight f);
          equal slant_t `Normal (Font.slant f));
      cases "share one advance among the ten digits" ~id:fst
        [ ("regular", Font.regular); ("bold", Font.bold) ]
        (fun (_, f) ->
          let advance c = Font.advance f (glyph f (Char.code c)) in
          String.iter
            (fun c ->
              equal ~msg:(String.make 1 c) float_exact (advance '0') (advance c))
            "123456789");
    ]

(* Loading *)

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
    ( "CFF of version 3",
      `Unsupported,
      cff_file ~glyphs:1 "\003\000\004\004",
      "'CFF ' version 3" );
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
    ( "flags that repeat past a glyph's points",
      `Malformed,
      tiny_file [ ""; repeated 4 ],
      "glyph 1: flags repeat past its points" );
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
   65535 components. *)
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
    (Font.of_string over)

(* Glyph [k + 1] is 100 copies of glyph [k], so glyph 3 expands to 1,010,100
   components. *)
let nested_components () =
  let nested =
    "" :: List.init 3 (fun k -> composite_of (List.init 100 (fun _ -> k)))
  in
  is_error `Malformed ~sub:"glyph 3: more than 65535 components"
    (Font.of_string (tiny_file nested))

(* A load's work is bounded by the file's size: 1000 copies of a glyph of 65534
   points expand to 65.5 million points from a file under 400 kB, past the
   16,777,216 + 64 n steps of a load. *)
let composite_work () =
  let copies = List.init 1000 (fun _ -> composite_of [ 1 ]) in
  let over = tiny_file ("" :: simple_line 65534 :: copies) in
  less int ~than:400_000 (String.length over);
  is_error `Unsupported ~sub:"expand to more than" (Font.of_string over)

(* Robustness: [of_string] checks all it reads, so a font it returns answers
   every query. The laws run over fonts with bytes overwritten, from the regular
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
    simple [ [ on 0 0; on 100 0 ] ];
    simple [];
    simple [ [ on 5 5 ] ];
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
    ("CFF", read "golden/cff-small.otf");
    ("CFF2", read "golden/cff2-small.otf");
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

(* [answers ~at f] checks that every query of [f] answers as font.mli states:
   none raises, [ink] is the bounds of [outline], and a subset is an OpenType
   file that keeps its glyphs' outlines. [at] names [f] in failures. *)
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

let of_string =
  group "of_string"
    [
      cases "returns an error for"
        ~id:(fun (n, _, _, _) -> n)
        rejected
        (fun (_, kind, data, sub) -> is_error kind ~sub (Font.of_string data));
      test "accepts 16 and 16384 units per em" (fun () ->
          equal float_exact (500. /. 16.)
            (Font.advance (tiny ~upem:16 [ ""; "" ]) 1);
          equal float_exact (500. /. 16384.)
            (Font.advance (tiny ~upem:16384 [ ""; "" ]) 1));
      test "a composite expands to at most 65535 points" composite_points;
      test "a composite expands to at most 65535 components"
        composite_components;
      test "nested composites count every component they expand to"
        nested_components;
      test "a load's work is bounded by the file's size" composite_work;
      cases "returns a font that answers every query, for" ~id:fst seeds
        (fun (at, data) -> answers ~at (decoded data));
      test "rejects or answers every query, one byte away from a small font"
        one_byte_away;
      prop ~count:300 "rejects or answers every query, bytes overwritten"
        gen_damaged (fun (name, _, data) ->
          let verdict = damaged ~at:name data in
          cover "rejected" (verdict = `Rejected);
          cover "accepted" (verdict = `Accepted));
    ]

(* CFF and CFF2 outlines *)

(* [expected_outlines file upem] is each glyph's outline that the generator
   recorded from fontTools, in em with y down. Coordinates are written as
   multiples of 2⁻¹⁶. A moveto followed by no segment draws nothing. *)
let expected_outlines file upem =
  let v s = Float.of_string s /. 65536. in
  let pt x y = P2.v (v x /. upem) ((0. -. v y) /. upem) in
  let glyphs = ref [] and path = ref Path.empty and start = ref None in
  let draw () =
    Option.iter (fun p -> path := Path.move_to p !path) !start;
    start := None
  in
  let finish () = glyphs := !path :: !glyphs in
  List.iteri
    (fun i line ->
      match String.split_on_char ' ' line with
      | [ "glyph"; _ ] ->
          if i > 0 then finish ();
          path := Path.empty
      | [ "M"; x; y ] -> start := Some (pt x y)
      | [ "L"; x; y ] ->
          draw ();
          path := Path.line_to (pt x y) !path
      | [ "C"; x1; y1; x2; y2; x; y ] ->
          draw ();
          path := Path.cubic_to (pt x1 y1) (pt x2 y2) (pt x y) !path
      | [ "Z" ] ->
          if Option.is_none !start then path := Path.close !path;
          start := None
      | [ "" ] -> ()
      | _ -> failf "%s: line %S" file line)
    (String.split_on_char '\n' (read file));
  finish ();
  Array.of_list (List.rev !glyphs)

(* [n v] is the number [v] as a charstring writes an integer. *)
let n v =
  if v >= -107 && v <= 107 then String.make 1 (Char.chr (v + 139))
  else "\028" ^ be16 v

(* [fx v] is [v] as a 16.16 number. *)
let fx v = "\255" ^ be32 (Float.to_int (Float.round (v *. 65536.)))
let ns vs = cat (List.map n vs)
let ones k = ns (List.init k (fun _ -> 1))
let rmoveto = "\021"
let rlineto = "\005"
let hlineto = "\006"
let rrcurveto = "\008"
let hvcurveto = "\031"
let endchar = "\014"
let return = "\011"
let callsubr = "\010"
let callgsubr = "\029"
let esc k = "\012" ^ String.make 1 (Char.chr k)

(* A charstring's subroutine number [k] written with the bias of a font of fewer
   than 1240 subroutines. *)
let subr k = n (k - 107)

(* [cff_bytes ?gsubrs ?subrs glyphs] is the CFF font whose glyph 0 is empty and
   whose glyph [i + 1] is [glyphs.(i)]; [cff2_bytes] the CFF2 one. *)
let cff_bytes ?gsubrs ?subrs glyphs =
  cff_file
    ~glyphs:(List.length glyphs + 1)
    (cff_table ?gsubrs ?subrs (endchar :: glyphs))

let cff2_bytes ?regions ?subrs glyphs =
  cff_file ~cff2:true
    ~glyphs:(List.length glyphs + 1)
    (cff2_table ?regions ?subrs ("" :: glyphs))

let ptf x y = P2.v (x /. 1000.) ((0. -. y) /. 1000.)

(* [shape pts] is the closed polygon through [pts], in font units. *)
let shape ?(path = Path.empty) pts =
  match pts with
  | [] -> path
  | (x, y) :: rest ->
      List.fold_left
        (fun p (x, y) -> Path.line_to (ptf x y) p)
        (Path.move_to (ptf x y) path)
        rest
      |> Path.close

(* [steps k] is the polygon of [k] unit steps from the origin, along x first and
   then alternately along y and x, as [k] arguments of hlineto draw it. *)
let steps k =
  shape
    (List.init (k + 1) (fun i ->
         (Float.of_int ((i + 1) / 2), Float.of_int (i / 2))))

(* Glyphs whose outlines the Type 2 and CFF2 definitions fix. *)
let charstring_cases =
  let curve ?(path = Path.empty) start cs =
    let x0, y0 = start in
    List.fold_left
      (fun p ((a, b), (c, d), (e, f)) ->
        Path.cubic_to (ptf a b) (ptf c d) (ptf e f) p)
      (Path.move_to (ptf x0 y0) path)
      cs
    |> Path.close
  in
  [
    ( "a width before the first moveto is dropped",
      cat [ ns [ 500; 10; 20 ]; rmoveto; ns [ 30; 0 ]; rlineto; endchar ],
      shape [ (10., 20.); (40., 20.) ] );
    ( "hlineto alternates from horizontal",
      cat [ ns [ 0; 0 ]; rmoveto; ns [ 10; 20; 30 ]; hlineto; endchar ],
      shape [ (0., 0.); (10., 0.); (10., 20.); (40., 20.) ] );
    ( "an operator takes 48 arguments, the stack's limit",
      cat [ ns [ 0; 0 ]; rmoveto; ones 48; hlineto; endchar ],
      steps 48 );
    ( "a moveto closes the contour and moves from its last point",
      cat
        [
          ns [ 0; 0 ];
          rmoveto;
          ns [ 10; 0 ];
          rlineto;
          ns [ 5; 5 ];
          rmoveto;
          ns [ 0; 10 ];
          rlineto;
          endchar;
        ],
      shape ~path:(shape [ (0., 0.); (10., 0.) ]) [ (15., 5.); (15., 15.) ] );
    ( "rrcurveto takes 16.16 deltas",
      cat
        [
          ns [ 0; 0 ];
          rmoveto;
          fx 1.5;
          fx (-2.25);
          ns [ 3; 4 ];
          fx 0.125;
          ns [ 6 ];
          rrcurveto;
          endchar;
        ],
      curve (0., 0.) [ ((1.5, -2.25), (4.5, 1.75), (4.625, 7.75)) ] );
    ( "hvcurveto ends a last curve by its fifth argument",
      cat [ ns [ 0; 0 ]; rmoveto; ns [ 10; 20; 30; 40; 5 ]; hvcurveto; endchar ],
      curve (0., 0.) [ ((10., 0.), (30., 30.), (35., 70.)) ] );
    ( "hflex returns to its starting y",
      cat
        [
          ns [ 0; 0 ];
          rmoveto;
          ns [ 10; 20; 30; 40; 50; 60; 70 ];
          esc 34;
          endchar;
        ],
      curve (0., 0.)
        [
          ((10., 0.), (30., 30.), (70., 30.));
          ((120., 30.), (180., 0.), (250., 0.));
        ] );
    ( "flex1 takes its last delta along the longer of its spans",
      cat
        [
          ns [ 0; 0 ];
          rmoveto;
          ns [ 10; 1; 10; 1; 10; 1; 10; 1; 10; 1; 7 ];
          esc 37;
          endchar;
        ],
      curve (0., 0.)
        [ ((10., 1.), (20., 2.), (30., 3.)); ((40., 4.), (50., 5.), (57., 0.)) ]
    );
    ( "arithmetic results are 16.16 numbers",
      cat
        [
          ns [ 1; 3 ];
          esc 12;
          ns [ 0 ];
          rmoveto;
          ns [ 2 ];
          esc 26;
          ns [ 0 ];
          rlineto;
          ns [ 4; 5 ];
          esc 28;
          esc 11;
          ns [ 0 ];
          esc 28;
          rlineto;
          endchar;
        ],
      (* 1/3 and √2 round to the nearest 2⁻¹⁶; [4 5 exch sub] is 1. *)
      let third = Float.round (65536. /. 3.) /. 65536.
      and root = Float.round (Float.sqrt 2. *. 65536.) /. 65536. in
      shape [ (third, 0.); (third +. root, 0.); (third +. root, 1.) ] );
    ( "the stack operators and the transient array",
      cat
        [
          ns [ 7; 0 ];
          esc 20;
          ns [ 1; 2; 3; 3; 1 ];
          esc 30;
          esc 18;
          rmoveto;
          ns [ 0 ];
          esc 21;
          ns [ 0 ];
          rlineto;
          ns [ 1; 2; 5; 4 ];
          esc 22;
          ns [ 9; 1 ];
          esc 29;
          esc 18;
          rlineto;
          endchar;
        ],
      (* [1 2 3 3 1 roll] is [3 1 2]; [drop] leaves [3 1]. *)
      shape [ (3., 1.); (10., 1.); (12., 10.) ] );
    ( "hintmask counts the stems before it, implicit vstems included",
      cat
        [
          ns [ 0; 10; 20; 10 ];
          "\018";
          ns [ 30; 10; 50; 10 ];
          "\019\255";
          ns [ 0; 0 ];
          rmoveto;
          ns [ 5; 5 ];
          rlineto;
          endchar;
        ],
      shape [ (0., 0.); (5., 5.) ] );
    ("an endchar alone draws nothing", endchar, Path.empty);
    ( "a moveto that draws nothing leaves no subpath",
      cat [ ns [ 5; 5 ]; rmoveto; endchar ],
      Path.empty );
  ]

(* Subroutines and their calls: (name, global ones, local ones, the glyph, its
   outline). *)
let subroutine_cases =
  [
    ( "a local subroutine draws what it holds",
      [],
      [ cat [ ns [ 10; 0 ]; rlineto; return ] ],
      cat [ ns [ 0; 0 ]; rmoveto; subr 0; callsubr; endchar ],
      shape [ (0., 0.); (10., 0.) ] );
    ( "a global subroutine may end the glyph",
      [ cat [ ns [ 0; 10 ]; rlineto; endchar ] ],
      [],
      cat [ ns [ 0; 0 ]; rmoveto; subr 0; callgsubr ],
      shape [ (0., 0.); (0., 10.) ] );
    ( "subroutines nest ten deep",
      [],
      List.init 10 (fun k ->
          if k = 9 then cat [ ns [ 1; 0 ]; rlineto; return ]
          else cat [ subr (k + 1); callsubr; return ]),
      cat [ ns [ 0; 0 ]; rmoveto; subr 0; callsubr; endchar ],
      shape [ (0., 0.); (1., 0.) ] );
  ]

(* Data no font may hold, and the error of_string returns for it. *)
let hostile_cases =
  let chain k =
    List.init k (fun i ->
        if i = k - 1 then return else cat [ subr (i + 1); callsubr; return ])
  in
  let call0 = cat [ subr 0; callsubr; endchar ] in
  [
    ( "49 arguments",
      cff_bytes [ cat [ ns (List.init 49 Fun.id); endchar ] ],
      `Malformed,
      "more than 48 arguments" );
    ( "subroutines nested eleven deep",
      cff_bytes ~subrs:(chain 11) [ call0 ],
      `Malformed,
      "deeper than 10" );
    ( "a subroutine calling itself",
      cff_bytes ~subrs:[ cat [ subr 0; callsubr ] ] [ call0 ],
      `Malformed,
      "deeper than 10" );
    ( "calls that fan out past 65535 bytes",
      cff_bytes
        ~subrs:
          (List.init 10 (fun k ->
               if k = 9 then return
               else cat (List.init 8 (fun _ -> subr (k + 1) ^ callsubr))))
        [ call0 ],
      `Unsupported,
      "more than 65535 bytes" );
    ( "an outline one charstring writes in more than 65535 bytes",
      (* 33600 segments of one-byte arguments to hlineto run 34304 bytes, and
         take two bytes each as rlineto writes them. *)
      cff_bytes
        [
          cat
            ([ ns [ 0; 0 ]; rmoveto ]
            @ List.init 700 (fun _ -> ones 48 ^ hlineto)
            @ [ endchar ]);
        ],
      `Unsupported,
      "its outline takes more than 65535 bytes" );
    ( "glyphs whose runs add up past the load's bound",
      (* Each glyph runs 49155 bytes through subroutines that call the next four
         times, so 400 of them pass 16,777,216 + 64 n steps. *)
      cff_bytes
        ~subrs:
          (List.init 8 (fun k ->
               if k = 7 then ""
               else
                 cat (List.init 4 (fun _ -> subr (k + 1) ^ callsubr)) ^ return))
        (List.init 400 (fun _ -> cat [ ns [ 0; 0 ]; rmoveto; call0 ])),
      `Unsupported,
      "expand to more than" );
    ( "a call of a missing subroutine",
      cff_bytes [ call0 ],
      `Malformed,
      "subroutine 0 of 0" );
    ( "a line before any moveto",
      cff_bytes [ cat [ ns [ 1; 1 ]; rlineto; endchar ] ],
      `Malformed,
      "before a moveto" );
    ( "a charstring without endchar",
      cff_bytes [ cat [ ns [ 1; 1 ]; rmoveto ] ],
      `Malformed,
      "does not end with endchar" );
    ( "a hint mask past the charstring's end",
      cff_bytes [ cat [ ns [ 0; 10 ]; "\001"; "\019" ] ],
      `Malformed,
      "hint mask" );
    ( "a division by zero",
      cff_bytes [ cat [ ns [ 1; 0 ]; esc 12; endchar ] ],
      `Malformed,
      "divides by zero" );
    ( "a product past the 16.16 range",
      (* Written as C's [%g] writes it, the same on every system: the C
         library's printf writes 9e+008 under mingw-w64. *)
      cff_bytes [ cat [ ns [ 30000; 30000 ]; esc 24; endchar ] ],
      `Malformed,
      "glyph 1: an arithmetic result 9e+08 out of range" );
    ( "the random operator",
      cff_bytes [ cat [ esc 23; endchar ] ],
      `Unsupported,
      "random" );
    ( "an accented glyph built by endchar",
      cff_bytes [ cat [ ns [ 0; 0; 65; 97 ]; endchar ] ],
      `Unsupported,
      "accented" );
    ( "514 arguments in CFF2",
      cff2_bytes [ ones 514 ],
      `Malformed,
      "more than 513 arguments" );
    ( "a CFF2 blend without a VariationStore",
      cff2_bytes [ cat [ ns [ 1; 1 ]; "\016" ] ],
      `Malformed,
      "without a VariationStore" );
    ( "a CFF2 INDEX of 2³² − 1 items",
      cff_file ~cff2:true ~glyphs:1
        (let t = cff2_table [ "" ] in
         (* The CharStrings INDEX follows the header, Top DICT and the empty
            global subroutines. *)
         set_bytes t (5 + 15 + 4) "\255\255\255\255"),
      `Malformed,
      "CFF2" );
    ( "fewer charstrings than glyphs",
      cff_file ~glyphs:3 (cff_table [ endchar; endchar ]),
      `Malformed,
      "2 charstrings for 3 glyphs" );
  ]

let cff_outlines =
  group "CFF outlines"
    [
      cases "are those fontTools decodes, for"
        ~id:(fun (n, _, _, _) -> n)
        [
          ("CFF", cff_otf, "golden/cff.expected", 1000.);
          ("CFF2", cff2_otf, "golden/cff2.expected", 1000.);
          ( "CFF of 2048 units per em",
            cff_2048_otf,
            "golden/cff-2048.expected",
            2048. );
        ]
        (fun (_, data, expected, upem) ->
          let f = decoded data in
          let expected = expected_outlines expected upem in
          equal int (Array.length expected) (Font.glyph_count f);
          Array.iteri
            (fun g p ->
              equal
                ~msg:(Printf.sprintf "glyph %d" g)
                path_t p (Font.outline f g))
            expected);
      cases "draw what Type 2 defines"
        ~id:(fun (n, _, _) -> n)
        charstring_cases
        (fun (_, cs, expected) ->
          let f = decoded (cff_bytes [ cs ]) in
          equal path_t expected (Font.outline f 1));
      cases "run their subroutines"
        ~id:(fun (n, _, _, _, _) -> n)
        subroutine_cases
        (fun (_, gsubrs, subrs, cs, expected) ->
          let f = decoded (cff_bytes ~gsubrs ~subrs [ cs ]) in
          equal path_t expected (Font.outline f 1));
      test "a CFF2 operator takes 513 arguments, the stack's limit" (fun () ->
          let glyph = cat [ ns [ 0; 0 ]; rmoveto; ones 513; hlineto ] in
          let f = decoded (cff2_bytes [ glyph ]) in
          equal path_t (steps 513) (Font.outline f 1));
      test "a CFF2 blend gives its defaults at the default instance" (fun () ->
          let f =
            decoded
              (cff2_bytes ~regions:2
                 [
                   cat
                     [
                       ns [ 10; 20; 1; 2; 3; 4; 2 ];
                       "\016";
                       rmoveto;
                       ns [ 5; 0 ];
                       rlineto;
                     ];
                 ])
          in
          equal path_t (shape [ (10., 20.); (15., 20.) ]) (Font.outline f 1));
      (* OpenType's CFF2 table: "if an unrecognized operator is encountered
         while decoding, it is ignored and the stack is cleared." *)
      test "CFF2 ignores an operator it does not know and clears the stack"
        (fun () ->
          let f =
            decoded
              (cff2_bytes
                 [
                   cat
                     [
                       ns [ 7 ];
                       "\009";
                       ns [ 0; 0 ];
                       rmoveto;
                       ns [ 3; 0 ];
                       rlineto;
                     ];
                 ])
          in
          equal path_t (shape [ (0., 0.); (3., 0.) ]) (Font.outline f 1));
      cases "make of_string return an error for"
        ~id:(fun (n, _, _, _) -> n)
        hostile_cases
        (fun (_, data, kind, sub) -> is_error kind ~sub (Font.of_string data));
    ]

let of_file =
  group "of_file"
    [
      test "decodes the file's contents" (fun () ->
          let path = Filename.concat (temp_dir ()) "inter.ttf" in
          Out_channel.with_open_bin path (fun oc ->
              Out_channel.output_string oc regular_ttf);
          equal font_t Font.regular
            (require_ok ~pp:Font.pp_error (Font.of_file path)));
      test "returns Io naming a file it cannot read" (fun () ->
          is_error `Io ~sub:"missing.ttf"
            (Font.of_file (Filename.concat (temp_dir ()) "missing.ttf")));
    ]

let pp_error =
  group "pp_error"
    [
      cases "formats the message of" ~id:fst
        [
          ("Io", Font.Io "the message");
          ("Malformed", Font.Malformed "the message");
          ("Unsupported", Font.Unsupported "the message");
        ]
        (fun (_, e) ->
          contains ~sub:"the message" (Format.asprintf "%a" Font.pp_error e));
    ]

let loading = group "loading" [ of_string; cff_outlines; of_file; pp_error ]

(* Identity *)

let family_of records =
  Font.family (tiny ~tables:[ ("name", name_table records) ] [ "" ])

let mac = (1, 0, 0, 1, "Caf\x8E")
let fr = (3, 1, 0x40C, 1, utf16 "Famille")
let de = (3, 1, 0x407, 1, utf16 "Familie")
let us = (3, 1, 0x409, 1, utf16 "Family")

(* Records of a name table and the family read from them. *)
let names =
  [
    ("a Macintosh Roman record", [ mac ], "Caf\u{E9}");
    ("Macintosh Roman 0x80", [ (1, 0, 0, 1, "\x80") ], "\u{C4}");
    ("a Windows record over a Macintosh one", [ mac; fr ], "Famille");
    ("the first Windows record in table order", [ mac; fr; de ], "Famille");
    ("the US English record first", [ mac; fr; us ], "Family");
    ( "the typographic family over the family",
      [ us; (3, 10, 0x409, 16, utf16 "Typo") ],
      "Typo" );
    ("nothing from Macintosh Japanese", [ (1, 1, 0, 1, "X") ], "");
    ("nothing from the Unicode platform", [ (0, 3, 0, 1, utf16 "X") ], "");
  ]

let slants =
  let sel = os2_pos regular_ttf + 62 in
  let without_os2 = rename bold_ttf "OS/2" "OS/X" in
  let head_italic data = set_u16 data (head_pos data + 44) 0x3 in
  [
    ("OS/2's italic bit", set_u16 regular_ttf sel 0x81, `Italic);
    ("OS/2's oblique bit", set_u16 regular_ttf sel 0x280, `Oblique);
    ("head's italic bit without OS/2", head_italic without_os2, `Italic);
    ("head's italic bit beside OS/2", head_italic regular_ttf, `Normal);
    ("no bit, without OS/2", without_os2, `Normal);
  ]

let identity =
  group "identity"
    [
      group "bytes"
        [
          test "of a bundled face is the file it embeds" (fun () ->
              equal string regular_ttf (Font.bytes Font.regular);
              equal string bold_ttf (Font.bytes Font.bold));
          test "of a font from of_string is the string decoded" (fun () ->
              let data = tiny_file [ ""; simple [ square ] ] in
              equal string data (Font.bytes (decoded data)));
        ];
      group "outlines"
        [
          cases "is the format of the outlines of"
            ~id:(fun (n, _, _) -> n)
            [
              ("Inter", regular_ttf, `Truetype);
              ("the CFF fixture", cff_otf, `Cff);
              ("the CFF2 fixture", cff2_otf, `Cff);
            ]
            (fun (_, data, o) ->
              equal outlines_t o (Font.outlines (decoded data)));
        ];
      group "equal"
        [
          test "compares bytes" (fun () ->
              equal font_t Font.regular (decoded regular_ttf);
              not_equal font_t Font.regular Font.bold);
        ];
      group "compare"
        [
          test "is a total order compatible with equal" (fun () ->
              Law.order
                (Testable.with_compare Font.compare font_t)
                (Font.regular, Font.bold, decoded regular_ttf));
        ];
      group "family"
        [
          cases "reads"
            ~id:(fun (n, _, _) -> n)
            names
            (fun (_, records, name) -> equal string name (family_of records));
          test "is empty without a name table" (fun () ->
              equal string ""
                (Font.family (decoded (rename regular_ttf "name" "namX"))));
        ];
      group "postscript_name"
        [
          test "reads name ID 6 as family reads names" (fun () ->
              let records =
                [ (1, 0, 0, 6, "Mac-PS"); (3, 1, 0x409, 6, utf16 "Win-PS") ]
              in
              equal string "Win-PS"
                (Font.postscript_name
                   (tiny ~tables:[ ("name", name_table records) ] [ "" ])));
          test "is empty without a name table" (fun () ->
              equal string ""
                (Font.postscript_name
                   (decoded (rename regular_ttf "name" "namX"))));
        ];
      group "weight"
        [
          test "is clamped to [1, 1000]" (fun () ->
              let w = os2_pos regular_ttf + 4 in
              equal int 1000
                (Font.weight (decoded (set_u16 regular_ttf w 5000)));
              equal int 1 (Font.weight (decoded (set_u16 regular_ttf w 0))));
          test "is 400 without OS/2" (fun () ->
              equal int 400
                (Font.weight (decoded (rename bold_ttf "OS/2" "OS/X"))));
        ];
      group "slant"
        [
          cases "reads"
            ~id:(fun (n, _, _) -> n)
            slants
            (fun (_, data, s) -> equal slant_t s (Font.slant (decoded data)));
        ];
      group "pp"
        [
          test "formats the family, weight and slant" (fun () ->
              expect (Format.asprintf "%a" Font.pp Font.bold)
              @@ __POS_OF__ {|(font "Inter" 700 normal)|});
          test "escapes only quotes, backslashes and controls in the family"
            (fun () ->
              let name = name_table [ (1, 0, 0, 1, "Caf\x8E\"\n") ] in
              contains ~sub:{|"Café\"\n"|}
                (Format.asprintf "%a" Font.pp
                   (tiny ~tables:[ ("name", name) ] [ "" ])));
        ];
    ]

(* Character maps of tiny fonts of 40 glyphs *)

let glyph_count = 40
let empty_glyphs = List.init glyph_count (fun _ -> "")
let all_glyphs = List.init glyph_count Fun.id

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

let cover_segments segments =
  cover "a segment of glyph ids"
    (List.exists (fun (_, _, _, ids) -> ids <> None) segments);
  cover "a segment near U+FFFF"
    (List.exists (fun (_, last, _, _) -> last > 0xFF00) segments)

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

let format_4_ranges segments =
  List.map (fun (first, last, _, _) -> (first, last)) segments

let format_4_font segments =
  tiny ~cmap:(cmap_table [ (3, 1, format_4 segments) ]) empty_glyphs

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

let cover_groups groups =
  cover "a group past the BMP"
    (List.exists (fun (_, last, _) -> last > 0xFFFF) groups)

let format_12_glyph groups c =
  match
    List.find_opt (fun (first, last, _) -> first <= c && c <= last) groups
  with
  | Some (first, _, g) when g + (c - first) < glyph_count -> g + (c - first)
  | _ -> 0

(* Nine characters at most at each end of each group. *)
let format_12_ranges groups =
  List.concat_map
    (fun (first, last, _) ->
      [ (first, Int.min last (first + 8)); (Int.max first (last - 8), last) ])
    groups

(* [maps f ranges expected] checks the glyph of every character around each of
   [ranges], and of U+FFFF, against [expected]. *)
let maps f ranges expected =
  List.iter
    (fun (first, last) ->
      for c = first - 1 to last + 1 do
        if Uchar.is_valid c then
          equal ~msg:(Printf.sprintf "U+%04X" c) int (expected c) (glyph f c)
      done)
    ((0xFFFF, 0xFFFF) :: ranges)

(* Subsetting *)

let eacute = glyph Font.regular 0xE9

let gen_subset =
  let open Gen in
  let* f =
    map Lazy.force
      (of_list
         [
           lazy Font.regular;
           lazy Font.bold;
           lazy (decoded cff_otf);
           lazy (decoded cff2_otf);
           lazy (decoded cff_2048_otf);
         ])
  in
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

let rec consecutive = function
  | a :: (b :: _ as rest) -> (a, b) :: consecutive rest
  | _ -> []

(* The tags of the tables of the OpenType file [data]. *)
let tags data =
  List.init (get_u16 data 4) (fun i -> String.sub data (12 + (16 * i)) 4)

let more_components = 0x20

(* The glyphs a subset of [f] to [gs] keeps, as [Font.subset] states them: glyph
   0, the glyphs [gs], and the components of every composite glyph it keeps,
   read from the [glyf] table of [f] as OpenType lays it out. *)
let kept_glyphs f gs =
  let data = Font.bytes f in
  let keep = Array.make (Font.glyph_count f) false in
  match Font.outlines f with
  | `Cff ->
      List.iter (fun g -> keep.(g) <- true) (0 :: gs);
      keep
  | `Truetype ->
      let long = get_u16 data (head_pos data + 50) = 1 in
      let loca = snd (table data "loca") and glyf = snd (table data "glyf") in
      let offset g =
        if long then get_u32 data (loca + (4 * g))
        else 2 * get_u16 data (loca + (2 * g))
      in
      (* A composite's records follow its ten bytes of header: flags, glyph, two
         arguments of a byte or a word each, and a transform of zero to four
         F2Dot14 numbers. *)
      let rec visit g =
        if not keep.(g) then begin
          keep.(g) <- true;
          let at = glyf + offset g in
          if offset (g + 1) > offset g && String.get_int16_be data at < 0 then
            components (at + 10)
        end
      and components at =
        let flags = get_u16 data at in
        visit (get_u16 data (at + 2));
        let args = if flags land words <> 0 then 4 else 2 in
        let transform =
          if flags land two_by_two <> 0 then 8
          else if flags land xy_scales <> 0 then 4
          else if flags land one_scale <> 0 then 2
          else 0
        in
        if flags land more_components <> 0 then
          components (at + 4 + args + transform)
      in
      List.iter visit (0 :: gs);
      keep

(* The tables font.mli says the subset of [f] holds. *)
let subset_tables f =
  let has tag = List.mem tag (tags (Font.bytes f)) in
  let outlines =
    match Font.outlines f with
    | `Truetype ->
        [ "glyf"; "loca" ] @ List.filter has [ "cvt "; "fpgm"; "gasp"; "prep" ]
    | `Cff -> [ "CFF " ]
  in
  List.sort String.compare
    ([ "cmap"; "head"; "hhea"; "hmtx"; "maxp"; "post" ]
    @ List.filter has [ "OS/2"; "name" ]
    @ outlines)

(* The face-wide maxima of a font's [head], [hhea] and [maxp] tables: its
   bounds; the largest advance, smallest side bearings and largest extent; and,
   for TrueType outlines, the largest points, contours, components and hinting
   resources a glyph takes. *)
let maxima ~cff data =
  let field (tag, at, len) = String.sub data (snd (table data tag) + at) len in
  List.map field
    ([ ("head", 36, 8); ("hhea", 10, 8) ]
    @ if cff then [] else [ ("maxp", 6, 26) ])

(* What the subset of [f] to [gs] keeps of [f], as [Font.subset] states it. *)
let subset_keeps (f, gs) =
  let s = Font.subset f gs in
  check_sfnt s;
  let f' = decoded s in
  let n = Font.glyph_count f' in
  let cff = Font.outlines f = `Cff in
  let keep = kept_glyphs f gs in
  let kept g = keep.(g) in
  cover "a composite" (List.mem eacute gs);
  cover "CFF outlines" cff;
  cover "a kerned pair kept"
    (List.exists (fun (g, g') -> Font.kerning f g g' <> 0.) (consecutive gs));
  let largest = ref 0 in
  Array.iteri (fun g k -> if k then largest := g) keep;
  equal ~msg:"glyph count" int (!largest + 1) n;
  equal ~msg:"tables" (list string) (subset_tables f) (tags s);
  equal ~msg:"face-wide maxima" (list string)
    (maxima ~cff (Font.bytes f))
    (maxima ~cff s);
  for g = 0 to n - 1 do
    let msg = Printf.sprintf "glyph %d" g in
    if kept g then begin
      equal ~msg path_t (Font.outline f g) (Font.outline f' g);
      equal ~msg float_exact (Font.advance f g) (Font.advance f' g)
    end
    else begin
      equal ~msg path_t Path.empty (Font.outline f' g);
      equal ~msg float_exact 0. (Font.advance f' g)
    end
  done;
  for c = 0 to 0x3000 do
    if Uchar.is_valid c then begin
      let g = glyph f c in
      let expected = if g < n && kept g then g else 0 in
      equal ~msg:(Printf.sprintf "U+%04X" c) int expected (glyph f' c)
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
    (fun (g, g') ->
      equal
        ~msg:(Printf.sprintf "kerning %d %d" g g')
        float_exact 0. (Font.kerning f' g g'))
    (consecutive gs);
  equal ~msg:"the glyphs reversed and repeated" string s
    (Font.subset f (List.rev_append gs gs))

let subsetting =
  group "subsetting"
    [
      prop "a subset keeps its glyphs, their characters and the face's metrics"
        ~examples:
          [
            (Font.regular, [ eacute ]);
            (Font.regular, [ glyph Font.regular 0x41; glyph Font.regular 0x56 ]);
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
      prop "a subset of every glyph maps characters as a format 4 map does"
        gen_segments (fun segments ->
          cover_segments segments;
          let f' = subset (format_4_font segments) all_glyphs in
          maps f' (format_4_ranges segments) (format_4_glyph segments));
      prop "a subset of every glyph maps characters as a format 12 map does"
        gen_groups (fun groups ->
          cover_groups groups;
          let f = tiny ~cmap:(cmap_12 groups) empty_glyphs in
          maps (subset f all_glyphs) (format_12_ranges groups)
            (format_12_glyph groups));
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
      test "a subset of more than 128 kB of outlines keeps them" (fun () ->
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
          let f =
            format_4_font
              [ (0x41, 0x43, 1 - 0x41, None); (0x42, 0x44, 3 - 0x42, None) ]
          in
          maps (subset f all_glyphs) [ (0x41, 0x44) ] (glyph f));
      test "a CFF subset's font matrix is the head table's em" (fun () ->
          let s = Font.subset (decoded cff_2048_otf) [ 1 ] in
          (* [0.00048828125 0 0 0.00048828125 0 0 FontMatrix], the reals in
             nibbles. *)
          let scale = "\030\010\000\004\136\040\018\095" in
          contains ~sub:(cat [ scale; "\139\139"; scale; "\139\139\012\007" ]) s);
      test "subset raises on a glyph out of range" (fun () ->
          let last = Font.glyph_count Font.regular - 1 in
          let raises_on g =
            raises_match
              (Exn.invalid_arg
                 ~sub:
                   (Printf.sprintf "Font.subset: glyph %d not in [0, %d]" g last))
              (fun () -> Font.subset Font.regular [ 0; g ])
          in
          raises_on (-1);
          raises_on (last + 1));
    ]

(* Metrics: the bundled faces' values read with fontTools. *)

let metric_cases =
  [
    ("ascent", Font.ascent, em 1984);
    ("descent", Font.descent, em 494);
    ("line_gap", Font.line_gap, 0.);
    ("cap_height", Font.cap_height, em 1490);
    ("x_height", Font.x_height, em 1118);
    ("italic_angle", Font.italic_angle, 0.);
  ]

(* Inter Regular with OS/2 version 1, version 2 and 4 with a cap height of 1400
   units, a cap height of 0, and no OS/2. *)
let cap_heights =
  let os2 = os2_pos regular_ttf in
  let cap = set_u16 regular_ttf (os2 + 88) 1400 in
  [
    ("OS/2 version 4", cap, em 1400);
    ("OS/2 version 2", set_u16 cap os2 2, em 1400);
    ("OS/2 version 1, the top of H", set_u16 cap os2 1, em 1490);
    ("a zero in OS/2, the top of H", set_u16 regular_ttf (os2 + 88) 0, em 1490);
    ("no OS/2, the top of H", rename regular_ttf "OS/2" "OS/X", em 1490);
  ]

(* Inter Regular with OS/2's typographic ascender, descender and line gap set to
   2000, -500 and 100 units, which differ from its hhea's 1984, -494 and 0:
   [vertical (typo, hhea, tiny)] are the rows that read [typo] from these under
   USE_TYPO_METRICS, [hhea] once the flag is cleared, and [tiny] from a tiny
   font without OS/2, whose hhea gives 800, -200 and 100 of 1000 units. *)
let vertical (typo, hhea, small) =
  let os2 = os2_pos regular_ttf in
  let set data (at, v) = set_u16 data (os2 + at) v in
  let data =
    List.fold_left set regular_ttf [ (68, 2000); (70, -500); (72, 100) ]
  in
  [
    ("OS/2 under USE_TYPO_METRICS", ((fun () -> decoded data), typo));
    ( "hhea when USE_TYPO_METRICS is clear",
      ((fun () -> decoded (set data (62, 0x40))), hhea) );
    ("hhea without OS/2", ((fun () -> tiny [ ""; simple [ square ] ]), small));
  ]

let metrics =
  group "metrics"
    [
      cases "of the bundled faces are fontTools'"
        ~id:(fun (n, _, _) -> n)
        metric_cases
        (fun (_, m, v) ->
          equal ~msg:"regular" float_exact v (m (regular ()));
          equal ~msg:"bold" float_exact v (m (bold ())));
      group "ascent"
        [
          cases "reads"
            ~id:(fun (n, _) -> n)
            (vertical (em 2000, em 1984, 0.8))
            (fun (_, (data, v)) -> equal float_exact v (Font.ascent (data ())));
        ];
      group "descent"
        [
          cases "reads"
            ~id:(fun (n, _) -> n)
            (vertical (em 500, em 494, 0.2))
            (fun (_, (data, v)) -> equal float_exact v (Font.descent (data ())));
        ];
      group "line_gap"
        [
          cases "reads"
            ~id:(fun (n, _) -> n)
            (vertical (em 100, 0., 0.1))
            (fun (_, (data, v)) ->
              equal float_exact v (Font.line_gap (data ())));
        ];
      group "cap_height"
        [
          cases "reads"
            ~id:(fun (n, _, _) -> n)
            cap_heights
            (fun (_, data, h) ->
              equal float_exact h (Font.cap_height (decoded data)));
          test "is the top of H in a font without OS/2" (fun () ->
              let f =
                tiny
                  ~cmap:(cmap_12 [ (0x48, 0x48, 1) ])
                  [ ""; simple [ square ] ]
              in
              equal float_exact 0.1 (Font.cap_height f));
          test "is the ascent without OS/2 and an H with ink" (fun () ->
              equal float_exact 0.8
                (Font.cap_height (tiny [ ""; simple [ square ] ])));
        ];
      group "x_height"
        [
          test "is OS/2's when positive" (fun () ->
              let data = set_u16 regular_ttf (os2_pos regular_ttf + 86) 1000 in
              equal float_exact (em 1000) (Font.x_height (decoded data)));
          test "is the top of x's ink without OS/2" (fun () ->
              equal float_exact (em 1118)
                (Font.x_height (decoded (rename regular_ttf "OS/2" "OS/X"))));
          test "is 0.5 without OS/2 and an x with ink" (fun () ->
              equal float_exact 0.5
                (Font.x_height (tiny [ ""; simple [ square ] ])));
        ];
      group "italic_angle"
        [
          test "is post's angle in radians" (fun () ->
              let post = snd (table regular_ttf "post") in
              let f = decoded (set_u32 regular_ttf (post + 4) (-12 * 65536)) in
              (* -12° is -π/15. Each side rounds twice, so they agree within 2
                 ulp, 4 ε relative. *)
              equal
                (float_rel ~rel:(4. *. epsilon_float) ~abs:0.)
                (-.(Float.pi /. 15.))
                (Font.italic_angle f));
          test "is 0 without post" (fun () ->
              equal float_exact 0.
                (Font.italic_angle (tiny [ ""; simple [ square ] ])));
        ];
      group "bounds"
        [
          test "is the head table's box, as fontTools reads it" (fun () ->
              equal ~msg:"regular" box_t
                (Box2.of_pts
                   (P2.v (em (-1513)) (em (-2269)))
                   (P2.v (em 5290) (em 660)))
                (Font.bounds (regular ()));
              equal ~msg:"bold" box_t
                (Box2.of_pts
                   (P2.v (em (-1613)) (em (-2279)))
                   (P2.v (em 5290) (em 685)))
                (Font.bounds (bold ())));
        ];
    ]

(* Glyphs *)

let glyph_of_a_in cmap = glyph (tiny ~cmap empty_glyphs) 0x41
let a_to g = format_4 [ (0x41, 0x41, g - 0x41, None) ]

(* Character maps of A, and the glyph font.mli says A gets. *)
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

(* Every mapping of the character map, as gen/gen.py lists fontTools'. *)
let cmap_listing f =
  let b = Buffer.create 8192 in
  for c = 0 to 0x10FFFF do
    if Uchar.is_valid c then
      let g = glyph f c in
      if g <> 0 then Printf.bprintf b "U+%04X %d\n" c g
  done;
  Buffer.contents b

let character_map =
  group "glyph"
    [
      prop "reads format 4 segments as OpenType defines them" gen_segments
        (fun segments ->
          cover_segments segments;
          maps (format_4_font segments) (format_4_ranges segments)
            (format_4_glyph segments));
      prop "reads format 12 groups as OpenType defines them" gen_groups
        (fun groups ->
          cover_groups groups;
          maps
            (tiny ~cmap:(cmap_12 groups) empty_glyphs)
            (format_12_ranges groups) (format_12_glyph groups));
      cases "prefers the character map of"
        ~id:(fun (n, _, _) -> n)
        selection
        (fun (_, subtables, g) ->
          equal int g (glyph_of_a_in (cmap_table subtables)));
      test "is 0 for a glyph the font does not have" (fun () ->
          equal int 0
            (glyph (tiny ~cmap:(cmap_12 [ (0x41, 0x41, 2) ]) [ ""; "" ]) 0x41));
      test "of Inter Regular is fontTools'" (fun () ->
          equal text (read "golden/cmap.expected") (cmap_listing (regular ())));
    ]

let advance =
  group "advance"
    [
      cases "of H is hmtx's, as fontTools reads it, in"
        ~id:(fun (n, _, _) -> n)
        [ ("regular", regular_ttf, 1522); ("bold", bold_ttf, 1530) ]
        (fun (_, data, units) ->
          let f = decoded data in
          equal float_exact (em units) (Font.advance f (glyph f 0x48)));
      test "of a glyph past hhea's count of metrics is the last advance"
        (fun () ->
          equal float_exact 0.5
            (Font.advance (tiny [ ""; simple [ square ] ]) 1));
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

(* Whether a kern subtable of [coverage] kerns: horizontal (bit 0), of kerning
   values (bit 1 clear) along the text (bit 2 clear), of format 0 (bits 8 to
   15). *)
let kerns coverage =
  coverage land 1 = 1 && coverage land 6 = 0 && coverage lsr 8 = 0

(* The kerning of [g] and [g'] from kern [subtables], as OpenType's kern table
   defines coverage bit 3: "If this bit is set to 1 the value in this table
   should replace the value currently being accumulated." *)
let kern_table_value subtables g g' =
  List.fold_left
    (fun acc (c, pairs) ->
      match List.find_opt (fun (a, b, _) -> (a, b) = (g, g')) pairs with
      | Some (_, _, v) when kerns c -> if c land 8 <> 0 then v else acc + v
      | _ -> acc)
    0 subtables

let kern_table_law subtables =
  let sum g g' =
    List.fold_left
      (fun acc (c, pairs) ->
        if kerns c then acc + pair_value pairs g g' else acc)
      0 subtables
  in
  cover "a subtable left out"
    (List.exists (fun (c, _) -> not (kerns c)) subtables);
  cover "an override replacing a value"
    (List.exists
       (fun (g, g') -> kern_table_value subtables g g' <> sum g g')
       (List.init 64 (fun i -> (i / 8, i mod 8))));
  kerns_as
    (kerning_font [ ("kern", kern_table subtables) ])
    (kern_table_value subtables)

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

(* Every kerned pair of printable ASCII, as gen/gen.py lists HarfBuzz's. *)
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
      prop "reads GPOS pair adjustments of format 1" gen_format_1
        (fun (pairs, (ranges, placements, second), st) ->
          cover "a coverage of ranges" ranges;
          cover "placements before the advance" placements;
          cover "second value records" second;
          kerns_as (kerning_font [ ("GPOS", one_lookup st) ]) (pair_value pairs));
      prop "reads GPOS pair adjustments of format 2" gen_format_2
        (fun (_, _, _, _, _, expected, st) ->
          kerns_as (kerning_font [ ("GPOS", one_lookup st) ]) expected);
      prop "without GPOS, accumulates the horizontal format 0 kern subtables"
        ~examples:[ [ (0x1, [ (0, 0, 0); (2, 2, 1) ]); (0x9, [ (2, 2, 0) ]) ] ]
        gen_kern kern_table_law;
      cases "takes"
        ~id:(fun (n, _, _, _) -> n)
        lookup_rules
        (fun (_, features, lookups, expected) ->
          let f = kerning_font [ ("GPOS", gpos_table features lookups) ] in
          List.iter
            (fun (g, g', k) ->
              equal ~msg:(Printf.sprintf "%d %d" g g') int k (units f g g'))
            expected);
      cases "reads GPOS beside a kern table when"
        ~id:(fun (n, _, _) -> n)
        kern_fallback
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
      test "is 0 from a kern table of version 1" (fun () ->
          let table = set_u16 (kern_table [ (0x1, [ (1, 2, -100) ]) ]) 0 1 in
          equal int 0 (units (kerning_font [ ("kern", table) ]) 1 2));
      test "is 0 without GPOS and kern tables" (fun () ->
          equal int 0 (units (kerning_font []) 1 2));
      (* (pair, kerning) in font units, read with HarfBuzz as gen/gen.py reads
         Inter Regular's. *)
      cases "of Inter Bold is HarfBuzz's for" ~id:fst
        [ ("AV", -162); ("To", -160); ("LT", -186); ("Yo", -224); ("HH", 0) ]
        (fun (s, k) -> equal float_exact (em k) (kern_of (bold ()) s));
      test "of Inter Regular's printable ASCII is HarfBuzz's" (fun () ->
          equal text
            (read "golden/kerning.expected")
            (kerning_listing (regular ())));
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

(* Paths of the same commands whose coordinates differ by at most 1e-12. A
   coordinate is a few products and sums of F2Dot14 numbers, exact in binary,
   and integers below 2¹⁷, divided by 1000: it is below 140 in magnitude, where
   each of the few roundings either side makes costs at most 2⁻⁴⁶, about
   1.4e-14. *)
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
  equal path_t expected (Font.outline f 1)

let starts_on_curve () =
  let f = tiny [ ""; simple [ [ off 0 0; on 100 0; on 100 100 ] ] ] in
  let expected =
    Path.empty
    |> Path.move_to (pt 100 0)
    |> Path.line_to (pt 100 100)
    |> Path.quad_to (pt 0 0) (pt 100 0)
    |> Path.close
  in
  equal path_t expected (Font.outline f 1)

let one_point = [ ""; simple [ [ on 5 5 ]; [ on 7 7 ] ] ]

let outline =
  group "outline"
    [
      prop "of a composite is its components placed as OpenType defines"
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
      test "starts a contour without on-curve points between its first two"
        all_off_curve;
      test "starts a contour at its first on-curve point" starts_on_curve;
      test "reads flags repeated up to the last point" (fun () ->
          equal path_t square_path (Font.outline (tiny [ ""; repeated 3 ]) 1));
      test "skips instructions" (fun () ->
          let f = tiny [ ""; simple ~instructions:"\x01\x02\x03" [ square ] ] in
          equal path_t square_path (Font.outline f 1));
      test "draws nothing for a contour of one point" (fun () ->
          equal path_t Path.empty (Font.outline (tiny one_point) 1));
      test "draws a line for a contour of two points" (fun () ->
          let f = tiny [ ""; simple [ [ on 0 0; on 100 0 ] ] ] in
          let expected =
            Path.empty
            |> Path.move_to (pt 0 0)
            |> Path.line_to (pt 100 0)
            |> Path.close
          in
          equal path_t expected (Font.outline f 1));
      test "of H in Inter Regular is its contour of lines" (fun () ->
          expect
            (Format.asprintf "%a" Path.pp
               (Font.outline (regular ()) (glyph Font.regular 0x48)))
          @@ __POS_OF__
               {|
            M 0.0878906 0 L 0.0878906 -0.727539 L 0.180664 -0.727539 L 0.180664 -0.413086
            L 0.5625 -0.413086 L 0.5625 -0.727539 L 0.655273 -0.727539 L 0.655273 0
            L 0.5625 0 L 0.5625 -0.331055 L 0.180664 -0.331055 L 0.180664 0 Z
            |});
      cases "is empty for a glyph without ink," ~id:fst
        [ (".notdef", 0); ("space", glyph Font.regular 0x20) ]
        (fun (_, g) -> equal path_t Path.empty (Font.outline (regular ()) g));
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

let ink =
  group "ink"
    [
      cases "is fontTools' box of"
        ~id:(fun (n, _, _) -> n)
        inks
        (fun (_, c, (x0, y0, x1, y1)) ->
          let f = regular () in
          let box =
            Box2.of_pts
              (P2.v (em x0) (0. -. em y1))
              (P2.v (em x1) (0. -. em y0))
          in
          equal (option box_t) (Some box) (Font.ink f (glyph f c)));
      cases "is the bounds of what the outline draws, for"
        ~id:(fun (n, _, _) -> n)
        [
          ( "a contour of two points",
            simple [ [ on 0 0; on 100 0 ] ],
            Some (Box2.v 0. 0. 0.1 0.) );
          ("a glyph of no contour", simple [], None);
          ("a contour of one point", simple [ [ on 5 5 ] ], None);
        ]
        (fun (_, g, box) ->
          equal (option box_t) box (Font.ink (tiny [ ""; g ]) 1));
      cases "is None for a glyph without ink," ~id:fst
        [ (".notdef", 0); ("space", glyph Font.regular 0x20) ]
        (fun (_, g) -> is_none ~pp:pp_box (Font.ink (regular ()) g));
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

let glyphs =
  group "glyphs"
    [
      cases "raise on a glyph out of range in"
        ~id:(fun (n, _, _) -> n)
        glyph_range
        (fun (_, fn, f) ->
          let last = Font.glyph_count Font.regular - 1 in
          let msg g = Printf.sprintf "%s: glyph %d not in [0, %d]" fn g last in
          raises_match
            (Exn.invalid_arg ~sub:(msg (-1)))
            (fun () -> f Font.regular (-1));
          raises_match
            (Exn.invalid_arg ~sub:(msg (last + 1)))
            (fun () -> f Font.regular (last + 1)));
      group "glyph_count"
        [
          cases "is maxp's, as fontTools reads it, for" ~id:fst
            [ ("regular", regular_ttf); ("bold", bold_ttf) ]
            (fun (_, data) -> equal int 730 (Font.glyph_count (decoded data)));
        ];
      character_map;
      advance;
      kerning;
      outline;
      ink;
    ]

let () =
  exit
    (run "hugin.font"
       [ bundled; loading; identity; subsetting; metrics; glyphs ])
