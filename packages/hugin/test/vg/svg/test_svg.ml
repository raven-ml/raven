(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_gg
open Hugin_font
open Hugin_vg

let svg ?(w = 100.) ?(h = 100.) p = Hugin_vg_svg.render (Renderable.v w h p)

(* Reading documents *)

let find_from s i sub =
  let n = String.length sub in
  let rec loop i =
    if i + n > String.length s then None
    else if String.sub s i n = sub then Some i
    else loop (i + 1)
  in
  loop i

(* [count s sub] is the number of occurrences of [sub] in [s]. *)
let count s sub =
  let rec loop i k =
    match find_from s i sub with None -> k | Some j -> loop (j + 1) (k + 1)
  in
  loop 0 0

(* [attrs name s] is the values of the attributes [name] in [s], in order. *)
let attrs name s =
  let key = " " ^ name ^ "=\"" in
  let rec loop i acc =
    match find_from s i key with
    | None -> List.rev acc
    | Some j ->
        let start = j + String.length key in
        let stop = String.index_from s start '"' in
        loop stop (String.sub s start (stop - start) :: acc)
  in
  loop 0 []

let attr name s =
  match attrs name s with
  | v :: _ -> v
  | [] -> failf "no attribute %s in:\n%s" name s

(* [attr_or initial name s] is the first attribute [name] of [s], or [initial],
   its initial value in SVG 2, if [s] has none. *)
let attr_or initial name s =
  match attrs name s with v :: _ -> v | [] -> initial

(* [elements tag s] is the start tags of the elements [tag] of [s], in order. *)
let elements tag s =
  let rec loop i acc =
    match find_from s i ("<" ^ tag ^ " ") with
    | None -> List.rev acc
    | Some i ->
        let stop = String.index_from s i '>' in
        loop stop (String.sub s i (stop - i + 1) :: acc)
  in
  loop 0 []

let element tag s =
  match elements tag s with
  | e :: _ -> e
  | [] -> failf "no element %s in:\n%s" tag s

(* [numbers d] is the numbers of the path data [d], in order. *)
let numbers d =
  let b = Buffer.create 16 and acc = ref [] in
  let flush () =
    if Buffer.length b > 0 then begin
      acc := float_of_string (Buffer.contents b) :: !acc;
      Buffer.clear b
    end
  in
  String.iter
    (fun c ->
      match c with
      | '0' .. '9' | '.' | 'e' -> Buffer.add_char b c
      | '-' ->
          flush ();
          Buffer.add_char b c
      | _ -> flush ())
    d;
  flush ();
  List.rev !acc

(* [points d] is the points of the path data [d] of moves and lines. *)
let points d =
  let rec pairs = function x :: y :: rest -> (x, y) :: pairs rest | _ -> [] in
  pairs (numbers d)

(* [path_of_data d] is the path of the SVG path data [d] of absolute moves,
   lines, cubics and closes. *)
let path_of_data d =
  let tokens = ref [] and b = Buffer.create 16 in
  let flush () =
    if Buffer.length b > 0 then begin
      tokens := `Num (float_of_string (Buffer.contents b)) :: !tokens;
      Buffer.clear b
    end
  in
  String.iter
    (fun c ->
      match c with
      | '0' .. '9' | '.' | 'e' -> Buffer.add_char b c
      | '-' ->
          flush ();
          Buffer.add_char b c
      | 'M' | 'L' | 'C' | 'Z' ->
          flush ();
          tokens := `Cmd c :: !tokens
      | _ -> flush ())
    d;
  flush ();
  let rec go q = function
    | `Cmd 'M' :: `Num x :: `Num y :: rest ->
        go (Path.move_to (P2.v x y) q) rest
    | `Cmd 'L' :: `Num x :: `Num y :: rest ->
        go (Path.line_to (P2.v x y) q) rest
    | `Cmd 'C'
      :: `Num a
      :: `Num b
      :: `Num c
      :: `Num d
      :: `Num x
      :: `Num y
      :: rest ->
        go (Path.cubic_to (P2.v a b) (P2.v c d) (P2.v x y) q) rest
    | `Cmd 'Z' :: rest -> go (Path.close q) rest
    | [] -> q
    | _ -> invalid_arg ("path_of_data: " ^ d)
  in
  go Path.empty (List.rev !tokens)

(* [transform_of v] is the map of the SVG transform list [v] of [translate],
   [scale] and [matrix] functions. *)
let transform_of v =
  let one f =
    let i = String.index f '(' in
    let args =
      String.sub f (i + 1) (String.length f - i - 1)
      |> String.map (function ',' -> ' ' | c -> c)
      |> String.split_on_char ' '
      |> List.filter (( <> ) "")
      |> List.map float_of_string
    in
    match (String.trim (String.sub f 0 i), args) with
    | "translate", [ x ] -> Affine.translate x 0.
    | "translate", [ x; y ] -> Affine.translate x y
    | "scale", [ k ] -> Affine.scale k k
    | "scale", [ sx; sy ] -> Affine.scale sx sy
    | "matrix", [ xx; yx; xy; yy; x0; y0 ] -> { Affine.xx; yx; xy; yy; x0; y0 }
    | _ -> failf "transform function %S" f
  in
  String.split_on_char ')' v
  |> List.filter (fun f -> String.trim f <> "")
  |> List.fold_left (fun m f -> Affine.(m * one f)) Affine.id

(* [matrix e] is the map of the [transform] of the start tag [e]. *)
let matrix e = transform_of (attr "transform" e)

(* [placement u] is the map by which the use element [u] places its definition:
   its [transform], then its [x] and [y]. *)
let placement u =
  let x = float_of_string (attr_or "0" "x" u)
  and y = float_of_string (attr_or "0" "y" u) in
  Affine.(transform_of (attr_or "" "transform" u) * translate x y)

let map m (x, y) =
  let p = P2.transform m (P2.v x y) in
  (P2.x p, P2.y p)

(* [reach m w (dx, dy)] is how far a pen of width [w] under the linear part of
   [m] reaches along the unit direction [(dx, dy)]: [w /. 2.] times the length
   of the transpose of [m] applied to it. *)
let reach (m : Affine.t) w (dx, dy) =
  w /. 2.
  *. Float.hypot ((m.xx *. dx) +. (m.yx *. dy)) ((m.xy *. dx) +. (m.yy *. dy))

(* The reaches along four directions fix an elliptic pen. *)
let directions =
  let h = Float.sqrt 0.5 in
  [ (1., 0.); (0., 1.); (h, h); (h, -.h) ]

(* The accuracy the document keeps: 0.001 point horizontally and vertically. *)
let point_within tol =
  Testable.make
    ~pp:(fun ppf (x, y) -> Format.fprintf ppf "(%.17g, %.17g)" x y)
    ~equal:(fun (x, y) (x', y') ->
      Float.abs (x -. x') <= tol && Float.abs (y -. y') <= tol)

let point = point_within 0.001

(* Unmapped coordinates written with three decimals: rounded by at most 0.0005,
   and their difference from the picture's, below 200 in magnitude, computed
   within 1e-12. *)
let decimals = point_within (0.0005 +. 1e-12)

let of_base64 s =
  let value c =
    match c with
    | 'A' .. 'Z' -> Char.code c - 65
    | 'a' .. 'z' -> Char.code c - 71
    | '0' .. '9' -> Char.code c + 4
    | '+' -> 62
    | '/' -> 63
    | _ -> invalid_arg "of_base64"
  in
  let b = Buffer.create (String.length s * 3 / 4) in
  let acc = ref 0 and bits = ref 0 in
  String.iter
    (fun c ->
      if c <> '=' then begin
        acc := (!acc lsl 6) lor value c;
        bits := !bits + 6;
        if !bits >= 8 then begin
          bits := !bits - 8;
          Buffer.add_char b (Char.chr ((!acc lsr !bits) land 0xff))
        end
      end)
    s;
  Buffer.contents b

(* [png_of image] is the pixels of the PNG data URI of the [image] element. *)
let png_of image =
  let h = attr "href" image in
  starts_with ~prefix:"data:image/png;base64," h;
  Vg_corpus.load_png (of_base64 (String.sub h 22 (String.length h - 22)))

(* [fonts s] is the family, media type and bytes of each [@font-face] rule of
   [s]. *)
let fonts s =
  let field i key stop =
    let start = Option.get (find_from s i key) + String.length key in
    let j = String.index_from s start stop in
    (String.sub s start (j - start), j)
  in
  let rec loop i acc =
    match find_from s i "@font-face{" with
    | None -> List.rev acc
    | Some i ->
        let family, i = field i "font-family:" ';' in
        let media, i = field i "src:url(data:" ';' in
        let data, i = field i "base64," ')' in
        loop i ((family, media, of_base64 data) :: acc)
  in
  loop 0 []

(* [masked s] is [s] with the data of its fonts replaced by their length, which
   leaves the goldens readable. *)
let masked s =
  let key = "base64," and b = Buffer.create (String.length s) in
  let rec loop i =
    match find_from s i "@font-face{" with
    | None -> Buffer.add_string b (String.sub s i (String.length s - i))
    | Some j ->
        let start = Option.get (find_from s j key) + String.length key in
        let stop = String.index_from s start ')' in
        Buffer.add_string b (String.sub s i (start - i));
        Printf.bprintf b "<%d bytes>" (stop - start);
        loop stop
  in
  loop 0;
  Buffer.contents b

(* Pictures *)

let red = Color.red
let rect x y w h = Path.rect (Box2.v x y w h)
let square = rect 10. 20. 30. 40.
let glyph c = Font.glyph Font.regular (Uchar.of_char c)
let disc = Path.circle (P2.v 0. 0.) 2.

let polyline pts =
  Path.polyline
    (Array.of_list (List.map fst pts))
    (Array.of_list (List.map snd pts))

let stroked ?(cap = `Butt) ?(join = `Miter) ?(miter_limit = 4.) ?dash
    ?dash_offset w =
  svg
    (Picture.stroke
       (Stroke.v ~cap ~join ~miter_limit ?dash ?dash_offset w)
       red
       (Path.polyline [| 10.; 50.; 50. |] [| 10.; 10.; 50. |]))

let zero_length =
  Path.empty |> Path.move_to (P2.v 5. 5.) |> Path.line_to (P2.v 5. 5.)

let marker =
  Picture.group
    [
      Picture.fill Color.black disc;
      Picture.stroke (Stroke.v 0.5) Color.white disc;
    ]

(* Strokes of differing widths, which a scaling stamp writes in full. *)
let two_pens =
  Picture.group
    [
      Picture.stroke (Stroke.v 1.) red disc;
      Picture.stroke (Stroke.v 2.) red disc;
    ]

let cff_small () =
  Result.get_ok
    (Font.of_string
       (In_channel.with_open_bin "../../font/golden/cff-small.otf"
          In_channel.input_all))

let runs_ab () =
  let a = Picture.glyphs red (P2.v 5. 20.) (Vg_corpus.typeset "a" 10.)
  and b = Picture.glyphs red (P2.v 5. 40.) (Vg_corpus.typeset "b" 10.) in
  svg (Picture.group [ a; b ])

(* [unmapped text] is a run of [text] whose one glyph, a, renders none of its
   characters. *)
let unmapped text =
  Run.v ~clusters:[| 0 |] ~font:Font.regular ~size:10. ~text
    ~glyphs:[| glyph 'a' |]
    ~xs:[| 0. |] ()

(* Pictures as SVG *)

let typeset_ab () =
  Picture.glyphs red (P2.v 5. 50.) (Vg_corpus.typeset "AV" 10.)

let scaled_uses () =
  let ring = Picture.stroke (Stroke.v ~dash:[ 2.; 1. ] 1.) red disc in
  let s =
    svg (Picture.stamp ~scales:[| 2.; 4. |] [| 10.; 20. |] [| 5.; 5. |] ring)
  in
  (* Where each use puts the definition's origin and the point a unit right of
     it, and its pen on the page: widths to 0.001 and dash lengths to a
     thousandth of that, as the accuracy section states. *)
  let on_page u =
    let m = placement u in
    let k = Affine.stretch m in
    let page v = k *. float_of_string v in
    ( [ map m (0., 0.); map m (1., 0.) ],
      ( page (attr "stroke-width" u),
        List.map page (String.split_on_char ' ' (attr "stroke-dasharray" u)) )
    )
  in
  equal
    (list (pair (list point) (pair (float 0.001) (list (float 1e-6)))))
    [
      ([ (10., 5.); (12., 5.) ], (1., [ 2.; 1. ]));
      ([ (20., 5.); (24., 5.) ], (1., [ 2.; 1. ]));
    ]
    (List.map on_page (elements "use" s))

let nested_scales () =
  let inner =
    Picture.stamp ~scales:[| 2. |] [| 0. |] [| 0. |] (Picture.fill red disc)
  in
  let s = svg (Picture.stamp ~scales:[| 3. |] [| 10. |] [| 5. |] inner) in
  (* The disc of radius 2, scaled by 2 times 3, has radius 12, at (10, 5). The
     ends of its cubics lie on it. *)
  let m = placement (element "use" s) in
  let ends = List.filteri (fun i _ -> i mod 3 = 0) (points (attr "d" s)) in
  List.iter
    (fun p ->
      let x, y = map m p in
      equal (float 0.001) 12. (Float.hypot (x -. 10.) (y -. 5.)))
    ends

let mapping =
  group "Pictures as SVG"
    ([
       test "a document is w by h points with a point per user unit" (fun () ->
           expect (svg ~w:360.5 ~h:240. Picture.empty)
           @@ __POS_OF__
                {|
            <?xml version="1.0" encoding="UTF-8"?>
            <svg xmlns="http://www.w3.org/2000/svg" width="360.5pt" height="240pt" viewBox="0 0 360.5 240">
            </svg>
          |});
       test "a fill is a path" (fun () ->
           expect (svg (Picture.fill red square))
           @@ __POS_OF__
                {|
            <?xml version="1.0" encoding="UTF-8"?>
            <svg xmlns="http://www.w3.org/2000/svg" width="100pt" height="100pt" viewBox="0 0 100 100">
            <path d="M10 20L40 20L40 60L10 60Z" fill="#ff0000"/>
            </svg>
          |});
       test "a colour is written #rrggbb" (fun () ->
           (* 0.2, 0.4 and 1 times 255 are 0x33, 0x66 and 0xff. *)
           equal string "#3366ff"
             (attr "fill" (svg (Picture.fill (Color.v 0.2 0.4 1.) square))));
       cases "an alpha below 1 is written as" ~id:fst
         [
           ("fill-opacity", Picture.fill (Color.with_alpha 0.25 red) square);
           ( "stroke-opacity",
             Picture.stroke (Stroke.v 1.) (Color.with_alpha 0.25 red) square );
         ]
         (fun (name, p) -> equal string "0.25" (attr name (svg p)));
       test "an opaque colour writes no opacity" (fun () ->
           not_contains ~sub:"opacity" (svg (Picture.fill red square)));
       test "an even-odd fill writes its rule" (fun () ->
           equal string "evenodd"
             (attr "fill-rule" (svg (Picture.fill ~rule:`Even_odd red square))));
       (* SVG 2's initial values are stroke-linecap butt, stroke-linejoin miter
          and stroke-miterlimit 4. *)
       cases "a stroke writes its pen, read with SVG's initial values" ~id:fst
         [
           ( "butt caps and miter joins",
             ((`Butt, `Miter, 4., 2.), [ "2"; "butt"; "miter"; "4" ]) );
           ( "square caps and round joins",
             ((`Square, `Round, 4., 2.), [ "2"; "square"; "round"; "4" ]) );
           ( "a miter limit of 2.5",
             ((`Butt, `Miter, 2.5, 3.), [ "3"; "butt"; "miter"; "2.5" ]) );
         ]
         (fun (_, ((cap, join, miter_limit, w), pen)) ->
           let s = stroked ~cap ~join ~miter_limit w in
           equal (list string)
             ([ "none"; "#ff0000" ] @ pen)
             [
               attr "fill" s;
               attr "stroke" s;
               attr "stroke-width" s;
               attr_or "butt" "stroke-linecap" s;
               attr_or "miter" "stroke-linejoin" s;
               attr_or "4" "stroke-miterlimit" s;
             ]);
       test "a dashed stroke writes its pattern and offset" (fun () ->
           let s = stroked ~dash:[ 4.; 2. ] ~dash_offset:1. 2. in
           equal (pair string string) ("4 2", "1")
             (attr "stroke-dasharray" s, attr "stroke-dashoffset" s));
       (* [1 2 3] dashes as [1 2 3 1 2 3], of period 12. *)
       cases
         "a dash offset keeps its phase in an odd pattern, which repeats twice"
         ~id:(Printf.sprintf "offset %g") [ 0.5; 7.; 13. ] (fun o ->
           let s = stroked ~dash:[ 1.; 2.; 3. ] ~dash_offset:o 2. in
           equal (float 1e-9) (Float.rem o 12.)
             (Float.rem (float_of_string (attr "stroke-dashoffset" s)) 12.));
       test "a subpath of zero length is left out with square caps" (fun () ->
           not_contains ~sub:"<path"
             (svg (Picture.stroke (Stroke.v ~cap:`Square 2.) red zero_length)));
       test "a subpath of zero length is kept with round caps" (fun () ->
           let round =
             Picture.stroke (Stroke.v ~cap:`Round 2.) red zero_length
           in
           equal string "M5 5L5 5" (attr "d" (svg round)));
       test "a subpath that repeats its start, then moves, is stroked"
         (fun () ->
           let q = Path.polyline [| 5.; 5.; 6. |] [| 5.; 5.; 6. |] in
           let s = svg (Picture.stroke (Stroke.v ~cap:`Square 2.) red q) in
           equal string "M5 5L5 5L6 6" (attr "d" s));
       test "a subpath of a start alone is a dot with round caps" (fun () ->
           let lone = Path.empty |> Path.move_to (P2.v 5. 6.) in
           let s = svg (Picture.stroke (Stroke.v 2.) red lone) in
           equal string "M5 6L5 6" (attr "d" s));
       test "a run is a text element holding its text" (fun () ->
           in_order ~subs:[ "<text "; ">AV</text>" ] (svg (typeset_ab ())));
       test "a run places each character at its glyph by x and y lists"
         (fun () ->
           let r = Vg_corpus.typeset "AV" 10. in
           let s = element "text" (svg (typeset_ab ())) in
           equal (pair string string)
             ( Printf.sprintf "5 %g"
                 (Float.round ((5. +. Run.x r 1) *. 1000.) /. 1000.),
               "50" )
             (attr "x" s, attr "y" s));
       test "a run off one baseline gives each glyph its y" (fun () ->
           let r =
             Run.v ~ys:[| 0.; 2.5 |] ~font:Font.regular ~size:10. ~text:"ab"
               ~glyphs:[| glyph 'a'; glyph 'b' |]
               ~xs:[| 0.; 6. |] ()
           in
           let s = element "text" (svg (Picture.glyphs red (P2.v 5. 50.) r)) in
           equal (pair string string) ("5 11", "50 52.5")
             (attr "x" s, attr "y" s));
       test "a run is at its size" (fun () ->
           let r = Vg_corpus.typeset "AV" 12.5 in
           equal string "12.5"
             (attr "font-size" (svg (Picture.glyphs red (P2.v 5. 50.) r))));
       test "a run turns off ligatures, kerning and other shaping" (fun () ->
           in_order
             ~subs:
               [
                 "font-variant-ligatures:none";
                 "font-kerning:none";
                 "font-feature-settings:'liga' 0";
               ]
             (attr "style" (svg (typeset_ab ()))));
       test "a font is embedded once for all its runs" (fun () ->
           equal int 1 (List.length (fonts (runs_ab ()))));
       test "an embedded font is the subset of the glyphs its runs write"
         (fun () ->
           equal (list string)
             [ Font.subset Font.regular [ glyph 'a'; glyph 'b' ] ]
             (List.map (fun (_, _, bytes) -> bytes) (fonts (runs_ab ()))));
       test "runs name the family of their font's rule" (fun () ->
           let s = runs_ab () in
           let family = List.map (fun (f, _, _) -> f) (fonts s) in
           equal (list string) (family @ family) (attrs "font-family" s));
       cases "a font's data URI is" ~id:fst
         [
           ( "font/ttf for TrueType outlines",
             (Fun.const Font.regular, "font/ttf") );
           ("font/otf for CFF outlines", (cff_small, "font/otf"));
         ]
         (fun (_, (font, media)) ->
           let font = font () in
           let s =
             svg
               (Picture.glyphs red (P2.v 5. 20.)
                  (Vg_corpus.typeset ~font "H" 10.))
           in
           equal (list string) [ media ]
             (List.map (fun (_, m, _) -> m) (fonts s)));
       test "a run with a glyph the cmap does not give is labelled outlines"
         (fun () ->
           let r =
             Run.v ~font:Font.regular ~size:10. ~text:"a"
               ~glyphs:[| glyph 'b' |]
               ~xs:[| 0. |] ()
           in
           let s = svg (Picture.glyphs red (P2.v 5. 50.) r) in
           not_contains ~sub:"<text" s;
           in_order ~subs:[ "<g aria-label=\"a\""; "<path" ] s);
       cases "a run is outlines when a glyph renders" ~id:fst
         [
           ("two characters", "fi");
           ("a character beyond the Basic Multilingual Plane", "\u{1D465}");
           ("a tab", "\t");
           ("a character the font lacks", "\u{4E2D}");
         ]
         (fun (_, text) ->
           let u = String.get_utf_8_uchar text 0 |> Uchar.utf_decode_uchar in
           let r =
             Run.v ~clusters:[| 0 |] ~font:Font.regular ~size:10. ~text
               ~glyphs:[| Font.glyph Font.regular u |]
               ~xs:[| 0. |] ()
           in
           let s = svg (Picture.glyphs red (P2.v 5. 50.) r) in
           not_contains ~sub:"<text" s;
           contains ~sub:"aria-label=" s);
       test "a run of glyphs without text is outlines" (fun () ->
           let s = svg (Picture.glyphs red (P2.v 5. 50.) (unmapped "")) in
           not_contains ~sub:"<text" s;
           in_order ~subs:[ "<g aria-label=\"\""; "<path" ] s);
       test "a run's label holds what XML can hold, U+FFFD the rest" (fun () ->
           let text = "<&\u{1} \t\u{FFFD}\u{FFFE}\u{FFFF}\u{10000}" in
           equal string
             "&lt;&amp;\u{FFFD} &#9;\u{FFFD}\u{FFFD}\u{FFFD}\u{10000}"
             (attr "aria-label"
                (svg (Picture.glyphs red (P2.v 5. 50.) (unmapped text)))));
       test "an image is a PNG data URI of its pixels" (fun () ->
           let px =
             Nx.init Nx.uint8 [| 2; 3; 3 |] (fun i ->
                 (i.(0) * 100) + (i.(1) * 50) + (i.(2) * 9))
           in
           let s =
             element "image" (svg (Picture.image (Box2.v 10. 20. 30. 40.) px))
           in
           equal (array int) (Nx.to_array px) (Nx.to_array (png_of s)));
       test "an image is stretched over its box" (fun () ->
           let px = Nx.zeros Nx.uint8 [| 2; 3; 3 |] in
           let s =
             element "image" (svg (Picture.image (Box2.v 10. 20. 30. 40.) px))
           in
           equal (list string)
             [ "10"; "20"; "30"; "40"; "none" ]
             [
               attr "x" s;
               attr "y" s;
               attr "width" s;
               attr "height" s;
               attr "preserveAspectRatio" s;
             ]);
       test "an image is drawn pixelated" (fun () ->
           let px = Nx.zeros Nx.uint8 [| 2; 3; 3 |] in
           let s =
             element "image" (svg (Picture.image (Box2.v 10. 20. 30. 40.) px))
           in
           equal string "image-rendering:pixelated" (attr "style" s));
       test "a clip is a clipPath that the group of its picture refers to"
         (fun () ->
           let s =
             svg (Picture.clip square (Picture.fill red (rect 0. 0. 100. 100.)))
           in
           let id = attr "id" (element "clipPath" s) in
           equal string ("url(#" ^ id ^ ")") (attr "clip-path" (element "g" s)));
       test "a clip writes its rule" (fun () ->
           let s =
             svg
               (Picture.clip ~rule:`Even_odd square
                  (Picture.fill red (rect 0. 0. 100. 100.)))
           in
           equal string "evenodd" (attr "clip-rule" s));
       test "a transform writes no element of its own" (fun () ->
           not_contains ~sub:"<g"
             (svg
                (Picture.transform (Affine.translate 5. 7.)
                   (Picture.fill red square))));
       test "an opacity is a group with an opacity" (fun () ->
           let s = svg (Picture.opacity 0.5 (Picture.fill red square)) in
           equal string "0.5" (attr "opacity" (element "g" s)));
       test "a stamp writes its picture once, as a definition" (fun () ->
           let s =
             svg (Picture.stamp [| 10.; 20.; 30. |] [| 5.; 6.; 7. |] marker)
           in
           equal int 1 (count s "<g id="));
       test "a stamp draws a use per instance, at its position" (fun () ->
           let s =
             svg (Picture.stamp [| 10.; 20.; 30. |] [| 5.; 6.; 7. |] marker)
           in
           equal (list point)
             [ (10., 5.); (20., 6.); (30., 7.) ]
             (List.map (fun u -> map (placement u) (0., 0.)) (elements "use" s)));
       cases "an instance holding a stroke that paints nothing is used" ~id:fst
         [ ("an underflow to zero", 1e-200); ("an overflow", 1e200) ]
         (fun (_, k) ->
           (* Two scales, each with an inverse, compose to a map without one,
              under which the stroke paints nothing. *)
           let lost = Picture.transform (Affine.scale k k) in
           let p =
             Picture.group
               [
                 Picture.fill red disc;
                 lost (lost (Picture.stroke (Stroke.v 1.) red disc));
               ]
           in
           let s = svg (Picture.stamp [| 10.; 20. |] [| 5.; 5. |] p) in
           equal int 2 (count s "<use"));
       test "a use sets the colours and alphas of the leaves it replaces"
         (fun () ->
           let half = Color.with_alpha 0.5 red in
           let s =
             svg
               (Picture.stamp ~fills:[| red; half |]
                  ~strokes:[| Color.blue; red |] [| 10.; 20. |] [| 5.; 5. |]
                  marker)
           in
           equal
             (list (triple string string string))
             [ ("#ff0000", "1", "#0000ff"); ("#ff0000", "0.5", "#ff0000") ]
             (List.map
                (fun u ->
                  (attr "fill" u, attr_or "1" "fill-opacity" u, attr "stroke" u))
                (elements "use" s)));
       test "a definition leaves the colours its uses replace unset" (fun () ->
           let s =
             svg
               (Picture.stamp ~fills:[| red |] ~strokes:[| Color.blue |]
                  [| 10. |] [| 5. |] marker)
           in
           (* The fill leaf has no fill and the stroke leaf no stroke, so they
              inherit the colours the use sets. Each turns off the other. *)
           equal
             (list (pair (list string) (list string)))
             [ ([], [ "none" ]); ([ "none" ], []) ]
             (List.map
                (fun p -> (attrs "fill" p, attrs "stroke" p))
                (elements "path" s)));
       test "a fill set by a use is not stroked by another's stroke" (fun () ->
           let s =
             svg
               (Picture.stamp ~strokes:[| red |] [| 10. |] [| 5. |]
                  (Picture.fill red disc))
           in
           equal string "none" (attr "stroke" (element "path" s)));
       test "a run as outlines within a use that sets strokes is not stroked"
         (fun () ->
           let p = Picture.glyphs red (P2.v 0. 0.) (unmapped "fi") in
           let s =
             svg (Picture.stamp ~strokes:[| red |] [| 10. |] [| 5. |] p)
           in
           (* The group of the outlines, then the use. *)
           equal (list string) [ "none"; "#ff0000" ] (attrs "stroke" s));
       test "a use within a definition that sets fills sets its alpha"
         (fun () ->
           let inner =
             Picture.stamp ~fills:[| red |] [| 0. |] [| 0. |]
               (Picture.fill red disc)
           in
           let s =
             svg
               (Picture.stamp
                  ~fills:[| Color.with_alpha 0.5 red |]
                  [| 10. |] [| 5. |] inner)
           in
           equal (list string) [ "1"; "0.5" ] (attrs "fill-opacity" s));
       test "a scaling stamp scales its uses and keeps their pens" scaled_uses;
       cases "a scaling stamp is written in full when its picture holds" ~id:fst
         [
           ( "strokes of differing widths",
             (two_pens, (0, ("stroke-width", [ "1"; "2" ]))) );
           ( "strokes of differing dash offsets",
             ( Picture.group
                 [
                   Picture.stroke (Stroke.v ~dash:[ 2.; 1. ] 1.) red disc;
                   Picture.stroke
                     (Stroke.v ~dash:[ 2.; 1. ] ~dash_offset:1. 1.)
                     red disc;
                 ],
               (0, ("stroke-dashoffset", [ "1" ])) ) );
           (* Its one use is the inner stamp's. *)
           ( "a stamp that scales",
             ( Picture.stamp ~scales:[| 2. |] [| 0. |] [| 0. |]
                 (Picture.fill red disc),
               (1, ("stroke-width", [])) ) );
         ]
         (fun (_, (p, (uses, (name, pens)))) ->
           (* Written in full, its copy keeps the pens of [p]. *)
           let s = svg (Picture.stamp ~scales:[| 3. |] [| 10. |] [| 5. |] p) in
           equal
             (pair int (list string))
             (uses, pens)
             (count s "<use", attrs name s));
       test "a stamp in a scaling stamp paints at the product of their scales"
         nested_scales;
       test "equal stamps share one definition" (fun () ->
           let st = Picture.stamp [| 10. |] [| 5. |] marker in
           let s =
             svg
               (Picture.group
                  [ st; Picture.transform (Affine.translate 0. 20.) st ])
           in
           equal (pair int int) (1, 2) (count s "<g id=", count s "<use"));
       test "styles are attributes, never selectors" (fun () ->
           let s = svg Vg_corpus.(Renderable.picture marks) in
           (* The one style sheet holds [@font-face] rules. *)
           equal int (count s "<style>") (count s "<style>@font-face{");
           not_contains ~sub:"class=" s);
     ]
    @ List.map
        (fun (name, r) ->
          test (name ^ " is written as its golden document") (fun () ->
              expect_file
                ~file:("packages/hugin/test/vg/svg/golden/" ^ name ^ ".svg")
                (masked (Hugin_vg_svg.render r))))
        Vg_corpus.pages)

(* Accuracy *)

(* [written ~w m pts] checks that a polygon through the points [pts] of a page
   [w] wide, drawn as their preimages under [m], is written within 0.001 of
   each. *)
let written ?(w = 100.) m pts =
  let inv = Option.get (Affine.invert m) in
  let q = polyline (List.map (map inv) pts) in
  let s = svg ~w (Picture.transform m (Picture.fill red q)) in
  equal (list point) pts (points (attr "d" s))

let gen_page_points =
  Gen.list ~size:(Gen.int_range 3 6)
    (Gen.pair (Gen.float_range 0. 100.) (Gen.float_range 0. 100.))

let gen_far_map =
  Gen.map
    (fun ((ox, oy), (sx, sy)) -> Affine.(scale sx sy * translate (-.ox) (-.oy)))
    (Gen.pair
       (Gen.pair (Gen.float_range (-2e9) 2e9) (Gen.float_range (-2e9) 2e9))
       (Gen.pair (Gen.float_range 0.01 400.) (Gen.float_range 0.01 400.)))

(* [gen_marker_points] is the corners of a quadrilateral around the origin, one
   in each quadrant, so that it has an area to fill. *)
let gen_marker_points =
  let coord = Gen.float_range 0.1 1. in
  Gen.map
    (fun ((a, b), (c, d), (e, f), (g, h)) ->
      [ (-.a, -.b); (c, -.d); (e, f); (-.g, h) ])
    (Gen.quad (Gen.pair coord coord) (Gen.pair coord coord)
       (Gen.pair coord coord) (Gen.pair coord coord))

(* [placed (scale, (x, y), pts)] checks that the points [pts] of a polygon
   stamped at [(x, y)] and scaled by [scale] are placed within 0.001 of where
   the stamp puts them. *)
let placed (scale, (x, y), pts) =
  let s =
    svg
      (Picture.stamp ~scales:[| scale |] [| x |] [| y |]
         (Picture.fill red (polyline pts)))
  in
  let m = placement (element "use" s) in
  equal (list point)
    (List.map (fun (px, py) -> (x +. (scale *. px), y +. (scale *. py))) pts)
    (List.map (map m) (points (attr "d" s)))

(* [bezier p0 p1 p2 p3 t] is the point at [t] of the cubic of control points
   [p0] to [p3]. *)
let bezier (x0, y0) (x1, y1) (x2, y2) (x3, y3) t =
  let u = 1. -. t in
  let f a b c d =
    (u *. u *. u *. a)
    +. (3. *. u *. u *. t *. b)
    +. (3. *. u *. t *. t *. c)
    +. (t *. t *. t *. d)
  in
  (f x0 x1 x2 x3, f y0 y1 y2 y3)

(* [to_curve c (x, y)] is the distance from [(x, y)] to the curve [c], a
   function from [\[0;1\]]: sampled, then refined around the nearest sample. *)
let to_curve c (x, y) =
  let n = 4000 in
  let dist t =
    let cx, cy = c t in
    Float.hypot (cx -. x) (cy -. y)
  in
  let best = ref 0 and nearest = ref (dist 0.) in
  for i = 1 to n do
    let d = dist (Float.of_int i /. Float.of_int n) in
    if d < !nearest then begin
      best := i;
      nearest := d
    end
  done;
  let lo = ref (Float.max 0. (Float.of_int (!best - 1) /. Float.of_int n))
  and hi = ref (Float.min 1. (Float.of_int (!best + 1) /. Float.of_int n)) in
  for _ = 1 to 60 do
    let a = !lo +. ((!hi -. !lo) /. 3.) and b = !hi -. ((!hi -. !lo) /. 3.) in
    if dist a < dist b then hi := b else lo := a
  done;
  dist !lo

(* [to_polyline pts (x, y)] is the distance from [(x, y)] to the polyline
   through [pts]. *)
let to_polyline pts (x, y) =
  let seg (ax, ay) (bx, by) =
    let dx = bx -. ax and dy = by -. ay in
    let l = (dx *. dx) +. (dy *. dy) in
    let t =
      if l = 0. then 0.
      else
        Float.max 0.
          (Float.min 1. ((((x -. ax) *. dx) +. ((y -. ay) *. dy)) /. l))
    in
    Float.hypot (ax +. (t *. dx) -. x) (ay +. (t *. dy) -. y)
  in
  let rec loop acc = function
    | a :: (b :: _ as rest) -> loop (Float.min acc (seg a b)) rest
    | _ -> acc
  in
  loop infinity pts

(* [drawn d] is the curves of the path data [d], each the control points of a
   cubic, its lines as cubics. *)
let drawn d =
  let curves = ref [] and cur = ref (0., 0.) and start = ref (0., 0.) in
  let line p =
    curves := (!cur, !cur, p, p) :: !curves;
    cur := p
  in
  let command c args =
    match (c, args) with
    | 'M', [ x; y ] ->
        cur := (x, y);
        start := (x, y)
    | 'L', [ x; y ] -> line (x, y)
    | 'C', [ x1; y1; x2; y2; x; y ] ->
        curves := (!cur, (x1, y1), (x2, y2), (x, y)) :: !curves;
        cur := (x, y)
    | 'Z', [] -> line !start
    | _ -> failf "path data %c with %d numbers" c (List.length args)
  in
  let n = String.length d in
  let rec loop i =
    if i < n then begin
      let j = ref (i + 1) in
      while !j < n && not (String.contains "MLCZ" d.[!j]) do
        incr j
      done;
      command d.[i] (numbers (String.sub d (i + 1) (!j - i - 1)));
      loop !j
    end
  in
  loop 0;
  List.rev !curves

(* [boundary d] is the polyline through the curves of [d] sampled finely. *)
let boundary d =
  List.concat_map
    (fun (p0, p1, p2, p3) ->
      List.init 257 (fun i -> bezier p0 p1 p2 p3 (Float.of_int i /. 256.)))
    (drawn d)

(* A point within 0.001 of another along each axis lies within 0.001 sqrt 2 of
   it. *)
let diagonal = 0.001 *. Float.sqrt 2.

let cubic (x0, y0) (x1, y1) (x2, y2) (x3, y3) =
  Path.empty
  |> Path.move_to (P2.v x0 y0)
  |> Path.cubic_to (P2.v x1 y1) (P2.v x2 y2) (P2.v x3 y3)

let crossing (_, (p0, p1, p2, p3)) =
  let curve = bezier p0 p1 p2 p3 in
  let d =
    attr "d" (svg (Picture.stroke (Stroke.v 0.1) red (cubic p0 p1 p2 p3)))
  in
  let got = boundary d in
  List.iter
    (fun v ->
      at_most ~msg:"piece to curve" float_exact ~than:diagonal
        (to_curve curve v))
    got;
  for i = 0 to 2000 do
    let ((x, y) as c) = curve (Float.of_int i /. 2000.) in
    if x > -99. && x < 199. && y > -99. && y < 199. then
      at_most ~msg:"curve to pieces" float_exact ~than:diagonal
        (to_polyline got c)
  done

(* [phase lin s] is the phase in the dash pattern of period 4, in the units of
   the picture, at the start of the path of the document [s] of a dashed line
   from x = -999999.5 drawn under [lin]: where the picture puts it, and as the
   document writes it under its matrix. *)
let phase (lin : Affine.t) s =
  let e = element "path" s in
  let m = matrix e in
  match points (attr "d" e) with
  | (x0, y0) :: (x1, y1) :: _ ->
      let ux, _ = map (Option.get (Affine.invert lin)) (map m (x0, y0)) in
      (* Lengths along the path on the page, per unit of the document and per
         unit of the picture. *)
      let l = Float.hypot (x1 -. x0) (y1 -. y0) in
      let dx = (x1 -. x0) /. l and dy = (y1 -. y0) /. l in
      let written =
        Float.hypot ((m.xx *. dx) +. (m.xy *. dy)) ((m.yx *. dx) +. (m.yy *. dy))
      in
      let picture = Float.hypot lin.xx lin.yx in
      let offset = float_of_string (attr "stroke-dashoffset" e) in
      ( Float.rem (ux +. 999999.5) 4.,
        Float.rem (offset *. written /. picture) 4. )
  | _ -> failf "fewer than two points in %s" e

let dashed_line ?(cap = `Butt) lin y =
  Picture.transform lin
    (Picture.stroke
       (Stroke.v ~cap ~dash:[ 3.; 1. ] 1.)
       red
       (Path.polyline [| -999999.5; 1e6 |] [| y; y |]))

let stretched_pen (lin, pts) =
  let e =
    element "path"
      (svg
         (Picture.transform lin
            (Picture.stroke (Stroke.v 1.) red (polyline pts))))
  in
  let m = matrix e and w = float_of_string (attr "stroke-width" e) in
  equal ~msg:"points" (list point)
    (List.map (map lin) pts)
    (List.map (map m) (points (attr "d" e)));
  equal ~msg:"reach of the pen"
    (list (float 0.001))
    (List.map (reach lin 1.) directions)
    (List.map (reach m w) directions)

(* [alpha_written a] is the fill opacity written for an alpha of [a]. *)
let alpha_written a =
  attr_or "1" "fill-opacity"
    (svg (Picture.fill (Color.with_alpha a red) square))

(* [numbers_finite s] checks that the document [s] holds no number that is not
   finite, its embedded fonts and images left out. *)
let numbers_finite s =
  let b = Buffer.create (String.length s) in
  let rec loop i =
    match find_from s i "base64," with
    | None -> Buffer.add_string b (String.sub s i (String.length s - i))
    | Some j ->
        Buffer.add_string b (String.sub s i (j - i));
        let k = ref (j + 7) in
        while s.[!k] <> '"' && s.[!k] <> ')' do
          incr k
        done;
        loop !k
  in
  loop 0;
  let text = Buffer.contents b in
  not_contains ~sub:"inf" text;
  not_contains ~sub:"nan" text

let gaps =
  Picture.stroke (Stroke.v 1.) red
    (Path.polyline [| 10.; Float.nan; 50.; 60. |] [| 10.; 20.; 50.; infinity |])

let accuracy =
  group "Accuracy"
    [
      prop "points are written within 0.001 of where the page puts them"
        (Gen.pair gen_far_map gen_page_points) (fun (m, pts) -> written m pts);
      test "an offset of 1.6e9 under a scale keeps a 400-point line level"
        (fun () ->
          let m = Affine.(scale 4. 1. * translate (-1.6e9) 0.) in
          written ~w:400. m [ (0., 50.); (400., 50.); (400., 60.) ]);
      test "a line crossing the page far beyond it crosses where it should"
        (fun () ->
          (* [far_line] runs from (-123456789, -123456700) to (123456789,
             123456900): it crosses x = 0 at y = 100. *)
          let far_line =
            Path.polyline
              [| -123456789.; 123456789. |]
              [| -123456700.; 123456900. |]
          in
          let s =
            svg ~w:200. ~h:200. (Picture.stroke (Stroke.v 1.) red far_line)
          in
          match points (attr "d" s) with
          | [ (x0, y0); (x1, y1) ] ->
              (* Cut at the margin, with numbers of the order of the page. *)
              List.iter
                (fun v ->
                  at_most ~msg:"magnitude" float_exact ~than:1000. (Float.abs v))
                [ x0; y0; x1; y1 ];
              let y = y0 +. ((y1 -. y0) *. (0. -. x0) /. (x1 -. x0)) in
              equal (float 0.001) 100. y
          | pts -> failf "%d points" (List.length pts));
      cases "a line to a point far beyond the page keeps its direction"
        ~id:(Printf.sprintf "to %g") [ 1e9; 1e16; 1e300; 1e308 ] (fun l ->
          (* From (10, 20) along (2, 1): it leaves the page's margin on its own
             line, whether its far end maps near the page, past 1e15 or, under
             the scale, beyond the floats. *)
          let line =
            Path.polyline [| 5.; 5. +. l |] [| 10.; 10. +. (l /. 2.) |]
          in
          let s =
            svg ~w:200. ~h:200.
              (Picture.transform (Affine.scale 2. 2.)
                 (Picture.stroke (Stroke.v 1.) red line))
          in
          match points (attr "d" s) with
          | [ (x0, y0); (x1, y1) ] ->
              equal (pair (float 0.001) (float 0.001)) (10., 20.) (x0, y0);
              equal (float 1e-6) 0.5 ((y1 -. y0) /. (x1 -. x0))
          | pts -> failf "%d points" (List.length pts));
      prop "dashes keep within the accuracy over a thousand lengths"
        (Gen.pair (Gen.float_range 0.3 3.)
           (Gen.pair (Gen.float_range 0.1 10.) (Gen.float_range 0.1 10.)))
        (fun (k, (a, b)) ->
          let s =
            svg
              (Picture.transform (Affine.scale k k)
                 (Picture.stroke
                    (Stroke.v ~dash:[ a; b ] 1.)
                    red
                    (Path.polyline [| 10.; 20. |] [| 10.; 10. |])))
          in
          match
            List.map float_of_string
              (String.split_on_char ' ' (attr "stroke-dasharray" s))
          with
          | [ a'; b' ] ->
              (* A thousand lengths, 500 periods, end within 0.001. *)
              at_most float_exact ~than:0.001
                (500. *. Float.abs (a' +. b' -. (k *. (a +. b))))
          | l -> failf "%d lengths" (List.length l));
      test "a dashed line cut at the margin keeps the phase of its dashes"
        (fun () ->
          let s = svg (dashed_line Affine.id 50.) in
          (* The page, its margin and the pen's reach of 0.5 start 999899 along
             the line, 3 into the pattern of period 4. *)
          equal (pair string string)
            ("M-100.5 50L200.5 50", "3")
            (attr "d" s, attr "stroke-dashoffset" s));
      cases "a closed subpath cut at the margin keeps its join at its start"
        ~id:fst
        [
          ( "on the right",
            ( [| 50.; 1e5; 1e5; 50. |],
              [| 50.; 50.; 60.; 60. |],
              "M50 50L202 50L202 60L50 60Z" ) );
          ( "below",
            ( [| 50.; 50.; 60.; 60. |],
              [| 50.; 1e5; 1e5; 50. |],
              "M50 50L50 202L60 202L60 50Z" ) );
          ("of no area", ([| 50.; 1e5 |], [| 50.; 50. |], "M50 50L202 50Z"));
        ]
        (fun (_, (xs, ys, d)) ->
          let q = Path.polygon xs ys in
          let s =
            svg (Picture.stroke (Stroke.v ~cap:`Butt ~join:`Miter 1.) red q)
          in
          (* Cut where the margin and the miter's reach of 2 end, the part
             beyond running along that edge, whose ink lies off the page. *)
          equal string d (attr "d" s));
      prop "paths within the margin are written as they are"
        (Gen.list ~size:(Gen.int_range 2 40)
           (Gen.pair
              (Gen.float_range (-90.) 190.)
              (Gen.float_range (-90.) 190.)))
        (fun pts ->
          let s = svg (Picture.stroke (Stroke.v 1.) red (polyline pts)) in
          equal (list decimals) pts (points (attr "d" s)));
      cases "a stroked curve crossing the margin keeps its part within" ~id:fst
        [
          ("bending once", ((-300., 50.), (10., -60.), (90., 170.), (400., 40.)));
          (* Curves whose second differences, [p0 - 2 p1 + p2] and [p1 - 2 p2 +
             p3], a sign away from zero, test the estimate of chords. *)
          ( "swerving along x",
            ((0., -500.), (100., -100.), (-200., 300.), (500., 700.)) );
          ( "swerving back along x",
            ((0., -500.), (100., -100.), (-200., 300.), (300., 700.)) );
          ( "swerving along y",
            ((-500., 0.), (-100., 100.), (300., -200.), (700., 500.)) );
          ( "swerving back along y",
            ((-500., 0.), (-100., 100.), (300., -200.), (700., 300.)) );
        ]
        crossing;
      test "an area bounded by a curve crossing the margin keeps its part"
        (fun () ->
          let p0 = (-300., 50.)
          and p1 = (10., -60.)
          and p2 = (90., 170.)
          and p3 = (400., 40.) in
          let q =
            cubic p0 p1 p2 p3
            |> Path.line_to (P2.v 400. 1e5)
            |> Path.line_to (P2.v (-300.) 1e5)
            |> Path.close
          in
          let d = attr "d" (svg (Picture.fill red q)) in
          contains ~sub:"C" d;
          List.iter
            (fun (q0, q1, q2, q3) ->
              if not (q0 = q1 && q2 = q3) then
                for i = 0 to 64 do
                  let v = bezier q0 q1 q2 q3 (Float.of_int i /. 64.) in
                  at_most float_exact ~than:diagonal
                    (to_curve (bezier p0 p1 p2 p3) v)
                done)
            (drawn d));
      (* The curve's second control point lies beyond the margin. *)
      cases "a clipped subpath's numbers lie within the margin"
        ~id:(fun (_, c) -> Format.asprintf "control point %a" P2.pp c)
        [ ("below", P2.v 70. 1000.); ("right", P2.v 1000. 60.) ]
        (fun (_, c2) ->
          let q =
            Path.empty
            |> Path.move_to (P2.v 50. 50.)
            |> Path.cubic_to (P2.v 60. 40.) c2 (P2.v 80. 50.)
            |> Path.line_to (P2.v 50. 60.)
            |> Path.close
          in
          List.iter
            (fun v ->
              at_least ~msg:"low" float_exact ~than:(-100.) v;
              at_most ~msg:"high" float_exact ~than:200. v)
            (numbers (attr "d" (svg (Picture.fill red q)))));
      test "a curve leaving the margin keeps its bend within it" (fun () ->
          let p0 = (50., 50.)
          and p1 = (60., 0.)
          and p2 = (150., 0.)
          and p3 = (1e5, 50.) in
          let q =
            cubic p0 p1 p2 p3 |> Path.line_to (P2.v 50. 90.) |> Path.close
          in
          let boundary = boundary (attr "d" (svg (Picture.fill red q))) in
          for i = 0 to 2000 do
            let ((x, _) as c) = bezier p0 p1 p2 p3 (Float.of_int i /. 2000.) in
            if x < 200. then
              at_most float_exact ~than:diagonal (to_polyline boundary c)
          done);
      test "a vertical edge of a clipped area keeps its ends" (fun () ->
          let q =
            Path.polygon [| 10.; 10.; 1e5; 1e5 |] [| 50.; 0.; 0.; 50. |]
          in
          equal
            (list (pair float_exact float_exact))
            [ (10., 0.); (10., 50.); (200., 0.); (200., 50.) ]
            (List.sort compare (points (attr "d" (svg (Picture.fill red q))))));
      test "a curve within the margin is kept when its subpath is clipped"
        (fun () ->
          let q =
            cubic (50., 50.) (60., 40.) (70., 40.) (80., 50.)
            |> Path.line_to (P2.v 1e5 50.)
            |> Path.line_to (P2.v 1e5 60.)
            |> Path.line_to (P2.v 50. 60.)
            |> Path.close
          in
          equal string "M50 50C60 40 70 40 80 50L200 50L200 60L50 60Z"
            (attr "d" (svg (Picture.fill red q))));
      test "a closed subpath whose curve's points cross the margin is closed"
        (fun () ->
          (* The control points lie beyond the margin, the curve within it. *)
          let q =
            cubic (50., 150.) (50., 210.) (150., 210.) (150., 150.)
            |> Path.close
          in
          equal string "M50 150C50 210 150 210 150 150Z"
            (attr "d" (svg (Picture.stroke (Stroke.v 1.) red q))));
      test "a slanted edge clipped at the margin crosses it where it should"
        (fun () ->
          let q =
            Path.polygon [| 50.; 150.; 50. |] [| 50.; 100050.; 100050. |]
          in
          (* The edge from (50, 50) to (150, 100050) crosses y = 200 at x =
             50.15. *)
          let d = attr "d" (svg (Picture.fill red q)) in
          equal
            (list (pair float_exact float_exact))
            [ (50., 50.); (50., 200.); (50.15, 200.) ]
            (List.sort compare (points d)));
      test "a dashed curve keeps its length in the phase of later pieces"
        (fun () ->
          let p0 = (50., 50.)
          and p1 = (60., 40.)
          and p2 = (70., 40.)
          and p3 = (80., 50.) in
          let q =
            cubic p0 p1 p2 p3
            |> Path.line_to (P2.v 1e5 50.)
            |> Path.line_to (P2.v 1e5 60.)
            |> Path.line_to (P2.v 50. 60.)
          in
          let s =
            svg (Picture.stroke (Stroke.v ~cap:`Butt ~dash:[ 3.; 1. ] 1.) red q)
          in
          (* The second piece starts where the line back meets the margin and
             the pen's reach, 200.5. *)
          let curve =
            let l = ref 0. and prev = ref p0 in
            for i = 1 to 100000 do
              let ((x, y) as c) =
                bezier p0 p1 p2 p3 (Float.of_int i /. 100000.)
              in
              l := !l +. Float.hypot (x -. fst !prev) (y -. snd !prev);
              prev := c
            done;
            !l
          in
          let along = curve +. (1e5 -. 80.) +. 10. +. (1e5 -. 200.5) in
          match attrs "stroke-dashoffset" s with
          | [ o ] ->
              equal (float 0.001) (Float.rem along 4.) (float_of_string o)
          | l -> failf "%d offsets" (List.length l));
      test "a dashed polyline cut at the margin keeps the phase of its dashes"
        (fun () ->
          let s =
            svg
              (Picture.stroke
                 (Stroke.v ~cap:`Butt ~dash:[ 3.; 1. ] 1.)
                 red
                 (Path.polyline
                    [| -999999.5; -1000.; 1e6 |]
                    [| 50.; 50.; 50. |]))
          in
          (* The cut, at -100.5, is 999899 along the line, 3 into the
             pattern. *)
          equal (pair string string)
            ("M-100.5 50L200.5 50", "3")
            (attr "d" s, attr "stroke-dashoffset" s));
      (* The start is written within 0.001 of where the picture puts it, which
         moves the phase by as much. *)
      cases "a dashed line drawn under a matrix keeps its phase at the margin"
        ~id:fst
        [
          ("stretched", Affine.scale 1. 4.);
          ("sheared", Affine.(scale 1. 4. * rotate 0.001));
        ]
        (fun (_, lin) ->
          let expected, got = phase lin (svg (dashed_line lin 12.5)) in
          equal (float 0.001) expected got);
      test "a closed subpath that starts beyond the margin closes along it"
        (fun () ->
          let q =
            Path.polygon [| 50.; 50.; 60.; 60. |] [| 1e5; 50.; 50.; 1e5 |]
          in
          let s =
            svg (Picture.stroke (Stroke.v ~cap:`Butt ~join:`Miter 1.) red q)
          in
          equal string "M50 202L50 50L60 50L60 202Z" (attr "d" s));
      test "an open subpath back at its start keeps its pieces apart" (fun () ->
          let q =
            Path.polyline
              [| 50.; 1e5; 1e5; 50.; 50. |]
              [| 50.; 50.; 60.; 60.; 50. |]
          in
          let s =
            svg (Picture.stroke (Stroke.v ~cap:`Butt ~join:`Miter 1.) red q)
          in
          equal string "M50 50L202 50M202 60L50 60L50 50" (attr "d" s));
      test "an area's curve crossing the margin is cut at it" (fun () ->
          let big = Path.circle (P2.v 50. 1e6) (1e6 -. 50.) in
          List.iter
            (fun (x, y) ->
              at_most ~msg:"magnitude" float_exact ~than:1000.
                (Float.max (Float.abs x) (Float.abs y)))
            (points (attr "d" (svg (Picture.fill red big)))));
      test "an area's curves within the margin are kept" (fun () ->
          let small = svg (Picture.fill red (Path.circle (P2.v 50. 50.) 10.)) in
          equal int 4 (count (attr "d" small) "C"));
      test "the area within the margin is kept when a path is clipped"
        (fun () ->
          (* A triangle mostly beyond the margin: the cut keeps the page's part,
             here the half of the page below the diagonal. *)
          let q = Path.polygon [| -1e7; 1e7; -1e7 |] [| -1e7; 1e7; 1e7 |] in
          let raster p =
            Nx.to_array
              (Hugin_vg_raster.render ~density:1. (Renderable.v 100. 100. p))
          in
          let d = attr "d" (svg (Picture.fill red q)) in
          let a = raster (Picture.fill red q)
          and b = raster (Picture.fill red (path_of_data d)) in
          at_most ~msg:"largest difference in levels" int ~than:1
            (Array.fold_left Int.max 0
               (Array.mapi (fun i v -> abs (v - b.(i))) a)));
      test "an image is cropped to the pixels that meet the margin" (fun () ->
          let px = Nx.zeros Nx.uint8 [| 1; 1000; 3 |] in
          let s =
            element "image" (svg (Picture.image (Box2.v (-1e6) 0. 2e6 10.) px))
          in
          (* Each pixel is 2000 points wide: the margin of 100 points around the
             page meets two, from -2000 to 2000. *)
          equal
            (triple (array int) string string)
            ([| 1; 2; 3 |], "-2000", "4000")
            (Nx.shape (png_of s), attr "x" s, attr "width" s));
      test "a cropped image keeps the pixels it shows" (fun () ->
          let px =
            Nx.init Nx.uint8 [| 3; 1000; 3 |] (fun i ->
                ((i.(0) * 100) + i.(1)) mod 256)
          in
          let s =
            element "image" (svg (Picture.image (Box2.v (-1e6) 0. 2e6 30.) px))
          in
          (* Columns 499 and 500 meet the margin. *)
          equal (array int)
            (Array.init 18 (fun k ->
                 ((k / 6 * 100) + 499 + (k / 3 mod 2)) mod 256))
            (Nx.to_array (png_of s)));
      test "an image is cropped to the rows that meet the margin" (fun () ->
          let px = Nx.zeros Nx.uint8 [| 1000; 1; 3 |] in
          let s =
            element "image" (svg (Picture.image (Box2.v 0. (-1e6) 10. 2e6) px))
          in
          (* Each row is 2000 points high: the margin meets two, from -2000 to
             2000. *)
          equal
            (triple (array int) string string)
            ([| 2; 1; 3 |], "-2000", "4000")
            (Nx.shape (png_of s), attr "y" s, attr "height" s));
      cases "an image beyond the margin on one side is cropped there" ~id:fst
        [
          ("left", (1, 1000, Box2.v (-999900.) 0. 1e6 10.));
          ("right", (1, 1000, Box2.v (-100.) 0. 1e6 10.));
          ("top", (1000, 1, Box2.v 0. (-999900.) 10. 1e6));
          ("bottom", (1000, 1, Box2.v 0. (-100.) 10. 1e6));
        ]
        (fun (_, (h, w, box)) ->
          let px = Nx.zeros Nx.uint8 [| h; w; 3 |] in
          equal (array int) [| 1; 1; 3 |]
            (Nx.shape (png_of (element "image" (svg (Picture.image box px))))));
      cases "an image beyond the margin is left out" ~id:fst
        [
          ("to the right", Box2.v 300. 0. 10. 10.);
          ("above", Box2.v 0. (-300.) 10. 10.);
        ]
        (fun (_, box) ->
          let px = Nx.zeros Nx.uint8 [| 2; 2; 3 |] in
          not_contains ~sub:"<image" (svg (Picture.image box px)));
      test "a run beyond the margin is left out" (fun () ->
          not_contains ~sub:"<text"
            (svg
               (Picture.glyphs red (P2.v 1e5 50.) (Vg_corpus.typeset "far" 10.))));
      cases "a pen stretched unevenly is written under a matrix" ~id:fst
        [
          ("along y", (Affine.scale 1. 4., [ (0., 0.); (10., 10.) ]));
          ( "turned and along y",
            ( Affine.(scale 1. 4. * rotate (Float.pi /. 6.)),
              [ (0., 0.); (10., 0.) ] ) );
        ]
        (fun (_, row) -> stretched_pen row);
      test "a pen turned and scaled evenly is written on the page" (fun () ->
          let p =
            Picture.transform
              Affine.(rotate (Float.pi /. 3.) * scale 2. 2.)
              (Picture.stroke (Stroke.v 1.) red
                 (Path.polyline [| 0.; 10. |] [| 0.; 0. |]))
          in
          let s = svg p in
          equal
            (pair (list string) string)
            ([], "2")
            (attrs "transform" s, attr "stroke-width" s));
      test "a run scaled evenly is written on the page" (fun () ->
          let p =
            Picture.transform (Affine.scale 2. 2.)
              (Picture.glyphs red (P2.v 5. 10.) (Vg_corpus.typeset "a" 10.))
          in
          let s = element "text" (svg p) in
          equal
            (triple (list string) string string)
            ([], "20", "10")
            (attrs "transform" s, attr "font-size" s, attr "x" s));
      test
        "a turned run is written under a matrix that keeps its place and size"
        (fun () ->
          let lin =
            Affine.(translate 50. 50. * rotate (Float.pi /. 2.) * scale 2. 2.)
          in
          let p =
            Picture.transform lin
              (Picture.glyphs red (P2.v 0. 0.) (Vg_corpus.typeset "a" 10.))
          in
          let e = element "text" (svg p) in
          let m = matrix e and size = float_of_string (attr "font-size" e) in
          let x = float_of_string (attr "x" e)
          and y = float_of_string (attr "y" e) in
          (* The origin, and the em square's sides along x and y. *)
          equal (list point)
            [
              (50., 50.);
              (10. *. lin.xx, 10. *. lin.yx);
              (10. *. lin.xy, 10. *. lin.yy);
            ]
            [
              map m (x, y);
              (size *. m.xx, size *. m.yx);
              (size *. m.xy, size *. m.yy);
            ]);
      test "an image turned is written under a matrix that keeps its box"
        (fun () ->
          let px = Nx.zeros Nx.uint8 [| 1; 1; 3 |] in
          let lin = Affine.rotate (Float.pi /. 2.) in
          let e =
            element "image"
              (svg
                 (Picture.transform lin
                    (Picture.image (Box2.v 10. 0. 20. 10.) px)))
          in
          let n name = float_of_string (attr_or "0" name e) in
          let x = n "x" and y = n "y" and w = n "width" and h = n "height" in
          let corners x y w h =
            [ (x, y); (x +. w, y); (x +. w, y +. h); (x, y +. h) ]
          in
          equal (list point)
            (List.map (map lin) (corners 10. 0. 20. 10.))
            (List.map (map (matrix e)) (corners x y w h)));
      test "an image scaled along the axes is written on the page" (fun () ->
          let px = Nx.zeros Nx.uint8 [| 1; 1; 3 |] in
          let e =
            element "image"
              (svg
                 (Picture.transform (Affine.scale 2. 3.)
                    (Picture.image (Box2.v 10. 0. 20. 10.) px)))
          in
          equal (list string) [ "20"; "0"; "40"; "30" ]
            (List.map
               (fun n -> attr_or "0" n e)
               [ "x"; "y"; "width"; "height" ]);
          not_contains ~sub:"transform=" e);
      prop
        "a stamp's points are placed within 0.001 of where the page puts them"
        (Gen.triple
           (Gen.float_range 0.01 1000.)
           (Gen.pair (Gen.float_range 0. 100.) (Gen.float_range 0. 100.))
           gen_marker_points)
        placed;
      test "a scaled instance whose pen alone meets a clip is written"
        (fun () ->
          (* The stroke misses the clip at scale 1, and meets it at 0.5 with the
             pen the instance keeps. *)
          let p =
            Picture.clip (rect 0. 0. 1. 1.)
              (Picture.stroke (Stroke.v ~cap:`Butt 2.) red
                 (Path.close (Path.polyline [| 0.; 0. |] [| 2.5; 3.5 |])))
          in
          contains ~sub:"stroke="
            (svg (Picture.stamp ~scales:[| 0.5 |] [| 0. |] [| 0. |] p)));
      test "an instance beyond the margin is left out, however near" (fun () ->
          (* At scale 2, each disc reaches 4 from its centre, 1 short of the
             margin, which spans -100 to 200. *)
          let s =
            svg
              (Picture.stamp ~scales:(Array.make 5 2.)
                 [| 205.; -105.; 50.; 50.; 50. |]
                 [| 50.; 50.; 205.; -105.; 50. |]
                 (Picture.fill red disc))
          in
          equal int 1 (count s "<use"));
      test "a shrunk instance whose pen reaches within the margin is drawn"
        (fun () ->
          (* At a tenth of its size, each ring of radius 2.5 keeps its pen of
             width 1, which reaches 0.75 from its centre, 0.25 within the
             margin. *)
          let ring =
            Picture.stroke (Stroke.v 1.) red (Path.circle (P2.v 0. 0.) 2.5)
          in
          let s =
            svg
              (Picture.stamp ~scales:(Array.make 5 0.1)
                 [| 200.5; -100.5; 50.; 50.; 300. |]
                 [| 50.; 50.; 200.5; -100.5; 50. |]
                 ring)
          in
          equal int 4 (count s "<use"));
      cases "an instance that paints nothing is left out:" ~id:fst
        [
          ("at x = NaN", (marker, (Float.nan, 5., 1000.)));
          ("at y = NaN", (marker, (20., Float.nan, 1000.)));
          ("at y = infinity", (marker, (20., Float.infinity, 1000.)));
          ("of scale NaN", (marker, (20., 5., Float.nan)));
          ("of scale infinity", (marker, (20., 5., Float.infinity)));
          ("of scale 0", (marker, (20., 5., 0.)));
          ("at x = NaN, written in full", (two_pens, (Float.nan, 5., 1.)));
          ("of scale 0, written in full", (two_pens, (20., 5., 0.)));
        ]
        (fun (_, (p, (x, y, k))) ->
          equal string
            (svg (Picture.stamp ~scales:[| 2. |] [| 10. |] [| 5. |] p))
            (svg (Picture.stamp ~scales:[| 2.; k |] [| 10.; x |] [| 5.; y |] p)));
      cases "an alpha is written to three decimals:" ~id:fst
        [
          ("0.0004", (0.0004, "0"));
          ("0.1234", (0.1234, "0.123"));
          ("0.9996", (0.9996, "1"));
        ]
        (fun (_, (a, written)) -> equal string written (alpha_written a));
      cases "a pen keeps its width" ~id:(Printf.sprintf "under a scale of %g")
        [ 1e-170; 1e170 ] (fun k ->
          let p =
            Picture.transform (Affine.scale k k)
              (Picture.stroke
                 (Stroke.v (10. /. k))
                 red
                 (Path.polyline [| 0.; 50. /. k |] [| 50. /. k; 50. /. k |]))
          in
          let s = svg p in
          equal (pair string string) ("10", "M0 50L50 50")
            (attr "stroke-width" s, attr "d" s));
      cases "transforms whose composition overflows or underflows paint nothing"
        ~id:fst
        [ ("overflows", 1e200); ("underflows", 1e-200) ]
        (fun (_, k) ->
          let twice p =
            Picture.transform (Affine.scale k k)
              (Picture.transform (Affine.scale k k) p)
          in
          not_contains ~sub:"<path" (svg (twice (Picture.fill red square))));
      test "a number above 1e15 is written as 1e15" (fun () ->
          equal string "1000000000000000" (attr "stroke-width" (stroked 1e300)));
      test "a number below -1e15 is written as -1e15" (fun () ->
          (* Nothing cuts a stamp's picture, written relative to its
             instances. *)
          let far =
            Picture.fill red
              (Path.polygon [| -1e300; 10.; 10. |] [| 0.; 0.; 10. |])
          in
          let d = attr "d" (svg (Picture.stamp [| 0. |] [| 0. |] far)) in
          equal (list float_exact) [ -1e15 ] [ List.hd (numbers d) ]);
      prop ~examples:(gaps :: Vg_corpus.extremes)
        "numbers are finite whatever the scales" Vg_corpus.gen_extreme (fun p ->
          numbers_finite (svg p));
    ]

(* Tags *)

let tagged rows p =
  Picture.tag
    { id = Nx.Ptree.Path.v [ Index 0; Field "a.b"; Field "q\"" ]; rows }
    p

let cells () =
  let px = Nx.zeros Nx.uint8 [| 3; 4; 3 |] in
  let box = Box2.v 0.5 1. 4. 3. in
  tagged (Cells { box; width = 4; height = 3 }) (Picture.image box px)

let tags =
  group "Tags"
    [
      test "data-id is the id's segments as JSON" (fun () ->
          let s = svg (tagged (Rows [| 1 |]) (Picture.fill red square)) in
          equal string "[0,&quot;a.b&quot;,&quot;q\\&quot;&quot;]"
            (attr "data-id" s));
      test "data-id escapes in JSON what XML cannot hold" (fun () ->
          let id = Nx.Ptree.Path.v [ Field "a b\u{1}\u{FFFE}\u{FFFF}" ] in
          let s =
            svg (Picture.tag { id; rows = Rows [||] } (Picture.fill red square))
          in
          equal string "[&quot;a b\\u0001\\ufffe\\uffff&quot;]"
            (attr "data-id" s));
      test "rows of a picture are listed on its group" (fun () ->
          equal string "3 1 4"
            (attr "data-rows"
               (svg (tagged (Rows [| 3; 1; 4 |]) (Picture.fill red square)))));
      test "no rows write no data-rows" (fun () ->
          not_contains ~sub:"data-rows"
            (svg (tagged (Rows [||]) (Picture.fill red square))));
      test "rows of a stamp are given instance by instance" (fun () ->
          let st = Picture.stamp [| 10.; 20. |] [| 5.; 5. |] marker in
          let s = svg (tagged (Rows [| 7; 9 |]) st) in
          equal (list string) [ "7"; "9" ] (attrs "data-row" s));
      test "rows of a stamp written in full are given instance by instance"
        (fun () ->
          let st =
            Picture.stamp ~scales:[| 1.; 2. |] [| 10.; 20. |] [| 5.; 5. |]
              two_pens
          in
          equal (list string) [ "7"; "9" ]
            (attrs "data-row" (svg (tagged (Rows [| 7; 9 |]) st))));
      test "rows over a transform of a stamp are listed together" (fun () ->
          let st = Picture.stamp [| 10.; 20. |] [| 5.; 5. |] marker in
          let s =
            svg
              (tagged
                 (Rows [| 7; 9 |])
                 (Picture.transform (Affine.translate 1. 1.) st))
          in
          equal
            (pair string (list string))
            ("7 9", [])
            (attr "data-rows" s, attrs "data-row" s));
      test "cells give their box and grid" (fun () ->
          equal string "0.5 1 4 3 4 3" (attr "data-cells" (svg (cells ()))));
      test "cells without transforms above write no data-matrix" (fun () ->
          not_contains ~sub:"data-matrix" (svg (cells ())));
      test "cells under a transform write its matrix" (fun () ->
          let s =
            svg (Picture.transform (Affine.translate 10. 0.1) (cells ()))
          in
          equal string "1 0 0 1 10 0.1" (attr "data-matrix" s));
      test "cells in a stamp's picture written once map to the instance"
        (fun () ->
          let s =
            svg
              (Picture.stamp [| 10.; 20. |] [| 5.; 5. |]
                 (Picture.transform (Affine.scale 2. 2.) (cells ())))
          in
          equal (pair string int) ("2 0 0 2 0 0", 2)
            (attr "data-matrix" s, count s "<use"));
    ]

(* Determinism *)

let determinism =
  group "Determinism"
    [
      prop "equal renderables give equal documents" Vg_corpus.gen_picture
        (Law.ignores
           (Testable.make ~pp:Picture.pp ~equal:Picture.equal)
           string svg Vg_corpus.respell);
      test "a number that rounds to zero is written 0" (fun () ->
          let s =
            svg
              (Picture.fill red
                 (Path.polygon [| -0.; 10.; -0.0001 |] [| -0.; 0.; 10. |]))
          in
          equal string "M0 0L10 0L0 10Z" (attr "d" s));
      test "a clip's id is alike across documents for equal content" (fun () ->
          let clip q =
            Picture.clip q (Picture.fill red (rect 0. 0. 100. 100.))
          in
          let a = svg (clip square)
          and b =
            svg (Picture.group [ Picture.fill red square; clip square ])
          in
          equal string (attr "id" a) (attr "id" b));
      test "a clip of other content has another id" (fun () ->
          let clip q =
            Picture.clip q (Picture.fill red (rect 0. 0. 100. 100.))
          in
          not_equal string
            (attr "id" (svg (clip square)))
            (attr "id" (svg (clip (rect 0. 0. 1. 1.)))));
      test "a stamp's definition id is alike across documents" (fun () ->
          let st = Picture.stamp [| 10. |] [| 5. |] marker in
          let a = svg st
          and b = svg (Picture.group [ Picture.fill red square; st ]) in
          equal string (attr "id" a) (attr "id" b));
      test "a font family is alike across documents" (fun () ->
          let a = svg (typeset_ab ())
          and b =
            svg (Picture.group [ Picture.fill red square; typeset_ab () ])
          in
          equal string (attr "font-family" a) (attr "font-family" b));
    ]

let () = exit (run "hugin.vg svg" [ mapping; accuracy; tags; determinism ])
