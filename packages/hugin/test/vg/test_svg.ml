(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_gg
open Hugin_font
open Hugin_vg

let svg ?(w = 100.) ?(h = 100.) p =
  Hugin_vg_svg.render (Renderable.v w h p)

(* Reading documents *)

let find_from s i sub =
  let n = String.length sub in
  let rec loop i =
    if i + n > String.length s then None
    else if String.sub s i n = sub then Some i
    else loop (i + 1)
  in
  loop i

let contains s sub = find_from s 0 sub <> None

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

(* [element tag s] is the start tag of the first element [tag] of [s]. *)
let element tag s =
  match find_from s 0 ("<" ^ tag ^ " ") with
  | None -> failf "no element %s in:\n%s" tag s
  | Some i -> String.sub s i (String.index_from s i '>' - i + 1)

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

(* [png_of image] is the pixels of the PNG data URI of the [image] element. *)
let png_of image =
  let h = attr "href" image in
  equal string "data:image/png;base64," (String.sub h 0 22);
  Vg_corpus.load_png
    (Vg_corpus.of_base64 (String.sub h 22 (String.length h - 22)))

(* [points d] is the points of the path data [d] of moves and lines. *)
let points d =
  let rec pairs = function x :: y :: rest -> (x, y) :: pairs rest | _ -> [] in
  pairs (numbers d)

(* Pages *)

let red = Color.red
let rect x y w h = Path.rect (Box2.v x y w h)
let square = rect 10. 20. 30. 40.
let glyph c = Font.glyph Font.regular (Uchar.of_char c)

let document =
  group "document"
    [
      test "is a page in points with a point per user unit" (fun () ->
          expect (svg ~w:360.5 ~h:240. Picture.empty)
          @@ __POS_OF__
               {|
            <?xml version="1.0" encoding="UTF-8"?>
            <svg xmlns="http://www.w3.org/2000/svg" width="360.5pt" height="240pt" viewBox="0 0 360.5 240">
            </svg>
          |});
      test "writes a fill as a path" (fun () ->
          expect (svg (Picture.fill red square))
          @@ __POS_OF__
               {|
            <?xml version="1.0" encoding="UTF-8"?>
            <svg xmlns="http://www.w3.org/2000/svg" width="100pt" height="100pt" viewBox="0 0 100 100">
            <path d="M10 20L40 20L40 60L10 60Z" fill="#ff0000"/>
            </svg>
          |});
      test "styles elements by attributes, never by selectors" (fun () ->
          let s = svg Vg_corpus.(Renderable.picture marks) in
          (* The one style sheet holds [@font-face] rules. *)
          equal int (count s "<style>") (count s "<style>@font-face{");
          equal int 0 (count s "class="));
    ]

(* Leaves *)

let stroked ?(cap = `Butt) ?(join = `Miter) ?(miter_limit = 4.) ?dash
    ?dash_offset w =
  svg
    (Picture.stroke
       (Stroke.v ~cap ~join ~miter_limit ?dash ?dash_offset w)
       red
       (Path.polyline [| 10.; 50.; 50. |] [| 10.; 10.; 50. |]))

let point = Path.empty |> Path.move_to (P2.v 5. 5.) |> Path.line_to (P2.v 5. 5.)

let leaves =
  group "leaves"
    [
      test "a colour is #rrggbb and an alpha below 1 an opacity" (fun () ->
          let s = svg (Picture.fill (Color.v ~alpha:0.25 0.2 0.4 1.) square) in
          equal string "#3366ff" (attr "fill" s);
          equal string "0.25" (attr "fill-opacity" s));
      test "an opaque colour writes no opacity" (fun () ->
          equal (list string) []
            (attrs "fill-opacity" (svg (Picture.fill red square))));
      test "an even-odd fill says so" (fun () ->
          equal string "evenodd"
            (attr "fill-rule" (svg (Picture.fill ~rule:`Even_odd red square))));
      test "a stroke writes its pen, defaults left out" (fun () ->
          let s = stroked 2. in
          equal string "none" (attr "fill" s);
          equal string "#ff0000" (attr "stroke" s);
          equal string "2" (attr "stroke-width" s);
          equal (list string) [] (attrs "stroke-linecap" s);
          equal (list string) [] (attrs "stroke-linejoin" s);
          equal (list string) [] (attrs "stroke-miterlimit" s));
      test "a stroke writes caps, joins and limits that differ" (fun () ->
          let s = stroked ~cap:`Square ~join:`Round 2. in
          equal string "square" (attr "stroke-linecap" s);
          equal string "round" (attr "stroke-linejoin" s);
          equal string "2.5"
            (attr "stroke-miterlimit" (stroked ~miter_limit:2.5 2.)));
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
              at_most float_exact ~than:0.0005
                (500. *. Float.abs (a' +. b' -. (k *. (a +. b))))
          | l -> failf "%d lengths" (List.length l));
      test "a dashed stroke writes its pattern and offset" (fun () ->
          let s = stroked ~dash:[ 4.; 2. ] ~dash_offset:1. 2. in
          equal string "4 2" (attr "stroke-dasharray" s);
          equal string "1" (attr "stroke-dashoffset" s));
      test
        "a dash offset keeps its phase in an odd pattern, which repeats twice"
        (fun () ->
          (* [1 2 3] dashes as [1 2 3 1 2 3], of period 12. *)
          List.iter
            (fun o ->
              let s = stroked ~dash:[ 1.; 2.; 3. ] ~dash_offset:o 2. in
              equal
                ~msg:(Printf.sprintf "offset %g" o)
                (float 1e-9) (Float.rem o 12.)
                (Float.rem (float_of_string (attr "stroke-dashoffset" s)) 12.))
            [ 0.5; 7.; 13. ]);
      cases ~name:fst "a subpath of zero length"
        [
          ("is left out with square caps", (`Square, 0));
          ("is kept with round caps", (`Round, 1));
        ]
        (fun (_, (cap, n)) ->
          let s = svg (Picture.stroke (Stroke.v ~cap 2.) red point) in
          equal int n (count s "<path"));
      test "a subpath that repeats its start, then moves, is stroked" (fun () ->
          let q = Path.polyline [| 5.; 5.; 6. |] [| 5.; 5.; 6. |] in
          let s = svg (Picture.stroke (Stroke.v ~cap:`Square 2.) red q) in
          equal string "M5 5L5 5L6 6" (attr "d" s));
      test "a subpath of a start alone is a dot with round caps" (fun () ->
          let lone = Path.empty |> Path.move_to (P2.v 5. 6.) in
          let s = svg (Picture.stroke (Stroke.v 2.) red lone) in
          equal string "M5 6L5 6" (attr "d" s));
      test "a run is text with its glyphs' positions" (fun () ->
          let r = Vg_corpus.typeset "AV" 10. in
          let s = svg (Picture.glyphs red (P2.v 5. 50.) r) in
          equal int 1 (count s "<text");
          equal string
            (Printf.sprintf "5 %g"
               (Float.round ((5. +. Run.x r 1) *. 1000.) /. 1000.))
            (attr "x" s);
          equal string "50" (attr "y" s);
          equal string "10" (attr "font-size" s);
          is_true ~msg:"text" (contains s ">AV</text>");
          is_true ~msg:"shaping off" (contains s "font-kerning:none"));
      test "a run off one baseline gives each glyph its y" (fun () ->
          let r =
            Run.v ~ys:[| 0.; 2.5 |] ~font:Font.regular ~size:10. ~text:"ab"
              ~glyphs:[| glyph 'a'; glyph 'b' |]
              ~xs:[| 0.; 6. |] ()
          in
          let s = svg (Picture.glyphs red (P2.v 5. 50.) r) in
          equal string "5 11" (attr "x" s);
          equal string "50 52.5" (attr "y" s));
      test
        "a font is embedded once, as the subset of its text, under a name from \
         its bytes" (fun () ->
          let a = Picture.glyphs red (P2.v 5. 20.) (Vg_corpus.typeset "a" 10.)
          and b =
            Picture.glyphs red (P2.v 5. 40.) (Vg_corpus.typeset "b" 10.)
          in
          let s = svg (Picture.group [ a; b ]) in
          equal int 1 (count s "@font-face");
          let family = attr "font-family" s in
          let uri =
            "font-family:" ^ family ^ ";src:url(data:font/ttf;base64,"
          in
          let start = Option.get (find_from s 0 uri) + String.length uri in
          let stop = Option.get (find_from s start ")") in
          equal ~msg:"the subset of a and b" string
            (Font.subset Font.regular [ glyph 'a'; glyph 'b' ])
            (Vg_corpus.of_base64 (String.sub s start (stop - start)));
          equal (list string) [ family; family ] (attrs "font-family" s));
      test "a run with a glyph the cmap does not give is outlines" (fun () ->
          let font = Font.regular in
          let r =
            Run.v ~font ~size:10. ~text:"a"
              ~glyphs:[| Font.glyph font (Uchar.of_char 'b') |]
              ~xs:[| 0. |] ()
          in
          let s = svg (Picture.glyphs red (P2.v 5. 50.) r) in
          equal int 0 (count s "<text");
          equal string "a" (attr "aria-label" s);
          equal int 1 (count s "<path"));
      cases ~name:fst "a run is outlines when a glyph renders"
        [
          ("two characters", ("fi", [| 0 |]));
          ( "a character beyond the Basic Multilingual Plane",
            ("\u{1D465}", [| 0 |]) );
          ("a tab", ("\t", [| 0 |]));
          ("a character the font lacks", ("\u{4E2D}", [| 0 |]));
        ]
        (fun (_, (text, clusters)) ->
          let font = Font.regular in
          let u = String.get_utf_8_uchar text 0 |> Uchar.utf_decode_uchar in
          let r =
            Run.v ~clusters ~font ~size:10. ~text
              ~glyphs:[| Font.glyph font u |]
              ~xs:[| 0. |] ()
          in
          let s = svg (Picture.glyphs red (P2.v 5. 50.) r) in
          equal int 0 (count s "<text");
          equal int 1 (count s "aria-label="));
      test "a run of glyphs without text is outlines" (fun () ->
          let r =
            Run.v ~clusters:[| 0 |] ~font:Font.regular ~size:10. ~text:""
              ~glyphs:[| glyph 'a' |]
              ~xs:[| 0. |] ()
          in
          let s = svg (Picture.glyphs red (P2.v 5. 50.) r) in
          equal int 0 (count s "<text");
          equal string "" (attr "aria-label" s);
          equal int 1 (count s "<path"));
      test "a run's label holds what XML can hold, U+FFFD the rest" (fun () ->
          let text = "<&\u{1} \t\u{FFFD}\u{FFFE}\u{FFFF}\u{10000}" in
          let r =
            Run.v ~clusters:[| 0 |] ~font:Font.regular ~size:10. ~text
              ~glyphs:[| glyph 'a' |]
              ~xs:[| 0. |] ()
          in
          equal string "&lt;&amp;\u{FFFD} &#9;\u{FFFD}\u{FFFD}\u{FFFD}\u{10000}"
            (attr "aria-label" (svg (Picture.glyphs red (P2.v 5. 50.) r))));
      test "an image is a PNG stretched over its box, not smoothed" (fun () ->
          let px =
            Nx.init Nx.uint8 [| 2; 3; 3 |] (fun i ->
                (i.(0) * 100) + (i.(1) * 50) + (i.(2) * 9))
          in
          let s =
            element "image" (svg (Picture.image (Box2.v 10. 20. 30. 40.) px))
          in
          equal (list string) [ "10"; "20"; "30"; "40" ]
            [ attr "x" s; attr "y" s; attr "width" s; attr "height" s ];
          equal string "none" (attr "preserveAspectRatio" s);
          equal string "image-rendering:pixelated" (attr "style" s);
          equal (array int) (Nx.to_array px) (Nx.to_array (png_of s)));
      test "a clip is a clip path the group of its picture refers to" (fun () ->
          let s =
            svg
              (Picture.clip ~rule:`Even_odd square
                 (Picture.fill red (rect 0. 0. 100. 100.)))
          in
          let id = attr "id" s in
          equal string ("url(#" ^ id ^ ")") (attr "clip-path" s);
          equal string "evenodd" (attr "clip-rule" s));
      test "a transform writes no element of its own" (fun () ->
          let s =
            svg
              (Picture.transform (Affine.translate 5. 7.)
                 (Picture.fill red square))
          in
          equal int 0 (count s "<g");
          equal string "M15 27L45 27L45 67L15 67Z" (attr "d" s));
      test "an opacity is a group" (fun () ->
          let s = svg (Picture.opacity 0.5 (Picture.fill red square)) in
          equal string "0.5" (attr "opacity" s));
    ]

(* Accuracy *)

(* [written ~w m pts] checks that a polygon through the points [pts] of a page
   [w] wide, drawn as their preimages under [m], is written within 0.001 of
   each. *)
let written ?(w = 100.) m pts =
  let inv = Option.get (Affine.invert m) in
  let user = List.map (fun (x, y) -> P2.transform inv (P2.v x y)) pts in
  let q =
    Path.polyline
      (Array.of_list (List.map P2.x user))
      (Array.of_list (List.map P2.y user))
  in
  let s = svg ~w (Picture.transform m (Picture.fill red q)) in
  let got = points (attr "d" s) in
  equal int (List.length pts) (List.length got);
  List.iter2
    (fun (x, y) (x', y') ->
      at_most ~msg:"x" float_exact ~than:0.001 (Float.abs (x -. x'));
      at_most ~msg:"y" float_exact ~than:0.001 (Float.abs (y -. y')))
    pts got

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

(* [placed scale x y pts] checks that the points [pts] of a polygon stamped at
   [(x, y)] and scaled by [scale] are placed within 0.001 of where they are. *)
let placed scale x y pts =
  let q =
    Path.polyline
      (Array.of_list (List.map fst pts))
      (Array.of_list (List.map snd pts))
  in
  let s =
    svg (Picture.stamp ~scales:[| scale |] [| x |] [| y |] (Picture.fill red q))
  in
  let floats v =
    List.filter_map float_of_string_opt
      (String.split_on_char ' '
         (String.map (function '(' | ')' -> ' ' | c -> c) v))
  in
  match floats (attr "transform" (element "use" s)) with
  | [ tx; ty; k ] ->
      let got = points (attr "d" s) in
      equal int (List.length pts) (List.length got);
      List.iter2
        (fun (px, py) (fx, fy) ->
          at_most ~msg:"x" float_exact ~than:0.001
            (Float.abs (tx +. (k *. fx) -. (x +. (scale *. px))));
          at_most ~msg:"y" float_exact ~than:0.001
            (Float.abs (ty +. (k *. fy) -. (y +. (scale *. py)))))
        pts got
  | _ -> failf "one translation and scale"

(* [matrix s] is the coefficients of the matrix of the [transform] of [s]. *)
let matrix s =
  let v = attr "transform" s in
  let inner = String.sub v 7 (String.length v - 8) in
  List.map float_of_string (String.split_on_char ' ' inner)

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

(* [far_line] runs from (-123456789, -123456700) to (123456789, 123456900): it
   crosses x = 0 at y = 100. *)
let far_line =
  Path.polyline [| -123456789.; 123456789. |] [| -123456700.; 123456900. |]

let accuracy =
  group "accuracy"
    [
      prop "points are written within 0.001 of where the page puts them"
        (Gen.pair gen_far_map gen_page_points) (fun (m, pts) -> written m pts);
      test "an offset of 1.6e9 under a scale keeps a 400-point line level"
        (fun () ->
          let m = Affine.(scale 4. 1. * translate (-1.6e9) 0.) in
          written ~w:400. m [ (0., 50.); (400., 50.); (400., 60.) ]);
      test "a line crossing the page far beyond it crosses where it should"
        (fun () ->
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
              at_most float_exact ~than:0.001 (Float.abs (y -. 100.))
          | pts -> failf "%d points" (List.length pts));
      test "a dashed line cut at the margin keeps the phase of its dashes"
        (fun () ->
          let s =
            svg
              (Picture.stroke
                 (Stroke.v ~cap:`Butt ~dash:[ 3.; 1. ] 1.)
                 red
                 (Path.polyline [| -999999.5; 1e6 |] [| 50.; 50. |]))
          in
          (* The page, its margin and the pen's reach of 0.5 start 999899 along
             the line, 3 into the pattern of period 4. *)
          equal string "M-100.5 50L200.5 50" (attr "d" s);
          equal string "3" (attr "stroke-dashoffset" s));
      cases ~name:fst
        "a closed subpath cut at the margin keeps its join at its start"
        [
          ( "on the right",
            ( [| 50.; 1e5; 1e5; 50. |],
              [| 50.; 50.; 60.; 60. |],
              "M50 50L202 50L202 60L50 60Z" ) );
          ( "below",
            ( [| 50.; 50.; 60.; 60. |],
              [| 50.; 1e5; 1e5; 50. |],
              "M50 50L50 202L60 202L60 50Z" ) );
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
          let q =
            Path.polyline
              (Array.of_list (List.map fst pts))
              (Array.of_list (List.map snd pts))
          in
          let got =
            points (attr "d" (svg (Picture.stroke (Stroke.v 1.) red q)))
          in
          equal int (List.length pts) (List.length got);
          List.iter2
            (fun (x, y) (x', y') ->
              at_most ~msg:"x" float_exact ~than:0.00051 (Float.abs (x -. x'));
              at_most ~msg:"y" float_exact ~than:0.00051 (Float.abs (y -. y')))
            pts got);
      cases ~name:fst
        "a stroked curve crossing the margin keeps its part within"
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
        (fun (_, (p0, p1, p2, p3)) ->
          let curve = bezier p0 p1 p2 p3 in
          let q =
            Path.empty
            |> Path.move_to (P2.v (fst p0) (snd p0))
            |> Path.cubic_to
                 (P2.v (fst p1) (snd p1))
                 (P2.v (fst p2) (snd p2))
                 (P2.v (fst p3) (snd p3))
          in
          let d = attr "d" (svg (Picture.stroke (Stroke.v 0.1) red q)) in
          let got = boundary d in
          (* The pieces lie on the curve, their numbers rounded by up to 0.0005
             along each axis. *)
          List.iter
            (fun v ->
              at_most ~msg:"piece to curve" float_exact ~than:0.0013
                (to_curve curve v))
            got;
          for i = 0 to 2000 do
            let ((x, y) as c) = curve (Float.of_int i /. 2000.) in
            if x > -99. && x < 199. && y > -99. && y < 199. then
              at_most ~msg:"curve to pieces" float_exact ~than:0.0013
                (to_polyline got c)
          done);
      test "an area bounded by a curve crossing the margin keeps its part"
        (fun () ->
          let p0 = (-300., 50.)
          and p1 = (10., -60.)
          and p2 = (90., 170.)
          and p3 = (400., 40.) in
          let q =
            Path.empty
            |> Path.move_to (P2.v (-300.) 50.)
            |> Path.cubic_to (P2.v 10. (-60.)) (P2.v 90. 170.) (P2.v 400. 40.)
            |> Path.line_to (P2.v 400. 1e5)
            |> Path.line_to (P2.v (-300.) 1e5)
            |> Path.close
          in
          let d = attr "d" (svg (Picture.fill red q)) in
          at_least ~msg:"curves" int ~than:1 (count d "C");
          (* The curves written lie on the curve, their numbers rounded by up to
             0.0005 along each axis. *)
          List.iter
            (fun (q0, q1, q2, q3) ->
              if not (q0 = q1 && q2 = q3) then
                for i = 0 to 64 do
                  let v = bezier q0 q1 q2 q3 (Float.of_int i /. 64.) in
                  at_most float_exact ~than:0.0013
                    (to_curve (bezier p0 p1 p2 p3) v)
                done)
            (drawn d));
      test "a clipped subpath's numbers lie within the margin" (fun () ->
          (* The curve's second control point lies beyond the margin, below it
             or to its right. *)
          List.iter
            (fun c2 ->
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
                (numbers (attr "d" (svg (Picture.fill red q)))))
            [ P2.v 70. 1000.; P2.v 1000. 60. ]);
      test "a curve leaving the margin keeps its bend within it" (fun () ->
          let p0 = (50., 50.)
          and p1 = (60., 0.)
          and p2 = (150., 0.)
          and p3 = (1e5, 50.) in
          let q =
            Path.empty
            |> Path.move_to (P2.v 50. 50.)
            |> Path.cubic_to (P2.v 60. 0.) (P2.v 150. 0.) (P2.v 1e5 50.)
            |> Path.line_to (P2.v 50. 90.)
            |> Path.close
          in
          let boundary = boundary (attr "d" (svg (Picture.fill red q))) in
          for i = 0 to 2000 do
            let ((x, _) as c) = bezier p0 p1 p2 p3 (Float.of_int i /. 2000.) in
            if x < 200. then
              at_most float_exact ~than:0.0013 (to_polyline boundary c)
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
            Path.empty
            |> Path.move_to (P2.v 50. 50.)
            |> Path.cubic_to (P2.v 60. 40.) (P2.v 70. 40.) (P2.v 80. 50.)
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
            Path.empty
            |> Path.move_to (P2.v 50. 150.)
            |> Path.cubic_to (P2.v 50. 210.) (P2.v 150. 210.) (P2.v 150. 150.)
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
            Path.empty
            |> Path.move_to (P2.v 50. 50.)
            |> Path.cubic_to (P2.v 60. 40.) (P2.v 70. 40.) (P2.v 80. 50.)
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
              at_most float_exact ~than:0.001
                (Float.abs (float_of_string o -. Float.rem along 4.))
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
          equal string "M-100.5 50L200.5 50" (attr "d" s);
          equal string "3" (attr "stroke-dashoffset" s));
      test "a dashed line stretched unevenly keeps its phase at the margin"
        (fun () ->
          let s =
            svg
              (Picture.transform (Affine.scale 1. 4.)
                 (Picture.stroke
                    (Stroke.v ~cap:`Butt ~dash:[ 3.; 1. ] 1.)
                    red
                    (Path.polyline [| -999999.5; 1e6 |] [| 12.5; 12.5 |])))
          in
          (* Under the matrix, lengths along x are 4 times the page's: the cut,
             where the margin and the pen's reach of 2 end, is 3999590 along the
             line, 6 into the pattern of period 16. *)
          equal string "matrix(0.25 0 0 1 0 0)" (attr "transform" s);
          equal string "M-408 50L808 50" (attr "d" s);
          equal string "6" (attr "stroke-dashoffset" s));
      test "a closed subpath that starts beyond the margin closes along it"
        (fun () ->
          let q =
            Path.polygon [| 50.; 50.; 60.; 60. |] [| 1e5; 50.; 50.; 1e5 |]
          in
          let s =
            svg (Picture.stroke (Stroke.v ~cap:`Butt ~join:`Miter 1.) red q)
          in
          equal string "M50 202L50 50L60 50L60 202Z" (attr "d" s));
      test "a dashed line sheared at the margin keeps its phase" (fun () ->
          let a = 0.001 in
          let lin = Affine.(scale 1. 4. * rotate a) in
          let s =
            svg
              (Picture.transform lin
                 (Picture.stroke
                    (Stroke.v ~cap:`Butt ~dash:[ 3.; 1. ] 1.)
                    red
                    (Path.polyline [| -999999.5; 1e6 |] [| 12.5; 12.5 |])))
          in
          (* The line enters where the margin and the pen's reach of 2 end, at x
             = -102 on the page; dashes are 4 times as long under the matrix as
             along the line. *)
          let u = (-102. +. (12.5 *. sin a)) /. cos a in
          match attrs "stroke-dashoffset" s with
          | [ o ] ->
              at_most float_exact ~than:0.01
                (Float.abs
                   (float_of_string o -. Float.rem (4. *. (u +. 999999.5)) 16.))
          | l -> failf "%d offsets" (List.length l));
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
          let s = svg (Picture.fill red big) in
          List.iter
            (fun (x, y) ->
              at_most ~msg:"magnitude" float_exact ~than:1000.
                (Float.max (Float.abs x) (Float.abs y)))
            (points (attr "d" s));
          let small = svg (Picture.fill red (Path.circle (P2.v 50. 50.) 10.)) in
          equal int 4 (count (attr "d" small) "C"));
      test "the area within the margin is kept when a path is clipped"
        (fun () ->
          (* A triangle mostly beyond the margin: the cut keeps the page's part,
             here the half of the page below the diagonal. *)
          let q = Path.polygon [| -1e7; 1e7; -1e7 |] [| -1e7; 1e7; 1e7 |] in
          let raster =
            Hugin_vg_raster.render ~density:1.
              (Renderable.v 100. 100. (Picture.fill red q))
          in
          let d = attr "d" (svg (Picture.fill red q)) in
          let again =
            Hugin_vg_raster.render ~density:1.
              (Renderable.v 100. 100.
                 (Picture.fill red (Vg_corpus.path_of_data d)))
          in
          let a = Nx.to_array raster and b = Nx.to_array again in
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
          equal (array int) [| 1; 2; 3 |] (Nx.shape (png_of s));
          equal string "-2000" (attr "x" s);
          equal string "4000" (attr "width" s));
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
          equal (array int) [| 2; 1; 3 |] (Nx.shape (png_of s));
          equal string "-2000" (attr "y" s);
          equal string "4000" (attr "height" s));
      cases ~name:fst "an image beyond the margin on one side is cropped there"
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
      test "a run or an instance beyond the margin is left out" (fun () ->
          let r =
            Picture.glyphs red (P2.v 1e5 50.) (Vg_corpus.typeset "far" 10.)
          in
          equal int 0 (count (svg r) "<text");
          let st =
            Picture.stamp [| 50.; 1e5 |] [| 50.; 50. |]
              (Picture.fill red square)
          in
          equal int 1 (count (svg st) "<use"));
      test "a pen stretched unevenly is written under a matrix" (fun () ->
          let p =
            Picture.transform (Affine.scale 1. 4.)
              (Picture.stroke (Stroke.v 1.) red
                 (Path.polyline [| 0.; 10. |] [| 0.; 10. |]))
          in
          let s = svg p in
          equal string "matrix(0.25 0 0 1 0 0)" (attr "transform" s);
          equal string "4" (attr "stroke-width" s);
          (* Under the matrix, the end (10, 40) on the page is (40, 40). *)
          equal string "M0 0L40 40" (attr "d" s));
      test "a pen turned and stretched unevenly is written under a matrix"
        (fun () ->
          let a = Float.pi /. 6. in
          let lin = Affine.(scale 1. 4. * rotate a) in
          let p =
            Picture.transform lin
              (Picture.stroke (Stroke.v 1.) red
                 (Path.polyline [| 0.; 10. |] [| 0.; 0. |]))
          in
          let s = svg p in
          (* The map stretches by 4 at most, and the matrix is its linear part
             divided by 4, under which the path's points land on the page. *)
          equal string "4" (attr "stroke-width" s);
          let m = matrix s in
          equal
            (list (float 1e-12))
            [ cos a /. 4.; sin a; -.sin a /. 4.; cos a; 0.; 0. ]
            m;
          match (m, points (attr "d" s)) with
          | [ xx; yx; xy; yy; _; _ ], [ (x0, y0); (x1, y1) ] ->
              let at x y = ((xx *. x) +. (xy *. y), (yx *. x) +. (yy *. y)) in
              let page = P2.transform lin (P2.v 10. 0.) in
              equal (pair (float 0.002) (float 0.002)) (0., 0.) (at x0 y0);
              equal
                (pair (float 0.002) (float 0.002))
                (P2.x page, P2.y page)
                (at x1 y1)
          | _ -> failf "one matrix and two points");
      test "a pen turned and scaled evenly is written on the page" (fun () ->
          let p =
            Picture.transform
              Affine.(rotate (Float.pi /. 3.) * scale 2. 2.)
              (Picture.stroke (Stroke.v 1.) red
                 (Path.polyline [| 0.; 10. |] [| 0.; 0. |]))
          in
          let s = svg p in
          equal (list string) [] (attrs "transform" s);
          equal string "2" (attr "stroke-width" s));
      test "a run scaled evenly is written on the page" (fun () ->
          let p =
            Picture.transform (Affine.scale 2. 2.)
              (Picture.glyphs red (P2.v 5. 10.) (Vg_corpus.typeset "a" 10.))
          in
          let s = svg p in
          equal (list string) [] (attrs "transform" s);
          equal string "20" (attr "font-size" s);
          equal string "10" (attr "x" s));
      test "a turned run is written under a matrix at its origin" (fun () ->
          let p =
            Picture.transform
              Affine.(translate 50. 50. * rotate (Float.pi /. 2.) * scale 2. 2.)
              (Picture.glyphs red (P2.v 0. 0.) (Vg_corpus.typeset "a" 10.))
          in
          let s = svg p in
          equal string "matrix(0 1 -1 0 50 50)" (attr "transform" s);
          equal string "20" (attr "font-size" s));
      test "an image beyond the margin is left out" (fun () ->
          let px = Nx.zeros Nx.uint8 [| 2; 2; 3 |] in
          List.iter
            (fun box ->
              equal int 0 (count (svg (Picture.image box px)) "<image"))
            [ Box2.v 300. 0. 10. 10.; Box2.v 0. (-300.) 10. 10. ]);
      test "an image under a turn is written under a matrix" (fun () ->
          let px = Nx.zeros Nx.uint8 [| 1; 1; 3 |] in
          let p =
            Picture.transform
              (Affine.rotate (Float.pi /. 2.))
              (Picture.image (Box2.v 10. 0. 20. 10.) px)
          in
          let s = element "image" (svg p) in
          equal string "matrix(0 1 -1 0 0 10)" (attr "transform" s);
          equal (list string) [ "20"; "10" ] [ attr "width" s; attr "height" s ]);
      prop
        "a stamp's points are placed within 0.001 of where the page puts them"
        (Gen.triple
           (Gen.float_range 0.01 1000.)
           (Gen.pair (Gen.float_range 0. 100.) (Gen.float_range 0. 100.))
           gen_marker_points)
        (fun (scale, (x, y), pts) -> placed scale x y pts);
    ]

(* Stamps *)

let disc = Path.circle (P2.v 0. 0.) 2.

let marker =
  Picture.group
    [
      Picture.fill Color.black disc;
      Picture.stroke (Stroke.v 0.5) Color.white disc;
    ]

let stamps =
  group "stamps"
    [
      test "a stamp writes its picture once and uses it per instance" (fun () ->
          let s =
            svg (Picture.stamp [| 10.; 20.; 30. |] [| 5.; 6.; 7. |] marker)
          in
          equal int 1 (count s "<g id=");
          equal int 3 (count s "<use");
          equal (list string) [ "10"; "20"; "30" ] (attrs "x" s);
          equal (list string) [ "5"; "6"; "7" ] (attrs "y" s));
      test "fills and strokes are set on each use and inherited" (fun () ->
          let half = Color.with_alpha 0.5 red in
          let s =
            svg
              (Picture.stamp ~fills:[| red; half |]
                 ~strokes:[| Color.blue; red |] [| 10.; 20. |] [| 5.; 5. |]
                 marker)
          in
          (* The definition's leaves leave their colours to the uses. *)
          equal (list string) [ "none"; "#ff0000"; "#ff0000" ] (attrs "fill" s);
          equal (list string)
            [ "none"; "#0000ff"; "#ff0000" ]
            (attrs "stroke" s);
          equal (list string) [ "0.5" ] (attrs "fill-opacity" s));
      test "a fill set by a use is not stroked by another's stroke" (fun () ->
          let s =
            svg
              (Picture.stamp ~strokes:[| red |] [| 10. |] [| 5. |]
                 (Picture.fill red disc))
          in
          equal (list string) [ "none"; "#ff0000" ] (attrs "stroke" s));
      test "a scaling stamp scales its uses and keeps their pens" (fun () ->
          let ring = Picture.stroke (Stroke.v ~dash:[ 2.; 1. ] 1.) red disc in
          let s =
            svg
              (Picture.stamp ~scales:[| 2.; 4. |] [| 10.; 20. |] [| 5.; 5. |]
                 ring)
          in
          equal (list string)
            [ "translate(10 5) scale(2)"; "translate(20 5) scale(4)" ]
            (attrs "transform" s);
          equal (list string) [ "0.5"; "0.25" ] (attrs "stroke-width" s);
          equal (list string) [ "1 0.5"; "0.5 0.25" ]
            (attrs "stroke-dasharray" s));
      test "a scaling stamp of strokes of differing pens is written in full"
        (fun () ->
          let two =
            Picture.group
              [
                Picture.stroke (Stroke.v 1.) red disc;
                Picture.stroke (Stroke.v 2.) red disc;
              ]
          in
          let s = svg (Picture.stamp ~scales:[| 2. |] [| 10. |] [| 5. |] two) in
          equal int 0 (count s "<use");
          equal (list string) [ "1"; "2" ] (attrs "stroke-width" s));
      test "a scaling stamp of strokes of differing dash offsets is in full"
        (fun () ->
          let two =
            Picture.group
              [
                Picture.stroke (Stroke.v ~dash:[ 2.; 1. ] 1.) red disc;
                Picture.stroke
                  (Stroke.v ~dash:[ 2.; 1. ] ~dash_offset:1. 1.)
                  red disc;
              ]
          in
          let s = svg (Picture.stamp ~scales:[| 2. |] [| 10. |] [| 5. |] two) in
          equal int 0 (count s "<use");
          equal (list string) [ "1" ] (attrs "stroke-dashoffset" s));
      test "a scaling stamp holding a scaling stamp is written in full"
        (fun () ->
          let inner =
            Picture.stamp ~scales:[| 2. |] [| 0. |] [| 0. |]
              (Picture.fill red disc)
          in
          let s =
            svg (Picture.stamp ~scales:[| 3. |] [| 10. |] [| 5. |] inner)
          in
          equal int 1 (count s "<use");
          (* The inner definition is drawn at the outer instance's scale. *)
          equal (list string)
            [ "translate(10 5) scale(2)" ]
            (attrs "transform" s));
      test "a use within a definition that sets fills sets its alpha" (fun () ->
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
      test "instances at non-finite positions or of scale 0 are left out"
        (fun () ->
          let s =
            svg
              (Picture.stamp ~scales:[| 1.; 0.; 1. |] [| 10.; 20.; Float.nan |]
                 [| 5.; 5.; 5. |] marker)
          in
          equal int 1 (count s "<use");
          let two =
            Picture.group
              [
                Picture.stroke (Stroke.v 1.) red disc;
                Picture.stroke (Stroke.v 2.) red disc;
              ]
          in
          let s =
            svg
              (Picture.stamp ~scales:[| 1.; 0.; 1. |] [| 10.; 20.; Float.nan |]
                 [| 5.; 5.; 5. |] two)
          in
          equal int ~msg:"written in full" 2 (count s "<path"));
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
      test "a run as outlines within a use that sets strokes is not stroked"
        (fun () ->
          let r =
            Run.v ~clusters:[| 0 |] ~font:Font.regular ~size:10. ~text:"fi"
              ~glyphs:[| glyph 'f' |]
              ~xs:[| 0. |] ()
          in
          let p = Picture.glyphs red (P2.v 0. 0.) r in
          equal (list string) [] (attrs "stroke" (svg p));
          equal (list string) [ "none"; "#ff0000" ]
            (attrs "stroke"
               (svg (Picture.stamp ~strokes:[| red |] [| 10. |] [| 5. |] p))));
      cases ~name:fst "an instance changes nothing"
        [
          ("at x = NaN", (Float.nan, 5., 1000.));
          ("at y = NaN", (20., Float.nan, 1000.));
          ("at y = infinity", (20., Float.infinity, 1000.));
          ("of scale NaN", (20., 5., Float.nan));
          ("of scale infinity", (20., 5., Float.infinity));
        ]
        (fun (_, (x, y, k)) ->
          equal string
            (svg (Picture.stamp ~scales:[| 2. |] [| 10. |] [| 5. |] marker))
            (svg
               (Picture.stamp ~scales:[| 2.; k |] [| 10.; x |] [| 5.; y |]
                  marker)));
      test "equal stamps share their definition" (fun () ->
          let st = Picture.stamp [| 10. |] [| 5. |] marker in
          let s =
            svg
              (Picture.group
                 [ st; Picture.transform (Affine.translate 0. 20.) st ])
          in
          equal int 1 (count s "<g id=");
          equal int 2 (count s "<use"));
    ]

(* Tags *)

let tagged rows p =
  Picture.tag
    { id = Nx.Ptree.Path.v [ Index 0; Field "a.b"; Field "q\"" ]; rows }
    p

let tags =
  group "tags"
    [
      test "data-id is the id's segments as JSON" (fun () ->
          let s = svg (tagged (Rows [||]) (Picture.fill red square)) in
          equal string "[0,&quot;a.b&quot;,&quot;q\\&quot;&quot;]"
            (attr "data-id" s);
          equal (list string) [] (attrs "data-rows" s));
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
      test "rows of a stamp are given instance by instance" (fun () ->
          let st = Picture.stamp [| 10.; 20. |] [| 5.; 5. |] marker in
          let s = svg (tagged (Rows [| 7; 9 |]) st) in
          equal (list string) [ "7"; "9" ] (attrs "data-row" s));
      test "rows of a stamp written in full are given instance by instance"
        (fun () ->
          let two =
            Picture.group
              [
                Picture.stroke (Stroke.v 1.) red disc;
                Picture.stroke (Stroke.v 2.) red disc;
              ]
          in
          let st =
            Picture.stamp ~scales:[| 1.; 2. |] [| 10.; 20. |] [| 5.; 5. |] two
          in
          equal (list string) [ "7"; "9" ]
            (attrs "data-row" (svg (tagged (Rows [| 7; 9 |]) st))));
      test "cells give their box and grid, and the map above them" (fun () ->
          let px = Nx.zeros Nx.uint8 [| 3; 4; 3 |] in
          let box = Box2.v 0.5 1. 4. 3. in
          let p =
            tagged (Cells { box; width = 4; height = 3 }) (Picture.image box px)
          in
          let s = svg p in
          equal string "0.5 1 4 3 4 3" (attr "data-cells" s);
          equal (list string) [] (attrs "data-matrix" s);
          let s = svg (Picture.transform (Affine.translate 10. 0.1) p) in
          equal string "1 0 0 1 10 0.1" (attr "data-matrix" s));
      test "cells in a stamp's picture written once map to the instance"
        (fun () ->
          let px = Nx.zeros Nx.uint8 [| 3; 4; 3 |] in
          let box = Box2.v 0.5 1. 4. 3. in
          let p =
            tagged (Cells { box; width = 4; height = 3 }) (Picture.image box px)
          in
          let s =
            svg
              (Picture.stamp [| 10.; 20. |] [| 5.; 5. |]
                 (Picture.transform (Affine.scale 2. 2.) p))
          in
          equal string "2 0 0 2 0 0" (attr "data-matrix" s);
          equal int 2 (count s "<use"));
    ]

(* Limits *)

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
  equal int ~msg:"inf" 0 (count text "inf");
  equal int ~msg:"nan" 0 (count text "nan")

let limits =
  group "limits"
    [
      cases ~name:fst
        "transforms whose composition overflows or underflows paint nothing"
        [ ("overflows", 1e200); ("underflows", 1e-200) ]
        (fun (_, k) ->
          let twice p =
            Picture.transform (Affine.scale k k)
              (Picture.transform (Affine.scale k k) p)
          in
          equal int 0 (count (svg (twice (Picture.fill red square))) "<path"));
      test "a number beyond 1e15 is written as 1e15" (fun () ->
          equal string "1000000000000000" (attr "stroke-width" (stroked 1e300)));
      prop ~examples:Vg_corpus.extremes "numbers are finite whatever the scales"
        Vg_corpus.gen_extreme (fun p -> numbers_finite (svg p));
      cases ~name:(Printf.sprintf "under a scale of %g")
        "a pen keeps its width" [ 1e-170; 1e170 ] (fun k ->
          let p =
            Picture.transform (Affine.scale k k)
              (Picture.stroke
                 (Stroke.v (10. /. k))
                 red
                 (Path.polyline [| 0.; 50. /. k |] [| 50. /. k; 50. /. k |]))
          in
          let s = svg p in
          equal string "10" (attr "stroke-width" s);
          equal string "M0 50L50 50" (attr "d" s));
    ]

(* Determinism *)

let determinism =
  group "determinism"
    [
      prop "equal renderables give equal documents" Vg_corpus.gen_picture
        (fun p -> equal string (svg p) (svg (Vg_corpus.respell p)));
      test "a number that rounds to zero is written 0" (fun () ->
          let s =
            svg
              (Picture.fill red
                 (Path.polygon [| -0.; 10.; -0.0001 |] [| -0.; 0.; 10. |]))
          in
          equal string "M0 0L10 0L0 10Z" (attr "d" s));
      test "ids derive from content, alike across documents" (fun () ->
          let clip q =
            Picture.clip q (Picture.fill red (rect 0. 0. 100. 100.))
          in
          let a = svg (clip square)
          and b =
            svg (Picture.group [ Picture.fill red square; clip square ])
          in
          equal string (attr "id" a) (attr "id" b);
          let c = svg (clip (rect 0. 0. 1. 1.)) in
          is_false ~msg:"other content, other id"
            (String.equal (attr "id" a) (attr "id" c)));
    ]

(* Goldens *)

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

let goldens =
  group "goldens"
    (List.map
       (fun (name, r) ->
         test (name ^ " writes its golden document") (fun () ->
             expect_file
               (masked (Hugin_vg_svg.render r))
               ("packages/hugin/test/vg/golden/" ^ name ^ ".svg")))
       Vg_corpus.pages)

let () =
  exit
    (run "hugin.vg svg"
       [
         document; leaves; accuracy; stamps; tags; limits; determinism; goldens;
       ])
