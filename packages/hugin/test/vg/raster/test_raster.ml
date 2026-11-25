(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_gg
open Hugin_font
open Hugin_vg

let render ?(density = 1.) w h p =
  Hugin_vg_raster.render ~density (Renderable.v w h p)

let rgba img y x =
  ( Nx.item [ y; x; 0 ] img,
    Nx.item [ y; x; 1 ] img,
    Nx.item [ y; x; 2 ] img,
    Nx.item [ y; x; 3 ] img )

let alpha img y x = Nx.item [ y; x; 3 ] img
let color = quad int int int int
let picture = Testable.make ~pp:Picture.pp ~equal:Picture.equal

(* The sum of the alphas of [img], in pixels. *)
let coverage img =
  let a = Nx.to_array img in
  let s = ref 0 in
  Array.iteri (fun i v -> if i mod 4 = 3 then s := !s + v) a;
  Float.of_int !s /. 255.

(* [levels a b] is the largest difference between the bytes of [a] and [b]. *)
let levels a b =
  let a = Nx.to_array a and b = Nx.to_array b in
  let d = ref 0 in
  Array.iteri (fun i v -> d := Int.max !d (abs (v - b.(i)))) a;
  !d

let same_image ?(within = 0) a b =
  equal (array int) (Nx.shape a) (Nx.shape b);
  at_most ~msg:"largest difference in levels" int ~than:within (levels a b)

(* Images compare level by level, and print as their size and the pixels that
   hold ink. *)
let raster =
  let pp ppf img =
    let s = Nx.shape img and a = Nx.to_array img in
    Format.fprintf ppf "@[<v>%d by %d pixels" s.(1) s.(0);
    Array.iteri
      (fun k v ->
        if k mod 4 = 3 && v > 0 then
          Format.fprintf ppf "@,(%d, %d) %d %d %d %d"
            (k / 4 / s.(1))
            (k / 4 mod s.(1))
            a.(k - 3)
            a.(k - 2)
            a.(k - 1)
            v)
      a;
    Format.fprintf ppf "@]"
  in
  Testable.make ~pp ~equal:(fun a b ->
      Nx.shape a = Nx.shape b && levels a b = 0)

(* [area ?arc ?drift ?pieces ~box exact got] asserts that the coverage [got] is
   the area [exact] of [pieces] shapes, each within a box [box] pixels wide and
   high and crossing each row and column of pixels in one segment, whose round
   parts, [arc] long, are flattened, and whose edges may lie off by [drift]
   square pixels in all. Each such outline crosses each grid line through its
   box at most twice, so it enters at most 2 (w + h) + 4 pixels, each rounded to
   its nearest level, by half a level at most; the pixels within are whole.
   Flattening to within a tenth of a pixel loses at most a tenth of the arc's
   length. *)
let area ?(arc = 0.) ?(drift = 0.) ?(pieces = 1) ~box:(w, h) exact got =
  let rounding =
    (Float.of_int pieces *. ((2. *. (w +. h)) +. 4.) *. 0.5 /. 255.) +. drift
  in
  at_most ~msg:"no more than the shape" float_exact ~than:(exact +. rounding)
    got;
  at_least ~msg:"no less than its chords" float_exact
    ~than:(exact -. (0.1 *. arc) -. rounding)
    got

(* [near name exact got] asserts that the level [got] is the level nearest the
   value [exact] in [0;1], up to the drift of compositing in single precision,
   about a thousandth of a level over a thousand blends. *)
let near name exact got =
  at_most ~msg:name float_exact ~than:(0.5 +. 1e-3)
    (Float.abs (Float.of_int got -. (255. *. exact)))

let red = Color.red
let blue = Color.blue
let rect x y w h = Path.rect (Box2.v x y w h)
let solid w h = Picture.fill red (rect 0. 0. w h)
let segment x0 y0 x1 y1 = Path.polyline [| x0; x1 |] [| y0; y1 |]

(* Generators *)

let gen_color = Gen.with_pp Color.pp Vg_corpus.gen_color

let pp_rule ppf r =
  Format.pp_print_string ppf
    (match r with `Nonzero -> "nonzero" | `Even_odd -> "even-odd")

let gen_rule = Gen.of_list ~pp:pp_rule [ `Nonzero; `Even_odd ]
let gen_in lo hi = Gen.float_range lo hi
let gen_points n lo hi = Gen.array ~size:(Gen.constant n) (gen_in lo hi)

(* Closed paths that bound areas of the page [0;30] with straight edges:
   triangles, rectangles and quadrilaterals that may cross themselves. *)
let polygons =
  let open Gen in
  let coord = gen_in 0. 30. in
  let points n = pair (gen_points n 0. 30.) (gen_points n 0. 30.) in
  [
    map (fun (xs, ys) -> Path.polygon xs ys) (points 3);
    map (fun (xs, ys) -> Path.polygon xs ys) (points 4);
    map
      (fun ((x, y), (w, h)) -> rect x y w h)
      (pair (pair coord coord) (pair (gen_in 0. 20.) (gen_in 0. 20.)));
  ]

let gen_polygon = Gen.with_pp Path.pp (Gen.one_of polygons)

(* The paths of [gen_polygon], open triangles, circles and rings. *)
let gen_area =
  let open Gen in
  let centre = pair (gen_in 0. 30.) (gen_in 0. 30.) and r = gen_in 0. 15. in
  let circle ((x, y), r) = Path.circle (P2.v x y) r in
  with_pp Path.pp
    (one_of
       (map
          (fun (xs, ys) -> Path.polyline xs ys)
          (pair (gen_points 3 0. 30.) (gen_points 3 0. 30.))
       :: map circle (pair centre r)
       :: map
            (fun (c, (r, r')) -> Path.append (circle (c, r)) (circle (c, r')))
            (pair centre (pair r r))
       :: polygons))

(* Boxes with whole coordinates, on pixel boundaries. *)
let gen_cell_box =
  let open Gen in
  let+ x = int_range (-2) 30
  and+ y = int_range (-2) 30
  and+ w = int_range 0 20
  and+ h = int_range 0 20 in
  Box2.v (Float.of_int x) (Float.of_int y) (Float.of_int w) (Float.of_int h)

let gen_cell_box = Gen.with_pp Box2.pp gen_cell_box

(* Fills *)

let rect_area (x, y, w, h) =
  let img = render 30. 30. (Picture.fill red (rect x y w h)) in
  area ~box:(w, h) (w *. h) (coverage img)

let gen_rect =
  let open Gen in
  let+ x = gen_in 0. 15.
  and+ y = gen_in 0. 15.
  and+ w = gen_in 0. 14.
  and+ h = gen_in 0. 14. in
  (x, y, w, h)

(* A triangle, an open subpath half of the time, covers its area under either
   rule, whichever way it turns. *)
let triangle_area ((xs, ys), closed, rule) =
  let q = if closed then Path.polygon xs ys else Path.polyline xs ys in
  let twice =
    ((xs.(1) -. xs.(0)) *. (ys.(2) -. ys.(0)))
    -. ((xs.(2) -. xs.(0)) *. (ys.(1) -. ys.(0)))
  in
  cover "clockwise" (twice > 0.);
  cover "counter-clockwise" (twice < 0.);
  cover "open" (not closed);
  let extent a =
    Array.fold_left Float.max a.(0) a -. Array.fold_left Float.min a.(0) a
  in
  area
    ~box:(extent xs, extent ys)
    (Float.abs twice /. 2.)
    (coverage (render 30. 30. (Picture.fill ~rule red q)))

let fills =
  group "fills"
    [
      prop "a filled rectangle covers its area" gen_rect rect_area;
      prop "a filled triangle covers its area"
        (Gen.triple
           (Gen.pair (gen_points 3 0. 30.) (gen_points 3 0. 30.))
           Gen.bool gen_rule)
        triangle_area;
      test "a pixel takes the fraction of its square that a fill covers"
        (fun () ->
          let img = render 1. 1. (Picture.fill red (rect 0.25 0. 0.75 1.)) in
          equal color (255, 0, 0, 191) (rgba img 0 0));
      test "even-odd leaves a hole where nonzero fills" (fun () ->
          (* Squares on whole pixels: 100 of them, and 84 around the hole of
             16. *)
          let both = Path.append (rect 3. 3. 4. 4.) (rect 0. 0. 10. 10.) in
          equal float_exact 100.
            (coverage (render 10. 10. (Picture.fill red both)));
          equal float_exact 84.
            (coverage (render 10. 10. (Picture.fill ~rule:`Even_odd red both))));
      test "a circle covers its area, its chords within a tenth of a pixel"
        (fun () ->
          area
            ~arc:(2. *. Float.pi *. 15.)
            ~box:(30., 30.)
            (Float.pi *. 15. *. 15.)
            (coverage
               (render 40. 40.
                  (Picture.fill red (Path.circle (P2.v 20. 20.) 15.)))));
    ]

(* Glyphs *)

let glyphs =
  group "glyphs"
    [
      test "glyphs are their outlines, each at its origin" (fun () ->
          let font = Font.regular in
          let typeset = Vg_corpus.typeset "Hug" 16. in
          let r =
            Run.v ~ys:[| 0.; 6.; -3. |] ~font ~size:16. ~text:"Hug"
              ~glyphs:(Array.init 3 (Run.glyph typeset))
              ~xs:(Array.init 3 (Run.x typeset))
              ()
          in
          let outline i =
            Picture.transform
              Affine.(
                translate (4. +. Run.x r i) (20. +. Run.y r i) * scale 16. 16.)
              (Picture.fill red (Font.outline font (Run.glyph r i)))
          in
          same_image ~within:1
            (render 60. 30. (Picture.group (List.init 3 outline)))
            (render 60. 30. (Picture.glyphs red (P2.v 4. 20.) r)));
      test "overlapping translucent glyphs composite twice" (fun () ->
          let g = Font.glyph Font.regular (Uchar.of_char 'I') in
          let r =
            Run.v ~font:Font.regular ~size:20. ~text:"II" ~glyphs:[| g; g |]
              ~xs:[| 0.; 0. |] ()
          in
          let img =
            render 20. 30.
              (Picture.glyphs (Color.with_alpha 0.5 red) (P2.v 5. 25.) r)
          in
          equal ~msg:"the most opaque pixel, 0.75 rounded once" int 191
            (Nx.item [] (Nx.max (Nx.slice [ A; A; I 3 ] img))));
    ]

(* Clips *)

(* Clips of axis-aligned sides around no area, in each order of their turns. *)
let flat_clips =
  List.map
    (fun (xs, ys) -> (`Nonzero, Path.polygon xs ys))
    [
      ([| 0.; 4.; 0.; 0. |], [| 0.; 0.; 0.; 4. |]);
      ([| 0.; 0.; 4.; 0. |], [| 0.; 4.; 4.; 4. |]);
      ([| 0.; 0.; 0.; 4. |], [| 0.; 4.; 0.; 0. |]);
      ([| 0.; 4.; 4.; 4. |], [| 0.; 0.; 4.; 0. |]);
    ]

(* Boxes on pixel boundaries beside and within circles. *)
let boxed_circles =
  let circle x y r = (`Nonzero, Path.circle (P2.v x y) r) in
  let box x y w h = (`Nonzero, rect x y w h) in
  [
    (box 6. 0. 2. 10., circle 3. 5. 2.);
    (box 8. 0. 2. 10., circle 3. 5. 2.);
    (box 2. 3. 6. 5., circle 5. 5. 4.);
  ]

(* A fill under a clip whose box misses its own: they meet the same row of
   pixels, but share no area. *)
let apart =
  Picture.clip (rect 2. 10.6 6. 0.4) (Picture.fill red (rect 2. 10. 6. 0.5))

let clips =
  group "clips"
    [
      prop "a clip lets through the area of its path"
        ~examples:
          ((`Nonzero, rect 40. 40. 5. 5.)
          :: ( `Nonzero,
               Path.polygon [| 0.; 10.; 10.; 0. |] [| 0.; 10.; 0.; 10. |] )
          :: (`Even_odd, Path.append (rect 3. 3. 4. 4.) (rect 0. 0. 10. 10.))
          :: flat_clips)
        (Gen.pair gen_rule gen_area)
        (fun (rule, q) ->
          same_image ~within:1
            (render 30. 30. (Picture.fill ~rule red q))
            (render 30. 30. (Picture.clip ~rule q (solid 30. 30.))));
      prop "a clip to a box draws what cropping to it draws"
        (Gen.pair gen_cell_box (Gen.pair gen_rule gen_polygon))
        (fun (b, (rule, q)) ->
          same_image ~within:1
            (render 30. 30. (Picture.fill ~rule red (Path.crop b q)))
            (render 30. 30.
               (Picture.clip (Path.rect b) (Picture.fill ~rule red q))));
      prop "nested clips commute" ~examples:boxed_circles
        (Gen.pair (Gen.pair gen_rule gen_area) (Gen.pair gen_rule gen_area))
        (fun ((r, a), (r', b)) ->
          let page = solid 30. 30. in
          same_image ~within:1
            (render 30. 30.
               (Picture.clip ~rule:r a (Picture.clip ~rule:r' b page)))
            (render 30. 30.
               (Picture.clip ~rule:r' b (Picture.clip ~rule:r a page))));
      test "nested clips let through the intersection of their areas" (fun () ->
          let p =
            Picture.clip (rect 0. 0. 6. 10.)
              (Picture.clip (rect 3. 0. 7. 10.) (solid 10. 10.))
          in
          equal float_exact 30. (coverage (render 10. 10. p)));
      cases "a box beside a curved clip shows nothing" ~id:fst
        [ ("next to its pixels", 6.); ("apart from its pixels", 8.) ]
        (fun (_, x) ->
          let circle = Path.circle (P2.v 3. 5.) 2. in
          let p =
            Picture.clip circle
              (Picture.clip (rect x 0. 2. 10.) (solid 10. 10.))
          in
          equal float_exact 0. (coverage (render 10. 10. p)));
      test "a fill beside its clip shows nothing" (fun () ->
          equal float_exact 0. (coverage (render 20. 20. apart)));
      test "a fill whose box only touches its clip's shows nothing" (fun () ->
          (* Both cover half of each pixel of row 10, which their product would
             paint a quarter. *)
          let p =
            Picture.clip (rect 2. 10.5 6. 0.5)
              (Picture.fill red (rect 2. 10. 6. 0.5))
          in
          equal float_exact 0. (coverage (render 20. 20. p)));
      test "a clipped fill takes the product of its coverage and the clip's"
        (fun () ->
          (* Each covers the left half of the pixel: a product of 0.25, where
             their intersection would be 0.5. *)
          equal color (255, 0, 0, 64)
            (rgba
               (render 1. 1.
                  (Picture.clip (rect 0. 0. 0.5 1.)
                     (Picture.fill red (rect 0. 0. 0.5 1.))))
               0 0));
    ]

(* Strokes *)

(* The shapes below put the edges where a stroke's pieces meet on pixel
   boundaries, where the sum of their coverages is that of their union. *)

let line = segment 5. 10. 15. 10.
let stroked s = coverage (render 20. 20. (Picture.stroke s red line))

let caps =
  [
    ("butt caps end at the ends", `Butt, 40., 0., 10.);
    ("square caps add half a square at each end", `Square, 56., 0., 14.);
    ( "round caps add half a disc at each end",
      `Round,
      40. +. (Float.pi *. 4.),
      Float.pi *. 4.,
      14. );
  ]

(* A right-angle corner at (10, 10), drawn 4 wide with butt caps: two bodies of
   32 sharing a square of 4, and the join in the outer corner's square, all
   within a box of 10 by 10. *)
let corner = Path.polyline [| 2.; 10.; 10. |] [| 10.; 10.; 2. |]

let joins =
  [
    ("a miter fills the outer square", (`Miter, 4.), 64., 0.);
    ("a bevel fills half of it", (`Bevel, 4.), 62., 0.);
    ( "a round join fills a quarter disc",
      (`Round, 4.),
      60. +. Float.pi,
      Float.pi );
    ("a miter beyond the limit is bevelled", (`Miter, 1.2), 62., 0.);
  ]

(* Dashes of a line 2 wide along y = 5, on whole pixels. *)
let dashes =
  [
    ("dashes and gaps", ([ 4.; 2. ], 0.), 28.);
    ("an offset into the pattern", ([ 4.; 2. ], 1.), 28.);
    ("an odd pattern, repeated", ([ 3. ], 0.), 22.);
    ("a negative offset", ([ 4.; 2. ], -5.), 28.);
  ]

let dashed (pattern, dash_offset) =
  let s = Stroke.v ~cap:`Butt ~dash:pattern ~dash_offset 2. in
  coverage (render 20. 10. (Picture.stroke s red (segment 0. 5. 20. 5.)))

let point =
  Path.empty |> Path.move_to (P2.v 10. 10.) |> Path.line_to (P2.v 10. 10.)

(* Dashes of zero length every 4 along a line 8 long: three dots, each within a
   box of 2 by 2. *)
let zero_dashes =
  [
    ( "squares along a line with square caps",
      `Square,
      segment 2. 5. 10. 5.,
      (12., 0., 3) );
    ( "squares along a vertical line",
      `Square,
      segment 5. 2. 5. 10.,
      (12., 0., 3) );
    ( "discs with round caps",
      `Round,
      segment 2. 5. 10. 5.,
      (3. *. Float.pi, 6. *. Float.pi, 3) );
    ("nothing with butt caps", `Butt, segment 2. 5. 10. 5., (0., 0., 0));
  ]

(* [shrunk join r] is the ink of a circle of radius [r] stroked 4 wide. *)
let shrunk join r =
  coverage
    (render 20. 20.
       (Picture.stroke (Stroke.v ~join 4.) red (Path.circle (P2.v 10. 10.) r)))

let sharp_corner () =
  (* Turning by 160 degrees, a miter would reach 5.8 widths out. *)
  let a = 160. *. Float.pi /. 180. in
  let q =
    Path.polyline
      [| 2.; 12.; 12. +. (10. *. cos a) |]
      [| 10.; 10.; 10. +. (10. *. sin a) |]
  in
  let draw join =
    render 30. 20.
      (Picture.stroke (Stroke.v ~cap:`Butt ~join ~miter_limit:4. 2.) red q)
  in
  same_image (draw `Bevel) (draw `Miter)

let turned_dash () =
  (* A square turned by 45 degrees about (5, 5), within a box 2 sqrt 2 wide,
     covers part of the pixel at its centre's lower right, which an upright one
     covers whole. *)
  let s = Stroke.v ~cap:`Square ~dash:[ 0.; 100. ] 2. in
  let img =
    render 20. 20.
      (Picture.transform (Affine.translate 5. 5.)
         (Picture.stroke s red (segment 0. 0. 10. 10.)))
  in
  let side = 2. *. Float.sqrt 2. in
  area ~box:(side, side) 4. (coverage img);
  less ~msg:"alpha of pixel (5, 5)" int ~than:255 (alpha img 5 5)

let turned_zigzag () =
  (* Its segments run along the diagonals, which a turn by 45 degrees lays along
     the axes. Butt caps and bevels have no round part to flatten, so the turned
     pieces are the pieces of the turned path. *)
  let zig =
    Path.polyline [| 0.; 3.; 6.; 9.; 12. |] [| 0.; -3.; 0.; -3.; 0. |]
  in
  let s = Stroke.v ~cap:`Butt ~join:`Bevel 1. in
  let m = Affine.(translate 14. 20. * rotate (Float.pi /. 4.)) in
  same_image ~within:1
    (render 40. 40. (Picture.stroke s red (Path.transform m zig)))
    (render 40. 40. (Picture.transform m (Picture.stroke s red zig)))

let repeated_point (s, xs, again) =
  let draw xs =
    render 20. 10.
      (Picture.stroke s red
         (Path.polyline xs (Array.make (Array.length xs) 5.)))
  in
  let repeated =
    Array.concat
      (Array.to_list
         (Array.map (fun x -> if x = again then [| x; x |] else [| x |]) xs))
  in
  same_image (draw xs) (draw repeated)

(* The map stretches by 3 along the diagonal the line follows, so a period of
   [p] is [3 p] pixels there. *)
let fine_dash dash =
  let m = { Affine.xx = 1.; yx = 2.; xy = 2.; yy = 1.; x0 = 5.; y0 = 5. } in
  render 30. 30.
    (Picture.transform m
       (Picture.stroke (Stroke.v ~cap:`Butt ~dash 1.) red (segment 0. 0. 4. 4.)))

let tiny_polygons join =
  let draw xs ys =
    render 20. 20. (Picture.stroke (Stroke.v ~join 4.) red (Path.polygon xs ys))
  in
  let triangle = draw [| 10.; 10.01; 10. |] [| 10.; 10.; 10.01 |] in
  same_image triangle
    (draw [| 10.; 10.; 10.01; 10. |] [| 10.; 10.; 10.; 10.01 |]);
  same_image triangle
    (draw [| 10.; 10.01; 10.01; 10. |] [| 10.; 10.; 10.; 10.01 |])

let growing join =
  ignore
    (List.fold_left
       (fun before r ->
         let after = shrunk join r in
         at_least
           ~msg:(Printf.sprintf "radius %g" r)
           float_exact ~than:before after;
         after)
       (shrunk join 0.005) [ 0.01; 0.03; 0.06; 0.12 ])

let small_joins =
  [ ("miter joins", `Miter); ("round joins", `Round); ("bevels", `Bevel) ]

let strokes =
  group "strokes"
    [
      cases "a line 4 wide:"
        ~id:(fun (n, _, _, _, _) -> n)
        caps
        (fun (_, cap, a, arc, w) ->
          area ~arc ~box:(w, 4.) a (stroked (Stroke.v ~cap 4.)));
      cases "at a right-angle corner,"
        ~id:(fun (n, _, _, _) -> n)
        joins
        (fun (_, (join, miter_limit), a, arc) ->
          let s = Stroke.v ~cap:`Butt ~join ~miter_limit 4. in
          let p = Picture.stroke s red corner in
          area ~arc ~box:(10., 10.) a (coverage (render 20. 20. p)));
      test "a corner sharper than the miter limit is bevelled" sharp_corner;
      test "a closed subpath joins at its start" (fun () ->
          (* An 8 by 8 square stroked 2 wide with mitred corners is a 10 by 10
             frame around a 6 by 6 hole, on whole pixels. *)
          let p =
            Picture.stroke (Stroke.v ~join:`Miter 2.) red (rect 4. 4. 8. 8.)
          in
          equal float_exact 64. (coverage (render 20. 20. p)));
      cases "a dashed line 20 long and 2 wide, with"
        ~id:(fun (n, _, _) -> n)
        dashes
        (fun (_, d, a) -> equal float_exact a (dashed d));
      test "a dashed closed subpath dashes its closing side" (fun () ->
          (* Each side of 10 holds two dashes of 3 on whole points, 1 wide
             across a pixel boundary: 8 dashes of 6 half pixels, each rounded to
             128 of 255. *)
          let s = Stroke.v ~cap:`Butt ~dash:[ 3.; 2. ] 1. in
          equal float_exact
            (Float.of_int (8 * 6 * 128) /. 255.)
            (coverage
               (render 20. 20. (Picture.stroke s red (rect 5. 5. 10. 10.)))));
      test "a dash pattern of a thousandth of a pixel or more is dashed"
        (fun () ->
          let solid = coverage (fine_dash []) in
          greater ~msg:"solid" float_exact ~than:10. solid;
          at_most ~msg:"dashed" float_exact ~than:(0.5 *. solid)
            (coverage (fine_dash [ 0.0002; 0.0002 ])));
      test "a dash pattern of period below a thousandth of a pixel is solid"
        (fun () -> same_image (fine_dash []) (fine_dash [ 0.0001; 0.0001 ]));
      test "a subpath of zero length is a disc with round caps" (fun () ->
          area ~arc:(4. *. Float.pi) ~box:(4., 4.) (4. *. Float.pi)
            (coverage
               (render 20. 20.
                  (Picture.stroke (Stroke.v ~cap:`Round 4.) red point))));
      cases "a subpath of zero length is nothing with" ~id:fst
        [ ("butt caps", `Butt); ("square caps, having no direction", `Square) ]
        (fun (_, cap) ->
          equal float_exact 0.
            (coverage
               (render 20. 20. (Picture.stroke (Stroke.v ~cap 4.) red point))));
      test "a disc of a subpath of zero length adds to the other subpaths"
        (fun () ->
          (* A line 12 long and 4 wide holds the disc of the point at its
             middle, so that the union is the line. *)
          let q =
            Path.(
              empty
              |> move_to (P2.v 4. 10.)
              |> line_to (P2.v 16. 10.)
              |> move_to (P2.v 10. 10.)
              |> line_to (P2.v 10. 10.))
          in
          let p = Picture.stroke (Stroke.v ~cap:`Round 4.) red q in
          area ~arc:(4. *. Float.pi) ~box:(16., 4.)
            (48. +. (4. *. Float.pi))
            (coverage (render 20. 20. p)));
      test "a dashed subpath of a start alone is a disc with round caps"
        (fun () ->
          let lone = Path.empty |> Path.move_to (P2.v 10. 10.) in
          let p =
            Picture.stroke (Stroke.v ~cap:`Round ~dash:[ 2.; 2. ] 4.) red lone
          in
          area ~arc:(4. *. Float.pi) ~box:(4., 4.) (4. *. Float.pi)
            (coverage (render 20. 20. p)));
      cases "dashes of zero length are"
        ~id:(fun (n, _, _, _) -> n)
        zero_dashes
        (fun (_, cap, q, (a, arc, pieces)) ->
          let p = Picture.stroke (Stroke.v ~cap ~dash:[ 0.; 4. ] 2.) red q in
          area ~arc ~pieces ~box:(2., 2.) a (coverage (render 20. 20. p)));
      test "a dash of zero length takes the direction of its line" turned_dash;
      cases "a line too short to see is a square with square caps" ~id:fst
        [ ("vertical", (0., 0.01)); ("horizontal", (0.01, 0.)) ]
        (fun (_, (dx, dy)) ->
          (* Its end, within a twentieth of a pixel, is dropped: its caps alone
             make a square of 2 on whole pixels. *)
          let p =
            Picture.stroke (Stroke.v ~cap:`Square 2.) red
              (segment 10. 10. (10. +. dx) (10. +. dy))
          in
          equal float_exact 4. (coverage (render 20. 20. p)));
      cases "a repeated point leaves a dashed line as it is" ~id:fst
        [
          ( "within it",
            (Stroke.v ~cap:`Butt ~dash:[ 4.; 2. ] 2., [| 0.; 10.; 20. |], 10.)
          );
          (* A dash starts at its end, where its square caps make a square. *)
          ( "at its end",
            (Stroke.v ~cap:`Square ~dash:[ 0.01; 4.99 ] 2., [| 2.; 12. |], 12.)
          );
        ]
        (fun (_, c) -> repeated_point c);
      test "a closed subpath smaller than a pixel keeps its pen" (fun () ->
          (* Stroked 8 wide with mitred corners, a square of side 0.004 is a
             square of side 8.004, without a hole. *)
          let s = Stroke.v ~cap:`Butt ~join:`Miter 8. in
          let p = Picture.stroke s red (rect 9.998 9.998 0.004 0.004) in
          area ~box:(8.004, 8.004) (8.004 *. 8.004)
            (coverage (render 20. 20. p)));
      cases "a closed subpath smaller than a pixel ignores a repeated vertex,"
        ~id:fst small_joins (fun (_, join) -> tiny_polygons join);
      cases "a closed subpath smaller than a pixel draws more ink as it grows,"
        ~id:fst small_joins (fun (_, join) -> growing join);
      cases "a mitred circle smaller than a pixel covers at least its pen"
        ~id:(Printf.sprintf "radius %g") [ 0.001; 0.01; 0.03; 0.06 ] (fun r ->
          at_least float_exact ~than:(4. *. Float.pi) (shrunk `Miter r));
      test "a transform maps the pen" (fun () ->
          let p =
            Picture.transform (Affine.scale 1. 4.)
              (Picture.stroke (Stroke.v ~cap:`Butt 1.) red line)
          in
          (* A rectangle of 10 by 4 on whole pixels. *)
          equal float_exact 40. (coverage (render 20. 60. p)));
      test "a pen under a shear covers the area the map gives it" (fun () ->
          (* A ring of radius 5 and width 1 has area 10 pi, which the map
             multiplies by its determinant, 4: its outlines turn into ellipses
             within boxes of 11 by 44 and 9 by 36, at most 110 and 90 pixels
             around. Flattened to within a tenth of a pixel, the outer one loses
             at most a tenth of its length and the inner one gains as much.
             Where the pieces of two segments overlap, inside their join, a
             pixel both their edges cross counts the overlap twice: (1/2)^2 tan
             (t/2) for a turn of t, pi / 4 over the whole turn, mapped to pi. *)
          let p =
            Picture.transform
              Affine.(translate 20. 30. * scale 1. 4. * rotate (Float.pi /. 6.))
              (Picture.stroke (Stroke.v 1.) red (Path.circle (P2.v 0. 0.) 5.))
          in
          let got = coverage (render 40. 60. p) in
          let rounding =
            ((2. *. (11. +. 44.)) +. (2. *. (9. +. 36.)) +. 8.) *. 0.5 /. 255.
          in
          let ring = 40. *. Float.pi in
          at_least ~msg:"the outer chords" float_exact
            ~than:(ring -. 11. -. rounding)
            got;
          at_most ~msg:"the inner chords and the joins" float_exact
            ~than:(ring +. 9. +. Float.pi +. rounding)
            got);
      test "a turn draws a stroke as the stroke of the turned path"
        turned_zigzag;
    ]

(* Compositing *)

(* [over_formula (c, c')] checks a pixel painted [c'] then [c] against the
   formula of source-over, in premultiplied components: each component is the
   level nearest the exact colour, straight. *)
let over_formula (c, c') =
  let img =
    render 1. 1.
      (Picture.group
         [
           Picture.fill c' (rect 0. 0. 1. 1.); Picture.fill c (rect 0. 0. 1. 1.);
         ])
  in
  let a = Color.alpha c and a' = Color.alpha c' in
  let alpha = a +. (a' *. (1. -. a)) in
  let straight k =
    (* A pixel whose alpha rounds to zero is (0, 0, 0, 0). *)
    if 255. *. alpha < 0.5 then 0.
    else ((a *. k c) +. (a' *. (1. -. a) *. k c')) /. alpha
  in
  let r, g, b, alpha' = rgba img 0 0 in
  near "alpha" alpha alpha';
  near "red" (straight Color.r) r;
  near "green" (straight Color.g) g;
  near "blue" (straight Color.b) b

(* A page painted with a backdrop, if any, then with [count] fills of the whole
   page, each a colour of [palette] in turn faded by [opacity], as the alpha of
   its colour or as a group opacity. *)
type dense = {
  backdrop : Color.t option;
  palette : Color.t list;
  opacity : float;
  count : int;
  as_group : bool;
}

let pp_dense ppf d =
  Format.fprintf ppf
    "@[<v>backdrop %a@,palette %a@,opacity %g@,count %d@,as group %b@]"
    (Format.pp_print_option
       ~none:(fun ppf () -> Format.pp_print_string ppf "none")
       Color.pp)
    d.backdrop
    (Format.pp_print_list ~pp_sep:Format.pp_print_space Color.pp)
    d.palette d.opacity d.count d.as_group

let gen_dense =
  let open Gen in
  let opaque =
    map
      (fun (r, g, b) -> Color.v r g b)
      (triple (gen_in 0. 1.) (gen_in 0. 1.) (gen_in 0. 1.))
  in
  let opacity =
    frequency
      [
        (1, of_list ~pp:Format.pp_print_float [ 0.02; 1. ]); (4, gen_in 0.02 1.);
      ]
  in
  let count = frequency [ (1, int_range 1 4); (3, int_range 1 1000) ] in
  with_pp pp_dense
    (map
       (fun ((backdrop, palette), (opacity, count, as_group)) ->
         { backdrop; palette; opacity; count; as_group })
       (pair
          (pair (option opaque) (list ~size:(int_range 1 3) opaque))
          (triple opacity count bool)))

(* The ends of [gen_dense]'s range, which it draws too rarely to reach on every
   run: a thousand of the lightest fills over nothing, and opaque groups over a
   backdrop. *)
let dense_ends =
  [
    {
      backdrop = None;
      palette = [ Color.v 0.9 0.3 0.1; Color.v 0.2 0.6 0.8 ];
      opacity = 0.02;
      count = 1000;
      as_group = false;
    };
    {
      backdrop = Some (Color.v 0.1 0.2 0.3);
      palette = [ Color.v 0.7 0.5 0.2 ];
      opacity = 1.;
      count = 3;
      as_group = true;
    };
  ]

(* [dense_within_a_level d] checks the page of [d] against source-over in exact
   arithmetic: each component of the pixel is the level nearest the exact
   colour, straight. *)
let dense_within_a_level d =
  cover "the lightest opacity" (d.opacity = 0.02);
  cover "an opaque opacity" (d.opacity = 1.);
  cover "a transparent backdrop" (d.backdrop = None);
  cover "an opaque backdrop" (d.backdrop <> None);
  cover "group opacities" d.as_group;
  cover "hundreds of light fills" (d.opacity < 0.05 && d.count > 200);
  let whole = rect 0. 0. 1. 1. in
  let palette = Array.of_list d.palette in
  let colour i = palette.(i mod Array.length palette) in
  let shape i =
    if d.as_group then Picture.opacity d.opacity (Picture.fill (colour i) whole)
    else Picture.fill (Color.with_alpha d.opacity (colour i)) whole
  in
  let backdrop =
    match d.backdrop with None -> [] | Some c -> [ Picture.fill c whole ]
  in
  let img = render 1. 1. (Picture.group (backdrop @ List.init d.count shape)) in
  (* Premultiplied, in [0;1]. *)
  let over (r, g, b, a) c =
    let k = d.opacity and k' = 1. -. d.opacity in
    ( (k *. Color.r c) +. (k' *. r),
      (k *. Color.g c) +. (k' *. g),
      (k *. Color.b c) +. (k' *. b),
      k +. (k' *. a) )
  in
  let start =
    match d.backdrop with
    | None -> (0., 0., 0., 0.)
    | Some c -> (Color.r c, Color.g c, Color.b c, 1.)
  in
  let r, g, b, a = List.fold_left over start (List.init d.count colour) in
  let r', g', b', a' = rgba img 0 0 in
  near "alpha" a a';
  near "red" (r /. a) r';
  near "green" (g /. a) g';
  near "blue" (b /. a) b'

(* [pixel x p] is pixel [x] of the page 3 wide and 1 high that [p] draws. *)
let pixel x p = rgba (render 3. 1. p) 0 x
let px x c = Picture.fill c (rect x 0. 1. 1.)
let bar x c = Picture.fill c (rect x 0. 2. 1.)
let half = Color.with_alpha 0.5

(* Levels of one pixel that [render] states: (name, colour drawn, level). *)
let nearest_levels =
  let grey g a = Color.v ~alpha:a g g g in
  [
    ("with straight alpha", half red, (255, 0, 0, 128));
    ( "the level nearest each component",
      grey (200.7 /. 255.) (100.8 /. 255.),
      (201, 201, 201, 101) );
    (* Premultiplied by an alpha of 153 levels, a grey of 31/153 is 31 levels
       exactly; straight, it is 51.67 levels. *)
    ( "the nearest level of a translucent colour, straight",
      grey (31. /. 153.) 0.6,
      (52, 52, 52, 153) );
  ]

(* Opacities over pixels: (name, picture, pixel column, colour). *)
let layers =
  [
    ( "a translucent fill fades by it",
      Picture.opacity 0.5 (px 0. (half red)),
      0,
      (255, 0, 0, 64) );
    ( "the overlap of a group shows its later part only",
      Picture.opacity 0.5 (Picture.group [ bar 0. red; bar 1. blue ]),
      1,
      (0, 0, 255, 128) );
    ( "each part of a group fades alike",
      Picture.opacity 0.5 (Picture.group [ bar 0. red; bar 1. blue ]),
      0,
      (255, 0, 0, 128) );
    ( "an opacity on each part lets the earlier show through",
      Picture.group
        [ Picture.opacity 0.5 (bar 0. red); Picture.opacity 0.5 (bar 1. blue) ],
      1,
      (85, 0, 170, 191) );
    ( "an opacity of 0 paints nothing",
      Picture.opacity 0. (px 0. red),
      0,
      (0, 0, 0, 0) );
  ]

let compositing =
  group "compositing"
    [
      prop "fills composite source-over on encoded components"
        (Gen.pair gen_color gen_color)
        over_formula;
      prop "many translucent fills composite to within a level"
        ~examples:dense_ends gen_dense dense_within_a_level;
      test "source-over composites encoded components" (fun () ->
          equal color (128, 0, 128, 255)
            (pixel 0 (Picture.group [ px 0. blue; px 0. (half red) ])));
      test "shapes abutting across a pixel let a little of the backdrop through"
        (fun () ->
          (* Each covers half the pixel: the second takes half of what the first
             left, 1 - 0.5 * 0.5 of the colour. *)
          equal color (255, 0, 0, 191)
            (pixel 0
               (Picture.group
                  [
                    Picture.fill red (rect 0. 0. 0.5 1.);
                    Picture.fill red (rect 0.5 0. 0.5 1.);
                  ])));
      cases "an opacity composites its layer as one primitive:"
        ~id:(fun (n, _, _, _) -> n)
        layers
        (fun (_, p, x, c) -> equal color c (pixel x p));
    ]

(* Images *)

let image ?(box = Box2.v 0. 0. 1. 1.) shape values =
  Picture.image box (Nx.create Nx.uint8 shape values)

let images =
  group "images"
    [
      test "an image of no width paints nothing" (fun () ->
          let flat =
            image ~box:(Box2.v 10.5 0. 0. 20.) [| 1; 1; 3 |] [| 255; 0; 0 |]
          in
          equal float_exact 0. (coverage (render 20. 20. flat)));
      test "a clip around a thin image leaves it as it is" (fun () ->
          (* The image is 5e-10 wide, on the centres of column 10, and lies
             inside the clip's box. *)
          let thin =
            image ~box:(Box2.v 10.5 0. 5e-10 20.) [| 1; 1; 3 |] [| 255; 0; 0 |]
          in
          same_image (render 20. 20. thin)
            (render 20. 20. (Picture.clip (rect 0. 0. 20. 20.) thin)));
      cases "an image shown huge turns colour at its pixels' boundary"
        ~id:Float.to_string [ 1e3; 2e9; 1e10 ] (fun w ->
          (* The boundary between the two columns lands on x = 10. *)
          let p = image ~box:(Box2.v 0. 0. 1. 1.) [| 1; 2; 1 |] [| 10; 200 |] in
          let img =
            render 20. 20.
              (Picture.transform
                 Affine.(translate (10. -. (w /. 2.)) 0. * scale w 20.)
                 p)
          in
          equal (array int)
            (Array.init 20 (fun x -> if x < 10 then 10 else 200))
            (Array.init 20 (fun x -> Nx.item [ 10; x; 0 ] img)));
      test "each cell takes its pixel, without interpolation" (fun () ->
          let p =
            image ~box:(Box2.v 0. 0. 4. 4.) [| 2; 2; 3 |]
              [| 255; 0; 0; 0; 255; 0; 0; 0; 255; 10; 20; 30 |]
          in
          let img = render 4. 4. p in
          equal (list color)
            [
              (255, 0, 0, 255);
              (0, 255, 0, 255);
              (0, 0, 255, 255);
              (10, 20, 30, 255);
            ]
            [ rgba img 1 1; rgba img 0 3; rgba img 3 0; rgba img 2 2 ]);
      test "grey images are grey and opaque" (fun () ->
          equal color (77, 77, 77, 255)
            (rgba (render 1. 1. (image [| 1; 1; 1 |] [| 77 |])) 0 0));
      test "RGBA images keep their straight alpha" (fun () ->
          equal color (200, 100, 50, 128)
            (rgba
               (render 1. 1. (image [| 1; 1; 4 |] [| 200; 100; 50; 128 |]))
               0 0));
      test "a cell holds its left and top edges" (fun () ->
          (* Two cells over 3 pixels meet at 1.5, the centre of pixel 1. *)
          let across =
            render 3. 1.
              (image ~box:(Box2.v 0. 0. 3. 1.) [| 1; 2; 1 |] [| 0; 255 |])
          and down =
            render 1. 3.
              (image ~box:(Box2.v 0. 0. 1. 3.) [| 2; 1; 1 |] [| 0; 255 |])
          in
          equal (pair color color)
            ((255, 255, 255, 255), (255, 255, 255, 255))
            (rgba across 0 1, rgba down 1 0));
      test "the last row and column hold their bottom and right edges"
        (fun () ->
          (* The box ends at the centre of pixel (1, 1). *)
          let img =
            render 3. 3.
              (image ~box:(Box2.v 0. 0. 1.5 1.5) [| 1; 1; 1 |] [| 255 |])
          in
          equal (pair color color)
            ((255, 255, 255, 255), (0, 0, 0, 0))
            (rgba img 1 1, rgba img 2 2));
      test "an image paints only the pixels whose centres it covers" (fun () ->
          (* Centres at 0.5 and 1.5 lie within 0.25 to 2.25, those at 2.5 do
             not. *)
          let img =
            render 10. 10.
              (image ~box:(Box2.v 0.25 0.25 2. 2.) [| 1; 1; 3 |]
                 [| 255; 255; 255 |])
          in
          equal (list int) [ 255; 255; 0; 0; 0 ]
            (List.map
               (fun (y, x) -> alpha img y x)
               [ (0, 0); (1, 1); (2, 1); (1, 2); (2, 2) ]));
      test "an image shown smaller than its pixels averages them" (fun () ->
          (* Its 16 samples fall on 8 black and 8 white pixels: the mean, 127.5,
             lies halfway between two levels. *)
          let px =
            Nx.init Nx.uint8 [| 4; 4; 1 |] (fun i ->
                if (i.(0) + i.(1)) mod 2 = 0 then 0 else 255)
          in
          let r, _, _, _ =
            rgba (render 1. 1. (Picture.image (Box2.v 0. 0. 1. 1.) px)) 0 0
          in
          mem int r [ 127; 128 ]);
      test "a pixel averages at most 4 by 4 image pixels" (fun () ->
          (* Eight columns in one pixel: 4 samples a row, at columns 1, 3, 5 and
             7, all white, where all eight average to grey. *)
          let px =
            Nx.init Nx.uint8 [| 8; 8; 1 |] (fun i ->
                if i.(1) mod 2 = 1 then 255 else 0)
          in
          equal color (255, 255, 255, 255)
            (rgba (render 1. 1. (Picture.image (Box2.v 0. 0. 1. 1.) px)) 0 0));
      test "a view of a tensor is read as its elements" (fun () ->
          let px =
            Nx.init Nx.uint8 [| 3; 2; 3 |] (fun i ->
                (i.(0) * 70) + (i.(1) * 30) + (i.(2) * 9))
          in
          let view = Nx.transpose ~axes:[ 1; 0; 2 ] px in
          let draw px = render 6. 4. (Picture.image (Box2.v 0. 0. 6. 4.) px) in
          same_image (draw (Nx.copy view)) (draw view));
    ]

(* Stamps *)

(* [restyle ?fill ?stroke k p] is [p] with the colours of its fills and glyphs
   replaced by [fill], those of its strokes by [stroke], and its pens divided by
   [k]: what a stamp draws for an instance of scale [k]. *)
let rec restyle ?fill ?stroke k (p : Picture.t) =
  let go = restyle ?fill ?stroke k in
  let pick c = Option.value ~default:c in
  match p with
  | Empty | Image _ -> p
  | Fill { rule; color; path } -> Picture.fill ~rule (pick color fill) path
  | Glyphs { color; at; run } -> Picture.glyphs (pick color fill) at run
  | Stroke { stroke = s; color; path } ->
      let pen =
        Stroke.v ~cap:(Stroke.cap s) ~join:(Stroke.join s)
          ~miter_limit:(Stroke.miter_limit s)
          ~dash:(List.map (fun d -> d /. k) (Stroke.dash s))
          ~dash_offset:(Stroke.dash_offset s /. k)
          (Stroke.width s /. k)
      in
      Picture.stroke pen (pick color stroke) path
  | Group ps -> Picture.group (List.map go ps)
  | Clip { rule; path; picture } -> Picture.clip ~rule path (go picture)
  | Transform { m; picture } -> Picture.transform m (go picture)
  | Opacity { opacity; picture } -> Picture.opacity opacity (go picture)
  | Tag { picture; _ } -> go picture
  | Stamp { picture; xs; ys; scales; fills; strokes } ->
      Picture.stamp ?fills ?strokes ?scales xs ys (go picture)

(* [instances s] is the group of the instances of the stamp [s], each built with
   [restyle] and placed by a transform. *)
let instances (s : Picture.t) =
  match s with
  | Stamp { picture; xs; ys; scales; fills; strokes } ->
      let at a i = Option.map (fun a -> a.(i)) a in
      Picture.group
        (List.init (Array.length xs) (fun i ->
             let k = Option.value ~default:1. (at scales i) in
             Picture.transform
               Affine.(translate xs.(i) ys.(i) * scale k k)
               (restyle ?fill:(at fills i) ?stroke:(at strokes i) k picture)))
  | Empty -> Picture.empty
  | p -> failf "not a stamp: %a" Picture.pp p

(* Stamps of up to four instances of a generated picture on the quarter-pixel
   grid, where [instances] places them as the stamp does. A stamp nested in the
   picture stays one in [instances], and rounds its positions alike. *)
let gen_stamp =
  let open Gen in
  let quarter = map (fun k -> Float.of_int k /. 4.) (int_range 0 240) in
  let stamp =
    let* n = int_range 1 4 in
    let each g = array ~size:(constant n) g in
    let+ xs = each quarter
    and+ ys = each quarter
    and+ fills = option (each gen_color)
    and+ strokes = option (each gen_color)
    and+ scales = option (each (of_list [ 0.5; 1.; 2.; 3. ]))
    and+ p = Vg_corpus.gen_picture in
    Picture.stamp ?fills ?strokes ?scales xs ys p
  in
  with_pp Picture.pp stamp

(* Shrunk instances of an opacity, whose pens their transforms grow. *)
let shrunk_opacities =
  let ring w r = Picture.stroke (Stroke.v w) red (Path.circle (P2.v 0. 0.) r) in
  let dot = Picture.fill blue (Path.circle (P2.v 0. 0.) 3.) in
  List.map
    (fun p ->
      Picture.stamp ~scales:[| 0.1 |] [| 10. |] [| 10. |]
        (Picture.opacity 0.5 p))
    [
      ring 4. 5.;
      Picture.transform (Affine.scale 10. 10.) (ring 0.4 0.5);
      Picture.group [ dot; ring 4. 5. ];
      Picture.group [ ring 4. 5.; dot ];
    ]

let marker =
  let disc = Path.circle (P2.v 0. 0.) 3. in
  Picture.group
    [
      Picture.fill (Color.v 0.2 0.4 0.8) disc;
      Picture.stroke (Stroke.v 1.) (Color.v ~alpha:0.8 0.9 0.1 0.1) disc;
    ]

let quarter v = Float.round (v *. 4.) /. 4.
let ring w = Picture.stroke (Stroke.v w) red (Path.circle (P2.v 0. 0.) 5.)

(* Factors of scales each with an inverse, two of which compose to a map without
   one, under which a picture paints nothing. *)
let lost_maps = [ ("an underflow to zero", 1e-200); ("an overflow", 1e200) ]

(* Instances of a fill whose whole-pixel coverage a clip cuts to a sliver: a
   product of coverage and clip below a two-thousandth, which adds up to a level
   over the instances. *)
let sliver =
  let n = 8 in
  let origin = Array.make n 0. in
  Picture.stamp origin origin
    (Picture.clip (rect 1.98 9.98 1. 1.) (solid 3. 12.))

(* A stroked circle of the smallest normal radius, at a scale that rounds the
   coordinates of its path: its stroke is that of the group's instance. *)
let tiny_circle =
  Picture.stamp ~scales:[| 0.5 |] [| 0. |] [| 0. |]
    (Picture.stroke
       (Stroke.v ~cap:`Butt ~join:`Miter 1.)
       red
       (Path.circle (P2.v 0. 0.) Float.min_float))

(* An image turned half a turn whose edge lies on a row of pixel centres:
   rounding in the composed maps must not decide whether that row is painted. *)
let image_on_centres =
  let pixels =
    Nx.init Nx.uint8 [| 2; 3; 4 |] (fun i ->
        (40 * i.(1)) + (90 * i.(0)) + (20 * i.(2)))
  in
  Picture.stamp [| 13.75 |] [| 19.5 |]
    (Picture.transform
       Affine.(translate 0. (-5.) * rotate Float.pi)
       (Picture.image (Box2.v (-9.) (-5.) 14. 16.) pixels))

(* Fills that overlap within an opacity within a clip whose edge crosses pixels:
   the layer is clipped once, as one primitive. *)
let clipped_opacity =
  Picture.stamp [| 0. |] [| 0. |]
    (Picture.clip
       (Path.circle (P2.v 2. 2.) 1.5)
       (Picture.opacity 0.5 (Picture.group [ solid 4. 4.; solid 4. 4. ])))

(* A clip whose box a stroke meets at the instance's scale, with the pen the
   stamp keeps, and misses at scale 1. *)
let pen_meets_clip =
  Picture.stamp ~scales:[| 0.5 |] [| 0. |] [| 0. |]
    (Picture.clip (rect 0. 0. 1. 1.)
       (Picture.stroke (Stroke.v ~cap:`Butt 2.) red
          (Path.close (segment 0. 2.5 0. 3.5))))

(* A dash whose run of points closer than a twentieth of a pixel starts inside
   the pen's reach of the page, and which ends with a short turn: the page cuts
   the dash where a tile does not, and which points survive must not depend on
   the cut. *)
let dash_points_survive =
  let n = 701 in
  let x i = -10.96 +. (0.04 *. Float.of_int i) in
  let xs =
    Array.concat [ [| -30. |]; Array.init n x; [| x (n - 1) +. 0.02 |] ]
  in
  let ys = Array.make (n + 2) 30. in
  ys.(n + 1) <- 30.04;
  Picture.stamp [| 0. |] [| 0. |]
    (Picture.stroke
       (Stroke.v ~cap:`Butt ~join:`Round ~dash:[ 1000.; 1. ] 20.)
       red (Path.polyline xs ys))

(* An image narrower than a pixel, turned half a turn. *)
let thin_image =
  let pixels =
    Nx.init Nx.uint8 [| 2; 3; 4 |] (fun i ->
        (40 * i.(1)) + (90 * i.(0)) + (20 * i.(2)))
  in
  Picture.stamp ~scales:[| 0.5 |] [| 2.5 |] [| 8.25 |]
    (Picture.transform (Affine.rotate Float.pi)
       (Picture.stamp [| 0. |] [| 0. |]
          (Picture.image (Box2.v 0. 0. 1.90735e-06 16.) pixels)))

(* An opacity over a clip whose box a turned fill's box touches along a side. *)
let touching_clip =
  Picture.stamp ~scales:[| 0.5; 0.5; 0.5 |] [| 0.; 2.25; 0. |] [| 0.; 0.; 0. |]
    (Picture.opacity 0.5
       (Picture.clip (rect 0. 0. 1. 1.)
          (Picture.transform
             Affine.(translate 0. 2.78947 * rotate Float.pi)
             (Picture.fill red (rect 0. 1.9443 1. 1.)))))

(* The same stroke under an opacity, whose layer must hold what it paints. *)
let opacity_pen_meets_clip =
  match pen_meets_clip with
  | Stamp { picture; xs; ys; scales; _ } ->
      Picture.stamp ?scales xs ys (Picture.opacity 0.5 picture)
  | p -> p

(* A nested stamp whose fill ends above the page at its position, and reaches
   the first row of pixels at that position rounded to a quarter pixel. *)
let rounded_onto_page =
  Picture.stamp [| 0. |] [| 0. |]
    (Picture.stamp [| 0. |] [| -0.1 |]
       (Picture.fill red (rect 0. (-1.) 5. 1.05)))

let stamps =
  group "stamps"
    [
      prop "a stamp draws the group of its instances"
        ~examples:
          (Picture.stamp [| 0. |] [| 0. |] apart
          :: sliver :: tiny_circle :: image_on_centres :: pen_meets_clip
          :: rounded_onto_page :: clipped_opacity :: opacity_pen_meets_clip
          :: dash_points_survive :: thin_image :: touching_clip
          :: shrunk_opacities) gen_stamp (fun s ->
          same_image ~within:1 (render 60. 60. (instances s)) (render 60. 60. s));
      test "a stamp under a clip draws the group of its instances under it"
        (fun () ->
          (* The second fill of the instance at (5, 5) misses the clip's box,
             and the first meets it. *)
          let s =
            Picture.stamp [| 5.; 12. |] [| 5.; 1. |]
              (Picture.group
                 [ solid 2. 2.; Picture.fill blue (rect 0. 2.6 2. 0.4) ])
          in
          let under p = render 20. 20. (Picture.clip (rect 0. 0. 20. 7.5) p) in
          same_image ~within:1 (under (instances s)) (under s));
      prop "an instance is drawn at its position rounded to a quarter pixel"
        (Gen.pair (gen_in 5. 15.) (gen_in 5. 15.))
        (fun (x, y) ->
          same_image ~within:1
            (render 20. 20.
               (Picture.transform
                  (Affine.translate (quarter x) (quarter y))
                  marker))
            (render 20. 20. (Picture.stamp [| x |] [| y |] marker)));
      test "instances at non-finite or far positions are skipped" (fun () ->
          let p =
            Picture.stamp
              [| Float.nan; 1e300; infinity; 10. |]
              [| 0.; 0.; 0.; 10. |] marker
          in
          same_image
            (render 20. 20.
               (Picture.transform (Affine.translate 10. 10.) marker))
            (render 20. 20. p));
      test "an instance scaled to a point keeps its pen" (fun () ->
          let img =
            render 20. 20.
              (Picture.stamp ~scales:[| 1e-305 |] [| 10. |] [| 10. |] (ring 4.))
          in
          area ~arc:(4. *. Float.pi) ~box:(4., 4.) (4. *. Float.pi)
            (coverage img));
      test "an instance whose map has no inverse paints nothing" (fun () ->
          let p =
            Picture.stamp ~scales:[| 1e-310 |] [| 10. |] [| 10. |] (ring 4.)
          in
          equal float_exact 0. (coverage (render 20. 20. p)));
      test "an outlined square shrunk below a pixel keeps its outline"
        (fun () ->
          let square =
            Picture.stroke
              (Stroke.v ~cap:`Butt ~join:`Miter 2.)
              red (rect (-5.) (-5.) 10. 10.)
          in
          let img =
            render 20. 20.
              (Picture.stamp ~scales:[| 0.001 |] [| 10. |] [| 10. |] square)
          in
          (* A square of side 0.01 stroked 2 wide with mitred corners. *)
          area ~box:(2.01, 2.01) (2.01 *. 2.01) (coverage img));
      test "a shrunk instance whose pen alone reaches the page is drawn"
        (fun () ->
          (* A ring of radius 0.25 half a pixel left of the page, whose pen
             reaches 0.25 into it. *)
          let p =
            Picture.stamp ~scales:[| 0.1 |] [| -0.5 |] [| 10. |]
              (Picture.stroke (Stroke.v 1.) red (Path.circle (P2.v 0. 0.) 2.5))
          in
          greater float_exact ~than:0. (coverage (render 20. 20. p)));
      cases "an instance holding a stroke that paints nothing is drawn" ~id:fst
        lost_maps (fun (_, k) ->
          let lost = Picture.transform (Affine.scale k k) in
          let s =
            Picture.stamp ~scales:[| 1.; 2. |] [| 5.; 12. |] [| 5.; 12. |]
              (Picture.group [ solid 2. 2.; lost (lost (ring 1.)) ])
          in
          same_image ~within:1 (render 20. 20. (instances s)) (render 20. 20. s));
    ]

(* Transforms *)

let gen_map =
  let open Gen in
  let k = gen_in (-1.5) 1.5 in
  with_pp Affine.pp
    (let+ xx = k and+ yx = k and+ xy = k and+ yy = k in
     { Affine.xx; yx; xy; yy; x0 = 15.; y0 = 15. })

let transforms =
  group "transforms"
    [
      prop "a transform draws a fill as the fill of the mapped path"
        (Gen.pair gen_map (Gen.pair gen_rule gen_area))
        (fun (m, (rule, q)) ->
          let q = Path.transform (Affine.translate (-15.) (-15.)) q in
          same_image ~within:1
            (render 30. 30. (Picture.fill ~rule red (Path.transform m q)))
            (render 30. 30. (Picture.transform m (Picture.fill ~rule red q))));
      test "an image under a translation is drawn where it puts it" (fun () ->
          let px =
            Nx.init Nx.uint8 [| 2; 2; 3 |] (fun i ->
                (i.(0) * 100) + (i.(1) * 50) + (i.(2) * 30))
          in
          same_image
            (render 20. 20. (Picture.image (Box2.v 4.5 3.25 8. 8.) px))
            (render 20. 20.
               (Picture.transform
                  (Affine.translate 3.5 2.25)
                  (Picture.image (Box2.v 1. 1. 8. 8.) px))));
      cases "transforms whose composition overflows or underflows paint nothing"
        ~id:fst
        [ ("overflows", 1e200); ("underflows", 1e-200) ]
        (fun (_, k) ->
          let twice p =
            Picture.transform (Affine.scale k k)
              (Picture.transform (Affine.scale k k) p)
          in
          equal float_exact 0.
            (coverage
               (render 20. 20.
                  (twice (Picture.fill red (rect 0. 0. 1e300 1e300))))));
      cases "a picture beyond the range of floats paints nothing"
        ~id:(Format.asprintf "%a" Picture.pp) Vg_corpus.extremes (fun p ->
          equal float_exact 0. (coverage (render 20. 20. p)));
      cases "a pen beyond the range of floats paints nothing" ~id:fst
        (let disc = Path.circle (P2.v 0. 0.) 2. in
         [
           ( "a width magnified past the range",
             Picture.transform (Affine.scale 1e300 1e300)
               (Picture.stroke (Stroke.v 1e300) red disc) );
           ( "dashes an instance's scale lengthens past the range",
             Picture.stamp ~scales:[| 1e-307 |] [| 10. |] [| 10. |]
               (Picture.stroke (Stroke.v ~dash:[ 1e3; 1. ] 1.) red disc) );
         ])
        (fun (_, p) -> equal float_exact 0. (coverage (render 20. 20. p)));
      cases "a pen keeps its width" ~id:(Printf.sprintf "under a scale of %g")
        [ 1e-170; 1e170 ] (fun k ->
          (* A line 10 wide across the page, drawn in units of [1 /. k]. *)
          let line u =
            Picture.stroke
              (Stroke.v ~cap:`Butt (10. *. u))
              red
              (segment 0. (10. *. u) (20. *. u) (10. *. u))
          in
          same_image
            (render 20. 20. (line 1.))
            (render 20. 20.
               (Picture.transform (Affine.scale k k) (line (1. /. k)))));
      prop ~examples:Vg_corpus.extremes "any scale draws without failing"
        Vg_corpus.gen_extreme (fun p ->
          equal (array int) [| 20; 20; 4 |] (Nx.shape (render 20. 20. p)));
    ]

(* Tags *)

let tags =
  group "tags"
    [
      prop "are ignored" Vg_corpus.gen_picture (fun p ->
          let cells =
            Picture.Cells { box = Box2.v 0. 0. 1. 1.; width = 1; height = 1 }
          in
          let t =
            { Picture.id = Nx.Ptree.Path.v [ Field "a" ]; rows = cells }
          in
          Law.ignores picture raster (render 100. 100.) (Picture.tag t) p);
    ]

(* Render *)

(* [windowed (side, p)] checks that the page shows what it cuts from a larger
   one: [p] moved by [margin] points on a page [2 margin] points larger shows
   the pixels of the page [side] points wide, [margin] pixels in. *)
let margin = 8

let windowed (side, p) =
  let n = Float.to_int side and m = Float.of_int margin in
  let big =
    render
      (side +. (2. *. m))
      (side +. (2. *. m))
      (Picture.transform (Affine.translate m m) p)
  in
  let cut =
    Nx.slice [ R (margin, margin + n); R (margin, margin + n); A ] big
  in
  same_image ~within:1 cut (render side side p)

(* Pictures whose ink the edges of a page of 20 cut. *)
let cut_by_the_page =
  let butt ?(w = 2.) dash = Stroke.v ~cap:`Butt ~dash w in
  let stroke s q = Picture.stroke s red q in
  List.map
    (fun p -> (20., p))
    [
      Picture.fill red (Path.polygon [| -10.; 10.; -10. |] [| 0.; 0.; 20. |]);
      stroke (Stroke.v 2.) (segment (-1e9) 5. 1e9 5.);
      stroke (butt [ 2.; 2. ]) (segment (-1e9) 5. 1e9 5.);
      stroke (butt [ 2.; 2. ]) (segment (-7.5) 5. 20. 5.);
      stroke (butt [ 2.; 2. ]) (segment 27.5 5. 0. 5.);
      stroke (butt [ 2.; 2. ]) (segment 5. (-7.5) 5. 20.);
      stroke (butt [ 2.; 2. ]) (segment 5. 27.5 5. 0.);
      stroke
        (butt [ 2.; 2. ])
        (Path.polyline [| 2.; 34.; 34.; 2. |] [| 5.; 5.; 15.; 15. |]);
      stroke (butt ~w:0.4 [ 2.; 2. ]) (segment 0.3 0. 0.3 20.);
      stroke (butt ~w:0.4 [ 2.; 2. ]) (segment 19.7 0. 19.7 20.);
      stroke (butt ~w:0.4 [ 2.; 2. ]) (segment 0. 19.7 20. 19.7);
      (* A pen stretched 4 times reaching the page from 6 above it. *)
      Picture.transform (Affine.scale 1. 4.)
        (stroke (butt ~w:4. [ 2.; 2. ]) (segment (-10.) (-1.5) 30. (-1.5)));
      (* A sheared pen reaching the page from 6 above it. *)
      Picture.transform
        { Affine.xx = 1.; yx = 4.; xy = 0.; yy = 1.; x0 = 0.; y0 = 0. }
        (stroke (butt ~w:4. [ 2.; 2. ]) (segment (-10.) 34. 30. (-126.)));
    ]

(* [ink_outside p] is the pixels of [p] drawn on a page of 100 whose alpha is
   not zero and whose square does not meet [Picture.bounds p], grown by the
   eighth of a pixel by which stamps move their instances. *)
let ink_outside p =
  let n = 100 in
  let px = Nx.to_array (render (Float.of_int n) (Float.of_int n) p) in
  let bounds = Picture.bounds p in
  let meets i j =
    match bounds with
    | None -> false
    | Some b ->
        let e = 0.125 +. 1e-9 in
        Float.of_int j <= Box2.maxx b +. e
        && Box2.minx b -. e <= Float.of_int (j + 1)
        && Float.of_int i <= Box2.maxy b +. e
        && Box2.miny b -. e <= Float.of_int (i + 1)
  in
  let outside = ref [] in
  for i = n - 1 downto 0 do
    for j = n - 1 downto 0 do
      let alpha = px.((((i * n) + j) * 4) + 3) in
      if alpha > 0 && not (meets i j) then outside := (i, j) :: !outside
    done
  done;
  !outside

let render_group =
  group "render"
    [
      cases "the pixel grid of"
        ~id:(fun (n, _, _) -> n)
        [
          ("whole points", (10., 5., 1.), [| 5; 10; 4 |]);
          ("fractions rounded to the nearest", (10.4, 5.6, 1.), [| 6; 10; 4 |]);
          ("a density of two", (360., 240., 2.), [| 480; 720; 4 |]);
          ( "a product a hair above an integer",
            (100.00000000000001, 1., 2.),
            [| 2; 200; 4 |] );
        ]
        (fun (_, (w, h, density), shape) ->
          equal (array int) shape (Nx.shape (render ~density w h Picture.empty)));
      test "pixel (i, j) shows the square at (j / d, i / d)" (fun () ->
          let img =
            render ~density:2. 3. 2. (Picture.fill red (rect 1.5 0.5 0.5 0.5))
          in
          equal color (255, 0, 0, 255) (rgba img 1 3);
          equal float_exact 1. (coverage img));
      prop "a page shows what it cuts from a larger page"
        ~examples:cut_by_the_page
        (Gen.with_pp
           (fun ppf (_, p) -> Picture.pp ppf p)
           (Gen.map (fun p -> (100., p)) Vg_corpus.gen_picture))
        windowed;
      test "a line far beyond the page is drawn across it" (fun () ->
          (* A band 2 high on whole pixels, 20 long. *)
          let far = segment (-1e9) 5. 1e9 5. in
          equal float_exact 40.
            (coverage (render 20. 10. (Picture.stroke (Stroke.v 2.) red far))));
      test "a dashed line far beyond the page keeps its phase across it"
        (fun () ->
          (* 1e9 is a whole number of periods of 4: dashes from 0 to 2, 4 to 6,
             and so on, ten of them 2 by 2 on whole pixels. *)
          let far = segment (-1e9) 5. 1e9 5. in
          let dashed = Stroke.v ~cap:`Butt ~dash:[ 2.; 2. ] 2. in
          equal float_exact 40.
            (coverage (render 40. 10. (Picture.stroke dashed red far))));
      test "a line to near max_float is drawn up to the page's edge" (fun () ->
          (* Its far end, at density 2, is beyond [max_float], held at 1e15
             along x: a band 2 pixels high across the 20 of the page. *)
          let p =
            Picture.stroke (Stroke.v ~cap:`Butt 1.) red (segment 0. 5. 1e308 5.)
          in
          equal float_exact 40. (coverage (render ~density:2. 10. 10. p)));
      cases "a fill with a vertex far beyond the page covers its area on it"
        ~id:(Printf.sprintf "%g") [ 1e16; 1e300 ] (fun k ->
          (* On the page, the triangle is the band between x = y / 3 and x = 20.
             The edges have an end on the page, so they lie where they should to
             within rounding. *)
          let far = Path.polygon [| 0.; 20.; k |] [| 0.; 0.; 3. *. k |] in
          area ~box:(20., 20.)
            (400. -. (200. /. 3.))
            (coverage (render 20. 20. (Picture.fill red far))));
      cases "a turned line far beyond the page crosses it within the precision"
        ~id:(Printf.sprintf "%g") [ 1e12; 1e14 ] (fun e ->
          (* The line through the page's centre leans 0.3 from a diagonal, so
             each side crosses the page, 20 pixels wide at density 2, over 20 /
             cos (pi / 4 - 0.3) pixels, and they are 2 pixels apart. The ends of
             its sides lie within 2 sqrt 2 e pixels of the origin, so each side
             may lie 2^-50 of that off along its length. *)
          let p =
            Picture.transform
              Affine.(translate 5. 5. * rotate 0.3)
              (Picture.stroke (Stroke.v 1.) red (segment (-.e) (-.e) e e))
          in
          let side = 20. /. Float.cos ((Float.pi /. 4.) -. 0.3) in
          let off = Float.ldexp (2. *. Float.sqrt 2. *. e) (-50) in
          area ~box:(20., 20.)
            ~drift:(2. *. side *. off)
            (2. *. side)
            (coverage (render ~density:2. 10. 10. p)));
      test "an edge cancelled by a translation keeps the rounding of its terms"
        (fun () ->
          (* The left and right sides of a rect 10 wide, 1e15 out and brought
             back by the translation: each side, 20 pixels long, may lie 2^-50
             of the largest term, about 1.1e15, off. *)
          let p =
            Picture.transform
              Affine.(translate (-1.1e15) 0. * scale 1.1 1.)
              (Picture.fill red (rect (1e15 +. 5.) 0. 10. 20.))
          in
          let off = Float.ldexp 1.1e15 (-50) in
          area ~box:(11., 20.)
            ~drift:(2. *. 20. *. off)
            (11. *. 20.)
            (coverage (render 20. 20. p)));
      test
        "a fill's end that the density maps beyond the range of floats is a gap"
        (fun () ->
          (* At density 2 the far vertex maps beyond max_float, so the triangle
             loses it and draws no area. *)
          let p =
            Picture.fill red
              (Path.polygon [| 0.; 10.; 1e308 |] [| 0.; 0.; 3e307 |])
          in
          equal float_exact 0. (coverage (render ~density:2. 10. 10. p)));
      test "a slanted stroke to near max_float draws without failing" (fun () ->
          (* Both ends map beyond max_float at density 2 and are held at 1e15,
             which may bend its sides: only the drawing is stated. *)
          let far = segment (-1e308) (-1e308) 1e308 1e308 in
          let p =
            Picture.transform
              Affine.(translate 5. 5. * rotate 0.3)
              (Picture.stroke (Stroke.v 1.) red far)
          in
          equal (array int) [| 20; 20; 4 |]
            (Nx.shape (render ~density:2. 10. 10. p)));
      test "an image to near max_float covers the page" (fun () ->
          let p =
            image ~box:(Box2.v 0. 0. 1e308 1e308) [| 1; 1; 1 |] [| 255 |]
          in
          equal float_exact 400. (coverage (render ~density:2. 10. 10. p)));
      cases "a pixel shows its colour"
        ~id:(fun (n, _, _) -> n)
        nearest_levels
        (fun (_, c, level) -> equal color level (pixel 0 (px 0. c)));
      test "pixels painted nothing are transparent black" (fun () ->
          equal (array int) (Array.make 84 0)
            (Nx.to_array (render 7. 3. Picture.empty)));
      test "a pixel whose alpha rounds to 0 is transparent black" (fun () ->
          (* An alpha of 0.001 is a quarter of a level. *)
          equal color (0, 0, 0, 0)
            (rgba
               (render 1. 1.
                  (Picture.fill (Color.with_alpha 0.001 red) (rect 0. 0. 1. 1.)))
               0 0));
      prop "a picture paints within its bounds" Vg_corpus.gen_picture (fun p ->
          equal (list (pair int int)) [] (ink_outside p));
      prop "equal renderables give equal pixels" Vg_corpus.gen_picture (fun p ->
          Law.ignores picture raster (render 100. 100.) Vg_corpus.respell p);
      (* The PNGs are written by gen/gen.exe. A level of difference is allowed:
         the arcs go through libm, whose last bits differ between platforms. *)
      cases "draws each page of the corpus as its golden PNG" ~id:fst
        Vg_corpus.pages (fun (name, r) ->
          same_image ~within:1
            (Nx_io.load_image (Filename.concat "golden" (name ^ ".png")))
            (Hugin_vg_raster.render ~density:2. r));
      cases "render raises on the density" ~id:(Format.asprintf "%g")
        [ 0.; -1.; Float.nan; infinity ] (fun density ->
          raises_match (Exn.invalid_arg ~sub:"density") (fun () ->
              render ~density 10. 10. Picture.empty));
      cases "render raises on a page of" ~id:fst
        [
          ("no pixel wide", (0.4, 10.));
          ("no pixel high", (10., 0.4));
          ("more than 2^31 - 1 pixels", (3e9, 1.));
        ]
        (fun (_, (w, h)) ->
          raises_match (Exn.invalid_arg ~sub:"pixels") (fun () ->
              render w h Picture.empty));
    ]

(* PNG *)

let chunk ty png =
  let rec find i =
    if i >= String.length png then None
    else
      let n = Int32.to_int (String.get_int32_be png i) in
      if String.sub png (i + 4) 4 = ty then Some (String.sub png (i + 8) n)
      else find (i + 12 + n)
  in
  find 8

let be32 n = String.init 4 (fun i -> Char.chr ((n lsr (8 * (3 - i))) land 0xff))

(* The [pHYs] chunk of [n] pixels per metre in both directions. *)
let per_metre n = Some (be32 n ^ be32 n ^ "\001")

let png_group =
  let page =
    Renderable.v 20. 10. (Picture.fill red (Path.circle (P2.v 10. 5.) 4.))
  in
  group "png"
    [
      test "png holds the pixels of render" (fun () ->
          same_image
            (Hugin_vg_raster.render ~density:2. page)
            (Vg_corpus.load_png (Hugin_vg_raster.png ~density:2. page)));
      test "png records the density in pixels per metre" (fun () ->
          (* 144 dpi is 5669.29 pixels per metre. *)
          equal (option string) (per_metre 5669)
            (chunk "pHYs" (Hugin_vg_raster.png ~density:2. page)));
      test "png records sRGB with the perceptual rendering intent" (fun () ->
          equal (option string) (Some "\000")
            (chunk "sRGB" (Hugin_vg_raster.png ~density:2. page)));
      cases "png raises on the density as render does"
        ~id:(Format.asprintf "%g") [ 0.; -1.; Float.nan; infinity ]
        (fun density ->
          raises_match (Exn.invalid_arg ~sub:"density") (fun () ->
              Hugin_vg_raster.png ~density page));
      cases "png raises on a density beyond a PNG's resolution, unrendered"
        ~id:(Format.asprintf "%g") [ 1e6; 1e-4 ] (fun density ->
          (* A page of 1e10 by 1e10 pixels at either density, which render would
             refuse for its size. *)
          let side = 1e10 /. density in
          raises_match (Exn.invalid_arg ~sub:"resolution") (fun () ->
              Hugin_vg_raster.png ~density
                (Renderable.v side side Picture.empty)));
      cases "png accepts the ends of a PNG's resolution" ~id:fst
        [
          ("1 pixel per metre", (0.0254 /. 72., 3000., 1));
          ( "2^31 - 1 pixels per metre",
            (2147483647. *. 0.0254 /. 72., 1e-6, 2147483647) );
        ]
        (fun (_, (density, side, n)) ->
          let png =
            Hugin_vg_raster.png ~density (Renderable.v side side Picture.empty)
          in
          equal (option string) (per_metre n) (chunk "pHYs" png));
    ]

let () =
  exit
    (run "hugin.vg raster"
       [
         fills;
         glyphs;
         clips;
         strokes;
         compositing;
         images;
         stamps;
         transforms;
         tags;
         render_group;
         png_group;
       ])
