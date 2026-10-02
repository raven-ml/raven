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

(* Pages *)

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
  same_image ~within:1 (Nx.contiguous cut) (render side side p)

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

let pages =
  group "pages"
    [
      cases
        ~name:(fun (n, _, _) -> n)
        "the pixel grid of"
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
      test "pixels painted nothing are transparent black" (fun () ->
          equal float_exact 0. (coverage (render 7. 3. Picture.empty));
          equal color (0, 0, 0, 0) (rgba (render 7. 3. Picture.empty) 1 1));
      test "pixel (i, j) shows the square at (j / d, i / d)" (fun () ->
          let img =
            render ~density:2. 3. 2. (Picture.fill red (rect 1.5 0.5 0.5 0.5))
          in
          equal color (255, 0, 0, 255) (rgba img 1 3);
          equal (float 1e-9) 1. (coverage img));
      prop "a page shows what it cuts from a larger page"
        ~examples:cut_by_the_page
        (Gen.with_pp
           (fun ppf (_, p) -> Picture.pp ppf p)
           (Gen.map (fun p -> (100., p)) Vg_corpus.gen_picture))
        windowed;
      test "a line far beyond the page is drawn across it" (fun () ->
          let far = segment (-1e9) 5. 1e9 5. in
          equal (float 0.05) 40.
            (coverage (render 20. 10. (Picture.stroke (Stroke.v 2.) red far)));
          let dashed = Stroke.v ~cap:`Butt ~dash:[ 2.; 2. ] 2. in
          equal (float 0.05) 40.
            (coverage (render 40. 10. (Picture.stroke dashed red far))));
      cases ~name:(Format.asprintf "%g") "render raises on the density"
        [ 0.; -1.; Float.nan; infinity ] (fun density ->
          raises_match (Exn.invalid_arg ~substring:"density") (fun () ->
              render ~density 10. 10. Picture.empty));
      cases ~name:fst "render raises on a page of"
        [
          ("no pixel wide", (0.4, 10.));
          ("no pixel high", (10., 0.4));
          ("more than 2^31 - 1 pixels", (3e9, 1.));
        ]
        (fun (_, (w, h)) ->
          raises_match (Exn.invalid_arg ~substring:"pixels") (fun () ->
              render w h Picture.empty));
    ]

(* Areas *)

(* [within_edges ~edges exact got] asserts that [got] is the area [exact] up to
   the pixels that edges [edges] long cross, each rounded to 8 bits, half a
   level at most. *)
let within_edges ~edges exact got =
  let tolerance = (edges +. 8.) *. 0.5 /. 255. in
  at_most
    ~msg:(Printf.sprintf "covers %g for an area of %g" got exact)
    float_exact ~than:tolerance
    (Float.abs (got -. exact))

let rect_area (x, y, w, h) =
  let img = render 30. 30. (Picture.fill red (rect x y w h)) in
  within_edges ~edges:(2. *. (w +. h)) (w *. h) (coverage img)

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
  let edges =
    List.fold_left
      (fun l (i, j) ->
        l +. Float.abs (xs.(j) -. xs.(i)) +. Float.abs (ys.(j) -. ys.(i)))
      0.
      [ (0, 1); (1, 2); (2, 0) ]
  in
  within_edges ~edges
    (Float.abs twice /. 2.)
    (coverage (render 30. 30. (Picture.fill ~rule red q)))

let areas =
  group "areas"
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
          let both = Path.append (rect 3. 3. 4. 4.) (rect 0. 0. 10. 10.) in
          equal (float 0.01) 100.
            (coverage (render 10. 10. (Picture.fill red both)));
          equal (float 0.01) 84.
            (coverage (render 10. 10. (Picture.fill ~rule:`Even_odd red both))));
      test "a circle covers its area, its chords within a tenth of a pixel"
        (fun () ->
          let got =
            coverage
              (render 40. 40.
                 (Picture.fill red (Path.circle (P2.v 20. 20.) 15.)))
          in
          let disc = Float.pi *. 15. *. 15. in
          at_most float_exact ~than:disc got;
          at_least float_exact
            ~than:(disc -. (0.1 *. 2. *. Float.pi *. 15.))
            got);
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
          equal (float 1e-9) 30. (coverage (render 10. 10. p)));
      cases ~name:fst "a box beside a curved clip shows nothing"
        [ ("next to its pixels", 6.); ("apart from its pixels", 8.) ]
        (fun (_, x) ->
          let circle = Path.circle (P2.v 3. 5.) 2. in
          let p =
            Picture.clip circle
              (Picture.clip (rect x 0. 2. 10.) (solid 10. 10.))
          in
          equal float_exact 0. (coverage (render 10. 10. p)));
    ]

(* Strokes *)

let line = segment 5. 10. 15. 10.
let stroked s = coverage (render 20. 20. (Picture.stroke s red line))

(* [curved ~arc exact got] asserts that [got] is the area [exact] of a shape
   whose round parts, [arc] long, are flattened inside it to within a tenth of a
   pixel, up to rounding. *)
let curved ~arc exact got =
  at_most ~msg:"no more than the shape" float_exact ~than:(exact +. 0.05) got;
  at_least ~msg:"no less than its chords" float_exact
    ~than:(exact -. (0.1 *. arc) -. 0.05)
    got

let caps =
  [
    ("butt caps end at the ends", `Butt, 40., 0.);
    ("square caps add half a square at each end", `Square, 56., 0.);
    ( "round caps add half a disc at each end",
      `Round,
      40. +. (Float.pi *. 4.),
      Float.pi *. 4. );
  ]

(* A right-angle corner at (10, 10), drawn 4 wide with butt caps: two bodies of
   32 sharing a square of 4, and the join in the outer corner's square. *)
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

let zero_dashes =
  [
    ( "squares along a line with square caps",
      `Square,
      segment 2. 5. 10. 5.,
      (12., 0.) );
    ("squares along a vertical line", `Square, segment 5. 2. 5. 10., (12., 0.));
    ( "discs with round caps",
      `Round,
      segment 2. 5. 10. 5.,
      (3. *. Float.pi, 6. *. Float.pi) );
    ("nothing with butt caps", `Butt, segment 2. 5. 10. 5., (0., 0.));
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
    coverage
      (render 30. 20.
         (Picture.stroke (Stroke.v ~cap:`Butt ~join ~miter_limit:4. 2.) red q))
  in
  equal (float 1e-9) (draw `Bevel) (draw `Miter)

let turned_dash () =
  (* A square turned by 45 degrees about (5, 5) covers part of the pixel at its
     centre's lower right, which an upright one covers whole. *)
  let s = Stroke.v ~cap:`Square ~dash:[ 0.; 100. ] 2. in
  let img =
    render 20. 20.
      (Picture.transform (Affine.translate 5. 5.)
         (Picture.stroke s red (segment 0. 0. 10. 10.)))
  in
  equal (float 0.1) 4. (coverage img);
  less ~msg:"alpha of pixel (5, 5)" int ~than:255 (alpha img 5 5)

let turned_zigzag () =
  (* Its segments run along the diagonals, which a turn by 45 degrees lays along
     the axes. *)
  let zig =
    Path.polyline [| 0.; 3.; 6.; 9.; 12. |] [| 0.; -3.; 0.; -3.; 0. |]
  in
  let draw m =
    coverage
      (render 40. 40.
         (Picture.transform m
            (Picture.stroke (Stroke.v ~cap:`Butt ~join:`Bevel 1.) red zig)))
  in
  equal (float 0.2)
    (draw (Affine.translate 14. 20.))
    (draw Affine.(translate 14. 20. * rotate (Float.pi /. 4.)))

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

let fine_dashes () =
  (* The map stretches by 3 along the diagonal the line follows, so the period
     of 0.0004 is 0.0012 of a pixel there. *)
  let m = { Affine.xx = 1.; yx = 2.; xy = 2.; yy = 1.; x0 = 5.; y0 = 5. } in
  let draw dash =
    coverage
      (render 30. 30.
         (Picture.transform m
            (Picture.stroke
               (Stroke.v ~cap:`Butt ~dash 1.)
               red (segment 0. 0. 4. 4.))))
  in
  let solid = draw [] in
  greater ~msg:"solid" float_exact ~than:10. solid;
  at_most ~msg:"dashed" float_exact ~than:(0.5 *. solid)
    (draw [ 0.0002; 0.0002 ])

let tiny_polygons join =
  let draw xs ys =
    render 20. 20. (Picture.stroke (Stroke.v ~join 4.) red (Path.polygon xs ys))
  in
  let triangle = draw [| 10.; 10.01; 10. |] [| 10.; 10.; 10.01 |] in
  same_image triangle
    (draw [| 10.; 10.; 10.01; 10. |] [| 10.; 10.; 10.; 10.01 |]);
  same_image triangle
    (draw [| 10.; 10.01; 10.01; 10. |] [| 10.; 10.; 10.; 10.01 |])

let strokes =
  group "strokes"
    [
      cases
        ~name:(fun (n, _, _, _) -> n)
        "a line 4 wide:" caps
        (fun (_, cap, a, arc) -> curved ~arc a (stroked (Stroke.v ~cap 4.)));
      cases
        ~name:(fun (n, _, _, _) -> n)
        "at a right-angle corner," joins
        (fun (_, (join, miter_limit), a, arc) ->
          let s = Stroke.v ~cap:`Butt ~join ~miter_limit 4. in
          let p = Picture.stroke s red corner in
          curved ~arc a (coverage (render 20. 20. p)));
      test "a corner sharper than the miter limit is bevelled" sharp_corner;
      test "a closed subpath joins at its start" (fun () ->
          (* An 8 by 8 square stroked 2 wide with mitred corners is a 10 by 10
             frame around a 6 by 6 hole. *)
          let p =
            Picture.stroke (Stroke.v ~join:`Miter 2.) red (rect 4. 4. 8. 8.)
          in
          equal (float 0.05) 64. (coverage (render 20. 20. p)));
      cases
        ~name:(fun (n, _, _) -> n)
        "a dashed line 20 long and 2 wide, with" dashes
        (fun (_, d, a) -> equal (float 0.05) a (dashed d));
      test "a dashed closed subpath dashes its closing side" (fun () ->
          (* Each side of 10 holds two dashes of 3; the half pixels along its
             edges round to 128 of 255 each. *)
          let s = Stroke.v ~cap:`Butt ~dash:[ 3.; 2. ] 1. in
          equal (float 0.1) 24.
            (coverage
               (render 20. 20. (Picture.stroke s red (rect 5. 5. 10. 10.)))));
      test "a dash pattern longer than a thousandth of a pixel is dashed"
        fine_dashes;
      cases
        ~name:(fun (n, _, _) -> n)
        "a subpath of zero length"
        [
          ("is a disc with round caps", `Round, Float.pi *. 4.);
          ("is nothing with butt caps", `Butt, 0.);
          ("is nothing with square caps, having no direction", `Square, 0.);
        ]
        (fun (_, cap, a) ->
          curved ~arc:a a
            (coverage
               (render 20. 20. (Picture.stroke (Stroke.v ~cap 4.) red point))));
      test "a dashed subpath of a start alone is a disc with round caps"
        (fun () ->
          let lone = Path.empty |> Path.move_to (P2.v 10. 10.) in
          let p =
            Picture.stroke (Stroke.v ~cap:`Round ~dash:[ 2.; 2. ] 4.) red lone
          in
          curved ~arc:(4. *. Float.pi) (4. *. Float.pi)
            (coverage (render 20. 20. p)));
      cases
        ~name:(fun (n, _, _, _) -> n)
        "dashes of zero length are" zero_dashes
        (fun (_, cap, q, (a, arc)) ->
          let p = Picture.stroke (Stroke.v ~cap ~dash:[ 0.; 4. ] 2.) red q in
          curved ~arc a (coverage (render 20. 20. p)));
      test "a dash of zero length takes the direction of its line" turned_dash;
      cases ~name:fst "a line too short to see is a square with square caps"
        [ ("vertical", (0., 0.01)); ("horizontal", (0.01, 0.)) ]
        (fun (_, (dx, dy)) ->
          let p =
            Picture.stroke (Stroke.v ~cap:`Square 2.) red
              (segment 10. 10. (10. +. dx) (10. +. dy))
          in
          equal (float 0.05) 4. (coverage (render 20. 20. p)));
      cases ~name:fst "a repeated point leaves a dashed line as it is"
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
          equal (float 0.1) (8.004 *. 8.004) (coverage (render 20. 20. p)));
      cases ~name:fst "a closed subpath smaller than a pixel, with"
        [ ("miter joins", `Miter); ("round joins", `Round); ("bevels", `Bevel) ]
        (fun (_, join) ->
          subtest "ignores a repeated vertex" (fun () -> tiny_polygons join);
          subtest "draws more ink as it grows" (fun () ->
              ignore
                (List.fold_left
                   (fun before r ->
                     let after = shrunk join r in
                     at_least
                       ~msg:(Printf.sprintf "radius %g" r)
                       float_exact ~than:before after;
                     after)
                   (shrunk join 0.005) [ 0.01; 0.03; 0.06; 0.12 ])));
      cases ~name:(Printf.sprintf "radius %g")
        "a mitred circle smaller than a pixel covers at least its pen"
        [ 0.001; 0.01; 0.03; 0.06 ] (fun r ->
          at_least float_exact ~than:(4. *. Float.pi) (shrunk `Miter r));
      test "a transform maps the pen" (fun () ->
          let p =
            Picture.transform (Affine.scale 1. 4.)
              (Picture.stroke (Stroke.v ~cap:`Butt 1.) red line)
          in
          equal (float 0.05) 40. (coverage (render 20. 60. p)));
      test "a pen under a shear covers the area the map gives it" (fun () ->
          (* A ring of radius 5 and width 1 has area 10 pi, which the map
             multiplies by its determinant, 4, up to the pixels that the
             overlapping pieces of the outline cover twice. *)
          let p =
            Picture.transform
              Affine.(translate 20. 30. * scale 1. 4. * rotate (Float.pi /. 6.))
              (Picture.stroke (Stroke.v 1.) red (Path.circle (P2.v 0. 0.) 5.))
          in
          equal (float 4.) (40. *. Float.pi) (coverage (render 40. 60. p)));
      test "a turn keeps a zigzag's area" turned_zigzag;
    ]

(* Compositing *)

(* [over_formula (c, c')] checks a pixel painted [c'] then [c] against the
   formula of source-over, in premultiplied levels, which the output's 8-bit
   straight alpha keeps to a level and a half. *)
let over_formula (c, c') =
  let img =
    render 1. 1.
      (Picture.group
         [
           Picture.fill c' (rect 0. 0. 1. 1.); Picture.fill c (rect 0. 0. 1. 1.);
         ])
  in
  let a = Color.alpha c and a' = Color.alpha c' in
  let r, g, b, alpha = rgba img 0 0 in
  at_most ~msg:"alpha" (float 1e-9) ~than:1.
    (Float.abs (Float.of_int alpha -. (255. *. (a +. (a' *. (1. -. a))))));
  List.iter
    (fun (name, got, k) ->
      let premultiplied = 255. *. ((a *. k c) +. (a' *. (1. -. a) *. k c')) in
      at_most ~msg:name (float 1e-9) ~than:1.5
        (Float.abs
           ((Float.of_int got *. Float.of_int alpha /. 255.) -. premultiplied)))
    [ ("red", r, Color.r); ("green", g, Color.g); ("blue", b, Color.b) ]

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

(* [dense_within_a_level d] checks the page of [d] against source-over in exact
   arithmetic: each component of the pixel is the level nearest the exact
   colour, straight, up to the drift of compositing in single precision, about a
   thousandth of a level over a thousand blends. *)
let dense_within_a_level d =
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
  let near name exact got =
    at_most ~msg:name (float 1e-9) ~than:(0.5 +. 1e-3)
      (Float.abs (Float.of_int got -. (255. *. exact)))
  in
  near "alpha" a a';
  near "red" (r /. a) r';
  near "green" (g /. a) g';
  near "blue" (b /. a) b';
  cover "the lightest opacity" (d.opacity = 0.02);
  cover "an opaque opacity" (d.opacity = 1.);
  cover "a transparent backdrop" (d.backdrop = None);
  cover "an opaque backdrop" (d.backdrop <> None);
  cover "group opacities" d.as_group;
  cover "hundreds of light fills" (d.opacity < 0.05 && d.count > 200)

(* Pixels whose levels the spec states: (name, picture of a page 3 wide, pixel
   column, colour). *)
let stated =
  let px x c = Picture.fill c (rect x 0. 1. 1.)
  and bar x c = Picture.fill c (rect x 0. 2. 1.) in
  let grey g a = Color.v ~alpha:a g g g in
  let half = Color.with_alpha 0.5 in
  [
    ("output has straight alpha", px 0. (half red), 0, (255, 0, 0, 128));
    ( "a component reads back as the level nearest it",
      px 0. (grey (200.7 /. 255.) (100.8 /. 255.)),
      0,
      (201, 201, 201, 101) );
    (* Premultiplied by an alpha of 153 levels, a grey of 31/153 is 31 levels
       exactly; straight, it is 51.67 levels. *)
    ( "a translucent colour reads back as its nearest level",
      px 0. (grey (31. /. 153.) 0.6),
      0,
      (52, 52, 52, 153) );
    ( "source-over composites encoded components",
      Picture.group [ px 0. blue; px 0. (half red) ],
      0,
      (128, 0, 128, 255) );
    ( "an opacity multiplies a translucent fill",
      Picture.opacity 0.5 (px 0. (half red)),
      0,
      (255, 0, 0, 64) );
    ( "group opacity fades the group as one: the overlap shows the later only",
      Picture.opacity 0.5 (Picture.group [ bar 0. red; bar 1. blue ]),
      1,
      (0, 0, 255, 128) );
    ( "group opacity fades each part alike",
      Picture.opacity 0.5 (Picture.group [ bar 0. red; bar 1. blue ]),
      0,
      (255, 0, 0, 128) );
    ( "per-leaf opacity lets the earlier show through",
      Picture.group
        [ Picture.opacity 0.5 (bar 0. red); Picture.opacity 0.5 (bar 1. blue) ],
      1,
      (85, 0, 170, 191) );
    ( "opacity 0. paints nothing",
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
      prop "many translucent fills composite to within a level" gen_dense
        dense_within_a_level;
      cases
        ~name:(fun (n, _, _, _) -> n)
        "pixels:" stated
        (fun (_, p, x, c) -> equal color c (rgba (render 3. 1. p) 0 x));
      prop "a tag changes no pixel" Vg_corpus.gen_picture (fun p ->
          let cells =
            Picture.Cells { box = Box2.v 0. 0. 1. 1.; width = 1; height = 1 }
          in
          let t =
            { Picture.id = Nx.Ptree.Path.v [ Field "a" ]; rows = cells }
          in
          same_image (render 100. 100. p) (render 100. 100. (Picture.tag t p)));
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
  | Tag { tag; picture } -> Picture.tag tag (go picture)
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
   grid, where stamps place instances exactly. *)
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

let stamps =
  group "stamps"
    [
      prop "a stamp draws the group of its instances" ~examples:shrunk_opacities
        gen_stamp (fun s ->
          same_image ~within:1 (render 60. 60. (instances s)) (render 60. 60. s));
      prop "an instance is drawn at its position rounded to a quarter pixel"
        (Gen.pair (gen_in 5. 15.) (gen_in 5. 15.))
        (fun (x, y) ->
          same_image ~within:1
            (render 20. 20.
               (Picture.transform
                  (Affine.translate (quarter x) (quarter y))
                  marker))
            (render 20. 20. (Picture.stamp [| x |] [| y |] marker)));
      test "positions round to the nearest quarter pixel" (fun () ->
          let at x = render 20. 10. (Picture.stamp [| x |] [| 5. |] marker) in
          same_image (at 10.) (at 10.1);
          same_image (at 10.25) (at 10.2);
          greater ~msg:"levels between 10. and 10.25" int ~than:0
            (levels (at 10.) (at 10.25)));
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
          curved ~arc:(4. *. Float.pi) (4. *. Float.pi) (coverage img));
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
          equal (float 0.1) (2.01 *. 2.01) (coverage img));
      test "a shrunk instance whose pen alone reaches the page is drawn"
        (fun () ->
          (* A ring of radius 0.25 half a pixel left of the page, whose pen
             reaches 0.25 into it. *)
          let p =
            Picture.stamp ~scales:[| 0.1 |] [| -0.5 |] [| 10. |]
              (Picture.stroke (Stroke.v 1.) red (Path.circle (P2.v 0. 0.) 2.5))
          in
          greater (float 1e-9) ~than:0. (coverage (render 20. 20. p)));
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

(* Images *)

let image ?(box = Box2.v 0. 0. 1. 1.) shape values =
  Picture.image box (Nx.create Nx.uint8 shape values)

let images =
  group "images"
    [
      test "each cell takes its pixel, without interpolation" (fun () ->
          let p =
            image ~box:(Box2.v 0. 0. 4. 4.) [| 2; 2; 3 |]
              [| 255; 0; 0; 0; 255; 0; 0; 0; 255; 10; 20; 30 |]
          in
          let img = render 4. 4. p in
          equal color (255, 0, 0, 255) (rgba img 1 1);
          equal color (0, 255, 0, 255) (rgba img 0 3);
          equal color (0, 0, 255, 255) (rgba img 3 0);
          equal color (10, 20, 30, 255) (rgba img 2 2));
      test "grey images are grey and opaque" (fun () ->
          equal color (77, 77, 77, 255)
            (rgba (render 1. 1. (image [| 1; 1; 1 |] [| 77 |])) 0 0));
      test "RGBA images keep their straight alpha" (fun () ->
          let r, g, b, a =
            rgba
              (render 1. 1. (image [| 1; 1; 4 |] [| 200; 100; 50; 128 |]))
              0 0
          in
          equal int 128 a;
          equal
            (list (float 1.))
            [ 200.; 100.; 50. ]
            (List.map Float.of_int [ r; g; b ]));
      test "a cell holds its left and top edges" (fun () ->
          (* Two cells over 3 pixels meet at 1.5, the centre of pixel 1. *)
          let across =
            render 3. 1.
              (image ~box:(Box2.v 0. 0. 3. 1.) [| 1; 2; 1 |] [| 0; 255 |])
          in
          equal color (255, 255, 255, 255) (rgba across 0 1);
          let down =
            render 1. 3.
              (image ~box:(Box2.v 0. 0. 1. 3.) [| 2; 1; 1 |] [| 0; 255 |])
          in
          equal color (255, 255, 255, 255) (rgba down 1 0));
      test "the last row and column hold their bottom and right edges"
        (fun () ->
          (* The box ends at the centre of pixel (1, 1). *)
          let img =
            render 3. 3.
              (image ~box:(Box2.v 0. 0. 1.5 1.5) [| 1; 1; 1 |] [| 255 |])
          in
          equal color (255, 255, 255, 255) (rgba img 1 1);
          equal color (0, 0, 0, 0) (rgba img 2 2));
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
          let px =
            Nx.init Nx.uint8 [| 4; 4; 1 |] (fun i ->
                if (i.(0) + i.(1)) mod 2 = 0 then 0 else 255)
          in
          let r, _, _, _ =
            rgba (render 1. 1. (Picture.image (Box2.v 0. 0. 1. 1.) px)) 0 0
          in
          equal (float 1.) 128. (Float.of_int r));
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

(* Bounds and limits *)

(* [ink_outside p] is the pixels of [p] drawn on a page of 100 whose alpha is
   not zero and whose square does not meet [Picture.bounds p], grown by the
   eighth of a pixel by which stamps move their instances. *)
let ink_outside p =
  let img = render 100. 100. p in
  let meets i j =
    match Picture.bounds p with
    | None -> false
    | Some b ->
        let e = 0.125 +. 1e-9 in
        Float.of_int j <= Box2.maxx b +. e
        && Box2.minx b -. e <= Float.of_int (j + 1)
        && Float.of_int i <= Box2.maxy b +. e
        && Box2.miny b -. e <= Float.of_int (i + 1)
  in
  let outside = ref [] in
  for i = 99 downto 0 do
    for j = 99 downto 0 do
      if alpha img i j > 0 && not (meets i j) then outside := (i, j) :: !outside
    done
  done;
  !outside

let limits =
  group "bounds and limits"
    [
      prop "a picture paints within its bounds" Vg_corpus.gen_picture (fun p ->
          equal (list (pair int int)) [] (ink_outside p));
      test "a line to near max_float is drawn up to the page's edge" (fun () ->
          (* Its far end, at density 2, is beyond [max_float]. *)
          let p =
            Picture.stroke (Stroke.v ~cap:`Butt 1.) red (segment 0. 5. 1e308 5.)
          in
          equal (float 0.05) 40. (coverage (render ~density:2. 10. 10. p)));
      test "an image to near max_float covers the page" (fun () ->
          let p =
            image ~box:(Box2.v 0. 0. 1e308 1e308) [| 1; 1; 1 |] [| 255 |]
          in
          equal float_exact 400. (coverage (render ~density:2. 10. 10. p)));
      cases ~name:(Printf.sprintf "under a scale of %g")
        "a pen keeps its width" [ 1e-170; 1e170 ] (fun k ->
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
      cases ~name:fst
        "transforms whose composition overflows or underflows paint nothing"
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
      prop ~tags:[ "slow" ] ~examples:Vg_corpus.extremes
        "any scale draws without failing" Vg_corpus.gen_extreme (fun p ->
          equal (array int) [| 20; 20; 4 |] (Nx.shape (render 20. 20. p)));
      test "a rotated stroke near max_float draws without failing" (fun () ->
          let far = segment (-1e308) (-1e308) 1e308 1e308 in
          let p =
            Picture.transform
              Affine.(translate 5. 5. * rotate 0.3)
              (Picture.stroke (Stroke.v 1.) red far)
          in
          equal (array int) [| 20; 20; 4 |]
            (Nx.shape (render ~density:2. 10. 10. p)));
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

let png_group =
  let page =
    Renderable.v 20. 10. (Picture.fill red (Path.circle (P2.v 10. 5.) 4.))
  in
  group "png"
    [
      test "png records the density and sRGB" (fun () ->
          let png = Hugin_vg_raster.png ~density:2. page in
          (* 144 dpi is 5669.29 pixels per metre. *)
          equal (option string)
            (Some (be32 5669 ^ be32 5669 ^ "\001"))
            (chunk "pHYs" png);
          equal (option string) (Some "\000") (chunk "sRGB" png));
      test "png holds the pixels of render" (fun () ->
          let rgb =
            Nx.slice
              [ A; A; R (0, 3) ]
              (Hugin_vg_raster.render ~density:2. page)
          in
          same_image (Nx.contiguous rgb)
            (Vg_corpus.load_png (Hugin_vg_raster.png ~density:2. page)));
      cases ~name:(Format.asprintf "%g")
        "png raises on a density beyond a PNG's resolution, unrendered"
        [ 1e6; 1e-4 ] (fun density ->
          (* A page of 1e10 by 1e10 pixels at either density, which render would
             refuse for its size. *)
          let side = 1e10 /. density in
          raises_match (Exn.invalid_arg ~substring:"resolution") (fun () ->
              Hugin_vg_raster.png ~density
                (Renderable.v side side Picture.empty)));
      cases ~name:fst "png accepts the ends of a PNG's resolution"
        [
          ("1 pixel per metre", (0.0254 /. 72., 3000.));
          ("2^31 - 1 pixels per metre", (2147483647. *. 0.0254 /. 72., 1e-6));
        ]
        (fun (_, (density, side)) ->
          let png =
            Hugin_vg_raster.png ~density
              (Renderable.v side side Picture.empty)
          in
          equal string "\137PNG" (String.sub png 0 4));
    ]

(* Goldens *)

let goldens =
  group "goldens"
    (prop "equal renderables give equal pixels" Vg_corpus.gen_picture (fun p ->
         same_image (render 100. 100. p)
           (render 100. 100. (Vg_corpus.respell p)))
    :: List.map
         (fun (name, r) ->
           test (name ^ " draws its golden PNG") (fun () ->
               let png = Hugin_vg_raster.png ~density:2. r in
               equal ~msg:"twice" string png
                 (Hugin_vg_raster.png ~density:2. r);
               (* The page is opaque, so the colours its PNG loads as are all
                  there is to compare. *)
               let img = Hugin_vg_raster.render ~density:2. r in
               equal ~msg:"least alpha" int 255
                 (Nx.item [] (Nx.min (Nx.slice [ A; A; I 3 ] img)));
               Vg_corpus.golden ("golden/" ^ name ^ ".png") png))
         Vg_corpus.pages)

let () =
  exit
    (run "hugin.vg raster"
       [
         pages;
         areas;
         clips;
         strokes;
         compositing;
         transforms;
         stamps;
         glyphs;
         images;
         limits;
         png_group;
         goldens;
       ])
