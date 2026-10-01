(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_next_gg
open Hugin_next_font
open Hugin_next_vg

let render ?(density = 1.) w h p =
  Hugin_next_vg_raster.render ~density (Renderable.v w h p)

let rgba img y x =
  ( Nx.item [ y; x; 0 ] img,
    Nx.item [ y; x; 1 ] img,
    Nx.item [ y; x; 2 ] img,
    Nx.item [ y; x; 3 ] img )

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

(* Pages *)

let sizes =
  [
    ("whole points", (10., 5., 1.), [| 5; 10; 4 |]);
    ("fractions rounded to the nearest", (10.4, 5.6, 1.), [| 6; 10; 4 |]);
    ("a density of two", (360., 240., 2.), [| 480; 720; 4 |]);
    ( "a product a hair above an integer",
      (100.00000000000001, 1., 2.),
      [| 2; 200; 4 |] );
  ]

let pages =
  group "pages"
    [
      cases
        ~name:(fun (n, _, _) -> n)
        "the pixel grid of" sizes
        (fun (_, (w, h, density), shape) ->
          equal (array int) shape (Nx.shape (render ~density w h Picture.empty)));
      test "pixels painted nothing are transparent black" (fun () ->
          equal float_exact 0. (coverage (render 7. 3. Picture.empty));
          equal color (0, 0, 0, 0) (rgba (render 7. 3. Picture.empty) 1 1));
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
      test "pixel (i, j) shows the square at (j / d, i / d)" (fun () ->
          let img =
            render ~density:2. 3. 2. (Picture.fill red (rect 1.5 0.5 0.5 0.5))
          in
          equal color (255, 0, 0, 255) (rgba img 1 3);
          equal (float 1e-9) 1. (coverage img));
    ]

(* Coverage *)

let rect_area (x, y, w, h) =
  let img = render 30. 30. (Picture.fill red (rect x y w h)) in
  (* Pixels the edges cross round to 8 bits, half a level each at most. *)
  let tolerance = ((2. *. (w +. h)) +. 8.) *. 0.5 /. 255. in
  let got = coverage img in
  is_true
    ~msg:(Printf.sprintf "covers %g for an area of %g" got (w *. h))
    (Float.abs (got -. (w *. h)) <= tolerance)

let gen_rect =
  let open Gen in
  let+ x = float_range 0. 15.
  and+ y = float_range 0. 15.
  and+ w = float_range 0. 14.
  and+ h = float_range 0. 14. in
  (x, y, w, h)

let triangle = Path.polygon [| -10.; 10.; -10. |] [| 0.; 0.; 20. |]

let coverage_group =
  group "coverage"
    [
      prop "a filled rectangle covers its area" gen_rect rect_area;
      test "a pixel takes the fraction of its square that a fill covers"
        (fun () ->
          let img = render 1. 1. (Picture.fill red (rect 0.25 0. 0.75 1.)) in
          equal color (255, 0, 0, 191) (rgba img 0 0));
      test "a slanted edge covers exactly" (fun () ->
          let img =
            render 10. 10.
              (Picture.fill red
                 (Path.polygon [| 0.; 10.; 0. |] [| 0.; 0.; 10. |]))
          in
          equal (float 0.05) 50. (coverage img));
      test "an edge leaving the page is cut where it leaves" (fun () ->
          equal (float 0.05) 50.
            (coverage (render 10. 10. (Picture.fill red triangle))));
      test "an edge leaving a clip is cut where it leaves" (fun () ->
          let p =
            Picture.clip (rect 2. 2. 4. 4.)
              (Picture.fill red
                 (Path.polygon [| 0.; 8.; 0. |] [| 0.; 0.; 8. |]))
          in
          (* The triangle's edge x + y = 8 halves the clip's square. *)
          equal (float 0.05) 8. (coverage (render 10. 10. p)));
      test "even-odd leaves a hole where nonzero fills" (fun () ->
          let both = Path.append (rect 3. 3. 4. 4.) (rect 0. 0. 10. 10.) in
          equal (float 0.01) 100.
            (coverage (render 10. 10. (Picture.fill red both)));
          equal (float 0.01) 84.
            (coverage (render 10. 10. (Picture.fill ~rule:`Even_odd red both))));
      test "nonzero fills both orientations alike" (fun () ->
          let cw = Path.polygon [| 1.; 5.; 5.; 1. |] [| 1.; 1.; 5.; 5. |] in
          let ccw = Path.polygon [| 1.; 1.; 5.; 5. |] [| 1.; 5.; 5.; 1. |] in
          same_image
            (render 6. 6. (Picture.fill red cw))
            (render 6. 6. (Picture.fill red ccw)));
      test "an area closes its open subpaths" (fun () ->
          let xs = [| 2.; 18.; 2. |] and ys = [| 2.; 2.; 18. |] in
          same_image
            (render 20. 20. (Picture.fill red (Path.polygon xs ys)))
            (render 20. 20. (Picture.fill red (Path.polyline xs ys))));
      test "a circle covers its area, its chords within a tenth of a pixel"
        (fun () ->
          let img =
            render 40. 40. (Picture.fill red (Path.circle (P2.v 20. 20.) 15.))
          in
          let disc = Float.pi *. 15. *. 15. in
          (* Chords within 0.1 of the circle lose at most that times its
             perimeter. *)
          at_most float_exact ~than:disc (coverage img);
          at_least float_exact
            ~than:(disc -. (0.1 *. 2. *. Float.pi *. 15.))
            (coverage img));
    ]

(* Clips *)

let clips =
  group "clips"
    [
      test "a clip lets through its area only" (fun () ->
          let img =
            render 10. 10.
              (Picture.clip (rect 0. 0. 5. 10.)
                 (Picture.fill red (rect 0. 0. 10. 10.)))
          in
          equal (float 1e-9) 50. (coverage img);
          equal color (0, 0, 0, 0) (rgba img 5 7));
      test "clips nest by intersection" (fun () ->
          let p =
            Picture.clip (rect 0. 0. 6. 10.)
              (Picture.clip (rect 3. 0. 7. 10.)
                 (Picture.fill red (rect 0. 0. 10. 10.)))
          in
          equal (float 1e-9) 30. (coverage (render 10. 10. p)));
      test "a curved clip lets through the fraction it covers" (fun () ->
          let circle = Path.circle (P2.v 20. 20.) 10. in
          let clipped =
            render 40. 40.
              (Picture.clip circle (Picture.fill red (rect 0. 0. 40. 40.)))
          in
          let filled = render 40. 40. (Picture.fill red circle) in
          equal (float 0.5) (coverage filled) (coverage clipped));
      test "an even-odd clip lets through its ring" (fun () ->
          let ring = Path.append (rect 3. 3. 4. 4.) (rect 0. 0. 10. 10.) in
          let p =
            Picture.clip ~rule:`Even_odd ring
              (Picture.fill red (rect 0. 0. 10. 10.))
          in
          equal (float 0.01) 84. (coverage (render 10. 10. p)));
      test "a crossed quadrilateral clips as its two triangles" (fun () ->
          let bowtie =
            Path.polygon [| 0.; 10.; 10.; 0. |] [| 0.; 10.; 0.; 10. |]
          in
          let p = Picture.clip bowtie (Picture.fill red (rect 0. 0. 10. 10.)) in
          equal (float 0.05) 50. (coverage (render 10. 10. p)));
      cases ~name:fst
        "a clip of axis-aligned sides around no area shows nothing"
        [
          ("right, left, down, up", ([| 0.; 4.; 0.; 0. |], [| 0.; 0.; 0.; 4. |]));
          ("down, right, left, up", ([| 0.; 0.; 4.; 0. |], [| 0.; 4.; 4.; 4. |]));
          ("down, up, right, left", ([| 0.; 0.; 0.; 4. |], [| 0.; 4.; 0.; 0. |]));
          ("right, down, up, left", ([| 0.; 4.; 4.; 4. |], [| 0.; 0.; 4.; 0. |]));
        ]
        (fun (_, (xs, ys)) ->
          let p =
            Picture.clip (Path.polygon xs ys)
              (Picture.fill red (rect 0. 0. 10. 10.))
          in
          equal float_exact 0. (coverage (render 10. 10. p)));
      test "a clip off pixel boundaries covers its edge pixels in part"
        (fun () ->
          let img =
            render 10. 10.
              (Picture.clip (rect 0.5 0. 9.5 10.)
                 (Picture.fill red (rect 0. 0. 10. 10.)))
          in
          let _, _, _, a = rgba img 5 0 in
          equal int 128 a);
      cases ~name:fst
        "a clip on pixel boundaries beside a curved clip shows nothing"
        [ ("next to its pixels", 6.); ("apart from its pixels", 8.) ]
        (fun (_, x) ->
          let p =
            Picture.clip
              (Path.circle (P2.v 3. 5.) 2.)
              (Picture.clip (rect x 0. 2. 10.)
                 (Picture.fill red (rect 0. 0. 10. 10.)))
          in
          equal float_exact 0. (coverage (render 10. 10. p)));
      test "a clip on pixel boundaries within a curved clip keeps its curve"
        (fun () ->
          let circle = Path.circle (P2.v 5. 5.) 4. and box = rect 2. 3. 6. 5. in
          let page = Picture.fill red (rect 0. 0. 10. 10.) in
          same_image ~within:1
            (render 10. 10. (Picture.clip box (Picture.clip circle page)))
            (render 10. 10. (Picture.clip circle (Picture.clip box page))));
      test "a clip off the page shows nothing" (fun () ->
          let p =
            Picture.clip (rect 20. 20. 5. 5.)
              (Picture.fill red (rect 0. 0. 30. 30.))
          in
          equal float_exact 0. (coverage (render 10. 10. p)));
    ]

(* Strokes *)

let line = Path.polyline [| 5.; 15. |] [| 10.; 10. |]
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
    ("a miter fills the outer square", `Miter, 64., 0.);
    ("a bevel fills half of it", `Bevel, 62., 0.);
    ("a round join fills a quarter disc", `Round, 60. +. Float.pi, Float.pi);
  ]

let dashes =
  [
    ("dashes and gaps", ([ 4.; 2. ], 0.), 28.);
    ("an offset into the pattern", ([ 4.; 2. ], 1.), 28.);
    ("an odd pattern, repeated", ([ 3. ], 0.), 22.);
    ("a negative offset", ([ 4.; 2. ], -5.), 28.);
  ]

let dashed (pattern, dash_offset) =
  let line = Path.polyline [| 0.; 20. |] [| 5.; 5. |] in
  coverage
    (render 20. 10.
       (Picture.stroke
          (Stroke.v ~cap:`Butt ~dash:pattern ~dash_offset 2.)
          red line))

let point =
  Path.empty |> Path.move_to (P2.v 10. 10.) |> Path.line_to (P2.v 10. 10.)

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
        (fun (_, join, a, arc) ->
          let p = Picture.stroke (Stroke.v ~cap:`Butt ~join 4.) red corner in
          curved ~arc a (coverage (render 20. 20. p)));
      test "a miter beyond the limit is bevelled" (fun () ->
          let p =
            Picture.stroke
              (Stroke.v ~cap:`Butt ~join:`Miter ~miter_limit:1.2 4.)
              red corner
          in
          equal (float 0.1) 62. (coverage (render 20. 20. p)));
      cases
        ~name:(fun (n, _, _) -> n)
        "a dashed line 20 long and 2 wide, with" dashes
        (fun (_, d, a) -> equal (float 0.05) a (dashed d));
      cases
        ~name:(fun (n, _, _) -> n)
        "a subpath of zero length"
        [
          ("is a disc with round caps", `Round, Float.pi *. 4.);
          ("is nothing with butt caps", `Butt, 0.);
          ("is nothing with square caps, having no direction", `Square, 0.);
        ]
        (fun (_, cap, a) ->
          let p = Picture.stroke (Stroke.v ~cap 4.) red point in
          curved ~arc:a a (coverage (render 20. 20. p)));
      cases
        ~name:(fun (n, _, _, _) -> n)
        "dashes of zero length on a line"
        [
          ("are squares along it with square caps", `Square, 12., 0.);
          ("are discs with round caps", `Round, 3. *. Float.pi, 6. *. Float.pi);
          ("are nothing with butt caps", `Butt, 0., 0.);
        ]
        (fun (_, cap, a, arc) ->
          let line = Path.polyline [| 2.; 10. |] [| 5.; 5. |] in
          let p = Picture.stroke (Stroke.v ~cap ~dash:[ 0.; 4. ] 2.) red line in
          curved ~arc a (coverage (render 20. 10. p)));
      test "a zero-length dash takes the direction of its line" (fun () ->
          let diagonal = Path.polyline [| 0.; 10. |] [| 0.; 10. |] in
          let s = Stroke.v ~cap:`Square ~dash:[ 0.; 100. ] 2. in
          let img =
            render 20. 20.
              (Picture.transform (Affine.translate 5. 5.)
                 (Picture.stroke s red diagonal))
          in
          (* A square turned by 45 degrees about (5, 5) covers part of the pixel
             at its centre's lower right, which an upright one covers whole. *)
          equal (float 0.1) 4. (coverage img);
          let _, _, _, a = rgba img 5 5 in
          less ~msg:"alpha of pixel (5, 5)" int ~than:255 a);
      test "a corner sharper than the miter limit is bevelled" (fun () ->
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
                 (Picture.stroke
                    (Stroke.v ~cap:`Butt ~join ~miter_limit:4. 2.)
                    red q))
          in
          equal (float 1e-9) (draw `Bevel) (draw `Miter));
      test "dashes of zero length on a vertical line are squares along it"
        (fun () ->
          let line = Path.polyline [| 5.; 5. |] [| 2.; 10. |] in
          let p =
            Picture.stroke (Stroke.v ~cap:`Square ~dash:[ 0.; 4. ] 2.) red line
          in
          equal (float 0.05) 12. (coverage (render 10. 20. p)));
      cases ~name:fst "a line too short to see is a square with square caps"
        [ ("vertical", (0., 0.01)); ("horizontal", (0.01, 0.)) ]
        (fun (_, (dx, dy)) ->
          let q = Path.polyline [| 10.; 10. +. dx |] [| 10.; 10. +. dy |] in
          let p = Picture.stroke (Stroke.v ~cap:`Square 2.) red q in
          equal (float 0.05) 4. (coverage (render 20. 20. p)));
      test "a turn keeps a zigzag's area" (fun () ->
          (* Its segments run along the diagonals, which a turn by 45 degrees
             lays along the axes. *)
          let zig =
            Path.polyline [| 0.; 3.; 6.; 9.; 12. |] [| 0.; -3.; 0.; -3.; 0. |]
          in
          let draw m =
            coverage
              (render 40. 40.
                 (Picture.transform m
                    (Picture.stroke
                       (Stroke.v ~cap:`Butt ~join:`Bevel 1.)
                       red zig)))
          in
          equal (float 0.2)
            (draw (Affine.translate 14. 20.))
            (draw Affine.(translate 14. 20. * rotate (Float.pi /. 4.))));
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
      test "a dashed line beyond the page reaches it with a stretched pen"
        (fun () ->
          (* Stretched 4 times vertically, the pen 4 wide reaches 8 from the
             line, which lies 6 above the page: dashes of 2 every 4 cover two
             rows of it. *)
          let p =
            Picture.transform (Affine.scale 1. 4.)
              (Picture.stroke
                 (Stroke.v ~cap:`Butt ~dash:[ 2.; 2. ] 4.)
                 red
                 (Path.polyline [| -10.; 30. |] [| -1.5; -1.5 |]))
          in
          equal (float 0.05) 20. (coverage (render 20. 20. p)));
      test "a dashed line beyond the page reaches it with a sheared pen"
        (fun () ->
          (* The shear lays the line 6 above the page and the pen 4 wide reaches
             8.25 from it: half of its solid coverage, about. *)
          let shear =
            { Affine.xx = 1.; yx = 4.; xy = 0.; yy = 1.; x0 = 0.; y0 = 0. }
          in
          let draw dash =
            coverage
              (render 20. 20.
                 (Picture.transform shear
                    (Picture.stroke
                       (Stroke.v ~cap:`Butt ~dash 4.)
                       red
                       (Path.polyline [| -10.; 30. |] [| 34.; -126. |]))))
          in
          let solid = draw [] and dashed = draw [ 2.; 2. ] in
          greater ~msg:"solid" float_exact ~than:10. solid;
          greater ~msg:"dashed" float_exact ~than:(0.3 *. solid) dashed;
          less ~msg:"dashed" float_exact ~than:(0.7 *. solid) dashed);
      cases ~name:fst "a thin dashed line along the page's edge is drawn"
        [
          ("left", (0.3, 0., 0.3, 20.));
          ("top", (0., 0.3, 20., 0.3));
          ("right", (19.7, 0., 19.7, 20.));
          ("bottom", (0., 19.7, 20., 19.7));
        ]
        (fun (_, (x0, y0, x1, y1)) ->
          let p =
            Picture.stroke
              (Stroke.v ~cap:`Butt ~dash:[ 2.; 2. ] 0.4)
              red
              (Path.polyline [| x0; x1 |] [| y0; y1 |])
          in
          equal (float 0.05) 4. (coverage (render 20. 20. p)));
      cases ~name:fst "a dashed line entering the page keeps its phase"
        [
          ("from the left", ((-7.5, 5.), (20., 5.), fun i -> (4, i)));
          ("from the right", ((27.5, 5.), (0., 5.), fun i -> (4, 19 - i)));
          ("from above", ((5., -7.5), (5., 20.), fun i -> (i, 4)));
          ("from below", ((5., 27.5), (5., 0.), fun i -> (19 - i, 4)));
        ]
        (fun (_, ((x0, y0), (x1, y1), at)) ->
          (* Dashes of 2 every 4 from 7.5 before the page: on it, from 0.5 to
             2.5, 4.5 to 6.5 and so on. *)
          let img =
            render 20. 20.
              (Picture.stroke
                 (Stroke.v ~cap:`Butt ~dash:[ 2.; 2. ] 2.)
                 red
                 (Path.polyline [| x0; x1 |] [| y0; y1 |]))
          in
          List.iter
            (fun (i, a) ->
              let y, x = at i in
              let _, _, _, got = rgba img y x in
              equal ~msg:(Printf.sprintf "pixel %d along" i) int a got)
            [
              (0, 128);
              (1, 255);
              (2, 128);
              (3, 0);
              (5, 255);
              (7, 0);
              (9, 255);
              (11, 0);
              (13, 255);
              (15, 0);
              (17, 255);
              (19, 0);
            ]);
      test "a dashed line back on the page keeps the phase of its way out"
        (fun () ->
          (* Along the way back, the point at x lies 76 - x along the path. *)
          let q = Path.polyline [| 2.; 34.; 34.; 2. |] [| 5.; 5.; 15.; 15. |] in
          let img =
            render 20. 20.
              (Picture.stroke (Stroke.v ~cap:`Butt ~dash:[ 2.; 2. ] 2.) red q)
          in
          List.iter
            (fun (x, a) ->
              let _, _, _, got = rgba img 14 x in
              equal ~msg:(Printf.sprintf "column %d" x) int a got)
            [ (19, 255); (18, 255); (17, 0); (16, 0); (15, 255); (14, 255) ]);
      test "a dashed subpath of a start alone is a disc with round caps"
        (fun () ->
          let lone = Path.empty |> Path.move_to (P2.v 10. 10.) in
          let p =
            Picture.stroke (Stroke.v ~cap:`Round ~dash:[ 2.; 2. ] 4.) red lone
          in
          curved ~arc:(4. *. Float.pi) (4. *. Float.pi)
            (coverage (render 20. 20. p)));
      test "a dashed closed subpath dashes its closing side" (fun () ->
          (* Each side of 10 holds two dashes of 3. *)
          let p =
            Picture.stroke
              (Stroke.v ~cap:`Butt ~dash:[ 3.; 2. ] 1.)
              red (rect 5. 5. 10. 10.)
          in
          (* The half pixels along its edges round to 128 of 255 each. *)
          equal (float 0.1) 24. (coverage (render 20. 20. p)));
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
        (fun (_, (s, xs, again)) ->
          let draw xs =
            render 20. 10.
              (Picture.stroke s red
                 (Path.polyline xs (Array.make (Array.length xs) 5.)))
          in
          let repeated =
            Array.of_list
              (List.concat_map
                 (fun x -> if x = again then [ x; x ] else [ x ])
                 (Array.to_list xs))
          in
          same_image (draw xs) (draw repeated));
      test "a dash pattern longer than a thousandth of a pixel is dashed"
        (fun () ->
          (* The map stretches by 3 along the diagonal the line follows, so the
             period of 0.0004 is 0.0012 of a pixel there. *)
          let m =
            { Affine.xx = 1.; yx = 2.; xy = 2.; yy = 1.; x0 = 5.; y0 = 5. }
          in
          let draw dash =
            coverage
              (render 30. 30.
                 (Picture.transform m
                    (Picture.stroke
                       (Stroke.v ~cap:`Butt ~dash 1.)
                       red
                       (Path.polyline [| 0.; 4. |] [| 0.; 4. |]))))
          in
          let solid = draw [] in
          greater ~msg:"solid" float_exact ~than:10. solid;
          at_most ~msg:"dashed" float_exact ~than:(0.5 *. solid)
            (draw [ 0.0002; 0.0002 ]));
      test "a closed subpath joins at its start" (fun () ->
          let square = rect 4. 4. 8. 8. in
          let miter = Picture.stroke (Stroke.v ~join:`Miter 2.) red square in
          (* An 8 by 8 square stroked 2 wide with mitred corners is a 10 by 10
             frame around a 6 by 6 hole. *)
          equal (float 0.05) 64. (coverage (render 20. 20. miter)));
      test "a closed subpath smaller than a pixel keeps its pen" (fun () ->
          (* Stroked 8 wide with mitred corners, a square of side 0.004 is a
             square of side 8.004, without a hole. *)
          let tiny = rect 9.998 9.998 0.004 0.004 in
          let p =
            Picture.stroke (Stroke.v ~cap:`Butt ~join:`Miter 8.) red tiny
          in
          equal (float 0.1) (8.004 *. 8.004) (coverage (render 20. 20. p)));
      cases ~name:fst
        "a circle smaller than a pixel draws more ink as it grows, with"
        [ ("miter joins", `Miter); ("round joins", `Round); ("bevels", `Bevel) ]
        (fun (_, join) ->
          let ink r =
            let circle = Path.circle (P2.v 10. 10.) r in
            coverage
              (render 20. 20. (Picture.stroke (Stroke.v ~join 4.) red circle))
          in
          ignore
            (List.fold_left
               (fun (r, before) r' ->
                 let after = ink r' in
                 at_least
                   ~msg:(Printf.sprintf "radius %g against %g" r' r)
                   float_exact ~than:before after;
                 (r', after))
               (0.005, ink 0.005)
               [ 0.01; 0.03; 0.06; 0.12 ]));
      test "a circle smaller than a pixel covers at least its pen" (fun () ->
          List.iter
            (fun r ->
              let circle = Path.circle (P2.v 10. 10.) r in
              let p = Picture.stroke (Stroke.v ~join:`Miter 4.) red circle in
              at_least
                ~msg:(Printf.sprintf "radius %g" r)
                float_exact ~than:(4. *. Float.pi)
                (coverage (render 20. 20. p)))
            [ 0.001; 0.01; 0.03; 0.06 ]);
      test "a transform maps the pen" (fun () ->
          let p =
            Picture.transform (Affine.scale 1. 4.)
              (Picture.stroke (Stroke.v ~cap:`Butt 1.) red line)
          in
          equal (float 0.05) 40. (coverage (render 20. 60. p)));
      test "a line far beyond the page is cut to it" (fun () ->
          let far = Path.polyline [| -1e9; 1e9 |] [| 5.; 5. |] in
          equal (float 0.05) 40.
            (coverage (render 20. 10. (Picture.stroke (Stroke.v 2.) red far))));
      test "a dashed line far beyond the page dashes only the page" (fun () ->
          let far = Path.polyline [| -1e9; 1e9 |] [| 5.; 5. |] in
          let s = Stroke.v ~cap:`Butt ~dash:[ 2.; 2. ] 2. in
          equal (float 0.05) 40.
            (coverage (render 40. 10. (Picture.stroke s red far))));
    ]

(* Compositing *)

let gen_color =
  let unit = Gen.float_range 0. 1. in
  Gen.with_pp Color.pp
    (Gen.map
       (fun (r, g, b, alpha) -> Color.v ~alpha r g b)
       (Gen.quad unit unit unit unit))

(* [over_formula (c, c')] checks a pixel painted [c'] then [c] against the
   formula of source-over, in premultiplied levels, which the output's 8-bit
   straight alpha keeps to a level and a half. *)
let over_formula (c, c') =
  let whole = rect 0. 0. 1. 1. in
  let img =
    render 1. 1. (Picture.group [ Picture.fill c' whole; Picture.fill c whole ])
  in
  let a = Color.alpha c and a' = Color.alpha c' in
  let a'' = a +. (a' *. (1. -. a)) in
  let r, g, b, alpha = rgba img 0 0 in
  at_most ~msg:"alpha" (float 1e-9) ~than:1.
    (Float.abs (Float.of_int alpha -. (255. *. a'')));
  List.iter
    (fun (name, got, k) ->
      let premultiplied = 255. *. ((a *. k c) +. (a' *. (1. -. a) *. k c')) in
      at_most ~msg:name (float 1e-9) ~than:1.5
        (Float.abs
           ((Float.of_int got *. Float.of_int alpha /. 255.) -. premultiplied)))
    [ ("red", r, Color.r); ("green", g, Color.g); ("blue", b, Color.b) ]

let compositing =
  group "compositing"
    [
      prop "fills composite source-over on encoded components"
        (Gen.pair gen_color gen_color)
        over_formula;
      test "source-over on encoded components" (fun () ->
          let half = Color.with_alpha 0.5 red in
          let img =
            render 1. 1.
              (Picture.group
                 [
                   Picture.fill blue (rect 0. 0. 1. 1.);
                   Picture.fill half (rect 0. 0. 1. 1.);
                 ])
          in
          let r, g, b, a = rgba img 0 0 in
          equal ~msg:"red" int 128 r;
          equal ~msg:"green" int 0 g;
          equal ~msg:"blue" int 128 b;
          equal ~msg:"alpha" int 255 a);
      test "output has straight alpha" (fun () ->
          let img =
            render 1. 1.
              (Picture.fill (Color.with_alpha 0.5 red) (rect 0. 0. 1. 1.))
          in
          equal color (255, 0, 0, 128) (rgba img 0 0));
      test "a translucent colour reads back as its nearest level" (fun () ->
          (* Premultiplied by an alpha of 153 levels, a grey of 31/153 is 31
             levels exactly; straight, it is 51.67 levels. *)
          let g = 31. /. 153. in
          let img =
            render 1. 1.
              (Picture.fill (Color.v ~alpha:0.6 g g g) (rect 0. 0. 1. 1.))
          in
          equal color (52, 52, 52, 153) (rgba img 0 0));
      test "group opacity fades the group as one" (fun () ->
          let two =
            Picture.group
              [
                Picture.fill red (rect 0. 0. 2. 1.);
                Picture.fill blue (rect 1. 0. 2. 1.);
              ]
          in
          let img = render 3. 1. (Picture.opacity 0.5 two) in
          equal ~msg:"the overlap shows the later only" color (0, 0, 255, 128)
            (rgba img 0 1);
          equal color (255, 0, 0, 128) (rgba img 0 0));
      test "per-leaf opacity lets the earlier show through" (fun () ->
          let img =
            render 3. 1.
              (Picture.group
                 [
                   Picture.opacity 0.5 (Picture.fill red (rect 0. 0. 2. 1.));
                   Picture.opacity 0.5 (Picture.fill blue (rect 1. 0. 2. 1.));
                 ])
          in
          let r, _, b, a = rgba img 0 1 in
          is_true ~msg:"red shows through" (r > 0 && b > r);
          at_most ~msg:"alpha, 0.75 rounded per primitive" int ~than:1
            (abs (a - 191)));
      test "opacity 0. paints nothing" (fun () ->
          equal float_exact 0.
            (coverage
               (render 4. 4.
                  (Picture.opacity 0. (Picture.fill red (rect 0. 0. 4. 4.))))));
      test "tags change nothing" (fun () ->
          let p = Picture.fill red (Path.circle (P2.v 5. 5.) 3.) in
          let t =
            { Picture.id = Nx.Ptree.Path.v [ Field "a" ]; rows = Rows [| 1 |] }
          in
          same_image (render 10. 10. p) (render 10. 10. (Picture.tag t p)));
      test "a translucent opacity of a translucent fill multiplies" (fun () ->
          let img =
            render 1. 1.
              (Picture.opacity 0.5
                 (Picture.fill (Color.with_alpha 0.5 red) (rect 0. 0. 1. 1.)))
          in
          let _, _, _, a = rgba img 0 0 in
          equal int 64 a);
    ]

(* Transforms *)

let transforms =
  group "transforms"
    [
      test "a scale scales the area" (fun () ->
          let p =
            Picture.transform (Affine.scale 2. 3.)
              (Picture.fill red (rect 1. 1. 2. 2.))
          in
          equal (float 0.01) 24. (coverage (render 20. 20. p)));
      test "a rotation keeps the area" (fun () ->
          let p =
            Picture.transform
              Affine.(translate 10. 10. * rotate (Float.pi /. 6.))
              (Picture.fill red (rect (-3.) (-3.) 6. 6.))
          in
          equal (float 0.1) 36. (coverage (render 20. 20. p)));
      test "a mirror keeps the area" (fun () ->
          let p =
            Picture.transform
              Affine.(translate 10. 0. * scale (-1.) 1.)
              (Picture.fill red (rect 1. 1. 3. 3.))
          in
          equal (float 0.01) 9. (coverage (render 10. 10. p)));
    ]

(* Stamps *)

let marker_of fill stroke =
  Picture.group
    [
      Picture.fill fill (Path.circle (P2.v 0. 0.) 3.);
      Picture.stroke (Stroke.v 1.) stroke (Path.circle (P2.v 0. 0.) 3.);
    ]

let marker = marker_of (Color.v 0.2 0.4 0.8) (Color.v ~alpha:0.8 0.9 0.1 0.1)

(* A square filled with [fill] and outlined with [stroke] [pen] wide, beside a
   letter in [fill]: a leaf of each colour a stamp replaces. *)
let lettered ?(pen = 1.) fill stroke =
  let i = Font.glyph Font.regular (Uchar.of_char 'I') in
  let run =
    Run.v ~font:Font.regular ~size:8. ~text:"I" ~glyphs:[| i |] ~xs:[| 0. |] ()
  in
  Picture.group
    [
      Picture.fill fill (rect (-3.) (-3.) 3. 6.);
      Picture.stroke (Stroke.v pen) stroke (rect (-3.) (-3.) 3. 6.);
      Picture.glyphs fill (P2.v 1. 3.) run;
    ]

(* Positions on the quarter-pixel grid, where stamps place instances exactly. *)
let gen_quarters n =
  Gen.array ~size:(Gen.constant n)
    (Gen.map (fun k -> Float.of_int k /. 4.) (Gen.int_range 0 160))

let instances xs ys p =
  Picture.group
    (List.init (Array.length xs) (fun i ->
         Picture.transform (Affine.translate xs.(i) ys.(i)) (p i)))

let stamps =
  group "stamps"
    [
      prop "a stamp draws its instances, placed on quarter pixels"
        (Gen.pair (gen_quarters 6) (gen_quarters 6))
        (fun (xs, ys) ->
          same_image ~within:1
            (render 40. 40. (instances xs ys (Fun.const marker)))
            (render 40. 40. (Picture.stamp xs ys marker)));
      prop "an instance is drawn at its position rounded to a quarter pixel"
        (Gen.pair (Gen.float_range 5. 15.) (Gen.float_range 5. 15.))
        (fun (x, y) ->
          let quarter v = Float.round (v *. 4.) /. 4. in
          same_image ~within:1
            (render 20. 20.
               (Picture.transform
                  (Affine.translate (quarter x) (quarter y))
                  marker))
            (render 20. 20. (Picture.stamp [| x |] [| y |] marker)));
      test "instances round to the nearest quarter pixel" (fun () ->
          let at x = render 20. 10. (Picture.stamp [| x |] [| 5. |] marker) in
          same_image (at 10.) (at 10.1);
          same_image (at 10.25) (at 10.2);
          is_true ~msg:"10.25 differs from 10." (levels (at 10.) (at 10.25) > 0));
      test "instances take their fills and strokes" (fun () ->
          let xs = [| 10.; 20.; 30. |] and ys = [| 10.; 10.; 10. |] in
          let fills = [| red; Color.green; blue |]
          and strokes = [| blue; red; Color.black |] in
          same_image ~within:1
            (render 40. 20.
               (instances xs ys (fun i -> marker_of fills.(i) strokes.(i))))
            (render 40. 20. (Picture.stamp ~fills ~strokes xs ys marker)));
      test "scaled instances take their fills and strokes" (fun () ->
          let xs = [| 10.; 20.; 30. |] and ys = [| 10.; 10.; 10. |] in
          let scales = [| 1.; 2.; 0.5 |] in
          let fills = [| red; Color.green; blue |]
          and strokes = [| blue; red; Color.black |] in
          same_image ~within:1
            (render 40. 20.
               (instances xs ys (fun i ->
                    let s = scales.(i) in
                    Picture.transform (Affine.scale s s)
                      (lettered ~pen:(1. /. s) fills.(i) strokes.(i)))))
            (render 40. 20.
               (Picture.stamp ~fills ~strokes ~scales xs ys
                  (lettered Color.white Color.white))));
      test "instances of a stamp of stamps take their fills and strokes"
        (fun () ->
          let pair p = Picture.stamp [| 0.; 4. |] [| 0.; 0. |] p in
          let xs = [| 10.; 30. |] and ys = [| 10.; 10. |] in
          let fills = [| red; blue |] and strokes = [| blue; Color.green |] in
          same_image ~within:1
            (render 40. 20.
               (instances xs ys (fun i -> pair (lettered fills.(i) strokes.(i)))))
            (render 40. 20.
               (Picture.stamp ~fills ~strokes xs ys
                  (pair (lettered Color.white Color.white)))));
      test "a stamp of a faded marker fades each instance" (fun () ->
          let faded = Picture.opacity 0.5 marker in
          let xs = [| 10.; 13. |] and ys = [| 10.; 10. |] in
          same_image ~within:1
            (render 30. 20. (instances xs ys (Fun.const faded)))
            (render 30. 20. (Picture.stamp xs ys faded)));
      test "a stamp of images, glyphs and clips draws its instances" (fun () ->
          let px =
            Nx.init Nx.uint8 [| 2; 2; 4 |] (fun i ->
                (i.(0) * 90) + (i.(1) * 40) + (i.(2) * 30))
          in
          let font = Font.regular in
          let g = Font.glyph font (Uchar.of_char 'A') in
          let a =
            Run.v ~font ~size:8. ~text:"A" ~glyphs:[| g |] ~xs:[| 0. |] ()
          in
          let template =
            Picture.group
              [
                Picture.clip
                  (Path.circle (P2.v 0. 0.) 3.)
                  (Picture.image (Box2.v (-3.) (-3.) 6. 6.) px);
                Picture.glyphs Color.black (P2.v (-2.) 3.) a;
              ]
          in
          let xs = [| 6.25; 17.5; 30. |] and ys = [| 8.; 9.75; 8.5 |] in
          let fills = [| red; blue; Color.green |] in
          same_image ~within:1
            (render 40. 20.
               (instances xs ys (fun i ->
                    Picture.stamp
                      ~fills:[| fills.(i) |]
                      [| 0. |] [| 0. |] template)))
            (render 40. 20. (Picture.stamp ~fills xs ys template)));
      test "scaled instances keep their pens" (fun () ->
          let ring =
            Picture.stroke (Stroke.v 1.) red (Path.circle (P2.v 0. 0.) 5.)
          in
          let img =
            render 40. 40.
              (Picture.stamp ~scales:[| 2. |] [| 20. |] [| 20. |] ring)
          in
          (* A ring of radius 10 and width 1. *)
          equal (float 1.) (2. *. Float.pi *. 10.) (coverage img));
      test "an instance scaled to a point keeps its pen" (fun () ->
          let ring =
            Picture.stroke (Stroke.v 4.) red (Path.circle (P2.v 0. 0.) 5.)
          in
          let img =
            render 20. 20.
              (Picture.stamp ~scales:[| 1e-305 |] [| 10. |] [| 10. |] ring)
          in
          curved ~arc:(4. *. Float.pi) (4. *. Float.pi) (coverage img));
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
      test "an instance whose map has no inverse paints nothing" (fun () ->
          let ring =
            Picture.stroke (Stroke.v 4.) red (Path.circle (P2.v 0. 0.) 5.)
          in
          equal float_exact 0.
            (coverage
               (render 20. 20.
                  (Picture.stamp ~scales:[| 1e-310 |] [| 10. |] [| 10. |] ring))));
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
      test "a stamp of an image draws it at each instance" (fun () ->
          let px =
            Nx.init Nx.uint8 [| 2; 2; 4 |] (fun i ->
                if i.(2) = 3 then 200
                else (i.(0) * 120) + (i.(1) * 60) + (i.(2) * 30))
          in
          let img = Picture.image (Box2.v 0. 0. 4. 4.) px in
          let disc = Picture.clip (Path.circle (P2.v 2. 2.) 1.5) img in
          List.iter
            (fun p ->
              same_image ~within:1
                (render 30. 20.
                   (instances [| 5.; 13.25 |] [| 5.; 6.5 |] (Fun.const p)))
                (render 30. 20. (Picture.stamp [| 5.; 13.25 |] [| 5.; 6.5 |] p)))
            [ img; disc ]);
      test "a shrunk instance of an opacity keeps its pen" (fun () ->
          (* At a tenth of its size, the ring keeps its pen of 4: the same as a
             ring of a tenth of the size drawn with a pen of 40. *)
          let ring w =
            Picture.stroke (Stroke.v w) red (Path.circle (P2.v 0. 0.) 5.)
          in
          same_image ~within:1
            (render 20. 20.
               (Picture.opacity 0.5
                  (Picture.transform
                     Affine.(translate 10. 10. * scale 0.1 0.1)
                     (ring 40.))))
            (render 20. 20.
               (Picture.stamp ~scales:[| 0.1 |] [| 10. |] [| 10. |]
                  (Picture.opacity 0.5 (ring 4.)))));
      test "a shrunk instance of an opacity keeps a pen its transforms grow"
        (fun () ->
          (* The ring of radius 0.5 and pen 0.4, magnified 10 times within the
             instance, is the ring of radius 5 and pen 4 of the test above. *)
          let ring =
            Picture.transform (Affine.scale 10. 10.)
              (Picture.stroke (Stroke.v 0.4) red (Path.circle (P2.v 0. 0.) 0.5))
          in
          same_image ~within:1
            (render 20. 20.
               (Picture.opacity 0.5
                  (Picture.transform
                     Affine.(translate 10. 10. * scale 0.1 0.1)
                     (Picture.stroke (Stroke.v 40.) red
                        (Path.circle (P2.v 0. 0.) 5.)))))
            (render 20. 20.
               (Picture.stamp ~scales:[| 0.1 |] [| 10. |] [| 10. |]
                  (Picture.opacity 0.5 ring))));
      test "a stamp of stamps draws them all" (fun () ->
          let pair = Picture.stamp [| 0.; 8. |] [| 0.; 0. |] marker in
          same_image ~within:1
            (render 40. 20.
               (instances [| 5.; 13.; 21.; 29. |] [| 10.; 10.; 10.; 10. |]
                  (Fun.const marker)))
            (render 40. 20. (Picture.stamp [| 5.; 21. |] [| 10.; 10. |] pair)));
    ]

(* Glyphs *)

let set text size =
  let font = Font.regular in
  let n = String.length text in
  let glyphs =
    Array.init n (fun i -> Font.glyph font (Uchar.of_char text.[i]))
  in
  let xs = Array.make n 0. in
  for i = 1 to n - 1 do
    xs.(i) <- xs.(i - 1) +. (size *. Font.advance font glyphs.(i - 1))
  done;
  Run.v ~font ~size ~text ~glyphs ~xs ()

let glyphs =
  group "glyphs"
    [
      test "glyphs are their outlines, filled one by one" (fun () ->
          let r = set "Hug" 16. in
          let outlines =
            List.init (Run.length r) (fun i ->
                Picture.transform
                  Affine.(translate (4. +. Run.x r i) 20. * scale 16. 16.)
                  (Picture.fill red (Font.outline Font.regular (Run.glyph r i))))
          in
          same_image ~within:1
            (render 60. 30. (Picture.group outlines))
            (render 60. 30. (Picture.glyphs red (P2.v 4. 20.) r)));
      test "each glyph of a run is drawn at its own origin" (fun () ->
          let font = Font.regular in
          let g c = Font.glyph font (Uchar.of_char c) in
          let one c x y =
            Picture.glyphs red
              (P2.v (4. +. x) (14. +. y))
              (Run.v ~font ~size:12. ~text:(String.make 1 c)
                 ~glyphs:[| g c |]
                 ~xs:[| 0. |] ())
          in
          let run =
            Run.v ~ys:[| 0.; 6. |] ~font ~size:12. ~text:"ab"
              ~glyphs:[| g 'a'; g 'b' |]
              ~xs:[| 0.; 9. |] ()
          in
          same_image
            (render 30. 30. (Picture.group [ one 'a' 0. 0.; one 'b' 9. 6. ]))
            (render 30. 30. (Picture.glyphs red (P2.v 4. 14.) run)));
      test "overlapping translucent glyphs composite twice" (fun () ->
          let font = Font.regular
          and g = Font.glyph Font.regular (Uchar.of_char 'I') in
          let r =
            Run.v ~font ~size:20. ~text:"II" ~glyphs:[| g; g |] ~xs:[| 0.; 0. |]
              ()
          in
          let img =
            render 20. 30.
              (Picture.glyphs (Color.with_alpha 0.5 red) (P2.v 5. 25.) r)
          in
          let alphas =
            Array.to_list (Nx.to_array (Nx.slice [ A; A; I 3 ] img))
          in
          at_most ~msg:"the most opaque pixel, 0.75 rounded per glyph" int
            ~than:1
            (abs (List.fold_left Int.max 0 alphas - 191)));
    ]

(* Images *)

let images =
  group "images"
    [
      test "each cell takes its pixel, without interpolation" (fun () ->
          let px =
            Nx.create Nx.uint8 [| 2; 2; 3 |]
              [| 255; 0; 0; 0; 255; 0; 0; 0; 255; 10; 20; 30 |]
          in
          let img = render 4. 4. (Picture.image (Box2.v 0. 0. 4. 4.) px) in
          equal color (255, 0, 0, 255) (rgba img 1 1);
          equal color (0, 255, 0, 255) (rgba img 0 3);
          equal color (0, 0, 255, 255) (rgba img 3 0);
          equal color (10, 20, 30, 255) (rgba img 2 2));
      test "grey images are grey and opaque" (fun () ->
          let img =
            render 1. 1.
              (Picture.image (Box2.v 0. 0. 1. 1.)
                 (Nx.full Nx.uint8 [| 1; 1; 1 |] 77))
          in
          equal color (77, 77, 77, 255) (rgba img 0 0));
      test "RGBA images keep their straight alpha" (fun () ->
          let px = Nx.create Nx.uint8 [| 1; 1; 4 |] [| 200; 100; 50; 128 |] in
          let img = render 1. 1. (Picture.image (Box2.v 0. 0. 1. 1.) px) in
          let r, g, b, a = rgba img 0 0 in
          equal int 128 a;
          at_most ~msg:"red" int ~than:1 (abs (r - 200));
          at_most ~msg:"green" int ~than:1 (abs (g - 100));
          at_most ~msg:"blue" int ~than:1 (abs (b - 50)));
      test "a cell holds its left edge" (fun () ->
          (* Two columns over 3 pixels meet at 1.5, the centre of pixel 1. *)
          let px = Nx.create Nx.uint8 [| 1; 2; 1 |] [| 0; 255 |] in
          let img = render 3. 1. (Picture.image (Box2.v 0. 0. 3. 1.) px) in
          equal color (255, 255, 255, 255) (rgba img 0 1));
      test "a cell holds its top edge" (fun () ->
          (* Two rows over 3 pixels meet at 1.5, the centre of pixel row 1. *)
          let px = Nx.create Nx.uint8 [| 2; 1; 1 |] [| 0; 255 |] in
          let img = render 1. 3. (Picture.image (Box2.v 0. 0. 1. 3.) px) in
          equal color (255, 255, 255, 255) (rgba img 1 0));
      test "the last row and column hold their bottom and right edges"
        (fun () ->
          (* The box ends at the centre of pixel (1, 1). *)
          let px = Nx.full Nx.uint8 [| 1; 1; 1 |] 255 in
          let img = render 3. 3. (Picture.image (Box2.v 0. 0. 1.5 1.5) px) in
          equal color (255, 255, 255, 255) (rgba img 1 1);
          equal color (0, 0, 0, 0) (rgba img 2 2));
      test "an image shown smaller than its pixels averages them" (fun () ->
          let px =
            Nx.init Nx.uint8 [| 4; 4; 1 |] (fun i ->
                if (i.(0) + i.(1)) mod 2 = 0 then 0 else 255)
          in
          let img = render 1. 1. (Picture.image (Box2.v 0. 0. 1. 1.) px) in
          let r, _, _, _ = rgba img 0 0 in
          at_most int ~than:1 (abs (r - 128)));
      test "a pixel averages at most 4 by 4 image pixels" (fun () ->
          (* Eight columns in one pixel: 4 samples a row, at columns 1, 3, 5 and
             7, all white, where all eight average to grey. *)
          let px =
            Nx.init Nx.uint8 [| 8; 8; 1 |] (fun i ->
                if i.(1) mod 2 = 1 then 255 else 0)
          in
          let img = render 1. 1. (Picture.image (Box2.v 0. 0. 1. 1.) px) in
          equal color (255, 255, 255, 255) (rgba img 0 0));
      test "an image under a transform is drawn where the transform puts it"
        (fun () ->
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
      test "an image paints only the pixels whose centres it covers" (fun () ->
          let px = Nx.full Nx.uint8 [| 1; 1; 3 |] 255 in
          let img =
            render 10. 10. (Picture.image (Box2.v 0.25 0.25 2. 2.) px)
          in
          (* Centres at 0.5 and 1.5 lie within 0.25 to 2.25, those at 2.5 do
             not. *)
          List.iter
            (fun (y, x, a) ->
              let _, _, _, got = rgba img y x in
              equal ~msg:(Printf.sprintf "(%d, %d)" x y) int a got)
            [ (0, 0, 255); (1, 1, 255); (2, 1, 0); (1, 2, 0); (2, 2, 0) ]);
      test "a view of a tensor is read as its elements" (fun () ->
          let px =
            Nx.init Nx.uint8 [| 3; 2; 3 |] (fun i ->
                (i.(0) * 70) + (i.(1) * 30) + (i.(2) * 9))
          in
          let view = Nx.transpose ~axes:[ 1; 0; 2 ] px in
          let draw px = render 6. 4. (Picture.image (Box2.v 0. 0. 6. 4.) px) in
          same_image (draw (Nx.copy view)) (draw view));
    ]

(* Limits *)

let limits =
  group "limits"
    [
      test "a line to near max_float is drawn up to the page's edge" (fun () ->
          (* Its far end, at density 2, is beyond [max_float]. *)
          let far = Path.polyline [| 0.; 1e308 |] [| 5.; 5. |] in
          let p = Picture.stroke (Stroke.v ~cap:`Butt 1.) red far in
          equal (float 0.05) 40. (coverage (render ~density:2. 10. 10. p)));
      test "an image to near max_float covers the page" (fun () ->
          let px = Nx.full Nx.uint8 [| 1; 1; 1 |] 255 in
          let p = Picture.image (Box2.v 0. 0. 1e308 1e308) px in
          equal float_exact 400. (coverage (render ~density:2. 10. 10. p)));
      cases ~name:(Printf.sprintf "under a scale of %g")
        "a pen keeps its width" [ 1e-170; 1e170 ] (fun k ->
          (* A line 10 wide across the page, drawn in units of [1 /. k]. *)
          let line u =
            Picture.stroke
              (Stroke.v ~cap:`Butt (10. *. u))
              red
              (Path.polyline [| 0.; 20. *. u |] [| 10. *. u; 10. *. u |])
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
          let far = Path.polyline [| -1e308; 1e308 |] [| -1e308; 1e308 |] in
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
          let png = Hugin_next_vg_raster.png ~density:2. page in
          (* 144 dpi is 5669.29 pixels per metre. *)
          equal (option string)
            (Some (be32 5669 ^ be32 5669 ^ "\001"))
            (chunk "pHYs" png);
          equal (option string) (Some "\000") (chunk "sRGB" png));
      test "png holds the pixels of render" (fun () ->
          let png = Hugin_next_vg_raster.png ~density:2. page in
          let path = Filename.temp_file "hugin" ".png" in
          Fun.protect
            ~finally:(fun () -> Sys.remove path)
            (fun () ->
              Out_channel.with_open_bin path (fun oc -> output_string oc png);
              let rgb =
                Nx.slice
                  [ A; A; R (0, 3) ]
                  (Hugin_next_vg_raster.render ~density:2. page)
              in
              let back = Nx_io.load_image path in
              same_image (Nx.contiguous rgb) back));
      cases ~name:(Format.asprintf "%g")
        "png raises on a density beyond a PNG's resolution, unrendered"
        [ 1e6; 1e-4 ] (fun density ->
          (* A page of 1e10 by 1e10 pixels at either density, which render would
             refuse for its size. *)
          let side = 1e10 /. density in
          let page = Renderable.v side side Picture.empty in
          raises_match (Exn.invalid_arg ~substring:"resolution") (fun () ->
              Hugin_next_vg_raster.png ~density page));
      cases ~name:fst "png accepts the ends of a PNG's resolution"
        [
          ("1 pixel per metre", (0.0254 /. 72., 3000.));
          ("2^31 - 1 pixels per metre", (2147483647. *. 0.0254 /. 72., 1e-6));
        ]
        (fun (_, (density, side)) ->
          let png =
            Hugin_next_vg_raster.png ~density
              (Renderable.v side side Picture.empty)
          in
          equal string "\137PNG" (String.sub png 0 4));
    ]

(* Goldens *)

let goldens =
  group "goldens"
    (test "rendering twice gives the same bytes" (fun () ->
         List.iter
           (fun (name, r) ->
             equal ~msg:name string
               (Hugin_next_vg_raster.png ~density:2. r)
               (Hugin_next_vg_raster.png ~density:2. r))
           Vg_corpus.pages)
    :: prop "equal renderables give equal pixels" Vg_corpus.gen_picture
         (fun p ->
           same_image (render 100. 100. p)
             (render 100. 100. (Vg_corpus.respell p)))
    :: List.map
         (fun (name, r) ->
           test (name ^ " draws its golden PNG") (fun () ->
               (* The page is opaque, so the colours its PNG loads as are all
                  there is to compare. *)
               let a =
                 Nx.to_array (Hugin_next_vg_raster.render ~density:2. r)
               in
               Array.iteri
                 (fun i v -> if i mod 4 = 3 then equal ~msg:"alpha" int 255 v)
                 a;
               Vg_corpus.golden
                 ("golden/" ^ name ^ ".png")
                 (Hugin_next_vg_raster.png ~density:2. r)))
         Vg_corpus.pages)

let () =
  exit
    (run "hugin.next.vg raster"
       [
         pages;
         coverage_group;
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
