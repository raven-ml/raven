(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_next_gg
open Hugin_next_font
open Hugin_next_vg

let pp_float ppf x = Format.fprintf ppf "%.17g" x

let pp_box ppf b =
  Format.fprintf ppf "[%a, %a; %a, %a]" pp_float (Box2.minx b) pp_float
    (Box2.miny b) pp_float (Box2.maxx b) pp_float (Box2.maxy b)

let box2 = Testable.make ~pp:pp_box ~equal:Box2.equal

let near a b =
  Float.equal a b
  || Float.abs (a -. b) <= 1e-9 *. (1. +. Float.abs a +. Float.abs b)

let box2_near =
  Testable.make ~pp:pp_box ~equal:(fun a b ->
      near (Box2.minx a) (Box2.minx b)
      && near (Box2.miny a) (Box2.miny b)
      && near (Box2.maxx a) (Box2.maxx b)
      && near (Box2.maxy a) (Box2.maxy b))

let picture = Testable.make ~pp:Picture.pp ~equal:Picture.equal
let red = Color.red
let square = Path.rect (Box2.v 0. 0. 10. 10.)
let dot = Picture.fill red square
let bar = Picture.fill Color.blue (Path.rect (Box2.v 20. 0. 5. 30.))
let line = Path.polyline [| 0.; 10. |] [| 0.; 0. |]
let pixels = Nx.zeros Nx.uint8 [| 2; 3; 4 |]

let glyph_run ?(size = 10.) text =
  let font = Font.regular in
  let glyphs =
    Array.init (String.length text) (fun i ->
        Font.glyph font (Uchar.of_char text.[i]))
  in
  let xs = Array.mapi (fun i _ -> size *. Float.of_int i) glyphs in
  Run.v ~font ~size ~text ~glyphs ~xs ()

let union a b =
  match (a, b) with
  | None, x | x, None -> x
  | Some a, Some b -> Some (Box2.union a b)

let shift dx dy b =
  Box2.v (Box2.minx b +. dx) (Box2.miny b +. dy) (Box2.w b) (Box2.h b)

let contains outer inner =
  Box2.minx outer <= Box2.minx inner
  && Box2.miny outer <= Box2.miny inner
  && Box2.maxx inner <= Box2.maxx outer
  && Box2.maxy inner <= Box2.maxy outer

(* Generators *)

let coord = Gen.float_range (-100.) 100.
let unit_float = Gen.float_range 0. 1.

let gen_color =
  Gen.with_pp Color.pp
    (Gen.map
       (fun (r, g, b, alpha) -> Color.v ~alpha r g b)
       (Gen.quad unit_float unit_float unit_float unit_float))

let gen_rect =
  Gen.map
    (fun ((x, y), (w, h)) -> Path.rect (Box2.v x y w h))
    (Gen.pair (Gen.pair coord coord)
       (Gen.pair (Gen.float_range 0.5 50.) (Gen.float_range 0.5 50.)))

let gen_polyline =
  Gen.map
    (fun pts ->
      Path.polyline
        (Array.of_list (List.map fst pts))
        (Array.of_list (List.map snd pts)))
    (Gen.list ~size:(Gen.int_range 2 5) (Gen.pair coord coord))

let gen_path = Gen.one_of [ gen_rect; gen_polyline ]

let gen_stroke =
  Gen.map
    (fun (w, cap, join) -> Stroke.v ~cap ~join w)
    (Gen.triple (Gen.float_range 0.5 5.)
       (Gen.of_list [ `Butt; `Round; `Square ])
       (Gen.of_list [ `Miter; `Round; `Bevel ]))

let gen_leaf =
  Gen.one_of
    [
      Gen.map
        (fun (rule, c, q) -> Picture.fill ~rule c q)
        (Gen.triple (Gen.of_list [ `Nonzero; `Even_odd ]) gen_color gen_path);
      Gen.map
        (fun (s, c, q) -> Picture.stroke s c q)
        (Gen.triple gen_stroke gen_color gen_path);
    ]

let pp_positions ppf xs =
  let pp_sep ppf () = Format.fprintf ppf ";@ " in
  Format.fprintf ppf "@[<1>[|%a|]@]" (Format.pp_print_array ~pp_sep pp_float) xs

let gen_positions =
  Gen.with_pp pp_positions
    (Gen.map Array.of_list (Gen.list ~size:(Gen.int_range 1 4) coord))

let rec gen_picture depth =
  if depth = 0 then gen_leaf
  else
    let sub = gen_picture (depth - 1) in
    Gen.frequency
      [
        (2, gen_leaf);
        (1, Gen.map Picture.group (Gen.list ~size:(Gen.int_range 0 3) sub));
        (1, Gen.map (fun (q, p) -> Picture.clip q p) (Gen.pair gen_rect sub));
        ( 1,
          Gen.map
            (fun ((dx, dy), s, p) ->
              Picture.transform Affine.(translate dx dy * scale s s) p)
            (Gen.triple (Gen.pair coord coord) (Gen.float_range 0.5 2.) sub) );
        ( 1,
          Gen.map (fun (a, p) -> Picture.opacity a p) (Gen.pair unit_float sub)
        );
        ( 1,
          Gen.map
            (fun (xs, p) ->
              let ys = Array.map (fun x -> x /. 2.) xs in
              Picture.stamp xs ys p)
            (Gen.pair gen_positions sub) );
      ]

let gen_picture = Gen.with_pp Picture.pp (gen_picture 3)

(* Pictures of rectangles whose numbers are quarters below 2^9, stroked with
   pens that reach half their width, placed by quarters and scaled by 0.5 or 2,
   so that each sum a box takes is exact in any order: whether a box touches a
   clip then does not hang on a rounding. *)
let quarter = Gen.map (fun n -> Float.of_int n /. 4.) (Gen.int_range (-400) 400)
let quarter_size = Gen.map (fun n -> Float.of_int n /. 4.) (Gen.int_range 1 200)

let gen_quarter_rect =
  Gen.map
    (fun ((x, y), (w, h)) -> Path.rect (Box2.v x y w h))
    (Gen.pair (Gen.pair quarter quarter) (Gen.pair quarter_size quarter_size))

let gen_quarter_positions =
  Gen.with_pp pp_positions
    (Gen.map Array.of_list (Gen.list ~size:(Gen.int_range 1 4) quarter))

let gen_quarter_placement =
  Gen.with_pp Affine.pp
    (Gen.map
       (fun ((dx, dy), s) -> Affine.(translate dx dy * scale s s))
       (Gen.pair (Gen.pair quarter quarter) (Gen.of_list [ 0.5; 1.; 2. ])))

let rec gen_quarter_picture depth =
  let leaf =
    Gen.one_of
      [
        Gen.map (Picture.fill red) gen_quarter_rect;
        Gen.map
          (fun (w, q) -> Picture.stroke (Stroke.v ~join:`Round w) red q)
          (Gen.pair quarter_size gen_quarter_rect);
      ]
  in
  if depth = 0 then leaf
  else
    let sub = gen_quarter_picture (depth - 1) in
    Gen.frequency
      [
        (2, leaf);
        (1, Gen.map Picture.group (Gen.list ~size:(Gen.int_range 0 3) sub));
        ( 1,
          Gen.map
            (fun (q, p) -> Picture.clip q p)
            (Gen.pair gen_quarter_rect sub) );
        ( 1,
          Gen.map
            (fun (m, p) -> Picture.transform m p)
            (Gen.pair gen_quarter_placement sub) );
        ( 1,
          Gen.map (fun (a, p) -> Picture.opacity a p) (Gen.pair unit_float sub)
        );
        ( 1,
          Gen.map
            (fun (xs, p) -> Picture.stamp xs (Array.map Float.neg xs) p)
            (Gen.pair gen_quarter_positions sub) );
      ]

let gen_quarter_picture = Gen.with_pp Picture.pp (gen_quarter_picture 3)

(* Leaves *)

let empty_cases =
  [
    ("a fill of the empty path", fun () -> Picture.fill red Path.empty);
    ( "a stroke of the empty path",
      fun () -> Picture.stroke (Stroke.v 1.) red Path.empty );
    ("a stroke of width 0", fun () -> Picture.stroke (Stroke.v 0.) red line);
    ( "glyphs of a run without glyphs",
      fun () ->
        Picture.glyphs red (P2.v 0. 0.)
          (Run.v ~font:Font.regular ~size:10. ~text:"" ~glyphs:[||] ~xs:[||] ())
    );
    ( "glyphs of size 0",
      fun () -> Picture.glyphs red (P2.v 0. 0.) (glyph_run ~size:0. "ab") );
    ( "glyphs at nan",
      fun () -> Picture.glyphs red (P2.v Float.nan 0.) (glyph_run "ab") );
    ( "glyphs at infinity",
      fun () -> Picture.glyphs red (P2.v 0. infinity) (glyph_run "ab") );
    ( "an image without rows",
      fun () ->
        Picture.image (Box2.v 0. 0. 1. 1.) (Nx.zeros Nx.uint8 [| 0; 2; 3 |]) );
    ( "an image without columns",
      fun () ->
        Picture.image (Box2.v 0. 0. 1. 1.) (Nx.zeros Nx.uint8 [| 2; 0; 1 |]) );
    ("an image of width 0", fun () -> Picture.image (Box2.v 0. 0. 0. 1.) pixels);
    ("an image of height 0", fun () -> Picture.image (Box2.v 0. 0. 1. 0.) pixels);
    ("an empty group", fun () -> Picture.group []);
    ( "a group of empties",
      fun () -> Picture.group [ Picture.empty; Picture.empty ] );
    ("a clip of empty", fun () -> Picture.clip square Picture.empty);
    ("a clip by the empty path", fun () -> Picture.clip Path.empty dot);
    ( "a transform of empty",
      fun () -> Picture.transform (Affine.translate 1. 1.) Picture.empty );
    ( "a singular transform",
      fun () -> Picture.transform (Affine.scale 0. 1.) dot );
    ( "a transform with a nan coefficient",
      fun () -> Picture.transform { Affine.id with x0 = Float.nan } dot );
    ("an opacity of empty", fun () -> Picture.opacity 0.5 Picture.empty);
    ("a stamp of empty", fun () -> Picture.stamp [| 0. |] [| 0. |] Picture.empty);
    ("a stamp without positions", fun () -> Picture.stamp [||] [||] dot);
    ( "a tag of empty",
      fun () ->
        Picture.tag { id = Nx.Ptree.Path.root; rows = Rows [||] } Picture.empty
    );
  ]

let image_shapes =
  [
    ("rank 2", [| 2; 3 |]);
    ("two channels", [| 2; 3; 2 |]);
    ("five channels", [| 2; 3; 5 |]);
    ("rank 4", [| 1; 2; 3; 4 |]);
  ]

let leaves =
  group "leaves"
    [
      cases ~name:fst "is empty:" empty_cases (fun (_, p) ->
          equal picture Picture.empty (p ()));
      cases ~name:fst "image raises on a shape of" image_shapes (fun (_, s) ->
          raises_match (Exn.invalid_arg ~substring:"Picture.image") (fun () ->
              Picture.image (Box2.v 0. 0. 1. 1.) (Nx.zeros Nx.uint8 s)));
      test "image keeps its tensor, physically" (fun () ->
          match Picture.image (Box2.v 0. 0. 3. 2.) pixels with
          | Image { pixels = px; _ } -> is_true (px == pixels)
          | p -> failf "not an image: %a" Picture.pp p);
      test "a transparent fill is a fill, with bounds" (fun () ->
          let p = Picture.fill Color.transparent square in
          is_false (Picture.equal p Picture.empty);
          equal (option box2) (Some (Box2.v 0. 0. 10. 10.)) (Picture.bounds p));
    ]

(* Composing *)

let stamp_lengths =
  [
    ("ys", fun () -> Picture.stamp [| 0.; 1. |] [| 0. |] dot);
    ( "fills",
      fun () -> Picture.stamp ~fills:[| red |] [| 0.; 1. |] [| 0.; 1. |] dot );
    ("strokes", fun () -> Picture.stamp ~strokes:[||] [| 0. |] [| 0. |] dot);
    ( "scales",
      fun () -> Picture.stamp ~scales:[| 1.; 1. |] [| 0. |] [| 0. |] dot );
  ]

let stamp_copies () =
  let xs = [| 1.; 2. |] and ys = [| 3.; 4. |] in
  let fills = [| red; Color.blue |] and scales = [| 1.; 2. |] in
  let p = Picture.stamp ~fills ~scales xs ys dot in
  let expected =
    Picture.stamp ~fills:[| red; Color.blue |] ~scales:[| 1.; 2. |] [| 1.; 2. |]
      [| 3.; 4. |] dot
  in
  xs.(0) <- 9.;
  ys.(0) <- 9.;
  fills.(0) <- Color.green;
  scales.(0) <- 9.;
  equal picture expected p

let composing =
  group "composing"
    [
      test "group drops empties and unwraps a single picture" (fun () ->
          equal picture dot
            (Picture.group [ Picture.empty; dot; Picture.empty ]);
          equal picture dot (Picture.group [ dot ]));
      test "group keeps its nesting" (fun () ->
          is_false
            (Picture.equal
               (Picture.group [ Picture.group [ dot; bar ]; dot ])
               (Picture.group [ dot; bar; dot ])));
      test "opacity 1. is the identity" (fun () ->
          equal picture dot (Picture.opacity 1. dot));
      test "opacity 0. keeps the picture's bounds and tags" (fun () ->
          let tagged =
            Picture.tag
              { id = Nx.Ptree.Path.v [ Field "a" ]; rows = Rows [| 3 |] }
              dot
          in
          let p = Picture.opacity 0. tagged in
          is_false (Picture.equal Picture.empty p);
          equal (option box2) (Picture.bounds tagged) (Picture.bounds p);
          match p with
          | Opacity { opacity = 0.; picture = q } -> equal picture tagged q
          | p -> failf "not an opacity: %a" Picture.pp p);
      cases ~name:(Format.asprintf "%g") "opacity raises on"
        [ -0.1; 1.1; Float.nan; infinity ] (fun a ->
          raises_match (Exn.invalid_arg ~substring:"Picture.opacity") (fun () ->
              Picture.opacity a dot));
      cases ~name:fst "stamp raises when the positions and its" stamp_lengths
        (fun (_, f) ->
          raises_match (Exn.invalid_arg ~substring:"Picture.stamp") f);
      test "stamp raises on a finite negative scale" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"negative scale") (fun () ->
              Picture.stamp ~scales:[| -1. |] [| 0. |] [| 0. |] dot));
      test "stamp accepts the scales it skips: nan and infinities" (fun () ->
          let p =
            Picture.stamp
              ~scales:[| neg_infinity; Float.nan; infinity; 1. |]
              [| 0.; 0.; 0.; 50. |] [| 0.; 0.; 0.; 0. |] dot
          in
          equal (option box2) (Some (Box2.v 50. 0. 10. 10.)) (Picture.bounds p));
      test "stamp copies its arrays" stamp_copies;
    ]

(* Tags *)

let tag_of rows =
  { Picture.id = Nx.Ptree.Path.v [ Index 0; Field "dots" ]; rows }

let tags =
  group "tags"
    [
      test "tag raises when the rows of a stamp differ from its positions"
        (fun () ->
          raises_match (Exn.invalid_arg ~substring:"Picture.tag") (fun () ->
              Picture.tag (tag_of (Rows [| 1 |]))
                (Picture.stamp [| 0.; 1. |] [| 0.; 1. |] dot)));
      test "a tag on anything else takes rows of any length" (fun () ->
          let p = Picture.tag (tag_of (Rows [| 1; 2; 3 |])) dot in
          equal (option box2) (Picture.bounds dot) (Picture.bounds p));
      test "a tag over a transform of a stamp takes rows of any length"
        (fun () ->
          let s = Picture.stamp [| 0.; 1. |] [| 0.; 1. |] dot in
          let p = Picture.transform (Affine.translate 1. 0.) s in
          is_false
            (Picture.equal Picture.empty
               (Picture.tag (tag_of (Rows [| 7 |])) p)));
      cases
        ~name:(fun (w, h) -> Printf.sprintf "%d by %d" w h)
        "tag raises on a grid of"
        [ (0, 1); (1, 0); (-1, 3) ]
        (fun (width, height) ->
          raises_match (Exn.invalid_arg ~substring:"Picture.tag") (fun () ->
              Picture.tag
                (tag_of (Cells { box = Box2.v 0. 0. 1. 1.; width; height }))
                dot));
      test "tag copies its rows" (fun () ->
          let a = [| 4; 5 |] in
          let p = Picture.tag (tag_of (Rows a)) dot in
          a.(0) <- 0;
          equal picture (Picture.tag (tag_of (Rows [| 4; 5 |])) dot) p);
      test "tags of different ids or rows differ" (fun () ->
          let t rows id = Picture.tag { id = Nx.Ptree.Path.v id; rows } dot in
          is_false
            (Picture.equal
               (t (Rows [| 1 |]) [ Field "a" ])
               (t (Rows [| 1 |]) [ Field "b" ]));
          is_false
            (Picture.equal
               (t (Rows [| 1 |]) [ Field "a" ])
               (t (Rows [| 2 |]) [ Field "a" ]));
          is_false
            (Picture.equal
               (t
                  (Cells { box = Box2.v 0. 0. 1. 1.; width = 1; height = 1 })
                  [])
               (t
                  (Cells { box = Box2.v 0. 0. 1. 2.; width = 1; height = 1 })
                  [])));
    ]

(* Bounds *)

let pen_reaches =
  let r2 = Float.sqrt 2. in
  [
    ("round caps and joins reach half the width", Stroke.v 2., 1.);
    ("butt caps and bevel joins too", Stroke.v ~cap:`Butt ~join:`Bevel 2., 1.);
    ("square caps reach sqrt 2 times further", Stroke.v ~cap:`Square 2., r2);
    ( "miter joins reach the miter limit times further",
      Stroke.v ~join:`Miter ~miter_limit:3. 2.,
      3. );
    ( "square caps and miter joins reach the further of both",
      Stroke.v ~cap:`Square ~join:`Miter ~miter_limit:1. 2.,
      r2 );
  ]

let glyph_bounds () =
  let r = glyph_run "Hi" in
  let ink = Option.get (Run.bounds r) in
  equal (option box2_near)
    (Some (shift 5. 7. ink))
    (Picture.bounds (Picture.glyphs red (P2.v 5. 7.) r))

let scaled_stamp_keeps_pens () =
  let p = Picture.stroke (Stroke.v 2.) red line in
  equal (option box2)
    (Some (Box2.v (-1.) (-1.) 22. 2.))
    (Picture.bounds (Picture.stamp ~scales:[| 2. |] [| 0. |] [| 0. |] p))

let stamp_is_its_instances (m, (xs, p)) =
  let ys = Array.map (fun x -> 1. -. x) xs in
  let instances =
    Array.to_list
      (Array.mapi
         (fun i x ->
           Picture.bounds
             (Picture.transform Affine.(m * translate x ys.(i)) p))
         xs)
  in
  equal (option box2)
    (List.fold_left union None instances)
    (Picture.bounds (Picture.transform m (Picture.stamp xs ys p)))

let clipped_stamp_is_its_instances (q, (xs, p)) =
  let ys = Array.map (fun x -> 1. -. x) xs in
  let instances =
    Array.to_list
      (Array.mapi
         (fun i x ->
           Picture.bounds
             (Picture.clip q (Picture.transform (Affine.translate x ys.(i)) p)))
         xs)
  in
  equal (option box2)
    (List.fold_left union None instances)
    (Picture.bounds (Picture.clip q (Picture.stamp xs ys p)))

let bounds =
  group "bounds"
    [
      test "a fill is bounded by its path" (fun () ->
          equal (option box2) (Some (Box2.v 0. 0. 10. 10.)) (Picture.bounds dot));
      cases
        ~name:(fun (n, _, _) -> n)
        "a stroke's path is grown by its pen:" pen_reaches
        (fun (_, s, r) ->
          equal (option box2_near)
            (Some (Box2.v (-.r) (-.r) (10. +. (2. *. r)) (2. *. r)))
            (Picture.bounds (Picture.stroke s red line)));
      test "glyphs are bounded by their ink at their origin" glyph_bounds;
      test "an image is bounded by its box" (fun () ->
          equal (option box2)
            (Some (Box2.v 1. 2. 3. 4.))
            (Picture.bounds (Picture.image (Box2.v 1. 2. 3. 4.) pixels)));
      test "an empty picture has no bounds" (fun () ->
          is_none ~pp:pp_box (Picture.bounds Picture.empty));
      test "a path of gaps has no bounds" (fun () ->
          let gaps = Path.polyline [| Float.nan; Float.nan |] [| 0.; 1. |] in
          is_none ~pp:pp_box (Picture.bounds (Picture.fill red gaps)));
      prop "the bounds of a group are the union of its pictures'"
        (Gen.pair gen_picture gen_picture) (fun (a, b) ->
          equal (option box2)
            (union (Picture.bounds a) (Picture.bounds b))
            (Picture.bounds (Picture.group [ a; b ])));
      prop "a translation moves the bounds"
        (Gen.triple quarter quarter gen_quarter_picture) (fun (dx, dy, p) ->
          equal (option box2)
            (Option.map (shift dx dy) (Picture.bounds p))
            (Picture.bounds (Picture.transform (Affine.translate dx dy) p)));
      test "a rotation bounds the rotated box" (fun () ->
          let r = Float.sqrt 2. *. 5. in
          equal (option box2_near)
            (Some (Box2.v (-.r) 0. (2. *. r) (2. *. r)))
            (Picture.bounds
               (Picture.transform (Affine.rotate (Float.pi /. 4.)) dot)));
      prop "a clip cuts the bounds of its picture to within its path's"
        (Gen.pair gen_rect gen_picture) (fun (q, p) ->
          match Picture.bounds (Picture.clip q p) with
          | None -> ()
          | Some b ->
              let within = function Some o -> contains o b | None -> false in
              is_true ~msg:"within the picture's bounds"
                (within (Picture.bounds p));
              is_true ~msg:"within the path's" (within (Path.bounds q)));
      test "a clip of a single leaf is the meet of their boxes" (fun () ->
          let q = Path.rect (Box2.v 5. (-5.) 20. 10.) in
          equal (option box2)
            (Some (Box2.v 5. 0. 5. 5.))
            (Picture.bounds (Picture.clip q dot)));
      test "a clip touching its picture along an edge bounds the edge"
        (fun () ->
          let q = Path.rect (Box2.v 10. 0. 5. 5.) in
          equal (option box2)
            (Some (Box2.v 10. 0. 0. 5.))
            (Picture.bounds (Picture.clip q dot)));
      test "a clip touching its picture along a horizontal edge bounds it"
        (fun () ->
          let q = Path.rect (Box2.v 0. 10. 5. 5.) in
          equal (option box2)
            (Some (Box2.v 0. 10. 5. 0.))
            (Picture.bounds (Picture.clip q dot)));
      test "a picture of no area has a flat box" (fun () ->
          let p = Picture.fill red (Path.polyline [| 5.; 5. |] [| 0.; 10. |]) in
          equal (option box2) (Some (Box2.v 5. 0. 0. 10.)) (Picture.bounds p));
      test "a clip apart from its picture has no extent" (fun () ->
          let q = Path.rect (Box2.v 50. 50. 1. 1.) in
          is_none ~pp:pp_box (Picture.bounds (Picture.clip q dot)));
      test "a clip cuts each leaf, not their union" (fun () ->
          let q = Path.rect (Box2.v 0. 0. 30. 15.) in
          let far =
            Picture.fill Color.blue (Path.rect (Box2.v 20. 20. 10. 10.))
          in
          equal (option box2)
            (Some (Box2.v 0. 0. 10. 10.))
            (Picture.bounds (Picture.clip q (Picture.group [ dot; far ]))));
      prop "a stamp is bounded by its instances"
        (Gen.pair gen_quarter_placement
           (Gen.pair gen_quarter_positions gen_quarter_picture))
        stamp_is_its_instances;
      prop "a stamp under a clip is bounded by its instances under the clip"
        (Gen.pair gen_quarter_rect
           (Gen.pair gen_quarter_positions gen_quarter_picture))
        clipped_stamp_is_its_instances;
      test "a clip cuts each leaf of a stamp's instances" (fun () ->
          let two =
            Picture.group
              [ dot; Picture.fill red (Path.rect (Box2.v 20. 20. 10. 10.)) ]
          in
          let q = Path.rect (Box2.v 15. 15. 30. 30.) in
          let clipped p = Picture.bounds (Picture.clip q p) in
          equal (option box2)
            (Some (Box2.v 20. 20. 10. 10.))
            (clipped (Picture.stamp [| 0. |] [| 0. |] two));
          equal (option box2)
            (clipped (Picture.stamp ~scales:[| 1. |] [| 0. |] [| 0. |] two))
            (clipped (Picture.stamp [| 0. |] [| 0. |] two)));
      test "a clip cuts each instance of a stamp in a stamp" (fun () ->
          let q = Path.rect (Box2.v 12. 12. 6. 6.) in
          let inner = Picture.stamp [| 0.; 20. |] [| 20.; 0. |] dot in
          is_none ~pp:pp_box
            (Picture.bounds
               (Picture.clip q (Picture.stamp [| 0. |] [| 0. |] inner))));
      test "a stamp skips instances at non-finite positions" (fun () ->
          let p =
            Picture.stamp [| Float.nan; 50.; infinity |] [| 0.; 0.; 0. |] dot
          in
          equal (option box2) (Some (Box2.v 50. 0. 10. 10.)) (Picture.bounds p));
      test "a scaled stamp scales its geometry and keeps its pens"
        scaled_stamp_keeps_pens;
      test "an instance of scale 0. has no extent" (fun () ->
          is_none ~pp:pp_box
            (Picture.bounds
               (Picture.stamp ~scales:[| 0. |] [| 0. |] [| 0. |] dot)));
      test "bounds raises when a corner is not finite" (fun () ->
          let far =
            Picture.stroke (Stroke.v 1.) red
              (Path.polyline [| 0.; 1e308 |] [| 0.; 0. |])
          in
          raises_match (Exn.invalid_arg ~substring:"not finite") (fun () ->
              Picture.bounds (Picture.transform (Affine.scale 10. 10.) far)));
    ]

(* Comparing and formatting *)

let font_copy =
  match Font.of_string (Font.bytes Font.regular) with
  | Ok f -> f
  | Error e -> Format.kasprintf failwith "%a" Font.pp_error e

let comparing =
  group "comparing"
    [
      prop "equal is an equivalence"
        (Gen.pair gen_picture gen_picture)
        (Law.equivalence ~respell:Vg_corpus.respell picture);
      test "images compare tensors physically" (fun () ->
          let b = Box2.v 0. 0. 1. 1. in
          let a = Nx.zeros Nx.uint8 [| 1; 1; 3 |] in
          equal picture (Picture.image b a) (Picture.image b a);
          is_false
            (Picture.equal (Picture.image b a)
               (Picture.image b (Nx.zeros Nx.uint8 [| 1; 1; 3 |]))));
      test "glyph runs compare fonts by their bytes" (fun () ->
          let glyphs = [| Font.glyph Font.regular (Uchar.of_char 'a') |] in
          let r f = Run.v ~font:f ~size:9. ~text:"a" ~glyphs ~xs:[| 0. |] () in
          equal picture
            (Picture.glyphs red (P2.v 0. 0.) (r Font.regular))
            (Picture.glyphs red (P2.v 0. 0.) (r font_copy)));
      test "every field counts" (fun () ->
          let s ?fills ?strokes ?scales xs =
            Picture.stamp ?fills ?strokes ?scales xs [| 0. |] dot
          in
          let differ a b =
            is_false
              ~msg:(Format.asprintf "%a" Picture.pp a)
              (Picture.equal a b)
          in
          differ (Picture.fill ~rule:`Even_odd red square) dot;
          differ (Picture.fill Color.blue square) dot;
          differ (Picture.stroke (Stroke.v 1.) red square) dot;
          differ (s [| 0. |]) (s [| 1. |]);
          differ (s ~fills:[| red |] [| 0. |]) (s [| 0. |]);
          differ (s ~strokes:[| red |] [| 0. |]) (s [| 0. |]);
          differ (s ~scales:[| 1. |] [| 0. |]) (s [| 0. |]);
          differ (Picture.opacity 0.5 dot) (Picture.opacity 0.25 dot);
          differ
            (Picture.clip ~rule:`Even_odd square dot)
            (Picture.clip square dot);
          differ
            (Picture.transform (Affine.translate 1. 0.) dot)
            (Picture.transform (Affine.translate 0. 1.) dot);
          let g text = Picture.glyphs red (P2.v 0. 0.) (glyph_run text) in
          differ (g "ab") (g "ac");
          differ (Picture.glyphs red (P2.v 1. 0.) (glyph_run "ab")) (g "ab");
          differ
            (Picture.image (Box2.v 0. 0. 3. 2.) pixels)
            (Picture.image (Box2.v 0. 0. 3. 4.) pixels);
          differ (Picture.clip square dot) (Picture.clip square bar);
          differ (Picture.clip line dot) (Picture.clip square dot);
          differ (s [| 0. |]) (Picture.stamp [| 0. |] [| 0. |] bar));
      test "pp formats each case" (fun () ->
          let p =
            Picture.group
              [
                Picture.clip
                  (Path.rect (Box2.v 0. 0. 4. 4.))
                  (Picture.transform (Affine.translate 1. 2.)
                     (Picture.opacity 0.5
                        (Picture.stroke (Stroke.v 1.) red line)));
                Picture.tag
                  {
                    id = Nx.Ptree.Path.v [ Field "dots" ];
                    rows = Rows [| 7; 8 |];
                  }
                  (Picture.stamp ~fills:[| red; Color.blue |] [| 1.; 2. |]
                     [| 3.; 4. |]
                     (Picture.fill ~rule:`Even_odd red square));
                Picture.image (Box2.v 0. 0. 3. 2.) pixels;
              ]
          in
          expect (Format.asprintf "%a" Picture.pp p)
          @@ __POS_OF__
               {|
            (group
             (clip nonzero "M 0 0 L 4 0 L 4 4 L 0 4 Z"
              (transform (1 0 0 1 1 2)
               (opacity 0.5
                (stroke (stroke 1 cap round join round miter 4) #ff0000 "M 0 0 L 10 0"))))
             (tag dots (rows (7 8))
              (stamp (xs (1 2)) (ys (3 4)) (fills (#ff0000 #0000ff))
               (fill even-odd #ff0000 "M 0 0 L 10 0 L 10 10 L 0 10 Z")))
             (image [(0, 0) (3, 2)] (2 3 4)))
            |});
    ]

(* Renderables *)

let renderables =
  group "renderables"
    [
      cases
        ~name:(fun (w, h) -> Printf.sprintf "%g by %g" w h)
        "v raises on a page of"
        [ (0., 1.); (1., -1.); (Float.nan, 1.); (1., infinity) ]
        (fun (w, h) ->
          raises_match (Exn.invalid_arg ~substring:"Renderable.v") (fun () ->
              Renderable.v w h dot));
      test "v keeps its page and picture" (fun () ->
          let r = Renderable.v 360. 240. dot in
          equal float_exact 360. (Renderable.w r);
          equal float_exact 240. (Renderable.h r);
          equal picture dot (Renderable.picture r));
      test "equal compares sizes and pictures" (fun () ->
          let r = Renderable.v 1. 2. dot in
          is_true (Renderable.equal r (Renderable.v 1. 2. dot));
          is_false (Renderable.equal r (Renderable.v 2. 2. dot));
          is_false (Renderable.equal r (Renderable.v 1. 2. bar)));
    ]

let () =
  exit
    (run "hugin.next.vg picture"
       [ leaves; composing; tags; bounds; comparing; renderables ])
