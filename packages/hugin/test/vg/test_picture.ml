(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_gg
open Hugin_font
open Hugin_vg

let invalid substring f = raises_match (Exn.invalid_arg ~substring) f
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
let bounds p = Picture.bounds p
let red = Color.red
let rect x y w h = Path.rect (Box2.v x y w h)
let square = rect 0. 0. 10. 10.
let dot = Picture.fill red square
let bar = Picture.fill Color.blue (rect 20. 0. 5. 30.)
let line = Path.polyline [| 0.; 10. |] [| 0.; 0. |]
let pixels = Nx.zeros Nx.uint8 [| 2; 3; 4 |]
let typeset text = Vg_corpus.typeset text 10.
let tag_of rows = { Picture.id = Nx.Ptree.Path.v [ Field "dots" ]; rows }

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

(* Pictures of rectangles whose numbers are quarters below 2^9, stroked with
   pens that reach half their width, placed by quarters and scaled by 0.5 or 2,
   so that each sum a box takes is exact in any order: whether a box touches a
   clip then does not hang on a rounding. *)

let quarter = Gen.map (fun n -> Float.of_int n /. 4.) (Gen.int_range (-400) 400)
let quarter_size = Gen.map (fun n -> Float.of_int n /. 4.) (Gen.int_range 1 200)

let gen_rect =
  Gen.map
    (fun ((x, y), (w, h)) -> rect x y w h)
    (Gen.pair (Gen.pair quarter quarter) (Gen.pair quarter_size quarter_size))

let pp_positions ppf xs =
  let pp_sep ppf () = Format.fprintf ppf ";@ " in
  Format.fprintf ppf "@[<1>[|%a|]@]" (Format.pp_print_array ~pp_sep pp_float) xs

let gen_positions =
  Gen.with_pp pp_positions
    (Gen.map Array.of_list (Gen.list ~size:(Gen.int_range 1 4) quarter))

let gen_placement =
  Gen.with_pp Affine.pp
    (Gen.map
       (fun ((dx, dy), s) -> Affine.(translate dx dy * scale s s))
       (Gen.pair (Gen.pair quarter quarter) (Gen.of_list [ 0.5; 1.; 2. ])))

let rec gen_picture depth =
  let leaf =
    Gen.one_of
      [
        Gen.map (Picture.fill red) gen_rect;
        Gen.map
          (fun (w, q) -> Picture.stroke (Stroke.v ~join:`Round w) red q)
          (Gen.pair quarter_size gen_rect);
      ]
  in
  if depth = 0 then leaf
  else
    let sub = gen_picture (depth - 1) in
    Gen.frequency
      [
        (2, leaf);
        (1, Gen.map Picture.group (Gen.list ~size:(Gen.int_range 0 3) sub));
        (1, Gen.map (fun (q, p) -> Picture.clip q p) (Gen.pair gen_rect sub));
        ( 1,
          Gen.map
            (fun (m, p) -> Picture.transform m p)
            (Gen.pair gen_placement sub) );
        ( 1,
          Gen.map
            (fun (a, p) -> Picture.opacity a p)
            (Gen.pair (Gen.float_range 0. 1.) sub) );
        ( 1,
          Gen.map
            (fun (xs, p) -> Picture.stamp xs (Array.map Float.neg xs) p)
            (Gen.pair gen_positions sub) );
      ]

let gen_picture = Gen.with_pp Picture.pp (gen_picture 3)

(* Constructors *)

let empties =
  [
    ("a fill of the empty path", fun () -> Picture.fill red Path.empty);
    ( "a stroke of the empty path",
      fun () -> Picture.stroke (Stroke.v 1.) red Path.empty );
    ("a stroke of width 0", fun () -> Picture.stroke (Stroke.v 0.) red line);
    ( "glyphs of a run without glyphs",
      fun () -> Picture.glyphs red (P2.v 0. 0.) (typeset "") );
    ( "glyphs of size 0",
      fun () -> Picture.glyphs red (P2.v 0. 0.) (Vg_corpus.typeset "ab" 0.) );
    ( "glyphs at nan",
      fun () -> Picture.glyphs red (P2.v Float.nan 0.) (typeset "ab") );
    ( "glyphs at infinity",
      fun () -> Picture.glyphs red (P2.v 0. infinity) (typeset "ab") );
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
    ("a tag of empty", fun () -> Picture.tag (tag_of (Rows [||])) Picture.empty);
  ]

let raises =
  let image shape () =
    Picture.image (Box2.v 0. 0. 1. 1.) (Nx.zeros Nx.uint8 shape)
  in
  let stamp ?fills ?strokes ?scales xs ys () =
    Picture.stamp ?fills ?strokes ?scales xs ys dot
  in
  let grid width height () =
    Picture.tag (tag_of (Cells { box = Box2.v 0. 0. 1. 1.; width; height })) dot
  in
  [
    ("image of rank 2", "Picture.image", image [| 2; 3 |]);
    ("image of two channels", "Picture.image", image [| 2; 3; 2 |]);
    ("image of five channels", "Picture.image", image [| 2; 3; 5 |]);
    ("image of rank 4", "Picture.image", image [| 1; 2; 3; 4 |]);
    ("opacity -0.1", "Picture.opacity", fun () -> Picture.opacity (-0.1) dot);
    ("opacity 1.1", "Picture.opacity", fun () -> Picture.opacity 1.1 dot);
    ("opacity nan", "Picture.opacity", fun () -> Picture.opacity Float.nan dot);
    ("stamp with fewer ys", "Picture.stamp", stamp [| 0.; 1. |] [| 0. |]);
    ( "stamp with fewer fills",
      "Picture.stamp",
      stamp ~fills:[| red |] [| 0.; 1. |] [| 0.; 1. |] );
    ( "stamp with no strokes",
      "Picture.stamp",
      stamp ~strokes:[||] [| 0. |] [| 0. |] );
    ( "stamp with more scales",
      "Picture.stamp",
      stamp ~scales:[| 1.; 1. |] [| 0. |] [| 0. |] );
    ( "stamp with a negative scale",
      "negative scale",
      stamp ~scales:[| -1. |] [| 0. |] [| 0. |] );
    ( "tag of a stamp with fewer rows",
      "Picture.tag",
      fun () ->
        Picture.tag (tag_of (Rows [| 1 |]))
          (Picture.stamp [| 0.; 1. |] [| 0.; 1. |] dot) );
    ("tag of a grid 0 wide", "Picture.tag", grid 0 1);
    ("tag of a grid 0 high", "Picture.tag", grid 1 0);
    ("tag of a grid -1 wide", "Picture.tag", grid (-1) 3);
  ]

let copies () =
  let xs = [| 1.; 2. |] and ys = [| 3.; 4. |] and rows = [| 4; 5 |] in
  let fills = [| red; Color.blue |] and scales = [| 1.; 2. |] in
  let p =
    Picture.tag (tag_of (Rows rows)) (Picture.stamp ~fills ~scales xs ys dot)
  in
  let expected =
    Picture.tag
      (tag_of (Rows [| 4; 5 |]))
      (Picture.stamp ~fills:[| red; Color.blue |] ~scales:[| 1.; 2. |]
         [| 1.; 2. |] [| 3.; 4. |] dot)
  in
  List.iter (fun a -> a.(0) <- 9.) [ xs; ys; scales ];
  fills.(0) <- Color.green;
  rows.(0) <- 0;
  equal picture expected p

let opacity_zero () =
  let tagged = Picture.tag (tag_of (Rows [| 3 |])) dot in
  let p = Picture.opacity 0. tagged in
  equal (option box2) (bounds tagged) (bounds p);
  match p with
  | Opacity { opacity = 0.; picture = q } -> equal picture tagged q
  | p -> failf "not an opacity: %a" Picture.pp p

let constructors =
  group "constructors"
    [
      cases ~name:fst "are empty:" empties (fun (_, p) ->
          equal picture Picture.empty (p ()));
      cases
        ~name:(fun (n, _, _) -> n)
        "raise Invalid_argument on" raises
        (fun (_, substring, f) -> invalid substring f);
      test "image keeps its tensor, physically" (fun () ->
          match Picture.image (Box2.v 0. 0. 3. 2.) pixels with
          | Image { pixels = px; _ } ->
              equal ~msg:"physically" bool true (px == pixels)
          | p -> failf "not an image: %a" Picture.pp p);
      test "a transparent fill is not empty" (fun () ->
          not_equal picture Picture.empty
            (Picture.fill Color.transparent square));
      test "group drops empties and unwraps a single picture" (fun () ->
          equal picture dot
            (Picture.group [ Picture.empty; dot; Picture.empty ]);
          equal picture dot (Picture.group [ dot ]));
      test "group keeps its nesting" (fun () ->
          not_equal picture
            (Picture.group [ Picture.group [ dot; bar ]; dot ])
            (Picture.group [ dot; bar; dot ]));
      test "opacity 1. is the identity" (fun () ->
          equal picture dot (Picture.opacity 1. dot));
      test "opacity 0. keeps the picture, its bounds and its tags" opacity_zero;
      test "a stamp's scales may be nan or infinite" (fun () ->
          let p =
            Picture.stamp
              ~scales:[| neg_infinity; Float.nan; infinity; 1. |]
              [| 0.; 0.; 0.; 50. |] [| 0.; 0.; 0.; 0. |] dot
          in
          equal (option box2) (Some (Box2.v 50. 0. 10. 10.)) (bounds p));
      test "a tag of anything but a stamp takes rows of any length" (fun () ->
          let s = Picture.stamp [| 0.; 1. |] [| 0.; 1. |] dot in
          let moved = Picture.transform (Affine.translate 1. 0.) s in
          not_equal picture Picture.empty
            (Picture.tag (tag_of (Rows [| 7 |])) moved);
          not_equal picture Picture.empty
            (Picture.tag (tag_of (Rows [| 1; 2; 3 |])) dot));
      test "stamp and tag copy their arrays" copies;
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

(* [instances wrap (xs, p)] is the union of the bounds of the instances of
   [stamp xs ys p], each wrapped by [wrap], where [ys] is [1 - xs]. *)
let instances wrap (xs, p) =
  let ys = Array.map (fun x -> 1. -. x) xs in
  let one i x =
    bounds (wrap (Picture.transform (Affine.translate x ys.(i)) p))
  in
  ( Array.fold_left union None (Array.mapi one xs),
    bounds (wrap (Picture.stamp xs ys p)) )

let stamp_examples =
  let two = Picture.group [ dot; Picture.fill red (rect 20. 20. 10. 10.) ] in
  let nested = Picture.stamp [| 0.; 20. |] [| 20.; 0. |] dot in
  [
    (rect 15. 15. 30. 30., ([| 0. |], two));
    (rect 12. 13. 6. 6., ([| 0. |], nested));
  ]

let bounds_group =
  group "bounds"
    [
      test "a fill is bounded by its path" (fun () ->
          equal (option box2) (Some (Box2.v 0. 0. 10. 10.)) (bounds dot));
      test "a transparent fill has bounds" (fun () ->
          equal (option box2) (bounds dot)
            (bounds (Picture.fill Color.transparent square)));
      cases
        ~name:(fun (n, _, _) -> n)
        "a stroke's path is grown by its pen:" pen_reaches
        (fun (_, s, r) ->
          equal (option box2_near)
            (Some (Box2.v (-.r) (-.r) (10. +. (2. *. r)) (2. *. r)))
            (bounds (Picture.stroke s red line)));
      test "glyphs are bounded by their ink at their origin" (fun () ->
          let ink = Option.get (Run.bounds (typeset "Hi")) in
          equal (option box2_near)
            (Some (shift 5. 7. ink))
            (bounds (Picture.glyphs red (P2.v 5. 7.) (typeset "Hi"))));
      test "an image is bounded by its box" (fun () ->
          equal (option box2)
            (Some (Box2.v 1. 2. 3. 4.))
            (bounds (Picture.image (Box2.v 1. 2. 3. 4.) pixels)));
      test "a path of no area has a flat box" (fun () ->
          let p = Picture.fill red (Path.polyline [| 5.; 5. |] [| 0.; 10. |]) in
          equal (option box2) (Some (Box2.v 5. 0. 0. 10.)) (bounds p));
      cases ~name:fst "has no extent:"
        [
          ("empty", Picture.empty);
          ( "a path of gaps",
            Picture.fill red
              (Path.polyline [| Float.nan; Float.nan |] [| 0.; 1. |]) );
          ( "a clip apart from its picture",
            Picture.clip (rect 50. 50. 1. 1.) dot );
          ( "an instance of scale 0.",
            Picture.stamp ~scales:[| 0. |] [| 0. |] [| 0. |] dot );
        ]
        (fun (_, p) -> is_none ~pp:pp_box (bounds p));
      prop "a group is bounded by the union of its pictures'"
        (Gen.pair Vg_corpus.gen_picture Vg_corpus.gen_picture) (fun (a, b) ->
          equal (option box2)
            (union (bounds a) (bounds b))
            (bounds (Picture.group [ a; b ])));
      prop "a translation moves the bounds"
        (Gen.triple quarter quarter gen_picture) (fun (dx, dy, p) ->
          equal (option box2)
            (Option.map (shift dx dy) (bounds p))
            (bounds (Picture.transform (Affine.translate dx dy) p)));
      test "a rotation bounds the rotated box" (fun () ->
          let r = Float.sqrt 2. *. 5. in
          equal (option box2_near)
            (Some (Box2.v (-.r) 0. (2. *. r) (2. *. r)))
            (bounds (Picture.transform (Affine.rotate (Float.pi /. 4.)) dot)));
      prop "a clip is bounded within its picture's bounds and its path's"
        (Gen.pair
           (Gen.with_pp Path.pp Vg_corpus.gen_path)
           Vg_corpus.gen_picture)
        (fun (q, p) ->
          match bounds (Picture.clip q p) with
          | None -> ()
          | Some b ->
              let within = function Some o -> contains o b | None -> false in
              satisfies ~claim:"within the picture's bounds" box2
                (fun _ -> within (bounds p))
                b;
              satisfies ~claim:"within the path's bounds" box2
                (fun _ -> within (Path.bounds q))
                b);
      cases ~name:fst "a clip of a leaf bounds the meet of their boxes:"
        [
          ("overlapping", (rect 5. (-5.) 20. 10., Box2.v 5. 0. 5. 5.));
          ( "touching along a vertical edge",
            (rect 10. 0. 5. 5., Box2.v 10. 0. 0. 5.) );
          ( "touching along a horizontal edge",
            (rect 0. 10. 5. 5., Box2.v 0. 10. 5. 0.) );
        ]
        (fun (_, (q, b)) ->
          equal (option box2) (Some b) (bounds (Picture.clip q dot)));
      test "a clip cuts each leaf, not their union" (fun () ->
          let far = Picture.fill Color.blue (rect 20. 20. 10. 10.) in
          equal (option box2)
            (Some (Box2.v 0. 0. 10. 10.))
            (bounds
               (Picture.clip (rect 0. 0. 30. 15.) (Picture.group [ dot; far ]))));
      prop "a stamp is bounded by its instances"
        (Gen.pair gen_placement (Gen.pair gen_positions gen_picture))
        (fun (m, s) ->
          let expected, got = instances (Picture.transform m) s in
          equal (option box2) expected got);
      prop "a stamp under a clip is bounded by its instances under the clip"
        ~examples:stamp_examples
        (Gen.pair gen_rect (Gen.pair gen_positions gen_picture))
        (fun (q, s) ->
          let expected, got = instances (Picture.clip q) s in
          equal (option box2) expected got);
      test "a stamp skips instances at non-finite positions" (fun () ->
          let p =
            Picture.stamp [| Float.nan; 50.; infinity |] [| 0.; 0.; 0. |] dot
          in
          equal (option box2) (Some (Box2.v 50. 0. 10. 10.)) (bounds p));
      test "a scaled stamp scales its geometry and keeps its pens" (fun () ->
          let p = Picture.stroke (Stroke.v 2.) red line in
          equal (option box2)
            (Some (Box2.v (-1.) (-1.) 22. 2.))
            (bounds (Picture.stamp ~scales:[| 2. |] [| 0. |] [| 0. |] p)));
      test "bounds raises when a corner is not finite" (fun () ->
          let far =
            Picture.stroke (Stroke.v 1.) red
              (Path.polyline [| 0.; 1e308 |] [| 0.; 0. |])
          in
          invalid "not finite" (fun () ->
              bounds (Picture.transform (Affine.scale 10. 10.) far)));
    ]

(* Comparing and formatting *)

let font_copy =
  match Font.of_string (Font.bytes Font.regular) with
  | Ok f -> f
  | Error e -> Format.kasprintf failwith "%a" Font.pp_error e

(* Pairs of pictures that differ in one field. *)
let differ =
  let s ?fills ?strokes ?scales xs =
    Picture.stamp ?fills ?strokes ?scales xs [| 0. |] dot
  in
  let g ?(at = P2.v 0. 0.) text = Picture.glyphs red at (typeset text) in
  let img h = Picture.image (Box2.v 0. 0. 3. h) pixels in
  let px v shape =
    Picture.image (Box2.v 0. 0. 1. 1.) (Nx.full Nx.uint8 shape v)
  in
  let tag id rows = Picture.tag { id = Nx.Ptree.Path.v id; rows } dot in
  let cells h =
    Picture.Cells { box = Box2.v 0. 0. 1. h; width = 1; height = 1 }
  in
  [
    ("fill rule", Picture.fill ~rule:`Even_odd red square, dot);
    ("fill colour", Picture.fill Color.blue square, dot);
    ("fill and stroke", Picture.stroke (Stroke.v 1.) red square, dot);
    ("stamp positions", s [| 0. |], s [| 1. |]);
    ("stamp fills", s ~fills:[| red |] [| 0. |], s [| 0. |]);
    ("stamp strokes", s ~strokes:[| red |] [| 0. |], s [| 0. |]);
    ("stamp scales", s ~scales:[| 1. |] [| 0. |], s [| 0. |]);
    ("stamp picture", s [| 0. |], Picture.stamp [| 0. |] [| 0. |] bar);
    ("opacity", Picture.opacity 0.5 dot, Picture.opacity 0.25 dot);
    ( "clip rule",
      Picture.clip ~rule:`Even_odd square dot,
      Picture.clip square dot );
    ("clip path", Picture.clip line dot, Picture.clip square dot);
    ("clip picture", Picture.clip square dot, Picture.clip square bar);
    ( "transform",
      Picture.transform (Affine.translate 1. 0.) dot,
      Picture.transform (Affine.translate 0. 1.) dot );
    ("glyph run", g "ab", g "ac");
    ("glyph origin", g ~at:(P2.v 1. 0.) "ab", g "ab");
    ("image box", img 2., img 4.);
    ("image elements", px 7 [| 1; 2; 3 |], px 8 [| 1; 2; 3 |]);
    ("image shape", px 7 [| 1; 2; 3 |], px 7 [| 2; 1; 3 |]);
    ( "tag id",
      tag [ Field "a" ] (Rows [| 1 |]),
      tag [ Field "b" ] (Rows [| 1 |]) );
    ( "tag rows",
      tag [ Field "a" ] (Rows [| 1 |]),
      tag [ Field "a" ] (Rows [| 2 |]) );
    ("tag cells", tag [] (cells 1.), tag [] (cells 2.));
  ]

let comparing =
  group "comparing"
    [
      prop "equal is an equivalence"
        (Gen.pair Vg_corpus.gen_picture Vg_corpus.gen_picture)
        (Law.equivalence ~respell:Vg_corpus.respell picture);
      cases
        ~name:(fun (n, _, _) -> n)
        "equal tells apart" differ
        (fun (_, a, b) -> not_equal picture a b);
      test "equal compares images by elements and fonts by bytes" (fun () ->
          let px () = Nx.full Nx.uint8 [| 1; 2; 3 |] 7 in
          let b = Box2.v 0. 0. 1. 1. in
          equal picture (Picture.image b (px ())) (Picture.image b (px ()));
          let glyphs = [| Font.glyph Font.regular (Uchar.of_char 'a') |] in
          let r f = Run.v ~font:f ~size:9. ~text:"a" ~glyphs ~xs:[| 0. |] () in
          equal picture
            (Picture.glyphs red (P2.v 0. 0.) (r Font.regular))
            (Picture.glyphs red (P2.v 0. 0.) (r font_copy)));
      test "pp formats each case" (fun () ->
          let p =
            Picture.group
              [
                Picture.clip (rect 0. 0. 4. 4.)
                  (Picture.transform (Affine.translate 1. 2.)
                     (Picture.opacity 0.5
                        (Picture.stroke (Stroke.v 1.) red line)));
                Picture.tag
                  (tag_of (Rows [| 7; 8 |]))
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

let renderable = Testable.make ~pp:Renderable.pp ~equal:Renderable.equal

let renderables =
  group "renderables"
    [
      cases
        ~name:(fun (w, h) -> Printf.sprintf "%g by %g" w h)
        "v raises on a page of"
        [ (0., 1.); (1., -1.); (Float.nan, 1.); (1., infinity) ]
        (fun (w, h) -> invalid "Renderable.v" (fun () -> Renderable.v w h dot));
      test "v keeps its page and picture" (fun () ->
          let r = Renderable.v 360. 240. dot in
          equal float_exact 360. (Renderable.w r);
          equal float_exact 240. (Renderable.h r);
          equal picture dot (Renderable.picture r));
      test "equal compares sizes and pictures" (fun () ->
          let r = Renderable.v 1. 2. dot in
          equal renderable r (Renderable.v 1. 2. dot);
          not_equal renderable r (Renderable.v 2. 2. dot);
          not_equal renderable r (Renderable.v 1. 2. bar));
    ]

let () =
  exit
    (run "hugin.vg picture"
       [ constructors; bounds_group; comparing; renderables ])
