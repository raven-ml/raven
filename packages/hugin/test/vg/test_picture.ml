(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_gg
open Hugin_font
open Hugin_vg

let invalid f = raises_match Exn.invalid_arg f
let pp_float ppf x = Format.fprintf ppf "%.17g" x

let pp_box ppf b =
  Format.fprintf ppf "[%a, %a; %a, %a]" pp_float (Box2.minx b) pp_float
    (Box2.miny b) pp_float (Box2.maxx b) pp_float (Box2.maxy b)

let box2 = Testable.make ~pp:pp_box ~equal:Box2.equal

(* The boxes compared with it come from [sqrt 2], the sine and cosine of a
   quarter turn, or a glyph's ink moved by a few points: a few roundings each,
   within 8 ulps of the expected value. *)
let near a b =
  Float.equal a b
  || Float.abs (a -. b) <= 8. *. epsilon_float *. Float.max 1. (Float.abs a)

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
let pixels () = Nx.zeros Nx.uint8 [| 2; 3; 4 |]
let typeset text = Vg_corpus.typeset text 10.
let tag_of rows = { Picture.id = Nx.Ptree.Path.v [ Field "dots" ]; rows }

let union a b =
  match (a, b) with
  | None, x | x, None -> x
  | Some a, Some b -> Some (Box2.union a b)

let shift dx dy b =
  Box2.v (Box2.minx b +. dx) (Box2.miny b +. dy) (Box2.w b) (Box2.h b)

let encloses outer inner =
  Box2.minx outer <= Box2.minx inner
  && Box2.miny outer <= Box2.miny inner
  && Box2.maxx inner <= Box2.maxx outer
  && Box2.maxy inner <= Box2.maxy outer

(* [tags p] is the ids of the tags in [p], outermost first. *)
let rec tags (p : Picture.t) =
  match p with
  | Empty | Fill _ | Stroke _ | Glyphs _ | Image _ -> []
  | Group ps -> List.concat_map tags ps
  | Clip { picture; _ }
  | Transform { picture; _ }
  | Opacity { picture; _ }
  | Stamp { picture; _ } ->
      tags picture
  | Tag { tag; picture } -> Nx.Ptree.Path.to_string tag.id :: tags picture

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

(* Leaves *)

let fill =
  group "fill"
    [
      test "takes the rule `Nonzero by default" (fun () ->
          equal picture (Picture.fill ~rule:`Nonzero red square) dot);
      test "is empty for the empty path" (fun () ->
          equal picture Picture.empty (Picture.fill red Path.empty));
    ]

let stroke =
  group "stroke"
    [
      cases "is empty for" ~id:fst
        [
          ("the empty path", (Stroke.v 1., Path.empty));
          ("width 0", (Stroke.v 0., line));
        ]
        (fun (_, (s, q)) ->
          equal picture Picture.empty (Picture.stroke s red q));
    ]

let glyphs =
  group "glyphs"
    [
      cases "is empty for" ~id:fst
        [
          ("a run without glyphs", (P2.v 0. 0., ("", 10.)));
          ("a run of size 0", (P2.v 0. 0., ("ab", 0.)));
          ("an origin at nan", (P2.v Float.nan 0., ("ab", 10.)));
          ("an origin at infinity", (P2.v 0. infinity, ("ab", 10.)));
        ]
        (fun (_, (at, (text, size))) ->
          let r = Vg_corpus.typeset text size in
          equal picture Picture.empty (Picture.glyphs red at r));
    ]

let image =
  let image box shape = Picture.image box (Nx.zeros Nx.uint8 shape) in
  let unit = Box2.v 0. 0. 1. 1. in
  group "image"
    [
      test "keeps its tensor as given" (fun () ->
          let px = pixels () in
          let kept =
            require_match
              (function
                | Picture.Image { pixels; _ } -> Some pixels | _ -> None)
              (Picture.image (Box2.v 0. 0. 3. 2.) px)
          in
          let same = Testable.make ~pp:Nx.pp ~equal:( == ) in
          equal ~msg:"the tensor given, physically" same px kept);
      cases "is empty for" ~id:fst
        [
          ("a tensor without rows", (unit, [| 0; 2; 3 |]));
          ("a tensor without columns", (unit, [| 2; 0; 1 |]));
          ("a box of width 0", (Box2.v 0. 0. 0. 1., [| 2; 3; 4 |]));
          ("a box of height 0", (Box2.v 0. 0. 1. 0., [| 2; 3; 4 |]));
        ]
        (fun (_, (box, shape)) -> equal picture Picture.empty (image box shape));
      cases "raises on a tensor of shape"
        ~id:(fun s ->
          String.concat "; " (Array.to_list (Array.map string_of_int s)))
        [
          [| 2; 3 |];
          [| 2; 3; 0 |];
          [| 2; 3; 2 |];
          [| 2; 3; 5 |];
          [| 1; 2; 3; 4 |];
          (* Without a pixel, which would make it empty. *)
          [| 0; 2; 2 |];
        ]
        (fun shape -> invalid (fun () -> image unit shape));
    ]

(* Composing *)

let group_ =
  group "group"
    [
      cases "is empty for" ~id:fst
        [
          ("no picture", []); ("empties only", [ Picture.empty; Picture.empty ]);
        ]
        (fun (_, ps) -> equal picture Picture.empty (Picture.group ps));
      prop "drops the empty pictures"
        (Gen.pair Vg_corpus.gen_picture Vg_corpus.gen_picture) (fun (a, b) ->
          let e = Picture.empty in
          equal picture
            (Picture.group [ a; b ])
            (Picture.group [ e; a; e; b; e ]));
      prop "of one picture is that picture" Vg_corpus.gen_picture (fun p ->
          equal picture p (Picture.group [ p ]));
    ]

let clip =
  group "clip"
    [
      test "takes the rule `Nonzero by default" (fun () ->
          equal picture
            (Picture.clip ~rule:`Nonzero square bar)
            (Picture.clip square bar));
      cases "is empty for" ~id:fst
        [
          ("an empty picture", (square, Picture.empty));
          ("the empty path", (Path.empty, dot));
        ]
        (fun (_, (q, p)) -> equal picture Picture.empty (Picture.clip q p));
    ]

let transform =
  group "transform"
    [
      cases "is empty for" ~id:fst
        [
          ("an empty picture", (Affine.translate 1. 1., Picture.empty));
          ("a map without inverse", (Affine.scale 0. 1., dot));
          ( "a map with a nan coefficient",
            ({ Affine.id with x0 = Float.nan }, dot) );
        ]
        (fun (_, (m, p)) -> equal picture Picture.empty (Picture.transform m p));
    ]

let opacity =
  group "opacity"
    [
      prop "1. leaves the picture as it is" Vg_corpus.gen_picture (fun p ->
          equal picture p (Picture.opacity 1. p));
      prop "0. keeps the bounds" Vg_corpus.gen_picture (fun p ->
          cover "the picture has extent" (Option.is_some (bounds p));
          equal (option box2) (bounds p) (bounds (Picture.opacity 0. p)));
      prop "0. keeps the tags"
        ~examples:[ Picture.tag (tag_of (Rows [| 3 |])) dot ]
        Vg_corpus.gen_picture
        (fun p ->
          cover "the picture has a tag" (tags p <> []);
          equal (list string) (tags p) (tags (Picture.opacity 0. p)));
      test "is empty for an empty picture" (fun () ->
          equal picture Picture.empty (Picture.opacity 0.5 Picture.empty));
      cases "raises on" ~id:fst
        [
          ("-0.1", (-0.1, dot));
          ("the float below 0.", (Float.pred 0., dot));
          ("the float above 1.", (Float.succ 1., dot));
          ("1.1", (1.1, dot));
          ("nan", (Float.nan, dot));
          ("infinity", (infinity, dot));
          ("2. for an empty picture", (2., Picture.empty));
        ]
        (fun (_, (a, p)) -> invalid (fun () -> Picture.opacity a p));
    ]

let render p =
  Nx.to_array (Hugin_vg_raster.render ~density:1. (Renderable.v 10. 10. p))

let copies () =
  let xs = [| 1.; 2. |] and ys = [| 3.; 4. |] in
  let fills = [| red; Color.blue |] and scales = [| 1.; 2. |] in
  let p = Picture.stamp ~fills ~scales xs ys dot in
  let expected =
    Picture.stamp ~fills:[| red; Color.blue |] ~scales:[| 1.; 2. |] [| 1.; 2. |]
      [| 3.; 4. |] dot
  in
  List.iter (fun a -> a.(0) <- 9.) [ xs; ys; scales ];
  fills.(0) <- Color.green;
  equal picture expected p

let stamp =
  group "stamp"
    [
      test "replaces the colour of a transparent fill" (fun () ->
          let p = Picture.fill Color.transparent square in
          equal (array int) (render dot)
            (render (Picture.stamp ~fills:[| red |] [| 0. |] [| 0. |] p)));
      cases "skips an instance whose position or scale is" ~id:fst
        [
          ("x nan", (Float.nan, 0., 1.));
          ("y infinity", (50., infinity, 1.));
          ("nan", (50., 0., Float.nan));
          ("infinity", (50., 0., infinity));
          ("-infinity", (50., 0., neg_infinity));
        ]
        (fun (_, (x, y, k)) ->
          equal (option box2)
            (Some (Box2.v 50. 0. 10. 10.))
            (bounds
               (Picture.stamp ~scales:[| 1.; k |] [| 50.; x |] [| 0.; y |] dot)));
      test "copies its arrays" copies;
      cases "is empty for" ~id:fst
        [
          ("an empty picture", ([| 0. |], Picture.empty));
          ("no position", ([||], dot));
        ]
        (fun (_, (xs, p)) ->
          equal picture Picture.empty (Picture.stamp xs xs p));
      cases "raises on" ~id:fst
        [
          ("fewer ys", fun () -> Picture.stamp [| 0.; 1. |] [| 0. |] dot);
          ( "fewer fills",
            fun () ->
              Picture.stamp ~fills:[| red |] [| 0.; 1. |] [| 0.; 1. |] dot );
          ( "no strokes",
            fun () -> Picture.stamp ~strokes:[||] [| 0. |] [| 0. |] dot );
          ( "more scales",
            fun () -> Picture.stamp ~scales:[| 1.; 1. |] [| 0. |] [| 0. |] dot
          );
          ( "a scale of -1",
            fun () -> Picture.stamp ~scales:[| -1. |] [| 0. |] [| 0. |] dot );
          ( "a scale of the float below 0.",
            fun () ->
              Picture.stamp ~scales:[| Float.pred 0. |] [| 0. |] [| 0. |] dot );
          (* Arguments that would otherwise make it empty. *)
          ( "fills without position",
            fun () -> Picture.stamp ~fills:[| red |] [||] [||] dot );
          ( "a scale of -1 on an empty picture",
            fun () ->
              Picture.stamp ~scales:[| -1. |] [| 0. |] [| 0. |] Picture.empty );
        ]
        (fun (_, f) -> invalid f);
    ]

(* Tags *)

let tag =
  let grid width height p =
    Picture.tag (tag_of (Cells { box = Box2.v 0. 0. 1. 1.; width; height })) p
  in
  group "tag"
    [
      test "copies its rows" (fun () ->
          let rows = [| 4; 5 |] in
          let p = Picture.tag (tag_of (Rows rows)) dot in
          rows.(0) <- 0;
          equal picture (Picture.tag (tag_of (Rows [| 4; 5 |])) dot) p);
      test "takes rows of any length on a picture other than a stamp" (fun () ->
          let s = Picture.stamp [| 0.; 1. |] [| 0.; 1. |] dot in
          let moved = Picture.transform (Affine.translate 1. 0.) s in
          not_equal picture Picture.empty
            (Picture.tag (tag_of (Rows [| 7 |])) moved);
          not_equal picture Picture.empty
            (Picture.tag (tag_of (Rows [| 1; 2; 3 |])) dot));
      test "is empty for an empty picture" (fun () ->
          equal picture Picture.empty
            (Picture.tag (tag_of (Rows [||])) Picture.empty));
      cases "raises on" ~id:fst
        [
          ( "fewer rows than a stamp's positions",
            fun () ->
              Picture.tag (tag_of (Rows [| 1 |]))
                (Picture.stamp [| 0.; 1. |] [| 0.; 1. |] dot) );
          ( "more rows than a stamp's positions",
            fun () ->
              Picture.tag
                (tag_of (Rows [| 1; 2 |]))
                (Picture.stamp [| 0. |] [| 0. |] dot) );
          ("a grid 0 wide", fun () -> grid 0 1 dot);
          ("a grid 0 high", fun () -> grid 1 0 dot);
          ("a grid -1 wide", fun () -> grid (-1) 3 dot);
          ("a grid 0 wide on an empty picture", fun () -> grid 0 1 Picture.empty);
        ]
        (fun (_, f) -> invalid f);
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
   [stamp xs ys p], each wrapped by [wrap], where [ys] is [1 - xs], and the
   bounds of the stamp wrapped. *)
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

let clip_within (q, p) =
  let b = bounds (Picture.clip q p) in
  cover "the clip has extent" (Option.is_some b);
  match b with
  | None -> ()
  | Some b ->
      let within = function Some o -> encloses o b | None -> false in
      satisfies ~claim:"within the picture's bounds" box2
        (fun _ -> within (bounds p))
        b;
      satisfies ~claim:"within the path's bounds" box2
        (fun _ -> within (Path.bounds q))
        b

let bounds_ =
  group "bounds"
    [
      test "of a fill is its path's box" (fun () ->
          equal (option box2) (Some (Box2.v 0. 0. 10. 10.)) (bounds dot));
      test "of a transparent fill is its path's box" (fun () ->
          equal (option box2) (bounds dot)
            (bounds (Picture.fill Color.transparent square)));
      cases "of a stroke is its path's box grown by its pen's reach:"
        ~id:(fun (n, _, _) -> n)
        pen_reaches
        (fun (_, s, r) ->
          equal (option box2_near)
            (Some (Box2.v (-.r) (-.r) (10. +. (2. *. r)) (2. *. r)))
            (bounds (Picture.stroke s red line)));
      test "of glyphs is their ink at their origin" (fun () ->
          let ink = Option.get (Run.bounds (typeset "Hi")) in
          equal (option box2_near)
            (Some (shift 5. 7. ink))
            (bounds (Picture.glyphs red (P2.v 5. 7.) (typeset "Hi"))));
      test "of an image is its box" (fun () ->
          equal (option box2)
            (Some (Box2.v 1. 2. 3. 4.))
            (bounds (Picture.image (Box2.v 1. 2. 3. 4.) (pixels ()))));
      test "of a path of no area is a flat box" (fun () ->
          let p = Picture.fill red (Path.polyline [| 5.; 5. |] [| 0.; 10. |]) in
          equal (option box2) (Some (Box2.v 5. 0. 0. 10.)) (bounds p));
      cases "is None without extent:" ~id:fst
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
      prop "of a group is the union of its pictures'"
        (Gen.pair Vg_corpus.gen_picture Vg_corpus.gen_picture)
        (Law.homomorphic picture (option box2) bounds
           (fun a b -> Picture.group [ a; b ])
           union);
      prop "of a translation are the bounds moved"
        (Gen.triple quarter quarter gen_picture) (fun (dx, dy, p) ->
          equal (option box2)
            (Option.map (shift dx dy) (bounds p))
            (bounds (Picture.transform (Affine.translate dx dy) p)));
      test "of a rotation is the box of the rotated box" (fun () ->
          let r = Float.sqrt 2. *. 5. in
          equal (option box2_near)
            (Some (Box2.v (-.r) 0. (2. *. r) (2. *. r)))
            (bounds (Picture.transform (Affine.rotate (Float.pi /. 4.)) dot)));
      prop "of a clip lie within its picture's and its path's"
        ~examples:[ (rect 5. 5. 20. 20., dot) ]
        (Gen.pair
           (Gen.with_pp Path.pp Vg_corpus.gen_path)
           Vg_corpus.gen_picture)
        clip_within;
      cases "of a clip of a leaf is the meet of their boxes:" ~id:fst
        [
          ("overlapping", (rect 5. (-5.) 20. 10., Box2.v 5. 0. 5. 5.));
          ( "touching along a vertical edge",
            (rect 10. 0. 5. 5., Box2.v 10. 0. 0. 5.) );
          ( "touching along a horizontal edge",
            (rect 0. 10. 5. 5., Box2.v 0. 10. 5. 0.) );
        ]
        (fun (_, (q, b)) ->
          equal (option box2) (Some b) (bounds (Picture.clip q dot)));
      test "of a clip cut each leaf apart" (fun () ->
          let far = Picture.fill Color.blue (rect 20. 20. 10. 10.) in
          equal (option box2)
            (Some (Box2.v 0. 0. 10. 10.))
            (bounds
               (Picture.clip (rect 0. 0. 30. 15.) (Picture.group [ dot; far ]))));
      prop "of a stamp are the union of its instances'"
        (Gen.pair gen_placement (Gen.pair gen_positions gen_picture))
        (fun (m, s) ->
          let expected, got = instances (Picture.transform m) s in
          equal (option box2) expected got);
      prop "of a stamp under a clip are the union of its instances' under it"
        ~examples:stamp_examples
        (Gen.pair gen_rect (Gen.pair gen_positions gen_picture))
        (fun (q, s) ->
          let expected, got = instances (Picture.clip q) s in
          cover "an instance meets the clip" (Option.is_some got);
          equal (option box2) expected got);
      test "of a stroke in a scaled stamp keep the pen's reach" (fun () ->
          let p = Picture.stroke (Stroke.v 2.) red line in
          equal (option box2)
            (Some (Box2.v (-1.) (-1.) 22. 2.))
            (bounds (Picture.stamp ~scales:[| 2. |] [| 0. |] [| 0. |] p)));
      test "raises when a corner is not finite" (fun () ->
          let far =
            Picture.stroke (Stroke.v 1.) red
              (Path.polyline [| 0.; 1e308 |] [| 0.; 0. |])
          in
          invalid (fun () ->
              bounds (Picture.transform (Affine.scale 10. 10.) far)));
    ]

(* Comparing and formatting *)

(* Pairs of pictures that differ in one field. *)
let differ =
  let s ?fills ?strokes ?scales xs =
    Picture.stamp ?fills ?strokes ?scales xs [| 0. |] dot
  in
  let g ?(at = P2.v 0. 0.) text = Picture.glyphs red at (typeset text) in
  let img h = Picture.image (Box2.v 0. 0. 3. h) (pixels ()) in
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
    ( "group nesting",
      Picture.group [ Picture.group [ dot; bar ]; dot ],
      Picture.group [ dot; bar; dot ] );
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

let fonts_by_bytes () =
  let copy = Result.get_ok (Font.of_string (Font.bytes Font.regular)) in
  let glyphs = [| Font.glyph Font.regular (Uchar.of_char 'a') |] in
  let run font = Run.v ~font ~size:9. ~text:"a" ~glyphs ~xs:[| 0. |] () in
  equal picture
    (Picture.glyphs red (P2.v 0. 0.) (run Font.regular))
    (Picture.glyphs red (P2.v 0. 0.) (run copy))

let equal_ =
  group "equal"
    [
      prop "is an equivalence"
        (Gen.pair Vg_corpus.gen_picture Vg_corpus.gen_picture)
        (Law.equivalence ~respell:Vg_corpus.respell picture);
      cases "tells apart pictures that differ in"
        ~id:(fun (n, _, _) -> n)
        differ
        (fun (_, a, b) -> not_equal picture a b);
      test "compares image tensors by their elements" (fun () ->
          let px () = Nx.full Nx.uint8 [| 1; 2; 3 |] 7 in
          let b = Box2.v 0. 0. 1. 1. in
          equal picture (Picture.image b (px ())) (Picture.image b (px ())));
      test "compares fonts by their bytes" fonts_by_bytes;
      cases "compares numbers with Float.equal:" ~id:fst
        [
          ("nan equals nan", (Float.nan, Float.nan));
          ("-0. equals 0.", (-0., 0.));
        ]
        (fun (_, (x, x')) ->
          equal picture
            (Picture.stamp [| x |] [| 0. |] dot)
            (Picture.stamp [| x' |] [| 0. |] dot));
    ]

let pp =
  group "pp"
    [
      test "formats each case as an s-expression, paths as path data" (fun () ->
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
                Picture.image (Box2.v 0. 0. 3. 2.) (pixels ());
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
  let r = Renderable.v 360. 240. dot in
  group "Renderable"
    [
      cases "v raises on a page of"
        ~id:(fun (w, h) -> Printf.sprintf "%g by %g" w h)
        [
          (0., 1.);
          (-0., 1.);
          (1., -1.);
          (Float.nan, 1.);
          (1., infinity);
          (neg_infinity, 1.);
        ]
        (fun (w, h) -> invalid (fun () -> Renderable.v w h dot));
      test "w is the width of the page" (fun () ->
          equal float_exact 360. (Renderable.w r));
      test "h is the height of the page" (fun () ->
          equal float_exact 240. (Renderable.h r));
      test "picture is the picture on the page" (fun () ->
          equal picture dot (Renderable.picture r));
      test "equal holds for equal sizes and pictures" (fun () ->
          equal renderable r (Renderable.v 360. 240. dot));
      cases "equal tells apart renderables that differ in" ~id:fst
        [
          ("width", Renderable.v 1. 240. dot);
          ("height", Renderable.v 360. 1. dot);
          ("picture", Renderable.v 360. 240. bar);
        ]
        (fun (_, r') -> not_equal renderable r r');
    ]

let () =
  exit
    (run "hugin.vg picture"
       [
         fill;
         stroke;
         glyphs;
         image;
         group_;
         clip;
         transform;
         opacity;
         stamp;
         tag;
         bounds_;
         equal_;
         pp;
         renderables;
       ])
