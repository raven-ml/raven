(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_gg

(* Witnesses print every digit, so that a failure shows the difference. *)

let pp_float ppf x = Format.fprintf ppf "%.17g" x

let pp_affine ppf (m : Affine.t) =
  Format.fprintf ppf "{xx=%a; yx=%a; xy=%a; yy=%a; x0=%a; y0=%a}" pp_float m.xx
    pp_float m.yx pp_float m.xy pp_float m.yy pp_float m.x0 pp_float m.y0

let pp_p2 ppf p =
  Format.fprintf ppf "(%a, %a)" pp_float (P2.x p) pp_float (P2.y p)

let pp_box ppf b =
  Format.fprintf ppf "[%a, %a; %a, %a]" pp_float (Box2.minx b) pp_float
    (Box2.miny b) pp_float (Box2.maxx b) pp_float (Box2.maxy b)

let affine =
  Testable.with_compare Affine.compare
    (Testable.make ~pp:pp_affine ~equal:Affine.equal)

let p2 =
  Testable.with_compare P2.compare (Testable.make ~pp:pp_p2 ~equal:P2.equal)

let box2 =
  Testable.with_compare Box2.compare
    (Testable.make ~pp:pp_box ~equal:Box2.equal)

let near a b =
  Float.equal a b
  || Float.abs (a -. b) <= 1e-9 *. (1. +. Float.abs a +. Float.abs b)

let affine_near =
  Testable.make ~pp:pp_affine ~equal:(fun (m : Affine.t) (n : Affine.t) ->
      near m.xx n.xx && near m.yx n.yx && near m.xy n.xy && near m.yy n.yy
      && near m.x0 n.x0 && near m.y0 n.y0)

let p2_near =
  Testable.make ~pp:pp_p2 ~equal:(fun p q ->
      near (P2.x p) (P2.x q) && near (P2.y p) (P2.y q))

(* Generators *)

let coord = Gen.float_range (-100.) 100.
let size = Gen.float_range 0. 100.

(* Floats whose equality the conventions single out: zeros of both signs, NaN
   and the infinities. *)
let special =
  Gen.frequency
    [
      (4, Gen.float_range (-2.) 2.);
      ( 1,
        Gen.of_list ~pp:pp_float
          [ 0.; -0.; Float.nan; infinity; neg_infinity; 1. ] );
    ]

let gen_p2 =
  Gen.with_pp pp_p2 (Gen.map (fun (x, y) -> P2.v x y) (Gen.pair coord coord))

let gen_affine_of g =
  Gen.with_pp pp_affine
    (Gen.map
       (fun ((xx, yx, xy), (yy, x0, y0)) -> { Affine.xx; yx; xy; yy; x0; y0 })
       (Gen.pair (Gen.triple g g g) (Gen.triple g g g)))

let gen_affine = gen_affine_of (Gen.float_range (-10.) 10.)

(* Rotations, positive scales and translations: inverses are well
   conditioned. *)
let gen_invertible =
  Gen.with_pp pp_affine
    (Gen.map
       (fun ((a, sx, sy), (dx, dy)) ->
         Affine.(translate dx dy * rotate a * scale sx sy))
       (Gen.pair
          (Gen.triple (Gen.float_range (-7.) 7.) (Gen.float_range 0.1 10.)
             (Gen.float_range (-10.) (-0.1)))
          (Gen.pair coord coord)))

let gen_box =
  Gen.with_pp pp_box
    (Gen.map
       (fun ((x, y), (w, h)) -> Box2.v x y w h)
       (Gen.pair (Gen.pair coord coord) (Gen.pair size size)))

(* Affine *)

let id_is_neutral () =
  let m = { Affine.xx = 2.; yx = 3.; xy = 5.; yy = 7.; x0 = 11.; y0 = 13. } in
  Law.neutral affine Affine.( * ) Affine.id m

let rotation_quarter_turn () =
  let m = Affine.rotate (Float.pi /. 2.) in
  equal p2_near (P2.v 0. 1.) (P2.transform m (P2.v 1. 0.));
  equal p2_near (P2.v (-1.) 0.) (P2.transform m (P2.v 0. 1.))

let composition_order () =
  let p = P2.v 1. 2. in
  equal ~msg:"scale, then translate" p2 (P2.v 12. 26.)
    (P2.transform Affine.(translate 10. 20. * scale 2. 3.) p);
  equal ~msg:"translate, then scale" p2 (P2.v 22. 66.)
    (P2.transform Affine.(scale 2. 3. * translate 10. 20.) p)

let singular_maps =
  [
    ("zero x scale", Affine.scale 0. 1.);
    ("parallel columns", { Affine.id with xx = 2.; yx = 4.; xy = 1.; yy = 2. });
    ("NaN translation", { Affine.id with x0 = Float.nan });
    ("infinite coefficient", { Affine.id with xx = infinity });
    ( "inverse translation overflows",
      { (Affine.scale 1e-5 1e-5) with x0 = 1e308 } );
  ]

let invert_extreme_scales () =
  equal affine_near (Affine.scale 1e200 1e200)
    (require_some (Affine.invert (Affine.scale 1e-200 1e-200)));
  equal affine_near
    (Affine.scale 1e-200 1e-200)
    (require_some (Affine.invert (Affine.scale 1e200 1e200)))

let equality_of_floats () =
  equal affine
    { Affine.id with x0 = Float.nan }
    { Affine.id with x0 = Float.nan };
  equal affine { Affine.id with x0 = 0. } { Affine.id with x0 = -0. };
  not_equal ~msg:"maps differing in y0 alone" affine Affine.id
    { Affine.id with y0 = 1. }

let lexicographic_order () =
  let ordered =
    Affine.
      [
        { id with y0 = 1. };
        { id with x0 = 1. };
        { id with yy = 2. };
        { id with xy = 1. };
        { id with yx = 1. };
        { id with xx = 2. };
      ]
  in
  equal (list affine) ordered (List.sort Affine.compare (List.rev ordered))

(* The largest length of the image of a unit vector over a fine sweep of
   directions: the stretch's definition, computed the slow way. *)
let swept_stretch (m : Affine.t) =
  let best = ref 0. in
  for i = 0 to 3599 do
    let a = Float.pi *. Float.of_int i /. 3600. in
    let c = Float.cos a and s = Float.sin a in
    best :=
      Float.max !best
        (Float.hypot ((m.xx *. c) +. (m.xy *. s)) ((m.yx *. c) +. (m.yy *. s)))
  done;
  !best

(* The sweep misses the largest image by a relative 1e-7 at most. *)
let stretch_is_largest_image m =
  let s = Affine.stretch m and swept = swept_stretch m in
  at_least float_exact ~than:(swept *. (1. -. 1e-12)) s;
  at_most float_exact ~than:(swept *. (1. +. 1e-6)) s

let stretch_cases =
  [
    ("scale 2 -3", Affine.scale 2. (-3.), 3.);
    ("a rotation", Affine.rotate 1., 1.);
    ("a translation", Affine.translate 5. 7., 1.);
    ("a shear", { Affine.id with xy = 1. }, (1. +. Float.sqrt 5.) /. 2.);
    ("scale 1e-200", Affine.scale 1e-200 1e-200, 1e-200);
    ( "scale 1e200 under a rotation",
      Affine.(rotate 1. * scale 1e200 1e200),
      1e200 );
  ]

let affine_tests =
  group "Affine"
    [
      test "id is neutral for composition" id_is_neutral;
      prop "translate moves points by its offsets"
        (Gen.triple coord coord gen_p2) (fun (dx, dy, p) ->
          equal p2
            (P2.v (P2.x p +. dx) (P2.y p +. dy))
            (P2.transform (Affine.translate dx dy) p));
      prop "scale multiplies coordinates by its factors"
        (Gen.triple coord coord gen_p2) (fun (sx, sy, p) ->
          equal p2
            (P2.v (P2.x p *. sx) (P2.y p *. sy))
            (P2.transform (Affine.scale sx sy) p));
      test "rotate turns the x axis towards the y axis" rotation_quarter_turn;
      prop "rotate keeps distances to the origin"
        (Gen.pair (Gen.float_range (-7.) 7.) gen_p2)
        (fun (a, p) ->
          let q = P2.transform (Affine.rotate a) p in
          equal (float 1e-9)
            (Float.hypot (P2.x p) (P2.y p))
            (Float.hypot (P2.x q) (P2.y q)));
      test "( * ) applies its right operand first" composition_order;
      prop "( * ) maps points as successive maps"
        (Gen.triple gen_affine gen_affine gen_p2) (fun (m, n, p) ->
          equal p2_near
            (P2.transform m (P2.transform n p))
            (P2.transform Affine.(m * n) p));
      prop "( * ) is associative"
        (Gen.triple gen_affine gen_affine gen_affine)
        (Law.associative affine_near Affine.( * ));
      prop "invert gives the inverse map" gen_invertible
        (Law.invertible affine_near Affine.( * ) Affine.id (fun m ->
             require_some (Affine.invert m)));
      cases ~name:fst "invert is None for" singular_maps (fun (_, m) ->
          is_none ~pp:pp_affine (Affine.invert m));
      test "invert neither overflows nor underflows on extreme scales"
        invert_extreme_scales;
      prop "linear drops the translation of a composition"
        (Gen.pair gen_affine gen_affine)
        (Law.homomorphic affine affine Affine.linear Affine.( * ) Affine.( * ));
      test "linear keeps the linear coefficients" (fun () ->
          let m =
            { Affine.xx = 2.; yx = 3.; xy = 5.; yy = 7.; x0 = 1.; y0 = 1. }
          in
          equal affine { m with x0 = 0.; y0 = 0. } (Affine.linear m));
      cases
        ~name:(fun (n, _, _) -> n)
        "stretch of" stretch_cases
        (fun (_, m, s) ->
          equal (float_rel ~rel:1e-15 ~abs:0.) s (Affine.stretch m));
      cases ~name:fst "stretch is NaN for"
        [
          ("the zero map", Affine.scale 0. 0.);
          ("an infinite coefficient", Affine.scale infinity 1.);
        ]
        (fun (_, m) -> equal float_exact Float.nan (Affine.stretch m));
      prop "stretch is the largest image of a unit vector" gen_affine
        stretch_is_largest_image;
      test "equal follows Float.equal" equality_of_floats;
      prop "equal is an equivalence"
        (Gen.pair (gen_affine_of special) (gen_affine_of special))
        (Law.equivalence affine);
      prop "compare is a total order compatible with equal"
        (Gen.triple (gen_affine_of special) (gen_affine_of special)
           (gen_affine_of special))
        (Law.order affine);
      test "compare is lexicographic by xx, yx, xy, yy, x0 and y0"
        lexicographic_order;
    ]

(* P2 *)

let p2_tests =
  group "P2"
    [
      prop "x and y give back the coordinates of v"
        (Gen.pair Gen.any_float Gen.any_float) (fun (x, y) ->
          let p = P2.v x y in
          equal float_exact x (P2.x p);
          equal float_exact y (P2.y p));
      prop "transform is the map of the affine record"
        (Gen.pair gen_affine gen_p2) (fun ((m : Affine.t), p) ->
          let x = P2.x p and y = P2.y p in
          equal p2
            (P2.v
               ((m.xx *. x) +. (m.xy *. y) +. m.x0)
               ((m.yx *. x) +. (m.yy *. y) +. m.y0))
            (P2.transform m p));
      test "transform propagates non-finite coefficients" (fun () ->
          let q = P2.transform { Affine.id with x0 = Float.nan } (P2.v 1. 2.) in
          equal float_exact Float.nan (P2.x q);
          equal float_exact 2. (P2.y q));
      prop "equal is an equivalence"
        (Gen.pair (Gen.pair special special) (Gen.pair special special))
        (fun ((a, b), (c, d)) -> Law.equivalence p2 (P2.v a b, P2.v c d));
      prop "compare is a total order compatible with equal"
        (Gen.triple (Gen.pair special special) (Gen.pair special special)
           (Gen.pair special special))
        (fun ((a, b), (c, d), (e, f)) ->
          Law.order p2 (P2.v a b, P2.v c d, P2.v e f));
      test "compare orders by x, then by y" (fun () ->
          less p2 ~than:(P2.v 1. 0.) (P2.v 0. 5.);
          less p2 ~than:(P2.v 0. 1.) (P2.v 0. 0.));
    ]

(* Box2 *)

let box_corners () =
  let b = Box2.v 1. 2. 3. 4. in
  equal (list float_exact) [ 1.; 2.; 4.; 6.; 3.; 4. ]
    [ Box2.minx b; Box2.miny b; Box2.maxx b; Box2.maxy b; Box2.w b; Box2.h b ];
  equal p2 (P2.v 2.5 4.) (Box2.mid b)

let invalid_boxes =
  [
    ("negative width", (0., 0., -1., 1.));
    ("negative height", (0., 0., 1., -1.));
    ("NaN width", (0., 0., Float.nan, 1.));
    ("infinite corner", (infinity, 0., 1., 1.));
    ("right edge overflows", (Float.max_float, 0., Float.max_float, 1.));
  ]

let mid_of_huge_box () =
  let b = Box2.of_pts (P2.v (-.Float.max_float) 0.) (P2.v Float.max_float 0.) in
  equal p2 (P2.v 0. 0.) (Box2.mid b);
  let b = Box2.of_pts (P2.v Float.max_float 0.) (P2.v Float.max_float 0.) in
  equal p2 (P2.v Float.max_float 0.) (Box2.mid b)

let corners b =
  [
    P2.v (Box2.minx b) (Box2.miny b);
    P2.v (Box2.maxx b) (Box2.miny b);
    P2.v (Box2.maxx b) (Box2.maxy b);
    P2.v (Box2.minx b) (Box2.maxy b);
  ]

(* The smallest box containing the images of the corners, computed from its
   definition. *)
let transform_is_corner_hull (m, b) =
  let images = List.map (P2.transform m) (corners b) in
  let xs = List.map P2.x images and ys = List.map P2.y images in
  let lo = List.fold_left Float.min infinity
  and hi = List.fold_left Float.max neg_infinity in
  equal box2
    (Box2.of_pts (P2.v (lo xs) (lo ys)) (P2.v (hi xs) (hi ys)))
    (Box2.transform m b)

let rotation_enlarges () =
  let b =
    Box2.transform (Affine.rotate (Float.pi /. 4.)) (Box2.v (-1.) (-1.) 2. 2.)
  in
  equal (float 1e-12) (2. *. Float.sqrt 2.) (Box2.w b);
  equal (float 1e-12) (2. *. Float.sqrt 2.) (Box2.h b)

(* Boxes on a small grid of integers: they often touch, nest and miss each
   other, and their corners are exact. *)
let gen_grid_box =
  let c = Gen.map Float.of_int (Gen.int_range (-3) 3) in
  Gen.with_pp pp_box
    (Gen.map
       (fun (p, q) -> Box2.of_pts p q)
       (Gen.pair
          (Gen.map (fun (x, y) -> P2.v x y) (Gen.pair c c))
          (Gen.map (fun (x, y) -> P2.v x y) (Gen.pair c c))))

let gen_half_pt =
  let c = Gen.map (fun k -> Float.of_int k /. 2.) (Gen.int_range (-7) 7) in
  Gen.with_pp pp_p2 (Gen.map (fun (x, y) -> P2.v x y) (Gen.pair c c))

let holds b p =
  Box2.minx b <= P2.x p
  && P2.x p <= Box2.maxx b
  && Box2.miny b <= P2.y p
  && P2.y p <= Box2.maxy b

(* A point is in [inter a b] iff it is in [a] and in [b]. *)
let inter_holds_common_points (a, b, p) =
  let both = holds a p && holds b p in
  cover "a common point" both;
  cover "touching boxes"
    (Option.is_some (Box2.inter a b) && Box2.maxx a = Box2.minx b);
  match Box2.inter a b with
  | None -> is_false ~msg:"no common point" both
  | Some i -> equal bool both (holds i p)

let invalid_grows =
  [
    ("shrinking a side past the other", -0.75, Box2.v 0. 0. 1. 2.);
    ("a NaN distance", Float.nan, Box2.v 0. 0. 1. 1.);
    ("an infinite distance", infinity, Box2.v 0. 0. 1. 1.);
    ( "a corner beyond max_float",
      Float.max_float,
      Box2.v Float.max_float 0. 0. 0. );
  ]

let box_tests =
  group "Box2"
    [
      test "v spans from the corner by the size" box_corners;
      cases ~name:fst "v raises on" invalid_boxes (fun (_, (x, y, w, h)) ->
          raises_match Exn.invalid_arg (fun () -> Box2.v x y w h));
      test "v accepts a zero size of either sign" (fun () ->
          equal float_exact 0. (Box2.w (Box2.v 1. 1. (-0.) 0.)));
      prop "of_pts takes opposite corners in any order" (Gen.pair gen_p2 gen_p2)
        (fun (p, q) ->
          equal box2 (Box2.of_pts p q) (Box2.of_pts q p);
          equal box2 (Box2.of_pts p q)
            (Box2.of_pts (P2.v (P2.x p) (P2.y q)) (P2.v (P2.x q) (P2.y p))));
      cases ~name:fst "of_pts raises on a corner with"
        [ ("NaN", P2.v Float.nan 0.); ("infinity", P2.v 0. neg_infinity) ]
        (fun (_, p) ->
          raises_match Exn.invalid_arg (fun () -> Box2.of_pts p (P2.v 1. 1.)));
      test "mid stays finite on the largest boxes" mid_of_huge_box;
      prop "union is commutative" (Gen.pair gen_box gen_box)
        (Law.commutative box2 Box2.union);
      prop "union is associative"
        (Gen.triple gen_box gen_box gen_box)
        (Law.associative box2 Box2.union);
      prop "union contains both boxes" (Gen.pair gen_box gen_box) (fun (a, b) ->
          let u = Box2.union a b in
          equal box2 u (Box2.union u a);
          equal box2 u (Box2.union u b));
      prop "inter holds the points common to both boxes"
        (Gen.triple gen_grid_box gen_grid_box gen_half_pt)
        inter_holds_common_points;
      prop "inter is commutative" (Gen.pair gen_grid_box gen_grid_box)
        (fun (a, b) -> equal (option box2) (Box2.inter a b) (Box2.inter b a));
      test "inter of boxes touching at a corner is that point" (fun () ->
          equal (option box2)
            (Some (Box2.v 1. 1. 0. 0.))
            (Box2.inter (Box2.v 0. 0. 1. 1.) (Box2.v 1. 1. 1. 1.)));
      prop "grow moves each side by the distance"
        (Gen.pair (Gen.map Float.of_int (Gen.int_range (-1) 3)) gen_grid_box)
        (fun (d, b) ->
          assume (2. *. d >= -.Box2.w b && 2. *. d >= -.Box2.h b);
          equal box2
            (Box2.of_pts
               (P2.v (Box2.minx b -. d) (Box2.miny b -. d))
               (P2.v (Box2.maxx b +. d) (Box2.maxy b +. d)))
            (Box2.grow d b));
      test "grow down to a zero size is allowed" (fun () ->
          equal box2 (Box2.v 1. 1. 0. 2.) (Box2.grow (-1.) (Box2.v 0. 0. 2. 4.)));
      cases
        ~name:(fun (n, _, _) -> n)
        "grow raises on" invalid_grows
        (fun (_, d, b) ->
          raises_match Exn.invalid_arg (fun () -> Box2.grow d b));
      prop "transform is the bounding box of the corners' images"
        (Gen.pair gen_affine gen_box)
        transform_is_corner_hull;
      test "transform under a rotation is larger than the box" rotation_enlarges;
      test "transform raises if an image is not finite" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Box2.transform (Affine.scale 1e308 1.) (Box2.v 0. 0. 10. 10.)));
      prop "equal is an equivalence" (Gen.pair gen_box gen_box)
        (Law.equivalence box2);
      prop "compare is a total order compatible with equal"
        (Gen.triple gen_box gen_box gen_box)
        (Law.order box2);
      test "equal compares every corner" (fun () ->
          not_equal box2 (Box2.v 0. 0. 1. 1.) (Box2.v 0. 0. 1. 2.));
      test "compare is lexicographic by minx, miny, maxx and maxy" (fun () ->
          let ordered =
            [
              Box2.v 0. 0. 1. 1.;
              Box2.v 0. 0. 1. 2.;
              Box2.v 0. 0. 2. 0.;
              Box2.v 0. 1. 0. 0.;
              Box2.v 1. 0. 0. 0.;
            ]
          in
          equal (list box2) ordered (List.sort Box2.compare (List.rev ordered)));
    ]

let () =
  exit (run "hugin.gg geometry" [ affine_tests; p2_tests; box_tests ])
