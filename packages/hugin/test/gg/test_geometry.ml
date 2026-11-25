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

let coefficients (m : Affine.t) = [ m.xx; m.yx; m.xy; m.yy; m.x0; m.y0 ]
let edges b = [ Box2.minx b; Box2.miny b; Box2.maxx b; Box2.maxy b ]

(* Witnesses within an absolute tolerance, which report the largest error they
   saw. *)
let affine_within eps = Testable.contramap coefficients (list (float eps))

let p2_within eps =
  Testable.contramap (fun p -> (P2.x p, P2.y p)) (pair (float eps) (float eps))

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

(* Rotations, scales of either sign and translations, whose condition number is
   at most 100. *)
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

(* Both sides round a few times within half an ulp of [|p| <= 142]: below 1e-13
   apart. *)
let rotate_turns (a, p) =
  let r = Float.hypot (P2.x p) (P2.y p) in
  let t = Float.atan2 (P2.y p) (P2.x p) +. a in
  equal (p2_within 1e-12)
    (P2.v (r *. Float.cos t) (r *. Float.sin t))
    (P2.transform (Affine.rotate a) p)

let singular_maps =
  [
    ("a zero x scale", Affine.scale 0. 1.);
    ("parallel columns", { Affine.id with xx = 2.; yx = 4.; xy = 1.; yy = 2. });
    ("a NaN translation", { Affine.id with x0 = Float.nan });
    ("an infinite coefficient", { Affine.id with xx = infinity });
    ("an inverse that overflows", { (Affine.scale 1e-5 1e-5) with x0 = 1e308 });
  ]

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

(* The least subnormal float. Below the least normal float, results are rounded
   to multiples of it, so relative precision runs out. *)
let least_subnormal = Float.succ 0.

(* Maps from [gen_affine] with their linear part scaled by [0.] or by the least
   subnormal float as often as left as drawn: the zero map and stretches that
   only a few subnormals represent are cases the law must hold on. *)
let gen_stretched =
  Gen.with_pp pp_affine
    (Gen.map
       (fun ((m : Affine.t), f) ->
         {
           m with
           xx = m.xx *. f;
           yx = m.yx *. f;
           xy = m.xy *. f;
           yy = m.yy *. f;
         })
       (Gen.pair gen_affine
          (Gen.of_list ~pp:pp_float [ 1.; 1.; 0.; least_subnormal ])))

(* The sweep misses the largest image by a relative 1e-7 at most. Among
   subnormals it rounds each image coordinate within one least subnormal of its
   value and each length within two, and the stretch rounds within half of one:
   the two agree within three least subnormals. *)
let stretch_is_largest_image (m : Affine.t) =
  let s = Affine.stretch m and swept = swept_stretch m in
  let slack = 3. *. least_subnormal in
  cover "the zero map" (m.xx = 0. && m.yx = 0. && m.xy = 0. && m.yy = 0.);
  cover "a subnormal stretch" (0. < swept && swept < Float.min_float);
  at_least float_exact ~than:((swept *. (1. -. 1e-12)) -. slack) s;
  at_most float_exact ~than:((swept *. (1. +. 1e-6)) +. slack) s

(* Largest singular values: of a diagonal map its largest factor's magnitude, of
   a rotation or a translation 1, of the shear [(1 1) (0 1)] the golden
   ratio. *)
let stretches =
  [
    ("scale 2 -3, 3", Affine.scale 2. (-3.), 3.);
    ("a rotation, 1", Affine.rotate 1., 1.);
    ("a translation, 1", Affine.translate 5. 7., 1.);
    ( "a shear, the golden ratio",
      { Affine.id with xy = 1. },
      (1. +. Float.sqrt 5.) /. 2. );
    ("scale 1e-200, 1e-200", Affine.scale 1e-200 1e-200, 1e-200);
    ( "scale 1e200 under a rotation, 1e200",
      Affine.(rotate 1. * scale 1e200 1e200),
      1e200 );
    ("the zero map, 0", Affine.scale 0. 0., 0.);
    ( "the least subnormal in every coefficient, twice it",
      {
        Affine.xx = least_subnormal;
        yx = least_subnormal;
        xy = least_subnormal;
        yy = least_subnormal;
        x0 = 0.;
        y0 = 0.;
      },
      2. *. least_subnormal );
  ]

let fields =
  [
    ("xx", { Affine.id with xx = 2. });
    ("yx", { Affine.id with yx = 2. });
    ("xy", { Affine.id with xy = 2. });
    ("yy", { Affine.id with yy = 2. });
    ("x0", { Affine.id with x0 = 2. });
    ("y0", { Affine.id with y0 = 2. });
  ]

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

let affine_tests =
  group "Affine"
    [
      prop "id is neutral for ( * )" gen_affine
        (Law.neutral affine Affine.( * ) Affine.id);
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
      prop "rotate turns points about the origin towards the y axis"
        (Gen.pair (Gen.float_range (-7.) 7.) gen_p2)
        rotate_turns;
      (* The terms of either side are below 4.1e4 in magnitude, and each side
         rounds at most eight times: within 4e-11 of each other. *)
      prop "( * ) applies its right operand first"
        (Gen.triple gen_affine gen_affine gen_p2) (fun (m, n, p) ->
          equal (p2_within 1e-10)
            (P2.transform m (P2.transform n p))
            (P2.transform Affine.(m * n) p));
      (* The composition rounds within a few ulps of the condition number (100)
         times the translation (142): about 3e-11. *)
      prop "invert gives the inverse map" gen_invertible
        (Law.invertible (affine_within 1e-10) Affine.( * ) Affine.id (fun m ->
             require_some (Affine.invert m)));
      cases "invert is None for" ~id:fst singular_maps (fun (_, m) ->
          is_none ~pp:pp_affine (Affine.invert m));
      (* Each inverse coefficient is one division by the scaled determinant and
         one by the scale: within a few ulps. *)
      cases "invert neither overflows nor underflows on"
        ~id:(fun (n, _, _) -> n)
        [
          ("scale 1e-200", Affine.scale 1e-200 1e-200, Affine.scale 1e200 1e200);
          ("scale 1e200", Affine.scale 1e200 1e200, Affine.scale 1e-200 1e-200);
        ]
        (fun (_, m, inverse) ->
          equal
            (list (float_rel ~rel:1e-15 ~abs:0.))
            (coefficients inverse)
            (coefficients (require_some (Affine.invert m))));
      prop "linear is m with x0 and y0 zero" gen_affine (fun m ->
          equal affine { m with x0 = 0.; y0 = 0. } (Affine.linear m));
      (* The formula rounds a handful of times within half an ulp, 1.1e-16
         relative. *)
      cases "stretch is the largest singular value of linear m"
        ~id:(fun (n, _, _) -> n)
        stretches
        (fun (_, m, s) ->
          equal (float_rel ~rel:1e-15 ~abs:0.) s (Affine.stretch m));
      cases "stretch is NaN for a non-finite coefficient" ~id:fst
        [
          ("an infinite xx", Affine.scale infinity 1.);
          ("a NaN xy", { Affine.id with xy = Float.nan });
        ]
        (fun (_, m) -> equal float_exact Float.nan (Affine.stretch m));
      prop "stretch is the largest image of a unit vector" gen_stretched
        stretch_is_largest_image;
      cases "equal compares coefficients with Float.equal" ~id:fst
        [
          ("NaN equals NaN", ({ Affine.id with x0 = Float.nan }, Float.nan));
          ("-0. equals 0.", ({ Affine.id with x0 = -0. }, 0.));
        ]
        (fun (_, (m, x0)) -> equal affine { Affine.id with x0 } m);
      cases "equal tells maps apart by" ~id:fst fields (fun (_, m) ->
          not_equal affine Affine.id m);
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

(* The terms of either side are below 2010 in magnitude: two evaluations of the
   formula differ by a few ulps of 2048, 1e-12 at most, whether or not the
   compiler fuses a multiply and an add. *)
let transform_maps ((m : Affine.t), p) =
  let x = P2.x p and y = P2.y p in
  equal (p2_within 1e-12)
    (P2.v
       ((m.xx *. x) +. (m.xy *. y) +. m.x0)
       ((m.yx *. x) +. (m.yy *. y) +. m.y0))
    (P2.transform m p)

let p2_tests =
  group "P2"
    [
      prop "x and y give back the coordinates of v"
        (Gen.pair Gen.any_float Gen.any_float) (fun (x, y) ->
          let p = P2.v x y in
          equal (pair float_exact float_exact) (x, y) (P2.x p, P2.y p));
      prop "transform maps a point as Affine.t states"
        (Gen.pair gen_affine gen_p2)
        transform_maps;
      test "transform makes the coordinates a NaN coefficient reaches NaN"
        (fun () ->
          let q = P2.transform { Affine.id with x0 = Float.nan } (P2.v 1. 2.) in
          equal (pair float_exact float_exact) (Float.nan, 2.) (P2.x q, P2.y q));
      cases "equal compares coordinates with Float.equal" ~id:fst
        [
          ("NaN equals NaN", (P2.v Float.nan 1., P2.v Float.nan 1.));
          ("-0. equals 0.", (P2.v (-0.) 1., P2.v 0. 1.));
        ]
        (fun (_, (p, q)) -> equal p2 p q);
      prop "equal is an equivalence"
        (Gen.pair (Gen.pair special special) (Gen.pair special special))
        (fun ((a, b), (c, d)) -> Law.equivalence p2 (P2.v a b, P2.v c d));
      prop "compare is a total order compatible with equal"
        (Gen.triple (Gen.pair special special) (Gen.pair special special)
           (Gen.pair special special))
        (fun ((a, b), (c, d), (e, f)) ->
          Law.order p2 (P2.v a b, P2.v c d, P2.v e f));
      test "compare orders by x, then by y" (fun () ->
          equal (list p2)
            [ P2.v 0. 0.; P2.v 0. 5.; P2.v 1. 0. ]
            (List.sort P2.compare [ P2.v 1. 0.; P2.v 0. 5.; P2.v 0. 0. ]));
    ]

(* Box2 *)

let invalid_boxes =
  [
    ("a negative width", (0., 0., -1., 1.));
    ("a negative height", (0., 0., 1., -1.));
    ("a NaN width", (0., 0., Float.nan, 1.));
    ("a NaN corner", (Float.nan, 0., 1., 1.));
    ("an infinite corner", (0., infinity, 1., 1.));
    ("a right edge beyond max_float", (Float.max_float, 0., Float.max_float, 1.));
  ]

let huge =
  Box2.of_pts
    (P2.v (-.Float.max_float) (-.Float.max_float))
    (P2.v Float.max_float Float.max_float)

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

(* The box of the points in both: from the larger of the minima to the smaller
   of the maxima, if they are ordered. *)
let common a b =
  let minx = Float.max (Box2.minx a) (Box2.minx b)
  and miny = Float.max (Box2.miny a) (Box2.miny b)
  and maxx = Float.min (Box2.maxx a) (Box2.maxx b)
  and maxy = Float.min (Box2.maxy a) (Box2.maxy b) in
  if minx <= maxx && miny <= maxy then
    Some (Box2.of_pts (P2.v minx miny) (P2.v maxx maxy))
  else None

let inter_is_common (a, b) =
  let i = common a b and solid b = Box2.w b > 0. && Box2.h b > 0. in
  cover "boxes apart" (Option.is_none i);
  cover "boxes that overlap" (Option.fold ~none:false ~some:solid i);
  cover "solid boxes that touch"
    (solid a && solid b && Option.fold ~none:false ~some:(Fun.negate solid) i);
  equal (option box2) i (Box2.inter a b)

let grow_moves_sides (d, b) =
  assume (2. *. d >= -.Box2.w b && 2. *. d >= -.Box2.h b);
  cover "a side shrunk to nothing" (2. *. d = -.Box2.w b || 2. *. d = -.Box2.h b);
  equal box2
    (Box2.of_pts
       (P2.v (Box2.minx b -. d) (Box2.miny b -. d))
       (P2.v (Box2.maxx b +. d) (Box2.maxy b +. d)))
    (Box2.grow d b)

let invalid_grows =
  [
    ("shrinking a side past the other", -0.75, Box2.v 0. 0. 1. 2.);
    ("a NaN distance", Float.nan, Box2.v 0. 0. 1. 1.);
    ("an infinite distance", infinity, Box2.v 0. 0. 1. 1.);
    ( "a corner beyond max_float",
      Float.max_float,
      Box2.v Float.max_float 0. 0. 0. );
  ]

let corners_differ =
  [
    ("minx", Box2.v (-1.) 0. 2. 1.);
    ("miny", Box2.v 0. (-1.) 1. 2.);
    ("maxx", Box2.v 0. 0. 2. 1.);
    ("maxy", Box2.v 0. 0. 1. 2.);
  ]

let box_lexicographic () =
  let ordered =
    [
      Box2.v 0. 0. 1. 1.;
      Box2.v 0. 0. 1. 2.;
      Box2.v 0. 0. 2. 0.;
      Box2.v 0. 1. 0. 0.;
      Box2.v 1. 0. 0. 0.;
    ]
  in
  equal (list box2) ordered (List.sort Box2.compare (List.rev ordered))

let box_tests =
  group "Box2"
    [
      prop "v spans from its corner by its size"
        (Gen.pair (Gen.pair coord coord) (Gen.pair size size))
        (fun ((x, y), (w, h)) ->
          equal (list float_exact)
            [ x; y; x +. w; y +. h ]
            (edges (Box2.v x y w h)));
      test "v accepts a zero size" (fun () ->
          equal (list float_exact) [ 1.; 2.; 1.; 2. ]
            (edges (Box2.v 1. 2. 0. 0.)));
      cases "v raises on" ~id:fst invalid_boxes (fun (_, (x, y, w, h)) ->
          raises_match Exn.invalid_arg (fun () -> Box2.v x y w h));
      prop "of_pts is the smallest box containing both points"
        (Gen.pair gen_p2 gen_p2) (fun (p, q) ->
          equal (list float_exact)
            [
              Float.min (P2.x p) (P2.x q);
              Float.min (P2.y p) (P2.y q);
              Float.max (P2.x p) (P2.x q);
              Float.max (P2.y p) (P2.y q);
            ]
            (edges (Box2.of_pts p q)));
      cases "of_pts raises on a corner with" ~id:fst
        [ ("NaN", P2.v Float.nan 0.); ("infinity", P2.v 0. neg_infinity) ]
        (fun (_, p) ->
          raises_match Exn.invalid_arg (fun () -> Box2.of_pts p (P2.v 1. 1.)));
      prop "w and h are maxx - minx and maxy - miny" ~examples:[ huge ] gen_box
        (fun b ->
          cover "an infinite size" (Box2.w b = infinity);
          equal
            (pair float_exact float_exact)
            (Box2.maxx b -. Box2.minx b, Box2.maxy b -. Box2.miny b)
            (Box2.w b, Box2.h b));
      (* The centre rounded once: halving a finite sum is exact unless it is
         subnormal, where the division rounds once; halving a sum that overflows
         is done on its terms, each halved exactly. *)
      prop "mid is the centre"
        ~examples:
          [
            Box2.v (Float.succ Float.min_float) 0. 0. 0.;
            Box2.v (3. *. least_subnormal) 0. 0. 0.;
          ]
        gen_box
        (fun b ->
          let centre a b =
            let s = a +. b in
            if Float.is_finite s then s /. 2. else (a /. 2.) +. (b /. 2.)
          in
          equal p2
            (P2.v
               (centre (Box2.minx b) (Box2.maxx b))
               (centre (Box2.miny b) (Box2.maxy b)))
            (Box2.mid b));
      prop "mid of a box of zero size is its corner"
        ~examples:[ (Float.succ Float.min_float, 3. *. least_subnormal) ]
        (Gen.pair Gen.float Gen.float)
        (fun (x, y) -> equal p2 (P2.v x y) (Box2.mid (Box2.v x y 0. 0.)));
      cases "mid is finite on" ~id:fst
        [
          ("the largest box", (huge, P2.v 0. 0.));
          ( "a box at max_float",
            ( Box2.of_pts
                (P2.v Float.max_float Float.max_float)
                (P2.v Float.max_float Float.max_float),
              P2.v Float.max_float Float.max_float ) );
        ]
        (fun (_, (b, centre)) -> equal p2 centre (Box2.mid b));
      prop "union is the smallest box containing both"
        (Gen.pair gen_box gen_box) (fun (a, b) ->
          equal box2
            (Box2.of_pts
               (P2.v
                  (Float.min (Box2.minx a) (Box2.minx b))
                  (Float.min (Box2.miny a) (Box2.miny b)))
               (P2.v
                  (Float.max (Box2.maxx a) (Box2.maxx b))
                  (Float.max (Box2.maxy a) (Box2.maxy b))))
            (Box2.union a b));
      prop "inter is the box of the points in both"
        ~examples:
          [
            (Box2.v 0. 0. 1. 1., Box2.v 1. 0. 1. 1.);
            (Box2.v 0. 0. 1. 1., Box2.v 1. 1. 1. 1.);
            (Box2.v 0. 0. 2. 2., Box2.v 1. 1. 2. 2.);
            (Box2.v 0. 0. 1. 1., Box2.v 2. 2. 1. 1.);
          ]
        (Gen.pair gen_grid_box gen_grid_box)
        inter_is_common;
      prop "grow moves each side by the distance"
        ~examples:[ (-1., Box2.v 0. 0. 2. 4.) ]
        (Gen.pair (Gen.map Float.of_int (Gen.int_range (-1) 3)) gen_grid_box)
        grow_moves_sides;
      cases "grow raises on"
        ~id:(fun (n, _, _) -> n)
        invalid_grows
        (fun (_, d, b) ->
          raises_match Exn.invalid_arg (fun () -> Box2.grow d b));
      prop "transform is the smallest box containing the corners' images"
        (Gen.pair gen_affine gen_box)
        transform_is_corner_hull;
      (* The box of the square of side 2 turned by an eighth turn has the
         square's diagonal as its side; cos and sin of pi/4 round within an
         ulp. *)
      test "transform under a rotation is larger than the box" (fun () ->
          let b =
            Box2.transform
              (Affine.rotate (Float.pi /. 4.))
              (Box2.v (-1.) (-1.) 2. 2.)
          in
          let side = 2. *. Float.sqrt 2. in
          equal
            (pair (float 1e-15) (float 1e-15))
            (side, side)
            (Box2.w b, Box2.h b));
      test "transform raises if an image is not finite" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Box2.transform (Affine.scale 1e308 1.) (Box2.v 0. 0. 10. 10.)));
      test "equal compares corners with Float.equal" (fun () ->
          equal box2 (Box2.v 0. 0. 1. 1.) (Box2.v (-0.) (-0.) 1. 1.));
      cases "equal tells boxes apart by" ~id:fst corners_differ (fun (_, b) ->
          not_equal box2 (Box2.v 0. 0. 1. 1.) b);
      prop "equal is an equivalence" (Gen.pair gen_box gen_box)
        (Law.equivalence box2);
      prop "compare is a total order compatible with equal"
        (Gen.triple gen_box gen_box gen_box)
        (Law.order box2);
      test "compare is lexicographic by minx, miny, maxx and maxy"
        box_lexicographic;
    ]

let () = exit (run "hugin.gg geometry" [ affine_tests; p2_tests; box_tests ])
