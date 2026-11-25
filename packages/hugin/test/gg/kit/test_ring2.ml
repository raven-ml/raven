(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_gg
open Hugin_gg_kit

(* Witnesses print every digit, so that a failure shows the difference. *)

let pp_ring ppf r =
  Format.fprintf ppf "@[<1>[";
  for i = 0 to Ring2.length r - 1 do
    if i > 0 then Format.fprintf ppf ";@ ";
    Format.fprintf ppf "(%.17g, %.17g)" (Ring2.x r i) (Ring2.y r i)
  done;
  Format.fprintf ppf "]@]"

let ring = Testable.make ~pp:pp_ring ~equal:Ring2.equal
let point = pair float_exact float_exact

(* Areas compared by [Float.equal], [0.] equal to [-0.]. *)
let area =
  Testable.make
    ~pp:(fun ppf x -> Format.fprintf ppf "%.17g" x)
    ~equal:Float.equal

let pp_box ppf b =
  Format.fprintf ppf "[%.17g, %.17g; %.17g, %.17g]" (Box2.minx b) (Box2.miny b)
    (Box2.maxx b) (Box2.maxy b)

let box2 = Testable.make ~pp:pp_box ~equal:Box2.equal
let path = Testable.make ~pp:Path.pp ~equal:Path.equal

let of_list pts =
  Ring2.v (Array.of_list (List.map fst pts)) (Array.of_list (List.map snd pts))

let points r = List.init (Ring2.length r) (fun i -> (Ring2.x r i, Ring2.y r i))

(* The square of side [s] from [(x0, y0)], positively oriented. *)
let square x0 y0 s =
  Ring2.v [| x0; x0 +. s; x0 +. s; x0 |] [| y0; y0; y0 +. s; y0 +. s |]

(* [r] from its point [k]: the same curve. *)
let rotate k r =
  let n = Ring2.length r in
  if n = 0 then r
  else
    Ring2.v
      (Array.init n (fun i -> Ring2.x r ((i + k) mod n)))
      (Array.init n (fun i -> Ring2.y r ((i + k) mod n)))

(* Generators *)

(* Integer coordinates make every product and sum of an area exact. *)
let int_coord = Gen.map Float.of_int (Gen.int_range (-3) 3)
let coord = Gen.frequency [ (3, int_coord); (1, Gen.float_range (-3.5) 3.5) ]
let gen_pts c = Gen.list ~size:(Gen.int_range 0 8) (Gen.pair c c)
let gen_ring_of c = Gen.with_pp pp_ring (Gen.map of_list (gen_pts c))
let gen_ring = gen_ring_of coord
let gen_int_ring = gen_ring_of int_coord

(* v *)

let copies_its_arrays () =
  let xs = [| 0.; 1.; 1. |] and ys = [| 0.; 0.; 1. |] in
  let r = Ring2.v xs ys in
  xs.(0) <- 5.;
  ys.(2) <- 5.;
  equal (list point) [ (0., 0.); (1., 0.); (1., 1.) ] (points r)

let non_finite =
  [
    ("NaN x", [| 0.; Float.nan |], [| 0.; 1. |]);
    ("infinite x", [| infinity; 0. |], [| 0.; 1. |]);
    ("NaN y", [| 0.; 1. |], [| Float.nan; 1. |]);
    ("negative infinite y", [| 0.; 1. |], [| 0.; neg_infinity |]);
  ]

let constructing =
  group "v"
    [
      test "copies its arrays" copies_its_arrays;
      test "raises on arrays of different lengths" (fun () ->
          raises_match Exn.invalid_arg (fun () -> Ring2.v [| 0.; 1. |] [| 0. |]));
      cases "raises on a coordinate that is not finite"
        ~id:(fun (n, _, _) -> n)
        non_finite
        (fun (_, xs, ys) ->
          raises_match Exn.invalid_arg (fun () -> Ring2.v xs ys));
    ]

(* Accessors *)

let index_errors = [ ("-1 of 3", 3, -1); ("3 of 3", 3, 3); ("0 of 0", 0, 0) ]

let accessors =
  group "accessors"
    [
      prop "length, x and y read the points given to v" (gen_pts coord)
        (fun pts -> equal (list point) pts (points (of_list pts)));
      cases "an index outside [0; length r - 1] raises"
        ~id:(fun (n, _, _) -> n)
        index_errors
        (fun (_, n, i) ->
          let r = Ring2.v (Array.make n 0.) (Array.make n 0.) in
          raises_match ~msg:"x" Exn.invalid_arg (fun () -> Ring2.x r i);
          raises_match ~msg:"y" Exn.invalid_arg (fun () -> Ring2.y r i));
    ]

(* area *)

(* Areas from the definition, the integral of the winding number. Integer
   coordinates make the computed ones exact. *)
let known_areas =
  [
    ("positive unit square", [ (0., 0.); (1., 0.); (1., 1.); (0., 1.) ], 1.);
    ("negative unit square", [ (0., 1.); (1., 1.); (1., 0.); (0., 0.) ], -1.);
    ("right triangle of legs 4 and 3", [ (0., 0.); (4., 0.); (0., 3.) ], 6.);
    ( "bow tie, lobes of opposite signs",
      [ (0., 0.); (2., 2.); (2., 0.); (0., 2.) ],
      0. );
    ( "unit square traversed twice",
      [
        (0., 0.);
        (1., 0.);
        (1., 1.);
        (0., 1.);
        (0., 0.);
        (1., 0.);
        (1., 1.);
        (0., 1.);
      ],
      2. );
  ]

let few_points =
  [
    ("no point", []);
    ("one point", [ (1., 1.) ]);
    ("two points", [ (0., 0.); (5., 5.) ]);
  ]

let areas =
  group "area"
    [
      cases "is the signed area"
        ~id:(fun (n, _, _) -> n)
        known_areas
        (fun (_, pts, a) -> equal float_exact a (Ring2.area (of_list pts)));
      cases "is 0. for fewer than three points" ~id:fst few_points
        (fun (_, pts) -> equal float_exact 0. (Ring2.area (of_list pts)));
      (* A unit square at 2^52: n = 4 and w = h = 1, so the stated bound is 16
         eps. Products of its coordinates, near 2^104, would err by 2^52. *)
      test "rounds by less than n^2 eps w h, however far from the origin"
        (fun () ->
          equal
            (float (16. *. epsilon_float))
            1.
            (Ring2.area (square 0x1p52 0x1p52 1.)));
      test "is not finite past a span of 1e154" (fun () ->
          satisfies float_exact ~claim:"an infinity or NaN"
            (fun a -> not (Float.is_finite a))
            (Ring2.area (square 0. 0. 1e160)));
      prop "does not depend on the first point" (Gen.pair gen_int_ring Gen.nat)
        (fun (r, k) -> equal area (Ring2.area r) (Ring2.area (rotate k r)));
    ]

(* bounds *)

(* The box from the least and greatest coordinates, [None] for no point. *)
let smallest_box = function
  | [] -> None
  | (x, y) :: pts ->
      let minx, miny, maxx, maxy =
        List.fold_left
          (fun (x0, y0, x1, y1) (x, y) ->
            (Float.min x0 x, Float.min y0 y, Float.max x1 x, Float.max y1 y))
          (x, y, x, y) pts
      in
      Some (Box2.of_pts (P2.v minx miny) (P2.v maxx maxy))

let bounding =
  group "bounds"
    [
      prop "is the smallest box containing the points"
        ~examples:
          [
            [];
            [ (2., 3.) ];
            [ (0., 5.); (-1., 2.); (3., 0.) ];
            [ (0., 0.); (-0., 0.) ];
          ]
        (gen_pts coord)
        (fun pts ->
          equal (option box2) (smallest_box pts) (Ring2.bounds (of_list pts)));
    ]

(* reverse *)

let reversing =
  group "reverse"
    [
      prop "puts point i at length r - 1 - i"
        ~examples:[ [ (0., 0.); (1., 2.); (3., 4.) ] ]
        (gen_pts coord)
        (fun pts ->
          equal (list point) (List.rev pts)
            (points (Ring2.reverse (of_list pts))));
      prop "negates the signed area" gen_int_ring (fun r ->
          equal area (-.Ring2.area r) (Ring2.area (Ring2.reverse r)));
    ]

(* to_path *)

let converting =
  group "to_path"
    [
      prop "is Path.polygon of the coordinates"
        ~examples:[ []; [ (1., 1.) ] ]
        (gen_pts coord)
        (fun pts ->
          let xs = Array.of_list (List.map fst pts)
          and ys = Array.of_list (List.map snd pts) in
          equal path (Path.polygon xs ys) (Ring2.to_path (of_list pts)));
      cases "is Path.empty for fewer than two points" ~id:fst
        [ ("no point", []); ("one point", [ (1., 1.) ]) ]
        (fun (_, pts) -> equal path Path.empty (Ring2.to_path (of_list pts)));
    ]

(* equal *)

let unequal =
  [
    ("a rotation", square 0. 0. 1., rotate 1 (square 0. 0. 1.));
    ( "equal xs, other ys",
      of_list [ (0., 0.); (1., 0.) ],
      of_list [ (0., 0.); (1., 1.) ] );
    ( "one more point",
      of_list [ (0., 0.); (1., 0.) ],
      of_list [ (0., 0.); (1., 0.); (1., 1.) ] );
  ]

let comparing =
  group "equal"
    [
      prop "is an equivalence"
        (Gen.pair gen_ring gen_ring)
        (Law.equivalence ring);
      cases "is false for points that differ, or differ in order"
        ~id:(fun (n, _, _) -> n)
        unequal
        (fun (_, a, b) -> not_equal ring a b);
      test "compares coordinates with Float.equal" (fun () ->
          equal ring (of_list [ (0., 1.) ]) (of_list [ (-0., 1.) ]));
    ]

(* pp *)

(* The format the .mli states: [M x y], [L x y] for each other point, [Z]. *)
let formatted =
  [
    ("three points", [ (0., 0.); (1., 0.); (1., 1.) ], "M 0 0 L 1 0 L 1 1 Z");
    ("one point", [ (2.5, -3.) ], "M 2.5 -3 Z");
    ("no point", [], "");
  ]

let printing =
  group "pp"
    [
      cases "formats the points as SVG path data"
        ~id:(fun (n, _, _) -> n)
        formatted
        (fun (_, pts, s) ->
          equal string s (Format.asprintf "%a" Ring2.pp (of_list pts)));
    ]

let () =
  exit
    (run "Ring2"
       [
         constructing;
         accessors;
         areas;
         bounding;
         reversing;
         converting;
         comparing;
         printing;
       ])
