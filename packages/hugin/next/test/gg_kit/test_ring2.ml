(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_next_gg
open Hugin_next_gg_kit

(* Witnesses print every digit, so that a failure shows the difference. *)

let pp_ring ppf r =
  Format.fprintf ppf "@[<1>[";
  for i = 0 to Ring2.length r - 1 do
    if i > 0 then Format.fprintf ppf ";@ ";
    Format.fprintf ppf "(%.17g, %.17g)" (Ring2.x r i) (Ring2.y r i)
  done;
  Format.fprintf ppf "]@]"

let ring = Testable.make ~pp:pp_ring ~equal:Ring2.equal

(* Numbers, [0.] equal to [-0.]. *)
let number =
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

(* The unit square from the origin, positively oriented. *)
let square x0 y0 s =
  Ring2.v [| x0; x0 +. s; x0 +. s; x0 |] [| y0; y0; y0 +. s; y0 +. s |]

let rotate k r =
  let n = Ring2.length r in
  if n = 0 then r
  else
    Ring2.v
      (Array.init n (fun i -> Ring2.x r ((i + k) mod n)))
      (Array.init n (fun i -> Ring2.y r ((i + k) mod n)))

(* Generators *)

(* Integer coordinates keep the area exact and put query points on edges and
   vertices often. *)
let int_coord = Gen.map Float.of_int (Gen.int_range (-3) 3)
let coord = Gen.frequency [ (3, int_coord); (1, Gen.float_range (-3.5) 3.5) ]

let gen_ring_of c =
  Gen.with_pp pp_ring
    (Gen.map
       (fun pts -> of_list pts)
       (Gen.list ~size:(Gen.int_range 0 8) (Gen.pair c c)))

let gen_ring = gen_ring_of coord
let gen_int_ring = gen_ring_of int_coord

let gen_pt =
  Gen.with_pp P2.pp (Gen.map (fun (x, y) -> P2.v x y) (Gen.pair coord coord))

(* Constructing *)

let copies_its_arrays () =
  let xs = [| 0.; 1.; 1. |] and ys = [| 0.; 0.; 1. |] in
  let r = Ring2.v xs ys in
  xs.(0) <- 5.;
  ys.(2) <- 5.;
  equal float_exact 0. (Ring2.x r 0);
  equal float_exact 1. (Ring2.y r 2)

let rejects_length_mismatch () =
  raises_match Exn.invalid_arg (fun () -> Ring2.v [| 0.; 1. |] [| 0. |])

let non_finite =
  [
    ("NaN x", [| 0.; Float.nan |], [| 0.; 1. |]);
    ("infinite x", [| infinity; 0. |], [| 0.; 1. |]);
    ("NaN y", [| 0.; 1. |], [| Float.nan; 1. |]);
    ("negative infinite y", [| 0.; 1. |], [| 0.; neg_infinity |]);
  ]

let constructing =
  group "constructing"
    [
      test "v copies its arrays" copies_its_arrays;
      test "v rejects arrays of different lengths" rejects_length_mismatch;
      cases
        ~name:(fun (n, _, _) -> n)
        "v rejects a coordinate that is not finite" non_finite
        (fun (_, xs, ys) ->
          raises_match Exn.invalid_arg (fun () -> Ring2.v xs ys));
      test "v accepts no point" (fun () ->
          equal int 0 (Ring2.length (Ring2.v [||] [||])));
    ]

(* Accessors *)

let reads_points () =
  let r = of_list [ (1., 2.); (3., 4.); (5., 6.) ] in
  equal int 3 (Ring2.length r);
  equal float_exact 3. (Ring2.x r 1);
  equal float_exact 6. (Ring2.y r 2)

let index_errors = [ ("-1", 3, -1); ("length", 3, 3); ("0 of no point", 0, 0) ]

let accessors =
  group "accessors"
    [
      test "x and y read the points in order" reads_points;
      cases
        ~name:(fun (n, _, _) -> n)
        "an index out of range raises" index_errors
        (fun (_, n, i) ->
          let r = Ring2.v (Array.make n 0.) (Array.make n 0.) in
          raises_match Exn.invalid_arg (fun () -> Ring2.x r i);
          raises_match Exn.invalid_arg (fun () -> Ring2.y r i));
    ]

(* Area *)

let known_areas =
  [
    ("positive unit square", square 0. 0. 1., 1.);
    ("negative unit square", Ring2.reverse (square 0. 0. 1.), -1.);
    ("right triangle", of_list [ (0., 0.); (4., 0.); (0., 3.) ], 6.);
    ("no point", Ring2.v [||] [||], 0.);
    ("one point", of_list [ (1., 1.) ], 0.);
    ("two points", of_list [ (0., 0.); (5., 5.) ], 0.);
    ( "bow tie, lobes of opposite signs",
      of_list [ (0., 0.); (2., 2.); (2., 0.); (0., 2.) ],
      0. );
    ( "square traversed twice",
      of_list
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
    ("unit square at 2^52", square 4503599627370496. 4503599627370496. 1., 1.);
    ("square of side 1e160", square 0. 0. 1e160, infinity);
  ]

let area =
  group "area"
    [
      cases
        ~name:(fun (n, _, _) -> n)
        "known areas" known_areas
        (fun (_, r, a) -> equal float_exact a (Ring2.area r));
      prop "reversing negates the area" gen_int_ring (fun r ->
          equal number (-.Ring2.area r) (Ring2.area (Ring2.reverse r)));
      prop "the first point does not change the area"
        (Gen.pair gen_int_ring Gen.nat) (fun (r, k) ->
          equal number (Ring2.area r) (Ring2.area (rotate k r)));
    ]

(* Membership *)

let unit_square_points =
  [
    ("centre", (0.5, 0.5), true);
    ("right of it", (2., 0.5), false);
    ("above it", (0.5, -1.), false);
    ("top left vertex", (0., 0.), true);
    ("top right vertex", (1., 0.), false);
    ("bottom right vertex", (1., 1.), false);
    ("bottom left vertex", (0., 1.), false);
    ("top edge", (0.5, 0.), true);
    ("left edge", (0., 0.5), true);
    ("bottom edge", (0.5, 1.), false);
    ("right edge", (1., 0.5), false);
    ("NaN x", (Float.nan, 0.5), false);
    ("infinite y", (0.5, infinity), false);
    ("negative infinite x", (neg_infinity, 0.5), false);
  ]

let mem_unit_square (_, (x, y), expected) =
  let p = P2.v x y in
  equal ~msg:"positive" bool expected (Ring2.mem p (square 0. 0. 1.));
  equal ~msg:"negative" bool expected
    (Ring2.mem p (Ring2.reverse (square 0. 0. 1.)))

let mem_degenerate () =
  List.iter
    (fun r ->
      List.iter
        (fun (x, y) -> is_false ~msg:"no enclosure" (Ring2.mem (P2.v x y) r))
        [ (0., 0.); (1., 1.); (0.5, 0.5); (-1., 0.) ])
    [ Ring2.v [||] [||]; of_list [ (0., 0.) ]; of_list [ (0., 0.); (1., 1.) ] ]

let mem_winding_two () =
  let twice =
    of_list
      [
        (0., 0.);
        (2., 0.);
        (2., 2.);
        (0., 2.);
        (0., 0.);
        (2., 0.);
        (2., 2.);
        (0., 2.);
      ]
  in
  is_true (Ring2.mem (P2.v 1. 1.) twice)

(* Nine unit squares tile [\[0;3)] by [\[0;3)]: every point of it is in exactly
   one of them and every other point in none. *)
let tiles =
  List.concat_map
    (fun i ->
      List.map
        (fun j -> square (Float.of_int j) (Float.of_int i) 1.)
        [ 0; 1; 2 ])
    [ 0; 1; 2 ]

let gen_tile_pt =
  let c =
    Gen.frequency
      [
        (3, Gen.map (fun k -> Float.of_int k /. 2.) (Gen.int_range (-1) 7));
        (1, Gen.float_range (-0.5) 3.5);
      ]
  in
  Gen.with_pp P2.pp (Gen.map (fun (x, y) -> P2.v x y) (Gen.pair c c))

let tiling_partitions p =
  let x = P2.x p and y = P2.y p in
  let inside = 0. <= x && x < 3. && 0. <= y && y < 3. in
  cover "on a shared edge" (Float.is_integer x && 0. < x && x < 3.);
  let n = List.length (List.filter (Ring2.mem p) tiles) in
  equal int (if inside then 1 else 0) n

(* Triangles around a common centre share their edges: a point on an edge they
   share lies in at most one, whatever the rounding of that slanted edge. *)
let fan_shares_edges (k, t) =
  let n = 7 in
  let px i =
    let a = 2. *. Float.pi *. Float.of_int i /. Float.of_int n in
    (3. *. Float.cos a, 3. *. Float.sin a)
  in
  let tri i =
    let x0, y0 = px i and x1, y1 = px (i + 1) in
    of_list [ (0.1, 0.2); (x0, y0); (x1, y1) ]
  in
  let x1, y1 = px k in
  let p = P2.v (0.1 +. (t *. (x1 -. 0.1))) (0.2 +. (t *. (y1 -. 0.2))) in
  let n_in = List.length (List.filter (Ring2.mem p) (List.init n tri)) in
  at_most int ~than:1 n_in

let mem_ignores_orientation_and_start (r, p, k) =
  let m = Ring2.mem p r in
  equal ~msg:"reversed" bool m (Ring2.mem p (Ring2.reverse r));
  equal ~msg:"rotated" bool m (Ring2.mem p (rotate k r))

(* A point within rounding error of a slanted edge, on its left: the rounded
   cross product is zero, the exact orientation positive. *)
let slanted_edge_decided_exactly () =
  let p = P2.v 0.49056068382391227 0.9160279203438393 in
  let left = of_list [ (0.1, 0.2); (0.7, 1.3); (-1., 1.3) ] in
  let right = of_list [ (0.7, 1.3); (0.1, 0.2); (1.5, 0.2) ] in
  is_true ~msg:"in the triangle on the left" (Ring2.mem p left);
  is_false ~msg:"not in the triangle on the right" (Ring2.mem p right)

(* Two triangles share the edge from [l] to [u], and [p] lies within rounding
   error of it, on the side [on_right] says by the exact orientation. The
   rounded cross product is not zero and has the other sign. *)
let near_shared_edge (_, l, u, (px, py), on_right) =
  let left = of_list [ l; u; (-40., snd u) ] in
  let right = of_list [ u; l; (40., snd l) ] in
  let p = P2.v px py in
  equal ~msg:"in the triangle on the right" bool on_right (Ring2.mem p right);
  equal ~msg:"in the triangle on the left" bool (not on_right)
    (Ring2.mem p left)

let near_shared_edges =
  [
    ( "an edge between points of like magnitude",
      (-21.938145353255926, 20.846024216233957),
      (15.826477138596843, 31.793716220071403),
      (-3.228226320010573, 26.26989495761091),
      true );
    ( "an edge from a point near the origin, inexact differences",
      (-0.0005240707458162172, 8.845845059190371e-05),
      (21.098654996442377, 28.117601157885833),
      (13.201660671605467, 17.59378705550328),
      false );
  ]

let membership =
  group "membership"
    [
      cases
        ~name:(fun (n, _, _) -> n)
        "on the unit square" unit_square_points mem_unit_square;
      test "a ring of fewer than three points holds nothing" mem_degenerate;
      test "a point the ring winds around twice is in it" mem_winding_two;
      test "a point within rounding error of a slanted edge is decided exactly"
        slanted_edge_decided_exactly;
      cases
        ~name:(fun (n, _, _, _, _) -> n)
        "a point the rounded cross product misplaces is decided exactly"
        near_shared_edges near_shared_edge;
      prop "unit squares partition the plane they tile" gen_tile_pt
        tiling_partitions;
      prop "triangles sharing slanted edges hold their points at most once"
        (Gen.pair (Gen.int_range 0 6) (Gen.float_range 0. 1.))
        fan_shares_edges;
      prop "membership ignores the orientation and the first point"
        (Gen.triple gen_ring gen_pt Gen.nat)
        mem_ignores_orientation_and_start;
    ]

(* Bounds, reversing and paths *)

let bounds_cases () =
  is_none (Ring2.bounds (Ring2.v [||] [||]));
  equal (option box2)
    (Some (Box2.v 2. 3. 0. 0.))
    (Ring2.bounds (of_list [ (2., 3.) ]));
  equal (option box2)
    (Some (Box2.v (-1.) 0. 4. 5.))
    (Ring2.bounds (of_list [ (0., 5.); (-1., 2.); (3., 0.) ]))

let bounds_contain r =
  match Ring2.bounds r with
  | None -> equal int 0 (Ring2.length r)
  | Some b ->
      let ok = ref true and minx = ref infinity and maxy = ref neg_infinity in
      for i = 0 to Ring2.length r - 1 do
        let x = Ring2.x r i and y = Ring2.y r i in
        ok :=
          !ok
          && Box2.minx b <= x
          && x <= Box2.maxx b
          && Box2.miny b <= y
          && y <= Box2.maxy b;
        minx := Float.min !minx x;
        maxy := Float.max !maxy y
      done;
      is_true ~msg:"every point is in the box" !ok;
      equal number ~msg:"the box is the smallest" !minx (Box2.minx b);
      equal number ~msg:"the box is the smallest" !maxy (Box2.maxy b)

let reverse_order () =
  let r = of_list [ (0., 0.); (1., 2.); (3., 4.) ] in
  equal ring (of_list [ (3., 4.); (1., 2.); (0., 0.) ]) (Ring2.reverse r)

let to_path_is_polygon r =
  let n = Ring2.length r in
  let xs = Array.init n (Ring2.x r) and ys = Array.init n (Ring2.y r) in
  equal path (Path.polygon xs ys) (Ring2.to_path r)

let transforming =
  group "bounds, reversing and paths"
    [
      test "bounds of no point, one point and several" bounds_cases;
      prop "bounds are the smallest box holding the points"
        ~examples:[ of_list [ (0., 0.); (-0., 0.) ] ]
        gen_ring bounds_contain;
      test "reverse puts point i at length - 1 - i" reverse_order;
      prop "reverse is an involution" gen_ring
        (Law.involutive ring Ring2.reverse);
      prop "to_path is the polygon of the points" gen_ring to_path_is_polygon;
      test "to_path of fewer than two points is empty" (fun () ->
          equal path Path.empty (Ring2.to_path (of_list [ (1., 1.) ])));
    ]

(* Comparing and formatting *)

let to_string r = Format.asprintf "%a" Ring2.pp r

let comparing =
  group "comparing and formatting"
    [
      prop "equal is an equivalence"
        (Gen.pair gen_ring gen_ring)
        (Law.equivalence ring);
      test "a rotation is not equal" (fun () ->
          let r = square 0. 0. 1. in
          is_false (Ring2.equal r (rotate 1 r)));
      test "rings with equal xs and different ys are not equal" (fun () ->
          is_false
            (Ring2.equal
               (Ring2.v [| 0.; 1. |] [| 0.; 0. |])
               (Ring2.v [| 0.; 1. |] [| 0.; 1. |])));
      test "pp writes SVG path data" (fun () ->
          equal string "M 0 0 L 1 0 L 1 1 Z"
            (to_string (of_list [ (0., 0.); (1., 0.); (1., 1.) ])));
      test "pp of one point closes it" (fun () ->
          equal string "M 2.5 -3 Z" (to_string (of_list [ (2.5, -3.) ])));
      test "pp of no point is empty" (fun () ->
          equal string "" (to_string (Ring2.v [||] [||])));
    ]

let () =
  exit
    (run "Ring2"
       [ constructing; accessors; area; membership; transforming; comparing ])
