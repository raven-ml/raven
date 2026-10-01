(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_next_gg
open Hugin_next_gg_kit

let pp_ring ppf r =
  Format.fprintf ppf "@[<1>[";
  for i = 0 to Ring2.length r - 1 do
    if i > 0 then Format.fprintf ppf ";@ ";
    Format.fprintf ppf "(%.17g, %.17g)" (Ring2.x r i) (Ring2.y r i)
  done;
  Format.fprintf ppf "]@]"

let pp_pgon ppf p =
  Format.fprintf ppf "@[<1>{%a}@]"
    (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_ring)
    (Pgon2.rings p)

let ring = Testable.make ~pp:pp_ring ~equal:Ring2.equal
let pgon = Testable.make ~pp:pp_pgon ~equal:Pgon2.equal

let pp_box ppf b =
  Format.fprintf ppf "[%.17g, %.17g; %.17g, %.17g]" (Box2.minx b) (Box2.miny b)
    (Box2.maxx b) (Box2.maxy b)

let box2 = Testable.make ~pp:pp_box ~equal:Box2.equal
let path = Testable.make ~pp:Path.pp ~equal:Path.equal

let of_list pts =
  Ring2.v (Array.of_list (List.map fst pts)) (Array.of_list (List.map snd pts))

let square x0 y0 s =
  Ring2.v [| x0; x0 +. s; x0 +. s; x0 |] [| y0; y0; y0 +. s; y0 +. s |]

let donut = Pgon2.v [ square 0. 0. 4.; Ring2.reverse (square 1. 1. 2.) ]

(* Generators *)

let coord =
  Gen.frequency
    [
      (3, Gen.map Float.of_int (Gen.int_range (-3) 3));
      (1, Gen.float_range (-3.5) 3.5);
    ]

let gen_ring =
  Gen.with_pp pp_ring
    (Gen.map of_list
       (Gen.list ~size:(Gen.int_range 0 6) (Gen.pair coord coord)))

let gen_rings = Gen.list ~size:(Gen.int_range 0 3) gen_ring
let gen_pgon = Gen.with_pp pp_pgon (Gen.map Pgon2.v gen_rings)

let gen_pt =
  Gen.with_pp P2.pp (Gen.map (fun (x, y) -> P2.v x y) (Gen.pair coord coord))

(* Tests *)

let donut_cases () =
  equal float_exact 12. (Pgon2.area donut);
  is_true ~msg:"on the ring" (Pgon2.mem (P2.v 0.5 0.5) donut);
  is_false ~msg:"in the hole" (Pgon2.mem (P2.v 2. 2.) donut);
  is_false ~msg:"outside" (Pgon2.mem (P2.v 5. 5.) donut);
  is_false ~msg:"not finite" (Pgon2.mem (P2.v neg_infinity 1.) donut);
  equal (option box2) (Some (Box2.v 0. 0. 4. 4.)) (Pgon2.bounds donut)

(* An inner ring wound like the outer one is no hole: the rings wind twice
   around its points, which the nonzero rule keeps in the surface. *)
let same_orientation_is_no_hole () =
  let p = Pgon2.v [ square 0. 0. 4.; square 1. 1. 2. ] in
  is_true (Pgon2.mem (P2.v 2. 2.) p);
  equal float_exact 20. (Pgon2.area p)

let empty_polygon () =
  let p = Pgon2.v [] in
  equal float_exact 0. (Pgon2.area p);
  is_false (Pgon2.mem (P2.v 0. 0.) p);
  is_none (Pgon2.bounds p);
  is_none (Pgon2.bounds (Pgon2.v [ Ring2.v [||] [||] ]));
  equal path Path.empty (Pgon2.to_path p)

let bounds_skip_empty_rings () =
  let p =
    Pgon2.v [ Ring2.v [||] [||]; square 1. 2. 3.; of_list [ (-1., 5.) ] ]
  in
  equal (option box2) (Some (Box2.v (-1.) 2. 5. 3.)) (Pgon2.bounds p)

let area_is_sum rs =
  equal float_exact
    (List.fold_left (fun a r -> a +. Ring2.area r) 0. rs)
    (Pgon2.area (Pgon2.v rs))

let one_ring_agrees (r, pt) =
  equal bool (Ring2.mem pt r) (Pgon2.mem pt (Pgon2.v [ r ]))

let windings_add (r, pt) =
  is_false ~msg:"a ring and its reverse cancel"
    (Pgon2.mem pt (Pgon2.v [ r; Ring2.reverse r ]));
  equal ~msg:"a ring twice" bool (Ring2.mem pt r)
    (Pgon2.mem pt (Pgon2.v [ r; r ]))

let to_path_in_order () =
  let a = square 0. 0. 1. and b = of_list [ (5., 5.); (6., 5.); (6., 7.) ] in
  let expected =
    Path.empty |> Path.append (Ring2.to_path a) |> Path.append (Ring2.to_path b)
  in
  equal path expected (Pgon2.to_path (Pgon2.v [ a; b ]))

let to_string p = Format.asprintf "%a" Pgon2.pp p

let tests =
  [
    group "rings"
      [
        prop "rings returns the rings given, in order" gen_rings (fun rs ->
            equal (list ring) rs (Pgon2.rings (Pgon2.v rs)));
      ];
    group "measures"
      [
        test "a square with a hole" donut_cases;
        test "an inner ring wound alike is no hole" same_orientation_is_no_hole;
        test "the polygon of no ring" empty_polygon;
        test "bounds skip rings of no point" bounds_skip_empty_rings;
        prop "area is the sum of the rings' areas" gen_rings area_is_sum;
        prop "a polygon of one ring holds what the ring holds"
          (Gen.pair gen_ring gen_pt) one_ring_agrees;
        prop "winding numbers of the rings add up" (Gen.pair gen_ring gen_pt)
          windings_add;
      ];
    group "converting"
      [ test "to_path appends the rings' paths in order" to_path_in_order ];
    group "comparing and formatting"
      [
        prop "equal is an equivalence"
          (Gen.pair gen_pgon gen_pgon)
          (Law.equivalence pgon);
        test "equal depends on the order of the rings" (fun () ->
            let a = square 0. 0. 1. and b = square 2. 2. 1. in
            is_false (Pgon2.equal (Pgon2.v [ a; b ]) (Pgon2.v [ b; a ])));
        test "pp writes the rings separated by spaces" (fun () ->
            equal string "M 0 0 L 1 0 L 1 1 Z M 5 5 L 6 5 L 6 6 Z"
              (to_string
                 (Pgon2.v
                    [
                      of_list [ (0., 0.); (1., 0.); (1., 1.) ];
                      of_list [ (5., 5.); (6., 5.); (6., 6.) ];
                    ])));
      ];
  ]

let () = exit (run "Pgon2" tests)
