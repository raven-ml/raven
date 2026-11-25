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

let pp_rings ppf rs =
  Format.fprintf ppf "@[<1>{%a}@]"
    (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_ring)
    rs

let ring = Testable.make ~pp:pp_ring ~equal:Ring2.equal

let pgon =
  Testable.make
    ~pp:(fun ppf p -> pp_rings ppf (Pgon2.rings p))
    ~equal:Pgon2.equal

let pp_box ppf b =
  Format.fprintf ppf "[%.17g, %.17g; %.17g, %.17g]" (Box2.minx b) (Box2.miny b)
    (Box2.maxx b) (Box2.maxy b)

let box2 = Testable.make ~pp:pp_box ~equal:Box2.equal
let path = Testable.make ~pp:Path.pp ~equal:Path.equal

let of_list pts =
  Ring2.v (Array.of_list (List.map fst pts)) (Array.of_list (List.map snd pts))

(* The square of side [s] from [(x0, y0)], positively oriented. *)
let square x0 y0 s =
  Ring2.v [| x0; x0 +. s; x0 +. s; x0 |] [| y0; y0; y0 +. s; y0 +. s |]

let no_point = Ring2.v [||] [||]

(* A square of side 4 with a hole of side 2: a surface of area 16 - 4. *)
let donut = [ square 0. 0. 4.; Ring2.reverse (square 1. 1. 2.) ]

(* Generators *)

let int_coord = Gen.map Float.of_int (Gen.int_range (-3) 3)
let coord = Gen.frequency [ (3, int_coord); (1, Gen.float_range (-3.5) 3.5) ]

let gen_rings_of c =
  Gen.with_pp pp_rings
    (Gen.list ~size:(Gen.int_range 0 3)
       (Gen.map of_list (Gen.list ~size:(Gen.int_range 0 6) (Gen.pair c c))))

let gen_rings = gen_rings_of coord

(* Integer coordinates make every ring's area, and their sum in any order,
   exact. *)
let gen_int_rings = gen_rings_of int_coord

(* The box from the least and greatest coordinates of the rings, [None] for no
   point. *)
let smallest_box rs =
  let pts =
    List.concat_map
      (fun r ->
        List.init (Ring2.length r) (fun i -> (Ring2.x r i, Ring2.y r i)))
      rs
  in
  match pts with
  | [] -> None
  | (x, y) :: pts ->
      let minx, miny, maxx, maxy =
        List.fold_left
          (fun (x0, y0, x1, y1) (x, y) ->
            (Float.min x0 x, Float.min y0 y, Float.max x1 x, Float.max y1 y))
          (x, y, x, y) pts
      in
      Some (Box2.of_pts (P2.v minx miny) (P2.v maxx maxy))

(* The rings' paths, each appended after the previous ones. *)
let paths rs =
  List.fold_left (fun p r -> p |> Path.append (Ring2.to_path r)) Path.empty rs

(* Tests *)

let to_string p = Format.asprintf "%a" Pgon2.pp p

let tests =
  [
    group "rings"
      [
        prop "are the rings given to v, in order" gen_rings (fun rs ->
            equal (list ring) rs (Pgon2.rings (Pgon2.v rs)));
      ];
    group "area"
      [
        prop "is the sum of the rings' areas"
          ~examples:[ []; donut; [ square 0. 0. 4.; square 1. 1. 2. ] ]
          gen_int_rings
          (fun rs ->
            equal float_exact
              (List.fold_left (fun a r -> a +. Ring2.area r) 0. rs)
              (Pgon2.area (Pgon2.v rs)));
        test "is the surface's area when rings wind 0 or 1 times" (fun () ->
            equal float_exact 12. (Pgon2.area (Pgon2.v donut)));
      ];
    group "bounds"
      [
        prop "is the smallest box containing the rings' points"
          ~examples:
            [
              [];
              [ no_point ];
              donut;
              [ no_point; square 1. 2. 3.; of_list [ (-1., 5.) ] ];
            ]
          gen_rings
          (fun rs ->
            equal (option box2) (smallest_box rs) (Pgon2.bounds (Pgon2.v rs)));
      ];
    group "to_path"
      [
        prop "is the rings' paths in order"
          ~examples:
            [
              []; [ square 0. 0. 1.; of_list [ (5., 5.); (6., 5.); (6., 7.) ] ];
            ]
          gen_rings
          (fun rs -> equal path (paths rs) (Pgon2.to_path (Pgon2.v rs)));
      ];
    group "equal"
      [
        prop "is an equivalence" (Gen.pair gen_rings gen_rings) (fun (a, b) ->
            Law.equivalence pgon (Pgon2.v a, Pgon2.v b));
        test "is false for the same rings in another order" (fun () ->
            let a = square 0. 0. 1. and b = square 2. 2. 1. in
            not_equal pgon (Pgon2.v [ a; b ]) (Pgon2.v [ b; a ]));
      ];
    group "pp"
      [
        test "formats the rings as Ring2.pp does, separated by spaces"
          (fun () ->
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
