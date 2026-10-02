(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_gg
open Hugin_vg
module Raster = Hugin_vg_raster

let stroke =
  Testable.with_compare Stroke.compare
    (Testable.make ~pp:Stroke.pp ~equal:Stroke.equal)

let cap =
  Testable.make ~equal:( = ) ~pp:(fun ppf c ->
      Format.pp_print_string ppf
        (match c with
        | `Butt -> "`Butt"
        | `Round -> "`Round"
        | `Square -> "`Square"))

let join =
  Testable.make ~equal:( = ) ~pp:(fun ppf j ->
      Format.pp_print_string ppf
        (match j with
        | `Miter -> "`Miter"
        | `Round -> "`Round"
        | `Bevel -> "`Bevel"))

let defaults () =
  let s = Stroke.v 2. in
  equal float_exact 2. (Stroke.width s);
  equal cap `Round (Stroke.cap s);
  equal join `Round (Stroke.join s);
  equal float_exact 4. (Stroke.miter_limit s);
  equal (list float_exact) [] (Stroke.dash s);
  equal float_exact 0. (Stroke.dash_offset s)

let fields () =
  let s =
    Stroke.v ~cap:`Square ~join:`Bevel ~miter_limit:1.5 ~dash:[ 1.; 2.; 3. ]
      ~dash_offset:1. 0.
  in
  equal float_exact 0. (Stroke.width s);
  equal cap `Square (Stroke.cap s);
  equal join `Bevel (Stroke.join s);
  equal float_exact 1.5 (Stroke.miter_limit s);
  equal ~msg:"the pattern as given" (list float_exact) [ 1.; 2.; 3. ]
    (Stroke.dash s)

(* (pattern, offset, offset reduced into the even pattern's length) *)
let offsets =
  [
    ([ 1.; 2. ], 4., 1.);
    ([ 1.; 2. ], -1., 2.);
    ([ 1.; 2. ], 3., 0.);
    ([ 1.; 2. ], -6., 0.);
    ([ 1.; 2. ], -1e-20, 0.);
    ([ 1. ], 3., 1.);
    ([ 0.; 2. ], 0.5, 0.5);
    ([], 5., 0.);
  ]

let pp_pattern ppf l =
  Format.fprintf ppf "[%a]"
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       (fun ppf x -> Format.fprintf ppf "%g" x))
    l

let invalid =
  [
    ("a negative width", fun () -> Stroke.v (-1.));
    ("a NaN width", fun () -> Stroke.v Float.nan);
    ("an infinite width", fun () -> Stroke.v infinity);
    ("a miter limit below 1", fun () -> Stroke.v ~miter_limit:0.5 1.);
    ("an infinite miter limit", fun () -> Stroke.v ~miter_limit:infinity 1.);
    ("a negative dash", fun () -> Stroke.v ~dash:[ 1.; -1. ] 1.);
    ("a NaN dash", fun () -> Stroke.v ~dash:[ Float.nan ] 1.);
    ("an infinite dash", fun () -> Stroke.v ~dash:[ infinity; 1. ] 1.);
    ("a pattern of zeros", fun () -> Stroke.v ~dash:[ 0.; 0. ] 1.);
    ( "a pattern whose sum overflows",
      fun () -> Stroke.v ~dash:[ Float.max_float; Float.max_float ] 1. );
    ( "an odd pattern whose doubled sum overflows",
      fun () -> Stroke.v ~dash:[ 1e308 ] 1. );
    ( "an infinite dash offset",
      fun () -> Stroke.v ~dash:[ 1. ] ~dash_offset:infinity 1. );
    ( "a NaN offset on a solid line",
      fun () -> Stroke.v ~dash_offset:Float.nan 1. );
  ]

(* Strokes from small sets, so that equal values are drawn. Patterns of integers
   keep the reduction of offsets exact. *)
let gen_stroke =
  let open Gen in
  let+ width = of_list [ 0.; 1.; 2. ]
  and+ cap = of_list [ `Butt; `Round; `Square ]
  and+ join = of_list [ `Miter; `Round; `Bevel ]
  and+ miter_limit = of_list [ 1.; 4. ]
  and+ dash = of_list [ []; [ 1. ]; [ 1.; 2. ]; [ 2.; 1. ] ]
  and+ dash_offset = int_range (-6) 6 in
  Stroke.v ~cap ~join ~miter_limit ~dash ~dash_offset:(Float.of_int dash_offset)
    width

let gen_stroke = Gen.with_pp Stroke.pp gen_stroke

let period s =
  let sum = List.fold_left ( +. ) 0. (Stroke.dash s) in
  if List.length (Stroke.dash s) mod 2 = 1 then 2. *. sum else sum

(* The same stroke given an offset one period further, or a solid line given an
   offset. *)
let respell s =
  Stroke.v ~cap:(Stroke.cap s) ~join:(Stroke.join s)
    ~miter_limit:(Stroke.miter_limit s) ~dash:(Stroke.dash s)
    ~dash_offset:(Stroke.dash_offset s +. if period s = 0. then 5. else period s)
    (Stroke.width s)

let offset_in_period (pattern, offset) =
  let s = Stroke.v ~dash:pattern ~dash_offset:offset 1. in
  let o = Stroke.dash_offset s in
  at_least float_exact ~than:0. o;
  less float_exact ~than:(period s) o

let gen_pattern =
  Gen.with_pp pp_pattern
    (Gen.list ~size:(Gen.int_range 1 5) (Gen.float_range 0.5 20.))

(* (cap, join, miter limit, reach of a stroke of width 2) *)
let reaches =
  [
    (`Butt, `Round, 4., 1.);
    (`Round, `Bevel, 4., 1.);
    (`Square, `Round, 4., Float.sqrt 2.);
    (`Butt, `Miter, 4., 4.);
    (`Square, `Miter, 4., 4.);
    (`Square, `Miter, 1., Float.sqrt 2.);
  ]

(* Strokes of polylines of three points, whose corner exercises the joins,
   rendered at one pixel per point. *)
let gen_ink =
  let open Gen in
  let pt = pair (float_range 20. 44.) (float_range 20. 44.) in
  let+ a, b, c = triple pt pt pt
  and+ width = float_range 0.5 8.
  and+ cap = of_list [ `Butt; `Round; `Square ]
  and+ join = of_list [ `Miter; `Round; `Bevel ]
  and+ miter_limit = float_range 1. 10.
  and+ dash = of_list [ []; [ 3.; 2. ]; [ 0.; 4. ] ] in
  ( Stroke.v ~cap ~join ~miter_limit ~dash width,
    Path.polyline [| fst a; fst b; fst c |] [| snd a; snd b; snd c |] )

let gen_ink =
  Gen.with_pp
    (fun ppf (s, p) -> Format.fprintf ppf "%a %a" Stroke.pp s Path.pp p)
    gen_ink

(* Every pixel with ink meets the path's box grown by the reach. *)
let reach_bounds_ink (s, path) =
  let picture = Picture.stroke s Color.black path in
  let px =
    Nx.to_array (Raster.render ~density:1. (Renderable.v 64. 64. picture))
  in
  let grown = Box2.grow (Stroke.reach s) (Option.get (Path.bounds path)) in
  cover "a miter beyond the square cap"
    (Stroke.join s = `Miter && Stroke.miter_limit s > 2.);
  for y = 0 to 63 do
    for x = 0 to 63 do
      if px.((((y * 64) + x) * 4) + 3) > 0 then
        is_some
          ~msg:(Printf.sprintf "pixel (%d, %d) has ink" x y)
          (Box2.inter (Box2.v (Float.of_int x) (Float.of_int y) 1. 1.) grown)
    done
  done

let stroke_tests =
  group "Stroke"
    [
      test "v defaults to round caps and joins, miter limit 4, solid" defaults;
      test "v keeps the given cap, join, miter limit and pattern" fields;
      cases
        ~name:(fun (p, o, _) -> Format.asprintf "%a at %g" pp_pattern p o)
        "dash_offset is reduced into the even pattern" offsets
        (fun (dash, dash_offset, expected) ->
          equal float_exact expected
            (Stroke.dash_offset (Stroke.v ~dash ~dash_offset 1.)));
      prop "dash_offset lies in [0, period[ for any finite offset"
        (Gen.pair gen_pattern Gen.float)
        offset_in_period;
      cases ~name:fst "v raises on" invalid (fun (_, f) ->
          raises_match Exn.invalid_arg f);
      prop "offsets one period apart give equal strokes" gen_stroke (fun s ->
          equal stroke s (respell s));
      cases
        ~name:(fun (c, j, l, _) ->
          Format.asprintf "%a caps, %a joins, miter limit %g" (Testable.pp cap)
            c (Testable.pp join) j l)
        "reach is" reaches
        (fun (cap, join, miter_limit, r) ->
          equal (float 1e-15) r
            (Stroke.reach (Stroke.v ~cap ~join ~miter_limit 2.)));
      prop "reach bounds the rasterised ink" ~count:200 gen_ink reach_bounds_ink;
      prop "equal is an equivalence"
        (Gen.pair gen_stroke gen_stroke)
        (Law.equivalence stroke);
      prop "compare is a total order compatible with equal"
        (Gen.triple gen_stroke gen_stroke gen_stroke)
        (Law.order stroke);
    ]

let () = exit (run "hugin.gg stroke" [ stroke_tests ])
