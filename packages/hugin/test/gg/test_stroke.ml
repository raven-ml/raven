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

let cap_name = function
  | `Butt -> "`Butt"
  | `Round -> "`Round"
  | `Square -> "`Square"

let join_name = function
  | `Miter -> "`Miter"
  | `Round -> "`Round"
  | `Bevel -> "`Bevel"

let pp_pattern ppf l =
  Format.fprintf ppf "[%a]"
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       (fun ppf x -> Format.fprintf ppf "%g" x))
    l

(* The six fields of a stroke style, as its accessors give them. *)
let fields s =
  ( Stroke.width s,
    Stroke.cap s,
    Stroke.join s,
    Stroke.miter_limit s,
    Stroke.dash s,
    Stroke.dash_offset s )

let fields_t =
  Testable.make ~equal:( = ) ~pp:(fun ppf (w, c, j, l, d, o) ->
      Format.fprintf ppf "(%g, %s, %s, %g, %a, %g)" w (cap_name c) (join_name j)
        l pp_pattern d o)

(* The length of the pattern made even. *)
let period s =
  let sum = List.fold_left ( +. ) 0. (Stroke.dash s) in
  if List.length (Stroke.dash s) mod 2 = 1 then 2. *. sum else sum

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

(* v *)

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

let gen_pattern =
  Gen.with_pp pp_pattern
    (Gen.list ~size:(Gen.int_range 1 5) (Gen.float_range 0.5 20.))

(* [Float.rem] is exact. The reduction adds the period, below 200, at most once
   to a remainder, which rounds within half an ulp of 256, and the check's own
   subtraction rounds once more: the offsets agree within 1e-13 modulo the
   period. *)
let offset_reduced (pattern, offset) =
  let s = Stroke.v ~dash:pattern ~dash_offset:offset 1. in
  let o = Stroke.dash_offset s and l = period s in
  cover "a negative offset" (offset < 0.);
  at_least float_exact ~than:0. o;
  less float_exact ~than:l o;
  satisfies ~claim:"0 or the period, within 1e-13" float_exact
    (fun d -> Float.abs d <= 1e-13 || Float.abs (d -. l) <= 1e-13)
    (o -. Float.rem offset l)

(* The same stroke given an offset one period further, or a solid line given an
   offset. *)
let respell s =
  Stroke.v ~cap:(Stroke.cap s) ~join:(Stroke.join s)
    ~miter_limit:(Stroke.miter_limit s) ~dash:(Stroke.dash s)
    ~dash_offset:(Stroke.dash_offset s +. if period s = 0. then 5. else period s)
    (Stroke.width s)

let invalid =
  [
    ("a negative width", fun () -> Stroke.v (-1.));
    ("a NaN width", fun () -> Stroke.v Float.nan);
    ("an infinite width", fun () -> Stroke.v infinity);
    ("a miter limit below 1", fun () -> Stroke.v ~miter_limit:0.5 1.);
    ("a NaN miter limit", fun () -> Stroke.v ~miter_limit:Float.nan 1.);
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
    ( "a NaN dash offset on a dashed line",
      fun () -> Stroke.v ~dash:[ 1. ] ~dash_offset:Float.nan 1. );
    ( "a NaN dash offset on a solid line",
      fun () -> Stroke.v ~dash_offset:Float.nan 1. );
  ]

(* reach *)

(* (cap, join, miter limit, reach of a stroke of width 2 by the formula) *)
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
  let+ pts = triple pt pt pt
  and+ width = float_range 0.5 8.
  and+ cap = of_list [ `Butt; `Round; `Square ]
  and+ join = of_list [ `Miter; `Round; `Bevel ]
  and+ miter_limit = float_range 1. 10.
  and+ dash = of_list [ []; [ 3.; 2. ]; [ 0.; 4. ] ] in
  (Stroke.v ~cap ~join ~miter_limit ~dash width, pts)

let gen_ink =
  Gen.with_pp
    (fun ppf (s, ((ax, ay), (bx, by), (cx, cy))) ->
      Format.fprintf ppf "%a through (%g, %g) (%g, %g) (%g, %g)" Stroke.pp s ax
        ay bx by cx cy)
    gen_ink

let dist_to_segment (px, py) ((x0, y0), (x1, y1)) =
  let dx = x1 -. x0 and dy = y1 -. y0 in
  let l = (dx *. dx) +. (dy *. dy) in
  let t =
    if l = 0. then 0.
    else
      Float.min 1.
        (Float.max 0. ((((px -. x0) *. dx) +. ((py -. y0) *. dy)) /. l))
  in
  Float.hypot (px -. (x0 +. (t *. dx))) (py -. (y0 +. (t *. dy)))

(* A pixel with ink holds a painted point, within [reach] of the path. Its
   centre lies within [sqrt 2 /. 2.] of that point, and the renderer flattens
   round pieces within a tenth of a pixel (hugin_vg_raster.mli, Accuracy). *)
let reach_bounds_ink (s, (a, b, c)) =
  let path =
    Path.polyline [| fst a; fst b; fst c |] [| snd a; snd b; snd c |]
  in
  let picture = Picture.stroke s Color.black path in
  let px =
    Nx.to_array (Raster.render ~density:1. (Renderable.v 64. 64. picture))
  in
  let bound = Stroke.reach s +. (Float.sqrt 2. /. 2.) +. 0.1 in
  cover "a miter beyond the square cap"
    (Stroke.join s = `Miter && Stroke.miter_limit s > 2.);
  for y = 0 to 63 do
    for x = 0 to 63 do
      if px.((((y * 64) + x) * 4) + 3) > 0 then
        let q = (Float.of_int x +. 0.5, Float.of_int y +. 0.5) in
        at_most
          ~msg:(Printf.sprintf "pixel (%d, %d) has ink" x y)
          float_exact ~than:bound
          (Float.min (dist_to_segment q (a, b)) (dist_to_segment q (b, c)))
    done
  done

(* equal *)

let equal_iff_fields (s, s') =
  let same = fields s = fields s' in
  cover "equal fields" same;
  if same then equal stroke s s' else not_equal stroke s s'

let stroke_tests =
  group "Stroke"
    [
      test "v defaults to round caps and joins, miter limit 4, solid" (fun () ->
          equal fields_t (2., `Round, `Round, 4., [], 0.) (fields (Stroke.v 2.)));
      test "v keeps the given cap, join, miter limit and pattern" (fun () ->
          equal fields_t
            (0., `Square, `Bevel, 1.5, [ 1.; 2.; 3. ], 1.)
            (fields
               (Stroke.v ~cap:`Square ~join:`Bevel ~miter_limit:1.5
                  ~dash:[ 1.; 2.; 3. ] ~dash_offset:1. 0.)));
      cases "v reduces the offset modulo the even pattern's length"
        ~id:(fun (p, o, _) -> Format.asprintf "%a at %g" pp_pattern p o)
        offsets
        (fun (dash, dash_offset, expected) ->
          equal float_exact expected
            (Stroke.dash_offset (Stroke.v ~dash ~dash_offset 1.)));
      prop "v reduces any finite offset into [0, period["
        (Gen.pair gen_pattern Gen.float)
        offset_reduced;
      prop "v gives equal strokes for offsets a period apart" gen_stroke
        (fun s -> equal stroke s (respell s));
      cases "v raises on" ~id:fst invalid (fun (_, f) ->
          raises_match Exn.invalid_arg f);
      cases "reach is half the width times the formula's factor"
        ~id:(fun (c, j, l, _) ->
          Printf.sprintf "%s caps, %s joins, miter limit %g" (cap_name c)
            (join_name j) l)
        reaches
        (fun (cap, join, miter_limit, r) ->
          equal float_exact r
            (Stroke.reach (Stroke.v ~cap ~join ~miter_limit 2.)));
      prop "reach bounds the rasterised ink" ~count:200 gen_ink reach_bounds_ink;
      prop "equal holds iff every field is equal"
        ~examples:[ (Stroke.v 1., Stroke.v 1.) ]
        (Gen.pair gen_stroke gen_stroke)
        equal_iff_fields;
      prop "equal is an equivalence"
        (Gen.pair gen_stroke gen_stroke)
        (Law.equivalence stroke);
      prop "compare is a total order compatible with equal"
        (Gen.triple gen_stroke gen_stroke gen_stroke)
        (Law.order stroke);
    ]

let () = exit (run "hugin.gg stroke" [ stroke_tests ])
