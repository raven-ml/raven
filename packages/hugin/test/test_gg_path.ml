(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_gg

let pp_float ppf x = Format.fprintf ppf "%.17g" x
let pt x y = P2.v x y
let float_eq = Testable.make ~pp:pp_float ~equal:Float.equal

(* What [Path.fold] visits *)

type seg =
  | M of float * float
  | L of float * float
  | C of float * float * float * float * float * float
  | Z

let pp_seg ppf = function
  | M (x, y) -> Format.fprintf ppf "M %a %a" pp_float x pp_float y
  | L (x, y) -> Format.fprintf ppf "L %a %a" pp_float x pp_float y
  | C (a, b, c, d, e, f) ->
      Format.fprintf ppf "C %a %a %a %a %a %a" pp_float a pp_float b pp_float c
        pp_float d pp_float e pp_float f
  | Z -> Format.pp_print_string ppf "Z"

let seg_equal close a b =
  match (a, b) with
  | M (x, y), M (x', y') | L (x, y), L (x', y') -> close x x' && close y y'
  | C (a, b, c, d, e, f), C (a', b', c', d', e', f') ->
      close a a' && close b b' && close c c' && close d d' && close e e'
      && close f f'
  | Z, Z -> true
  | (M _ | L _ | C _ | Z), _ -> false

let near a b =
  Float.equal a b
  || Float.abs (a -. b) <= 1e-9 *. (1. +. Float.abs a +. Float.abs b)

let segs_exact = list (Testable.make ~pp:pp_seg ~equal:(seg_equal Float.equal))
let segs_near = list (Testable.make ~pp:pp_seg ~equal:(seg_equal near))

let segs p =
  Path.fold
    ~move:(fun acc x y -> M (x, y) :: acc)
    ~line:(fun acc x y -> L (x, y) :: acc)
    ~cubic:(fun acc a b c d e f -> C (a, b, c, d, e, f) :: acc)
    ~close:(fun acc -> Z :: acc)
    [] p
  |> List.rev

let flat ?tolerance m p =
  Path.flatten ?tolerance m
    ~move:(fun acc x y -> M (x, y) :: acc)
    ~line:(fun acc x y -> L (x, y) :: acc)
    ~close:(fun acc -> Z :: acc)
    [] p
  |> List.rev

let path_t = Testable.make ~pp:Path.pp ~equal:Path.equal

(* Programs: paths described by the calls that build them *)

type cmd =
  | Move of float * float
  | Line of float * float
  | Quad of float * float * float * float
  | Cubic of float * float * float * float * float * float
  | Arc of float * float * float * float * float
  | Close
  | Poly of bool * (float * float) list

let pp_cmd ppf = function
  | Move (x, y) -> Format.fprintf ppf "move_to %a %a" pp_float x pp_float y
  | Line (x, y) -> Format.fprintf ppf "line_to %a %a" pp_float x pp_float y
  | Quad (a, b, c, d) ->
      Format.fprintf ppf "quad_to %a %a %a %a" pp_float a pp_float b pp_float c
        pp_float d
  | Cubic (a, b, c, d, e, f) ->
      Format.fprintf ppf "cubic_to %a %a %a %a %a %a" pp_float a pp_float b
        pp_float c pp_float d pp_float e pp_float f
  | Arc (x, y, r, s, w) ->
      Format.fprintf ppf "arc %a %a %a ~start:%a ~sweep:%a" pp_float x pp_float
        y pp_float r pp_float s pp_float w
  | Close -> Format.pp_print_string ppf "close"
  | Poly (closed, pts) ->
      Format.fprintf ppf "append (%s %a)"
        (if closed then "polygon" else "polyline")
        (Format.pp_print_list
           ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
           (fun ppf (x, y) -> Format.fprintf ppf "%a,%a" pp_float x pp_float y))
        pts

let pp_program =
  Format.pp_print_list ~pp_sep:(fun ppf () -> Format.fprintf ppf " |>@ ") pp_cmd

let step p = function
  | Move (x, y) -> Path.move_to (pt x y) p
  | Line (x, y) -> Path.line_to (pt x y) p
  | Quad (a, b, c, d) -> Path.quad_to (pt a b) (pt c d) p
  | Cubic (a, b, c, d, e, f) -> Path.cubic_to (pt a b) (pt c d) (pt e f) p
  | Arc (x, y, r, start, sweep) -> Path.arc (pt x y) r ~start ~sweep p
  | Close -> Path.close p
  | Poly (closed, pts) ->
      let xs = Array.of_list (List.map fst pts) in
      let ys = Array.of_list (List.map snd pts) in
      Path.append ((if closed then Path.polygon else Path.polyline) xs ys) p

let build program = List.fold_left step Path.empty program

let gen_program_of coord =
  let p2 = Gen.pair coord coord in
  let cmd =
    Gen.frequency
      [
        (2, Gen.map (fun (x, y) -> Move (x, y)) p2);
        (4, Gen.map (fun (x, y) -> Line (x, y)) p2);
        (2, Gen.map (fun ((a, b), (c, d)) -> Quad (a, b, c, d)) (Gen.pair p2 p2));
        ( 3,
          Gen.map
            (fun ((a, b), (c, d), (e, f)) -> Cubic (a, b, c, d, e, f))
            (Gen.triple p2 p2 p2) );
        ( 1,
          Gen.map
            (fun ((x, y), r, (start, sweep)) -> Arc (x, y, r, start, sweep))
            (Gen.triple
               (Gen.pair
                  (Gen.float_range (-50.) 50.)
                  (Gen.float_range (-50.) 50.))
               (Gen.float_range 0. 50.)
               (Gen.pair (Gen.float_range (-7.) 7.) (Gen.float_range (-7.) 7.)))
        );
        (1, Gen.constant Close);
        ( 1,
          Gen.map
            (fun (closed, pts) -> Poly (closed, pts))
            (Gen.pair Gen.bool (Gen.list ~size:(Gen.int_range 0 6) p2)) );
      ]
  in
  Gen.with_pp pp_program (Gen.list ~size:(Gen.int_range 0 12) cmd)

let gen_program = gen_program_of (Gen.float_range (-100.) 100.)

(* Coordinates that are sometimes non-finite, for the gap rule. *)
let gen_gappy_program =
  gen_program_of
    (Gen.frequency
       [
         (8, Gen.float_range (-100.) 100.);
         (1, Gen.of_list ~pp:pp_float [ Float.nan; infinity; neg_infinity ]);
       ])

let gen_affine =
  let c = Gen.float_range (-3.) 3. in
  Gen.map
    (fun ((xx, yx, xy), (yy, x0, y0)) -> { Affine.xx; yx; xy; yy; x0; y0 })
    (Gen.pair (Gen.triple c c c) (Gen.triple c c c))

let map_seg (m : Affine.t) s =
  let tx x y = (m.xx *. x) +. (m.xy *. y) +. m.x0
  and ty x y = (m.yx *. x) +. (m.yy *. y) +. m.y0 in
  match s with
  | M (x, y) -> M (tx x y, ty x y)
  | L (x, y) -> L (tx x y, ty x y)
  | C (a, b, c, d, e, f) -> C (tx a b, ty a b, tx c d, ty c d, tx e f, ty e f)
  | Z -> Z

let pp_box ppf b =
  Format.fprintf ppf "[%a, %a; %a, %a]" pp_float (Box2.minx b) pp_float
    (Box2.miny b) pp_float (Box2.maxx b) pp_float (Box2.maxy b)

let box_near eps =
  Testable.make ~pp:pp_box ~equal:(fun a b ->
      let close x y = Float.abs (x -. y) <= eps in
      close (Box2.minx a) (Box2.minx b)
      && close (Box2.miny a) (Box2.miny b)
      && close (Box2.maxx a) (Box2.maxx b)
      && close (Box2.maxy a) (Box2.maxy b))

(* Building *)

let starts_without_current_point () =
  let q = pt 1. 2. in
  equal segs_exact [ M (1., 2.) ] (segs (Path.line_to q Path.empty));
  equal segs_exact [ M (1., 2.) ] (segs (Path.cubic_to q q q Path.empty));
  equal segs_exact [ M (1., 2.) ] (segs (Path.quad_to q q Path.empty));
  let closed = Path.polygon [| 0.; 1.; 1. |] [| 0.; 0.; 1. |] in
  equal ~msg:"after a polygon" segs_exact
    [ M (0., 0.); L (1., 0.); L (1., 1.); Z; M (1., 2.) ]
    (segs (Path.line_to q closed))

let quad_elevation () =
  let p =
    Path.empty |> Path.move_to (pt 0. 0.) |> Path.quad_to (pt 3. 6.) (pt 6. 0.)
  in
  equal segs_exact [ M (0., 0.); C (2., 4., 4., 4., 6., 0.) ] (segs p)

let quadratic t a c b =
  let u = 1. -. t in
  (u *. u *. a) +. (2. *. u *. t *. c) +. (t *. t *. b)

let cubic t a b c d =
  let u = 1. -. t in
  (u *. u *. u *. a)
  +. (3. *. u *. u *. t *. b)
  +. (3. *. u *. t *. t *. c)
  +. (t *. t *. t *. d)

let quad_is_same_curve ((x0, y0), (cx, cy), (x, y), t) =
  match
    segs
      (Path.empty |> Path.move_to (pt x0 y0) |> Path.quad_to (pt cx cy) (pt x y))
  with
  | [ M _; C (a, b, c, d, e, f) ] ->
      equal (float 1e-9) (quadratic t x0 cx x) (cubic t x0 a c e);
      equal (float 1e-9) (quadratic t y0 cy y) (cubic t y0 b d f)
  | s -> failf "expected one cubic, got %a" (Format.pp_print_list pp_seg) s

let quad_near_max_float () =
  let p =
    Path.empty
    |> Path.move_to (pt 1e308 0.)
    |> Path.quad_to (pt (-1e308) 0.) (pt 0. 0.)
  in
  equal segs_near
    [ M (1e308, 0.); C (-1e308 /. 3., 0., -2. /. 3. *. 1e308, 0., 0., 0.) ]
    (segs p)

let quad_control_on_end_point ((x0, y0), (x, y)) =
  let from_start c =
    segs (Path.empty |> Path.move_to (pt x0 y0) |> Path.quad_to c (pt x y))
  in
  (match from_start (pt x0 y0) with
  | [ M _; C (a, b, _, _, _, _) ] ->
      equal ~msg:"control point on the start" (pair float_eq float_eq) (x0, y0)
        (a, b)
  | s -> failf "expected one cubic, got %a" (Format.pp_print_list pp_seg) s);
  match from_start (pt x y) with
  | [ M _; C (_, _, c, d, _, _) ] ->
      equal ~msg:"control point on the end" (pair float_eq float_eq) (x, y)
        (c, d)
  | s -> failf "expected one cubic, got %a" (Format.pp_print_list pp_seg) s

let quad_after_polyline () =
  let p =
    Path.polyline [| 0.; 0. |] [| 5.; 0. |]
    |> Path.quad_to (pt 3. 6.) (pt 6. 0.)
  in
  equal segs_exact
    [ M (0., 5.); L (0., 0.); C (2., 4., 4., 4., 6., 0.) ]
    (segs p)

let quad_after_gap () =
  let p =
    Path.empty
    |> Path.move_to (pt 0. 0.)
    |> Path.line_to (pt Float.nan 0.)
    |> Path.quad_to (pt 3. 6.) (pt 6. 0.)
    |> Path.line_to (pt 7. 0.)
  in
  equal segs_exact [ M (0., 0.); M (6., 0.); L (7., 0.) ] (segs p)

let cubics p = List.filter (function C _ -> true | _ -> false) (segs p)

let arc_counts =
  let q = Float.pi /. 2. in
  [
    (0., 0);
    (q, 1);
    (q +. 1e-9, 2);
    (2. *. q, 2);
    (-2. *. q, 2);
    (4. *. q, 4);
    ((4. *. q) +. 0.5, 5);
  ]

let arc_start_and_join () =
  let c = pt 10. 20. in
  let a = Path.arc c 5. ~start:0. ~sweep:Float.pi Path.empty in
  (match segs a with
  | M (x, y) :: _ -> equal (pair (float 1e-12) (float 1e-12)) (15., 20.) (x, y)
  | s -> failf "arc starts with %a" (Format.pp_print_list pp_seg) s);
  let w =
    Path.empty |> Path.move_to c |> Path.arc c 5. ~start:0. ~sweep:Float.pi
  in
  match segs w with
  | M (10., 20.) :: L (x, y) :: rest ->
      equal (pair (float 1e-12) (float 1e-12)) (15., 20.) (x, y);
      equal int 2 (List.length rest)
  | s -> failf "wedge is %a" (Format.pp_print_list pp_seg) s

(* Every cubic of an arc, sampled, stays within 0.03% of the radius. *)
let arc_within_radius ((cx, cy), r, (start, sweep)) =
  let p = Path.arc (pt cx cy) r ~start ~sweep Path.empty in
  let _ =
    Path.fold
      ~move:(fun _ x y -> (x, y))
      ~line:(fun _ x y -> (x, y))
      ~cubic:(fun (x0, y0) c1x c1y c2x c2y x y ->
        for i = 0 to 16 do
          let t = Float.of_int i /. 16. in
          let d =
            Float.hypot (cubic t x0 c1x c2x x -. cx) (cubic t y0 c1y c2y y -. cy)
          in
          at_most
            ~msg:(Printf.sprintf "t = %g" t)
            float_exact
            ~than:((3e-4 *. r) +. 1e-9)
            (Float.abs (d -. r))
        done;
        (x, y))
      ~close:Fun.id (0., 0.) p
  in
  ()

let arc_ends_at_angle ((cx, cy), r, (start, sweep)) =
  match List.rev (segs (Path.arc (pt cx cy) r ~start ~sweep Path.empty)) with
  | (C (_, _, _, _, x, y) | M (x, y)) :: _ ->
      let a = start +. sweep in
      equal
        (pair (float 1e-9) (float 1e-9))
        (cx +. (r *. Float.cos a), cy +. (r *. Float.sin a))
        (x, y)
  | s -> failf "arc ends with %a" (Format.pp_print_list pp_seg) s

let invalid_arcs =
  [
    ("a negative radius", (-1., 0., 1.));
    ("a NaN radius", (Float.nan, 0., 1.));
    ("an infinite radius", (infinity, 0., 1.));
    ("an infinite start", (1., infinity, 1.));
    ("a NaN sweep", (1., 0., Float.nan));
    ("a sweep beyond a thousand turns", (1., 0., 2001. *. Float.pi));
  ]

let close_cases =
  let open_line =
    Path.empty |> Path.move_to (pt 0. 0.) |> Path.line_to (pt 1. 0.)
  in
  [
    ("the empty path", Path.empty);
    ("a closed subpath", Path.close open_line);
    ("a subpath without segment", Path.move_to (pt 1. 1.) open_line);
    ("a polygon", Path.polygon [| 0.; 1. |] [| 0.; 1. |]);
  ]

let close_closes () =
  let p =
    Path.empty
    |> Path.move_to (pt 0. 0.)
    |> Path.line_to (pt 1. 0.)
    |> Path.close
  in
  equal segs_exact [ M (0., 0.); L (1., 0.); Z ] (segs p);
  equal ~msg:"no current point after close" segs_exact
    [ M (0., 0.); L (1., 0.); Z; M (2., 2.) ]
    (segs (Path.line_to (pt 2. 2.) p))

let building =
  group "building"
    [
      test "empty has no subpaths" (fun () ->
          is_true (Path.is_empty Path.empty);
          equal segs_exact [] (segs Path.empty));
      test "move_to starts a subpath" (fun () ->
          let p = Path.move_to (pt 1. 2.) Path.empty in
          is_false (Path.is_empty p);
          equal segs_exact [ M (1., 2.) ] (segs p));
      test "a segment without a current point starts a subpath"
        starts_without_current_point;
      test "quad_to elevates the quadratic exactly" quad_elevation;
      prop "quad_to's cubic is the quadratic's curve"
        (Gen.quad
           (Gen.pair
              (Gen.float_range (-100.) 100.)
              (Gen.float_range (-100.) 100.))
           (Gen.pair
              (Gen.float_range (-100.) 100.)
              (Gen.float_range (-100.) 100.))
           (Gen.pair
              (Gen.float_range (-100.) 100.)
              (Gen.float_range (-100.) 100.))
           (Gen.float_range 0. 1.))
        quad_is_same_curve;
      test "quad_to elevates a quadratic spanning nearly max_float"
        quad_near_max_float;
      prop "quad_to keeps a control point on the end point it coincides with"
        ~examples:[ ((0.21, 0.23), (5., 0.)); ((5., 0.), (0.42, 0.45)) ]
        (Gen.pair
           (Gen.pair
              (Gen.float_range (-1000.) 1000.)
              (Gen.float_range (-1000.) 1000.))
           (Gen.pair
              (Gen.float_range (-1000.) 1000.)
              (Gen.float_range (-1000.) 1000.)))
        quad_control_on_end_point;
      test "quad_to continues a polyline from its last point"
        quad_after_polyline;
      test "quad_to after a gap starts a subpath at its end point"
        quad_after_gap;
      test "arc starts a subpath or joins the current point with a line"
        arc_start_and_join;
      cases
        ~name:(fun (s, n) -> Printf.sprintf "a sweep of %g takes %d cubics" s n)
        "arc uses one cubic per quarter turn or less" arc_counts
        (fun (sweep, n) ->
          equal int n
            (List.length
               (cubics (Path.arc (pt 0. 0.) 1. ~start:0.3 ~sweep Path.empty))));
      prop "arc stays within 0.03% of the radius"
        (Gen.triple
           (Gen.pair (Gen.float_range (-50.) 50.) (Gen.float_range (-50.) 50.))
           (Gen.float_range 0. 100.)
           (Gen.pair (Gen.float_range (-7.) 7.) (Gen.float_range (-13.) 13.)))
        arc_within_radius;
      prop "arc ends at the angle start + sweep"
        (Gen.triple
           (Gen.pair (Gen.float_range (-50.) 50.) (Gen.float_range (-50.) 50.))
           (Gen.float_range 0. 100.)
           (Gen.pair (Gen.float_range (-7.) 7.) (Gen.float_range (-13.) 13.)))
        arc_ends_at_angle;
      test "arc accepts a thousand turns" (fun () ->
          is_false
            (Path.is_empty
               (Path.arc (pt 0. 0.) 1. ~start:0. ~sweep:(-2000. *. Float.pi)
                  Path.empty)));
      cases ~name:fst "arc raises on" invalid_arcs
        (fun (_, (r, start, sweep)) ->
          raises_match Exn.invalid_arg (fun () ->
              Path.arc (pt 0. 0.) r ~start ~sweep Path.empty));
      test "close closes the last subpath and drops the current point"
        close_closes;
      cases ~name:fst "close is the identity on" close_cases (fun (_, p) ->
          equal segs_exact (segs p) (segs (Path.close p)));
      prop "close is idempotent" gen_program (fun prog ->
          Law.idempotent path_t Path.close (build prog));
      prop "append q p draws p, then q"
        (Gen.pair gen_gappy_program gen_gappy_program) (fun (a, b) ->
          equal segs_exact
            (segs (build a) @ segs (build b))
            (segs (Path.append (build b) (build a))));
      prop "append is associative"
        (Gen.triple gen_program gen_program gen_program) (fun (a, b, c) ->
          Law.associative path_t Path.append (build a, build b, build c));
      prop "append has empty as neutral" gen_program (fun a ->
          Law.neutral path_t Path.append Path.empty (build a));
      prop "transform maps every point" (Gen.pair gen_affine gen_program)
        (fun (m, prog) ->
          equal segs_near
            (List.map (map_seg m) (segs (build prog)))
            (segs (Path.transform m (build prog))));
    ]

(* Shapes *)

let polyline_copies () =
  let xs = [| 0.; 1.; 2. |] and ys = [| 0.; 1.; 0. |] in
  let p = Path.polyline xs ys in
  xs.(1) <- 10.;
  ys.(1) <- 10.;
  equal segs_exact [ M (0., 0.); L (1., 1.); L (2., 0.) ] (segs p)

let shapes =
  group "shapes"
    [
      test "rect starts at the min corner and goes clockwise on screen"
        (fun () ->
          equal segs_exact
            [ M (1., 2.); L (4., 2.); L (4., 6.); L (1., 6.); Z ]
            (segs (Path.rect (Box2.v 1. 2. 3. 4.))));
      test "circle is arc's full turn, closed" (fun () ->
          let c = pt 3. 4. in
          equal path_t
            (Path.close
               (Path.arc c 2. ~start:0. ~sweep:(2. *. Float.pi) Path.empty))
            (Path.circle c 2.));
      test "circle turns clockwise on screen" (fun () ->
          match segs (Path.circle (pt 0. 0.) 1.) with
          | M (1., 0.) :: C (_, _, _, _, x, y) :: _ ->
              equal (pair (float 1e-12) (float 1e-12)) (0., 1.) (x, y)
          | s -> failf "circle is %a" (Format.pp_print_list pp_seg) s);
      cases ~name:(Printf.sprintf "circle raises on radius %g") "circle raises"
        [ -1.; Float.nan; infinity ] (fun r ->
          raises_match (Exn.invalid_arg ~substring:"Path.circle") (fun () ->
              Path.circle (pt 0. 0.) r));
      test "circle of radius 0 is a point" (fun () ->
          equal
            (option (box_near 0.))
            (Some (Box2.v 3. 4. 0. 0.))
            (Path.bounds (Path.circle (pt 3. 4.) 0.)));
      test "polyline copies its arrays" polyline_copies;
      test "polyline with fewer than two points is empty" (fun () ->
          is_true (Path.is_empty (Path.polyline [||] [||]));
          is_true (Path.is_empty (Path.polyline [| 1. |] [| 1. |]));
          is_true (Path.is_empty (Path.polygon [| 1. |] [| 1. |])));
      test "polyline raises on arrays of different lengths" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Path.polyline [| 1.; 2. |] [| 1. |]));
      test "polygon is the polyline closed" (fun () ->
          let xs = [| 0.; 1.; 1. |] and ys = [| 0.; 0.; 1. |] in
          equal path_t (Path.close (Path.polyline xs ys)) (Path.polygon xs ys);
          equal segs_exact
            [ M (0., 0.); L (1., 0.); L (1., 1.); Z ]
            (segs (Path.polygon xs ys)));
    ]

(* Gaps *)

let nan = Float.nan

let gap_cases =
  [
    ( "a segment after a gap starts a subpath at its end point",
      Path.polyline [| 0.; nan; 2.; 3. |] [| 0.; 0.; 0.; 0. |],
      [ M (0., 0.); M (2., 0.); L (3., 0.) ] );
    ( "a close closes the subpath the gap's successor started",
      Path.polygon
        [| 0.; 1.; 2.; nan; 4.; 5.; 6. |]
        [| 0.; 1.; 0.; 0.; 0.; 1.; 0. |],
      [
        M (0., 0.);
        L (1., 1.);
        L (2., 0.);
        M (4., 0.);
        L (5., 1.);
        L (6., 0.);
        Z;
      ] );
    ( "a close after a gap without successor does nothing",
      Path.polygon [| 0.; 1.; nan |] [| 0.; 1.; 0. |],
      [ M (0., 0.); L (1., 1.) ] );
    ( "a close of a subpath without segment does nothing",
      Path.polygon [| 0.; 1.; nan; 4. |] [| 0.; 1.; 0.; 0. |],
      [ M (0., 0.); L (1., 1.); M (4., 0.) ] );
    ( "a non-finite control point is a gap",
      Path.empty
      |> Path.move_to (pt 0. 0.)
      |> Path.cubic_to (pt infinity 0.) (pt 1. 1.) (pt 2. 0.)
      |> Path.line_to (pt 3. 0.)
      |> Path.line_to (pt 4. 0.),
      [ M (0., 0.); M (3., 0.); L (4., 0.) ] );
    ( "a non-finite subpath start is a gap",
      Path.empty
      |> Path.move_to (pt nan 0.)
      |> Path.line_to (pt 1. 0.)
      |> Path.line_to (pt 2. 0.),
      [ M (1., 0.); L (2., 0.) ] );
    ( "a path of gaps draws nothing",
      Path.polyline [| nan; infinity |] [| 0.; 0. |],
      [] );
  ]

let all_finite s =
  let f = Float.is_finite in
  match s with
  | M (x, y) | L (x, y) -> f x && f y
  | C (a, b, c, d, e, g) -> f a && f b && f c && f d && f e && f g
  | Z -> true

let gaps =
  group "gaps"
    [
      cases
        ~name:(fun (n, _, _) -> n)
        "fold applies the gap rule" gap_cases
        (fun (_, p, expected) -> equal segs_exact expected (segs p));
      prop "fold gives only finite numbers" gen_gappy_program (fun prog ->
          List.iter
            (fun s ->
              satisfies ~claim:"finite segment"
                (Testable.make ~pp:pp_seg ~equal:( = ))
                all_finite s)
            (segs (build prog)));
      prop "flatten gives only finite numbers, whatever the map"
        (Gen.pair gen_gappy_program
           (Gen.of_list ~pp:Affine.pp
              [
                Affine.id; Affine.scale 1e308 1e308; { Affine.id with x0 = nan };
              ]))
        (fun (prog, m) ->
          List.iter
            (fun s ->
              satisfies ~claim:"finite segment"
                (Testable.make ~pp:pp_seg ~equal:( = ))
                all_finite s)
            (flat m (build prog)));
    ]

(* Flattening *)

(* [chords_within tol] checks Wang's guarantee on a single cubic: the curve and
   its chords, both at uniform parameters, stay within [tol]. *)
let chords_within ((x0, y0), (x1, y1), (x2, y2), (x3, y3), tol) =
  let p =
    Path.empty
    |> Path.move_to (pt x0 y0)
    |> Path.cubic_to (pt x1 y1) (pt x2 y2) (pt x3 y3)
  in
  let pts =
    List.filter_map
      (function M (x, y) | L (x, y) -> Some (x, y) | _ -> None)
      (flat ~tolerance:tol Affine.id p)
    |> Array.of_list
  in
  let n = Array.length pts - 1 in
  greater int ~than:0 n;
  equal (pair float_eq float_eq) (x3, y3) pts.(n);
  for i = 0 to n - 1 do
    let t = (Float.of_int i +. 0.5) /. Float.of_int n in
    let ax, ay = pts.(i) and bx, by = pts.(i + 1) in
    let d =
      Float.hypot
        (cubic t x0 x1 x2 x3 -. ((ax +. bx) /. 2.))
        (cubic t y0 y1 y2 y3 -. ((ay +. by) /. 2.))
    in
    at_most
      ~msg:(Printf.sprintf "chord %d of %d" i n)
      float_exact ~than:(tol +. 1e-9) d
  done

let lines_only =
  Gen.with_pp pp_program
    (Gen.list ~size:(Gen.int_range 0 10)
       (Gen.frequency
          [
            ( 1,
              Gen.map
                (fun (x, y) -> Move (x, y))
                (Gen.pair (Gen.float_range (-9.) 9.) (Gen.float_range (-9.) 9.))
            );
            ( 4,
              Gen.map
                (fun (x, y) -> Line (x, y))
                (Gen.pair (Gen.float_range (-9.) 9.) (Gen.float_range (-9.) 9.))
            );
            (1, Gen.constant Close);
          ]))

let huge_cubic () =
  let p =
    Path.empty
    |> Path.move_to (pt 0. 0.)
    |> Path.cubic_to (pt 0. 1e12) (pt 1e12 1e12) (pt 1e12 0.)
  in
  equal int 65537 (List.length (flat ~tolerance:0.1 Affine.id p))

let flattening =
  group "flatten"
    [
      prop "flatten of lines is fold of the mapped path"
        (Gen.pair gen_affine lines_only) (fun (m, prog) ->
          equal segs_exact
            (segs (Path.transform m (build prog)))
            (flat m (build prog)));
      prop "flatten under a map is flatten of the mapped path"
        (Gen.pair gen_affine gen_gappy_program) (fun (m, prog) ->
          equal segs_exact
            (flat Affine.id (Path.transform m (build prog)))
            (flat m (build prog)));
      prop "flatten keeps chords within the tolerance"
        (Gen.quad
           (Gen.pair
              (Gen.float_range (-100.) 100.)
              (Gen.float_range (-100.) 100.))
           (Gen.pair
              (Gen.float_range (-100.) 100.)
              (Gen.float_range (-100.) 100.))
           (Gen.pair
              (Gen.float_range (-100.) 100.)
              (Gen.float_range (-100.) 100.))
           (Gen.pair
              (Gen.pair
                 (Gen.float_range (-100.) 100.)
                 (Gen.float_range (-100.) 100.))
              (Gen.float_range 0.01 2.)))
        (fun (a, b, c, (d, tol)) -> chords_within (a, b, c, d, tol));
      test "flatten with an infinite tolerance draws one chord per cubic"
        (fun () ->
          let p = Path.circle (pt 0. 0.) 10. in
          equal int 6 (List.length (flat ~tolerance:infinity Affine.id p)));
      cases ~name:(Printf.sprintf "tolerance %g")
        "flatten raises on" [ 0.; -1.; Float.nan ] (fun tolerance ->
          raises_match Exn.invalid_arg (fun () ->
              flat ~tolerance Affine.id (Path.circle (pt 0. 0.) 1.)));
      test "flatten cuts a cubic into at most 65536 chords" huge_cubic;
      test
        "flatten draws one chord for an overflowing cubic at infinite tolerance"
        (fun () ->
          let m = Float.max_float in
          let p =
            Path.empty
            |> Path.move_to (pt 0. 0.)
            |> Path.cubic_to (pt m m) (pt (-.m) (-.m)) (pt 0. 0.)
          in
          equal int 2 (List.length (flat ~tolerance:infinity Affine.id p)));
    ]

(* Bounds *)

let flat_box ?tolerance p =
  List.fold_left
    (fun acc s ->
      match s with
      | M (x, y) | L (x, y) -> (
          let q = Box2.of_pts (pt x y) (pt x y) in
          match acc with None -> Some q | Some b -> Some (Box2.union b q))
      | C _ | Z -> acc)
    None
    (flat ?tolerance Affine.id p)

let contains_points prog =
  let p = build prog in
  match Path.bounds p with
  | None -> equal segs_exact [] (flat Affine.id p)
  | Some b ->
      let slack = 1e-9 *. (1. +. Box2.w b +. Box2.h b) in
      List.iter
        (function
          | M (x, y) | L (x, y) ->
              at_least float_exact ~than:(Box2.minx b -. slack) x;
              at_most float_exact ~than:(Box2.maxx b +. slack) x;
              at_least float_exact ~than:(Box2.miny b -. slack) y;
              at_most float_exact ~than:(Box2.maxy b +. slack) y
          | C _ | Z -> ())
        (flat ~tolerance:0.01 Affine.id p)

let bounds_tight prog =
  let p = build prog in
  equal (option (box_near 1e-3)) (Path.bounds p) (flat_box ~tolerance:1e-4 p)

let bounds =
  group "bounds"
    [
      test "bounds of a path that draws nothing is None" (fun () ->
          is_none ~pp:pp_box (Path.bounds Path.empty);
          is_none ~pp:pp_box
            (Path.bounds (Path.polyline [| nan; nan |] [| 0.; 1. |])));
      test "bounds of a lone subpath start is its point" (fun () ->
          equal
            (option (box_near 0.))
            (Some (Box2.v 1. 2. 0. 0.))
            (Path.bounds (Path.move_to (pt 1. 2.) Path.empty)));
      test "bounds of a cubic is exact, not its control points" (fun () ->
          let p =
            Path.empty
            |> Path.move_to (pt 0. 0.)
            |> Path.cubic_to (pt 0. 10.) (pt 10. 10.) (pt 10. 0.)
          in
          equal
            (option (box_near 1e-12))
            (Some (Box2.v 0. 0. 10. 7.5))
            (Path.bounds p));
      test "bounds of a circle is its square" (fun () ->
          equal
            (option (box_near 1e-12))
            (Some (Box2.v 1. 2. 6. 6.))
            (Path.bounds (Path.circle (pt 4. 5.) 3.)));
      test "bounds ignore gaps" (fun () ->
          equal
            (option (box_near 0.))
            (Some (Box2.v 0. 0. 3. 1.))
            (Path.bounds (Path.polyline [| 0.; 1e9; 3. |] [| 0.; nan; 1. |])));
      prop "bounds contain every flattened point" gen_gappy_program
        contains_points;
      prop "bounds are those of a fine flattening" gen_program bounds_tight;
    ]

(* Cropping *)

let crop_box = Box2.v (-25.) (-15.) 55. 50.

let inside_box ~margin b (x, y) =
  Box2.minx b +. margin < x
  && x < Box2.maxx b -. margin
  && Box2.miny b +. margin < y
  && y < Box2.maxy b -. margin

(* [polygons segs] is the closed polygons of the flattened segments [segs], each
   subpath closed as a fill closes it. *)
let polygons segs =
  let close start pts acc =
    match (start, pts) with
    | Some s, _ :: _ -> List.rev (s :: pts) :: acc
    | _ -> acc
  in
  let rec go start pts acc = function
    | [] -> List.rev (close start pts acc)
    | M (x, y) :: rest -> go (Some (x, y)) [ (x, y) ] (close start pts acc) rest
    | L (x, y) :: rest -> go start ((x, y) :: pts) acc rest
    | Z :: rest -> go None [] (close start pts acc) rest
    | C _ :: rest -> go start pts acc rest
  in
  go None [] [] segs

(* [winding polys q] is the winding number of [polys] around [q]. *)
let winding polys (px, py) =
  let w = ref 0 in
  let edge (x0, y0) (x1, y1) =
    let side = ((x1 -. x0) *. (py -. y0)) -. ((px -. x0) *. (y1 -. y0)) in
    if y0 <= py then (if y1 > py && side > 0. then incr w)
    else if y1 <= py && side < 0. then decr w
  in
  List.iter
    (fun pts ->
      let rec loop = function
        | a :: (b :: _ as rest) ->
            edge a b;
            loop rest
        | [ _ ] | [] -> ()
      in
      loop pts)
    polys;
  !w

(* [area polys] is the signed area of [polys] by the shoelace formula. *)
let area polys =
  let a = ref 0. in
  List.iter
    (fun pts ->
      let rec loop = function
        | (x0, y0) :: ((x1, y1) :: _ as rest) ->
            a := !a +. ((x0 *. y1) -. (x1 *. y0));
            loop rest
        | [ _ ] | [] -> ()
      in
      loop pts)
    polys;
  Float.abs (!a /. 2.)

let fine p = flat ~tolerance:1e-6 Affine.id p
let gen_coord r = Gen.float_range (-.r) r
let gen_point r = Gen.pair (gen_coord r) (gen_coord r)

let gen_shapes ~closed =
  Gen.with_pp
    (Format.pp_print_list (fun ppf pts -> pp_cmd ppf (Poly (closed, pts))))
    (Gen.list ~size:(Gen.int_range 1 3)
       (Gen.list ~size:(Gen.int_range 2 7) (gen_point 60.)))

let shapes_path ~closed subs =
  build (List.map (fun pts -> Poly (closed, pts)) subs)

(* The region law: inside the box, a cropped polygon winds around every point as
   the polygon does, and outside it around none. *)
let crop_keeps_windings (subs, q) =
  let p = shapes_path ~closed:true subs in
  let w = winding (polygons (fine (Path.crop crop_box p))) q in
  let inside = inside_box ~margin:0. crop_box q in
  cover "a point inside the box" inside;
  cover "a point the polygons wind around" (winding (polygons (segs p)) q <> 0);
  equal int (if inside then winding (polygons (segs p)) q else 0) w

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

let edges segs =
  let rec go cur acc = function
    | [] -> List.rev acc
    | M (x, y) :: rest -> go (x, y) acc rest
    | L (x, y) :: rest -> go (x, y) ((cur, (x, y)) :: acc) rest
    | (C _ | Z) :: rest -> go cur acc rest
  in
  go (0., 0.) [] segs

let near_edges es q = List.exists (fun e -> dist_to_segment q e <= 1e-9) es

(* The line law: a cropped polyline draws the points of the polyline within the
   box, and no other. *)
let crop_keeps_lines subs =
  let p = shapes_path ~closed:false subs in
  let original = edges (segs p)
  and cropped = edges (segs (Path.crop crop_box p)) in
  let mid ((x0, y0), (x1, y1)) = ((x0 +. x1) /. 2., (y0 +. y1) /. 2.) in
  List.iter
    (fun e ->
      let q = mid e in
      equal bool ~msg:"within the box" true
        (inside_box ~margin:(-1e-9) crop_box q);
      equal bool ~msg:"on the polyline" true (near_edges original q))
    cropped;
  List.iter
    (fun ((x0, y0), (x1, y1)) ->
      for k = 0 to 8 do
        let t = Float.of_int k /. 8. in
        let q = (x0 +. (t *. (x1 -. x0)), y0 +. (t *. (y1 -. y0))) in
        if inside_box ~margin:1e-6 crop_box q then
          equal bool ~msg:"kept" true (near_edges cropped q)
      done)
    original;
  cover "a polyline crossing an edge"
    (List.length cropped > 0 && List.length cropped <> List.length original)

let unit_box = Box2.v 0. 0. 1. 1.

(* Shapes on a grid of fives around [crop_box], whose edges lie on it: they
   leave and enter the box at its edges and corners, and run along them. *)
let gen_grid_shapes =
  let coord = Gen.map (fun k -> 5. *. Float.of_int k) (Gen.int_range (-7) 8) in
  let pts = Gen.list ~size:(Gen.int_range 2 6) (Gen.pair coord coord) in
  Gen.with_pp
    (Format.pp_print_list (fun ppf (closed, pts) ->
         pp_cmd ppf (Poly (closed, pts))))
    (Gen.list ~size:(Gen.int_range 1 3) (Gen.pair Gen.bool pts))

(* [degenerate segs] is the segments of [segs] that draw nothing: lines to the
   point they start from, and lines back to the start just before a close. *)
let degenerate segs =
  let rec go start cur acc = function
    | [] -> List.rev acc
    | M (x, y) :: rest -> go (x, y) (x, y) acc rest
    | (L (x, y) as s) :: Z :: rest when (x, y) = start ->
        go start (x, y) (s :: acc) (Z :: rest)
    | (L (x, y) as s) :: rest when (x, y) = cur -> go start cur (s :: acc) rest
    | L (x, y) :: rest -> go start (x, y) acc rest
    | C (_, _, _, _, x, y) :: rest -> go start (x, y) acc rest
    | Z :: rest -> go start start acc rest
  in
  go (0., 0.) (0., 0.) [] segs

let crop_adds_nothing_degenerate shapes =
  let p = build (List.map (fun (closed, pts) -> Poly (closed, pts)) shapes) in
  assume (degenerate (segs p) = []);
  let cropped = segs (Path.crop crop_box p) in
  cover "a shape cut" (cropped <> segs p);
  equal (list (Testable.make ~pp:pp_seg ~equal:( = ))) [] (degenerate cropped)

let cropping =
  group "crop"
    [
      prop "crop is the path itself when it lies within the box" gen_program
        (fun prog ->
          let p = build prog in
          equal path_t p (Path.crop (Box2.v (-200.) (-200.) 400. 400.) p));
      prop "crop keeps the windings of polygons within the box"
        (Gen.pair (gen_shapes ~closed:true) (gen_point 70.))
        crop_keeps_windings;
      prop "crop keeps the points of polylines within the box"
        (gen_shapes ~closed:false) crop_keeps_lines;
      prop "crop adds no segment that draws nothing" gen_grid_shapes
        crop_adds_nothing_degenerate;
      test "crop of a closed subpath of no area within the box drops it"
        (fun () ->
          equal segs_exact []
            (segs
               (Path.crop unit_box
                  (Path.polygon [| -1.; 0.5; -1. |] [| 0.5; 0.5; 0.5 |]))));
      test "crop of a closed subpath along an edge drops it" (fun () ->
          equal segs_exact []
            (segs
               (Path.crop unit_box
                  (Path.polygon [| 1.; 2.; 2.; 1. |] [| 0.; 0.; 1.; 1. |]))));
      prop "crop draws within the box, finite points only" gen_gappy_program
        (fun prog ->
          List.iter
            (function
              | (M (x, y) | L (x, y)) as s ->
                  satisfies ~claim:"finite segment"
                    (Testable.make ~pp:pp_seg ~equal:( = ))
                    all_finite s;
                  equal bool ~msg:"within the box" true
                    (inside_box ~margin:(-1e-9) crop_box (x, y))
              | C _ | Z -> ())
            (fine (Path.crop crop_box (build prog))));
      test "crop puts a cut on the edge exactly" (fun () ->
          equal segs_exact
            [ M (0., 0.); L (1., 1. /. 3.) ]
            (segs
               (Path.crop unit_box (Path.polyline [| 0.; 3. |] [| 0.; 1. |]))));
      test "crop cuts a curve into the cubic of its part within the box"
        (fun () ->
          (* x = t and y = 9 t (1 - t), cut at t = 1/2. *)
          let p =
            Path.empty
            |> Path.move_to (pt 0. 0.)
            |> Path.cubic_to (pt (1. /. 3.) 3.) (pt (2. /. 3.) 3.) (pt 1. 0.)
          in
          equal segs_near
            [ M (0., 0.); C (1. /. 6., 1.5, 1. /. 3., 2.25, 0.5, 2.25) ]
            (segs (Path.crop (Box2.v 0. 0. 0.5 3.) p)));
      test "crop keeps the pieces of a curve that leaves and comes back"
        (fun () ->
          let p =
            Path.empty
            |> Path.move_to (pt 0. 0.)
            |> Path.cubic_to (pt 0. 4.) (pt 1. (-4.)) (pt 1. 0.)
          in
          match segs (Path.crop (Box2.v 0. (-0.5) 1. 1.) p) with
          | [ M _; C _; M (_, y0); C _; M (_, y1); C _ ] ->
              equal (list float_eq) [ 0.5; -0.5 ] [ y0; y1 ]
          | s -> failf "%d segments" (List.length s));
      test "crop of a closed subpath stays closed along the edges" (fun () ->
          let p = Path.rect (Box2.v (-1.) (-1.) 2. 2.) in
          let s = segs (Path.crop (Box2.v 0. 0. 2. 2.) p) in
          equal
            (option (Testable.make ~pp:pp_seg ~equal:( = )))
            (Some Z)
            (List.nth_opt s (List.length s - 1));
          equal (float 1e-12) 1. (area (polygons s)));
      (* The circle's cubics depart from it, so its share of the disc they bound
         is the reference. *)
      cases ~name:fst "crop of a circle fills its part within the box"
        [
          ("on an edge, half its disc", (Box2.v 0. (-5.) 5. 10., `Share 0.5));
          ("at a corner, a quarter", (Box2.v 0. 0. 5. 5., `Share 0.25));
          ("around the box, the box", (Box2.v (-0.5) (-0.5) 1. 1., `Area 1.));
        ]
        (fun (_, (b, expected)) ->
          let disc = Path.circle (pt 0. 0.) 1. in
          let expected =
            match expected with
            | `Share k -> k *. area (polygons (fine disc))
            | `Area a -> a
          in
          equal (float 1e-9) expected
            (area (polygons (fine (Path.crop b disc)))));
      test "crop of a ring around the box fills nothing" (fun () ->
          let outer = Path.rect (Box2.v (-4.) (-4.) 8. 8.) in
          let inner =
            Path.polygon [| -2.; -2.; 2.; 2. |] [| -2.; 2.; 2.; -2. |]
          in
          let ring = Path.append inner outer in
          equal (float 1e-12) 0.
            (area (polygons (segs (Path.crop unit_box ring)))));
      test "crop of a path outside the box is empty" (fun () ->
          equal segs_exact []
            (segs
               (Path.crop unit_box
                  (Path.polyline [| 2.; 3.; 3. |] [| 0.; 0.; 5. |]))));
    ]

(* Equality and printing *)

let gen_polys =
  let coord =
    Gen.frequency
      [
        (6, Gen.float_range (-9.) 9.);
        (1, Gen.of_list ~pp:pp_float [ 0.; -0.; nan; infinity ]);
      ]
  in
  Gen.list ~size:(Gen.int_range 0 4)
    (Gen.pair Gen.bool
       (Gen.list ~size:(Gen.int_range 2 6) (Gen.pair coord coord)))

let as_polylines subs =
  List.fold_left
    (fun p (closed, pts) ->
      let xs = Array.of_list (List.map fst pts)
      and ys = Array.of_list (List.map snd pts) in
      Path.append ((if closed then Path.polygon else Path.polyline) xs ys) p)
    Path.empty subs

let as_segments subs =
  List.fold_left
    (fun p (closed, pts) ->
      let p =
        match pts with
        | [] -> p
        | (x, y) :: rest ->
            List.fold_left
              (fun p (x, y) -> Path.line_to (pt x y) p)
              (Path.move_to (pt x y) p)
              rest
      in
      if closed then Path.close p else p)
    Path.empty subs

let equality =
  group "equal and pp"
    [
      prop "a polyline equals its points built segment by segment" gen_polys
        (fun subs -> equal path_t (as_segments subs) (as_polylines subs));
      prop "equal is an equivalence" (Gen.pair gen_polys gen_polys)
        (fun (a, b) -> Law.equivalence path_t (as_polylines a, as_polylines b));
      test "equal compares structure, not shapes" (fun () ->
          not_equal path_t
            (Path.rect (Box2.v 0. 0. 1. 1.))
            (Path.polygon [| 1.; 1.; 0.; 0. |] [| 0.; 1.; 1.; 0. |]));
      test "equal compares what follows equal curves" (fun () ->
          let c = Path.circle (pt 0. 0.) 1. in
          not_equal path_t c (Path.move_to (pt 5. 5.) c));
      test "equal compares non-finite points" (fun () ->
          equal path_t
            (Path.polyline [| nan; 1. |] [| 0.; 1. |])
            (Path.polyline [| nan; 1. |] [| 0.; 1. |]);
          not_equal path_t
            (Path.polyline [| nan; 1. |] [| 0.; 1. |])
            (Path.polyline [| 2.; 1. |] [| 0.; 1. |]));
      test "pp writes the built segments as SVG path data" (fun () ->
          let p =
            Path.empty
            |> Path.move_to (pt 1. 2.)
            |> Path.line_to (pt 3. 4.5)
            |> Path.cubic_to (pt 5. 6.) (pt 7. 8.) (pt 9. nan)
            |> Path.close
          in
          equal string "M 1 2 L 3 4.5 C 5 6 7 8 9 nan Z"
            (Format.asprintf "%a" Path.pp p));
      test "pp writes a polyline and a polygon point by point" (fun () ->
          let p =
            Path.polygon [| 0.; 1.; 1. |] [| 0.; 0.; 1. |]
            |> Path.append (Path.polyline [| 2.; 3. |] [| 2.; nan |])
          in
          equal string "M 0 0 L 1 0 L 1 1 Z M 2 2 L 3 nan"
            (Format.asprintf "%a" Path.pp p));
    ]

let () =
  exit
    (run "hugin.gg path"
       [ building; shapes; gaps; flattening; bounds; cropping; equality ])
