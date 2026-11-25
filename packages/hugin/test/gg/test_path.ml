(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_gg

let pp_float ppf x = Format.fprintf ppf "%.17g" x
let pt x y = P2.v x y
let nan = Float.nan
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

(* Coordinates the tests compare within rounding are computed with a few
   roundings of numbers below 1000, or of the coordinates themselves: within
   1e-12 of each other relative to [1 + |a| + |b|]. *)
let near a b =
  Float.equal a b
  || Float.abs (a -. b) <= 1e-12 *. (1. +. Float.abs a +. Float.abs b)

let seg_t = Testable.make ~pp:pp_seg ~equal:(seg_equal Float.equal)
let segs_exact = list seg_t
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
let printed p = Format.asprintf "%a" Path.pp p
let pp_point ppf (x, y) = Format.fprintf ppf "(%a, %a)" pp_float x pp_float y
let point_t = Testable.make ~pp:pp_point ~equal:( = )

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

let quadratic t a c b =
  let u = 1. -. t in
  (u *. u *. a) +. (2. *. u *. t *. c) +. (t *. t *. b)

let cubic t a b c d =
  let u = 1. -. t in
  (u *. u *. u *. a)
  +. (3. *. u *. u *. t *. b)
  +. (3. *. u *. t *. t *. c)
  +. (t *. t *. t *. d)

(* Types *)

let is_empty_is_equal_empty prog =
  let p = build prog in
  cover "the empty path" (Path.is_empty p);
  if Path.is_empty p then equal path_t Path.empty p
  else not_equal path_t Path.empty p

let types =
  group "types"
    [
      test "empty has no subpaths" (fun () ->
          equal segs_exact [] (segs Path.empty));
      prop "is_empty p is equal p empty" ~examples:[ [] ] gen_gappy_program
        is_empty_is_equal_empty;
      test "is_empty is false for a path of gaps only" (fun () ->
          not_equal path_t Path.empty
            (Path.polyline [| nan; nan |] [| 0.; 0. |]));
    ]

(* Building *)

let p12 = pt 1. 2.
let triangle = Path.polygon [| 0.; 1.; 1. |] [| 0.; 0.; 1. |]
let open_line = Path.empty |> Path.move_to (pt 0. 0.) |> Path.line_to (pt 1. 0.)

let without_current_point =
  [
    ("line_to on empty", Path.line_to p12 Path.empty, [ M (1., 2.) ]);
    ("cubic_to on empty", Path.cubic_to p12 p12 p12 Path.empty, [ M (1., 2.) ]);
    ("quad_to on empty", Path.quad_to p12 p12 Path.empty, [ M (1., 2.) ]);
    ( "line_to after a polygon",
      Path.line_to p12 triangle,
      [ M (0., 0.); L (1., 0.); L (1., 1.); Z; M (1., 2.) ] );
    ( "line_to after close",
      Path.line_to p12 (Path.close open_line),
      [ M (0., 0.); L (1., 0.); Z; M (1., 2.) ] );
  ]

let quad_is_same_curve ((x0, y0), (cx, cy), (x, y), t) =
  match
    segs
      (Path.empty |> Path.move_to (pt x0 y0) |> Path.quad_to (pt cx cy) (pt x y))
  with
  | [ M _; C (a, b, c, d, e, f) ] ->
      (* Each side rounds a few times within an ulp of 128. *)
      equal
        (pair (float 1e-12) (float 1e-12))
        (quadratic t x0 cx x, quadratic t y0 cy y)
        (cubic t x0 a c e, cubic t y0 b d f)
  | s -> failf "expected one cubic, got %a" (Format.pp_print_list pp_seg) s

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

(* Every cubic of an arc, sampled, stays within 0.03% of the radius, plus the
   rounding of the evaluation, a few ulps of coordinates below 150. *)
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
            ~than:((3e-4 *. r) +. 1e-12)
            (Float.abs (d -. r))
        done;
        (x, y))
      ~close:Fun.id (0., 0.) p
  in
  ()

(* The end point and its reference round a few times within an ulp of 128. *)
let arc_ends_at_angle ((cx, cy), r, (start, sweep)) =
  match List.rev (segs (Path.arc (pt cx cy) r ~start ~sweep Path.empty)) with
  | (C (_, _, _, _, x, y) | M (x, y)) :: _ ->
      let a = start +. sweep in
      equal
        (pair (float 1e-12) (float 1e-12))
        (cx +. (r *. Float.cos a), cy +. (r *. Float.sin a))
        (x, y)
  | s -> failf "arc ends with %a" (Format.pp_print_list pp_seg) s

let gen_arc =
  Gen.triple
    (Gen.pair (Gen.float_range (-50.) 50.) (Gen.float_range (-50.) 50.))
    (Gen.float_range 0. 100.)
    (Gen.pair (Gen.float_range (-7.) 7.) (Gen.float_range (-13.) 13.))

let invalid_arcs =
  [
    ("a negative radius", (-1., 0., 1.));
    ("a NaN radius", (Float.nan, 0., 1.));
    ("an infinite radius", (infinity, 0., 1.));
    ("a NaN start", (1., Float.nan, 1.));
    ("an infinite start", (1., infinity, 1.));
    ("a NaN sweep", (1., 0., Float.nan));
    ("an infinite sweep", (1., 0., neg_infinity));
    ("a sweep beyond a thousand turns", (1., 0., 2001. *. Float.pi));
  ]

let close_identities =
  [
    ("the empty path", Path.empty);
    ("a closed subpath", Path.close open_line);
    ("a subpath without segment", Path.move_to (pt 1. 1.) open_line);
    ("a polygon", Path.polygon [| 0.; 1. |] [| 0.; 1. |]);
  ]

let building =
  group "building"
    [
      cases "a segment without a current point starts a subpath at its end"
        ~id:(fun (n, _, _) -> n)
        without_current_point
        (fun (_, p, expected) -> equal segs_exact expected (segs p));
      test "move_to starts a subpath" (fun () ->
          equal segs_exact [ M (1., 2.) ] (segs (Path.move_to p12 Path.empty)));
      test "line_to adds a line from the current point" (fun () ->
          equal segs_exact [ M (0., 0.); L (1., 0.) ] (segs open_line));
      (* Degree elevation: the cubic's control points lie two thirds of the way
         from each end point to the quadratic's control point. *)
      test "quad_to stores the cubic of the quadratic" (fun () ->
          equal segs_exact
            [ M (0., 0.); C (2., 4., 4., 4., 6., 0.) ]
            (segs
               (Path.empty
               |> Path.move_to (pt 0. 0.)
               |> Path.quad_to (pt 3. 6.) (pt 6. 0.))));
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
      test "quad_to stores a quadratic spanning nearly max_float" (fun () ->
          equal segs_near
            [
              M (1e308, 0.); C (-1e308 /. 3., 0., -2. /. 3. *. 1e308, 0., 0., 0.);
            ]
            (segs
               (Path.empty
               |> Path.move_to (pt 1e308 0.)
               |> Path.quad_to (pt (-1e308) 0.) (pt 0. 0.))));
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
      test "quad_to continues a polyline from its last point" (fun () ->
          equal segs_exact
            [ M (0., 5.); L (0., 0.); C (2., 4., 4., 4., 6., 0.) ]
            (segs
               (Path.polyline [| 0.; 0. |] [| 5.; 0. |]
               |> Path.quad_to (pt 3. 6.) (pt 6. 0.))));
      test "quad_to after a gap starts a subpath at its end point" (fun () ->
          equal segs_exact
            [ M (0., 0.); M (6., 0.); L (7., 0.) ]
            (segs
               (Path.empty
               |> Path.move_to (pt 0. 0.)
               |> Path.line_to (pt Float.nan 0.)
               |> Path.quad_to (pt 3. 6.) (pt 6. 0.)
               |> Path.line_to (pt 7. 0.))));
      test "cubic_to adds a cubic from the current point" (fun () ->
          equal segs_exact
            [ M (0., 0.); C (1., 2., 3., 4., 5., 6.) ]
            (segs
               (Path.empty
               |> Path.move_to (pt 0. 0.)
               |> Path.cubic_to (pt 1. 2.) (pt 3. 4.) (pt 5. 6.))));
      (* cos 0 and sin 0 are exact. *)
      test "arc without a current point starts a subpath at its start"
        (fun () ->
          match
            segs (Path.arc (pt 10. 20.) 5. ~start:0. ~sweep:1. Path.empty)
          with
          | M (x, y) :: _ -> equal point_t (15., 20.) (x, y)
          | s -> failf "arc starts with %a" (Format.pp_print_list pp_seg) s);
      test "arc joins the current point with a line to its start" (fun () ->
          let c = pt 10. 20. in
          match
            segs
              (Path.empty |> Path.move_to c
              |> Path.arc c 5. ~start:0. ~sweep:Float.pi)
          with
          | M (10., 20.) :: L (x, y) :: _ -> equal point_t (15., 20.) (x, y)
          | s -> failf "wedge is %a" (Format.pp_print_list pp_seg) s);
      cases "arc uses one cubic per quarter turn or less"
        ~id:(fun (s, n) -> Printf.sprintf "a sweep of %g takes %d cubics" s n)
        arc_counts
        (fun (sweep, n) ->
          equal int n
            (List.length
               (cubics (Path.arc (pt 0. 0.) 1. ~start:0.3 ~sweep Path.empty))));
      prop "arc stays within 0.03% of the radius" gen_arc arc_within_radius;
      prop "arc ends at the angle start + sweep" gen_arc arc_ends_at_angle;
      test "arc accepts a sweep of a thousand turns" (fun () ->
          not_equal path_t Path.empty
            (Path.arc (pt 0. 0.) 1. ~start:0. ~sweep:(-2000. *. Float.pi)
               Path.empty));
      cases "arc raises on" ~id:fst invalid_arcs (fun (_, (r, start, sweep)) ->
          raises_match Exn.invalid_arg (fun () ->
              Path.arc (pt 0. 0.) r ~start ~sweep Path.empty));
      test "close closes the last subpath with a line back to its start"
        (fun () ->
          equal segs_exact
            [ M (0., 0.); L (1., 0.); Z ]
            (segs (Path.close open_line)));
      cases "close is the identity on" ~id:fst close_identities (fun (_, p) ->
          equal path_t p (Path.close p));
      prop "close is idempotent" gen_program (fun prog ->
          Law.idempotent path_t Path.close (build prog));
      prop "append q p draws p, then q"
        (Gen.pair gen_gappy_program gen_gappy_program) (fun (a, b) ->
          equal segs_exact
            (segs (build a) @ segs (build b))
            (segs (Path.append (build b) (build a))));
      prop "append has empty as neutral" gen_gappy_program (fun a ->
          Law.neutral path_t Path.append Path.empty (build a));
      prop "transform maps every point" (Gen.pair gen_affine gen_program)
        (fun (m, prog) ->
          equal segs_near
            (List.map (map_seg m) (segs (build prog)))
            (segs (Path.transform m (build prog))));
    ]

(* Cropping *)

let crop_box = Box2.v (-25.) (-15.) 55. 50.
let unit_box = Box2.v 0. 0. 1. 1.

let inside_box ~margin b (x, y) =
  Box2.minx b +. margin < x
  && x < Box2.maxx b -. margin
  && Box2.miny b +. margin < y
  && y < Box2.maxy b -. margin

let lies_within b pb =
  Box2.minx b <= Box2.minx pb
  && Box2.maxx pb <= Box2.maxx b
  && Box2.miny b <= Box2.miny pb
  && Box2.maxy pb <= Box2.maxy b

(* Boxes around a gappy program's coordinates: one that cuts what it draws, one
   that holds all of it and one of zero width. *)
let gen_crop_box =
  Gen.of_list
    ~pp:(fun ppf (n, _) -> Format.pp_print_string ppf n)
    [
      ("crop_box", crop_box);
      ("Box2.v (-200.) (-200.) 400. 400.", Box2.v (-200.) (-200.) 400. 400.);
      ("Box2.v 0. (-200.) 0. 400.", Box2.v 0. (-200.) 0. 400.);
    ]

(* [Path.pp] prints every point a path was built with, non-finite ones as [nan],
   [inf] and [-inf], the only words it prints that hold an [n]. *)
let crop_is_finite ((_, b), prog) =
  let p = build prog in
  let has_gap = String.contains (printed p) 'n' in
  cover "a path with a gap" has_gap;
  cover "a gap, and what fold visits within the box"
    (has_gap
    && match Path.bounds p with Some pb -> lies_within b pb | None -> false);
  cover "a path of gaps only" (Path.bounds p = None && not (Path.is_empty p));
  let s = printed (Path.crop b p) in
  not_contains ~sub:"nan" s;
  not_contains ~sub:"inf" s

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

(* [area polys] is the absolute value of the signed area of [polys] by the
   shoelace formula. *)
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

let rec pairs = function
  | a :: (b :: _ as rest) -> (a, b) :: pairs rest
  | [ _ ] | [] -> []

(* [clear polys q] is [true] iff [q] lies more than [1e-9] from every edge of
   the closed polygons [polys] and of [crop_box]. *)
let clear polys q =
  let x0 = Box2.minx crop_box and y0 = Box2.miny crop_box in
  let x1 = Box2.maxx crop_box and y1 = Box2.maxy crop_box in
  let box = [ (x0, y0); (x1, y0); (x1, y1); (x0, y1); (x0, y0) ] in
  List.for_all
    (fun e -> dist_to_segment q e > 1e-9)
    (List.concat_map pairs (box :: polys))

(* The region law: inside the box, a cropped polygon winds around every point as
   the polygon does, and outside it around none. A point on an edge, of a
   polygon or of the box, or within rounding of one, is inside or outside a fill
   as its arithmetic rounds, and the oracle's winding no better: the law holds
   for points clear of every edge, which the generator's corners, such as the
   least subnormal, would otherwise draw beside a vertex. *)
let crop_keeps_windings (subs, q) =
  let p = shapes_path ~closed:true subs in
  let around = winding (polygons (segs p)) q in
  let inside = inside_box ~margin:0. crop_box q in
  cover "a point inside the box" inside;
  cover "a point the polygons wind around" (around <> 0);
  assume (clear (polygons (segs p)) q);
  let w = winding (polygons (fine (Path.crop crop_box p))) q in
  equal int (if inside then around else 0) w

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
   box, and no other. Cut points lie on the box's edges exactly and on the
   polyline within a few ulps of 60: 1e-9 covers both. *)
let crop_keeps_lines subs =
  let p = shapes_path ~closed:false subs in
  let original = edges (segs p)
  and cropped = edges (segs (Path.crop crop_box p)) in
  let mid ((x0, y0), (x1, y1)) = ((x0 +. x1) /. 2., (y0 +. y1) /. 2.) in
  cover "a polyline crossing an edge"
    (List.length cropped > 0 && List.length cropped <> List.length original);
  List.iter
    (fun e ->
      let q = mid e in
      satisfies ~claim:"within the box" point_t
        (inside_box ~margin:(-1e-9) crop_box)
        q;
      satisfies ~claim:"on the polyline" point_t (near_edges original) q)
    cropped;
  List.iter
    (fun ((x0, y0), (x1, y1)) ->
      for k = 0 to 8 do
        let t = Float.of_int k /. 8. in
        let q = (x0 +. (t *. (x1 -. x0)), y0 +. (t *. (y1 -. y0))) in
        if inside_box ~margin:1e-6 crop_box q then
          satisfies ~claim:"kept by the crop" point_t (near_edges cropped) q
      done)
    original

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
  equal segs_exact [] (degenerate cropped)

(* Two points lie on one line, however near zero their differences are, so
   whatever crop computes of them has no area. *)
let tiny = Float.min_float *. Float.epsilon

let no_area =
  [
    ( "of no area within the box",
      (unit_box, Path.polygon [| -1.; 0.5; -1. |] [| 0.5; 0.5; 0.5 |]) );
    ( "cut to two points",
      ( crop_box,
        Path.polygon
          [| 30.001059649773559; -.tiny |]
          [| 2.1822677440245566e-174; tiny |] ) );
    ( "along an edge",
      (unit_box, Path.polygon [| 1.; 2.; 2.; 1. |] [| 0.; 0.; 1.; 1. |]) );
  ]

(* The crop of a square across the box's corner ends with its close, after a
   line to a point other than its start. *)
let crop_ends_with_close () =
  let s =
    segs (Path.crop (Box2.v 0. 0. 2. 2.) (Path.rect (Box2.v (-1.) (-1.) 2. 2.)))
  in
  match (s, List.rev s) with
  | M (sx, sy) :: _, Z :: L (x, y) :: _ -> not_equal point_t (sx, sy) (x, y)
  | _ -> failf "crop is %a" (Format.pp_print_list pp_seg) s

(* The circle's cubics depart from it, so its share of the disc they bound is
   the reference. The boxes' edges meet the circle at the ends of its cubics, so
   the crop keeps whole cubics, which flatten alike: the shoelace sums, each of
   a few thousand products below 1, round within 1e-12. *)
let circle_crops =
  [
    ("on an edge, half its disc", (Box2.v 0. (-5.) 5. 10., `Share 0.5));
    ("at a corner, a quarter", (Box2.v 0. 0. 5. 5., `Share 0.25));
    ("around the box, the box", (Box2.v (-0.5) (-0.5) 1. 1., `Area 1.));
  ]

let crop_fills_circle_part (_, (b, expected)) =
  let disc = Path.circle (pt 0. 0.) 1. in
  let expected =
    match expected with
    | `Share k -> k *. area (polygons (fine disc))
    | `Area a -> a
  in
  equal (float 1e-12) expected (area (polygons (fine (Path.crop b disc))))

(* The line law under a map: a polyline cropped to the points a map takes into
   the box draws, mapped, the points of the mapped polyline within the box, and
   no other. *)
let crop_keeps_lines_under ((m : Affine.t), subs) =
  assume (Float.abs ((m.xx *. m.yy) -. (m.xy *. m.yx)) > 0.1);
  let p = shapes_path ~closed:false subs in
  let original = edges (segs (Path.transform m p))
  and cropped = edges (segs (Path.transform m (Path.crop ~m crop_box p))) in
  let mid ((x0, y0), (x1, y1)) = ((x0 +. x1) /. 2., (y0 +. y1) /. 2.) in
  cover "a polyline crossing an edge"
    (List.length cropped > 0 && List.length cropped <> List.length original);
  List.iter
    (fun e ->
      let q = mid e in
      satisfies ~claim:"within the box" point_t
        (inside_box ~margin:(-1e-9) crop_box)
        q;
      satisfies ~claim:"on the polyline" point_t (near_edges original) q)
    cropped;
  List.iter
    (fun ((x0, y0), (x1, y1)) ->
      for k = 0 to 8 do
        let t = Float.of_int k /. 8. in
        let q = (x0 +. (t *. (x1 -. x0)), y0 +. (t *. (y1 -. y0))) in
        if inside_box ~margin:1e-6 crop_box q then
          satisfies ~claim:"kept by the crop" point_t (near_edges cropped) q
      done)
    original

(* A cubic whose control values reach 1 in magnitude, and a box that cuts it
   several times. *)
let wavy =
  Path.empty
  |> Path.move_to (pt (-1.) 0.5)
  |> Path.cubic_to (pt 1. 0.5) (pt (-1.) 0.5) (pt 1. 0.5)

let wavy_box = Box2.v 0. 0. 0.25 1.

(* A triangle that the unit box cuts to a square. *)
let corner = Path.polygon [| 0.; 7.; 0. |] [| 0.; 0.; 7. |]

let cropping =
  group "crop"
    [
      prop "crop gives only finite points"
        ~examples:
          [
            (("crop_box", crop_box), [ Poly (false, [ (nan, 0.); (nan, 0.) ]) ]);
            ( ("crop_box", crop_box),
              [ Poly (false, [ (1., 1.); (nan, 1.); (2., 1.); (3., 1.) ]) ] );
          ]
        (Gen.pair gen_crop_box gen_gappy_program)
        crop_is_finite;
      prop "crop is p when p has no gap and lies within the box" gen_program
        (fun prog ->
          let p = build prog in
          equal path_t p (Path.crop (Box2.v (-200.) (-200.) 400. 400.) p));
      prop "crop keeps the windings of polygons within the box"
        ~examples:
          [
            ( [ [ (-10., -10.); (10., -10.); (10., 10.); (-10., 10.) ] ],
              (0., 0.) );
          ]
        (Gen.pair (gen_shapes ~closed:true) (gen_point 70.))
        crop_keeps_windings;
      prop "crop keeps the points of polylines within the box"
        (gen_shapes ~closed:false) crop_keeps_lines;
      prop "crop adds no segment that draws nothing" gen_grid_shapes
        crop_adds_nothing_degenerate;
      cases "crop drops a closed subpath left with no area" ~id:fst no_area
        (fun (_, (b, p)) -> equal segs_exact [] (segs (Path.crop b p)));
      test "crop keeps a flat closed subpath within the box" (fun () ->
          let within = Path.polygon [| 0.2; 0.5; 0.8 |] [| 0.5; 0.5; 0.5 |]
          and crossing = Path.polyline [| 0.5; 2. |] [| 0.2; 0.2 |]
          and cut_flat = Path.polygon [| -1.; 0.5; -1. |] [| 0.8; 0.8; 0.8 |] in
          equal segs_exact
            [
              M (0.2, 0.5);
              L (0.5, 0.5);
              L (0.8, 0.5);
              Z;
              M (0.5, 0.2);
              L (1., 0.2);
            ]
            (segs
               (Path.crop unit_box
                  (within |> Path.append crossing |> Path.append cut_flat))));
      (* Points of the cubics of a piece within the box round within a few ulps
         of 100. *)
      prop "crop draws only within the box" gen_gappy_program (fun prog ->
          List.iter
            (function
              | M (x, y) | L (x, y) ->
                  satisfies ~claim:"within the box" point_t
                    (inside_box ~margin:(-1e-12) crop_box)
                    (x, y)
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
          | s -> failf "crop is %a" (Format.pp_print_list pp_seg) s);
      test "crop ends a cut closed subpath with its close, no line back"
        crop_ends_with_close;
      cases "crop of a circle fills its part within the box" ~id:fst
        circle_crops crop_fills_circle_part;
      test "crop of a ring around the box fills nothing" (fun () ->
          let outer = Path.rect (Box2.v (-4.) (-4.) 8. 8.) in
          let inner =
            Path.polygon [| -2.; -2.; 2.; 2. |] [| -2.; 2.; 2.; -2. |]
          in
          let ring = Path.append inner outer in
          equal float_exact 0.
            (area (polygons (segs (Path.crop unit_box ring)))));
      test "crop of a path outside the box is empty" (fun () ->
          equal segs_exact []
            (segs
               (Path.crop unit_box
                  (Path.polyline [| 2.; 3.; 3. |] [| 0.; 0.; 5. |]))));
      cases "crop cuts a segment whose differences overflow along its line"
        ~id:fst
        [
          ( "a diagonal across max_float",
            ( Box2.v 0. 0. 10. 10.,
              ( -.Float.max_float,
                -.Float.max_float,
                Float.max_float,
                Float.max_float ),
              [ M (0., 0.); L (10., 10.) ] ) );
          ( "a diagonal from 1e300",
            (unit_box, (1e300, 1e300, 0.5, 0.5), [ M (1., 1.); L (0.5, 0.5) ])
          );
        ]
        (fun (_, (b, (x0, y0, x1, y1), expected)) ->
          equal segs_near expected
            (segs (Path.crop b (Path.polyline [| x0; x1 |] [| y0; y1 |]))));
      cases "crop commutes with scaling by a power of two"
        ~id:(fun (n, _, _, e) -> Printf.sprintf "%s by 2^%d" n e)
        [
          ("a wavy cubic", wavy, wavy_box, 1);
          ("a wavy cubic", wavy, wavy_box, 996);
          ("a triangle", corner, unit_box, 1);
          ("a triangle", corner, unit_box, 1020);
        ]
        (fun (_, p, b, e) ->
          (* Scaling by a power of two is exact, so the crops agree bit for bit,
             however large the path's values. *)
          let k = Float.ldexp 1. e in
          let m = Affine.scale k k in
          equal segs_exact
            (segs (Path.transform m (Path.crop b p)))
            (segs (Path.crop (Box2.transform m b) (Path.transform m p))));
      prop "crop with a map keeps the points of polylines within the box"
        (Gen.pair gen_affine (gen_shapes ~closed:false))
        crop_keeps_lines_under;
      cases "crop with a map keeps a line's direction past the floats"
        ~id:(Printf.sprintf "to %g") [ 100.; 1e300; 1e308 ] (fun l ->
          (* The image runs from (4, 4) along (2, 1) and leaves the box at x =
             10, wherever its far end lies, beyond the floats included. *)
          let m = Affine.scale 4. 4. in
          let p = Path.polyline [| 1.; 1. +. l |] [| 1.; 1. +. (l /. 2.) |] in
          equal segs_near
            [ M (4., 4.); L (10., 7.) ]
            (segs (Path.transform m (Path.crop ~m (Box2.v 0. 0. 10. 10.) p))));
      test "cut keeps a closed subpath's start, running along the box beyond"
        (fun () ->
          (* From (5, 5) out past x = 10 and back: the part beyond runs along x
             = 10, and the subpath still starts at (5, 5), so a stroke's dashes
             keep their phase there. *)
          let p = Path.polygon [| 5.; 20.; 20.; 5. |] [| 5.; 5.; 8.; 8. |] in
          equal segs_exact
            [ M (5., 5.); L (10., 5.); L (10., 8.); L (5., 8.); Z ]
            (segs (Path.cut (Box2.v 0. 0. 10. 10.) p)));
      test "cut keeps a flat closed subpath that crop drops" (fun () ->
          let p = Path.polygon [| 5.; 20. |] [| 5.; 5. |] in
          let b = Box2.v 0. 0. 10. 10. in
          equal segs_exact [] (segs (Path.crop b p));
          equal segs_exact [ M (5., 5.); L (10., 5.); Z ] (segs (Path.cut b p)));
      test "cut keeps a closed subpath within the box closed" (fun () ->
          let p = Path.rect (Box2.v 1. 1. 2. 2.) in
          equal path_t p (Path.cut (Box2.v 0. 0. 10. 10.) p));
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
      (* cos and sin of pi/2 round within an ulp. *)
      test "circle turns clockwise on screen" (fun () ->
          match segs (Path.circle (pt 0. 0.) 1.) with
          | M (1., 0.) :: C (_, _, _, _, x, y) :: _ ->
              equal (pair (float 1e-15) (float 1e-15)) (0., 1.) (x, y)
          | s -> failf "circle is %a" (Format.pp_print_list pp_seg) s);
      cases "circle raises on a radius of" ~id:(Printf.sprintf "%g")
        [ -1.; Float.nan; infinity ] (fun r ->
          raises_match (Exn.invalid_arg ~sub:"Path.circle") (fun () ->
              Path.circle (pt 0. 0.) r));
      test "circle of radius 0 is a point" (fun () ->
          equal
            (option (box_near 0.))
            (Some (Box2.v 3. 4. 0. 0.))
            (Path.bounds (Path.circle (pt 3. 4.) 0.)));
      test "polyline copies its arrays" polyline_copies;
      cases "polyline and polygon of fewer than two points are empty" ~id:fst
        [
          ("polyline of none", Path.polyline [||] [||]);
          ("polyline of one", Path.polyline [| 1. |] [| 1. |]);
          ("polygon of one", Path.polygon [| 1. |] [| 1. |]);
        ]
        (fun (_, p) -> equal path_t Path.empty p);
      test "polyline raises on arrays of different lengths" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              Path.polyline [| 1.; 2. |] [| 1. |]));
      test "polygon is the polyline closed" (fun () ->
          let xs = [| 0.; 1.; 1. |] and ys = [| 0.; 0.; 1. |] in
          equal path_t (Path.close (Path.polyline xs ys)) (Path.polygon xs ys));
    ]

(* Gaps *)

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
      cases "fold applies the gap rule"
        ~id:(fun (n, _, _) -> n)
        gap_cases
        (fun (_, p, expected) -> equal segs_exact expected (segs p));
      prop "fold gives only finite numbers" gen_gappy_program (fun prog ->
          List.iter
            (satisfies ~claim:"a finite segment" seg_t all_finite)
            (segs (build prog)));
      prop "flatten gives only finite numbers, whatever the map"
        (Gen.pair gen_gappy_program
           (Gen.of_list ~pp:Affine.pp
              [
                Affine.id; Affine.scale 1e308 1e308; { Affine.id with x0 = nan };
              ]))
        (fun (prog, m) ->
          List.iter
            (satisfies ~claim:"a finite segment" seg_t all_finite)
            (flat m (build prog)));
    ]

(* Flattening *)

(* [curve_samples] cuts a cubic into this many chords at uniform parameters.
   Wang's bound puts them within [0.75 d /. n²] of it, [d] the largest second
   difference of its control points, below 400 here: within 3e-4. *)
let curve_samples = 1024

let sampled (x0, y0) (x1, y1) (x2, y2) (x3, y3) =
  let n = curve_samples in
  List.init (n + 1) (fun i ->
      let t = Float.of_int i /. Float.of_int n in
      (cubic t x0 x1 x2 x3, cubic t y0 y1 y2 y3))

let second_difference (ax, ay) (bx, by) (cx, cy) =
  Float.hypot (ax -. (2. *. bx) +. cx) (ay -. (2. *. by) +. cy)

(* Flattened, a cubic from its start to its end: the points of each chord, at
   eighths of its length, lie within the tolerance of the curve. They are
   measured against the curve's fine sampling, which departs from it by at most
   [slack], and rounded within a few ulps of 100. A single chord across an S
   checked at its midpoint alone would pass. *)
let chords_within ((a, b, c, d), tol) =
  let p =
    Path.empty
    |> Path.move_to (pt (fst a) (snd a))
    |> Path.cubic_to
         (pt (fst b) (snd b))
         (pt (fst c) (snd c))
         (pt (fst d) (snd d))
  in
  let pts =
    List.filter_map
      (function M (x, y) | L (x, y) -> Some (x, y) | _ -> None)
      (flat ~tolerance:tol Affine.id p)
  in
  equal ~msg:"from the start to the end" (pair point_t point_t) (a, d)
    (List.hd pts, List.hd (List.rev pts));
  let curve = pairs (sampled a b c d) in
  let dd = Float.max (second_difference a b c) (second_difference b c d) in
  let slack = 0.75 *. dd /. Float.of_int (curve_samples * curve_samples) in
  List.iter
    (fun ((ax, ay), (bx, by)) ->
      for k = 1 to 7 do
        let t = Float.of_int k /. 8. in
        let q = (ax +. (t *. (bx -. ax)), ay +. (t *. (by -. ay))) in
        let dist =
          List.fold_left
            (fun d e -> Float.min d (dist_to_segment q e))
            infinity curve
        in
        at_most
          ~msg:(Printf.sprintf "point %d/8 of a chord from the curve" k)
          float_exact
          ~than:(tol +. slack +. 1e-12)
          dist
      done)
    (pairs pts)

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

let last_point segs =
  List.fold_left
    (fun acc s ->
      match s with
      | M (x, y) | L (x, y) | C (_, _, _, _, x, y) -> Some (x, y)
      | Z -> acc)
    None segs

let gen_curve =
  let c =
    Gen.pair (Gen.float_range (-100.) 100.) (Gen.float_range (-100.) 100.)
  in
  Gen.pair (Gen.quad c c c c) (Gen.float_range 0.01 2.)

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
        ~examples:[ (((0., 0.), (0., 100.), (100., -100.), (100., 0.)), 0.01) ]
        gen_curve chords_within;
      test "flatten's tolerance defaults to 0.1" (fun () ->
          let c = Path.circle (pt 0. 0.) 100. in
          equal segs_exact (flat ~tolerance:0.1 Affine.id c) (flat Affine.id c));
      cases "flatten accepts an infinite tolerance" ~id:fst
        [
          ("a circle", Path.circle (pt 0. 0.) 10.);
          ( "a cubic whose control points span the floats",
            Path.empty
            |> Path.move_to (pt 0. 0.)
            |> Path.cubic_to
                 (pt Float.max_float Float.max_float)
                 (pt (-.Float.max_float) (-.Float.max_float))
                 (pt 0. 0.) );
        ]
        (fun (_, p) ->
          equal (option point_t)
            (last_point (segs p))
            (last_point (flat ~tolerance:infinity Affine.id p)));
      cases "flatten raises on a tolerance of" ~id:(Printf.sprintf "%g")
        [ 0.; -1.; Float.nan ] (fun tolerance ->
          raises_match Exn.invalid_arg (fun () ->
              flat ~tolerance Affine.id (Path.circle (pt 0. 0.) 1.)));
      test "flatten cuts a cubic into at most 65536 chords" (fun () ->
          let p =
            Path.empty
            |> Path.move_to (pt 0. 0.)
            |> Path.cubic_to (pt 0. 1e12) (pt 1e12 1e12) (pt 1e12 0.)
          in
          at_most int ~than:(1 + 65536)
            (List.length (flat ~tolerance:0.1 Affine.id p)));
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

(* Flattened points round within a few ulps of coordinates below 150. *)
let contains_points prog =
  let p = build prog in
  match Path.bounds p with
  | None -> equal segs_exact [] (flat Affine.id p)
  | Some b ->
      let grown = Box2.grow 1e-12 b in
      List.iter
        (function
          | M (x, y) | L (x, y) ->
              satisfies ~claim:"within the bounds" point_t
                (fun q -> inside_box ~margin:0. grown q)
                (x, y)
          | C _ | Z -> ())
        (flat ~tolerance:0.01 Affine.id p)

(* A flattening's points lie on the curves and its chords within its tolerance
   of them, so the two boxes are within that tolerance, plus rounding. *)
let bounds_tight prog =
  let p = build prog in
  equal
    (option (box_near (1e-4 +. 1e-12)))
    (Path.bounds p)
    (flat_box ~tolerance:1e-4 p)

let bounds =
  group "bounds"
    [
      cases "bounds of a path that draws nothing is None" ~id:fst
        [
          ("empty", Path.empty);
          ("gaps only", Path.polyline [| nan; nan |] [| 0.; 1. |]);
        ]
        (fun (_, p) -> is_none ~pp:pp_box (Path.bounds p));
      test "bounds of a lone subpath start is its point" (fun () ->
          equal
            (option (box_near 0.))
            (Some (Box2.v 1. 2. 0. 0.))
            (Path.bounds (Path.move_to (pt 1. 2.) Path.empty)));
      (* y = 30 t (1 - t), whose largest value, at t = 1/2, is 7.5. *)
      test "bounds of a cubic is exact, not its control points" (fun () ->
          let p =
            Path.empty
            |> Path.move_to (pt 0. 0.)
            |> Path.cubic_to (pt 0. 10.) (pt 10. 10.) (pt 10. 0.)
          in
          equal
            (option (box_near 1e-14))
            (Some (Box2.v 0. 0. 10. 7.5))
            (Path.bounds p));
      (* The circle's cubics end at its extremes, which cos and sin put within a
         few ulps of 8. *)
      test "bounds of a circle is its square" (fun () ->
          equal
            (option (box_near 1e-14))
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

(* Comparing and formatting *)

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

let comparing =
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
      cases "equal compares points with Float.equal" ~id:fst
        [ ("NaN equals NaN", (nan, nan)); ("-0. equals 0.", (-0., 0.)) ]
        (fun (_, (x, x')) ->
          equal path_t
            (Path.polyline [| x; 1. |] [| 0.; 1. |])
            (Path.polyline [| x'; 1. |] [| 0.; 1. |]));
      test "equal tells a non-finite point from a finite one" (fun () ->
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
          equal string "M 1 2 L 3 4.5 C 5 6 7 8 9 nan Z" (printed p));
      test "pp writes a polyline and a polygon point by point" (fun () ->
          let p =
            Path.polygon [| 0.; 1.; 1. |] [| 0.; 0.; 1. |]
            |> Path.append (Path.polyline [| 2.; 3. |] [| 2.; nan |])
          in
          equal string "M 0 0 L 1 0 L 1 1 Z M 2 2 L 3 nan" (printed p));
      test "pp writes numbers as %g does, rounded once, ties to even" (fun () ->
          let p =
            Path.empty
            |> Path.move_to (pt 135.5625 1e16)
            |> Path.line_to (pt 1.2345678e-5 (-0.000123456789))
          in
          equal string "M 135.562 1e+16 L 1.23457e-05 -0.000123457" (printed p));
    ]

let () =
  exit
    (run "hugin.gg path"
       [
         types; building; cropping; shapes; gaps; flattening; bounds; comparing;
       ])
