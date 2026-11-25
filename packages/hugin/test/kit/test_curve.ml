(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_gg
open Hugin_kit

let invalid f = raises_match (Exn.invalid_arg ?sub:None) f
let pp_float ppf x = Format.fprintf ppf "%.17g" x

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

(* [Float.equal] takes [-0.] equal to [0.]: [Path.fold] gives [0.] for a stored
   [-0.]. *)
let segs_exact = list (Testable.make ~pp:pp_seg ~equal:(seg_equal Float.equal))

(* Segments whose coordinates differ by at most [tol]. *)
let segs_within tol =
  let close a b = Float.abs (a -. b) <= tol in
  list (Testable.make ~pp:pp_seg ~equal:(seg_equal close))

let point =
  let coord = Testable.make ~pp:pp_float ~equal:Float.equal in
  pair coord coord

let path = Testable.make ~pp:Path.pp ~equal:Path.equal
let curve = Testable.make ~pp:Curve.pp ~equal:Curve.equal

let segs p =
  Path.fold
    ~move:(fun acc x y -> M (x, y) :: acc)
    ~line:(fun acc x y -> L (x, y) :: acc)
    ~cubic:(fun acc a b c d e f -> C (a, b, c, d, e, f) :: acc)
    ~close:(fun acc -> Z :: acc)
    [] p
  |> List.rev

let seg_end = function
  | M (x, y) | L (x, y) | C (_, _, _, _, x, y) -> Some (x, y)
  | Z -> None

let cubics s =
  List.filter_map
    (function C (a, b, c, d, e, f) -> Some (a, b, c, d, e, f) | _ -> None)
    s
  |> Array.of_list

let name c = Format.asprintf "%a" Curve.pp c
let lines = Curve.[ linear; step_after; step_before; step_mid ]
let smooth = Curve.[ monotone_x; monotone_y; natural; catmull_rom; basis ]
let curves = lines @ smooth

(* The largest magnitude of the coordinates, at least [1.]. *)
let magnitude (xs, ys) =
  Array.fold_left (fun m v -> Float.max m (Float.abs v)) 1. (Array.append xs ys)

(* Generators *)

let pp_points ppf (xs, ys) =
  Format.fprintf ppf "@[<1>[%a]@]"
    (Format.pp_print_array ~pp_sep:Format.pp_print_space (fun ppf (x, y) ->
         Format.fprintf ppf "(%h, %h)" x y))
    (Array.map2 (fun x y -> (x, y)) xs ys)

(* Coordinates of a page, in points. Those within [1e-6] of [0.] are [0.]:
   smooth curves divide by differences of coordinates, and subnormal ones
   overflow ({!Curve}). *)
let coord =
  Gen.map
    (fun x -> if Float.abs x < 1e-6 then 0. else x)
    (Gen.float_range (-1000.) 1000.)

(* Coordinates from a small set, so that points repeat, x values tie and
   segments are flat. *)
let small = Gen.map float_of_int (Gen.int_range (-3) 3)

let gen_points ?(coord = coord) size =
  Gen.with_pp pp_points
    (Gen.map
       (fun pts -> (Array.map fst pts, Array.map snd pts))
       (Gen.array ~size (Gen.pair coord coord)))

(* Runs of [least] to [10] points, half of them of small coordinates. *)
let gen_finite least =
  Gen.one_of
    [
      gen_points (Gen.int_range least 10);
      gen_points ~coord:small (Gen.int_range least 10);
    ]

(* Points whose coordinates are multiples of 1/8, which affine maps of moderate
   factors neither underflow nor make equal. *)
let gen_moderate =
  let eighths =
    Gen.map (fun k -> float_of_int k /. 8.) (Gen.int_range (-8000) 8000)
  in
  Gen.one_of
    [
      gen_points ~coord:eighths (Gen.int_range 2 10);
      gen_points ~coord:small (Gen.int_range 2 10);
    ]

(* Points of which some are missing, by a non-finite x or y. *)
let gen_gappy =
  let missing = Gen.of_list ~pp:pp_float [ nan; infinity; neg_infinity ] in
  let point =
    Gen.frequency
      [
        (4, Gen.pair coord coord);
        (1, Gen.pair missing coord);
        (1, Gen.pair coord missing);
      ]
  in
  Gen.with_pp pp_points
    (Gen.map
       (fun pts -> (Array.map fst pts, Array.map snd pts))
       (Gen.array ~size:(Gen.int_range 0 15) point))

(* Affine maps *)

let map_points m (xs, ys) =
  let p = Array.map2 (fun x y -> P2.transform m (P2.v x y)) xs ys in
  (Array.map P2.x p, Array.map P2.y p)

(* A bound on the magnitude of the terms of [m] applied to the points, and so of
   every value the two sides compute before they divide. *)
let terms (m : Affine.t) pts =
  let { Affine.xx; yx; xy; yy; x0; y0 } = m in
  let factor = Float.(abs xx +. abs yx +. abs xy +. abs yy) in
  (magnitude pts *. factor) +. Float.abs x0 +. Float.abs y0

let commutes ~tol c m pts =
  let xs', ys' = map_points m pts in
  equal (segs_within tol)
    (segs (Path.transform m (Curve.path c (fst pts) (snd pts))))
    (segs (Curve.path c xs' ys'))

let gen_affine =
  let k = Gen.float_range (-3.) 3. in
  Gen.with_pp Affine.pp
    (Gen.such_that
       (fun (m : Affine.t) ->
         Float.abs ((m.xx *. m.yy) -. (m.xy *. m.yx)) >= 0.1)
       (Gen.map
          (fun ((xx, yx, xy), (yy, x0, y0)) ->
            { Affine.xx; yx; xy; yy; x0; y0 })
          (Gen.pair (Gen.triple k k k) (Gen.triple k coord coord))))

let gen_axes =
  let factor =
    Gen.map
      (fun (s, neg) -> if neg then -.s else s)
      (Gen.pair (Gen.float_range 0.1 10.) Gen.bool)
  in
  Gen.with_pp Affine.pp
    (Gen.map
       (fun ((sx, sy), (x0, y0)) -> Affine.(translate x0 y0 * scale sx sy))
       (Gen.pair (Gen.pair factor factor) (Gen.pair coord coord)))

let gen_similarity =
  Gen.with_pp Affine.pp
    (Gen.map
       (fun ((a, k, mirror), (x0, y0)) ->
         Affine.(
           translate x0 y0 * rotate a * scale k (if mirror then -.k else k)))
       (Gen.pair
          (Gen.triple
             (Gen.float_range 0. (2. *. Float.pi))
             (Gen.float_range 0.1 10.) Gen.bool)
          (Gen.pair coord coord)))

(* [spread pts] is the largest difference of the coordinates of consecutive
   points over the smallest nonzero one, at least [1.]: how much a slope or a
   ratio of distances between the points magnifies an error of their
   coordinates. *)
let spread (xs, ys) =
  let lo = ref infinity and hi = ref 0. in
  let see d =
    let d = Float.abs d in
    if d > 0. then begin
      lo := Float.min !lo d;
      hi := Float.max !hi d
    end
  in
  for k = 0 to Array.length xs - 2 do
    see (xs.(k + 1) -. xs.(k));
    see (ys.(k + 1) -. ys.(k))
  done;
  if !hi = 0. then 1. else Float.max 1. (!hi /. !lo)

(* Each side maps a point with two products and two sums, so it errs by two
   rounding errors of [terms]. [linear] adds nothing to that, and [basis] takes
   means of three points, two more. [natural] solves a tridiagonal system by
   elimination with multipliers of at most 1/2 and pivots of at least 2, so an
   error shrinks from row to row and each control errs by at most about 16
   rounding errors of [terms]. *)
let affine_law (c, m, pts) =
  commutes ~tol:(0x1p5 *. epsilon_float *. terms m pts) c m pts

(* The mapped points err by a rounding error of [terms], and a difference [d] of
   their coordinates then errs by [terms / d] rounding errors relative to
   itself. A monotone control point is its end point plus [h/3 t], at most two
   thirds of [|y1 - y0|], with [t] taken from such differences in about ten
   operations. A Catmull–Rom control point is its end point plus differences of
   points weighted by ratios of distances, each at most [spread]. Either errs by
   at most about 2^5 [spread] rounding errors of [terms]; a step corner is a
   mapped coordinate, and a midpoint two roundings. *)
let conditioned_law (c, m, pts) =
  let tol = 0x1p5 *. epsilon_float *. terms m pts in
  commutes ~tol:(tol *. (1. +. spread (map_points m pts))) c m pts

let affine =
  group "affine maps"
    [
      prop "linear, natural and basis commute with every affine map"
        (Gen.triple
           (Gen.of_list ~pp:Curve.pp Curve.[ linear; natural; basis ])
           gen_affine gen_moderate)
        affine_law;
      prop "steps and monotone curves commute with scaling each axis"
        (Gen.triple
           (Gen.of_list ~pp:Curve.pp
              Curve.
                [ step_after; step_before; step_mid; monotone_x; monotone_y ])
           gen_axes gen_moderate)
        conditioned_law;
      prop "catmull_rom commutes with similarities"
        (Gen.pair gen_similarity gen_moderate) (fun (m, pts) ->
          conditioned_law (Curve.catmull_rom, m, pts));
      prop "catmull_rom commutes with scaling by 1e-30" gen_moderate (fun pts ->
          conditioned_law (Curve.catmull_rom, Affine.scale 1e-30 1e-30, pts));
    ]

(* Missing points *)

(* [runs xs ys] is the runs of the points, cut at missing points. *)
let runs (xs, ys) =
  let n = Array.length xs in
  let rec go i first acc =
    if i = n || not (Float.is_finite xs.(i) && Float.is_finite ys.(i)) then
      let acc =
        if i > first then
          (Array.sub xs first (i - first), Array.sub ys first (i - first))
          :: acc
        else acc
      in
      if i = n then List.rev acc else go (i + 1) (i + 1) acc
    else go (i + 1) first acc
  in
  go 0 0 []

let runs_alone pts =
  let has v = Array.exists (Float.equal v) (Array.append (fst pts) (snd pts)) in
  cover "a run breaks" (List.length (runs pts) >= 2);
  cover "a nan is missing" (has nan);
  cover "an infinity is missing" (has infinity);
  cover "a negative infinity is missing" (has neg_infinity);
  List.iter
    (fun c ->
      let alone =
        List.fold_left
          (fun p (xs, ys) -> Path.append (Curve.path c xs ys) p)
          Path.empty (runs pts)
      in
      equal ~msg:(name c) path alone (Curve.path c (fst pts) (snd pts)))
    curves

let missing =
  group "missing points"
    [
      prop "each run is drawn alone" gen_gappy runs_alone;
      cases "a run of one point draws nothing" ~id:name curves (fun c ->
          equal path Path.empty (Curve.path c [| 1. |] [| 2. |]);
          equal path Path.empty
            (Curve.path c [| nan; 1.; 2.; 3. |] [| 0.; 1.; infinity; 3. |]));
    ]

(* Curves *)

let xs5 = [| 0.; 1.; 2.; 3.; 4. |]
let ys5 = [| 0.; 2.; 1.; 3.; 3. |]

(* d3-shape 3.2 draws [linear], the steps, [natural], [catmull_rom] and [basis]
   through these points with the same segments, but for the data points that
   [step_mid] passes through on its flat parts and d3 leaves out. Its
   [monotone_x] takes another slope at the ends. *)
let d3 () =
  List.iter
    (fun c -> Format.printf "%s: %a@." (name c) Path.pp (Curve.path c xs5 ys5))
    curves;
  expect (output ())
  @@ __POS_OF__
       {|
    linear: M 0 0 L 1 2 L 2 1 L 3 3 L 4 3
    step_after: M 0 0 L 1 0 L 1 2 L 2 2 L 2 1 L 3 1 L 3 3 L 4 3 L 4 3
    step_before: M 0 0 L 0 2 L 1 2 L 1 1 L 2 1 L 2 3 L 3 3 L 3 3 L 4 3
    step_mid: M 0 0 L 0.5 0 L 0.5 2 L 1 2 L 1.5 2 L 1.5 1 L 2 1 L 2.5 1 L 2.5 3
              L 3 3 L 3.5 3 L 3.5 3 L 4 3
    monotone_x: M 0 0 C 0.333333 1.16667 0.666667 2 1 2 C 1.33333 2 1.66667 1 2 1
                C 2.33333 1 2.66667 3 3 3 C 3.33333 3 3.66667 3 4 3
    monotone_y: M 0 0 L 1 2 L 2 1 L 3 3 L 4 3
    natural: M 0 0 C 0.333333 1.01786 0.666667 2.03571 1 2
             C 1.33333 1.96429 1.66667 0.875 2 1
             C 2.33333 1.125 2.66667 2.46429 3 3
             C 3.33333 3.53571 3.66667 3.26786 4 3
    catmull_rom: M 0 0 C 0 0 0.618868 1.93815 1 2
                 C 1.3031 2.04919 1.6969 0.950813 2 1
                 C 2.38113 1.06185 2.56772 2.73284 3 3 C 3.28908 3.17866 4 3 4 3
    basis: M 0 0 L 0.166667 0.333333 C 0.333333 0.666667 0.666667 1.33333 1 1.5
           C 1.33333 1.66667 1.66667 1.33333 2 1.5
           C 2.33333 1.66667 2.66667 2.33333 3 2.66667
           C 3.33333 3 3.66667 3 3.83333 3 L 4 3
    |}

let two_points (xs, ys) =
  List.iter
    (fun c ->
      equal ~msg:(name c) segs_exact
        [ M (xs.(0), ys.(0)); L (xs.(1), ys.(1)) ]
        (segs (Curve.path c xs ys)))
    (Curve.linear :: smooth)

(* The end points of the segments include every point, in order, a point equal
   to the one before it counting as reached. *)
let passes_through (xs, ys) =
  List.iter
    (fun c ->
      let ends = List.filter_map seg_end (segs (Curve.path c xs ys)) in
      let at i (x, y) = Float.equal x xs.(i) && Float.equal y ys.(i) in
      let rec sub i ends =
        if i = Array.length xs then ()
        else if i > 0 && at i (xs.(i - 1), ys.(i - 1)) then sub (i + 1) ends
        else
          match ends with
          | [] ->
              failf "%s: point %d (%h, %h) is not on the path" (name c) i xs.(i)
                ys.(i)
          | e :: rest -> if at i e then sub (i + 1) rest else sub i rest
      in
      sub 0 ends)
    Curve.[ monotone_x; monotone_y; natural; catmull_rom ]

let curves_group =
  group "curves"
    [
      test "each curve draws five points as baselined" d3;
      prop "every curve but the steps draws two points as a straight segment"
        (gen_points ~coord:small (Gen.int_range 2 2))
        ~examples:[ ([| 0.; 1. |], [| 0.; 0. |]); ([| 1.; 1. |], [| 2.; 2. |]) ]
        two_points;
      prop
        "monotone_x, monotone_y, natural and catmull_rom pass through every \
         point"
        (gen_finite 2) passes_through;
    ]

(* The polylines the four curves of lines state: from each point, the corners
   [corner p p'] on the way to the next point [p'], then [p']. *)
let polyline corner (xs, ys) =
  let n = Array.length xs in
  M (xs.(0), ys.(0))
  :: List.concat
       (List.init (n - 1) (fun k ->
            let p = (xs.(k), ys.(k)) and p' = (xs.(k + 1), ys.(k + 1)) in
            corner p p' @ [ L (fst p', snd p') ]))

let draws_polyline c corner pts =
  equal segs_exact (polyline corner pts)
    (segs (Curve.path c (fst pts) (snd pts)))

(* Two points that a step joins with a corner, as a run of two. *)
let step_example = ([| 1.; 4. |], [| 2.; 6. |])

let linear =
  group "linear"
    [
      prop "joins consecutive points with straight segments" (gen_finite 2)
        (draws_polyline Curve.linear (fun _ _ -> []));
    ]

let steps =
  [
    group "step_after"
      [
        prop
          "goes from each point along x to the next point's x, then along y to \
           it"
          ~examples:[ step_example ] (gen_finite 2)
          (draws_polyline Curve.step_after (fun (_, y) (x', _) -> [ L (x', y) ]));
      ];
    group "step_before"
      [
        prop
          "goes from each point along y to the next point's y, then along x to \
           it"
          ~examples:[ step_example ] (gen_finite 2)
          (draws_polyline Curve.step_before (fun (x, _) (_, y') ->
               [ L (x, y') ]));
      ];
    (* Halfway is [(x + x') / 2] rounded once: coordinates within a thousand
       neither overflow the sum nor make its half subnormal. *)
    group "step_mid"
      [
        prop
          "goes from each point along x halfway to the next point's x, along y \
           to its y, then along x to it"
          ~examples:[ step_example ] (gen_finite 2)
          (draws_polyline Curve.step_mid (fun (x, y) (x', y') ->
               let m = (x +. x') /. 2. in
               [ L (m, y); L (m, y') ]));
      ];
  ]

(* Monotone curves *)

(* [bezier t a b c d] is the cubic Bézier of control values [a] to [d] at
   [t]. *)
let bezier t a b c d =
  let u = 1. -. t in
  (u *. u *. u *. a)
  +. (3. *. u *. u *. t *. b)
  +. (3. *. u *. t *. t *. c)
  +. (t *. t *. t *. d)

(* Between consecutive points the curve stays within their x and their y.
   Steffen's slopes keep each control value strictly between the segment's end
   values, by at least a third of their difference, so the cubic is a convex
   combination of values in [lo, hi]; evaluating it at [t] takes about ten
   roundings of the largest of them. *)
let never_overshoots (xs, ys) =
  let s = segs (Curve.path Curve.monotone_x xs ys) in
  cover "a cubic" (Array.length (cubics s) > 0);
  let within ~msg lo hi v =
    let lo, hi = (Float.min lo hi, Float.max lo hi) in
    let slack =
      16. *. epsilon_float *. Float.max (Float.abs lo) (Float.abs hi)
    in
    at_least ~msg float_exact ~than:(lo -. slack) v;
    at_most ~msg float_exact ~than:(hi +. slack) v
  in
  ignore
    (List.fold_left
       (fun (x0, y0) seg ->
         match seg with
         | M (x, y) | L (x, y) -> (x, y)
         | C (ax, ay, bx, by, x, y) ->
             let msg = Format.asprintf "from (%h, %h) to (%h, %h)" x0 y0 x y in
             for k = 0 to 32 do
               let t = float_of_int k /. 32. in
               within ~msg x0 x (bezier t x0 ax bx x);
               within ~msg y0 y (bezier t y0 ay by y)
             done;
             (x, y)
         | Z -> (x0, y0))
       (nan, nan) s)

(* At a point inside a piece, the cubic arriving and the one leaving have
   parallel tangents [a] and [b], of the same direction in x. A control point is
   the end point less a third of the x step and its slope, so each coordinate of
   a tangent errs by about two roundings of the largest coordinate [m], and
   their cross product by [4 m |a|] and [4 m |b|] rounding errors; its own
   products round by [|a| |b|], below [8 m] times the larger. *)
let slope_continuous (xs, ys) =
  let m = magnitude (xs, ys) in
  let rec joints prev acc = function
    | C (_, _, c2x, c2y, px, py) :: (C (c1x, c1y, _, _, x', _) :: _ as rest)
      when Float.compare px prev = Float.compare x' px ->
        joints px (((px -. c2x, py -. c2y), (c1x -. px, c1y -. py)) :: acc) rest
    | seg :: rest ->
        let prev = match seg_end seg with Some (x, _) -> x | None -> prev in
        joints prev acc rest
    | [] -> acc
  in
  let inner = joints nan [] (segs (Curve.path Curve.monotone_x xs ys)) in
  cover "a point inside a piece" (inner <> []);
  List.iter
    (fun ((ax, ay), (bx, by)) ->
      let size = Float.abs ax +. Float.abs ay +. Float.abs bx +. Float.abs by in
      at_most
        ~msg:(Format.asprintf "tangents (%h, %h) and (%h, %h)" ax ay bx by)
        float_exact
        ~than:(16. *. epsilon_float *. m *. size)
        (Float.abs ((ax *. by) -. (ay *. bx))))
    inner

(* [steffen (xs, ys)] is the cubics of the one piece [xs] makes, from the slopes
   curve.mli states, with [h] and [s] the difference of x and the slope of a
   segment and [h'] and [s'] those of the next, or at the last point the one
   before. *)
let steffen (xs, ys) =
  let n = Array.length xs in
  let h k = xs.(k + 1) -. xs.(k) in
  let s k = (ys.(k + 1) -. ys.(k)) /. h k in
  let sign v = if v > 0. then 1. else if v < 0. then -1. else 0. in
  let at_end h s h' s' =
    let p = (s *. (1. +. (h /. (h +. h')))) -. (s' *. h /. (h +. h')) in
    if p *. s <= 0. then 0.
    else if Float.abs p > 2. *. Float.abs s then 2. *. s
    else p
  in
  let inner k =
    let p = ((s (k - 1) *. h k) +. (s k *. h (k - 1))) /. (h (k - 1) +. h k) in
    (sign (s (k - 1)) +. sign (s k))
    *. Float.min
         (Float.min (Float.abs (s (k - 1))) (Float.abs (s k)))
         (Float.abs p /. 2.)
  in
  let slope k =
    if k = 0 then at_end (h 0) (s 0) (h 1) (s 1)
    else if k = n - 1 then
      at_end (h (n - 2)) (s (n - 2)) (h (n - 3)) (s (n - 3))
    else inner k
  in
  List.init (n - 1) (fun k ->
      let d = h k /. 3. in
      C
        ( xs.(k) +. d,
          ys.(k) +. (d *. slope k),
          xs.(k + 1) -. d,
          ys.(k + 1) -. (d *. slope (k + 1)),
          xs.(k + 1),
          ys.(k + 1) ))

(* Runs of 3 to 8 points whose x strictly increase, or decrease, by steps of 1/8
   to 8, multiples of 1/8 so that every difference of x is exact. *)
let gen_piece =
  let piece (steps, ys, down) =
    let xs = Array.make (Array.length steps + 1) 0. in
    Array.iteri
      (fun k w ->
        let w = Float.of_int w /. 8. in
        xs.(k + 1) <- (if down then xs.(k) -. w else xs.(k) +. w))
      steps;
    (xs, Array.sub ys 0 (Array.length xs))
  in
  Gen.with_pp pp_points
    (Gen.map piece
       (Gen.triple
          (Gen.array ~size:(Gen.int_range 2 7) (Gen.int_range 1 64))
          (Gen.array ~size:(Gen.int_range 8 8) small)
          Gen.bool))

(* The slopes are a few operations on exact differences, and a control point is
   its end point plus [h/3] times one, at most two thirds of the segment's
   difference of y: each control errs by a few roundings of the largest
   coordinate. *)
let takes_steffen (xs, ys) =
  let n = Array.length xs in
  let uneven = ref false in
  for k = 0 to n - 3 do
    if xs.(k + 1) -. xs.(k) <> xs.(k + 2) -. xs.(k + 1) then uneven := true
  done;
  cover "unequal differences of x" !uneven;
  equal
    (segs_within (0x1p4 *. epsilon_float *. magnitude (xs, ys)))
    (steffen (xs, ys))
    (List.tl (segs (Curve.path Curve.monotone_x xs ys)))

(* [by_pieces xs ys] is [monotone_x]'s path built as the run's pieces drawn
   alone, joined at their shared end points, and a straight segment between two
   points of equal x. *)
let by_pieces (xs, ys) =
  let n = Array.length xs in
  let dir k = Float.compare xs.(k + 1) xs.(k) in
  let rec go a acc =
    if a >= n - 1 then List.concat (List.rev acc)
    else if dir a = 0 then go (a + 1) ([ L (xs.(a + 1), ys.(a + 1)) ] :: acc)
    else
      let rec stop b = if b < n - 1 && dir b = dir a then stop (b + 1) else b in
      let b = stop (a + 1) in
      let piece a' = Array.sub a' a (b - a + 1) in
      let drawn = segs (Curve.path Curve.monotone_x (piece xs) (piece ys)) in
      go b (List.tl drawn :: acc)
  in
  M (xs.(0), ys.(0)) :: go 0 []

let piece_by_piece (xs, ys) =
  let n = Array.length xs in
  let dirs = List.init (n - 1) (fun k -> Float.compare xs.(k + 1) xs.(k)) in
  let rec turns = function
    | d :: (d' :: _ as rest) -> d * d' < 0 || turns rest
    | _ -> false
  in
  cover "x turn back" (turns (List.filter (( <> ) 0) dirs));
  cover "x repeat" (List.mem 0 dirs);
  equal segs_exact
    (by_pieces (xs, ys))
    (segs (Curve.path Curve.monotone_x xs ys))

let swap = { Affine.xx = 0.; yx = 1.; xy = 1.; yy = 0.; x0 = 0.; y0 = 0. }

let monotone =
  [
    group "monotone_x"
      [
        prop "never overshoots" (gen_finite 2) never_overshoots;
        prop "has a continuous slope inside a piece" (gen_finite 3)
          slope_continuous;
        prop "takes Steffen's slopes"
          ~examples:
            [
              (* Collinear: the middle slope is [|p| / 2] times two, p = 1. *)
              ([| 0.; 1.; 2. |], [| 0.; 1.; 2. |]);
              (* The end slope p = 1 × 1.5 - 9 × 0.5 = -3 is against s = 1, so
                 it is 0; the middle slope is [|s|] times two. *)
              ([| 0.; 1.; 2. |], [| 0.; 1.; 10. |]);
              (* The end slope p = 1.5 + 1 = 2.5 exceeds 2 s = 2, so it is 2;
                 the middle slope is 0 as the slopes' signs differ. *)
              ([| 0.; 1.; 2. |], [| 0.; 1.; -1. |]);
            ]
          gen_piece takes_steffen;
        prop "draws a run piece by piece"
          ~examples:
            [
              ([| 0.; 1.; 2.; 1.; 0. |], [| 0.; 1.; 0.; -1.; 0. |]);
              ([| 0.; 1.; 1.; 2. |], [| 0.; 1.; 2.; 3. |]);
            ]
          (gen_finite 2) piece_by_piece;
      ];
    group "monotone_y"
      [
        prop "is monotone_x with x and y exchanged" (gen_finite 2)
          (fun (xs, ys) ->
            equal path
              (Path.transform swap (Curve.path Curve.monotone_x ys xs))
              (Curve.path Curve.monotone_y xs ys));
      ];
  ]

(* Natural *)

(* The controls err by [affine_law]'s bound, 2^4 rounding errors of the largest
   coordinate, and each check sums three or four of them. *)
let natural_tol pts = 0x1p6 *. epsilon_float *. magnitude pts
let natural_cubics (xs, ys) = cubics (segs (Curve.path Curve.natural xs ys))

(* At each inner point the tangent arriving equals the one leaving: [P - c2] of
   one cubic is [c1 - P] of the next. *)
let first_derivative pts =
  let c = natural_cubics pts and w = float (natural_tol pts) in
  for k = 0 to Array.length c - 2 do
    let _, _, c2x, c2y, px, py = c.(k) and c1x, c1y, _, _, _, _ = c.(k + 1) in
    equal w (px -. c2x) (c1x -. px);
    equal w (py -. c2y) (c1y -. py)
  done

(* At each inner point the second difference arriving, [c1 - 2 c2 + P], equals
   the one leaving, [P - 2 c1 + c2]. *)
let second_derivative pts =
  let c = natural_cubics pts and w = float (natural_tol pts) in
  for k = 0 to Array.length c - 2 do
    let c1x, c1y, c2x, c2y, px, py = c.(k)
    and c1x', c1y', c2x', c2y', _, _ = c.(k + 1) in
    equal w (c1x -. (2. *. c2x) +. px) (px -. (2. *. c1x') +. c2x');
    equal w (c1y -. (2. *. c2y) +. py) (py -. (2. *. c1y') +. c2y')
  done

let free_ends ((xs, ys) as pts) =
  let c = natural_cubics pts and w = float (natural_tol pts) in
  let n = Array.length c in
  let c1x, c1y, c2x, c2y, _, _ = c.(0) in
  equal w 0. (xs.(0) -. (2. *. c1x) +. c2x);
  equal w 0. (ys.(0) -. (2. *. c1y) +. c2y);
  let c1x, c1y, c2x, c2y, x, y = c.(n - 1) in
  equal w 0. (c1x -. (2. *. c2x) +. x);
  equal w 0. (c1y -. (2. *. c2y) +. y)

let natural =
  group "natural"
    [
      prop "has a continuous first derivative" (gen_finite 3) first_derivative;
      prop "has a continuous second derivative" (gen_finite 3) second_derivative;
      prop "has a zero second derivative at both ends" (gen_finite 3) free_ends;
    ]

(* Catmull–Rom and basis *)

let dedup (xs, ys) =
  let n = Array.length xs in
  let keep =
    List.filter
      (fun i -> i = 0 || xs.(i) <> xs.(i - 1) || ys.(i) <> ys.(i - 1))
      (List.init n Fun.id)
  in
  ( Array.of_list (List.map (fun i -> xs.(i)) keep),
    Array.of_list (List.map (fun i -> ys.(i)) keep) )

let catmull_rom_ends pts =
  let xs, ys = dedup pts in
  assume (Array.length xs >= 3);
  let c = cubics (segs (Curve.path Curve.catmull_rom xs ys)) in
  let n = Array.length xs and m = Array.length c in
  let c1x, c1y, _, _, _, _ = c.(0) and _, _, c2x, c2y, _, _ = c.(m - 1) in
  equal (pair point point)
    ((xs.(0), ys.(0)), (xs.(n - 1), ys.(n - 1)))
    ((c1x, c1y), (c2x, c2y))

let catmull_rom_repeats (xs, ys) =
  let xs', ys' = dedup (xs, ys) in
  assume (Array.length xs' >= 2);
  cover "repeats" (Array.length xs' < Array.length xs);
  equal path
    (Curve.path Curve.catmull_rom xs' ys')
    (Curve.path Curve.catmull_rom xs ys)

let catmull_rom =
  group "catmull_rom"
    [
      prop "puts its end control points on its end points" (gen_finite 3)
        catmull_rom_ends;
      prop "counts consecutive equal points once"
        (gen_points ~coord:small (Gen.int_range 2 10))
        catmull_rom_repeats;
      prop "draws a run of equal points as linear does"
        (Gen.pair (Gen.pair coord coord) (Gen.int_range 2 6))
        (fun ((x, y), n) ->
          let xs = Array.make n x and ys = Array.make n y in
          equal path
            (Curve.path Curve.linear xs ys)
            (Curve.path Curve.catmull_rom xs ys));
    ]

let basis_ends (xs, ys) =
  let s = segs (Curve.path Curve.basis xs ys) in
  let n = Array.length xs in
  equal
    (pair (option point) (option point))
    (Some (xs.(0), ys.(0)), Some (xs.(n - 1), ys.(n - 1)))
    (seg_end (List.hd s), seg_end (List.nth s (List.length s - 1)))

(* A uniform cubic B-spline segment of control points [q0] to [q3] is the cubic
   Bézier from [(q0 + 4 q1 + q2) / 6], of control points [(2 q1 + q2) / 3] and
   [(q1 + 2 q2) / 3], to [(q1 + 4 q2 + q3) / 6]. With the ends counted three
   times, the first and last segments have collinear controls, which a path may
   draw as lines, so the law compares the cubics of the inner segments, whose
   [q1] is each point but the last. Each value is a mean of three coordinates,
   erring by a few roundings of the largest. *)
let b_spline (xs, ys) =
  let n = Array.length xs in
  let mean a j w1 w2 w3 =
    (w1 *. a.(j)) +. (w2 *. a.(j + 1)) +. (w3 *. a.(Int.min (n - 1) (j + 2)))
  in
  let expected =
    List.init (n - 1) (fun j ->
        let c1 a = mean a j (2. /. 3.) (1. /. 3.) 0.
        and c2 a = mean a j (1. /. 3.) (2. /. 3.) 0.
        and e a = mean a j (1. /. 6.) (4. /. 6.) (1. /. 6.) in
        C (c1 xs, c1 ys, c2 xs, c2 ys, e xs, e ys))
  in
  let actual =
    List.filter
      (function C _ -> true | _ -> false)
      (segs (Curve.path Curve.basis xs ys))
  in
  equal
    (segs_within (8. *. epsilon_float *. magnitude (xs, ys)))
    expected actual

let basis =
  group "basis"
    [
      prop "starts at the first point and ends at the last" (gen_finite 2)
        basis_ends;
      prop "is the B-spline of its points, the ends counted three times"
        ~examples:[ (xs5, ys5) ]
        (gen_finite 3) b_spline;
    ]

(* Drawing *)

let path_group =
  group "path"
    [
      cases "of no points is empty" ~id:name curves (fun c ->
          equal path Path.empty (Curve.path c [||] [||]));
      cases "raises on lengths that differ" ~id:name curves (fun c ->
          invalid (fun () -> Curve.path c [| 1.; 2. |] [| 1. |]);
          invalid (fun () -> Curve.path c [| 1. |] [| 1.; 2. |]));
    ]

(* [reversed start segs] is the segments [segs], drawn from [start], drawn
   backwards. *)
let reversed start segs =
  let rec go from acc = function
    | [] -> acc
    | L (x, y) :: rest -> go (x, y) (L (fst from, snd from) :: acc) rest
    | C (a, b, c, d, x, y) :: rest ->
        go (x, y) (C (c, d, a, b, fst from, snd from) :: acc) rest
    | (M _ | Z) :: _ -> fail "not a segment"
  in
  go start [] segs

let area_law ((xs, ys), (x0, y0)) =
  let n = Int.min (Array.length xs) (Array.length x0) in
  let xs = Array.sub xs 0 n and ys = Array.sub ys 0 n in
  let x0 = Array.sub x0 0 n and y0 = Array.sub y0 0 n in
  List.iter
    (fun c ->
      let top = segs (Curve.path c xs ys)
      and base = segs (Curve.path c x0 y0) in
      let expected =
        top
        @ L (x0.(n - 1), y0.(n - 1))
          :: reversed (x0.(0), y0.(0)) (List.tl base)
        @ [ Z ]
      in
      equal ~msg:(name c) segs_exact expected
        (segs (Curve.area c ~x0 ~y0 xs ys)))
    curves

(* Areas over points of which some are missing, each by one non-finite
   coordinate among its four, as [(x0, y0, xs, ys)]. *)
let gen_gappy_area =
  let missing = Gen.of_list ~pp:pp_float [ nan; infinity; neg_infinity ] in
  let quad = Gen.pair (Gen.pair coord coord) (Gen.pair coord coord) in
  let point =
    Gen.frequency
      [
        (2, quad);
        ( 1,
          Gen.map
            (fun ((k, v), ((a, b), (c, d))) ->
              match k with
              | 0 -> ((v, b), (c, d))
              | 1 -> ((a, v), (c, d))
              | 2 -> ((a, b), (v, d))
              | _ -> ((a, b), (c, v)))
            (Gen.pair (Gen.pair (Gen.int_range 0 3) missing) quad) );
      ]
  in
  let pp ppf (x0, y0, xs, ys) =
    Format.fprintf ppf "baseline %a,@ top %a" pp_points (x0, y0) pp_points
      (xs, ys)
  in
  Gen.with_pp pp
    (Gen.map
       (fun pts ->
         ( Array.map (fun ((x0, _), _) -> x0) pts,
           Array.map (fun ((_, y0), _) -> y0) pts,
           Array.map (fun (_, (x, _)) -> x) pts,
           Array.map (fun (_, (_, y)) -> y) pts ))
       (Gen.array ~size:(Gen.int_range 0 15) point))

let area_runs_alone (x0, y0, xs, ys) =
  let n = Array.length xs in
  let held i =
    Float.is_finite x0.(i)
    && Float.is_finite y0.(i)
    && Float.is_finite xs.(i)
    && Float.is_finite ys.(i)
  in
  let breaks a = Array.exists (fun v -> not (Float.is_finite v)) a in
  cover "x0 breaks" (breaks x0);
  cover "y0 breaks" (breaks y0);
  cover "xs breaks" (breaks xs);
  cover "ys breaks" (breaks ys);
  List.iter
    (fun c ->
      let rec go i first p =
        if i = n || not (held i) then
          let p =
            if i > first then
              let sub a = Array.sub a first (i - first) in
              Path.append
                (Curve.area c ~x0:(sub x0) ~y0:(sub y0) (sub xs) (sub ys))
                p
            else p
          in
          if i = n then p else go (i + 1) (i + 1) p
        else go (i + 1) first p
      in
      equal ~msg:(name c) path (go 0 0 Path.empty) (Curve.area c ~x0 ~y0 xs ys))
    curves

(* [shoelace segs] is the area that the polygon of the points of [segs]
   encloses, by the shoelace formula. *)
let shoelace segs =
  let pts =
    List.filter_map
      (function M (x, y) | L (x, y) -> Some (x, y) | _ -> None)
      segs
  in
  match pts with
  | [] -> 0.
  | first :: _ ->
      let rec go acc = function
        | (x, y) :: ((x', y') :: _ as rest) ->
            go (acc +. ((x *. y') -. (x' *. y))) rest
        | [ (x, y) ] -> acc +. ((x *. snd first) -. (fst first *. y))
        | [] -> acc
      in
      Float.abs (go 0. pts) /. 2.

(* Columns of a linear area over a shared x, its top at or above its baseline:
   increasing [xs], each baseline height, and each height above it. *)
let gen_columns =
  let column =
    Gen.triple (Gen.int_range 1 3) (Gen.int_range (-3) 3) (Gen.int_range 0 4)
  in
  let pp ppf cols =
    Format.pp_print_list
      (fun ppf (w, b, h) -> Format.fprintf ppf "(%d, %d, %d)" w b h)
      ppf cols
  in
  Gen.with_pp pp (Gen.list ~size:(Gen.int_range 2 8) column)

(* Integer coordinates below a hundred make every product and sum, and their
   halves, exact. *)
let shoelace_law cols =
  let n = List.length cols in
  let xs = Array.make n 0. and y0 = Array.make n 0. and ys = Array.make n 0. in
  List.iteri
    (fun i (w, b, h) ->
      xs.(i) <- (if i = 0 then 0. else xs.(i - 1)) +. Float.of_int w;
      y0.(i) <- Float.of_int b;
      ys.(i) <- Float.of_int (b + h))
    cols;
  let trapezoids = ref 0. in
  for i = 0 to n - 2 do
    let h i = ys.(i) -. y0.(i) in
    trapezoids :=
      !trapezoids +. ((xs.(i + 1) -. xs.(i)) *. (h i +. h (i + 1)) /. 2.)
  done;
  equal float_exact !trapezoids
    (shoelace (segs (Curve.area Curve.linear ~x0:xs ~y0 xs ys)))

(* Each array shorter, then longer, than the three others. *)
let area_lengths c =
  let a = [| 0.; 1. |] and b = [| 0. |] in
  invalid (fun () -> Curve.area c ~x0:b ~y0:a a a);
  invalid (fun () -> Curve.area c ~x0:a ~y0:b a a);
  invalid (fun () -> Curve.area c ~x0:a ~y0:a b a);
  invalid (fun () -> Curve.area c ~x0:a ~y0:a a b);
  invalid (fun () -> Curve.area c ~x0:a ~y0:b b b);
  invalid (fun () -> Curve.area c ~x0:b ~y0:a b b);
  invalid (fun () -> Curve.area c ~x0:b ~y0:b a b);
  invalid (fun () -> Curve.area c ~x0:b ~y0:b b a)

let area =
  group "area"
    [
      prop "is its top, a line, its baseline backwards and a close"
        ~examples:
          [
            ( ([| 0.; 1.; 2. |], [| 1.; 2.; 3. |]),
              ([| 0.; 1.; 2. |], [| 0.; -1.; -2. |]) );
          ]
        (Gen.pair (gen_finite 2) (gen_finite 2))
        area_law;
      prop "over a shared x encloses the integral between its linear curves"
        gen_columns shoelace_law;
      prop "draws each run alone"
        ~examples:
          [
            ( [| 0.; 1.; 2.; 3.; 4. |],
              [| 0.; 0.; nan; 0.; 0. |],
              [| 0.; 1.; 2.; 3.; 4. |],
              [| 1.; 1.; 1.; 1.; 1. |] );
          ]
        gen_gappy_area area_runs_alone;
      cases "is empty without a run of two points" ~id:name curves (fun c ->
          equal path Path.empty
            (Curve.area c ~x0:[| 0.; 1. |] ~y0:[| 0.; nan |] [| 0.; 1. |]
               [| 1.; 1. |]));
      cases "raises on lengths that differ" ~id:name curves area_lengths;
    ]

(* Comparing and formatting *)

let comparing =
  [
    group "equal"
      [
        prop "is true iff the curves are the same"
          (Gen.pair (Gen.int_range 0 8) (Gen.int_range 0 8))
          (fun (i, j) ->
            let c = List.nth curves i and c' = List.nth curves j in
            if i = j then equal curve c c' else not_equal curve c c');
      ];
    group "pp"
      [
        cases "formats the name of the curve" ~id:snd
          (List.combine curves
             [
               "linear";
               "step_after";
               "step_before";
               "step_mid";
               "monotone_x";
               "monotone_y";
               "natural";
               "catmull_rom";
               "basis";
             ])
          (fun (c, n) -> equal string n (name c));
      ];
  ]

let () =
  exit
    (run "Curve"
       ([ affine; missing; curves_group; linear ]
       @ steps @ monotone
       @ [ natural; catmull_rom; basis; path_group; area ]
       @ comparing))
