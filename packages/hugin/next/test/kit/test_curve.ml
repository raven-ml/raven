(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_next_gg
open Hugin_next_kit

let invalid f = raises_match (Exn.invalid_arg ?substring:None) f
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

let near a b =
  Float.equal a b
  || Float.abs (a -. b) <= 1e-9 *. (1. +. Float.abs a +. Float.abs b)

let segs_exact = list (Testable.make ~pp:pp_seg ~equal:(seg_equal Float.equal))
let segs_near = list (Testable.make ~pp:pp_seg ~equal:(seg_equal near))
let path = Testable.make ~pp:Path.pp ~equal:Path.equal

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

(* Points, [-0.] equal to [0.]: [Path.fold] gives [0.] for a stored [-0.]. *)
let point =
  let coord = Testable.make ~pp:pp_float ~equal:Float.equal in
  pair coord coord

let curve = Testable.make ~pp:Curve.pp ~equal:Curve.equal
let name c = Format.asprintf "%a" Curve.pp c
let lines = Curve.[ linear; step_after; step_before; step_mid ]
let smooth = Curve.[ monotone_x; monotone_y; natural; catmull_rom; basis ]
let curves = lines @ smooth
let through = List.filter (fun c -> not (Curve.equal c Curve.basis)) curves

(* Generators *)

let pp_points ppf (xs, ys) =
  Format.fprintf ppf "@[<1>[%a]@]"
    (Format.pp_print_array ~pp_sep:Format.pp_print_space (fun ppf (x, y) ->
         Format.fprintf ppf "(%h, %h)" x y))
    (Array.map2 (fun x y -> (x, y)) xs ys)

let gen_curve = Gen.of_list ~pp:Curve.pp curves

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

let gen_finite =
  Gen.one_of
    [
      gen_points (Gen.int_range 2 10);
      gen_points ~coord:small (Gen.int_range 2 10);
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
  let missing = Gen.of_list [ nan; infinity; neg_infinity ] in
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

(* Goldens *)

let xs5 = [| 0.; 1.; 2.; 3.; 4. |]
let ys5 = [| 0.; 2.; 1.; 3.; 3. |]

(* d3-shape 3.2 draws [linear], the steps, [natural], [catmull_rom] and [basis]
   through these points with the same segments, but for the data points that
   [step_mid] passes through on its flat parts and d3 leaves out. Its
   [monotone_x] takes another slope at the ends. *)
let goldens () =
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

(* Runs *)

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

let runs_alone (c, pts) =
  let expected =
    List.fold_left
      (fun p (xs, ys) -> Path.append (Curve.path c xs ys) p)
      Path.empty (runs pts)
  in
  cover "a run breaks" (List.length (runs pts) >= 2);
  equal path expected (Curve.path c (fst pts) (snd pts))

let two_points c =
  let expected =
    match name c with
    | "step_after" -> [ M (1., 2.); L (4., 2.); L (4., 6.) ]
    | "step_before" -> [ M (1., 2.); L (1., 6.); L (4., 6.) ]
    | "step_mid" -> [ M (1., 2.); L (2.5, 2.); L (2.5, 6.); L (4., 6.) ]
    | _ -> [ M (1., 2.); L (4., 6.) ]
  in
  equal segs_exact expected (segs (Curve.path c [| 1.; 4. |] [| 2.; 6. |]))

(* The end points of the segments include every point, in order, a point equal
   to the one before it counting as reached. *)
let passes_through (c, (xs, ys)) =
  let ends = List.filter_map seg_end (segs (Curve.path c xs ys)) in
  let at i (x, y) = Float.equal x xs.(i) && Float.equal y ys.(i) in
  let rec sub i ends =
    if i = Array.length xs then ()
    else if i > 0 && at i (xs.(i - 1), ys.(i - 1)) then sub (i + 1) ends
    else
      match ends with
      | [] -> failf "point %d (%h, %h) is not on the path" i xs.(i) ys.(i)
      | e :: rest -> if at i e then sub (i + 1) rest else sub i rest
  in
  sub 0 ends

let basis_ends (xs, ys) =
  let s = segs (Curve.path Curve.basis xs ys) in
  let n = Array.length xs in
  equal (option point) (Some (xs.(0), ys.(0))) (seg_end (List.hd s));
  equal (option point)
    (Some (xs.(n - 1), ys.(n - 1)))
    (seg_end (List.nth s (List.length s - 1)))

let runs_group =
  group "runs"
    [
      test "path draws every curve as d3 does" goldens;
      prop "each run is drawn alone" (Gen.pair gen_curve gen_gappy) runs_alone;
      cases "a run of one point draws nothing" ~name curves (fun c ->
          equal path Path.empty (Curve.path c [| 1. |] [| 2. |]);
          equal path Path.empty
            (Curve.path c [| nan; 1.; 2.; 3. |] [| 0.; 1.; infinity; 3. |]));
      cases "no points draw nothing" ~name curves (fun c ->
          equal path Path.empty (Curve.path c [||] [||]));
      cases "two points" ~name curves two_points;
      prop "every curve but basis passes through its points"
        (Gen.pair (Gen.of_list ~pp:Curve.pp through) gen_finite)
        passes_through;
      prop "basis starts at the first point and ends at the last" gen_finite
        basis_ends;
      cases "path raises on lengths that differ" ~name curves (fun c ->
          invalid (fun () -> Curve.path c [| 1.; 2. |] [| 1. |]);
          invalid (fun () -> Curve.path c [| 1. |] [| 1.; 2. |]));
    ]

(* Monotone curves *)

let first_cubic c xs ys =
  match segs (Curve.path c xs ys) with
  | _ :: (C _ as seg) :: _ -> seg
  | s -> failf "no first cubic in %a" (Format.pp_print_list pp_seg) s

(* Hand cases of Steffen's slopes, from the formulas. *)
let steffen_cases =
  [
    (* Collinear points: every slope is the line's. *)
    ( "collinear",
      [| 0.; 1.; 2. |],
      [| 0.; 1.; 2. |],
      C (1. /. 3., 1. /. 3., 2. /. 3., 2. /. 3., 1., 1.) );
    (* p = 1 × 1.5 - 9 × 0.5 = -3 is against s = 1: the end slope is 0. *)
    ( "end slope against the segment",
      [| 0.; 1.; 2. |],
      [| 0.; 1.; 10. |],
      C (1. /. 3., 0., 2. /. 3., 1. -. (2. /. 3.), 1., 1.) );
    (* p = 1.5 + 1 = 2.5 exceeds 2 s = 2: the end slope is 2. The interior slope
       is 0 since the segments' slopes have opposite signs. *)
    ( "end slope capped at twice the segment's",
      [| 0.; 1.; 2. |],
      [| 0.; 1.; -1. |],
      C (1. /. 3., 2. /. 3., 2. /. 3., 1., 1., 1.) );
  ]

let steffen_case (_, xs, ys, expected) =
  equal segs_near [ expected ] [ first_cubic Curve.monotone_x xs ys ]

(* [bezier t a b c d] is the cubic Bézier of control values [a] to [d] at
   [t]. *)
let bezier t a b c d =
  let u = 1. -. t in
  (u *. u *. u *. a)
  +. (3. *. u *. u *. t *. b)
  +. (3. *. u *. t *. t *. c)
  +. (t *. t *. t *. d)

(* Between consecutive points the curve stays within their x and their y. *)
let never_overshoots (xs, ys) =
  let s = segs (Curve.path Curve.monotone_x xs ys) in
  let within lo hi v =
    let lo, hi = (Float.min lo hi, Float.max lo hi) in
    let slack = 1e-9 *. (1. +. Float.abs lo +. Float.abs hi) in
    lo -. slack <= v && v <= hi +. slack
  in
  ignore
    (List.fold_left
       (fun (x0, y0) seg ->
         match seg with
         | M (x, y) | L (x, y) -> (x, y)
         | C (ax, ay, bx, by, x, y) ->
             for k = 0 to 32 do
               let t = float_of_int k /. 32. in
               let bx' = bezier t x0 ax bx x and by' = bezier t y0 ay by y in
               if not (within x0 x bx' && within y0 y by') then
                 failf "the cubic from (%h, %h) to (%h, %h) reaches (%h, %h)" x0
                   y0 x y bx' by'
             done;
             cover "a cubic" true;
             (x, y)
         | Z -> (x0, y0))
       (nan, nan) s)

let swap = { Affine.xx = 0.; yx = 1.; xy = 1.; yy = 0.; x0 = 0.; y0 = 0. }

let monotone_y_swaps (xs, ys) =
  equal path
    (Path.transform swap (Curve.path Curve.monotone_x ys xs))
    (Curve.path Curve.monotone_y xs ys)

let pieces () =
  let xs = [| 0.; 1.; 2.; 1.; 0. |] and ys = [| 0.; 1.; 0.; -1.; 0. |] in
  let s = segs (Curve.path Curve.monotone_x xs ys) in
  equal int 5 (List.length s);
  let ends = List.filter_map seg_end s in
  equal (list point) [ (0., 0.); (1., 1.); (2., 0.); (1., -1.); (0., 0.) ] ends

let equal_x () =
  let xs = [| 0.; 1.; 1.; 2. |] and ys = [| 0.; 1.; 2.; 3. |] in
  match segs (Curve.path Curve.monotone_x xs ys) with
  | [ M (0., 0.); L (1., 1.); L (1., 2.); L (2., 3.) ] -> ()
  | s ->
      failf "pieces of one segment and equal x are lines, got %a"
        (Format.pp_print_list pp_seg)
        s

let monotone =
  group "monotone"
    [
      cases "Steffen's slopes"
        ~name:(fun (n, _, _, _) -> n)
        steffen_cases steffen_case;
      prop "monotone_x never overshoots" gen_finite never_overshoots;
      prop "monotone_y is monotone_x with x and y exchanged" gen_finite
        monotone_y_swaps;
      test "x turning back starts a piece" pieces;
      test "a piece of one segment and a segment between equal x are lines"
        equal_x;
    ]

(* Natural, Catmull–Rom and basis *)

(* The natural spline has a continuous first derivative at interior points, and
   a zero second derivative at both ends. *)
let natural_smooth (xs, ys) =
  assume (Array.length xs >= 3);
  let s = List.tl (segs (Curve.path Curve.natural xs ys)) in
  let cubics =
    Array.of_list
      (List.map
         (function
           | C (a, b, c, d, e, f) -> (a, b, c, d, e, f)
           | _ -> fail "not a cubic")
         s)
  in
  let n = Array.length cubics in
  let close a b =
    equal (float (1e-9 *. (1. +. Float.abs a +. Float.abs b))) a b
  in
  for k = 0 to n - 2 do
    let _, _, c2x, c2y, px, py = cubics.(k) in
    let c1x, c1y, _, _, _, _ = cubics.(k + 1) in
    close (px -. c2x) (c1x -. px);
    close (py -. c2y) (c1y -. py)
  done;
  let c1x, c1y, c2x, c2y, _, _ = cubics.(0) in
  close 0. (xs.(0) -. (2. *. c1x) +. c2x);
  close 0. (ys.(0) -. (2. *. c1y) +. c2y);
  let c1x, c1y, c2x, c2y, x, y = cubics.(n - 1) in
  close 0. (c1x -. (2. *. c2x) +. x);
  close 0. (c1y -. (2. *. c2y) +. y)

let dedup (xs, ys) =
  let n = Array.length xs in
  let keep =
    List.filter
      (fun i -> i = 0 || xs.(i) <> xs.(i - 1) || ys.(i) <> ys.(i - 1))
      (List.init n Fun.id)
  in
  ( Array.of_list (List.map (fun i -> xs.(i)) keep),
    Array.of_list (List.map (fun i -> ys.(i)) keep) )

let catmull_rom_repeats (xs, ys) =
  let xs', ys' = dedup (xs, ys) in
  assume (Array.length xs' >= 2);
  cover "repeats" (Array.length xs' < Array.length xs);
  equal path
    (Curve.path Curve.catmull_rom xs' ys')
    (Curve.path Curve.catmull_rom xs ys)

let catmull_rom_ends (xs, ys) =
  let xs, ys = dedup (xs, ys) in
  assume (Array.length xs >= 3);
  let n = Array.length xs in
  let s = segs (Curve.path Curve.catmull_rom xs ys) in
  (match List.nth s 1 with
  | C (c1x, c1y, _, _, _, _) -> equal point (xs.(0), ys.(0)) (c1x, c1y)
  | _ -> fail "not a cubic");
  match List.nth s (List.length s - 1) with
  | C (_, _, c2x, c2y, _, _) -> equal point (xs.(n - 1), ys.(n - 1)) (c2x, c2y)
  | _ -> fail "not a cubic"

let others =
  group "natural, catmull_rom and basis"
    [
      prop "natural is smooth with free ends" gen_finite natural_smooth;
      prop "catmull_rom counts consecutive equal points once"
        (gen_points ~coord:small (Gen.int_range 2 10))
        catmull_rom_repeats;
      prop "catmull_rom puts its end control points on its end points"
        gen_finite catmull_rom_ends;
      test "catmull_rom draws equal points as linear does" (fun () ->
          let xs = [| 1.; 1.; 1. |] and ys = [| 2.; 2.; 2. |] in
          equal path
            (Curve.path Curve.linear xs ys)
            (Curve.path Curve.catmull_rom xs ys));
    ]

(* Affine maps *)

let map_points m (xs, ys) =
  let p = Array.map2 (fun x y -> P2.transform m (P2.v x y)) xs ys in
  (Array.map P2.x p, Array.map P2.y p)

let pp_affine = Affine.pp

let commutes c m pts =
  let xs', ys' = map_points m pts in
  equal segs_near
    (segs (Path.transform m (Curve.path c (fst pts) (snd pts))))
    (segs (Curve.path c xs' ys'))

let gen_affine =
  let k = Gen.float_range (-3.) 3. in
  Gen.with_pp pp_affine
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
  Gen.with_pp pp_affine
    (Gen.map
       (fun ((sx, sy), (x0, y0)) -> Affine.(translate x0 y0 * scale sx sy))
       (Gen.pair (Gen.pair factor factor) (Gen.pair coord coord)))

let gen_similarity =
  Gen.with_pp pp_affine
    (Gen.map
       (fun ((a, k, mirror), (x0, y0)) ->
         Affine.(
           translate x0 y0 * rotate a * scale k (if mirror then -.k else k)))
       (Gen.pair
          (Gen.triple
             (Gen.float_range 0. (2. *. Float.pi))
             (Gen.float_range 0.1 10.) Gen.bool)
          (Gen.pair coord coord)))

(* At [1e-30] the absolute floor of [segs_near] accepts any path, so the path
   through the scaled points is mapped back before comparing. *)
let catmull_rom_tiny pts =
  let xs, ys = map_points (Affine.scale 1e-30 1e-30) pts in
  equal segs_near
    (segs (Curve.path Curve.catmull_rom (fst pts) (snd pts)))
    (segs
       (Path.transform (Affine.scale 1e30 1e30)
          (Curve.path Curve.catmull_rom xs ys)))

let affine =
  group "affine maps"
    [
      prop "linear, natural and basis commute with every affine map"
        (Gen.triple
           (Gen.of_list ~pp:Curve.pp Curve.[ linear; natural; basis ])
           gen_affine gen_moderate)
        (fun (c, m, pts) -> commutes c m pts);
      prop "steps and monotone curves commute with scaling each axis"
        (Gen.triple
           (Gen.of_list ~pp:Curve.pp
              Curve.
                [ step_after; step_before; step_mid; monotone_x; monotone_y ])
           gen_axes gen_moderate)
        (fun (c, m, pts) -> commutes c m pts);
      prop "catmull_rom commutes with similarities"
        (Gen.pair gen_similarity gen_moderate) (fun (m, pts) ->
          commutes Curve.catmull_rom m pts);
      prop "catmull_rom commutes with scaling by 1e-30" gen_moderate
        catmull_rom_tiny;
    ]

(* Comparing and formatting *)

let comparing =
  group "comparing and formatting"
    [
      cases "a curve equals itself" ~name curves (fun c -> equal curve c c);
      test "distinct curves are unequal" (fun () ->
          List.iteri
            (fun i c ->
              List.iteri
                (fun j c' ->
                  if i <> j then
                    not_equal ~msg:(name c ^ " " ^ name c') curve c c')
                curves)
            curves);
      test "pp names a curve" (fun () ->
          expect (String.concat " " (List.map name curves))
          @@ __POS_OF__
               {| linear step_after step_before step_mid monotone_x monotone_y natural catmull_rom basis |});
    ]

let () = exit (run "Curve" [ runs_group; monotone; others; affine; comparing ])
