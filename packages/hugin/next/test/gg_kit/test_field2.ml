(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_next_gg
open Hugin_next_gg_kit

let nan = Float.nan

(* Fields from rows of samples. *)
let tensor rows =
  let ny = List.length rows in
  let nx = match rows with [] -> 0 | r :: _ -> Array.length r in
  Nx.create Nx.float64 [| ny; nx |] (Array.concat rows)

let field ?xs ?ys rows = Field2.v ?xs ?ys (tensor rows)
let domain f = Field2.isoband ~lo:neg_infinity ~hi:infinity f

(* Witnesses print every digit, so that a failure shows the difference. *)

let pp_pts ppf pts =
  Format.fprintf ppf "@[<1>[";
  List.iteri
    (fun i (x, y) ->
      if i > 0 then Format.fprintf ppf ";@ ";
      Format.fprintf ppf "(%.17g, %.17g)" x y)
    pts;
  Format.fprintf ppf "]@]"

let ring_pts r =
  List.init (Ring2.length r) (fun i -> (Ring2.x r i, Ring2.y r i))

let pp_pgon ppf p =
  Format.fprintf ppf "@[<v>%a@]"
    (Format.pp_print_list (fun ppf r -> pp_pts ppf (ring_pts r)))
    (Pgon2.rings p)

let pgon = Testable.make ~pp:pp_pgon ~equal:Pgon2.equal
let path = Testable.make ~pp:Path.pp ~equal:Path.equal
let near = float_rel ~rel:1e-9 ~abs:1e-9

(* The curves of a path: points, and whether the curve is closed. *)
let curves p =
  let flush cur acc =
    match cur with
    | None -> acc
    | Some (pts, closed) -> (List.rev pts, closed) :: acc
  in
  let cur, acc =
    Path.fold
      ~move:(fun (cur, acc) x y -> (Some ([ (x, y) ], false), flush cur acc))
      ~line:(fun (cur, acc) x y ->
        match cur with
        | Some (pts, c) -> (Some ((x, y) :: pts, c), acc)
        | None -> failf "a line without a start")
      ~cubic:(fun _ _ _ _ _ _ _ -> failf "a curve in an isoline")
      ~close:(fun (cur, acc) ->
        match cur with
        | Some (pts, _) -> (Some (pts, true), acc)
        | None -> (cur, acc))
      (None, []) p
  in
  List.rev (flush cur acc)

let pp_curves ppf cs =
  Format.fprintf ppf "@[<v>%a@]"
    (Format.pp_print_list (fun ppf (pts, closed) ->
         Format.fprintf ppf "%s %a"
           (if closed then "closed" else "open")
           pp_pts pts))
    cs

(* A ring or a closed curve compared up to its first point. *)
let rotations pts =
  let n = List.length pts in
  let a = Array.of_list pts in
  List.init n (fun k -> List.init n (fun i -> a.((i + k) mod n)))

let same_cycle a b = List.mem b (rotations a)

let same_rings expected p =
  let actual = List.map ring_pts (Pgon2.rings p) in
  let pp = Format.pp_print_list pp_pts in
  let ok =
    List.length expected = List.length actual
    && List.for_all (fun e -> List.exists (same_cycle e) actual) expected
  in
  if not ok then
    failf
      "@[<v>rings differ up to order and first points@,\
       expected@,\
      \  %a@,\
       actual@,\
      \  %a@]"
      pp expected pp actual

(* Curves compared up to their order and the first point of closed ones. *)
let same_curves expected p =
  let actual = curves p in
  let same (e, c) (a, d) = c = d && if c then same_cycle e a else e = a in
  let ok =
    List.length expected = List.length actual
    && List.for_all (fun e -> List.exists (same e) actual) expected
  in
  if not ok then
    failf "@[<v>curves differ up to order@,expected@,  %a@,actual@,  %a@]"
      pp_curves expected pp_curves actual

let signed_area pts =
  let a = Array.of_list pts in
  let n = Array.length a in
  let s = ref 0. in
  for i = 0 to n - 1 do
    let x0, y0 = a.(i) and x1, y1 = a.((i + 1) mod n) in
    s := !s +. ((x0 *. y1) -. (x1 *. y0))
  done;
  0.5 *. !s

(* Constructing *)

let rejects name z = test name (fun () -> raises_match Exn.invalid_arg z)

let constructing =
  group "constructing"
    [
      rejects "v rejects a vector" (fun () ->
          Field2.v (Nx.zeros Nx.float64 [| 4 |]));
      rejects "v rejects three dimensions" (fun () ->
          Field2.v (Nx.zeros Nx.float64 [| 2; 2; 2 |]));
      rejects "v rejects a complex tensor" (fun () ->
          Field2.v (Nx.zeros Nx.complex64 [| 2; 2 |]));
      rejects "v rejects a boolean tensor" (fun () ->
          Field2.v (Nx.zeros Nx.bool [| 2; 2 |]));
      rejects "v rejects xs of another length" (fun () ->
          field ~xs:[| 0.; 1. |] [ [| 0.; 0.; 0. |] ]);
      rejects "v rejects ys of another length" (fun () ->
          field ~ys:[| 0. |] [ [| 0. |]; [| 0. |] ]);
      rejects "v rejects a NaN coordinate" (fun () ->
          field ~xs:[| 0.; nan |] [ [| 0.; 0. |] ]);
      rejects "v rejects an infinite coordinate" (fun () ->
          field ~ys:[| 0.; infinity |] [ [| 0. |]; [| 0. |] ]);
      rejects "v rejects repeated coordinates" (fun () ->
          field ~xs:[| 0.; 1.; 1. |] [ [| 0.; 0.; 0. |] ]);
      rejects "v rejects coordinates that turn back" (fun () ->
          field ~xs:[| 0.; 2.; 1. |] [ [| 0.; 0.; 0. |] ]);
      rejects "v rejects coordinates spanning more than the largest float"
        (fun () -> field ~xs:[| -1e308; 1e308 |] [ [| 0.; 0. |] ]);
      test "v reads integer and float16 tensors" (fun () ->
          let i = Nx.create Nx.int32 [| 2; 2 |] [| 0l; 0l; 2l; 2l |] in
          let h = Nx.cast Nx.float16 (tensor [ [| 0.; 0. |]; [| 2.; 2. |] ]) in
          equal near 0.5
            (Pgon2.area (Field2.isoband ~lo:1. ~hi:infinity (Field2.v i)));
          equal near 0.5
            (Pgon2.area (Field2.isoband ~lo:1. ~hi:infinity (Field2.v h))));
      test "v defaults coordinates to indices" (fun () ->
          let f = field [ [| 0.; 0.; 0. |]; [| 0.; 0.; 0. |] ] in
          let b = require_some (Pgon2.bounds (domain f)) in
          equal float_exact 0. (Box2.minx b);
          equal float_exact 0. (Box2.miny b);
          equal float_exact 2. (Box2.maxx b);
          equal float_exact 1. (Box2.maxy b));
      test "v copies the coordinates" (fun () ->
          let xs = [| 0.; 1. |] in
          let f = field ~xs [ [| 0.; 0. |]; [| 0.; 0. |] ] in
          xs.(1) <- 10.;
          equal near 1. (Pgon2.area (domain f)));
    ]

(* Golden cases, expected values from the definitions. *)

(* The L1 cone [4 - |x - 2| - |y - 2|] is linear on every cell, so its regions
   are exact diamonds. *)
let cone =
  field
    (List.init 5 (fun i ->
         Array.init 5 (fun j ->
             4.
             -. Float.abs (Float.of_int (i - 2))
             -. Float.abs (Float.of_int (j - 2)))))

let cone_cases () =
  equal near 4.5 (Pgon2.area (Field2.isoband ~lo:2.5 ~hi:infinity cone));
  (* The diamond of radius 2.5 less its four tips beyond the grid. *)
  equal near
    (12.5 -. 1. -. 4.5)
    (Pgon2.area (Field2.isoband ~lo:1.5 ~hi:2.5 cone));
  match curves (Field2.isoline 2.5 cone) with
  | [ (pts, true) ] ->
      equal int 12 (List.length pts);
      greater float_exact ~than:0. (signed_area pts)
  | cs -> failf "one closed curve expected, got %a" pp_curves cs

let saddle = field [ [| 1.; 0. |]; [| 0.; 1. |] ]

let saddle_cases () =
  equal ~msg:"joined below the mean" near 0.84
    (Pgon2.area (Field2.isoband ~lo:0.4 ~hi:infinity saddle));
  equal ~msg:"joined at the mean" near 0.75
    (Pgon2.area (Field2.isoband ~lo:0.5 ~hi:infinity saddle));
  equal ~msg:"split above the mean" near 0.16
    (Pgon2.area (Field2.isoband ~lo:0.6 ~hi:infinity saddle));
  equal ~msg:"joined: one ring" int 1
    (List.length (Pgon2.rings (Field2.isoband ~lo:0.5 ~hi:infinity saddle)));
  equal ~msg:"split: two rings" int 2
    (List.length (Pgon2.rings (Field2.isoband ~lo:0.6 ~hi:infinity saddle)))

let saddle_joined_curves () =
  same_curves
    [ ([ (0.5, 0.); (1., 0.5) ], false); ([ (0.5, 1.); (0., 0.5) ], false) ]
    (Field2.isoline 0.5 saddle)

let saddle_split_curves () =
  same_curves
    [ ([ (0.4, 0.); (0., 0.4) ], false); ([ (0.6, 1.); (1., 0.6) ], false) ]
    (Field2.isoline 0.6 saddle)

let other_saddle () =
  let f = field [ [| 0.; 1. |]; [| 1.; 0. |] ] in
  equal near 0.75 (Pgon2.area (Field2.isoband ~lo:0.5 ~hi:infinity f));
  equal near 0.16 (Pgon2.area (Field2.isoband ~lo:0.6 ~hi:infinity f))

let constant_cases () =
  let f = field [ [| 1.; 1.; 1. |]; [| 1.; 1.; 1. |] ] in
  let border = [ (0., 0.); (1., 0.); (2., 0.); (2., 1.); (1., 1.); (0., 1.) ] in
  same_rings [ border ] (Field2.isoband ~lo:0. ~hi:infinity f);
  same_rings [ border ] (Field2.isoband ~lo:1. ~hi:infinity f);
  same_rings [] (Field2.isoband ~lo:2. ~hi:infinity f);
  equal path Path.empty (Field2.isoline 1. f);
  equal path Path.empty (Field2.isoline 0. f)

let ties () =
  let ridge = field [ [| 0.; 0.; 0. |]; [| 1.; 1.; 1. |]; [| 0.; 0.; 0. |] ] in
  subtest "a ridge at the level has no region and no isoline" (fun () ->
      same_rings [] (Field2.isoband ~lo:1. ~hi:infinity ridge);
      equal path Path.empty (Field2.isoline 1. ridge);
      equal near 4. (Pgon2.area (Field2.isoband ~lo:neg_infinity ~hi:1. ridge)));
  subtest "a border row at the level draws no line" (fun () ->
      let f = field [ [| 1.; 1.; 1. |]; [| 0.; 0.; 0. |] ] in
      equal path Path.empty (Field2.isoline 1. f));
  subtest "an isolated peak at the level is not in its region" (fun () ->
      let f = field [ [| 0.; 0.; 0. |]; [| 0.; 1.; 0. |]; [| 0.; 0.; 0. |] ] in
      same_rings [] (Field2.isoband ~lo:1. ~hi:infinity f);
      equal path Path.empty (Field2.isoline 1. f));
  subtest "a peak at the level is not in its region at any coordinates"
    (fun () ->
      let c = [| -0.5; 0.3; 1.1 |] in
      let f =
        field ~xs:c ~ys:c
          [ [| 0.; 0.; 0. |]; [| 0.; 1.; 0. |]; [| 0.; 0.; 0. |] ]
      in
      same_rings [] (Field2.isoband ~lo:1. ~hi:infinity f));
  subtest "near levels whose crossings round onto a corner leave no band"
    (fun () ->
      let f =
        field ~ys:[| 0x1p52; 0x1p52 +. 1. |] [ [| 3.; 0. |]; [| 3.; 10. |] ]
      in
      same_rings [] (Field2.isoband ~lo:1. ~hi:2. f);
      equal near 1. (Pgon2.area (Field2.isoband ~lo:2. ~hi:infinity f)));
  subtest "an isolated pit at a level is a band's diamond" (fun () ->
      let f = field [ [| 2.; 2.; 2. |]; [| 2.; 1.; 2. |]; [| 2.; 2.; 2. |] ] in
      same_rings [] (Field2.isoband ~lo:neg_infinity ~hi:1. f);
      equal path Path.empty (Field2.isoline 1. f);
      same_rings
        [ [ (1., 0.); (2., 1.); (1., 2.); (0., 1.) ] ]
        (Field2.isoband ~lo:1. ~hi:2. f));
  subtest "a plateau's edge at the level is an isoline" (fun () ->
      let f = field [ [| 2.; 2.; 2. |]; [| 1.; 1.; 1. |]; [| 0.; 0.; 0. |] ] in
      same_curves
        [ ([ (2., 1.); (1., 1.); (0., 1.) ], false) ]
        (Field2.isoline 1. f);
      equal near 2. (Pgon2.area (Field2.isoband ~lo:1. ~hi:infinity f)));
  subtest "an isoline through corners at the level is one segment" (fun () ->
      let f = field [ [| 1.; 0. |]; [| 2.; 1. |] ] in
      same_curves [ ([ (0., 0.); (1., 1.) ], false) ] (Field2.isoline 1. f);
      equal near 0.5 (Pgon2.area (Field2.isoband ~lo:1. ~hi:infinity f)))

let missing () =
  subtest "a NaN cuts a diamond and opens the isolines at it" (fun () ->
      let f =
        field
          (List.init 5 (fun i ->
               Array.init 5 (fun j ->
                   if i = 2 && j = 2 then nan else Float.of_int j)))
      in
      equal near 14. (Pgon2.area (domain f));
      same_curves
        [
          ([ (2.5, 4.); (2.5, 3.); (2.5, 2.5) ], false);
          ([ (2.5, 1.5); (2.5, 1.); (2.5, 0.) ], false);
        ]
        (Field2.isoline 2.5 f));
  subtest "a NaN at a corner of the grid cuts its triangle" (fun () ->
      let f = field [ [| nan; 0.; 0. |]; [| 0.; 0.; 0. |]; [| 0.; 0.; 0. |] ] in
      equal near 3.5 (Pgon2.area (domain f)));
  subtest "cells with two missing samples are left out" (fun () ->
      let row k =
        Array.init 4 (fun j -> if k = 1 && (j = 1 || j = 2) then nan else 0.)
      in
      equal near 5. (Pgon2.area (domain (field (List.init 4 row)))));
  subtest "infinite samples are missing as NaN is" (fun () ->
      let with_ v =
        field [ [| 0.; 1.; 2. |]; [| 1.; v; 1. |]; [| 2.; 1.; 0. |] ]
      in
      equal pgon
        (Field2.isoband ~lo:0.5 ~hi:1.5 (with_ nan))
        (Field2.isoband ~lo:0.5 ~hi:1.5 (with_ infinity));
      equal pgon
        (Field2.isoband ~lo:0.5 ~hi:1.5 (with_ nan))
        (Field2.isoband ~lo:0.5 ~hi:1.5 (with_ neg_infinity)))

let degenerate () =
  subtest "a field of one row has no domain" (fun () ->
      let f = field [ [| 0.; 1.; 2. |] ] in
      same_rings [] (domain f);
      equal path Path.empty (Field2.isoline 1. f));
  subtest "a field of no sample has no domain" (fun () ->
      same_rings [] (domain (Field2.v (Nx.zeros Nx.float64 [| 0; 3 |]))));
  subtest "infinite levels" (fun () ->
      equal path Path.empty (Field2.isoline infinity cone);
      equal path Path.empty (Field2.isoline neg_infinity cone);
      same_rings [] (Field2.isoband ~lo:infinity ~hi:infinity cone);
      equal near 16. (Pgon2.area (domain cone)));
  subtest "lo = hi is empty" (fun () ->
      same_rings [] (Field2.isoband ~lo:2. ~hi:2. cone));
  subtest "NaN levels and lo > hi raise" (fun () ->
      raises_match Exn.invalid_arg (fun () -> Field2.isoline nan cone);
      raises_match Exn.invalid_arg (fun () ->
          Field2.isoband ~lo:nan ~hi:1. cone);
      raises_match Exn.invalid_arg (fun () ->
          Field2.isoband ~lo:0. ~hi:nan cone);
      raises_match Exn.invalid_arg (fun () -> Field2.isoband ~lo:2. ~hi:1. cone))

let overflowing_samples () =
  let f = field [ [| 1.5e308; -1.5e308 |]; [| 1.5e308; -1.5e308 |] ] in
  equal near (1. /. 6.) (Pgon2.area (Field2.isoband ~lo:1e308 ~hi:infinity f))

let mirrored () =
  let rows = [ [| 0.; 1.; 3. |]; [| 1.; 2.; 1. |]; [| 3.; 1.; 0. |] ] in
  let flip a =
    Array.init (Array.length a) (fun i -> a.(Array.length a - 1 - i))
  in
  let up = field rows in
  let left = field ~xs:[| 2.; 1.; 0. |] (List.map flip rows) in
  let down = field ~ys:[| 2.; 1.; 0. |] (List.rev rows) in
  let both =
    field ~xs:[| 2.; 1.; 0. |] ~ys:[| 2.; 1.; 0. |] (List.rev_map flip rows)
  in
  let bands f =
    List.map
      (fun (lo, hi) -> Field2.isoband ~lo ~hi f)
      [ (0.5, 1.5); (1.5, infinity) ]
  in
  List.iter
    (fun (name, f) ->
      List.iter2
        (fun expected actual ->
          equal ~msg:name near (Pgon2.area expected) (Pgon2.area actual);
          List.iter
            (fun r -> greater ~msg:name float_exact ~than:0. (Ring2.area r))
            (Pgon2.rings actual))
        (bands up) (bands f))
    [
      ("xs decreasing", left); ("ys decreasing", down); ("both decreasing", both);
    ]

let goldens =
  group "cases"
    [
      test "a cone's region is a diamond, its isoline closed and positive"
        cone_cases;
      test "a saddle joins its in corners iff their mean reaches the level"
        saddle_cases;
      test "a joined saddle's isolines cut off the out corners"
        saddle_joined_curves;
      test "a split saddle's isolines cut off the in corners"
        saddle_split_curves;
      test "the other saddle" other_saddle;
      test "a constant field is all in, or all out" constant_cases;
      test "samples equal to the level" ties;
      test "missing samples" missing;
      test "degenerate fields and levels" degenerate;
      test "decreasing coordinates mirror the bands" mirrored;
      test "samples whose difference overflows cross where they reach the level"
        overflowing_samples;
    ]

(* Agreement with d3-contour on its test fields, at 0.5, none of which has a
   saddle. d3 puts samples at half-integers and winds outer rings the other way;
   its rings repeat their first point. *)

let d3_ring pts =
  (* Reversed, without the repeated point. *)
  match List.rev pts with
  | _ :: rest -> List.map (fun (x, y) -> (x -. 0.5, y -. 0.5)) rest
  | [] -> []

let grid10 rows =
  field (List.map (fun r -> Array.of_list (List.map Float.of_int r)) rows)

let d3_cases =
  let z = [ 0; 0; 0; 0; 0; 0; 0; 0; 0; 0 ] in
  [
    ( "simple polygon",
      [
        z;
        z;
        z;
        [ 0; 0; 0; 1; 1; 1; 0; 0; 0; 0 ];
        [ 0; 0; 0; 1; 1; 1; 0; 0; 0; 0 ];
        [ 0; 0; 0; 1; 1; 1; 0; 0; 0; 0 ];
        [ 0; 0; 0; 1; 1; 1; 0; 0; 0; 0 ];
        [ 0; 0; 0; 1; 1; 1; 0; 0; 0; 0 ];
        z;
        z;
      ],
      [
        [
          (6., 7.5);
          (6., 6.5);
          (6., 5.5);
          (6., 4.5);
          (6., 3.5);
          (5.5, 3.);
          (4.5, 3.);
          (3.5, 3.);
          (3., 3.5);
          (3., 4.5);
          (3., 5.5);
          (3., 6.5);
          (3., 7.5);
          (3.5, 8.);
          (4.5, 8.);
          (5.5, 8.);
          (6., 7.5);
        ];
      ] );
    ( "polygon with a hole",
      [
        z;
        z;
        z;
        [ 0; 0; 0; 1; 1; 1; 0; 0; 0; 0 ];
        [ 0; 0; 0; 1; 0; 1; 0; 0; 0; 0 ];
        [ 0; 0; 0; 1; 0; 1; 0; 0; 0; 0 ];
        [ 0; 0; 0; 1; 0; 1; 0; 0; 0; 0 ];
        [ 0; 0; 0; 1; 1; 1; 0; 0; 0; 0 ];
        z;
        z;
      ],
      [
        [
          (6., 7.5);
          (6., 6.5);
          (6., 5.5);
          (6., 4.5);
          (6., 3.5);
          (5.5, 3.);
          (4.5, 3.);
          (3.5, 3.);
          (3., 3.5);
          (3., 4.5);
          (3., 5.5);
          (3., 6.5);
          (3., 7.5);
          (3.5, 8.);
          (4.5, 8.);
          (5.5, 8.);
          (6., 7.5);
        ];
        [
          (4.5, 7.);
          (4., 6.5);
          (4., 5.5);
          (4., 4.5);
          (4.5, 4.);
          (5., 4.5);
          (5., 5.5);
          (5., 6.5);
          (4.5, 7.);
        ];
      ] );
    ( "multipolygon",
      [
        z;
        z;
        z;
        [ 0; 0; 0; 1; 1; 0; 1; 0; 0; 0 ];
        [ 0; 0; 0; 1; 1; 0; 1; 0; 0; 0 ];
        [ 0; 0; 0; 1; 1; 0; 1; 0; 0; 0 ];
        [ 0; 0; 0; 1; 1; 0; 1; 0; 0; 0 ];
        [ 0; 0; 0; 1; 1; 0; 1; 0; 0; 0 ];
        z;
        z;
      ],
      [
        [
          (5., 7.5);
          (5., 6.5);
          (5., 5.5);
          (5., 4.5);
          (5., 3.5);
          (4.5, 3.);
          (3.5, 3.);
          (3., 3.5);
          (3., 4.5);
          (3., 5.5);
          (3., 6.5);
          (3., 7.5);
          (3.5, 8.);
          (4.5, 8.);
          (5., 7.5);
        ];
        [
          (7., 7.5);
          (7., 6.5);
          (7., 5.5);
          (7., 4.5);
          (7., 3.5);
          (6.5, 3.);
          (6., 3.5);
          (6., 4.5);
          (6., 5.5);
          (6., 6.5);
          (6., 7.5);
          (6.5, 8.);
          (7., 7.5);
        ];
      ] );
    ( "multipolygon with holes",
      [
        z;
        z;
        z;
        [ 0; 1; 1; 1; 0; 1; 1; 1; 0; 0 ];
        [ 0; 1; 0; 1; 0; 1; 0; 1; 0; 0 ];
        [ 0; 1; 1; 1; 0; 1; 1; 1; 0; 0 ];
        z;
        z;
        z;
        z;
      ],
      [
        [
          (4., 5.5);
          (4., 4.5);
          (4., 3.5);
          (3.5, 3.);
          (2.5, 3.);
          (1.5, 3.);
          (1., 3.5);
          (1., 4.5);
          (1., 5.5);
          (1.5, 6.);
          (2.5, 6.);
          (3.5, 6.);
          (4., 5.5);
        ];
        [ (2.5, 5.); (2., 4.5); (2.5, 4.); (3., 4.5); (2.5, 5.) ];
        [
          (8., 5.5);
          (8., 4.5);
          (8., 3.5);
          (7.5, 3.);
          (6.5, 3.);
          (5.5, 3.);
          (5., 3.5);
          (5., 4.5);
          (5., 5.5);
          (5.5, 6.);
          (6.5, 6.);
          (7.5, 6.);
          (8., 5.5);
        ];
        [ (6.5, 5.); (6., 4.5); (6.5, 4.); (7., 4.5); (6.5, 5.) ];
      ] );
  ]

let d3_agreement =
  cases
    ~name:(fun (n, _, _) -> n)
    "agreement with d3-contour" d3_cases
    (fun (_, rows, rings) ->
      let f = grid10 rows in
      same_rings (List.map d3_ring rings)
        (Field2.isoband ~lo:0.5 ~hi:infinity f);
      let closed =
        List.map
          (fun (pts, closed) ->
            is_true ~msg:"isolines away from the border are closed" closed;
            pts)
          (curves (Field2.isoline 0.5 f))
      in
      same_rings closed (Field2.isoband ~lo:0.5 ~hi:infinity f))

(* Laws over small random fields. *)

type sample = {
  rows : float array list;
  xs : float array option;
  ys : float array option;
}

let pp_sample ppf s =
  let pp_arr ppf a =
    Format.fprintf ppf "[|%a|]"
      (Format.pp_print_array
         ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
         (fun ppf x -> Format.fprintf ppf "%.17g" x))
      a
  in
  let pp_opt name ppf = function
    | None -> ()
    | Some a -> Format.fprintf ppf "~%s:%a " name pp_arr a
  in
  Format.fprintf ppf "@[<v>%a%a@,%a@]" (pp_opt "xs") s.xs (pp_opt "ys") s.ys
    (Format.pp_print_list pp_arr)
    s.rows

let sample_field s = field ?xs:s.xs ?ys:s.ys s.rows
let coord a k = match a with None -> Float.of_int k | Some a -> a.(k)

(* The diagonals of the cells with exactly one missing sample: the segments
   between the two corners next to the missing one. *)
let cut_diagonals s =
  let rows = Array.of_list s.rows in
  let ny = Array.length rows in
  let nx = if ny = 0 then 0 else Array.length rows.(0) in
  let pt (i, j) = (coord s.xs j, coord s.ys i) in
  List.concat_map
    (fun i ->
      List.concat_map
        (fun j ->
          let corners = [ (i, j); (i, j + 1); (i + 1, j + 1); (i + 1, j) ] in
          let missing (i, j) = not (Float.is_finite rows.(i).(j)) in
          match List.filter missing corners with
          | [ (mi, mj) ] -> (
              match List.filter (fun (i, j) -> i = mi <> (j = mj)) corners with
              | [ a; b ] -> [ (pt a, pt b) ]
              | _ -> assert false)
          | _ -> [])
        (List.init (Int.max 0 (nx - 1)) Fun.id))
    (List.init (Int.max 0 (ny - 1)) Fun.id)

(* Whether [(x, y)] lies within [tol] of segment [a, b], [tol] a billionth of
   the coordinates' magnitude: far wider than the rounding error by which
   crossings lie off a diagonal. *)
let near_segment (x, y) ((ax, ay), (bx, by)) =
  let tol =
    1e-9 *. (1. +. Float.abs ax +. Float.abs ay +. Float.abs bx +. Float.abs by)
  in
  Float.min ax bx -. tol <= x
  && x <= Float.max ax bx +. tol
  && Float.min ay by -. tol <= y
  && y <= Float.max ay by +. tol
  && Float.abs (((bx -. ax) *. (y -. ay)) -. ((by -. ay) *. (x -. ax)))
     <= tol *. Float.hypot (bx -. ax) (by -. ay)

(* Whether two of the finite levels lie within rounding error of each other, so
   that their crossings on a side do. *)
let close_levels levels =
  let rec loop = function
    | a :: (b :: _ as rest) ->
        (Float.is_finite b && b -. a <= 4. *. epsilon_float *. Float.abs b)
        || loop rest
    | _ -> false
  in
  loop levels

(* Values cluster on the levels the laws use, with their neighbours, so that
   ties, crossings at corners and nearly equal levels are common. Other values
   lie on a grid of 1/64: below it, coordinates of crossings and their
   differences become subnormal, where products underflow and the conventions'
   rule for points on rings is no longer exact. *)
let level_values = [ 0.; 1.; 2.; 3.; 1.5; Float.succ 1.; Float.pred 2.; 2.5 ]

let on_grid lo hi =
  Gen.map (fun x -> Float.round (x *. 64.) /. 64.) (Gen.float_range lo hi)

let gen_value =
  Gen.frequency
    [
      (8, Gen.of_list level_values);
      (3, on_grid (-0.5) 3.5);
      (1, Gen.of_list [ nan; infinity; neg_infinity ]);
    ]

let gen_coords n =
  let open Gen in
  frequency
    [
      (2, constant None);
      ( 1,
        let+ start = float_range (-3.) 3.
        and+ steps =
          array ~size:(constant (max 0 (n - 1))) (float_range 0.25 2.)
        and+ down = bool in
        let a = Array.make n start in
        for i = 1 to n - 1 do
          a.(i) <-
            (if down then a.(i - 1) -. steps.(i - 1)
             else a.(i - 1) +. steps.(i - 1))
        done;
        Some a );
    ]

let gen_sample =
  let open Gen in
  with_pp pp_sample
    (let* ny, nx = pair (int_range 1 5) (int_range 1 5) in
     let+ rows = list ~size:(constant ny) (array ~size:(constant nx) gen_value)
     and+ xs = gen_coords nx
     and+ ys = gen_coords ny in
     { rows; xs; ys })

let gen_levels =
  Gen.map
    (fun ls -> List.sort_uniq Float.compare ls)
    (Gen.list ~size:(Gen.int_range 1 4)
       (Gen.frequency [ (3, Gen.of_list level_values); (1, on_grid 0. 3.) ]))

(* The winding number by the conventions' rule for points on rings. *)
let winding pts (px, py) =
  let a = Array.of_list pts in
  let n = Array.length a in
  let w = ref 0 in
  for i = 0 to n - 1 do
    let ax, ay = a.(i) and bx, by = a.((i + 1) mod n) in
    let a_above = ay > py and b_above = by > py in
    if b_above && not a_above then
      begin if ((bx -. ax) *. (py -. ay)) -. ((px -. ax) *. (by -. ay)) > 0.
      then incr w
      end
    else if a_above && not b_above then
      if ((ax -. bx) *. (py -. by)) -. ((px -. bx) *. (ay -. by)) > 0. then
        decr w
  done;
  !w

let segments_of_rings p =
  List.concat_map
    (fun r ->
      let pts = Array.of_list (ring_pts r) in
      let n = Array.length pts in
      List.init n (fun i -> (pts.(i), pts.((i + 1) mod n))))
    (Pgon2.rings p)

let segments_of_curves cs =
  List.concat_map
    (fun (pts, closed) ->
      let a = Array.of_list pts in
      let n = Array.length a in
      let m = if closed then n else n - 1 in
      List.init m (fun i -> (a.(i), a.((i + 1) mod n))))
    cs

(* The sign of the orientation of [c] from [a] to [b], [0] within rounding
   error. *)
let orient (ax, ay) (bx, by) (cx, cy) =
  let p = (bx -. ax) *. (cy -. ay) and q = (by -. ay) *. (cx -. ax) in
  let tol = 1e-9 *. (Float.abs p +. Float.abs q) in
  if p -. q > tol then 1 else if q -. p > tol then -1 else 0

(* Whether segments share more than a point. Collinear overlaps arise only along
   the horizontal and vertical lines of the grid, where they are exact; a
   crossing is reported only when every orientation is clear. *)
let cross_or_overlap ((a, b) as s) ((c, d) as t) =
  let overlap coord =
    let lo1 = Float.min (coord a) (coord b)
    and hi1 = Float.max (coord a) (coord b) in
    let lo2 = Float.min (coord c) (coord d)
    and hi2 = Float.max (coord c) (coord d) in
    Float.min hi1 hi2 > Float.max lo1 lo2
  in
  let horizontal (p, q) = snd p = snd q and vertical (p, q) = fst p = fst q in
  (a = c && b = d)
  || (a = d && b = c)
  || (horizontal s && horizontal t && snd a = snd c && overlap fst)
  || (vertical s && vertical t && fst a = fst c && overlap snd)
  || (orient a b c * orient a b d < 0 && orient c d a * orient c d b < 0)

(* Whether segment [p, q] lies on segment [a, b], running the same way. *)
let lies_on (p, q) (a, b) =
  (p = a && q = b)
  ||
  let (px, py), (qx, qy), (ax, ay), (bx, by) = (p, q, a, b) in
  let within u v w = Float.min v w <= u && u <= Float.max v w in
  let same_way =
    ((qx -. px) *. (bx -. ax)) +. ((qy -. py) *. (by -. ay)) > 0.
  in
  let tol =
    1e-12
    *. (1. +. Float.abs ax +. Float.abs ay +. Float.abs bx +. Float.abs by)
       ** 2.
  in
  let on (ux, uy) =
    within ux ax bx && within uy ay by
    && Float.abs (((bx -. ax) *. (uy -. ay)) -. ((by -. ay) *. (ux -. ax)))
       <= tol
  in
  same_way && on p && on q

let check_rings ~exempt what p =
  List.iter
    (fun r ->
      let pts = ring_pts r in
      let n = List.length pts in
      at_least ~msg:(what ^ ": points of a ring") int ~than:3 n;
      let a = Array.of_list pts in
      for i = 0 to n - 1 do
        if a.(i) = a.((i + 1) mod n) then
          failf "%s: a ring repeats a point %a" what pp_pts pts
      done)
    (Pgon2.rings p);
  let segs = Array.of_list (segments_of_rings p) in
  let n = Array.length segs in
  for i = 0 to n - 1 do
    for j = i + 1 to n - 1 do
      if cross_or_overlap segs.(i) segs.(j) && not (exempt segs.(i) segs.(j))
      then
        failf "%s: segments %a and %a cross or overlap" what pp_pts
          [ fst segs.(i); snd segs.(i) ]
          pp_pts
          [ fst segs.(j); snd segs.(j) ]
    done
  done

let probes s f =
  let b = Pgon2.bounds (domain f) in
  match b with
  | None -> []
  | Some b ->
      let r = Random.State.make [| List.length s.rows |] in
      let rand lo hi = lo +. Random.State.float r (hi -. lo) in
      List.init 30 (fun _ ->
          (rand (Box2.minx b) (Box2.maxx b), rand (Box2.miny b) (Box2.maxy b)))
      @ List.concat_map (fun r -> ring_pts r) (Pgon2.rings (domain f))

(* The bands between [levels] and infinite ends. *)
let bands levels f =
  let rec pairs = function
    | a :: (b :: _ as rest) -> (a, b) :: pairs rest
    | _ -> []
  in
  List.map
    (fun (lo, hi) -> Field2.isoband ~lo ~hi f)
    (pairs ((neg_infinity :: levels) @ [ infinity ]))

(* The bands of [levels] with open ends tile the domain: their areas add up to
   the domain's, and a point, decided by the conventions' rule for points on
   rings, is in exactly one band if it is in the domain and in none otherwise.
   Near a cut cell's diagonal, where bands bend at their crossings and the
   domain does not, a point is in at most one band, unless two levels lie within
   rounding error of each other. *)
let bands_tile (s, levels) =
  let f = sample_field s in
  cover "a missing sample"
    (List.exists (Array.exists (fun v -> not (Float.is_finite v))) s.rows);
  cover "a sample equal to a level"
    (List.exists (Array.exists (fun v -> List.mem v levels)) s.rows);
  cover "two adjacent levels"
    (List.exists (fun l -> List.mem (Float.succ l) levels) levels
    || (List.mem 1. levels && List.mem (Float.succ 1.) levels));
  let bands = bands levels f in
  let d = domain f in
  let total = List.fold_left (fun a b -> a +. Pgon2.area b) 0. bands in
  equal ~msg:"areas add up to the domain's" near (Pgon2.area d) total;
  let vertices =
    List.concat_map (fun b -> List.concat_map ring_pts (Pgon2.rings b)) bands
  in
  let diagonals = cut_diagonals s and close = close_levels levels in
  List.iter
    (fun (x, y) ->
      let p = P2.v x y in
      let n = List.length (List.filter (Pgon2.mem p) bands) in
      if not (List.exists (near_segment (x, y)) diagonals) then begin
        let expected = if Pgon2.mem p d then 1 else 0 in
        if n <> expected then
          failf "(%.17g, %.17g) is in %d bands, %d expected" x y n expected
      end
      else if n > 1 && not close then
        failf "(%.17g, %.17g) is in %d bands" x y n)
    (probes s f @ vertices)

(* Each band's rings are simple, wind 0 or 1 times, lie in the domain's box and
   run along the domain's boundary or the isolines of its bounds. Near a cut
   cell's diagonal, rings of levels within rounding error of each other may
   cross. *)
let band_rings (s, levels) =
  let f = sample_field s in
  let lo = List.hd levels and hi = List.nth levels (List.length levels - 1) in
  let hi = if hi = lo then infinity else hi in
  let band = Field2.isoband ~lo ~hi f in
  let diagonals = cut_diagonals s and close = close_levels [ lo; hi ] in
  let near pts =
    close
    && List.exists
         (fun dg -> List.for_all (fun pt -> near_segment pt dg) pts)
         diagonals
  in
  check_rings ~exempt:(fun (a, b) (c, d) -> near [ a; b; c; d ]) "band" band;
  List.iter
    (fun pt ->
      if not (near [ pt ]) then
        let w =
          List.fold_left
            (fun w r -> w + winding (ring_pts r) pt)
            0 (Pgon2.rings band)
        in
        if w <> 0 && w <> 1 then
          failf "the rings wind %d times around (%g, %g)" w (fst pt) (snd pt))
    (probes s f);
  (match (Pgon2.bounds band, Pgon2.bounds (domain f)) with
  | Some b, Some d ->
      is_true ~msg:"rings lie in the domain's box"
        (Box2.minx d <= Box2.minx b
        && Box2.maxx b <= Box2.maxx d
        && Box2.miny d <= Box2.miny b
        && Box2.maxy b <= Box2.maxy d)
  | Some _, None -> fail "a band of an empty domain"
  | None, _ -> ());
  let dom = segments_of_rings (domain f) in
  let lo_line = segments_of_curves (curves (Field2.isoline lo f)) in
  let hi_line =
    List.map
      (fun (a, b) -> (b, a))
      (segments_of_curves (curves (Field2.isoline hi f)))
  in
  List.iter
    (fun seg ->
      let on l = List.exists (lies_on seg) l in
      if not (on dom || on lo_line || on hi_line) then
        failf "segment %a is on neither the domain's boundary nor an isoline"
          pp_pts
          [ fst seg; snd seg ])
    (segments_of_rings band)

(* The isoline at [l] is the boundary of [R l] off the domain's boundary: its
   segments are segments of [isoband ~lo:l ~hi:infinity], the same way, and
   reversed of [isoband ~lo:neg_infinity ~hi:l]. *)
let isolines_bound_regions (s, levels) =
  let f = sample_field s in
  let l = List.hd levels in
  let cs = curves (Field2.isoline l f) in
  List.iter
    (fun (pts, closed) ->
      let n = List.length pts in
      at_least ~msg:"points of a curve" int ~than:(if closed then 3 else 2) n;
      let a = Array.of_list pts in
      let m = if closed then n else n - 1 in
      for i = 0 to m - 1 do
        if a.(i) = a.((i + 1) mod n) then
          failf "a curve repeats a point %a" pp_pts pts
      done)
    cs;
  let segs = segments_of_curves cs in
  let above = segments_of_rings (Field2.isoband ~lo:l ~hi:infinity f) in
  let below = segments_of_rings (Field2.isoband ~lo:neg_infinity ~hi:l f) in
  let dom = segments_of_rings (domain f) in
  List.iter
    (fun ((a, b) as seg) ->
      if not (List.mem seg above) then
        failf "isoline segment %a is not on R l" pp_pts [ a; b ];
      if not (List.exists (lies_on (b, a)) below) then
        failf "isoline segment %a reversed is not on the band below" pp_pts
          [ a; b ])
    segs;
  List.iter
    (fun seg ->
      if not (List.mem seg segs || List.exists (lies_on seg) dom) then
        failf
          "segment %a of R l is neither the isoline nor the domain's boundary"
          pp_pts
          [ fst seg; snd seg ])
    above;
  let rings = Array.of_list segs in
  for i = 0 to Array.length rings - 1 do
    for j = i + 1 to Array.length rings - 1 do
      if cross_or_overlap rings.(i) rings.(j) then
        failf "isoline segments %a and %a cross or overlap" pp_pts
          [ fst rings.(i); snd rings.(i) ]
          pp_pts
          [ fst rings.(j); snd rings.(j) ]
    done
  done

(* The centre of a whole cell whose samples are all at least [l] is in [R l],
   and that of one whose samples are all below [l] is not. *)
let isolines_separate (s, levels) =
  let f = sample_field s in
  let l = List.hd levels in
  let r = Field2.isoband ~lo:l ~hi:infinity f in
  let rows = Array.of_list s.rows in
  let ny = Array.length rows and nx = Array.length rows.(0) in
  for i = 0 to ny - 2 do
    for j = 0 to nx - 2 do
      let vs =
        [
          rows.(i).(j); rows.(i).(j + 1); rows.(i + 1).(j); rows.(i + 1).(j + 1);
        ]
      in
      if List.for_all Float.is_finite vs then begin
        let c =
          P2.v
            ((coord s.xs j +. coord s.xs (j + 1)) /. 2.)
            ((coord s.ys i +. coord s.ys (i + 1)) /. 2.)
        in
        if List.for_all (fun v -> v >= l) vs then
          is_true ~msg:"a cell above is in R l" (Pgon2.mem c r);
        if List.for_all (fun v -> v < l) vs then
          is_false ~msg:"a cell below is not in R l" (Pgon2.mem c r)
      end
    done
  done

let deterministic (s, levels) =
  let f = sample_field s in
  let l = List.hd levels in
  equal pgon
    (Field2.isoband ~lo:l ~hi:infinity f)
    (Field2.isoband ~lo:l ~hi:infinity (sample_field s));
  equal path (Field2.isoline l f) (Field2.isoline l (sample_field s))

(* A cut cell 5.6e-6 high at y = 792240, where a unit in the last place of y is
   2e-5 of the cell's height. *)
let far_ys = [| 792240.; 792240.0000056 |]

(* A cut cell whose bands of two levels within rounding error of each other
   overlap at [near_levels_pt], within rounding error of its diagonal: the chord
   of the lower level from a crossing next to the top left corner to the bottom
   right corner runs within rounding error of the diagonal, and the upper
   level's crossing on the diagonal falls on its wrong side. *)
let near_levels_xs = [| -16.814359448746295; -16.607636386910656 |]
let near_levels_ys = [| -5.1714227594632955; -3.5619193239620968 |]

let near_levels_rows =
  [ [| 2.000000000000001; nan |]; [| 0.6999999999999996; 1.9999999999999998 |] ]

let near_levels = [ 1.9999999999999996; 2.0000000000000004 ]
let near_levels_pt = P2.v (-16.73167022401174) (-4.527621385260474)

let gen_case =
  Gen.with_pp
    (fun ppf (s, ls) ->
      Format.fprintf ppf "@[<v>%a@,levels %a@]" pp_sample s
        (Format.pp_print_list ~pp_sep:Format.pp_print_space (fun ppf l ->
             Format.fprintf ppf "%.17g" l))
        ls)
    (Gen.pair gen_sample gen_levels)

(* Fields on which the laws once failed: crossings on a diagonal rounding onto
   the line of another side, crossings of near levels on a diagonal that
   inverted a band, a saddle whose region collapses onto its diagonal, a point
   within rounding error of edges of three bands. Then cut cells far from the
   origin against their height, whose diagonals bend most at their crossings and
   whose step is set by y: one with bands overlapping on its diagonal, one whose
   crossing near an end would otherwise round onto the top side. *)
let regressions =
  let p2 = Float.pred 2. and s1 = Float.succ 1. in
  let case ?xs ?ys rows levels = ({ rows; xs; ys }, levels) in
  [
    case
      ~xs:[| 0.2502392362475277; 0.50023923624752764 |]
      ~ys:[| 0.; 0.25; 0.5 |]
      [ [| 0.; 0. |]; [| nan; 1. |]; [| 2.; 2. |] ]
      [ s1; 2. ];
    case
      [ [| 0.; 0.; 2.; 1.; nan |]; [| 0.; 0.; 0.; 2.; 2.0000660703615822 |] ]
      [ s1 ];
    case
      ~ys:[| 0.; 1.0000359577848634; 1.2500359577848634 |]
      [ [| 0.; 0. |]; [| 3.; p2 |]; [| 2.5; nan |] ]
      [ 0.; 2. ];
    case
      [
        [| 0.; 0.; 0.; 0. |];
        [| 0.; nan; 2.5; nan |];
        [| 0.; 0.; 2.; 1. |];
        [| 0.; nan; 2.5; 0. |];
      ]
      [ p2; 2.; 3. ];
    case
      ~ys:[| 0.; 0.25; 0.5; 0.75068742307833647 |]
      [
        [| 0.; 0.; 0.; 0. |];
        [| 0.; 0.; 0.; 0. |];
        [| 0.; 0.; 9.9920962214652108e-16; nan |];
        [| 0.; 0.; 0.; 2.5935403645858823 |];
      ]
      [ 0.; p2; 2. ];
    case ~xs:[| 0.; 0.25; 0.5 |]
      [ [| 0.; 2.; p2 |]; [| 0.; p2; 2. |] ]
      [ 0.; 2. ];
    case
      ~xs:
        [|
          -0.989882094496219;
          -1.5000075269943542;
          -1.7500075269943542;
          -2.000007526994354;
        |]
      ~ys:[| -0.00039530343356600488; 1.50019762959069 |]
      [ [| 0.; nan; 3.; 0. |]; [| 0.; 0.; 0.; 0. |] ]
      [ 1.; s1 ];
    case ~xs:[| 2.; 3. |] ~ys:far_ys [ [| 3.; 0. |]; [| 3.; nan |] ] [ 1. ];
    case ~xs:near_levels_xs ~ys:near_levels_ys near_levels_rows near_levels;
    case ~xs:[| 2.; 3. |] ~ys:far_ys [ [| 3.; 0. |]; [| 1.5; nan |] ] [ 3e-6 ];
  ]

let laws =
  let prop name law = prop ~count:300 ~examples:regressions name gen_case law in
  group "laws"
    [
      prop "bands tile the domain exactly away from cut cells' diagonals"
        bands_tile;
      prop "a band's rings are simple and follow its isolines" band_rings;
      prop "an isoline bounds its region" isolines_bound_regions;
      prop "isolines separate cells above from cells below" isolines_separate;
      prop "equal arguments give equal results" deterministic;
    ]

(* Near levels on cut cells. The contract lets their bands overlap within
   rounding error of a diagonal. *)

let overlap_near_diagonal () =
  let s =
    {
      rows = near_levels_rows;
      xs = Some near_levels_xs;
      ys = Some near_levels_ys;
    }
  in
  let pt = (P2.x near_levels_pt, P2.y near_levels_pt) in
  is_true ~msg:"near the diagonal"
    (List.exists (near_segment pt) (cut_diagonals s))

let near_levels_apart () =
  let n =
    List.length
      (List.filter (Pgon2.mem near_levels_pt)
         (bands near_levels
            (field ~xs:near_levels_xs ~ys:near_levels_ys near_levels_rows)))
  in
  at_most int ~than:1 n

(* Levels a few units in the last place apart whose crossings on the diagonal
   round to neighbouring multiples of the step, so that they lie a step apart.
   Their bands overlap with a step of 8 or 32 rounding errors and stay apart
   with 256. *)
let step_cases =
  [
    ( [| 0.797302913532759; 1.0679835797987471 |],
      [| 1.3616971934929403; 3.155449259613466 |],
      [| 2.18277012134777; nan |],
      [| 0.19155382558576306; 2.1502660026096754 |],
      [ 2.1667778990365343; 2.1667778990365347; 2.1667778990366466 ] );
    ( [| -2.14711828494428; -1.0354536283897178 |],
      [| 0.9115560817856547; 1.834905994438033 |],
      [| nan; 1.3185791774140974 |],
      [| 1.277457855137686; 2.6177877563401766 |],
      [ 1.2880633573784808; 1.288063357378481 ] );
    ( [| 2.494447018285971; 3.109902385233981 |],
      [| -0.13578220799741825; 0.5133614194118594 |],
      [| nan; 0.9213107305341494 |],
      [| 0.9101629531291233; 2.887993084758688 |],
      [ 0.9117739595635503; 0.9117739595635516 ] );
  ]

let step_keeps_bands_apart (xs, ys, top, bottom, levels) =
  let bs = bands levels (field ~xs ~ys [ top; bottom ]) in
  let count x y = List.length (List.filter (Pgon2.mem (P2.v x y)) bs) in
  List.iter
    (fun b ->
      List.iter
        (fun r ->
          let n = Ring2.length r in
          for i = 0 to n - 1 do
            let x = Ring2.x r i and y = Ring2.y r i in
            let j = (i + 1) mod n in
            let mx = (x +. Ring2.x r j) /. 2. in
            let my = (y +. Ring2.y r j) /. 2. in
            at_most ~msg:"a vertex" int ~than:1 (count x y);
            at_most ~msg:"a midpoint" int ~than:1 (count mx my)
          done)
        (Pgon2.rings b))
    bs

let near_level_cases =
  group "near levels on cut cells"
    [
      test "the overlap of near levels is within rounding error of the diagonal"
        overlap_near_diagonal;
      xfail ~reason:"crossings on a diagonal cannot lie on it exactly"
        (test "bands of near levels do not overlap at a cut cell's diagonal"
           near_levels_apart);
      cases
        ~name:(fun (xs, _, _, _, _) -> Printf.sprintf "xs.(0) = %g" xs.(0))
        "the diagonal's step keeps bands of near levels apart" step_cases
        step_keeps_bands_apart;
    ]

(* The diagonal of the [far_ys] cell runs one unit left from its top right
   corner, sample 0, to its bottom left one, sample 3. The crossing of a level
   [l] lies at x = 3 - l / 3 before it moves, and moves by at most the fraction
   256 ε (c + d) / d of the diagonal, which the cell's height sets: about 0.008
   in x. A crossing farther than that from both ends stays inside the cell. *)
let far_crossings_move_within_bound () =
  let f = field ~xs:[| 2.; 3. |] ~ys:far_ys [ [| 3.; 0. |]; [| 3.; nan |] ] in
  let y0 = far_ys.(0) and y1 = far_ys.(1) in
  let fraction c0 c1 =
    let d = Float.abs (c1 -. c0) in
    256. *. epsilon_float *. (Float.max (Float.abs c0) (Float.abs c1) +. d) /. d
  in
  let bound = Float.max (fraction 2. 3.) (fraction y0 y1) in
  for k = 1 to 999 do
    let l = 3. *. Float.of_int k /. 1000. in
    let exact = 3. -. (l /. 3.) in
    let inside =
      List.concat_map
        (fun r ->
          List.filter
            (fun (x, y) -> 2. < x && x < 3. && y0 < y && y < y1)
            (ring_pts r))
        (Pgon2.rings (Field2.isoband ~lo:l ~hi:infinity f))
    in
    let msg = Printf.sprintf "level %.17g" l in
    if exact -. 2. > bound && 3. -. exact > bound then
      equal ~msg int 1 (List.length inside);
    List.iter
      (fun (x, _) ->
        at_most ~msg float_exact ~than:bound (Float.abs (x -. exact)))
      inside
  done

(* Golden output. *)

let golden_field =
  field
    [
      [| 0.; 1.; 2.; 1.; 0. |];
      [| 1.; 3.; 1.; 3.; 1. |];
      [| 2.; 1.; nan; 1.; 2. |];
      [| 1.; 0.; 1.; 2.; 3. |];
      [| 0.; 1.; 2.; 3.; 3. |];
    ]

let golden () =
  let pp_band (lo, hi) =
    Format.printf "@[<v 2>band [%g, %g):@,%a@]@." lo hi Pgon2.pp
      (Field2.isoband ~lo ~hi golden_field)
  in
  List.iter pp_band [ (neg_infinity, 1.); (1., 2.); (2., infinity) ];
  List.iter
    (fun l ->
      Format.printf "@[<v 2>isoline %g:@,%a@]@." l Path.pp
        (Field2.isoline l golden_field))
    [ 1.; 2.; 2.5 ];
  expect (output ())
  @@ __POS_OF__
       {|
    band [-inf, 1):
      M 0 1 L 0 0 L 1 0 Z M 3 0 L 4 0 L 4 1 Z M 0 3 L 1 2 L 2 3 L 1 4 L 0 4 Z
    band [1, 2):
      M 1 0.5 L 0.5 1 L 1 1.5 L 1.5 1 Z
      M 0 1 L 1 0 L 2 0 L 3 0 L 4 1 L 4 2 L 3 3 L 2 4 L 1 4 L 2 3 L 3 2 L 2 1
      L 1 2 L 0 3 L 0 2 Z M 3 0.5 L 2.5 1 L 3 1.5 L 3.5 1 Z
    band [2, inf):
      M 0.5 1 L 1 0.5 L 1.5 1 L 1 1.5 Z M 2.5 1 L 3 0.5 L 3.5 1 L 3 1.5 Z
      M 4 2 L 4 3 L 4 4 L 3 4 L 2 4 L 3 3 Z
    isoline 1:
      M 3 0 L 4 1 M 1 4 L 2 3 M 1 2 L 0 3 M 0 1 L 1 0
    isoline 2:
      M 0.5 1 L 1 0.5 L 1.5 1 L 1 1.5 Z M 2.5 1 L 3 0.5 L 3.5 1 L 3 1.5 Z M 2 4
      L 3 3 L 4 2
    isoline 2.5:
      M 0.75 1 L 1 0.75 L 1.25 1 L 1 1.25 Z M 2.75 1 L 3 0.75 L 3.25 1 L 3 1.25 Z
      M 2.5 4 L 3 3.5 L 3.5 3 L 4 2.5
    |}

let () =
  exit
    (run "Field2"
       [
         constructing;
         goldens;
         d3_agreement;
         laws;
         near_level_cases;
         group "crossings on diagonals"
           [
             test
               "a crossing moves by at most its stated fraction of the diagonal"
               far_crossings_move_within_bound;
           ];
         group "golden" [ test "bands and isolines of a 5 x 5 field" golden ];
       ])
