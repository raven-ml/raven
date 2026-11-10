(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Jord

let tau = 1e-6
let utm33 = Crs.utm ~zone:33 ~north:true

(* A grid whose world point (x, y) is pixel point (-y, x), exactly. *)
let rows = 12
let cols = 10

let grid =
  Grid.v utm33 ~rows ~cols ~origin:(0., 0.) ~row_step:(0., -1.)
    ~col_step:(1., 0.)

(* Features given as rings of pixel points (r, c). *)
let features ?(g = grid) fs =
  let rings = List.concat fs in
  let pts = List.concat_map (List.concat_map (fun (r, c) -> [ r; c ])) rings in
  let pts =
    Nx.create Nx.float64 [| List.length pts / 2; 2 |] (Array.of_list pts)
  in
  let offsets lens =
    Array.of_list
      (List.rev
         (List.fold_left (fun acc n -> (List.hd acc + n) :: acc) [ 0 ] lens))
  in
  Polygons.v g.Grid.crs (Grid.world g pts)
    ~rings:(offsets (List.map List.length rings))
    ~features:(offsets (List.map List.length fs))

let relation (f, c) =
  let ints t = List.map Int64.to_int (Array.to_list (Nx.to_array t)) in
  List.combine (ints f) (ints c)

let rel = list (pair int int)
let centres ?(g = grid) fs = relation (Rasterize.centres g (features ~g fs))
let touched ?(g = grid) fs = relation (Rasterize.touched g (features ~g fs))

let cells ?(rows = rows) ?(cols = cols) f =
  List.concat
    (List.init rows (fun i ->
         List.filter_map
           (fun j -> if f i j then Some ((i * cols) + j) else None)
           (List.init cols Fun.id)))

(* Rectangles, by the .mli's rules *)

let rect (r0, c0, r1, c1) = [ (r0, c0); (r0, c1); (r1, c1); (r1, c0) ]
let fi = float_of_int

(* The centre moved left by δ and down by ε << δ: r0 <= i + 1/2 < r1 and c0 < j
   + 1/2 <= c1. *)
let rect_centres (r0, c0, r1, c1) =
  cells (fun i j ->
      r0 <= fi i +. 0.5
      && fi i +. 0.5 < r1
      && c0 < fi j +. 0.5
      && fi j +. 0.5 <= c1)

(* The rectangle meets the cell's open square shrunk by τ. *)
let rect_touched (r0, c0, r1, c1) =
  cells (fun i j ->
      r0 < fi (i + 1) -. tau
      && r1 > fi i +. tau
      && c0 < fi (j + 1) -. tau
      && c1 > fi j +. tau)

let one f = List.map (fun c -> (0, c)) f

(* Corners on cell borders, on centre lines, or anywhere, past the grid too. *)
let coordinate n =
  Gen.(
    frequency
      [
        (2, map fi (int_range (-2) (n + 2)));
        (2, map (fun k -> fi k +. 0.5) (int_range (-2) (n + 2)));
        (3, float_range (-2.) (fi n +. 2.));
      ])

let rectangle =
  Gen.(
    let+ a = coordinate rows
    and+ b = coordinate rows
    and+ c = coordinate cols
    and+ d = coordinate cols in
    (Float.min a b, Float.min c d, Float.max a b, Float.max c d))

let nondegenerate (r0, c0, r1, c1) = r0 < r1 && c0 < c1
let on_border (r0, c0, r1, c1) = List.exists Float.is_integer [ r0; c0; r1; c1 ]

let on_centre_line (r0, c0, r1, c1) =
  List.exists (fun x -> Float.is_integer (x -. 0.5)) [ r0; c0; r1; c1 ]

let rectangles =
  group "rectangles"
    [
      prop "centres is the rectangle's rule" rectangle (fun r ->
          assume (nondegenerate r);
          cover "an edge on a cell border" (on_border r);
          cover "an edge on a line of centres" (on_centre_line r);
          equal rel (one (rect_centres r)) (centres [ [ rect r ] ]));
      prop "touched is the rectangle's rule" rectangle (fun r ->
          assume (nondegenerate r);
          cover "an edge on a cell border" (on_border r);
          cover "an edge on a line of centres" (on_centre_line r);
          equal rel (one (rect_touched r)) (touched [ [ rect r ] ]));
      test "a rectangle on cell borders covers its cells and no neighbour"
        (fun () ->
          let inside = cells (fun i j -> i >= 2 && i < 5 && j >= 3 && j < 7) in
          equal rel (one inside) (centres [ [ rect (2., 3., 5., 7.) ] ]);
          equal rel (one inside) (touched [ [ rect (2., 3., 5., 7.) ] ]));
      test "an overlap thicker than τ covers a cell, a thinner one does not"
        (fun () ->
          let next = (2 * cols) + 7 in
          equal ~msg:"2e-6" bool true
            (List.mem (0, next) (touched [ [ rect (2., 3., 5., 7. +. 2e-6) ] ]));
          equal ~msg:"5e-7" bool false
            (List.mem (0, next) (touched [ [ rect (2., 3., 5., 7. +. 5e-7) ] ])));
    ]

(* Polygons with float vertices, against a direct reference *)

(* A simple polygon: points at increasing angles around a centre. *)
let star =
  Gen.(
    let+ cr = float_range 1. (fi rows -. 1.)
    and+ cc = float_range 1. (fi cols -. 1.)
    and+ radii = list ~size:(int_range 3 9) (float_range 0.3 6.) in
    let n = List.length radii in
    List.mapi
      (fun k rad ->
        let a = 2. *. Float.pi *. fi k /. fi n in
        (cr +. (rad *. Float.sin a), cc +. (rad *. Float.cos a)))
      radii)

let edges ring = List.combine ring (List.tl ring @ [ List.hd ring ])

(* Even-odd at a centre by W. R. Franklin's crossing test. Float vertices put no
   edge through a centre. *)
let inside rings (y, x) =
  List.fold_left
    (fun inside ((ri, ci), (rj, cj)) ->
      if ri > y <> (rj > y) && x < ((cj -. ci) *. (y -. ri) /. (rj -. ri)) +. ci
      then not inside
      else inside)
    false
    (List.concat_map edges rings)

(* Whether the segment meets the open box, by Liang and Barsky's clip. *)
let meets ((pr, pc), (qr, qc)) (r0, c0, r1, c1) =
  let dr = qr -. pr and dc = qc -. pc in
  let clip (t0, t1) (p, q) =
    if p = 0. then if q > 0. then (t0, t1) else (1., 0.)
    else
      let t = q /. p in
      if p < 0. then (Float.max t0 t, t1) else (t0, Float.min t1 t)
  in
  let t0, t1 =
    List.fold_left clip (0., 1.)
      [ (-.dr, pr -. r0); (dr, r1 -. pr); (-.dc, pc -. c0); (dc, c1 -. pc) ]
  in
  t0 < t1
  || t0 = t1
     &&
     let r = pr +. (t0 *. dr) and c = pc +. (t0 *. dc) in
     r0 < r && r < r1 && c0 < c && c < c1

let direct_centres rings =
  cells (fun i j -> inside rings (fi i +. 0.5, fi j +. 0.5))

let direct_touched rings =
  let shrunk i j =
    (fi i +. tau, fi j +. tau, fi (i + 1) -. tau, fi (j + 1) -. tau)
  in
  cells (fun i j ->
      inside rings (fi i +. 0.5, fi j +. 0.5)
      || List.exists
           (fun e -> meets e (shrunk i j))
           (List.concat_map edges rings))

let polygons =
  group "polygons"
    [
      prop "centres is the even-odd test at each centre" star (fun s ->
          equal rel (one (direct_centres [ s ])) (centres [ [ s ] ]));
      prop "touched is the centres and the cells an edge enters" star (fun s ->
          equal rel (one (direct_touched [ s ])) (touched [ [ s ] ]));
      prop "touched contains centres"
        (Gen.list ~size:(Gen.int_range 1 4) star)
        (fun ss ->
          let fs = List.map (fun s -> [ s ]) ss in
          let t = touched fs in
          List.iter (fun p -> mem (pair int int) p t) (centres fs));
    ]

(* Rings and features *)

let parity l =
  List.sort_uniq compare
    (List.filter (fun c -> List.length (List.filter (( = ) c) l) mod 2 = 1) l)

let cells_of r = List.map snd r
let rects = Gen.list ~size:(Gen.int_range 1 5) rectangle

let rings =
  group "rings"
    [
      prop "a feature's centres are those of an odd number of its rings" rects
        (fun rs ->
          let rs = List.filter nondegenerate rs in
          assume (rs <> []);
          cover "several rings" (List.length rs >= 2);
          let each =
            List.concat_map (fun r -> cells_of (centres [ [ rect r ] ])) rs
          in
          equal (list int) (parity each)
            (cells_of (centres [ List.map rect rs ])));
      prop "a feature's touched cells lie between its rings' bounds" rects
        (fun rs ->
          let rs = List.filter nondegenerate rs in
          assume (rs <> []);
          let c = cells_of (centres [ List.map rect rs ]) in
          let t = cells_of (touched [ List.map rect rs ]) in
          let ring_t =
            List.map (fun r -> cells_of (touched [ [ rect r ] ])) rs
          in
          let ring_c =
            List.map (fun r -> cells_of (centres [ [ rect r ] ])) rs
          in
          let edge_only =
            List.concat
              (List.map2
                 (fun t c -> List.filter (fun x -> not (List.mem x c)) t)
                 ring_t ring_c)
          in
          List.iter (fun x -> mem int x t) (c @ edge_only);
          List.iter (fun x -> mem int x (c @ List.concat ring_t)) t);
      test "a hole removes its cells, whichever way it turns" (fun () ->
          let shell = rect (1., 1., 9., 8.) and hole = rect (3., 3., 6., 5.) in
          let expect =
            cells (fun i j ->
                i >= 1 && i < 9 && j >= 1 && j < 8
                && not (i >= 3 && i < 6 && j >= 3 && j < 5))
          in
          equal rel (one expect) (centres [ [ shell; List.rev hole ] ]);
          equal rel (one expect) (touched [ [ shell; hole ] ]));
      test "a feature without rings covers nothing" (fun () ->
          equal rel [ (1, 0) ] (centres [ []; [ rect (0., 0., 1., 1.) ] ]);
          equal rel [] (touched [ [] ]));
      test "no feature covers nothing" (fun () -> equal rel [] (touched []));
      test "cells outside the grid are in no pair" (fun () ->
          equal rel [] (touched [ [ rect (-5., -5., -1., -1.) ] ]);
          equal rel
            [ (0, (11 * cols) + 9) ]
            (touched [ [ rect (11.5, 9.5, 20., 20.) ] ]));
    ]

(* Order *)

let owner rel =
  let o = Array.make (rows * cols) (-1) in
  List.iter (fun (f, c) -> o.(c) <- max o.(c) f) rel;
  o

let count rel =
  let m = Array.make (rows * cols) 0 in
  List.iter (fun (_, c) -> m.(c) <- m.(c) + 1) rel;
  m

let shuffle seed n =
  let perm = Array.init n Fun.id in
  let st = Random.State.make [| seed |] in
  for k = n - 1 downto 1 do
    let j = Random.State.int st (k + 1) in
    let t = perm.(k) in
    perm.(k) <- perm.(j);
    perm.(j) <- t
  done;
  perm

let order =
  group "order"
    [
      prop "permuting the features changes only the owners of shared cells"
        Gen.(pair rects (int_range 0 1000))
        (fun (rs, seed) ->
          let rs = List.filter nondegenerate rs in
          assume (rs <> []);
          let n = List.length rs in
          (* Feature k of the permuted list is feature perm.(k) of rs. *)
          let perm = shuffle seed n in
          let a = touched (List.map (fun r -> [ rect r ]) rs) in
          let b =
            touched (List.init n (fun k -> [ rect (List.nth rs perm.(k)) ]))
          in
          equal ~msg:"the relation" rel a
            (List.sort compare (List.map (fun (f, c) -> (perm.(f), c)) b));
          let ca = count a in
          equal ~msg:"the counts" (array int) ca (count b);
          cover "a shared cell" (Array.exists (fun k -> k >= 2) ca);
          let oa = owner a and ob = owner b in
          Array.iteri
            (fun c k ->
              if k = 1 then
                equal ~msg:(Printf.sprintf "cell %d" c) int oa.(c) perm.(ob.(c)))
            ca);
      test "the owner composition gives the last covering feature" (fun () ->
          let fs =
            [
              [ rect (0., 0., 6., 6.) ];
              [ rect (3., 3., 9., 9.) ];
              [ rect (4., 4., 5., 5.) ];
            ]
          in
          let f, c = Rasterize.touched grid (features fs) in
          let none = Nx.full Nx.int64 [| rows * cols |] (-1L) in
          let o = Nx.to_array (Nx.scatter ~axis:0 c f none) in
          equal (array int) (owner (relation (f, c))) (Array.map Int64.to_int o);
          let n =
            Nx.scatter ~combine:Add ~axis:0 c (Nx.ones_like f)
              (Nx.zeros Nx.int64 [| rows * cols |])
          in
          equal (array int)
            (count (relation (f, c)))
            (Array.map Int64.to_int (Nx.to_array n)));
    ]

(* Tilings *)

let once rel within =
  Array.iteri
    (fun c k ->
      if within (c / cols) (c mod cols) then
        equal ~msg:(Printf.sprintf "cell %d" c) int 1 k)
    (count rel)

let tilings =
  group "tilings"
    [
      test "squares with corners on centres cover each centre once" (fun () ->
          let square i j =
            [ rect (fi i +. 0.5, fi j +. 0.5, fi i +. 1.5, fi j +. 1.5) ]
          in
          let squares =
            List.concat (List.init 11 (fun i -> List.init 9 (square i)))
          in
          once (centres squares) (fun i j ->
              i >= 1 && i <= 10 && j >= 1 && j <= 8));
      test "triangles around a centre cover each centre once" (fun () ->
          (* A fan around the centre of cell (5, 5), out to a square ring of
             other centres. *)
          let side = List.init 4 (fun k -> fi k +. 2.5) in
          let border =
            List.map (fun c -> (2.5, c)) side
            @ List.map (fun r -> (r, 6.5)) side
            @ List.map (fun c -> (6.5, 9. -. c)) side
            @ List.map (fun r -> (9. -. r, 2.5)) side
          in
          let fan =
            List.map (fun (p, q) -> [ [ (5.5, 5.5); p; q ] ]) (edges border)
          in
          once (centres fan) (fun i j -> i >= 3 && i <= 5 && j >= 3 && j <= 5));
      test "two triangles split by a diagonal through centres share none"
        (fun () ->
          let a = [ (0., 0.); (0., 10.); (10., 0.) ]
          and b = [ (0., 10.); (10., 10.); (10., 0.) ] in
          once (centres [ [ a ]; [ b ] ]) (fun i j -> i < 10 && j < 10));
    ]

(* The tolerance *)

let rectilinear =
  Gen.(
    let+ r0 = int_range 0 5
    and+ c0 = int_range 0 4
    and+ h = int_range 1 6
    and+ w = int_range 1 5 in
    (fi r0, fi c0, fi (r0 + h), fi (c0 + w)))

let jitter amount =
  Gen.(list ~size:(constant 8) (float_range (-.amount) amount))

let move ring d =
  List.mapi
    (fun k (r, c) -> (r +. List.nth d (2 * k), c +. List.nth d ((2 * k) + 1)))
    ring

(* The cells of the rectangles' pairs, feature by feature. *)
let traced ~rows ~cols rs =
  List.concat
    (List.mapi
       (fun f (r0, c0, r1, c1) ->
         List.map
           (fun c -> (f, c))
           (cells ~rows ~cols (fun i j ->
                fi i >= r0 && fi i < r1 && fi j >= c0 && fi j < c1)))
       rs)

(* A field on cell borders of a 200 by 200 grid. *)
let metre_field =
  Gen.(
    let+ r0 = int_range 0 190
    and+ c0 = int_range 0 190
    and+ h = int_range 1 40
    and+ w = int_range 1 40 in
    (fi r0, fi c0, fi (min 200 (r0 + h)), fi (min 200 (c0 + w))))

let tolerance =
  group "tolerance"
    [
      prop "vertices moved by less than τ give the same relation"
        Gen.(list ~size:(int_range 1 4) (pair rectilinear (jitter 9e-7)))
        (fun rs ->
          let want = traced ~rows ~cols (List.map fst rs) in
          let moved = List.map (fun (r, d) -> [ move (rect r) d ]) rs in
          equal ~msg:"centres" rel want (centres moved);
          equal ~msg:"touched" rel want (touched moved));
      test "fields traced on a tile's borders cover their cells after CRS84"
        (fun () ->
          (* A 10 m grid whose origin is a landmask's. The fields' corners are
             its cell corners, computed in float64, stored in CRS84 as GeoJSON
             stores them, and brought back. *)
          let g =
            Grid.v
              (Crs.utm ~zone:31 ~north:true)
              ~rows:40 ~cols:40
              ~origin:(301822.86256412434, 5798366.72759726)
              ~row_step:(0., -10.) ~col_step:(10., 0.)
          in
          let fields =
            [
              (0., 0., 7., 9.);
              (7., 0., 20., 9.);
              (0., 9., 20., 25.);
              (21., 3., 39., 38.);
            ]
          in
          let utm = features ~g (List.map (fun r -> [ rect r ]) fields) in
          let back = Polygons.to_crs g.crs (Polygons.to_crs Crs.lonlat utm) in
          let want = traced ~rows:40 ~cols:40 fields in
          equal ~msg:"touched" rel want (relation (Rasterize.touched g back));
          equal ~msg:"centres" rel want (relation (Rasterize.centres g back)));
      prop "fields traced on a 1 m grid's borders cover their cells after CRS84"
        Gen.(list ~size:(int_range 1 6) metre_field)
        (fun fields ->
          (* A 1 m grid at a UTM 33N landmask's origin, 5.3e6 m north, where a
             round trip moves a vertex by a few ulps of 5.3e6 m, about 3e-9
             pixel. *)
          let g =
            Grid.v utm33 ~rows:200 ~cols:200
              ~origin:(596785.7115403633, 5328362.815087351)
              ~row_step:(0., -1.) ~col_step:(1., 0.)
          in
          let utm = features ~g (List.map (fun r -> [ rect r ]) fields) in
          let back = Polygons.to_crs g.crs (Polygons.to_crs Crs.lonlat utm) in
          let want = traced ~rows:200 ~cols:200 fields in
          equal ~msg:"touched" rel want (relation (Rasterize.touched g back));
          equal ~msg:"centres" rel want (relation (Rasterize.centres g back)));
    ]

let systems =
  test "the features' system must be the grid's" (fun () ->
      let p =
        Polygons.v Crs.lonlat
          (Nx.zeros Nx.float64 [| 0; 2 |])
          ~rings:[| 0 |] ~features:[| 0 |]
      in
      raises_match (Exn.invalid_arg ~sub:"WGS 84 / UTM zone 33N") (fun () ->
          Rasterize.touched grid p);
      raises_match (Exn.invalid_arg ~sub:"centres") (fun () ->
          Rasterize.centres grid p))

let () =
  exit
    (run "jord.rasterize"
       [
         rectangles;
         polygons;
         rings;
         order;
         tilings;
         tolerance;
         group "systems" [ systems ];
       ])
