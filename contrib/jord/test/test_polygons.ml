(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Jord

let points l = Nx.create Nx.float64 [| List.length l / 2; 2 |] (Array.of_list l)
let square = [ 15.; 48.; 15.1; 48.; 15.1; 48.1; 15.; 48.1 ]

let layout =
  test "a feature is its rings, a ring its points" (fun () ->
      let p =
        Polygons.v Crs.lonlat
          (points (square @ square))
          ~rings:[| 0; 4; 8 |] ~features:[| 0; 2; 2 |]
      in
      equal int 2 (Polygons.count p);
      equal int 0
        (Polygons.count
           (Polygons.v Crs.lonlat (points []) ~rings:[| 0 |] ~features:[| 0 |])))

let refusals =
  cases "v refuses a broken layout" ~id:fst
    [
      ("rings from 1", (square, [| 1; 4 |], [| 0; 1 |], "rings"));
      ("rings short of the points", (square, [| 0; 3 |], [| 0; 1 |], "rings"));
      ( "decreasing rings",
        (square @ square, [| 0; 5; 4; 8 |], [| 0; 3 |], "decreases") );
      ("features past the rings", (square, [| 0; 4 |], [| 0; 2 |], "features"));
      ("no offsets", (square, [||], [| 0 |], "rings"));
      ( "a NaN point",
        (Float.nan :: List.tl square, [| 0; 4 |], [| 0; 1 |], "finite") );
    ]
    (fun (_, (pts, rings, features, sub)) ->
      raises_match (Exn.invalid_arg ~sub) (fun () ->
          Polygons.v Crs.lonlat (points pts) ~rings ~features))

let shape =
  test "v refuses points that are not [n; 2]" (fun () ->
      raises_match (Exn.invalid_arg ~sub:"[n; 2]") (fun () ->
          Polygons.v Crs.lonlat
            (Nx.zeros Nx.float64 [| 4; 3 |])
            ~rings:[| 0; 4 |] ~features:[| 0; 1 |]))

let to_crs =
  test "to_crs moves the points and keeps the offsets" (fun () ->
      let utm = Crs.utm ~zone:33 ~north:true in
      let p =
        Polygons.v Crs.lonlat (points square) ~rings:[| 0; 4 |]
          ~features:[| 0; 1 |]
      in
      let q = Polygons.to_crs utm p in
      equal bool true (Crs.equal utm q.crs);
      equal (array int) p.rings q.rings;
      equal (array float_exact)
        (Nx.to_array (Crs.transform ~src:Crs.lonlat ~dst:utm p.points))
        (Nx.to_array q.points))

(* Areas *)

let utm = Crs.utm ~zone:33 ~north:true

(* Features of rings of (x, y) from (500000, 5300000), so that the plane's
   numbers are a UTM zone's. *)
let features crs fs =
  let rings = List.concat fs in
  let at (x, y) = [ 500000. +. x; 5300000. +. y ] in
  let pts = List.concat_map (List.concat_map at) rings in
  let offsets lens =
    Array.of_list
      (List.rev
         (List.fold_left (fun acc n -> (List.hd acc + n) :: acc) [ 0 ] lens))
  in
  Polygons.v crs (points pts)
    ~rings:(offsets (List.map List.length rings))
    ~features:(offsets (List.map List.length fs))

let areas fs = Array.to_list (Nx.to_array (Polygons.area (features utm fs)))
let box x y d = [ (x, y); (x +. d, y); (x +. d, y +. d); (x, y +. d) ]

let even_odd =
  test "a feature's area is its even-odd region's, whichever way rings turn"
    (fun () ->
      let shell = box 0. 0. 100. and lake = box 20. 20. 60. in
      let island = box 40. 40. 20. in
      equal (list (float 1e-9))
        [ 10000.; 6400.; 6400.; 6800.; 2.; 6.; 0. ]
        (areas
           [
             [ shell ];
             [ shell; lake ];
             [ List.rev shell; lake ];
             [ shell; List.rev lake; island ];
             (* A bow-tie, two triangles crossing at (1, 1). *)
             [ [ (0., 0.); (2., 2.); (2., 0.); (0., 2.) ] ];
             (* Two squares overlapping by one in one feature. *)
             [ box 0. 0. 2.; box 1. 1. 2. ];
             [];
           ]))

(* A 10 m grid of UTM 33N turned by 0.3 radians, and a star of 17 points
   inside it. *)
let cells_sum =
  test "a feature's area is its cells' areas times a cell's" (fun () ->
      let t = 0.3 and d = 10. in
      let g =
        Grid.v utm ~rows:80 ~cols:80 ~origin:(499700., 5300300.)
          ~row_step:(d *. sin t, -.d *. cos t)
          ~col_step:(d *. cos t, d *. sin t)
      in
      let star =
        List.init 17 (fun k ->
            let a = 2. *. Float.pi *. float_of_int k /. 17. in
            let r = if k mod 2 = 0 then 140. else 70. in
            (150. +. (r *. cos a), -150. +. (r *. sin a)))
      in
      let p = features utm [ [ star ] ] in
      let _, _, a = Rasterize.areas g p in
      let cells = Array.fold_left ( +. ) 0. (Nx.to_array a) in
      let whole = Nx.item [ 0 ] (Polygons.area p) in
      (* Rasterize leaves out slivers thinner than τ = 1e-6 pixel along the
         edges, so beyond rounding the sums may differ by τ times the
         perimeter in pixels. *)
      let edges = List.combine star (List.tl star @ [ List.hd star ]) in
      let length ((x, y), (x', y')) = Float.hypot (x' -. x) (y' -. y) /. d in
      let perimeter = List.fold_left (fun acc e -> acc +. length e) 0. edges in
      let tol = (4e-6 *. perimeter *. d *. d) +. (1e-12 *. whole) in
      equal (float tol) whole (cells *. d *. d))

let geographic =
  test "area refuses a geographic system" (fun () ->
      raises_match (Exn.invalid_arg ~sub:"geographic") (fun () ->
          Polygons.area
            (Polygons.v Crs.lonlat (points square) ~rings:[| 0; 4 |]
               ~features:[| 0; 1 |])))

(* plane_areas.dat: features near UTM's numbers with their exact areas and
   crossings, by its header's format. polygons.mli's bound for each, with K
   from above, every slanted edge one window and each crossing two more, and
   each window's width at most 2D. *)
let plane_features =
  let lines =
    In_channel.with_open_text "plane_areas.dat" In_channel.input_lines
  in
  let words l = List.filter (( <> ) "") (String.split_on_char ' ' l) in
  let rec pairs = function
    | x :: y :: rest -> (float_of_string x, float_of_string y) :: pairs rest
    | _ -> []
  in
  List.rev
    (List.fold_left
       (fun acc l ->
         match (words l, acc) with
         | [], _ -> acc
         | w :: _, _ when w.[0] = '#' -> acc
         | [ "feature"; a; x ], _ ->
             (float_of_string a, int_of_string x, []) :: acc
         | "ring" :: pts, (a, x, rings) :: rest ->
             (a, x, rings @ [ pairs pts ]) :: rest
         | _ -> failwith ("plane_areas.dat: " ^ l))
       [] lines)

let stated_bound =
  test "a feature's area is its exact one's within the stated bound" (fun () ->
      List.iteri
        (fun k (exact, crossings, rings) ->
          let p =
            let xy (x, y) = [ x; y ] in
            let pts = List.concat_map (List.concat_map xy) rings in
            let ends =
              List.fold_left
                (fun acc r -> (List.hd acc + List.length r) :: acc)
                [ 0 ] rings
            in
            Polygons.v utm (points pts)
              ~rings:(Array.of_list (List.rev ends))
              ~features:[| 0; List.length rings |]
          in
          let x0, y0 = List.hd (List.hd rings) in
          let all = List.concat rings in
          let far acc (x, y) =
            let dx = Float.abs (x -. x0) and dy = Float.abs (y -. y0) in
            Float.max acc (Float.max dx dy)
          in
          let d = List.fold_left far 0. all in
          let ring_edges r = List.combine r (List.tl r @ [ List.hd r ]) in
          let edges = List.concat_map ring_edges rings in
          let slanted = List.filter (fun ((x, _), (x', _)) -> x <> x') edges in
          let e = float_of_int (List.length edges) in
          let x = float_of_int crossings in
          let windows = float_of_int (List.length slanted) +. (2. *. x) in
          let w = 2. *. d *. windows in
          let u = epsilon_float /. 2. in
          let bound =
            ((94. +. windows) *. u *. d *. w)
            +. (((4. *. e) +. (36. *. x)) *. u *. d *. d)
          in
          let msg = Printf.sprintf "feature %d" k in
          equal ~msg (float bound) exact (Nx.item [ 0 ] (Polygons.area p)))
        plane_features)

let () =
  exit
    (run "jord.polygons"
       [
         group "layout" [ layout; refusals; shape; to_crs ];
         group "area" [ even_odd; cells_sum; stated_bound; geographic ];
       ])
