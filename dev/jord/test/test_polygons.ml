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

let () =
  exit
    (run "jord.polygons" [ group "layout" [ layout; refusals; shape; to_crs ] ])
