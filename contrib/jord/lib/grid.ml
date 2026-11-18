(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = {
  crs : Crs.t;
  rows : int;
  cols : int;
  origin : float * float;
  row_step : float * float;
  col_step : float * float;
}

let strf = Printf.sprintf

(* A product that OCaml's arm64 backend cannot fuse into the addition that uses
   it, as amd64's does not, so the result is the same bits on both. *)
let mul a b = Sys.opaque_identity (a *. b)
let det (rx, ry) (cx, cy) = mul rx cy -. mul cx ry

let v crs ~rows ~cols ~origin ~row_step ~col_step =
  if rows < 0 || cols < 0 then
    invalid_arg (strf "Jord.Grid.v: size %d × %d is negative" rows cols);
  let finite (a, b) = Float.is_finite a && Float.is_finite b in
  if not (finite origin && finite row_step && finite col_step) then
    invalid_arg "Jord.Grid.v: a number is not finite";
  if det row_step col_step = 0. then
    invalid_arg "Jord.Grid.v: the steps are parallel (determinant 0)";
  { crs; rows; cols; origin; row_step; col_step }

let world g p =
  let batch, r, c = Points.split "Jord.Grid.world" p in
  let (ox, oy), (rx, ry), (cx, cy) = (g.origin, g.row_step, g.col_step) in
  let along a b = Nx.add (Nx.mul_s r a) (Nx.mul_s c b) in
  Points.join batch (Nx.add_s (along rx cx) ox) (Nx.add_s (along ry cy) oy)

let pixel g p =
  let batch, x, y = Points.split "Jord.Grid.pixel" p in
  let (ox, oy), (rx, ry), (cx, cy) = (g.origin, g.row_step, g.col_step) in
  let dx = Nx.sub_s x ox and dy = Nx.sub_s y oy in
  let r, c =
    (* An axis-aligned grid divides once per coordinate, so a vertex on a cell
       border of an exactly stated grid maps to an integer. *)
    if rx = 0. && cy = 0. then (Nx.div_s dy ry, Nx.div_s dx cx)
    else if ry = 0. && cx = 0. then (Nx.div_s dx rx, Nx.div_s dy cy)
    else
      let d = det g.row_step g.col_step in
      ( Nx.div_s (Nx.sub (Nx.mul_s dx cy) (Nx.mul_s dy cx)) d,
        Nx.div_s (Nx.sub (Nx.mul_s dy rx) (Nx.mul_s dx ry)) d )
  in
  Points.join batch r c

let centres ~at g =
  let half n =
    Nx.add_s (Nx.arange_f_at at Nx.float64 0. (float_of_int n) 1.) 0.5
  in
  let shape = [| g.rows; g.cols |] in
  let r = Nx.broadcast_to shape (Nx.reshape [| g.rows; 1 |] (half g.rows)) in
  let c = Nx.broadcast_to shape (Nx.reshape [| 1; g.cols |] (half g.cols)) in
  world g (Nx.stack ~axis:2 [ r; c ])

let window ~row ~col ~rows ~cols g =
  if rows < 0 || cols < 0 then
    invalid_arg (strf "Jord.Grid.window: size %d × %d is negative" rows cols);
  let (ox, oy), (rx, ry), (cx, cy) = (g.origin, g.row_step, g.col_step) in
  let r = float_of_int row and c = float_of_int col in
  (* As [world] computes it, each product rounded alone. *)
  let origin = (mul r rx +. mul c cx +. ox, mul r ry +. mul c cy +. oy) in
  { g with rows; cols; origin }

let equal a b =
  Crs.equal a.crs b.crs && a.rows = b.rows && a.cols = b.cols
  && a.origin = b.origin && a.row_step = b.row_step && a.col_step = b.col_step
