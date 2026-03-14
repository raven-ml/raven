(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let pi = Float.pi
let f64 = Nx.float64
let galcen_distance_default = Unit.Length.of_kpc (Nx.scalar f64 8.122)
let z_sun_default = Unit.Length.of_kpc (Nx.scalar f64 0.0208)

type t = { x : Nx.float64_t; y : Nx.float64_t; z : Nx.float64_t }

let x t = Unit.Length.of_kpc t.x
let y t = Unit.Length.of_kpc t.y
let z t = Unit.Length.of_kpc t.z

(* Convert via Galactic coordinates. In Galactic (l,b) the GC is at l=0, b=0, so
   heliocentric Galactic Cartesian is: x_h = d cos(b) cos(l) toward GC y_h = d
   cos(b) sin(l) toward rotation z_h = d sin(b) toward NGP

   Galactocentric = heliocentric shifted by Sun's position: x_gc = x_h -
   galcen_distance y_gc = y_h z_gc = z_h + z_sun *)

let of_coord ?(galcen_distance = galcen_distance_default)
    ?(z_sun = z_sun_default) ~distance c =
  let galcen_distance = Nx.item [] (Unit.Length.in_kpc galcen_distance) in
  let z_sun = Nx.item [] (Unit.Length.in_kpc z_sun) in
  let gal = Coord.galactic c in
  let l_rad = Unit.Angle.to_tensor (Coord.lon gal) in
  let b_rad = Unit.Angle.to_tensor (Coord.lat gal) in
  let d_kpc = Unit.Length.in_kpc distance in
  let n = Nx.numel l_rad in
  let x_out = Nx.zeros Nx.float64 [| n |] in
  let y_out = Nx.zeros Nx.float64 [| n |] in
  let z_out = Nx.zeros Nx.float64 [| n |] in
  for i = 0 to n - 1 do
    let l = Nx.item [ i ] l_rad in
    let b = Nx.item [ i ] b_rad in
    let d = Nx.item [ i ] d_kpc in
    let cb = Float.cos b in
    let xh = d *. cb *. Float.cos l in
    let yh = d *. cb *. Float.sin l in
    let zh = d *. Float.sin b in
    Nx.set_item [ i ] (xh -. galcen_distance) x_out;
    Nx.set_item [ i ] yh y_out;
    Nx.set_item [ i ] (zh +. z_sun) z_out
  done;
  { x = x_out; y = y_out; z = z_out }

let to_coord ?(galcen_distance = galcen_distance_default)
    ?(z_sun = z_sun_default) t =
  let galcen_distance = Nx.item [] (Unit.Length.in_kpc galcen_distance) in
  let z_sun = Nx.item [] (Unit.Length.in_kpc z_sun) in
  let n = Nx.numel t.x in
  let l_out = Nx.zeros Nx.float64 [| n |] in
  let b_out = Nx.zeros Nx.float64 [| n |] in
  let d_out = Nx.zeros Nx.float64 [| n |] in
  for i = 0 to n - 1 do
    let xg = Nx.item [ i ] t.x in
    let yg = Nx.item [ i ] t.y in
    let zg = Nx.item [ i ] t.z in
    let xh = xg +. galcen_distance in
    let yh = yg in
    let zh = zg -. z_sun in
    let d = Float.sqrt ((xh *. xh) +. (yh *. yh) +. (zh *. zh)) in
    let b = Float.asin (Float.max ~-.1.0 (Float.min 1.0 (zh /. d))) in
    let l = Float.atan2 yh xh in
    let l = if l < 0.0 then l +. (2.0 *. pi) else l in
    Nx.set_item [ i ] l l_out;
    Nx.set_item [ i ] b b_out;
    Nx.set_item [ i ] d d_out
  done;
  let coord =
    Coord.of_galactic
      ~l:(Unit.Angle.of_tensor l_out)
      ~b:(Unit.Angle.of_tensor b_out)
  in
  let distance = Unit.Length.of_kpc d_out in
  (coord, distance)
