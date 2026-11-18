(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Jord

let deg = Float.pi /. 180.

let instant s =
  match Time.of_rfc3339 s with Ok t -> t | Error e -> fail e

let floats x = Nx.to_array (Nx.reshape [| -1 |] x)
let times ts = Nx.create Nx.float64 [| Array.length ts |] ts

let points ps =
  Nx.create Nx.float64 [| Array.length ps; 2 |]
    (Array.concat (Array.to_list (Array.map (fun (a, b) -> [| a; b |]) ps)))

(* The difference of two azimuths, in (-180, 180]. *)
let turn a b =
  let d = Float.rem (a -. b) 360. in
  if d > 180. then d -. 360. else if d <= -180. then d +. 360. else d

(* The bounds Sun states: the direction 0.86', the distance 1.1e-4 au, the
   zenith 0.014°, and the azimuth's arc 0.014° plus 0.6 arcseconds over the
   cosine of the latitude. Measured against DE441: the error of the
   direction in the Earth's frame, which depends on the instant alone, is at
   most 0.0109° over 57,676 daylight instants from 1972 to 2100 at 15 points
   within 75° of the equator; at 17 points from 88°S to 80°N, 1972 to 2100,
   the largest errors seen are 0.0125° in zenith and, at 88°S, 0.0134° of
   arc in azimuth. *)
let direction_bound = 0.86 /. 60.
let distance_bound = 1.1e-4
let zenith_bound = 0.014
let azimuth_bound lat = 0.014 +. (0.6 /. 3600. /. Float.cos (lat *. deg))

(* The rows of sun_horizons.dat: [g] rows (instant, right ascension,
   declination, distance), [t] rows (instant, longitude, latitude, azimuth,
   elevation). *)
let geocentric, topocentric =
  let ic = open_in "sun_horizons.dat" in
  let rec go g t =
    match input_line ic with
    | exception End_of_file ->
        close_in ic;
        (Array.of_list (List.rev g), Array.of_list (List.rev t))
    | l when String.length l = 0 || l.[0] = '#' -> go g t
    | l when l.[0] = 'g' ->
        go (Scanf.sscanf l "g %f %f %f %f" (fun a b c d -> (a, b, c, d)) :: g) t
    | l ->
        go g
          (Scanf.sscanf l "t %f %f %f %f %f" (fun a b c d e -> (a, b, c, d, e))
          :: t)
  in
  go [] []

(* USNO's Greenwich mean sidereal time, in degrees, as Sun states it. *)
let gmst t =
  let d = (t -. 946728000.) /. 86400. in
  15. *. (18.697375 +. (24.065709824279 *. d))

(* Seen from the point whose zenith points at Horizons' (RA, Dec) of date,
   latitude Dec and longitude RA less sidereal time, the Sun's zenith is the
   angle between its direction and Horizons'. *)
let direction =
  test "the direction is within 0.86' of DE441's, 1950 to 2100" (fun () ->
      let t = Array.map (fun (t, _, _, _) -> t) geocentric in
      let p =
        Array.map
          (fun (t, ra, dec, _) -> (turn (ra -. gmst t) 0., dec))
          geocentric
      in
      let z, _ = Sun.angles (times t) (points p) in
      Array.iteri
        (fun i z ->
          let msg = Printf.sprintf "instant %.0f" t.(i) in
          equal ~msg (float direction_bound) 0. z)
        (floats z))

let distance =
  test "the distance is within 1.1e-4 au of DE441's" (fun () ->
      let t = Array.map (fun (t, _, _, _) -> t) geocentric in
      let r = floats (Sun.distance (times t)) in
      Array.iteri
        (fun i (t, _, _, want) ->
          let msg = Printf.sprintf "instant %.0f" t in
          equal ~msg (float distance_bound) want r.(i))
        geocentric)

let sites =
  test "zenith and azimuth within their bounds at nine sites" (fun () ->
      let t = Array.map (fun (t, _, _, _, _) -> t) topocentric in
      let p = Array.map (fun (_, lon, lat, _, _) -> (lon, lat)) topocentric in
      let z, a = Sun.angles (times t) (points p) in
      let z = floats z and a = floats a in
      Array.iteri
        (fun i (t, lon, lat, az, el) ->
          let msg = Printf.sprintf "%g, %g at %.0f" lon lat t in
          equal ~msg (float zenith_bound) (90. -. el) z.(i);
          let arc = turn a.(i) az *. Float.sin (z.(i) *. deg) in
          equal ~msg (float (azimuth_bound lat)) 0. arc;
          is_true ~msg (a.(i) >= 0. && a.(i) <= 360.))
        topocentric)

(* Catalogues *)

let utm33 = Crs.utm ~zone:33 ~north:true

let to_lonlat xy =
  Crs.transform ~src:utm33 ~dst:Crs.lonlat (points xy)

(* Microsoft's Planetary Computer item LC09_L2SP_193024_20240629_02_T1
   (stac/mpc_landsat_c2_l2.json): its datetime, and the centre of its grid,
   proj:transform [30, 0, 228885, 0, -30, 5850915] and proj:shape [8141,
   8061], where Landsat states view:sun_elevation and view:sun_azimuth. Its
   MTL.txt (asset mtl.txt) states EARTH_SUN_DISTANCE 1.0166259. *)
let landsat =
  test "Landsat 9's scene angles and distance, from its item" (fun () ->
      let t = Nx.scalar Nx.float64 (instant "2024-06-29T10:02:04.303380Z") in
      let z, a = Sun.angles t (to_lonlat [| (349800., 5728800.) |]) in
      let z = (floats z).(0) and a = (floats a).(0) in
      equal (float zenith_bound) (90. -. 58.45652183) z;
      equal (float (azimuth_bound 52.)) 0.
        (turn a 148.0085495 *. Float.sin (z *. deg));
      equal (float distance_bound) 1.0166259
        (Nx.item [] (Sun.distance t)))

(* Shapes and refusals *)

let broadcast =
  test "a column of instants serves a tile's rows" (fun () ->
      let t0 = instant "2024-06-29T10:26:13Z" in
      let t = Nx.create Nx.float64 [| 3; 1 |] [| t0; t0 +. 5.; t0 +. 10. |] in
      let p =
        Nx.init Nx.float64 [| 3; 4; 2 |] (fun ix ->
            if ix.(2) = 0 then 12. +. Float.of_int ix.(1)
            else 52. -. Float.of_int ix.(0))
      in
      let z, a = Sun.angles t p in
      equal (array int) [| 3; 4 |] (Nx.shape z);
      equal (array int) [| 3; 4 |] (Nx.shape a);
      for i = 0 to 2 do
        for j = 0 to 3 do
          let zi, ai =
            Sun.angles
              (Nx.scalar Nx.float64 (t0 +. (5. *. Float.of_int i)))
              (points [| (12. +. Float.of_int j, 52. -. Float.of_int i) |])
          in
          let msg = Printf.sprintf "(%d, %d)" i j in
          equal ~msg float_exact (floats zi).(0) (Nx.item [ i; j ] z);
          equal ~msg float_exact (floats ai).(0) (Nx.item [ i; j ] a)
        done
      done)

let refusals =
  test "points that are not a longitude and latitude" (fun () ->
      let t = Nx.scalar Nx.float64 0. in
      raises_match (Exn.invalid_arg ~sub:"last axis") (fun () ->
          Sun.angles t (Nx.zeros Nx.float64 [| 4; 3 |]));
      raises_match (Exn.invalid_arg ~sub:"Jord.Sun.angles: point [1]")
        (fun () -> Sun.angles t (points [| (0., 0.); (10., 91.) |]));
      raises_match (Exn.invalid_arg ~sub:"Jord.Sun.angles: point [0]")
        (fun () -> Sun.angles t (points [| (Float.infinity, 0.) |]));
      let z, a = Sun.angles t (points [| (Float.nan, 10.); (10., 90.) |]) in
      is_true (Float.is_nan (floats z).(0) && Float.is_nan (floats a).(0));
      is_true (Float.is_finite (floats z).(1)))

(* Gradients *)

let gradient =
  test "the gradients are the angles' derivatives" (fun () ->
      let t0 = instant "2024-06-29T10:26:13Z" in
      let p0 = [| (12.8, 52.7); (-60., -3.); (140., -35.) |] in
      let pick which = if which = 0 then fst else snd in
      List.iter
        (fun which ->
          let name = if which = 0 then "zenith" else "azimuth" in
          let of_t t = Nx.sum ((pick which) (Sun.angles t (points p0))) in
          let of_p p =
            Nx.sum ((pick which) (Sun.angles (Nx.scalar Nx.float64 t0) p))
          in
          let at_t t = Nx.item [] (of_t (Nx.scalar Nx.float64 t)) in
          let h = 0.5 in
          let want = (at_t (t0 +. h) -. at_t (t0 -. h)) /. (2. *. h) in
          let g = Nx.item [] (Rune.grad' of_t (Nx.scalar Nx.float64 t0)) in
          equal ~msg:(name ^ ", time") (float_rel ~rel:1e-6 ~abs:1e-12) want g;
          let g = floats (Rune.grad' of_p (points p0)) in
          Array.iteri
            (fun k g ->
              let i = k / 2 and c = k mod 2 in
              (* A longitude is added to sidereal time, some 10⁶ degrees, so a
                 step below 1e-3° rounds. *)
              let h = 1e-3 in
              let moved s =
                points
                  (Array.mapi
                     (fun i' (a, b) ->
                       if i' <> i then (a, b)
                       else if c = 0 then (a +. s, b)
                       else (a, b +. s))
                     p0)
              in
              let at s = Nx.item [] (of_p (moved s)) in
              let want = (at h -. at (-.h)) /. (2. *. h) in
              let msg = Printf.sprintf "%s, point %d axis %d" name i c in
              equal ~msg (float_rel ~rel:1e-6 ~abs:1e-9) want g)
            g)
        [ 0; 1 ])

let () =
  exit
    (run "jord.sun"
       [
         group "Horizons" [ direction; distance; sites ];
         group "catalogues" [ landsat ];
         group "shapes" [ broadcast; refusals ];
         gradient;
       ])
