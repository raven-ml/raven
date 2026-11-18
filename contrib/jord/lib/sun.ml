(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let deg = Float.pi /. 180.

(* Days from J2000.0, 2000-01-01T12:00:00, which is instant 946728000. *)
let days t = Nx.div_s (Nx.sub_s t 946728000.) 86400.

(* USNO's mean anomaly g, in radians. *)
let anomaly d = Nx.mul_s (Nx.add_s (Nx.mul_s d 0.98560028) 357.529) deg

let distance t =
  let g = anomaly (days t) in
  Nx.sub
    (Nx.rsub_s 1.00014 (Nx.mul_s (Nx.cos g) 0.01671))
    (Nx.mul_s (Nx.cos (Nx.mul_s g 2.)) 0.00014)

(* Raises through [Nx.check] where a point is neither NaN nor a longitude and
   latitude. *)
let require p lon lat =
  let nan = Nx.logical_or (Nx.isnan lon) (Nx.isnan lat) in
  let valid = Crs.in_domain ~src:Crs.lonlat ~dst:Crs.lonlat p in
  let ok = Nx.logical_or nan valid in
  Nx.check
    Nx.Ptree.(pair tensor tensor)
    ok (lon, lat)
    (fun i (lon, lat) ->
      let at = String.concat ", " (List.map string_of_int (Array.to_list i)) in
      Invalid_argument
        (Printf.sprintf
           "Jord.Sun.angles: point [%s] (%.17g, %.17g) is not a longitude and \
            latitude"
           at (Nx.item [] lon) (Nx.item [] lat)))

let angles (type d) (t : d Nx.float64_t) (p : d Nx.float64_t) =
  let batch, lon, lat = Points.split "Jord.Sun.angles" p in
  let lon = Nx.reshape batch lon and lat = Nx.reshape batch lat in
  require p lon lat;
  let d = days t in
  (* The ecliptic longitude L and the obliquity e, and the Sun's direction
     in the equator and equinox of date: (cos L, cos e sin L, sin e sin L). *)
  let g = anomaly d in
  let q = Nx.add_s (Nx.mul_s d 0.98564736) 280.459 in
  let l =
    Nx.add q
      (Nx.add
         (Nx.mul_s (Nx.sin g) 1.915)
         (Nx.mul_s (Nx.sin (Nx.mul_s g 2.)) 0.020))
  in
  let l = Nx.mul_s l deg in
  let e = Nx.mul_s (Nx.rsub_s 23.439 (Nx.mul_s d 0.00000036)) deg in
  let sl = Nx.sin l in
  let x = Nx.cos l and y = Nx.mul (Nx.cos e) sl and z = Nx.mul (Nx.sin e) sl in
  (* Turned by the sidereal angle of the point's meridian, GMST plus its
     longitude: c towards that meridian, s 90° east of it, in the equator. *)
  let gmst = Nx.mul_s (Nx.add_s (Nx.mul_s d 24.065709824279) 18.697375) 15. in
  let psi = Nx.mul_s (Nx.add gmst lon) deg in
  let cp = Nx.cos psi and sp = Nx.sin psi in
  let c = Nx.add (Nx.mul x cp) (Nx.mul y sp) in
  let s = Nx.sub (Nx.mul y cp) (Nx.mul x sp) in
  (* GN 7-2's local axes at latitude phi: U east, V north, W up. *)
  let phi = Nx.mul_s lat deg in
  let cf = Nx.cos phi and sf = Nx.sin phi in
  let east = s in
  let north = Nx.sub (Nx.mul cf z) (Nx.mul sf c) in
  let up = Nx.add (Nx.mul sf z) (Nx.mul cf c) in
  let zenith = Nx.div_s (Nx.atan2 (Nx.hypot east north) up) deg in
  let azimuth = Nx.div_s (Nx.atan2 east north) deg in
  let azimuth =
    Nx.where (Nx.less_s azimuth 0.) (Nx.add_s azimuth 360.) azimuth
  in
  (zenith, azimuth)
