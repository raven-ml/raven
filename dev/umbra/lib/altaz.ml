(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let pi = Float.pi

type observer = { lat : float; lon : float; height : float }

let make_observer ~lat ~lon ?(height = Unit.Length.m 0.0) () =
  let lat = Nx.item [] (Unit.Angle.to_tensor lat) in
  let lon = Nx.item [] (Unit.Angle.to_tensor lon) in
  let height = Nx.item [] (Unit.Length.to_tensor height) in
  { lat; lon; height }

let observer_height obs =
  Unit.Length.of_tensor (Nx.scalar Nx.float64 obs.height)

type t = { az : Nx.float64_t; alt : Nx.float64_t }

let alt t = Unit.Angle.of_tensor t.alt
let az t = Unit.Angle.of_tensor t.az

(* Earth Rotation Angle from UT1 Julian Date. ERA = 2π(0.7790572732640 +
   1.00273781191135448 * Du) where Du = JD_UT1 - 2451545.0 *)
let era jd_ut1 =
  let du = jd_ut1 -. 2_451_545.0 in
  let theta =
    2.0 *. pi *. (0.779_057_273_264_0 +. (1.002_737_811_911_354_48 *. du))
  in
  Float.rem theta (2.0 *. pi)

(* IAU 2006 precession angles (Capitaine et al. 2003). T = Julian centuries from
   J2000.0 TT. Returns (zeta_A, z_A, theta_A) in radians. *)
let precession_angles t_cy =
  let arcsec_to_rad x = x *. pi /. 648_000.0 in
  let t2 = t_cy *. t_cy in
  let t3 = t2 *. t_cy in
  (* zeta_A = 2.5976176'' + 2306.0809506''T + 1.0109032''T² + 0.0182337''T³ *)
  let zeta_a =
    arcsec_to_rad
      (2.597_617_6 +. (2306.080_950_6 *. t_cy) +. (1.010_903_2 *. t2)
     +. (0.018_233_7 *. t3))
  in
  (* z_A = -2.5976176'' + 2306.0803226''T + 1.0947790''T² + 0.0182273''T³ *)
  let z_a =
    arcsec_to_rad
      (~-.2.597_617_6 +. (2306.080_322_6 *. t_cy) +. (1.094_779_0 *. t2)
     +. (0.018_227_3 *. t3))
  in
  (* theta_A = 2004.1917476''T - 0.4269353''T² - 0.0418251''T³ *)
  let theta_a =
    arcsec_to_rad
      ((2004.191_747_6 *. t_cy) -. (0.426_935_3 *. t2) -. (0.041_825_1 *. t3))
  in
  (zeta_a, z_a, theta_a)

(* Apply IAU 2006 precession matrix to ICRS (RA, Dec) → mean (RA, Dec) of date.
   R = Rz(-z_A) · Ry(theta_A) · Rz(-zeta_A) *)
let precess_to_date ra dec t_cy =
  let zeta_a, z_a, theta_a = precession_angles t_cy in
  let sz = Float.sin zeta_a and cz = Float.cos zeta_a in
  let sa = Float.sin z_a and ca = Float.cos z_a in
  let st = Float.sin theta_a and ct = Float.cos theta_a in
  (* Rotation matrix elements *)
  let r11 = (ca *. ct *. cz) -. (sa *. sz) in
  let r12 = ~-.((ca *. ct *. sz) +. (sa *. cz)) in
  let r13 = ~-.(ca *. st) in
  let r21 = (sa *. ct *. cz) +. (ca *. sz) in
  let r22 = ~-.((sa *. ct *. sz) -. (ca *. cz)) in
  let r23 = ~-.(sa *. st) in
  let r31 = st *. cz in
  let r32 = ~-.(st *. sz) in
  let r33 = ct in
  let n = Nx.numel ra in
  let ra_out = Nx.zeros Nx.float64 [| n |] in
  let dec_out = Nx.zeros Nx.float64 [| n |] in
  for i = 0 to n - 1 do
    let r = Nx.item [ i ] ra in
    let d = Nx.item [ i ] dec in
    let cd = Float.cos d in
    let x = cd *. Float.cos r in
    let y = cd *. Float.sin r in
    let z = Float.sin d in
    let x' = (r11 *. x) +. (r12 *. y) +. (r13 *. z) in
    let y' = (r21 *. x) +. (r22 *. y) +. (r23 *. z) in
    let z' = (r31 *. x) +. (r32 *. y) +. (r33 *. z) in
    Nx.set_item [ i ] (Float.atan2 y' x') ra_out;
    Nx.set_item [ i ] (Float.asin (Float.max ~-.1.0 (Float.min 1.0 z'))) dec_out
  done;
  (ra_out, dec_out)

let airmass hz =
  let n = Nx.numel hz.alt in
  let out = Nx.zeros Nx.float64 [| n |] in
  let to_deg = 180.0 /. pi in
  for i = 0 to n - 1 do
    let alt_deg = Nx.item [ i ] hz.alt *. to_deg in
    (* Pickering (2002): X = 1 / sin(h + 244/(165 + 47h^1.1)) where h in deg *)
    let arg =
      alt_deg
      +. (244.0 /. (165.0 +. (47.0 *. Float.pow (Float.abs alt_deg) 1.1)))
    in
    let x = 1.0 /. Float.sin (arg *. pi /. 180.0) in
    Nx.set_item [ i ] (Float.max 1.0 x) out
  done;
  out

(* Bennett (1982) atmospheric refraction for geometric altitude. R (arcmin) =
   cot(h + 7.31/(h + 4.4)) where h in degrees. Returns refraction in radians.
   Clamps to 0 below -1°. *)
let refraction_correction alt_rad =
  let h = alt_rad *. 180.0 /. pi in
  if h < -1.0 then 0.0
  else
    let arg = (h +. (7.31 /. (h +. 4.4))) *. pi /. 180.0 in
    let r_arcmin = 1.0 /. Float.tan arg in
    r_arcmin *. pi /. (180.0 *. 60.0)

let refraction hz =
  let n = Nx.numel hz.alt in
  let out = Nx.zeros Nx.float64 [| n |] in
  for i = 0 to n - 1 do
    Nx.set_item [ i ] (refraction_correction (Nx.item [ i ] hz.alt)) out
  done;
  Unit.Angle.of_tensor out

let of_coord ?(refraction = false) ~obstime ~observer c =
  let icrs = Coord.icrs c in
  let ra_rad = Unit.Angle.to_tensor (Coord.lon icrs) in
  let dec_rad = Unit.Angle.to_tensor (Coord.lat icrs) in
  (* Convert UTC → UT1 (ignoring DUT1 < 1s) then to TT for precession *)
  let jd_utc = Time.to_jd obstime in
  let jd_ut1 = jd_utc in
  let jd_tt = Time.to_jd (Time.tai_to_tt (Time.utc_to_tai obstime)) in
  let t_cy = (jd_tt -. 2_451_545.0) /. 36_525.0 in
  (* Precess ICRS to mean RA/Dec of date *)
  let ra_date, dec_date = precess_to_date ra_rad dec_rad t_cy in
  (* Hour angle: HA = ERA + observer_lon - RA_date *)
  let era_val = era jd_ut1 in
  let n = Nx.numel ra_rad in
  let alt_out = Nx.zeros Nx.float64 [| n |] in
  let az_out = Nx.zeros Nx.float64 [| n |] in
  let slat = Float.sin observer.lat and clat = Float.cos observer.lat in
  for i = 0 to n - 1 do
    let ha = era_val +. observer.lon -. Nx.item [ i ] ra_date in
    let dec = Nx.item [ i ] dec_date in
    let sdec = Float.sin dec and cdec = Float.cos dec in
    let sha = Float.sin ha and cha = Float.cos ha in
    (* alt = asin(sin(lat)sin(dec) + cos(lat)cos(dec)cos(ha)) *)
    let sin_alt = (slat *. sdec) +. (clat *. cdec *. cha) in
    let alt = Float.asin (Float.max ~-.1.0 (Float.min 1.0 sin_alt)) in
    (* az = atan2(-cos(dec)sin(ha), cos(lat)sin(dec) -
       sin(lat)cos(dec)cos(ha)) *)
    let num = ~-.(cdec *. sha) in
    let den = (clat *. sdec) -. (slat *. cdec *. cha) in
    let az = Float.atan2 num den in
    let az = if az < 0.0 then az +. (2.0 *. pi) else az in
    let alt = if refraction then alt +. refraction_correction alt else alt in
    Nx.set_item [ i ] alt alt_out;
    Nx.set_item [ i ] az az_out
  done;
  { alt = alt_out; az = az_out }

let to_coord ~obstime ~observer t =
  let jd_utc = Time.to_jd obstime in
  let jd_ut1 = jd_utc in
  let jd_tt = Time.to_jd (Time.tai_to_tt (Time.utc_to_tai obstime)) in
  let t_cy = (jd_tt -. 2_451_545.0) /. 36_525.0 in
  let era_val = era jd_ut1 in
  let slat = Float.sin observer.lat and clat = Float.cos observer.lat in
  let zeta_a, z_a, theta_a = precession_angles t_cy in
  (* Inverse precession matrix = transpose of forward *)
  let sz = Float.sin zeta_a and cz = Float.cos zeta_a in
  let sa = Float.sin z_a and ca = Float.cos z_a in
  let st = Float.sin theta_a and ct = Float.cos theta_a in
  let r11 = (ca *. ct *. cz) -. (sa *. sz) in
  let r12 = ~-.((ca *. ct *. sz) +. (sa *. cz)) in
  let r13 = ~-.(ca *. st) in
  let r21 = (sa *. ct *. cz) +. (ca *. sz) in
  let r22 = ~-.((sa *. ct *. sz) -. (ca *. cz)) in
  let r23 = ~-.(sa *. st) in
  let r31 = st *. cz in
  let r32 = ~-.(st *. sz) in
  let r33 = ct in
  (* Transpose for inverse *)
  let ri11 = r11 and ri12 = r21 and ri13 = r31 in
  let ri21 = r12 and ri22 = r22 and ri23 = r32 in
  let ri31 = r13 and ri32 = r23 and ri33 = r33 in
  let n = Nx.numel t.alt in
  let ra_out = Nx.zeros Nx.float64 [| n |] in
  let dec_out = Nx.zeros Nx.float64 [| n |] in
  for i = 0 to n - 1 do
    let alt = Nx.item [ i ] t.alt in
    let az = Nx.item [ i ] t.az in
    let salt = Float.sin alt and calt = Float.cos alt in
    let saz = Float.sin az and caz = Float.cos az in
    (* (Alt, Az) → (HA, Dec) *)
    let sin_dec = (slat *. salt) +. (clat *. calt *. caz) in
    let dec = Float.asin (Float.max ~-.1.0 (Float.min 1.0 sin_dec)) in
    let num = ~-.(calt *. saz) in
    let den = (clat *. salt) -. (slat *. calt *. caz) in
    let ha = Float.atan2 num den in
    (* RA_date = ERA + observer_lon - HA *)
    let ra_date = era_val +. observer.lon -. ha in
    (* Deprecess: mean of date → ICRS *)
    let cd = Float.cos dec in
    let x = cd *. Float.cos ra_date in
    let y = cd *. Float.sin ra_date in
    let z = Float.sin dec in
    let x' = (ri11 *. x) +. (ri12 *. y) +. (ri13 *. z) in
    let y' = (ri21 *. x) +. (ri22 *. y) +. (ri23 *. z) in
    let z' = (ri31 *. x) +. (ri32 *. y) +. (ri33 *. z) in
    let ra = Float.atan2 y' x' in
    let ra = if ra < 0.0 then ra +. (2.0 *. pi) else ra in
    let dec = Float.asin (Float.max ~-.1.0 (Float.min 1.0 z')) in
    Nx.set_item [ i ] ra ra_out;
    Nx.set_item [ i ] dec dec_out
  done;
  Coord.of_radec
    ~ra:(Unit.Angle.of_tensor ra_out)
    ~dec:(Unit.Angle.of_tensor dec_out)
