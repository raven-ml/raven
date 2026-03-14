(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Coordinates, time scales, and survey selection.

   Demonstrates Umbra's coordinate, time, and observing modules: - Coord:
   celestial coordinates with frame transforms (ICRS, Galactic, Ecliptic,
   Supergalactic) and angular separation. - Time: astronomical time with
   type-safe scale conversions (UTC, TAI, TT, TDB) and ISO 8601 parsing. -
   Altaz: horizontal coordinates, airmass, and atmospheric refraction.

   Combines these into a survey selection function that determines which targets
   are observable given an observer, time, and observing constraints. *)

open Nx
open Umbra

let f64 = Nx.float64

let () =
  Printf.printf "Coordinates, time scales, and survey selection\n";
  Printf.printf "===============================================\n\n";

  (* --- Part 1: Coordinate frames --- *)
  Printf.printf "Part 1: Coordinate frame transforms\n";
  Printf.printf "------------------------------------\n\n";

  let targets =
    [|
      ("Galactic center", 266.417, -28.936);
      ("Vega", 279.235, 38.784);
      ("North Galactic Pole", 192.860, 27.128);
      ("LMC", 80.894, -69.756);
      ("M31 (Andromeda)", 10.685, 41.269);
    |]
  in

  Printf.printf "%20s  %8s  %8s  %8s  %8s\n" "Object" "RA" "Dec" "l" "b";
  Printf.printf "%20s  %8s  %8s  %8s  %8s\n" "--------------------" "--------"
    "--------" "--------" "--------";

  Array.iter
    (fun (name, ra_deg, dec_deg) ->
      let coord =
        Coord.of_radec
          ~ra:(Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| ra_deg |]))
          ~dec:(Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| dec_deg |]))
      in
      let gal = Coord.galactic coord in
      let l = item [ 0 ] (Unit.Angle.in_deg (Coord.lon gal)) in
      let b = item [ 0 ] (Unit.Angle.in_deg (Coord.lat gal)) in
      Printf.printf "%20s  %8.2f  %+8.2f  %8.2f  %+8.2f\n" name ra_deg dec_deg l
        b)
    targets;
  Printf.printf "\n";

  (* Angular separation *)
  Printf.printf "Angular separations:\n";
  let vega =
    Coord.of_radec
      ~ra:(Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| 279.235 |]))
      ~dec:(Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| 38.784 |]))
  in
  let altair =
    Coord.of_radec
      ~ra:(Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| 297.696 |]))
      ~dec:(Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| 8.868 |]))
  in
  let deneb =
    Coord.of_radec
      ~ra:(Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| 310.358 |]))
      ~dec:(Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| 45.280 |]))
  in
  let sep_va = item [ 0 ] (Unit.Angle.in_deg (Coord.separation vega altair)) in
  let sep_vd = item [ 0 ] (Unit.Angle.in_deg (Coord.separation vega deneb)) in
  let sep_ad = item [ 0 ] (Unit.Angle.in_deg (Coord.separation altair deneb)) in
  Printf.printf "  Vega - Altair:  %.2f deg\n" sep_va;
  Printf.printf "  Vega - Deneb:   %.2f deg\n" sep_vd;
  Printf.printf "  Altair - Deneb: %.2f deg\n" sep_ad;
  Printf.printf "  (Summer Triangle)\n\n";

  (* --- Part 2: Time scales --- *)
  Printf.printf "Part 2: Astronomical time scales\n";
  Printf.printf "--------------------------------\n\n";

  let t_utc = Time.of_iso "2024-06-21T04:00:00" in
  let t_tai = Time.utc_to_tai t_utc in
  let t_tt = Time.tai_to_tt t_tai in
  let t_tdb = Time.tt_to_tdb t_tt in

  Printf.printf "  UTC: %s\n" (Time.to_iso t_utc);
  Printf.printf "  JD (UTC): %.6f\n" (Time.to_jd t_utc);
  Printf.printf "  MJD (UTC): %.6f\n" (Time.to_mjd t_utc);
  Printf.printf "  JD (TAI): %.6f\n" (Time.to_jd t_tai);
  Printf.printf "  JD (TT):  %.6f\n" (Time.to_jd t_tt);
  Printf.printf "  JD (TDB): %.6f\n" (Time.to_jd t_tdb);

  let dt_tai_utc =
    Unit.to_float (Time.diff t_tai (Time.unsafe_of_jd (Time.to_jd t_utc)))
  in
  Printf.printf "\n  TAI - UTC = %.1f s (leap seconds)\n" (dt_tai_utc *. 86400.0);

  let t_j2000 = Time.of_iso "2000-01-01T12:00:00" in
  let dt_j2000 = Unit.to_float (Time.diff t_utc t_j2000) in
  Printf.printf "  Days since J2000.0: %.2f\n\n" (dt_j2000 *. 86400.0 /. 86400.0);

  (* --- Part 3: Horizontal coordinates and airmass --- *)
  Printf.printf "Part 3: Altitude-azimuth and airmass\n";
  Printf.printf "------------------------------------\n\n";

  (* Observer at Cerro Pachon (Rubin site) *)
  let obs =
    Altaz.make_observer
      ~lat:(Unit.Angle.deg (-30.2444))
      ~lon:(Unit.Angle.deg (-70.7494))
      ~height:(Unit.Length.m 2663.0) ()
  in
  let obstime = Time.of_iso "2024-06-21T04:00:00" in

  Printf.printf "  Observer: Cerro Pachon (Rubin Observatory)\n";
  Printf.printf "    Lat: %.4f deg\n" (-30.2444);
  Printf.printf "    Lon: %.4f deg\n" (-70.7494);
  Printf.printf "    Elevation: %.0f m\n" 2663.0;
  Printf.printf "  Time: 2024-06-21 04:00 UTC\n\n";

  let stars =
    [|
      ("Vega", 279.235, 38.784);
      ("Sirius", 101.287, -16.716);
      ("Canopus", 95.988, -52.696);
      ("Alpha Cen", 219.902, -60.834);
      ("Fomalhaut", 344.413, -29.622);
    |]
  in

  Printf.printf "%12s  %7s  %7s  %8s\n" "Star" "Alt" "Az" "Airmass";
  Printf.printf "%12s  %7s  %7s  %8s\n" "------------" "-------" "-------"
    "--------";

  Array.iter
    (fun (name, ra_deg, dec_deg) ->
      let coord =
        Coord.of_radec
          ~ra:(Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| ra_deg |]))
          ~dec:(Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| dec_deg |]))
      in
      let hz = Altaz.of_coord ~obstime ~observer:obs coord in
      let alt_deg =
        item [ 0 ] (Unit.Angle.to_tensor (Altaz.alt hz)) *. 180.0 /. Float.pi
      in
      let az_deg =
        item [ 0 ] (Unit.Angle.to_tensor (Altaz.az hz)) *. 180.0 /. Float.pi
      in
      let am = item [ 0 ] (Altaz.airmass hz) in
      Printf.printf "%12s  %+7.1f  %7.1f  %8.2f\n" name alt_deg az_deg am)
    stars;
  Printf.printf "\n";

  (* --- Part 4: Survey selection --- *)
  Printf.printf "Part 4: Survey selection function\n";
  Printf.printf "---------------------------------\n\n";

  let mag_limit = 20.0 in
  let airmass_cut = 2.0 in
  Printf.printf "  Selection criteria:\n";
  Printf.printf "    Magnitude limit: r < %.1f (AB)\n" mag_limit;
  Printf.printf "    Airmass cut: X < %.1f\n" airmass_cut;
  Printf.printf "    Above horizon: alt > 0 deg\n\n";

  let bp = Filters.rubin_r in
  let norm = Nx.scalar f64 (Float.exp (-49.0)) in

  let star_data =
    [|
      ("Vega", 279.235, 38.784, 5800.0);
      ("Sirius", 101.287, -16.716, 9940.0);
      ("Canopus", 95.988, -52.696, 7350.0);
      ("Alpha Cen", 219.902, -60.834, 5790.0);
      ("Fomalhaut", 344.413, -29.622, 8590.0);
    |]
  in

  Printf.printf "%12s  %7s  %8s  %6s  %s\n" "Star" "Alt" "Airmass" "r_mag"
    "Select?";
  Printf.printf "%12s  %7s  %8s  %6s  %s\n" "------------" "-------" "--------"
    "------" "-------";

  Array.iter
    (fun (name, ra_deg, dec_deg, temp_k) ->
      let coord =
        Coord.of_radec
          ~ra:(Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| ra_deg |]))
          ~dec:(Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| dec_deg |]))
      in
      let hz = Altaz.of_coord ~obstime ~observer:obs coord in
      let alt_deg =
        item [ 0 ] (Unit.Angle.to_tensor (Altaz.alt hz)) *. 180.0 /. Float.pi
      in
      let am = item [ 0 ] (Altaz.airmass hz) in

      (* Synthetic magnitude through Rubin r-band *)
      let temp = Unit.Temperature.of_kelvin (Nx.scalar f64 temp_k) in
      let bp_wave = Photometry.wavelength bp in
      let sed =
        Spectrum.blackbody ~temperature:temp ~wavelength:bp_wave
        |> Spectrum.scale norm |> Spectrum.as_flux_density
      in
      let r_mag = item [] (Photometry.ab_mag bp sed) in

      let selected = alt_deg > 0.0 && am < airmass_cut && r_mag < mag_limit in
      Printf.printf "%12s  %+7.1f  %8.2f  %6.2f  %s\n" name alt_deg am r_mag
        (if selected then "YES" else "no"))
    star_data;

  Printf.printf "\n  Height stored: %.0f m\n"
    (item [] (Unit.Length.to_tensor (Altaz.observer_height obs)))
