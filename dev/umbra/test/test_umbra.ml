(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Umbra

let eps = 1e-6
let f64 = Nx.float64
let v x = Nx.item [] x

(* Unit tests *)

let test_length_conversion () =
  let d = Unit.Length.kpc 10.0 in
  let mpc = v (Unit.Length.in_mpc d) in
  is_true ~msg:"10 kpc = 0.01 Mpc" (Float.abs (mpc -. 0.01) < eps);
  let m = v (Unit.Length.in_m d) in
  let back = v (Unit.Length.in_kpc (Unit.Length.m m)) in
  is_true ~msg:"kpc -> m -> kpc roundtrip" (Float.abs (back -. 10.0) < eps)

let test_length_arithmetic () =
  let open Unit in
  let d = Length.kpc 10.0 + Length.pc 500.0 in
  let kpc = v (Length.in_kpc d) in
  is_true ~msg:"10 kpc + 500 pc = 10.5 kpc" (Float.abs (kpc -. 10.5) < eps)

let test_mass_conversion () =
  let m = Unit.Mass.solar_mass 1.0 in
  let kg = v (Unit.Mass.in_kg m) in
  is_true ~msg:"1 Msun ~ 1.988e30 kg"
    (Float.abs (kg -. 1.9884e30) /. 1.9884e30 < 1e-4)

let test_velocity_cross_dim () =
  let d = Unit.Length.km 100.0 in
  let t = Unit.Time.s 10.0 in
  let vel = Unit.length_per_time d t in
  let km_s = v (Unit.Velocity.in_km_s vel) in
  is_true ~msg:"100 km / 10 s = 10 km/s" (Float.abs (km_s -. 10.0) < eps)

let test_angle_trig () =
  let a = Unit.Angle.deg 90.0 in
  is_true ~msg:"sin(90°) = 1"
    (Float.abs (Nx.item [] (Unit.Angle.sin a) -. 1.0) < eps);
  is_true ~msg:"cos(90°) = 0" (Float.abs (Nx.item [] (Unit.Angle.cos a)) < eps)

let test_wavelength_frequency () =
  let lam = Unit.Length.nm 500.0 in
  let nu = Unit.wavelength_to_frequency lam in
  let lam2 = Unit.frequency_to_wavelength nu in
  let nm2 = v (Unit.Length.in_nm lam2) in
  is_true ~msg:"wavelength -> freq -> wavelength roundtrip"
    (Float.abs (nm2 -. 500.0) < eps)

let test_phantom_type_safety () =
  (* This is a compile-time test: the following should NOT typecheck: let _ =
     Unit.(Length.m 1.0 + Mass.kg 1.0) The fact that this module compiles proves
     type safety. *)
  let _d = Unit.(Length.m 1.0 + Length.km 1.0) in
  let _m = Unit.(Mass.kg 1.0 + Mass.g 500.0) in
  ()

(* Const tests *)

let test_const_c () =
  let c_km_s = v (Unit.Velocity.in_km_s Const.c) in
  is_true ~msg:"c ~ 299792 km/s" (Float.abs (c_km_s -. 299792.458) < 1.0)

(* Coord tests *)

let deg_eps = 1e-6

let test_coord_roundtrip () =
  let ra =
    Unit.Angle.of_deg (Nx.create f64 [| 4 |] [| 180.0; 0.0; 90.0; 266.405 |])
  in
  let dec =
    Unit.Angle.of_deg (Nx.create f64 [| 4 |] [| 45.0; -30.0; 0.0; -28.936 |])
  in
  let c = Coord.of_radec ~ra ~dec in
  let gal = Coord.galactic c in
  let back = Coord.icrs gal in
  let ra' = Unit.Angle.in_deg (Coord.ra back) in
  let dec' = Unit.Angle.in_deg (Coord.dec back) in
  let ra_orig = Unit.Angle.in_deg ra in
  let dec_orig = Unit.Angle.in_deg dec in
  for i = 0 to 3 do
    is_true
      ~msg:(Printf.sprintf "RA roundtrip [%d]" i)
      (Float.abs (Nx.item [ i ] ra_orig -. Nx.item [ i ] ra') < deg_eps);
    is_true
      ~msg:(Printf.sprintf "Dec roundtrip [%d]" i)
      (Float.abs (Nx.item [ i ] dec_orig -. Nx.item [ i ] dec') < deg_eps)
  done

let test_separation_poles () =
  let c1 =
    Coord.of_radec
      ~ra:(Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| 0.0 |]))
      ~dec:(Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| 90.0 |]))
  in
  let c2 =
    Coord.of_radec
      ~ra:(Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| 0.0 |]))
      ~dec:(Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| -90.0 |]))
  in
  let sep = Coord.separation c1 c2 in
  is_true ~msg:"Pole separation = 180"
    (Float.abs (Nx.item [ 0 ] (Unit.Angle.in_deg sep) -. 180.0) < deg_eps)

(* Cosmo tests *)

let test_cosmo_distances () =
  let z = Nx.scalar f64 0.1 in
  let dc = v (Unit.Length.in_mpc (Cosmo.comoving_distance z)) in
  is_true
    ~msg:(Printf.sprintf "comoving(0.1) ~ 421 Mpc, got %.1f" dc)
    (Float.abs (dc -. 421.0) < 5.0);
  let dl = v (Unit.Length.in_mpc (Cosmo.luminosity_distance z)) in
  is_true
    ~msg:(Printf.sprintf "luminosity(0.1) ~ 463 Mpc, got %.1f" dl)
    (Float.abs (dl -. 463.0) < 5.0)

let test_cosmo_lookback () =
  let z = Nx.scalar f64 1.0 in
  let t = v (Unit.Time.in_gyr (Cosmo.lookback_time z)) in
  is_true
    ~msg:(Printf.sprintf "lookback(1.0) ~ 7.7 Gyr, got %.1f" t)
    (Float.abs (t -. 7.7) < 0.3)

let test_cosmo_angular_scale () =
  let phys = Unit.Length.kpc 1.0 in
  let z = Nx.scalar f64 0.022 in
  let ang = Cosmo.angular_size ~z phys in
  let arcsec = v (Unit.Angle.in_arcsec ang) in
  is_true
    ~msg:(Printf.sprintf "1 kpc at z=0.022 ~ 2.3 arcsec, got %.2f" arcsec)
    (Float.abs (arcsec -. 2.3) < 0.2)

(* Cosmo: high-z regression tests. These catch quadrature under-resolution at
   large z. *)

let test_cosmo_age_planck18 () =
  let p = Cosmo.planck18 in
  let t = v (Unit.Time.in_gyr (Cosmo.age ~p (Nx.scalar f64 0.0))) in
  is_true
    ~msg:(Printf.sprintf "age(Planck18, z=0) ~ 13.8 Gyr, got %.1f" t)
    (Float.abs (t -. 13.8) < 0.3)

let test_cosmo_age_at_z1 () =
  let p = Cosmo.planck18 in
  let age_0 = v (Unit.Time.in_gyr (Cosmo.age ~p (Nx.scalar f64 0.0))) in
  let age_1 = v (Unit.Time.in_gyr (Cosmo.age ~p (Nx.scalar f64 1.0))) in
  let lb_1 =
    v (Unit.Time.in_gyr (Cosmo.lookback_time ~p (Nx.scalar f64 1.0)))
  in
  is_true
    ~msg:
      (Printf.sprintf
         "age(z=0) - age(z=1) = lookback(z=1): %.2f - %.2f = %.2f vs %.2f" age_0
         age_1 (age_0 -. age_1) lb_1)
    (Float.abs (age_0 -. age_1 -. lb_1) < 0.05)

let test_cosmo_comoving_cmb () =
  let p = Cosmo.planck18 in
  let dc =
    v (Unit.Length.in_mpc (Cosmo.comoving_distance ~p (Nx.scalar f64 1089.0)))
  in
  is_true
    ~msg:(Printf.sprintf "comoving(z=1089) ~ 14000 Mpc, got %.0f" dc)
    (Float.abs (dc -. 14000.0) < 500.0)

let test_cosmo_comoving_high_z () =
  let p = Cosmo.planck18 in
  let dc_2 =
    v (Unit.Length.in_mpc (Cosmo.comoving_distance ~p (Nx.scalar f64 2.0)))
  in
  let dc_5 =
    v (Unit.Length.in_mpc (Cosmo.comoving_distance ~p (Nx.scalar f64 5.0)))
  in
  let dc_10 =
    v (Unit.Length.in_mpc (Cosmo.comoving_distance ~p (Nx.scalar f64 10.0)))
  in
  is_true ~msg:"comoving distances monotonically increase"
    (dc_2 < dc_5 && dc_5 < dc_10);
  is_true
    ~msg:(Printf.sprintf "comoving(z=10) ~ 9700 Mpc, got %.0f" dc_10)
    (Float.abs (dc_10 -. 9700.0) < 300.0)

let test_cosmo_lookback_high_z () =
  let p = Cosmo.planck18 in
  let lb_5 =
    v (Unit.Time.in_gyr (Cosmo.lookback_time ~p (Nx.scalar f64 5.0)))
  in
  is_true
    ~msg:(Printf.sprintf "lookback(z=5) ~ 12.5 Gyr, got %.1f" lb_5)
    (Float.abs (lb_5 -. 12.5) < 0.3)

(* FITS tests *)

let test_fits_image_roundtrip () =
  let path = "_test_image.fits" in
  Fun.protect
    ~finally:(fun () -> if Sys.file_exists path then Sys.remove path)
    (fun () ->
      let data =
        Nx.create Nx.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
      in
      Umbra_fits.write_image path data;
      let packed = Umbra_fits.read_image ~hdu:0 path in
      let result = Nx_io.to_typed Nx.float32 packed in
      is_true ~msg:"Image shape" (Nx.shape result = [| 2; 3 |]);
      for i = 0 to 5 do
        let row = i / 3 and col = i mod 3 in
        is_true
          ~msg:(Printf.sprintf "Image value [%d,%d]" row col)
          (Float.abs (Nx.item [ row; col ] data -. Nx.item [ row; col ] result)
          < 1e-6)
      done)

let test_fits_table_roundtrip () =
  let path = "_test_table.fits" in
  Fun.protect
    ~finally:(fun () -> if Sys.file_exists path then Sys.remove path)
    (fun () ->
      let df =
        Talon.create
          [
            ("ra", Talon.Col.float64 [| 10.0; 20.0; 30.0 |]);
            ("dec", Talon.Col.float64 [| -10.0; 0.0; 10.0 |]);
          ]
      in
      Umbra_fits.write_table path df;
      let df2 = Umbra_fits.read_table ~hdu:1 path in
      is_true ~msg:"Table rows" (Talon.num_rows df2 = 3);
      match Talon.to_array Nx.float64 df2 "ra" with
      | Some arr -> is_true ~msg:"ra[0]" (Float.abs (arr.(0) -. 10.0) < 1e-10)
      | None -> fail "ra column missing")

(* Coord cross-matching tests *)

let test_match_nearest_self () =
  let ra = Unit.Angle.of_deg (Nx.create f64 [| 3 |] [| 10.0; 20.0; 30.0 |]) in
  let dec = Unit.Angle.of_deg (Nx.create f64 [| 3 |] [| -10.0; 0.0; 10.0 |]) in
  let c = Coord.of_radec ~ra ~dec in
  let { Coord.indices; separations } = Coord.nearest c c in
  for i = 0 to 2 do
    is_true
      ~msg:(Printf.sprintf "Self-match index[%d]" i)
      (Int32.to_int (Nx.item [ i ] indices) = i);
    is_true
      ~msg:(Printf.sprintf "Self-match separation[%d]" i)
      (Nx.item [ i ] (Unit.Angle.in_rad separations) < 1e-10)
  done

(* Time tests *)

let test_time_jd_mjd () =
  let t = Time.unsafe_of_jd 2451545.0 in
  is_true ~msg:"J2000.0 JD" (Float.abs (Time.to_jd t -. 2451545.0) < 1e-10);
  is_true ~msg:"J2000.0 MJD" (Float.abs (Time.to_mjd t -. 51544.5) < 1e-10);
  let t2 = Time.unsafe_of_mjd 51544.5 in
  is_true ~msg:"MJD roundtrip" (Float.abs (Time.to_jd t2 -. 2451545.0) < 1e-10)

let test_time_iso () =
  let t = Time.of_iso "2000-01-01T12:00:00" in
  is_true ~msg:"J2000.0 from ISO" (Float.abs (Time.to_jd t -. 2451545.0) < 1e-6);
  let s = Time.to_iso t in
  is_true ~msg:"ISO roundtrip" (s = "2000-01-01T12:00:00Z")

let test_time_utc_tai_tt () =
  let utc = Time.unsafe_of_jd 2451545.0 in
  let tai = Time.utc_to_tai utc in
  let dt_s = (Time.to_jd tai -. Time.to_jd utc) *. 86400.0 in
  is_true
    ~msg:(Printf.sprintf "TAI-UTC at J2000 = 32s, got %.1f" dt_s)
    (Float.abs (dt_s -. 32.0) < 0.1);
  let tt = Time.tai_to_tt tai in
  let dt_tt = (Time.to_jd tt -. Time.to_jd tai) *. 86400.0 in
  is_true
    ~msg:(Printf.sprintf "TT-TAI = 32.184s, got %.6f" dt_tt)
    (Float.abs (dt_tt -. 32.184) < 1e-3);
  let tai' = Time.tt_to_tai tt in
  is_true ~msg:"TT->TAI roundtrip"
    (Float.abs (Time.to_jd tai' -. Time.to_jd tai) < 1e-12);
  let utc' = Time.tai_to_utc tai in
  is_true ~msg:"TAI->UTC roundtrip"
    (Float.abs (Time.to_jd utc' -. Time.to_jd utc) < 1e-10)

let test_time_tdb () =
  let tt = Time.unsafe_of_jd 2451545.0 in
  let tdb = Time.tt_to_tdb tt in
  let dt_ms = (Time.to_jd tdb -. Time.to_jd tt) *. 86400.0 *. 1000.0 in
  is_true
    ~msg:(Printf.sprintf "TDB-TT < 2ms, got %.3f ms" dt_ms)
    (Float.abs dt_ms < 2.0);
  let tt' = Time.tdb_to_tt tdb in
  is_true ~msg:"TDB->TT roundtrip"
    (Float.abs (Time.to_jd tt' -. Time.to_jd tt) < 1e-10)

let test_time_unix () =
  let t = Time.of_unix 0.0 in
  is_true ~msg:"Unix epoch JD" (Float.abs (Time.to_jd t -. 2440587.5) < 1e-10);
  let u = Time.to_unix t in
  is_true ~msg:"Unix roundtrip" (Float.abs u < 1e-6)

let test_time_diff_add () =
  let t1 = Time.unsafe_of_jd 2451545.0 in
  let t2 = Time.unsafe_of_jd 2451546.0 in
  let dt = Time.diff t2 t1 in
  is_true ~msg:"diff = 1 day"
    (Float.abs (v (Unit.Time.in_day dt) -. 1.0) < 1e-10);
  let t3 = Time.add t1 (Unit.Time.day 1.0) in
  is_true ~msg:"add 1 day" (Float.abs (Time.to_jd t3 -. 2451546.0) < 1e-10)

(* Cosmo preset tests *)

let test_cosmo_planck18 () =
  let z = Nx.scalar f64 0.5 in
  let dc =
    v (Unit.Length.in_mpc (Cosmo.comoving_distance ~p:Cosmo.planck18 z))
  in
  is_true
    ~msg:(Printf.sprintf "Planck18 comoving(0.5) ~ 1960 Mpc, got %.0f" dc)
    (Float.abs (dc -. 1960.0) < 30.0)

let test_cosmo_hubble () =
  let z = Nx.scalar f64 0.0 in
  let h0 = Nx.item [] (Cosmo.hubble z) in
  is_true
    ~msg:(Printf.sprintf "H(0) = H0 = 70, got %.1f" h0)
    (Float.abs (h0 -. 70.0) < 1e-6)

(* Coord FK5/Supergalactic tests *)

let test_coord_ecliptic_roundtrip () =
  let ra = Unit.Angle.of_deg (Nx.create f64 [| 2 |] [| 180.0; 45.0 |]) in
  let dec = Unit.Angle.of_deg (Nx.create f64 [| 2 |] [| 45.0; -30.0 |]) in
  let c = Coord.of_radec ~ra ~dec in
  let ecl = Coord.ecliptic_j2000 c in
  let back = Coord.icrs ecl in
  let ra' = Unit.Angle.in_deg (Coord.ra back) in
  let dec' = Unit.Angle.in_deg (Coord.dec back) in
  let ra_orig = Unit.Angle.in_deg ra in
  let dec_orig = Unit.Angle.in_deg dec in
  for i = 0 to 1 do
    is_true
      ~msg:(Printf.sprintf "Ecliptic RA roundtrip [%d]" i)
      (Float.abs (Nx.item [ i ] ra_orig -. Nx.item [ i ] ra') < deg_eps);
    is_true
      ~msg:(Printf.sprintf "Ecliptic Dec roundtrip [%d]" i)
      (Float.abs (Nx.item [ i ] dec_orig -. Nx.item [ i ] dec') < deg_eps)
  done

let test_coord_supergalactic_roundtrip () =
  let ra = Unit.Angle.of_deg (Nx.create f64 [| 2 |] [| 180.0; 45.0 |]) in
  let dec = Unit.Angle.of_deg (Nx.create f64 [| 2 |] [| 45.0; -30.0 |]) in
  let c = Coord.of_radec ~ra ~dec in
  let sg = Coord.supergalactic c in
  let back = Coord.icrs sg in
  let ra' = Unit.Angle.in_deg (Coord.ra back) in
  let dec' = Unit.Angle.in_deg (Coord.dec back) in
  let ra_orig = Unit.Angle.in_deg ra in
  let dec_orig = Unit.Angle.in_deg dec in
  for i = 0 to 1 do
    is_true
      ~msg:(Printf.sprintf "Supergalactic RA roundtrip [%d]" i)
      (Float.abs (Nx.item [ i ] ra_orig -. Nx.item [ i ] ra') < 1e-4);
    is_true
      ~msg:(Printf.sprintf "Supergalactic Dec roundtrip [%d]" i)
      (Float.abs (Nx.item [ i ] dec_orig -. Nx.item [ i ] dec') < 1e-4)
  done

(* Unit energy-wavelength-frequency tests *)

let test_energy_wavelength_frequency () =
  let e = Unit.Energy.ev 2.0 in
  let nu = Unit.energy_to_frequency e in
  let e2 = Unit.frequency_to_energy nu in
  is_true ~msg:"energy->freq->energy roundtrip"
    (Float.abs (v (Unit.Energy.in_ev e2) -. 2.0) < 1e-6);
  let lam = Unit.energy_to_wavelength e in
  let nu2 = Unit.wavelength_to_frequency lam in
  let e3 = Unit.frequency_to_energy nu2 in
  is_true ~msg:"energy->wavelength->freq->energy roundtrip"
    (Float.abs (v (Unit.Energy.in_ev e3) -. 2.0) < 1e-6)

(* Spectrum tests *)

let test_spectrum_blackbody_wien () =
  (* Wien's displacement law: λ_max * T = 2.898e-3 m·K *)
  let temp = Unit.Temperature.of_kelvin (Nx.scalar f64 5778.0) in
  let wave = Unit.Length.of_m (Nx.linspace f64 1e-7 3e-6 1000) in
  let spec = Spectrum.blackbody ~temperature:temp ~wavelength:wave in
  let vals = Spectrum.values spec in
  (* Find index of max value *)
  let peak_idx = ref 0 in
  let peak_val = ref (Nx.item [ 0 ] vals) in
  for i = 1 to 999 do
    let v = Nx.item [ i ] vals in
    if v > !peak_val then begin
      peak_val := v;
      peak_idx := i
    end
  done;
  let wave_m = Unit.Length.in_m (Spectrum.wavelength spec) in
  let peak_lam = Nx.item [ !peak_idx ] wave_m in
  let wien = peak_lam *. 5778.0 in
  is_true
    ~msg:(Printf.sprintf "Wien's law: λ_max*T ~ 2.898e-3, got %.4e" wien)
    (Float.abs (wien -. 2.898e-3) /. 2.898e-3 < 0.01)

let test_spectrum_redshift () =
  let wave = Unit.Length.of_m (Nx.linspace f64 1e-7 1e-6 100) in
  let values = Nx.ones f64 [| 100 |] in
  let spec =
    Spectrum.create ~wavelength:wave ~values |> Spectrum.as_flux_density
  in
  let z = Nx.scalar f64 1.0 in
  let shifted = Spectrum.redshift ~z spec in
  (* Wavelengths should double at z=1 *)
  let orig_wave = Unit.Length.in_m (Spectrum.wavelength spec) in
  let shifted_wave = Unit.Length.in_m (Spectrum.wavelength shifted) in
  let ratio = Nx.item [ 50 ] shifted_wave /. Nx.item [ 50 ] orig_wave in
  is_true ~msg:"Redshift z=1 doubles wavelength"
    (Float.abs (ratio -. 2.0) < 1e-10);
  (* Values should halve at z=1 *)
  let val_ratio =
    Nx.item [ 50 ] (Spectrum.values shifted) /. Nx.item [ 50 ] values
  in
  is_true ~msg:"Redshift z=1 halves values"
    (Float.abs (val_ratio -. 0.5) < 1e-10)

let test_spectrum_scale () =
  let wave = Unit.Length.of_m (Nx.linspace f64 1e-7 1e-6 10) in
  let values = Nx.ones f64 [| 10 |] in
  let spec = Spectrum.create ~wavelength:wave ~values in
  let scaled = Spectrum.scale (Nx.scalar f64 3.0) spec in
  is_true ~msg:"Scale by 3"
    (Float.abs (Nx.item [ 0 ] (Spectrum.values scaled) -. 3.0) < 1e-10)

(* Extinction tests *)

let test_extinction_ccm89_v_band () =
  (* At V-band (550nm), A_λ/A_V should be ~1.0 for R_V=3.1 *)
  let rv = Nx.scalar f64 3.1 in
  let wave_v = Unit.Length.of_m (Nx.create f64 [| 1 |] [| 5.5e-7 |]) in
  let alav = Extinction.curve (Extinction.ccm89 ~rv) ~wavelength:wave_v in
  let val_v = Nx.item [ 0 ] alav in
  is_true
    ~msg:(Printf.sprintf "CCM89 A_V/A_V ~ 1.0 at 550nm, got %.3f" val_v)
    (Float.abs (val_v -. 1.0) < 0.1)

let test_extinction_apply_unredden () =
  (* apply then unredden should recover original spectrum *)
  let rv = Nx.scalar f64 3.1 in
  let law = Extinction.ccm89 ~rv in
  let wave = Unit.Length.of_m (Nx.linspace f64 3e-7 1e-6 50) in
  let temp = Unit.Temperature.of_kelvin (Nx.scalar f64 6000.0) in
  let spec = Spectrum.blackbody ~temperature:temp ~wavelength:wave in
  let av = Nx.scalar f64 1.0 in
  let reddened = Extinction.apply law ~av spec in
  let recovered = Extinction.unredden law ~av reddened in
  (* Compare values *)
  let orig_val = Nx.item [ 25 ] (Spectrum.values spec) in
  let rec_val = Nx.item [ 25 ] (Spectrum.values recovered) in
  is_true ~msg:"apply + unredden roundtrip"
    (Float.abs (rec_val -. orig_val) /. orig_val < 1e-10)

let test_extinction_ccm89_monotonic () =
  (* Extinction should increase toward blue wavelengths (for optical) *)
  let rv = Nx.scalar f64 3.1 in
  let wave =
    Unit.Length.of_m (Nx.create f64 [| 3 |] [| 4e-7; 5.5e-7; 8e-7 |])
  in
  let alav = Extinction.curve (Extinction.ccm89 ~rv) ~wavelength:wave in
  let blue = Nx.item [ 0 ] alav in
  let green = Nx.item [ 1 ] alav in
  let red = Nx.item [ 2 ] alav in
  is_true ~msg:"CCM89: A_blue > A_green" (blue > green);
  is_true ~msg:"CCM89: A_green > A_red" (green > red)

(* Photometry tests *)

let test_photometry_ab_mag_flat () =
  (* A flat f_nu spectrum at 3631 Jy should give m_AB = 0 in any band. f_nu =
     3631e-26 W/m²/Hz, so f_lambda = f_nu * c / lambda² *)
  let n = 100 in
  let bp =
    Photometry.tophat ~lo:(Unit.Length.m 4e-7) ~hi:(Unit.Length.m 7e-7) ~n
  in
  let wave_m = Unit.Length.to_tensor (Photometry.wavelength bp) in
  let c = 299_792_458.0 in
  let ab_zp = 3631.0e-26 in
  (* f_lambda = f_nu * c / lambda^2 *)
  let f_lambda =
    Nx.div
      (Nx.mul_s (Nx.recip (Nx.square wave_m)) (ab_zp *. c))
      (Nx.scalar f64 1.0)
  in
  let spec =
    Spectrum.create ~wavelength:(Photometry.wavelength bp) ~values:f_lambda
    |> Spectrum.as_flux_density
  in
  let mag = Nx.item [] (Photometry.ab_mag bp spec) in
  is_true
    ~msg:(Printf.sprintf "Flat f_nu=3631Jy gives m_AB ~ 0, got %.3f" mag)
    (Float.abs mag < 0.05)

let test_photometry_color_same_band () =
  (* Color between same band should be 0 *)
  let bp =
    Photometry.tophat ~lo:(Unit.Length.m 4e-7) ~hi:(Unit.Length.m 5.5e-7) ~n:50
  in
  let spec =
    Spectrum.blackbody
      ~temperature:(Unit.Temperature.of_kelvin (Nx.scalar f64 5000.0))
      ~wavelength:(Photometry.wavelength bp)
    |> Spectrum.as_flux_density
  in
  let col = Nx.item [] (Photometry.color bp bp spec) in
  is_true ~msg:"Same-band color = 0" (Float.abs col < 1e-10)

let test_photometry_blue_star_color () =
  (* A hot star should be brighter (lower mag) in blue than red *)
  let n = 100 in
  let bp_b =
    Photometry.tophat ~lo:(Unit.Length.m 4e-7) ~hi:(Unit.Length.m 5e-7) ~n
  in
  let bp_r =
    Photometry.tophat ~lo:(Unit.Length.m 6e-7) ~hi:(Unit.Length.m 7e-7) ~n
  in
  let temp = Unit.Temperature.of_kelvin (Nx.scalar f64 20000.0) in
  let spec_b =
    Spectrum.blackbody ~temperature:temp
      ~wavelength:(Photometry.wavelength bp_b)
    |> Spectrum.as_flux_density
  in
  let spec_r =
    Spectrum.blackbody ~temperature:temp
      ~wavelength:(Photometry.wavelength bp_r)
    |> Spectrum.as_flux_density
  in
  let mag_b = Nx.item [] (Photometry.ab_mag bp_b spec_b) in
  let mag_r = Nx.item [] (Photometry.ab_mag bp_r spec_r) in
  is_true ~msg:"Hot star: blue mag < red mag (brighter in blue)" (mag_b < mag_r)

(* Cosmo: extended models *)

let test_cosmo_flat_lcdm_same_as_default () =
  let p = Cosmo.flat_lcdm ~h0:70.0 ~omega_m:0.3 in
  let z = Nx.scalar f64 0.5 in
  let dc_default = v (Unit.Length.in_mpc (Cosmo.comoving_distance z)) in
  let dc_flat = v (Unit.Length.in_mpc (Cosmo.comoving_distance ~p z)) in
  is_true ~msg:"flat_lcdm(70,0.3) = default"
    (Float.abs (dc_default -. dc_flat) < 1e-6)

let test_cosmo_nonflat_lcdm () =
  (* Open universe: omega_m=0.3, omega_l=0.5 → omega_k=0.2. Result should differ
     from flat LCDM. *)
  let p_flat = Cosmo.flat_lcdm ~h0:70.0 ~omega_m:0.3 in
  let p_open = Cosmo.lcdm ~h0:70.0 ~omega_m:0.3 ~omega_l:0.5 in
  let z = Nx.scalar f64 1.0 in
  let dl_flat =
    v (Unit.Length.in_mpc (Cosmo.luminosity_distance ~p:p_flat z))
  in
  let dl_open =
    v (Unit.Length.in_mpc (Cosmo.luminosity_distance ~p:p_open z))
  in
  is_true
    ~msg:
      (Printf.sprintf "Non-flat LCDM differs from flat: %.0f vs %.0f" dl_open
         dl_flat)
    (Float.abs (dl_open -. dl_flat) > 10.0)

let test_cosmo_wcdm () =
  (* w0 = -1 should be identical to ΛCDM *)
  let p_lcdm = Cosmo.flat_lcdm ~h0:70.0 ~omega_m:0.3 in
  let p_wcdm = Cosmo.wcdm ~h0:70.0 ~omega_m:0.3 ~w0:(-1.0) () in
  let z = Nx.scalar f64 0.5 in
  let dc_lcdm = v (Unit.Length.in_mpc (Cosmo.comoving_distance ~p:p_lcdm z)) in
  let dc_wcdm = v (Unit.Length.in_mpc (Cosmo.comoving_distance ~p:p_wcdm z)) in
  is_true
    ~msg:(Printf.sprintf "wCDM(w0=-1) = LCDM: %.1f vs %.1f" dc_wcdm dc_lcdm)
    (Float.abs (dc_wcdm -. dc_lcdm) < 1.0)

let test_cosmo_w0wacdm () =
  (* w0=-1, wa=0 should reduce to ΛCDM *)
  let p_lcdm = Cosmo.flat_lcdm ~h0:70.0 ~omega_m:0.3 in
  let p_cpl = Cosmo.w0wacdm ~h0:70.0 ~omega_m:0.3 ~w0:(-1.0) ~wa:0.0 () in
  let z = Nx.scalar f64 1.0 in
  let dl_lcdm =
    v (Unit.Length.in_mpc (Cosmo.luminosity_distance ~p:p_lcdm z))
  in
  let dl_cpl = v (Unit.Length.in_mpc (Cosmo.luminosity_distance ~p:p_cpl z)) in
  is_true
    ~msg:(Printf.sprintf "w0waCDM(-1,0) = LCDM: %.1f vs %.1f" dl_cpl dl_lcdm)
    (Float.abs (dl_cpl -. dl_lcdm) < 1.0)

let test_cosmo_e_of () =
  (* E(z=0) = 1 for any cosmology *)
  let p = Cosmo.planck18 in
  let z = Nx.scalar f64 0.0 in
  let e0 = v (Cosmo.e_of p z) in
  is_true
    ~msg:(Printf.sprintf "E(z=0) = 1, got %.6f" e0)
    (Float.abs (e0 -. 1.0) < 1e-6)

let test_cosmo_z_at_value () =
  (* Roundtrip: compute dl at z=0.5, then find z back *)
  let p = Cosmo.default in
  let z0 = 0.5 in
  let dl = Cosmo.luminosity_distance ~p (Nx.scalar f64 z0) in
  let z_found =
    v
      (Cosmo.z_at_value ~p
         (fun ~p z -> Unit.Length.to_tensor (Cosmo.luminosity_distance ~p z))
         (Unit.Length.to_tensor dl))
  in
  is_true
    ~msg:(Printf.sprintf "z_at_value roundtrip: expected 0.5, got %.6f" z_found)
    (Float.abs (z_found -. z0) < 1e-6)

(* AltAz tests *)

let test_altaz_zenith () =
  (* A star at the observer's zenith should have alt ~ 90° *)
  let obs =
    Altaz.make_observer ~lat:(Unit.Angle.deg 0.0) ~lon:(Unit.Angle.deg 0.0) ()
  in
  (* Use the vernal equinox time: RA=0, Dec=0 should be near zenith at sidereal
     midnight from lon=0, lat=0. At J2000.0 the ERA is ~280.46°, so RA ~ 280.46°
     should be near transit. Instead, test roundtrip. *)
  let t = Time.of_iso "2024-01-01T00:00:00" in
  let ra = Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| 180.0 |]) in
  let dec = Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| 45.0 |]) in
  let c = Coord.of_radec ~ra ~dec in
  let hz = Altaz.of_coord ~obstime:t ~observer:obs c in
  let back = Altaz.to_coord ~obstime:t ~observer:obs hz in
  let ra' = Nx.item [ 0 ] (Unit.Angle.in_deg (Coord.ra back)) in
  let dec' = Nx.item [ 0 ] (Unit.Angle.in_deg (Coord.dec back)) in
  is_true
    ~msg:(Printf.sprintf "AltAz RA roundtrip: 180 vs %.4f" ra')
    (Float.abs (ra' -. 180.0) < 0.1);
  is_true
    ~msg:(Printf.sprintf "AltAz Dec roundtrip: 45 vs %.4f" dec')
    (Float.abs (dec' -. 45.0) < 0.1)

let test_altaz_north_pole () =
  (* Polaris (dec ~ 90) should always be near alt = observer lat *)
  let obs =
    Altaz.make_observer ~lat:(Unit.Angle.deg 45.0) ~lon:(Unit.Angle.deg 0.0) ()
  in
  let t = Time.of_iso "2024-06-15T12:00:00" in
  let ra = Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| 37.95 |]) in
  let dec = Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| 89.264 |]) in
  let c = Coord.of_radec ~ra ~dec in
  let hz = Altaz.of_coord ~obstime:t ~observer:obs c in
  let alt_deg = Nx.item [ 0 ] (Unit.Angle.in_deg (Altaz.alt hz)) in
  is_true
    ~msg:(Printf.sprintf "Polaris alt ~ 45° from lat=45°, got %.1f" alt_deg)
    (Float.abs (alt_deg -. 45.0) < 2.0)

(* Galactocentric tests *)

let test_galactocentric_gc_position () =
  (* A point at l=0, b=0, d=galcen_distance should map to near (0, 0, z_sun) in
     Galactocentric. *)
  let c =
    Coord.of_galactic
      ~l:(Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| 0.0 |]))
      ~b:(Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| 0.0 |]))
  in
  let gc =
    Galactocentric.of_coord
      ~distance:(Unit.Length.of_kpc (Nx.create f64 [| 1 |] [| 8.122 |]))
      c
  in
  let xv = Nx.item [ 0 ] (Unit.Length.in_kpc (Galactocentric.x gc)) in
  let yv = Nx.item [ 0 ] (Unit.Length.in_kpc (Galactocentric.y gc)) in
  let zv = Nx.item [ 0 ] (Unit.Length.in_kpc (Galactocentric.z gc)) in
  is_true
    ~msg:(Printf.sprintf "GC x ~ 0 kpc, got %.6f" xv)
    (Float.abs xv < 1e-10);
  is_true
    ~msg:(Printf.sprintf "GC y ~ 0 kpc, got %.6f" yv)
    (Float.abs yv < 1e-10);
  is_true
    ~msg:(Printf.sprintf "GC z ~ z_sun=0.0208 kpc, got %.4f" zv)
    (Float.abs (zv -. 0.0208) < 1e-10)

let test_galactocentric_roundtrip () =
  let ra = Unit.Angle.of_deg (Nx.create f64 [| 2 |] [| 180.0; 45.0 |]) in
  let dec = Unit.Angle.of_deg (Nx.create f64 [| 2 |] [| 30.0; -15.0 |]) in
  let c = Coord.of_radec ~ra ~dec in
  let d = Unit.Length.of_kpc (Nx.create f64 [| 2 |] [| 5.0; 12.0 |]) in
  let gc = Galactocentric.of_coord ~distance:d c in
  let c', d' = Galactocentric.to_coord gc in
  let ra' = Unit.Angle.in_deg (Coord.ra c') in
  let dec' = Unit.Angle.in_deg (Coord.dec c') in
  let d_kpc' = Unit.Length.in_kpc d' in
  let ra_orig = Unit.Angle.in_deg ra in
  let dec_orig = Unit.Angle.in_deg dec in
  let d_orig = Unit.Length.in_kpc d in
  for i = 0 to 1 do
    is_true
      ~msg:(Printf.sprintf "Galactocentric RA roundtrip [%d]" i)
      (Float.abs (Nx.item [ i ] ra' -. Nx.item [ i ] ra_orig) < 0.01);
    is_true
      ~msg:(Printf.sprintf "Galactocentric Dec roundtrip [%d]" i)
      (Float.abs (Nx.item [ i ] dec' -. Nx.item [ i ] dec_orig) < 0.01);
    is_true
      ~msg:(Printf.sprintf "Galactocentric distance roundtrip [%d]" i)
      (Float.abs (Nx.item [ i ] d_kpc' -. Nx.item [ i ] d_orig) < 0.01)
  done

(* Cosmo: growth and power spectrum *)

let test_cosmo_growth_factor_z0 () =
  let g = v (Cosmo.growth_factor ~p:Cosmo.planck18 (Nx.scalar f64 0.0)) in
  is_true
    ~msg:(Printf.sprintf "D(z=0) = 1.0, got %.6f" g)
    (Float.abs (g -. 1.0) < 1e-4)

let test_cosmo_growth_factor_z1 () =
  let g = v (Cosmo.growth_factor ~p:Cosmo.planck18 (Nx.scalar f64 1.0)) in
  is_true
    ~msg:(Printf.sprintf "D(z=1) ~ 0.61, got %.4f" g)
    (Float.abs (g -. 0.61) < 0.02)

let test_cosmo_growth_rate_z0 () =
  let f = v (Cosmo.growth_rate ~p:Cosmo.planck18 (Nx.scalar f64 0.0)) in
  (* f(z=0) ~ Ω_m^0.55 ~ 0.524 for Planck18 *)
  is_true
    ~msg:(Printf.sprintf "f(z=0) ~ 0.52, got %.4f" f)
    (Float.abs (f -. 0.52) < 0.02)

let test_cosmo_growth_monotonic () =
  let p = Cosmo.planck18 in
  let d0 = v (Cosmo.growth_factor ~p (Nx.scalar f64 0.0)) in
  let d1 = v (Cosmo.growth_factor ~p (Nx.scalar f64 0.5)) in
  let d2 = v (Cosmo.growth_factor ~p (Nx.scalar f64 1.0)) in
  is_true ~msg:"D(0) > D(0.5) > D(1)" (d0 > d1 && d1 > d2)

let test_cosmo_linear_power () =
  let p = Cosmo.planck18 in
  let k = Nx.scalar f64 0.1 in
  let pk = v (Cosmo.linear_power ~p k (Nx.scalar f64 0.0)) in
  is_true ~msg:(Printf.sprintf "P_lin(k=0.1, z=0) > 0, got %.1f" pk) (pk > 0.0);
  (* P(k, z=1) should be less than P(k, z=0) *)
  let pk1 = v (Cosmo.linear_power ~p k (Nx.scalar f64 1.0)) in
  is_true ~msg:"P_lin(z=1) < P_lin(z=0)" (pk1 < pk)

let test_cosmo_nonlinear_power () =
  let p = Cosmo.planck18 in
  let k = Nx.scalar f64 1.0 in
  let pk_nl = v (Cosmo.nonlinear_power ~p k (Nx.scalar f64 0.0)) in
  let pk_lin = v (Cosmo.linear_power ~p k (Nx.scalar f64 0.0)) in
  is_true ~msg:(Printf.sprintf "P_nl(k=1) > 0, got %.1f" pk_nl) (pk_nl > 0.0);
  (* At k=1 h/Mpc, nonlinear should exceed linear *)
  is_true
    ~msg:(Printf.sprintf "P_nl(k=1) > P_lin(k=1): %.1f > %.1f" pk_nl pk_lin)
    (pk_nl > pk_lin)

let test_cosmo_params_accessors () =
  let p = Cosmo.planck18 in
  let ob = v (Cosmo.omega_b p) in
  let ns = v (Cosmo.n_s p) in
  let s8 = v (Cosmo.sigma8 p) in
  is_true ~msg:"Planck18 omega_b = 0.049" (Float.abs (ob -. 0.049) < 1e-6);
  is_true ~msg:"Planck18 n_s = 0.9665" (Float.abs (ns -. 0.9665) < 1e-6);
  is_true ~msg:"Planck18 sigma8 = 0.8102" (Float.abs (s8 -. 0.8102) < 1e-6)

(* Survey tests *)

let test_survey_smail_normalized () =
  let nz = Survey.smail ~a:2.0 ~b:1.5 ~z0:0.3 () in
  let n = 1000 in
  let zmax = Survey.nz_zmax nz in
  let dz = zmax /. Float.of_int n in
  let sum = ref 0.0 in
  for i = 0 to n do
    let z = Float.of_int i *. dz in
    let nz_val = v (Survey.eval_nz nz (Nx.scalar f64 z)) in
    let w = if i = 0 || i = n then 0.5 else 1.0 in
    sum := !sum +. (w *. nz_val *. dz)
  done;
  is_true
    ~msg:(Printf.sprintf "smail integrates to 1.0, got %.6f" !sum)
    (Float.abs (!sum -. 1.0) < 1e-3)

let test_survey_tabulated () =
  let z = Nx.create f64 [| 5 |] [| 0.0; 0.25; 0.5; 0.75; 1.0 |] in
  let pz = Nx.create f64 [| 5 |] [| 0.0; 1.0; 2.0; 1.0; 0.0 |] in
  let nz = Survey.tabulated ~z ~pz () in
  let mid = v (Survey.eval_nz nz (Nx.scalar f64 0.5)) in
  is_true ~msg:"tabulated mid > 0" (mid > 0.0);
  let out = v (Survey.eval_nz nz (Nx.scalar f64 1.5)) in
  is_true ~msg:"tabulated outside = 0" (Float.abs out < eps)

let test_survey_cl_shape () =
  let p = Cosmo.planck18 in
  let nz1 = Survey.smail ~a:2.0 ~b:1.5 ~z0:0.3 () in
  let nz2 = Survey.smail ~a:2.0 ~b:1.5 ~z0:0.7 () in
  let wl1 = Survey.weak_lensing ~n_gal:26.0 nz1 in
  let wl2 = Survey.weak_lensing ~n_gal:26.0 nz2 in
  let ell = Nx.create f64 [| 3 |] [| 100.0; 300.0; 1000.0 |] in
  let cls = Survey.angular_cl ~p ~power:Survey.linear ~ell [ wl1; wl2 ] in
  let shape = Nx.shape (Survey.Cls.to_tensor cls) in
  is_true
    ~msg:(Printf.sprintf "C_l shape = [3; 3], got [%d; %d]" shape.(0) shape.(1))
    (shape.(0) = 3 && shape.(1) = 3);
  is_true
    ~msg:(Printf.sprintf "n_tracers = 2, got %d" (Survey.Cls.n_tracers cls))
    (Survey.Cls.n_tracers cls = 2)

let test_survey_cl_positive () =
  let p = Cosmo.planck18 in
  let nz1 = Survey.smail ~a:2.0 ~b:1.5 ~z0:0.5 () in
  let wl = Survey.weak_lensing ~n_gal:26.0 nz1 in
  let ell = Nx.create f64 [| 3 |] [| 100.0; 500.0; 1000.0 |] in
  let cls = Survey.angular_cl ~p ~power:Survey.linear ~ell [ wl ] in
  let cl_auto = Survey.Cls.get cls ~i:0 ~j:0 in
  for l = 0 to 2 do
    let cl_val = Nx.item [ l ] cl_auto in
    is_true ~msg:(Printf.sprintf "C_l[%d] = %.2e > 0" l cl_val) (cl_val > 0.0)
  done

let test_survey_noise_wl () =
  let sigma_e = 0.26 in
  let n_gal = 30.0 in
  let nz1 = Survey.smail ~a:2.0 ~b:1.5 ~z0:0.3 () in
  let wl = Survey.weak_lensing ~sigma_e ~n_gal nz1 in
  let ell = Nx.create f64 [| 3 |] [| 100.0; 500.0; 1000.0 |] in
  let cls = Survey.angular_cl ~ell [ wl ] in
  let nl = Survey.Cls.noise cls in
  let n0 = Nx.item [ 0; 0 ] nl in
  let n1 = Nx.item [ 0; 1 ] nl in
  let n2 = Nx.item [ 0; 2 ] nl in
  is_true ~msg:"WL noise > 0" (n0 > 0.0);
  is_true
    ~msg:(Printf.sprintf "WL noise constant in ℓ: %.2e vs %.2e" n0 n1)
    (Float.abs (n0 -. n1) < 1e-20);
  is_true ~msg:"WL noise constant in ℓ (2)" (Float.abs (n1 -. n2) < 1e-20)

(* Spectrum: mul/div *)

let test_spectrum_mul () =
  let wave = Unit.Length.of_m (Nx.linspace f64 3e-7 1e-6 10) in
  let values =
    Nx.create f64 [| 10 |]
      [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0; 10.0 |]
  in
  let a =
    Spectrum.create ~wavelength:wave ~values |> Spectrum.as_flux_density
  in
  let trans =
    Nx.create f64 [| 10 |]
      [| 0.5; 0.5; 0.5; 0.5; 0.5; 0.5; 0.5; 0.5; 0.5; 0.5 |]
  in
  let b = Spectrum.create ~wavelength:wave ~values:trans in
  let result = Spectrum.mul a b in
  is_true ~msg:"mul: 2.0 * 0.5 = 1.0"
    (Float.abs (Nx.item [ 1 ] (Spectrum.values result) -. 1.0) < eps);
  is_true ~msg:"mul: 10.0 * 0.5 = 5.0"
    (Float.abs (Nx.item [ 9 ] (Spectrum.values result) -. 5.0) < eps)

let test_spectrum_div () =
  let wave = Unit.Length.of_m (Nx.linspace f64 3e-7 1e-6 10) in
  let values =
    Nx.create f64 [| 10 |]
      [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0; 10.0 |]
  in
  let a =
    Spectrum.create ~wavelength:wave ~values |> Spectrum.as_flux_density
  in
  let flat =
    Nx.create f64 [| 10 |]
      [| 2.0; 2.0; 2.0; 2.0; 2.0; 2.0; 2.0; 2.0; 2.0; 2.0 |]
  in
  let b = Spectrum.create ~wavelength:wave ~values:flat in
  let result = Spectrum.div a b in
  is_true ~msg:"div: 4.0 / 2.0 = 2.0"
    (Float.abs (Nx.item [ 3 ] (Spectrum.values result) -. 2.0) < eps);
  is_true ~msg:"div: 10.0 / 2.0 = 5.0"
    (Float.abs (Nx.item [ 9 ] (Spectrum.values result) -. 5.0) < eps)

let test_spectrum_mul_div_roundtrip () =
  let wave = Unit.Length.of_m (Nx.linspace f64 3e-7 1e-6 50) in
  let temp = Unit.Temperature.of_kelvin (Nx.scalar f64 6000.0) in
  let spec =
    Spectrum.blackbody ~temperature:temp ~wavelength:wave
    |> Spectrum.as_flux_density
  in
  let trans_vals =
    Nx.create f64 [| 50 |]
      (Array.init 50 (fun i ->
           0.5 +. (0.3 *. Float.sin (Float.of_int i *. 0.2))))
  in
  let trans = Spectrum.create ~wavelength:wave ~values:trans_vals in
  let mulled = Spectrum.mul spec trans in
  let recovered = Spectrum.div mulled trans in
  let orig_val = Nx.item [ 25 ] (Spectrum.values spec) in
  let rec_val = Nx.item [ 25 ] (Spectrum.values recovered) in
  is_true ~msg:"mul then div roundtrip"
    (Float.abs (rec_val -. orig_val) /. orig_val < 1e-10)

(* Spectrum: line profiles *)

let test_spectrum_gaussian_peak () =
  let wave = Unit.Length.of_m (Nx.linspace f64 6.4e-7 6.7e-7 1000) in
  let center = Unit.Length.nm 656.3 in
  let stddev = Unit.Length.nm 1.0 in
  let amplitude = Nx.scalar f64 1.0 in
  let g = Spectrum.gaussian ~amplitude ~center ~stddev ~wavelength:wave in
  let vals = Spectrum.values g in
  let peak_idx = ref 0 in
  let peak_val = ref (Nx.item [ 0 ] vals) in
  for i = 1 to 999 do
    let vi = Nx.item [ i ] vals in
    if vi > !peak_val then begin
      peak_val := vi;
      peak_idx := i
    end
  done;
  let wave_m = Unit.Length.in_m (Spectrum.wavelength g) in
  let peak_lam_nm = Nx.item [ !peak_idx ] wave_m *. 1e9 in
  is_true
    ~msg:(Printf.sprintf "Gaussian peak near 656.3 nm, got %.1f" peak_lam_nm)
    (Float.abs (peak_lam_nm -. 656.3) < 0.5);
  is_true
    ~msg:(Printf.sprintf "Gaussian peak amplitude ~ 1.0, got %.4f" !peak_val)
    (Float.abs (!peak_val -. 1.0) < 0.01)

let test_spectrum_lorentzian_peak () =
  let wave = Unit.Length.of_m (Nx.linspace f64 4.8e-7 5.2e-7 1000) in
  let center = Unit.Length.nm 500.0 in
  let fwhm = Unit.Length.nm 2.0 in
  let amplitude = Nx.scalar f64 3.0 in
  let l = Spectrum.lorentzian ~amplitude ~center ~fwhm ~wavelength:wave in
  let vals = Spectrum.values l in
  let peak_idx = ref 0 in
  let peak_val = ref (Nx.item [ 0 ] vals) in
  for i = 1 to 999 do
    let vi = Nx.item [ i ] vals in
    if vi > !peak_val then begin
      peak_val := vi;
      peak_idx := i
    end
  done;
  let wave_m = Unit.Length.in_m (Spectrum.wavelength l) in
  let peak_lam_nm = Nx.item [ !peak_idx ] wave_m *. 1e9 in
  is_true
    ~msg:(Printf.sprintf "Lorentzian peak near 500 nm, got %.1f" peak_lam_nm)
    (Float.abs (peak_lam_nm -. 500.0) < 0.5);
  is_true
    ~msg:(Printf.sprintf "Lorentzian peak ~ 3.0, got %.4f" !peak_val)
    (Float.abs (!peak_val -. 3.0) < 0.05)

let test_spectrum_voigt_limits () =
  let wave = Unit.Length.of_m (Nx.linspace f64 4.8e-7 5.2e-7 1000) in
  let center = Unit.Length.nm 500.0 in
  let amplitude = Nx.scalar f64 1.0 in
  (* Gaussian limit: sigma >> gamma *)
  let sigma_big = Unit.Length.nm 2.0 in
  let gamma_tiny = Unit.Length.nm 0.001 in
  let voigt_g =
    Spectrum.voigt ~amplitude ~center ~sigma:sigma_big ~gamma:gamma_tiny
      ~wavelength:wave
  in
  let gauss =
    Spectrum.gaussian ~amplitude ~center ~stddev:sigma_big ~wavelength:wave
  in
  let vg_peak = ref 0.0 in
  let g_peak = ref 0.0 in
  for i = 0 to 999 do
    let vv = Nx.item [ i ] (Spectrum.values voigt_g) in
    let gv = Nx.item [ i ] (Spectrum.values gauss) in
    if vv > !vg_peak then vg_peak := vv;
    if gv > !g_peak then g_peak := gv
  done;
  is_true
    ~msg:
      (Printf.sprintf "Voigt(sigma>>gamma) peak ~ Gaussian peak: %.4f vs %.4f"
         !vg_peak !g_peak)
    (Float.abs (!vg_peak -. !g_peak) /. !g_peak < 0.05)

let test_spectrum_line_composability () =
  let wave = Unit.Length.of_m (Nx.linspace f64 6e-7 7e-7 500) in
  let continuum =
    Spectrum.power_law ~amplitude:(Nx.scalar f64 1e-15)
      ~index:(Nx.scalar f64 (-2.0)) ~pivot:(Unit.Length.nm 650.0)
      ~wavelength:wave
  in
  let ha =
    Spectrum.gaussian ~amplitude:(Nx.scalar f64 1e-15)
      ~center:(Unit.Length.nm 656.3) ~stddev:(Unit.Length.nm 0.5)
      ~wavelength:wave
  in
  let composite = Spectrum.add continuum ha in
  let cont_val = Nx.item [ 0 ] (Spectrum.values continuum) in
  let comp_val = Nx.item [ 0 ] (Spectrum.values composite) in
  is_true ~msg:"Composite spectrum at wing ~ continuum"
    (Float.abs (comp_val -. cont_val) /. cont_val < 0.01)

(* Altaz: airmass *)

let test_altaz_airmass_zenith () =
  let hz =
    Altaz.of_coord
      ~obstime:(Time.of_iso "2024-06-21T12:00:00")
      ~observer:
        (Altaz.make_observer ~lat:(Unit.Angle.deg 45.0)
           ~lon:(Unit.Angle.deg 0.0) ())
      (Coord.of_radec
         ~ra:(Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| 0.0 |]))
         ~dec:(Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| 89.0 |])))
  in
  let x = Altaz.airmass hz in
  let x0 = Nx.item [ 0 ] x in
  is_true ~msg:(Printf.sprintf "Airmass >= 1.0, got %.4f" x0) (x0 >= 1.0)

let test_altaz_airmass_low_alt () =
  let obs =
    Altaz.make_observer ~lat:(Unit.Angle.deg 30.0) ~lon:(Unit.Angle.deg 0.0) ()
  in
  let t = Time.of_iso "2024-06-21T22:00:00" in
  let star_a =
    Coord.of_radec
      ~ra:(Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| 0.0 |]))
      ~dec:(Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| 80.0 |]))
  in
  let star_b =
    Coord.of_radec
      ~ra:(Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| 180.0 |]))
      ~dec:(Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| 10.0 |]))
  in
  let hz_a = Altaz.of_coord ~obstime:t ~observer:obs star_a in
  let hz_b = Altaz.of_coord ~obstime:t ~observer:obs star_b in
  let x_a = Nx.item [ 0 ] (Altaz.airmass hz_a) in
  let x_b = Nx.item [ 0 ] (Altaz.airmass hz_b) in
  is_true
    ~msg:(Printf.sprintf "Both airmasses >= 1: %.2f, %.2f" x_a x_b)
    (x_a >= 1.0 && x_b >= 1.0);
  is_true
    ~msg:(Printf.sprintf "Different airmasses: %.2f vs %.2f" x_a x_b)
    (Float.abs (x_a -. x_b) > 0.01)

(* Cosmo: BAO distances *)

let test_cosmo_dh () =
  let p = Cosmo.planck18 in
  let z = Nx.scalar f64 0.0 in
  let dh0 = v (Unit.Length.in_mpc (Cosmo.dh ~p z)) in
  let h0 = Nx.item [] (Cosmo.h0 p) in
  let expected = 299792.458 /. h0 in
  is_true
    ~msg:(Printf.sprintf "D_H(0) = c/H0 ~ %.1f Mpc, got %.1f" expected dh0)
    (Float.abs (dh0 -. expected) /. expected < 1e-4)

let test_cosmo_dm_flat () =
  let p = Cosmo.planck18 in
  let z = Nx.scalar f64 0.5 in
  let dm_val = v (Unit.Length.in_mpc (Cosmo.dm ~p z)) in
  let dc_val = v (Unit.Length.in_mpc (Cosmo.comoving_distance ~p z)) in
  is_true
    ~msg:(Printf.sprintf "D_M = D_C for flat: %.1f vs %.1f" dm_val dc_val)
    (Float.abs (dm_val -. dc_val) /. dc_val < 1e-4)

let test_cosmo_dv () =
  let p = Cosmo.planck18 in
  let z = Nx.scalar f64 0.5 in
  let dv_val = v (Unit.Length.in_mpc (Cosmo.dv ~p z)) in
  is_true ~msg:(Printf.sprintf "D_V(0.5) > 0, got %.1f" dv_val) (dv_val > 0.0);
  let dh_val = v (Unit.Length.in_mpc (Cosmo.dh ~p z)) in
  let dm_val = v (Unit.Length.in_mpc (Cosmo.dm ~p z)) in
  let z_f = 0.5 in
  let expected = (z_f *. dh_val *. dm_val *. dm_val) ** (1.0 /. 3.0) in
  is_true
    ~msg:
      (Printf.sprintf "D_V = (z D_H D_M^2)^{1/3}: %.1f vs %.1f" dv_val expected)
    (Float.abs (dv_val -. expected) /. expected < 1e-3)

let test_cosmo_sound_horizon () =
  let p = Cosmo.planck18 in
  let rs = v (Unit.Length.in_mpc (Cosmo.sound_horizon ~p ())) in
  is_true
    ~msg:(Printf.sprintf "r_s(Planck18) ~ 147 Mpc, got %.1f" rs)
    (Float.abs (rs -. 147.0) < 5.0)

(* Filters *)

let test_filters_sdss_pivot () =
  let bp = Filters.sdss_r in
  let lam_p = v (Unit.Length.in_nm (Photometry.pivot_wavelength bp)) in
  is_true
    ~msg:(Printf.sprintf "SDSS r pivot ~ 620 nm, got %.0f" lam_p)
    (Float.abs (lam_p -. 620.0) < 30.0)

let test_filters_johnson_v_pivot () =
  let bp = Filters.johnson_v in
  let lam_p = v (Unit.Length.in_nm (Photometry.pivot_wavelength bp)) in
  is_true
    ~msg:(Printf.sprintf "Johnson V pivot ~ 551 nm, got %.0f" lam_p)
    (Float.abs (lam_p -. 551.0) < 20.0)

let test_filters_twomass_j_pivot () =
  let bp = Filters.twomass_j in
  let lam_p = v (Unit.Length.in_nm (Photometry.pivot_wavelength bp)) in
  is_true
    ~msg:(Printf.sprintf "2MASS J pivot ~ 1235 nm, got %.0f" lam_p)
    (Float.abs (lam_p -. 1235.0) < 30.0)

let test_filters_gaia_ordering () =
  let bp_p =
    v (Unit.Length.in_nm (Photometry.pivot_wavelength Filters.gaia_bp))
  in
  let g_p =
    v (Unit.Length.in_nm (Photometry.pivot_wavelength Filters.gaia_g))
  in
  let rp_p =
    v (Unit.Length.in_nm (Photometry.pivot_wavelength Filters.gaia_rp))
  in
  is_true
    ~msg:(Printf.sprintf "Gaia: BP < G < RP: %.0f < %.0f < %.0f" bp_p g_p rp_p)
    (bp_p < g_p && g_p < rp_p)

let test_filters_photometry () =
  let bp = Filters.sdss_g in
  let wave = Photometry.wavelength bp in
  let temp = Unit.Temperature.of_kelvin (Nx.scalar f64 5800.0) in
  let sed =
    Spectrum.blackbody ~temperature:temp ~wavelength:wave
    |> Spectrum.as_flux_density
  in
  let mag = Nx.item [] (Photometry.ab_mag bp sed) in
  is_true
    ~msg:(Printf.sprintf "BB(5800K) through SDSS g is finite, got %.2f" mag)
    (Float.is_finite mag)

(* Photometry: auto-resample *)

let test_photometry_auto_resample () =
  let bp = Filters.sdss_g in
  let temp = Unit.Temperature.of_kelvin (Nx.scalar f64 5800.0) in
  let wave_fine = Unit.Length.of_m (Nx.linspace f64 3e-7 1.1e-6 1000) in
  let sed =
    Spectrum.blackbody ~temperature:temp ~wavelength:wave_fine
    |> Spectrum.as_flux_density
  in
  let mag = Nx.item [] (Photometry.ab_mag bp sed) in
  is_true
    ~msg:
      (Printf.sprintf "Auto-resample: BB(5800K) through SDSS g finite, got %.2f"
         mag)
    (Float.is_finite mag);
  let manual = Spectrum.resample ~wavelength:(Photometry.wavelength bp) sed in
  let mag_manual = Nx.item [] (Photometry.ab_mag bp manual) in
  is_true
    ~msg:
      (Printf.sprintf "Auto-resample matches manual: %.4f vs %.4f" mag
         mag_manual)
    (Float.abs (mag -. mag_manual) < 1e-10)

(* Photometry: ST magnitude *)

let test_photometry_st_mag () =
  let bp =
    Photometry.tophat ~lo:(Unit.Length.nm 400.0) ~hi:(Unit.Length.nm 700.0)
      ~n:100
  in
  let wave = Photometry.wavelength bp in
  let temp = Unit.Temperature.of_kelvin (Nx.scalar f64 5800.0) in
  let sed =
    Spectrum.blackbody ~temperature:temp ~wavelength:wave
    |> Spectrum.as_flux_density
  in
  let st = Nx.item [] (Photometry.st_mag bp sed) in
  let ab = Nx.item [] (Photometry.ab_mag bp sed) in
  is_true ~msg:(Printf.sprintf "ST mag is finite: %.2f" st) (Float.is_finite st);
  is_true
    ~msg:(Printf.sprintf "ST and AB differ: ST=%.2f AB=%.2f" st ab)
    (Float.abs (st -. ab) > 0.01)

(* Photometry: Vega magnitude *)

let test_photometry_vega_mag () =
  let bp = Filters.johnson_v in
  let wave = Photometry.wavelength bp in
  let temp = Unit.Temperature.of_kelvin (Nx.scalar f64 9600.0) in
  let sed =
    Spectrum.blackbody ~temperature:temp ~wavelength:wave
    |> Spectrum.as_flux_density
  in
  let vm = Nx.item [] (Photometry.vega_mag bp sed) in
  is_true
    ~msg:(Printf.sprintf "Vega mag of hot BB through V is finite: %.2f" vm)
    (Float.is_finite vm);
  let ab = Nx.item [] (Photometry.ab_mag bp sed) in
  is_true
    ~msg:(Printf.sprintf "Vega and AB differ: V=%.2f AB=%.2f" vm ab)
    (Float.abs (vm -. ab) > 0.001)

(* Photometry: effective wavelength *)

let test_photometry_effective_wavelength () =
  let bp =
    Photometry.tophat ~lo:(Unit.Length.nm 400.0) ~hi:(Unit.Length.nm 700.0)
      ~n:100
  in
  let wave = Photometry.wavelength bp in
  let flat_vals = Nx.ones f64 [| 100 |] in
  let flat =
    Spectrum.create ~wavelength:wave ~values:flat_vals
    |> Spectrum.as_flux_density
  in
  let lam_eff =
    v (Unit.Length.in_nm (Photometry.effective_wavelength bp flat))
  in
  let lam_pivot = v (Unit.Length.in_nm (Photometry.pivot_wavelength bp)) in
  is_true
    ~msg:
      (Printf.sprintf "Flat spectrum: eff_wavelength in range: %.1f nm" lam_eff)
    (lam_eff > 500.0 && lam_eff < 600.0);
  is_true
    ~msg:
      (Printf.sprintf "eff_wavelength >= pivot for flat/tophat: %.1f vs %.1f"
         lam_eff lam_pivot)
    (lam_eff >= lam_pivot)

(* Altaz: atmospheric refraction *)

let test_altaz_refraction () =
  let obs =
    Altaz.make_observer ~lat:(Unit.Angle.deg 45.0) ~lon:(Unit.Angle.deg 0.0) ()
  in
  let t = Time.of_iso "2024-06-15T12:00:00" in
  let ra = Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| 37.95 |]) in
  let dec = Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| 89.264 |]) in
  let c = Coord.of_radec ~ra ~dec in
  let hz_no = Altaz.of_coord ~refraction:false ~obstime:t ~observer:obs c in
  let hz_yes = Altaz.of_coord ~refraction:true ~obstime:t ~observer:obs c in
  let alt_no = Nx.item [ 0 ] (Unit.Angle.in_deg (Altaz.alt hz_no)) in
  let alt_yes = Nx.item [ 0 ] (Unit.Angle.in_deg (Altaz.alt hz_yes)) in
  (* Refraction makes objects appear higher *)
  is_true
    ~msg:
      (Printf.sprintf "Refraction raises altitude: %.4f > %.4f" alt_yes alt_no)
    (alt_yes > alt_no);
  (* At ~45° alt, refraction is ~1 arcmin = 0.017° *)
  let diff = alt_yes -. alt_no in
  is_true
    ~msg:(Printf.sprintf "Refraction at ~45° is small (< 0.1°): %.4f" diff)
    (diff > 0.0 && diff < 0.1)

let test_altaz_refraction_standalone () =
  let obs =
    Altaz.make_observer ~lat:(Unit.Angle.deg 45.0) ~lon:(Unit.Angle.deg 0.0) ()
  in
  let t = Time.of_iso "2024-06-15T12:00:00" in
  let ra = Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| 37.95 |]) in
  let dec = Unit.Angle.of_deg (Nx.create f64 [| 1 |] [| 89.264 |]) in
  let c = Coord.of_radec ~ra ~dec in
  let hz = Altaz.of_coord ~obstime:t ~observer:obs c in
  let r = Altaz.refraction hz in
  let r_arcmin = Nx.item [ 0 ] (Unit.Angle.in_deg r) *. 60.0 in
  is_true
    ~msg:(Printf.sprintf "Refraction > 0 arcmin: %.2f" r_arcmin)
    (r_arcmin > 0.0);
  is_true
    ~msg:(Printf.sprintf "Refraction < 2 arcmin at high alt: %.2f" r_arcmin)
    (r_arcmin < 2.0)

(* Survey: shear multiplicative bias *)

let test_survey_m_bias () =
  let nz = Survey.smail ~a:2.0 ~b:1.5 ~z0:0.5 () in
  let ell = Nx.logspace f64 1.0 3.0 20 in
  let wl_no_bias = Survey.weak_lensing ~n_gal:26.0 nz in
  let wl_with_bias = Survey.weak_lensing ~m_bias:0.02 ~n_gal:26.0 nz in
  let cls_no = Survey.angular_cl ~ell [ wl_no_bias ] in
  let cls_yes = Survey.angular_cl ~ell [ wl_with_bias ] in
  let cl_no = Survey.Cls.get cls_no ~i:0 ~j:0 in
  let cl_yes = Survey.Cls.get cls_yes ~i:0 ~j:0 in
  (* Auto-spectrum scales as (1+m)^2 = 1.0404 *)
  let ratio = Nx.item [ 10 ] (Nx.div cl_yes cl_no) in
  let expected = 1.02 *. 1.02 in
  is_true
    ~msg:
      (Printf.sprintf
         "m_bias=0.02 scales auto-Cl by (1+m)^2: ratio=%.4f vs %.4f" ratio
         expected)
    (Float.abs (ratio -. expected) < 1e-4);
  (* m_bias=0.0 gives same result as no bias *)
  let wl_zero_bias = Survey.weak_lensing ~m_bias:0.0 ~n_gal:26.0 nz in
  let cls_zero = Survey.angular_cl ~ell [ wl_zero_bias ] in
  let cl_zero = Survey.Cls.get cls_zero ~i:0 ~j:0 in
  let diff = Nx.item [] (Nx.max (Nx.abs (Nx.sub cl_zero cl_no))) in
  is_true
    ~msg:(Printf.sprintf "m_bias=0.0 matches no bias: max_diff=%.2e" diff)
    (diff < 1e-30)

(* Spectrum: differentiable resample *)

let test_spectrum_resample_values () =
  let wave = Unit.Length.of_m (Nx.linspace f64 1e-7 1e-5 100) in
  let sed =
    Spectrum.blackbody
      ~temperature:(Unit.Temperature.of_kelvin (Nx.scalar f64 5800.0))
      ~wavelength:wave
  in
  let new_wave = Unit.Length.of_m (Nx.linspace f64 2e-7 9e-6 50) in
  let resampled = Spectrum.resample ~wavelength:new_wave sed in
  let vals = Spectrum.values resampled in
  let n = Nx.numel vals in
  is_true ~msg:(Printf.sprintf "Resampled has %d points" n) (n = 50);
  let v0 = Nx.item [ 0 ] vals in
  is_true ~msg:(Printf.sprintf "Resampled values positive: %.2e" v0) (v0 > 0.0);
  let vmax = Nx.item [] (Nx.max vals) in
  is_true
    ~msg:(Printf.sprintf "Resampled max is finite: %.2e" vmax)
    (Float.is_finite vmax)

(* Survey: baryonic feedback *)

let test_survey_baryonic_feedback () =
  let nz = Survey.smail ~a:2.0 ~b:1.5 ~z0:0.5 () in
  let ell = Nx.logspace f64 1.0 3.0 20 in
  let wl = Survey.weak_lensing ~n_gal:26.0 nz in
  let cls_dm = Survey.angular_cl ~power:Survey.nonlinear ~ell [ wl ] in
  let power_bary = Survey.baryonic_feedback ~a_bary:0.2 Survey.nonlinear in
  let cls_bary = Survey.angular_cl ~power:power_bary ~ell [ wl ] in
  let cl_dm = Survey.Cls.get cls_dm ~i:0 ~j:0 in
  let cl_bary = Survey.Cls.get cls_bary ~i:0 ~j:0 in
  (* Baryonic feedback suppresses small-scale (high-ell) power *)
  let ratio_high = Nx.item [ 19 ] (Nx.div cl_bary cl_dm) in
  is_true
    ~msg:
      (Printf.sprintf "Baryonic suppression at high ell: ratio=%.4f < 1"
         ratio_high)
    (ratio_high < 1.0);
  (* a_bary=0 gives same result as no feedback *)
  let power_zero = Survey.baryonic_feedback ~a_bary:0.0 Survey.nonlinear in
  let cls_zero = Survey.angular_cl ~power:power_zero ~ell [ wl ] in
  let cl_zero = Survey.Cls.get cls_zero ~i:0 ~j:0 in
  let diff = Nx.item [] (Nx.max (Nx.abs (Nx.sub cl_zero cl_dm))) in
  is_true
    ~msg:(Printf.sprintf "a_bary=0 matches DM-only: max_diff=%.2e" diff)
    (diff < 1e-30)

(* Batched spectra *)

let test_batch_create () =
  let wave = Unit.Length.of_m (Nx.linspace f64 1e-7 1e-6 100) in
  let v1 = Nx.ones f64 [| 100 |] in
  let v2 = Nx.full f64 [| 100 |] 2.0 in
  let values = Nx.stack [ v1; v2 ] in
  let s =
    Spectrum.create ~wavelength:wave ~values |> Spectrum.as_flux_density
  in
  let sh = Nx.shape (Spectrum.values s) in
  is_true ~msg:"batch values shape [2; 100]"
    (Array.length sh = 2 && sh.(0) = 2 && sh.(1) = 100)

let test_batch_resample () =
  let wave = Unit.Length.of_m (Nx.linspace f64 1e-7 1e-6 100) in
  let wave2 = Unit.Length.of_m (Nx.linspace f64 2e-7 8e-7 50) in
  let bb1 =
    Spectrum.blackbody
      ~temperature:(Unit.Temperature.of_kelvin (Nx.scalar f64 5000.0))
      ~wavelength:wave
    |> Spectrum.as_sampled
  in
  let bb2 =
    Spectrum.blackbody
      ~temperature:(Unit.Temperature.of_kelvin (Nx.scalar f64 8000.0))
      ~wavelength:wave
    |> Spectrum.as_sampled
  in
  let values = Nx.stack [ Spectrum.values bb1; Spectrum.values bb2 ] in
  let batch = Spectrum.create ~wavelength:wave ~values |> Spectrum.as_sampled in
  let resampled = Spectrum.resample ~wavelength:wave2 batch in
  let r_shape = Nx.shape (Spectrum.values resampled) in
  is_true ~msg:"batch resample shape [2; 50]"
    (Array.length r_shape = 2 && r_shape.(0) = 2 && r_shape.(1) = 50);
  let r1 = Spectrum.resample ~wavelength:wave2 bb1 in
  let r2 = Spectrum.resample ~wavelength:wave2 bb2 in
  let expected = Nx.stack [ Spectrum.values r1; Spectrum.values r2 ] in
  let diff =
    Nx.item [] (Nx.max (Nx.abs (Nx.sub (Spectrum.values resampled) expected)))
  in
  is_true
    ~msg:
      (Printf.sprintf "batch resample matches individual: max_diff=%.2e" diff)
    (diff < 1e-20)

let test_batch_ab_mag () =
  let wave = Unit.Length.of_m (Nx.linspace f64 3e-7 8e-7 200) in
  let bp =
    Photometry.tophat ~lo:(Unit.Length.nm 400.0) ~hi:(Unit.Length.nm 600.0)
      ~n:100
  in
  let bb1 =
    Spectrum.blackbody
      ~temperature:(Unit.Temperature.of_kelvin (Nx.scalar f64 5000.0))
      ~wavelength:wave
    |> Spectrum.as_flux_density
  in
  let bb2 =
    Spectrum.blackbody
      ~temperature:(Unit.Temperature.of_kelvin (Nx.scalar f64 8000.0))
      ~wavelength:wave
    |> Spectrum.as_flux_density
  in
  let values = Nx.stack [ Spectrum.values bb1; Spectrum.values bb2 ] in
  let batch =
    Spectrum.create ~wavelength:wave ~values |> Spectrum.as_flux_density
  in
  let mags_batch = Photometry.ab_mag bp batch in
  let mag1 = Photometry.ab_mag bp bb1 in
  let mag2 = Photometry.ab_mag bp bb2 in
  let expected = Nx.stack [ mag1; mag2 ] in
  let diff = Nx.item [] (Nx.max (Nx.abs (Nx.sub mags_batch expected))) in
  is_true
    ~msg:(Printf.sprintf "batch ab_mag matches individual: max_diff=%.2e" diff)
    (diff < 1e-10)

let test_batch_extinction () =
  let wave = Unit.Length.of_m (Nx.linspace f64 3e-7 8e-7 200) in
  let rv = Nx.scalar f64 3.1 in
  let av = Nx.scalar f64 0.5 in
  let bb1 =
    Spectrum.blackbody
      ~temperature:(Unit.Temperature.of_kelvin (Nx.scalar f64 5000.0))
      ~wavelength:wave
    |> Spectrum.as_flux_density
  in
  let bb2 =
    Spectrum.blackbody
      ~temperature:(Unit.Temperature.of_kelvin (Nx.scalar f64 8000.0))
      ~wavelength:wave
    |> Spectrum.as_flux_density
  in
  let values = Nx.stack [ Spectrum.values bb1; Spectrum.values bb2 ] in
  let batch =
    Spectrum.create ~wavelength:wave ~values |> Spectrum.as_flux_density
  in
  let reddened = Extinction.apply (Extinction.ccm89 ~rv) ~av batch in
  let r1 = Extinction.apply (Extinction.ccm89 ~rv) ~av bb1 in
  let r2 = Extinction.apply (Extinction.ccm89 ~rv) ~av bb2 in
  let expected = Nx.stack [ Spectrum.values r1; Spectrum.values r2 ] in
  let diff =
    Nx.item [] (Nx.max (Nx.abs (Nx.sub (Spectrum.values reddened) expected)))
  in
  is_true
    ~msg:
      (Printf.sprintf "batch extinction matches individual: max_diff=%.2e" diff)
    (diff < 1e-25)

let test_batch_scale () =
  let wave = Unit.Length.of_m (Nx.linspace f64 1e-7 1e-6 100) in
  let v1 = Nx.ones f64 [| 100 |] in
  let v2 = Nx.full f64 [| 100 |] 2.0 in
  let values = Nx.stack [ v1; v2 ] in
  let batch = Spectrum.create ~wavelength:wave ~values in
  let scaled = Spectrum.scale (Nx.scalar f64 3.0) batch in
  let sv = Spectrum.values scaled in
  let expected =
    Nx.stack [ Nx.full f64 [| 100 |] 3.0; Nx.full f64 [| 100 |] 6.0 ]
  in
  let diff = Nx.item [] (Nx.max (Nx.abs (Nx.sub sv expected))) in
  is_true ~msg:"batch scalar scale" (diff < 1e-15)

let test_batch_redshift_scalar () =
  let wave = Unit.Length.of_m (Nx.linspace f64 1e-7 1e-6 100) in
  let bb1 =
    Spectrum.blackbody
      ~temperature:(Unit.Temperature.of_kelvin (Nx.scalar f64 5000.0))
      ~wavelength:wave
    |> Spectrum.as_flux_density
  in
  let bb2 =
    Spectrum.blackbody
      ~temperature:(Unit.Temperature.of_kelvin (Nx.scalar f64 8000.0))
      ~wavelength:wave
    |> Spectrum.as_flux_density
  in
  let values = Nx.stack [ Spectrum.values bb1; Spectrum.values bb2 ] in
  let batch =
    Spectrum.create ~wavelength:wave ~values |> Spectrum.as_flux_density
  in
  let z = Nx.scalar f64 0.5 in
  let shifted = Spectrum.redshift ~z batch in
  let s1 = Spectrum.redshift ~z bb1 in
  let s2 = Spectrum.redshift ~z bb2 in
  let expected = Nx.stack [ Spectrum.values s1; Spectrum.values s2 ] in
  let diff =
    Nx.item [] (Nx.max (Nx.abs (Nx.sub (Spectrum.values shifted) expected)))
  in
  is_true
    ~msg:
      (Printf.sprintf "batch redshift matches individual: max_diff=%.2e" diff)
    (diff < 1e-20)

let test_batch_create_mismatch () =
  let wave = Unit.Length.of_m (Nx.linspace f64 1e-7 1e-6 100) in
  let values = Nx.ones f64 [| 3; 50 |] in
  let raised =
    try
      ignore (Spectrum.create ~wavelength:wave ~values);
      false
    with Invalid_argument _ -> true
  in
  is_true ~msg:"mismatched last dim raises" raised

let test_batch_roundtrip () =
  let wave = Unit.Length.of_m (Nx.linspace f64 1e-7 1e-6 100) in
  let v1 = Nx.ones f64 [| 100 |] in
  let v2 = Nx.full f64 [| 100 |] 2.0 in
  let v3 = Nx.full f64 [| 100 |] 3.0 in
  let values = Nx.stack [ v1; v2; v3 ] in
  let batch = Spectrum.create ~wavelength:wave ~values in
  let extracted = Nx.get [ 1 ] (Spectrum.values batch) in
  let diff = Nx.item [] (Nx.max (Nx.abs (Nx.sub extracted v2))) in
  is_true ~msg:"extract second spectrum from batch" (diff < 1e-15)

let () =
  run "Umbra"
    [
      group "Unit"
        [
          test "10 kpc converts to 0.01 Mpc" test_length_conversion;
          test "10 kpc + 500 pc = 10.5 kpc" test_length_arithmetic;
          test "1 solar mass is ~1.988e30 kg" test_mass_conversion;
          test "100 km / 10 s = 10 km/s" test_velocity_cross_dim;
          test "sin(90) = 1 and cos(90) = 0" test_angle_trig;
          test "wavelength to frequency roundtrips" test_wavelength_frequency;
          test "phantom types prevent adding length and mass"
            test_phantom_type_safety;
          test "2 eV survives energy-wavelength-frequency roundtrip"
            test_energy_wavelength_frequency;
        ];
      group "Const" [ test "speed of light is ~299792 km/s" test_const_c ];
      group "Time"
        [
          test "J2000.0 JD and MJD values are correct" test_time_jd_mjd;
          test "ISO 8601 parse and format roundtrip" test_time_iso;
          test "UTC to TAI offset is 32s at J2000" test_time_utc_tai_tt;
          test "TDB-TT difference is less than 2 ms" test_time_tdb;
          test "Unix epoch maps to JD 2440587.5" test_time_unix;
          test "diff and add with 1-day offset" test_time_diff_add;
        ];
      group "Coord"
        [
          test "ICRS to Galactic and back preserves RA/Dec" test_coord_roundtrip;
          test "ICRS to Ecliptic and back preserves RA/Dec"
            test_coord_ecliptic_roundtrip;
          test "ICRS to Supergalactic and back preserves RA/Dec"
            test_coord_supergalactic_roundtrip;
          test "north pole to south pole separation is 180 deg"
            test_separation_poles;
          test "nearest self-match returns identity indices"
            test_match_nearest_self;
        ];
      group "Cosmo"
        [
          test "H(0) equals H0 = 70 km/s/Mpc" test_cosmo_hubble;
          test "E(z=0) = 1 for any cosmology" test_cosmo_e_of;
          test "comoving(0.1) ~ 421 Mpc and luminosity(0.1) ~ 463 Mpc"
            test_cosmo_distances;
          test "lookback time at z=1 is ~7.7 Gyr" test_cosmo_lookback;
          test "1 kpc at z=0.022 subtends ~2.3 arcsec" test_cosmo_angular_scale;
          test "Planck18 comoving(0.5) ~ 1960 Mpc" test_cosmo_planck18;
          test "flat_lcdm(70, 0.3) matches default cosmology"
            test_cosmo_flat_lcdm_same_as_default;
          test "non-flat LCDM differs from flat" test_cosmo_nonflat_lcdm;
          test "wCDM with w0=-1 reduces to LCDM" test_cosmo_wcdm;
          test "w0waCDM with w0=-1 wa=0 reduces to LCDM" test_cosmo_w0wacdm;
          test "z_at_value roundtrips luminosity distance" test_cosmo_z_at_value;
          test "growth factor D(z=0) = 1" test_cosmo_growth_factor_z0;
          test "growth factor D(z=1) ~ 0.61" test_cosmo_growth_factor_z1;
          test "growth rate f(z=0) ~ 0.52" test_cosmo_growth_rate_z0;
          test "growth factor decreases with redshift"
            test_cosmo_growth_monotonic;
          test "linear power spectrum is positive and decreases with z"
            test_cosmo_linear_power;
          test "nonlinear power exceeds linear at k=1 h/Mpc"
            test_cosmo_nonlinear_power;
          test "Planck18 omega_b, n_s, and sigma8 accessors"
            test_cosmo_params_accessors;
          test "D_H(0) = c/H0" test_cosmo_dh;
          test "D_M equals D_C for flat geometry" test_cosmo_dm_flat;
          test "D_V = (z * D_H * D_M^2)^(1/3)" test_cosmo_dv;
          test "sound horizon r_s ~ 147 Mpc for Planck18"
            test_cosmo_sound_horizon;
          test "age of universe ~ 13.8 Gyr for Planck18" test_cosmo_age_planck18;
          test "age(z=0) - age(z=1) = lookback(z=1)" test_cosmo_age_at_z1;
          test "comoving distance to CMB ~ 14000 Mpc" test_cosmo_comoving_cmb;
          test "comoving distances increase at z = 2, 5, 10"
            test_cosmo_comoving_high_z;
          test "lookback time at z=5 ~ 12.5 Gyr" test_cosmo_lookback_high_z;
        ];
      group "Altaz"
        [
          test "ICRS to AltAz and back preserves RA/Dec" test_altaz_zenith;
          test "Polaris altitude ~ observer latitude from lat=45"
            test_altaz_north_pole;
          test "airmass is >= 1.0 near zenith" test_altaz_airmass_zenith;
          test "airmass differs for high vs low altitude stars"
            test_altaz_airmass_low_alt;
          test "refraction raises apparent altitude" test_altaz_refraction;
          test "standalone refraction is between 0 and 2 arcmin at high alt"
            test_altaz_refraction_standalone;
        ];
      group "Galactocentric"
        [
          test "l=0 b=0 at galcen_distance maps to origin"
            test_galactocentric_gc_position;
          test "Galactocentric to ICRS roundtrips RA/Dec/distance"
            test_galactocentric_roundtrip;
        ];
      group "Spectrum"
        [
          test "scale by 3 multiplies all values" test_spectrum_scale;
          test "multiply spectrum by transmission" test_spectrum_mul;
          test "divide spectrum by flat transmission" test_spectrum_div;
          test "mul then div roundtrips to original"
            test_spectrum_mul_div_roundtrip;
          test "blackbody peak obeys Wien's displacement law"
            test_spectrum_blackbody_wien;
          test "redshift z=1 doubles wavelength and halves flux"
            test_spectrum_redshift;
          test "resample preserves positivity and finiteness"
            test_spectrum_resample_values;
          test "Gaussian line peaks at 656.3 nm with unit amplitude"
            test_spectrum_gaussian_peak;
          test "Lorentzian line peaks at 500 nm with amplitude 3"
            test_spectrum_lorentzian_peak;
          test "Voigt with sigma >> gamma matches Gaussian"
            test_spectrum_voigt_limits;
          test "power-law continuum plus Gaussian line composes cleanly"
            test_spectrum_line_composability;
        ];
      group "Extinction"
        [
          test "CCM89 A_V/A_V ~ 1.0 at V-band 550 nm"
            test_extinction_ccm89_v_band;
          test "CCM89 extinction increases toward blue"
            test_extinction_ccm89_monotonic;
          test "apply then unredden recovers original spectrum"
            test_extinction_apply_unredden;
        ];
      group "Photometry"
        [
          test "flat f_nu = 3631 Jy gives m_AB ~ 0" test_photometry_ab_mag_flat;
          test "same-band color is zero" test_photometry_color_same_band;
          test "hot star is brighter in blue than red"
            test_photometry_blue_star_color;
          test "auto-resample matches manual resample"
            test_photometry_auto_resample;
          test "ST and AB magnitudes differ for a blackbody"
            test_photometry_st_mag;
          test "Vega and AB magnitudes differ through Johnson V"
            test_photometry_vega_mag;
          test "effective wavelength is in range for flat tophat spectrum"
            test_photometry_effective_wavelength;
        ];
      group "Filters"
        [
          test "SDSS r pivot wavelength ~ 620 nm" test_filters_sdss_pivot;
          test "Johnson V pivot wavelength ~ 551 nm"
            test_filters_johnson_v_pivot;
          test "2MASS J pivot wavelength ~ 1235 nm" test_filters_twomass_j_pivot;
          test "Gaia BP < G < RP pivot ordering" test_filters_gaia_ordering;
          test "5800 K blackbody through SDSS g yields finite magnitude"
            test_filters_photometry;
        ];
      group "Survey"
        [
          test "Smail n(z) integrates to 1.0" test_survey_smail_normalized;
          test "tabulated n(z) is positive at midpoint and zero outside"
            test_survey_tabulated;
          test "C_l matrix has correct shape for 2 tracers" test_survey_cl_shape;
          test "auto C_l is positive at all ell" test_survey_cl_positive;
          test "weak lensing noise is constant in ell" test_survey_noise_wl;
          test "shear m_bias=0.02 scales auto C_l by (1+m)^2" test_survey_m_bias;
          test "baryonic feedback suppresses high-ell power"
            test_survey_baryonic_feedback;
        ];
      group "FITS"
        [
          test "2x3 float32 image writes and reads back"
            test_fits_image_roundtrip;
          test "3-row table with ra/dec writes and reads back"
            test_fits_table_roundtrip;
        ];
      group "Batch"
        [
          test "batch of 2 spectra has shape [2; 100]" test_batch_create;
          test "mismatched wavelength and values dims raises"
            test_batch_create_mismatch;
          test "extract second spectrum from batch" test_batch_roundtrip;
          test "scalar scale applies to all spectra in batch" test_batch_scale;
          test "batch resample matches per-spectrum resample"
            test_batch_resample;
          test "batch AB magnitudes match per-spectrum magnitudes"
            test_batch_ab_mag;
          test "batch extinction matches per-spectrum extinction"
            test_batch_extinction;
          test "batch redshift matches per-spectrum redshift"
            test_batch_redshift_scalar;
        ];
    ]
