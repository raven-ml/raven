(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Type-safe units and physical constants.

   Introduces Umbra's dimensional type system: quantities carry phantom types
   that prevent mixing incompatible dimensions at compile time. Shows how to
   create, convert, and combine quantities in different units, and how to use
   physical and astronomical constants. *)

open Nx
open Umbra

let f64 = Nx.float64

let () =
  Printf.printf "Type-safe units and physical constants\n";
  Printf.printf "======================================\n\n";

  (* --- Length: metres, parsecs, AU, light-years --- *)
  Printf.printf "Length conversions\n";
  Printf.printf "------------------\n";
  let d_pc = Unit.Length.pc 1.0 in
  Printf.printf "  1 parsec = %.4e m\n" (item [] (Unit.Length.in_m d_pc));
  Printf.printf "  1 parsec = %.6f ly\n" (item [] (Unit.Length.in_ly d_pc));
  Printf.printf "  1 parsec = %.0f AU\n" (item [] (Unit.Length.in_au d_pc));

  let d_au = Unit.Length.au 1.0 in
  Printf.printf "  1 AU     = %.4e m\n" (item [] (Unit.Length.in_m d_au));
  Printf.printf "  1 AU     = %.4e pc\n\n" (item [] (Unit.Length.in_pc d_au));

  (* Adding lengths of different units — the type system ensures consistency *)
  let d_total = Unit.( + ) (Unit.Length.kpc 10.0) (Unit.Length.pc 500.0) in
  Printf.printf "  10 kpc + 500 pc = %.3f kpc\n\n"
    (item [] (Unit.Length.in_kpc d_total));

  (* --- Angle: degrees, radians, arcseconds --- *)
  Printf.printf "Angle conversions\n";
  Printf.printf "-----------------\n";
  let a_deg = Unit.Angle.deg 1.0 in
  Printf.printf "  1 degree = %.6f rad\n" (item [] (Unit.Angle.in_rad a_deg));
  Printf.printf "  1 degree = %.1f arcmin\n"
    (item [] (Unit.Angle.in_arcmin a_deg));
  Printf.printf "  1 degree = %.1f arcsec\n"
    (item [] (Unit.Angle.in_arcsec a_deg));

  let a_mas = Unit.Angle.mas 1.0 in
  Printf.printf "  1 mas    = %.4e arcsec\n\n"
    (item [] (Unit.Angle.in_arcsec a_mas));

  (* --- Temperature --- *)
  Printf.printf "Temperature\n";
  Printf.printf "-----------\n";
  let sun_t = Unit.Temperature.kelvin 5778.0 in
  Printf.printf "  Sun surface: %.0f K\n"
    (item [] (Unit.Temperature.in_kelvin sun_t));
  let sirius_t = Unit.Temperature.kelvin 9940.0 in
  Printf.printf "  Sirius:      %.0f K\n\n"
    (item [] (Unit.Temperature.in_kelvin sirius_t));

  (* --- Time durations --- *)
  Printf.printf "Time durations\n";
  Printf.printf "--------------\n";
  let t_yr = Unit.Time.yr 1.0 in
  Printf.printf "  1 Julian year = %.0f days\n"
    (item [] (Unit.Time.in_day t_yr));
  Printf.printf "  1 Julian year = %.2f s\n" (item [] (Unit.Time.in_s t_yr));

  let t_gyr = Unit.Time.gyr 13.8 in
  Printf.printf "  Age of universe ~ %.2e yr\n\n"
    (item [] (Unit.Time.in_yr t_gyr));

  (* --- Mass: kg, solar masses, Earth masses --- *)
  Printf.printf "Mass conversions\n";
  Printf.printf "----------------\n";
  let m_sun = Unit.Mass.solar_mass 1.0 in
  Printf.printf "  1 solar mass = %.4e kg\n" (item [] (Unit.Mass.in_kg m_sun));
  Printf.printf "  1 solar mass = %.0f Earth masses\n"
    (item [] (Unit.Mass.in_earth_mass m_sun));
  Printf.printf "  1 solar mass = %.1f Jupiter masses\n\n"
    (item [] (Unit.Mass.in_jupiter_mass m_sun));

  (* --- Physical constants --- *)
  Printf.printf "Physical constants\n";
  Printf.printf "------------------\n";
  Printf.printf "  c     = %.0f m/s\n" (Unit.to_float Const.c);
  Printf.printf "  h     = %.4e J s\n" Const.h_si;
  Printf.printf "  k_B   = %.4e J/K\n" Const.k_b_si;
  Printf.printf "  G     = %.4e m^3 kg^-1 s^-2\n" Const.g_si;
  Printf.printf "  sigma = %.4e W m^-2 K^-4\n\n" Const.sigma_sb_si;

  (* --- Astronomical constants --- *)
  Printf.printf "Astronomical constants\n";
  Printf.printf "----------------------\n";
  Printf.printf "  L_sun = %.4e W\n"
    (item [] (Unit.Power.in_w Const.solar_luminosity));
  Printf.printf "  R_sun = %.4e m\n"
    (item [] (Unit.Length.in_m Const.solar_radius));
  Printf.printf "  M_sun = %.4e kg\n"
    (item [] (Unit.Mass.in_kg Const.solar_mass));
  Printf.printf "  1 AU  = %.4e m\n" (item [] (Unit.Length.in_m Const.au));
  Printf.printf "  1 pc  = %.4e m\n\n" (item [] (Unit.Length.in_m Const.pc));

  (* --- Cross-dimension: parallax to distance --- *)
  Printf.printf "Parallax to distance\n";
  Printf.printf "--------------------\n";
  let parallax = Unit.Angle.arcsec 1.0 in
  let dist = Unit.parallax_to_distance parallax in
  Printf.printf "  1 arcsec parallax -> %.3f pc\n"
    (item [] (Unit.Length.in_pc dist));

  let proxima_parallax = Unit.Angle.mas 768.5 in
  let proxima_dist = Unit.parallax_to_distance proxima_parallax in
  Printf.printf "  Proxima Cen (768.5 mas) -> %.3f pc\n"
    (item [] (Unit.Length.in_pc proxima_dist));

  (* --- Tensor operations: batch unit conversions --- *)
  Printf.printf "\nBatch operations\n";
  Printf.printf "----------------\n";
  let wavelengths_nm =
    create f64 [| 5 |] [| 380.0; 450.0; 550.0; 650.0; 750.0 |]
  in
  let wavelengths = Unit.Length.of_nm wavelengths_nm in
  let wavelengths_angstrom = Unit.Length.in_angstrom wavelengths in
  Printf.printf "  Wavelengths (nm):       %s\n"
    (Nx.data_to_string wavelengths_nm);
  Printf.printf "  Wavelengths (angstrom): %s\n"
    (Nx.data_to_string wavelengths_angstrom);

  (* Convert wavelength to frequency *)
  let freqs = Unit.wavelength_to_frequency wavelengths in
  Printf.printf "  Frequencies (Hz):       %s\n"
    (Nx.data_to_string (Unit.Frequency.in_hz freqs));

  (* Wien's law: peak wavelength of a blackbody *)
  Printf.printf "\nWien's displacement law\n";
  Printf.printf "----------------------\n";
  let b_wien = Const.b_wien_si in
  let sun_peak_m = b_wien /. item [] (Unit.Temperature.in_kelvin sun_t) in
  Printf.printf "  Sun (T=%.0f K): peak at %.0f nm\n"
    (item [] (Unit.Temperature.in_kelvin sun_t))
    (sun_peak_m *. 1e9);
  let sirius_peak_m = b_wien /. item [] (Unit.Temperature.in_kelvin sirius_t) in
  Printf.printf "  Sirius (T=%.0f K): peak at %.0f nm\n"
    (item [] (Unit.Temperature.in_kelvin sirius_t))
    (sirius_peak_m *. 1e9)
