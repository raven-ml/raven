(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* K-corrections, extinction, and magnitude systems.

   Demonstrates three key photometric concepts:

   1. Magnitude systems: AB, ST, and Vega magnitudes through real SDSS filters.
   2. K-corrections: the difference between observed and rest-frame magnitudes
   due to redshift shifting the SED across the bandpass. 3. Extinction: how
   interstellar dust reddens and dims stellar light, comparing CCM89 and
   Fitzpatrick99 extinction laws. *)

open Nx
open Umbra

let f64 = Nx.float64

let () =
  Printf.printf "Extinction, K-corrections, and magnitude systems\n";
  Printf.printf "=================================================\n\n";

  (* --- Part 1: Magnitude systems --- *)
  Printf.printf "Part 1: AB, ST, and Vega magnitudes\n";
  Printf.printf "------------------------------------\n\n";

  let temp = Unit.Temperature.of_kelvin (Nx.scalar f64 6000.0) in
  let norm = Nx.scalar f64 (Float.exp (-50.0)) in
  let bands =
    [|
      ("SDSS u", Filters.sdss_u);
      ("SDSS g", Filters.sdss_g);
      ("SDSS r", Filters.sdss_r);
      ("SDSS i", Filters.sdss_i);
      ("SDSS z", Filters.sdss_z);
    |]
  in

  Printf.printf "  Source: T=6000 K blackbody\n\n";
  Printf.printf "%8s  %8s  %8s  %8s\n" "Band" "AB" "ST" "Vega";
  Printf.printf "%8s  %8s  %8s  %8s\n" "--------" "--------" "--------"
    "--------";
  Array.iter
    (fun (name, bp) ->
      let bp_wave = Photometry.wavelength bp in
      let sed =
        Spectrum.blackbody ~temperature:temp ~wavelength:bp_wave
        |> Spectrum.scale norm |> Spectrum.as_flux_density
      in
      let m_ab = item [] (Photometry.ab_mag bp sed) in
      let m_st = item [] (Photometry.st_mag bp sed) in
      let m_vega = item [] (Photometry.vega_mag bp sed) in
      Printf.printf "%8s  %+8.3f  %+8.3f  %+8.3f\n" name m_ab m_st m_vega)
    bands;

  Printf.printf "\n  Note: AB and ST systems are defined by reference flux\n";
  Printf.printf "  densities; Vega magnitudes use the alpha Lyr spectrum.\n\n";

  (* --- Part 2: K-corrections --- *)
  Printf.printf "Part 2: K-corrections\n";
  Printf.printf "---------------------\n\n";

  let bp = Filters.sdss_r in
  let bp_wave = Photometry.wavelength bp in

  let rest_sed =
    Spectrum.blackbody ~temperature:temp ~wavelength:bp_wave
    |> Spectrum.scale norm |> Spectrum.as_flux_density
  in
  let m_ab_rest = item [] (Photometry.ab_mag bp rest_sed) in
  let m_st_rest = item [] (Photometry.st_mag bp rest_sed) in
  let m_vega_rest = item [] (Photometry.vega_mag bp rest_sed) in

  Printf.printf "  Rest-frame SDSS r-band:\n";
  Printf.printf "    AB   = %.3f\n" m_ab_rest;
  Printf.printf "    ST   = %.3f\n" m_st_rest;
  Printf.printf "    Vega = %.3f\n\n" m_vega_rest;

  Printf.printf "  K-correction = m_obs(z) - m_rest\n\n";
  Printf.printf "%6s  %8s  %8s  %8s\n" "z" "K_AB" "K_ST" "K_Vega";
  Printf.printf "%6s  %8s  %8s  %8s\n" "------" "--------" "--------" "--------";

  let redshifts = [| 0.1; 0.2; 0.3; 0.5; 0.7; 1.0 |] in
  Array.iter
    (fun z ->
      let zv = Nx.scalar f64 z in
      let obs_sed =
        Spectrum.blackbody ~temperature:temp ~wavelength:bp_wave
        |> Spectrum.scale norm |> Spectrum.as_flux_density
        |> Spectrum.redshift ~z:zv
      in
      let k_ab = item [] (Photometry.ab_mag bp obs_sed) -. m_ab_rest in
      let k_st = item [] (Photometry.st_mag bp obs_sed) -. m_st_rest in
      let k_vega = item [] (Photometry.vega_mag bp obs_sed) -. m_vega_rest in
      Printf.printf "%6.2f  %+8.3f  %+8.3f  %+8.3f\n" z k_ab k_st k_vega)
    redshifts;

  Printf.printf "\n";

  (* --- Part 3: Color evolution with redshift --- *)
  Printf.printf "Part 3: Color evolution (u-r) with redshift\n";
  Printf.printf "-------------------------------------------\n\n";
  Printf.printf "%6s  %8s\n" "z" "u-r (AB)";
  Printf.printf "%6s  %8s\n" "------" "--------";
  Array.iter
    (fun z ->
      let zv = Nx.scalar f64 z in
      let color =
        item []
          (Photometry.color Filters.sdss_u Filters.sdss_r
             (Spectrum.blackbody ~temperature:temp ~wavelength:bp_wave
             |> Spectrum.scale norm |> Spectrum.as_flux_density
             |> Spectrum.redshift ~z:zv))
      in
      Printf.printf "%6.2f  %+8.3f\n" z color)
    redshifts;

  Printf.printf "\n";

  (* --- Part 4: Extinction --- *)
  Printf.printf "Part 4: Dust extinction\n";
  Printf.printf "-----------------------\n\n";

  let rv = Nx.scalar f64 3.1 in
  let av_values = [| 0.0; 0.5; 1.0; 2.0; 3.0 |] in

  Printf.printf "  CCM89 extinction law (R_V = 3.1)\n";
  Printf.printf "  Reddening a T=6000 K blackbody through SDSS r-band\n\n";
  Printf.printf "%6s  %8s  %8s  %8s\n" "A_V" "m_AB" "delta_m" "E(u-r)";
  Printf.printf "%6s  %8s  %8s  %8s\n" "------" "--------" "--------" "--------";

  let unreddened_sed =
    Spectrum.blackbody ~temperature:temp ~wavelength:bp_wave
    |> Spectrum.scale norm |> Spectrum.as_flux_density
  in
  let m0 = item [] (Photometry.ab_mag bp unreddened_sed) in
  let color0 =
    item [] (Photometry.color Filters.sdss_u Filters.sdss_r unreddened_sed)
  in

  Array.iter
    (fun av_f ->
      let av = Nx.scalar f64 av_f in
      let reddened =
        Spectrum.blackbody ~temperature:temp ~wavelength:bp_wave
        |> Spectrum.scale norm
        |> Extinction.apply (Extinction.ccm89 ~rv) ~av
        |> Spectrum.as_flux_density
      in
      let m = item [] (Photometry.ab_mag bp reddened) in
      let color =
        item [] (Photometry.color Filters.sdss_u Filters.sdss_r reddened)
      in
      Printf.printf "%6.1f  %8.3f  %+8.3f  %+8.3f\n" av_f m (m -. m0)
        (color -. color0))
    av_values;

  Printf.printf "\n";

  (* Compare extinction laws *)
  Printf.printf "  Comparing extinction laws at A_V = 1.0:\n\n";
  Printf.printf "%16s  %8s  %8s\n" "Law" "r-band" "E(u-r)";
  Printf.printf "%16s  %8s  %8s\n" "----------------" "--------" "--------";

  let av_one = Nx.scalar f64 1.0 in
  let laws =
    [|
      ("CCM89", Extinction.ccm89 ~rv);
      ("Fitzpatrick99", Extinction.fitzpatrick99 ~rv);
      ("O'Donnell94", Extinction.odonnell94 ~rv);
    |]
  in
  Array.iter
    (fun (name, law) ->
      let reddened =
        Spectrum.blackbody ~temperature:temp ~wavelength:bp_wave
        |> Spectrum.scale norm
        |> Extinction.apply law ~av:av_one
        |> Spectrum.as_flux_density
      in
      let m = item [] (Photometry.ab_mag bp reddened) in
      let color =
        item [] (Photometry.color Filters.sdss_u Filters.sdss_r reddened)
      in
      Printf.printf "%16s  %+8.3f  %+8.3f\n" name (m -. m0) (color -. color0))
    laws
