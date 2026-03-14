(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Batch template photometry.

   Computes SDSS g-r colors for a grid of blackbody templates at different
   temperatures and dust extinctions in a single pass, demonstrating batched
   spectra. Instead of looping over individual spectra, the values tensor has a
   leading batch dimension and all photometry operations broadcast over it. *)

open Nx
open Umbra

let f64 = Nx.float64

let () =
  Printf.printf "Batch Template Photometry\n";
  Printf.printf "=========================\n\n";

  (* Temperature grid: 20 blackbodies from 3000K to 30000K *)
  let n_temp = 20 in
  let temps =
    Array.init n_temp (fun i ->
        3000.0
        +. (Float.of_int i *. (30000.0 -. 3000.0) /. Float.of_int (n_temp - 1)))
  in

  (* Shared wavelength grid covering SDSS g and r *)
  let wavelength = Unit.Length.of_m (Nx.linspace f64 3e-7 1.1e-6 500) in

  (* Build batch spectrum: stack individual blackbodies into [n_temp; 500] *)
  let values =
    Nx.stack
      (List.init n_temp (fun i ->
           let temp = Unit.Temperature.of_kelvin (Nx.scalar f64 temps.(i)) in
           Spectrum.values (Spectrum.blackbody ~temperature:temp ~wavelength)))
  in
  let batch = Spectrum.create ~wavelength ~values |> Spectrum.as_flux_density in

  (* AB magnitudes in g and r — returns shape [n_temp] each *)
  let g_mag = Photometry.ab_mag Filters.sdss_g batch in
  let r_mag = Photometry.ab_mag Filters.sdss_r batch in
  let g_r = Nx.sub g_mag r_mag in

  Printf.printf "Unreddened blackbody colors (SDSS g-r):\n";
  Printf.printf "%8s  %8s  %8s  %8s\n" "T (K)" "g" "r" "g-r";
  Printf.printf "%8s  %8s  %8s  %8s\n" "--------" "--------" "--------"
    "--------";
  Array.iteri
    (fun i t ->
      if i mod 4 = 0 || i = n_temp - 1 then
        Printf.printf "%8.0f  %+8.3f  %+8.3f  %+8.3f\n" t (item [ i ] g_mag)
          (item [ i ] r_mag) (item [ i ] g_r))
    temps;

  (* Now apply per-spectrum extinction: A_V from 0.0 to 2.0 *)
  Printf.printf "\nReddening a T=6000K blackbody (SDSS g-r vs A_V):\n";
  let n_av = 10 in
  let av_values = Nx.linspace f64 0.0 2.0 n_av in

  (* Single-temperature spectrum, batched over A_V *)
  let temp_6k = Unit.Temperature.of_kelvin (Nx.scalar f64 6000.0) in
  let sed_1d =
    Spectrum.blackbody ~temperature:temp_6k ~wavelength
    |> Spectrum.as_flux_density
  in
  (* Replicate into [n_av; 500] *)
  let sed_values =
    Nx.stack (List.init n_av (fun _ -> Spectrum.values sed_1d))
  in
  let sed_batch =
    Spectrum.create ~wavelength ~values:sed_values |> Spectrum.as_flux_density
  in
  (* Per-spectrum A_V: reshape to [n_av; 1] to broadcast with [n_av; 500] *)
  let rv = Nx.scalar f64 3.1 in
  let av_col = Nx.reshape [| n_av; 1 |] av_values in
  let reddened = Extinction.apply (Extinction.ccm89 ~rv) ~av:av_col sed_batch in
  let g_red = Photometry.ab_mag Filters.sdss_g reddened in
  let r_red = Photometry.ab_mag Filters.sdss_r reddened in
  let g_r_red = Nx.sub g_red r_red in

  Printf.printf "%8s  %8s\n" "A_V" "g-r";
  Printf.printf "%8s  %8s\n" "--------" "--------";
  for i = 0 to n_av - 1 do
    Printf.printf "%8.2f  %+8.3f\n" (item [ i ] av_values) (item [ i ] g_r_red)
  done
