(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Photometric redshift estimation via template fitting.

   Demonstrates composing Spectrum.redshift -> Extinction.apply ->
   Photometry.ab_mag through real SDSS filters, with gradient refinement via
   Rune's autodiff. Auto-resampling makes the pipeline seamless.

   Stage 1: Grid search over redshift to find a coarse estimate. Stage 2: Adam
   optimizer refines z and normalization using AD gradients. *)

open Nx
open Umbra

let f64 = Nx.float64

let bands =
  [
    Filters.sdss_u;
    Filters.sdss_g;
    Filters.sdss_r;
    Filters.sdss_i;
    Filters.sdss_z;
  ]

let band_names = [| "u"; "g"; "r"; "i"; "z" |]

(* True parameters for synthetic galaxy *)
let true_z = 0.3
let true_temp = 5500.0
let true_av = 0.2
let true_log_norm = -50.0
let rv = Nx.scalar f64 3.1

(* Synthetic observed magnitudes *)
let obs_mags =
  let temp = Unit.Temperature.of_kelvin (Nx.scalar f64 true_temp) in
  let z = Nx.scalar f64 true_z in
  let av = Nx.scalar f64 true_av in
  let norm = Nx.scalar f64 (Float.exp true_log_norm) in
  List.map
    (fun bp ->
      let bp_wave = Photometry.wavelength bp in
      let sed =
        Spectrum.blackbody ~temperature:temp ~wavelength:bp_wave
        |> Spectrum.scale norm
        |> Extinction.apply (Extinction.ccm89 ~rv) ~av
        |> Spectrum.as_flux_density |> Spectrum.redshift ~z
      in
      Photometry.ab_mag bp sed)
    bands

(* Grid search: coarse scan over z *)
let grid_search () =
  let best_z = ref 0.0 in
  let best_chi2 = ref Float.infinity in
  let n_z = 30 in
  for iz = 0 to n_z - 1 do
    let z = Nx.scalar f64 (0.01 +. (Float.of_int iz *. 0.03)) in
    let temp = Unit.Temperature.of_kelvin (Nx.scalar f64 5000.0) in
    let norm = Nx.scalar f64 (Float.exp (-50.0)) in
    let pred =
      List.map
        (fun bp ->
          let bp_wave = Photometry.wavelength bp in
          let sed =
            Spectrum.blackbody ~temperature:temp ~wavelength:bp_wave
            |> Spectrum.scale norm |> Spectrum.as_flux_density
            |> Spectrum.redshift ~z
          in
          Photometry.ab_mag bp sed)
        bands
    in
    (* Color-based chi-squared: compare color differences *)
    let chi2 =
      List.fold_left2
        (fun acc p o -> add acc (square (sub p o)))
        (scalar f64 0.0) pred obs_mags
    in
    let chi2_v = item [] chi2 in
    if chi2_v < !best_chi2 then begin
      best_chi2 := chi2_v;
      best_z := item [] z
    end
  done;
  !best_z

(* Gradient refinement around grid minimum *)
let refine z0 =
  let loss params =
    match params with
    | [ log_z1; log_norm ] ->
        let z = sub (exp log_z1) (scalar f64 1.0) in
        let temp = Unit.Temperature.of_kelvin (Nx.scalar f64 5500.0) in
        let norm = exp log_norm in
        let pred =
          List.map
            (fun bp ->
              let bp_wave = Photometry.wavelength bp in
              let sed =
                Spectrum.blackbody ~temperature:temp ~wavelength:bp_wave
                |> Spectrum.scale norm |> Spectrum.as_flux_density
                |> Spectrum.redshift ~z
              in
              Photometry.ab_mag bp sed)
            bands
        in
        List.fold_left2
          (fun acc p o -> add acc (square (sub p o)))
          (scalar f64 0.0) pred obs_mags
    | _ -> failwith "expected [log_z1; log_norm]"
  in
  let algo = Vega.adam (Vega.Schedule.constant 5e-4) in
  let log_z1 = ref (scalar f64 (Float.log (1.0 +. z0))) in
  let log_norm = ref (scalar f64 (-50.0)) in
  let states = [| Vega.init algo !log_z1; Vega.init algo !log_norm |] in
  let refs = [| log_z1; log_norm |] in
  for _ = 0 to 499 do
    let _loss_val, grads = Rune.value_and_grads loss [ !log_z1; !log_norm ] in
    List.iteri
      (fun j g ->
        let p, s = Vega.step states.(j) ~grad:g ~param:!(refs.(j)) in
        refs.(j) := p;
        states.(j) <- s)
      grads
  done;
  Float.exp (item [] !log_z1) -. 1.0

let () =
  Printf.printf "Photometric Redshift Estimation\n";
  Printf.printf "===============================\n";
  Printf.printf
    "Pipeline: blackbody -> redshift -> extinction -> ab_mag (SDSS)\n\n";
  Printf.printf "True: z=%.3f  T=%.0fK  A_V=%.2f\n\n" true_z true_temp true_av;
  Printf.printf "Observed magnitudes:\n";
  List.iteri
    (fun i m -> Printf.printf "  %s = %.3f\n" band_names.(i) (item [] m))
    obs_mags;
  Printf.printf "\nStep 1: Grid search (z = 0.01 to 0.90)...\n";
  let z_grid = grid_search () in
  Printf.printf "  Best grid z = %.3f\n" z_grid;
  Printf.printf "\nStep 2: Gradient refinement (500 Adam steps)...\n";
  let z_fit = refine z_grid in
  Printf.printf "  Refined z = %.4f  (true: %.3f)\n" z_fit true_z;
  Printf.printf "  Error = %.4f\n" (Float.abs (z_fit -. true_z))
