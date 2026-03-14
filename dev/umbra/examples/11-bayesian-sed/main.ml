(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Fisher information and HMC sampling for SED parameter estimation.

   Demonstrates two capabilities:

   1. Fisher matrix: compute the Cramer-Rao bounds on temperature and extinction
   -- "how well CAN I constrain these parameters from UGRIZ photometry?" --
   before taking any data. Computed inline from Rune.jacrev + linear algebra.

   2. HMC sampling: full Bayesian posterior through the differentiable Spectrum
   -> Extinction -> Photometry pipeline, via Norn.hmc. *)

open Nx
open Umbra

let f64 = Nx.float64

(* Bandpasses *)
let n_bp = 20

let bands =
  [
    Photometry.tophat ~lo:(Unit.Length.m 3.0e-7) ~hi:(Unit.Length.m 4.0e-7)
      ~n:n_bp;
    Photometry.tophat ~lo:(Unit.Length.m 4.0e-7) ~hi:(Unit.Length.m 5.5e-7)
      ~n:n_bp;
    Photometry.tophat ~lo:(Unit.Length.m 5.5e-7) ~hi:(Unit.Length.m 7.0e-7)
      ~n:n_bp;
    Photometry.tophat ~lo:(Unit.Length.m 7.0e-7) ~hi:(Unit.Length.m 8.5e-7)
      ~n:n_bp;
    Photometry.tophat ~lo:(Unit.Length.m 8.5e-7) ~hi:(Unit.Length.m 1.0e-6)
      ~n:n_bp;
  ]

let band_names = [| "U"; "G"; "R"; "I"; "Z" |]
let rv = Nx.scalar f64 3.1

(* Forward model: [log_T, A_V] -> 5 magnitudes *)
let model params =
  let log_temp = Nx.reshape [||] (Nx.slice [ I 0 ] params) in
  let av = Nx.reshape [||] (Nx.slice [ I 1 ] params) in
  let temp = Unit.Temperature.of_kelvin (Nx.exp log_temp) in
  let mags =
    List.map
      (fun bp ->
        let wave = Photometry.wavelength bp in
        let sed =
          Spectrum.blackbody ~temperature:temp ~wavelength:wave
          |> Extinction.apply (Extinction.ccm89 ~rv) ~av
          |> Spectrum.as_flux_density
        in
        Photometry.ab_mag bp sed)
      bands
  in
  Nx.stack ~axis:0 mags

(* True parameters *)
let true_log_temp = Float.log 6500.0
let true_av = 0.5
let true_params = Nx.create f64 [| 2 |] [| true_log_temp; true_av |]

(* Synthetic observations *)
let obs_errs = Nx.create f64 [| 5 |] [| 0.05; 0.03; 0.03; 0.04; 0.05 |]

let obs_mags =
  let true_mags = model true_params in
  let noise = Nx.create f64 [| 5 |] [| 0.03; -0.02; 0.01; -0.01; 0.02 |] in
  Nx.add true_mags noise

(* Fisher information: F = J^T C^-1 J *)
let fisher f ~params ~obs_cov =
  let j = Rune.jacrev f params in
  let jt = Nx.matrix_transpose j in
  Nx.matmul (Nx.matmul jt (Nx.inv obs_cov)) j

(* Cramer-Rao bounds: sigma = sqrt(diag(F^-1)) *)
let marginal_sigma f = Nx.sqrt (Nx.diagonal (Nx.inv f))

let () =
  Printf.printf "Fisher Information & HMC for SED Fitting\n";
  Printf.printf "=========================================\n\n";

  Printf.printf "True parameters:\n";
  Printf.printf "  T   = %.0f K  (log_T = %.4f)\n" (Float.exp true_log_temp)
    true_log_temp;
  Printf.printf "  A_V = %.2f\n\n" true_av;

  Printf.printf "Observed magnitudes:\n";
  Array.iteri
    (fun i name ->
      Printf.printf "  %s = %.3f +/- %.3f\n" name (item [ i ] obs_mags)
        (item [ i ] obs_errs))
    band_names;
  Printf.printf "\n";

  (* --- Fisher Information --- *)
  Printf.printf "=== Fisher Information ===\n\n";

  let obs_cov = Nx.diag (Nx.square obs_errs) in
  let f = fisher model ~params:true_params ~obs_cov in
  let sigma = marginal_sigma f in

  Printf.printf "Fisher matrix:\n";
  Printf.printf "  F = [[ %10.2f  %10.2f ]\n"
    (item [ 0; 0 ] f)
    (item [ 0; 1 ] f);
  Printf.printf "       [ %10.2f  %10.2f ]]\n\n"
    (item [ 1; 0 ] f)
    (item [ 1; 1 ] f);

  Printf.printf "Cramer-Rao bounds (best achievable 1-sigma):\n";
  let sigma_log_t = item [ 0 ] sigma in
  let sigma_av = item [ 1 ] sigma in
  Printf.printf "  sigma(log_T) = %.4f  ->  sigma(T) ~ %.0f K\n" sigma_log_t
    (sigma_log_t *. Float.exp true_log_temp);
  Printf.printf "  sigma(A_V)   = %.4f\n\n" sigma_av;

  (* --- HMC Sampling --- *)
  Printf.printf "=== HMC Posterior Sampling ===\n\n";

  (* Log-posterior: Gaussian likelihood, flat prior *)
  let log_posterior params =
    let pred = model params in
    let residuals = Nx.div (Nx.sub pred obs_mags) obs_errs in
    Nx.mul_s (Nx.sum (Nx.square residuals)) (-0.5)
  in

  let init = Nx.create f64 [| 2 |] [| Float.log 7000.0; 0.3 |] in
  let result =
    Norn.hmc ~step_size:0.001 ~num_leapfrog:10 ~num_warmup:200 ~n:500
      log_posterior init
  in

  Printf.printf "HMC diagnostics:\n";
  Printf.printf "  Accept rate: %.1f%%\n\n" (result.stats.accept_rate *. 100.);

  (* Sample statistics *)
  let sample_mean = Nx.mean ~axes:[ 0 ] result.samples in
  let centered = Nx.sub result.samples sample_mean in
  let sample_cov =
    Nx.div_s
      (Nx.matmul (Nx.matrix_transpose centered) centered)
      (Float.of_int 499)
  in
  let sample_std = Nx.sqrt (Nx.diagonal sample_cov) in

  let hmc_log_t = item [ 0 ] sample_mean in
  let hmc_av = item [ 1 ] sample_mean in
  let hmc_sigma_log_t = item [ 0 ] sample_std in
  let hmc_sigma_av = item [ 1 ] sample_std in

  Printf.printf "Posterior (HMC):\n";
  Printf.printf "  log_T = %.4f +/- %.4f  ->  T ~ %.0f K\n" hmc_log_t
    hmc_sigma_log_t (Float.exp hmc_log_t);
  Printf.printf "  A_V   = %.4f +/- %.4f\n\n" hmc_av hmc_sigma_av;

  (* --- Comparison --- *)
  Printf.printf "=== Fisher vs HMC Comparison ===\n\n";
  Printf.printf "  %12s  %10s  %10s\n" "" "Fisher s" "HMC s";
  Printf.printf "  %12s  %10s  %10s\n" "------------" "----------" "----------";
  Printf.printf "  %12s  %10.4f  %10.4f\n" "s(log_T)" sigma_log_t
    hmc_sigma_log_t;
  Printf.printf "  %12s  %10.4f  %10.4f\n\n" "s(A_V)" sigma_av hmc_sigma_av;

  Printf.printf "Fisher gives the theoretical minimum uncertainty.\n";
  Printf.printf "HMC gives the actual posterior width.\n";
  Printf.printf "Agreement confirms the model is well-behaved (near-linear).\n"
