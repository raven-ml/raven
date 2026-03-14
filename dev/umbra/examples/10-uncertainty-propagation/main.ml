(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Automatic uncertainty propagation through cosmological distances.

   Demonstrates propagating H0 and Omega_m uncertainties through
   Umbra.Cosmo.distance_modulus using exact AD Jacobians. The linear error
   propagation formula (Sigma_out = J Sigma_in J^T) is computed automatically
   via Rune.jacfwd. Results are validated against Monte Carlo sampling.

   Fisher, propagation, and Monte Carlo are all trivial given Rune's jacfwd --
   no dedicated library needed. *)

open Nx
open Umbra

let f64 = Nx.float64

(* Redshifts to evaluate *)
let redshifts = [| 0.1; 0.3; 0.5; 0.7; 1.0 |]

(* Forward model: given [H0; Omega_m], compute distance modulus at z *)
let distance_modulus_at z p =
  let h0 = Nx.reshape [||] (Nx.slice [ I 0 ] p) in
  let om = Nx.reshape [||] (Nx.slice [ I 1 ] p) in
  let cosmo = Cosmo.create_flat_lcdm ~h0 ~omega_m:om in
  Cosmo.distance_modulus ~p:cosmo (Nx.scalar f64 z)

(* Linear error propagation: Sigma_out = J Sigma_in J^T *)
let propagate f ~mean ~cov =
  let j = Rune.jacfwd f mean in
  let mean_out = f mean in
  let cov_out = Nx.matmul (Nx.matmul j cov) (Nx.matrix_transpose j) in
  let cov_out = Nx.div_s (Nx.add cov_out (Nx.matrix_transpose cov_out)) 2.0 in
  (mean_out, cov_out)

(* Monte Carlo validation *)
let monte_carlo ?(n_samples = 50_000) f ~mean ~cov =
  let n = Nx.numel mean in
  let l = Nx.cholesky cov in
  let z = Nx.randn f64 [| n_samples; n |] in
  let samples = Nx.add (Nx.matmul z (Nx.matrix_transpose l)) mean in
  let y0 = f (Nx.slice [ I 0 ] samples) in
  let m = Nx.numel y0 in
  let outputs = Nx.zeros f64 [| n_samples; m |] in
  Nx.set_slice [ I 0 ] outputs y0;
  for i = 1 to n_samples - 1 do
    Nx.set_slice [ I i ] outputs (f (Nx.slice [ I i ] samples))
  done;
  let mean_out = Nx.mean ~axes:[ 0 ] outputs in
  let centered = Nx.sub outputs mean_out in
  let cov_out =
    Nx.div_s
      (Nx.matmul (Nx.matrix_transpose centered) centered)
      (Float.of_int (n_samples - 1))
  in
  (mean_out, cov_out)

let () =
  Printf.printf "Automatic Uncertainty Propagation through Cosmology\n";
  Printf.printf "====================================================\n\n";

  (* Parameters with uncertainties *)
  let h0_mean = 70.0 and h0_std = 1.0 in
  let om_mean = 0.30 and om_std = 0.01 in

  Printf.printf "Input parameters:\n";
  Printf.printf "  H0      = %.1f +/- %.1f km/s/Mpc\n" h0_mean h0_std;
  Printf.printf "  Omega_m = %.2f +/- %.2f\n\n" om_mean om_std;

  let mean = Nx.create f64 [| 2 |] [| h0_mean; om_mean |] in
  let std = Nx.create f64 [| 2 |] [| h0_std; om_std |] in
  let cov = Nx.diag (Nx.square std) in

  Printf.printf "%5s  %10s  %10s  %10s  %10s\n" "z" "mu (AD)" "sigma (AD)"
    "sigma (MC)" "agreement";
  Printf.printf "%5s  %10s  %10s  %10s  %10s\n" "-----" "----------"
    "----------" "----------" "----------";

  Array.iter
    (fun z ->
      (* AD-based propagation *)
      let f p = Nx.reshape [| 1 |] (distance_modulus_at z p) in
      let mean_ad, cov_ad = propagate f ~mean ~cov in
      let mu_ad = item [ 0 ] mean_ad in
      let std_ad = Float.sqrt (item [ 0; 0 ] cov_ad) in

      (* Monte Carlo validation *)
      let _, cov_mc = monte_carlo f ~mean ~cov in
      let std_mc = Float.sqrt (item [ 0; 0 ] cov_mc) in

      let agreement = Float.abs (std_ad -. std_mc) /. std_mc *. 100.0 in
      Printf.printf "%5.1f  %10.4f  %10.4f  %10.4f  %9.1f%%\n" z mu_ad std_ad
        std_mc agreement)
    redshifts;

  Printf.printf "\n";
  Printf.printf "AD uses exact Jacobians (2 JVP calls for 2 parameters).\n";
  Printf.printf "MC uses 50,000 samples for validation.\n";
  Printf.printf "Agreement < 1%% confirms linear propagation is accurate.\n";

  (* Also demonstrate the simple scalar API *)
  Printf.printf "\n--- Scalar API demo ---\n\n";
  Printf.printf "Propagating z = 0.5 +/- 0.01 through distance_modulus:\n";
  let x = Nx.scalar f64 0.5 in
  let y, dy =
    Rune.jvp (fun z -> Cosmo.distance_modulus z) x (Nx.scalar f64 1.0)
  in
  let mu_mean = Nx.item [] y in
  let mu_std = Float.abs (Nx.item [] dy) *. 0.01 in
  Printf.printf "  mu = %.4f +/- %.4f\n" mu_mean mu_std
