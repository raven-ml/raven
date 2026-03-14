(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Bayesian linear regression: infer slope and intercept from noisy data.

   Model: y_i = slope * x_i + intercept + eps_i, eps_i ~ N(0, sigma^2)

   True parameters: slope = 2.0, intercept = 1.0, sigma = 0.5

   Priors: slope ~ N(0, 10) intercept ~ N(0, 10)

   We sample the posterior with NUTS and report credible intervals. *)

let f64 = Nx.float64

(* Generate synthetic data: y = 2x + 1 + noise *)
let n_data = 50
let true_slope = 2.0
let true_intercept = 1.0
let noise_sigma = 0.5

let gen_data () =
  let x = Nx.linspace f64 (-2.0) 2.0 n_data in
  let noise = Nx.mul_s (Nx.randn f64 [| n_data |]) noise_sigma in
  let y =
    Nx.add (Nx.add (Nx.mul_s x true_slope) (Nx.scalar f64 true_intercept)) noise
  in
  (x, y)

(* Log-posterior: Gaussian likelihood + normal prior. params = [slope;
   intercept] *)
let log_posterior x_data y_data params =
  let slope = Nx.slice [ I 0 ] params in
  let intercept = Nx.slice [ I 1 ] params in
  (* Predicted values *)
  let y_pred = Nx.add (Nx.mul x_data slope) intercept in
  let residuals = Nx.sub y_data y_pred in
  (* Log-likelihood: -0.5 * sum((y - y_pred)^2) / sigma^2 *)
  let ll =
    Nx.div_s
      (Nx.mul_s (Nx.sum (Nx.square residuals)) (-0.5))
      (noise_sigma *. noise_sigma)
  in
  (* Log-prior: N(0, 10) on each parameter *)
  let lp_slope = Nx.mul_s (Nx.square slope) (-0.5 /. 100.0) in
  let lp_intercept = Nx.mul_s (Nx.square intercept) (-0.5 /. 100.0) in
  Nx.add ll (Nx.add lp_slope lp_intercept)

let percentile samples frac =
  let n = (Nx.shape samples).(0) in
  let sorted, _ = Nx.sort samples in
  let idx = Float.to_int (frac *. Float.of_int (n - 1)) in
  Nx.item [ idx ] sorted

let () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let x_data, y_data = gen_data () in
  let init = Nx.zeros f64 [| 2 |] in
  let log_prob = log_posterior x_data y_data in
  let result = Norn.nuts ~n:2000 ~num_warmup:1000 log_prob init in

  Printf.printf "--- Bayesian Linear Regression (NUTS, 2000 samples) ---\n\n";
  Printf.printf "True:      slope = %.2f, intercept = %.2f\n" true_slope
    true_intercept;

  let sample_mean = Nx.mean ~axes:[ 0 ] result.samples in
  Printf.printf "Posterior: slope = %.3f, intercept = %.3f\n"
    (Nx.item [ 0 ] sample_mean)
    (Nx.item [ 1 ] sample_mean);

  (* 95%% credible intervals *)
  Printf.printf "\n95%% credible intervals:\n";
  let slope_samples = Nx.slice [ A; I 0 ] result.samples in
  let intercept_samples = Nx.slice [ A; I 1 ] result.samples in
  Printf.printf "  slope:     [%.3f, %.3f]\n"
    (percentile slope_samples 0.025)
    (percentile slope_samples 0.975);
  Printf.printf "  intercept: [%.3f, %.3f]\n"
    (percentile intercept_samples 0.025)
    (percentile intercept_samples 0.975);

  (* Diagnostics *)
  let e = Norn.ess result.samples in
  Printf.printf "\nESS:         [%.1f, %.1f]\n" (Nx.item [ 0 ] e)
    (Nx.item [ 1 ] e);
  Printf.printf "Accept rate: %.3f\n" result.stats.accept_rate;
  Printf.printf "Step size:   %.4f\n" result.stats.step_size;
  Printf.printf "Divergent:   %d\n" result.stats.num_divergent;

  (* Also demonstrate the configurable API *)
  Printf.printf "\n--- Same model with configurable sample API ---\n";
  let result2 =
    Norn.sample ~n:1000 ~num_warmup:500 log_prob init (fun ~step_size ~metric ->
        Norn.nuts_kernel ~step_size ~metric ())
  in
  let mean2 = Nx.mean ~axes:[ 0 ] result2.samples in
  Printf.printf "Posterior: slope = %.3f, intercept = %.3f\n"
    (Nx.item [ 0 ] mean2) (Nx.item [ 1 ] mean2);
  Printf.printf "Accept rate: %.3f\n" result2.stats.accept_rate
