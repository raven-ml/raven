(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Multi-chain convergence diagnostics.

   Run 4 chains on a 3D target distribution, then compute ESS and R-hat to
   assess whether the chains have converged and mixed well.

   Guidelines: - ESS > 100 per parameter for reliable estimates - R-hat < 1.01
   indicates convergence across chains *)

let f64 = Nx.float64

(* Target: 3D Gaussian with different scales per dimension. mu = [1, -2, 0.5]
   sigma = diag([1, 4, 0.25]) *)
let mu = Nx.create f64 [| 3 |] [| 1.0; -2.0; 0.5 |]
let inv_var = Nx.create f64 [| 3 |] [| 1.0; 0.25; 4.0 |]
(* 1/sigma^2 for each dim *)

let log_prob x =
  let d = Nx.sub x mu in
  Nx.mul_s (Nx.sum (Nx.mul (Nx.square d) inv_var)) (-0.5)

let dim = 3
let n_chains = 4
let n_samples = 1000
let param_names = [| "x0"; "x1"; "x2" |]

let () =
  Printf.printf "--- Multi-Chain Diagnostics (%d chains x %d samples) ---\n\n"
    n_chains n_samples;

  (* Run chains with different seeds *)
  let chains =
    Array.init n_chains (fun i ->
        Nx.Rng.run ~seed:(i + 1) @@ fun () ->
        let init = Nx.zeros f64 [| dim |] in
        Norn.nuts ~n:n_samples ~num_warmup:500 log_prob init)
  in

  (* Per-chain summary *)
  Printf.printf "Per-chain summary:\n";
  Printf.printf "  %-8s  %-12s  %-12s  %-8s\n" "Chain" "Accept Rate" "Step Size"
    "Diverg.";
  Array.iteri
    (fun i r ->
      Printf.printf "  %-8d  %-12.3f  %-12.4f  %-8d\n" (i + 1)
        r.Norn.stats.accept_rate r.stats.step_size r.stats.num_divergent)
    chains;

  (* Per-chain ESS *)
  Printf.printf "\nEffective Sample Size (ESS) per chain:\n";
  Printf.printf "  %-8s" "Chain";
  Array.iter (fun name -> Printf.printf "  %-8s" name) param_names;
  Printf.printf "\n";
  Array.iteri
    (fun i r ->
      let e = Norn.ess r.Norn.samples in
      Printf.printf "  %-8d" (i + 1);
      for d = 0 to dim - 1 do
        Printf.printf "  %-8.1f" (Nx.item [ d ] e)
      done;
      Printf.printf "\n")
    chains;

  (* R-hat across chains *)
  let chain_samples = Array.map (fun r -> r.Norn.samples) chains in
  let r = Norn.rhat chain_samples in
  Printf.printf "\nSplit R-hat (target: < 1.01):\n";
  for d = 0 to dim - 1 do
    let rv = Nx.item [ d ] r in
    let status = if rv < 1.01 then "OK" else "WARNING" in
    Printf.printf "  %s: %.4f  [%s]\n" param_names.(d) rv status
  done;

  (* Grand summary *)
  let all_converged = ref true in
  for d = 0 to dim - 1 do
    if Nx.item [ d ] r >= 1.01 then all_converged := false
  done;
  Printf.printf "\nConvergence: %s\n"
    (if !all_converged then "All parameters converged (R-hat < 1.01)"
     else
       "Some parameters have not converged -- increase samples or check model");

  (* Pooled posterior summary *)
  Printf.printf "\nPooled posterior (all chains):\n";
  Printf.printf "  %-8s  %-10s  %-10s  %-10s\n" "Param" "True" "Mean" "Std";
  let all_samples =
    Nx.concatenate ~axis:0
      (Array.to_list (Array.map (fun r -> r.Norn.samples) chains))
  in
  let pooled_mean = Nx.mean ~axes:[ 0 ] all_samples in
  let pooled_centered = Nx.sub all_samples pooled_mean in
  let nf = Float.of_int ((Nx.shape all_samples).(0) - 1) in
  let pooled_var =
    Nx.div_s (Nx.sum ~axes:[ 0 ] (Nx.square pooled_centered)) nf
  in
  let pooled_std = Nx.sqrt pooled_var in
  for d = 0 to dim - 1 do
    Printf.printf "  %-8s  %-10.3f  %-10.3f  %-10.3f\n" param_names.(d)
      (Nx.item [ d ] mu)
      (Nx.item [ d ] pooled_mean)
      (Nx.item [ d ] pooled_std)
  done
