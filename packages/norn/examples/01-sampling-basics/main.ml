(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Sample from a 2D correlated Gaussian using NUTS.

   Target distribution: N(mu, Sigma) with mu = [2.0; -1.0] Sigma = [[1.0, 0.8],
   [0.8, 2.0]]

   NUTS automatically adapts step size and trajectory length during warmup, so a
   single call to [Norn.nuts] is all you need. *)

let f64 = Nx.float64

(* Target parameters *)
let mu = Nx.create f64 [| 2 |] [| 2.0; -1.0 |]
let sigma_inv = Nx.inv (Nx.create f64 [| 2; 2 |] [| 1.0; 0.8; 0.8; 2.0 |])

(* Log-density of the target (unnormalized). *)
let log_prob x =
  let d = Nx.sub x mu in
  let dt = Nx.reshape [| 1; 2 |] d in
  let mahal = Nx.matmul (Nx.matmul dt sigma_inv) (Nx.reshape [| 2; 1 |] d) in
  Nx.mul_s (Nx.reshape [||] mahal) (-0.5)

let () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let init = Nx.zeros f64 [| 2 |] in
  let result = Norn.nuts ~n:1000 log_prob init in

  (* Sample mean *)
  let sample_mean = Nx.mean ~axes:[ 0 ] result.samples in
  Printf.printf "--- 2D Correlated Gaussian (NUTS, 1000 samples) ---\n\n";
  Printf.printf "True mean:    [%6.3f, %6.3f]\n" (Nx.item [ 0 ] mu)
    (Nx.item [ 1 ] mu);
  Printf.printf "Sample mean:  [%6.3f, %6.3f]\n"
    (Nx.item [ 0 ] sample_mean)
    (Nx.item [ 1 ] sample_mean);

  (* Sample variance *)
  let centered = Nx.sub result.samples sample_mean in
  let n = Float.of_int ((Nx.shape result.samples).(0) - 1) in
  let sample_cov =
    Nx.div_s (Nx.matmul (Nx.matrix_transpose centered) centered) n
  in
  Printf.printf "\nTrue var:     [%6.3f, %6.3f]\n" 1.0 2.0;
  Printf.printf "Sample var:   [%6.3f, %6.3f]\n"
    (Nx.item [ 0; 0 ] sample_cov)
    (Nx.item [ 1; 1 ] sample_cov);
  Printf.printf "True cov:     %6.3f\n" 0.8;
  Printf.printf "Sample cov:   %6.3f\n" (Nx.item [ 0; 1 ] sample_cov);

  (* Diagnostics *)
  let e = Norn.ess result.samples in
  Printf.printf "\nESS:          [%6.1f, %6.1f]\n" (Nx.item [ 0 ] e)
    (Nx.item [ 1 ] e);
  Printf.printf "Accept rate:  %.3f\n" result.stats.accept_rate;
  Printf.printf "Step size:    %.4f\n" result.stats.step_size;
  Printf.printf "Divergent:    %d\n" result.stats.num_divergent
