(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

let f64 = Nx.float64

(* 2D correlated Gaussian: mean [3, -1], covariance [[1, 0.5], [0.5, 2]] *)
let true_mean = Nx.create f64 [| 2 |] [| 3.0; -1.0 |]
let true_cov = Nx.create f64 [| 2; 2 |] [| 1.0; 0.5; 0.5; 2.0 |]
let cov_inv = Nx.inv true_cov

let log_prob x =
  let d = Nx.sub x true_mean in
  let dt = Nx.reshape [| 1; 2 |] d in
  let mahal = Nx.matmul (Nx.matmul dt cov_inv) (Nx.reshape [| 2; 1 |] d) in
  Nx.mul_s (Nx.reshape [||] mahal) (-0.5)

let check_result msg result =
  is_true
    ~msg:
      (Printf.sprintf "%s: accept rate %.2f > 0.4" msg
         result.Norn.stats.accept_rate)
    (result.stats.accept_rate > 0.4);
  let sample_mean = Nx.mean ~axes:[ 0 ] result.samples in
  for i = 0 to 1 do
    let sm = Nx.item [ i ] sample_mean in
    let tm = Nx.item [ i ] true_mean in
    is_true
      ~msg:(Printf.sprintf "%s: mean[%d]: %.2f ~ %.2f" msg i sm tm)
      (Float.abs (sm -. tm) < 0.5)
  done;
  let centered = Nx.sub result.samples sample_mean in
  let n = Float.of_int ((Nx.shape result.samples).(0) - 1) in
  let sample_cov =
    Nx.div_s (Nx.matmul (Nx.matrix_transpose centered) centered) n
  in
  for i = 0 to 1 do
    let sc = Nx.item [ i; i ] sample_cov in
    let tc = Nx.item [ i; i ] true_cov in
    is_true
      ~msg:(Printf.sprintf "%s: var[%d]: %.2f ~ %.2f (within 60%%)" msg i sc tc)
      (Float.abs (sc -. tc) /. tc < 0.6)
  done

let test_hmc () =
  Nx.Rng.run ~seed:42 (fun () ->
      let init = Nx.zeros f64 [| 2 |] in
      let result =
        Norn.hmc ~step_size:0.1 ~num_leapfrog:20 ~num_warmup:200 ~n:500 log_prob
          init
      in
      check_result "HMC" result)

let test_nuts () =
  Nx.Rng.run ~seed:42 (fun () ->
      let init = Nx.zeros f64 [| 2 |] in
      let result =
        Norn.nuts ~step_size:0.5 ~max_depth:6 ~num_warmup:500 ~n:800 log_prob
          init
      in
      check_result "NUTS" result)

let test_kernel_api () =
  Nx.Rng.run ~seed:42 (fun () ->
      let init = Nx.zeros f64 [| 2 |] in
      let metric = Norn.unit_metric 2 in
      let kernel = Norn.hmc_kernel ~step_size:0.1 ~metric () in
      let state = kernel.init init log_prob in
      is_true ~msg:"init log_density is finite"
        (Float.is_finite state.log_density);
      let state', info = kernel.step state log_prob in
      is_true ~msg:"step produces finite log_density"
        (Float.is_finite state'.log_density);
      is_true ~msg:"acceptance_rate in [0, 1]"
        (info.acceptance_rate >= 0.0 && info.acceptance_rate <= 1.0))

let test_sample_with_kernel () =
  Nx.Rng.run ~seed:42 (fun () ->
      let init = Nx.zeros f64 [| 2 |] in
      let result =
        Norn.sample ~step_size:0.1 ~num_warmup:200 ~n:500 log_prob init
          (fun ~step_size ~metric -> Norn.hmc_kernel ~step_size ~metric ())
      in
      check_result "sample+kernel" result)

let test_diagnostics () =
  Nx.Rng.run ~seed:42 (fun () ->
      let init = Nx.zeros f64 [| 2 |] in
      let chains =
        Array.init 4 (fun i ->
            Nx.Rng.run ~seed:i (fun () ->
                Norn.nuts ~step_size:0.1 ~num_warmup:500 ~n:1000 log_prob init))
      in
      let chain_samples = Array.map (fun r -> r.Norn.samples) chains in
      let r = Norn.rhat chain_samples in
      for d = 0 to 1 do
        let rv = Nx.item [ d ] r in
        is_true ~msg:(Printf.sprintf "rhat[%d]: %.3f < 1.1" d rv) (rv < 1.1)
      done;
      let e = Norn.ess chain_samples.(0) in
      for d = 0 to 1 do
        let ev = Nx.item [ d ] e in
        is_true ~msg:(Printf.sprintf "ess[%d]: %.0f > 50" d ev) (ev > 50.0)
      done)

let () =
  run "Norn"
    [
      test "HMC: 2D Gaussian" test_hmc;
      test "NUTS: 2D Gaussian" test_nuts;
      test "Kernel API" test_kernel_api;
      test "Sample with kernel" test_sample_with_kernel;
      test "Diagnostics" test_diagnostics;
    ]
