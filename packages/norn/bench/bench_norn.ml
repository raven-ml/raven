(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let f64 = Nx.float64

let make_log_prob dim =
  let mean = Nx.zeros f64 [| dim |] in
  fun x ->
    let d = Nx.sub x mean in
    Nx.mul_s (Nx.sum (Nx.mul d d)) (-0.5)

let hmc_benches () =
  let cases = [ ("2D_3lf", 2, 3); ("5D_3lf", 5, 3) ] in
  List.map
    (fun (label, dim, num_leapfrog) ->
      let log_prob = make_log_prob dim in
      let metric = Norn.unit_metric dim in
      let kernel =
        Norn.hmc_kernel ~num_leapfrog ~step_size:0.1 ~metric ()
      in
      let init = Nx.zeros f64 [| dim |] in
      let state = ref (kernel.init init log_prob) in
      Thumper.bench (Printf.sprintf "HMC/%s" label) (fun () ->
          let new_state, info = kernel.step !state log_prob in
          state := new_state;
          Thumper.consume info.Norn.acceptance_rate))
    cases

let nuts_benches () =
  let cases = [ ("2D_d3", 2, 3); ("5D_d3", 5, 3) ] in
  List.map
    (fun (label, dim, max_depth) ->
      let log_prob = make_log_prob dim in
      let metric = Norn.unit_metric dim in
      let kernel =
        Norn.nuts_kernel ~max_depth ~step_size:0.1 ~metric ()
      in
      let init = Nx.zeros f64 [| dim |] in
      let state = ref (kernel.init init log_prob) in
      Thumper.bench (Printf.sprintf "NUTS/%s" label) (fun () ->
          let new_state, info = kernel.step !state log_prob in
          state := new_state;
          Thumper.consume info.Norn.acceptance_rate))
    cases

let ess_benches () =
  let cases = [ ("2D_n100", 2, 100); ("5D_n100", 5, 100) ] in
  List.map
    (fun (label, dim, n) ->
      let samples = Nx.randn f64 [| n; dim |] in
      Thumper.bench (Printf.sprintf "ESS/%s" label) (fun () ->
          Thumper.consume (Norn.ess samples)))
    cases

let rhat_benches () =
  let cases = [ ("2D_n100", 2, 100); ("5D_n100", 5, 100) ] in
  List.map
    (fun (label, dim, n) ->
      let chains = Array.init 4 (fun _ -> Nx.randn f64 [| n; dim |]) in
      Thumper.bench (Printf.sprintf "Rhat/%s" label) (fun () ->
          Thumper.consume (Norn.rhat chains)))
    cases

let () =
  Nx.Rng.run ~seed:42 (fun () ->
      Thumper.run "norn"
        [
          Thumper.group "HMC" (hmc_benches ());
          Thumper.group "NUTS" (nuts_benches ());
          Thumper.group "ESS" (ess_benches ());
          Thumper.group "Rhat" (rhat_benches ());
        ])
