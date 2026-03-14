(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

include Internal

let f64 = Nx.float64

(* Integrators *)

let leapfrog kinetic_energy_grad q p g grad_log_prob eps =
  let p = Nx.add p (Nx.mul_s g (eps /. 2.0)) in
  let q = Nx.add q (Nx.mul_s (kinetic_energy_grad p) eps) in
  let lp, g = grad_log_prob q in
  let p = Nx.add p (Nx.mul_s g (eps /. 2.0)) in
  (q, p, lp, g)

let palindromic ~momentum_coeffs ~position_coeffs kinetic_energy_grad q p g
    grad_log_prob eps =
  let n_pos = Array.length position_coeffs in
  let q = ref q in
  let p = ref (Nx.add p (Nx.mul_s g (momentum_coeffs.(0) *. eps))) in
  let lp = ref 0.0 in
  let g = ref g in
  for i = 0 to n_pos - 1 do
    q :=
      Nx.add !q (Nx.mul_s (kinetic_energy_grad !p) (position_coeffs.(i) *. eps));
    let lp', g' = grad_log_prob !q in
    lp := lp';
    g := g';
    p := Nx.add !p (Nx.mul_s !g (momentum_coeffs.(i + 1) *. eps))
  done;
  (!q, !p, !lp, !g)

let mclachlan =
  let l = 0.1932093174209856 in
  palindromic
    ~momentum_coeffs:[| l; 1.0 -. (2.0 *. l); l |]
    ~position_coeffs:[| 0.5; 0.5 |]

let yoshida =
  let cbrt2 = 2.0 ** (1.0 /. 3.0) in
  let w1 = 1.0 /. (2.0 -. cbrt2) in
  let w0 = -.cbrt2 /. (2.0 -. cbrt2) in
  palindromic
    ~momentum_coeffs:
      [| w1 /. 2.0; (w1 +. w0) /. 2.0; (w0 +. w1) /. 2.0; w1 /. 2.0 |]
    ~position_coeffs:[| w1; w0; w1 |]

(* Metrics *)

let euclidean_is_turning scale left_p right_p momentum_sum =
  let rho = Nx.sub momentum_sum (Nx.div_s (Nx.add left_p right_p) 2.0) in
  Nx.item [] (Nx.vdot (scale left_p) rho) <= 0.0
  || Nx.item [] (Nx.vdot (scale right_p) rho) <= 0.0

let unit_metric dim =
  {
    sample_momentum = (fun _dim -> Nx.randn f64 [| dim |]);
    kinetic_energy = (fun p -> 0.5 *. Nx.item [] (Nx.sum (Nx.square p)));
    scale = Fun.id;
    is_turning = euclidean_is_turning Fun.id;
  }

let diagonal_metric inv_mass_diag =
  let mass_diag = Nx.recip inv_mass_diag in
  let sqrt_mass = Nx.sqrt mass_diag in
  let scale v = Nx.mul v inv_mass_diag in
  {
    sample_momentum = (fun dim -> Nx.mul (Nx.randn f64 [| dim |]) sqrt_mass);
    kinetic_energy =
      (fun p -> 0.5 *. Nx.item [] (Nx.sum (Nx.mul (Nx.square p) inv_mass_diag)));
    scale;
    is_turning = euclidean_is_turning scale;
  }

let dense_metric inv_mass_matrix =
  let dim = (Nx.shape inv_mass_matrix).(0) in
  let mass_matrix = Nx.inv inv_mass_matrix in
  let chol = Nx.cholesky mass_matrix in
  let scale v =
    let v_col = Nx.reshape [| dim; 1 |] v in
    Nx.reshape [| dim |] (Nx.matmul inv_mass_matrix v_col)
  in
  {
    sample_momentum =
      (fun _dim ->
        let z = Nx.randn f64 [| dim; 1 |] in
        Nx.reshape [| dim |] (Nx.matmul chol z));
    kinetic_energy =
      (fun p ->
        let p_col = Nx.reshape [| dim; 1 |] p in
        0.5
        *. Nx.item []
             (Nx.matmul
                (Nx.matrix_transpose p_col)
                (Nx.matmul inv_mass_matrix p_col)));
    scale;
    is_turning = euclidean_is_turning scale;
  }

(* Kernels *)

let grad_log_prob log_density_fn q =
  let lp, g = Rune.value_and_grad log_density_fn q in
  (Nx.item [] lp, g)

let init_state position log_density_fn =
  let lp, g = Rune.value_and_grad log_density_fn position in
  { position; log_density = Nx.item [] lp; grad_log_density = g }

let hmc_kernel ?(integrator : integrator = leapfrog) ?(num_leapfrog = 20)
    ~step_size ~(metric : metric) () =
  let step (state : state) log_density_fn =
    let dim = Nx.numel state.position in
    let glp = grad_log_prob log_density_fn in
    let p0 = metric.sample_momentum dim in
    let ke_current = metric.kinetic_energy p0 in
    let q = ref state.position in
    let p = ref p0 in
    let lp = ref state.log_density in
    let g = ref state.grad_log_density in
    for _ = 1 to num_leapfrog do
      let q', p', lp', g' = integrator metric.scale !q !p !g glp step_size in
      q := q';
      p := p';
      lp := lp';
      g := g'
    done;
    let ke_proposed = metric.kinetic_energy !p in
    let delta = !lp -. state.log_density -. (ke_proposed -. ke_current) in
    let log_accept = if Float.is_nan delta then Float.neg_infinity else delta in
    let acceptance_rate = Float.min 1.0 (Float.exp log_accept) in
    let accepted = Float.log (Nx.item [] (Nx.rand f64 [||])) < log_accept in
    let new_state =
      if accepted then
        { position = !q; log_density = !lp; grad_log_density = !g }
      else state
    in
    let info =
      {
        acceptance_rate;
        is_divergent = Float.abs (ke_proposed -. ke_current) > 1000.0;
        energy = -. !lp +. ke_proposed;
        num_integration_steps = num_leapfrog;
      }
    in
    (new_state, info)
  in
  { init = init_state; step }

let nuts_kernel ?(integrator : integrator = leapfrog) ?(max_depth = 10)
    ~step_size ~(metric : metric) () =
  let step state log_density_fn =
    Nuts.step integrator metric step_size max_depth state log_density_fn
  in
  { init = init_state; step }

(* Sampling *)

let metric_of_mass_matrix dim mm =
  match Adapt.mass_matrix_inv_diag mm with
  | None -> unit_metric dim
  | Some inv_mass_diag -> diagonal_metric inv_mass_diag

let sample ?(step_size = 0.01) ?(target_accept = 0.65) ?num_warmup ?report ~n
    log_density_fn init make_kernel =
  let num_warmup = match num_warmup with Some w -> w | None -> n / 2 in
  let dim = Nx.numel init in
  let schedule = Adapt.build_schedule num_warmup in
  let met = ref (unit_metric dim) in
  let kern = ref (make_kernel ~step_size ~metric:!met) in
  let state = ref (!kern.init init log_density_fn) in
  let ss = ref (Adapt.step_size_init ~target_accept step_size) in
  let mm = ref (Adapt.mass_matrix_init dim) in
  for i = 1 to num_warmup do
    let eps = Adapt.step_size_current !ss in
    kern := make_kernel ~step_size:eps ~metric:!met;
    let new_state, info = !kern.step !state log_density_fn in
    state := new_state;
    (match schedule.(i - 1) with
    | Adapt.Fast ->
        ss := Adapt.step_size_update !ss ~acceptance_rate:info.acceptance_rate
    | Adapt.Slow ->
        ss := Adapt.step_size_update !ss ~acceptance_rate:info.acceptance_rate;
        mm := Adapt.mass_matrix_update !mm new_state.position
    | Adapt.Slow_end ->
        ss := Adapt.step_size_update !ss ~acceptance_rate:info.acceptance_rate;
        mm := Adapt.mass_matrix_update !mm new_state.position;
        met := metric_of_mass_matrix dim !mm;
        mm := Adapt.mass_matrix_reset !mm;
        ss := Adapt.step_size_reset !ss);
    match report with
    | Some f -> f ~step:(-(num_warmup - i + 1)) new_state info
    | None -> ()
  done;
  let final_step_size = Adapt.step_size_final !ss in
  kern := make_kernel ~step_size:final_step_size ~metric:!met;
  let samples = Nx.zeros f64 [| n; dim |] in
  let log_densities = Nx.zeros f64 [| n |] in
  let total_accept = ref 0.0 in
  let num_divergent = ref 0 in
  for i = 0 to n - 1 do
    let new_state, info = !kern.step !state log_density_fn in
    state := new_state;
    total_accept := !total_accept +. info.acceptance_rate;
    if info.is_divergent then incr num_divergent;
    Nx.set_slice [ I i ] samples new_state.position;
    Nx.set_item [ i ] new_state.log_density log_densities;
    match report with Some f -> f ~step:i new_state info | None -> ()
  done;
  {
    samples;
    log_densities;
    stats =
      {
        accept_rate = !total_accept /. Float.of_int n;
        step_size = final_step_size;
        num_divergent = !num_divergent;
      };
  }

let hmc ?(step_size = 0.01) ?(target_accept = 0.65) ?num_leapfrog ?num_warmup ~n
    log_prob init =
  sample ~step_size ~target_accept ?num_warmup ~n log_prob init
    (fun ~step_size ~metric ->
      hmc_kernel ?integrator:None ?num_leapfrog ~step_size ~metric ())

let nuts ?(step_size = 0.01) ?(target_accept = 0.80) ?max_depth ?num_warmup ~n
    log_prob init =
  sample ~step_size ~target_accept ?num_warmup ~n log_prob init
    (fun ~step_size ~metric ->
      nuts_kernel ?integrator:None ?max_depth ~step_size ~metric ())

(* Diagnostics *)

let autocorr samples =
  let n = (Nx.shape samples).(0) in
  let dim = (Nx.shape samples).(1) in
  let mean = Nx.mean ~axes:[ 0 ] samples in
  let centered = Nx.sub samples mean in
  let max_lag = n / 2 in
  let acf = Nx.zeros f64 [| max_lag; dim |] in
  for d = 0 to dim - 1 do
    let col = Nx.slice [ A; I d ] centered in
    let v = ref 0.0 in
    for i = 0 to n - 1 do
      let x = Nx.item [ i ] col in
      v := !v +. (x *. x)
    done;
    v := !v /. Float.of_int n;
    for lag = 0 to max_lag - 1 do
      let c = ref 0.0 in
      for i = 0 to n - 1 - lag do
        c := !c +. (Nx.item [ i ] col *. Nx.item [ i + lag ] col)
      done;
      Nx.set_item [ lag; d ] (!c /. (Float.of_int n *. !v)) acf
    done
  done;
  acf

let ess samples =
  let n = (Nx.shape samples).(0) in
  let dim = (Nx.shape samples).(1) in
  let acf = autocorr samples in
  let max_lag = n / 2 in
  let result = Nx.zeros f64 [| dim |] in
  for d = 0 to dim - 1 do
    let tau = ref 1.0 in
    let lag = ref 1 in
    let stop = ref false in
    while !lag < max_lag - 1 && not !stop do
      let rho1 = Nx.item [ !lag; d ] acf in
      let rho2 = Nx.item [ !lag + 1; d ] acf in
      if rho1 +. rho2 < 0.0 then stop := true
      else begin
        tau := !tau +. (2.0 *. rho1);
        incr lag
      end
    done;
    Nx.set_item [ d ] (Float.of_int n /. !tau) result
  done;
  result

let rhat chains =
  let m = Array.length chains in
  let n = (Nx.shape chains.(0)).(0) in
  let dim = (Nx.shape chains.(0)).(1) in
  let half = n / 2 in
  let split_chains = Array.make (2 * m) chains.(0) in
  for i = 0 to m - 1 do
    split_chains.(2 * i) <- Nx.slice [ R (0, half - 1) ] chains.(i);
    split_chains.((2 * i) + 1) <- Nx.slice [ R (half, n - 1) ] chains.(i)
  done;
  let nf = Float.of_int half in
  let mf = Float.of_int (2 * m) in
  let chain_means = Array.map (Nx.mean ~axes:[ 0 ]) split_chains in
  let grand_mean =
    Array.fold_left Nx.add (Nx.zeros f64 [| dim |]) chain_means |> fun s ->
    Nx.div_s s mf
  in
  let b =
    Array.fold_left
      (fun acc cm ->
        let diff = Nx.sub cm grand_mean in
        Nx.add acc (Nx.square diff))
      (Nx.zeros f64 [| dim |]) chain_means
    |> fun s -> Nx.mul_s s (nf /. (mf -. 1.0))
  in
  let w =
    Array.fold_left
      (fun acc chain ->
        let cm = Nx.mean ~axes:[ 0 ] chain in
        let centered = Nx.sub chain cm in
        let s2 =
          Nx.div_s (Nx.sum ~axes:[ 0 ] (Nx.square centered)) (nf -. 1.0)
        in
        Nx.add acc s2)
      (Nx.zeros f64 [| dim |]) split_chains
    |> fun s -> Nx.div_s s mf
  in
  let var_hat = Nx.add (Nx.mul_s w ((nf -. 1.0) /. nf)) (Nx.div_s b nf) in
  Nx.sqrt (Nx.div var_hat w)
