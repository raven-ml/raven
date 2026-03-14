(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Step-size adaptation via dual averaging (Nesterov 2009). *)

let f64 = Nx.float64

type step_size = {
  target_accept : float;
  mu : float;
  log_eps : float;
  log_eps_bar : float;
  h_bar : float;
  count : int;
}

let step_size_init ?(target_accept = 0.65) eps =
  {
    target_accept;
    mu = Float.log (10.0 *. eps);
    log_eps = Float.log eps;
    log_eps_bar = 0.0;
    h_bar = 0.0;
    count = 0;
  }

let step_size_update ss ~acceptance_rate =
  let gamma = 0.05 in
  let t0 = 10.0 in
  let kappa = 0.75 in
  let m = Float.of_int (ss.count + 1) in
  let w = 1.0 /. (m +. t0) in
  let h_bar =
    ((1.0 -. w) *. ss.h_bar) +. (w *. (ss.target_accept -. acceptance_rate))
  in
  let log_eps = ss.mu -. (Float.sqrt m /. gamma *. h_bar) in
  let m_pow = m ** -.kappa in
  (* BlackJAX: log_x_avg uses PREVIOUS log_x, not the newly computed one. *)
  let log_eps_bar =
    (m_pow *. ss.log_eps) +. ((1.0 -. m_pow) *. ss.log_eps_bar)
  in
  { ss with h_bar; log_eps; log_eps_bar; count = ss.count + 1 }

let step_size_current ss = Float.exp ss.log_eps
let step_size_final ss = Float.exp ss.log_eps_bar

(* Mass-matrix adaptation via Welford's online algorithm. *)

type mass_matrix = {
  dim : int;
  count : int;
  mean : Nx.float64_t;
  m2 : Nx.float64_t;
}

let mass_matrix_init dim =
  { dim; count = 0; mean = Nx.zeros f64 [| dim |]; m2 = Nx.zeros f64 [| dim |] }

let mass_matrix_update mm position =
  let count = mm.count + 1 in
  let delta = Nx.sub position mm.mean in
  let mean = Nx.add mm.mean (Nx.div_s delta (Float.of_int count)) in
  let delta2 = Nx.sub position mean in
  let m2 = Nx.add mm.m2 (Nx.mul delta delta2) in
  { mm with count; mean; m2 }

let mass_matrix_inv_diag mm =
  if mm.count < 2 then None
  else
    let n = Float.of_int mm.count in
    let variance = Nx.div_s mm.m2 (n -. 1.0) in
    let w = n /. (n +. 5.0) in
    let shrinkage = 1e-3 *. 5.0 /. (n +. 5.0) in
    Some (Nx.add_s (Nx.mul_s variance w) shrinkage)

let mass_matrix_reset mm =
  {
    mm with
    count = 0;
    mean = Nx.zeros f64 [| mm.dim |];
    m2 = Nx.zeros f64 [| mm.dim |];
  }

let step_size_reset ss =
  step_size_init ~target_accept:ss.target_accept (step_size_final ss)

(* Window adaptation schedule (Stan warmup).

   Three phases: - Fast (initial buffer): adapt step size only. - Slow (doubling
   windows): adapt step size + mass matrix. At each window boundary the mass
   matrix is finalized (regularized) and both estimators are reset. - Fast
   (final buffer): adapt step size only with the final mass matrix. *)

type warmup_action = Fast | Slow | Slow_end

let build_schedule num_warmup =
  if num_warmup < 20 then Array.make num_warmup Fast
  else
    let initial_buffer, final_buffer, first_window =
      if 75 + 50 + 25 > num_warmup then
        let ib = Float.to_int (0.15 *. Float.of_int num_warmup) in
        let fb = Float.to_int (0.10 *. Float.of_int num_warmup) in
        (ib, fb, num_warmup - ib - fb)
      else (75, 50, 25)
    in
    let schedule = Array.make num_warmup Fast in
    let slow_end_pos = num_warmup - final_buffer in
    let pos = ref initial_buffer in
    let window_size = ref first_window in
    while !pos < slow_end_pos do
      let end_pos =
        if !pos + (3 * !window_size) <= slow_end_pos then !pos + !window_size
        else slow_end_pos
      in
      for j = !pos to end_pos - 1 do
        schedule.(j) <- (if j = end_pos - 1 then Slow_end else Slow)
      done;
      pos := end_pos;
      window_size := !window_size * 2
    done;
    schedule
