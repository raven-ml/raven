(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* No-U-Turn Sampler (Hoffman & Gelman 2014). Recursive binary tree expansion
   with multinomial sampling and generalised U-turn detection (Betancourt 2013).
   Follows BlackJAX's implementation strictly: - inner recursion:
   progressive_uniform_sampling (multinomial) - outer expansion:
   progressive_biased_sampling - U-turn: rho = m_sum - (m_right + m_left)/2,
   dot(v, rho) <= 0 *)

let f64 = Nx.float64

(* Proposal: sampled state and acceptance statistics. *)

type proposal = {
  q : Nx.float64_t;
  lp : float;
  g : Nx.float64_t;
  energy : float;
  weight : float;
  sum_log_p_accept : float;
}

(* Trajectory: endpoints and accumulated momentum. *)

type trajectory = {
  left_q : Nx.float64_t;
  left_p : Nx.float64_t;
  left_g : Nx.float64_t;
  right_q : Nx.float64_t;
  right_p : Nx.float64_t;
  right_g : Nx.float64_t;
  momentum_sum : Nx.float64_t;
  num_states : int;
}

let log_add_exp a b =
  if a = Float.neg_infinity then b
  else if b = Float.neg_infinity then a
  else if a >= b then a +. Float.log (1.0 +. Float.exp (b -. a))
  else b +. Float.log (1.0 +. Float.exp (a -. b))

(* One leapfrog step → leaf proposal + trajectory. *)
let build_leaf (integrator : Internal.integrator) (metric : Internal.metric)
    direction step_size grad_log_prob q p g initial_energy =
  let eps = Float.of_int direction *. step_size in
  let q', p', lp', g' = integrator metric.scale q p g grad_log_prob eps in
  let energy = -.lp' +. metric.kinetic_energy p' in
  let delta = initial_energy -. energy in
  let delta = if Float.is_nan delta then Float.neg_infinity else delta in
  let proposal =
    {
      q = q';
      lp = lp';
      g = g';
      energy;
      weight = delta;
      sum_log_p_accept = Float.min delta 0.0;
    }
  in
  let trajectory =
    {
      left_q = q';
      left_p = p';
      left_g = g';
      right_q = q';
      right_p = p';
      right_g = g';
      momentum_sum = p';
      num_states = 1;
    }
  in
  let is_diverging = delta < -1000.0 in
  (proposal, trajectory, is_diverging)

(* progressive_uniform_sampling: multinomial for inner tree building. *)
let uniform_sample prop new_prop =
  let p_accept = 1.0 /. (1.0 +. Float.exp (prop.weight -. new_prop.weight)) in
  let u = Nx.item [] (Nx.rand f64 [||]) in
  let weight = log_add_exp prop.weight new_prop.weight in
  let sum_log_p_accept =
    log_add_exp prop.sum_log_p_accept new_prop.sum_log_p_accept
  in
  if u < p_accept then { new_prop with weight; sum_log_p_accept }
  else { prop with weight; sum_log_p_accept }

(* progressive_biased_sampling: for outer tree doubling. *)
let biased_sample prop new_prop =
  let p_accept = Float.min 1.0 (Float.exp (new_prop.weight -. prop.weight)) in
  let u = Nx.item [] (Nx.rand f64 [||]) in
  let weight = log_add_exp prop.weight new_prop.weight in
  let sum_log_p_accept =
    log_add_exp prop.sum_log_p_accept new_prop.sum_log_p_accept
  in
  if u < p_accept then { new_prop with weight; sum_log_p_accept }
  else { prop with weight; sum_log_p_accept }

let merge_trajectories direction traj new_traj =
  let l, r = if direction > 0 then (traj, new_traj) else (new_traj, traj) in
  {
    left_q = l.left_q;
    left_p = l.left_p;
    left_g = l.left_g;
    right_q = r.right_q;
    right_p = r.right_p;
    right_g = r.right_g;
    momentum_sum = Nx.add l.momentum_sum r.momentum_sum;
    num_states = l.num_states + r.num_states;
  }

(* Recursive tree building — buildtree_integrate from BlackJAX
   trajectory.py:dynamic_recursive_integration. *)
let rec build_tree integrator metric step_size grad_log_prob q p g depth
    direction initial_energy =
  if depth = 0 then
    let prop, traj, is_div =
      build_leaf integrator metric direction step_size grad_log_prob q p g
        initial_energy
    in
    (prop, traj, is_div, false)
  else
    let half = depth - 1 in
    let prop, traj, is_div, is_turn =
      build_tree integrator metric step_size grad_log_prob q p g half direction
        initial_energy
    in
    if is_div || is_turn then (prop, traj, is_div, is_turn)
    else
      let q', p', g' =
        if direction > 0 then (traj.right_q, traj.right_p, traj.right_g)
        else (traj.left_q, traj.left_p, traj.left_g)
      in
      let new_prop, new_traj, new_div, new_turn =
        build_tree integrator metric step_size grad_log_prob q' p' g' half
          direction initial_energy
      in
      let merged = merge_trajectories direction traj new_traj in
      if new_turn then
        (* Second half turning: keep old proposal state/weight, accumulate
           sum_log_p_accept for consistent acceptance_rate. *)
        let slpa =
          log_add_exp prop.sum_log_p_accept new_prop.sum_log_p_accept
        in
        ({ prop with sum_log_p_accept = slpa }, merged, new_div, true)
      else
        (* Check U-turn on merged trajectory *)
        let turning =
          metric.is_turning merged.left_p merged.right_p merged.momentum_sum
        in
        (* Always sample when second half is not turning *)
        let sampled = uniform_sample prop new_prop in
        (sampled, merged, new_div, turning)

(* Outer expansion loop — dynamic_multiplicative_expansion from BlackJAX
   trajectory.py. *)
let step (integrator : Internal.integrator) (metric : Internal.metric) step_size
    max_depth (state : Internal.state) log_density_fn =
  let grad_log_prob q =
    let lp, g = Rune.value_and_grad log_density_fn q in
    (Nx.item [] lp, g)
  in
  let dim = Nx.numel state.position in
  let p0 = metric.sample_momentum dim in
  let ke0 = metric.kinetic_energy p0 in
  let initial_energy = -.state.log_density +. ke0 in
  let proposal =
    ref
      {
        q = state.position;
        lp = state.log_density;
        g = state.grad_log_density;
        energy = initial_energy;
        weight = 0.0;
        sum_log_p_accept = Float.neg_infinity;
      }
  in
  let trajectory =
    ref
      {
        left_q = state.position;
        left_p = p0;
        left_g = state.grad_log_density;
        right_q = state.position;
        right_p = p0;
        right_g = state.grad_log_density;
        momentum_sum = p0;
        num_states = 0;
      }
  in
  let depth = ref 0 in
  let diverging = ref false in
  let turning = ref false in
  while !depth < max_depth && (not !diverging) && not !turning do
    let direction = if Nx.item [] (Nx.rand f64 [||]) < 0.5 then -1 else 1 in
    let q, p, g =
      if direction > 0 then
        (!trajectory.right_q, !trajectory.right_p, !trajectory.right_g)
      else (!trajectory.left_q, !trajectory.left_p, !trajectory.left_g)
    in
    let sub_prop, sub_traj, sub_div, sub_turn =
      build_tree integrator metric step_size grad_log_prob q p g !depth
        direction initial_energy
    in
    (* Update proposal: biased sampling unless subtree diverged or turned *)
    if sub_div || sub_turn then
      proposal :=
        {
          !proposal with
          sum_log_p_accept =
            log_add_exp !proposal.sum_log_p_accept sub_prop.sum_log_p_accept;
        }
    else proposal := biased_sample !proposal sub_prop;
    (* Always merge trajectory *)
    trajectory := merge_trajectories direction !trajectory sub_traj;
    (* Check U-turn on full trajectory *)
    let full_turn =
      metric.is_turning !trajectory.left_p !trajectory.right_p
        !trajectory.momentum_sum
    in
    diverging := sub_div;
    turning := sub_turn || full_turn;
    incr depth
  done;
  let p = !proposal in
  let t = !trajectory in
  let new_state : Internal.state =
    { position = p.q; log_density = p.lp; grad_log_density = p.g }
  in
  let n_states = Float.of_int (max 1 t.num_states) in
  let acceptance_rate = Float.exp p.sum_log_p_accept /. n_states in
  let info : Internal.info =
    {
      acceptance_rate;
      is_divergent = !diverging;
      energy = initial_energy;
      num_integration_steps = t.num_states;
    }
  in
  (new_state, info)
