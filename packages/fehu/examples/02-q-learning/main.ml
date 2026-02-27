(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tabular Q-learning on CartPole-v1.

   Discretizes the continuous 4D observation into bins, learns a Q-table with
   epsilon-greedy exploration and temporal difference updates. Uses Eval.run for
   periodic evaluation. *)

open Fehu

(* Hyperparameters *)

let n_bins = 12
let n_actions = 2
let alpha = 0.1
let gamma = 0.99
let epsilon_start = 1.0
let epsilon_end = 0.01
let epsilon_decay = 2000.0
let n_episodes = 10_000
let eval_interval = 500

(* Sparkline *)

let sparkline values =
  let blocks =
    [|
      "\xe2\x96\x81";
      "\xe2\x96\x82";
      "\xe2\x96\x83";
      "\xe2\x96\x84";
      "\xe2\x96\x85";
      "\xe2\x96\x86";
      "\xe2\x96\x87";
      "\xe2\x96\x88";
    |]
  in
  let lo = Array.fold_left Float.min Float.infinity values in
  let hi = Array.fold_left Float.max Float.neg_infinity values in
  let range = hi -. lo in
  if range < 1e-9 then
    String.concat "" (Array.to_list (Array.map (fun _ -> blocks.(4)) values))
  else
    String.concat ""
      (Array.to_list
         (Array.map
            (fun v ->
              let idx = Float.to_int ((v -. lo) /. range *. 7.0) in
              blocks.(max 0 (min 7 idx)))
            values))

(* Q-table *)

let n_states = n_bins * n_bins * n_bins * n_bins
let q = Array.make (n_states * n_actions) 0.0
let q_get s a = q.((s * n_actions) + a)
let q_set s a v = q.((s * n_actions) + a) <- v

(* Discretize: clip each of the 4 obs dimensions into bins. CartPole obs: [x,
   x_dot, theta, theta_dot] We use generous clip ranges that cover typical
   CartPole trajectories. *)

let clip_ranges = [| (-2.4, 2.4); (-3.0, 3.0); (-0.21, 0.21); (-3.0, 3.0) |]

let discretize obs =
  let arr = (Nx.to_array obs : float array) in
  let bin i =
    let lo, hi = clip_ranges.(i) in
    let v = Float.max lo (Float.min hi arr.(i)) in
    let normalized = (v -. lo) /. (hi -. lo) in
    Float.to_int (normalized *. Float.of_int (n_bins - 1))
    |> max 0
    |> min (n_bins - 1)
  in
  let b0 = bin 0 in
  let b1 = bin 1 in
  let b2 = bin 2 in
  let b3 = bin 3 in
  (b0 * n_bins * n_bins * n_bins) + (b1 * n_bins * n_bins) + (b2 * n_bins) + b3

let best_action s = if q_get s 0 >= q_get s 1 then 0 else 1

(* Training *)

let () =
  Printf.printf "Q-Learning on CartPole-v1\n";
  Printf.printf "==========================\n\n";
  Printf.printf "States: %d bins/dim (%d total), Actions: left/right\n" n_bins
    n_states;
  Printf.printf "alpha = %.2f, gamma = %.2f, episodes = %d\n\n" alpha gamma
    n_episodes;

  Nx.Rng.run ~seed:42 @@ fun () ->
  let sample_uniform () =
    let t = Nx.rand Nx.float32 [| 1 |] in
    (Nx.to_array t : float array).(0)
  in

  let sample_random_action () =
    let t = Nx.randint Nx.int32 ~high:n_actions [| 1 |] 0 in
    Int32.to_int (Nx.to_array t : Int32.t array).(0)
  in

  let env = Fehu_envs.Cartpole.make () in

  let n_evals = n_episodes / eval_interval in
  let reward_history = Array.make n_evals 0.0 in
  let eval_idx = ref 0 in

  Printf.printf "Training...\n\n";

  for episode = 1 to n_episodes do
    let epsilon =
      epsilon_end
      +. (epsilon_start -. epsilon_end)
         *. exp (-.Float.of_int episode /. epsilon_decay)
    in

    let obs, _info = Env.reset env () in
    let state = ref (discretize obs) in
    let done_ = ref false in

    while not !done_ do
      let a =
        if sample_uniform () < epsilon then sample_random_action ()
        else best_action !state
      in
      let s = Env.step env (Space.Discrete.of_int a) in
      let next_state = discretize s.observation in
      let done_flag = s.terminated || s.truncated in

      let bootstrap =
        if done_flag then 0.0
        else Float.max (q_get next_state 0) (q_get next_state 1)
      in
      let target = s.reward +. (gamma *. bootstrap) in
      let old_q = q_get !state a in
      q_set !state a (old_q +. (alpha *. (target -. old_q)));

      state := next_state;
      done_ := done_flag
    done;

    if episode mod eval_interval = 0 then begin
      let greedy_policy obs =
        Space.Discrete.of_int (best_action (discretize obs))
      in
      let stats = Eval.run env ~policy:greedy_policy ~n_episodes:20 () in
      Printf.printf
        "  episode %5d  eps = %.2f  eval: reward = %5.1f +/- %4.1f\n%!" episode
        epsilon stats.mean_reward stats.std_reward;
      reward_history.(!eval_idx) <- stats.mean_reward;
      incr eval_idx
    end
  done;

  Printf.printf "\n  reward: %s\n" (sparkline reward_history);

  (* Final evaluation *)
  Printf.printf "\nFinal evaluation (100 episodes):\n";
  let greedy_policy obs =
    Space.Discrete.of_int (best_action (discretize obs))
  in
  let stats = Eval.run env ~policy:greedy_policy ~n_episodes:100 () in
  Printf.printf "  mean reward: %5.1f +/- %.1f\n" stats.mean_reward
    stats.std_reward;
  Printf.printf "  mean length: %5.1f\n" stats.mean_length;

  if stats.mean_reward >= 195.0 then
    Printf.printf "\nSolved! (mean reward >= 195)\n"
  else Printf.printf "\nNot solved yet (mean reward < 195).\n";

  Env.close env
