(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* DQN on CartPole-v1.

   Deep Q-Network with experience replay and a target network. Epsilon-greedy
   exploration decays linearly. The target network is hard-copied every
   target_update_interval steps. *)

open Fehu
open Kaun

(* Hyperparameters *)

let buffer_capacity = 50_000
let batch_size = 64
let gamma = 0.99
let lr = 5e-4
let epsilon_start = 1.0
let epsilon_end = 0.05
let epsilon_decay_steps = 10_000
let target_update_interval = 250
let learning_starts = 1000
let n_steps = 50_000
let eval_interval = 2000
let eval_episodes = 20

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

(* Network *)

let q_network =
  Layer.sequential
    [
      Layer.linear ~in_features:4 ~out_features:128 ();
      Layer.relu ();
      Layer.linear ~in_features:128 ~out_features:128 ();
      Layer.relu ();
      Layer.linear ~in_features:128 ~out_features:2 ();
    ]

(* Forward pass: obs [batch; 4] -> q_values [batch; 2] *)

let forward params net_state obs =
  let vars = Layer.make_vars ~params ~state:net_state ~dtype:Rune.float32 in
  fst (Layer.apply q_network vars ~training:false obs)

(* Epsilon schedule: linear decay *)

let epsilon step =
  let t =
    Float.min 1.0 (Float.of_int step /. Float.of_int epsilon_decay_steps)
  in
  epsilon_start +. (t *. (epsilon_end -. epsilon_start))

(* Copy parameters for the target network *)

let copy_params params = Ptree.map { run = (fun t -> Rune.copy t) } params

(* Main *)

let () =
  Printf.printf "DQN on CartPole-v1\n";
  Printf.printf "===================\n\n";
  Printf.printf
    "Network: Linear(4 -> 128) -> ReLU -> Linear(128 -> 128) -> ReLU -> \
     Linear(128 -> 2)\n";
  Printf.printf "Buffer: %d, batch: %d, gamma = %.2f, lr = %.4f\n"
    buffer_capacity batch_size gamma lr;
  Printf.printf
    "Epsilon: %.2f -> %.2f over %d steps, target update every %d steps\n\n"
    epsilon_start epsilon_end epsilon_decay_steps target_update_interval;

  let env = Fehu_envs.Cartpole.make ~rng:(Rune.Rng.key 0) () in

  (* Initialize network *)
  let vars = Layer.init q_network ~rngs:(Rune.Rng.key 42) ~dtype:Rune.float32 in
  let params = ref (Layer.params vars) in
  let net_state = Layer.state vars in
  let target_params = ref (copy_params !params) in

  Printf.printf "Parameters: %d\n\n" (Ptree.count_parameters !params);

  (* Optimizer *)
  let algo = Optim.adam ~lr:(Optim.Schedule.constant lr) () in
  let opt_state = ref (Optim.init algo !params) in

  (* Replay buffer *)
  let buffer = Buffer.create ~capacity:buffer_capacity in

  (* RNG for exploration *)
  let agent_rng = ref (Rune.Rng.key 1) in
  let take_rng () =
    let keys = Rune.Rng.split !agent_rng in
    agent_rng := keys.(0);
    keys.(1)
  in

  let sample_uniform () =
    let t = Rune.rand Rune.float32 ~key:(take_rng ()) [| 1 |] in
    (Rune.to_array t : float array).(0)
  in

  (* Epsilon-greedy action selection *)
  let select_action obs eps =
    if sample_uniform () < eps then
      fst (Space.sample (Env.action_space env) ~rng:(take_rng ()))
    else begin
      let obs_batch = Rune.reshape [| 1; 4 |] obs in
      let q_values =
        Rune.no_grad (fun () -> forward !params net_state obs_batch)
      in
      let action_idx =
        Rune.argmax q_values ~axis:(-1) ~keepdims:false |> Rune.cast Rune.int32
      in
      Rune.reshape [||] action_idx
    end
  in

  (* Greedy policy for evaluation *)
  let greedy_policy obs =
    let obs_batch = Rune.reshape [| 1; 4 |] obs in
    let q_values =
      Rune.no_grad (fun () -> forward !params net_state obs_batch)
    in
    let action_idx =
      Rune.argmax q_values ~axis:(-1) ~keepdims:false |> Rune.cast Rune.int32
    in
    Rune.reshape [||] action_idx
  in

  (* Training step *)
  let train_step rng_key =
    let (obs_arr, act_arr, rew_arr, next_obs_arr, term_arr, trunc_arr), _rng' =
      Buffer.sample_arrays buffer ~rng:rng_key ~batch_size
    in
    let n = Array.length obs_arr in

    (* Stack into batch tensors *)
    let obs_batch = Rune.stack (Array.to_list obs_arr) in
    let next_obs_batch = Rune.stack (Array.to_list next_obs_arr) in
    let actions_batch =
      Rune.stack
        (Array.to_list (Array.map (fun a -> Rune.reshape [| 1 |] a) act_arr))
    in
    let rewards_t = Rune.create Rune.float32 [| n |] rew_arr in

    (* Done mask: 1.0 if not done, 0.0 if done *)
    let done_mask =
      Array.init n (fun i -> if term_arr.(i) || trunc_arr.(i) then 0.0 else 1.0)
    in
    let done_mask_t = Rune.create Rune.float32 [| n |] done_mask in

    (* Compute TD target with target network (no gradient) *)
    let td_target =
      Rune.no_grad (fun () ->
          let target_q = forward !target_params net_state next_obs_batch in
          let max_q = Rune.max target_q ~axes:[ 1 ] ~keepdims:false in
          Rune.add rewards_t
            (Rune.mul
               (Rune.scalar Rune.float32 gamma)
               (Rune.mul max_q done_mask_t)))
    in
    let td_target = Rune.detach td_target in

    (* Loss: MSE between predicted Q and TD target *)
    let loss_fn p =
      let q_values = forward p net_state obs_batch in
      let q_selected = Rune.take_along_axis ~axis:1 actions_batch q_values in
      let q_selected = Rune.reshape [| n |] q_selected in
      let diff = Rune.sub q_selected td_target in
      Rune.mean (Rune.mul diff diff)
    in

    let loss, grads = Grad.value_and_grad loss_fn !params in
    let new_params, new_opt_state =
      Optim.update algo !opt_state !params grads
    in
    params := new_params;
    opt_state := new_opt_state;
    Rune.item [] loss
  in

  (* Main training loop *)
  Printf.printf "Filling buffer (%d steps)...\n\n" learning_starts;

  let obs = ref (fst (Env.reset env ())) in
  let last_loss = ref 0.0 in

  let n_evals = n_steps / eval_interval in
  let reward_history = Array.make n_evals 0.0 in
  let eval_idx = ref 0 in

  Printf.printf "Training...\n\n";

  for step = 1 to n_steps do
    let eps = epsilon step in
    let action = select_action !obs eps in
    let s = Env.step env action in

    Buffer.add buffer
      {
        observation = !obs;
        action;
        reward = s.reward;
        next_observation = s.observation;
        terminated = s.terminated;
        truncated = s.truncated;
      };

    if s.terminated || s.truncated then obs := fst (Env.reset env ())
    else obs := s.observation;

    (* Train *)
    if step >= learning_starts then begin
      last_loss := train_step (take_rng ());

      (* Update target network *)
      if step mod target_update_interval = 0 then
        target_params := copy_params !params
    end;

    (* Evaluate periodically *)
    if step mod eval_interval = 0 then begin
      let stats =
        Eval.run env ~policy:greedy_policy ~n_episodes:eval_episodes ()
      in
      Printf.printf
        "  step %5d  epsilon = %.2f  loss = %6.4f  eval: reward = %5.1f +/- \
         %4.1f\n\
         %!"
        step eps !last_loss stats.mean_reward stats.std_reward;
      reward_history.(!eval_idx) <- stats.mean_reward;
      incr eval_idx;
      (* Eval.run leaves the env in a done state; reset for training *)
      obs := fst (Env.reset env ())
    end
  done;

  Printf.printf "\n  reward: %s\n" (sparkline reward_history);

  (* Final evaluation *)
  Printf.printf "\nFinal evaluation (%d episodes):\n" 50;
  let stats = Eval.run env ~policy:greedy_policy ~n_episodes:50 () in
  Printf.printf "  mean reward: %5.1f +/- %.1f\n" stats.mean_reward
    stats.std_reward;
  Printf.printf "  mean length: %5.1f\n" stats.mean_length;

  if stats.mean_reward >= 475.0 then
    Printf.printf "\nSolved! (mean reward >= 475)\n"
  else Printf.printf "\nNot solved yet (mean reward < 475).\n";

  Env.close env
