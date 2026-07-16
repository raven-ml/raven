(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* REINFORCE on CartPole-v1.

   Policy gradient with a small neural network. Collects rollouts, computes
   discounted returns, and updates the policy by maximizing the expected return
   weighted by log-probabilities. *)

open Fehu
open Kaun

(* Hyperparameters *)

let gamma = 0.99
let lr = 1e-3
let n_steps = 2048
let n_updates = 250
let eval_interval = 10
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

(* Network: Linear(4 -> 64) -> ReLU -> Linear(64 -> 2), as a plain record of
   Linear layers with hand-written traversals (the Nx.Ptree.S contract). *)

module Policy = struct
  type t = { l1 : Linear.t; l2 : Linear.t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { l1; l2 } =
    { l1 = Linear.map f l1; l2 = Linear.map f l2 }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { l1 = Linear.map2 f p.l1 q.l1; l2 = Linear.map2 f p.l2 q.l2 }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { l1; l2 } =
    Linear.iter f l1;
    Linear.iter f l2

  (* Forward pass: obs [batch; 4] -> logits [batch; 2] *)
  let apply p obs = Linear.apply p.l2 (Fn.relu (Linear.apply p.l1 obs))
end

let count_parameters params =
  let n = ref 0 in
  Policy.iter (fun t -> n := !n + Nx.numel t) params;
  !n

(* Main *)

let () =
  Printf.printf "REINFORCE on CartPole-v1\n";
  Printf.printf "=========================\n\n";
  Printf.printf "Network: Linear(4 -> 64) -> ReLU -> Linear(64 -> 2)\n";
  Printf.printf "Rollout: %d steps/update, gamma = %.2f, lr = %.4f\n\n" n_steps
    gamma lr;

  Nx.Rng.run ~seed:42 @@ fun () ->
  let env = Fehu_envs.Cartpole.make () in

  (* Initialize network *)
  let params =
    ref
      {
        Policy.l1 = Linear.init ~inputs:4 ~outputs:64;
        l2 = Linear.init ~inputs:64 ~outputs:2;
      }
  in

  Printf.printf "Parameters: %d\n\n" (count_parameters !params);

  (* Optimizer *)
  let opt_state = ref (Vega.adam_init (module Policy) !params) in

  let policy obs =
    let obs_batch = Nx.reshape [| 1; 4 |] obs in
    let logits = Rune.no_grad (fun () -> Policy.apply !params obs_batch) in
    let action_idx = Nx.categorical logits in
    let action = Nx.reshape [||] action_idx in
    let log_probs = Nx.log_softmax logits in
    let action_1 = Nx.reshape [| 1; 1 |] action_idx in
    let log_prob = Nx.take_along_axis ~axis:1 ~indices:action_1 log_probs in
    let lp = Nx.item [ 0; 0 ] log_prob in
    (action, Some lp, None)
  in

  (* Greedy policy for evaluation *)
  let greedy_policy obs =
    let obs_batch = Nx.reshape [| 1; 4 |] obs in
    let logits = Rune.no_grad (fun () -> Policy.apply !params obs_batch) in
    let action_idx =
      Nx.argmax logits ~axis:(-1) ~keepdims:false |> Nx.cast Nx.int32
    in
    Nx.reshape [||] action_idx
  in

  (* Training loop *)
  Printf.printf "Training...\n\n";

  let n_evals = n_updates / eval_interval in
  let reward_history = Array.make n_evals 0.0 in
  let eval_idx = ref 0 in

  for update = 1 to n_updates do
    (* Collect rollout *)
    let traj = Collect.rollout env ~policy ~n_steps in
    let n = Collect.length traj in

    (* Compute discounted returns and normalize *)
    let returns =
      Gae.returns ~rewards:traj.rewards ~terminated:traj.terminated
        ~truncated:traj.truncated ~gamma
    in
    let returns = Gae.normalize returns in

    (* Stack observations and actions into batch tensors *)
    let obs_batch = Nx.stack (Array.to_list traj.observations) in
    let actions_batch =
      Nx.stack
        (Array.to_list (Array.map (fun a -> Nx.reshape [| 1 |] a) traj.actions))
    in
    let returns_t = Nx.create Nx.float32 [| n |] returns in

    (* Policy gradient loss *)
    let loss_fn p =
      let logits = Policy.apply p obs_batch in
      let log_probs = Nx.log_softmax logits in
      let action_log_probs =
        Nx.take_along_axis ~axis:1 ~indices:actions_batch log_probs
      in
      let action_log_probs = Nx.reshape [| n |] action_log_probs in
      let weighted = Nx.mul action_log_probs returns_t in
      Nx.neg (Nx.mean weighted)
    in

    let loss, grads = Rune.value_and_grad (module Policy) loss_fn !params in
    let new_params, new_opt_state =
      Vega.adam_step (module Policy) ~lr !opt_state ~params:!params ~grads
    in
    params := new_params;
    opt_state := new_opt_state;

    (* Evaluate periodically *)
    if update mod eval_interval = 0 then begin
      let stats =
        Eval.run env ~policy:greedy_policy ~n_episodes:eval_episodes ()
      in
      Printf.printf
        "  update %3d  loss = %6.3f  eval: reward = %5.1f +/- %4.1f\n%!" update
        (Nx.item [] loss) stats.mean_reward stats.std_reward;
      reward_history.(!eval_idx) <- stats.mean_reward;
      incr eval_idx
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
