(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Policy gradient training for RandomWalk using Fehu Algorithms *)

open Fehu
open Kaun

(** Create a simple policy network *)
let create_policy_network () =
  Layer.sequential
    [
      Layer.linear ~in_features:1 ~out_features:32 ();
      Layer.tanh ();
      Layer.linear ~in_features:32 ~out_features:2 ();
    ]

let run_policy_gradient () =
  Printf.printf "Training policy gradient on RandomWalk environment...\n";

  (* Create environment *)
  let env = Fehu_envs.Random_walk.make ~rng:(Rune.Rng.key 2) () in

  (* Create policy network *)
  let policy_net = create_policy_network () in

  (* Train REINFORCE *)
  let rng = Rune.Rng.key 42 in
  let config =
    {
      Fehu_algorithms.Reinforce.default_config with
      learning_rate = 0.001;
      gamma = 0.95;
      reward_scale = 0.1;
      entropy_coef = 0.01;
      max_episode_steps = 200;
    }
  in
  let params, _state =
    Fehu_algorithms.Reinforce.train ~env ~policy_network:policy_net ~rng ~config
      ~total_timesteps:50_000
      ~callback:(fun metrics ->
        if metrics.total_episodes > 0 && metrics.total_episodes mod 50 = 0 then
          Printf.printf "Episode %d: Return = %.2f, Length = %d\n%!"
            metrics.total_episodes metrics.episode_return metrics.episode_length;
        `Continue)
      ()
  in

  (* Evaluate learned policy *)
  Printf.printf "\n=== Evaluation ===\n";
  let eval_episodes = 20 in
  let total_rewards = ref 0.0 in

  for episode = 1 to eval_episodes do
    let obs, _info = Env.reset env () in
    let obs_ref = ref obs in
    let done_flag = ref false in
    let episode_reward = ref 0.0 in

    while not !done_flag do
      (* Use greedy policy *)
      let obs_batched =
        match Rune.shape !obs_ref with
        | [| features |] -> Rune.reshape [| 1; features |] !obs_ref
        | [| 1; _ |] -> !obs_ref
        | _ -> !obs_ref
      in
      let logits = apply policy_net params ~training:false obs_batched in
      let action_idx =
        Rune.argmax logits ~axis:(-1) ~keepdims:false |> Rune.cast Rune.int32
      in
      let action_scalar =
        Rune.reshape [||] action_idx |> Rune.to_array |> fun arr -> arr.(0)
      in
      let action = Rune.scalar Rune.int32 action_scalar in
      let transition = Env.step env action in
      episode_reward := !episode_reward +. transition.reward;
      obs_ref := transition.observation;
      done_flag := transition.terminated || transition.truncated
    done;

    total_rewards := !total_rewards +. !episode_reward;
    Printf.printf "Eval episode %d: reward = %.2f\n" episode !episode_reward
  done;

  Printf.printf "Average evaluation reward: %.2f\n"
    (!total_rewards /. float_of_int eval_episodes);

  Env.close env

let () = run_policy_gradient ()
