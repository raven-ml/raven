(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Fehu
open Kaun
module Dqn = Fehu_algorithms.Dqn
module Policy = Fehu.Policy
module Wrapper = Fehu.Wrapper
module FV = Fehu_visualize

let grid_size = Fehu_envs.Grid_world.grid_size

let float_grid_env ?render_mode rng =
  let base = Fehu_envs.Grid_world.make ?render_mode ~rng () in
  let max_coord = float_of_int (grid_size - 1) in
  let observation_space =
    Space.Box.create ~low:[| 0.0; 0.0 |] ~high:[| max_coord; max_coord |]
  in
  Wrapper.map_observation ~observation_space
    ~f:(fun observation info ->
      let ints : Int32.t array = Rune.to_array observation in
      let floats = Array.map (fun v -> Int32.to_float v) ints in
      let tensor = Rune.create Rune.float32 [| 2 |] floats in
      (tensor, info))
    base

let create_q_network () =
  Layer.sequential
    [
      Layer.linear ~in_features:2 ~out_features:64 ();
      Layer.relu ();
      Layer.linear ~in_features:64 ~out_features:64 ();
      Layer.relu ();
      Layer.linear ~in_features:64 ~out_features:4 ();
    ]

let q_scores q_network params obs =
  let obs_f = Rune.cast Rune.float32 obs in
  let shape = Rune.shape obs_f in
  let obs_batch =
    match shape with
    | [||] -> Rune.reshape [| 1 |] obs_f
    | [| features |] -> Rune.reshape [| 1; features |] obs_f
    | shape when shape.(0) = 1 -> obs_f
    | _ -> Rune.expand_dims [ 0 ] obs_f
  in
  let q_values = Kaun.apply q_network params ~training:false obs_batch in
  Rune.to_array q_values

let record_guard description f =
  try f ()
  with Invalid_argument msg ->
    Printf.eprintf "Warning: %s (%s)\n%!" description msg

let record_random_rollout ~path ~steps =
  let env = float_grid_env ~render_mode:`Rgb_array (Rune.Rng.key 73) in
  let policy = Policy.random env in
  FV.Sink.with_ffmpeg ~fps:6 ~path (fun sink ->
      FV.record_rollout ~env ~policy ~steps ~sink ());
  Env.close env

let record_trained_rollout ~path ~steps ~q_network ~params =
  let env = float_grid_env ~render_mode:`Rgb_array (Rune.Rng.key 137) in
  let policy =
    Policy.greedy_discrete env ~score:(fun obs -> q_scores q_network params obs)
  in
  FV.Sink.with_ffmpeg ~fps:6 ~path (fun sink ->
      FV.record_rollout ~env ~policy ~steps ~sink ());
  Env.close env

let run_dqn () =
  let record_dir = Sys.getenv_opt "FEHU_DQN_RECORD_DIR" in
  let record_path suffix =
    Option.map (fun dir -> Filename.concat dir suffix) record_dir
  in
  Option.iter
    (fun path ->
      Printf.printf "Recording untrained rollout to %s\n%!" path;
      record_guard "recording untrained rollout" (fun () ->
          record_random_rollout ~path ~steps:100))
    (record_path "gridworld_random.mp4");

  let training_env = float_grid_env (Rune.Rng.key 3) in
  let q_network = create_q_network () in
  let config =
    {
      Dqn.default_config with
      batch_size = 32;
      buffer_capacity = 10_000;
      target_update_freq = 10;
    }
  in
  let target_episodes = 400 in
  let total_timesteps = target_episodes * 64 in

  Printf.printf "Training DQN on GridWorld via Fehu_algorithms.Dqn...\n%!";
  Printf.printf "Total timesteps: %d, batch size: %d, replay capacity: %d\n%!"
    total_timesteps config.batch_size config.buffer_capacity;

  let log_every = 50 in
  let last_logged_episode = ref 0 in
  let params, state =
    Dqn.train ~env:training_env ~q_network ~rng:(Rune.Rng.key 42) ~config
      ~total_timesteps
      ~callback:(fun metrics ->
        (match (metrics.episode_return, metrics.episode_length) with
        | Some reward, Some length
          when metrics.total_episodes > 0
               && metrics.total_episodes mod log_every = 0
               && metrics.total_episodes <> !last_logged_episode ->
            last_logged_episode := metrics.total_episodes;
            Printf.printf
              "Episode %d: Reward = %.2f, Steps = %d, Epsilon = %.3f, Loss = \
               %.4f\n\
               %!"
              metrics.total_episodes reward length metrics.epsilon metrics.loss
        | _ -> ());
        if metrics.total_episodes >= target_episodes then `Stop else `Continue)
      ()
  in
  let final_metrics = Dqn.metrics state in
  Printf.printf
    "Training finished after %d steps (%d/%d episodes). Final epsilon: %.3f, \
     last loss: %.4f, avg Q: %.4f\n\
     %!"
    final_metrics.total_steps final_metrics.total_episodes target_episodes
    final_metrics.epsilon final_metrics.loss final_metrics.avg_q_value;
  Env.close training_env;

  let eval_env = float_grid_env (Rune.Rng.key 1337) in
  let greedy_policy =
    Policy.greedy_discrete eval_env ~score:(fun obs ->
        q_scores q_network params obs)
  in
  let eval_stats =
    Fehu.Training.evaluate eval_env
      ~policy:(fun obs ->
        let action, _, _ = greedy_policy obs in
        action)
      ~n_episodes:20 ~max_steps:200 ()
  in
  Printf.printf
    "\n=== Evaluation ===\nMean reward: %.2f ± %.2f (length: %.1f)\n%!"
    eval_stats.mean_reward eval_stats.std_reward eval_stats.mean_length;
  Env.close eval_env;

  Option.iter
    (fun path ->
      Printf.printf "Recording trained rollout to %s\n%!" path;
      record_guard "recording trained rollout" (fun () ->
          record_trained_rollout ~path ~steps:100 ~q_network ~params))
    (record_path "gridworld_trained.mp4")

let () = run_dqn ()
