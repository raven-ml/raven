(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Fehu
open Windtrap
module Reinforce = Fehu_algorithms.Reinforce

let make_env () =
  let obs_space = Space.Box.create ~low:[| 0.0; 0.0 |] ~high:[| 10.0; 10.0 |] in
  let act_space = Space.Discrete.create 2 in
  let rng = Rune.Rng.key 0 in
  let step_count = ref 0 in
  Env.create ~rng ~observation_space:obs_space ~action_space:act_space
    ~reset:(fun _ ?options:_ () ->
      step_count := 0;
      (Rune.create Rune.float32 [| 2 |] [| 0.0; 0.0 |], Info.empty))
    ~step:(fun _ _action ->
      incr step_count;
      let terminated = !step_count >= 3 in
      let obs =
        Rune.create Rune.float32 [| 2 |] [| float_of_int !step_count; 0.0 |]
      in
      Env.transition ~observation:obs ~reward:1.0 ~terminated ~truncated:false
        ())
    ()

let make_policy_network ~obs_dim ~n_actions =
  Kaun.Layer.sequential
    [
      Kaun.Layer.linear ~in_features:obs_dim ~out_features:8 ();
      Kaun.Layer.relu ();
      Kaun.Layer.linear ~in_features:8 ~out_features:n_actions ();
    ]

let make_baseline_network ~obs_dim =
  Kaun.Layer.sequential
    [
      Kaun.Layer.linear ~in_features:obs_dim ~out_features:8 ();
      Kaun.Layer.relu ();
      Kaun.Layer.linear ~in_features:8 ~out_features:1 ();
    ]

let test_init () =
  let env = make_env () in
  let rng = Rune.Rng.key 42 in
  let policy_net = make_policy_network ~obs_dim:2 ~n_actions:2 in
  let config = { Reinforce.default_config with max_episode_steps = 5 } in
  let params, state =
    Reinforce.init ~env ~policy_network:policy_net ~rng ~config ()
  in
  ignore params;
  let metrics = Reinforce.metrics state in
  equal ~msg:"initial total steps" int 0 metrics.total_steps;
  equal ~msg:"initial episodes" int 0 metrics.total_episodes

let test_init_requires_baseline () =
  let env = make_env () in
  let rng = Rune.Rng.key 11 in
  let policy_net = make_policy_network ~obs_dim:2 ~n_actions:2 in
  let config = { Reinforce.default_config with use_baseline = true } in
  raises ~msg:"baseline required when flag is set"
    (Invalid_argument
       "Reinforce.init: baseline_network required when use_baseline = true")
    (fun () ->
      ignore (Reinforce.init ~env ~policy_network:policy_net ~rng ~config ()))

let test_init_with_baseline () =
  let env = make_env () in
  let rng = Rune.Rng.key 17 in
  let policy_net = make_policy_network ~obs_dim:2 ~n_actions:2 in
  let baseline_net = make_baseline_network ~obs_dim:2 in
  let config =
    { Reinforce.default_config with use_baseline = true; max_episode_steps = 5 }
  in
  let params, state =
    Reinforce.init ~env ~policy_network:policy_net
      ~baseline_network:baseline_net ~rng ~config ()
  in
  ignore params;
  let metrics = Reinforce.metrics state in
  equal ~msg:"baseline init total steps" int 0 metrics.total_steps

let finish_episode env params state =
  let rec loop params state =
    let params, state = Reinforce.step ~env ~params ~state in
    let metrics = Reinforce.metrics state in
    if metrics.total_episodes > 0 then (params, state) else loop params state
  in
  loop params state

let test_step_completes_episode () =
  let env = make_env () in
  let rng = Rune.Rng.key 5 in
  let policy_net = make_policy_network ~obs_dim:2 ~n_actions:2 in
  let config = { Reinforce.default_config with max_episode_steps = 5 } in
  let params, state =
    Reinforce.init ~env ~policy_network:policy_net ~rng ~config ()
  in
  let _params, state = finish_episode env params state in
  let metrics = Reinforce.metrics state in
  equal ~msg:"episode length" int 3 metrics.episode_length;
  equal ~msg:"episodes counted" int 1 metrics.total_episodes;
  equal ~msg:"episode return positive" bool true
    (metrics.episode_return > 0.0)

let test_train_runs () =
  let env = make_env () in
  let rng = Rune.Rng.key 9 in
  let policy_net = make_policy_network ~obs_dim:2 ~n_actions:2 in
  let config = { Reinforce.default_config with max_episode_steps = 5 } in
  match
    Reinforce.train ~env ~policy_network:policy_net ~rng ~config
      ~total_timesteps:15 ()
  with
  | params, state ->
      ignore params;
      let metrics = Reinforce.metrics state in
      equal ~msg:"train advanced steps" bool true
        (metrics.total_steps >= 15)
  | exception Fehu.Errors.Error err ->
      failf "train raised env error: %s" (Fehu.Errors.to_string err)

let test_save_and_load () =
  let env = make_env () in
  let rng = Rune.Rng.key 19 in
  let policy_net = make_policy_network ~obs_dim:2 ~n_actions:2 in
  let config = { Reinforce.default_config with max_episode_steps = 5 } in
  let params, state =
    Reinforce.train ~env ~policy_network:policy_net ~rng ~config
      ~total_timesteps:12 ()
  in
  let saved_metrics = Reinforce.metrics state in
  let path = Filename.concat (Filename.get_temp_dir_name ()) "reinforce.snap" in
  Reinforce.save ~path ~params ~state;
  let env' = make_env () in
  match
    Reinforce.load ~path ~env:env' ~policy_network:policy_net ~config ()
  with
  | Ok (_params', state') ->
      let metrics = Reinforce.metrics state' in
      equal ~msg:"loaded preserves total steps" int saved_metrics.total_steps
        metrics.total_steps;
      equal ~msg:"loaded preserves total episodes" int saved_metrics.total_episodes
        metrics.total_episodes
  | Error msg -> failf "load failed: %s" msg

let test_train_with_baseline () =
  let env = make_env () in
  let rng = Rune.Rng.key 27 in
  let policy_net = make_policy_network ~obs_dim:2 ~n_actions:2 in
  let baseline_net = make_baseline_network ~obs_dim:2 in
  let config =
    { Reinforce.default_config with use_baseline = true; max_episode_steps = 5 }
  in
  let params, state =
    Reinforce.train ~env ~policy_network:policy_net
      ~baseline_network:baseline_net ~rng ~config ~total_timesteps:12 ()
  in
  ignore params;
  let metrics = Reinforce.metrics state in
  equal ~msg:"baseline train episodes" bool true
    (metrics.total_episodes > 0)

let () =
  run "REINFORCE"
    [
      group "Core"
        [
          test "init" test_init;
          test "init requires baseline" test_init_requires_baseline;
          test "init with baseline" test_init_with_baseline;
          test "step completes episode" test_step_completes_episode;
          test "train" test_train_runs;
          test "save/load" test_save_and_load;
          test "train with baseline" test_train_with_baseline;
        ];
    ]
