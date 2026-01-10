(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Fehu
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
  Alcotest.(check int) "initial total steps" 0 metrics.total_steps;
  Alcotest.(check int) "initial episodes" 0 metrics.total_episodes

let test_init_requires_baseline () =
  let env = make_env () in
  let rng = Rune.Rng.key 11 in
  let policy_net = make_policy_network ~obs_dim:2 ~n_actions:2 in
  let config = { Reinforce.default_config with use_baseline = true } in
  Alcotest.check_raises "baseline required when flag is set"
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
  Alcotest.(check int) "baseline init total steps" 0 metrics.total_steps

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
  Alcotest.(check int) "episode length" 3 metrics.episode_length;
  Alcotest.(check int) "episodes counted" 1 metrics.total_episodes;
  Alcotest.(check bool)
    "episode return positive" true
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
      Alcotest.(check bool)
        "train advanced steps" true
        (metrics.total_steps >= 15)
  | exception Fehu.Errors.Error err ->
      Alcotest.failf "train raised env error: %s" (Fehu.Errors.to_string err)

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
      Alcotest.(check int)
        "loaded preserves total steps" saved_metrics.total_steps
        metrics.total_steps;
      Alcotest.(check int)
        "loaded preserves total episodes" saved_metrics.total_episodes
        metrics.total_episodes
  | Error msg -> Alcotest.failf "load failed: %s" msg

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
  Alcotest.(check bool)
    "baseline train episodes" true
    (metrics.total_episodes > 0)

let () =
  let open Alcotest in
  run "REINFORCE"
    [
      ( "Core",
        [
          test_case "init" `Quick test_init;
          test_case "init requires baseline" `Quick test_init_requires_baseline;
          test_case "init with baseline" `Quick test_init_with_baseline;
          test_case "step completes episode" `Quick test_step_completes_episode;
          test_case "train" `Quick test_train_runs;
          test_case "save/load" `Quick test_save_and_load;
          test_case "train with baseline" `Quick test_train_with_baseline;
        ] );
    ]
