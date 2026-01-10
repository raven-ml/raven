(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Fehu
module Dqn = Fehu_algorithms.Dqn

let make_env () =
  let obs_space = Space.Box.create ~low:[| 0.0 |] ~high:[| 10.0 |] in
  let act_space = Space.Discrete.create 2 in
  let rng = Rune.Rng.key 0 in
  let step_count = ref 0 in
  Env.create ~rng ~observation_space:obs_space ~action_space:act_space
    ~reset:(fun _ ?options:_ () ->
      step_count := 0;
      (Rune.create Rune.float32 [| 1 |] [| 0.0 |], Info.empty))
    ~step:(fun _ _ ->
      incr step_count;
      let terminated = !step_count >= 3 in
      let obs =
        Rune.create Rune.float32 [| 1 |] [| float_of_int !step_count |]
      in
      Env.transition ~observation:obs ~reward:1.0 ~terminated ~truncated:false
        ())
    ()

let make_network ~in_features ~hidden =
  Kaun.Layer.sequential
    [
      Kaun.Layer.linear ~in_features ~out_features:hidden ();
      Kaun.Layer.relu ();
      Kaun.Layer.linear ~in_features:hidden ~out_features:2 ();
    ]

let test_init () =
  let env = make_env () in
  let rng = Rune.Rng.key 42 in
  let q_net = make_network ~in_features:1 ~hidden:8 in
  let config = { Dqn.default_config with warmup_steps = 0 } in
  let params, state = Dqn.init ~env ~q_network:q_net ~rng ~config in
  ignore params;
  let metrics = Dqn.metrics state in
  Alcotest.(check int) "initial steps" 0 metrics.total_steps;
  Alcotest.(check int) "initial episodes" 0 metrics.total_episodes

let test_step_progresses () =
  let env = make_env () in
  let rng = Rune.Rng.key 7 in
  let q_net = make_network ~in_features:1 ~hidden:8 in
  let config = { Dqn.default_config with warmup_steps = 0 } in
  let params, state = Dqn.init ~env ~q_network:q_net ~rng ~config in
  let params', state' = Dqn.step ~env ~params ~state in
  ignore params';
  let metrics = Dqn.metrics state' in
  Alcotest.(check bool) "one step taken" true (metrics.total_steps = 1)

let test_train_runs () =
  let env = make_env () in
  let rng = Rune.Rng.key 11 in
  let q_net = make_network ~in_features:1 ~hidden:8 in
  let config = { Dqn.default_config with warmup_steps = 0 } in
  match Dqn.train ~env ~q_network:q_net ~rng ~config ~total_timesteps:20 () with
  | exception Fehu.Errors.Error err ->
      Alcotest.failf "train failed: %s" (Fehu.Errors.to_string err)
  | params, state ->
      ignore params;
      let metrics = Dqn.metrics state in
      Alcotest.(check bool) "trained steps" true (metrics.total_steps >= 20)

let test_save_and_load () =
  let env = make_env () in
  let rng = Rune.Rng.key 21 in
  let q_net = make_network ~in_features:1 ~hidden:8 in
  let config = { Dqn.default_config with warmup_steps = 0 } in
  let params, state =
    Dqn.train ~env ~q_network:q_net ~rng ~config ~total_timesteps:10 ()
  in
  let path = Filename.concat (Filename.get_temp_dir_name ()) "dqn_state" in
  Dqn.save ~path ~params ~state;
  let env2 = make_env () in
  match Dqn.load ~path ~env:env2 ~q_network:q_net ~config with
  | Ok (_params', state') ->
      let metrics = Dqn.metrics state' in
      Alcotest.(check int) "loaded steps" 0 metrics.total_steps
  | Error msg -> Alcotest.failf "failed to load dqn snapshot: %s" msg

let () =
  let open Alcotest in
  run "DQN"
    [
      ( "Core",
        [
          test_case "init" `Quick test_init;
          test_case "step" `Quick test_step_progresses;
          test_case "train" `Quick test_train_runs;
          test_case "save/load" `Quick test_save_and_load;
        ] );
    ]
