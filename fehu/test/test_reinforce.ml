open Fehu
module Reinforce = Fehu_algorithms.Reinforce

let test_create () =
  let rng = Rune.Rng.key 42 in

  (* Create simple policy network *)
  let policy_net =
    Kaun.Layer.sequential
      [
        Kaun.Layer.linear ~in_features:4 ~out_features:8 ();
        Kaun.Layer.relu ();
        Kaun.Layer.linear ~in_features:8 ~out_features:2 ();
      ]
  in

  let _agent =
    Reinforce.create ~policy_network:policy_net ~n_actions:2 ~rng
      Reinforce.default_config
  in

  (* Just verify it was created without crashing *)
  Alcotest.(check bool) "agent created" true true

let test_create_with_baseline () =
  let rng = Rune.Rng.key 42 in

  let policy_net =
    Kaun.Layer.sequential
      [
        Kaun.Layer.linear ~in_features:4 ~out_features:8 ();
        Kaun.Layer.relu ();
        Kaun.Layer.linear ~in_features:8 ~out_features:2 ();
      ]
  in

  let value_net =
    Kaun.Layer.sequential
      [
        Kaun.Layer.linear ~in_features:4 ~out_features:8 ();
        Kaun.Layer.relu ();
        Kaun.Layer.linear ~in_features:8 ~out_features:1 ();
      ]
  in

  let _agent =
    Reinforce.create ~policy_network:policy_net ~baseline_network:value_net
      ~n_actions:2 ~rng
      Reinforce.{ default_config with use_baseline = true }
  in

  Alcotest.(check bool) "agent with baseline created" true true

let test_create_baseline_missing () =
  let rng = Rune.Rng.key 42 in

  let policy_net =
    Kaun.Layer.sequential
      [ Kaun.Layer.linear ~in_features:4 ~out_features:2 () ]
  in

  let test_fn () =
    let _agent =
      Reinforce.create ~policy_network:policy_net ~n_actions:2 ~rng
        Reinforce.{ default_config with use_baseline = true }
    in
    ()
  in

  Alcotest.check_raises "baseline required when use_baseline = true"
    (Invalid_argument
       "Reinforce.create: baseline_network required when use_baseline = true")
    test_fn

let test_predict_greedy () =
  let rng = Rune.Rng.key 42 in

  let policy_net =
    Kaun.Layer.sequential
      [ Kaun.Layer.linear ~in_features:4 ~out_features:2 () ]
  in

  let agent =
    Reinforce.create ~policy_network:policy_net ~n_actions:2 ~rng
      Reinforce.default_config
  in

  let obs = Rune.create Rune.float32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let action, log_prob = Reinforce.predict agent obs ~training:false in

  (* Greedy action should have log_prob = 0.0 *)
  Alcotest.(check (float 0.0001)) "log_prob is 0" 0.0 log_prob;

  (* Action should be int32 *)
  let action_val = (Rune.to_array action).(0) in
  Alcotest.(check bool)
    "action is 0 or 1" true
    (action_val = 0l || action_val = 1l)

let test_predict_stochastic () =
  let rng = Rune.Rng.key 42 in

  let policy_net =
    Kaun.Layer.sequential
      [ Kaun.Layer.linear ~in_features:4 ~out_features:2 () ]
  in

  let agent =
    Reinforce.create ~policy_network:policy_net ~n_actions:2 ~rng
      Reinforce.default_config
  in

  let obs = Rune.create Rune.float32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let _action1, log_prob1 = Reinforce.predict agent obs ~training:true in
  let _action2, log_prob2 = Reinforce.predict agent obs ~training:true in

  (* Stochastic sampling should have negative log_prob *)
  Alcotest.(check bool) "log_prob1 < 0" true (log_prob1 < 0.0);
  Alcotest.(check bool) "log_prob2 < 0" true (log_prob2 < 0.0);

  (* Different samples may give different actions (though not guaranteed) *)
  (* Just verify both calls succeed *)
  Alcotest.(check bool) "sampling works" true true

let test_update () =
  let rng = Rune.Rng.key 42 in

  let policy_net =
    Kaun.Layer.sequential
      [
        Kaun.Layer.linear ~in_features:2 ~out_features:8 ();
        Kaun.Layer.relu ();
        Kaun.Layer.linear ~in_features:8 ~out_features:2 ();
      ]
  in

  let agent =
    Reinforce.create ~policy_network:policy_net ~n_actions:2 ~rng
      Reinforce.default_config
  in

  (* Create a simple trajectory *)
  let observations =
    [|
      Rune.create Rune.float32 [| 2 |] [| 0.0; 0.0 |];
      Rune.create Rune.float32 [| 2 |] [| 1.0; 0.0 |];
      Rune.create Rune.float32 [| 2 |] [| 2.0; 0.0 |];
    |]
  in
  let actions =
    [|
      Rune.scalar Rune.int32 0l;
      Rune.scalar Rune.int32 1l;
      Rune.scalar Rune.int32 0l;
    |]
  in
  let rewards = [| 1.0; 0.0; 1.0 |] in
  let terminateds = [| false; false; true |] in
  let truncateds = [| false; false; false |] in
  let log_probs = [| -0.5; -0.7; -0.6 |] in

  let trajectory =
    Trajectory.create ~observations ~actions ~rewards ~terminateds ~truncateds
      ~log_probs ()
  in

  let _agent, metrics = Reinforce.update agent trajectory in

  (* Verify metrics are reasonable *)
  Alcotest.(check bool)
    "episode_return is finite" true
    (not (Float.is_nan metrics.episode_return));
  Alcotest.(check int) "episode_length is 3" 3 metrics.episode_length;
  Alcotest.(check bool) "avg_entropy >= 0" true (metrics.avg_entropy >= 0.0);
  Alcotest.(check bool) "avg_log_prob < 0" true (metrics.avg_log_prob < 0.0)

let test_learn () =
  let rng = Rune.Rng.key 42 in
  let env_rng = Rune.Rng.key 123 in

  (* Create simple environment with slightly longer episodes *)
  let obs_space = Space.Box.create ~low:[| 0.0; 0.0 |] ~high:[| 10.0; 10.0 |] in
  let act_space = Space.Discrete.create 2 in

  let step_count = ref 0 in

  let env =
    Env.create ~rng:env_rng ~observation_space:obs_space ~action_space:act_space
      ~reset:(fun _env ?options:_ () ->
        step_count := 0;
        let obs = Rune.create Rune.float32 [| 2 |] [| 0.0; 0.0 |] in
        (obs, Info.empty))
      ~step:(fun _env _action ->
        step_count := !step_count + 1;
        let terminated = !step_count >= 5 in
        let obs =
          Rune.create Rune.float32 [| 2 |] [| float_of_int !step_count; 0.0 |]
        in
        Env.transition ~observation:obs ~reward:1.0 ~terminated ())
      ()
  in

  let policy_net =
    Kaun.Layer.sequential
      [
        Kaun.Layer.linear ~in_features:2 ~out_features:8 ();
        Kaun.Layer.relu ();
        Kaun.Layer.linear ~in_features:8 ~out_features:2 ();
      ]
  in

  let agent =
    Reinforce.create ~policy_network:policy_net ~n_actions:2 ~rng
      Reinforce.default_config
  in

  let iterations_called = ref 0 in

  let agent =
    Reinforce.learn agent ~env ~total_timesteps:15
      ~callback:(fun ~iteration:_ ~metrics:_ ->
        incr iterations_called;
        true)
      ()
  in

  (* Should have run multiple episodes to reach 15 timesteps (3 episodes of 5
     steps each) *)
  Alcotest.(check bool)
    "callback called at least twice" true (!iterations_called >= 2);

  (* Agent should still be usable after training *)
  let obs = Rune.create Rune.float32 [| 2 |] [| 0.5; 0.5 |] in
  let _action, _log_prob = Reinforce.predict agent obs ~training:false in
  Alcotest.(check bool) "agent usable after training" true true

let test_learn_early_stop () =
  let rng = Rune.Rng.key 42 in
  let env_rng = Rune.Rng.key 123 in

  let obs_space = Space.Box.create ~low:[| 0.0 |] ~high:[| 10.0 |] in
  let act_space = Space.Discrete.create 2 in

  let step_count = ref 0 in

  let env =
    Env.create ~rng:env_rng ~observation_space:obs_space ~action_space:act_space
      ~reset:(fun _env ?options:_ () ->
        step_count := 0;
        let obs = Rune.create Rune.float32 [| 1 |] [| 0.0 |] in
        (obs, Info.empty))
      ~step:(fun _env _action ->
        step_count := !step_count + 1;
        let terminated = !step_count >= 3 in
        let obs =
          Rune.create Rune.float32 [| 1 |] [| float_of_int !step_count |]
        in
        Env.transition ~observation:obs ~reward:1.0 ~terminated ())
      ()
  in

  let policy_net =
    Kaun.Layer.sequential
      [ Kaun.Layer.linear ~in_features:1 ~out_features:2 () ]
  in

  let agent =
    Reinforce.create ~policy_network:policy_net ~n_actions:2 ~rng
      Reinforce.default_config
  in

  let iterations = ref 0 in

  let _agent =
    Reinforce.learn agent ~env ~total_timesteps:1000
      ~callback:(fun ~iteration:_ ~metrics:_ ->
        incr iterations;
        !iterations < 3 (* Stop after 3 iterations *))
      ()
  in

  (* Should have stopped early *)
  Alcotest.(check int) "stopped after 3 iterations" 3 !iterations

let () =
  let open Alcotest in
  run "Reinforce"
    [
      ( "Creation",
        [
          test_case "create agent" `Quick test_create;
          test_case "create with baseline" `Quick test_create_with_baseline;
          test_case "create baseline missing raises" `Quick
            test_create_baseline_missing;
        ] );
      ( "Prediction",
        [
          test_case "predict greedy" `Quick test_predict_greedy;
          test_case "predict stochastic" `Quick test_predict_stochastic;
        ] );
      ( "Training",
        [
          test_case "update" `Quick test_update;
          test_case "learn" `Quick test_learn;
          test_case "learn with early stop" `Quick test_learn_early_stop;
        ] );
    ]
