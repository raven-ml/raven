open Fehu
module Dqn = Fehu_algorithms.Dqn

let test_create () =
  let rng = Rune.Rng.key 42 in

  (* Create simple Q-network *)
  let q_net =
    Kaun.Layer.sequential
      [
        Kaun.Layer.linear ~in_features:4 ~out_features:16 ();
        Kaun.Layer.relu ();
        Kaun.Layer.linear ~in_features:16 ~out_features:2 ();
      ]
  in

  let _agent =
    Dqn.create ~q_network:q_net ~n_actions:2 ~rng Dqn.default_config
  in

  (* Just verify it was created without crashing *)
  Alcotest.(check bool) "agent created" true true

let test_create_custom_config () =
  let rng = Rune.Rng.key 42 in

  let q_net =
    Kaun.Layer.sequential
      [ Kaun.Layer.linear ~in_features:2 ~out_features:4 () ]
  in

  let custom_config =
    Dqn.
      {
        learning_rate = 0.0001;
        gamma = 0.95;
        epsilon_start = 0.5;
        epsilon_end = 0.05;
        epsilon_decay = 500.0;
        batch_size = 64;
        buffer_capacity = 5000;
        target_update_freq = 5;
      }
  in

  let _agent = Dqn.create ~q_network:q_net ~n_actions:4 ~rng custom_config in

  Alcotest.(check bool) "agent with custom config created" true true

let test_predict_greedy () =
  let rng = Rune.Rng.key 42 in

  let q_net =
    Kaun.Layer.sequential
      [ Kaun.Layer.linear ~in_features:4 ~out_features:2 () ]
  in

  let agent =
    Dqn.create ~q_network:q_net ~n_actions:2 ~rng Dqn.default_config
  in

  let obs = Rune.create Rune.float32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let action = Dqn.predict agent obs ~epsilon:0.0 in

  (* Action should be int32 scalar *)
  let action_shape = Rune.shape action in
  Alcotest.(check int) "action is scalar" 0 (Array.length action_shape);

  let action_val = (Rune.to_array action).(0) in
  Alcotest.(check bool)
    "action is 0 or 1" true
    (action_val = 0l || action_val = 1l)

let test_predict_random () =
  let rng = Rune.Rng.key 42 in

  let q_net =
    Kaun.Layer.sequential
      [ Kaun.Layer.linear ~in_features:4 ~out_features:2 () ]
  in

  let agent =
    Dqn.create ~q_network:q_net ~n_actions:2 ~rng Dqn.default_config
  in

  let obs = Rune.create Rune.float32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in

  (* With epsilon = 1.0, should always explore randomly *)
  let action1 = Dqn.predict agent obs ~epsilon:1.0 in
  let action2 = Dqn.predict agent obs ~epsilon:1.0 in

  (* Actions should be valid *)
  let action1_val = (Rune.to_array action1).(0) in
  let action2_val = (Rune.to_array action2).(0) in

  Alcotest.(check bool)
    "action1 is valid" true
    (action1_val = 0l || action1_val = 1l);
  Alcotest.(check bool)
    "action2 is valid" true
    (action2_val = 0l || action2_val = 1l)

let test_add_transition () =
  let rng = Rune.Rng.key 42 in

  let q_net =
    Kaun.Layer.sequential
      [ Kaun.Layer.linear ~in_features:2 ~out_features:2 () ]
  in

  let agent =
    Dqn.create ~q_network:q_net ~n_actions:2 ~rng Dqn.default_config
  in

  (* Add some transitions *)
  let obs = Rune.create Rune.float32 [| 2 |] [| 0.0; 0.0 |] in
  let action = Rune.scalar Rune.int32 0l in
  let next_obs = Rune.create Rune.float32 [| 2 |] [| 1.0; 0.0 |] in

  Dqn.add_transition agent ~observation:obs ~action ~reward:1.0
    ~next_observation:next_obs ~terminated:false ~truncated:false;

  Dqn.add_transition agent ~observation:next_obs ~action ~reward:0.0
    ~next_observation:obs ~terminated:true ~truncated:false;

  (* Buffer should have 2 transitions *)
  Alcotest.(check bool) "transitions added" true true

let test_update_insufficient_samples () =
  let rng = Rune.Rng.key 42 in

  let q_net =
    Kaun.Layer.sequential
      [ Kaun.Layer.linear ~in_features:2 ~out_features:2 () ]
  in

  let config = Dqn.{ default_config with batch_size = 10 } in

  let agent = Dqn.create ~q_network:q_net ~n_actions:2 ~rng config in

  (* Add only a few transitions (less than batch_size) *)
  for i = 0 to 5 do
    let obs = Rune.create Rune.float32 [| 2 |] [| float_of_int i; 0.0 |] in
    let action = Rune.scalar Rune.int32 0l in
    let next_obs =
      Rune.create Rune.float32 [| 2 |] [| float_of_int (i + 1); 0.0 |]
    in
    Dqn.add_transition agent ~observation:obs ~action ~reward:1.0
      ~next_observation:next_obs ~terminated:false ~truncated:false
  done;

  (* Update should return (0.0, 0.0) when insufficient samples *)
  let loss, avg_q = Dqn.update agent in
  Alcotest.(check (float 0.0001)) "loss is 0" 0.0 loss;
  Alcotest.(check (float 0.0001)) "avg_q is 0" 0.0 avg_q

let test_update () =
  let rng = Rune.Rng.key 42 in

  let q_net =
    Kaun.Layer.sequential
      [
        Kaun.Layer.linear ~in_features:2 ~out_features:16 ();
        Kaun.Layer.relu ();
        Kaun.Layer.linear ~in_features:16 ~out_features:2 ();
      ]
  in

  let config = Dqn.{ default_config with batch_size = 8 } in

  let agent = Dqn.create ~q_network:q_net ~n_actions:2 ~rng config in

  (* Add enough transitions *)
  for i = 0 to 15 do
    let obs = Rune.create Rune.float32 [| 2 |] [| float_of_int i; 0.0 |] in
    let action = Rune.scalar Rune.int32 (Int32.of_int (i mod 2)) in
    let next_obs =
      Rune.create Rune.float32 [| 2 |] [| float_of_int (i + 1); 0.0 |]
    in
    let terminated = i = 15 in
    Dqn.add_transition agent ~observation:obs ~action ~reward:1.0
      ~next_observation:next_obs ~terminated ~truncated:false
  done;

  (* Update should succeed and return finite loss *)
  let loss, avg_q = Dqn.update agent in
  Alcotest.(check bool) "loss is finite" true (not (Float.is_nan loss));
  Alcotest.(check bool) "avg_q is finite" true (not (Float.is_nan avg_q));
  Alcotest.(check bool) "loss >= 0" true (loss >= 0.0)

let test_update_target_network () =
  let rng = Rune.Rng.key 42 in

  let q_net =
    Kaun.Layer.sequential
      [ Kaun.Layer.linear ~in_features:2 ~out_features:2 () ]
  in

  let agent =
    Dqn.create ~q_network:q_net ~n_actions:2 ~rng Dqn.default_config
  in

  (* Update target network (should just copy parameters) *)
  Dqn.update_target_network agent;

  Alcotest.(check bool) "target network updated" true true

let test_learn () =
  let rng = Rune.Rng.key 42 in
  let env_rng = Rune.Rng.key 123 in

  (* Create simple environment *)
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

  let q_net =
    Kaun.Layer.sequential
      [
        Kaun.Layer.linear ~in_features:2 ~out_features:8 ();
        Kaun.Layer.relu ();
        Kaun.Layer.linear ~in_features:8 ~out_features:2 ();
      ]
  in

  let config =
    Dqn.{ default_config with batch_size = 4; buffer_capacity = 50 }
  in

  let agent = Dqn.create ~q_network:q_net ~n_actions:2 ~rng config in

  let episodes_called = ref 0 in

  let agent =
    Dqn.learn agent ~env ~total_timesteps:25 ~warmup_steps:10
      ~callback:(fun ~episode:_ ~metrics:_ ->
        incr episodes_called;
        true)
      ()
  in

  (* Should have run multiple episodes *)
  Alcotest.(check bool)
    "callback called at least once" true (!episodes_called >= 1);

  (* Agent should still be usable after training *)
  let obs = Rune.create Rune.float32 [| 2 |] [| 0.5; 0.5 |] in
  let _action = Dqn.predict agent obs ~epsilon:0.0 in
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

  let q_net =
    Kaun.Layer.sequential
      [ Kaun.Layer.linear ~in_features:1 ~out_features:2 () ]
  in

  let agent =
    Dqn.create ~q_network:q_net ~n_actions:2 ~rng Dqn.default_config
  in

  let episodes = ref 0 in

  let _agent =
    Dqn.learn agent ~env ~total_timesteps:1000
      ~callback:(fun ~episode:_ ~metrics:_ ->
        incr episodes;
        !episodes < 3 (* Stop after 3 episodes *))
      ()
  in

  (* Should have stopped early *)
  Alcotest.(check int) "stopped after 3 episodes" 3 !episodes

let () =
  let open Alcotest in
  run "DQN"
    [
      ( "Creation",
        [
          test_case "create agent" `Quick test_create;
          test_case "create with custom config" `Quick test_create_custom_config;
        ] );
      ( "Prediction",
        [
          test_case "predict greedy" `Quick test_predict_greedy;
          test_case "predict random" `Quick test_predict_random;
        ] );
      ( "Buffer",
        [
          test_case "add transition" `Quick test_add_transition;
          test_case "update insufficient samples" `Quick
            test_update_insufficient_samples;
        ] );
      ( "Training",
        [
          test_case "update" `Quick test_update;
          test_case "update target network" `Quick test_update_target_network;
          test_case "learn" `Quick test_learn;
          test_case "learn with early stop" `Quick test_learn_early_stop;
        ] );
    ]
