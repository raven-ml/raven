open Fehu

let test_compute_gae () =
  let rewards = [| 1.0; 1.0; 1.0 |] in
  let values = [| 0.0; 0.0; 0.0 |] in
  let dones = [| false; false; true |] in
  let advantages, returns =
    Training.compute_gae ~rewards ~values ~dones ~gamma:0.99 ~gae_lambda:0.95
  in

  Alcotest.(check int) "advantages length" 3 (Array.length advantages);
  Alcotest.(check int) "returns length" 3 (Array.length returns);

  (* All advantages should be positive since rewards are positive *)
  Array.iter
    (fun adv -> Alcotest.(check bool) "advantage > 0" true (adv > 0.0))
    advantages

let test_compute_returns () =
  let rewards = [| 1.0; 1.0; 1.0 |] in
  let dones = [| false; false; true |] in
  let returns = Training.compute_returns ~rewards ~dones ~gamma:0.99 in

  Alcotest.(check int) "returns length" 3 (Array.length returns);

  (* First return should be largest (sum of all future rewards) *)
  Alcotest.(check bool)
    "returns[0] >= returns[1]" true
    (returns.(0) >= returns.(1));
  Alcotest.(check bool)
    "returns[1] >= returns[2]" true
    (returns.(1) >= returns.(2))

let test_normalize () =
  let arr = [| 1.0; 2.0; 3.0; 4.0; 5.0 |] in
  let normalized = Training.normalize arr () in

  Alcotest.(check int) "normalized length" 5 (Array.length normalized);

  (* Check mean is close to 0 *)
  let sum = Array.fold_left ( +. ) 0.0 normalized in
  let mean = sum /. 5.0 in
  Alcotest.(check bool) "mean ~= 0" true (abs_float mean < 1e-6);

  (* Check std is close to 1 *)
  let var_sum = Array.fold_left (fun acc x -> acc +. (x *. x)) 0.0 normalized in
  let std = sqrt (var_sum /. 5.0) in
  Alcotest.(check bool) "std ~= 1" true (abs_float (std -. 1.0) < 1e-6)

let test_policy_gradient_loss () =
  let log_probs = [| -0.5; -0.3; -0.7 |] in
  let advantages = [| 1.0; -1.0; 0.5 |] in

  let loss = Training.policy_gradient_loss ~log_probs ~advantages () in

  (* Loss should be finite *)
  Alcotest.(check bool) "loss is finite" true (not (Float.is_nan loss));
  Alcotest.(check bool) "loss is finite" true (not (Float.is_infinite loss))

let test_policy_gradient_loss_no_normalize () =
  let log_probs = [| -0.5; -0.3; -0.7 |] in
  let advantages = [| 1.0; -1.0; 0.5 |] in

  let loss =
    Training.policy_gradient_loss ~log_probs ~advantages ~normalize:false ()
  in

  Alcotest.(check bool) "loss is finite" true (not (Float.is_nan loss));
  Alcotest.(check bool) "loss is finite" true (not (Float.is_infinite loss))

let test_ppo_clip_loss () =
  let log_probs = [| -0.5; -0.3; -0.7 |] in
  let old_log_probs = [| -0.6; -0.4; -0.8 |] in
  let advantages = [| 1.0; -1.0; 0.5 |] in

  let loss =
    Training.ppo_clip_loss ~log_probs ~old_log_probs ~advantages ~clip_range:0.2
  in

  Alcotest.(check bool) "loss is finite" true (not (Float.is_nan loss));
  Alcotest.(check bool) "loss is finite" true (not (Float.is_infinite loss))

let test_value_loss () =
  let values = [| 1.0; 2.0; 3.0 |] in
  let returns = [| 1.5; 2.5; 2.8 |] in

  let loss = Training.value_loss ~values ~returns () in

  Alcotest.(check bool) "loss >= 0" true (loss >= 0.0);
  Alcotest.(check bool) "loss is finite" true (not (Float.is_nan loss))

let test_value_loss_clipped () =
  let values = [| 1.2; 2.8; 3.5 |] in
  let old_values = [| 1.0; 3.0; 3.0 |] in
  let returns = [| 1.1; 2.7; 3.2 |] in
  let clip_range = 0.1 in

  (* Compute expected value manually*)
  let compute_one v ov r =
    let delta = v -. ov in
    let clipped_delta = max (-.clip_range) (min clip_range delta) in
    let v_clipped = ov +. clipped_delta in
    let unclipped = (v -. r) ** 2.0 in
    let clipped = (v_clipped -. r) ** 2.0 in
    max unclipped clipped
  in

  (* Compute expected loss using the arrays *)
  let n = Array.length values in
  let sum = ref 0.0 in
  for i = 0 to n - 1 do
    sum := !sum +. compute_one values.(i) old_values.(i) returns.(i)
  done;
  let expected = !sum /. float_of_int n in
  let loss =
    Training.value_loss ~values ~returns ~clip:(clip_range, old_values) ()
  in

  Alcotest.(check (float 1e-6))
    "clipped value loss matches expected" expected loss;

  Alcotest.check_raises "clip_range should be positive"
    (Invalid_argument "Training.value_loss: clip_range must be non-negative")
    (fun () ->
      ignore (Training.value_loss ~values ~returns ~clip:(-0.2, old_values) ()));

  let old_values_small = [| 1.0; 3.0 |] in
  Alcotest.check_raises "old_values should be have the same shape"
    (Invalid_argument
       "Training.value_loss: old_values must have same length as arrays")
    (fun () ->
      ignore
        (Training.value_loss ~values ~returns
           ~clip:(clip_range, old_values_small)
           ()));

  (*Change the shape of returns and check it raises error*)
  let returns = [| 1.1; 2.7 |] in
  Alcotest.check_raises "old_values should be have the same shape"
    (Invalid_argument "Training.value_loss: arrays must have same length")
    (fun () ->
      ignore
        (Training.value_loss ~values ~returns ~clip:(clip_range, old_values) ()))

let test_explained_variance () =
  let y_pred = [| 1.0; 2.0; 3.0; 4.0 |] in
  let y_true = [| 1.1; 1.9; 3.1; 3.9 |] in

  let ev = Training.explained_variance ~y_pred ~y_true in

  (* Explained variance should be between -inf and 1 *)
  Alcotest.(check bool) "ev <= 1" true (ev <= 1.0);
  Alcotest.(check bool) "ev is finite" true (not (Float.is_nan ev));

  (* For good predictions, should be close to 1 *)
  Alcotest.(check bool) "ev > 0.9" true (ev > 0.9)

let test_evaluate () =
  (* Create a simple environment *)
  let rng = Rune.Rng.key 42 in
  let obs_space = Space.Box.create ~low:[| -1.0 |] ~high:[| 1.0 |] in
  let act_space = Space.Discrete.create 2 in

  let env =
    Env.create ~rng ~observation_space:obs_space ~action_space:act_space
      ~reset:(fun _env ?options:_ () ->
        let obs = Rune.create Rune.float32 [| 1 |] [| 0.0 |] in
        (obs, Info.empty))
      ~step:(fun _env _action ->
        let obs = Rune.create Rune.float32 [| 1 |] [| 0.0 |] in
        Env.transition ~observation:obs ~reward:1.0 ~terminated:true ())
      ()
  in

  (* Simple policy that always returns action 0 *)
  let policy _obs = Rune.create Rune.int32 [| 1 |] [| 0l |] in

  let stats = Training.evaluate env ~policy ~n_episodes:5 () in

  Alcotest.(check int) "n_episodes" 5 stats.n_episodes;
  Alcotest.(check (float 0.0001)) "mean_reward" 1.0 stats.mean_reward;
  Alcotest.(check bool) "mean_length > 0" true (stats.mean_length > 0.0)

let () =
  let open Alcotest in
  run "Training"
    [
      ( "Advantage Estimation",
        [
          test_case "compute GAE" `Quick test_compute_gae;
          test_case "compute returns" `Quick test_compute_returns;
        ] );
      ( "Loss Functions",
        [
          test_case "policy gradient loss" `Quick test_policy_gradient_loss;
          test_case "policy gradient loss (no normalize)" `Quick
            test_policy_gradient_loss_no_normalize;
          test_case "PPO clip loss" `Quick test_ppo_clip_loss;
          test_case "value loss" `Quick test_value_loss;
          test_case "value loss (clipped)" `Quick test_value_loss_clipped;
        ] );
      ( "Utilities",
        [
          test_case "normalize" `Quick test_normalize;
          test_case "explained variance" `Quick test_explained_variance;
        ] );
      ("Evaluation", [ test_case "evaluate policy" `Quick test_evaluate ]);
    ]
