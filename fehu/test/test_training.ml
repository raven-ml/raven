(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Fehu
open Windtrap

let test_compute_gae () =
  let rewards = [| 1.0; 1.0; 1.0 |] in
  let values = [| 0.0; 0.0; 0.0 |] in
  let dones = [| false; false; true |] in
  let advantages, returns =
    Training.compute_gae ~rewards ~values ~dones ~last_value:0.0 ~last_done:true
      ~gamma:0.99 ~gae_lambda:0.95
  in

  equal ~msg:"advantages length" int 3 (Array.length advantages);
  equal ~msg:"returns length" int 3 (Array.length returns);

  (* All advantages should be positive since rewards are positive *)
  Array.iter
    (fun adv -> equal ~msg:"advantage > 0" bool true (adv > 0.0))
    advantages

let test_compute_gae_bootstrap () =
  let rewards = [| 1.0 |] in
  let values = [| 0.5 |] in
  let dones = [| false |] in
  let advantages, returns =
    Training.compute_gae ~rewards ~values ~dones ~last_value:0.25
      ~last_done:false ~gamma:0.99 ~gae_lambda:1.0
  in
  let expected_adv = 1.0 +. (0.99 *. 0.25) -. 0.5 in
  let expected_return = expected_adv +. 0.5 in
  equal ~msg:"bootstrapped advantage" (float 1e-6)
    expected_adv advantages.(0);
  equal ~msg:"bootstrapped return" (float 1e-6)
    expected_return returns.(0)

let test_compute_returns () =
  let rewards = [| 1.0; 1.0; 1.0 |] in
  let dones = [| false; false; true |] in
  let returns = Training.compute_returns ~rewards ~dones ~gamma:0.99 in

  equal ~msg:"returns length" int 3 (Array.length returns);

  (* First return should be largest (sum of all future rewards) *)
  equal ~msg:"returns[0] >= returns[1]" bool true
    (returns.(0) >= returns.(1));
  equal ~msg:"returns[1] >= returns[2]" bool true
    (returns.(1) >= returns.(2))

let test_normalize () =
  let arr = [| 1.0; 2.0; 3.0; 4.0; 5.0 |] in
  let normalized = Training.normalize arr () in

  equal ~msg:"normalized length" int 5 (Array.length normalized);

  (* Check mean is close to 0 *)
  let sum = Array.fold_left ( +. ) 0.0 normalized in
  let mean = sum /. 5.0 in
  equal ~msg:"mean ~= 0" bool true (abs_float mean < 1e-6);

  (* Check std is close to 1 *)
  let var_sum = Array.fold_left (fun acc x -> acc +. (x *. x)) 0.0 normalized in
  let std = sqrt (var_sum /. 5.0) in
  equal ~msg:"std ~= 1" bool true (abs_float (std -. 1.0) < 1e-6)

let test_policy_gradient_loss () =
  let log_probs = [| -0.5; -0.3; -0.7 |] in
  let advantages = [| 1.0; -1.0; 0.5 |] in

  let loss = Training.policy_gradient_loss ~log_probs ~advantages () in

  (* Loss should be finite *)
  equal ~msg:"loss is finite" bool true (not (Float.is_nan loss));
  equal ~msg:"loss is finite" bool true (not (Float.is_infinite loss))

let test_policy_gradient_loss_no_normalize () =
  let log_probs = [| -0.5; -0.3; -0.7 |] in
  let advantages = [| 1.0; -1.0; 0.5 |] in

  let loss =
    Training.policy_gradient_loss ~log_probs ~advantages ~normalize:false ()
  in

  equal ~msg:"loss is finite" bool true (not (Float.is_nan loss));
  equal ~msg:"loss is finite" bool true (not (Float.is_infinite loss))

let test_ppo_clip_loss () =
  let log_probs = [| -0.5; -0.3; -0.7 |] in
  let old_log_probs = [| -0.6; -0.4; -0.8 |] in
  let advantages = [| 1.0; -1.0; 0.5 |] in

  let loss =
    Training.ppo_clip_loss ~log_probs ~old_log_probs ~advantages ~clip_range:0.2
  in

  equal ~msg:"loss is finite" bool true (not (Float.is_nan loss));
  equal ~msg:"loss is finite" bool true (not (Float.is_infinite loss))

let test_value_loss () =
  let values = [| 1.0; 2.0; 3.0 |] in
  let returns = [| 1.5; 2.5; 2.8 |] in

  let loss = Training.value_loss ~values ~returns () in

  equal ~msg:"loss >= 0" bool true (loss >= 0.0);
  equal ~msg:"loss is finite" bool true (not (Float.is_nan loss))

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

  equal ~msg:"clipped value loss matches expected" (float 1e-6) expected loss;

  raises ~msg:"clip_range should be positive"
    (Invalid_argument "Training.value_loss: clip_range must be non-negative")
    (fun () ->
      ignore (Training.value_loss ~values ~returns ~clip:(-0.2, old_values) ()));

  let old_values_small = [| 1.0; 3.0 |] in
  raises ~msg:"old_values should be have the same shape"
    (Invalid_argument
       "Training.value_loss: old_values must have same length as arrays")
    (fun () ->
      ignore
        (Training.value_loss ~values ~returns
           ~clip:(clip_range, old_values_small)
           ()));

  (*Change the shape of returns and check it raises error*)
  let returns = [| 1.1; 2.7 |] in
  raises ~msg:"returns and values should be have the same shape"
    (Invalid_argument "Training.value_loss: arrays must have same length")
    (fun () ->
      ignore
        (Training.value_loss ~values ~returns ~clip:(clip_range, old_values) ()))

let test_explained_variance () =
  let y_pred = [| 1.0; 2.0; 3.0; 4.0 |] in
  let y_true = [| 1.1; 1.9; 3.1; 3.9 |] in

  let ev = Training.explained_variance ~y_pred ~y_true in

  (* Explained variance should be between -inf and 1 *)
  equal ~msg:"ev <= 1" bool true (ev <= 1.0);
  equal ~msg:"ev is finite" bool true (not (Float.is_nan ev));

  (* For good predictions, should be close to 1 *)
  equal ~msg:"ev > 0.9" bool true (ev > 0.9)

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

  equal ~msg:"n_episodes" int 5 stats.n_episodes;
  equal ~msg:"mean_reward" (float 0.0001) 1.0 stats.mean_reward;
  equal ~msg:"mean_length > 0" bool true (stats.mean_length > 0.0)

let test_evaluate_updates_observation () =
  let rng = Rune.Rng.key 123 in
  let obs_space = Space.Box.create ~low:[| 0.0 |] ~high:[| 5.0 |] in
  let act_space = Space.Discrete.create 1 in
  let state = ref 0 in
  let env =
    Env.create ~rng ~observation_space:obs_space ~action_space:act_space
      ~reset:(fun _ ?options:_ () ->
        state := 0;
        let obs = Rune.create Rune.float32 [| 1 |] [| float_of_int !state |] in
        (obs, Info.empty))
      ~step:(fun _ _ ->
        state := !state + 1;
        let terminated = !state >= 3 in
        let obs = Rune.create Rune.float32 [| 1 |] [| float_of_int !state |] in
        Env.transition ~observation:obs ~reward:1.0 ~terminated ())
      ()
  in
  let seen = ref [] in
  let policy obs =
    let arr = Rune.to_array (Rune.reshape [| 1 |] obs) in
    seen := arr.(0) :: !seen;
    Rune.create Rune.int32 [| 1 |] [| 0l |]
  in
  ignore (Training.evaluate env ~policy ~n_episodes:1 ~max_steps:10 ());
  let observed = List.rev !seen in
  equal ~msg:"policy saw evolving observations" (list (float 1e-6))
    [ 0.0; 1.0; 2.0 ] observed

let () =
  run "Training"
    [
      group "Advantage Estimation"
        [
          test "compute GAE" test_compute_gae;
          test "compute GAE with bootstrap"
            test_compute_gae_bootstrap;
          test "compute returns" test_compute_returns;
        ];
      group "Loss Functions"
        [
          test "policy gradient loss" test_policy_gradient_loss;
          test "policy gradient loss (no normalize)"
            test_policy_gradient_loss_no_normalize;
          test "PPO clip loss" test_ppo_clip_loss;
          test "value loss" test_value_loss;
          test "value loss (clipped)" test_value_loss_clipped;
        ];
      group "Utilities"
        [
          test "normalize" test_normalize;
          test "explained variance" test_explained_variance;
        ];
      group "Evaluation"
        [
          test "evaluate policy" test_evaluate;
          test "evaluate updates observation"
            test_evaluate_updates_observation;
        ];
    ]
