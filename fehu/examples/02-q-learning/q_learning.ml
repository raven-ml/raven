(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Fehu

(* Discretize the continuous position space for Q-learning *)
let discretize_position pos =
  (* Map [-10, 10] to discrete states 0-20 *)
  let discretized = int_of_float ((pos +. 10.0) /. 1.0) in
  max 0 (min 20 discretized)

let state_of_observation obs =
  let values = Rune.to_array obs in
  if Array.length values <> 1 then
    invalid_arg "RandomWalk observation must have length 1"
  else discretize_position values.(0)

let run_q_learning () =
  (* Q-learning parameters *)
  let alpha = 0.2 in
  (* learning rate (increased for faster learning) *)
  let gamma = 0.95 in
  (* discount factor (reduced to value immediate rewards more) *)
  let epsilon_start = 1.0 in
  (* initial exploration rate *)
  let epsilon_end = 0.01 in
  (* final exploration rate *)
  let epsilon_decay = 0.998 in
  (* exploration decay rate (slower decay) *)
  let episodes = 5_000 in

  (* Q-table: 21 states (discretized positions) x 2 actions *)
  let q_table = Array.make_matrix 21 2 0.0 in

  (* Create environment *)
  let root_key = Rune.Rng.key 1 in
  let split_keys = Rune.Rng.split root_key in
  let env = Fehu_envs.Random_walk.make ~rng:split_keys.(0) () in
  let agent_key = ref split_keys.(1) in

  let take_key () =
    let next = Rune.Rng.split !agent_key in
    agent_key := next.(0);
    next.(1)
  in

  let sample_uniform () =
    let tensor = Rune.rand Rune.float32 ~key:(take_key ()) [| 1 |] in
    let values = Rune.to_array tensor in
    values.(0)
  in

  let sample_action () =
    let tensor = Rune.randint Rune.int32 ~key:(take_key ()) ~high:2 [| 1 |] 0 in
    let values : Int32.t array = Rune.to_array tensor in
    Int32.to_int values.(0)
  in

  (* Training loop *)
  let epsilon = ref epsilon_start in

  for episode = 1 to episodes do
    (* Decay epsilon *)
    epsilon := Float.max epsilon_end (!epsilon *. epsilon_decay);

    (* Reset environment *)
    let obs, _info = Env.reset env () in
    let state = ref (state_of_observation obs) in
    let done_ = ref false in
    let steps = ref 0 in

    while not !done_ do
      incr steps;

      (* Epsilon-greedy action selection *)
      let action_index =
        if sample_uniform () < !epsilon then sample_action () (* explore *)
        else if q_table.(!state).(0) > q_table.(!state).(1) then 0
        else 1
      in
      let action = Rune.scalar Rune.int32 (Int32.of_int action_index) in

      (* Take action *)
      let transition = Env.step env action in
      let reward = transition.reward in
      let terminated = transition.terminated in
      let truncated = transition.truncated in
      let next_state = state_of_observation transition.observation in

      done_ := terminated || truncated;

      (* Q-learning update *)
      let max_next_q =
        if terminated || truncated then 0.0
        else max q_table.(next_state).(0) q_table.(next_state).(1)
      in
      let target = reward +. (gamma *. max_next_q) in
      q_table.(!state).(action_index) <-
        q_table.(!state).(action_index)
        +. (alpha *. (target -. q_table.(!state).(action_index)));

      state := next_state
    done;

    if episode mod 500 = 0 then
      Printf.printf "Episode %d: steps=%d, epsilon=%.3f\n%!" episode !steps
        !epsilon
  done;

  (* Evaluate learned policy *)
  Printf.printf "\n=== Evaluation ===\n%!";
  let eval_episodes = 10 in
  let total_rewards = ref 0.0 in
  let total_steps = ref 0 in

  for episode = 1 to eval_episodes do
    let obs, _info = Env.reset env () in
    let state = ref (state_of_observation obs) in
    let done_ = ref false in
    let episode_reward = ref 0.0 in
    let steps = ref 0 in

    while not !done_ do
      incr steps;
      (* Greedy policy *)
      let action_index =
        if q_table.(!state).(0) > q_table.(!state).(1) then 0 else 1
      in
      let transition =
        Env.step env (Rune.scalar Rune.int32 (Int32.of_int action_index))
      in
      episode_reward := !episode_reward +. transition.reward;
      done_ := transition.terminated || transition.truncated;

      if not !done_ then state := state_of_observation transition.observation
    done;

    total_rewards := !total_rewards +. !episode_reward;
    total_steps := !total_steps + !steps;
    Printf.printf "Eval episode %d: reward = %.2f, steps = %d\n%!" episode
      !episode_reward !steps
  done;

  Printf.printf "Average evaluation reward: %.2f (avg steps: %.1f)\n%!"
    (!total_rewards /. float_of_int eval_episodes)
    (float_of_int !total_steps /. float_of_int eval_episodes);

  (* Print learned Q-table *)
  Printf.printf "\nLearned Q-table:\n%!";
  for state = 0 to 20 do
    let pos = float_of_int state -. 10.0 in
    let best_action =
      if q_table.(state).(0) > q_table.(state).(1) then "←" else "→"
    in
    Printf.printf "Position %+.1f: Left=%.3f, Right=%.3f [%s]\n%!" pos
      q_table.(state).(0)
      q_table.(state).(1)
      best_action
  done;

  (* Verify the learned policy makes sense *)
  Printf.printf "\n=== Policy Summary ===\n%!";
  let correct_actions = ref 0 in
  for state = 0 to 20 do
    let pos = float_of_int state -. 10.0 in
    let learned_action =
      if q_table.(state).(0) > q_table.(state).(1) then 0 else 1
    in
    (* Optimal policy: go toward 0 *)
    let optimal_action = if pos < 0.0 then 1 else 0 in
    (* negative: go right, positive: go left *)
    if learned_action = optimal_action then incr correct_actions
  done;
  Printf.printf "Learned policy matches optimal for %d/21 states (%.1f%%)\n%!"
    !correct_actions
    (float_of_int !correct_actions /. 21.0 *. 100.0);

  Env.close env

let () = run_q_learning ()
