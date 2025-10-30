open Fehu
open Kaun
module FV = Fehu_visualize
module Policy = Fehu.Policy

let q_scores q_net params obs =
  let obs_f = Rune.cast Rune.float32 obs in
  let state_tensor = Rune.reshape [| 1; 2 |] obs_f in
  let q_values = Kaun.apply q_net params ~training:false state_tensor in
  Rune.to_array (Rune.reshape [| 4 |] q_values)

let record_guard description f =
  try f ()
  with Invalid_argument msg ->
    Printf.eprintf "Warning: %s (%s)\n%!" description msg

let record_random_rollout ~path ~steps =
  let env =
    Fehu_envs.Grid_world.make ~rng:(Rune.Rng.key 73) ~render_mode:`Rgb_array ()
  in
  let policy = Policy.random env in
  FV.Sink.with_ffmpeg ~fps:6 ~path (fun sink ->
      FV.record_rollout ~env ~policy ~steps ~sink ());
  Env.close env

let record_trained_rollout ~path ~steps ~q_net ~params =
  let env =
    Fehu_envs.Grid_world.make ~rng:(Rune.Rng.key 137) ~render_mode:`Rgb_array ()
  in
  let policy =
    Policy.greedy_discrete env ~score:(fun obs -> q_scores q_net params obs)
  in
  FV.Sink.with_ffmpeg ~fps:6 ~path (fun sink ->
      FV.record_rollout ~env ~policy ~steps ~sink ());
  Env.close env

(* Q-network: takes state (flattened grid position) and outputs Q-values for
   each action *)
let create_q_network () =
  Layer.sequential
    [
      Layer.linear ~in_features:2 ~out_features:64 ();
      Layer.relu ();
      Layer.linear ~in_features:64 ~out_features:64 ();
      Layer.relu ();
      Layer.linear ~in_features:64 ~out_features:4 ();
      (* 4 actions *)
    ]

let run_dqn () =
  (* Create RNG *)
  let rngs = Rune.Rng.key 42 in

  (* Create Q-network and target network *)
  let q_net = create_q_network () in
  let target_net = create_q_network () in

  let params = Kaun.init q_net ~rngs ~dtype:Rune.float32 in
  let target_params = ref (Ptree.copy params) in

  (* Optimizer *)
  let lr = Optimizer.Schedule.constant 0.001 in
  let optimizer = Optimizer.adam ~lr () in
  let opt_state = ref (Optimizer.init optimizer params) in

  (* Experience replay *)
  let replay_buffer = Buffer.Replay.create ~capacity:10_000 in

  (* Training parameters *)
  let episodes = 500 in
  let batch_size = 32 in
  let gamma = 0.99 in
  let epsilon_start = 1.0 in
  let epsilon_end = 0.01 in
  let epsilon_decay = 1000.0 in
  let target_update_freq = 10 in

  (* Create environment and agent RNGs *)
  let root_key = Rune.Rng.key 3 in
  let split_keys = Rune.Rng.split root_key in
  let env = Fehu_envs.Grid_world.make ~rng:split_keys.(0) () in
  let agent_key = ref split_keys.(1) in

  let take_key () =
    let next = Rune.Rng.split !agent_key in
    agent_key := next.(0);
    next.(1)
  in

  let epsilon_schedule =
    Optimizer.Schedule.polynomial_decay ~init_value:epsilon_start
      ~end_value:epsilon_end ~power:1.0
      ~decay_steps:(int_of_float epsilon_decay)
  in
  let epsilon_step = ref 0 in
  let epsilon_rng = ref (take_key ()) in
  let random_policy = Policy.random env in
  let last_epsilon = ref (epsilon_schedule 0) in

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

  Printf.printf "Training DQN on GridWorld...\n%!";
  Printf.printf "Episodes: %d, Batch size: %d, Replay capacity: 10000\n%!"
    episodes batch_size;

  (* Training loop *)
  for episode = 1 to episodes do
    let obs, _info = Env.reset env () in
    let current_obs = ref obs in
    let done_ = ref false in
    let episode_reward = ref 0.0 in
    let steps = ref 0 in

    let greedy_policy observation =
      let scores = q_scores q_net params observation in
      let best_idx = ref 0 in
      let best_val = ref scores.(0) in
      for i = 1 to Array.length scores - 1 do
        if scores.(i) > !best_val then (
          best_idx := i;
          best_val := scores.(i))
      done;
      Rune.scalar Rune.int32 (Int32.of_int !best_idx)
    in

    while not !done_ do
      incr steps;

      let epsilon = epsilon_schedule !epsilon_step in
      let keys = Rune.Rng.split ~n:2 !epsilon_rng in
      epsilon_rng := keys.(0);
      let sample_key = keys.(1) in
      let coin =
        Rune.Rng.uniform sample_key Rune.float32 [| 1 |]
        |> Rune.to_array |> fun arr -> arr.(0)
      in
      let action =
        if coin < epsilon then
          let action, _, _ = random_policy !current_obs in
          action
        else greedy_policy !current_obs
      in
      last_epsilon := epsilon;
      incr epsilon_step;

      (* Take action *)
      let transition = Env.step env action in
      let reward = transition.reward in
      let terminated = transition.terminated in
      let truncated = transition.truncated in

      episode_reward := !episode_reward +. reward;
      done_ := terminated || truncated;

      (* Store experience - Fehu's Buffer.Replay uses tensors *)
      Buffer.Replay.add replay_buffer
        Buffer.
          {
            observation = !current_obs;
            action;
            reward;
            next_observation = transition.observation;
            terminated;
            truncated;
          };
      current_obs := transition.observation;

      (* Train if enough experiences *)
      if Buffer.Replay.size replay_buffer >= batch_size then (
        let ( observations_t,
              actions_t,
              rewards_t,
              next_obs_t,
              terminated_t,
              truncated_t ) =
          Buffer.Replay.sample_tensors replay_buffer ~rng:(take_key ())
            ~batch_size
        in
        let batch_len =
          match Rune.shape rewards_t with
          | [| len |] -> len
          | _ -> invalid_arg "sample_tensors: rewards tensor must be rank-1"
        in
        let _loss, grads =
          value_and_grad
            (fun params ->
              let observations_f = Rune.cast Rune.float32 observations_t in
              let next_obs_f = Rune.cast Rune.float32 next_obs_t in
              let q_values =
                Kaun.apply q_net params ~training:true observations_f
              in
              let action_indices = Rune.reshape [| batch_len; 1 |] actions_t in
              let chosen_q =
                Rune.take_along_axis ~axis:1 action_indices q_values
                |> Rune.squeeze ~axes:[ 1 ]
              in
              let next_q_values =
                Kaun.apply target_net !target_params ~training:false next_obs_f
              in
              let max_next_q = Rune.max next_q_values ~axes:[ 1 ] in
              let done_mask = Rune.logical_or terminated_t truncated_t in
              let done_float = Rune.cast Rune.float32 done_mask in
              let not_done =
                let ones = Rune.full_like done_float 1.0 in
                Rune.sub ones done_float
              in
              let discount =
                Rune.mul
                  (Rune.scalar Rune.float32 gamma)
                  (Rune.mul not_done max_next_q)
              in
              let target_q = Rune.add rewards_t discount in
              let td_error = Rune.sub chosen_q target_q in
              Rune.mean (Rune.square td_error))
            params
        in

        (* Update Q-network *)
        let updates, new_state =
          Optimizer.step optimizer !opt_state params grads
        in
        opt_state := new_state;
        Optimizer.apply_updates_inplace params updates)
    done;

    (* Update target network *)
    if episode mod target_update_freq = 0 then
      target_params := Ptree.copy params;

    if episode mod 50 = 0 then
      Printf.printf "Episode %d: Reward = %.2f, Steps = %d, Epsilon = %.3f\n%!"
        episode !episode_reward !steps !last_epsilon
  done;

  (* Evaluate learned policy *)
  Printf.printf "\n=== Evaluation ===\n%!";
  let eval_episodes = 20 in
  let total_rewards = ref 0.0 in
  let eval_policy =
    Policy.greedy_discrete env ~score:(fun obs -> q_scores q_net params obs)
  in

  for episode = 1 to eval_episodes do
    let obs, _info = Env.reset env () in
    let current_obs = ref obs in
    let done_ = ref false in
    let episode_reward = ref 0.0 in

    while not !done_ do
      let action, _, _ = eval_policy !current_obs in
      let transition = Env.step env action in
      episode_reward := !episode_reward +. transition.reward;
      done_ := transition.terminated || transition.truncated;
      if not !done_ then current_obs := transition.observation
    done;

    total_rewards := !total_rewards +. !episode_reward;
    Printf.printf "Eval episode %d: reward = %.2f\n%!" episode !episode_reward
  done;

  Printf.printf "Average evaluation reward: %.2f\n%!"
    (!total_rewards /. float_of_int eval_episodes);

  Option.iter
    (fun path ->
      Printf.printf "Recording trained rollout to %s\n%!" path;
      record_guard "recording trained rollout" (fun () ->
          record_trained_rollout ~path ~steps:100 ~q_net ~params))
    (record_path "gridworld_trained.mp4");

  Env.close env

let () = run_dqn ()
