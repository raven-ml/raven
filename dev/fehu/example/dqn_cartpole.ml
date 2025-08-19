(** Deep Q-Network (DQN) example using Fehu + Kaun for CartPole *)

open Fehu
module Rng = Rune.Rng

(** DQN Agent using Kaun for the Q-network *)
module DQN = struct
  type t = {
    q_network : Kaun.model;
    mutable q_params : (Rune.float32_elt, [ `c ]) Kaun.params;
    target_network : Kaun.model;
    mutable target_params : (Rune.float32_elt, [ `c ]) Kaun.params;
    optimizer :
      (Rune.float32_elt, [ `c ]) Kaun.Optimizer.gradient_transformation;
    mutable opt_state : (Rune.float32_elt, [ `c ]) Kaun.Optimizer.opt_state;
    buffer : [ `c ] Buffer.t;
    rng : Rng.key;
    mutable epsilon : float;
    epsilon_min : float;
    epsilon_decay : float;
    gamma : float;
    _learning_rate : float;
    batch_size : int;
    target_update_freq : int;
    mutable steps : int;
  }

  let create_q_network obs_dim n_actions =
    (* Simple MLP for Q-function *)
    Kaun.Layer.sequential
      [
        Kaun.Layer.linear ~in_features:obs_dim ~out_features:128 ();
        Kaun.Layer.relu ();
        Kaun.Layer.linear ~in_features:128 ~out_features:128 ();
        Kaun.Layer.relu ();
        Kaun.Layer.linear ~in_features:128 ~out_features:n_actions ();
      ]

  let create ~obs_dim ~n_actions ~buffer_size ~learning_rate ~gamma ~epsilon
      ~epsilon_min ~epsilon_decay ~batch_size ~target_update_freq ~seed =
    let rng = Rng.key seed in
    let keys = Rng.split ~n:2 rng in

    (* Create Q-network and target network *)
    let q_network = create_q_network obs_dim n_actions in
    let target_network = create_q_network obs_dim n_actions in

    (* Initialize parameters *)
    let dummy_input = Rune.zeros Rune.c Rune.float32 [| obs_dim |] in
    let q_params = Kaun.init q_network ~rngs:keys.(0) dummy_input in
    let target_params = Kaun.init target_network ~rngs:keys.(1) dummy_input in

    (* Create optimizer *)
    let optimizer = Kaun.Optimizer.adam ~lr:learning_rate () in
    let opt_state = optimizer.init q_params in

    (* Create replay buffer *)
    let buffer = Buffer.create ~capacity:buffer_size in

    {
      q_network;
      q_params;
      target_network;
      target_params;
      optimizer;
      opt_state;
      buffer;
      rng = keys.(0);
      epsilon;
      epsilon_min;
      epsilon_decay;
      gamma;
      _learning_rate = learning_rate;
      batch_size;
      target_update_freq;
      steps = 0;
    }

  let select_action t obs ~training =
    if training && Random.float 1.0 < t.epsilon then
      (* Exploration: random action *)
      let n_actions =
        match
          Kaun.apply t.q_network t.q_params ~training:false obs |> Rune.shape
        with
        | [| n |] -> n
        | _ -> failwith "Unexpected Q-values shape"
      in
      Rune.scalar Rune.c Rune.float32 (float_of_int (Random.int n_actions))
    else
      (* Exploitation: greedy action *)
      let q_values = Kaun.apply t.q_network t.q_params ~training:false obs in
      let action_idx = Rune.argmax q_values ~axis:(-1) ~keepdims:false in
      Rune.cast Rune.float32 action_idx

  let update_target_network t =
    (* Copy Q-network parameters to target network *)
    t.target_params <- t.q_params;
    Printf.printf "Target network updated at step %d\n" t.steps

  let train_step t =
    if Buffer.size t.buffer < t.batch_size then None
    else
      (* Sample batch from buffer *)
      let batch = Buffer.sample t.buffer ~rng:t.rng ~batch_size:t.batch_size in

      (* Extract batch components *)
      let obs_batch = Array.map (fun tr -> tr.Buffer.obs) batch in
      let action_batch = Array.map (fun tr -> tr.Buffer.action) batch in
      let reward_batch = Array.map (fun tr -> tr.Buffer.reward) batch in
      let next_obs_batch = Array.map (fun tr -> tr.Buffer.next_obs) batch in
      let terminated_batch = Array.map (fun tr -> tr.Buffer.terminated) batch in

      (* Stack observations into batch tensors *)
      let obs_tensor = Rune.stack (Array.to_list obs_batch) ~axis:0 in
      let next_obs_tensor = Rune.stack (Array.to_list next_obs_batch) ~axis:0 in
      let _action_tensor = Rune.stack (Array.to_list action_batch) ~axis:0 in
      let reward_tensor =
        Rune.create Rune.c Rune.float32 [| t.batch_size |] reward_batch
      in
      let terminated_tensor =
        Rune.create Rune.c Rune.float32 [| t.batch_size |]
          (Array.map (fun b -> if b then 1.0 else 0.0) terminated_batch)
      in

      (* Compute loss and update *)
      let loss_fn params =
        (* Current Q-values for taken actions *)
        let q_values =
          Kaun.apply t.q_network params ~training:true obs_tensor
        in

        (* Extract Q-values for taken actions using one-hot encoding *)
        let n_actions =
          match Rune.shape q_values with
          | [| _; n |] -> n
          | _ -> failwith "Unexpected shape"
        in
        let action_indices =
          Array.map
            (fun a ->
              let v = Rune.unsafe_to_array a in
              int_of_float v.(0))
            action_batch
        in
        let one_hot =
          Array.init t.batch_size (fun i ->
              Array.init n_actions (fun j ->
                  if j = action_indices.(i) then 1.0 else 0.0))
        in
        let one_hot_tensor =
          Rune.create Rune.c Rune.float32
            [| t.batch_size; n_actions |]
            (Array.concat (Array.to_list one_hot))
        in
        let current_q_values =
          Rune.sum
            (Rune.mul q_values one_hot_tensor)
            ~axes:[| 1 |] ~keepdims:false
        in

        (* Target Q-values using target network *)
        let next_q_values =
          Kaun.apply t.target_network t.target_params ~training:false
            next_obs_tensor
        in
        let max_next_q = Rune.max next_q_values ~axes:[| 1 |] ~keepdims:false in
        let not_terminated =
          Rune.sub (Rune.scalar Rune.c Rune.float32 1.0) terminated_tensor
        in
        let targets =
          Rune.add reward_tensor
            (Rune.mul
               (Rune.mul (Rune.scalar Rune.c Rune.float32 t.gamma) max_next_q)
               not_terminated)
        in

        (* MSE loss *)
        let diff = Rune.sub current_q_values targets in
        let loss =
          Rune.mean (Rune.mul diff diff) ~axes:[| 0 |] ~keepdims:false
        in
        loss
      in

      let loss, grads = Kaun.value_and_grad loss_fn t.q_params in
      let updated_params, new_opt_state =
        t.optimizer.update t.opt_state t.q_params grads
      in
      t.q_params <- updated_params;
      t.opt_state <- new_opt_state;

      (* Update target network periodically *)
      t.steps <- t.steps + 1;
      if t.steps mod t.target_update_freq = 0 then update_target_network t;

      (* Decay epsilon *)
      t.epsilon <- max t.epsilon_min (t.epsilon *. t.epsilon_decay);

      Some (Rune.unsafe_to_array loss).(0)

  let add_experience t obs action reward next_obs terminated =
    Buffer.add t.buffer { obs; action; reward; next_obs; terminated }
end

(** Training loop *)
let train_dqn () =
  (* Environment *)
  let env = Envs.cartpole () in

  (* Hyperparameters *)
  let obs_dim = 4 in
  let n_actions = 2 in
  let n_episodes = 500 in
  let max_steps_per_episode = 500 in

  (* Create DQN agent *)
  let agent =
    DQN.create ~obs_dim ~n_actions ~buffer_size:10000 ~learning_rate:0.001
      ~gamma:0.99 ~epsilon:1.0 ~epsilon_min:0.01 ~epsilon_decay:0.995
      ~batch_size:32 ~target_update_freq:100 ~seed:42
  in

  Printf.printf "Starting DQN training on CartPole...\n";
  Printf.printf "Episodes: %d, Max steps: %d\n\n" n_episodes
    max_steps_per_episode;

  (* Training loop *)
  let episode_rewards = ref [] in

  for episode = 1 to n_episodes do
    let obs, _ = env.Env.reset () in
    let episode_reward = ref 0.0 in
    let mutable_obs = ref obs in
    let steps_taken = ref 0 in

    try
      for _step = 1 to max_steps_per_episode do
        (* Select action *)
        let action = DQN.select_action agent !mutable_obs ~training:true in

        (* Take step in environment *)
        let next_obs, reward, terminated, truncated, _ = env.Env.step action in

        (* Store experience *)
        DQN.add_experience agent !mutable_obs action reward next_obs terminated;

        (* Train the agent *)
        let _loss = DQN.train_step agent in

        (* Update state *)
        episode_reward := !episode_reward +. reward;
        mutable_obs := next_obs;
        incr steps_taken;

        if terminated || truncated then (
          episode_rewards := !episode_reward :: !episode_rewards;

          (* Print progress every 10 episodes *)
          if episode mod 10 = 0 then (
            let recent_rewards =
              List.take (min 10 (List.length !episode_rewards)) !episode_rewards
            in
            let avg_reward =
              List.fold_left ( +. ) 0.0 recent_rewards
              /. float_of_int (List.length recent_rewards)
            in
            Printf.printf
              "Episode %3d | Reward: %6.1f | Avg(10): %6.1f | Epsilon: %.3f\n"
              episode !episode_reward avg_reward agent.epsilon;
            flush stdout);

          (* Early stopping if solved *)
          (if List.length !episode_rewards >= 100 then
             let last_100 = List.take 100 !episode_rewards in
             let avg_100 = List.fold_left ( +. ) 0.0 last_100 /. 100.0 in
             if avg_100 >= 195.0 then (
               Printf.printf
                 "\n✓ Solved! Average reward over last 100 episodes: %.1f\n"
                 avg_100;
               raise Exit));

          (* Break from step loop *)
          raise Exit)
      done
    with Exit -> ()
  done;

  Printf.printf "\nTraining completed after %d episodes\n" n_episodes

(** Evaluation *)
let _evaluate_agent agent env n_episodes =
  Printf.printf "\nEvaluating trained agent for %d episodes...\n" n_episodes;

  let total_reward = ref 0.0 in

  for episode = 1 to n_episodes do
    let obs, _ = env.Env.reset () in
    let episode_reward = ref 0.0 in
    let mutable_obs = ref obs in
    let finished = ref false in

    while not !finished do
      (* Select action (no exploration) *)
      let action = DQN.select_action agent !mutable_obs ~training:false in

      (* Take step *)
      let next_obs, reward, terminated, truncated, _ = env.step action in

      episode_reward := !episode_reward +. reward;
      mutable_obs := next_obs;
      finished := terminated || truncated
    done;

    Printf.printf "  Episode %d: Reward = %.1f\n" episode !episode_reward;
    total_reward := !total_reward +. !episode_reward
  done;

  let avg_reward = !total_reward /. float_of_int n_episodes in
  Printf.printf "\nAverage evaluation reward: %.1f\n" avg_reward

(** Main *)
let () =
  Random.self_init ();

  try train_dqn ()
  with Exit ->
    Printf.printf "Training stopped early (solved or interrupted)\n"
