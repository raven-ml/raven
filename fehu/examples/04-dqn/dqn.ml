open Fehu
open Kaun

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

(* Use Fehu's built-in replay buffer *)

let float_state obs =
  let arr : Int32.t array = Rune.to_array obs in
  Array.map (fun x -> float_of_int (Int32.to_int x)) arr

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
  let last_epsilon = ref epsilon_start in

  Printf.printf "Training DQN on GridWorld...\n%!";
  Printf.printf "Episodes: %d, Batch size: %d, Replay capacity: 10000\n%!"
    episodes batch_size;

  (* Training loop *)
  for episode = 1 to episodes do
    let obs, _info = Env.reset env () in
    let state = ref (float_state obs) in
    let done_ = ref false in
    let episode_reward = ref 0.0 in
    let steps = ref 0 in

    while not !done_ do
      incr steps;

      (* Epsilon-greedy exploration *)
      let epsilon =
        epsilon_end
        +. (epsilon_start -. epsilon_end)
           *. exp (-.float_of_int ((episode * 10) + !steps) /. epsilon_decay)
      in
      last_epsilon := epsilon;

      let action_index =
        if
          let sample =
            Rune.Rng.uniform (take_key ()) Rune.float32 [| 1 |] |> Rune.to_array
          in
          sample.(0) < epsilon
        then
          let tensor = Rune.Rng.randint (take_key ()) ~min:0 ~max:4 [| 1 |] in
          let values : Int32.t array = Rune.to_array tensor in
          Int32.to_int values.(0)
        else
          let state_tensor = Rune.create Rune.float32 [| 1; 2 |] !state in
          let q_values = Kaun.apply q_net params ~training:false state_tensor in
          let q0 = Rune.item [ 0; 0 ] q_values in
          let q1 = Rune.item [ 0; 1 ] q_values in
          let q2 = Rune.item [ 0; 2 ] q_values in
          let q3 = Rune.item [ 0; 3 ] q_values in
          let best = ref 0 in
          let best_val = ref q0 in
          let candidates = [| q0; q1; q2; q3 |] in
          for i = 1 to Array.length candidates - 1 do
            if candidates.(i) > !best_val then (
              best := i;
              best_val := candidates.(i))
          done;
          !best
      in
      let action = Rune.scalar Rune.int32 (Int32.of_int action_index) in

      (* Take action *)
      let transition = Env.step env action in
      let reward = transition.reward in
      let terminated = transition.terminated in
      let truncated = transition.truncated in
      let next_state = float_state transition.observation in

      episode_reward := !episode_reward +. reward;
      done_ := terminated || truncated;

      (* Store experience - Fehu's Buffer.Replay uses tensors *)
      let obs_tensor = Rune.create Rune.float32 [| 2 |] !state in
      let action_tensor = Rune.scalar Rune.int32 (Int32.of_int action_index) in
      let next_obs_tensor = Rune.create Rune.float32 [| 2 |] next_state in

      Buffer.Replay.add replay_buffer
        Buffer.
          {
            observation = obs_tensor;
            action = action_tensor;
            reward;
            next_observation = next_obs_tensor;
            terminated;
            truncated;
          };

      state := next_state;

      (* Train if enough experiences *)
      if Buffer.Replay.size replay_buffer >= batch_size then (
        let batch =
          Buffer.Replay.sample replay_buffer ~rng:(take_key ()) ~batch_size
        in

        let _loss, grads =
          value_and_grad
            (fun params ->
              let total_loss = ref 0.0 in

              Array.iter
                (fun Buffer.
                       {
                         observation;
                         action;
                         reward;
                         next_observation;
                         terminated;
                         _;
                       } ->
                  (* Current Q-value *)
                  let obs_batch = Rune.reshape [| 1; 2 |] observation in
                  let q_values =
                    Kaun.apply q_net params ~training:false obs_batch
                  in
                  let act_idx = Int32.to_int (Rune.to_array action).(0) in
                  let current_q = Rune.item [ 0; act_idx ] q_values in

                  (* Target Q-value *)
                  let target_q =
                    if terminated then reward
                    else
                      let next_obs_batch =
                        Rune.reshape [| 1; 2 |] next_observation
                      in
                      let next_q_values =
                        Kaun.apply target_net !target_params ~training:false
                          next_obs_batch
                      in
                      let q0 = Rune.item [ 0; 0 ] next_q_values in
                      let q1 = Rune.item [ 0; 1 ] next_q_values in
                      let q2 = Rune.item [ 0; 2 ] next_q_values in
                      let q3 = Rune.item [ 0; 3 ] next_q_values in
                      let max_next_q = max (max q0 q1) (max q2 q3) in
                      reward +. (gamma *. max_next_q)
                  in

                  let diff = current_q -. target_q in
                  total_loss := !total_loss +. (diff *. diff))
                batch;

              let loss_val = !total_loss /. float_of_int (Array.length batch) in
              Rune.create Rune.float32 [||] [| loss_val |])
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

  for episode = 1 to eval_episodes do
    let obs, _info = Env.reset env () in
    let state = ref (float_state obs) in
    let done_ = ref false in
    let episode_reward = ref 0.0 in

    while not !done_ do
      (* Greedy policy *)
      let state_tensor = Rune.create Rune.float32 [| 1; 2 |] !state in
      let q_values = Kaun.apply q_net params ~training:false state_tensor in
      let q0 = Rune.item [ 0; 0 ] q_values in
      let q1 = Rune.item [ 0; 1 ] q_values in
      let q2 = Rune.item [ 0; 2 ] q_values in
      let q3 = Rune.item [ 0; 3 ] q_values in
      let action =
        let best = ref 0 in
        let best_val = ref q0 in
        let candidates = [| q0; q1; q2; q3 |] in
        for i = 1 to Array.length candidates - 1 do
          if candidates.(i) > !best_val then (
            best := i;
            best_val := candidates.(i))
        done;
        !best
      in

      let transition =
        Env.step env (Rune.scalar Rune.int32 (Int32.of_int action))
      in
      episode_reward := !episode_reward +. transition.reward;
      done_ := transition.terminated || transition.truncated;
      if not !done_ then state := float_state transition.observation
    done;

    total_rewards := !total_rewards +. !episode_reward;
    Printf.printf "Eval episode %d: reward = %.2f\n%!" episode !episode_reward
  done;

  Printf.printf "Average evaluation reward: %.2f\n%!"
    (!total_rewards /. float_of_int eval_episodes);

  Env.close env

let () = run_dqn ()
