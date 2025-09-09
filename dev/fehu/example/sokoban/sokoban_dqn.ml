(** DQN (Deep Q-Network) implementation for Sokoban using Fehu + Kaun *)

open Fehu
module Rng = Rune.Rng

(** Experience replay buffer for DQN *)
module ReplayBuffer = struct
  type transition = {
    state: (float, Rune.float32_elt, [ `c ]) Rune.t;
    action: int;
    reward: float;
    next_state: (float, Rune.float32_elt, [ `c ]) Rune.t;
    is_done: bool;
  }

  type t = {
    buffer: transition array;
    mutable size: int;
    mutable position: int;
    capacity: int;
  }

  let create capacity =
    (* Create dummy transition for initialization *)
    let dummy_state = Rune.zeros Rune.c Rune.float32 [| 1 |] in
    let dummy = {
      state = dummy_state;
      action = 0;
      reward = 0.0;
      next_state = dummy_state;
      is_done = false;
    } in
    {
      buffer = Array.make capacity dummy;
      size = 0;
      position = 0;
      capacity;
    }

  let add t transition =
    t.buffer.(t.position) <- transition;
    t.position <- (t.position + 1) mod t.capacity;
    t.size <- min t.capacity (t.size + 1)

  let sample t batch_size =
    if t.size < batch_size then
      failwith "Not enough samples in replay buffer";
    
    let indices = Array.init batch_size (fun _ -> Random.int t.size) in
    Array.map (fun i -> t.buffer.(i)) indices

  let is_ready t min_size = t.size >= min_size
end

(** DQN Agent *)
module DQNAgent = struct
  type t = {
    q_network: Kaun.module_;
    mutable q_params: (Rune.float32_elt, [ `c ]) Kaun.params;
    target_network: Kaun.module_;
    mutable target_params: (Rune.float32_elt, [ `c ]) Kaun.params;
    optimizer: (Rune.float32_elt, [ `c ]) Kaun.Optimizer.gradient_transformation;
    mutable opt_state: (Rune.float32_elt, [ `c ]) Kaun.Optimizer.opt_state;
    replay_buffer: ReplayBuffer.t;
    gamma: float;
    epsilon: float ref;
    epsilon_decay: float;
    epsilon_min: float;
    _learning_rate: float;
  }

  let create_q_network obs_dim n_actions =
    Kaun.Layer.(
      sequential [
        linear ~in_features:obs_dim ~out_features:256 ();
        relu ();
        linear ~in_features:256 ~out_features:256 ();
        relu ();
        linear ~in_features:256 ~out_features:128 ();
        relu ();
        linear ~in_features:128 ~out_features:n_actions ();
      ]
    )

  let create ~obs_dim ~n_actions ~learning_rate ~gamma ~buffer_size ~seed =
    let rng = Rng.key seed in
    let keys = Rng.split ~n:2 rng in

    let q_network = create_q_network obs_dim n_actions in
    let target_network = create_q_network obs_dim n_actions in
    
    let dummy_input = Rune.zeros Rune.c Rune.float32 [| obs_dim |] in
    let q_params = Kaun.init q_network ~rngs:keys.(0) dummy_input in
    let target_params = Kaun.init target_network ~rngs:keys.(1) dummy_input in

    let optimizer = Kaun.Optimizer.adam ~lr:learning_rate () in
    let opt_state = optimizer.init q_params in

    {
      q_network;
      q_params;
      target_network;
      target_params;
      optimizer;
      opt_state;
      replay_buffer = ReplayBuffer.create buffer_size;
      gamma;
      epsilon = ref 1.0;
      epsilon_decay = 0.995;
      epsilon_min = 0.01;
      _learning_rate = learning_rate;
    }

  let select_action t obs ~training =
    if training && Random.float 1.0 < !(t.epsilon) then
      (* Epsilon-greedy exploration *)
      let n_actions = 
        let q_values = Kaun.apply t.q_network t.q_params ~training:false obs in
        Array.length (Rune.shape q_values) |> fun _ -> 
        Rune.shape q_values |> Array.fold_left (fun _ x -> x) 1
      in
      Random.int n_actions
    else
      (* Greedy action selection *)
      let q_values = Kaun.apply t.q_network t.q_params ~training:false obs in
      let action_idx = Rune.argmax q_values ~axis:(-1) ~keepdims:false in
      int_of_float (Rune.to_array (Rune.cast Rune.float32 action_idx)).(0)

  let update_target_network t =
    t.target_params <- t.q_params

  let train_step t batch_size =
    if not (ReplayBuffer.is_ready t.replay_buffer batch_size) then
      0.0
    else
      let batch = ReplayBuffer.sample t.replay_buffer batch_size in
      
      (* Prepare batch tensors *)
      let states = Array.map (fun trans -> trans.ReplayBuffer.state) batch in
      let actions = Array.map (fun trans -> trans.ReplayBuffer.action) batch in
      let rewards = Array.map (fun trans -> trans.ReplayBuffer.reward) batch in
      let next_states = Array.map (fun trans -> trans.ReplayBuffer.next_state) batch in
      let dones = Array.map (fun trans -> if trans.ReplayBuffer.is_done then 1.0 else 0.0) batch in
      
      (* Stack states into batches *)
      let batch_states = Rune.stack ~axis:0 (Array.to_list states) in
      let batch_next_states = Rune.stack ~axis:0 (Array.to_list next_states) in
      
      (* Compute loss and gradients *)
      let loss_fn params =
        let q_values = Kaun.apply t.q_network params ~training:true batch_states in
        
        (* Get Q-values for taken actions *)
        let batch_q_values = ref [] in
        for i = 0 to batch_size - 1 do
          let q_row = Rune.slice [ Rune.I i; Rune.R [] ] q_values in
          let q_array = Rune.to_array q_row in
          let q_value = q_array.(actions.(i)) in
          batch_q_values := q_value :: !batch_q_values
        done;
        let current_q = Array.of_list (List.rev !batch_q_values) in
        
        (* Compute target Q-values using target network *)
        let target_q_values = Kaun.apply t.target_network t.target_params 
          ~training:false batch_next_states in
        let max_next_q = Rune.max target_q_values ~axes:[| -1 |] ~keepdims:false in
        let max_next_q_array = Rune.to_array max_next_q in
        
        (* Compute TD targets *)
        let targets = Array.mapi (fun i r ->
          r +. t.gamma *. max_next_q_array.(i) *. (1.0 -. dones.(i))
        ) rewards in
        
        (* Compute MSE loss *)
        let total_loss = ref 0.0 in
        for i = 0 to batch_size - 1 do
          let diff = current_q.(i) -. targets.(i) in
          total_loss := !total_loss +. diff *. diff
        done;
        
        Rune.scalar Rune.c Rune.float32 (!total_loss /. float_of_int batch_size)
      in
      
      let (loss_value, grads) = Kaun.value_and_grad loss_fn t.q_params in
      let (updates, new_opt_state) = t.optimizer.update t.opt_state t.q_params grads in
      
      t.q_params <- Kaun.Optimizer.apply_updates t.q_params updates;
      t.opt_state <- new_opt_state;
      
      (* Decay epsilon *)
      t.epsilon := max t.epsilon_min (!(t.epsilon) *. t.epsilon_decay);
      
      Rune.to_array loss_value |> Array.fold_left (+.) 0.0

  let store_transition t state action reward next_state is_done =
    let transition = ReplayBuffer.{
      state;
      action;
      reward;
      next_state;
      is_done;
    } in
    ReplayBuffer.add t.replay_buffer transition
end

(** Training function *)
let train_dqn env ~episodes ~max_steps ~learning_rate ~gamma 
    ~buffer_size ~batch_size ~update_target_freq ~seed ~verbose =
  
  let obs_dim = match env.Env.observation_space with
    | Space.Box { shape; _ } -> shape.(0)
    | _ -> failwith "Expected Box observation space"
  in
  
  let n_actions = match env.Env.action_space with
    | Space.Discrete n -> n
    | _ -> failwith "Expected Discrete action space"
  in
  
  let agent = DQNAgent.create ~obs_dim ~n_actions ~learning_rate 
    ~gamma ~buffer_size ~seed in
  
  let rewards_history = ref [] in
  let wins_history = ref [] in
  let total_wins = ref 0 in
  let total_steps = ref 0 in
  
  Printf.printf "Starting DQN training\n";
  Printf.printf "Environment: Sokoban, Episodes: %d, LR: %.4f\n" 
    episodes learning_rate;
  Printf.printf "Buffer size: %d, Batch size: %d\n" buffer_size batch_size;
  
  for episode = 1 to episodes do
    let obs, info = env.Env.reset () in
    let obs_ref = ref obs in
    let episode_reward = ref 0.0 in
    let episode_length = ref 0 in
    let is_done = ref false in
    let stage_info = ref info in
    
    for _step = 1 to max_steps do
      if not !is_done then begin
        let obs = !obs_ref in
        let action = DQNAgent.select_action agent obs ~training:true in
        let action_tensor = Rune.scalar Rune.c Rune.float32 (float_of_int action) in
        let next_obs, reward, terminated, truncated, info = env.Env.step action_tensor in
        
        episode_reward := !episode_reward +. reward;
        incr episode_length;
        incr total_steps;
        
        (* Store transition in replay buffer *)
        DQNAgent.store_transition agent obs action reward next_obs (terminated || truncated);
        
        (* Update observation *)
        obs_ref := next_obs;
        
        (* Train the network *)
        if !total_steps mod 4 = 0 then
          let _loss = DQNAgent.train_step agent batch_size in
          ();
        
        (* Update target network *)
        if !total_steps mod update_target_freq = 0 then
          DQNAgent.update_target_network agent;
        
        if terminated || truncated then begin
          is_done := true;
          stage_info := info  (* Capture final info with stage *)
        end
      end
    done;
    
    let won = !episode_reward > 50.0 in
    if won then incr total_wins;
    rewards_history := !episode_reward :: !rewards_history;
    wins_history := (if won then 1.0 else 0.0) :: !wins_history;
    
    (* Keep only recent history *)
    if List.length !rewards_history > 100 then begin
      rewards_history := List.filteri (fun i _ -> i < 100) !rewards_history;
      wins_history := List.filteri (fun i _ -> i < 100) !wins_history
    end;
    
    (* Print progress *)
    if verbose && (episode = 1 || episode mod 10 = 0) then begin
      let recent_rewards = !rewards_history in
      let avg_reward = 
        (List.fold_left (+.) 0.0 recent_rewards) /. 
        float_of_int (List.length recent_rewards) in
      let recent_wins = !wins_history in
      let win_rate = 
        (List.fold_left (+.) 0.0 recent_wins) /. 
        float_of_int (List.length recent_wins) in
      
      (* Extract stage information *)
      let stage_str = 
        match List.assoc_opt "stage" !stage_info with
        | Some (`String s) -> s
        | _ -> "N/A"
      in
      
      let advanced = 
        match List.assoc_opt "advanced" !stage_info with
        | Some (`Bool true) -> " (ADVANCED!)"
        | _ -> ""
      in
      
      Printf.printf "Episode %d: Avg Reward = %.2f, Win Rate = %.1f%%, Epsilon = %.3f, Steps = %d, Stage = %s%s\n"
        episode avg_reward (win_rate *. 100.0) !(agent.epsilon) !episode_length stage_str advanced;
      flush stdout
    end
  done;
  
  Printf.printf "Training complete! Final win rate: %.1f%%\n"
    (float_of_int !total_wins /. float_of_int episodes *. 100.0)

(** Main entry point *)
let () =
  (* Test on simple corridor first *)
  Printf.printf "Training DQN on simple corridor...\n";
  let initial_state = Sokoban.LevelGen.generate_corridor 3 in
  let env = Sokoban.sokoban ~width:5 ~height:3 ~max_steps:50 
    ~initial_state () in
  
  train_dqn env
    ~episodes:100
    ~max_steps:50
    ~learning_rate:0.0001
    ~gamma:0.99
    ~buffer_size:10000
    ~batch_size:32
    ~update_target_freq:100
    ~seed:42
    ~verbose:true;
  
  Printf.printf "\nNow training with curriculum learning...\n";
  
  let curriculum_env = Sokoban.sokoban_curriculum ~max_steps:200 () in
  
  train_dqn curriculum_env
    ~episodes:5000
    ~max_steps:200
    ~learning_rate:0.0001
    ~gamma:0.99
    ~buffer_size:50000
    ~batch_size:64
    ~update_target_freq:500
    ~seed:43
    ~verbose:true;
  
  Printf.printf "\nDone!\n"