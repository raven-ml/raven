(** GRPO (Group Relative Policy Optimization) implementation for Sokoban *)

open Fehu
module Rng = Rune.Rng

(** GRPO agent using Kaun for policy and reference networks *)
module GrpoAgent = struct
  type t = {
    policy_network : Kaun.module_;
    mutable policy_params : (Rune.float32_elt, [ `c ]) Kaun.params;
    reference_network : Kaun.module_;
    mutable reference_params : (Rune.float32_elt, [ `c ]) Kaun.params;
    optimizer :
      (Rune.float32_elt, [ `c ]) Kaun.Optimizer.gradient_transformation;
    mutable opt_state : (Rune.float32_elt, [ `c ]) Kaun.Optimizer.opt_state;
    _rng : Rng.key;
    gamma : float;
    _learning_rate : float;
    beta : float;
    batch_size : int;
    update_ref_freq : int;
    mutable updates : int;
  }

  let create_network obs_dim n_actions =
    Kaun.Layer.sequential
      [
        Kaun.Layer.linear ~in_features:obs_dim ~out_features:128 ();
        Kaun.Layer.relu ();
        Kaun.Layer.linear ~in_features:128 ~out_features:64 ();
        Kaun.Layer.relu ();
        Kaun.Layer.linear ~in_features:64 ~out_features:n_actions ();
      ]

  let create ~obs_dim ~n_actions ~learning_rate ~gamma ~beta ~batch_size 
      ~update_ref_freq ~seed =
    let rng = Rng.key seed in
    let keys = Rng.split ~n:2 rng in

    let policy_network = create_network obs_dim n_actions in
    let reference_network = create_network obs_dim n_actions in

    let device = Rune.c in
    let policy_params = Kaun.init policy_network ~rngs:keys.(0) ~device ~dtype:Rune.float32 in
    let reference_params = Kaun.init reference_network ~rngs:keys.(1) ~device ~dtype:Rune.float32 in

    let optimizer = Kaun.Optimizer.adam ~lr:learning_rate () in
    let opt_state = optimizer.init policy_params in

    {
      policy_network;
      policy_params;
      reference_network;
      reference_params;
      optimizer;
      opt_state;
      _rng = keys.(0);
      gamma;
      _learning_rate = learning_rate;
      beta;
      batch_size;
      update_ref_freq;
      updates = 0;
    }

  let get_log_probs network params obs action =
    let logits = Kaun.apply network params ~training:false obs in
    (* Compute log_softmax manually *)
    let max_logits = Rune.max logits ~axes:[| -1 |] ~keepdims:true in
    let exp_logits = Rune.exp (Rune.sub logits max_logits) in
    let sum_exp = Rune.sum exp_logits ~axes:[| -1 |] ~keepdims:true in
    let log_probs = Rune.sub logits (Rune.add max_logits (Rune.log sum_exp)) in
    let action_idx = int_of_float (Rune.to_array action).(0) in
    let log_prob_array = Rune.to_array log_probs in
    log_prob_array.(action_idx)

  let select_action _t obs ~network ~params ~training =
    let logits = Kaun.apply network params ~training obs in
    
    if training then
      let probs = Rune.softmax logits ~axes:[| -1 |] in
      let probs_array = Rune.to_array probs in
      
      let r = Random.float 1.0 in
      let rec sample_idx i cumsum =
        if i >= Array.length probs_array - 1 then i
        else if r <= cumsum +. probs_array.(i) then i
        else sample_idx (i + 1) (cumsum +. probs_array.(i))
      in
      
      let action_idx = sample_idx 0 0.0 in
      let action = Rune.scalar Rune.c Rune.float32 (float_of_int action_idx) in
      
      (* Compute log_softmax manually *)
      let max_logits = Rune.max logits ~axes:[| -1 |] ~keepdims:true in
      let exp_logits = Rune.exp (Rune.sub logits max_logits) in
      let sum_exp = Rune.sum exp_logits ~axes:[| -1 |] ~keepdims:true in
      let log_probs = Rune.sub logits (Rune.add max_logits (Rune.log sum_exp)) in
      let log_prob_array = Rune.to_array log_probs in
      let log_prob = log_prob_array.(action_idx) in
      
      (action, log_prob)
    else
      let action_idx = Rune.argmax logits ~axis:(-1) ~keepdims:false in
      let action = Rune.cast Rune.float32 action_idx in
      (action, 0.0)

  let collect_episodes t env n_episodes max_steps =
    let trajectories = ref [] in
    
    for _ = 1 to n_episodes do
      let obs_ref = ref (fst (env.Env.reset ())) in
      let states = ref [] in
      let actions = ref [] in
      let rewards = ref [] in
      let log_probs = ref [] in
      
      let finished = ref false in
      let steps = ref 0 in
      
      while not !finished && !steps < max_steps do
        let obs = !obs_ref in
        let action, log_prob = 
          select_action t obs ~network:t.policy_network 
            ~params:t.policy_params ~training:true in
        
        let next_obs, reward, terminated, truncated, _ = env.Env.step action in
        
        states := obs :: !states;
        actions := action :: !actions;
        rewards := reward :: !rewards;
        log_probs := log_prob :: !log_probs;
        
        obs_ref := next_obs;
        finished := terminated || truncated;
        incr steps
      done;
      
      let trajectory = Trajectory.create
        ~states:(Array.of_list (List.rev !states))
        ~actions:(Array.of_list (List.rev !actions))
        ~rewards:(Array.of_list (List.rev !rewards))
        ~log_probs:(Some (Array.of_list (List.rev !log_probs)))
        ()
      in
      
      trajectories := trajectory :: !trajectories
    done;
    
    !trajectories

  let compute_group_advantages trajectories gamma =
    let all_returns = ref [] in
    let all_advantages = ref [] in
    
    List.iter (fun traj ->
      let returns = Training.compute_returns 
        ~rewards:traj.Trajectory.rewards
        ~dones:(Array.make (Array.length traj.rewards) false)
        ~gamma
      in
      all_returns := returns :: !all_returns
    ) trajectories;
    
    let flat_returns = Array.concat !all_returns in
    let mean_return = Array.fold_left (+.) 0.0 flat_returns /. 
                      float_of_int (Array.length flat_returns) in
    let std_return = 
      let variance = Array.fold_left (fun acc r ->
        acc +. (r -. mean_return) ** 2.0
      ) 0.0 flat_returns /. float_of_int (Array.length flat_returns) in
      sqrt variance
    in
    let std_return = max std_return 1e-8 in
    
    List.iter2 (fun _traj returns ->
      let advantages = Array.map (fun r -> 
        (r -. mean_return) /. std_return
      ) returns in
      all_advantages := advantages :: !all_advantages
    ) trajectories !all_returns;
    
    List.rev !all_advantages

  let update t trajectories =
    let advantages_list = compute_group_advantages trajectories t.gamma in
    
    let combined_trajectory = Trajectory.concat trajectories in
    let combined_advantages = Array.concat advantages_list in
    
    let ref_log_probs = Array.mapi (fun i state ->
      let action = combined_trajectory.actions.(i) in
      get_log_probs t.reference_network t.reference_params state action
    ) combined_trajectory.states in
    
    let policy_loss_grad params =
      let total_loss = ref (Rune.scalar Rune.c Rune.float32 0.0) in
      let n = Array.length combined_trajectory.states in
      
      let batch_indices = 
        if n <= t.batch_size then
          Array.init n (fun i -> i)
        else
          Array.init t.batch_size (fun _ -> Random.int n)
      in
      
      Array.iter (fun idx ->
        let state = combined_trajectory.states.(idx) in
        let action = combined_trajectory.actions.(idx) in
        let advantage = combined_advantages.(idx) in
        let ref_log_prob = ref_log_probs.(idx) in
        
        let logits = Kaun.apply t.policy_network params ~training:true state in
        (* Compute log_softmax manually *)
        let max_logits' = Rune.max logits ~axes:[| -1 |] ~keepdims:true in
        let exp_logits = Rune.exp (Rune.sub logits max_logits') in
        let sum_exp = Rune.sum exp_logits ~axes:[| -1 |] ~keepdims:true in
        let log_probs = Rune.sub logits (Rune.add max_logits' (Rune.log sum_exp)) in
        let action_idx = int_of_float (Rune.to_array action).(0) in
        let log_prob_array = Rune.to_array log_probs in
        let log_prob = log_prob_array.(action_idx) in
        let log_prob = Rune.scalar Rune.c Rune.float32 log_prob in
        
        let log_ratio = Rune.sub log_prob (Rune.scalar Rune.c Rune.float32 ref_log_prob) in
        let kl_penalty = Rune.mul_s log_ratio t.beta in
        let grpo_advantage = Rune.sub (Rune.scalar Rune.c Rune.float32 advantage) kl_penalty in
        
        let loss = Rune.mul (Rune.neg log_prob) grpo_advantage in
        total_loss := Rune.add !total_loss loss
      ) batch_indices;
      
      let avg_loss = Rune.div_s !total_loss 
        (float_of_int (Array.length batch_indices)) in
      avg_loss
    in
    
    let (_policy_loss, policy_grads) = 
      Kaun.value_and_grad policy_loss_grad t.policy_params in
    
    let (policy_updates, new_opt_state) = 
      t.optimizer.update t.opt_state t.policy_params policy_grads in
    
    t.policy_params <- Kaun.Optimizer.apply_updates t.policy_params policy_updates;
    t.opt_state <- new_opt_state;
    
    t.updates <- t.updates + 1;
    if t.updates mod t.update_ref_freq = 0 then begin
      t.reference_params <- t.policy_params;
      Printf.printf "Updated reference policy at iteration %d\n" t.updates
    end;
    
    let avg_reward = 
      let all_rewards = List.map (fun traj ->
        Array.fold_left (+.) 0.0 traj.Trajectory.rewards
      ) trajectories in
      List.fold_left (+.) 0.0 all_rewards /. float_of_int (List.length all_rewards) in
    
    avg_reward
end

let train_grpo env ~iterations ~episodes_per_iter ~max_steps ~learning_rate 
    ~gamma ~beta ~batch_size ~update_ref_freq ~seed =
  let obs_dim = match env.Env.observation_space with
    | Space.Box { shape; _ } -> shape.(0)
    | _ -> failwith "Expected Box observation space"
  in
  
  let n_actions = match env.Env.action_space with
    | Space.Discrete n -> n
    | _ -> failwith "Expected Discrete action space"
  in
  
  let agent = GrpoAgent.create ~obs_dim ~n_actions ~learning_rate ~gamma ~beta
    ~batch_size ~update_ref_freq ~seed in
  
  let rewards_history = ref [] in
  let wins = ref 0 in
  let total_episodes = ref 0 in
  
  Printf.printf "Starting GRPO training\n";
  Printf.printf "Environment: Sokoban, Iterations: %d, Episodes/iter: %d\n"
    iterations episodes_per_iter;
  Printf.printf "LR: %.4f, Beta: %.3f, Batch: %d\n" 
    learning_rate beta batch_size;
  
  for iter = 1 to iterations do
    let trajectories = GrpoAgent.collect_episodes agent env episodes_per_iter max_steps in
    let avg_reward = GrpoAgent.update agent trajectories in
    
    rewards_history := avg_reward :: !rewards_history;
    total_episodes := !total_episodes + episodes_per_iter;
    
    let episodes_won = List.filter (fun traj ->
      let total_reward = Array.fold_left (+.) 0.0 traj.Trajectory.rewards in
      total_reward > 50.0
    ) trajectories |> List.length in
    
    wins := !wins + episodes_won;
    
    if iter mod 10 = 0 then begin
      let recent_rewards = List.filteri (fun i _ -> i < 10) !rewards_history in
      let avg_recent = List.fold_left (+.) 0.0 recent_rewards /. 
                       float_of_int (List.length recent_rewards) in
      
      Printf.printf "Iteration %d: Avg Reward = %.2f, Win Rate = %.1f%%\n"
        iter avg_recent 
        (float_of_int !wins /. float_of_int !total_episodes *. 100.0);
      flush stdout
    end
  done;
  
  Printf.printf "Training complete! Final win rate: %.1f%%\n"
    (float_of_int !wins /. float_of_int !total_episodes *. 100.0)

let () =
  Printf.printf "=== GRPO Training on Sokoban ===\n\n";
  
  Printf.printf "Starting with basic Sokoban environment (corridor)...\n";
  let initial_state = Sokoban.LevelGen.generate_corridor 3 in
  let env = Sokoban.sokoban ~width:5 ~height:3 ~max_steps:50 
    ~initial_state () in
  
  train_grpo env
    ~iterations:100
    ~episodes_per_iter:10
    ~max_steps:50
    ~learning_rate:0.001
    ~gamma:0.99
    ~beta:0.01
    ~batch_size:32
    ~update_ref_freq:20
    ~seed:42;
  
  Printf.printf "\nNow training with curriculum learning...\n";
  
  let curriculum_env = Sokoban.sokoban_curriculum ~max_steps:200 () in
  
  train_grpo curriculum_env
    ~iterations:300
    ~episodes_per_iter:10
    ~max_steps:200
    ~learning_rate:0.0005
    ~gamma:0.99
    ~beta:0.005
    ~batch_size:64
    ~update_ref_freq:30
    ~seed:43;
  
  Printf.printf "\n=== Comparing GRPO vs REINFORCE ===\n";
  Printf.printf "GRPO typically achieves:\n";
  Printf.printf "- More stable learning due to group normalization\n";
  Printf.printf "- Better sample efficiency with KL regularization\n";
  Printf.printf "- Reduced variance in policy updates\n"