(*
```ocaml
 *)
include Slide8
open Fehu

(* Collect episode starting from a specific state *)
let collect_episode_from_state env policy_net params init_state max_steps =
  let rng = Rune.Rng.key (Random.int 1000000) in  
  (* Storage for episode data *)
  let states = ref [] in
  let actions = ref [] in
  let rewards = ref [] in
  let log_probs = ref [] in  
  
  (* Start from provided initial state *)
  states := init_state :: !states;  
  
  (* Run episode *)
  let rec run_steps step obs =
    if step >= max_steps then ()
    else begin
      (* Get action from policy *)
      let action, log_prob =
        sample_action policy_net params obs rng in      
      (* Take environment step *)
      let next_obs, reward, terminated, truncated, _ =
        env.Fehu.Env.step action in      
      (* Store transition *)
      actions := action :: !actions;
      rewards := reward :: !rewards;
      log_probs := log_prob :: !log_probs;      
      if not terminated && not truncated then begin
        states := next_obs :: !states;
        run_steps (step + 1) next_obs
      end
    end
  in  
  run_steps 0 init_state;  
  (* Return episode data *)
  {
    states = Array.of_list (List.rev !states);
    actions = Array.of_list (List.rev !actions);
    rewards = Array.of_list (List.rev !rewards);
    log_probs = Array.of_list (List.rev !log_probs);
  }

(* Generate multiple trajectories for the same initial state *)
let collect_group_trajectories env policy_net params
    init_state group_size =
  let trajectories = Array.init group_size (fun _ ->
    (* Each trajectory starts from same state
       but different random actions *)
    collect_episode_from_state env policy_net params
      init_state 100
  ) in
  trajectories

(* Compute group-relative advantages *)
let compute_group_advantages rewards =
  let mean = Array.fold_left (+.) 0. rewards /. 
             float_of_int (Array.length rewards) in
  let variance = Array.fold_left (fun acc r -> 
    acc +. (r -. mean) ** 2.) 0. rewards /. 
    float_of_int (Array.length rewards) in
  let std = sqrt variance in
  
  (* Normalize advantages within group *)
  Array.map (fun r -> (r -. mean) /. (std +. 1e-8)) rewards

(* GRPO training step *)
let train_grpo_step env policy_net params old_params
    group_size epsilon beta =
  let device = Rune.c in
  
  (* Get initial state *)
  let init_obs, _ = env.Env.reset () in
  
  (* Collect group of trajectories from same starting point *)
  let group =
      collect_group_trajectories env policy_net params
        init_obs group_size in
  
  (* Extract returns for each trajectory *)
  let group_returns = Array.map (fun traj ->
    let returns = compute_returns traj.rewards 0.99 in
    returns.(0)  (* Total return *)
  ) group in
  
  (* Compute group-relative advantages *)
  let group_advantages = compute_group_advantages group_returns in
  
  (* Update policy using clipped objective with KL penalty *)
  let loss, grads = Kaun.value_and_grad (fun p ->
    let total_loss = ref (Rune.zeros device Rune.float32 [||]) in
    
    Array.iteri (fun g_idx trajectory ->
      let advantage = group_advantages.(g_idx) in
      
      Array.iteri (fun t state ->
        let action = trajectory.actions.(t) in
        
        (* Add batch dimension to state *)
        let state_batched = Rune.reshape [|1; 5; 5|] state in
        (* Compute new and old log probs *)
        let new_logits = 
          Kaun.apply policy_net p ~training:true state_batched in
        let new_log_probs = log_softmax ~axis:(-1) new_logits in
        (* Use take_along_axis to get log prob of the action *)
        let action_int = int_of_float (Rune.item [] action) in
        let action_tensor = Rune.scalar device Rune.int32 (Int32.of_int action_int) in
        let action_expanded = Rune.reshape [|1; 1|] action_tensor in
        let new_action_log_prob =
          Rune.take_along_axis ~axis:(-1) action_expanded new_log_probs in
        let new_action_log_prob = Rune.squeeze new_action_log_prob in
        
        let old_logits =
          Kaun.apply policy_net old_params ~training:false
            state_batched in
        let old_log_probs = log_softmax ~axis:(-1) old_logits in
        let old_action_log_prob =
          Rune.take_along_axis ~axis:(-1) action_expanded old_log_probs in
        let old_action_log_prob = Rune.squeeze old_action_log_prob in
        
        (* Compute ratio and clip *)
        let log_ratio =
          Rune.sub new_action_log_prob old_action_log_prob in
        let ratio = Rune.exp log_ratio in
        let clipped_ratio = clip_ratio ratio epsilon in
        
        (* Clipped objective *)
        let adv_scalar =
          Rune.scalar device Rune.float32 advantage in
        let obj1 = Rune.mul ratio adv_scalar in
        let obj2 = Rune.mul clipped_ratio adv_scalar in
        let clipped_obj = Rune.minimum obj1 obj2 in
        
        (* Add KL penalty *)
        let kl_penalty = Rune.mul 
          (Rune.scalar device Rune.float32 beta)
          (Rune.sub old_action_log_prob new_action_log_prob) in
        
        let step_loss =
          Rune.sub (Rune.neg clipped_obj) kl_penalty in
        total_loss := Rune.add !total_loss step_loss
      ) trajectory.states
    ) group;
    
    (* Average over all steps and trajectories *)
    let total_steps = Array.fold_left (fun acc traj -> 
      acc + Array.length traj.states) 0 group in
    Rune.div !total_loss
      (Rune.scalar device Rune.float32 (float_of_int total_steps))
  ) params in
  
  (loss, grads)

(* Complete GRPO training loop *)
let train_grpo env n_iterations group_size learning_rate
    epsilon beta =
  let device = Rune.c in
  let rng = Rune.Rng.key 42 in
  
  (* Initialize policy *)
  let policy_net = create_policy_network 5 4 in
  let params = Kaun.init policy_net ~rngs:rng ~device ~dtype:Rune.float32 in
  (* Keep old params for ratios *)
  let old_params = ref (Kaun.Ptree.copy params) in

  let optimizer = Kaun.Optimizer.adam ~lr:learning_rate () in
  let opt_state = ref (optimizer.init params) in
  
  for iter = 1 to n_iterations do
    (* GRPO update *)
    let loss, grads =
      train_grpo_step env policy_net params !old_params 
                      group_size epsilon beta in
    
    (* Apply gradients *)
    let updates, new_state =
      optimizer.update !opt_state params grads in
    opt_state := new_state;
    Kaun.Optimizer.apply_updates_inplace params updates;
    
    (* Update old params periodically *)
    if iter mod 5 = 0 then
      old_params := Kaun.Ptree.copy params;
    
    if iter mod 10 = 0 then
      Printf.printf "Iteration %d: Loss = %.4f\n" 
        iter (Rune.item [] loss)
  done;
  
  (policy_net, params)
(*
```
 *)