(*
```ocaml
 *)
include Slide4
(* REINFORCE with running average baseline *)
let train_reinforce_with_baseline env n_episodes learning_rate
    gamma =
  (* Initialize policy *)
  let policy_net, params = initialize_policy () in
  let optimizer = Kaun.Optimizer.adam ~lr:learning_rate () in
  let opt_state = ref (optimizer.init params) in  
  (* Running average baseline *)
  let baseline = ref 0.0 in
  (* Exponential moving average factor *)
  let baseline_alpha = 0.01 in  
  for episode = 1 to n_episodes do
    (* Collect episode *)
    let episode_data =
      collect_episode env policy_net params 100 in    
    (* Compute returns *)
    let returns = compute_returns episode_data.rewards gamma in    
    (* Update baseline (exponential moving average) *)
    (* Total episode return *)
    let episode_return = returns.(0) in
    baseline := !baseline *. (1.0 -. baseline_alpha) +. 
                episode_return *. baseline_alpha;    
    (* Compute advantages (returns - baseline) *)
    let advantages =
      Array.map (fun r -> r -. !baseline) returns in    
    (* Compute policy gradient with baseline *)
    let _loss, grads = Kaun.value_and_grad (fun p ->
      let total_loss =
        ref (Rune.zeros device Rune.float32 [||]) in      
      Array.iteri (fun t state ->
        let action = episode_data.actions.(t) in
        (* Use advantage instead of return *)
        let advantage = advantages.(t) in        
        let logits =
          Kaun.apply policy_net p ~training:true state in
        let log_probs = log_softmax ~axis:(-1) logits in
        (* Get log prob of selected action -
           convert action back to int32 for indexing *)
        let action_idx = Rune.cast Rune.int32 action in
        let action_expanded =
          Rune.reshape [|1; 1|] action_idx in
        let action_log_prob =
          Rune.take_along_axis ~axis:(-1) action_expanded log_probs in
        let action_log_prob =
          Rune.squeeze action_log_prob in        
        (* Loss: -advantage * log π(a_t|s_t) *)
        let step_loss = Rune.mul
          (Rune.scalar device Rune.float32 (-. advantage))
          action_log_prob in        
        total_loss := Rune.add !total_loss step_loss
      ) episode_data.states;      
      let n_steps =
        float_of_int (Array.length episode_data.states) in
      Rune.div !total_loss
        (Rune.scalar device Rune.float32 n_steps)
    ) params in    
    (* Update parameters *)
    let updates, new_state =
      optimizer.update !opt_state params grads in
    opt_state := new_state;
    Kaun.Optimizer.apply_updates_inplace params updates;    
    if episode mod 10 = 0 then
      Printf.printf "Episode %d: Return = %.2f, Baseline = %.2f\n"
        episode episode_return !baseline
  done;  
  (policy_net, params)
(* Main function to test REINFORCE with baseline *)
let main () =
  print_endline "=== Slide 5: REINFORCE with Baseline ===";
  let env = create_simple_gridworld 5 in
  
  (* Train for a few episodes *)
  let _policy_net, _params = 
    train_reinforce_with_baseline env 20 0.01 0.99 in
  
  print_endline "REINFORCE with baseline training complete!"

(*
```
 *)
