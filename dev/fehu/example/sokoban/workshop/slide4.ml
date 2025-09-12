(*
```ocaml
 *)
include Slide3
(* REINFORCE training loop *)
let train_reinforce env n_episodes learning_rate gamma =
  (* Initialize policy *)
  let policy_net, params = initialize_policy () in  
  (* Create optimizer *)
  let optimizer = Kaun.Optimizer.adam ~lr:learning_rate () in
  let opt_state = ref (optimizer.init params) in  
  (* Training loop *)
  for episode = 1 to n_episodes do
    (* Collect episode *)
    let episode_data =
      collect_episode env policy_net params 100 in    
    (* Compute returns *)
    let returns = compute_returns episode_data.rewards gamma in    
    (* Compute policy gradient loss *)
    let loss, grads = Kaun.value_and_grad (fun p ->
      (* Recompute log probabilities with current parameters *)
      let total_loss =
        ref (Rune.zeros device Rune.float32 [||]) in      
      Array.iteri (fun t state ->
        let action = episode_data.actions.(t) in
        let g_t = returns.(t) in        
        (* Forward pass through policy *)
        let logits =
          Kaun.apply policy_net p ~training:true state in
        let log_probs = log_softmax ~axis:(-1) logits in
        (* Get log prob of selected action -
           convert action back to int32 for indexing *)
        let action_idx = Rune.cast Rune.int32 action in
        let action_expanded = Rune.reshape [|1; 1|] action_idx in
        let action_log_prob =
          Rune.take_along_axis ~axis:(-1)
            action_expanded log_probs in
        let action_log_prob = Rune.squeeze action_log_prob in        
        (* REINFORCE loss: -G_t * log π(a_t|s_t) *)
        let step_loss = Rune.mul
          (Rune.scalar device Rune.float32 (-. g_t))
          action_log_prob in        
        total_loss := Rune.add !total_loss step_loss
      ) episode_data.states;      
      (* Average over episode *)
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
    (* Log progress *)
    if episode mod 10 = 0 then
      let total_reward =
        Array.fold_left (+.) 0. episode_data.rewards in
      Printf.printf "Episode %d: Return = %.2f, Loss = %.4f\n"
        episode total_reward (Rune.item [] loss)
  done;  
  (policy_net, params)
(* Run the training *)
let main () =
  let env = create_simple_gridworld 5 in
  let _trained_policy, _trained_params = 
    train_reinforce env 100 0.01 0.99 in
  print_endline "Training complete!"
(*
```
 *)