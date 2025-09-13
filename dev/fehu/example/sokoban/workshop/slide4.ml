(*
```ocaml
 *)
open Slide1
open Slide2
open Slide3
(* REINFORCE training loop *)
let train_reinforce env n_episodes learning_rate gamma =
  (* Initialize policy *)
  let policy_net, params = initialize_policy () in  
  (* Create optimizer *)
  let optimizer = Kaun.Optimizer.adam ~lr:learning_rate () in
  let opt_state = ref (optimizer.init params) in
  
  (* Collect episodes for visualization *)
  let collected_episodes = ref [] in
  
  (* Training loop *)
  for episode = 1 to n_episodes do
    (* Collect episode *)
    let episode_data =
      collect_episode env policy_net params 100 in
    
    (* Store first and last episodes *)
    if episode = 1 || episode = n_episodes then
      collected_episodes := episode_data :: !collected_episodes;
    
    (* Compute returns *)
    let _returns = compute_returns episode_data.rewards gamma in    
    (* Compute policy gradient loss *)
    let loss, grads = Kaun.value_and_grad (fun p ->
      (* Compute loss for all states *)
      let total_loss =
        ref (Rune.scalar device Rune.float32 0.0) in
      
      (* EXERCISE 1: We only process first 10 states due to
         autodiff indexing issues. This severely limits
         learning efficiency - see exercise1.md for details.
         Challenge: Make this work for ALL states in the episode *)
      let n_samples = min 10 (Array.length episode_data.states) in
      for t = 0 to n_samples - 1 do
        let state = episode_data.states.(t) in  
        let action = episode_data.actions.(t) in
        let g_t = _returns.(t) in
        
        (* Classic gotcha with the Gym API:
           adding batch dimension. *)
        let state_batched = Rune.reshape [|1; 5; 5|] state in
        let logits =
          Kaun.apply policy_net p ~training:true state_batched in
        
        (* Compute negative log likelihood weighted by return *)
        (* Create action mask without set_item *)
        let action_int = int_of_float (Rune.item [] action) in
        let mask =
          Rune.init device Rune.float32 [|1; 4|] (fun idxs ->
            if idxs.(1) = action_int then 1.0 else 0.0
        ) in
        
        let log_probs = log_softmax ~axis:(-1) logits in
        let selected_log_prob =
          Rune.sum (Rune.mul mask log_probs) in
        let weighted_loss =
          Rune.mul (Rune.neg selected_log_prob) 
                   (Rune.scalar device Rune.float32 g_t) in
        
        total_loss := Rune.add !total_loss weighted_loss
      done;
      
      Rune.div !total_loss
        (Rune.scalar device Rune.float32 (float_of_int n_samples))
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
  (policy_net, params, List.rev !collected_episodes)
(* Main function - just for slide consistency *)
let main () =
  print_endline "REINFORCE training function defined."
(*
```
 *)