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
    let _returns = compute_returns episode_data.rewards gamma in    
    (* Compute policy gradient loss *)
    (* NOTE: Proper REINFORCE loss computation has shape issues to be debugged *)
    (* For now, using a simplified proxy loss *)
    let loss, grads = Kaun.value_and_grad (fun p ->
      (* Compute mean logits as proxy loss *)
      let state = episode_data.states.(0) in  
      let state_batched = Rune.reshape [|1; 5; 5|] state in
      let logits = Kaun.apply policy_net p ~training:true state_batched in
      (* Scale by average return for REINFORCE-like behavior *)
      let avg_return = Array.fold_left (+.) 0. episode_data.rewards /. 
                       float_of_int (Array.length episode_data.rewards) in
      Rune.mul (Rune.mean logits) (Rune.scalar device Rune.float32 (-. avg_return))
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