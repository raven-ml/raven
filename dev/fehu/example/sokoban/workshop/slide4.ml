(*
```ocaml
 *)
open Slide1
open Slide2
open Slide3

(* Training history type *)
type training_history = {
  returns: float array;
  losses: float array;
}

(* REINFORCE training loop *)
let train_reinforce env n_episodes learning_rate gamma ?(grid_size=5) () =
  (* Initialize policy *)
  let policy_net, params = initialize_policy ~grid_size () in
  (* Create optimizer *)
  let optimizer = Kaun.Optimizer.adam ~lr:learning_rate () in
  let opt_state = ref (optimizer.init params) in

  (* Collect episodes for visualization *)
  let collected_episodes = ref [] in

  (* History tracking *)
  let history_returns = Array.make n_episodes 0.0 in
  let history_losses = Array.make n_episodes 0.0 in

  (* Training loop *)
  for episode = 1 to n_episodes do
    (* Collect episode *)
    let episode_data =
      collect_episode env policy_net params 100 in
    
    (* Store first and last episodes *)
    if episode % (n_episodes / 10) = 0 || episode = n_episodes then
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
        let state_batched = Rune.reshape [|1; grid_size; grid_size|] state in
        let logits =
          Kaun.apply policy_net p ~training:true state_batched in
        
        (* Compute negative log likelihood weighted by return *)
        (* Get the log probability of the action that was taken *)
        let log_probs = log_softmax ~axis:(-1) logits in

        (* Convert action to one-hot encoding to stay on device *)
        (* First convert float action to int tensor *)
        let action_int_tensor =
          Rune.astype Rune.int32 action in
        let action_one_hot =
          Rune.one_hot ~num_classes:4 action_int_tensor in
        (* Reshape to match log_probs shape [1, 4] *)
        let action_one_hot =
          Rune.reshape [|1; 4|] action_one_hot |>
          Rune.astype Rune.float32 in

        (* Select the log prob using element-wise multiply and sum *)
        let selected_log_prob =
          Rune.sum (Rune.mul action_one_hot log_probs) in
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

    (* Track history *)
    let total_reward =
      Array.fold_left (+.) 0. episode_data.rewards in
    history_returns.(episode - 1) <- total_reward;
    history_losses.(episode - 1) <- Rune.item [] loss;

    (* Log progress *)
    if episode mod 10 = 0 then
      Printf.printf "Episode %d: Return = %.2f, Loss = %.4f\n"
        episode total_reward (Rune.item [] loss)
  done;

  (* Return everything including history *)
  (policy_net, params, List.rev !collected_episodes,
   {returns = history_returns; losses = history_losses})
(* Main function - just for slide consistency *)
let main () =
  print_endline "REINFORCE training function defined."
(*
```
 *)