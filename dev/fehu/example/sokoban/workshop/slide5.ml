(*
```ocaml
 *)
open Slide1
open Slide2
open Slide3
open Slide4  (* For training_history type *)
(* REINFORCE with running average baseline *)
let train_reinforce_with_baseline env n_episodes learning_rate
     gamma ?(grid_size=5) () =
  let policy_net, params = initialize_policy ~grid_size () in
  let optimizer = Kaun.Optimizer.adam ~lr:learning_rate () in
  let opt_state = ref (optimizer.init params) in
  let baseline = ref 0.0 in
  (* Exponential moving average factor *)
  let baseline_alpha = 0.01 in

  (* History tracking *)
  let history_returns = Array.make n_episodes 0.0 in
  let history_losses = Array.make n_episodes 0.0 in

  (* Collect episodes for visualization *)
  let collected_episodes = ref [] in

  for episode = 1 to n_episodes do
    (* Collect episode *)
    let episode_data =
      collect_episode env policy_net params 100 in

    (* Store selected episodes *)
    if episode mod (n_episodes / 10) = 0 || episode = n_episodes then
      collected_episodes := episode_data :: !collected_episodes;
    let returns = compute_returns episode_data.rewards gamma in
    (* Update baseline (exponential moving average) *)
    let episode_return = if Array.length returns > 0 then returns.(0) else 0.0 in
    baseline := !baseline *. (1.0 -. baseline_alpha) +. 
                episode_return *. baseline_alpha;    
    (* Compute advantages (returns - baseline) *)
    let advantages =
      Array.map (fun r -> r -. !baseline) returns in    
    (* Compute policy gradient with baseline *)
    let loss, grads = Kaun.value_and_grad (fun p ->
      let total_loss =
        ref (Rune.zeros device Rune.float32 [||]) in
      (* Only iterate over steps where we have advantages *)
      let n_steps = Array.length advantages in
      for t = 0 to n_steps - 1 do
        let state = episode_data.states.(t) in
        let action = episode_data.actions.(t) in
        (* Use advantage instead of return *)
        let advantage = advantages.(t) in
        (* Add batch dimension to state *)
        let state_batched = Rune.reshape [|1; grid_size; grid_size|] state in
        let logits =
          Kaun.apply policy_net p ~training:true state_batched in
        let log_probs = log_softmax ~axis:(-1) logits in

        (* Convert action to one-hot encoding to stay on device *)
        let action_int_tensor = Rune.astype Rune.int32 action in
        let action_one_hot =
           Rune.one_hot ~num_classes:4 action_int_tensor in
        let action_one_hot =
          Rune.reshape [|1; 4|] action_one_hot |>
          Rune.astype Rune.float32 in

        (* Select log prob using element-wise multiply and sum *)
        let action_log_prob =
          Rune.sum (Rune.mul action_one_hot log_probs) in
        (* Loss: -advantage * log π(a_t|s_t) *)
        let step_loss = Rune.mul
          (Rune.scalar device Rune.float32 (-. advantage))
          action_log_prob in
        total_loss := Rune.add !total_loss step_loss
      done;
      let n_steps_float =
        float_of_int n_steps in
      Rune.div !total_loss
        (Rune.scalar device Rune.float32 n_steps_float)
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

    if episode mod 10 = 0 then
      Printf.printf "Episode %d: Return = %.2f, Baseline = %.2f\n"
        episode episode_return !baseline
  done;

  (* Return with history *)
  (policy_net, params, {returns = history_returns; losses = history_losses; collected_episodes = List.rev !collected_episodes})
(* Main function to test REINFORCE with baseline *)
let main () =
  print_endline "=== Slide 5: REINFORCE with Baseline ===";
  let env = create_simple_gridworld 5 in
  
  (* Train for a few episodes *)
  let _policy_net, _params, _history =
    train_reinforce_with_baseline env 20 0.01 0.99 () in
  
  print_endline "REINFORCE with baseline training complete!"

(*
```
 *)
