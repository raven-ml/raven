(*
```ocaml
 *)
open Slide1
open Slide2
open Slide3
open Slide4  (* For training_history type *)
(* Create a value network for the baseline *)
let create_value_network grid_size =
  Kaun.Layer.sequential [
    Kaun.Layer.flatten ();
    Kaun.Layer.linear ~in_features:(grid_size * grid_size)
                 ~out_features:32 ();
    Kaun.Layer.relu ();
    Kaun.Layer.linear ~in_features:32 ~out_features:16 ();
    Kaun.Layer.relu ();
    (* Single value output *)
    Kaun.Layer.linear ~in_features:16 ~out_features:1 ();
  ]

(* REINFORCE with learned baseline (Actor-Critic) *)
let train_actor_critic env n_episodes lr_actor lr_critic gamma =
  (* History tracking *)
  let history_returns = Array.make n_episodes 0.0 in
  let history_losses = Array.make n_episodes 0.0 in
  let rng = Rune.Rng.key 42 in  
  (* Initialize actor (policy) and critic (value) networks *)
  let policy_net = create_policy_network 5 4 in
  let value_net = create_value_network 5 in  
  let keys = Rune.Rng.split ~n:2 rng in
  let policy_params =
    Kaun.init policy_net ~rngs:keys.(0) ~device
      ~dtype:Rune.float32 in
  let value_params =
    Kaun.init value_net ~rngs:keys.(1) ~device
      ~dtype:Rune.float32 in  
  (* Separate optimizers for actor and critic *)
  let policy_opt = Kaun.Optimizer.adam ~lr:lr_actor () in
  let value_opt = Kaun.Optimizer.adam ~lr:lr_critic () in
  let policy_opt_state = ref (policy_opt.init policy_params) in
  let value_opt_state = ref (value_opt.init value_params) in  
  for episode = 1 to n_episodes do
    (* Collect episode with value estimates *)
    let episode_data =
      collect_episode env policy_net policy_params 100 in
    let returns = compute_returns episode_data.rewards gamma in    
    (* Compute value estimates for states that have returns *)
    let n_steps = Array.length returns in
    let values = Array.init n_steps (fun i ->
      let state = episode_data.states.(i) in
      (* Add batch dimension to state *)
      let state_batched = Rune.reshape [|1; 5; 5|] state in
      let v =
        Kaun.apply value_net value_params ~training:false state_batched in
      Rune.item [] v
    ) in    
    (* Compute advantages (TD error) *)
    let advantages =
      Array.mapi (fun i r -> r -. values.(i)) returns in    
    (* Update critic (value network) *)
    let value_loss, value_grads = Kaun.value_and_grad (fun vp ->
      (* Only compute predictions for states that have returns *)
      let n_steps = Array.length returns in
      let predictions = Array.init n_steps (fun i ->
        let state = episode_data.states.(i) in
        let state_batched = Rune.reshape [|1; 5; 5|] state in
        let pred = Kaun.apply value_net vp ~training:true state_batched in
        (* Squeeze to remove batch dimension *)
        Rune.squeeze ~axes:[|0; 1|] pred
      ) in
      (* MSE loss between predictions and returns *)
      let pred_tensor =
        Rune.stack ~axis:0 (Array.to_list predictions) in
      let returns_tensor = Rune.create device Rune.float32
        [|n_steps|] returns in
      Kaun.Loss.mse pred_tensor returns_tensor
    ) value_params in    
    let value_updates, new_value_state = 
      value_opt.update !value_opt_state
        value_params value_grads in
    value_opt_state := new_value_state;
    Kaun.Optimizer.apply_updates_inplace
        value_params value_updates;    
    (* Update actor (policy network) *)
    let _policy_loss, policy_grads =
      Kaun.value_and_grad (fun pp ->
      let total_loss =
        ref (Rune.zeros device Rune.float32 [||]) in      
      for t = 0 to n_steps - 1 do
        let state = episode_data.states.(t) in
        let action = episode_data.actions.(t) in
        let advantage = advantages.(t) in
        (* Add batch dimension to state *)
        let state_batched = Rune.reshape [|1; 5; 5|] state in
        let logits =
          Kaun.apply policy_net pp ~training:true state_batched in
        let log_probs = log_softmax ~axis:(-1) logits in

        (* Convert action to one-hot encoding to stay on device *)
        let action_int_tensor = Rune.astype Rune.int32 action in
        let action_one_hot = Rune.one_hot ~num_classes:4 action_int_tensor in
        let action_one_hot =
          Rune.reshape [|1; 4|] action_one_hot |>
          Rune.astype Rune.float32 in

        (* Select the log prob using element-wise multiply and sum *)
        let action_log_prob = Rune.sum (Rune.mul action_one_hot log_probs) in
        let step_loss = Rune.mul
          (Rune.scalar device Rune.float32 (-. advantage))
          action_log_prob in
        total_loss := Rune.add !total_loss step_loss
      done;
      let n_steps_float = float_of_int n_steps in
      Rune.div !total_loss
        (Rune.scalar device Rune.float32 n_steps_float)
    ) policy_params in    
    let policy_updates, new_policy_state =
      policy_opt.update !policy_opt_state
        policy_params policy_grads in
    policy_opt_state := new_policy_state;
    Kaun.Optimizer.apply_updates_inplace
      policy_params policy_updates;

    (* Track history *)
    let total_reward = Array.fold_left (+.) 0. episode_data.rewards in
    history_returns.(episode - 1) <- total_reward;
    history_losses.(episode - 1) <- Rune.item [] value_loss;  (* Using value loss as primary metric *)

    if episode mod 10 = 0 then
      Printf.printf
        "Episode %d: Return = %.2f, Value Loss = %.4f\n"
        episode returns.(0) (Rune.item [] value_loss)
  done;

  (* Return with history *)
  (policy_net, policy_params, value_net, value_params,
   {returns = history_returns; losses = history_losses})
(* Main function to test Actor-Critic *)
let main () =
  print_endline "=== Slide 6: Actor-Critic ===";
  let env = create_simple_gridworld 5 in
  
  (* Train for a few episodes *)
  let _policy_net, _policy_params, _value_net, _value_params, _history =
    train_actor_critic env 20 0.01 0.005 0.99 in
  
  print_endline "Actor-Critic training complete!"

(*
```
 *)