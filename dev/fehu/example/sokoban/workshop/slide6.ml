(*
```ocaml
 *)
include Slide5
(* Create a value network for the baseline *)
let create_value_network grid_size =
  Layer.sequential [
    Layer.flatten ();
    Layer.linear ~in_features:(grid_size * grid_size)
                 ~out_features:32 ();
    Layer.relu ();
    Layer.linear ~in_features:32 ~out_features:16 ();
    Layer.relu ();
    (* Single value output *)
    Layer.linear ~in_features:16 ~out_features:1 ();
  ]
(* REINFORCE with learned baseline (Actor-Critic) *)
let train_actor_critic env n_episodes lr_actor lr_critic gamma =
  let rng = Rune.Rng.key 42 in  
  (* Initialize actor (policy) and critic (value) networks *)
  let policy_net = create_policy_network 5 4 in
  let value_net = create_value_network 5 in  
  let dummy_obs = Rune.zeros device Rune.float32 [|5; 5|] in
  let policy_params =
    Kaun.init policy_net ~rngs:(Rune.Rng.split ~n:1 rng).(0)
      dummy_obs in
  let value_params =
    Kaun.init value_net ~rngs:(Rune.Rng.split ~n:1 rng).(0)
      dummy_obs in  
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
    (* Compute value estimates for all states *)
    let values = Array.map (fun state ->
      let v =
        Kaun.apply value_net value_params ~training:false state in
      Rune.unsafe_get [] v
    ) episode_data.states in    
    (* Compute advantages (TD error) *)
    let advantages =
      Array.mapi (fun i r -> r -. values.(i)) returns in    
    (* Update critic (value network) *)
    let value_loss, value_grads = Kaun.value_and_grad (fun vp ->
      let predictions = Array.map (fun state ->
        Kaun.apply value_net vp ~training:true state
      ) episode_data.states in      
      (* MSE loss between predictions and returns *)
      let pred_tensor = Rune.stack predictions in
      let returns_tensor = Rune.create device Rune.float32 
        [|Array.length returns|] returns in
      Kaun.Loss.mse pred_tensor returns_tensor
    ) value_params in    
    let value_updates, new_value_state = 
      value_opt.update !value_opt_state
        value_params value_grads in
    value_opt_state := new_value_state;
    Kaun.Optimizer.apply_updates_inplace
        value_params value_updates;    
    (* Update actor (policy network) *)
    let policy_loss, policy_grads = Kaun.value_and_grad (fun pp ->
      let total_loss =
        ref (Rune.zeros device Rune.float32 [||]) in      
      Array.iteri (fun t state ->
        let action = episode_data.actions.(t) in
        let advantage = advantages.(t) in        
        let logits =
          Kaun.apply policy_net pp ~training:true state in
        let log_probs = Rune.log_softmax ~axis:(-1) logits in
        let action_log_prob = Rune.gather log_probs action in        
        let step_loss = Rune.mul
          (Rune.scalar device Rune.float32 (-. advantage))
          action_log_prob in        
        total_loss := Rune.add !total_loss step_loss
      ) episode_data.states;      
      let n_steps =
        float_of_int (Array.length episode_data.states) in
      Rune.div !total_loss
        (Rune.scalar device Rune.float32 n_steps)
    ) policy_params in    
    let policy_updates, new_policy_state = 
      policy_opt.update !policy_opt_state
        policy_params policy_grads in
    policy_opt_state := new_policy_state;
    Kaun.Optimizer.apply_updates_inplace
      policy_params policy_updates;    
    if episode mod 10 = 0 then
      Printf.printf
        "Episode %d: Return = %.2f, Value Loss = %.4f\n"
        episode returns.(0) (Rune.unsafe_get [] value_loss)
  done;  
  (policy_net, policy_params, value_net, value_params)
(*
```
 *)