open Kaun

type config = {
  learning_rate : float;
  gamma : float;
  use_baseline : bool;
  reward_scale : float;
  entropy_coef : float;
  max_episode_steps : int;
}

let default_config =
  {
    learning_rate = 0.001;
    gamma = 0.99;
    use_baseline = false;
    reward_scale = 1.0;
    entropy_coef = 0.01;
    max_episode_steps = 1000;
  }

type t = {
  policy_network : module_;
  mutable policy_params : Rune.float32_elt params;
  baseline_network : module_ option;
  mutable baseline_params : Rune.float32_elt params option;
  policy_optimizer : Rune.float32_elt Optimizer.gradient_transformation;
  mutable policy_opt_state : Rune.float32_elt Optimizer.opt_state;
  baseline_optimizer :
    Rune.float32_elt Optimizer.gradient_transformation option;
  mutable baseline_opt_state : Rune.float32_elt Optimizer.opt_state option;
  mutable rng : Rune.Rng.key;
  n_actions : int;
  config : config;
}

type update_metrics = {
  episode_return : float;
  episode_length : int;
  avg_entropy : float;
  avg_log_prob : float;
  adv_mean : float;
  adv_std : float;
  value_loss : float option;
}

let create ~policy_network ?baseline_network ~n_actions ~rng config =
  if config.use_baseline && Option.is_none baseline_network then
    invalid_arg
      "Reinforce.create: baseline_network required when use_baseline = true";

  let keys = Rune.Rng.split ~n:2 rng in

  let policy_params = init policy_network ~rngs:keys.(0) ~dtype:Rune.float32 in
  let policy_optimizer = Optimizer.adam ~lr:config.learning_rate () in
  let policy_opt_state = policy_optimizer.init policy_params in

  let ( baseline_network_state,
        baseline_params,
        baseline_optimizer,
        baseline_opt_state ) =
    match baseline_network with
    | Some net when config.use_baseline ->
        let params = init net ~rngs:keys.(1) ~dtype:Rune.float32 in
        let opt = Optimizer.adam ~lr:config.learning_rate () in
        let opt_state = opt.init params in
        (Some net, Some params, Some opt, Some opt_state)
    | _ -> (None, None, None, None)
  in

  {
    policy_network;
    policy_params;
    baseline_network = baseline_network_state;
    baseline_params;
    policy_optimizer;
    policy_opt_state;
    baseline_optimizer;
    baseline_opt_state;
    rng = keys.(0);
    n_actions;
    config;
  }

let predict t obs ~training =
  (* Add batch dimension if needed: [features] -> [1, features] *)
  let obs_shape = Rune.shape obs in
  let obs_batched =
    if Array.length obs_shape = 1 then
      (* No batch dimension, add it *)
      let features = obs_shape.(0) in
      Rune.reshape [| 1; features |] obs
    else
      (* Already has batch dimension *)
      obs
  in

  let logits = apply t.policy_network t.policy_params ~training obs_batched in

  if training then (
    (* Sample from policy distribution using Rune RNG *)
    let probs = Rune.softmax logits ~axes:[ -1 ] in
    let probs_array = Rune.to_array probs in

    (* Split RNG and sample *)
    let keys = Rune.Rng.split t.rng ~n:2 in
    t.rng <- keys.(0);
    let sample_rng = keys.(1) in

    let uniform_sample = Rune.Rng.uniform sample_rng Rune.float32 [| 1 |] in
    let r = (Rune.to_array uniform_sample).(0) in

    let rec sample_idx i cumsum =
      if i >= Array.length probs_array - 1 then i
      else if r <= cumsum +. probs_array.(i) then i
      else sample_idx (i + 1) (cumsum +. probs_array.(i))
    in

    let action_idx = sample_idx 0 0.0 in
    let action = Rune.scalar Rune.int32 (Int32.of_int action_idx) in

    (* Compute log_softmax manually *)
    let max_logits = Rune.max logits ~axes:[ -1 ] ~keepdims:true in
    let exp_logits = Rune.exp (Rune.sub logits max_logits) in
    let sum_exp = Rune.sum exp_logits ~axes:[ -1 ] ~keepdims:true in
    let log_probs = Rune.sub logits (Rune.add max_logits (Rune.log sum_exp)) in
    let log_prob_array = Rune.to_array log_probs in
    let log_prob = log_prob_array.(action_idx) in

    (action, log_prob))
  else
    (* Greedy selection *)
    let action_idx = Rune.argmax logits ~axis:(-1) ~keepdims:false in
    let action = Rune.cast Rune.int32 action_idx in
    (action, 0.0)

let predict_value t obs =
  match (t.baseline_network, t.baseline_params) with
  | Some net, Some params ->
      (* Add batch dimension if needed *)
      let obs_shape = Rune.shape obs in
      let obs_batched =
        if Array.length obs_shape = 1 then
          let features = obs_shape.(0) in
          Rune.reshape [| 1; features |] obs
        else obs
      in
      let value = apply net params ~training:false obs_batched in
      let value_array = Rune.to_array value in
      value_array.(0)
  | _ -> 0.0

let update t trajectory =
  let open Fehu in
  (* Compute returns using Monte Carlo *)
  let compute_returns ~rewards ~dones ~gamma =
    let n = Array.length rewards in
    let returns = Array.make n 0.0 in
    let running_return = ref 0.0 in

    for i = n - 1 downto 0 do
      if dones.(i) then running_return := 0.0;
      running_return := rewards.(i) +. (gamma *. !running_return);
      returns.(i) <- !running_return
    done;

    returns
  in

  let returns_raw =
    compute_returns ~rewards:trajectory.Trajectory.rewards
      ~dones:trajectory.Trajectory.terminateds ~gamma:t.config.gamma
  in

  let returns = Array.map (fun r -> r *. t.config.reward_scale) returns_raw in

  (* Compute advantages *)
  let advantages =
    if t.config.use_baseline then
      match trajectory.Trajectory.values with
      | Some values -> Array.mapi (fun i r -> r -. values.(i)) returns
      | None -> returns
    else returns
  in

  let n_steps = Array.length advantages in
  let steps_f = float_of_int n_steps in

  (* Compute advantage statistics *)
  let adv_mean =
    if n_steps = 0 then 0.0
    else Array.fold_left ( +. ) 0.0 advantages /. steps_f
  in
  let adv_var =
    if n_steps = 0 then 0.0
    else
      Array.fold_left
        (fun acc a ->
          let diff = a -. adv_mean in
          acc +. (diff *. diff))
        0.0 advantages
      /. steps_f
  in
  let adv_std = sqrt adv_var in

  (* Normalize advantages *)
  let advantages_norm =
    if n_steps = 0 then [||]
    else
      let denom = if adv_std < 1e-6 then 1.0 else adv_std in
      Array.map (fun a -> (a -. adv_mean) /. denom) advantages
  in

  let entropy_acc = ref 0.0 in
  let log_prob_acc = ref 0.0 in

  (* Policy gradient loss *)
  let policy_loss_grad params =
    let total_loss = ref (Rune.scalar Rune.float32 0.0) in

    for i = 0 to Array.length trajectory.Trajectory.observations - 1 do
      let obs = trajectory.Trajectory.observations.(i) in
      let action = trajectory.Trajectory.actions.(i) in
      let advantage = advantages_norm.(i) in

      (* Add batch dimension if needed: [features] -> [1, features] *)
      let obs_shape = Rune.shape obs in
      let obs_batched =
        if Array.length obs_shape = 1 then
          let features = obs_shape.(0) in
          Rune.reshape [| 1; features |] obs
        else obs
      in

      let logits = apply t.policy_network params ~training:true obs_batched in

      (* Compute log_softmax *)
      let max_logits = Rune.max logits ~axes:[ -1 ] ~keepdims:true in
      let exp_logits = Rune.exp (Rune.sub logits max_logits) in
      let sum_exp = Rune.sum exp_logits ~axes:[ -1 ] ~keepdims:true in
      let log_probs_pred =
        Rune.sub logits (Rune.add max_logits (Rune.log sum_exp))
      in

      (* Compute entropy for logging *)
      let probs = Rune.softmax logits ~axes:[ -1 ] in
      let probs_arr = Rune.to_array (Rune.reshape [| t.n_actions |] probs) in
      let entropy =
        Array.fold_left
          (fun acc p -> if p > 0. then acc -. (p *. log p) else acc)
          0.0 probs_arr
      in
      entropy_acc := !entropy_acc +. entropy;

      (* Select log_prob for taken action *)
      let log_probs_flat = Rune.reshape [| t.n_actions |] log_probs_pred in
      let indices = Rune.reshape [| 1 |] (Rune.cast Rune.int32 action) in
      let selected = Rune.take indices log_probs_flat |> Rune.reshape [||] in

      let log_prob_float = (Rune.to_array selected).(0) in
      log_prob_acc := !log_prob_acc +. log_prob_float;

      (* Policy gradient: -log π(a|s) × advantage *)
      let loss = Rune.mul_s (Rune.neg selected) advantage in
      total_loss := Rune.add !total_loss loss;

      (* Entropy regularization *)
      if t.config.entropy_coef <> 0. then
        let entropy_scalar =
          Rune.sum (Rune.mul probs log_probs_pred) ~axes:[ -1 ] ~keepdims:false
          |> Rune.reshape [||]
        in
        let entropy_term =
          Rune.mul_s entropy_scalar (-.t.config.entropy_coef)
        in
        total_loss := Rune.add !total_loss entropy_term
    done;

    let avg_loss =
      Rune.div_s !total_loss
        (float_of_int (Array.length trajectory.Trajectory.observations))
    in
    avg_loss
  in

  (* Update policy *)
  let _policy_loss, policy_grads =
    value_and_grad policy_loss_grad t.policy_params
  in

  let policy_updates, new_policy_opt_state =
    t.policy_optimizer.update t.policy_opt_state t.policy_params policy_grads
  in

  t.policy_params <- Optimizer.apply_updates t.policy_params policy_updates;
  t.policy_opt_state <- new_policy_opt_state;

  (* Update baseline if present *)
  let value_loss_acc = ref 0.0 in

  (if t.config.use_baseline then
     match
       ( t.baseline_network,
         t.baseline_params,
         t.baseline_optimizer,
         t.baseline_opt_state )
     with
     | Some net, Some params, Some opt, Some opt_state ->
         let baseline_loss_grad params =
           let total_loss = ref (Rune.scalar Rune.float32 0.0) in

           for i = 0 to Array.length trajectory.Trajectory.observations - 1 do
             let obs = trajectory.Trajectory.observations.(i) in
             let return = returns.(i) in

             (* Add batch dimension if needed: [features] -> [1, features] *)
             let obs_shape = Rune.shape obs in
             let obs_batched =
               if Array.length obs_shape = 1 then
                 let features = obs_shape.(0) in
                 Rune.reshape [| 1; features |] obs
               else obs
             in

             let value_pred = apply net params ~training:true obs_batched in
             let value = Rune.reshape [||] value_pred in
             let value_float = (Rune.to_array value).(0) in
             let diff = value_float -. return in
             value_loss_acc := !value_loss_acc +. (diff *. diff);
             let target = Rune.scalar Rune.float32 return in
             let loss = Rune.square (Rune.sub value target) in

             total_loss := Rune.add !total_loss loss
           done;

           let avg_loss =
             Rune.div_s !total_loss
               (float_of_int (Array.length trajectory.Trajectory.observations))
           in
           avg_loss
         in

         let _, baseline_grads = value_and_grad baseline_loss_grad params in
         let baseline_updates, new_baseline_opt_state =
           opt.update opt_state params baseline_grads
         in

         t.baseline_params <-
           Some (Optimizer.apply_updates params baseline_updates);
         t.baseline_opt_state <- Some new_baseline_opt_state
     | _ -> ());

  (* Compute metrics *)
  let episode_return = if n_steps > 0 then returns_raw.(0) else 0.0 in
  let avg_entropy = if n_steps = 0 then 0.0 else !entropy_acc /. steps_f in
  let avg_log_prob = if n_steps = 0 then 0.0 else !log_prob_acc /. steps_f in
  let value_loss_avg =
    if n_steps = 0 then None
    else if t.config.use_baseline then Some (!value_loss_acc /. steps_f)
    else None
  in

  let metrics =
    {
      episode_return;
      episode_length = n_steps;
      avg_entropy;
      avg_log_prob;
      adv_mean;
      adv_std;
      value_loss = value_loss_avg;
    }
  in

  (t, metrics)

let learn t ~env ~total_timesteps
    ?(callback = fun ~iteration:_ ~metrics:_ -> true) () =
  let open Fehu in
  let timesteps = ref 0 in
  let iteration = ref 0 in

  while !timesteps < total_timesteps do
    (* Collect episode *)
    let observations = ref [] in
    let actions = ref [] in
    let rewards = ref [] in
    let terminateds = ref [] in
    let truncateds = ref [] in
    let log_probs = ref [] in
    let values = ref [] in

    let obs, _info = Env.reset env () in
    let current_obs = ref obs in
    let steps = ref 0 in
    let done_flag = ref false in

    while !steps < t.config.max_episode_steps && not !done_flag do
      let action, log_prob = predict t !current_obs ~training:true in
      let value =
        if t.config.use_baseline then predict_value t !current_obs else 0.0
      in

      observations := !current_obs :: !observations;
      actions := action :: !actions;
      log_probs := log_prob :: !log_probs;
      values := value :: !values;

      let transition = Env.step env action in

      rewards := transition.Env.reward :: !rewards;
      terminateds := transition.Env.terminated :: !terminateds;
      truncateds := transition.Env.truncated :: !truncateds;

      current_obs := transition.Env.observation;
      steps := !steps + 1;
      done_flag := transition.Env.terminated || transition.Env.truncated
    done;

    timesteps := !timesteps + !steps;

    (* Create trajectory *)
    let log_probs_array = Array.of_list (List.rev !log_probs) in
    let values_array = Array.of_list (List.rev !values) in
    let trajectory =
      Trajectory.create
        ~observations:(Array.of_list (List.rev !observations))
        ~actions:(Array.of_list (List.rev !actions))
        ~rewards:(Array.of_list (List.rev !rewards))
        ~terminateds:(Array.of_list (List.rev !terminateds))
        ~truncateds:(Array.of_list (List.rev !truncateds))
        ~log_probs:log_probs_array
        ?values:(if t.config.use_baseline then Some values_array else None)
        ()
    in

    (* Update *)
    let _t, metrics = update t trajectory in

    iteration := !iteration + 1;

    (* Call callback *)
    let continue = callback ~iteration:!iteration ~metrics in
    if not continue then timesteps := total_timesteps
  done;

  t

let save _t _path =
  (* TODO: Implement serialization *)
  failwith "Reinforce.save: not yet implemented"

let load _path =
  (* TODO: Implement deserialization *)
  failwith "Reinforce.load: not yet implemented"
