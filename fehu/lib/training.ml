let compute_gae ~rewards ~values ~dones ~last_value ~last_done ~gamma
    ~gae_lambda =
  let n = Array.length rewards in
  if n <> Array.length values || n <> Array.length dones then
    invalid_arg "Training.compute_gae: arrays must have same length";

  let advantages = Array.make n 0.0 in
  let returns = Array.make n 0.0 in
  let last_gae_lam = ref 0.0 in

  for t = n - 1 downto 0 do
    let next_value =
      if t = n - 1 then if last_done then 0.0 else last_value
      else values.(t + 1)
    in
    let next_non_terminal =
      if t = n - 1 then if last_done then 0.0 else 1.0
      else if dones.(t) then 0.0
      else 1.0
    in
    let delta =
      rewards.(t) +. (gamma *. next_value *. next_non_terminal) -. values.(t)
    in
    last_gae_lam :=
      delta +. (gamma *. gae_lambda *. next_non_terminal *. !last_gae_lam);
    advantages.(t) <- !last_gae_lam;
    returns.(t) <- !last_gae_lam +. values.(t)
  done;

  (advantages, returns)

let compute_returns ~rewards ~dones ~gamma =
  let n = Array.length rewards in
  if n <> Array.length dones then
    invalid_arg "Training.compute_returns: arrays must have same length";

  let returns = Array.make n 0.0 in
  let return_acc = ref 0.0 in

  for t = n - 1 downto 0 do
    return_acc :=
      rewards.(t) +. (gamma *. !return_acc *. if dones.(t) then 0.0 else 1.0);
    returns.(t) <- !return_acc
  done;

  returns

let normalize_array arr ?(eps = 1e-8) () =
  let n = Array.length arr in
  if n = 0 then arr
  else
    let sum = Array.fold_left ( +. ) 0.0 arr in
    let mean = sum /. float_of_int n in
    let variance_sum =
      Array.fold_left (fun acc x -> acc +. ((x -. mean) ** 2.0)) 0.0 arr
    in
    let std = sqrt (variance_sum /. float_of_int n) +. eps in
    Array.map (fun x -> (x -. mean) /. std) arr

let normalize arr ?(eps = 1e-8) () = normalize_array arr ~eps ()

let policy_gradient_loss ~log_probs ~advantages ?(normalize = true) () =
  let n = Array.length log_probs in
  if n <> Array.length advantages then
    invalid_arg "Training.policy_gradient_loss: arrays must have same length";
  if n = 0 then 0.0
  else
    let adv = if normalize then normalize_array advantages () else advantages in
    let sum = ref 0.0 in
    for i = 0 to n - 1 do
      sum := !sum +. (log_probs.(i) *. adv.(i))
    done;
    -. !sum /. float_of_int n

let ppo_clip_loss ~log_probs ~old_log_probs ~advantages ~clip_range =
  let n = Array.length log_probs in
  if n <> Array.length old_log_probs || n <> Array.length advantages then
    invalid_arg "Training.ppo_clip_loss: arrays must have same length";
  if n = 0 then 0.0
  else
    let adv = normalize_array advantages () in
    let sum = ref 0.0 in
    for i = 0 to n - 1 do
      let ratio = exp (log_probs.(i) -. old_log_probs.(i)) in
      let surr1 = ratio *. adv.(i) in
      let ratio_clipped =
        max (1.0 -. clip_range) (min (1.0 +. clip_range) ratio)
      in
      let surr2 = ratio_clipped *. adv.(i) in
      sum := !sum +. min surr1 surr2
    done;
    -. !sum /. float_of_int n

let value_loss ~values ~returns ?clip () =
  let n = Array.length values in
  if n <> Array.length returns then
    invalid_arg "Training.value_loss: arrays must have same length";

  if n = 0 then 0.0
  else
    match clip with
    | None ->
        let sum = ref 0.0 in
        for i = 0 to n - 1 do
          let diff = values.(i) -. returns.(i) in
          sum := !sum +. (diff *. diff)
        done;
        !sum /. float_of_int n
    | Some (clip_range, old_values) ->
        if clip_range < 0.0 then
          invalid_arg "Training.value_loss: clip_range must be non-negative";
        if Array.length old_values <> n then
          invalid_arg
            "Training.value_loss: old_values must have same length as arrays";

        let sum = ref 0.0 in
        for i = 0 to n - 1 do
          let delta = values.(i) -. old_values.(i) in
          let clipped_delta = max (-.clip_range) (min clip_range delta) in
          let value_clipped = old_values.(i) +. clipped_delta in
          let unclipped = (values.(i) -. returns.(i)) ** 2.0 in
          let clipped = (value_clipped -. returns.(i)) ** 2.0 in
          sum := !sum +. max unclipped clipped
        done;
        !sum /. float_of_int n

let explained_variance ~y_pred ~y_true =
  let n = Array.length y_pred in
  if n <> Array.length y_true then
    invalid_arg "Training.explained_variance: arrays must have same length";
  if n = 0 then 0.0
  else
    let sum_true = Array.fold_left ( +. ) 0.0 y_true in
    let mean_true = sum_true /. float_of_int n in

    let var_y = ref 0.0 in
    let var_diff = ref 0.0 in
    for i = 0 to n - 1 do
      let diff = y_true.(i) -. y_pred.(i) in
      var_diff := !var_diff +. (diff *. diff);
      var_y := !var_y +. ((y_true.(i) -. mean_true) ** 2.0)
    done;

    if !var_y = 0.0 then 0.0 else 1.0 -. (!var_diff /. !var_y)

type stats = {
  mean_reward : float;
  std_reward : float;
  mean_length : float;
  n_episodes : int;
}

let evaluate env ~policy ?(n_episodes = 10) ?(max_steps = 1000) () =
  let episode_rewards = ref [] in
  let episode_lengths = ref [] in

  for _ = 1 to n_episodes do
    let obs, _ = Env.reset env () in
    let total_reward = ref 0.0 in
    let steps = ref 0 in
    let done_flag = ref false in

    while !steps < max_steps && not !done_flag do
      let action = policy obs in
      let transition = Env.step env action in
      total_reward := !total_reward +. transition.Env.reward;
      steps := !steps + 1;
      done_flag := transition.Env.terminated || transition.Env.truncated
    done;

    episode_rewards := !total_reward :: !episode_rewards;
    episode_lengths := float_of_int !steps :: !episode_lengths
  done;

  let rewards_arr = Array.of_list (List.rev !episode_rewards) in
  let lengths_arr = Array.of_list (List.rev !episode_lengths) in

  let mean_reward =
    Array.fold_left ( +. ) 0.0 rewards_arr /. float_of_int n_episodes
  in
  let mean_length =
    Array.fold_left ( +. ) 0.0 lengths_arr /. float_of_int n_episodes
  in

  let var_sum =
    Array.fold_left
      (fun acc r -> acc +. ((r -. mean_reward) ** 2.0))
      0.0 rewards_arr
  in
  let std_reward = sqrt (var_sum /. float_of_int n_episodes) in

  { mean_reward; std_reward; mean_length; n_episodes }
