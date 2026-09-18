(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type stats = {
  mean_reward : float;
  std_reward : float;
  mean_length : float;
  n_episodes : int;
}

let run env ~policy ?(n_episodes = 10) ?(max_steps = 1000) () =
  let ep_rewards = Array.make n_episodes 0.0 in
  let ep_lengths = Array.make n_episodes 0.0 in
  for ep = 0 to n_episodes - 1 do
    let obs, _info = Env.reset env () in
    let current_obs = ref obs in
    let total_reward = ref 0.0 in
    let steps = ref 0 in
    let done_flag = ref false in
    while !steps < max_steps && not !done_flag do
      let action = policy !current_obs in
      let s = Env.step env action in
      total_reward := !total_reward +. s.reward;
      steps := !steps + 1;
      current_obs := s.observation;
      done_flag := s.terminated || s.truncated
    done;
    ep_rewards.(ep) <- !total_reward;
    ep_lengths.(ep) <- Float.of_int !steps
  done;
  let n = Float.of_int n_episodes in
  let mean_reward = ref 0.0 in
  let mean_length = ref 0.0 in
  for i = 0 to n_episodes - 1 do
    mean_reward := !mean_reward +. ep_rewards.(i);
    mean_length := !mean_length +. ep_lengths.(i)
  done;
  mean_reward := !mean_reward /. n;
  mean_length := !mean_length /. n;
  let var_sum = ref 0.0 in
  for i = 0 to n_episodes - 1 do
    let d = ep_rewards.(i) -. !mean_reward in
    var_sum := !var_sum +. (d *. d)
  done;
  let std_reward = sqrt (!var_sum /. n) in
  {
    mean_reward = !mean_reward;
    std_reward;
    mean_length = !mean_length;
    n_episodes;
  }
