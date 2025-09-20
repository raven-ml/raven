open Fehu

let run_random_agent () =
  (* Derive independent keys for the environment and the agent policy *)
  let root_key = Rune.Rng.key 0 in
  let keys = Rune.Rng.split root_key in
  let env = Fehu_envs.Random_walk.make ~rng:keys.(0) () in
  let agent_key = ref keys.(1) in

  let take_agent_key () =
    let split = Rune.Rng.split !agent_key in
    agent_key := split.(0);
    split.(1)
  in

  (* Reset the environment with a reproducible key *)
  Env.set_rng env (Rune.Rng.key 42);
  let obs, _info = Env.reset env () in
  let obs_vals = Rune.to_array obs in
  Printf.printf "Initial observation: [|%f|]\n%!" obs_vals.(0);

  (* Run for 10 episodes *)
  for episode = 1 to 10 do
    Printf.printf "\n=== Episode %d ===\n%!" episode;

    (* Reset for new episode *)
    let _obs, _info = Env.reset env () in
    let done_ = ref false in
    let total_reward = ref 0.0 in
    let step_count = ref 0 in

    while not !done_ do
      incr step_count;

      (* Sample random action (RandomWalk has 2 actions: 0=left, 1=right) *)
      let action_tensor =
        Rune.Rng.randint (take_agent_key ()) ~min:0 ~max:2 [| 1 |]
      in
      let action_values : Int32.t array = Rune.to_array action_tensor in
      let action_index = Int32.to_int action_values.(0) in
      let action = Rune.scalar Rune.int32 action_values.(0) in

      (* Take step in environment *)
      let transition = Env.step env action in
      let obs = transition.observation in
      let obs_vals = Rune.to_array obs in
      let reward = transition.reward in
      let terminated = transition.terminated in
      let truncated = transition.truncated in

      total_reward := !total_reward +. reward;
      done_ := terminated || truncated;

      Printf.printf "Step %d: Action=%d, Reward=%.2f, Done=%b, Obs=[|%f|]\n%!"
        !step_count action_index reward !done_ obs_vals.(0);

      (* Render environment if available *)
      match Env.render env with
      | Some render -> Printf.printf "%s\n%!" render
      | None -> ()
    done;

    Printf.printf
      "Episode %d finished after %d steps with total reward %.2f\n%!" episode
      !step_count !total_reward
  done;

  (* Close environment *)
  Env.close env

let () = run_random_agent ()
