(** Simple example running CartPole with random actions *)

open Fehu
module Rng = Rune.Rng

let () =
  (* Create the CartPole environment *)
  let env = Envs.cartpole () in

  Printf.printf "CartPole environment created!\n";
  Printf.printf "Observation space: Box with shape [|4|]\n";
  Printf.printf "Action space: Discrete(2)\n\n";

  (* Create RNG for random actions *)
  let rng = Rng.key 42 in

  (* Run a few episodes with random actions *)
  let n_episodes = 5 in

  for episode = 1 to n_episodes do
    let _obs, _info = env.reset () in
    let episode_reward = ref 0.0 in
    let step_count = ref 0 in
    let finished = ref false in

    Printf.printf "Episode %d:\n" episode;

    while not !finished do
      (* Take a random action *)
      let action = Space.sample ~rng env.action_space in

      (* Step the environment *)
      let _next_obs, reward, terminated, truncated, _info = env.step action in

      episode_reward := !episode_reward +. reward;
      incr step_count;
      finished := terminated || truncated;

      if !step_count mod 20 = 0 then
        Printf.printf "  Step %d: reward = %.2f\n" !step_count reward
    done;

    Printf.printf "  Episode finished after %d steps\n" !step_count;
    Printf.printf "  Total reward: %.2f\n\n" !episode_reward
  done;

  Printf.printf "Random agent demonstration completed!\n"
