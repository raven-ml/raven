(** Policy gradient training for RandomWalk using Fehu Algorithms *)

open Fehu
open Kaun

(** Create a simple policy network *)
let create_policy_network () =
  Layer.sequential
    [
      Layer.linear ~in_features:1 ~out_features:32 ();
      Layer.tanh ();
      Layer.linear ~in_features:32 ~out_features:2 ();
    ]

let run_policy_gradient () =
  Printf.printf "Training policy gradient on RandomWalk environment...\n";

  (* Create environment *)
  let env = Fehu_envs.Random_walk.make ~rng:(Rune.Rng.key 2) () in

  (* Create policy network *)
  let policy_net = create_policy_network () in

  (* Initialize REINFORCE agent *)
  let rng = Rune.Rng.key 42 in
  let agent =
    Fehu_algorithms.Reinforce.create ~policy_network:policy_net ~n_actions:2
      ~rng
      Fehu_algorithms.Reinforce.
        {
          learning_rate = 0.001;
          (* Reduced for more stable learning *)
          gamma = 0.95;
          (* Lower gamma to value immediate rewards *)
          use_baseline = false;
          reward_scale = 0.1;
          (* Scale down rewards to reduce variance *)
          entropy_coef = 0.01;
          max_episode_steps = 200;
        }
  in

  (* Train for 500 episodes *)
  let agent =
    Fehu_algorithms.Reinforce.learn agent ~env ~total_timesteps:50_000
      ~callback:(fun ~iteration ~metrics ->
        if iteration mod 50 = 0 then
          Printf.printf "Episode %d: Return = %.2f, Length = %d\n%!" iteration
            metrics.episode_return metrics.episode_length;
        true)
      ()
  in

  (* Evaluate learned policy *)
  Printf.printf "\n=== Evaluation ===\n";
  let eval_episodes = 20 in
  let total_rewards = ref 0.0 in

  for episode = 1 to eval_episodes do
    let obs, _info = Env.reset env () in
    let obs_ref = ref obs in
    let done_flag = ref false in
    let episode_reward = ref 0.0 in

    while not !done_flag do
      (* Use greedy policy *)
      let action, _ =
        Fehu_algorithms.Reinforce.predict agent !obs_ref ~training:false
      in
      let transition = Env.step env action in
      episode_reward := !episode_reward +. transition.reward;
      obs_ref := transition.observation;
      done_flag := transition.terminated || transition.truncated
    done;

    total_rewards := !total_rewards +. !episode_reward;
    Printf.printf "Eval episode %d: reward = %.2f\n" episode !episode_reward
  done;

  Printf.printf "Average evaluation reward: %.2f\n"
    (!total_rewards /. float_of_int eval_episodes);

  Env.close env

let () = run_policy_gradient ()
