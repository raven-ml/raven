(** REINFORCE training for Sokoban using Fehu Algorithms *)

open Fehu
open Kaun

let grid_size = Sokoban_env.max_grid_size

(** Custom layer to reshape flat observations into 2D grid for conv layers *)
let add_channel_dim () =
  {
    init = (fun ~rngs:_ ~dtype:_ -> Ptree.List []);
    apply =
      (fun _ ~training:_ ?rngs:_ x ->
        (* Input: [batch, grid_size^2] -> Output: [batch, 1, grid_size,
           grid_size] *)
        let batch_size = (Rune.shape x).(0) in
        Rune.reshape [| batch_size; 1; grid_size; grid_size |] x);
  }

(** Create policy network with CNN architecture *)
let create_policy_network n_actions =
  Layer.sequential
    [
      add_channel_dim ();
      Layer.conv2d ~in_channels:1 ~out_channels:16 ~kernel_size:(3, 3) ();
      Layer.relu ();
      Layer.conv2d ~in_channels:16 ~out_channels:32 ~kernel_size:(3, 3) ();
      Layer.relu ();
      Layer.flatten ();
      Layer.linear
        ~in_features:(32 * grid_size * grid_size)
        ~out_features:128 ();
      Layer.relu ();
      Layer.linear ~in_features:128 ~out_features:n_actions ();
    ]

(** Create value network (baseline) *)
let create_value_network () =
  Layer.sequential
    [
      add_channel_dim ();
      Layer.conv2d ~in_channels:1 ~out_channels:16 ~kernel_size:(3, 3) ();
      Layer.relu ();
      Layer.conv2d ~in_channels:16 ~out_channels:32 ~kernel_size:(3, 3) ();
      Layer.relu ();
      Layer.flatten ();
      Layer.linear
        ~in_features:(32 * grid_size * grid_size)
        ~out_features:128 ();
      Layer.relu ();
      Layer.linear ~in_features:128 ~out_features:1 ();
    ]

type train_config = {
  episodes : int;
  learning_rate : float;
  gamma : float;
  use_baseline : bool;
  reward_scale : float;
  entropy_coef : float;
  max_steps : int;
  seed : int;
}
(** Training configuration *)

let default_config =
  {
    episodes = 500;
    learning_rate = 0.001;
    gamma = 0.99;
    use_baseline = true;
    reward_scale = 0.01;
    entropy_coef = 0.01;
    max_steps = 40;
    seed = 42;
  }

(** Train REINFORCE agent *)
let train env config =
  Printf.printf "Starting REINFORCE training%s\n%!"
    (if config.use_baseline then " with baseline" else "");
  Printf.printf "Episodes: %d, LR: %.4f, Gamma: %.2f\n%!" config.episodes
    config.learning_rate config.gamma;

  (* Create networks *)
  let policy_net = create_policy_network 4 in
  let value_net =
    if config.use_baseline then Some (create_value_network ()) else None
  in

  (* Initialize agent *)
  let rng = Rune.Rng.key config.seed in
  let agent =
    Fehu_algorithms.Reinforce.create ~policy_network:policy_net
      ?baseline_network:value_net ~n_actions:4 ~rng
      Fehu_algorithms.Reinforce.
        {
          learning_rate = config.learning_rate;
          gamma = config.gamma;
          use_baseline = config.use_baseline;
          reward_scale = config.reward_scale;
          entropy_coef = config.entropy_coef;
          max_episode_steps = config.max_steps;
        }
  in

  (* Training metrics *)
  let rewards_history = ref [] in
  let wins_history = ref [] in
  let total_wins = ref 0 in

  (* Training loop with callback *)
  let episode = ref 0 in
  let agent =
    Fehu_algorithms.Reinforce.learn agent ~env
      ~total_timesteps:(config.episodes * config.max_steps)
      ~callback:(fun ~iteration ~metrics ->
        episode := iteration;

        (* Track metrics *)
        rewards_history := metrics.episode_return :: !rewards_history;
        let won = metrics.episode_return > 50.0 in
        if won then incr total_wins;
        wins_history := (if won then 1.0 else 0.0) :: !wins_history;

        (* Log progress *)
        if iteration mod 100 = 0 then (
          let take_recent lst = List.filteri (fun i _ -> i < 100) lst in
          let recent_rewards = take_recent !rewards_history in
          let avg_reward =
            if recent_rewards = [] then 0.0
            else
              List.fold_left ( +. ) 0.0 recent_rewards
              /. float_of_int (List.length recent_rewards)
          in

          let recent_wins = take_recent !wins_history in
          let recent_win_rate =
            if recent_wins = [] then 0.0
            else
              List.fold_left ( +. ) 0.0 recent_wins
              /. float_of_int (List.length recent_wins)
              *. 100.0
          in

          Printf.printf
            "Episode %d: Avg Reward = %.2f, Win Rate = %.1f%% (%.1f%%), Length \
             = %d\n\
             %!"
            iteration avg_reward recent_win_rate
            (float_of_int !total_wins /. float_of_int iteration *. 100.0)
            metrics.episode_length;
          Printf.printf
            "           Entropy = %.3f, Log Prob = %.3f, Adv Mean = %.3f, Adv \
             Std = %.3f"
            metrics.avg_entropy metrics.avg_log_prob
            (metrics.adv_mean /. config.reward_scale)
            (metrics.adv_std /. config.reward_scale);
          (match metrics.value_loss with
          | Some v -> Printf.printf ", Value Loss = %.3f" v
          | None -> ());
          Printf.printf "\n%!";
          flush stdout);

        (* Continue training *)
        true)
      ()
  in

  Printf.printf "Training complete! Final win rate: %.1f%%\n%!"
    (float_of_int !total_wins /. float_of_int !episode *. 100.0);

  agent

(** Evaluate policy and compute win rate *)
let evaluate_with_wins agent env ~n_episodes =
  let wins = ref 0 in
  let total_reward = ref 0.0 in

  for _ = 1 to n_episodes do
    let obs, _ = Env.reset env () in
    let obs_ref = ref obs in
    let finished = ref false in
    let episode_reward = ref 0.0 in

    while not !finished do
      let action, _ =
        Fehu_algorithms.Reinforce.predict agent !obs_ref ~training:false
      in
      let transition = Env.step env action in
      episode_reward := !episode_reward +. transition.reward;
      obs_ref := transition.observation;
      finished := transition.terminated || transition.truncated
    done;

    total_reward := !total_reward +. !episode_reward;
    if !episode_reward > 50.0 then incr wins
  done;

  let n = float_of_int n_episodes in
  (float_of_int !wins /. n *. 100.0, !total_reward /. n)

(** Main entry point *)
let () =
  Printexc.record_backtrace true;

  Printf.printf "Training REINFORCE on Sokoban corridor levels...\n%!";

  (* Create training environment *)
  let corridor_initial_state () = Sokoban_env.Level_gen.generate_corridor 7 in
  let env =
    Sokoban_env.sokoban ~max_steps:40 ~initial_state_fn:corridor_initial_state
      ()
  in

  (* Train *)
  let config = default_config in
  let agent = train env config in

  (* Evaluate trained policy *)
  Printf.printf "\nEvaluating trained policy...\n%!";
  let win_rate, avg_reward = evaluate_with_wins agent env ~n_episodes:100 in
  Printf.printf "Evaluation: Win rate = %.1f%%, Avg reward = %.2f\n%!" win_rate
    avg_reward;

  (* Compare with random policy *)
  Printf.printf "\nEvaluating random policy...\n%!";
  let rng = ref (Rune.Rng.key 0xC0FFEE) in
  let random_policy _obs =
    let keys = Rune.Rng.split !rng ~n:2 in
    rng := keys.(0);
    fst (Space.sample ~rng:keys.(1) (Env.action_space env))
  in

  let random_stats =
    Fehu.Training.evaluate env ~policy:random_policy ~n_episodes:100 ()
  in
  Printf.printf "Random policy: Win rate = %.1f%%, Avg reward = %.2f\n%!" 0.0
    random_stats.mean_reward;

  Env.close env;
  Printf.printf "\nDone!\n%!"
