(** REINFORCE training for Sokoban using Fehu Algorithms *)

open Fehu
open Kaun

let grid_size = Sokoban_env.max_grid_size

let record_guard description f =
  try f ()
  with Invalid_argument msg ->
    Printf.eprintf "Warning: %s (%s)\n%!" description msg

let make_training_env ?render_mode ~max_steps () =
  Sokoban_env.sokoban_curriculum ?render_mode ~max_steps ()

let demo_generators =
  [
    (fun () -> Sokoban_env.Level_gen.generate_corridor 5);
    (fun () -> Sokoban_env.Level_gen.generate_room 5);
    (fun () -> Sokoban_env.Level_gen.generate_room 7);
    (fun () -> Sokoban_env.Level_gen.generate_multi_box 3);
    (fun () -> Sokoban_env.Level_gen.generate_complex ());
  ]

let demo_random = Random.State.make [| 0x51C0B |]

let random_demo_state () =
  let idx = Random.State.int demo_random (List.length demo_generators) in
  (List.nth demo_generators idx) ()

let make_demo_env ?render_mode ~max_steps () =
  Sokoban_env.sokoban ?render_mode ~max_steps
    ~initial_state_fn:random_demo_state ()

(** Custom layer to reshape flat observations into 2D grid for conv layers *)
let add_channel_dim ~n_channels () =
  {
    init = (fun ~rngs:_ ~dtype:_ -> Ptree.List []);
    apply =
      (fun _ ~training:_ ?rngs:_ x ->
        let shape = Rune.shape x in
        let batch =
          match Array.length shape with 1 -> 1 | 0 -> 1 | _ -> shape.(0)
        in
        Rune.reshape [| batch; n_channels; grid_size; grid_size |] x);
  }

(** Create policy network with CNN architecture *)
let create_policy_network n_actions =
  Layer.sequential
    [
      add_channel_dim ~n_channels:8 ();
      Layer.conv2d ~in_channels:8 ~out_channels:16 ~kernel_size:(3, 3) ();
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
      add_channel_dim ~n_channels:8 ();
      Layer.conv2d ~in_channels:8 ~out_channels:16 ~kernel_size:(3, 3) ();
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

let mask_from_info info =
  match Info.find "action_mask" info with
  | Some (Info.Bool_array arr) -> Some arr
  | _ -> None

let apply_action_mask logits = function
  | Some mask ->
      let shape = Rune.shape logits in
      let n_actions = shape.(Array.length shape - 1) in
      let len = Array.length mask in
      let mask_offsets =
        Array.init n_actions (fun idx ->
            if idx < len && mask.(idx) then 0.0 else -1e9)
      in
      let mask_tensor =
        Rune.create Rune.float32 [| 1; n_actions |] mask_offsets
      in
      Rune.add logits mask_tensor
  | None -> logits

let greedy_action ?mask policy_net params obs =
  let obs_batched =
    match Rune.shape obs with
    | [| features |] -> Rune.reshape [| 1; features |] obs
    | [| 1; _ |] -> obs
    | _ -> obs
  in
  let obs_batched = Rune.cast Rune.float32 obs_batched in
  let logits = apply policy_net params ~training:false obs_batched in
  let logits = apply_action_mask logits mask in
  let action_idx =
    Rune.argmax logits ~axis:(-1) ~keepdims:false |> Rune.cast Rune.int32
  in
  let action_scalar =
    Rune.reshape [||] action_idx |> Rune.to_array |> fun arr -> arr.(0)
  in
  Rune.scalar Rune.int32 action_scalar

let safe_stage env =
  match Sokoban_env.current_stage_descriptor_opt env with
  | Some s -> s
  | None -> "unknown-stage"

let record_random_rollout ~path ~max_steps =
  let env = make_demo_env ~render_mode:`Rgb_array ~max_steps () in
  let policy = Policy.random env in
  Fehu_visualize.Sink.with_ffmpeg ~fps:12 ~path (fun sink ->
      Fehu_visualize.record_rollout ~env ~policy ~steps:max_steps ~sink ());
  Env.close env

let record_trained_rollout ~level ~path ~max_steps ~policy_net ~params =
  let env =
    Sokoban_env.sokoban ~render_mode:`Rgb_array ~max_steps
      ~initial_state:(Sokoban_env.Core.copy_state level)
      ()
  in
  let policy =
    Policy.deterministic (fun obs ->
        let mask =
          try
            let state = Sokoban_env.current_game_state env in
            Some (Sokoban_env.action_mask state)
          with Invalid_argument _ -> None
        in
        greedy_action ?mask policy_net params obs)
  in
  Fehu_visualize.Sink.with_ffmpeg ~fps:12 ~path (fun sink ->
      Fehu_visualize.record_rollout ~env ~policy ~steps:max_steps ~sink ());
  Env.close env

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
    reward_scale = 0.1;
    entropy_coef = 0.02;
    max_steps = 40;
    seed = 42;
  }

(** Train REINFORCE agent *)
let train ?record_dir env config =
  Printf.printf "Starting REINFORCE training%s\n%!"
    (if config.use_baseline then " with baseline" else "");
  Printf.printf "Episodes: %d, LR: %.4f, Gamma: %.2f\n%!" config.episodes
    config.learning_rate config.gamma;

  (* Create networks *)
  let policy_net = create_policy_network 4 in
  let value_net =
    if config.use_baseline then Some (create_value_network ()) else None
  in

  let rng = Rune.Rng.key config.seed in
  let alg_config =
    {
      Fehu_algorithms.Reinforce.learning_rate = config.learning_rate;
      gamma = config.gamma;
      use_baseline = config.use_baseline;
      reward_scale = config.reward_scale;
      entropy_coef = config.entropy_coef;
      max_episode_steps = config.max_steps;
    }
  in

  (* Tracking metrics *)
  let rewards_history = ref [] in
  let wins_history = ref [] in
  let total_wins = ref 0 in
  let last_episode = ref 0 in
  let params0, state0 =
    Fehu_algorithms.Reinforce.init ~env ~policy_network:policy_net
      ?baseline_network:value_net ~rng ~config:alg_config ()
  in
  let params_ref = ref params0 in
  let state_ref = ref state0 in

  Option.iter
    (fun dir ->
      let stage_desc = safe_stage env in
      let level =
        match Sokoban_env.current_game_state_opt env with
        | Some lvl -> lvl
        | None -> Sokoban_env.Core.copy_state (random_demo_state ())
      in
      let path =
        Filename.concat dir
          (Printf.sprintf "sokoban_train_ep%04d_%s.mp4" 0 stage_desc)
      in
      Printf.printf "Recording rollout at episode 0 (%s) to %s\n%!" stage_desc
        path;
      record_guard "recording initial rollout" (fun () ->
          record_trained_rollout ~level ~path ~max_steps:config.max_steps
            ~policy_net ~params:!params_ref))
    record_dir;

  let target_timesteps = config.episodes * config.max_steps in
  let continue = ref true in
  while !continue do
    let params', state' =
      Fehu_algorithms.Reinforce.step ~env ~params:!params_ref ~state:!state_ref
    in
    params_ref := params';
    state_ref := state';
    let metrics = Fehu_algorithms.Reinforce.metrics state' in
    if metrics.total_episodes > !last_episode then (
      let stage_desc = metrics.stage_desc in
      last_episode := metrics.total_episodes;
      rewards_history := metrics.episode_return :: !rewards_history;
      let won = metrics.episode_won in
      if won then incr total_wins;
      wins_history := (if won then 1.0 else 0.0) :: !wins_history;

      if metrics.total_episodes mod 100 = 0 then (
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
          "Episode %d (Stage %s): Avg Reward = %.2f, Win Rate = %.1f%% \
           (%.1f%%), Length = %d\n\
           %!"
          metrics.total_episodes stage_desc avg_reward recent_win_rate
          (float_of_int !total_wins
          /. float_of_int metrics.total_episodes
          *. 100.0)
          metrics.episode_length;
        Printf.printf
          "           Entropy = %.3f, Log Prob = %.3f, Adv Mean = %.3f, Adv \
           Std = %.3f"
          metrics.avg_entropy metrics.avg_log_prob metrics.adv_mean
          metrics.adv_std;
        (match metrics.value_loss with
        | Some v -> Printf.printf ", Value Loss = %.3f" v
        | None -> ());
        Printf.printf "\n%!";
        flush stdout);

      Option.iter
        (fun dir ->
          if metrics.total_episodes mod 50 = 0 then (
            let stage_desc = metrics.stage_desc in
            let level =
              match Sokoban_env.current_game_state_opt env with
              | Some lvl -> lvl
              | None -> Sokoban_env.Core.copy_state (random_demo_state ())
            in
            let path =
              Filename.concat dir
                (Printf.sprintf "sokoban_train_ep%04d_%s.mp4"
                   metrics.total_episodes stage_desc)
            in
            Printf.printf "Recording rollout at episode %d (Stage %s) to %s\n%!"
              metrics.total_episodes stage_desc path;
            record_guard "recording training rollout" (fun () ->
                record_trained_rollout ~level ~path ~max_steps:config.max_steps
                  ~policy_net ~params:!params_ref)))
        record_dir);

    if
      metrics.total_episodes >= config.episodes
      || metrics.total_steps >= target_timesteps
    then continue := false
  done;

  let params = !params_ref in
  let state = !state_ref in

  let final_episodes = if !last_episode = 0 then 1 else !last_episode in
  Printf.printf "Training complete! Final win rate: %.1f%%\n%!"
    (float_of_int !total_wins /. float_of_int final_episodes *. 100.0);

  (params, state, policy_net)

(** Evaluate policy and compute win rate *)
let evaluate_with_wins ~policy_net ~params env ~n_episodes =
  let wins = ref 0 in
  let total_reward = ref 0.0 in

  for _ = 1 to n_episodes do
    let obs, info = Env.reset env () in
    let obs_ref = ref obs in
    let info_ref = ref info in
    let finished = ref false in
    let episode_reward = ref 0.0 in
    let episode_won = ref false in

    while not !finished do
      let mask = mask_from_info !info_ref in
      let action = greedy_action ?mask policy_net params !obs_ref in
      let transition = Env.step env action in
      episode_reward := !episode_reward +. transition.reward;
      obs_ref := transition.observation;
      info_ref := transition.info;
      episode_won := transition.terminated;
      finished := transition.terminated || transition.truncated
    done;

    total_reward := !total_reward +. !episode_reward;
    if !episode_won then incr wins
  done;

  let n = float_of_int n_episodes in
  (float_of_int !wins /. n *. 100.0, !total_reward /. n)

(** Main entry point *)
let () =
  Printexc.record_backtrace true;

  Printf.printf "Training REINFORCE on Sokoban corridor levels...\n%!";

  let config = default_config in
  let record_dir = Sys.getenv_opt "FEHU_SOKOBAN_RECORD_DIR" in

  Option.iter
    (fun dir ->
      let random_path = Filename.concat dir "sokoban_random.mp4" in
      Printf.printf "Recording random rollout to %s\n%!" random_path;
      record_guard "recording random rollout" (fun () ->
          record_random_rollout ~path:random_path ~max_steps:config.max_steps))
    record_dir;

  (* Create training environment *)
  let env = make_training_env ~max_steps:config.max_steps () in

  (* Train *)
  let params, state, policy_net = train ?record_dir env config in
  let final_metrics = Fehu_algorithms.Reinforce.metrics state in
  let final_stage_desc = final_metrics.stage_desc in
  let final_level =
    match Sokoban_env.current_game_state_opt env with
    | Some lvl -> lvl
    | None -> Sokoban_env.Core.copy_state (random_demo_state ())
  in

  (* Evaluate trained policy *)
  Printf.printf "\nEvaluating trained policy across curriculum...\n%!";
  let eval_env = make_training_env ~max_steps:config.max_steps () in
  let win_rate, avg_reward =
    evaluate_with_wins ~policy_net ~params eval_env ~n_episodes:100
  in
  Printf.printf "Evaluation: Win rate = %.1f%%, Avg reward = %.2f\n%!" win_rate
    avg_reward;
  Env.close eval_env;

  Option.iter
    (fun dir ->
      let trained_path =
        Filename.concat dir
          (Printf.sprintf "sokoban_trained_%s.mp4" final_stage_desc)
      in
      Printf.printf "Recording trained rollout (%s) to %s\n%!" final_stage_desc
        trained_path;
      record_guard "recording trained rollout" (fun () ->
          record_trained_rollout ~level:final_level ~path:trained_path
            ~max_steps:config.max_steps ~policy_net ~params))
    record_dir;

  (* Compare with random policy *)
  Printf.printf "\nEvaluating random policy on stage %s...\n%!" final_stage_desc;
  let random_env =
    Sokoban_env.sokoban ~max_steps:config.max_steps ~initial_state:final_level
      ()
  in
  let random_policy = Policy.random random_env in
  let random_stats =
    Fehu.Training.evaluate random_env
      ~policy:(fun obs ->
        let action, _, _ = random_policy obs in
        action)
      ~n_episodes:100 ~max_steps:config.max_steps ()
  in
  Printf.printf "Random policy: Win rate = %.1f%%, Avg reward = %.2f\n%!" 0.0
    random_stats.mean_reward;

  Env.close random_env;
  Env.close env;
  Printf.printf "\nDone!\n%!"
