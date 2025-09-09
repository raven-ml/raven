(** REINFORCE algorithm implementation for Sokoban using Fehu + Kaun *)

open Fehu
module Rng = Rune.Rng
open Rune

(** REINFORCE agent using Kaun for policy network *)
module ReinforceAgent = struct
  type t = {
    policy_network : Kaun.module_;
    mutable policy_params : (Rune.float32_elt, [ `c ]) Kaun.params;
    baseline_network : Kaun.module_ option;
    mutable baseline_params : (Rune.float32_elt, [ `c ]) Kaun.params option;
    policy_optimizer :
      (Rune.float32_elt, [ `c ]) Kaun.Optimizer.gradient_transformation;
    mutable policy_opt_state : (Rune.float32_elt, [ `c ]) Kaun.Optimizer.opt_state;
    baseline_optimizer :
      (Rune.float32_elt, [ `c ]) Kaun.Optimizer.gradient_transformation option;
    mutable baseline_opt_state : (Rune.float32_elt, [ `c ]) Kaun.Optimizer.opt_state option;
    _rng : Rng.key;
    gamma : float;
    _learning_rate : float;
    use_baseline : bool;
  }

  (* Custom module to add channel dimension for conv2d *)
  let add_channel_dim () = {
    Kaun.Module.init = (fun ~rngs:_ _x -> Kaun.Ptree.List []);
    Kaun.Module.apply = (fun _ ~training:_ ?rngs:_ x ->
      (* Reshape from [H, W] to [1, H, W] for single channel *)
      Rune.reshape [| 1; 10; 10 |] x
    );
  }

  let create_policy_network n_actions =
    Kaun.Layer.sequential
      [
        add_channel_dim ();
        (* Conv layers to extract spatial features *)
        Kaun.Layer.conv2d ~in_channels:1 ~out_channels:16 ~kernel_size:(3, 3) ();
        Kaun.Layer.relu ();
        Kaun.Layer.conv2d ~in_channels:16 ~out_channels:32 ~kernel_size:(3, 3) ();
        Kaun.Layer.relu ();
        (* Flatten for fully connected layers *)
        Kaun.Layer.flatten ();
        (* After two 3x3 convolutions with default stride=1, output is 6x6 *)
        Kaun.Layer.linear ~in_features:(32 * 6 * 6) ~out_features:128 ();
        Kaun.Layer.relu ();
        Kaun.Layer.linear ~in_features:128 ~out_features:n_actions ();
      ]

  let create_baseline_network () =
    Kaun.Layer.sequential
      [
        add_channel_dim ();
        (* Conv layers *)
        Kaun.Layer.conv2d ~in_channels:1 ~out_channels:16 ~kernel_size:(3, 3) ();
        Kaun.Layer.relu ();
        Kaun.Layer.conv2d ~in_channels:16 ~out_channels:32 ~kernel_size:(3, 3) ();
        Kaun.Layer.relu ();
        (* Flatten and fully connected *)
        Kaun.Layer.flatten ();
        Kaun.Layer.linear ~in_features:(32 * 6 * 6) ~out_features:128 ();
        Kaun.Layer.relu ();
        Kaun.Layer.linear ~in_features:128 ~out_features:1 ();
      ]

  let create ~obs_dim:_ ~n_actions ~learning_rate ~gamma ~use_baseline ~seed =
    let rng = Rng.key seed in
    let keys = Rng.split ~n:2 rng in

    let policy_network = create_policy_network n_actions in
    (* Dummy input is 2D (10x10 grid) *)
    let dummy_input = Rune.zeros Rune.c Rune.float32 [| 10; 10 |] in
    let policy_params = Kaun.init policy_network ~rngs:keys.(0) dummy_input in

    let policy_optimizer = Kaun.Optimizer.adam ~lr:learning_rate () in
    let policy_opt_state = policy_optimizer.init policy_params in

    let baseline_network, baseline_params, baseline_optimizer, baseline_opt_state =
      if use_baseline then
        let net = create_baseline_network () in
        let params = Kaun.init net ~rngs:keys.(1) dummy_input in
        let opt = Kaun.Optimizer.adam ~lr:(learning_rate *. 10.0) () in
        let opt_state = opt.init params in
        (Some net, Some params, Some opt, Some opt_state)
      else
        (None, None, None, None)
    in

    {
      policy_network;
      policy_params;
      baseline_network;
      baseline_params;
      policy_optimizer;
      policy_opt_state;
      baseline_optimizer;
      baseline_opt_state;
      _rng = keys.(0);
      gamma;
      _learning_rate = learning_rate;
      use_baseline;
    }

  let select_action t obs ~training =
    let logits = Kaun.apply t.policy_network t.policy_params ~training obs in
    
    if training then
      let probs = Rune.softmax logits ~axes:[| -1 |] in
      let probs_array = Rune.to_array probs in
      
      let r = Random.float 1.0 in
      let rec sample_idx i cumsum =
        if i >= Array.length probs_array - 1 then i
        else if r <= cumsum +. probs_array.(i) then i
        else sample_idx (i + 1) (cumsum +. probs_array.(i))
      in
      
      let action_idx = sample_idx 0 0.0 in
      let action = Rune.scalar Rune.c Rune.float32 (float_of_int action_idx) in
      
      (* Compute log_softmax manually *)
      let max_logits = Rune.max logits ~axes:[| -1 |] ~keepdims:true in
      let exp_logits = Rune.exp (Rune.sub logits max_logits) in
      let sum_exp = Rune.sum exp_logits ~axes:[| -1 |] ~keepdims:true in
      let log_probs = Rune.sub logits (Rune.add max_logits (Rune.log sum_exp)) in
      let log_prob_array = Rune.to_array log_probs in
      let log_prob = log_prob_array.(action_idx) in
      
      (action, log_prob)
    else
      let action_idx = Rune.argmax logits ~axis:(-1) ~keepdims:false in
      let action = Rune.cast Rune.float32 action_idx in
      (action, 0.0)

  let predict_value t obs =
    match t.baseline_network, t.baseline_params with
    | Some net, Some params ->
      let value = Kaun.apply net params ~training:false obs in
      let value_array = Rune.to_array value in
      value_array.(0)
    | _ -> 0.0

  let collect_episode t env max_steps ?(log_frames=false) ?(render_fn=None) () =
    let obs, info = env.Env.reset () in
    let obs_ref = ref obs in
    let states = ref [] in
    let actions = ref [] in
    let rewards = ref [] in
    let log_probs = ref [] in
    let values = ref [] in
    let frames = ref [] in
    
    let finished = ref false in
    let steps = ref 0 in
    let stage_info = ref info in
    
    (* Capture initial state if logging *)
    if log_frames then begin
      let state_repr = match render_fn with
        | Some fn -> 
          let buffer = Stdlib.Buffer.create 256 in
          let old_stdout = Unix.dup Unix.stdout in
          let tmp_file = Filename.temp_file "sokoban" ".txt" in
          let fd = Unix.openfile tmp_file [Unix.O_RDWR; Unix.O_CREAT; Unix.O_TRUNC] 0o600 in
          Unix.dup2 fd Unix.stdout;
          Unix.close fd;
          fn ();
          flush stdout;
          Unix.dup2 old_stdout Unix.stdout;
          Unix.close old_stdout;
          let ic = open_in tmp_file in
          let result = try
            while true do
              Stdlib.Buffer.add_string buffer (input_line ic);
              Stdlib.Buffer.add_char buffer '\n'
            done;
            ""
          with End_of_file ->
            Stdlib.Buffer.contents buffer in
          close_in ic;
          Sys.remove tmp_file;
          result
        | None -> ""
      in
      let frame = Fehu.Visualization.{
        step = 0;
        state_repr;
        action = "Initial";
        reward = 0.0;
        value = None;
      } in
      frames := frame :: !frames
    end;
    
    while not !finished && !steps < max_steps do
      let obs = !obs_ref in
      let action, log_prob = select_action t obs ~training:true in
      let value = if t.use_baseline then predict_value t obs else 0.0 in
      
      let action_idx = int_of_float (Rune.to_array action).(0) in
      
      let next_obs, reward, terminated, truncated, info = env.Env.step action in
      if terminated || truncated then stage_info := info;
      
      (* Log frame after step if requested *)
      if log_frames then begin
        let state_repr = match render_fn with
          | Some fn -> 
            let buffer = Stdlib.Buffer.create 256 in
            let old_stdout = Unix.dup Unix.stdout in
            let tmp_file = Filename.temp_file "sokoban" ".txt" in
            let fd = Unix.openfile tmp_file [Unix.O_RDWR; Unix.O_CREAT; Unix.O_TRUNC] 0o600 in
            Unix.dup2 fd Unix.stdout;
            Unix.close fd;
            fn ();
            flush stdout;
            Unix.dup2 old_stdout Unix.stdout;
            Unix.close old_stdout;
            let ic = open_in tmp_file in
            let result = try
              while true do
                Stdlib.Buffer.add_string buffer (input_line ic);
                Stdlib.Buffer.add_char buffer '\n'
              done;
              ""
            with End_of_file ->
              Stdlib.Buffer.contents buffer in
            close_in ic;
            Sys.remove tmp_file;
            result
          | None -> ""
        in
        let frame = Fehu.Visualization.{
          step = !steps + 1;
          state_repr;
          action = Fehu.Visualization.action_to_string action_idx;
          reward;
          value = if t.use_baseline then Some value else None;
        } in
        frames := frame :: !frames
      end;
      
      states := obs :: !states;
      actions := action :: !actions;
      rewards := reward :: !rewards;
      log_probs := log_prob :: !log_probs;
      values := value :: !values;
      
      obs_ref := next_obs;
      finished := terminated || truncated;
      incr steps
    done;
    
    let trajectory = Trajectory.create
      ~states:(Array.of_list (List.rev !states))
      ~actions:(Array.of_list (List.rev !actions))
      ~rewards:(Array.of_list (List.rev !rewards))
      ~log_probs:(Some (Array.of_list (List.rev !log_probs)))
      ~values:(if t.use_baseline then Some (Array.of_list (List.rev !values)) else None)
      ()
    in
    
    (trajectory, !stage_info, List.rev !frames)

  let update t trajectory =
    let returns = Training.compute_returns 
      ~rewards:trajectory.Trajectory.rewards
      ~dones:(Array.make (Array.length trajectory.rewards) false)
      ~gamma:t.gamma
    in
    
    let advantages = 
      if t.use_baseline then
        match trajectory.values with
        | Some values ->
          Array.mapi (fun i r -> r -. values.(i)) returns
        | None -> returns
      else returns
    in
    
    let _log_probs = Option.get trajectory.log_probs in
    
    let policy_loss_grad params =
      let total_loss = ref (Rune.scalar Rune.c Rune.float32 0.0) in
      
      for i = 0 to Array.length trajectory.states - 1 do
        let state = trajectory.states.(i) in
        let action = trajectory.actions.(i) in
        let advantage = advantages.(i) in
        
        let logits = Kaun.apply t.policy_network params ~training:true state in
        (* Compute log_softmax manually *)
        let max_logits = Rune.max logits ~axes:[| -1 |] ~keepdims:true in
        let exp_logits = Rune.exp (Rune.sub logits max_logits) in
        let sum_exp = Rune.sum exp_logits ~axes:[| -1 |] ~keepdims:true in
        let log_probs_pred = Rune.sub logits (Rune.add max_logits (Rune.log sum_exp)) in
        let action_idx = int_of_float (Rune.to_array action).(0) in
        let log_prob_array = Rune.to_array log_probs_pred in
        let log_prob = log_prob_array.(action_idx) in
        let log_prob = Rune.scalar Rune.c Rune.float32 log_prob in
        
        let loss = Rune.mul_s (Rune.neg log_prob) advantage in
        total_loss := Rune.add !total_loss loss
      done;
      
      let avg_loss = Rune.div_s !total_loss 
        (float_of_int (Array.length trajectory.states)) in
      avg_loss
    in
    
    let (_policy_loss, policy_grads) = 
      Kaun.value_and_grad policy_loss_grad t.policy_params in
    
    let (policy_updates, new_policy_opt_state) = 
      t.policy_optimizer.update t.policy_opt_state t.policy_params policy_grads in
    
    t.policy_params <- Kaun.Optimizer.apply_updates t.policy_params policy_updates;
    t.policy_opt_state <- new_policy_opt_state;
    
    (if t.use_baseline then
      match t.baseline_network, t.baseline_params, t.baseline_optimizer, t.baseline_opt_state with
      | Some net, Some params, Some opt, Some opt_state ->
        let baseline_loss_grad params =
          let total_loss = ref (Rune.scalar Rune.c Rune.float32 0.0) in
          
          for i = 0 to Array.length trajectory.states - 1 do
            let state = trajectory.states.(i) in
            let return = returns.(i) in
            
            let value_pred = Kaun.apply net params ~training:true state in
            let value_array = Rune.to_array value_pred in
            let value = Rune.scalar Rune.c Rune.float32 value_array.(0) in
            let target = Rune.scalar Rune.c Rune.float32 return in
            let loss = Rune.square (Rune.sub value target) in
            
            total_loss := Rune.add !total_loss loss
          done;
          
          let avg_loss = Rune.div_s !total_loss 
            (float_of_int (Array.length trajectory.states)) in
          avg_loss
        in
        
        let (_, baseline_grads) = Kaun.value_and_grad baseline_loss_grad params in
        let (baseline_updates, new_baseline_opt_state) = opt.update opt_state params baseline_grads in
        
        t.baseline_params <- Some (Kaun.Optimizer.apply_updates params baseline_updates);
        t.baseline_opt_state <- Some new_baseline_opt_state
      | _ -> ());
    
    (* Return the episode return (from start to end) *)
    let episode_return = if Array.length returns > 0 then returns.(0) else 0.0 in
    episode_return
end

let train_reinforce env ~episodes ~max_steps ~learning_rate ~gamma ~use_baseline ~seed ?(log_episodes=false) () =
  let obs_dim = match env.Env.observation_space with
    | Space.Box { shape; _ } -> 
      (* For 2D observations, flatten to get total dimension *)
      Array.fold_left ( * ) 1 shape
    | _ -> failwith "Expected Box observation space"
  in
  
  let n_actions = match env.Env.action_space with
    | Space.Discrete n -> n
    | _ -> failwith "Expected Discrete action space"
  in
  
  let agent = ReinforceAgent.create ~obs_dim ~n_actions ~learning_rate ~gamma 
    ~use_baseline ~seed in
  
  let rewards_history = ref [] in
  let wins_history = ref [] in
  let total_wins = ref 0 in
  
  Printf.printf "Starting REINFORCE training%s\n" 
    (if use_baseline then " with baseline" else "");
  Printf.printf "Environment: Sokoban, Episodes: %d, LR: %.4f\n" 
    episodes learning_rate;
  
  let logged_episodes = ref [] in
  
  for episode = 1 to episodes do
    (* Determine if we should log this episode *)
    let should_log = log_episodes && (
      episode = 1 || 
      episode = episodes ||
      episode mod 100 = 0 ||
      (episode > episodes - 5)  (* Log last 5 episodes *)
    ) in
    
    let trajectory, info, frames = ReinforceAgent.collect_episode agent env max_steps 
      ~log_frames:should_log ~render_fn:(Some env.render) () in
    let episode_return = ReinforceAgent.update agent trajectory in
    
    let episode_reward = Array.fold_left (+.) 0.0 trajectory.rewards in
    let episode_length = Array.length trajectory.rewards in
    rewards_history := episode_reward :: !rewards_history;
    
    let won = episode_reward > 50.0 in
    if won then incr total_wins;
    wins_history := (if won then 1.0 else 0.0) :: !wins_history;
    
    (* Log episode if needed *)
    if should_log && frames <> [] then begin
      let stage = match List.assoc_opt "stage" info with
        | Some (`String s) -> Some s
        | _ -> None in
      let log = Fehu.Visualization.{
        episode_num = episode;
        total_reward = episode_reward;
        total_steps = Array.length trajectory.rewards;
        won;
        frames;
        stage;
      } in
      logged_episodes := log :: !logged_episodes;
      
      (* Save important episodes to file *)
      if episode = 1 || episode = episodes || (episode mod 500 = 0) then
        let filename = Printf.sprintf "logs/episode_%d_%s.log" episode
          (if won then "win" else "loss") in
        (try Unix.mkdir "logs" 0o755 with Unix.Unix_error _ -> ());
        Fehu.Visualization.save_episode_log log filename
    end;
    
    if episode mod 100 = 0 then begin
      let recent_rewards = List.filteri (fun i _ -> i < 100) !rewards_history in
      let avg_reward = List.fold_left (+.) 0.0 recent_rewards /. 
                       float_of_int (List.length recent_rewards) in
      
      let recent_wins = List.filteri (fun i _ -> i < 100) !wins_history in
      let recent_win_rate = List.fold_left (+.) 0.0 recent_wins /. 
                            float_of_int (List.length recent_wins) *. 100.0 in
      
      let stage_str = try 
        match List.assoc "stage" info with
        | `String s -> s
        | _ -> "unknown"
      with Not_found -> "N/A" in
      
      let advanced = try 
        match List.assoc "advanced" info with
        | `Bool true -> " (ADVANCED!)"
        | _ -> ""
      with Not_found -> "" in
      
      Printf.printf "Episode %d: Avg Reward = %.2f, Win Rate = %.1f%% (%.1f%%), Return = %.2f, Steps = %d, Stage = %s%s\n"
        episode avg_reward 
        recent_win_rate
        (float_of_int !total_wins /. float_of_int episode *. 100.0)
        episode_return
        episode_length
        stage_str
        advanced;
      flush stdout
    end
  done;
  
  Printf.printf "Training complete! Final win rate: %.1f%%\n"
    (float_of_int !total_wins /. float_of_int episodes *. 100.0);
  
  List.rev !logged_episodes

let () =
  (* Start with a simple corridor for testing *)
  Printf.printf "Training REINFORCE on simple corridor...\n";
  let initial_state = Sokoban.LevelGen.generate_corridor 3 in
  let env = Sokoban.sokoban ~width:5 ~height:3 ~max_steps:50 
    ~initial_state () in
  
  let corridor_logs = train_reinforce env
    ~episodes:500
    ~max_steps:50
    ~learning_rate:0.001
    ~gamma:0.99
    ~use_baseline:true
    ~seed:42
    ~log_episodes:true
    () in
  
  Printf.printf "\nNow training with curriculum learning...\n";
  
  let curriculum_env = Sokoban.sokoban_curriculum ~max_steps:200 () in
  
  let curriculum_logs = train_reinforce curriculum_env
    ~episodes:2000
    ~max_steps:200
    ~learning_rate:0.0005
    ~gamma:0.99
    ~use_baseline:true
    ~seed:43
    ~log_episodes:true
    () in
  
  (* Print summaries *)
  Printf.printf "\n=== Corridor Training Summary ===\n";
  Fehu.Visualization.summary_stats corridor_logs;
  
  Printf.printf "\n=== Curriculum Training Summary ===\n";
  Fehu.Visualization.summary_stats curriculum_logs;
  
  (* Offer to visualize episodes *)
  Printf.printf "\nVisualize sample episodes? (y/n): ";
  flush stdout;
  let response = try input_line stdin with _ -> "n" in
  if String.lowercase_ascii response = "y" then begin
    Printf.printf "Showing sample episodes...\n";
    let interesting = List.filter (fun log -> 
      log.Fehu.Visualization.episode_num = 1 || 
      log.episode_num mod 500 = 0 ||
      log.won
    ) curriculum_logs in
    
    List.iter (fun log ->
      Fehu.Visualization.animate_episode log 0.3;
      Printf.printf "Press Enter to continue...";
      flush stdout;
      let _ = try input_line stdin with _ -> "" in ()
    ) (match interesting with
       | a :: b :: c :: _ -> [a; b; c]
       | lst -> lst)
  end;
  
  Printf.printf "\nDone!\n"