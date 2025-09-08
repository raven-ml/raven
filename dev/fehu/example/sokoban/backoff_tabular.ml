(** Backoff-Tabular Q-learning baseline for Sokoban 
    Ported from sokoban-rl-playground to work with Fehu/Gymnasium API *)

open Fehu

module StateHash = struct
  type t = string
  let equal = String.equal
  let hash = Hashtbl.hash
end

module QTable = Hashtbl.Make(StateHash)
module CountTable = Hashtbl.Make(StateHash)

type q_learning_params = {
  learning_rate: float;
  discount_factor: float;
  epsilon_start: float;
  epsilon_end: float;
  epsilon_decay: float;
  max_steps: int [@warning "-69"];
  window_sizes: int list;  (* List of window sizes, e.g., [3; 5; 7; -1] where -1 means full state *)
}

type q_table_level = {
  q_table: float QTable.t;
  count_table: int CountTable.t;
  window_size: int;  (* -1 for full state *)
}

type backoff_agent = {
  q_levels: q_table_level list;  (* Ordered from smallest to largest window *)
  params: q_learning_params;
  mutable epsilon: float;
  mutable steps: int;
}

let default_params = {
  learning_rate = 0.1;
  discount_factor = 0.95;
  epsilon_start = 1.0;
  epsilon_end = 0.01;
  epsilon_decay = 0.995;
  max_steps = 500;
  window_sizes = [3; 5; 7; -1];  (* 3x3, 5x5, 7x7 windows, then full state *)
}

(** Convert observation array to game state for processing *)
let obs_to_state obs =
  (* Get shape of observation tensor *)
  let shape = Rune.shape obs in
  let height = shape.(0) in
  let width = shape.(1) in
  
  let features = Rune.unsafe_to_array obs in
  
  let state = Sokoban.Core.{
    grid = Array.make_matrix height width Sokoban.Core.Empty;
    player_pos = (0, 0);
    width;
    height;
  } in
  
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let idx = y * width + x in  (* Flat index for underlying array *)
      let cell = match int_of_float features.(idx) with
        | 0 -> Sokoban.Core.Empty
        | 1 -> Sokoban.Core.Wall
        | 2 -> Sokoban.Core.Box
        | 3 -> Sokoban.Core.Target
        | 4 -> Sokoban.Core.BoxOnTarget
        | 5 -> Sokoban.Core.Player
        | 6 -> Sokoban.Core.PlayerOnTarget
        | _ -> Sokoban.Core.Empty
      in
      state.grid.(y).(x) <- cell
    done
  done;
  
  (* Find player position *)
  let player_pos = ref (0, 0) in
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      match state.grid.(y).(x) with
      | Sokoban.Core.Player | Sokoban.Core.PlayerOnTarget ->
        player_pos := (x, y)
      | _ -> ()
    done
  done;
  
  { state with player_pos = !player_pos }

(** Extract a window of the state centered on the player *)
let extract_window state window_size =
  if window_size = -1 then
    (* Full state *)
    state
  else
    let (px, py) = state.Sokoban.Core.player_pos in
    let half_window = window_size / 2 in
    let window_state = Sokoban.Core.{
      grid = Array.make_matrix window_size window_size Wall;
      player_pos = (half_window, half_window);
      width = window_size;
      height = window_size;
    } in
    
    (* Copy visible portion of the state *)
    for dy = -half_window to half_window do
      for dx = -half_window to half_window do
        let world_x = px + dx in
        let world_y = py + dy in
        let window_x = dx + half_window in
        let window_y = dy + half_window in
        
        if world_x >= 0 && world_x < state.width && 
           world_y >= 0 && world_y < state.height then
          window_state.grid.(window_y).(window_x) <- state.grid.(world_y).(world_x)
      done
    done;
    
    window_state

let state_to_key state =
  (* Find the actual game bounds (non-empty area) *)
  let min_x = ref state.Sokoban.Core.width in
  let max_x = ref (-1) in
  let min_y = ref state.Sokoban.Core.height in
  let max_y = ref (-1) in
  
  for y = 0 to state.Sokoban.Core.height - 1 do
    for x = 0 to state.Sokoban.Core.width - 1 do
      match state.grid.(y).(x) with
      | Empty -> ()
      | _ ->
        min_x := min !min_x x;
        max_x := max !max_x x;
        min_y := min !min_y y;
        max_y := max !max_y y
    done
  done;
  
  (* If completely empty, show full grid *)
  if !max_x < 0 then begin
    min_x := 0;
    max_x := state.width - 1;
    min_y := 0;
    max_y := state.height - 1
  end;
  
  (* Build string representation of active area only *)
  let buffer = Stdlib.Buffer.create 256 in
  for y = !min_y to !max_y do
    for x = !min_x to !max_x do
      let c = match state.grid.(y).(x) with
        | Empty -> ' '
        | Wall -> '#'
        | Box -> '$'
        | Target -> '.'
        | BoxOnTarget -> '*'
        | Player -> '@'
        | PlayerOnTarget -> '+'
      in
      Stdlib.Buffer.add_char buffer c
    done;
    Stdlib.Buffer.add_char buffer '|';
    Stdlib.Buffer.add_char buffer '\n'
  done;
  Stdlib.Buffer.contents buffer

(** Get Q-value using backoff strategy *)
let get_q_value_backoff agent state action =
  (* Try each level from most specific (largest window) to least specific *)
  let rec try_levels = function
    | [] -> 0.0  (* Default Q-value *)
    | level :: rest ->
      let window_state = extract_window state level.window_size in
      let key = state_to_key window_state ^ "_" ^ string_of_int action in
      
      (* Check if we've seen this state before *)
      match CountTable.find_opt level.count_table (state_to_key window_state) with
      | Some count when count > 0 ->
        (* We've seen this state, use its Q-value *)
        (try QTable.find level.q_table key with Not_found -> 0.0)
      | _ ->
        (* Haven't seen this state at this level, try next level *)
        try_levels rest
  in
  (* Start with the most specific (largest window/full state) *)
  try_levels (List.rev agent.q_levels)

(** Update Q-value at all levels *)
let update_q_values agent state action next_state reward terminated =
  let old_q = get_q_value_backoff agent state action in
  
  (* Get best next Q-value using backoff *)
  let max_next_q = 
    if terminated then 0.0
    else
      let best_q = ref neg_infinity in
      for a = 0 to 3 do  (* 4 actions: up, down, left, right *)
        let q = get_q_value_backoff agent next_state a in
        best_q := max !best_q q
      done;
      !best_q
  in
  
  let new_q = old_q +. agent.params.learning_rate *. 
    (reward +. agent.params.discount_factor *. max_next_q -. old_q) in
  
  (* Update all levels *)
  List.iter (fun level ->
    let window_state = extract_window state level.window_size in
    let window_next_state = extract_window next_state level.window_size in
    let key = state_to_key window_state ^ "_" ^ string_of_int action in
    
    (* Update Q-value *)
    QTable.replace level.q_table key new_q;
    
    (* Update counts *)
    let state_key = state_to_key window_state in
    let count = try CountTable.find level.count_table state_key with Not_found -> 0 in
    CountTable.replace level.count_table state_key (count + 1);
    
    (* Also update count for next state *)
    let next_state_key = state_to_key window_next_state in
    let next_count = try CountTable.find level.count_table next_state_key with Not_found -> 0 in
    CountTable.replace level.count_table next_state_key (next_count + 1)
  ) agent.q_levels

(** Get best action using backoff strategy *)
let get_best_action_backoff agent state =
  let best_action = ref 0 in
  let best_value = ref neg_infinity in
  
  for action = 0 to 3 do
    let q_val = get_q_value_backoff agent state action in
    if q_val > !best_value then begin
      best_value := q_val;
      best_action := action
    end
  done;
  
  !best_action

let epsilon_greedy_action agent state =
  if Random.float 1.0 < agent.epsilon then
    Random.int 4  (* Random action *)
  else
    get_best_action_backoff agent state

let create_agent ?(params=default_params) () = 
  let q_levels = List.map (fun window_size ->
    {
      q_table = QTable.create 10000;
      count_table = CountTable.create 10000;
      window_size;
    }
  ) params.window_sizes in
  {
    q_levels;
    params;
    epsilon = params.epsilon_start;
    steps = 0;
  }

(** Train the backoff-tabular agent on a Fehu environment *)
let train_backoff env ~episodes ~max_steps ~seed ~verbose ?(log_trajectories=false) () =
  Random.init seed;
  let agent = create_agent () in
  
  let rewards_history = ref [] in
  let wins_history = ref [] in
  let total_wins = ref 0 in
  let logged_episodes = ref [] in
  
  Printf.printf "Starting Backoff-Tabular training\n";
  Printf.printf "Window sizes: %s\n" 
    (String.concat ", " (List.map string_of_int agent.params.window_sizes));
  
  for episode = 1 to episodes do
    let obs, info = env.Env.reset () in
    let obs_ref = ref obs in
    let episode_reward = ref 0.0 in
    let episode_steps = ref 0 in
    let is_done = ref false in
    let stage_info = ref info in
    
    (* For logging trajectories *)
    let log_frames = ref [] in
    let should_log = log_trajectories && (episode = 1 || episode mod 500 = 0) in
    
    (* Log initial state *)
    if should_log then begin
      let initial_state = obs_to_state obs in
      let initial_frame = Fehu.Visualization.{
        step = 0;
        state_repr = state_to_key initial_state;
        action = "Initial";
        reward = 0.0;
        value = None;
      } in
      log_frames := initial_frame :: !log_frames
    end;
    
    while not !is_done && !episode_steps < max_steps do
      let state = obs_to_state !obs_ref in
      
      let action = epsilon_greedy_action agent state in
      let action_tensor = Rune.scalar Rune.c Rune.float32 (float_of_int action) in
      
      let next_obs, reward, terminated, truncated, info = env.Env.step action_tensor in
      let next_state = obs_to_state next_obs in
      
      (* Log frame if needed - show next state after action *)
      if should_log then begin
        let frame = Fehu.Visualization.{
          step = !episode_steps + 1;  (* Step number after taking action *)
          state_repr = state_to_key next_state;  (* Show state AFTER action *)
          action = Fehu.Visualization.action_to_string action;
          reward = reward;
          value = None;
        } in
        log_frames := frame :: !log_frames
      end;
      
      (* Update Q-values *)
      update_q_values agent state action next_state reward (terminated || truncated);
      
      obs_ref := next_obs;
      episode_reward := !episode_reward +. reward;
      incr episode_steps;
      
      if terminated || truncated then begin
        is_done := true;
        stage_info := info
      end
    done;
    
    (* Save trajectory if logging *)
    if should_log && List.length !log_frames > 0 then begin
      let won = !episode_reward > 50.0 in
      let stage_str = match List.assoc_opt "stage" !stage_info with
        | Some (`String s) -> Some s
        | _ -> None
      in
      
      let log = Fehu.Visualization.{
        episode_num = episode;
        frames = List.rev !log_frames;
        total_reward = !episode_reward;
        total_steps = !episode_steps;
        won = won;
        stage = stage_str;
      } in
      logged_episodes := log :: !logged_episodes;
      
      (* Save to file *)
      let filename = Printf.sprintf "logs/backoff_episode_%d.json" episode in
      (try Unix.mkdir "logs" 0o755 with Unix.Unix_error _ -> ());
      Fehu.Visualization.save_episode_log log filename
    end;
    
    (* Update epsilon *)
    agent.epsilon <- max agent.params.epsilon_end 
      (agent.epsilon *. agent.params.epsilon_decay);
    agent.steps <- agent.steps + 1;
    
    let won = !episode_reward > 50.0 in
    if won then incr total_wins;
    rewards_history := !episode_reward :: !rewards_history;
    wins_history := (if won then 1.0 else 0.0) :: !wins_history;
    
    (* Keep only recent history *)
    if List.length !rewards_history > 100 then begin
      rewards_history := List.filteri (fun i _ -> i < 100) !rewards_history;
      wins_history := List.filteri (fun i _ -> i < 100) !wins_history
    end;
    
    (* Print progress *)
    if verbose && episode mod 100 = 0 then begin
      let recent_rewards = !rewards_history in
      let avg_reward = 
        (List.fold_left (+.) 0.0 recent_rewards) /. 
        float_of_int (List.length recent_rewards) in
      let recent_wins = !wins_history in
      let win_rate = 
        (List.fold_left (+.) 0.0 recent_wins) /. 
        float_of_int (List.length recent_wins) in
      
      let stage = 
        match List.assoc_opt "stage" !stage_info with
        | Some (`String s) -> s
        | _ -> "N/A"
      in
      
      Printf.printf "Backoff Episode %d: Avg Reward = %.2f, Win Rate = %.1f%%, Epsilon = %.3f, Steps = %d, Stage = %s\n%!"
        episode avg_reward (win_rate *. 100.0) agent.epsilon !episode_steps stage;
      flush stdout
    end
  done;
  
  Printf.printf "Training complete! Final win rate: %.1f%%\n"
    (float_of_int !total_wins /. float_of_int episodes *. 100.0);
  
  agent, !logged_episodes

(** Main entry point to test backoff-tabular as baseline *)
let () =
  (* Test on simple corridor *)
  Printf.printf "\n=== Backoff-Tabular on Simple Corridor ===\n";
  let initial_state = Sokoban.LevelGen.generate_corridor 3 in
  let env = Sokoban.sokoban ~width:5 ~height:3 ~max_steps:50 
    ~initial_state () in
  
  let _agent, _logs = train_backoff env
    ~episodes:1000
    ~max_steps:50
    ~seed:42
    ~verbose:true
    ~log_trajectories:false () in
  
  Printf.printf "\n=== Backoff-Tabular with Curriculum ===\n";
  let curriculum_env = Sokoban.sokoban_curriculum ~max_steps:200 () in
  
  let _agent, _logs = train_backoff curriculum_env
    ~episodes:10000
    ~max_steps:200
    ~seed:43
    ~verbose:true
    ~log_trajectories:true () in
  
  Printf.printf "\nBackoff baseline testing complete!\n"