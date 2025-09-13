(*
Advanced Sokoban curriculum for RL workshop
Based on dev/fehu/envs/sokoban.ml but adapted for workshop use
*)

open Fehu
let device = Rune.c

(* Core Sokoban game types and logic *)
module Core = struct
  type cell =
    | Empty
    | Wall
    | Box
    | Target
    | BoxOnTarget
    | Player
    | PlayerOnTarget

  type position = int * int

  type direction = Up | Down | Left | Right

  type game_state = {
    grid: cell array array;
    player_pos: position;
    width: int;
    height: int;
  }

  let copy_state state = {
    grid = Array.map Array.copy state.grid;
    player_pos = state.player_pos;
    width = state.width;
    height = state.height;
  }

  let get_cell state (x, y) =
    if x >= 0 && x < state.width && y >= 0 && y < state.height then
      state.grid.(y).(x)
    else
      Wall

  let set_cell state (x, y) cell =
    if x >= 0 && x < state.width && y >= 0 && y < state.height then
      state.grid.(y).(x) <- cell

  let move_position (x, y) = function
    | Up -> (x, y - 1)
    | Down -> (x, y + 1)
    | Left -> (x - 1, y)
    | Right -> (x + 1, y)

  let is_box = function
    | Box | BoxOnTarget -> true
    | _ -> false

  let is_empty = function
    | Empty | Target -> true
    | _ -> false

  let cell_without_player = function
    | Player -> Empty
    | PlayerOnTarget -> Target
    | c -> c

  let cell_with_player = function
    | Empty -> Player
    | Target -> PlayerOnTarget
    | c -> c

  let cell_without_box = function
    | Box -> Empty
    | BoxOnTarget -> Target
    | c -> c

  let cell_with_box = function
    | Empty -> Box
    | Target -> BoxOnTarget
    | c -> c

  let check_win state =
    let has_box = ref false in
    let all_on_target = ref true in
    for y = 0 to state.height - 1 do
      for x = 0 to state.width - 1 do
        match state.grid.(y).(x) with
        | Box -> has_box := true; all_on_target := false
        | BoxOnTarget -> has_box := true
        | _ -> ()
      done
    done;
    !has_box && !all_on_target

  let apply_action state direction =
    let new_state = copy_state state in
    let new_pos = move_position state.player_pos direction in
    let next_cell = get_cell state new_pos in

    match next_cell with
    | Wall -> state
    | Empty | Target ->
      set_cell new_state state.player_pos (cell_without_player (get_cell state state.player_pos));
      set_cell new_state new_pos (cell_with_player next_cell);
      { new_state with player_pos = new_pos }
    | Box | BoxOnTarget ->
      let box_new_pos = move_position new_pos direction in
      let box_next_cell = get_cell state box_new_pos in
      if is_empty box_next_cell then
        begin
          set_cell new_state state.player_pos (cell_without_player (get_cell state state.player_pos));
          set_cell new_state new_pos (cell_with_player (cell_without_box next_cell));
          set_cell new_state box_new_pos (cell_with_box box_next_cell);
          { new_state with player_pos = new_pos }
        end
      else
        state
    | _ -> state

  let render state =
    for y = 0 to state.height - 1 do
      for x = 0 to state.width - 1 do
        let c = match state.grid.(y).(x) with
          | Empty -> ' '
          | Wall -> '#'
          | Box -> '$'
          | Target -> '.'
          | BoxOnTarget -> '*'
          | Player -> '@'
          | PlayerOnTarget -> '+'
        in
        print_char c
      done;
      print_newline ()
    done;
    print_newline ()

  (* Simple deadlock detection *)
  let is_corner pos state =
    let (x, y) = pos in
    let is_wall_or_boundary (px, py) =
      px < 0 || px >= state.width || py < 0 || py >= state.height ||
      state.grid.(py).(px) = Wall
    in
    (* Check if position is in a corner (two adjacent walls) *)
    (is_wall_or_boundary (x-1, y) && is_wall_or_boundary (x, y-1)) ||
    (is_wall_or_boundary (x-1, y) && is_wall_or_boundary (x, y+1)) ||
    (is_wall_or_boundary (x+1, y) && is_wall_or_boundary (x, y-1)) ||
    (is_wall_or_boundary (x+1, y) && is_wall_or_boundary (x, y+1))

  let count_targets state =
    let count = ref 0 in
    for y = 0 to state.height - 1 do
      for x = 0 to state.width - 1 do
        match state.grid.(y).(x) with
        | Target | BoxOnTarget | PlayerOnTarget -> incr count
        | _ -> ()
      done
    done;
    !count

  let count_boxes state =
    let count = ref 0 in
    for y = 0 to state.height - 1 do
      for x = 0 to state.width - 1 do
        match state.grid.(y).(x) with
        | Box | BoxOnTarget -> incr count
        | _ -> ()
      done
    done;
    !count

  let is_potentially_solvable state =
    (* Basic check: equal boxes and targets *)
    let num_boxes = count_boxes state in
    let num_targets = count_targets state in
    num_boxes = num_targets && num_boxes > 0
end

(* Level generation module *)
module LevelGen = struct
  open Core

  let make_empty width height = {
    grid = Array.make_matrix height width Empty;
    player_pos = (0, 0);
    width;
    height;
  }

  let add_walls state =
    for x = 0 to state.width - 1 do
      state.grid.(0).(x) <- Wall;
      state.grid.(state.height - 1).(x) <- Wall
    done;
    for y = 0 to state.height - 1 do
      state.grid.(y).(0) <- Wall;
      state.grid.(y).(state.width - 1) <- Wall
    done

  (* Simple corridor level *)
  let generate_corridor length =
    let width = length + 2 in
    let height = 3 in
    let state = make_empty width height in
    add_walls state;

    (* Simple configuration: player, box, target in a line *)
    state.grid.(1).(1) <- Player;
    state.grid.(1).(2) <- Box;
    state.grid.(1).(length) <- Target;
    { state with player_pos = (1, 1) }

  (* Room with single box and target *)
  let generate_room size =
    let state = make_empty size size in
    add_walls state;

    (* Place player in corner *)
    state.grid.(1).(1) <- Player;

    (* Place box in center *)
    let center = size / 2 in
    state.grid.(center).(center) <- Box;

    (* Place target in opposite corner *)
    state.grid.(size - 2).(size - 2) <- Target;

    { state with player_pos = (1, 1) }

  (* Room with multiple boxes *)
  let generate_multi_box num_boxes =
    let size = max 5 (num_boxes + 3) in
    let state = make_empty size size in
    add_walls state;

    (* Place player *)
    state.grid.(1).(1) <- Player;

    (* Place boxes and targets symmetrically *)
    for i = 0 to num_boxes - 1 do
      let offset = i + 2 in
      if offset < size - 1 then begin
        state.grid.(offset).(2) <- Box;
        state.grid.(offset).(size - 2) <- Target
      end
    done;

    { state with player_pos = (1, 1) }

  (* Fixed complex level *)
  let generate_complex () =
    let layout = [|
      [| Wall; Wall; Wall; Wall; Wall; Wall; Wall |];
      [| Wall; Empty; Empty; Target; Empty; Empty; Wall |];
      [| Wall; Empty; Box; Empty; Box; Empty; Wall |];
      [| Wall; Target; Empty; Player; Empty; Target; Wall |];
      [| Wall; Empty; Box; Empty; Box; Empty; Wall |];
      [| Wall; Empty; Empty; Target; Empty; Empty; Wall |];
      [| Wall; Wall; Wall; Wall; Wall; Wall; Wall |];
    |] in

    { grid = layout;
      player_pos = (3, 3);
      width = 7;
      height = 7 }
end

(* Curriculum learning module *)
module Curriculum = struct
  type stage =
    | Corridor of int     (* Length of corridor *)
    | Room of int        (* Size of room *)
    | MultiBox of int    (* Number of boxes *)
    | Complex           (* Fixed complex level *)

  type config = {
    stages: stage list;
    current_idx: int ref;
    success_threshold: float;
    window_size: int;
    recent_wins: bool list ref;
    total_episodes: int ref;
  }

  let create_config ?(success_threshold=0.7) ?(window_size=50) stages = {
    stages;
    current_idx = ref 0;
    success_threshold;
    window_size;
    recent_wins = ref [];
    total_episodes = ref 0;
  }

  let generate_level = function
    | Corridor len -> LevelGen.generate_corridor len
    | Room size -> LevelGen.generate_room size
    | MultiBox n -> LevelGen.generate_multi_box n
    | Complex -> LevelGen.generate_complex ()

  let update_and_check_advance config won =
    incr config.total_episodes;
    config.recent_wins := won :: !(config.recent_wins);

    (* Keep only window_size most recent *)
    if List.length !(config.recent_wins) > config.window_size then
      config.recent_wins :=
        List.filteri (fun i _ -> i < config.window_size) !(config.recent_wins);

    (* Check if ready to advance *)
    if List.length !(config.recent_wins) >= config.window_size then
      let wins = List.filter (fun x -> x) !(config.recent_wins) in
      let success_rate = float_of_int (List.length wins) /. float_of_int config.window_size in

      if success_rate >= config.success_threshold &&
         !(config.current_idx) < List.length config.stages - 1 then begin
        incr config.current_idx;
        config.recent_wins := [];
        Printf.printf "Advancing to stage %d after %d episodes (success rate: %.2f)\n"
          !(config.current_idx) !(config.total_episodes) success_rate;
        true
      end else
        false
    else
      false

  let get_current_stage config =
    List.nth config.stages !(config.current_idx)

  let get_stage_name = function
    | Corridor n -> Printf.sprintf "Corridor(%d)" n
    | Room n -> Printf.sprintf "Room(%d)" n
    | MultiBox n -> Printf.sprintf "MultiBox(%d)" n
    | Complex -> "Complex"

  let reset_curriculum config =
    config.current_idx := 0;
    config.recent_wins := [];
    config.total_episodes := 0
end

(* Create Sokoban environment with curriculum for workshop *)
let create_sokoban_curriculum ?(max_steps=100) ?curriculum_config () =
  (* Default curriculum stages *)
  let default_stages = [
    Curriculum.Corridor 3;   (* Very simple: push box down corridor *)
    Curriculum.Corridor 5;   (* Longer corridor *)
    Curriculum.Room 5;       (* Small room *)
    Curriculum.Room 7;       (* Larger room *)
    Curriculum.MultiBox 2;   (* Two boxes *)
    Curriculum.Complex;      (* Complex multi-box puzzle *)
  ] in

  let config = match curriculum_config with
    | Some c -> c
    | None -> Curriculum.create_config default_stages
  in

  let current_state = ref (Curriculum.generate_level (Curriculum.get_current_stage config)) in
  let steps = ref 0 in

  (* Max grid size for observation space *)
  let max_size = 9 in

  let observation_space =
    Space.Box {
      low = Rune.zeros device Rune.float32 [| max_size; max_size |];
      high = Rune.full device Rune.float32 [| max_size; max_size |] 6.0;
      shape = [| max_size; max_size |];
    }
  in

  let action_space = Space.Discrete 4 in

  let reset ?seed () =
    let () = match seed with Some s -> Random.init s | None -> () in
    steps := 0;

    (* Generate new level for current stage *)
    current_state := Curriculum.generate_level (Curriculum.get_current_stage config);

    (* Create observation - pad to max_size if needed *)
    let actual_height = !current_state.height in
    let actual_width = !current_state.width in

    let features = Array.make (max_size * max_size) 0.0 in
    for y = 0 to min (actual_height - 1) (max_size - 1) do
      for x = 0 to min (actual_width - 1) (max_size - 1) do
        let idx = y * max_size + x in
        if y < actual_height && x < actual_width then
          features.(idx) <- match !current_state.grid.(y).(x) with
            | Core.Empty -> 0.0
            | Core.Wall -> 1.0
            | Core.Box -> 2.0
            | Core.Target -> 3.0
            | Core.BoxOnTarget -> 4.0
            | Core.Player -> 5.0
            | Core.PlayerOnTarget -> 6.0
        else
          features.(idx) <- 1.0  (* Pad with walls *)
      done
    done;

    let obs = Rune.create device Rune.float32 [| max_size; max_size |] features in
    (obs, [])
  in

  let step action =
    incr steps;
    let action_val = int_of_float (Rune.item [] action) in
    let direction = match action_val with
      | 0 -> Core.Up
      | 1 -> Core.Down
      | 2 -> Core.Left
      | 3 -> Core.Right
      | _ -> Core.Up
    in

    let old_state = !current_state in
    current_state := Core.apply_action !current_state direction;

    let moved = !current_state.player_pos <> old_state.player_pos in
    let won = Core.check_win !current_state in
    let truncated = !steps >= max_steps in
    let terminated = won in

    (* Reward shaping *)
    let reward =
      if won then 10.0           (* Win bonus *)
      else if not moved then -0.1 (* Penalty for invalid move *)
      else -0.01                  (* Small step penalty *)
    in

    (* Update curriculum on episode end *)
    if terminated || truncated then (
      let _ = Curriculum.update_and_check_advance config won in
      ()
    );

    (* Create observation - pad to max_size if needed *)
    let actual_height = !current_state.height in
    let actual_width = !current_state.width in

    let features = Array.make (max_size * max_size) 0.0 in
    for y = 0 to min (actual_height - 1) (max_size - 1) do
      for x = 0 to min (actual_width - 1) (max_size - 1) do
        let idx = y * max_size + x in
        if y < actual_height && x < actual_width then
          features.(idx) <- match !current_state.grid.(y).(x) with
            | Core.Empty -> 0.0
            | Core.Wall -> 1.0
            | Core.Box -> 2.0
            | Core.Target -> 3.0
            | Core.BoxOnTarget -> 4.0
            | Core.Player -> 5.0
            | Core.PlayerOnTarget -> 6.0
        else
          features.(idx) <- 1.0  (* Pad with walls *)
      done
    done;

    let obs = Rune.create device Rune.float32 [| max_size; max_size |] features in

    (obs, reward, terminated, truncated,
     [("steps", `Int !steps);
      ("stage", `String (Curriculum.get_stage_name (Curriculum.get_current_stage config)))])
  in

  let render () = Core.render !current_state in

  Env.make ~observation_space ~action_space ~reset ~step ~render ()

(* Main test function *)
let main () =
  print_endline "=== Advanced Sokoban Curriculum ===";

  let env = create_sokoban_curriculum () in
  let obs, _ = env.reset () in

  print_endline "Initial state:";
  env.render ();

  print_endline "Observation shape:";
  let shape = Rune.shape obs in
  Printf.printf "[%d, %d]\n" shape.(0) shape.(1)