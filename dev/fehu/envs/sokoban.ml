(** Sokoban environment for RL with curriculum learning support *)

open Fehu

(** Core Sokoban game types *)
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

  let state_to_features state =
    let features = Array.make (state.width * state.height) 0.0 in
    for y = 0 to state.height - 1 do
      for x = 0 to state.width - 1 do
        let idx = y * state.width + x in
        features.(idx) <- match state.grid.(y).(x) with
          | Empty -> 0.0
          | Wall -> 1.0
          | Box -> 2.0
          | Target -> 3.0
          | BoxOnTarget -> 4.0
          | Player -> 5.0
          | PlayerOnTarget -> 6.0
      done
    done;
    features

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

  (** Deadlock detection for unsolvable states *)
  
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

  let is_box_deadlocked (x, y) state =
    let cell = get_cell state (x, y) in
    if not (is_box cell) then false
    else if cell = BoxOnTarget then false  (* Box is already on target *)
    else
      (* For deadlock checking, treat other boxes as walls *)
      let is_wall_or_box (px, py) =
        if px < 0 || px >= state.width || py < 0 || py >= state.height then true
        else 
          let c = state.grid.(py).(px) in
          c = Wall || is_box c
      in
      
      (* Check if box is in a corner (considering other boxes as walls) *)
      let is_in_corner =
        (is_wall_or_box (x-1, y) && is_wall_or_box (x, y-1)) ||
        (is_wall_or_box (x-1, y) && is_wall_or_box (x, y+1)) ||
        (is_wall_or_box (x+1, y) && is_wall_or_box (x, y-1)) ||
        (is_wall_or_box (x+1, y) && is_wall_or_box (x, y+1))
      in
      
      if is_in_corner then
        true  (* Corner without target is always deadlock *)
      else
        (* Check for 2x2 box pattern (4 boxes in a square = deadlock) *)
        let check_2x2_pattern =
          (* Check if this box is part of a 2x2 box formation *)
          let positions = [
            (* Check all 4 positions where current box could be part of 2x2 *)
            [(0,0); (1,0); (0,1); (1,1)];    (* Top-left *)
            [(-1,0); (0,0); (-1,1); (0,1)];  (* Top-right *)
            [(0,-1); (1,-1); (0,0); (1,0)];  (* Bottom-left *)
            [(-1,-1); (0,-1); (-1,0); (0,0)] (* Bottom-right *)
          ] in
          List.exists (fun offsets ->
            List.for_all (fun (dx, dy) ->
              let nx, ny = x + dx, y + dy in
              nx >= 0 && nx < state.width && ny >= 0 && ny < state.height &&
              is_box (get_cell state (nx, ny))
            ) offsets
          ) positions
        in
        
        if check_2x2_pattern then
          true  (* 2x2 box formation is always deadlock *)
        else
          (* Check for deadlock patterns along walls *)
        let is_wall_at (dx, dy) = 
          get_cell state (x + dx, y + dy) = Wall 
        in
        (* Box against wall - check if it can reach a target on the SAME wall segment *)
        (* A wall segment continues until hitting a corner or wall break *)
        
        (* Check which wall the box is against *)
        let against_top = is_wall_at (0, -1) in
        let against_bottom = is_wall_at (0, 1) in
        let against_left = is_wall_at (-1, 0) in
        let against_right = is_wall_at (1, 0) in
        
        if against_top || against_bottom then
          (* Box against horizontal wall - check same horizontal wall segment *)
          let wall_y = if against_top then y - 1 else y + 1 in
          let rec check_horizontal_segment pos dir =
            let nx = pos + dir in
            if nx < 0 || nx >= state.width then true  (* Hit boundary *)
            else
              (* Check if we're still along the same wall *)
              let wall_continues = get_cell state (nx, wall_y) = Wall in
              let cell_at_pos = get_cell state (nx, y) in
              
              if not wall_continues then true  (* Wall segment ended *)
              else if cell_at_pos = Wall then true  (* Hit perpendicular wall *)
              else if cell_at_pos = Target || cell_at_pos = BoxOnTarget || cell_at_pos = PlayerOnTarget then false  (* Found target! *)
              else if is_box cell_at_pos then 
                (* Found another box adjacent along the wall *)
                (* Two boxes side by side against a wall are BOTH deadlocked *)
                (* unless there's a target that one of them can reach *)
                true  (* Two boxes adjacent along wall = deadlock *)
              else check_horizontal_segment nx dir
          in
          check_horizontal_segment x 1 && check_horizontal_segment x (-1)
          
        else if against_left || against_right then
          (* Box against vertical wall - check same vertical wall segment *)
          let wall_x = if against_left then x - 1 else x + 1 in
          let rec check_vertical_segment pos dir =
            let ny = pos + dir in
            if ny < 0 || ny >= state.height then true  (* Hit boundary *)
            else
              (* Check if we're still along the same wall *)
              let wall_continues = get_cell state (wall_x, ny) = Wall in
              let cell_at_pos = get_cell state (x, ny) in
              
              if not wall_continues then true  (* Wall segment ended *)
              else if cell_at_pos = Wall then true  (* Hit perpendicular wall *)
              else if cell_at_pos = Target || cell_at_pos = BoxOnTarget || cell_at_pos = PlayerOnTarget then false  (* Found target! *)
              else if is_box cell_at_pos then
                (* Found another box adjacent along the wall *)
                (* Two boxes side by side against a wall are BOTH deadlocked *)
                (* unless there's a target that one of them can reach *)
                true  (* Two boxes adjacent along wall = deadlock *)
              else check_vertical_segment ny dir
          in
          check_vertical_segment y 1 && check_vertical_segment y (-1)
        else
          false  (* Not against a wall *)

  let has_deadlock state =
    try
      for y = 0 to state.height - 1 do
        for x = 0 to state.width - 1 do
          if is_box_deadlocked (x, y) state then
            raise Exit
        done
      done;
      false
    with Exit -> true

  let is_potentially_solvable state =
    (* Basic checks *)
    let num_boxes = count_boxes state in
    let num_targets = count_targets state in
    num_boxes = num_targets && num_boxes > 0 && not (has_deadlock state)
end

(** Level generation module *)
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

  let generate_corridor length =
    (* Randomly choose horizontal or vertical orientation *)
    let horizontal = Random.bool () in
    
    if horizontal then
      (* Horizontal corridor *)
      let width = length + 2 in
      let height = 3 in
      let state = make_empty width height in
      add_walls state;
      
      (* Randomly choose push direction *)
      if Random.bool () then
        (* Push right *)
        begin
          state.grid.(1).(1) <- Player;
          (* Randomly place box between player and target *)
          let box_pos = 2 + Random.int (max 1 (length - 2)) in
          state.grid.(1).(box_pos) <- Box;
          state.grid.(1).(length) <- Target;
          { state with player_pos = (1, 1) }
        end
      else
        (* Push left *)
        begin
          state.grid.(1).(length) <- Player;
          (* Randomly place box between player and target *)
          let box_pos = 2 + Random.int (max 1 (length - 2)) in
          state.grid.(1).(box_pos) <- Box;
          state.grid.(1).(1) <- Target;
          { state with player_pos = (length, 1) }
        end
    else
      (* Vertical corridor *)
      let width = 3 in
      let height = length + 2 in
      let state = make_empty width height in
      add_walls state;
      
      (* Randomly choose push direction *)
      if Random.bool () then
        (* Push down *)
        begin
          state.grid.(1).(1) <- Player;
          (* Randomly place box between player and target *)
          let box_pos = 2 + Random.int (max 1 (length - 2)) in
          state.grid.(box_pos).(1) <- Box;
          state.grid.(length).(1) <- Target;
          { state with player_pos = (1, 1) }
        end
      else
        (* Push up *)
        begin
          state.grid.(length).(1) <- Player;
          (* Randomly place box between player and target *)
          let box_pos = 2 + Random.int (max 1 (length - 2)) in
          state.grid.(box_pos).(1) <- Box;
          state.grid.(1).(1) <- Target;
          { state with player_pos = (1, length) }
        end

  let generate_room size =
    let max_attempts = 50 in
    
    let rec attempt n =
      if n >= max_attempts then
        (* Fall back to simple line configuration *)
        let state = make_empty size size in
        add_walls state;
        let center = size / 2 in
        state.grid.(center).(center - 1) <- Player;
        state.grid.(center).(center) <- Box;
        state.grid.(center).(center + 1) <- Target;
        { state with player_pos = (center - 1, center) }
      else
        let state = make_empty size size in
        add_walls state;
        
        (* Get all inner positions *)
        let all_positions = ref [] in
        for x = 1 to size - 2 do
          for y = 1 to size - 2 do
            all_positions := (x, y) :: !all_positions
          done
        done;
        
        (* Shuffle positions *)
        let shuffle lst =
          let arr = Array.of_list lst in
          for i = Array.length arr - 1 downto 1 do
            let j = Random.int (i + 1) in
            let tmp = arr.(i) in
            arr.(i) <- arr.(j);
            arr.(j) <- tmp
          done;
          Array.to_list arr
        in
        
        let positions = shuffle !all_positions in
        
        (* Try to place player, box, and target *)
        match positions with
        | [] -> attempt (n + 1)
        | p1 :: rest_positions ->
          (* Place player first *)
          let (px, py) = p1 in
          state.grid.(py).(px) <- Player;
          
          (* Try different combinations for box and target *)
          let try_placement (bx, by) (tx, ty) =
            (* Don't place box and target in same position *)
            if (bx, by) = (tx, ty) then false
            else begin
              (* Temporarily place box *)
              state.grid.(by).(bx) <- Box;
              
              (* Check corner placement rules for small rooms *)
              let corner_ok = 
                if size <= 5 then
                  (* In small rooms (3x3 playable), box in corner is OK only if target is also in a corner *)
                  if Core.is_corner (bx, by) state then
                    Core.is_corner (tx, ty) state && (bx, by) <> (tx, ty)
                  else
                    true
                else
                  (* In larger rooms, avoid corners for boxes *)
                  not (Core.is_corner (bx, by) state)
              in
              
              if not corner_ok then begin
                state.grid.(by).(bx) <- Empty;
                false
              end else begin
                (* Place target *)
                state.grid.(ty).(tx) <- Target;
                
                (* Check if configuration is potentially solvable *)
                let test_state = { state with player_pos = (px, py) } in
                if Core.is_potentially_solvable test_state then
                  true
                else begin
                  (* Revert changes *)
                  state.grid.(by).(bx) <- Empty;
                  state.grid.(ty).(tx) <- Empty;
                  false
                end
              end
            end
          in
          
          (* Try multiple placement combinations *)
          let rec try_combinations = function
            | [] -> None
            | (b, t) :: rest ->
              if try_placement b t then
                Some { state with player_pos = (px, py) }
              else begin
                (* Clean up any leftover state *)
                let (bx, by) = b in
                let (tx, ty) = t in
                state.grid.(by).(bx) <- Empty;
                state.grid.(ty).(tx) <- Empty;
                try_combinations rest
              end
          in
          
          (* Generate different position combinations from remaining positions *)
          let combinations = match rest_positions with
            | p2 :: p3 :: p4 :: _ -> 
              [(p2, p3); (p2, p4); (p3, p2); (p3, p4); (p4, p2); (p4, p3)]
            | p2 :: p3 :: [] ->
              [(p2, p3); (p3, p2)]
            | _ -> []
          in
          
          match try_combinations combinations with
          | Some final_state -> final_state
          | None -> 
            (* Clean up player and try again *)
            state.grid.(py).(px) <- Empty;
            attempt (n + 1)
    in
    attempt 0

  let generate_multi_box num_boxes =
    let max_attempts = 100 in
    
    let rec attempt n =
      if n >= max_attempts then
        (* Fall back to simple configuration *)
        let size = max 6 (num_boxes + 4) in
        let state = make_empty size size in
        add_walls state;
        (* Place player at center *)
        let center = size / 2 in
        state.grid.(center).(center) <- Player;
        (* Place boxes and targets in a line *)
        for i = 0 to num_boxes - 1 do
          state.grid.(center).(center - i - 1) <- Target;
          state.grid.(center).(center + i + 1) <- Box
        done;
        { state with player_pos = (center, center) }
      else
        let size = max 6 (num_boxes + 4) in
        let state = make_empty size size in
        add_walls state;
        
        (* Get all valid positions *)
        let positions = ref [] in
        for x = 1 to size - 2 do
          for y = 1 to size - 2 do
            positions := (x, y) :: !positions
          done
        done;
        
        (* Shuffle positions *)
        let shuffle lst =
          let arr = Array.of_list lst in
          for i = Array.length arr - 1 downto 1 do
            let j = Random.int (i + 1) in
            let tmp = arr.(i) in
            arr.(i) <- arr.(j);
            arr.(j) <- tmp
          done;
          Array.to_list arr
        in
        let shuffled = shuffle !positions in
        
        (* Place player first *)
        match shuffled with
        | [] -> attempt (n + 1)
        | (px, py) :: rest ->
          state.grid.(py).(px) <- Player;
          
          (* Try to place boxes and targets, checking for deadlocks after each box *)
          let rec place_items positions_left boxes_to_place targets_to_place =
            if boxes_to_place = 0 && targets_to_place = 0 then
              (* Success! *)
              Some { state with player_pos = (px, py) }
            else if positions_left = [] then
              None  (* Not enough positions *)
            else if boxes_to_place > 0 then
              (* Try to place a box *)
              let rec try_box_positions = function
                | [] -> None
                | (x, y) :: rest_pos ->
                  state.grid.(y).(x) <- Box;
                  (* Check if this creates a deadlock *)
                  if Core.is_box_deadlocked (x, y) state then begin
                    (* This box is deadlocked, try next position *)
                    state.grid.(y).(x) <- Empty;
                    try_box_positions rest_pos
                  end else begin
                    (* Box is OK, continue placing *)
                    match place_items rest_pos (boxes_to_place - 1) targets_to_place with
                    | Some result -> Some result
                    | None ->
                      (* Backtrack *)
                      state.grid.(y).(x) <- Empty;
                      try_box_positions rest_pos
                  end
              in
              try_box_positions positions_left
            else
              (* Place remaining targets *)
              match positions_left with
              | (x, y) :: rest ->
                state.grid.(y).(x) <- Target;
                place_items rest 0 (targets_to_place - 1)
              | [] -> None
          in
          
          match place_items rest num_boxes num_boxes with
          | Some result -> result
          | None ->
            (* Clean up and try again *)
            state.grid.(py).(px) <- Empty;
            attempt (n + 1)
    in
    attempt 0

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

(** Curriculum learning module *)
module Curriculum = struct
  type stage = 
    | Corridor of int
    | Room of int  
    | MultiBox of int
    | Complex

  type config = {
    stages: stage list;
    current_idx: int ref;
    success_threshold: float;
    window_size: int;
    recent_rewards: float list ref;
  }

  let create_config ?(success_threshold=0.8) ?(window_size=100) stages = {
    stages;
    current_idx = ref 0;
    success_threshold;
    window_size;
    recent_rewards = ref [];
  }

  let generate_level = function
    | Corridor len -> LevelGen.generate_corridor len
    | Room size -> LevelGen.generate_room size
    | MultiBox n -> LevelGen.generate_multi_box n
    | Complex -> LevelGen.generate_complex ()

  let update_and_check_advance config _reward won =
    (* Track win (1.0) or loss (0.0) for success rate calculation *)
    let outcome = if won then 1.0 else 0.0 in
    config.recent_rewards := outcome :: !(config.recent_rewards);
    
    if List.length !(config.recent_rewards) > config.window_size then
      config.recent_rewards := 
        List.filteri (fun i _ -> i < config.window_size) !(config.recent_rewards);
    
    if List.length !(config.recent_rewards) >= config.window_size then
      let wins = List.filter (fun r -> r > 0.5) !(config.recent_rewards) in
      let success_rate = float_of_int (List.length wins) /. float_of_int config.window_size in
      
      if success_rate >= config.success_threshold && 
         !(config.current_idx) < List.length config.stages - 1 then begin
        incr config.current_idx;
        config.recent_rewards := [];
        true
      end else
        false
    else
      false

  let get_current_stage config =
    List.nth config.stages !(config.current_idx)

  let reset_curriculum config =
    config.current_idx := 0;
    config.recent_rewards := []
end

(** Create basic Sokoban environment *)
let sokoban ?(width=7) ?(height=7) ?(max_steps=200) ?initial_state () =
  let state = ref (match initial_state with
    | Some s -> s
    | None -> LevelGen.generate_room 5) in
  
  let steps = ref 0 in
  
  let observation_space = 
    Space.Box {
      low = Rune.zeros Rune.c Rune.float32 [| height; width |];
      high = Rune.full Rune.c Rune.float32 [| height; width |] 6.0;
      shape = [| height; width |];
    }
  in
  
  let action_space = Space.Discrete 4 in
  
  let reset ?seed () =
    let () = match seed with Some s -> Random.init s | None -> () in
    steps := 0;
    state := (match initial_state with
      | Some s -> Core.copy_state s
      | None -> LevelGen.generate_room 5);
    
    (* Create 2D observation array using actual state dimensions *)
    let actual_height = !state.height in
    let actual_width = !state.width in
    let features = Array.make (actual_height * actual_width) 0.0 in
    for y = 0 to actual_height - 1 do
      for x = 0 to actual_width - 1 do
        let idx = y * actual_width + x in
        features.(idx) <- match !state.grid.(y).(x) with
          | Core.Empty -> 0.0
          | Core.Wall -> 1.0
          | Core.Box -> 2.0
          | Core.Target -> 3.0
          | Core.BoxOnTarget -> 4.0
          | Core.Player -> 5.0
          | Core.PlayerOnTarget -> 6.0
      done
    done;
    let obs = Rune.create Rune.c Rune.float32 [| actual_height; actual_width |] features in
    
    (obs, [])
  in
  
  let step action =
    incr steps;
    let action_val = int_of_float (Rune.unsafe_to_array action).(0) in
    let direction = match action_val with
      | 0 -> Core.Up
      | 1 -> Core.Down
      | 2 -> Core.Left
      | 3 -> Core.Right
      | _ -> Core.Up
    in
    
    let old_state = !state in
    state := Core.apply_action !state direction;
    
    let moved = !state != old_state in
    let won = Core.check_win !state in
    let truncated = !steps >= max_steps in
    
    let reward = 
      if won then 100.0
      else if not moved then -0.1
      else -0.01
    in
    
    (* Create 2D observation array using actual state dimensions *)
    let actual_height = !state.height in
    let actual_width = !state.width in
    let features = Array.make (actual_height * actual_width) 0.0 in
    for y = 0 to actual_height - 1 do
      for x = 0 to actual_width - 1 do
        let idx = y * actual_width + x in
        features.(idx) <- match !state.grid.(y).(x) with
          | Core.Empty -> 0.0
          | Core.Wall -> 1.0
          | Core.Box -> 2.0
          | Core.Target -> 3.0
          | Core.BoxOnTarget -> 4.0
          | Core.Player -> 5.0
          | Core.PlayerOnTarget -> 6.0
      done
    done;
    let obs = Rune.create Rune.c Rune.float32 [| actual_height; actual_width |] features in
    
    (obs, reward, won, truncated, [("steps", `Int !steps)])
  in
  
  let render () = Core.render !state in
  
  Env.make ~observation_space ~action_space ~reset ~step ~render ()

(** Create Sokoban environment with curriculum learning *)
let sokoban_curriculum ?(max_steps=200) ?stages () =
  let default_stages = [
    Curriculum.Corridor 3;
    Curriculum.Corridor 5;
    Curriculum.Room 5;
    Curriculum.Room 7;
    Curriculum.MultiBox 2;
    Curriculum.MultiBox 3;
    Curriculum.Complex;
  ] in
  
  let stages = match stages with Some s -> s | None -> default_stages in
  let config = Curriculum.create_config stages in
  
  let current_level = ref (Curriculum.generate_level (Curriculum.get_current_stage config)) in
  let steps = ref 0 in
  
  let max_size = 10 in
  
  let observation_space = 
    Space.Box {
      low = Rune.zeros Rune.c Rune.float32 [| max_size; max_size |];
      high = Rune.full Rune.c Rune.float32 [| max_size; max_size |] 6.0;
      shape = [| max_size; max_size |];
    }
  in
  
  let action_space = Space.Discrete 4 in
  
  let reset ?seed () =
    let () = match seed with Some s -> Random.init s | None -> () in
    steps := 0;
    current_level := Curriculum.generate_level (Curriculum.get_current_stage config);
    
    (* Create 2D observation array *)
    let features = Array.make (max_size * max_size) 0.0 in
    for y = 0 to !current_level.height - 1 do
      for x = 0 to !current_level.width - 1 do
        let idx = y * max_size + x in
        features.(idx) <- match !current_level.grid.(y).(x) with
          | Core.Empty -> 0.0
          | Core.Wall -> 1.0
          | Core.Box -> 2.0
          | Core.Target -> 3.0
          | Core.BoxOnTarget -> 4.0
          | Core.Player -> 5.0
          | Core.PlayerOnTarget -> 6.0
      done
    done;
    let obs = Rune.create Rune.c Rune.float32 [| max_size; max_size |] features in
    
    (obs, 
     [("stage", `String (Printf.sprintf "%d/%d" (!(config.current_idx) + 1) (List.length config.stages)))])
  in
  
  let step action =
    incr steps;
    let action_val = int_of_float (Rune.unsafe_to_array action).(0) in
    let direction = match action_val with
      | 0 -> Core.Up
      | 1 -> Core.Down
      | 2 -> Core.Left
      | 3 -> Core.Right
      | _ -> Core.Up
    in
    
    let old_level = !current_level in
    current_level := Core.apply_action !current_level direction;
    
    let moved = !current_level != old_level in
    let won = Core.check_win !current_level in
    let truncated = !steps >= max_steps in
    
    let reward = 
      if won then 100.0
      else if not moved then -0.1
      else -0.01
    in
    
    (* Only check advancement when episode ends (won or truncated) *)
    let advanced = 
      if won || truncated then
        Curriculum.update_and_check_advance config reward won
      else false in
    
    (* Create 2D observation array *)
    let features = Array.make (max_size * max_size) 0.0 in
    for y = 0 to !current_level.height - 1 do
      for x = 0 to !current_level.width - 1 do
        let idx = y * max_size + x in  (* Index into flat array backing the 2D tensor *)
        features.(idx) <- match !current_level.grid.(y).(x) with
          | Core.Empty -> 0.0
          | Core.Wall -> 1.0
          | Core.Box -> 2.0
          | Core.Target -> 3.0
          | Core.BoxOnTarget -> 4.0
          | Core.Player -> 5.0
          | Core.PlayerOnTarget -> 6.0
      done
    done;
    let obs = Rune.create Rune.c Rune.float32 [| max_size; max_size |] features in
    
    let info = [
      ("steps", `Int !steps);
      ("stage", `String (Printf.sprintf "%d/%d" (!(config.current_idx) + 1) (List.length config.stages)));
      ("advanced", `Bool advanced)
    ] in
    
    (obs, reward, won, truncated, info)
  in
  
  let render () = Core.render !current_level in
  
  Env.make ~observation_space ~action_space ~reset ~step ~render ()