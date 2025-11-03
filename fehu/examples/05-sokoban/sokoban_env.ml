open Fehu

(** Core Sokoban game types *)
module Core = struct
  type cell =
    | Empty
    | Wall
    | Box
    | Target
    | Box_on_target
    | Player
    | Player_on_target

  type position = int * int
  type direction = Up | Down | Left | Right

  type game_state = {
    grid : cell array array;
    player_pos : position;
    width : int;
    height : int;
  }

  let copy_state state =
    {
      grid = Array.map Array.copy state.grid;
      player_pos = state.player_pos;
      width = state.width;
      height = state.height;
    }

  let get_cell state (x, y) =
    if x >= 0 && x < state.width && y >= 0 && y < state.height then
      state.grid.(y).(x)
    else Wall

  let set_cell state (x, y) cell =
    if x >= 0 && x < state.width && y >= 0 && y < state.height then
      state.grid.(y).(x) <- cell

  let move_position (x, y) = function
    | Up -> (x, y - 1)
    | Down -> (x, y + 1)
    | Left -> (x - 1, y)
    | Right -> (x + 1, y)

  let is_box = function Box | Box_on_target -> true | _ -> false
  let is_empty = function Empty | Target -> true | _ -> false

  let cell_without_player = function
    | Player -> Empty
    | Player_on_target -> Target
    | c -> c

  let cell_with_player = function
    | Empty -> Player
    | Target -> Player_on_target
    | c -> c

  let cell_without_box = function
    | Box -> Empty
    | Box_on_target -> Target
    | c -> c

  let cell_with_box = function Empty -> Box | Target -> Box_on_target | c -> c

  let check_win state =
    let has_box = ref false in
    let all_on_target = ref true in
    for y = 0 to state.height - 1 do
      for x = 0 to state.width - 1 do
        match state.grid.(y).(x) with
        | Box ->
            has_box := true;
            all_on_target := false
        | Box_on_target -> has_box := true
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
        set_cell new_state state.player_pos
          (cell_without_player (get_cell state state.player_pos));
        set_cell new_state new_pos (cell_with_player next_cell);
        { new_state with player_pos = new_pos }
    | Box | Box_on_target ->
        let box_new_pos = move_position new_pos direction in
        let box_next_cell = get_cell state box_new_pos in
        if is_empty box_next_cell then (
          set_cell new_state state.player_pos
            (cell_without_player (get_cell state state.player_pos));
          set_cell new_state new_pos
            (cell_with_player (cell_without_box next_cell));
          set_cell new_state box_new_pos (cell_with_box box_next_cell);
          { new_state with player_pos = new_pos })
        else state
    | _ -> state

  let state_to_features state =
    let features = Array.make (state.width * state.height) 0.0 in
    for y = 0 to state.height - 1 do
      for x = 0 to state.width - 1 do
        let idx = (y * state.width) + x in
        features.(idx) <-
          (match state.grid.(y).(x) with
          | Empty -> 0.0
          | Wall -> 1.0
          | Box -> 2.0
          | Target -> 3.0
          | Box_on_target -> 4.0
          | Player -> 5.0
          | Player_on_target -> 6.0)
      done
    done;
    features

  let render state =
    let buffer = Stdlib.Buffer.create 256 in
    for y = 0 to state.height - 1 do
      for x = 0 to state.width - 1 do
        let c =
          match state.grid.(y).(x) with
          | Empty -> ' '
          | Wall -> '#'
          | Box -> '$'
          | Target -> '.'
          | Box_on_target -> '*'
          | Player -> '@'
          | Player_on_target -> '+'
        in
        Stdlib.Buffer.add_char buffer c
      done;
      Stdlib.Buffer.add_char buffer '\n'
    done;
    Stdlib.Buffer.contents buffer

  (** Deadlock detection for unsolvable states *)

  let is_corner pos state =
    let x, y = pos in
    let is_wall_or_boundary (px, py) =
      px < 0 || px >= state.width || py < 0 || py >= state.height
      || state.grid.(py).(px) = Wall
    in
    (* Check if position is in a corner (two adjacent walls) *)
    (is_wall_or_boundary (x - 1, y) && is_wall_or_boundary (x, y - 1))
    || (is_wall_or_boundary (x - 1, y) && is_wall_or_boundary (x, y + 1))
    || (is_wall_or_boundary (x + 1, y) && is_wall_or_boundary (x, y - 1))
    || (is_wall_or_boundary (x + 1, y) && is_wall_or_boundary (x, y + 1))

  let count_targets state =
    let count = ref 0 in
    for y = 0 to state.height - 1 do
      for x = 0 to state.width - 1 do
        match state.grid.(y).(x) with
        | Target | Box_on_target | Player_on_target -> incr count
        | _ -> ()
      done
    done;
    !count

  let count_boxes state =
    let count = ref 0 in
    for y = 0 to state.height - 1 do
      for x = 0 to state.width - 1 do
        match state.grid.(y).(x) with
        | Box | Box_on_target -> incr count
        | _ -> ()
      done
    done;
    !count

  let is_box_deadlocked (x, y) state =
    let cell = get_cell state (x, y) in
    if not (is_box cell) then false
    else if cell = Box_on_target then false (* Box is already on target *)
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
        (is_wall_or_box (x - 1, y) && is_wall_or_box (x, y - 1))
        || (is_wall_or_box (x - 1, y) && is_wall_or_box (x, y + 1))
        || (is_wall_or_box (x + 1, y) && is_wall_or_box (x, y - 1))
        || (is_wall_or_box (x + 1, y) && is_wall_or_box (x, y + 1))
      in

      if is_in_corner then true (* Corner without target is always deadlock *)
      else
        (* Check for 2x2 box pattern (4 boxes in a square = deadlock) *)
        let check_2x2_pattern =
          (* Check if this box is part of a 2x2 box formation *)
          let positions =
            [
              (* Check all 4 positions where current box could be part of 2x2 *)
              [ (0, 0); (1, 0); (0, 1); (1, 1) ];
              (* Top-left *)
              [ (-1, 0); (0, 0); (-1, 1); (0, 1) ];
              (* Top-right *)
              [ (0, -1); (1, -1); (0, 0); (1, 0) ];
              (* Bottom-left *)
              [ (-1, -1); (0, -1); (-1, 0); (0, 0) ] (* Bottom-right *);
            ]
          in
          List.exists
            (fun offsets ->
              List.for_all
                (fun (dx, dy) ->
                  let nx, ny = (x + dx, y + dy) in
                  nx >= 0 && nx < state.width && ny >= 0 && ny < state.height
                  && is_box (get_cell state (nx, ny)))
                offsets)
            positions
        in

        if check_2x2_pattern then true
          (* 2x2 box formation is always deadlock *)
        else (* Check for deadlock patterns along walls *)
          let is_wall_at (dx, dy) = get_cell state (x + dx, y + dy) = Wall in

          (* Box against wall - check if it can reach a target on the SAME wall segment *)
          (* A wall segment continues until hitting a corner or wall break *)

          (* Check which wall the box is against *)
          let against_top = is_wall_at (0, -1) in
          let against_bottom = is_wall_at (0, 1) in
          let against_left = is_wall_at (-1, 0) in
          let against_right = is_wall_at (1, 0) in

          (* Helper to check wall segment for targets *)
          let check_wall_segment ~is_horizontal ~wall_coord ~box_coord
              ~max_coord =
            let rec scan pos dir =
              let next_pos = pos + dir in
              if next_pos < 0 || next_pos >= max_coord then true
              else
                let wall_pos, check_pos =
                  if is_horizontal then
                    ((next_pos, wall_coord), (next_pos, box_coord))
                  else ((wall_coord, next_pos), (box_coord, next_pos))
                in
                let wall_continues = get_cell state wall_pos = Wall in
                let cell_at_pos = get_cell state check_pos in

                if not wall_continues then true
                else if cell_at_pos = Wall then true
                else if
                  cell_at_pos = Target
                  || cell_at_pos = Box_on_target
                  || cell_at_pos = Player_on_target
                then false
                else if is_box cell_at_pos then true
                else scan next_pos dir
            in
            scan box_coord 1 && scan box_coord (-1)
          in

          if against_top || against_bottom then
            let wall_y = if against_top then y - 1 else y + 1 in
            check_wall_segment ~is_horizontal:true ~wall_coord:wall_y
              ~box_coord:y ~max_coord:state.width
          else if against_left || against_right then
            let wall_x = if against_left then x - 1 else x + 1 in
            check_wall_segment ~is_horizontal:false ~wall_coord:wall_x
              ~box_coord:x ~max_coord:state.height
          else false

  let has_deadlock state =
    try
      for y = 0 to state.height - 1 do
        for x = 0 to state.width - 1 do
          if is_box_deadlocked (x, y) state then raise Exit
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
module Level_gen = struct
  open Core

  let make_empty width height =
    {
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

  (** Shuffle a list using Fisher-Yates *)
  let shuffle lst =
    let arr = Array.of_list lst in
    for i = Array.length arr - 1 downto 1 do
      let j = Random.int (i + 1) in
      let tmp = arr.(i) in
      arr.(i) <- arr.(j);
      arr.(j) <- tmp
    done;
    Array.to_list arr

  let generate_corridor length =
    (* Randomly choose horizontal or vertical orientation *)
    let horizontal = Random.bool () in

    if horizontal then (
      (* Horizontal corridor *)
      let width = length + 2 in
      let height = 3 in
      let state = make_empty width height in
      add_walls state;

      (* Randomly choose push direction *)
      if Random.bool () then (* Push right *)
        (
        state.grid.(1).(1) <- Player;
        (* Randomly place box between player and target *)
        let box_pos = 2 + Random.int (max 1 (length - 2)) in
        state.grid.(1).(box_pos) <- Box;
        state.grid.(1).(length) <- Target;
        { state with player_pos = (1, 1) })
      else (* Push left *)
        (
        state.grid.(1).(length) <- Player;
        (* Randomly place box between player and target *)
        let box_pos = 2 + Random.int (max 1 (length - 2)) in
        state.grid.(1).(box_pos) <- Box;
        state.grid.(1).(1) <- Target;
        { state with player_pos = (length, 1) }))
    else
      (* Vertical corridor *)
      let width = 3 in
      let height = length + 2 in
      let state = make_empty width height in
      add_walls state;

      (* Randomly choose push direction *)
      if Random.bool () then (* Push down *)
        (
        state.grid.(1).(1) <- Player;
        (* Randomly place box between player and target *)
        let box_pos = 2 + Random.int (max 1 (length - 2)) in
        state.grid.(box_pos).(1) <- Box;
        state.grid.(length).(1) <- Target;
        { state with player_pos = (1, 1) })
      else (* Push up *)
        (
        state.grid.(length).(1) <- Player;
        (* Randomly place box between player and target *)
        let box_pos = 2 + Random.int (max 1 (length - 2)) in
        state.grid.(box_pos).(1) <- Box;
        state.grid.(1).(1) <- Target;
        { state with player_pos = (1, length) })

  let generate_room size =
    let max_attempts = 50 in

    let rec attempt n =
      if n >= max_attempts then (
        (* Fall back to simple line configuration *)
        let state = make_empty size size in
        add_walls state;
        let center = size / 2 in
        state.grid.(center).(center - 1) <- Player;
        state.grid.(center).(center) <- Box;
        state.grid.(center).(center + 1) <- Target;
        { state with player_pos = (center - 1, center) })
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

        let positions = shuffle !all_positions in

        (* Try to place player, box, and target *)
        match positions with
        | [] -> attempt (n + 1)
        | p1 :: rest_positions -> (
            (* Place player first *)
            let px, py = p1 in
            state.grid.(py).(px) <- Player;

            (* Try different combinations for box and target *)
            let try_placement (bx, by) (tx, ty) =
              (* Don't place box and target in same position *)
              if (bx, by) = (tx, ty) then false
              else (
                (* Temporarily place box *)
                state.grid.(by).(bx) <- Box;

                (* Check corner placement rules for small rooms *)
                let corner_ok =
                  if size <= 5 then
                    (* In small rooms (3x3 playable), box in corner is OK only
                       if target is also in a corner *)
                    if Core.is_corner (bx, by) state then
                      Core.is_corner (tx, ty) state && (bx, by) <> (tx, ty)
                    else true
                  else
                    (* In larger rooms, avoid corners for boxes *)
                    not (Core.is_corner (bx, by) state)
                in

                if not corner_ok then (
                  state.grid.(by).(bx) <- Empty;
                  false)
                else (
                  (* Place target *)
                  state.grid.(ty).(tx) <- Target;

                  (* Check if configuration is potentially solvable *)
                  let test_state = { state with player_pos = (px, py) } in
                  if Core.is_potentially_solvable test_state then true
                  else (
                    (* Revert changes *)
                    state.grid.(by).(bx) <- Empty;
                    state.grid.(ty).(tx) <- Empty;
                    false)))
            in

            (* Try multiple placement combinations from remaining positions *)
            let combinations =
              match rest_positions with
              | p2 :: p3 :: p4 :: _ ->
                  [ (p2, p3); (p2, p4); (p3, p2); (p3, p4); (p4, p2); (p4, p3) ]
              | [ p2; p3 ] -> [ (p2, p3); (p3, p2) ]
              | _ -> []
            in

            let rec try_combinations = function
              | [] -> None
              | (b, t) :: rest ->
                  if try_placement b t then
                    Some { state with player_pos = (px, py) }
                  else
                    (* Clean up any leftover state *)
                    let bx, by = b in
                    let tx, ty = t in
                    state.grid.(by).(bx) <- Empty;
                    state.grid.(ty).(tx) <- Empty;
                    try_combinations rest
            in

            match try_combinations combinations with
            | Some final_state -> final_state
            | None ->
                (* Clean up player and try again *)
                state.grid.(py).(px) <- Empty;
                attempt (n + 1))
    in
    attempt 0

  let generate_multi_box num_boxes =
    let max_attempts = 100 in

    let rec attempt n =
      if n >= max_attempts then (
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
        { state with player_pos = (center, center) })
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

        let shuffled = shuffle !positions in

        (* Place player first *)
        match shuffled with
        | [] -> attempt (n + 1)
        | (px, py) :: rest -> (
            state.grid.(py).(px) <- Player;

            (* Try to place boxes and targets, checking for deadlocks after each
               box *)
            let rec place_items positions_left boxes_to_place targets_to_place =
              if boxes_to_place = 0 && targets_to_place = 0 then
                (* Success! *)
                Some { state with player_pos = (px, py) }
              else if positions_left = [] then None (* Not enough positions *)
              else if boxes_to_place > 0 then
                (* Try to place a box *)
                let rec try_box_positions = function
                  | [] -> None
                  | (x, y) :: rest_pos -> (
                      state.grid.(y).(x) <- Box;
                      (* Check if this creates a deadlock *)
                      if Core.is_box_deadlocked (x, y) state then (
                        (* This box is deadlocked, try next position *)
                        state.grid.(y).(x) <- Empty;
                        try_box_positions rest_pos)
                      else
                        (* Box is OK, continue placing *)
                        match
                          place_items rest_pos (boxes_to_place - 1)
                            targets_to_place
                        with
                        | Some result -> Some result
                        | None ->
                            (* Backtrack *)
                            state.grid.(y).(x) <- Empty;
                            try_box_positions rest_pos)
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
                attempt (n + 1))
    in
    attempt 0

  let generate_complex () =
    let layout =
      [|
        [| Wall; Wall; Wall; Wall; Wall; Wall; Wall |];
        [| Wall; Empty; Empty; Target; Empty; Empty; Wall |];
        [| Wall; Empty; Box; Empty; Box; Empty; Wall |];
        [| Wall; Target; Empty; Player; Empty; Target; Wall |];
        [| Wall; Empty; Box; Empty; Box; Empty; Wall |];
        [| Wall; Empty; Empty; Target; Empty; Empty; Wall |];
        [| Wall; Wall; Wall; Wall; Wall; Wall; Wall |];
      |]
    in

    { grid = layout; player_pos = (3, 3); width = 7; height = 7 }
end

(** Curriculum learning module *)
module Curriculum = struct
  type stage = Corridor of int | Room of int | Multi_box of int | Complex

  type config = {
    stages : stage list;
    current_idx : int ref;
    success_threshold : float;
    window_size : int;
    recent_rewards : float list ref;
  }

  let create_config ?(success_threshold = 0.6) ?(window_size = 20) stages =
    {
      stages;
      current_idx = ref 0;
      success_threshold;
      window_size;
      recent_rewards = ref [];
    }

  let trim_to n lst =
    let rec aux idx acc = function
      | [] -> List.rev acc
      | _ when idx >= n -> List.rev acc
      | x :: xs -> aux (idx + 1) (x :: acc) xs
    in
    aux 0 [] lst

  let generate_level = function
    | Corridor len -> Level_gen.generate_corridor len
    | Room size -> Level_gen.generate_room size
    | Multi_box n -> Level_gen.generate_multi_box n
    | Complex -> Level_gen.generate_complex ()

  let update_and_check_advance config _reward won =
    (* Track win (1.0) or loss (0.0) for success rate calculation *)
    let outcome = if won then 1.0 else 0.0 in
    config.recent_rewards :=
      trim_to config.window_size (outcome :: !(config.recent_rewards));
    let count = List.length !(config.recent_rewards) in
    let minimum_samples = max 10 (config.window_size / 2) in
    if count >= minimum_samples then
      let wins = List.filter (fun r -> r > 0.5) !(config.recent_rewards) in
      let success_rate =
        float_of_int (List.length wins) /. float_of_int count
      in

      if
        success_rate >= config.success_threshold
        && !(config.current_idx) < List.length config.stages - 1
      then (
        incr config.current_idx;
        config.recent_rewards := [];
        true)
      else false
    else false

  let get_current_stage config = List.nth config.stages !(config.current_idx)

  let reset_curriculum config =
    config.current_idx := 0;
    config.recent_rewards := []
end

type observation = (float, Rune.float32_elt) Rune.t
type action = (int32, Rune.int32_elt) Rune.t
type render = Render.t

type state = {
  mutable game_state : Core.game_state;
  mutable steps : int;
  max_steps : int;
  curriculum_config : Curriculum.config option;
  initial_state : Core.game_state option;
  initial_state_fn : (unit -> Core.game_state) option;
}

module Env_table = Hashtbl.Make (struct
  type t = Obj.t

  let equal a b = a == b
  let hash = Hashtbl.hash
end)

let state_registry : state Env_table.t = Env_table.create 16

let register_env env state =
  Env_table.replace state_registry (Obj.repr env) state

let unregister_env env = Env_table.remove state_registry (Obj.repr env)

let lookup_state env =
  match Env_table.find_opt state_registry (Obj.repr env) with
  | Some state -> state
  | None -> invalid_arg "Sokoban_env: environment not registered"

let max_grid_size = 10
let observation_channels = 8
let observation_flat_size = observation_channels * max_grid_size * max_grid_size
let mask_channel_index = observation_channels - 1

let cell_to_channel = function
  | Core.Empty -> 0
  | Core.Wall -> 1
  | Core.Box -> 2
  | Core.Target -> 3
  | Core.Box_on_target -> 4
  | Core.Player -> 5
  | Core.Player_on_target -> 6

let render_text state = Core.render state.game_state

let tile_color = function
  | Core.Empty -> (40, 44, 52)
  | Core.Wall -> (80, 86, 96)
  | Core.Box -> (196, 142, 80)
  | Core.Target -> (220, 200, 130)
  | Core.Box_on_target -> (120, 190, 120)
  | Core.Player -> (82, 142, 212)
  | Core.Player_on_target -> (120, 200, 220)

let cell_size = 32
let frame_cells = max_grid_size
let frame_width = frame_cells * cell_size
let frame_height = frame_cells * cell_size
let clamp_color value = max 0 (min 255 value)

let fill_rect data ~width ~x0 ~y0 ~w ~h (r, g, b) =
  let r = Char.unsafe_chr (clamp_color r) in
  let g = Char.unsafe_chr (clamp_color g) in
  let b = Char.unsafe_chr (clamp_color b) in
  for dy = 0 to h - 1 do
    let y = y0 + dy in
    let row_offset = y * width * 3 in
    for dx = 0 to w - 1 do
      let x = x0 + dx in
      let base = row_offset + (x * 3) in
      Bigarray.Array1.unsafe_set data (base + 0) r;
      Bigarray.Array1.unsafe_set data (base + 1) g;
      Bigarray.Array1.unsafe_set data (base + 2) b
    done
  done

let render_image state =
  let data =
    Bigarray.Array1.create Bigarray.char Bigarray.c_layout
      (frame_width * frame_height * 3)
  in
  (* Background *)
  fill_rect data ~width:frame_width ~x0:0 ~y0:0 ~w:frame_width ~h:frame_height
    (32, 36, 44);
  for gy = 0 to frame_cells - 1 do
    for gx = 0 to frame_cells - 1 do
      let cell =
        if gx < state.game_state.Core.width && gy < state.game_state.Core.height
        then state.game_state.Core.grid.(gy).(gx)
        else Core.Empty
      in
      let x0 = gx * cell_size in
      let y0 = gy * cell_size in
      let base_color = tile_color cell in
      fill_rect data ~width:frame_width ~x0 ~y0 ~w:cell_size ~h:cell_size
        base_color;
      (* subtle inner highlight *)
      fill_rect data ~width:frame_width ~x0:(x0 + 2) ~y0:(y0 + 2)
        ~w:(cell_size - 4) ~h:(cell_size - 4)
        (match cell with
        | Core.Wall -> (90, 96, 108)
        | Core.Box -> (206, 152, 90)
        | Core.Box_on_target -> (130, 206, 130)
        | Core.Player -> (102, 162, 232)
        | Core.Player_on_target -> (135, 215, 230)
        | Core.Target -> (230, 210, 150)
        | Core.Empty -> (46, 50, 58))
    done
  done;
  Render.image_u8 ~width:frame_width ~height:frame_height ~pixel_format:`RGB8
    ~data ()

let render env state =
  match Env.render_mode env with
  | Some `Rgb_array -> Render.Image (render_image state)
  | Some `Human -> Render.Image (render_image state)
  | Some `Ansi -> Render.Text (render_text state)
  | Some (`Custom mode) when String.equal mode "rgb_array" ->
      Render.Image (render_image state)
  | Some (`Custom mode) when String.equal mode "ansi" ->
      Render.Text (render_text state)
  | _ -> Render.Text (render_text state)

let observation_space =
  Space.Box.create
    ~low:(Array.make observation_flat_size 0.0)
    ~high:(Array.make observation_flat_size 1.0)

let action_space = Space.Discrete.create 4

(** Create observation tensor from game state with padding *)
let make_observation game_state =
  let g = max_grid_size in
  let data = Array.make observation_flat_size 0.0 in
  let open Core in
  let height = game_state.height in
  let width = game_state.width in
  for y = 0 to height - 1 do
    for x = 0 to width - 1 do
      let base_index = (y * g) + x in
      let mask_offset = (mask_channel_index * g * g) + base_index in
      data.(mask_offset) <- 1.0;
      let cell = game_state.grid.(y).(x) in
      let channel = cell_to_channel cell in
      let offset = (channel * g * g) + base_index in
      data.(offset) <- 1.0
    done
  done;
  Rune.create Rune.float32 [| observation_flat_size |] data

let can_move direction state =
  let open Core in
  let next_pos = move_position state.player_pos direction in
  match get_cell state next_pos with
  | Empty | Target -> true
  | Box | Box_on_target ->
      let after_box = move_position next_pos direction in
      is_empty (get_cell state after_box)
  | Player | Player_on_target -> false
  | Wall -> false

let action_mask state =
  [|
    can_move Core.Up state;
    can_move Core.Down state;
    can_move Core.Left state;
    can_move Core.Right state;
  |]

let manhattan (x1, y1) (x2, y2) = Stdlib.abs (x1 - x2) + Stdlib.abs (y1 - y2)

let boxes_and_targets state =
  let open Core in
  let boxes = ref [] in
  let targets = ref [] in
  for y = 0 to state.height - 1 do
    for x = 0 to state.width - 1 do
      match state.grid.(y).(x) with
      | Box -> boxes := (x, y) :: !boxes
      | Box_on_target ->
          boxes := (x, y) :: !boxes;
          targets := (x, y) :: !targets
      | Target -> targets := (x, y) :: !targets
      | Player_on_target -> targets := (x, y) :: !targets
      | _ -> ()
    done
  done;
  (!boxes, !targets)

let sorted_boxes state =
  let boxes, _ = boxes_and_targets state in
  List.sort Stdlib.compare boxes

let potential state =
  let boxes, targets = boxes_and_targets state in
  match (boxes, targets) with
  | [], _ | _, [] -> 0.0
  | _ ->
      let best_distance (x, y) =
        List.fold_left
          (fun acc target -> Stdlib.min acc (manhattan (x, y) target))
          max_int targets
      in
      let total =
        List.fold_left (fun acc box -> acc + best_distance box) 0 boxes
      in
      let max_per_box = max 1 (state.width - 1 + (state.height - 1)) in
      let max_total = max_per_box * List.length boxes in
      let diff = max_total - total in
      float_of_int (if diff > 0 then diff else 0)

let shaping_gamma = 0.99
let shaping_beta = 0.1

let has_any_move state =
  let mask = action_mask state in
  mask.(0) || mask.(1) || mask.(2) || mask.(3)

(** Convert action tensor to direction *)
let action_to_direction action =
  let action_val =
    let tensor = Rune.reshape [| 1 |] action in
    let arr : Int32.t array = Rune.to_array tensor in
    Int32.to_int arr.(0)
  in
  match action_val with
  | 0 -> Core.Up
  | 1 -> Core.Down
  | 2 -> Core.Left
  | _ -> Core.Right

(** Get stage info string for curriculum *)
let stage_info curriculum_config =
  match curriculum_config with
  | Some config ->
      let total = List.length config.Curriculum.stages in
      let current = !(config.current_idx) + 1 in
      Printf.sprintf "%d/%d" current total
  | None -> "1/1"

let registered_curriculum env =
  try (lookup_state env).curriculum_config with Invalid_argument _ -> None

let stage_to_string = function
  | Curriculum.Corridor len -> Printf.sprintf "corridor-%d" len
  | Curriculum.Room size -> Printf.sprintf "room-%d" size
  | Curriculum.Multi_box n -> Printf.sprintf "multi-box-%d" n
  | Curriculum.Complex -> "complex"

let stage_descriptor curriculum_config =
  match curriculum_config with
  | Some config ->
      let stage = Curriculum.get_current_stage config in
      let idx = !(config.Curriculum.current_idx) in
      let total = List.length config.Curriculum.stages in
      Printf.sprintf "%s-%02d-of-%02d" (stage_to_string stage) (idx + 1) total
  | None -> "single-stage"

let current_game_state env = Core.copy_state (lookup_state env).game_state
let has_registered_state env = Env_table.mem state_registry (Obj.repr env)
let state_opt env = Env_table.find_opt state_registry (Obj.repr env)

let current_stage env =
  let state = lookup_state env in
  match state.curriculum_config with
  | Some config ->
      let idx = !(config.Curriculum.current_idx) in
      let stage = Curriculum.get_current_stage config in
      Some (stage, idx, List.length config.Curriculum.stages)
  | None -> None

let current_stage_label env = stage_info (registered_curriculum env)

let current_stage_descriptor env =
  match state_opt env with
  | Some state -> stage_descriptor state.curriculum_config
  | None -> "static-level"

let current_stage_descriptor_opt env =
  match state_opt env with
  | None -> None
  | Some state -> (
      match state.curriculum_config with
      | None -> None
      | Some _ -> Some (stage_descriptor state.curriculum_config))

let current_stage_label_opt env =
  match state_opt env with
  | None -> None
  | Some state -> Some (stage_info state.curriculum_config)

let current_game_state_opt env =
  match state_opt env with
  | None -> None
  | Some state -> Some (Core.copy_state state.game_state)

let metadata =
  Metadata.default
  |> Metadata.add_render_mode "rgb_array"
  |> Metadata.add_render_mode "ansi"
  |> Metadata.with_description
       (Some "Sokoban puzzle game with curriculum learning support")
  |> Metadata.add_author "Fehu Team"
  |> Metadata.with_version (Some "0.1.0")

let reset _env ?options:_ () state =
  state.steps <- 0;
  let level =
    match state.curriculum_config with
    | Some config ->
        Curriculum.generate_level (Curriculum.get_current_stage config)
    | None -> (
        match state.initial_state_fn with
        | Some gen -> gen ()
        | None -> (
            match state.initial_state with
            | Some initial -> Core.copy_state initial
            | None -> Core.copy_state state.game_state))
  in
  state.game_state <- level;
  let obs = make_observation level in
  let info =
    Info.empty
    |> Info.set "stage" (Info.string (stage_descriptor state.curriculum_config))
    |> Info.set "action_mask" (Info.bool_array (action_mask level))
  in
  (obs, info)

let step _env action state =
  state.steps <- state.steps + 1;
  let phi_s = potential state.game_state in
  let boxes_before = sorted_boxes state.game_state in
  let direction = action_to_direction action in
  let new_state = Core.apply_action state.game_state direction in
  let moved = not (Stdlib.( == ) new_state state.game_state) in
  state.game_state <- new_state;
  let boxes_after = sorted_boxes state.game_state in
  let pushed = boxes_before <> boxes_after in
  let won = Core.check_win state.game_state in
  let no_moves = not (has_any_move state.game_state) in
  let truncated = state.steps >= state.max_steps || no_moves in

  let base_reward = if won then 100.0 else if not moved then -0.2 else -0.01 in
  let phi_s' = potential state.game_state in
  let shaping =
    if pushed then shaping_beta *. ((shaping_gamma *. phi_s') -. phi_s) else 0.0
  in
  let reward = base_reward +. shaping in

  (* Only check advancement when episode ends (won or truncated) *)
  let advanced =
    match state.curriculum_config with
    | Some config when won || truncated ->
        Curriculum.update_and_check_advance config reward won
    | _ -> false
  in

  let obs = make_observation state.game_state in
  let info =
    Info.empty
    |> Info.set "steps" (Info.int state.steps)
    |> Info.set "stage" (Info.string (stage_descriptor state.curriculum_config))
    |> Info.set "advanced" (Info.bool advanced)
    |> Info.set "dead_end" (Info.bool no_moves)
    |> Info.set "action_mask" (Info.bool_array (action_mask state.game_state))
  in

  Env.transition ~observation:obs ~reward ~terminated:won ~truncated ~info ()

let make ?(max_steps = 200) ?(rng = Rune.Rng.key 42) ?curriculum_stages
    ?render_mode () =
  let curriculum_config =
    Option.map Curriculum.create_config curriculum_stages
  in

  let initial_game_state = Level_gen.generate_room 5 in
  let state =
    {
      game_state = initial_game_state;
      steps = 0;
      max_steps;
      curriculum_config;
      initial_state =
        (match curriculum_config with
        | Some _ -> None
        | None -> Some (Core.copy_state initial_game_state));
      initial_state_fn = None;
    }
  in

  let env =
    Env.create ~id:"Sokoban-v0" ~metadata ?render_mode ~rng ~observation_space
      ~action_space
      ~reset:(fun env ?options () -> reset env ?options () state)
      ~step:(fun env action -> step env action state)
      ~render:(fun env -> Some (render env state))
      ~close:(fun env -> unregister_env env)
      ()
  in
  register_env env state;
  env

let sokoban ?(max_steps = 200) ?initial_state ?initial_state_fn ?render_mode ()
    =
  (match (initial_state, initial_state_fn) with
  | Some _, Some _ ->
      invalid_arg
        "Sokoban_env.sokoban: provide either initial_state or initial_state_fn"
  | _ -> ());

  let base_state, stored_initial, stored_fn =
    match (initial_state, initial_state_fn) with
    | Some s, _ ->
        let copy = Core.copy_state s in
        (copy, Some (Core.copy_state copy), None)
    | None, Some gen ->
        let initial = gen () in
        (initial, None, Some gen)
    | None, None ->
        let generated = Level_gen.generate_room 5 in
        (generated, Some (Core.copy_state generated), None)
  in

  let state =
    {
      game_state = base_state;
      steps = 0;
      max_steps;
      curriculum_config = None;
      initial_state = stored_initial;
      initial_state_fn = stored_fn;
    }
  in

  let env =
    Env.create ~id:"Sokoban-v0" ~metadata ?render_mode ~rng:(Rune.Rng.key 42)
      ~observation_space ~action_space
      ~reset:(fun env ?options () -> reset env ?options () state)
      ~step:(fun env action -> step env action state)
      ~render:(fun env -> Some (render env state))
      ~close:(fun env -> unregister_env env)
      ()
  in
  register_env env state;
  env

let sokoban_curriculum ?max_steps ?render_mode () =
  let stages =
    [
      Curriculum.Corridor 3;
      Curriculum.Corridor 5;
      Curriculum.Room 5;
      Curriculum.Room 7;
      Curriculum.Multi_box 2;
      Curriculum.Multi_box 3;
      Curriculum.Complex;
    ]
  in
  make ?max_steps ?render_mode ~curriculum_stages:stages ()
