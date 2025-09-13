# Exercise 5: Egocentric Sokoban - Learning Position-Invariant Policies

## Background

Our current Sokoban implementation uses an **allocentric** (world-centered) view where the agent sees the entire grid from a fixed perspective. This means the agent must learn different policies for different starting positions - what works at position (1,1) won't work at position (3,3).

An **egocentric** (agent-centered) view always places the agent at the center of the observation, making the policy position-invariant. The agent learns relative rules like "push box forward" instead of absolute rules like "at (1,1), go right."

## Your Task

Transform the Sokoban environment to use egocentric observations while keeping the orientation allocentric (north is always up).

## Current Allocentric Implementation

```ocaml
(* Current: Agent sees full grid with absolute positions *)
let state = Rune.init device Rune.float32 [|5; 5|] (fun idxs ->
  let i, j = idxs.(0), idxs.(1) in
  match grid.(i).(j) with
  | '@' -> 0.9  (* Agent position varies *)
  | 'o' -> 0.5  (* Box *)
  | 'x' -> 0.75 (* Target *)
  | '#' -> 1.0  (* Wall *)
  | ' ' -> 0.0  (* Empty *)
)
```

## Target Egocentric Implementation

```ocaml
(* Goal: Agent always at center of 5x5 observation *)
let make_egocentric_observation grid agent_pos =
  let ay, ax = agent_pos in  (* Agent's position in world *)
  let view_size = 5 in
  let center = view_size / 2 in  (* Center = 2 for 5x5 *)

  Rune.init device Rune.float32 [|view_size; view_size|] (fun idxs ->
    let view_y, view_x = idxs.(0), idxs.(1) in

    (* Convert view coordinates to world coordinates *)
    let world_y = ay + (view_y - center) in
    let world_x = ax + (view_x - center) in

    (* Check if we're looking at the agent's position *)
    if view_y = center && view_x = center then
      0.9  (* Agent always at center *)
    else if world_y < 0 || world_y >= grid_height ||
            world_x < 0 || world_x >= grid_width then
      1.0  (* Out of bounds = wall *)
    else
      match grid.(world_y).(world_x) with
      | 'o' -> 0.5   (* Box *)
      | 'x' -> 0.75  (* Target *)
      | '#' -> 1.0   (* Wall *)
      | ' ' -> 0.0   (* Empty *)
      | '@' -> 0.0   (* Treat agent cell as empty *)
  )
```

## Implementation Steps

### Step 1: Modify Environment Creation

Update `create_curriculum_env` in slide11.ml to track agent position:

```ocaml
let create_egocentric_env curriculum_state =
  (* ... existing grid creation ... *)

  (* Find agent position *)
  let agent_pos = ref (0, 0) in
  for i = 0 to height - 1 do
    for j = 0 to width - 1 do
      if grid.(i).(j) = '@' then
        agent_pos := (i, j)
    done
  done;

  (* Create egocentric observation *)
  let make_observation () =
    make_egocentric_observation grid !agent_pos in

  (* Update reset and step to maintain agent_pos *)
  let reset ?seed:_ () =
    agent_pos := find_agent_position grid;
    (make_observation (), [])
  in

  let step action =
    (* Update agent_pos based on action *)
    let dy, dx = match int_of_float (Rune.item [] action) with
      | 0 -> (-1, 0)  (* Up *)
      | 1 -> (1, 0)   (* Down *)
      | 2 -> (0, -1)  (* Left *)
      | 3 -> (0, 1)   (* Right *)
      | _ -> (0, 0)
    in
    (* Simplified: just move agent, ignore collision *)
    let new_y = fst !agent_pos + dy in
    let new_x = snd !agent_pos + dx in
    if is_valid_move grid new_y new_x then
      agent_pos := (new_y, new_x);

    (* Return egocentric observation *)
    let obs = make_observation () in
    let reward = compute_reward () in
    let terminated = check_win () in
    (obs, reward, terminated, false, [])
  in

  Fehu.Env.make ~observation_space ~action_space ~reset ~step ~render ()
```

### Step 2: Test Position Invariance

Create two identical rooms with the agent at different positions:

```ocaml
let test_position_invariance () =
  (* Same layout, different agent positions *)
  let grid1 = [|
    [|'#'; '#'; '#'; '#'; '#'; '#'; '#'|];
    [|'#'; '@'; ' '; ' '; ' '; ' '; '#'|];  (* Agent at (1,1) *)
    [|'#'; ' '; ' '; 'o'; ' '; ' '; '#'|];
    [|'#'; ' '; ' '; ' '; 'x'; ' '; '#'|];
    [|'#'; '#'; '#'; '#'; '#'; '#'; '#'|];
  |] in

  let grid2 = [|
    [|'#'; '#'; '#'; '#'; '#'; '#'; '#'|];
    [|'#'; ' '; ' '; ' '; ' '; ' '; '#'|];
    [|'#'; ' '; ' '; 'o'; ' '; ' '; '#'|];
    [|'#'; ' '; ' '; ' '; 'x'; '@'; '#'|];  (* Agent at (3,5) *)
    [|'#'; '#'; '#'; '#'; '#'; '#'; '#'|];
  |] in

  (* With egocentric view, both should produce similar observations *)
  let obs1 = make_egocentric_observation grid1 (1, 1) in
  let obs2 = make_egocentric_observation grid2 (3, 5) in

  (* obs1 and obs2 should show box at same relative position *)
```

### Step 3: Compare Learning Curves

Train two agents and compare:

```ocaml
let compare_representations () =
  (* Train allocentric agent *)
  let allo_returns = train_with_allocentric env 1000 in

  (* Train egocentric agent *)
  let ego_returns = train_with_egocentric env 1000 in

  (* Test generalization to new positions *)
  let new_env = create_test_env_with_random_start () in
  let allo_test = evaluate allo_agent new_env 100 in
  let ego_test = evaluate ego_agent new_env 100 in

  Printf.printf "Allocentric test return: %.2f\n" allo_test;
  Printf.printf "Egocentric test return: %.2f\n" ego_test;
  (* Egocentric should generalize better *)
```

## Expected Benefits

1. **Better Generalization**: Same policy works from any starting position
2. **Faster Learning**: Fewer parameters needed (no position memorization)
3. **Transfer Learning**: Policy trained on small grids may work on larger ones

## Challenges to Consider

1. **Partial Observability**: Agent can't see the whole puzzle
   - Solution: Add memory (LSTM) or increase view radius

2. **Loss of Global Context**: Harder to plan long sequences
   - Solution: Hybrid approach with both views

3. **Symmetric Situations**: Same local view, different optimal actions
   - Solution: Include additional features (distance to nearest target)

## Evaluation Criteria

Your implementation should:

1. **Maintain 5×5 observation size** with agent always at center (2,2)
2. **Handle edge cases** (agent near walls)
3. **Show improved generalization** to new starting positions
4. **Work with existing policy network** (no architecture changes needed)

## Bonus Challenges

1. **Variable View Radius**: Make view size a parameter (3×3, 7×7, etc.)
2. **Orientation**: Add agent orientation and rotate view accordingly
3. **Fog of War**: Only show previously visited areas
4. **Multi-Scale**: Combine local egocentric with downsampled global view

## Testing Your Implementation

```ocaml
(* Test that agent is always centered *)
let test_agent_centering env =
  let obs, _ = env.reset () in
  assert (Rune.item [2; 2] obs = 0.9);  (* Agent at center *)

  (* Move in any direction *)
  let obs, _, _, _, _ = env.step (Rune.scalar device Rune.float32 0.) in
  assert (Rune.item [2; 2] obs = 0.9);  (* Still at center *)
```

## Hints

- Start with a simple 3×3 egocentric view before scaling to 5×5
- Draw out the coordinate transformations on paper first
- Test with a single box puzzle before multi-box scenarios
- Remember the agent position needs to be tracked separately from the grid

## Connection to Real-World Robotics

Egocentric representations are standard in robotics because:
- Robots have sensors mounted on themselves (cameras, lidar)
- They don't have access to a "god's eye view" of the world
- Policies must work regardless of starting position
- Local navigation is often more important than global planning

Your egocentric Sokoban agent is learning the same kind of position-invariant policies that real robots use!