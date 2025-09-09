(* 
```ocaml
 *)
open Fehu
let dev = Rune.metal ()
(* Workshop Part 1: Define a simple grid world *)
let create_simple_gridworld size =
  (* Mutable state for agent position *)
  let agent_pos = ref (0, 0) in
  let goal_pos = (size - 1, size - 1) in
  (* Define observation and action spaces *)
  let observation_space =
    Space.Box {
     low = Rune.zeros dev Rune.float32 [|size; size|];
     high = Rune.ones dev Rune.float32 [|size; size|];
     shape = [|size; size|];
    }
  in  
  (* Up, Down, Left, Right *) 
  let action_space = Space.Discrete 4 in
  (* Reset function - initialize episode *)
  let reset ?seed () =
    let _ = Option.map Random.init seed in
    agent_pos := (0, 0);
    let obs = Rune.zeros dev Rune.float32 [|size; size|] in
    (* Mark agent position *)
    Rune.unsafe_set [0; 0] 1.0 obs;
    (obs, [])
  in  
  (* Step function - take action and return new state *)
  let step action =
    let action_idx =
      Rune.unsafe_get [] action |> int_of_float in
    let x, y = !agent_pos in    
    (* Move based on action *)
    let new_pos = match action_idx with
      | 0 -> (x, max 0 (y - 1))  (* Up *)
      | 1 -> (x, min (size-1) (y + 1))  (* Down *)
      | 2 -> (max 0 (x - 1), y)  (* Left *)
      | 3 -> (min (size-1) (x + 1), y)  (* Right *)
      | _ -> (x, y)
    in    
    agent_pos := new_pos;    
    (* Create observation *)
    let obs = Rune.zeros dev Rune.float32 [|size; size|] in
    let x, y = !agent_pos in
    Rune.unsafe_set [x; y] 1.0 obs;   
    (* Compute reward *)
    let reward = if new_pos = goal_pos then 10.0 else -0.1 in
    let terminated = new_pos = goal_pos in
    (obs, reward, terminated, false, [])
  in
  Env.make ~observation_space ~action_space ~reset ~step ()
(* Test the environment *)
let () =
  let env = create_simple_gridworld 5 in
  let obs, _ = env.reset () in
  print_endline "Initial state:";
  Rune.print obs
(* 
```
 *)