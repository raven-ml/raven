(** Quick test of DQN with curriculum to verify stage reporting *)

open Fehu

let () =
  Printf.printf "Testing DQN with curriculum stage reporting...\n\n";
  
  (* Create curriculum environment *)
  let env = Sokoban.sokoban_curriculum ~max_steps:50 () in
  
  (* Just run a few episodes to see stage reporting *)
  for episode = 1 to 20 do
    let obs, info = env.Env.reset () in
    let obs_ref = ref obs in
    let total_reward = ref 0.0 in
    let steps = ref 0 in
    let stage_info = ref info in
    let is_done = ref false in
    
    (* Run episode *)
    while not !is_done && !steps < 50 do
      (* Random action for testing *)
      let action = Random.int 4 in
      let action_tensor = Rune.scalar Rune.c Rune.float32 (float_of_int action) in
      let next_obs, reward, terminated, truncated, info = env.Env.step action_tensor in
      
      obs_ref := next_obs;
      total_reward := !total_reward +. reward;
      incr steps;
      
      if terminated || truncated then begin
        is_done := true;
        stage_info := info
      end
    done;
    
    (* Extract stage information *)
    let stage = 
      match List.assoc_opt "stage" !stage_info with
      | Some (`String s) -> s
      | _ -> "N/A"
    in
    
    let advanced = 
      match List.assoc_opt "advanced" !stage_info with
      | Some (`Bool true) -> " (ADVANCED!)"
      | _ -> ""
    in
    
    let won = !total_reward > 50.0 in
    
    Printf.printf "Episode %2d: Reward = %6.2f, Won = %b, Steps = %2d, Stage = %s%s\n"
      episode !total_reward won !steps stage advanced
  done;
  
  Printf.printf "\nDone!\n"