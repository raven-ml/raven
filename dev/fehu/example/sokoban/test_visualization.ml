open Fehu

let () =
  Printf.printf "Testing Sokoban episode visualization...\n";
  
  (* Create a simple corridor level *)
  let initial_state = Sokoban.LevelGen.generate_corridor 3 in
  let env = Sokoban.sokoban ~width:5 ~height:3 ~max_steps:20 
    ~initial_state () in
  
  (* Create a dummy episode log *)
  let frames = ref [] in
  
  (* Collect a few steps *)
  let _obs, _info = env.reset () in
  
  for step = 1 to 5 do
    (* Render current state to string *)
    let state_repr = 
      let buffer = Stdlib.Buffer.create 256 in
      let old_stdout = Unix.dup Unix.stdout in
      let tmp_file = Filename.temp_file "sokoban" ".txt" in
      let fd = Unix.openfile tmp_file [Unix.O_RDWR; Unix.O_CREAT; Unix.O_TRUNC] 0o600 in
      Unix.dup2 fd Unix.stdout;
      Unix.close fd;
      env.render ();
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
    in
    
    (* Random action *)
    let action = Rune.scalar Rune.c Rune.float32 (float_of_int (Random.int 4)) in
    let _next_obs, reward, terminated, _truncated, _info = env.step action in
    
    let frame = Visualization.{
      step;
      state_repr;
      action = Visualization.action_to_string (Random.int 4);
      reward;
      value = None;
    } in
    frames := frame :: !frames;
    
    if terminated then (
      Printf.printf "Episode ended at step %d!\n" step;
      ()
    )
  done;
  
  let log = Visualization.{
    episode_num = 1;
    total_reward = 10.0;
    total_steps = 5;
    won = false;
    frames = List.rev !frames;
    stage = Some "Corridor(3)";
  } in
  
  Printf.printf "\nAnimating episode:\n";
  Visualization.animate_episode log 0.5;
  
  Printf.printf "\nSaving to file...\n";
  (try Unix.mkdir "logs" 0o755 with Unix.Unix_error _ -> ());
  Visualization.save_episode_log log "logs/test_episode.log";
  
  Printf.printf "Done!\n"