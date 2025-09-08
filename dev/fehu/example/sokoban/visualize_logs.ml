(** Visualize logged trajectories from training *)

open Fehu

let () =
  let args = Sys.argv in
  if Array.length args < 2 then begin
    Printf.printf "Usage: %s <log_file.json> [delay_ms]\n" args.(0);
    Printf.printf "Example: %s logs/backoff_episode_4500.json 500\n" args.(0);
    exit 1
  end;
  
  let log_file = args.(1) in
  let delay = 
    if Array.length args > 2 then 
      float_of_string args.(2) /. 1000.0
    else 
      0.5
  in
  
  Printf.printf "Loading episode log from %s...\n" log_file;
  let episode = Visualization.load_episode_log log_file in
  
  Printf.printf "\n=== Episode %d ===\n" episode.episode_num;
  Printf.printf "Total reward: %.2f\n" episode.total_reward;
  Printf.printf "Total steps: %d\n" episode.total_steps;
  Printf.printf "Won: %b\n" episode.won;
  Printf.printf "Stage: %s\n\n" (Option.value episode.stage ~default:"N/A");
  
  Printf.printf "Press Enter to start visualization...\n";
  let _ = read_line () in
  
  Visualization.animate_episode episode delay;
  
  Printf.printf "\nVisualization complete!\n"