(** Visualization utilities for RL training *)

type episode_frame = {
  step: int;
  state_repr: string;
  action: string;
  reward: float;
  value: float option;
}

type episode_log = {
  episode_num: int;
  total_reward: float;
  total_steps: int;
  won: bool;
  frames: episode_frame list;
  stage: string option;
}

let clear_screen () =
  print_string "\027[2J\027[H"

let action_to_string action_idx =
  match action_idx with
  | 0 -> "Up"
  | 1 -> "Down"
  | 2 -> "Left"
  | 3 -> "Right"
  | _ -> "Unknown"

let animate_episode log delay =
  clear_screen ();
  Printf.printf "=== Episode %d (Stage: %s) ===\n" 
    log.episode_num 
    (Option.value log.stage ~default:"N/A");
  Printf.printf "Won: %b | Total Reward: %.2f | Steps: %d\n\n" 
    log.won log.total_reward log.total_steps;
  
  List.iter (fun frame ->
    Printf.printf "Step %d - Action: %s - Reward: %.2f" 
      frame.step frame.action frame.reward;
    (match frame.value with
     | Some v -> Printf.printf " - Value: %.2f" v
     | None -> ());
    Printf.printf "\n%s\n" frame.state_repr;
    Unix.sleepf delay
  ) log.frames;
  
  if log.won then
    print_endline "🎉 Level Complete!"
  else
    print_endline "Episode ended (timeout or failure)"

let save_episode_log log filename =
  let oc = open_out filename in
  Printf.fprintf oc "Episode %d\n" log.episode_num;
  Printf.fprintf oc "Stage: %s\n" (Option.value log.stage ~default:"N/A");
  Printf.fprintf oc "Won: %b\n" log.won;
  Printf.fprintf oc "Total Reward: %.2f\n" log.total_reward;
  Printf.fprintf oc "Total Steps: %d\n\n" log.total_steps;
  
  List.iter (fun frame ->
    Printf.fprintf oc "Step %d\n" frame.step;
    Printf.fprintf oc "Action: %s\n" frame.action;
    Printf.fprintf oc "Reward: %.2f\n" frame.reward;
    (match frame.value with
     | Some v -> Printf.fprintf oc "Value: %.2f\n" v
     | None -> ());
    Printf.fprintf oc "%s\n" frame.state_repr
  ) log.frames;
  
  close_out oc

let load_episode_log filename =
  let ic = open_in filename in
  let rec read_lines acc =
    try
      let line = input_line ic in
      read_lines (line :: acc)
    with End_of_file ->
      close_in ic;
      List.rev acc
  in
  let lines = read_lines [] in
  
  (* Simple parser - just extract key information *)
  let episode_num = ref 0 in
  let stage = ref None in
  let won = ref false in
  let total_reward = ref 0.0 in
  let total_steps = ref 0 in
  
  List.iter (fun line ->
    if String.starts_with ~prefix:"Episode " line then
      Scanf.sscanf line "Episode %d" (fun n -> episode_num := n)
    else if String.starts_with ~prefix:"Stage: " line then
      stage := Some (String.sub line 7 (String.length line - 7))
    else if String.starts_with ~prefix:"Won: " line then
      won := String.ends_with ~suffix:"true" line
    else if String.starts_with ~prefix:"Total Reward: " line then
      Scanf.sscanf line "Total Reward: %f" (fun r -> total_reward := r)
    else if String.starts_with ~prefix:"Total Steps: " line then
      Scanf.sscanf line "Total Steps: %d" (fun s -> total_steps := s)
  ) lines;
  
  {
    episode_num = !episode_num;
    total_reward = !total_reward;
    total_steps = !total_steps;
    won = !won;
    frames = [];  (* Simplified - not parsing frames *)
    stage = !stage;
  }

let print_training_progress episode avg_reward win_rate stage =
  Printf.printf "Episode %d: Avg Reward = %.2f, Win Rate = %.1f%%, Stage = %s\n"
    episode avg_reward (win_rate *. 100.0) 
    (Option.value stage ~default:"N/A")

let visualize_multiple_episodes logs delay =
  List.iter (fun log -> animate_episode log delay) logs

let summary_stats logs =
  let total = List.length logs in
  let wins = List.filter (fun l -> l.won) logs |> List.length in
  let total_reward = List.fold_left (fun acc l -> acc +. l.total_reward) 0.0 logs in
  let avg_steps = 
    float_of_int (List.fold_left (fun acc l -> acc + l.total_steps) 0 logs) /. 
    float_of_int total in
  
  Printf.printf "\n=== Training Summary ===\n";
  Printf.printf "Total Episodes: %d\n" total;
  Printf.printf "Wins: %d (%.1f%%)\n" wins (float_of_int wins /. float_of_int total *. 100.0);
  Printf.printf "Average Reward: %.2f\n" (total_reward /. float_of_int total);
  Printf.printf "Average Steps: %.1f\n" avg_steps