open Slide3

(* Visualize an episode's trajectory *)
let visualize_episode episode_data episode_num =
  Printf.printf "\n=== Episode %d Trajectory ===\n" episode_num;
  Printf.printf "Total reward: %.2f\n" 
    (Array.fold_left (+.) 0. episode_data.rewards);
  Printf.printf "Episode length: %d steps\n\n" 
    (Array.length episode_data.actions);
  
  (* Create a 5x5 grid to show the path *)
  let grid = Array.make_matrix 5 5 ' ' in
  grid.(4).(4) <- 'G';  (* Goal at bottom-right *)
  
  (* Track agent's path *)
  let x = ref 0 in
  let y = ref 0 in
  grid.(!y).(!x) <- 'S';  (* Start position *)
  
  for i = 0 to min 20 (Array.length episode_data.actions - 1) do
    let action = int_of_float (Rune.item [] episode_data.actions.(i)) in
    (* Update position based on action *)
    let new_x, new_y = match action with
      | 0 -> !x, max 0 (!y - 1)        (* Up *)
      | 1 -> !x, min 4 (!y + 1)        (* Down *)
      | 2 -> max 0 (!x - 1), !y        (* Left *)
      | 3 -> min 4 (!x + 1), !y        (* Right *)
      | _ -> !x, !y
    in
    
    (* Mark the path *)
    if grid.(new_y).(new_x) = ' ' then
      grid.(new_y).(new_x) <- '.';
    
    x := new_x;
    y := new_y;
  done;
  
  (* Mark final position if not at goal *)
  if (!x, !y) <> (4, 4) && grid.(!y).(!x) = '.' then
    grid.(!y).(!x) <- 'A';
  
  (* Print the grid *)
  print_endline "Grid (S=start, G=goal, .=path, A=agent final pos):";
  print_endline "+-----+";
  for row = 0 to 4 do
    print_char '|';
    for col = 0 to 4 do
      print_char grid.(row).(col)
    done;
    print_endline "|"
  done;
  print_endline "+-----+";
  
  (* Print first few actions *)
  Printf.printf "\nFirst 10 actions: ";
  for i = 0 to min 9 (Array.length episode_data.actions - 1) do
    let action = int_of_float (Rune.item [] episode_data.actions.(i)) in
    let action_str = match action with
      | 0 -> "↑" | 1 -> "↓" | 2 -> "←" | 3 -> "→" | _ -> "?"
    in
    Printf.printf "%s " action_str
  done;
  print_endline "\n"

