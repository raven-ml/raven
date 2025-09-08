let () =
  Random.self_init();
  Printf.printf "Testing room generation solvability...\n\n";
  
  let test_size = 5 in  (* 5x5 room = 3x3 playable area *)
  let num_tests = 100 in
  let solvable_count = ref 0 in
  let corner_box_count = ref 0 in
  let corner_target_count = ref 0 in
  let both_corner_count = ref 0 in
  
  for i = 1 to num_tests do
    let state = Sokoban.LevelGen.generate_room test_size in
    
    (* Check solvability *)
    if Sokoban.Core.is_potentially_solvable state then
      incr solvable_count;
    
    (* Analyze placement *)
    let box_in_corner = ref false in
    let target_in_corner = ref false in
    
    for y = 1 to test_size - 2 do
      for x = 1 to test_size - 2 do
        match state.grid.(y).(x) with
        | Box -> 
          if Sokoban.Core.is_corner (x, y) state then begin
            box_in_corner := true;
            incr corner_box_count
          end
        | Target ->
          if Sokoban.Core.is_corner (x, y) state then begin
            target_in_corner := true;
            incr corner_target_count
          end
        | _ -> ()
      done
    done;
    
    if !box_in_corner && !target_in_corner then
      incr both_corner_count;
    
    (* Show a few examples *)
    if i <= 3 || (i > num_tests - 2) then begin
      Printf.printf "Room %d (Solvable: %b):\n" i 
        (Sokoban.Core.is_potentially_solvable state);
      Sokoban.Core.render state;
      
      if not (Sokoban.Core.is_potentially_solvable state) then begin
        Printf.printf "  UNSOLVABLE: ";
        if Sokoban.Core.has_deadlock state then
          Printf.printf "Has deadlock"
        else
          Printf.printf "Other issue";
        Printf.printf "\n"
      end;
      
      if !box_in_corner then
        Printf.printf "  Box in corner\n";
      if !target_in_corner then
        Printf.printf "  Target in corner\n";
      Printf.printf "\n"
    end
  done;
  
  Printf.printf "\nStatistics over %d rooms:\n" num_tests;
  Printf.printf "  Solvable: %d (%.1f%%)\n" !solvable_count
    (float_of_int !solvable_count /. float_of_int num_tests *. 100.0);
  Printf.printf "  Box in corner: %d\n" !corner_box_count;
  Printf.printf "  Target in corner: %d\n" !corner_target_count;
  Printf.printf "  Both in corners: %d\n" !both_corner_count;
  
  (* Test corner detection *)
  Printf.printf "\n\nTesting corner detection in 5x5 room:\n";
  let test_state = Sokoban.LevelGen.make_empty 5 5 in
  Sokoban.LevelGen.add_walls test_state;
  
  let corners = [(1,1); (3,1); (1,3); (3,3)] in
  let non_corners = [(2,1); (1,2); (2,2); (3,2); (2,3)] in
  
  Printf.printf "Corners: ";
  List.iter (fun (x,y) ->
    Printf.printf "(%d,%d)=%b " x y (Sokoban.Core.is_corner (x,y) test_state)
  ) corners;
  Printf.printf "\n";
  
  Printf.printf "Non-corners: ";
  List.iter (fun (x,y) ->
    Printf.printf "(%d,%d)=%b " x y (Sokoban.Core.is_corner (x,y) test_state)
  ) non_corners;
  Printf.printf "\n"