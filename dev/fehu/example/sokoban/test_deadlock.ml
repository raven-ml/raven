let test_deadlock_detection () =
  (* Test case from episode 5500: two boxes against a wall *)
  let state = Sokoban.Core.{
    grid = [|
      [| Wall; Wall; Wall; Wall; Wall; Wall |];
      [| Wall; Empty; Box; Box; Empty; Wall |];
      [| Wall; Empty; Empty; Target; Empty; Wall |];
      [| Wall; Empty; Player; Empty; Empty; Wall |];
      [| Wall; Empty; Empty; Empty; Empty; Wall |];
      [| Wall; Wall; Wall; Wall; Wall; Wall |];
    |];
    player_pos = (2, 3);
    width = 6;
    height = 6;
  } in
  
  Printf.printf "Testing deadlock detection:\n";
  Printf.printf "State:\n";
  for y = 0 to 5 do
    for x = 0 to 5 do
      let c = match state.grid.(y).(x) with
        | Sokoban.Core.Empty -> ' '
        | Sokoban.Core.Wall -> '#'
        | Sokoban.Core.Box -> '$'
        | Sokoban.Core.Target -> '.'
        | Sokoban.Core.BoxOnTarget -> '*'
        | Sokoban.Core.Player -> '@'
        | Sokoban.Core.PlayerOnTarget -> '+'
      in
      Printf.printf "%c" c
    done;
    Printf.printf "\n"
  done;
  
  (* Check if boxes are deadlocked *)
  let box1_deadlocked = Sokoban.Core.is_box_deadlocked (2, 1) state in
  let box2_deadlocked = Sokoban.Core.is_box_deadlocked (3, 1) state in
  let has_deadlock = Sokoban.Core.has_deadlock state in
  
  Printf.printf "\nBox at (2,1) deadlocked: %b\n" box1_deadlocked;
  Printf.printf "Box at (3,1) deadlocked: %b\n" box2_deadlocked;
  Printf.printf "Overall deadlock: %b\n" has_deadlock;
  Printf.printf "\nExpected: all should be true (deadlocked)\n"

let () = test_deadlock_detection ()
