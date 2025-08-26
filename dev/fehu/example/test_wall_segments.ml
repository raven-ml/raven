let () =
  Printf.printf "Testing wall segment deadlock detection...\n\n";
  
  (* Test 1: Box on top wall, target on side wall (different segments) *)
  Printf.printf "Test 1: Box on TOP wall, target on SIDE wall (should be deadlocked):\n";
  let state1 = Sokoban.LevelGen.make_empty 5 5 in
  Sokoban.LevelGen.add_walls state1;
  state1.grid.(1).(2) <- Sokoban.Core.Box;      (* Box against top wall *)
  state1.grid.(2).(3) <- Sokoban.Core.Target;   (* Target against right wall *)
  state1.grid.(2).(1) <- Sokoban.Core.Player;
  let state1 = { state1 with player_pos = (1, 2) } in
  Sokoban.Core.render state1;
  Printf.printf "Has deadlock: %b\n" (Sokoban.Core.has_deadlock state1);
  Printf.printf "Is solvable: %b\n\n" (Sokoban.Core.is_potentially_solvable state1);
  
  (* Test 2: Box and target on same wall segment *)
  Printf.printf "Test 2: Box and target on SAME top wall (should be solvable):\n";
  let state2 = Sokoban.LevelGen.make_empty 5 5 in
  Sokoban.LevelGen.add_walls state2;
  state2.grid.(1).(2) <- Sokoban.Core.Box;      (* Box against top wall *)
  state2.grid.(1).(1) <- Sokoban.Core.Target;   (* Target also against top wall *)
  state2.grid.(2).(2) <- Sokoban.Core.Player;
  let state2 = { state2 with player_pos = (2, 2) } in
  Sokoban.Core.render state2;
  Printf.printf "Has deadlock: %b\n" (Sokoban.Core.has_deadlock state2);
  Printf.printf "Is solvable: %b\n\n" (Sokoban.Core.is_potentially_solvable state2);
  
  (* Test 3: Box on left wall, target on bottom wall (different segments) *)
  Printf.printf "Test 3: Box on LEFT wall, target on BOTTOM wall (should be deadlocked):\n";
  let state3 = Sokoban.LevelGen.make_empty 5 5 in
  Sokoban.LevelGen.add_walls state3;
  state3.grid.(2).(1) <- Sokoban.Core.Box;      (* Box against left wall *)
  state3.grid.(3).(2) <- Sokoban.Core.Target;   (* Target against bottom wall *)
  state3.grid.(2).(2) <- Sokoban.Core.Player;
  let state3 = { state3 with player_pos = (2, 2) } in
  Sokoban.Core.render state3;
  Printf.printf "Has deadlock: %b\n" (Sokoban.Core.has_deadlock state3);
  Printf.printf "Is solvable: %b\n\n" (Sokoban.Core.is_potentially_solvable state3);
  
  (* Test 4: Box and target both on left wall *)
  Printf.printf "Test 4: Box and target on SAME left wall (should be solvable):\n";
  let state4 = Sokoban.LevelGen.make_empty 5 5 in
  Sokoban.LevelGen.add_walls state4;
  state4.grid.(2).(1) <- Sokoban.Core.Box;      (* Box against left wall *)
  state4.grid.(3).(1) <- Sokoban.Core.Target;   (* Target also against left wall *)
  state4.grid.(1).(2) <- Sokoban.Core.Player;
  let state4 = { state4 with player_pos = (2, 1) } in
  Sokoban.Core.render state4;
  Printf.printf "Has deadlock: %b\n" (Sokoban.Core.has_deadlock state4);
  Printf.printf "Is solvable: %b\n\n" (Sokoban.Core.is_potentially_solvable state4);
  
  (* Test 5: Box in corner (always deadlocked unless target is there) *)
  Printf.printf "Test 5: Box in corner, target on adjacent wall (should be deadlocked):\n";
  let state5 = Sokoban.LevelGen.make_empty 5 5 in
  Sokoban.LevelGen.add_walls state5;
  state5.grid.(1).(1) <- Sokoban.Core.Box;      (* Box in top-left corner *)
  state5.grid.(1).(2) <- Sokoban.Core.Target;   (* Target adjacent on top wall *)
  state5.grid.(2).(2) <- Sokoban.Core.Player;
  let state5 = { state5 with player_pos = (2, 2) } in
  Sokoban.Core.render state5;
  Printf.printf "Has deadlock: %b\n" (Sokoban.Core.has_deadlock state5);
  Printf.printf "Is solvable: %b\n\n" (Sokoban.Core.is_potentially_solvable state5);
  
  (* Test 6: In a larger room with wall break *)
  Printf.printf "Test 6: Larger room test - box and target on different wall segments:\n";
  let state6 = Sokoban.LevelGen.make_empty 7 7 in
  Sokoban.LevelGen.add_walls state6;
  state6.grid.(1).(3) <- Sokoban.Core.Box;      (* Box against top wall, middle *)
  state6.grid.(3).(5) <- Sokoban.Core.Target;   (* Target against right wall *)
  state6.grid.(3).(3) <- Sokoban.Core.Player;
  let state6 = { state6 with player_pos = (3, 3) } in
  Sokoban.Core.render state6;
  Printf.printf "Has deadlock: %b\n" (Sokoban.Core.has_deadlock state6);
  Printf.printf "Is solvable: %b\n" (Sokoban.Core.is_potentially_solvable state6)