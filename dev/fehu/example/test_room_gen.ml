let () =
  Random.self_init();
  Printf.printf "Testing diverse room generation...\n";
  for i = 1 to 10 do
    Printf.printf "\nRoom %d:\n" i;
    let state = Sokoban.LevelGen.generate_room 5 in
    Sokoban.Core.render state;
    Printf.printf "Solvable: %b\n" (Sokoban.Core.is_potentially_solvable state)
  done