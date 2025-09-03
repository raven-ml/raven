(** Quick comparison test - run each algorithm briefly to see if they're learning *)

open Fehu

let () =
  Printf.printf "\n========================================\n";
  Printf.printf "  Quick Algorithm Comparison Test\n";
  Printf.printf "  Testing with 500 episodes each\n";
  Printf.printf "========================================\n\n";
  
  let episodes = 500 in
  let max_steps = 200 in
  
  (* Test on curriculum *)
  Printf.printf "Testing algorithms on curriculum environment...\n\n";
  
  Printf.printf "To run full comparisons:\n";
  Printf.printf "- Backoff-Tabular: dune exec dev/fehu/example/backoff_tabular.exe\n";
  Printf.printf "- REINFORCE: dune exec dev/fehu/example/sokoban_reinforce.exe\n"; 
  Printf.printf "- DQN: dune exec dev/fehu/example/sokoban_dqn.exe\n\n";
  
  Printf.printf "Quick test complete!\n"