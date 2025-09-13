open Workshop

let () =
  print_endline "=== Run 11: REINFORCE with Curriculum Learning ===";
  print_endline "Training an agent with gradually increasing difficulty\n";

  (* Run curriculum training *)
  let _policy_net, _params = Slide11.main () in

  print_endline "\n=== Training Complete ===";
  print_endline "The agent learned progressively:";
  print_endline "1. Started with simple corridor pushing";
  print_endline "2. Advanced to room navigation";
  print_endline "3. Finally tackled multi-box puzzles";
  print_endline "\nThis is much more efficient than starting with hard puzzles!"