(* Run 4: Basic REINFORCE training with visualization *)
open Workshop.Slide1
open Workshop.Slide4

let () =
  print_endline "\n==== Workshop Run 4: Basic REINFORCE Training with Visualization ====\n";
  
  (* Create environment *)
  let env = create_simple_gridworld 5 in
  
  (* Train the agent and get collected episodes *)
  let _policy_net, _params, episodes, _history =
    train_reinforce env 100 0.01 0.99 () in
  
  (* Visualize first and last episodes *)
  (match episodes with
  | [] -> print_endline "No episodes collected!"
  | [first] -> Workshop.Helpers.visualize_episode first 1
  | first :: second :: rest ->
      Workshop.Helpers.visualize_episode first 1;
      Workshop.Helpers.visualize_episode second 2;
      (match List.rev rest with
       | [] -> ()
       | last :: _ -> Workshop.Helpers.visualize_episode last 100));
  
  print_endline "\nTraining complete!";
  print_endline "\n==== Run 4 Complete ====\n"