open Workshop

let () =
  print_endline "=== Run 9: REINFORCE++ with GRPO features ===";
  print_endline "Implements ratio clipping and KL penalty";
  print_endline "This serves as foundation for full GRPO (Exercise 2)\n";

  (* Create environment *)
  let env = Slide1.create_simple_gridworld 5 in

  (* Train REINFORCE++ *)
  print_endline "Training REINFORCE++ with:";
  print_endline "- Epsilon clipping: 0.2";
  print_endline "- KL penalty beta: 0.01";
  print_endline "- Learning rate: 0.01";
  print_endline "- Discount factor: 0.99\n";

  let policy_net, params =
    Slide9.train_reinforce_plus_plus env 200 0.01 0.99 0.2 0.01 in

  (* Evaluate final policy *)
  print_endline "\n=== Evaluating final policy ===";
  let test_episodes = 10 in
  let total_returns = ref 0.0 in

  for i = 1 to test_episodes do
    let episode_data =
      Slide3.collect_episode env policy_net params 100 in
    let returns = Slide3.compute_returns episode_data.rewards 0.99 in
    let episode_return = returns.(0) in
    total_returns := !total_returns +. episode_return;
    Printf.printf "Test episode %d: Return = %.2f\n" i episode_return
  done;

  let mean_return = !total_returns /. float_of_int test_episodes in
  Printf.printf "\nMean test return: %.2f\n" mean_return;

  print_endline "\n=== REINFORCE++ complete ===";
  print_endline "Next: Exercise 2 - Extend environment API for full GRPO";
  print_endline "Challenge: Add state save/restore to enable reference policies"