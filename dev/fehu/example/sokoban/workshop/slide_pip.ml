(*
```ocaml
 *)
(* Complete workshop pipeline *)
let run_complete_workshop () =
  (* 1. Create environment *)
  let env = create_simple_gridworld 5 in  
  (* 2. Try basic REINFORCE *)
  print_endline "Training with basic REINFORCE...";
  let _, _ = train_reinforce env 50 0.01 0.99 in  
  (* 3. Add baseline for variance reduction *)
  print_endline "\nTraining with baseline...";
  let _, _ = train_reinforce_with_baseline env 50 0.01 0.99 in  
  (* 4. Use learned baseline (Actor-Critic) *)
  print_endline "\nTraining Actor-Critic...";
  let _, _, _, _ = train_actor_critic env 50 0.01 0.005 0.99 in  
  (* 5. Advanced: GRPO with clipping and KL *)
  print_endline "\nTraining with GRPO...";
  let _, _ = train_grpo env 50 4 0.001 0.2 0.01 in  
  print_endline "\nWorkshop complete! You've implemented:";
  print_endline "- Basic REINFORCE";
  print_endline "- REINFORCE with baseline";
  print_endline "- Actor-Critic";
  print_endline "- GRPO with clipping and KL penalties"
(* Evaluation helper *)
let evaluate_policy env policy_net params n_episodes =
  let total_returns = ref 0.0 in  
  for _ = 1 to n_episodes do
    let episode = collect_episode env policy_net params 100 in
    let returns = compute_returns episode.rewards 0.99 in
    total_returns := !total_returns +. returns.(0)
  done;  
  let mean_return = !total_returns /. float_of_int n_episodes in
  Printf.printf 
    "Mean return over %d episodes: %.2f\n" n_episodes mean_return;
    mean_return
(* Compare all methods *)
let compare_methods () =
  let env = create_simple_gridworld 5 in  
  (* Train each method *)
  let methods = [
    ("REINFORCE",
     train_reinforce env 100 0.01 0.99);
    ("REINFORCE+Baseline",
     train_reinforce_with_baseline env 100 0.01 0.99);
    ("Actor-Critic", 
      let p_net, p_params, _, _ =
        train_actor_critic env 100 0.01 0.005 0.99 in
      (p_net, p_params));
    ("GRPO",
     train_grpo env 100 4 0.001 0.2 0.01);
  ] in  
  (* Evaluate each *)
  List.iter (fun (name, (net, params)) ->
    Printf.printf "\nEvaluating %s:\n" name;
    let _ = evaluate_policy env net params 20 in
    ()
  ) methods
(*
```
 *)