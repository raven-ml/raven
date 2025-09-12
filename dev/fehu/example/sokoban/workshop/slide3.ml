(*
```ocaml
 *)
open Slide1
open Slide2

(* Episode data type *)
type episode_data = {
  states : (float, Rune.float32_elt, [`c]) Rune.t array;
  actions : (float, Rune.float32_elt, [`c]) Rune.t array;
  rewards : float array;
  log_probs : (float, Rune.float32_elt, [`c]) Rune.t array;
}
(* Collect a complete episode using our policy *)
let collect_episode env policy_net params max_steps =
  let rng = Rune.Rng.key (Random.int 1000000) in  
  (* Storage for episode data *)
  let states = ref [] in
  let actions = ref [] in
  let rewards = ref [] in
  let log_probs = ref [] in  
  (* Reset environment *)
  let obs, _ = env.Fehu.Env.reset () in
  states := obs :: !states;  
  (* Run episode *)
  let rec run_steps step obs =
    if step >= max_steps then ()
    else begin
      (* Get action from policy *)
      let action, log_prob =
        sample_action policy_net params obs rng in      
      (* Take environment step *)
      let next_obs, reward, terminated, truncated, _ =
        env.Fehu.Env.step action in      
      (* Store transition *)
      actions := action :: !actions;
      rewards := reward :: !rewards;
      log_probs := log_prob :: !log_probs;      
      if not terminated && not truncated then begin
        states := next_obs :: !states;
        run_steps (step + 1) next_obs
      end
    end
  in  
  run_steps 0 obs;  
  (* Return episode data *)
  {
    states = Array.of_list (List.rev !states);
    actions = Array.of_list (List.rev !actions);
    rewards = Array.of_list (List.rev !rewards);
    log_probs = Array.of_list (List.rev !log_probs);
  }
(* Compute returns from rewards *)
let compute_returns rewards gamma =
  let n = Array.length rewards in
  let returns = Array.make n 0.0 in  
  (* Work backwards to compute cumulative discounted returns *)
  for i = n - 1 downto 0 do
    if i = n - 1 then
      returns.(i) <- rewards.(i)
    else
      returns.(i) <- rewards.(i) +. gamma *. returns.(i + 1)
  done;  
  returns
(* Main function to test episode collection *)
let main () =
  print_endline "=== Slide 3: Episode Collection ===";
  let env = create_simple_gridworld 5 in
  let policy_net, params = initialize_policy () in
  
  (* Collect a single episode *)
  let episode = collect_episode env policy_net params 50 in
  
  Printf.printf "Episode collected:\n";
  Printf.printf "  States: %d\n" (Array.length episode.states);
  Printf.printf "  Actions: %d\n" (Array.length episode.actions);
  Printf.printf "  Total reward: %.2f\n" 
    (Array.fold_left (+.) 0. episode.rewards);
  
  (* Compute and display returns *)
  let returns = compute_returns episode.rewards 0.99 in
  Printf.printf "  Return (G_0): %.2f\n" returns.(0);
  print_endline "Episode collection test complete!"

(*
```
 *)