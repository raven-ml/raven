(*
```ocaml
 *)
include Slide2
(* Collect a complete episode using our policy *)
let collect_episode env policy_net params max_steps =
  let rng = Rune.Rng.key (Random.int 1000000) in  
  (* Storage for episode data *)
  let states = ref [] in
  let actions = ref [] in
  let rewards = ref [] in
  let log_probs = ref [] in  
  (* Reset environment *)
  let obs, _ = env.Envs.reset () in
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
        env.step action in      
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
(*
```
 *)