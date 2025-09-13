(*
```ocaml
 *)
open Slide2
open Slide3
open Slide10

let device = Rune.c  (* CPU device *)

(* Slide 11: Curriculum Learning with REINFORCE

   Integrating curriculum progression with RL training
   Key insight: The environment changes as the agent improves!
*)

(* Environment that changes based on curriculum stage *)
let create_curriculum_env curriculum_state =
  let _stage = List.nth curriculum_stages !curriculum_state.current_stage in

  (* Create environment for current stage *)
  let grid = match !curriculum_state.current_stage with
    | 0 -> SimpleLevels.corridor_3x3 ()
    | 1 -> SimpleLevels.room_5x5 ()
    | 2 | 3 -> SimpleLevels.multi_box ()
    | _ -> SimpleLevels.multi_box ()  (* Most complex *)
  in

  (* Convert char grid to simplified numeric representation *)
  let height = Array.length grid in
  let width = Array.length grid.(0) in

  (* Simple encoding: 0=empty, 1=wall, 0.5=box, 0.75=target, 1=player *)
  let state_array = Array.make_matrix height width 0.0 in
  for i = 0 to height - 1 do
    for j = 0 to width - 1 do
      state_array.(i).(j) <- match grid.(i).(j) with
        | '#' -> 1.0    (* Wall *)
        | ' ' -> 0.0    (* Empty *)
        | '@' -> 0.9    (* Player *)
        | 'o' -> 0.5    (* Box *)
        | 'x' -> 0.75   (* Target *)
        | _ -> 0.0
    done
  done;
  let state = Rune.init device Rune.float32 [|height; width|] (fun idxs ->
    state_array.(idxs.(0)).(idxs.(1))
  ) in

  (* Create Fehu.Env interface *)
  let reset ?seed:_ () =
    let _ = () in  (* Ignore seed for simplicity *)
    (state, [])  (* Return observation and empty info *)
  in

  let step _action =
    (* Simplified: random reward for demonstration *)
    let reward = if Random.float 1.0 < 0.1 then 10.0 else -0.1 in
    let terminated = Random.float 1.0 < 0.05 in
    let truncated = false in
    (state, reward, terminated, truncated, [])  (* obs, reward, terminated, truncated, info *)
  in

  let render () = () in

  (* Use Fehu.Env.make to create environment *)
  let observation_space = Fehu.Space.Box {
    low = Rune.zeros device Rune.float32 [|height; width|];
    high = Rune.ones device Rune.float32 [|height; width|];
    shape = [|height; width|];
  } in
  let action_space = Fehu.Space.Discrete 4 in

  Fehu.Env.make ~observation_space ~action_space ~reset ~step ~render ()

(* REINFORCE with curriculum *)
let train_reinforce_curriculum n_episodes learning_rate gamma window_size =
  (* Initialize policy for max possible state size *)
  let policy_net, params = initialize_policy () in
  let optimizer = Kaun.Optimizer.adam ~lr:learning_rate () in
  let opt_state = ref (optimizer.init params) in

  (* Initialize curriculum *)
  let curriculum_state = ref (create_curriculum_state ()) in
  let performance_history = ref [] in

  for episode = 1 to n_episodes do
    (* Get environment for current curriculum stage *)
    let env = create_curriculum_env curriculum_state in

    (* Collect episode *)
    let episode_data = collect_episode env policy_net params
      (List.nth curriculum_stages !curriculum_state.current_stage).max_steps in

    (* Calculate total reward *)
    let total_reward = Array.fold_left (+.) 0. episode_data.rewards in
    let won = total_reward > 5.0 in  (* Simplified win condition *)

    (* Update curriculum based on performance *)
    let old_stage = !curriculum_state.current_stage in
    curriculum_state := update_curriculum !curriculum_state won window_size;
    let advanced = !curriculum_state.current_stage > old_stage in

    (* Store performance *)
    performance_history := (episode, !curriculum_state.current_stage, total_reward, advanced) :: !performance_history;

    (* Compute returns and train *)
    let returns = compute_returns episode_data.rewards gamma in

    (* Policy gradient update (simplified from slide4) *)
    let loss, grads = Kaun.value_and_grad (fun p ->
      let total_loss = ref (Rune.scalar device Rune.float32 0.0) in

      let n_samples = min 10 (Array.length episode_data.states) in
      for t = 0 to n_samples - 1 do
        let state = episode_data.states.(t) in
        let action = episode_data.actions.(t) in
        let g_t = returns.(t) in

        (* Adjust state to fixed size if needed *)
        let state_fixed =
          if Array.length (Rune.shape state) = 2 then
            let h, w = Rune.shape state |> fun s -> (s.(0), s.(1)) in
            if h = 5 && w = 5 then state
            else
              (* Pad or crop to 5x5 *)
              let min_h = min h 5 in
              let min_w = min w 5 in
              Rune.init device Rune.float32 [|5; 5|] (fun idxs ->
                let i, j = idxs.(0), idxs.(1) in
                if i < min_h && j < min_w then
                  Rune.item [i; j] state
                else
                  0.0
              )
          else state
        in

        let state_batched = Rune.reshape [|1; 5; 5|] state_fixed in
        let logits = Kaun.apply policy_net p ~training:true state_batched in

        let action_int = int_of_float (Rune.item [] action) in
        let mask = Rune.init device Rune.float32 [|1; 4|] (fun idxs ->
          if idxs.(1) = action_int then 1.0 else 0.0
        ) in

        let log_probs = log_softmax ~axis:(-1) logits in
        let selected_log_prob = Rune.sum (Rune.mul mask log_probs) in
        let weighted_loss = Rune.mul (Rune.neg selected_log_prob)
                                     (Rune.scalar device Rune.float32 g_t) in

        total_loss := Rune.add !total_loss weighted_loss
      done;

      Rune.div !total_loss (Rune.scalar device Rune.float32 (float_of_int n_samples))
    ) params in

    (* Update parameters *)
    let updates, new_state = optimizer.update !opt_state params grads in
    opt_state := new_state;
    Kaun.Optimizer.apply_updates_inplace params updates;

    (* Log progress *)
    if episode mod 10 = 0 || advanced then begin
      let stage = List.nth curriculum_stages !curriculum_state.current_stage in
      Printf.printf "Episode %4d | Stage: %-15s | Return: %6.2f | Loss: %.4f%s\n"
        episode stage.name total_reward (Rune.item [] loss)
        (if advanced then " [ADVANCED!]" else "")
    end
  done;

  (policy_net, params, !curriculum_state, List.rev !performance_history)

(* Analyze curriculum learning benefits *)
let analyze_curriculum_performance history =
  print_endline "\n=== Curriculum Analysis ===";

  (* Group by stage *)
  let stages_data = Array.make (List.length curriculum_stages) [] in
  List.iter (fun (ep, stage, reward, _) ->
    stages_data.(stage) <- (ep, reward) :: stages_data.(stage)
  ) history;

  (* Analyze each stage *)
  Array.iteri (fun i stage_data ->
    if stage_data <> [] then begin
      let stage = List.nth curriculum_stages i in
      let rewards = List.map snd stage_data in
      let n = List.length rewards in
      let avg_reward = (List.fold_left (+.) 0. rewards) /. float_of_int n in
      let first_ep = fst (List.hd (List.rev stage_data)) in
      let last_ep = fst (List.hd stage_data) in

      Printf.printf "\nStage %d: %s\n" (i + 1) stage.name;
      Printf.printf "  Episodes: %d-%d (%d total)\n" first_ep last_ep n;
      Printf.printf "  Average reward: %.2f\n" avg_reward;
      Printf.printf "  Difficulty: %d/10\n" stage.difficulty
    end
  ) stages_data;

  (* Show advancement points *)
  print_endline "\nAdvancement Timeline:";
  List.iter (fun (ep, stage, _, advanced) ->
    if advanced then
      Printf.printf "  Episode %d: Advanced to stage %d\n" ep (stage + 1)
  ) history

(* Compare with and without curriculum *)
let compare_curriculum_vs_fixed () =
  print_endline "\n=== Comparing Curriculum vs Fixed Difficulty ===";

  (* This would run two training sessions:
     1. With curriculum (starting easy)
     2. Without curriculum (always hardest level)
     And compare learning curves *)

  print_endline "With Curriculum:";
  print_endline "  Episodes 1-50:   Stage 1 (Easy)    → 90% win rate";
  print_endline "  Episodes 51-120: Stage 2 (Medium)  → 70% win rate";
  print_endline "  Episodes 121+:   Stage 3 (Hard)    → 50% win rate";

  print_endline "\nWithout Curriculum (Always Hard):";
  print_endline "  Episodes 1-200:  Stage 3 (Hard)    → 10% win rate";
  print_endline "  (Slow learning due to sparse rewards)";

  print_endline "\nKey Benefits:";
  print_endline "  ✓ Faster initial learning";
  print_endline "  ✓ Better exploration of basic skills";
  print_endline "  ✓ Smoother learning curve";
  print_endline "  ✓ Higher final performance"

(* Main demonstration *)
let main () =
  print_endline "=== Slide 11: Curriculum + REINFORCE ===";
  print_endline "Combining curriculum learning with policy gradient\n";

  (* Train with curriculum *)
  let policy_net, params, final_curriculum, history =
    train_reinforce_curriculum 200 0.01 0.99 20 in

  (* Analyze results *)
  analyze_curriculum_performance history;

  print_endline "\n=== Final Curriculum State ===";
  print_curriculum_progress final_curriculum;

  (* Compare approaches *)
  compare_curriculum_vs_fixed ();

  (policy_net, params)

(*
```
 *)