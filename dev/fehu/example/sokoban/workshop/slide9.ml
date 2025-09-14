(*
```ocaml
 *)
open Slide1
open Slide2
open Slide3
open Slide4  (* For training_history type *)
open Slide7
open Slide8

let copy_params params =
  Kaun.Ptree.map Rune.copy params

(* REINFORCE++ with baseline, clipping, and KL penalty *)
let train_reinforce_plus_plus env n_episodes learning_rate gamma
    epsilon beta ?(grid_size=5) () =
  let policy_net, params = initialize_policy ~grid_size () in
  let optimizer = Kaun.Optimizer.adam ~lr:learning_rate () in
  let opt_state = ref (optimizer.init params) in

  (* History tracking *)
  let history_returns = Array.make n_episodes 0.0 in
  let history_losses = Array.make n_episodes 0.0 in

  (* Collect episodes for visualization *)
  let collected_episodes = ref [] in

  (* Baseline for variance reduction *)
  let baseline = ref 0.0 in
  (* Exponential moving average factor *)
  let baseline_alpha = 0.01 in

  (* Store old policy for computing ratios *)
  let old_params = ref (copy_params params) in

  for episode = 1 to n_episodes do
    (* Collect episode with current policy *)
    let episode_data =
      collect_episode env policy_net params 100 in

    (* Store selected episodes *)
    if episode mod (n_episodes / 10) = 0 || episode = n_episodes then
      collected_episodes := episode_data :: !collected_episodes;
    let returns = compute_returns episode_data.rewards gamma in

    (* Update baseline (exponential moving average) *)
    let episode_return = 
      if Array.length returns > 0 then returns.(0) else 0.0 in
    baseline := !baseline *. (1.0 -. baseline_alpha) +.
                episode_return *. baseline_alpha;

    (* Compute advantages (returns - baseline) *)
    let advantages =
      Array.map (fun r -> r -. !baseline) returns in

    (* Compute old log probs for clipping *)
    let old_log_probs = ref [] in
    let n_states = min 10 (Array.length advantages) in
    for t = 0 to n_states - 1 do
      let state = episode_data.states.(t) in
      let state_batched = Rune.reshape [|1; grid_size; grid_size|] state in
      let old_logits =
        Kaun.apply policy_net !old_params ~training:false
          state_batched in
      let old_log_prob_dist = log_softmax ~axis:(-1) old_logits in
      let action_int =
        int_of_float (Rune.item [] episode_data.actions.(t)) in
      (* Ensure action is in bounds *)
      let action_int = max 0 (min 3 action_int) in
      let old_action_log_prob =
        Rune.item [0; action_int] old_log_prob_dist in
      old_log_probs := old_action_log_prob :: !old_log_probs
    done;
    let old_log_probs_array =
      Array.of_list (List.rev !old_log_probs) in

    (* Compute policy gradient with clipping and KL penalty *)
    let loss, grads = Kaun.value_and_grad (fun p ->
      let total_loss = ref (Rune.scalar device Rune.float32 0.0) in
      let total_kl = ref (Rune.scalar device Rune.float32 0.0) in

      (* Process first 10 states due to autodiff limitations *)
      let n_samples = min 10 (Array.length advantages) in
      for t = 0 to n_samples - 1 do
        let state = episode_data.states.(t) in
        let action = episode_data.actions.(t) in
        let advantage = advantages.(t) in

        (* Get current policy probabilities *)
        let state_batched = Rune.reshape [|1; grid_size; grid_size|] state in
        let logits =
          Kaun.apply policy_net p ~training:true state_batched in
        let log_probs = log_softmax ~axis:(-1) logits in
        let probs = Rune.exp log_probs in

        (* Get action log prob - stay on device using one-hot *)
        let action_int_tensor = Rune.astype Rune.int32 action in
        let action_one_hot =
          Rune.one_hot ~num_classes:4 action_int_tensor in
        let action_one_hot =
          Rune.reshape [|1; 4|] action_one_hot |>
          Rune.astype Rune.float32 in
        let new_log_prob =
          Rune.sum (Rune.mul action_one_hot log_probs) in
        let old_log_prob =
          Rune.scalar device Rune.float32 old_log_probs_array.(t) in

        (* Compute ratio with clipping *)
        let log_ratio = Rune.sub new_log_prob old_log_prob in
        let ratio = Rune.exp log_ratio in
        let clipped_ratio = clip_ratio ratio epsilon in

        (* Compute objectives with advantage *)
        let advantage_tensor =
          Rune.scalar device Rune.float32 advantage in
        let obj1 = Rune.mul ratio advantage_tensor in
        let obj2 = Rune.mul clipped_ratio advantage_tensor in
        let clipped_obj = Rune.minimum obj1 obj2 in

        (* Add to loss (negative for gradient descent) *)
        total_loss := Rune.sub !total_loss clipped_obj;

        (* Compute KL penalty between old and new distributions *)
        let old_logits =
          Kaun.apply policy_net !old_params ~training:false state_batched in
        let old_probs = Rune.exp (log_softmax ~axis:(-1) old_logits) in
        let kl_div = compute_kl_divergence old_probs probs in
        total_kl := Rune.add !total_kl kl_div
      done;

      (* Average loss and add KL penalty *)
      let avg_loss =
        Rune.div !total_loss
          (Rune.scalar device Rune.float32 (float_of_int n_samples)) in
      let kl_penalty =
        Rune.mul (Rune.scalar device Rune.float32 beta)
                 (Rune.div !total_kl
                  (Rune.scalar device Rune.float32
                     (float_of_int n_samples))) in

      Rune.add avg_loss kl_penalty
    ) params in

    (* Update parameters *)
    let updates, new_state =
      optimizer.update !opt_state params grads in
    opt_state := new_state;
    Kaun.Optimizer.apply_updates_inplace params updates;

    (* Periodically update old policy *)
    if episode mod 3 = 0 then
      old_params := copy_params params;

    (* Track history *)
    let total_reward =
      Array.fold_left (+.) 0. episode_data.rewards in
    history_returns.(episode - 1) <- total_reward;
    history_losses.(episode - 1) <- Rune.item [] loss;

    (* Log progress *)
    if episode mod 10 = 0 then
      Printf.printf
        "Episode %d: Return = %.2f, Baseline = %.2f, Loss = %.4f\n"
        episode episode_return !baseline (Rune.item [] loss)
  done;

  (* Return with history *)
  (policy_net, params,
   {returns = history_returns; losses = history_losses; collected_episodes = List.rev !collected_episodes})

(* Main function to test REINFORCE++ *)
let main () =
  print_endline "=== Slide 9: REINFORCE++ ===";
  print_endline "Features: baseline, ratio clipping, KL penalty";
  let env = create_simple_gridworld 5 in

  (* Train with clipping (epsilon=0.2) and KL penalty (beta=0.01) *)
  let _policy_net, _params, _history =
    train_reinforce_plus_plus env 50 0.01 0.99 0.2 0.01 () in

  print_endline "REINFORCE++ training complete!"

(*
```
 *)
