(*
```ocaml
 *)
open Slide1
open Slide2
open Slide3
open Slide7
open Slide8

(* Helper to copy parameters *)
let copy_params params =
  Kaun.Ptree.map Rune.copy params

(* REINFORCE++ with GRPO features: clipping and KL penalty *)
let train_reinforce_plus_plus env n_episodes learning_rate gamma
    epsilon beta =
  let policy_net, params = initialize_policy () in
  let optimizer = Kaun.Optimizer.adam ~lr:learning_rate () in
  let opt_state = ref (optimizer.init params) in

  (* Store old policy for computing ratios *)
  let old_params = ref (copy_params params) in

  for episode = 1 to n_episodes do
    (* Collect episode with current policy *)
    let episode_data =
      collect_episode env policy_net params 100 in
    let returns = compute_returns episode_data.rewards gamma in

    (* Compute old log probs for clipping *)
    let old_log_probs = ref [] in
    let n_states = min 10 (Array.length episode_data.states) in
    for t = 0 to n_states - 1 do
      let state = episode_data.states.(t) in
      let state_batched = Rune.reshape [|1; 5; 5|] state in
      let old_logits =
        Kaun.apply policy_net !old_params ~training:false state_batched in
      let old_log_prob_dist = log_softmax ~axis:(-1) old_logits in
      let action_int = int_of_float (Rune.item [] episode_data.actions.(t)) in
      (* Ensure action is in bounds *)
      let action_int = max 0 (min 3 action_int) in
      let old_action_log_prob = Rune.item [0; action_int] old_log_prob_dist in
      old_log_probs := old_action_log_prob :: !old_log_probs
    done;
    let old_log_probs_array = Array.of_list (List.rev !old_log_probs) in

    (* Compute policy gradient with clipping and KL penalty *)
    let loss, grads = Kaun.value_and_grad (fun p ->
      let total_loss = ref (Rune.scalar device Rune.float32 0.0) in
      let total_kl = ref (Rune.scalar device Rune.float32 0.0) in

      (* Process first 10 states due to autodiff limitations *)
      let n_samples = min 10 (Array.length episode_data.states) in
      for t = 0 to n_samples - 1 do
        let state = episode_data.states.(t) in
        let action = episode_data.actions.(t) in
        let g_t = returns.(t) in

        (* Get current policy probabilities *)
        let state_batched = Rune.reshape [|1; 5; 5|] state in
        let logits = Kaun.apply policy_net p ~training:true state_batched in
        let log_probs = log_softmax ~axis:(-1) logits in
        let probs = Rune.exp log_probs in

        (* Get action log prob *)
        let action_int = int_of_float (Rune.item [] action) in
        let mask = Rune.init device Rune.float32 [|1; 4|] (fun idxs ->
          if idxs.(1) = action_int then 1.0 else 0.0
        ) in
        let new_log_prob = Rune.sum (Rune.mul mask log_probs) in
        let old_log_prob =
          Rune.scalar device Rune.float32 old_log_probs_array.(t) in

        (* Compute ratio with clipping *)
        let log_ratio = Rune.sub new_log_prob old_log_prob in
        let ratio = Rune.exp log_ratio in
        let clipped_ratio = clip_ratio ratio epsilon in

        (* Compute objectives *)
        let g_t_tensor = Rune.scalar device Rune.float32 g_t in
        let obj1 = Rune.mul ratio g_t_tensor in
        let obj2 = Rune.mul clipped_ratio g_t_tensor in
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
        Rune.div !total_loss (Rune.scalar device Rune.float32 (float_of_int n_samples)) in
      let kl_penalty =
        Rune.mul (Rune.scalar device Rune.float32 beta)
                 (Rune.div !total_kl (Rune.scalar device Rune.float32 (float_of_int n_samples))) in

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

    (* Log progress *)
    if episode mod 10 = 0 then
      let total_reward =
        Array.fold_left (+.) 0. episode_data.rewards in
      Printf.printf "Episode %d: Return = %.2f, Loss = %.4f\n"
        episode total_reward (Rune.item [] loss)
  done;

  (policy_net, params)

(* Main function to test REINFORCE++ *)
let main () =
  print_endline "=== Slide 5: REINFORCE++ (with GRPO features) ===";
  print_endline "Features: ratio clipping, KL penalty";
  let env = create_simple_gridworld 5 in

  (* Train with clipping (epsilon=0.2) and KL penalty (beta=0.01) *)
  let _policy_net, _params =
    train_reinforce_plus_plus env 50 0.01 0.99 0.2 0.01 in

  print_endline "REINFORCE++ training complete!"

(*
```
 *)
