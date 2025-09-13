(*
```ocaml
 *)
open Slide1
open Slide2
open Slide3

let device = Rune.c  (* CPU device *)

(* Slide 12: Visualization and Algorithm Comparison

   Compare different REINFORCE variants with loss and return plots
*)

(* Store training history for multiple algorithms *)
type training_history = {
  name: string;
  returns: float array;
  losses: float array;
  color: string;  (* For plotting *)
}

(* Modified training functions that return history *)
let train_reinforce_with_history env n_episodes learning_rate gamma =
  let policy_net, params = initialize_policy () in
  let optimizer = Kaun.Optimizer.adam ~lr:learning_rate () in
  let opt_state = ref (optimizer.init params) in

  let returns = Array.make n_episodes 0.0 in
  let losses = Array.make n_episodes 0.0 in

  for episode = 0 to n_episodes - 1 do
    let episode_data = collect_episode env policy_net params 100 in
    let episode_returns = compute_returns episode_data.rewards gamma in
    let total_reward = Array.fold_left (+.) 0. episode_data.rewards in
    returns.(episode) <- total_reward;

    (* Compute policy gradient loss *)
    let loss, grads = Kaun.value_and_grad (fun p ->
      let total_loss = ref (Rune.scalar device Rune.float32 0.0) in

      let n_samples = min 30 (Array.length episode_data.states) in
      for t = 0 to n_samples - 1 do
        let state = episode_data.states.(t) in
        let action = episode_data.actions.(t) in
        let g_t = episode_returns.(t) in

        let state_batched = Rune.reshape [|1; 5; 5|] state in
        let logits = Kaun.apply policy_net p ~training:true state_batched in
        let log_probs = log_softmax ~axis:(-1) logits in

        (* Use one-hot encoding for action selection *)
        let action_int_tensor = Rune.astype Rune.int32 action in
        let action_one_hot = Rune.one_hot ~num_classes:4 action_int_tensor in
        let action_one_hot =
          Rune.reshape [|1; 4|] action_one_hot |>
          Rune.astype Rune.float32 in
        let selected_log_prob = Rune.sum (Rune.mul action_one_hot log_probs) in

        let weighted_loss = Rune.mul (Rune.neg selected_log_prob)
                                     (Rune.scalar device Rune.float32 g_t) in
        total_loss := Rune.add !total_loss weighted_loss
      done;

      Rune.div !total_loss (Rune.scalar device Rune.float32 (float_of_int n_samples))
    ) params in

    losses.(episode) <- Rune.item [] loss;

    (* Update parameters *)
    let updates, new_state = optimizer.update !opt_state params grads in
    opt_state := new_state;
    Kaun.Optimizer.apply_updates_inplace params updates;
  done;

  { name = "REINFORCE (no baseline)";
    returns = returns;
    losses = losses;
    color = "blue" }

let train_reinforce_baseline_with_history env n_episodes learning_rate gamma =
  let policy_net, params = initialize_policy () in
  let optimizer = Kaun.Optimizer.adam ~lr:learning_rate () in
  let opt_state = ref (optimizer.init params) in

  let returns = Array.make n_episodes 0.0 in
  let losses = Array.make n_episodes 0.0 in
  let baseline = ref 0.0 in  (* Running average baseline *)
  let alpha = 0.1 in  (* Baseline learning rate *)

  for episode = 0 to n_episodes - 1 do
    let episode_data = collect_episode env policy_net params 100 in
    let episode_returns = compute_returns episode_data.rewards gamma in
    let total_reward = Array.fold_left (+.) 0. episode_data.rewards in
    returns.(episode) <- total_reward;

    (* Update baseline with exponential moving average *)
    baseline := (1.0 -. alpha) *. !baseline +. alpha *. total_reward;

    (* Compute advantages *)
    let advantages = Array.map (fun g -> g -. !baseline) episode_returns in

    (* Compute policy gradient loss with baseline *)
    let loss, grads = Kaun.value_and_grad (fun p ->
      let total_loss = ref (Rune.scalar device Rune.float32 0.0) in

      let n_samples = min 30 (Array.length episode_data.states) in
      for t = 0 to n_samples - 1 do
        let state = episode_data.states.(t) in
        let action = episode_data.actions.(t) in
        let advantage = advantages.(t) in

        let state_batched = Rune.reshape [|1; 5; 5|] state in
        let logits = Kaun.apply policy_net p ~training:true state_batched in
        let log_probs = log_softmax ~axis:(-1) logits in

        (* Use one-hot encoding for action selection *)
        let action_int_tensor = Rune.astype Rune.int32 action in
        let action_one_hot = Rune.one_hot ~num_classes:4 action_int_tensor in
        let action_one_hot =
          Rune.reshape [|1; 4|] action_one_hot |>
          Rune.astype Rune.float32 in
        let selected_log_prob = Rune.sum (Rune.mul action_one_hot log_probs) in

        let weighted_loss = Rune.mul (Rune.neg selected_log_prob)
                                     (Rune.scalar device Rune.float32 advantage) in
        total_loss := Rune.add !total_loss weighted_loss
      done;

      Rune.div !total_loss (Rune.scalar device Rune.float32 (float_of_int n_samples))
    ) params in

    losses.(episode) <- Rune.item [] loss;

    (* Update parameters *)
    let updates, new_state = optimizer.update !opt_state params grads in
    opt_state := new_state;
    Kaun.Optimizer.apply_updates_inplace params updates;
  done;

  { name = "REINFORCE with baseline";
    returns = returns;
    losses = losses;
    color = "green" }

(* Compute moving average for smoothing plots *)
let moving_average data window_size =
  let n = Array.length data in
  let smoothed = Array.make n 0.0 in
  for i = 0 to n - 1 do
    let start_idx = max 0 (i - window_size / 2) in
    let end_idx = min (n - 1) (i + window_size / 2) in
    let sum = ref 0.0 in
    let count = ref 0 in
    for j = start_idx to end_idx do
      sum := !sum +. data.(j);
      incr count
    done;
    smoothed.(i) <- !sum /. float_of_int !count
  done;
  smoothed

(* ASCII plot for terminal output *)
let ascii_plot histories metric_name window_size =
  let height = 20 in
  let width = 60 in

  (* Get the data for the specified metric *)
  let get_data h = match metric_name with
    | "returns" -> h.returns
    | "losses" -> h.losses
    | _ -> failwith "Unknown metric"
  in

  (* Find global min/max across all histories *)
  let all_values = List.concat_map (fun h ->
    Array.to_list (moving_average (get_data h) window_size)
  ) histories in
  let min_val = List.fold_left min max_float all_values in
  let max_val = List.fold_left max min_float all_values in
  let range = max_val -. min_val in

  (* Create empty plot *)
  let plot = Array.make_matrix height width ' ' in

  (* Plot each history *)
  List.iteri (fun hist_idx h ->
    let data = moving_average (get_data h) window_size in
    let n_episodes = Array.length data in
    let char = match hist_idx with
      | 0 -> '*'
      | 1 -> '+'
      | 2 -> 'o'
      | _ -> '.'
    in

    for i = 0 to n_episodes - 1 do
      let x = i * width / n_episodes in
      let y_norm = (data.(i) -. min_val) /. range in
      let y = height - 1 - int_of_float (y_norm *. float_of_int (height - 1)) in
      if x < width && y >= 0 && y < height then
        plot.(y).(x) <- char
    done
  ) histories;

  (* Print plot *)
  Printf.printf "\n=== %s (smoothed with window=%d) ===\n"
    (String.capitalize_ascii metric_name) window_size;
  Printf.printf "Max: %.2f\n" max_val;

  (* Print plot grid *)
  for y = 0 to height - 1 do
    Printf.printf "|";
    for x = 0 to width - 1 do
      Printf.printf "%c" plot.(y).(x)
    done;
    Printf.printf "|\n"
  done;

  Printf.printf "Min: %.2f\n" min_val;
  Printf.printf "+%s+\n" (String.make width '-');
  Printf.printf " 0%sEpisodes\n" (String.make (width - 8) ' ');

  (* Legend *)
  Printf.printf "\nLegend:\n";
  List.iteri (fun i h ->
    let char = match i with
      | 0 -> '*'
      | 1 -> '+'
      | 2 -> 'o'
      | _ -> '.'
    in
    Printf.printf "  %c - %s\n" char h.name
  ) histories

(* Create comparison plots *)
let create_comparison_plots histories =
  (* Plot returns *)
  ascii_plot histories "returns" 10;

  (* Plot losses *)
  ascii_plot histories "losses" 10;

  (* Print statistics *)
  Printf.printf "\n=== Final Statistics ===\n";
  List.iter (fun h ->
    let n = Array.length h.returns in
    let last_10_returns = Array.sub h.returns (max 0 (n - 10)) (min 10 n) in
    let avg_return = Array.fold_left (+.) 0. last_10_returns /.
                     float_of_int (Array.length last_10_returns) in
    Printf.printf "%s:\n" h.name;
    Printf.printf "  Average last 10 returns: %.2f\n" avg_return;
    Printf.printf "  Final loss: %.4f\n" h.losses.(n - 1);
  ) histories

(* Main comparison function *)
let main () =
  print_endline "=== Slide 12: Algorithm Comparison ===";
  print_endline "Training multiple REINFORCE variants...\n";

  let env = create_simple_gridworld 5 in
  let n_episodes = 100 in
  let learning_rate = 0.01 in
  let gamma = 0.99 in

  (* Train different variants *)
  print_endline "Training REINFORCE without baseline...";
  let history_no_baseline =
    train_reinforce_with_history env n_episodes learning_rate gamma in

  print_endline "Training REINFORCE with baseline...";
  let history_baseline =
    train_reinforce_baseline_with_history env n_episodes learning_rate gamma in

  (* Create comparison plots *)
  let histories = [history_no_baseline; history_baseline] in
  create_comparison_plots histories;

  print_endline "\n=== Analysis ===";
  print_endline "Expected observations:";
  print_endline "1. Baseline reduces variance in returns (smoother curve)";
  print_endline "2. Baseline may converge faster to optimal policy";
  print_endline "3. Loss magnitude typically smaller with baseline";
  print_endline "4. Both should eventually reach similar performance"

(*
```
 *)