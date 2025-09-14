(* Algorithm comparison plots for RL workshop *)

open Workshop
open Visualizations  (* For episode visualization *)

(* Extended training history type for named plots *)
type named_history = {
  name: string;
  returns: float array;
  losses: float array;
  color: string;  (* SVG color *)
}

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

(* Generate SVG plot *)
let generate_svg_plot histories metric_name output_file =
  let width = 800 in
  let height = 400 in
  let margin = 50 in
  let plot_width = width - 2 * margin in
  let plot_height = height - 2 * margin in

  (* Get data *)
  let get_data h = match metric_name with
    | "returns" -> moving_average h.returns 10
    | "losses" -> moving_average h.losses 10
    | _ -> failwith "Unknown metric"
  in

  (* Find global min/max *)
  let all_values = List.concat_map (fun h ->
    Array.to_list (get_data h)
  ) histories in
  let min_val = List.fold_left min max_float all_values in
  let max_val = List.fold_left max min_float all_values in
  let range = if max_val -. min_val > 0.001 then max_val -. min_val else 1.0 in

  (* Coordinate transformation *)
  let n_episodes = Array.length (List.hd histories).returns in
  let x_scale x = margin + int_of_float (float_of_int x *. float_of_int plot_width /. float_of_int (n_episodes - 1)) in
  let y_scale y = margin + plot_height - int_of_float ((y -. min_val) /. range *. float_of_int plot_height) in

  (* Create SVG *)
  let oc = open_out output_file in
  Printf.fprintf oc "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
  Printf.fprintf oc "<svg width=\"%d\" height=\"%d\" xmlns=\"http://www.w3.org/2000/svg\">\n" width height;

  (* White background *)
  Printf.fprintf oc "  <rect width=\"%d\" height=\"%d\" fill=\"white\"/>\n" width height;

  (* Draw axes *)
  Printf.fprintf oc "  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"black\" stroke-width=\"2\"/>\n"
    margin (height - margin) (width - margin) (height - margin);  (* X-axis *)
  Printf.fprintf oc "  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"black\" stroke-width=\"2\"/>\n"
    margin margin margin (height - margin);  (* Y-axis *)

  (* Title *)
  let title = String.capitalize_ascii metric_name ^ " per Episode" in
  Printf.fprintf oc "  <text x=\"%d\" y=\"30\" text-anchor=\"middle\" font-size=\"20\" font-weight=\"bold\">%s</text>\n"
    (width / 2) title;

  (* Axis labels *)
  Printf.fprintf oc "  <text x=\"%d\" y=\"%d\" text-anchor=\"middle\" font-size=\"14\">Episode</text>\n"
    (width / 2) (height - 10);
  Printf.fprintf oc "  <text x=\"15\" y=\"%d\" text-anchor=\"middle\" font-size=\"14\" transform=\"rotate(-90 15 %d)\">%s</text>\n"
    (height / 2) (height / 2) (String.capitalize_ascii metric_name);

  (* Axis tick labels *)
  for i = 0 to 4 do
    let episode = i * n_episodes / 4 in
    let x = x_scale episode in
    Printf.fprintf oc "  <text x=\"%d\" y=\"%d\" text-anchor=\"middle\" font-size=\"12\">%d</text>\n"
      x (height - margin + 20) episode;

    let value = min_val +. (float_of_int i /. 4.0) *. range in
    let y = y_scale value in
    Printf.fprintf oc "  <text x=\"%d\" y=\"%d\" text-anchor=\"end\" font-size=\"12\">%.1f</text>\n"
      (margin - 10) y value;
  done;

  (* Plot lines *)
  List.iter (fun h ->
    let data = get_data h in
    let points = Array.mapi (fun i v ->
      Printf.sprintf "%d,%d" (x_scale i) (y_scale v)
    ) data in
    Printf.fprintf oc "  <polyline points=\"%s\" fill=\"none\" stroke=\"%s\" stroke-width=\"2\" opacity=\"0.8\"/>\n"
      (String.concat " " (Array.to_list points)) h.color
  ) histories;

  (* Legend *)
  let legend_y = ref (margin + 20) in
  List.iter (fun h ->
    Printf.fprintf oc "  <line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"2\"/>\n"
      (width - margin - 150) !legend_y (width - margin - 120) !legend_y h.color;
    let legend_text =
      if metric_name = "losses" && h.name = "Actor-Critic" then
        h.name ^ " (value loss)"
      else
        h.name in
    Printf.fprintf oc "  <text x=\"%d\" y=\"%d\" font-size=\"12\">%s</text>\n"
      (width - margin - 115) (!legend_y + 4) legend_text;
    legend_y := !legend_y + 20
  ) histories;

  Printf.fprintf oc "</svg>\n";
  close_out oc;
  Printf.printf "Plot saved to %s\n" output_file

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
      | 3 -> '#'
      | 4 -> 'x'
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
      | 3 -> '#'
      | 4 -> 'x'
      | _ -> '.'
    in
    Printf.printf "  %c - %s\n" char h.name
  ) histories

(* Parse command line arguments *)
let parse_args () =
  let algorithms = ref [] in
  let n_episodes = ref 200 in
  let learning_rate = ref 0.01 in
  let gamma = ref 0.99 in
  let grid_size = ref 5 in
  let env_type = ref "gridworld" in
  let max_steps = ref 200 in
  let help = ref false in

  let usage = Printf.sprintf
    "Usage: %s [OPTIONS]\n\
     Compare different REINFORCE variants.\n\n\
     Options:" Sys.argv.(0) in

  let spec = [
    ("-a", Arg.String (fun s -> algorithms := s :: !algorithms),
     "ALGO  Add algorithm to comparison: reinforce, baseline, actor-critic, reinforce++, backoff-tabular, all (default: reinforce,baseline)");
    ("-n", Arg.Set_int n_episodes,
     "N     Number of training episodes (default: 200)");
    ("-lr", Arg.Set_float learning_rate,
     "LR    Learning rate (default: 0.01)");
    ("-gamma", Arg.Set_float gamma,
     "G     Discount factor (default: 0.99)");
    ("-grid", Arg.Set_int grid_size,
     "SIZE  Grid size for environment (default: 5, try 7 or 9)");
    ("-env", Arg.Set_string env_type,
     "TYPE  Environment type: gridworld, curriculum, verified-curriculum (default: gridworld)");
    ("-max-steps", Arg.Set_int max_steps,
     "N     Maximum steps per episode for Verified Sokoban (default: 200)");
    ("-help", Arg.Set help,
     "      Display this help message");
    ("--help", Arg.Set help,
     "     Display this help message");
  ] in

  Arg.parse spec (fun _ -> ()) usage;

  if !help then begin
    Printf.printf "%s\n" usage;
    Printf.printf "  -a ALGO       Add algorithm to comparison\n";
    Printf.printf "                Available: reinforce, baseline, actor-critic, reinforce++, backoff-tabular, all\n";
    Printf.printf "  -n N          Number of training episodes (default: 200)\n";
    Printf.printf "  -lr LR        Learning rate (default: 0.01)\n";
    Printf.printf "  -gamma G      Discount factor (default: 0.99)\n";
    Printf.printf "  -grid SIZE    Grid size for environment (default: 5, try 7 or 9)\n";
    Printf.printf "  -env TYPE     Environment type: gridworld, curriculum, verified-curriculum (default: gridworld)\n";
    Printf.printf "  -max-steps N  Maximum steps per episode for Verified Sokoban (default: 200)\n";
    Printf.printf "  -help/--help  Display this help message\n\n";
    Printf.printf "Examples:\n";
    Printf.printf "  %s                          # Compare reinforce vs baseline\n" Sys.argv.(0);
    Printf.printf "  %s -a all                   # Compare all algorithms\n" Sys.argv.(0);
    Printf.printf "  %s -a reinforce -a actor-critic -n 300  # Custom comparison\n" Sys.argv.(0);
    Printf.printf "  %s -env curriculum -a baseline  # Use gridworld curriculum environment\n" Sys.argv.(0);
    Printf.printf "  %s -env verified-curriculum -a all -n 500  # Verified Sokoban with all algorithms\n" Sys.argv.(0);
    Printf.printf "  %s -env verified-curriculum -a baseline -n 1000  # Verified Sokoban curriculum\n" Sys.argv.(0);
    exit 0
  end;

  (* Default to reinforce and baseline if no algorithms specified *)
  let algos = if !algorithms = [] then ["reinforce"; "baseline"] else !algorithms in

  (* Handle "all" option *)
  let algos = if List.mem "all" algos then
    ["reinforce"; "baseline"; "actor-critic"; "reinforce++"; "backoff-tabular"]
  else
    algos in

  (List.rev algos, !n_episodes, !learning_rate, !gamma, !grid_size, !env_type, !max_steps)

(* Main function *)
let () =
  Random.self_init ();

  let algorithms, n_episodes, learning_rate, gamma, grid_size, env_type, max_steps = parse_args () in

  print_endline "=== RL Workshop: Algorithm Comparison ===";
  Printf.printf "Comparing algorithms: %s\n" (String.concat ", " algorithms);
  Printf.printf "Environment: %s\n" env_type;
  Printf.printf "Episodes: %d, Learning rate: %.3f, Gamma: %.2f"
    n_episodes learning_rate gamma;
  if env_type = "gridworld" then
    Printf.printf ", Grid: %dx%d" grid_size grid_size
  else if env_type = "verified-curriculum" then
    Printf.printf ", Max steps: %d" max_steps;
  Printf.printf "\n\n";

  let algo_ref = ref "unset_algorithm" in

  let env = match env_type with
    | "gridworld" -> Slide1.create_simple_gridworld grid_size
    | "curriculum" ->
        let curriculum_state = ref Slide10.{
          current_stage = 0;
          episodes_in_stage = 0;
          recent_wins = Queue.create ();
          total_episodes = 0;
          stage_transitions = [];
        } in
        Slide11.create_curriculum_env curriculum_state
    | "verified-curriculum" ->
        (* Verified Sokoban with curriculum learning *)
        Verified.sokoban_curriculum ~max_steps ()
    | _ ->
        Printf.eprintf "Unknown environment type: %s\n" env_type;
        exit 1
  in

  (* Train selected algorithms *)
  let histories = ref [] in
  let full_histories = ref [] in  (* Store full histories with episode data *)

  (* Grid sizes must match what the environment observation space provides *)
  let effective_grid_size =
    match env_type with
    | "curriculum" -> 5  (* Gridworld curriculum is always 5x5 *)
    | "verified-curriculum" -> 10  (* Verified curriculum uses fixed 10x10 max *)
    | _ -> grid_size
  in

  List.iter (fun algo ->
    match algo with
    | "reinforce" ->
        algo_ref := "reinforce";
        print_endline "Training REINFORCE without baseline...";
        let _policy_net, _params, _episodes, history =
          Slide4.train_reinforce env n_episodes learning_rate gamma ~grid_size:effective_grid_size () in
        full_histories := ("reinforce", history) :: !full_histories;
        histories := {
          name = "REINFORCE (no baseline)";
          returns = history.returns;
          losses = history.losses;
          color = "#1f77b4"  (* Blue *)
        } :: !histories

    | "baseline" ->
        algo_ref := "baseline";
        print_endline "Training REINFORCE with baseline...";
        let _policy_net, _params, history =
          Slide5.train_reinforce_with_baseline env n_episodes learning_rate gamma ~grid_size:effective_grid_size () in
        full_histories := ("baseline", history) :: !full_histories;
        histories := {
          name = "REINFORCE with baseline";
          returns = history.returns;
          losses = history.losses;
          color = "#2ca02c"  (* Green *)
        } :: !histories

    | "actor-critic" ->
        algo_ref := "actor-critic";
        print_endline "Training Actor-Critic...";
        (* Actor-Critic uses different learning rates for actor and critic *)
        let lr_critic = learning_rate *. 0.5 in  (* Critic often trains slower *)
        let _policy_net, _policy_params, _value_net, _value_params, history =
          Slide6.train_actor_critic env n_episodes learning_rate lr_critic gamma ~grid_size:effective_grid_size () in
        full_histories := ("actor-critic", history) :: !full_histories;
        histories := {
          name = "Actor-Critic";
          returns = history.returns;
          losses = history.losses;
          color = "#d62728"  (* Red *)
        } :: !histories

    | "reinforce++" ->
        algo_ref := "reinforce++";
        print_endline "Training REINFORCE++...";
        (* REINFORCE++ with clipping and KL penalty *)
        let epsilon = 0.2 in  (* Clipping parameter *)
        let beta = 0.01 in    (* KL penalty coefficient *)
        let _policy_net, _params, history =
          Slide9.train_reinforce_plus_plus env n_episodes learning_rate gamma epsilon beta ~grid_size:effective_grid_size () in
        full_histories := ("reinforce++", history) :: !full_histories;
        histories := {
          name = "REINFORCE++";
          returns = history.returns;
          losses = history.losses;
          color = "#ff7f0e"  (* Orange *)
        } :: !histories

    | "backoff-tabular" ->
        algo_ref := "backoff-tabular";
        print_endline "Training Backoff-Tabular Q-learning...";
        let _agent, history =
          Backoff_tabular.train_backoff env n_episodes learning_rate gamma ~_grid_size:effective_grid_size () in
        full_histories := ("backoff-tabular", history) :: !full_histories;
        histories := {
          name = "Backoff-Tabular Q-learning";
          returns = history.returns;
          losses = history.losses;  (* TD errors *)
          color = "#9467bd"  (* Purple *)
        } :: !histories

    | unknown ->
        Printf.eprintf "Warning: Unknown algorithm '%s', skipping\n" unknown
  ) algorithms;

  let histories = List.rev !histories in

  if histories = [] then begin
    Printf.eprintf "Error: No valid algorithms selected\n";
    exit 1
  end;

  (* ASCII plots for terminal *)
  ascii_plot histories "returns" 10;
  print_endline "\nNote: Actor-Critic shows value network loss, Backoff-Tabular shows TD errors, others show policy loss";
  ascii_plot histories "losses" 10;

  (* SVG plots for files *)
  print_endline "\nGenerating SVG plots...";
  generate_svg_plot histories "returns" "reinforce_returns.svg";
  generate_svg_plot histories "losses" "reinforce_losses.svg";

  (* Visualize collected episodes *)
  print_endline "\n=== Visualizing Episodes ===";
  let env_category =
    match env_type with
    | "gridworld" | "curriculum" -> `Gridworld
    | "verified-curriculum" | "sokoban" -> `Sokoban
    | _ -> `Gridworld  (* Default *)
  in
  List.iter (fun (algo_name, full_history) ->
    visualize_episodes full_history env_type algo_name env_category ~grid_size:effective_grid_size
  ) (List.rev !full_histories);

  (* Print statistics *)
  Printf.printf "\n=== Final Statistics ===\n";
  List.iter (fun h ->
    let n = Array.length h.returns in
    let last_10_returns = Array.sub h.returns (max 0 (n - 10)) (min 10 n) in
    let avg_return = Array.fold_left (+.) 0. last_10_returns /.
                     float_of_int (Array.length last_10_returns) in
    Printf.printf "%s:\n" h.name;
    Printf.printf "  Average last 10 returns: %.2f\n" avg_return;
    let loss_type =
      if h.name = "Actor-Critic" then "value loss"
      else if h.name = "Backoff-Tabular Q-learning" then "TD error"
      else "policy loss" in
    Printf.printf "  Final %s: %.4f\n" loss_type h.losses.(n - 1);
  ) histories;

  (* Episode generation is now done per-algorithm above *)

  print_endline "\n=== Analysis ===";
  print_endline "Expected observations:";
  if List.exists (fun h -> h.name = "REINFORCE with baseline") histories then
    print_endline "- Baseline reduces variance in returns (smoother curve)";
  if List.exists (fun h -> h.name = "Actor-Critic") histories then
    print_endline "- Actor-Critic uses learned baseline for better variance reduction";
  if List.exists (fun h -> h.name = "REINFORCE++") histories then
    print_endline "- REINFORCE++ adds stability through clipping and KL regularization";
  if List.exists (fun h -> h.name = "Backoff-Tabular Q-learning") histories then
    print_endline "- Backoff-Tabular uses hierarchical state abstraction for efficient Q-learning";
  print_endline "- All methods should converge to similar final performance";
  print_endline "- Training stability varies: Actor-Critic > REINFORCE++ > Baseline > REINFORCE"