(* Algorithm comparison plots for RL workshop *)

open Workshop

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
    Printf.fprintf oc "  <text x=\"%d\" y=\"%d\" font-size=\"12\">%s</text>\n"
      (width - margin - 115) (!legend_y + 4) h.name;
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

(* Main function *)
let () =
  Random.self_init ();
  print_endline "=== RL Workshop: Algorithm Comparison ===";
  print_endline "Training multiple REINFORCE variants...\n";

  let env = Slide1.create_simple_gridworld 5 in
  let n_episodes = 200 in
  let learning_rate = 0.01 in
  let gamma = 0.99 in

  (* Train different variants using slide functions *)
  print_endline "Training REINFORCE without baseline...";
  let _policy_net1, _params1, _episodes1, history_no_baseline =
    Slide4.train_reinforce env n_episodes learning_rate gamma in

  print_endline "\nTraining REINFORCE with baseline...";
  let _policy_net2, _params2, history_baseline =
    Slide5.train_reinforce_with_baseline env n_episodes learning_rate gamma in

  (* Convert to named histories *)
  let histories = [
    { name = "REINFORCE (no baseline)";
      returns = history_no_baseline.returns;
      losses = history_no_baseline.losses;
      color = "#1f77b4" };  (* Blue *)
    { name = "REINFORCE with baseline";
      returns = history_baseline.returns;
      losses = history_baseline.losses;
      color = "#2ca02c" };  (* Green *)
  ] in

  (* ASCII plots for terminal *)
  ascii_plot histories "returns" 10;
  ascii_plot histories "losses" 10;

  (* SVG plots for files *)
  print_endline "\nGenerating SVG plots...";
  generate_svg_plot histories "returns" "reinforce_returns.svg";
  generate_svg_plot histories "losses" "reinforce_losses.svg";

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
  ) histories;

  print_endline "\n=== Analysis ===";
  print_endline "Expected observations:";
  print_endline "1. Baseline reduces variance in returns (smoother curve)";
  print_endline "2. Baseline may converge faster to optimal policy";
  print_endline "3. Loss magnitude typically smaller with baseline";
  print_endline "4. Both should eventually reach similar performance"