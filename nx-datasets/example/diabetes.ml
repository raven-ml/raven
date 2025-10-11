(* examples/diabetes.ml *)
open Nx_datasets
open Hugin

let () =
  Printf.printf "Loading Diabetes dataset...\n%!";
  let features, labels = load_diabetes () in

  let n_samples = (Nx.shape features).(0) in
  let n_features = (Nx.shape features).(1) in

  Printf.printf "Dataset loaded successfully!\n";
  Printf.printf "Number of samples: %d\n" n_samples;
  Printf.printf "Number of features: %d\n" n_features;
  Printf.printf "\n%!";

  let age_col = Nx.slice [ Nx.R (0, n_samples); Nx.R (0, 1) ] features in
  let age_f64 = Nx.reshape [| n_samples |] age_col in
  let age = Nx.cast Nx.float32 age_f64 in

  let labels_f64 = Nx.reshape [| n_samples |] labels in
  let labels_reshaped = Nx.cast Nx.float32 labels_f64 in

  Printf.printf "Creating scatter plot: Age vs Disease Progression...\n%!";
  let fig = Figure.create () in
  let ax = Figure.add_subplot fig in

  let ax =
    Plotting.scatter ~label:"Samples" ~marker:Artist.Circle ~c:Artist.Color.blue
      ~x:age ~y:labels_reshaped ax
  in

  let _ax =
    ax
    |> Axes.set_title "Diabetes Dataset: Age vs Disease Progression"
    |> Axes.set_xlabel "Age (normalized)"
    |> Axes.set_ylabel "Disease Progression"
    |> Axes.grid true
  in

  Printf.printf "Displaying plot...\n%!";
  Hugin.show fig;
  Printf.printf "Plot window closed.\n%!";

  (* Use the reshaped 1-D labels for stats to avoid indexing a [n;1] array *)
  let label_data = Array.init n_samples (fun i -> Nx.item [ i ] labels_f64) in

  let sum = Array.fold_left ( +. ) 0.0 label_data in
  let mean = sum /. float_of_int n_samples in
  let min_val = Array.fold_left min max_float label_data in
  let max_val = Array.fold_left max neg_infinity label_data in

  Printf.printf "\nDataset Statistics:\n";
  Printf.printf "Target (Disease Progression):\n";
  Printf.printf "  Mean: %.2f\n" mean;
  Printf.printf "  Min: %.2f\n" min_val;
  Printf.printf "  Max: %.2f\n%!" max_val
