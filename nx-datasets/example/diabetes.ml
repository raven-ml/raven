(* examples/diabetes.ml *)
open Nx_datasets
open Hugin

let setup_logging () =
  Logs.set_reporter (Logs_fmt.reporter ());
  Logs.set_level (Some Logs.Info)

let () =
  setup_logging ();
  Logs.info (fun m -> m "Loading Diabetes dataset...");
  let features, labels = load_diabetes () in

  let n_samples = (Nx.shape features).(0) in
  let n_features = (Nx.shape features).(1) in

  Logs.info (fun m -> m "Dataset loaded successfully.");
  Logs.info (fun m -> m "Number of samples: %d" n_samples);
  Logs.info (fun m -> m "Number of features: %d" n_features);

  let age_col = Nx.slice [ Nx.R (0, n_samples); Nx.R (0, 1) ] features in
  let age_f64 = Nx.reshape [| n_samples |] age_col in
  let age = Nx.cast Nx.float32 age_f64 in

  let labels_f64 = Nx.reshape [| n_samples |] labels in
  let labels_reshaped = Nx.cast Nx.float32 labels_f64 in

  Logs.info (fun m -> m "Creating scatter plot: Age vs Disease Progression...");
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

  Logs.info (fun m -> m "Displaying plot...");
  Hugin.show fig;
  Logs.info (fun m -> m "Plot window closed.");

  (* Use the reshaped 1-D labels for stats to avoid indexing a [n;1] array *)
  let label_data = Array.init n_samples (fun i -> Nx.item [ i ] labels_f64) in

  let sum = Array.fold_left ( +. ) 0.0 label_data in
  let mean = sum /. float_of_int n_samples in
  let min_val = Array.fold_left min max_float label_data in
  let max_val = Array.fold_left max neg_infinity label_data in

  Logs.info (fun m -> m "Dataset Statistics:");
  Logs.info (fun m -> m "  Target (Disease Progression):");
  Logs.info (fun m -> m "    Mean: %.2f" mean);
  Logs.info (fun m -> m "    Min: %.2f" min_val);
  Logs.info (fun m -> m "    Max: %.2f" max_val)
