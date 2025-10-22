(* examples/breast_cancer.ml *)
open Nx_datasets
open Hugin

let setup_logging () =
  Logs.set_reporter (Logs_fmt.reporter ());
  Logs.set_level (Some Logs.Info)

let () =
  setup_logging ();
  Logs.info (fun m -> m "Loading Breast Cancer dataset...");
  let features, labels = load_breast_cancer () in

  let n_samples = (Nx.shape features).(0) in
  let n_features = (Nx.shape features).(1) in

  Logs.info (fun m -> m "Dataset loaded successfully.");
  Logs.info (fun m -> m "Number of samples: %d" n_samples);
  Logs.info (fun m -> m "Number of features: %d" n_features);

  let labels_1d =
    let shp = Nx.shape labels in
    if Array.length shp = 2 && shp.(1) = 1 then
      Nx.reshape [| n_samples |] labels
    else labels
  in

  let malignant_count = ref 0 in
  let benign_count = ref 0 in
  for i = 0 to n_samples - 1 do
    let label = Nx.item [ i ] labels_1d in
    if label = 1 then incr malignant_count else incr benign_count
  done;

  Logs.info (fun m -> m "Class distribution:");
  Logs.info (fun m -> m "  Malignant (M): %d samples" !malignant_count);
  Logs.info (fun m -> m "  Benign (B): %d samples" !benign_count);

  let radius_col = Nx.slice [ Nx.R (0, n_samples); Nx.R (0, 1) ] features in
  let texture_col = Nx.slice [ Nx.R (0, n_samples); Nx.R (1, 2) ] features in

  let radius = Nx.cast Nx.float32 (Nx.reshape [| n_samples |] radius_col) in
  let texture = Nx.cast Nx.float32 (Nx.reshape [| n_samples |] texture_col) in

  let malignant_indices = ref [] in
  let benign_indices = ref [] in
  for i = 0 to n_samples - 1 do
    let label = Nx.item [ i ] labels_1d in
    if label = 1 then malignant_indices := i :: !malignant_indices
    else benign_indices := i :: !benign_indices
  done;

  let extract_by_indices indices tensor =
    let arr = Array.of_list (List.rev indices) in
    let n = Array.length arr in
    let data = Array.init n (fun i -> Nx.item [ arr.(i) ] tensor) in
    Nx.create Nx.float32 [| n |] data
  in

  let malignant_radius = extract_by_indices !malignant_indices radius in
  let malignant_texture = extract_by_indices !malignant_indices texture in
  let benign_radius = extract_by_indices !benign_indices radius in
  let benign_texture = extract_by_indices !benign_indices texture in

  Logs.info (fun m ->
      m "Creating scatter plot: Mean Radius vs Mean Texture...");
  let fig = Figure.create () in
  let ax = Figure.add_subplot fig in

  let ax =
    Plotting.scatter ~label:"Malignant" ~marker:Artist.Circle
      ~c:Artist.Color.red ~x:malignant_radius ~y:malignant_texture ax
  in
  let ax =
    Plotting.scatter ~label:"Benign" ~marker:Artist.Square ~c:Artist.Color.blue
      ~x:benign_radius ~y:benign_texture ax
  in

  let _ax =
    ax
    |> Axes.set_title "Breast Cancer Dataset: Mean Radius vs Mean Texture"
    |> Axes.set_xlabel "Mean Radius"
    |> Axes.set_ylabel "Mean Texture"
    |> Axes.grid true
  in

  Logs.info (fun m -> m "Displaying plot...");
  Hugin.show fig;
  Logs.info (fun m -> m "Plot window closed.")
