(** A complete data preparation pipeline — load, clean, transform, and split.

    Load the Iris dataset, inspect its shape and statistics, standardize
    features, split into train/test, and run a nearest-centroid classifier to
    verify the pipeline produces usable data. *)

open Nx

let () =
  (* --- Load dataset --- *)
  let features, labels_raw = Nx_datasets.load_iris () in
  let labels = squeeze labels_raw in
  Printf.printf "Iris dataset loaded\n";
  Printf.printf "  Features: %s  (%s)\n"
    (shape_to_string (shape features))
    (dtype_to_string (dtype features));
  Printf.printf "  Labels:   %s  (%s)\n\n"
    (shape_to_string (shape labels))
    (dtype_to_string (dtype labels));

  (* --- Inspect: per-feature statistics --- *)
  let feature_names =
    [| "sepal length"; "sepal width"; "petal length"; "petal width" |]
  in
  let feat_mean = mean ~axes:[ 0 ] features in
  let feat_std = std ~axes:[ 0 ] features in
  let feat_min = min ~axes:[ 0 ] features in
  let feat_max = max ~axes:[ 0 ] features in
  Printf.printf "Feature statistics:\n";
  for i = 0 to 3 do
    Printf.printf "  %-13s  mean=%.2f  std=%.2f  min=%.2f  max=%.2f\n"
      feature_names.(i) (item [ i ] feat_mean) (item [ i ] feat_std)
      (item [ i ] feat_min) (item [ i ] feat_max)
  done;
  print_newline ();

  (* --- Standardize features (z-score) --- *)
  let x = standardize ~axes:[ 0 ] features in
  Printf.printf "After standardization:\n";
  Printf.printf "  Column means ≈ 0: %s\n" (data_to_string (mean ~axes:[ 0 ] x));
  Printf.printf "  Column stds  ≈ 1: %s\n\n" (data_to_string (std ~axes:[ 0 ] x));

  (* --- Train/test split (80/20) with shuffle --- *)
  let n = (shape features).(0) in
  (* Build a shuffled index array using a seeded OCaml RNG. *)
  let idx_arr = Array.init n Fun.id in
  let rng_state = Random.State.make [| 42 |] in
  for i = n - 1 downto 1 do
    let j = Random.State.int rng_state (i + 1) in
    let tmp = idx_arr.(i) in
    idx_arr.(i) <- idx_arr.(j);
    idx_arr.(j) <- tmp
  done;
  let perm = create int32 [| n |] (Array.map Int32.of_int idx_arr) in
  let x_shuffled = take ~axis:0 perm x in
  let y_shuffled = take ~axis:0 perm labels in

  let n_train = n * 80 / 100 in
  let x_train = slice [ R (0, n_train); A ] x_shuffled in
  let y_train = slice [ R (0, n_train) ] y_shuffled in
  let x_test = slice [ R (n_train, n); A ] x_shuffled in
  let y_test = slice [ R (n_train, n) ] y_shuffled in
  Printf.printf "Split: %d train, %d test\n\n" n_train (n - n_train);

  (* --- Nearest-centroid classifier ---

     Compute the centroid (mean) of each class in the training set. To predict,
     assign each test point to the class of the nearest centroid. *)
  let n_classes = 3 in
  let n_features = (shape x_train).(1) in
  let centroids = zeros float64 [| n_classes; n_features |] in

  for c = 0 to n_classes - 1 do
    let mask = equal_s y_train (Int32.of_int c) in
    let class_samples = compress ~axis:0 ~condition:mask x_train in
    let centroid = mean ~axes:[ 0 ] class_samples in
    for j = 0 to n_features - 1 do
      set_item [ c; j ] (item [ j ] centroid) centroids
    done
  done;

  Printf.printf "Class centroids:\n%s\n\n" (data_to_string centroids);

  (* Predict: for each test point, find nearest centroid. *)
  let n_test = (shape x_test).(0) in
  let predictions = zeros Int32 [| n_test |] in
  for i = 0 to n_test - 1 do
    let point = slice [ I i; A ] x_test in
    let diff = sub centroids (expand_dims [ 0 ] point) in
    let dists = sum ~axes:[ 1 ] (mul diff diff) in
    let nearest = argmin dists in
    set_item [ i ] (item [] nearest) predictions
  done;

  (* Accuracy. *)
  let correct = equal predictions y_test in
  let accuracy =
    item [] (sum (cast Float64 correct)) /. Float.of_int n_test
  in
  Printf.printf "Nearest-centroid accuracy: %.1f%% (%d/%d)\n"
    (accuracy *. 100.0)
    (Float.to_int (accuracy *. Float.of_int n_test))
    n_test
