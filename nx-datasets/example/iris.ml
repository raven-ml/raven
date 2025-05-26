(* example/iris.ml *)
open Nx_datasets
open Hugin

let nd_of_float_array data = Nx.create Nx.float32 [| Array.length data |] data

let get_class_features features labels target_label feature_idx1 feature_idx2 =
  let n_samples = (Nx.shape features).(0) in
  let indices = List.init n_samples (fun i -> i) in

  let filtered_pairs =
    List.filter_map
      (fun i ->
        let label = Nx.get_item [ i; 0 ] labels in
        if label = target_label then
          let f1 = Nx.get_item [ i; feature_idx1 ] features in
          let f2 = Nx.get_item [ i; feature_idx2 ] features in
          Some (f1, f2)
        else None)
      indices
  in
  let list1, list2 = List.split filtered_pairs in
  (Array.of_list list1, Array.of_list list2)

let () =
  Printf.printf "Loading Iris dataset...\n%!";
  let features, labels = load_iris () in

  let sepal_length_idx = 0 in
  let petal_length_idx = 2 in
  let feature1_name = "Sepal Length (cm)" in
  let feature2_name = "Petal Length (cm)" in

  Printf.printf "Preparing data for plotting...\n%!";

  let setosa_f1, setosa_f2 =
    get_class_features features labels 0l sepal_length_idx petal_length_idx
  in
  let versicolor_f1, versicolor_f2 =
    get_class_features features labels 1l sepal_length_idx petal_length_idx
  in
  let virginica_f1, virginica_f2 =
    get_class_features features labels 2l sepal_length_idx petal_length_idx
  in

  let nd_setosa_f1 = nd_of_float_array setosa_f1 in
  let nd_setosa_f2 = nd_of_float_array setosa_f2 in
  let nd_versicolor_f1 = nd_of_float_array versicolor_f1 in
  let nd_versicolor_f2 = nd_of_float_array versicolor_f2 in
  let nd_virginica_f1 = nd_of_float_array virginica_f1 in
  let nd_virginica_f2 = nd_of_float_array virginica_f2 in

  Printf.printf "Creating plot...\n%!";
  let fig = Figure.create () in
  let ax = Figure.add_subplot fig in

  let ax =
    Plotting.scatter ~label:"Setosa" ~marker:Artist.Circle ~c:Artist.Color.red
      ~x:nd_setosa_f1 ~y:nd_setosa_f2 ax
  in
  let ax =
    Plotting.scatter ~label:"Versicolor" ~marker:Artist.Square
      ~c:Artist.Color.green ~x:nd_versicolor_f1 ~y:nd_versicolor_f2 ax
  in
  let ax =
    Plotting.scatter ~label:"Virginica" ~marker:Artist.Triangle
      ~c:Artist.Color.blue ~x:nd_virginica_f1 ~y:nd_virginica_f2 ax
  in

  let _ax =
    ax
    |> Axes.set_title "Iris Dataset Features"
    |> Axes.set_xlabel feature1_name
    |> Axes.set_ylabel feature2_name
    |> Axes.grid true
  in

  Printf.printf "Displaying plot...\n%!";
  Hugin.show fig;
  Printf.printf "Plot window closed.\n%!"
