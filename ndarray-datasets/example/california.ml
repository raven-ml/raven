(* example/california.ml *)
open Ndarray_datasets
open Hugin

let () = Printexc.record_backtrace true
let astype_f32 arr = Ndarray.astype Ndarray.float32 arr

let () =
  Printf.printf "Loading California Housing dataset...\n%!";
  let features, labels = load_california_housing () in

  Printf.printf "Preparing data for plotting...\n%!";

  let n_samples = (Ndarray.shape labels).(0) in
  let labels_1d = Ndarray.reshape [| n_samples |] labels in
  let labels_f32 = astype_f32 labels_1d in

  let longitude = Ndarray.slice [| 0; 0 |] [| n_samples; 1 |] features in
  let latitude = Ndarray.slice [| 0; 1 |] [| n_samples; 2 |] features in
  let longitude_f32 = astype_f32 longitude in
  let latitude_f32 = astype_f32 latitude in

  Printf.printf "Creating figure with subplots...\n%!";
  let fig = Figure.create ~width:1200 ~height:600 () in

  let ax_hist = Figure.add_subplot ~nrows:1 ~ncols:2 ~index:1 fig in
  let ax_hist = Plotting.hist ~bins:(`Num 50) ~x:labels_f32 ax_hist in
  let ax_hist =
    ax_hist
    |> Axes.set_title "Distribution of Median House Value"
    |> Axes.set_xlabel "Median House Value ($)"
    |> Axes.set_ylabel "Number of Districts"
    |> Axes.grid true
  in
  ignore ax_hist;

  let ax_scatter = Figure.add_subplot ~nrows:1 ~ncols:2 ~index:2 fig in
  let ax_scatter =
    Plotting.scatter ~s:1.0 ~marker:Artist.Pixel ~x:longitude_f32
      ~y:latitude_f32 ax_scatter
  in
  let ax_scatter =
    ax_scatter
    |> Axes.set_title "Housing Locations"
    |> Axes.set_xlabel "Longitude"
    |> Axes.set_ylabel "Latitude" |> Axes.grid true
  in
  ignore ax_scatter;

  Printf.printf "Displaying combined plot...\n%!";
  Hugin.show fig;
  Printf.printf "Plot window closed.\n%!"
