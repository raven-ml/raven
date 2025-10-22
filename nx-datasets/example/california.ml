(* example/california.ml *)
open Nx_datasets
open Hugin

let setup_logging () =
  Logs.set_reporter (Logs_fmt.reporter ());
  Logs.set_level (Some Logs.Info)

let astype_f32 arr = Nx.astype Nx.float32 arr

let () =
  setup_logging ();
  Logs.info (fun m -> m "Loading California Housing dataset...");
  let features, labels = load_california_housing () in

  Logs.info (fun m -> m "Preparing data for plotting...");

  let n_samples = (Nx.shape labels).(0) in
  let labels_1d = Nx.reshape [| n_samples |] labels in
  let labels_f32 = astype_f32 labels_1d in

  (* slice produces shape [n;1] — reshape to 1-D [n] so Hugin's scatter receives
     a vector not a 2-D column *)
  let longitude_col = Nx.slice [ Nx.R (0, n_samples); Nx.R (0, 1) ] features in
  let latitude_col = Nx.slice [ Nx.R (0, n_samples); Nx.R (1, 2) ] features in

  let longitude = Nx.reshape [| n_samples |] longitude_col in
  let latitude = Nx.reshape [| n_samples |] latitude_col in

  let longitude_f32 = astype_f32 longitude in
  let latitude_f32 = astype_f32 latitude in

  Logs.info (fun m -> m "Creating figure with subplots...");
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

  Logs.info (fun m -> m "Displaying combined plot...");
  Hugin.show fig;
  Logs.info (fun m -> m "Plot window closed.")
