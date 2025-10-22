open Nx_datasets
open Hugin

let setup_logging () =
  Logs.set_reporter (Logs_fmt.reporter ());
  Logs.set_level (Some Logs.Info)

let label_names =
  [|
    "airplane";
    "automobile";
    "bird";
    "cat";
    "deer";
    "dog";
    "frog";
    "horse";
    "ship";
    "truck";
  |]

let class_name idx =
  if idx >= 0 && idx < Array.length label_names then label_names.(idx)
  else Printf.sprintf "class %d" idx

let () =
  setup_logging ();
  Logs.info (fun m -> m "Loading CIFAR-10 dataset...");
  let (x_train, y_train), _ = load_cifar10 () in

  Logs.info (fun m -> m "Preparing data for plotting...");
  let n_show = 9 in
  let fig = Figure.create ~width:600 ~height:600 () in

  Logs.info (fun m -> m "Creating image grid...");
  for i = 0 to n_show - 1 do
    let img_slice =
      Nx.slice
        [ Nx.R (i, i + 1); Nx.R (0, 32); Nx.R (0, 32); Nx.R (0, 3) ]
        x_train
    in
    let img = Nx.reshape [| 32; 32; 3 |] img_slice in
    let img = Nx.mul_s (Nx.cast Nx.float32 img) (1.0 /. 255.0) in
    let label_idx = Nx.item [ i; 0 ] y_train in
    let ax = Figure.add_subplot ~nrows:3 ~ncols:3 ~index:(i + 1) fig in
    let ax = Plotting.imshow ~data:img ax in
    let ax =
      Axes.set_title
        (Printf.sprintf "%s (%d)" (class_name label_idx) label_idx)
        ax
    in
    let ax = Axes.set_xticks [] ax in
    let ax = Axes.set_yticks [] ax in
    ignore ax
  done;

  Logs.info (fun m -> m "Displaying plot...");
  Hugin.show fig;
  Logs.info (fun m -> m "Plot window closed.")
