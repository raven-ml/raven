(* example/example_mnist.ml *)
open Nx_datasets
open Hugin

let () =
  Printf.printf "Loading MNIST dataset...\n%!";
  let (x_train, y_train), _ = load_mnist () in

  Printf.printf "Preparing data for plotting...\n%!";
  let n_show = 9 in
  let fig = Figure.create ~width:600 ~height:600 () in

  Printf.printf "Creating subplots...\n%!";
  for i = 0 to n_show - 1 do
    let img_slice =
      Nx.slice
        [ Nx.R (i, i + 1); Nx.R (0, 28); Nx.R (0, 28); Nx.R (0, 1) ]
        x_train
    in
    let img_2d = Nx.reshape [| 28; 28 |] img_slice in
    let label = Nx.item [ i; 0 ] y_train in

    let ax = Figure.add_subplot ~nrows:3 ~ncols:3 ~index:(i + 1) fig in

    let ax = Plotting.imshow ~data:img_2d ~cmap:Artist.Colormap.gray ax in

    let ax =
      ax
      |> Axes.set_title (Printf.sprintf "Label: %d" label)
      |> Axes.set_xticks [] |> Axes.set_yticks []
    in
    ignore ax
  done;

  Printf.printf "Displaying plot...\n%!";
  Hugin.show fig;
  Printf.printf "Plot window closed.\n%!"
