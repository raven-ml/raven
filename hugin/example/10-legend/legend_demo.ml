open Hugin

let () =
  (* Create figure *)
  let fig = Figure.create ~width:800 ~height:600 () in

  (* Add subplot *)
  let ax = Figure.add_subplot ~nrows:1 ~ncols:1 ~index:1 fig in

  (* Create some data *)
  let x = Nx.linspace Nx.float32 ~endpoint:true 0.0 10.0 100 in

  (* Plot multiple functions with labels *)
  let _ =
    Plotting.plot ~color:Artist.Color.blue ~label:"sin(x)" ~x ~y:(Nx.sin x) ax
  in
  let _ =
    Plotting.plot ~color:Artist.Color.red ~label:"cos(x)" ~x ~y:(Nx.cos x) ax
  in
  let _ =
    Plotting.plot ~color:Artist.Color.green ~linestyle:Artist.Dashed
      ~label:"sin(x) + cos(x)" ~x
      ~y:(Nx.add (Nx.sin x) (Nx.cos x))
      ax
  in

  (* Add scatter plot *)
  let x_scatter = Nx.linspace Nx.float32 ~endpoint:true 0.0 10.0 20 in
  let y_scatter = Nx.mul_s (Nx.sin x_scatter) 0.8 in
  let _ =
    Plotting.scatter ~c:Artist.Color.orange ~marker:Artist.Circle ~s:30.0
      ~label:"data points" ~x:x_scatter ~y:y_scatter ax
  in

  (* Enable legend *)
  let _ = Axes.legend true ax in

  (* Set labels and title *)
  let _ = Axes.set_xlabel "x" ax in
  let _ = Axes.set_ylabel "y" ax in
  let _ = Axes.set_title "Legend Demo: Trigonometric Functions" ax in

  (* Enable grid *)
  let _ = Axes.grid true ax in

  (* Save figure *)
  Printf.printf "Saving legend demo to legend_demo.png\n%!";
  Hugin.savefig "legend_demo.png" fig;
  Printf.printf "Done!\n%!"
