open Hugin

let () =
  (* Create figure *)
  let fig = Figure.create ~width:800 ~height:600 () in

  (* Add subplot *)
  let ax = Figure.add_subplot ~nrows:1 ~ncols:1 ~index:1 fig in

  (* Create grid *)
  let x = Nx.linspace Nx.float32 ~endpoint:true (-3.0) 3.0 100 in
  let y = Nx.linspace Nx.float32 ~endpoint:true (-3.0) 3.0 100 in
  let xx, yy = Nx.meshgrid x y in

  (* Create a simple 2D function: z = sin(sqrt(x^2 + y^2)) *)
  let x2 = Nx.mul xx xx in
  let y2 = Nx.mul yy yy in
  let r2 = Nx.add x2 y2 in
  let r = Nx.sqrt r2 in
  let z = Nx.sin r in

  (* Create contour levels *)
  let levels = [| -0.9; -0.6; -0.3; 0.0; 0.3; 0.6; 0.9 |] in

  (* Add filled contour *)
  let _ = Plotting.contourf ~levels ~x ~y ~z ax in

  (* Add contour lines *)
  let _ = Plotting.contour ~levels ~x ~y ~z ax in

  (* Set labels *)
  let _ = Axes.set_xlabel "X" ax in
  let _ = Axes.set_ylabel "Y" ax in
  let _ = Axes.set_title "Contour Plot Demo: sin(sqrt(x² + y²))" ax in

  (* Save figure *)
  Printf.printf "Saving contour plot to contour_demo.png\n%!";
  Hugin.savefig "contour_demo.png" fig;
  Printf.printf "Done!\n%!"
