(* sobel.ml *)
open Common
open Sowilo

let () =
  let img_orig = load_image_rune image_path in
  let img_gray = to_grayscale img_orig in
  let img_sobel_x =
    time_it "Sobel X (k=3)" (fun () -> sobel ~dx:1 ~dy:0 ~ksize:3 img_gray)
  in
  let img_sobel_y =
    time_it "Sobel Y (k=3)" (fun () -> sobel ~dx:0 ~dy:1 ~ksize:3 img_gray)
  in

  let img_sobel_x_vis =
    time_it "Visualize Sobel X" (fun () -> visualize_sobel img_sobel_x)
  in
  let img_sobel_y_vis =
    time_it "Visualize Sobel Y" (fun () -> visualize_sobel img_sobel_y)
  in

  (* Plotting Sobel results *)
  let fig = Hugin.figure ~width:1200 ~height:600 () in
  let ax1 = Hugin.subplot ~nrows:1 ~ncols:3 ~index:1 fig in
  ignore
    (ax1
    |> Plot.imshow ~data:(rune_to_nx img_gray) ~cmap:Art.Colormap.gray
    |> Ax.set_title "Grayscale" |> Ax.set_xticks [] |> Ax.set_yticks []);
  let ax2 = Hugin.subplot ~nrows:1 ~ncols:3 ~index:2 fig in
  ignore
    (ax2
    |> Plot.imshow ~data:(rune_to_nx img_sobel_x_vis) ~cmap:Art.Colormap.gray
    |> Ax.set_title "Sobel X (Vis)"
    |> Ax.set_xticks [] |> Ax.set_yticks []);
  let ax3 = Hugin.subplot ~nrows:1 ~ncols:3 ~index:3 fig in
  ignore
    (ax3
    |> Plot.imshow ~data:(rune_to_nx img_sobel_y_vis) ~cmap:Art.Colormap.gray
    |> Ax.set_title "Sobel Y (Vis)"
    |> Ax.set_xticks [] |> Ax.set_yticks []);
  Hugin.show fig;
  print_endline "Plot window closed."
