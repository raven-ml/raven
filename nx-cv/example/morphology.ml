(* morphology.ml *)
open Common
open Nx_cv

let () =
  let img_orig = Nx_io.load_image image_path in
  let img_gray = to_grayscale img_orig in
  (* Binarize for clearer morphology effect *)
  let img_thresh = threshold ~thresh:128 ~maxval:255 ~type_:Binary img_gray in

  let kernel =
    time_it "Get Structuring Element (Rect 5x5)" (fun () ->
        get_structuring_element ~shape:Rect ~ksize:(5, 5))
  in

  let img_eroded = time_it "Erode" (fun () -> erode ~kernel img_thresh) in
  let img_dilated = time_it "Dilate" (fun () -> dilate ~kernel img_thresh) in

  (* Plotting Morphology results *)
  let fig = Hugin.figure ~width:1200 ~height:600 () in
  let ax1 = Hugin.subplot ~nrows:1 ~ncols:3 ~index:1 fig in
  ignore
    (ax1
    |> Plot.imshow ~data:img_thresh ~cmap:Art.Colormap.gray
    |> Ax.set_title "Thresholded" |> Ax.set_xticks [] |> Ax.set_yticks []);
  let ax2 = Hugin.subplot ~nrows:1 ~ncols:3 ~index:2 fig in
  ignore
    (ax2
    |> Plot.imshow ~data:img_eroded ~cmap:Art.Colormap.gray
    |> Ax.set_title "Eroded (5x5 Rect)"
    |> Ax.set_xticks [] |> Ax.set_yticks []);
  let ax3 = Hugin.subplot ~nrows:1 ~ncols:3 ~index:3 fig in
  ignore
    (ax3
    |> Plot.imshow ~data:img_dilated ~cmap:Art.Colormap.gray
    |> Ax.set_title "Dilated (5x5 Rect)"
    |> Ax.set_xticks [] |> Ax.set_yticks []);
  Hugin.show fig;
  print_endline "Plot window closed."
