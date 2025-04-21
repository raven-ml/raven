(* canny.ml *)
open Common
open Ndarray_cv

let () =
  let img_orig = Ndarray_io.load_image image_path in
  let img_gray = to_grayscale img_orig in
  let img_canny =
    time_it "Canny Edges (50, 150)" (fun () ->
        canny ~threshold1:50.0 ~threshold2:150.0 img_gray)
  in
  plot_compare "Grayscale" img_gray ~cmap1:Art.Colormap.gray "Canny Edges"
    img_canny ~cmap2:Art.Colormap.gray
