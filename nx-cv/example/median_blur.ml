(* median_blur.ml *)
open Common
open Nx_cv

let () =
  let img_orig = Nx_io.load_image image_path in
  let img_gray = to_grayscale img_orig in
  let img_median =
    time_it "Median Blur (k=5)" (fun () -> median_blur ~ksize:5 img_gray)
  in
  plot_compare "Grayscale" img_gray ~cmap1:Art.Colormap.gray "Median Blur (k=5)"
    img_median ~cmap2:Art.Colormap.gray
