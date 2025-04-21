(* threshold.ml *)
open Common
open Ndarray_cv

let () =
  let img_orig = Ndarray_io.load_image image_path in
  let img_gray = to_grayscale img_orig in
  let img_thresh =
    time_it "Binary Threshold (128)" (fun () ->
        threshold ~thresh:128 ~maxval:255 ~type_:Binary img_gray)
  in
  plot_compare "Grayscale" img_gray ~cmap1:Art.Colormap.gray "Binary Threshold"
    img_thresh ~cmap2:Art.Colormap.gray
