(* grayscale.ml *)
open Common
open Sowilo

let () =
  let img_orig = load_image_rune image_path in
  let img_gray =
    time_it "Grayscale Conversion" (fun () -> to_grayscale img_orig)
  in
  plot_compare "Original" img_orig "Grayscale" img_gray ~cmap2:Art.Colormap.gray
