(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* gaussian_blur.ml *)
open Common
open Sowilo

let () =
  let img_orig = load_image_rune image_path in
  print_endline "Loaded image successfully";
  let img_gray = to_grayscale img_orig in
  print_endline "Converted to grayscale";
  (* Usually blur grayscale *)
  let img_blurred =
    time_it "Gaussian Blur (5x5, sigma=1.5)" (fun () ->
        gaussian_blur ~ksize:(5, 5) ~sigmaX:1.5 img_gray)
  in
  print_endline "Applied Gaussian Blur";
  (* Plot the original grayscale and blurred images *)
  plot_compare "Grayscale" img_gray ~cmap1:Art.Colormap.gray "Gaussian Blur"
    img_blurred ~cmap2:Art.Colormap.gray
