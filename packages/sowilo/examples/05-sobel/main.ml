(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let image_path = "sowilo/examples/lena.png"

let normalize_gradient img =
  let abs_img = Rune.abs img in
  let min_val = Rune.item [] (Rune.min ~keepdims:false abs_img) in
  let max_val = Rune.item [] (Rune.max ~keepdims:false abs_img) in
  let range = max_val -. min_val in
  if range <= 1e-6 then Rune.zeros_like img
  else
    Rune.div
      (Rune.sub abs_img (Rune.scalar Rune.float32 min_val))
      (Rune.scalar Rune.float32 range)

let () =
  let img = Sowilo.to_float (Rune.of_nx (Nx_io.load_image image_path)) in
  let gray = Sowilo.to_grayscale img in
  let gx, gy = Sowilo.sobel gray in
  let fig = Hugin.figure ~width:1200 ~height:400 () in
  let ax1 = Hugin.subplot ~nrows:1 ~ncols:3 ~index:1 fig in
  ignore
    (ax1
    |> Hugin.Plotting.imshow ~data:(Rune.to_nx gray)
         ~cmap:Hugin.Artist.Colormap.gray
    |> Hugin.Axes.set_title "Grayscale"
    |> Hugin.Axes.set_xticks [] |> Hugin.Axes.set_yticks []);
  let ax2 = Hugin.subplot ~nrows:1 ~ncols:3 ~index:2 fig in
  ignore
    (ax2
    |> Hugin.Plotting.imshow
         ~data:(Rune.to_nx (normalize_gradient gx))
         ~cmap:Hugin.Artist.Colormap.gray
    |> Hugin.Axes.set_title "Sobel X"
    |> Hugin.Axes.set_xticks [] |> Hugin.Axes.set_yticks []);
  let ax3 = Hugin.subplot ~nrows:1 ~ncols:3 ~index:3 fig in
  ignore
    (ax3
    |> Hugin.Plotting.imshow
         ~data:(Rune.to_nx (normalize_gradient gy))
         ~cmap:Hugin.Artist.Colormap.gray
    |> Hugin.Axes.set_title "Sobel Y"
    |> Hugin.Axes.set_xticks [] |> Hugin.Axes.set_yticks []);
  Hugin.show fig
