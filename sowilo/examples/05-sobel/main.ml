(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let image_path = "sowilo/examples/lena.png"

let visualize_sobel (sobel_img : Rune.int16_t) : Rune.uint8_t =
  let abs_sobel = Rune.abs sobel_img in
  let abs_f = Rune.astype Rune.float32 abs_sobel in
  let min_val =
    let t = Rune.min ~keepdims:false abs_f in
    if Rune.ndim t = 0 then Rune.item [] t else 0.0
  in
  let max_val =
    let t = Rune.max ~keepdims:false abs_f in
    if Rune.ndim t = 0 then Rune.item [] t else 255.0
  in
  let range = max_val -. min_val in
  if range <= 1e-6 then Rune.zeros Rune.uint8 (Rune.shape sobel_img)
  else
    let scaled =
      Rune.div
        (Rune.sub abs_f (Rune.scalar Rune.float32 min_val))
        (Rune.scalar Rune.float32 range)
    in
    Sowilo.to_uint8 scaled

let () =
  let img = Rune.of_nx (Nx_io.load_image image_path) in
  let gray = Sowilo.to_grayscale img in
  let sobel_x = Sowilo.sobel ~dx:1 ~dy:0 ~ksize:3 gray in
  let sobel_y = Sowilo.sobel ~dx:0 ~dy:1 ~ksize:3 gray in
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
    |> Hugin.Plotting.imshow ~data:(Rune.to_nx (visualize_sobel sobel_x))
         ~cmap:Hugin.Artist.Colormap.gray
    |> Hugin.Axes.set_title "Sobel X"
    |> Hugin.Axes.set_xticks [] |> Hugin.Axes.set_yticks []);
  let ax3 = Hugin.subplot ~nrows:1 ~ncols:3 ~index:3 fig in
  ignore
    (ax3
    |> Hugin.Plotting.imshow ~data:(Rune.to_nx (visualize_sobel sobel_y))
         ~cmap:Hugin.Artist.Colormap.gray
    |> Hugin.Axes.set_title "Sobel Y"
    |> Hugin.Axes.set_xticks [] |> Hugin.Axes.set_yticks []);
  Hugin.show fig
