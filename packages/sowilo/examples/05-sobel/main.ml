(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let image_path = "sowilo/examples/lena.png"

let normalize_gradient img =
  let abs_img = Nx.abs img in
  let min_val = Nx.item [] (Nx.min ~keepdims:false abs_img) in
  let max_val = Nx.item [] (Nx.max ~keepdims:false abs_img) in
  let range = max_val -. min_val in
  if range <= 1e-6 then Nx.zeros_like img
  else
    Nx.div
      (Nx.sub abs_img (Nx.scalar Nx.float32 min_val))
      (Nx.scalar Nx.float32 range)

let () =
  let img = Sowilo.to_float (Nx_io.load_image image_path) in
  let gray = Sowilo.to_grayscale img in
  let gx, gy = Sowilo.sobel gray in
  let fig = Hugin.figure ~width:1200 ~height:400 () in
  let ax1 = Hugin.subplot ~nrows:1 ~ncols:3 ~index:1 fig in
  ignore
    (ax1
    |> Hugin.Plotting.imshow ~data:gray ~cmap:Hugin.Artist.Colormap.gray
    |> Hugin.Axes.set_title "Grayscale"
    |> Hugin.Axes.set_xticks [] |> Hugin.Axes.set_yticks []);
  let ax2 = Hugin.subplot ~nrows:1 ~ncols:3 ~index:2 fig in
  ignore
    (ax2
    |> Hugin.Plotting.imshow ~data:(normalize_gradient gx)
         ~cmap:Hugin.Artist.Colormap.gray
    |> Hugin.Axes.set_title "Sobel X"
    |> Hugin.Axes.set_xticks [] |> Hugin.Axes.set_yticks []);
  let ax3 = Hugin.subplot ~nrows:1 ~ncols:3 ~index:3 fig in
  ignore
    (ax3
    |> Hugin.Plotting.imshow ~data:(normalize_gradient gy)
         ~cmap:Hugin.Artist.Colormap.gray
    |> Hugin.Axes.set_title "Sobel Y"
    |> Hugin.Axes.set_xticks [] |> Hugin.Axes.set_yticks []);
  Hugin.show fig
