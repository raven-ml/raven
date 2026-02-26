(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let image_path = "sowilo/examples/lena.png"

let () =
  let img = Sowilo.to_float (Rune.of_nx (Nx_io.load_image image_path)) in
  let gray = Sowilo.to_grayscale img in
  let thresh = Sowilo.threshold 0.5 gray in
  let kernel = Sowilo.structuring_element Rect (5, 5) in
  let eroded = Sowilo.erode ~kernel thresh in
  let dilated = Sowilo.dilate ~kernel thresh in
  let fig = Hugin.figure ~width:1200 ~height:400 () in
  let ax1 = Hugin.subplot ~nrows:1 ~ncols:3 ~index:1 fig in
  ignore
    (ax1
    |> Hugin.Plotting.imshow ~data:(Rune.to_nx thresh)
         ~cmap:Hugin.Artist.Colormap.gray
    |> Hugin.Axes.set_title "Thresholded"
    |> Hugin.Axes.set_xticks [] |> Hugin.Axes.set_yticks []);
  let ax2 = Hugin.subplot ~nrows:1 ~ncols:3 ~index:2 fig in
  ignore
    (ax2
    |> Hugin.Plotting.imshow ~data:(Rune.to_nx eroded)
         ~cmap:Hugin.Artist.Colormap.gray
    |> Hugin.Axes.set_title "Eroded (5x5)"
    |> Hugin.Axes.set_xticks [] |> Hugin.Axes.set_yticks []);
  let ax3 = Hugin.subplot ~nrows:1 ~ncols:3 ~index:3 fig in
  ignore
    (ax3
    |> Hugin.Plotting.imshow ~data:(Rune.to_nx dilated)
         ~cmap:Hugin.Artist.Colormap.gray
    |> Hugin.Axes.set_title "Dilated (5x5)"
    |> Hugin.Axes.set_xticks [] |> Hugin.Axes.set_yticks []);
  Hugin.show fig
