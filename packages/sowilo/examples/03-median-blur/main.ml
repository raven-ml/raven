(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let image_path = "sowilo/examples/lena.png"

let () =
  let img = Sowilo.to_float (Nx_io.load_image image_path) in
  let gray = Sowilo.to_grayscale img in
  let median = Sowilo.median_blur ~ksize:5 gray in
  let fig = Hugin.figure ~width:1000 ~height:500 () in
  let ax1 = Hugin.subplot ~nrows:1 ~ncols:2 ~index:1 fig in
  ignore
    (ax1
    |> Hugin.Plotting.imshow ~data:gray ~cmap:Hugin.Artist.Colormap.gray
    |> Hugin.Axes.set_title "Grayscale"
    |> Hugin.Axes.set_xticks [] |> Hugin.Axes.set_yticks []);
  let ax2 = Hugin.subplot ~nrows:1 ~ncols:2 ~index:2 fig in
  ignore
    (ax2
    |> Hugin.Plotting.imshow ~data:median ~cmap:Hugin.Artist.Colormap.gray
    |> Hugin.Axes.set_title "Median Blur (k=5)"
    |> Hugin.Axes.set_xticks [] |> Hugin.Axes.set_yticks []);
  Hugin.show fig
